from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Union
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import argparse
import os
import yaml
import logging
import re

from pathlib import Path
from src.model import get_model
from src.utils import *
from src.dataset import *
from src.prompt import *
from src.config import *
from src.utils import get_path_to_code_uncertainty
from ast import literal_eval


def target_replace(class_text, f, d):
    # Pattern to match both triple single and triple double quotes docstrings
    pattern = r'(?:\'\'\'[\s\S]*?\'\'\')|("""[\s\S]*?""")'

    # Find the docstring of the function f
    func_pattern = rf'(def\s+{f}\s*\([^)]*\)\s*:)\s*?("""[\s\S]*?""")|(\'\'\'[\s\S]*?\'\'\')'
    func_docstring_match = re.search(func_pattern, class_text)

    if func_docstring_match:
        # Extract the docstring of the function f
        func_docstring = func_docstring_match.group(2) or func_docstring_match.group(3)

        # Replace the docstring of the function f with the given docstring d
        modified_text = class_text.replace(func_docstring, d)
    else:
        # If the function f is not found, return the original text
        modified_text = class_text

    # Store all other docstrings
    docstrings = re.findall(pattern, modified_text)

    # Clean up docstrings (remove empty matches and strip whitespace)
    docstrings = [ds for ds in docstrings if ds]
    docstrings = [ds.strip() for ds in docstrings]

    return modified_text, docstrings

@dataclass
class TransformDataset():
    """
    Handles utilities for transforming dataset
    Attributes
    ----------
    oracle : OracleModel
        
    prompt_fn : Dict[str, str]
        Path to the prompt config file.
    prompt_config : Dict[str, str]
        Dictionary with all necessary prompts.

    Methods
    ----------
    mask(context: str) -> Dict[str, str]:
        Return sys + query prompts for answering asst model query.

    answer_query(context: str, asst_query: str) -> str:
        Generates answer to asst model query
    """
    config: Config
    sent_len: int
    num_questions: int
    
    def __post_init__(self):
        self.load()

    def load(self):
        if config.oracle_model is None:
            raise ValueError("Must specify oracle model")

        code_uncert_dir = get_path_to_code_uncertainty()
        self.prompt_config: dict[str, str] = yaml.safe_load(open(Path(code_uncert_dir, self.config.oracle_model.prompt_fn), "r"))
        self.oracle = get_model(config=self.config.oracle_model)
        self.transformation = self.config.dataset.input_transformation_type
    

    def mask(self, example: Dict) -> str:
        """
        Masks the input of a single example.
        """
        question_words: list[str] = example["orig_input"].split(" ")
        num_words = len(question_words)
        # mask_words = np.random.randint(0, num_words, int(num_words*mask_rate))
        mask_words = np.random.choice(num_words, 
                                      size=int(num_words*self.config.dataset.masking_rate), 
                                      replace=False)

        new_question: list[str] = []
        for i, word in enumerate(question_words):
            if i in mask_words:
                new_question.append("#"*len(word))
            else:
                new_question.append(word)

        return " ".join(new_question)

    def summarize_with_prompt(self, example: Dict) -> Union[str,list[str]]:
        """
        Summarize the question using self.prompt_config
        """
        # cap the question at 12 000 characters (I think some were over causing tokenizer errors) 
        args = {"question": example["orig_input"][:12000], "sent_len": self.sent_len, "task": "transform"}
        prompt_set = get_prompts(self.prompt_config, 
                                 self.transformation, 
                                 args,
                                 transform=True)
        generated = self.oracle.generate(prompt_set, None)
        if type(generated) is list:
            generated = generated[0]

        # diff formatting of header
        generated = generated.removeprefix('###SUMMARY\n')
        generated = generated.removeprefix('###SUMMARY \n')
        generated = generated.removeprefix('### SUMMARY\n')
        generated = generated.removeprefix('### SUMMARY \n')
        generated = generated.removeprefix('SUMMARY\n')
        generated = generated.removeprefix('SUMMARY \n')
        generated = generated.split("SUMMARY")[-1]
        return generated
    
    def summarize_code(self, solutions: list[str]) -> Dict[int, str]:
        """
        Summarize (at most) the first three solutions using self.prompt_config
        """
        # arb use first solution for now if exists
        to_summarize = solutions[0]
        summaries = {}
        for i, soln in enumerate(to_summarize):
            args = {"code": soln, "sent_len": self.sent_len, "task":"summarize"} 
            prompt_set = get_prompts(self.prompt_config, 
                                     self.config.dataset.solution_info_type, 
                                     args,
                                     transform=True)
            generated = self.oracle.generate(prompt_set, None)
            if type(generated) is list:
                generated = generated[0]
            # diff formatting of header
            generated = generated.removeprefix('###SUMMARY\n')
            generated = generated.removeprefix('###SUMMARY \n')
            generated = generated.removeprefix('### SUMMARY\n')
            generated = generated.removeprefix('### SUMMARY \n')
            summaries[i] = generated

        return summaries 

    def mask_and_naturalize(self, example: Dict) -> Union[str, list[str]]:
        """
        Mask and summarize the question using self.prompt_config
        """
        args = {"question": self.mask(example), "sent_len": self.sent_len}
        prompt_set = get_prompts(self.prompt_config, 
                                 "mask_naturalize", 
                                 args,
                                 transform=True)
        generated = self.oracle.generate(prompt_set, None)
        if type(generated) is list:
            generated = generated[0]

        # diff formatting of header
        generated = generated.removeprefix('###SUMMARY\n')
        generated = generated.removeprefix('###SUMMARY \n')
        generated = generated.removeprefix('### SUMMARY\n')
        generated = generated.removeprefix('### SUMMARY \n')
        return generated

    
    def summarize_funcs(self, example: Dict) -> str:
        """
        Replace function docstrings in the skeleton with sonnet generated summaries
        """
        skeleton = example['orig_input']
        for method_info in example['method_info']:
            fn_name = method_info['method_name']
            descp = method_info['method_description']
            args = {'function': descp}
            prompt = get_prompts(self.prompt_config, 
                                 self.transformation, 
                                 args,
                                 transform=True)
            generated = self.oracle.generate(prompt, None)
            if type(generated) is list:
                generated = generated[0]

            new_skel, _ = target_replace(skeleton, fn_name, f'"""{generated}"""')
            skeleton = new_skel

        return skeleton


    def get_dataset_dir(self) -> Path:
        code_uncert_dir = get_path_to_code_uncertainty()

        dataset_name = self.config.dataset.name
        difficulty = self.config.dataset.difficulty
        split = self.config.dataset.split

        if difficulty is not None:
            data_path = Path(code_uncert_dir, "data", dataset_name, difficulty, split)
        else:
            data_path = Path(code_uncert_dir, "data", dataset_name, split)
        if Path(data_path, "input", "vanilla").exists():
            return data_path
        else:
            print(f"{str(data_path)}")
            raise AssertionError(f"Vanilla verison of {str(data_path)} does not exist!")
    
    def get_transform_dataset_dir(self) -> Path:
        """
        Given a dataset config and assuming the dataset has already been transformed,
        returns the directory where the dataset is stored.
        
        Also handles error where dataset has not yet been transofrmed.
        """
        data_dir = self.get_dataset_dir()
        data_path = Path(data_dir, self.config.dataset.transformation_type)
        if data_path.exists():
            return data_path
        else:
            raise AssertionError(f"Transformation {transform} does not exist!")

    def apply(self):
        question_transforms = ["plain_summary", "fuzzy_question", "mask_naturalize", "func_summary"]
        code_transforms = ["plain_summary", "1_sentence_summary"]

        if self.transformation in question_transforms:
            self.transform_question()
        else:
            raise ValueError("question transformation does not exist")

        if self.config.dataset.solution_info_type in code_transforms:
            self.transform_code()
        elif self.config.dataset.solution_info_type == "vanilla":
            self.vanilla_code()
        else:
            raise ValueError("solution transformation does not exist")

    def vanilla_code(self):
        vanilla_jsons = list(Path(self.get_dataset_dir(), "input", "vanilla").glob("*.json"))
        # only do sum number of questions (testing sample)
        if self.num_questions != -1:
            vanilla_jsons = vanilla_jsons[:self.num_questions]
        
        for q_path in tqdm(vanilla_jsons):
            with open(q_path, "r") as f_orig:
                example = json.loads(f_orig.readline())
                qid = example['qid']
                new_ex = {
                        "qid": qid,
                        "split": example['split'],
                        "solutions": example['solutions'],
                    }

                transform_path = Path(self.get_dataset_dir(), 
                                      "solutions", 
                                      self.transformation, 
                                      f"{qid}.json")

                if not transform_path.parent.exists():
                    transform_path.parent.mkdir(parents=True)
                with transform_path.open("w+") as f_out:
                    json.dump(new_ex, f_out)

    def transform_code(self):
        vanilla_jsons = list(Path(self.get_dataset_dir(), "input", "vanilla").glob("*.json"))

        # only do sum number of questions (testing sample)
        if self.num_questions != -1:
            vanilla_jsons = vanilla_jsons[:self.num_questions]
        
        for q_path in tqdm(vanilla_jsons):
            with open(q_path, "r") as f_orig:
                example = json.loads(f_orig.readline())
                qid = example['qid']
                try:
                    new_ex = {
                            "qid": qid,
                            "difficulty": example['difficulty'],
                            "split": example['split'],
                            "solutions": example['solutions'],
                            "solution_info": self.summarize_code(literal_eval(example["solutions"]))
                        }
                except:
                    new_ex = {
                            "qid": qid,
                            "difficulty": example['difficulty'],
                            "split": example['split'],
                            "solutions": example['solutions'],
                            "solution_info": self.summarize_code(example["solutions"])
                        }


                transform_path = Path(self.get_dataset_dir(), 
                                      "solutions", 
                                      self.transformation, 
                                      f"{qid}.json")

                if not transform_path.parent.exists():
                    transform_path.parent.mkdir(parents=True)
                with transform_path.open("w+") as f_out:
                    json.dump(new_ex, f_out)

    def transform_question(self):
        transform_type = self.transformation

        if transform_type in ["plain_summary", "fuzzy_question"]:
            transform_func = self.summarize_with_prompt
        elif transform_type == "mask_naturalize":
            transform_func = self.mask_and_naturalize
        elif transform_type in ['func_summary']:
            transform_func = self.summarize_funcs
        else:
            raise ValueError(f"Transformation type {transform_type} is not supported")
    
        vanilla_jsons = list(Path(self.get_dataset_dir(), "input", "vanilla").glob("*.json"))

        # only do sum number of questions (testing sample)
        if self.num_questions != -1:
            vanilla_jsons = vanilla_jsons[:self.num_questions]
        
        for q_path in tqdm(vanilla_jsons):
            with open(q_path, "r") as f_orig:
                example = json.loads(f_orig.readline())
                qid = example['qid']
                transform_path = Path(self.get_dataset_dir(),
                                   "input", 
                                   transform_type, 
                                   f"{qid}.json")               
                if transform_path.exists():
                    continue

                example['input'] = transform_func(example)
                example['transformation_type'] = self.transformation

                transform_path = Path(self.get_dataset_dir(),
                                      "input", 
                                      transform_type, 
                                      f"{qid}.json")

                if not transform_path.parent.exists():
                    transform_path.parent.mkdir(parents=True)
                with transform_path.open("w+") as f_out:
                    json.dump(example, f_out)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_sent", type=int, default=1)
    parser.add_argument("--config_fn", type=str, default="configs/transform_dataset.yaml")

# just for testing, should not use in practice
    parser.add_argument("--num_questions", type=int, default=-1)

    args = parser.parse_args()

    config = parse_config(args.config_fn)
    transform = TransformDataset(config, args.num_sent, args.num_questions)
    transform.apply()

    






