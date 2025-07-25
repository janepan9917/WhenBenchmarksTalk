import argparse
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from src.utils import *
from src.model import *

# import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import itertools
from collections import defaultdict
# from sklearn.linear_model import LinearRegression
import torch
import math



def parse_model_output(model_output):
    """
    
    """
    if "Class:" in model_output:
        return model_output.split("Class:")[1].strip()
    
    return model_output


def get_prompts(ir_data, ir_step):
    user_feedback = ir_data["iter_refinement_data"][ir_step]["user_ir_answer"]
    config = yaml.safe_load(open("prompts/analysis/classify_correctness.yaml", "r"))
    sys_prompt = config[f"sys_prompt"]
    query_prompt = config["query_prompt"].format(user_feedback)   

    return sys_prompt, query_prompt
    
def get_llm_eval_classifications(data_dir):
    """
    This function saves the llm classification for each sample in each question
    in "llm_eval_correctness_classifications_4o.json" in each problem's directory.
    """
    data_dir = os.path.join(data_dir, "raw_output")

    # Get Together AI model
    model = OpenAIModel(config=None)
    model.load_without_config("gpt-4o", os.getenv("OPENAI_KEY"))

    logging.warning("Hardcoded sample list here")

    # For each question
    for qid in os.listdir(data_dir):
        logging.warning("Processing question %s", qid)
        
        qid_dir = os.path.join(data_dir, qid)
        qid_data_dir = os.path.join(qid_dir, "generation_data")

        data_fn = os.path.join(qid_dir, f"llm_eval_correctness_classifications_4o.json")

        if os.path.exists(data_fn):
            print(f"Found data for {data_fn}")
        else:
            # For each sample
            probs = {}
            for sample in os.listdir(qid_data_dir):
                sample_dir = os.path.join(qid_data_dir, sample)
                
                ir_data = read_json(os.path.join(sample_dir, "iter_refinement_data.json"))
                
                labels = []
                full_outputs = []

                # For each IR step, get edit distance between curr and last code iter
                for ir_step in "0123":
                    sys_prompt, query_prompt = get_prompts(ir_data, ir_step)
                    
                    model_output = model._generate(sys_prompt, query_prompt)[0]

                    model_label = parse_model_output(model_output)
                    labels.append(model_label)
                    full_outputs.append(model_output)

                probs[sample] = {
                    "labels": labels,
                    "full_outputs": full_outputs,
                }
            
            print(f"Saving llm classification data to {data_fn}")
            with open(data_fn, "w") as f:
                f.write(json.dumps(probs, default=np_encoder))
                

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Testing a Language Model on Python Code")

    # Data base directory
    parser.add_argument("--data_dir", type=str, default=None, help="Where the data is stored")
    args = parser.parse_args()
    get_llm_eval_classifications(args.data_dir)
