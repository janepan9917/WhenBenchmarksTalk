from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from datasets import load_dataset
import numpy as np
import argparse
import os
import yaml
import logging
import random

from src.model import *
from src.utils import *
from src.dataset import *
from src.prompt import *
from src.config import *
from eval.apps.test_one_solution import evaluate_generations

@dataclass
class ModelRole():
    """
    This class is a wrapper over a Model which provides prompting and other
    role-defining utilities.

    Attributes
    ----------
    config : ModelConfig
    model: Model
        Model that does the actual generation
    prompt_config : Dict[str, str]
        Dictionary with all necessary prompts.
    """
    config: ModelConfig

    def __post_init__(self):
        self.load()
    
    def load(self):
        # TODO: fix the nonsense below
        assert("Assumes that you are running from within /src.")

        self.prompt_config = yaml.safe_load(open(self.config.prompt_fn, "r"))
        self.model = get_model(config=self.config)
    
    def reset_messages(self):
        self.model.reset_messages()

@dataclass
class APPSOnlineCodeEval(ModelRole):

    def load(self):
        self.problem = None
    
    def reset_messages(self):
        return

    def get_problem(self, problem_id):
        problem = load_dataset("codeparrot/apps")["test"].select(range(problem_id, problem_id+1))[0]
        problem["solutions"] = json.loads(problem["solutions"])
        problem["input_output"] = json.loads(problem["input_output"])
        self.problem = problem

    def answer_query(self, prompt_args) -> Dict:
        code = prompt_args["full_solution"]
        if "python" in code.split("\n")[0]:
            code = "\n".join(code.split("\n")[1:])
        test_case_performance = evaluate_generations([code], self.problem)[0]

        inputs = self.problem["input_output"]["inputs"]
        outputs = self.problem["input_output"]["outputs"]

        # compile error
        if test_case_performance[0] == -2:
            return {
                "prompts": None,
                "response": f"\"Code failed to compile.\""
            }

        elif -1 in test_case_performance:
            idx = test_case_performance.index(-1)
            input = inputs[idx]
            output = outputs[idx]

            return {
                "prompts": None,
                "response": f"\"Runtime error.\n\nInput:\n{input}\nExpected Output:\n{output}\""
            }
        else:
            idx = random.choice(range(len(inputs)))
            input = inputs[idx]
            output = outputs[idx]

            return {
                "prompts": None,
                "response": f"\"Test case failed.\n\nInput:\n{input}\nExpected Output:\n{output}\""
            }

@dataclass
class UserModel(ModelRole):
    """
    Class for the simulated user.

    Methods
    ----------
    answer_query(prompt_args: Dict[str, str]) -> Dict:
        Generates answer to asst model query in the form of feedback_type.
        Also returns prompts used to generate model query
    """
    
    def answer_query(self, prompt_args) -> Dict:
        
        # logging.warning("Using a simplified version of user prompts... no solution_info.")
        feedback_type = prompt_args["feedback_type"]
        feedback_setting = prompt_args["feedback_setting"]

        prompts = get_prompts(
            config=self.prompt_config,
            prompt_type=f"user",
            prompt_args=prompt_args,
        )

        # logging.info("\n\n")
        # logging.info("USER ANSWERING QUERY...")
        # logging.info(f"########################") 
        logging.info(f"\n=========BEGIN USER PROMPT=========\n{prompts.query_prompt}\n=========END USER PROMPT=========\n\n")

        answer = self.model.generate(prompts, n_samples=1)[0]
        logging.info("\n=========BEGIN USER ANSWER=========\n\n" + answer + "\n=========END USER ANSWER=========\n\n")
        
        answer = answer.replace("```", "").rstrip()

        return {
            "prompts": prompts.__dict__,
            "response": answer
        }

@dataclass
class AssistantModel(ModelRole):
    """
    Class for the code assistant.

    Methods
    ----------
    check_if_solution_finished(self, code: str) -> bool:
        Returns true if solution is finished.
    
    check_if_query_needed(self, code: str) -> bool:
        Returns true if user guidance is needed.

    generate_code(prompt_args: Dict, stop_condition: StoppingCondition) -> Dict, bool:
        Generates code. It only stops when either 
            1) the solution is finished  
            2) the assistant needs a question from the user model.

        Returns the code and a boolean indicating if the solution is finished.

        If bool is False, the code is not finished and the assistant needs
        a question from the user model.

    query_user(self, prompt_args: Dict) -> Dict:   
        Generates query to user model.
    """

    def check_if_query_needed(self, code: str) -> bool:
        # logging.warning("Always returns True for now!")
        return True

    def check_if_solution_finished(self, code: str) -> bool:
        """ We use markdown tags to signal the end of code. """
        
        # TODO: If you prefill does the output also have \'\'\' twice or once?
        # If twice, we'll just look for that twice. this version is pretty brittle.
        # logging.warning("Assumes that code solutions contains ```")
        return  "```" in code


    def get_prompt_type(
        self,
        ir_idx: int
    ):
        """ Figure out which prompt to use based on the feedback type and setting. """

        if ir_idx == 0:
            return "asst_initial_code"

        else:
            return "asst_code"

    def generate_code(
            self, 
            prompt_args: Dict, 
            stopping_condition: Config,
            ir_idx: int,
        ) -> str:

        # Get current feedback setting
        asst_prompt_type = self.get_prompt_type(ir_idx)

        # Keep track of code generated so far.
        full_code = prompt_args["partial_solution"]
        old_code = None
        response_data = []
        stuck_counter = 0
        i = 0
        
        logging.info(f"\n\n")
        logging.info(f"Beginning code: \n {full_code} \n-------------------\n")
        logging.info(f"########################")
        
        while True:
            logging.info(f"\n")
            logging.info("ASST GENERATING CODE...")
            logging.info(f"TURN {i}...")

            # Generate code
            prompts = get_prompts(
                config=self.prompt_config,
                prompt_type=asst_prompt_type,
                prompt_args=prompt_args,
            )
            logging.info(f"===========CODE PROMPT===========\n{prompts.query_prompt}\n=========END CODE PROMPT=========\n\n")
                
            code = self.model.generate(
                prompts, 
                n_samples=1, 
                stopping_condition=stopping_condition,
            )[0].rstrip()

            response_data.append({
                "prompts": prompts.__dict__,
                "response": code,
            })

            full_code = add_code_to_context(full_code, code)

            # Check to see if solution is finished
            if self.check_if_solution_finished(full_code):
    
                logging.info(f"=========BEGIN FINAL CODE=========\n{full_code}\n=========END FINAL CODE=========\n\n")
                logging.info(f"Solution finished.")

                return {
                    "code": full_code,
                    "is_finished": True,
                    "response_data": response_data
                }
                        
            # Check to see if generation should stop
            if stopping_condition.stopping_condition == "always_stop" or \
               stopping_condition.stopping_condition == "ask_asst" and \
               self.check_if_query_needed(full_code):
               
                logging.info(f"=========BEGIN INCOMPLETE CODE=========\n{full_code}\n=========END INCOMPLETE CODE=========\n\n")
                logging.info(f"Solution incomplete. Requesting user query.")

                return {
                    "code": full_code,
                    "is_finished": False,
                    "response_data": response_data
                }

            # If solution incomplete and no query needed, repeat
            logging.info(f"==========CURRENT CODE=========\n{full_code}\n=========END CURRENT CODE=========\n\n")
            logging.info(f"Generating next section of code.")
            prompt_args["partial_solution"] = full_code

            if old_code is not None and old_code.strip() == full_code.strip():
                stuck_counter += 1
                logging.warning(f"Code generation is stuck in a loop! Stuck counter: {stuck_counter}")
                # import pdb; pdb.set_trace()
            
            if stuck_counter == 10:
                logging.error("---- STUCK MAX TIMES ---- SKIPPING QUESTION")
                return {"code": "", "is_finished": True, "response_data": None}
            
            old_code = full_code
            i +=1 


    def query_user(
        self, 
        prompt_args: Dict,
        feedback_type: str,
        feedback_setting: str,
    ) -> Dict:
        
        # If feedback type does not require model query
        if feedback_type in ["comment", "code", "code_or_comment", "nl_feedback", "test_case", "input_refinement"]:
            return {
                "prompts": None,
                "response": None
            }
        
        # If feedback type requires model query
        else:
            prompts = get_prompts(
                config=self.prompt_config,
                prompt_type=f"asst_question",
                prompt_args=prompt_args,
            )
            
            logging.info(f"\n\n")
            logging.info("ASST QUERYING USER...")
            logging.info(f"########################")  
            logging.info(f"\n=========BEGIN QUERY PROMPT=========\n\n{prompts.query_prompt}\n=========END QUERY PROMPT=========\n\n")
            query = self.model.generate(prompts, n_samples=1)[0]

            logging.info("\n=========BEGIN QUERY=========\n\nQuery: \n" + query + "\n=========END QUERY=========\n\n")
            return {
                "prompts": prompts.__dict__,
                "response": query
            }


if __name__ == "__main__":
    file = "configs/interaction.yaml"
    x = parse_config(file)
    logging.basicConfig(level="INFO")

    dataset = Dataset(x.dataset)
    example = dataset.examples[0]

    prompt_args = {
        "partial_solution": "# Beginning on code",
        "underspec_question": example["input"],
        "full_question": example["orig_input"],
        "asst_query": None,
        "feedback_type": "free_response_answer",
        "feedback_setting": "mid_generation",
    }
    
    
    # types of feedback: comment, free_response_answer
    # Test assistant model
    asst = AssistantModel(x.asst_model)
    asst_code_output = asst.generate_code(prompt_args, x.stopping_condition)
    
    prompt_args["partial_solution"] = asst_code_output["code"]

    asst_query_output = asst.query_user(prompt_args, "free_response_answer", "mid_generation")
    prompt_args["asst_query"] = asst_query_output["response"]

    # Test user model
    user = UserModel(x.user_model)
    user_answer_output = user.answer_query(
        prompt_args
    )

    import pdb; pdb.set_trace()



