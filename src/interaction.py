from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from datasets import load_dataset
import numpy as np
import argparse
import os
import logging
from collections import deque

from src.model_role import *
from src.utils import read_json
from src.config import *
from src.dataset import *

from datasets import load_dataset # GOTTA FIND MORE EFFICIENT SOLUTION

@dataclass
class Interaction():
    """
    This class handles the interaction between the user and assistant models.
    Interactions are series of rounds. See run_round() documentation.
    By the end of the interaction, a full code solution has been formed.

    Attributes
    ----------
        user (UserModel): Simulated user model.
        asst (AssistantModel): Code assistant model.
        stop_condition (StoppingCondition): Stopping condition for code assistant model

    """
    user: UserModel
    asst: AssistantModel
    config: InteractionConfig
    output_dir: str
    problem = None
     
    def get_user_feedback(self, prompt_args: Dict[str, str]) -> Dict[str, str]:
        """
        This function generates feedback for the user after some amount of code has been generated.
        It is called after the code has been generated.
        it also updates the prompt_args dictionary with the feedback.

        Args:
            prompt_args (Dict[str, str]): Dictionary containing all necessary information
                                          for prompt building.

        Returns:
            A dict containing the feedback generated for the user.

        """
        asst_query_output = self.asst.query_user(
            prompt_args,
            feedback_setting=prompt_args["feedback_setting"],
            feedback_type=prompt_args["feedback_type"],
        )
        prompt_args["asst_query"] = asst_query_output["response"]
        user_answer_output = self.user.answer_query(prompt_args)

        return prompt_args, asst_query_output, user_answer_output
    
    
    def run_ir_feedback(self, prompt_args: Dict[str, str]) -> Dict[str, str]:
        """
        After a full code solution is generated, this function returns human feedback.
        First, the assistant generates a query for the user model (if needed).
        Then, the user model generates a response to the query.

        Args:
            prompt_args (Dict[str, str]): Dictionary containing all necessary information
                                            for prompt building.

        Returns:
            A dict containing the results from 1 -> 4 above.

        """
        results = {}

        if prompt_args["feedback_type"] is not None:
            prompt_args, asst_query_output, user_answer_output = self.get_user_feedback(
                prompt_args,
            )

            results["asst_ir_query"] = asst_query_output["response"]
            results["user_ir_answer"] = user_answer_output["response"]
            results["asst_ir_query_full_output"] = asst_query_output
            results["user_ir_answer_full_output"] = user_answer_output

        else:
            results["asst_ir_query"] = None
            results["user_ir_answer"] = None
            results["asst_ir_query_full_output"] = None
            results["user_ir_answer_full_output"] = None
        
        return results


    def run_round(
        self,
        example,
        prompt_args: Dict[str, str],
        ir_idx: int,
    ) -> Dict[str, str]:
        """
        This function performs one round of the interaction. Each round consists of:
        1) Context is fed to code model
        2) Code model outputs code
        3) Code model queries user model (opt.)
        4) User model returns an answer to query (opt.)

        Args:
            prompt_args (Dict[str, str]): Dictionary containing all necessary information
                                          for prompt building.

        Returns:
            A dict containing the results from 1 -> 4 above.

        """
    
        # 0. Initialize results dict
        code = prompt_args["partial_solution"]

        results = {
            "prompt_args": prompt_args
        }


        # 1. Start generating code
        asst_code_output = self.asst.generate_code(prompt_args, self.config, ir_idx)
        prompt_args["partial_solution"] = asst_code_output["code"]

        # 2. Query user for guidance (unless code is complete)
        asst_query_output = {
            "response": None,
            "prompts": None,
        }
        user_answer_output = {
            "response": None,
            "prompts": None,
        }

        if not asst_code_output["is_finished"] \
           and self.config.feedback_setting == "mid_generation":
            prompt_args, asst_query_output, user_answer_output = self.get_user_feedback(
                prompt_args,
            )

        #4. Return results
        results["asst_code"] = asst_code_output["code"]
        results["asst_mg_query"] = asst_query_output["response"]
        results["user_mg_answer"] = user_answer_output["response"]

        results["asst_mg_code_full_output"] = asst_code_output
        results["asst_mg_query_full_output"] = asst_query_output
        results["user_mg_answer_full_output"] = user_answer_output

        results["is_finished"] = asst_code_output["is_finished"]

        # Do i need to return prompt args here?
        return results


    def qid_logger(self, output_dir, qid, sample_idx):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # make sure path exists
        log_dir = Path(f"{output_dir}", "raw_output", f"{qid}", "logs")
        if not log_dir.exists():
            log_dir.mkdir(parents=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: - %(message)s',
            handlers=[
                logging.FileHandler(Path(log_dir, f"{sample_idx}.log"), mode="w"),
                logging.StreamHandler(),
            ]
        )
    
    def run_rounds(
        self,
        example,
        mid_gen_prompt_args: Dict[str, str],
        ir_idx: int,
        sample_idx,
    ):
        is_finished = False
        round_idx = 0
        while not is_finished:
            logging.info(f"\n")
            logging.info(f"#- - - - - - - - - - - - - - - -#")
            logging.info(f"MID_GENERATION ROUND {round_idx}...")
            logging.info(f"#- - - - - - - - - - - - - - - -#")
            
            # 1. Run one round
            round_results = self.run_round(example, mid_gen_prompt_args, ir_idx)
            
            # 2. Add code + user info to previous context.
            is_finished = round_results["is_finished"]
            code = round_results["asst_code"]

            # 3. If not completed, prepare for the next round
            if not is_finished:
                # TODO: this is unclear since it's only used by comment and code.
                # For other kinds of user feedback, they are just passed into the
                # prompt via prompt args.
                code = add_user_info_to_context(
                    code,
                    round_results["user_mg_answer"],
                    self.config.feedback_type,
                )

                mid_gen_prompt_args["partial_solution"] = code
                mid_gen_prompt_args["user_response"] = round_results["user_mg_answer"]

            else:
                code = code.replace("```", "")
                
            # 4. Save results of round               
            save_results(
                self.output_dir, 
                example,
                round_results,
                sample_idx,
                is_round=True, 
                ir_idx=ir_idx,
                round_idx=round_idx,

            )
            round_idx += 1
        
        return code, round_idx

    def initialize_interaction(
        self, 
        example: Dict, 
        sample_idx,
        comparison_output_dir: str,
        ir_idx,
    ) -> Tuple[str, str, str]:
        
        ir_idx -= 1

        if comparison_output_dir is None:
            user_response = None
            asst_query = None
            prev_solution = None

        else:
            comparison_output_fn = os.path.join(
                    comparison_output_dir, 
                    "raw_output",
                    str(example["qid"]),
                    "generation_data",
                    f"sample_{sample_idx}",
                    "iter_refinement_data.json"
                )
            
            comparison_data = read_json(comparison_output_fn)
            logging.warning(f"#==================================#")
            logging.warning("SAMPLE {}".format(sample_idx))
            logging.warning(f"#==================================#")
        
            user_response = comparison_data["iter_refinement_data"][str(ir_idx)]["user_ir_answer"]
            user_response = "\n".join(user_response.strip().split("\n")[1:])
            asst_query = comparison_data["iter_refinement_data"][str(ir_idx)]["asst_ir_query"]
            prev_solution = comparison_data[str(ir_idx)]["final_code"]

        return user_response, asst_query, prev_solution

    def get_starter_code(self, example):
        logging.warning("Assuming that example is using ground truth solution as solution info!")

        lines = example["solution_info"].split("\n")
        return "\n".join(lines[:int(len(lines)*self.config.starter_code)])

    def run_interaction(
        self,
        example: Dict, 
        output_dir: str,
    ) -> Dict[str, str]:
        """
        This function handles the whole interaction between the user and assistant models.
        The input is the original code question. By the end of the interaction,
        a complete code solution should have formed.

        Args:
            context (str): Original context to feed into model.

        Returns:
            A dict containing the results from 1 -> 4 above.
        """
        
        if self.config.comparison_output_dir is not None:
            assert self.config.n_iter_refinement_steps == 2

        ir_results = {}
        ir_results["iter_refinement_data"] = {}

        for sample_idx in range(self.config.n_samples):
            
            # initialize history
            # import pdb; pdb.set_trace()
            if self.user is not None:
                self.user.reset_messages()
            self.asst.reset_messages()
        
            self.qid_logger(output_dir, example['qid'], sample_idx)

            logging.warning(f"#==================================#")
            logging.warning("SAMPLE {}".format(sample_idx))
            logging.warning(f"#==================================#")
        
            # Initialize solution, asst query, and user info
            user_response, asst_query, prev_solution = self.initialize_interaction(
                example,
                sample_idx,
                self.config.comparison_output_dir,
                self.config.starting_ir_idx,
            )

            # ir_history = deque()
            # max_ir_history = self.config.history_window 

            for ir_idx in range(self.config.starting_ir_idx, self.config.n_iter_refinement_steps):

                logging.info(f"\n\n\n")
                logging.info(f"#---------------------------#")
                logging.info("ITERATIVE REFINEMENT BEGINNING...")
                logging.info(f"STEP {ir_idx}...")
                logging.info(f"#---------------------------#")
                

                #0. Initialize code, + info used for assistant prompt
                if self.config.starter_code is None:
                    code = ""
                else:
                    code = self.get_starter_code(example)

                ir_results[ir_idx] = {}
            
                logging.warning(f"Using first example solution as user solution data")

                # Adjust input refinement if necessary
                underspec_question = example["input"]
                if self.config.feedback_type == "input_refinement" and ir_idx > 0:
                    underspec_question = user_response
                
                if example['task'] == "livecodebench" and example["starter"] != "":
                    starter = f"Here is the starter code that you must continue:\n\n```python\n{example['starter']}\n\n```"
                else:
                    starter = ""

                mid_gen_prompt_args = {
                    "partial_solution": code,
                    "underspec_question": underspec_question,
                    "full_question": example["orig_input"],
                    "solution_info": example["solution_info"],
                    "feedback_type": self.config.feedback_type,
                    "feedback_setting": self.config.feedback_setting,
                    "asst_query": asst_query,
                    "user_response": user_response,
                    "prev_solution": prev_solution,
                    "task": example["task"],
                    "starter": starter,
                }   

                # 1. Do mid-generation rounds until solution is complete      example,
                code, round_idx = self.run_rounds(example, mid_gen_prompt_args, ir_idx, sample_idx)

                # 2. When solution is complete, save final code
                ir_results[ir_idx]["final_code"] = code
                ir_results["total_rounds"] = round_idx + 1

                # 3. If there are more iterative refinement steps, get human feedback for next step
                if ir_idx < self.config.n_iter_refinement_steps - 1:

                    ir_prompt_args = {
                        "full_solution": code,
                        "underspec_question": underspec_question,
                        "full_question": example["orig_input"],
                        "solution_info": example["solution_info"],
                        "feedback_type": self.config.feedback_type,
                        "feedback_setting": "iter_refinement",
                        "asst_query": None,
                        "user_response": None,
                        "task": example["task"],  
                    }

                    ir_step_results = self.run_ir_feedback(ir_prompt_args)
                    ir_results["iter_refinement_data"][ir_idx] = ir_step_results
                    asst_query = ir_step_results["asst_ir_query"]
                    user_response = ir_step_results["user_ir_answer"]
                    prev_solution = code

                    
            # At the end of iterative refinement, save final code and context
            ir_results["final_code"] = code
            ir_results["total_ir_steps"] = ir_idx + 1
            
        
            
            save_results(
                self.output_dir, 
                example,
                ir_results,
                sample_idx,
            )

if __name__ == "__main__":
    file = "configs/interaction.yaml"
    x = parse_config(file)
    logging.basicConfig(
        level="INFO",
    )

    dataset = Dataset(x.dataset)
    example = dataset.examples[0]

    prompt_args = {
        "prev_context": "",
        "partial_solution": "",
        "underspec_question": example["input"],
        "full_question": example["orig_input"],
        "asst_query": None,
        "feedback_type": "free_response_answer",
        "feedback_setting": "mid_generation",
        "dataset": example["task"],
    }
    
    interaction = Interaction(
        user=UserModel(x.user_model),
        asst=AssistantModel(x.asst_model),
        config=x.interaction,
        output_dir=x.output_dir,
    )

    # out = interaction.get_user_feedback(prompt_args)
    round_out = interaction.run_interaction(
        example)

    # import pdb; pdb.set_trace()
