import argparse
import openai
import os
import json
import yaml
import logging
from tqdm import tqdm

from src.model import *
from src.dataset import *
from src.model_role import UserModel, AssistantModel
from src.interaction import *
from src.config import *
from src.utils import save_config

# def get_results_dir(args) -> str:
#     """
#     Gets the results directory. 
#     Comments, raw data, and analysis are all stored here.
#     """

#     fn = "output/"

#     if args.dataset == "apps_oracle":
#         fn += f"dataset_apps_code_completion_w_oracle"
#     else:
#         fn += f"dataset_{args.dataset}"

#     fn += f"-model_{args.model}"
#     fn += f"-n_examples_{args.n_examples}"
#     fn += f"-n_samples_{args.n_samples}"
#     fn += f"-split_type_{args.split_type}"
    
#     return fn


def _main(args):
    # 0. Set up some utilities
    np.random.seed(0)


    # 1. Load config
    config = parse_config(args.config_fn)
    config.data_dir = args.data_dir
    
    #2. Load dataset and model
    dataset = Dataset(config.dataset, args.data_dir, n_examples=args.n_examples)
    asst = AssistantModel(config.asst_model)   
    user = None
    if config.interaction.feedback_type == "test_case":
        user = APPSOnlineCodeEval(None)
    elif config.user_model is not None:
        user = UserModel(config.user_model)
    
    # 3. Get results_dir
    output_dir = args.output_dir
    save_config(
        output_dir,
        args.config_fn,
        "interaction",
    )
    
    # 4. Set up logging
    # logging.root.setLevel(logging.INFO)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: - %(message)s',
        handlers=[
            # logging.FileHandler(Path(log_dir, "non-examples.log"), mode="w"),
            logging.StreamHandler(),
        ]
    )

    interaction = Interaction(
        user=user,
        asst=asst,
        config=config.interaction,
        output_dir=output_dir,
    )

    for example in dataset.examples:
        qid = example["qid"]

        
        if args.qid is None or args.qid is not None and qid == args.qid:
            # Check to see output already exists
            if args.overwrite_old_output or \
                not is_example_completed(output_dir,
                    example):
                
                # Delete old output
                # TODO:

                logging.info(f"Now on example {qid}")
                
                if config.interaction.feedback_type == "test_case":
                    interaction.user.get_problem(qid)

                
                results = interaction.run_interaction(
                    example,
                    output_dir
                )
                
                save_summary_file(
                    output_dir,
                    example,
                    n_samples=config.interaction.n_samples,
                )
            else:
                logging.info(f"Skipping example {qid}, results already found")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str,
                        help="Path to config file")
    parser.add_argument("--config_fn", type=str, default="configs/interaction.yaml",
                        help="Path to config file")
    parser.add_argument("--model_config_fn", type=str, default=None)

    parser.add_argument("--log", type=str, default="INFO",
                        help="Logging config level: WARNING, INFO, DEBUG.")

    parser.add_argument("--qid", type=int, default=None,
                        help="Mostly for debugging -- only do a single QID.")


    # Dataset arguments
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Name of dataset.")
    parser.add_argument("--n_examples", type=int, default=None,
                        help="Number of examples.")
    
    parser.add_argument('-v', '--overwrite_old_output',
                    action='store_true')


    args = parser.parse_args()
    return _main(args)

if __name__ == "__main__":
    main()
