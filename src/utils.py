import json
import os
from pathlib import Path
import logging
import shutil
import numpy as np
import argparse



def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()
    if isinstance(object, np.ndarray):
        return object.tolist()


def read_jsonl(jsonl_fn: str):
    """Basic utility for reading jsonl files."""

    with open(jsonl_fn, 'r') as jsonl_file:
        json_list = list(jsonl_file)

    for i, json_str in enumerate(json_list):
        yield json.loads(json_str)


def read_json(json_fn: str):
    """Basic utility for reading json files."""

    with open(json_fn, 'r') as json_file:
        json_str = json_file.read()
    
    return json.loads(json_str)


def add_code_to_context(context: str, code: str) -> str:
    """Appends newly generated code to previous context."""

    return context + code


def add_comment_to_context(context: str, comment: str) -> str:
    """Appends user comment to previous context. Assumes that 
    the comment includes the correct indentation level."""

    return context + "\n" + comment


def add_user_info_to_context(context: str, user_info: str, user_info_type: str) -> str:
    """Adds user information to previous context."""

    if user_info_type == "code":
        return add_code_to_context(context, user_info)

    elif user_info_type == "comment":
        return add_comment_to_context(context, user_info)

    else:
        return context


def get_path_to_code_uncertainty() -> Path:
    def go_up_to_code_uncertainty(p: Path) -> Path:
        if p.name == "":
            raise ValueError("Cannot find code-uncertainty directory")

        if p.name == "code-uncertainty":
            return p
        else:
            return go_up_to_code_uncertainty(p.parent)
    
    p = Path.cwd()
    subdir = list(p.glob("**/code-uncertainty")) # check if code-uncert is subdir
    return go_up_to_code_uncertainty(p) if len(subdir) == 0 else subdir[0]

def save_summary_file(
    base_results_dir: Path,
    example: dict,
    n_samples: int,
):
    
    # Recover final code output
    code_outputs = []
    for sample_idx in range(n_samples):
        results_dir = os.path.join(
            base_results_dir, 
            f"raw_output/{example['qid']}",
        )

        results_subdir = os.path.join(
            results_dir,
            f"generation_data/sample_{sample_idx}",
        )

        results_fn = os.path.join(
            results_subdir,
            "iter_refinement_data.json",
        )

        results = read_json(results_fn)
        code_outputs.append(results["final_code"])

    
    # Save summary file
    summary_fn = os.path.join(results_dir, f"final_output.json")
    logging.info(f"Saving summary to %s", results_fn)

    try: 
        # additional fields needed for classeval
        final_output = {
            "qid": example["qid"],
            "code": code_outputs,
            "test_cases": example["test_cases"],
            "import_statement": example['import_statement'],
            "test_classes": example['test_classes']
        }
    except:
        final_output = {
            "qid": example["qid"],
            "code": code_outputs,
            "test_cases": example["test_cases"],
        }

    
    with open(summary_fn, "w") as f:
        json.dump(final_output, f)

def is_example_completed(
    base_results_dir: Path,
    example: dict,
):
    
    results_fn = os.path.join(
        base_results_dir, 
        f"raw_output/{example['qid']}",
        "final_output.json",
    )
        
    return os.path.exists(results_fn)

def save_config(
    base_results_dir: Path,
    config_fn: Path,
    config_type: str = None,
):
    """
    Save copy of config to file_directory.
    Also checks to see if files already exist.
    """
    
    if not os.path.exists(base_results_dir):
        os.makedirs(base_results_dir)

    results_fn = os.path.join(
        base_results_dir, 
        f"{config_type}_config.json",
    )

    logging.info(f"Copying {config_type} config to %s", results_fn)
    shutil.copyfile(config_fn, results_fn)

def save_results(
    base_results_dir: Path, 
    example: dict,
    results: dict, 
    sample_idx: int,
    is_round: bool = False, 
    ir_idx: int = None,
    round_idx: int = None,
    is_config: bool = False,
    config_type: str = None,
):
    """
    Save results to a file.

    Args:
        base_results_dir (Dict[str, str]): Path to the results directory.
        example (dict): Example for results
        results (dict): Results to save.
        is_round (bool): Whether the results are for a round.
        ir_idx (int): The index of the iterative refinement step.
        round_idx (int): For round data, the index of the round. 

    Returns:
        A dict containing the feedback generated for the user.
    
    """
    # TODO: Add check for whether results are already saved
    
    qid = example["qid"]

    # get round data filename
    if is_round:
        results_dir = os.path.join(
            base_results_dir, 
            f"raw_output/{qid}/generation_data/"
            f"sample_{sample_idx}/"
            f"iter_{ir_idx}",
        )
        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        results_fn = os.path.join(results_dir, f"round_{round_idx}.json")

    # get iterative refinement data directory
    else:
        results_dir = os.path.join(
            base_results_dir, 
            f"raw_output/{qid}/generation_data/sample_{sample_idx}",)

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        results_fn = os.path.join(results_dir, f"iter_refinement_data.json")

    logging.info("Saving results to %s", results_fn)
    with open(results_fn, "w") as f:
        json.dump(results, f)
