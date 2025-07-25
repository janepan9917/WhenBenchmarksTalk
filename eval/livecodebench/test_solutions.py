"""
Run solutions for all problems
"""
import argparse
import json
import numpy as np
import os
import pandas as pd
import pprint
import multiprocessing
import time
import eval.livecodebench.testing_util as test_util
import logging
import pickle
import zlib
import base64

# for timing debugging
from datetime import datetime, date
from tqdm import tqdm

from eval.livecodebench.compute_code_generation_metrics import check_correctness

from datasets import load_dataset
from types import SimpleNamespace
from typing import Dict
from pathlib import Path


EXAMPLE_RESULTS = {"0": [[-2]],"1": [[False,False,False]],"2": [[True,True]],"3": [[False,True,False,True,False,False,False,True,False,True,False,True,True,True,False,True]],"4": [[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]]}
EXAMPLE_ARGS = SimpleNamespace(debug=True)
TIMEOUT = 10

import ast
import re
def extract_python_code(input_str):
    # Remove possible Markdown code fences
    input_str = re.sub(r'^```python\s*|```$', '', input_str, flags=re.MULTILINE)

    # Split input into lines
    lines = input_str.split("\n")

    # Identify longest contiguous valid Python block
    best_code = []
    current_block = []

    for line in lines:
        current_block.append(line)
        try:
            # Validate if the current block is a valid Python program
            ast.parse("\n".join(current_block))
            best_code = list(current_block)  # Save the longest valid block
        except SyntaxError:
            continue  # Ignore invalid parts

    return "\n".join(best_code).strip()



def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

def print_results(results: Dict, args: argparse.Namespace=None, setting: str=None, partial_sol_idx: str=None):
    """
    Given the results evaluated against the testcases we output some statistics.

    >>> print_results(EXAMPLE_RESULTS, EXAMPLE_ARGS)
    number of compile errors = 1 avg = 0.2
    number of runtime errors = 1 avg = 0.2
    number of test cases run = 5
    Test Case Average (average accuracy over problems) = 0.3
    Strict Accuracy (all test cases passed / total problems) = 0.2
    """
    all_results = []
    per_prob_res = []
    all_correct = []
    for index in results:
        # problem_results = np.asarray(results[index])
        problem_results = results[index][setting][partial_sol_idx]
        all_results.extend(problem_results)
        # per_prob_res.append(np.mean([1 if i > 0))
        # all_correct.append(np.all(all))

    # For each question:
    # We count both compile errors and runtime errors for multiple tests as one error.

    results_dict = {}
    for i, res in enumerate(all_results):
        compile_errors = len([e for e in res if -2 == e])
        runtime_errors = len([e for e in res if -1 == e])

        total_testcases = len(res)
        if args and args.debug:
            print(f"number of compile errors = {compile_errors} avg = {compile_errors / total_testcases }")
            print(f"number of runtime errors = {runtime_errors} avg = {runtime_errors / total_testcases}")
            print(f"number of test cases run = {total_testcases}")

        n_compile_errors = compile_errors
        avg_compile_errors = compile_errors / total_testcases

        n_runtime_errors = runtime_errors
        avg_runtime_errors = runtime_errors / total_testcases

        n_test_cases_run = total_testcases

        test_case_average = np.mean(np.asarray(res) > 0)
        strict_accuracy = np.all(np.asarray(res) > 0)

        results_dict[i] = {
            "had_compile_errors": n_compile_errors > 0,
            "avg_compile_errors": avg_compile_errors,
            "had_runtime_errors": n_runtime_errors > 0,
            "avg_runtime_errors": avg_runtime_errors,
            "test_cases_run": n_test_cases_run,
            "test_case_average": test_case_average,
            "strict_accuracy": strict_accuracy,
        }
        
        # print(f"Test Case Average (average accuracy over problems) = {np.mean(per_prob_res)}")
        # print(f"Strict Accuracy (all test cases passed / total problems) = {np.mean(all_correct)}")

    
    aggregate_results = {
        "had_compile_errors": np.sum([results_dict[i]["had_compile_errors"] for i in results_dict]),
        "avg_compile_errors": np.mean([results_dict[i]["avg_compile_errors"] for i in results_dict]),
        "had_runtime_errors": np.sum([results_dict[i]["had_runtime_errors"] for i in results_dict]),
        "avg_runtime_errors": np.mean([results_dict[i]["avg_runtime_errors"] for i in results_dict]),
        "test_cases_run": np.mean([results_dict[i]["test_cases_run"] for i in results_dict]),
        "test_case_average": np.mean([results_dict[i]["test_case_average"] for i in results_dict]),
        "strict_accuracy": np.mean([results_dict[i]["strict_accuracy"] for i in results_dict]),
        "individual_results": results_dict,
        "num_problems": len(results_dict)
    }

    return aggregate_results


def evaluate_generations(generations, problem):
    results = []

    for generation_idx, generation in enumerate(generations):
        curr_res = [-2]
        try:
            curr_res, metadata = check_correctness(sample=problem, generation=generation, timeout=TIMEOUT, debug=False)
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
            if not np.all(curr_res):
                print(f"Results were not all True: {curr_res}")
        except Exception as e:
            print(f"test framework exception = {repr(e)}{e}\n")
            break
        finally:
            assert isinstance(curr_res, list)
            results.append(curr_res)

    return results


def get_final_results(args, final_ir_data, results_folder):

    # Get location of code
    codes_loc = os.path.join(
        results_folder,
        "final_output.json"
    )

    # Read code 
    with open(codes_loc, "r") as json_file: 
        data = json.loads(json_file.read())
        generations = [c.replace("```", "") for c in data["code"]]
        # assert args.qid == data["qid"]

    # Get results fn
    results_loc_name = os.path.join(results_folder, f"test_case_results.json") 

    # Only do the problems that are specified.
    problem_id = data["qid"]
    # assert args.qid == problem_id

    # problem = load_dataset("codeparrot/apps")["test"].select(range(problem_id, problem_id+1))[0]
    problem = load_dataset("livecodebench/code_generation_lite", 
                        version_tag="release_v4", 
                        split='test', 
                        trust_remote_code=True).filter(lambda x: x['question_id'] == problem_id)[0]
    pub_tests = problem['public_test_cases']
    pub_tests = json.loads(pub_tests)
    priv_tests = problem['private_test_cases']

    try:
        priv_tests = json.loads(priv_tests)  # type: ignore
    except:
        priv_tests = json.loads(
            pickle.loads(
                zlib.decompress(
                    base64.b64decode(priv_tests.encode("utf-8"))  # type: ignore
                )
            )
        )  # type: ignore

    problem['input_output'] = json.dumps(
            {
                "inputs": [
                    t['input']
                    for t in pub_tests + priv_tests 
                    ],
                "outputs": [
                    t['output']
                    for t in pub_tests + priv_tests 
                    ],
                "fn_name": json.loads(problem['metadata']).get("func_name", None),
                }
            )
    # problem["solutions"] = json.loads(problem["solutions"])
    # problem["input_output"] = json.loads(problem["input_output"])
    
    logging.warning("Assumes that first solution data is being used...")

    # # Evaluate the final generations for main function
    # results = evaluate_generations(generations, problem)

        
    results = {}
    for ir_idx, ir_data in final_ir_data.items():
        res = evaluate_generations(ir_data, problem)
        results[ir_idx] = res

    if args.debug:
        print(f"\nHow to read results [-2] = compile error, [-1] = runtime error, [False] = failed test case, [True] = passed test case")
        #print(f"results = {res}")

    # Save results 
    with open(results_loc_name, "w") as f:
        try:
            f.write(json.dumps(results))
        except Exception as e:
            # import pdb; pdb.set_trace()
            print("didn't save problem due to {e}")
    return 


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # get location to save results of running codce
    # results_folder = os.path.join(args.output_dir, "raw_output", str(args.qid))
    base_path = Path(args.output_dir, "raw_output")
    all_results = base_path.glob("*")

    for results_folder in all_results:
        # Make iter-refinement data
        all_data = []
        for sample in range(args.n_samples):
            sample_ir_data = {}
            iter_refinement_fn = os.path.join(results_folder, f"generation_data/sample_{sample}/iter_refinement_data.json")
            iter_refinement_data = json.load(open(iter_refinement_fn, "r"))
            n_steps = iter_refinement_data["total_ir_steps"]


            for step in range(n_steps):
                if str(step) in iter_refinement_data.keys():
                    code = iter_refinement_data[str(step)]["final_code"]
                    code = extract_python_code(code)

                    # sometimes the first line of the code includes the word "python" so we're just gonna remove it here
                    if "python" in code.split("\n")[0]:
                        code = "\n".join(code.split("\n")[1:])
                    
                    sample_ir_data[step] = code
        
            if len(sample_ir_data) == 0:
                raise ValueError("No data found for sample")

            all_data.append(sample_ir_data)

        final_ir_data = {i: [] for i in range(n_steps)}
        for sample_data in all_data:
            for step in sample_data.keys():
                final_ir_data[step].append(sample_data[step])

        get_final_results(
            args,
            final_ir_data,
            results_folder=results_folder,
        )  


if __name__ == "__main__":
    # import doctest
    # doctest.testmod()

    parser = argparse.ArgumentParser(description="Testing a Language Model on Python Code")
    # parser.add_argument("--qid", type=str, help="Question ID to evaluate.")
    parser.add_argument("--n_samples", type=int, help="n_samples.") #TODO: make this automatic
    
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--output_dir", type=str, help="Where the evaluated data is loaded from and results saved to.")

    
    args = parser.parse_args()

    main(args)
