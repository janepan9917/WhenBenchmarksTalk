"""
Run solutions from one problem.
"""
import argparse
import json
import numpy as np
import os
import pandas as pd
import pprint
import multiprocessing
import time
import testing_util as test_util
import logging

# for timing debugging
from datetime import datetime, date
from tqdm import tqdm

from datasets import load_dataset
from types import SimpleNamespace
from typing import Dict


EXAMPLE_RESULTS = {"0": [[-2]],"1": [[False,False,False]],"2": [[True,True]],"3": [[False,True,False,True,False,False,False,True,False,True,False,True,True,True,False,True]],"4": [[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]]}
EXAMPLE_ARGS = SimpleNamespace(debug=True)
TIMEOUT = 10

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

# Dummy `test_util.run_test` function for debugging multiprocessing.
def run_test(problem, test, debug):
    time.sleep(1)  # Simulate some work
    return [1]  # Dummy test result

def check_correctness(problem, generation, timeout, debug):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    def _temp_run(problem, generation, debug, result):
        try:
            if debug:
                print(f"Running test for problem: {problem}")
            result.append(test_util.run_test(problem=problem, test=generation, debug=debug))
            # Useful for debugging the multiprocessing.
            # result.append(run_test(problem=problem, test=generation, debug=debug))
            if debug:
                print(f"Test completed with result: {result}")
        except Exception as e:
            if debug:
                print(f"Error in _temp_run: {e}")

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(problem, generation, debug, result))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        if debug:
            print(f"Process is still alive. Killing the process.")
        p.kill()
    if not result:
        # Remark: ideally we would consider that all tests failed but we can't access number of tests here easily
        # so we use 21=the average number of tests for a smaple in the test split instead 
        avg_number_tests = 21
        result = [[-1] * avg_number_tests]
        if debug:
            print(f"Global timeout occurred, returning default result.")
    if debug:
        print(f"Final result: {result}")
    return result[0]


def eval_and_save_problems(args, results_folder):
    codes = {}
    gpt_bleu = {}
    gpt_codebleu = {}
    
    # set up results dict
    results = {}  

    # main eval loop
    settings = args.settings.split(",")
    for setting in settings:

        # get location of code
        codes_loc = os.path.join(args.save, args.codes_dir, f"raw_data/{setting}/{args.index}.json")

        # get results fn
        results_loc_name = os.path.join(results_folder, f"{setting}")
        if not os.path.exists(results_loc_name):
            os.makedirs(results_loc_name)
        results_loc_name = os.path.join(results_loc_name, f"{args.index}.json")     

        with open(codes_loc, "r") as json_file: 
            codes = json.loads(json_file.read())

        # Only do the problems that are specified.
        problem_id = codes["qid"]
        assert args.index == problem_id

        problem = load_dataset("codeparrot/apps")["test"].select(range(problem_id, problem_id+1))[0]
        
        # This dict holds all settings + partial solution index results for one question.
        problem["solutions"] = json.loads(problem["solutions"])
        problem["input_output"] = json.loads(problem["input_output"])
    
        results["qid"] = problem_id

        for partial_sol_idx in range(args.n_parts):
        
            # prepare some solution data
            logging.warning("Assumes that first solution data is being used...")
            solution_data = codes["solutions_data"][0] 
            n_parts = solution_data["n_parts"]
            partial_solutions = solution_data[f"partial_solution_by_{args.split_type}"]
            model_completions = codes["responses"][str(partial_sol_idx)]["responses"]
            queries = codes["responses"][str(partial_sol_idx)]["query"]
            starter_code = codes["responses"][str(partial_sol_idx)]["starter_code"]

            # Prepare the code completions
            output_strings = []
            for code in model_completions:
                # Remove code markdown
                assert code.count("```") == 1
                code = code.split("```")[0]

                code = starter_code.replace("```python", "").strip() + code

                # Add partial solution if it is not in the completion
                # if type(list(partial_solutions)[0]) is str:
                #     partial_sol_idx = str(partial_sol_idx)

                if partial_solutions[partial_sol_idx] not in code:
                    if args.debug:
                        print(f"Partial solution not in completion, adding it now.")
                        print(f"Before: \n\n {code} \n ---------------")
                        import pdb; pdb.set_trace()

                    code = partial_solutions[partial_sol_idx] + "\n" + code
   
                if args.debug:
                    print(f"After cleaning: \n\n{code} \n ---------------")
                
                output_strings.append(code)

            
            sols = problem["solutions"]

            if not os.path.exists(args.save):
                os.makedirs(args.save)

            res = []
            for generation_idx, generation in enumerate(output_strings):
                if args.debug:
                    print(f"\nTesting solution {generation_idx}, {generation=}")
                curr_res = [-2]
                try:
                    curr_res = check_correctness(problem, generation=generation, timeout=TIMEOUT, debug=args.debug)
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
                    res.append(curr_res)

            results[partial_sol_idx] = res

        # Save results
        with open(results_loc_name, "w") as f:
            try:
                f.write(json.dumps(results))
            except Exception as e:
                import pdb; pdb.set_trace()
                print("didn't save problem due to {e}")
            
        if args.debug:
            print(f"\nHow to read results [-2] = compile error, [-1] = runtime error, [False] = failed test case, [True] = passed test case")
            #print(f"results = {res}")
        
    

    return 


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # get location to save results of running codce
    results_folder = os.path.join(args.save, args.codes_dir, "test_cases_results")

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    eval_and_save_problems(
        args,
        results_folder=results_folder,
    )  
        
       


if __name__ == "__main__":
    # import doctest
    # doctest.testmod()

    parser = argparse.ArgumentParser(description="Testing a Language Model on Python Code")
    parser.add_argument("-t","--test_loc", default="../data_split/test.json", type=str, help="path to the json containing problem paths to be evaluated.")
    parser.add_argument("-r","--root", default="../", type=str, help="where the data is stored.")
    parser.add_argument("-s","--start", default=0, type=int)
    parser.add_argument("-e","--end", default=None, type=int, help="If you want to evaluate a subset of problems specify start and ending index. File with start and ending prefix must exist typically used with batch evaluation.")
    parser.add_argument("-i", "--index", type=int, default=None)
    
    parser.add_argument("--skip_evals", action="store_true", help="If you want to skip the evals similar to print results.")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--save", type=str, default="./results", help="Where the evaluated data is loaded from and results saved to.")
    parser.add_argument("--split", type=str, default="test", help="What split to use.")
    parser.add_argument("--stop-early", default=None, type=int)
    
    # stages of evaluation
    parser.add_argument("--evaluate", action="store_true", help="If you want to evaluate the results.")
    # parser.add_argument("-p", "--print_results", action="store_true", help="If you have already evaluated the results and only want to print them.")

    # oracle experiment utilities
    parser.add_argument("--codes_dir", type=str, default=None, help="Folder where raw_data.jsonl is stored.")
    parser.add_argument("--n_parts", type=int, default=None, help="How many problems to evaluate.")
    parser.add_argument("--split_type", type=str, default=None, help="How to split the code: line, paragraph")
    parser.add_argument("--settings", type=str,default=None,  
                        help="Comma-split string of settings to evaluate. E.g. 'code_completions_w_comments,code_completions_wo_comments'")
    
    args = parser.parse_args()

    main(args)
