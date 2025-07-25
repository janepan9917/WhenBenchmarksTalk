import argparse
import json
import numpy as np
import os
import pandas as pd
import pprint
import multiprocessing
import time
import testing_util as test_util

# for timing debugging
from datetime import datetime, date
from tqdm import tqdm

from datasets import load_dataset
from types import SimpleNamespace
from typing import Dict

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()
    

def compute_stats(all_res):
    """ 
    Res is a list of n lists, one per generation.
    Each sublist = test cases for an individual generation
    """
    results_dict = {
        "had_compile_errors": [],
        "avg_compile_errors": [],
        "had_runtime_errors": [],
        "avg_runtime_errors": [],
        "test_cases_run": [],
        "test_case_average": [],
        "strict_accuracy": [],
    }
    
    for res in all_res:
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

        results_dict["had_compile_errors"].append(n_compile_errors > 0)
        results_dict["avg_compile_errors"].append(avg_compile_errors)
        results_dict["had_runtime_errors"].append(n_runtime_errors > 0)
        results_dict["avg_runtime_errors"].append(avg_runtime_errors)
        results_dict["test_cases_run"].append(n_test_cases_run)
        results_dict["test_case_average"].append(test_case_average)
        results_dict["strict_accuracy"].append(strict_accuracy)


    return results_dict


def get_agg_averages(results_dict: Dict):
    """ Take averages over a dictionary of problem-level aggregated results """

    for i in results_dict:
        if "test_case_average_ignoring_compile_errors" not in results_dict[i].keys():
            problem_test_case_averages = results_dict[i]["test_case_average"]
            problem_compile_errors = results_dict[i]["had_compile_errors"]
            results_dict[i]["test_case_average_ignoring_compile_errors"] = np.nanmean([problem_test_case_averages[i] for i in range(len(problem_test_case_averages)) if problem_compile_errors[i] == 0])

    aggregate_results = {
        "had_compile_errors": np.mean([results_dict[i]["had_compile_errors"] for i in results_dict]),
        "avg_compile_errors": np.mean([results_dict[i]["avg_compile_errors"] for i in results_dict]),
        "had_runtime_errors": np.mean([results_dict[i]["had_runtime_errors"] for i in results_dict]),
        "avg_runtime_errors": np.mean([results_dict[i]["avg_runtime_errors"] for i in results_dict]),
        "test_cases_run": np.mean([results_dict[i]["test_cases_run"] for i in results_dict]),
        "test_case_average": np.mean([results_dict[i]["test_case_average"] for i in results_dict]),
        "test_case_average_ignoring_compile_errors": np.nanmean([results_dict[i]["test_case_average_ignoring_compile_errors"] for i in results_dict]),
        "strict_accuracy": np.mean([results_dict[i]["strict_accuracy"] for i in results_dict]),
        "individual_results": results_dict,
        "num_problems": len(results_dict)
    }

    return aggregate_results

def compute_problem_level_agg_stats(
        all_results: Dict, 
        args: argparse.Namespace, 
    ):
    """
    Computes the problem level aggregated stats (i.e. aggregate the performance across n_generations)
    for each problem.
    """
    
    # For each question:
    # We count both compile errors and runtime errors for multiple tests as one error.

    results_dict = {}
    
    # for each problem
    for i, res in enumerate(all_results):

        problem_test_case_averages = []
        problem_strict_accuracies = []
        problem_compile_errors = []
        problem_had_compile_errors = 0
        problem_runtime_errors = []
        problem_had_runtime_errors = 0


        # for each rollout
        for j, rollout_res in enumerate(res):
            compile_errors = len([e for e in rollout_res if -2 == e])
            runtime_errors = len([e for e in rollout_res if -1 == e])

            total_testcases = len(rollout_res)
            if args and args.debug:
                print(f"number of compile errors = {compile_errors} avg = {compile_errors / total_testcases }")
                print(f"number of runtime errors = {runtime_errors} avg = {runtime_errors / total_testcases}")
                print(f"number of test cases run = {total_testcases}")

            n_compile_errors = compile_errors
            avg_compile_errors = 1 if compile_errors > 0 else 0

            n_runtime_errors = runtime_errors
            avg_runtime_errors = runtime_errors / total_testcases

            n_test_cases_run = total_testcases

            test_case_average = np.mean(np.asarray(rollout_res) > 0)
            strict_accuracy = np.all(np.asarray(rollout_res) > 0)

            problem_test_case_averages.append(test_case_average)
            problem_strict_accuracies.append(strict_accuracy)
            
            problem_runtime_errors.append(avg_runtime_errors)
            problem_compile_errors.append(avg_compile_errors)

            if n_compile_errors > 0:
                problem_had_compile_errors += 1

            if n_runtime_errors > 0:
                problem_had_runtime_errors += 1
            

        results_dict[i] = {
            "had_compile_errors": problem_had_compile_errors > 0,
            "avg_compile_errors": problem_had_compile_errors/len(res),
            "had_runtime_errors": problem_had_runtime_errors > 0,
            "avg_runtime_errors": np.mean(problem_runtime_errors),
            "test_cases_run": n_test_cases_run,
            "test_case_average": np.mean(problem_test_case_averages),
            "test_case_average_ignoring_compile_errors": np.mean([problem_test_case_averages[i] for i in range(len(problem_test_case_averages)) if problem_compile_errors[i] == 0]),
            "strict_accuracy": np.mean(problem_strict_accuracies),
        }

    return results_dict


def _get_summary_stats(
        results: Dict, 
        args: argparse.Namespace,
    ):
    """
    For a particular setting and partial solution index, get the summary stats on three levels:

    1) individual problem + individual generations
    2) individual problem, aggregated across n_generations
    3) all problems (averaging across )
    """
    all_results = [] # [n problems] x [n_generations] x [n_test_cases]
    problem_level_results = {} # [n problems] : [n_generations] x [n_test_cases]
    
    # collect results with this setting and partial_sol_idx
    for index in results:
        problem_results = results[index]
        problem_level_results[index] = problem_results
        all_results.append(problem_results)
    
    # get results per problem aggregated over n_generations
    problem_level_agg_stats = compute_problem_level_agg_stats(
        all_results, 
        args,
    )

    # get results aggregated over all problems
    agg_stats = get_agg_averages(problem_level_agg_stats)
    
    # get problem_level statistics with individual generations held separately
    problem_level_stats = {}
    for q_idx, res in problem_level_results.items():
        problem_level_stats[q_idx] = compute_stats(res)

    return problem_level_stats, problem_level_agg_stats, agg_stats

# def _get_filtered_agg_stats(stats_dict, indices, args):
#     agg_stats_dict = {}

#     for setting in args.settings.split(","):
#         agg_stats_dict[setting] = {}

#         for partial_sol_idx in range(args.n_parts):
#             partial_sol_idx = str(partial_sol_idx)  

#             agg_stats_dict[setting][partial_sol_idx] = {}
#             filtered_results = {}

#             for k in indices:
#                 filtered_results[k] = stats_dict[setting][partial_sol_idx][k]
                
#             # get agg stats
#             filtered_agg_stats = get_agg_averages(filtered_results)

#             agg_stats_dict[setting][partial_sol_idx] = filtered_agg_stats

#     return agg_stats_dict

# def get_filtered_agg_stats(all_stats_dict, args):
#     # filter questions to get stats of those that were doable by the model with baseline, 0

#     perfect_indices = [] # perfect performance 
#     good_indices = [] # decent performance
#     bad_indices = [] # bad performance

#     # get indices
#     for k, v in all_stats_dict["baseline"]['2'].items():
#         if 1 in v["test_case_average"]:
#             perfect_indices.append(k)
    
#         for tca in v["test_case_average"]:
#             if tca > 0.6:
#                 good_indices.append(k)
#                 break

#         if k not in perfect_indices and k not in good_indices:
#             bad_indices.append(k)


#     return [_get_filtered_agg_stats(all_stats_dict, indices, args) for indices in [perfect_indices, good_indices, bad_indices]]
    
def save_stats_tsv(stats_fn, stats_dict, stats_loc_name, args):
    settings = args.settings.split(",")
    metrics = [
        "had_compile_errors",
        "avg_compile_errors",
        "had_runtime_errors",
        "avg_runtime_errors",
        "test_cases_run",
        "test_case_average",
        "test_case_average_ignoring_compile_errors",
        "strict_accuracy",
    ]

    agg_csv_lines = []
    for i in range(args.n_parts):
        line = [setting, i]
        for met in metrics:
            setting_mets = [stats_dict[setting][str(i)][met] for setting in settings]
            line.extend(setting_mets)

        agg_csv_lines.append(line)
    
    metric_names = []
    for met in metrics:
        for setting in settings:
            metric_names.append(f"{met}_{setting}")


    tsv_fn = stats_loc_name.replace("all_stats.json", f"{stats_fn}.tsv")
    print("Saving summary stats to csv file: ", tsv_fn)
    columns = metric_names
    
    df = pd.DataFrame(agg_csv_lines, columns=columns)
    df.to_csv(tsv_fn, sep="\t", index=False)

def get_summary_stats(all_results, args, results_folder):
    """

    all_results: Dict where each key is [q_idx][setting][partial_sol_idx]

    """

    # get stats results save location
    stats_loc_name = os.path.join(args.output_dir, "stats")
    if not os.path.exists(stats_loc_name):
        os.makedirs(stats_loc_name)
        
    stats_loc_name = os.path.join(stats_loc_name, "all_stats.json")
    # problem_level_stats_loc_name = stats_loc_name.replace("all_stats.json", "problem_level_stats.json")
    
    # import pdb; pdb.set_trace()
    agg_stats_dict = {}
    all_stats_dict = {}
    agg_csv_lines = []
    
    # for each setting + partial_sol_idx
    problem_level_stats, problem_level_agg_stats, agg_stats = _get_summary_stats(
        all_results, 
        args
    )   

    import pdb; pdb.set_trace()
    
    # filtered_agg_stats = get_filtered_agg_stats(all_stats_dict, args)

    # save stats tsv
    # for stats_fn, stats_dict in filtered_agg_stats + [
    #     ("agg_stats_dict", agg_stats_dict)
    # ]:  
    
    
    # save stat jsons
    # print("Saving stats to ", stats_loc_name)
    # with open(stats_loc_name, "w") as f:
    #     f.write(json.dumps(agg_stats_dict, default=np_encoder))

    # print("Saving problem_level stats to ", problem_level_stats_loc_name)
    # with open(problem_level_stats_loc_name, "w") as f:
    #     f.write(json.dumps(all_stats_dict, default=np_encoder))


def main(args):

    # get location to save results of running codes
    results_folder = os.path.join(args.output_dir, "raw_output")

    if not os.path.exists(results_folder):
        raise ValueError(f"Results folder {results_folder} does not exist")

    results = {}
    # only select one file
    if args.qid is not None:
        results_loc_name = os.path.join(results_folder, str(args.qid), f"test_case_results.json")
        with open(results_loc_name, "r") as f:
            res = json.loads(f.read())
            results[args.qid] = res


    # do all files
    else:  

        # read each file
        for example_subdir in os.listdir(results_folder):
            results = {}
            results_loc_name = os.path.join(example_subdir, f"test_case_results.json")
            with open(results_loc_name, "r") as f:
                res = json.loads(f.read())
                results[args.qid] = res

            
    # get histograms of summary starts
    get_summary_stats(results, args, results_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing a Language Model on Python Code")
    # parser.add_argument("-t","--test_loc", default="../data_split/test.json", type=str, help="path to the json containing problem paths to be evaluated.")
    # parser.add_argument("-r","--root", default="../", type=str, help="where the data is stored.")
    # parser.add_argument("-s","--start", default=0, type=int)
    # parser.add_argument("-e","--end", default=None, type=int, help="If you want to evaluate a subset of problems specify start and ending index. File with start and ending prefix must exist typically used with batch evaluation.")
    parser.add_argument("--qid", type=int, default=None)
    
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--output_dir", help="Where the evaluated data is loaded from and results saved to.")
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