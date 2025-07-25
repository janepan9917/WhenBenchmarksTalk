import argparse
import json
import numpy as np
import os
import pandas as pd
import pprint
import math
import time
from collections import defaultdict

# for timing debugging
from datetime import datetime, date
from tqdm import tqdm

from datasets import load_dataset
from types import SimpleNamespace
from typing import Dict
from pathlib import Path

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

        if None in results_dict[i].values():
            return {
                "had_compile_errors": None,
                "avg_compile_errors": None,
                "had_runtime_errors": None,
                "avg_runtime_errors": None,
                "test_cases_run": None,
                "test_case_average": None,
                "test_case_average_ignoring_compile_errors": None,
                "strict_accuracy": None,
            } 
        
        if "test_case_average_ignoring_compile_errors" not in results_dict[i].keys():
            problem_test_case_averages = results_dict[i]["test_case_average"]
            problem_compile_errors = results_dict[i]["had_compile_errors"]
            results_dict[i]["test_case_average_ignoring_compile_errors"] = np.nanmean([problem_test_case_averages[i] for i in range(len(problem_test_case_averages)) if problem_compile_errors[i] == 0])

    aggregate_results = {
        "had_compile_errors": np.mean([results_dict[i]["had_compile_errors"] for i in results_dict]),
        "avg_compile_errors": np.mean([results_dict[i]["avg_compile_errors"] for i in results_dict]),
        "had_runtime_errors": np.mean([results_dict[i]["had_runtime_errors"] for i in results_dict]),

        "avg_runtime_errors": np.mean([results_dict[i]["avg_runtime_errors"] for i in results_dict]),
        "avg_runtime_errors_std": np.mean([results_dict[i]["avg_runtime_errors_std"] for i in results_dict]),

        "test_cases_run": np.mean([results_dict[i]["test_cases_run"] for i in results_dict]),
        "test_cases_run_std": None,

        "test_case_average": np.mean([results_dict[i]["test_case_average"] for i in results_dict]),
        "test_case_average_std": np.mean([results_dict[i]["test_case_average_std"] for i in results_dict]),
        
        "test_case_average_ignoring_compile_errors": np.nanmean([results_dict[i]["test_case_average_ignoring_compile_errors"] for i in results_dict]),
        "test_case_average_ignoring_compile_errors_std": np.nanmean([results_dict[i]["test_case_average_ignoring_compile_errors_std"] for i in results_dict]),
        
        "strict_accuracy": np.mean([results_dict[i]["strict_accuracy"] for i in results_dict]),
        "strict_accuracy_std": np.mean([results_dict[i]["strict_accuracy_std"] for i in results_dict]),
        
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
    for res in all_results:
        qid, res = res
        problem_test_case_averages = []
        problem_strict_accuracies = []
        problem_compile_errors = []
        problem_had_compile_errors = 0
        problem_runtime_errors = []
        problem_had_runtime_errors = 0

        if len(res) == 0:
            results_dict[qid] = {
            "had_compile_errors": None,
            "avg_compile_errors": None,
            "had_runtime_errors": None,
            "avg_runtime_errors": None,
            "test_cases_run": None,
            "test_case_average": None,
            "test_case_average_ignoring_compile_errors": None,
            "strict_accuracy": None,
        }
        
        else:
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
                
            test_case_average_ignoring_compile_errors = [problem_test_case_averages[i] for i in range(len(problem_test_case_averages)) if problem_compile_errors[i] == 0]

            results_dict[qid] = {
                "had_compile_errors": problem_had_compile_errors > 0,
                "avg_compile_errors": problem_had_compile_errors/len(res),
                "had_runtime_errors": problem_had_runtime_errors > 0,
                "avg_runtime_errors": np.mean(problem_runtime_errors),
                "avg_runtime_errors_std": np.std(problem_runtime_errors)/math.sqrt(len(problem_runtime_errors)),
                "test_cases_run": n_test_cases_run,
                "test_case_average": np.mean(problem_test_case_averages),
                "test_case_average_std": np.std(problem_test_case_averages)/math.sqrt(len(problem_test_case_averages)),
                "test_case_average_ignoring_compile_errors": np.mean(test_case_average_ignoring_compile_errors),
                "test_case_average_ignoring_compile_errors_std": np.std(test_case_average_ignoring_compile_errors)/math.sqrt(len(test_case_average_ignoring_compile_errors)),
                "strict_accuracy": np.mean(problem_strict_accuracies),
                "strict_accuracy_std": np.std(problem_strict_accuracies)/math.sqrt(len(problem_strict_accuracies)),
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
        all_results.append((index, problem_results))


    # get results per problem aggregated over n_generations
    problem_level_agg_stats = compute_problem_level_agg_stats(
        all_results, 
        args,
    )

    # # get results aggregated over all problems
    # agg_stats = get_agg_averages(problem_level_agg_stats)
    
    # get problem_level statistics with individual generations held separately
    problem_level_stats = {}
    for q_idx, res in problem_level_results.items():
        problem_level_stats[q_idx] = compute_stats(res)

    # collect all stats for a particular settindg
    all_stats= defaultdict(list)
    for q_idx, res in problem_level_stats.items():
        # import pdb; pdb.set_trace()
        for stat in res.keys():            
            all_stats[stat].extend(res[stat])

        # get pass@k
        if 1.0 in res["test_case_average"]:
            all_stats["pass@4"].append(True)
        else:
            all_stats["pass@4"].append(False)
            
        all_stats["pass@1"] = all_stats['strict_accuracy']



    # get agg stats over all problems
    new_agg_stats = {}
    for stat, res in all_stats.items():
        new_agg_stats[stat] = np.mean(res)
        new_agg_stats[stat + "_std"] = np.std(res)/math.sqrt(len(res))
    
    return problem_level_stats, problem_level_agg_stats, all_stats, new_agg_stats


def get_summary_stats(all_results, key, args):
    # get stats results save location
    stats_loc_name = os.path.join(args.output_dir, "stats_test", str(key))
    if not os.path.exists(stats_loc_name):
        os.makedirs(stats_loc_name)
        
    # import pdb; pdb.set_trace()
    agg_stats_dict = {}
    all_stats_dict = {}
    agg_csv_lines = []
    
    # for each setting + partial_sol_idx
    problem_level_stats, problem_level_agg_stats, all_stats, agg_stats = _get_summary_stats(
        all_results, 
        args
    )   
    # import pdb; pdb.set_trace()
    
    # save  jsons
    for fn, stats_dict in zip([
        "problem_level_stats.json",
        "problem_level_agg_stats.json",
        "agg_results.json",
        "agg_stats.json",
        ], [
            problem_level_stats, 
            problem_level_agg_stats, 
            all_stats, 
            agg_stats]):

        print(f"Saving {fn} to {stats_loc_name}")
        stats_fn = os.path.join(stats_loc_name, fn)
        with open(stats_fn, "w") as f:
            f.write(json.dumps(stats_dict, default=np_encoder))

def stats_summary(args):
    def get_max_key(data_dir): 
        numeric_listdir = [s.name for s in Path(data_dir, "stats").iterdir() if s.name.isdigit()]
        return max(numeric_listdir, key=int)
    """
    Plots the standard tables with summary stat metrics.
    """
    stats = [
        # "had_compile_errors",
        # "avg_compile_errors",
        # "had_runtime_errors",
        "avg_runtime_errors",
        "test_case_average",
        # "test_case_average_ignoring_compile_errors",
        "strict_accuracy",
        # "pass@4"
    ]
    stds = [stat+"_std" for stat in stats]

    all_stats = []
    data_dir = Path(args.output_dir)
    max_key = get_max_key(data_dir)
    setting_dir = Path(data_dir, "stats")
    # numeric_listdir = [s for s in os.listdir(setting_dir) if s.isdigit()]
    # max_key = max(numeric_listdir, key=int)

    stat_fn = setting_dir / f"{max_key}/agg_stats.json"
    with open(stat_fn, "r") as f:
        setting_stats_dict = json.load(f)
    
    rounded_stats = {k: round(v, 4) for k, v in setting_stats_dict.items() if type(v) == float}

    setting_stats = [f"{rounded_stats[stat]} +/- {rounded_stats[std]}" for stat, std in zip(stats, stds)]
    # setting_stats = [f"{rounded_stats[stat]}" for stat, std in zip(stats, stds)]
    # setting_stats.insert(0, setting)

    all_stats.append(setting_stats)

    all_stats = sorted(all_stats)

    # Save TSV of general stats
    agg_stats_folder = Path(args.stats_dir)
    if not agg_stats_folder.exists():
        agg_stats_folder.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.stats_dir, "all_stats.tsv"), "w") as f:
        # stats.insert(0, "setting")
        f.write("\t".join(stats) + "\n")
        for setting_stats in all_stats:
            f.write("\t".join(setting_stats) + "\n")
    
    print("Saved summary stats to all_stats.tsv")
            
def main(args):
    # get location to save results of running codes
    results_folder = os.path.join(args.output_dir, "raw_output")

    if not os.path.exists(results_folder):
        raise ValueError(f"Results folder {results_folder} does not exist")

    results = {}
    # read each file
    for example_subdir in os.listdir(results_folder):
        results_loc_name = os.path.join(results_folder, example_subdir, f"test_case_results.json")
        if os.path.exists(results_loc_name):
            with open(results_loc_name, "r") as f:
                res = json.load(f)
                try:
                    results[int(example_subdir)] = res
                except:
                    ascii_str = "".join([str(ord(c)) for c in example_subdir])
                    results[int(ascii_str)] = res

    if results == {}:
        raise ValueError(f"Couldn't find any data in this directory: {args.output_dir}")
    
    print(f"Found {len(results)} results\n\n")
    for key in res.keys():
        temp_results = {i: results[i][key] for i in results.keys()}
        get_summary_stats(temp_results, key, args)
    
    stats_summary(args)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--output_dir", help="Where the evaluated data is loaded from and results saved to.")
    parser.add_argument("--stats_dir", help="Where the evaluated data is loaded from and results saved to.")
    args = parser.parse_args()

    main(args)
