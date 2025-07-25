import argparse
import json

from collections import defaultdict
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="right before raw output")
    parser.add_argument("--n_steps", type=int, default=5, help="number of steps in experiment")
    parser.add_argument("--n_samples", type=int, default=4, help="number of samples in experiment")


    args = parser.parse_args()
    

    # In each classeval qid have a file "test_case_results.json"
    # Json object with n_steps keys (from 0 to n_steps - 1)
    # Each step corresponds to list of lists of booleans or ints
    # Each list inside list of lists corresponds to a sample (retain order between steps)

    p = Path(args.dir)
    qid_paths = list(p.glob("raw_output/*"))
    print(f"found {len(qid_paths)} qids")

    for qid_path in qid_paths:
        curr_qid = qid_path.name

        test_case_results = {}

        for step in range(args.n_steps):
        # for step in [4]:
            test_case_results[step] = defaultdict(list)

            step_path = Path(p, "stats", str(step), "classeval_result.json")

            with open(step_path, 'r') as f:
                results = json.load(f)

            try:
                curr_q_results = results[curr_qid]
            except:
                # I think this is the catastrophic case, just skip I guess
                print(f"Didn't find {curr_qid}")
                continue

            list_of_list_of_bool_results = []
            for sample_num in range(args.n_samples):
                curr_sample_results = curr_q_results[f"{curr_qid}_{sample_num}"]
                
                errors = 0
                failures = 0
                tests_run = 0
                for test_name, test_result in curr_sample_results.items():
                    errors += test_result["errors"]
                    failures += test_result["failures"]
                    tests_run += test_result["testsRun"]
                
                correct = tests_run - errors - failures
                if tests_run > 0:
                    bool_result = [True]*correct + [False]*failures + [-1]*errors
                else:
                    bool_result = [-2] # assume if all 0 then nothing compiles

                list_of_list_of_bool_results.append(bool_result)
            if len(list_of_list_of_bool_results) < 4:
                print(f"DIDNT FIND ALL 4 SAMLPES FOR {curr_qid}")
                exit()

            test_case_results[step] = list_of_list_of_bool_results

        save_path = Path(qid_path, "test_case_results.json")
        # save_folder = Path(p, "stats", str(step), "qid_results", curr_qid)
        # save_folder.mkdir(parents=True, exist_ok=True)
        # save_path = Path(save_folder, "test_case_results.json")
        with open(save_path, "w+") as f:
            json.dump(test_case_results, f)
        print(f"Done {qid_path.name}")








