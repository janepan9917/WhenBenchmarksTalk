import argparse
import json
import os
import time

from pathlib import Path
from classeval_evaluation.test_pipeline import AutoTest

def reformat_outputs(base, args):

    for qid in range(100):
        qid_dir = Path(base, "raw_output", f"ClassEval_{qid}", "generation_data")
        all_sample_paths = []
        for i in range(4):
            all_sample_paths.append(Path(qid_dir, f"sample_{i}", "iter_refinement_data.json"))


        with open(Path(qid_dir.parent, 'final_output.json')) as f:
            meta_data = json.load(f)

        for step in range(args.n_steps):
            reformatted = {}
            reformatted['task_id'] = meta_data['qid']
            reformatted['test'] = meta_data['test_cases']
            reformatted["import_statement"] =  meta_data['import_statement']
            reformatted["test_classes"] = meta_data['test_classes']
           
            code_list = []
            for sample_path in all_sample_paths:
                with open(sample_path, 'r') as f:
                    sample_code_data = json.load(f)

                program = sample_code_data[str(step)]['final_code']
                curr = program.strip()
                curr = "```python\n" + curr + "\n```"
                code_list.append(curr)

            reformatted['predict'] = code_list
            
            with open(Path(qid_dir.parent, f"code_step_{step}.json"), "w+") as f_out:
                json.dump(reformatted, f_out)

def consolidate(base_dir, stat_dir, step):
    final_files = []
    for i in range(100):
        final_files.append(Path(base_dir, "raw_output", f"ClassEval_{i}", f"code_step_{step}.json"))
    print(f"found {len(final_files)} for step {step}")
    
    all = []
    for file in final_files:
        with open(file, 'r') as f:
            data = json.load(f)
            all.append(data)
    
    total_file = Path(stat_dir, f'results_{step}.json')
    with open(total_file, 'w+') as f:
        json.dump(all, f)
        


def main(args, step_num, stat_folder):
    p = Path(args.output_dir)
    model = p.parent.name
    setting = p.name

    source_file_name = 'classeval' # our name for this set of tests per model
    model_list = [source_file_name]

    # reformat created final_class_eval in each raw_output/ClassEval_{i}

    # the eval wants it all in one big json file, list of results in output_dir/results.json
    consolidate(p, stat_folder, step_num)

    tester = AutoTest(Path(stat_folder, f'results_{step_num}.json'))
    tester.test_pipeline(source_file_name, Path(stat_folder, f'results_{step_num}.json'))
    time.sleep(5)
    tester.evaluate(model_list) 

    result = {}

    result["pass_1_greedy"] = tester.cal_metrics_pass_at_k(model_list, 1, 1)
    result["pass_1_4"] = tester.cal_metrics_pass_at_k(model_list, 1, 4)
    result["pass_3_5"] = tester.cal_metrics_pass_at_k(model_list, 3, 5)

    save_path = Path(tester.path, "STEPIN_reformat_at_k_result.json")

    ori_data = result

    with open(save_path, 'w') as f:
        json.dump(ori_data, f, indent=4, sort_keys=True)
    
    print(f"saved to {save_path}")

def _main(args):
    os.sync() 
    reserved_files = ["evaluation.py", "path_util.py", "test_pipeline.py", "README.md", "incremental generation.png", "run.sh"]
    reserved_files += ["classeval_evaluation", "test_classes.py", "classeval_log_data.log", "results_to_tsv.py", "make_qid_test_case_jsons.py"]

    p = Path(args.output_dir)
    reformat_outputs(p, args)
    stats_folder = Path(p, "stats")
    stats_folder.mkdir(parents=True, exist_ok=True)

    for step_num in range(args.n_steps):
        step_i_dir = Path(stats_folder, f"{step_num}")
        step_i_dir.mkdir(parents=True, exist_ok=True)
        try:
            main(args, step_num, step_i_dir)
        except Exception as e:
            print(f"Failed processing step {step_num}: {e}")
            # Optional: decide if you want to continue or break
            continue
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help="just before raw_output")
    parser.add_argument('--greedy', type=int, default=1)
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--n_steps', type=int, default=5)

    args = parser.parse_args()
    _main(args)


