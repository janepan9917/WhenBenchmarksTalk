import argparse
import json
import csv

from pathlib import Path


def main(args):
    p = Path(args.base_dir)
    
    results = []
    for pass_file in p.glob("**/pass_at_k_result.json"):
        with open(pass_file, 'r') as f:
            data = json.load(f)
    
        result = data[f'pass_{args.pass_num}']['classeval']
        result['setting'] = pass_file.parent.name
        results.append(result)

    with open(Path(p,"final_classeval_results.tsv"), "w+", newline='') as f:
        writer = csv.DictWriter(f, ["setting", "class_partial_success", "class_success", "fun_partial_success", "fun_success"], delimiter='\t')
        writer.writeheader()
        writer.writerows(results)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, help="two above the pass@k results file")
    parser.add_argument("--pass_num", type=int, help="the k in pass @ k")
    
    main(parser.parse_args())
