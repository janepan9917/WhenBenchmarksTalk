# Just converting old data to new data format so we can test evaluation 


import os
import json

def read_json(json_fn: str):
    """Basic utility for reading jsonl files."""
    with open(json_fn, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        yield json.loads(json_str)

data_fn = "../output/dataset_apps_oracle-model_gpt-3.5-turbo-n_examples_100-n_samples_16-oracle_model_gpt-4-turbo-split_type_line/raw_data.jsonl"

new_data_folder = "../output/dataset_apps_oracle-model_gpt-3.5-turbo-n_examples_100-n_samples_16-oracle_model_gpt-4-turbo-split_type_line/raw_data/"

test_case_folder = "../output/dataset_apps_code_completion_w_oracle-model_claude-3.5-sonnet-n_examples_100-n_samples_16-split_type_paragraph/test_cases_results"

keys = {
    "oracle": "code_completions_w_comments",
    "baseline": "code_completions_wo_comments",
}

# assemble the data
for name, old_name in keys.items():
    data = read_json(data_fn)

    for line in data:
        responses = line["responses"]

        line["responses"] = {}
        for i in responses[old_name].keys():
            line["responses"][i] = {
                "query": responses[old_name][i]["query"],
                "responses": responses[old_name][i]["responses"],
            }


        # Save results to JSON file.
        results_fn = os.path.join(new_data_folder, f"{name}.jsonl")
        print(results_fn)
        with open(results_fn, "a+") as f:
            json.dump(line, f)
            f.write("\n")
