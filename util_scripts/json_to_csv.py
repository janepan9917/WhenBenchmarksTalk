import json
import pandas as pd

# json_fn = "../output/dataset_human_eval-model_gpt-3.5-turbo-n_examples_10-n_samples_None_new_prompt_code_snippet.jsonl"
json_fn = "output/dataset_apps_oracle-model_gpt-3.5-turbo-n_examples_50-n_samples_16-oracle_model_gpt-4-turbo-split_type_line_stats.json"

with open(json_fn, 'r') as json_file:
    r = json.loads(json_file)
    for setting in ["code_completion", "docstring_completion"]:
        lines = []
        for qid, response in r[setting].items():
            code_snippet = response["code_snippet"]
            output = response["response"]
            confidence = output.split("Confidence: ")[-1].split("\n")[0]
            description = output.split("\n")[0]
            lines.append((qid, code_snippet, description, confidence, output))
        
        columns = ["qid", "code_snippet", "model_description", "model_confidence", "full_model_output"]
        df = pd.DataFrame(lines, columns=columns)

        df.to_csv(json_fn.replace("jsonl", f"{setting}.csv"), sep="\t", index=False)

#         qid = r["qid"]
#         task = r["task"]
#         signature = r["signature"]
#         docstring = r["docstring"].split("Examples")[0]
        
#         responses = r["responses"]
#         for i, response in responses.items():
#             code_snippet = response["code_snippet"]
#             output = response["response"][0]
#             confidence = output.split("Confidence: ")[-1].split("\n")[0]
#             description = output.split("\n")[0]
#             line_number = i
            
            
#             lines.append((qid, signature, docstring, line_number, code_snippet, description, confidence, output))
        
#     columns = ["qid", "signature", "docstring", "line_number", "code_snippet", "model_description", "model_confidence", "full_model_output"]
#     df = pd.DataFrame(lines, columns=columns)

#     df.to_csv(json_fn.replace("jsonl", "csv"), sep="\t", index=False)

        
# json_fn = "../output/dataset_human_eval_plus_code_completion-model_vllm-llama-3-8b-n_examples_10-n_samples_1_no_docstring.jsonl"


# with open(json_fn, 'r') as json_file:
#     json_list = list(json_file)
#     lines = []
#     for json_str in json_list:
#         r = json.loads(json_str)

#         qid = r["qid"]
#         task = r["task"]
#         signature = r["signature"]
#         solution = r["full_code"]
#         docstring = r["docstring"].split("Example")[0]
        
        
#         responses = r["responses"]

#         response_num_to_string = {
#             0: "declaration only",
#             # 1: "declaration with model-generated docstring",
#             1: "declaration with ground-truth docstring",
#             2: "declaration with ground-truth + 2 test cases",
#             3: "declaration with ground-truth + 4 test cases",
#             4: "declaration with ground-truth + 8 test cases",
#             5: "declaration with ground-truth + 16 test cases",
#         }

#         for i, response in responses.items():
#             docstring = response["docstring"]
#             output = response["response"]
#             description = response_num_to_string[int(i)]
#             is_exact_match = output == solution or solution in output or output in solution 
#             lines.append((qid, signature, description, docstring, output, solution, is_exact_match))
        
#     columns = ["qid", "signature", "input_type", "model_input", "model_output", "solution", "is_exact_match"]
#     df = pd.DataFrame(lines, columns=columns)

#     df.to_csv(json_fn.replace("jsonl", "csv"), sep="\t", index=False)