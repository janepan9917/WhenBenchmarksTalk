import json
import os 

def read_json(json_fn: str):
    """Basic utility for reading jsonl files."""
    with open(json_fn, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        yield json.loads(json_str)

for t in [
        # "baseline", 
        # "oracle_comment", 
        # "oracle_comment_local_only"
    ]:

    results_subdir = f"../output/dataset_apps_code_completion_w_oracle-model_claude-3.5-sonnet-n_examples_100-n_samples_4-split_type_paragraph/raw_data/{t}"
    
    fn = results_subdir + ".jsonl"

    # Save results to JSON file.
    if not os.path.exists(results_subdir):
        os.makedirs(results_subdir)

    for line in read_json(fn):
        qid = line["qid"]

        results_fn = os.path.join(results_subdir, f"{qid}.json")
        with open(results_fn, "a+") as f:
                json.dump(line, f)



# for t in [
#         "explicit", 
#         "implicit", 
#         "ground_truth"
#     ]:
#     fn = f"../output/dataset_apps_implicit_intent-model_claude-3.5-sonnet-n_examples_100-n_samples_16-split_type_paragraph/intent_hypotheses/{t}.jsonl"

#     n = 0
#     for line in read_json(fn):
#         n+=1
        
#         # import pdb; pdb.set_trace()
#         data = {
#             "question": line["question"],
#             "summarized_question": line["summarized_question"],
#             "responses": line["responses"]
#         }
#         if n > 10:
#             break
#         new_fn = f"sample_intent_hypotheses/{t}_{n}.json"
#         with open(new_fn, 'w') as f:
#             json.dump(data, f)

# for t in [
#         # "summarized_code", 
#         # "local_only_summarized_code", 
#         "self_generated"
#     ]:

#     fn = f"../output/dataset_apps_code_completion_w_oracle-model_claude-3.5-sonnet-n_examples_100-n_samples_4-split_type_paragraph/comments/{t}.jsonl"

#     n = 0
#     for line in read_json(fn):
#         n+=1
        
#         data = {
#             "question": line["question"],
#             "summarized_question": line["summarized_question"],
#             "responses": line["responses"]
#         }
#         if n > 10:
#             break

#         txt = f"""Question: {line["question"]}\n\n
# Summarized Question: {line["summarized_question"]}\n\n
# Oracle Comment 0: {line["responses"]['0']}
# Oracle Comment 1: {line["responses"]['1']}
# Oracle Comment 2: {line["responses"]['2']}
# Oracle Comment 3: {line["responses"]['3']}"""

#         new_fn = f"sample_oracle_comments/{t}_{n}.txt"
#         with open(new_fn, 'w') as f:
#             f.write(txt)




for t in [
        "baseline", 
        "oracle_comment", 
        "oracle_comment_local_only"
    ]:

    for n in range(20,21):
        fn = f"../output/dataset_apps_code_completion_w_oracle-model_claude-3.5-sonnet-n_examples_100-n_samples_4-split_type_paragraph/raw_data/{t}/{n}.json"

        
        if os.path.exists(fn):
            line = json.load(open(fn, 'r'))
            complete_response_strings = {}
            starter_code = {} 
            for idx in line["responses"].keys():
                starter_code[idx] = line["responses"][idx]["starter_code"]
                responses = line["responses"][idx]["responses"]
                complete_responses = [r for r in responses]
                complete_response_strings[idx] = "\n---------\n".join(complete_responses)
            
            orig_solution = line["responses"]["0"]["solution_data"]["orig_solution"]
            summary = line["responses"]["0"]["solution_data"]["summary"]
    
            if "Masha" in line["question"]:
                import pdb; pdb.set_trace()
         
            data = {
                "question": line["question"],
                "summarized_question": line["summarized_question"],
                "responses": complete_responses
            }

            txt = f"""Question: {line["question"]}\n\n
Summarized Question: {line["summarized_question"]}\n\n
################FULL GROUND TRUTH SOLUTION ################\n\n
{orig_solution}

################SOLUTION SUMMARY################\n\n
{summary}


################PARTIAL SOLUTION 0: ################\n\n
Starter Code: 
{starter_code['0']}
--------------------------------------
{complete_response_strings['0']}
################PARTIAL SOLUTION 1: ################\n\n
Starter Code: 
{starter_code['1']}
--------------------------------------
{complete_response_strings['1']}
################PARTIAL SOLUTION 2: ################\n\n
Starter Code: 
{starter_code['2']}
--------------------------------------
{complete_response_strings['2']}
################PARTIAL SOLUTION 3: ################\n\n
Starter Code: 
{starter_code['3']}
--------------------------------------
{complete_response_strings['3']}"""

            new_fn = f"sample_code_output/{t}_{n}.txt"
            n+=1
            with open(new_fn, 'w') as f:
                f.write(txt)

