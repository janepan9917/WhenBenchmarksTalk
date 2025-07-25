from argparse import ArgumentParser
from json import loads
from os import getenv
from pathlib import Path
from anthropic import Anthropic
from tqdm import tqdm

def main(args):
    data_file = Path(args.in_path)
    model = Anthropic(
                api_key=getenv(args.api_key)
            )
    num_summaries = 3

    with open(data_file) as f, open(args.out_path, "w+") as o:
        for line in tqdm(f):
            data_obj = loads(line)
            gen_code = data_obj["gen_solutions"]

            o.write(f"---- QUESTION ----\n{data_obj['question']} \n----------\n")
            # Arb choose to summarize one gen_code for now
            code = gen_code[0]
            
            for i in range(num_summaries):
                prompt = f"""
                  Your goal is to provide a natural language summary of the code below. Try to provide information about the overall purpose of the code, as well as the individual steps needed in order to achieve this purpose. You are only allowed to use natural language to describe the code. Do not refer to any specific code syntax or code structure, including function names or variable names. Use incomplete sentences or more concise terminology. Try to be ambiguous about the exact implementation. Your summary should be shorter than the code, but you can be fairly detailed.

                  Here is the code: 
                  {code}
                  Now, please provide a natural language summary of the code.
                """
                response = model.messages.create(
                        model = "claude-3-5-sonnet-20240620",
                        max_tokens=4096,
                        messages= [
                            {"role": "user", "content":prompt}
                        ]
                )
                o.write(f"--- SUMMARY {i+1} ---\n")
                o.write(f"{response.content[0].text}\n\n")
                o.write("-----------")


            o.write(f"\n\n\n")





if __name__ == "__main__":
    parser = ArgumentParser(description="Testing a Language Model on Python Code")
    parser.add_argument("--in_path", default="../../data/livecodebench/livecodebench_oracle_data_leetcode_medium.jsonl", type=str, help="path to the json containing generated lcb code/data")
    parser.add_argument("--out_path", default="temp.txt", type=str, help="path of output")
    parser.add_argument("--api_key", type=str, help="Env var on system of api key")
    parser.add_argument("--num_sum", type=int, help="Number of summaries to generate")


    args = parser.parse_args()
    main(args)
