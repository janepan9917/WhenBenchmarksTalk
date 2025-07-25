import argparse
import multiprocessing 
import re

from pathlib import Path
from anthropic import Anthropic
from tqdm import tqdm
from json import loads, dump
from os import getenv
from copy import deepcopy


def verify(res, tests, return_dict):
    if "```" in res:
        code = res.split("python")[1][1:-3] # get rid of leading newline + ending code ticks
    else:
        code = res
    fn_name = code.split("def ")[1].split("(")[0]   # ) <-- for my indenting...
    program = f'{code}\n'
    correct = 0
    failure = 0
    try:
        exec(code)
    except:
        # syntax error in base code
        return_dict["compiles"] = False
        return
    
    return_dict["compiles"] = True
    for ex in tests:
        program_copy = deepcopy(program)
        program_copy += f"assert {fn_name}({ex['input'].replace('\n', ',')}) == {ex['output']}\n"
        try:
            exec(program)
            correct += 1
        except:
            failure += 1
    
    if correct + failure > 0:
        return_dict["pass_rate"] = correct / (correct + failure)
    else:
        return_dict["pass_rate"] = 0

def main(args):
    data_file = Path(args.data)
    model = Anthropic(
                api_key=getenv(args.api_key)
            )
    
    with open(data_file) as f, open(args.out_path, "w+") as o:
        for line in tqdm(f):
            data_obj = loads(line)
            question = data_obj["question"]
            llm_summary = data_obj["summarized_question"]
            examples = data_obj["examples"]
            const = data_obj["constraint"]
            tests = data_obj["pub_test_cases"] + data_obj["priv_test_cases"]

            prompt = f"Solve the following problem:\n {question}\n\n"
            if not examples == []:
                prompt += f"Given the following examples:\n {examples}\n\n"
            if not const == []:
                prompt += f"Given the following constraint(s):\n {const}\n\n"
            if args.with_summary:
                prompt += f"A summary of the solution is:\n {llm_summary}"
            response = model.messages.create(
                    system="Only write a code block, do not write any other text or justificaiton. Do not write a class, only a single python function.",
                    model = "claude-3-5-sonnet-20240620",
                    max_tokens=4096,
                    messages= [
                        {"role": "user", "content":prompt}
                    ])

            manager = multiprocessing.Manager()
            ret = manager.dict()

            p = multiprocessing.Process(target=verify, args=(response.content[0].text, tests, ret))
            p.start()
            p.join(15)
            if p.is_alive():
                runtime_fails = True
                compiles = True
                p.terminate()
            else:
                runtime_fails = False
                compiles = ret["compiles"]

            result_dict = {
                "runtime_fail": runtime_fails,
                "compiles": compiles,
                "percent_correct": ret["pass_rate"]
            }
            dump(result_dict, o)
            o.write(f"\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="File input")
    parser.add_argument("--out_path", type=str, help="File output", default="tmp.txt")
    parser.add_argument("--api_key", type=str, help="Path to sys api key")

    parser.add_argument("--with_summary", action="store_true", help="If the code generation should use the llm summary")

    main(parser.parse_args())

