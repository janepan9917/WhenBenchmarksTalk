from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from datasets import load_dataset
import numpy as np
import argparse
import os
import logging

from src.utils import *

@dataclass
class PromptSet():
    """
    Class to hold prompts for a particular user + application.
    Three kinds right now:
    1) User model response
    2) Code model for code completion
    3) Code model for querying user model

    Attributes
    ----------
    prompt_type: str
        Type of prompt. Can be one of the following:
        - user_answer
        - asst_question
        - asst_code
    sys_prompt : Dict[str, str]
        Path to the prompt config file.
    prompt_config : Dict[str, str]
        Dictionary with all necessary prompts.

    Methods
    ----------
    get_answer_query_prompts(context: str) -> Dict[str, str]:
        Return sys + query prompts for answering asst model query.

    answer_query(context: str, asst_query: str) -> str:
        Generates answer to asst model query
    """
    prompt_type: str
    sys_prompt: str
    query_prompt: str
    prefill: str = None

# def get_initial_prefix(
#         config: Dict[str, str],
#         prompt_type: str, 
#         prompt_args: Dict[str, str]
#     ) -> str:
#     """
#     Get prefix from config file.
#     """
#     query_prefix_key = f"{prompt_type}_prefix"
#     return config[query_prefix_key].format(**prompt_args)

def get_prompts(
    config: Dict[str, str],
    prompt_type: str, 
    prompt_args: Dict[str, str],
    transform=False
) -> PromptSet:
    """
    Get prompt from config file.
    """
    sys_key = f"{prompt_type}_sys"
    query_key = f"{prompt_type}_query"
    prefill_key = f"{prompt_type}_prefill"

    if prompt_type == "asst_initial_code":
        sys_key = "asst_code_sys"
        prefill_key = "asst_code_prefill"

    sys = config[sys_key].format(**prompt_args)
    query = config[query_key].format(**prompt_args)
    prefill = config[prefill_key].format(**prompt_args)

    task = prompt_args["task"]
    if f"{prompt_type}_{task}_sys_suffix" in config.keys():
        sys_suffix = config[f"{prompt_type}_{task}_sys_suffix"].format(**prompt_args)
        sys = " ".join([sys.rstrip(), sys_suffix])

    logging.warning("Stripping newlines from prompts.")
    return PromptSet(
        prompt_type=prompt_type,
        sys_prompt=sys.strip(),
        query_prompt=query.strip(),
        prefill=prefill.strip(),
    )
        
