from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from datasets import load_dataset
import numpy as np
import argparse
import os
import logging
import torch
from transformers import StoppingCriteria

from src.utils import *
from src.dataset import *

# @dataclass
# class StoppingCondition():
#     """
#     Determines:
    
#     1) How often the code model stops to see if it should query the user
#     2) How the code model queries the user

#     Attributes
#     ----------
#     n_lines : int
#         How many lines to generate before stopping.
#         This won't work for models accessible only through API. See n_tokens instead.
#         Note that only one of n_lines and n_tokens should be set to an actual number. 
#     n_tokens : int
#         How many tokens to generate before stopping.
#         This is a workaround since API models don't have a clean way to stop after >1
#         occurrences of a stop sequence.
#         Note that only one of n_lines and n_tokens should be set to an actual number. 

#     stopping_model_prompt : str
#         the sound that the animal makes
#     num_legs : int
        

#     Methods
#     -------
#     is_query_needed(context)

#     """
#     n_lines: int = None
#     n_tokens: int = None
#     model_prompt: str = None

#     def is_query_needed():
        

class MaxNewLinesCriteria(StoppingCriteria):
    """
    Stops the generation process after a specified number of newline ('\n') tokens 
    have been generated beyond the initial prompt.

    Args:
        start_length (int):
            The number of tokens in the initial prompt. This ensures only the generated 
            tokens are counted towards the newline limit.
        max_new_lines (int):
            The maximum number of newline ('\n') tokens to allow in the generated output.
        newline_token_id (int):
            The token ID that corresponds to the newline character ('\n'). This is 
            typically obtained from the tokenizer, e.g., `tokenizer.encode('\n')[0]`.
    """
    def __init__(self, start_length: int, max_new_lines: int, newline_token_id: int):
        self.start_length = start_length
        self.max_new_lines = max_new_lines
        self.newline_token_id = newline_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Determines whether the generation should stop based on the number of newline tokens.

        Args:
            input_ids (torch.LongTensor):
                The sequence of token IDs generated so far, including the prompt.
            scores (torch.FloatTensor):
                The prediction scores for the next token. Not used in this criteria.
            **kwargs:
                Additional arguments (unused).

        Returns:
            bool:
                `True` if the number of newline tokens in the generated output has 
                reached or exceeded `max_new_lines`, else `False`.
        """
        # Extract the generated tokens excluding the prompt
        generated_ids = input_ids[0, self.start_length:]
        
        # Count the number of newline tokens
        newline_count = (generated_ids == self.newline_token_id).sum().item()
        
        # Debug
        # print(f"Generated tokens: {generated_ids.tolist()}")
        # print(f"Current newline count: {newline_count}")

        # Determine if the stopping condition is met
        return newline_count >= self.max_new_lines
