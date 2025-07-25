import argparse
import json
import numpy as np
import os
from dataclasses import dataclass, field
import pandas as pd
from typing import Dict, List, Tuple
import typing
from openai import OpenAI
from anthropic import Anthropic

@dataclass
class Model():
    name: str
    model: 'typing.Any' = None
    api: str = None # anthropic / openai

    def load_model(self, api_key, model_name):
        """
        Load a model from the OpenAI API.
        """

        if "gpt" in model_name:
            self.model = OpenAI(
                api_key=os.getenv(api_key),
            )  
            self.api = "openai"

        elif "claude" in model_name:
            self.model = Anthropic(
                api_key=os.getenv(api_key),
            )
            self.api = "anthropic"
        

    def check_intent_diversity(self, text):
        """
        Use an OAI LLM to summarize code.
        """
        prompt = f"""

        Here is the code: 
        {code}

        Here is the list

        Now, please provide a natural language summary of the code. 
        """

        if self.api == "openai":
            response = self.model.chat.completions.create(
                model=self.name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                n=1,
            )
            return response.choices[0].message.content

        elif self.api == "anthropic":
            response = self.model.messages.create(
                model=self.name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4096, # just set to max possible tokens for now
            )

            return response.content[0].text

