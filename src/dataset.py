from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from datasets import load_dataset
import numpy as np
import argparse
import os
import logging

from src.utils import *
from src.config import *

@dataclass
class Dataset():
    """
    Holds dataset meta-information and examples.

    Attributes
    ----------
    config : DatasetConfig
        Lists the dataset name, split, and transformation type.

    examples : List[Dict]
        List of examples in the dataset.
        Always read from folder of json files (1 example per file)
        

    Methods
    -------
    shuffle_examples(self) -> None:
        Shuffles examples.

    def load_examples(self):
        Loads dataset from json and prepares examples.
    """

    config: DatasetConfig
    data_dir: str = None
    examples: List[Dict] = field(default_factory=list)
    n_examples: int = None

    def __post_init__(self):
        self.load_examples()
        if self.config.shuffle:
            self.shuffle_examples()

    def shuffle_examples(self) -> None:
        """Shuffles examples."""
        np.random.shuffle(self.examples)
    
    def load_examples(self):
        """Loads dataset from json and prepares examples."""

        data_dir = f"{self.data_dir}/"

        input_dir = os.path.join(data_dir, f"input/{self.config.input_transformation_type}/")
        solutions_dir = os.path.join(data_dir, f"solutions/{self.config.solution_info_type}/")

        for fn in os.listdir(input_dir): 
            example = read_json(os.path.join(input_dir, fn))

            if self.config.input_transformation_type == "vanilla":
                example["input"] = example["orig_input"]

            if self.config.solution_info_type == "no_solution":
                example["solution_info"] = "(No solution provided. Use the only information from the question to help the assistant.)"

            elif self.config.solution_info_type == "full_solution":
                try:
                    solution = eval(example["solutions"])[0]
                except:
                    solution = example['solutions'][0]

                example["solution_info"] = solution
            
            else:
                raise ValueError("Invalid solution_info_type (plain summary solutions still not supported)")

            self.examples.append(example)
        
        if self.n_examples is not None:
            self.examples = self.examples[:self.n_examples]

if __name__ == "__main__":
    file = "configs/interaction.yaml"

    x = parse_config(file)
    dataset = Dataset(x.dataset)
    import pdb; pdb.set_trace()




