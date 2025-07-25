from dataclasses import MISSING, dataclass, field, fields
from enum import Enum
from typing import Optional, Dict, Any
import yaml
import logging
import torch



@dataclass
class DatasetConfig:
    input_transformation_type: str  # Type of transformation in data/name/difficulty/split
    solution_info_type: str   # Type of transformation in data/name/difficulty/split
    shuffle: bool             # True to reorder
    masking_rate: float = 0    # Proportion of words to mask

    name: Optional[str] = None       # Name of directory in code-uncertainty/data with data 
    difficulty: Optional[str] = None    # Difficulty to use in data/name
    n_examples: Optional[int] = None    # If None, use all examples
    split: Optional[str] = None         # Split in data/name/difficulty
    start_idx: int = 0        # Example idx to start using

    def __post_init__(self):
        if self.n_examples is not None and self.n_examples <= 0:
            raise ValueError("Number of question must be positive!")


@dataclass
class InteractionConfig:
    stopping_condition: str           # "always_stop", "never_stop", "ask_asst"
    feedback_setting: str = None
    feedback_type: str = None
    starter_code: float = None
    
    # mid_generation_feedback_type: Optional[str] = None               # comment, code, free_response_answer
    # iter_refinement_feedback_type: Optional[str] = None               # nl_feedback, free_response_answer
    n_iter_refinement_steps: int = 1                        # How many iterative refinement steps
    n_samples: int = 1                                      # Number of solutions to generate on each call
    history_window: int = 1 # to be deprecated
        
    user_query_budget: Optional[int] = None # How many queries the user model can be asked
                                            # if None, no limit
                                            # if 0, no queries allowed
    n_tokens: Optional[int] = None        # How many tokens to generate before stopping
                                          # for nonlocal models.
    n_lines: Optional[int] = None             # How many lines of code to generate before stopping

    # for comparison experiments
    comparison_output_dir: str = None
    starting_ir_idx: int = 0
 
    def __post_init__(self):
        # if self.iter_refinement_feedback_type is None \
        #     and self.mid_generation_feedback_type is None:
        #     logging.warning("No human feedback being used!")
        
        # if self.iter_refinement_feedback_type is not None \
        #     and self.mid_generation_feedback_type is not None:
        #     raise ValueError(
        #         "Only one of iter_refinement_feedback_type or mid_generation_feedback_type should be set!"
        #     )
        
        # if self.n_iter_refinement_steps > 1 and self.iter_refinement_feedback_type is None:
        #     raise ValueError(
        #         "If doing iterative refinement, iter_refinement_feedback_type must be set!"
        #     )
        
        # if self.n_iter_refinement_steps > 1 and self.mid_generation_feedback_type is not None:
        #     raise ValueError(
        #         "Mid-generation w/ iterative refinement is not yet implemented!"
        #     )
        
        
        # if self.n_iter_refinement_steps == 1 and self.iter_refinement_feedback_type is not None:
        #     raise ValueError(
        #         "If doing iterative refinement, need more than 1 step!"
        #     )

        # if self.stopping_condition != "never_stop" and self.n_tokens is None and self.n_lines is None:
        #     raise ValueError("If setting a stopping condition, either n_tokens or n_lines must be set!")
        
        # if self.stopping_condition == "never_stop" and self.mid_generation_feedback_type is not None:
        #     raise ValueError("For mid-generation feedback, stopping condition must be set!")

        # if self.stopping_condition != "never_stop" and self.iter_refinement_feedback_type is not None:
        #     raise ValueError("For iterative refinement, stopping condition must be never_stop!")

        if self.n_tokens is not None and self.n_lines is not None:
            raise ValueError("Only one of n_tokens or n_lines should be set")

@dataclass
class HuggingFaceGenerationConfig:
    """Configuration for model generation settings"""
    do_sample: bool = True
    temperature: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary, excluding None values"""
        return {k: v for k, v in self.__dict__.items() if v is not None}

@dataclass
class ModelConfig:
    model_name: str       # Abreviated API name of model 
    prompt_fn: str        # File with prompts for the model
    n_samples: int = 1  # Number of samples to generate (shoudl we remove this?)
    api_key: str = None         # Name of ENV variable in OS
    include_history: bool = False


    # For local models
    cache_dir: Optional[str] = None
    device: Optional[str] = None # Will be auto-detected if None
    generation_config: HuggingFaceGenerationConfig = field(
        default_factory=HuggingFaceGenerationConfig
    )
    stopping_condition: Optional[dict] = None

@dataclass
class OracleConfig(ModelConfig):
    def __post_init__(self):
        if self.n_samples <= 0:
            print("Number of question must be positive!")
            exit(1)

@dataclass
class UserModelConfig(ModelConfig):
    def __post_init__(self):
        if self.n_samples <= 0:
            print("Number of user responses must be positive!")
            exit(1)
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class AssistantModelConfig(ModelConfig):
    def __post_init__(self):
        if self.n_samples <= 0:
            raise ValueError("Number of generated solns must be positive!")
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class HuggingFaceModelConfig(ModelConfig):
    """Extended configuration for HuggingFace models"""
    def __post_init__(self):
        # Auto-detect device if not specified
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class Config:
    # output_dir: str 
    # overwrite_old_output: bool # What to do if output exists

    # Configs
    dataset: DatasetConfig 
    oracle_model: Optional[OracleConfig] = None
    user_model: Optional[UserModelConfig] = None
    asst_model: Optional[AssistantModelConfig] = None
    interaction: Optional[InteractionConfig] = None
    
    # transform_params: Optional[TransformParams] = None
    settings: list[str] = field(default_factory=list)
    data_dir: str = None
    
def parse_config(config_file_name:str, binary=False) -> Config:
    """
    Input: string of path to config file
    Returns: Config object of the information in file_name
    """
    required_fields = []
    additional_fields = []
    for entry in list(fields(Config)):
        if entry.default_factory is MISSING and entry.default is MISSING:
            required_fields.append(entry.name)
        else:
            additional_fields.append(entry.name)

    with open(config_file_name, "rb" if binary else "r") as f:
        data: dict = yaml.safe_load(f)

    # Ensure required items are present
    if not set(required_fields).issubset(set(data.keys())):
        print(f"Config missing required fields: {set(required_fields) - set(data.keys())}")
        exit(1)

    data["dataset"] = DatasetConfig(**data["dataset"])

    if "oracle_model" in data:
        data["oracle_model"] = OracleConfig(**data["oracle_model"])
    if "interaction" in data: 
        data["interaction"] = InteractionConfig(**data["interaction"])
    if "user_model" in data:
        data["user_model"] = UserModelConfig(**data["user_model"])
    if "asst_model" in data:
        data["asst_model"] = AssistantModelConfig(**data["asst_model"])

    return Config(**data)


if __name__ == "__main__":
    # file = "example_additional_config.yaml"
    file = "configs/example_config.yaml"
    x = parse_config(file)
    print(x)
    import pdb; pdb.set_trace()






