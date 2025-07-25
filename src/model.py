from dataclasses import dataclass, field
from datasets.config import HF_DATASETS_MULTITHREADING_MAX_WORKERS
from openai import OpenAI
from anthropic import Anthropic
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, EosTokenCriteria
from typing import List, Dict, Optional, Any
import os

# from vllm import LLM, SamplingParams
from together import Together
import cohere
from reka.client import Reka
import argparse
from tenacity import retry, wait_exponential, stop_after_attempt
import torch
import logging

from src.config import *
from src.prompt import PromptSet
from src.utils import get_path_to_code_uncertainty
from src.stopping_criteria import MaxNewLinesCriteria

def get_n_lines(s, n):
    lines = s.split('\n')
    counter = 0
    ret_lines = []
    for line in lines:
        if counter >= n:
            return ret_lines
        
        # has characters
        if any(letter.isalnum() for letter in line):
            counter += 1

        ret_lines.append(line)

    return ret_lines

@dataclass
class Model():
    """
    Handles the actual generation of text from a model.

    Attributes
    ----------
    config : ModelConfig
        Config with all necessary model params
    model : typing.Any
        The actual model itself. You should be able to call self.model.generate() or
        self.model.messages.create() to directly get output from the model.
    tokenizer : typing.Any
        The tokenizer for the model.
        

    Methods
    -------
    load(): 
        Prepares model and tokenizer for generation.
        This is called in get_model() after instantiating the class.
    
    generate(prompts: PromptSet, n_samples: int, stopping_condition: Config) -> List[str]:
        Generates text given the query and system prompts.
        
    get_max_tokens(stopping_condition: Config) -> int:
        For API models only. Returns the max tokens to generate before stopping.

    get_max_lines(stopping_condition: Config) -> int:
        For API models only. Returns the max tokens to generate before stopping.

    """
    config: ModelConfig 
    model: 'typing.Any' = None
    tokenizer: 'typing.Any' = None
    messages: List[Dict[str, str]] = None

    def load(self, key):
        raise NotImplementedError("Subclass must implement abstract method")

    def generate(self, prompts: PromptSet, stopping_condition: Optional[Config] = None, n_samples: int = 1) -> list[str]:
        raise NotImplementedError("Subclass must implement abstract method")


    def get_messages(self, prompts):
        raise NotImplementedError("Subclass must implement abstract method")

    def get_max_tokens(self, stopping_condition: Optional[Config] = None) -> int:
        max_tokens = 4096

        if stopping_condition is not None:
            if stopping_condition.n_tokens is not None:
                max_tokens = stopping_condition.n_tokens
            elif stopping_condition.n_lines is not None:
                max_tokens = 100*stopping_condition.n_lines

        return max_tokens

    def get_max_lines(self, stopping_condition: Config) -> int:
        max_lines = None

        if stopping_condition is not None:
            if stopping_condition.n_lines is None:
                raise ValueError("Stopping condition must have n_lines set.")
            max_lines = stopping_condition.n_lines

        return max_lines

    def reset_messages(self):
        self.messages = None


@dataclass
class OpenAIModel(Model):

    def load(self, key: str) -> None:
        self.model = OpenAI(
            api_key=key,
            # organization="org-PapcuLLbBhaQaETxKn4weUvg",
        )

        self.name = self.config.model_name

    def load_without_config(self, model_name, key) -> None:
        self.model = OpenAI(api_key=key)
        self.name = model_name

    def get_messages(self, sys_prompt, query_prompt):
        if self.messages is None or self.config.include_history is False:
            self.messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": query_prompt}
            ]
        else:
            self.messages.append({"role": "user", "content": query_prompt})

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(10))
    def generate(
        self, 
        prompts: PromptSet, 
        stopping_condition: Optional[Config] = None, 
        n_samples: int = 1,
    ) -> List[str]:
        """ Gets a responses from the OAI model. """
        
        sys_prompt = prompts.sys_prompt
        query_prompt = prompts.query_prompt
        query_prompt = query_prompt + "\n\n" + prompts.prefill

        self.get_messages(sys_prompt, query_prompt)

        response = self.model.chat.completions.create(
            model=self.name,
            messages=self.messages,
            n=n_samples,
        )

        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return [response.choices[i].message.content for i in range(len(response.choices))]
    
    def _generate(
        self, 
        sys_prompt: str,
        query_prompt: str,
        n_samples: int = 1,
    ) -> List[str]:
        """ Gets a responses from the OAI model. """
        

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": query_prompt}
        ]

        response = self.model.chat.completions.create(
            model=self.name,
            messages=messages,
            n=n_samples,
        )

        return [response.choices[i].message.content for i in range(len(response.choices))]
    

class DeepSeekChatModel(OpenAIModel):
    def load(self, key: str) -> None:
        self.model = OpenAI(
            api_key=key, 
            base_url="https://api.deepseek.com",
        )

        self.name = self.config.model_name
        
        
    def get_messages(self, sys_prompt, query_prompt):
        if self.messages is None or self.config.include_history is False:
            self.messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": query_prompt}
            ]
        else:
            self.messages.append({"role": "user", "content": query_prompt})

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(10))
    def generate(
        self, 
        prompts: PromptSet, 
        stopping_condition: Optional[Config] = None, 
        n_samples: int = 1,
    ) -> List[str]:
        """ Gets a responses from the OAI model. """
        
        sys_prompt = prompts.sys_prompt
        query_prompt = prompts.query_prompt
        query_prompt = query_prompt + "\n\n" + prompts.prefill

        self.get_messages(sys_prompt, query_prompt)

        response = self.model.chat.completions.create(
            model=self.name,
            messages=self.messages,
            n=n_samples,
            stream=False,
        )

        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})

        return [response.choices[i].message.content for i in range(len(response.choices))]


class TogetherAIModel(OpenAIModel):
    full_model_names = {
        "qwen-2.5-coder-32B": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "codellama-34B-instruct": "codellama/CodeLlama-34b-Instruct-hf",
        "gemma-2-27B": "google/gemma-2-27b-it",
    }

    def load(self) -> None:
        self.model = Together()
        self.name = self.full_model_names[self.config.model_name]
    
    def load_without_config(self, model_name) -> None:
        self.model = Together()
        self.name = self.full_model_names[model_name]
        
    def get_messages(self, sys_prompt, query_prompt):
        if self.messages is None or self.config.include_history is False:
            self.messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": query_prompt}
            ]
        else:
            self.messages.append({"role": "user", "content": query_prompt})

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(10))
    def generate(
        self, 
        prompts: PromptSet, 
        stopping_condition: Optional[Config] = None, 
        n_samples: int = 1,
    ) -> List[str]:
        """ Gets a responses from the OAI model. """
        
        sys_prompt = prompts.sys_prompt
        query_prompt = prompts.query_prompt
        query_prompt = query_prompt + "\n\n" + prompts.prefill

        self.get_messages(sys_prompt, query_prompt)

        response = self.model.chat.completions.create(
            model=self.name,
            messages=self.messages,
            n=n_samples,
        )

        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})

        return [response.choices[i].message.content for i in range(len(response.choices))]

    def get_logprobs(
        self, 
        query_prompt: str,
        n_samples: int = 1,
    ) -> List[float]:
        """ Gets a responses from the OAI model. """
        
        self.messages = [
            {"role": "user", "content": query_prompt},
        ]

        response = self.model.chat.completions.create(
            model=self.name,
            messages=self.messages,
            n=n_samples,
            logprobs=True,
            echo=True,
            max_tokens=0,
        )


        logprobs = response.prompt[0].logprobs.token_logprobs
        tokens = response.prompt[0].logprobs.tokens

        return tokens, logprobs
    
@dataclass
class ClaudeModel(Model):
    full_model_names = {
        "claude-3.5-sonnet": "claude-3-5-sonnet-20240620",
    }

    def load(self, key: str) -> None:
        self.name = self.full_model_names[self.config.model_name]
        self.model = Anthropic(
            api_key=key,
        )

    def get_messages(self, prompts):
        if self.messages is None or self.config.include_history is False:
            self.messages = [{"role": "user", "content": prompts.query_prompt}]
        else:
            self.messages.append({"role": "user", "content": prompts.query_prompt})
        return

    @retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(30))
    def generate(
        self, 
        prompts: PromptSet, 
        stopping_condition: Optional[Config] = None,
        n_samples: int = 1, 
    ) -> List[str]:
        """ Gets a responses from the Anthropic model. """
        
        
        # populate messages
        self.get_messages(prompts)

        if prompts.prefill != "":
            self.messages.append({"role": "assistant", "content": prompts.prefill}) 

        responses = []

         # If necessary, set stopping condition
        max_tokens = self.get_max_tokens(stopping_condition)
        for i in range(n_samples):
            
            if prompts.sys_prompt is None:
                response = self.model.messages.create(
                    model=self.name,
                    messages=self.messages,
                    max_tokens=max_tokens, 

                )
            else:
               
                response = self.model.messages.create(
                    model=self.name,
                    system=prompts.sys_prompt,
                    messages=self.messages,
                    # messages=messages,
                    max_tokens=max_tokens,
                )

            response = response.content[0].text
            if stopping_condition is not None and stopping_condition.n_lines is not None:
                response = "\n".join(get_n_lines(response, stopping_condition.n_lines))


            responses.append(response)
            
        
        if prompts.prefill != "":
            self.messages.pop() 
        
        self.messages.append({"role": "assistant", "content": response}) # TODO: fix this, this is terrible

        return responses
    

@dataclass
class HuggingFaceModel(Model):
    config: HuggingFaceModelConfig
    tokenizer: Optional[AutoTokenizer] = None
    model: Optional[AutoModelForCausalLM] = None
    attention: Optional[str] = 'flash_attention_2'
    _device: Optional[torch.device] = None

    def __post_init__(self):
        self._device = torch.device(self.config.device)
        logging.info(f"Using device: {self._device}")

    def load(self, key: Optional[str] = None) -> None:
        """
        Loads the Hugging Face model and tokenizer based on the model name provided in config.
        """
        # Load the Hugging Face tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir
        )
        logging.info(f"Loaded tokenizer for model {self.config.model_name}")
    
        if 'Qwen' in self.config.model_name:
            self.stop_criteria = [EosTokenCriteria(self.tokenizer.encode('<|im_end|>', add_special_tokens=False)[0]), EosTokenCriteria(self.tokenizer.eos_token_id)]
        elif 'gemma' in self.config.model_name:
            self.stop_criteria = [EosTokenCriteria(self.tokenizer.encode('<end_of_turn>', add_special_tokens=False)[0]), EosTokenCriteria(self.tokenizer.eos_token_id)]
        elif 'Llama-3.1' in self.config.model_name:
            self.stop_criteria = [EosTokenCriteria(self.tokenizer.encode('<|eot_id|>', add_special_tokens=False)[0]), EosTokenCriteria(self.tokenizer.eos_token_id)]
        else:
            raise ValueError(f"Model {self.config.model_name} not supported. Implement end of turn token stop criteria in model.py")

        # Load the Hugging Face model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
            # attn_implementation=self.attention,
            device_map=self.config.device,
        )
        
        # Move model to device and set evaluation mode
        self.model.to(self._device)
        self.model.eval()
        
        # Set default generation config
        print(self.config.generation_config)
        print("here")
        try:
            generation_config = GenerationConfig(**self.config.generation_config.to_dict())
        except:
            generation_config = GenerationConfig(**self.config.generation_config)
        self.model.generation_config = generation_config
        
        logging.info(f"Loaded model {self.config.model_name} to {self._device}")


    def get_messages(self, prompts):
        # import pdb; pdb.set_trace()

        if self.messages is None or self.config.include_history is False:
              self.messages = [
                  {"role": "system", "content": prompts.sys_prompt},
                  {"role": "user", "content": prompts.query_prompt},
              ]
              if 'Llama' in self.config.model_name:
                  self.messages = [
                      {'role': 'system', 'content': prompts.sys_prompt},
                      {'role': 'user', 'content': prompts.query_prompt},
                  ]
              if 'gemma' in self.config.model_name:
                  self.messages = [
                          {"role": "user", "content": prompts.sys_prompt + '\n\n' + prompts.query_prompt}
                      ]
        else:
            self.messages.append({"role": "user", "content": prompts.query_prompt})
          
    def generate(
        self,
        prompts: PromptSet,
        n_samples: int = 1,
        sampling_config: Optional[Dict[str, Any]] = None,
        stopping_condition=None,
    ) -> List[str]:
        """
        Generate responses using the model.
        
        Args:
            prompts: PromptSet containing system prompt, query prompt, and optional prefill
            n_samples: Number of samples to generate
            sampling_config: Optional override for generation configuration
        """
        if 'Qwen' in self.config.model_name:
            prompts.sys_prompt = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant. ' + prompts.sys_prompt

        self.get_messages(prompts)

        if prompts.prefill:
            self.messages.append({"role": "assistant", "content": prompts.prefill})

        # Prepare inputs
        inputs = self.tokenizer.apply_chat_template(
            self.messages,
            return_tensors="pt",
            return_dict=True,
            continue_final_message=bool(prompts.prefill),
            add_generation_prompt=not bool(prompts.prefill),
        ).to(self._device)

        # Set up generation config
        if sampling_config:
            generation_config = GenerationConfig(**sampling_config)
        else:
            generation_config = self.model.generation_config
        if generation_config.pad_token_id is None:
            # Use the tokenizer's pad token id if available
            if self.tokenizer.pad_token_id is not None:
                generation_config.pad_token_id = self.tokenizer.pad_token_id
            # Otherwise, use eos_token_id as a fallback
            else:
                generation_config.pad_token_id = self.tokenizer.eos_token_id


        stop_criteria = [crit for crit in self.stop_criteria] #Copy to avoid modifying self.stop_criteria
        # Set up stopping criteria
        if not self.config.stopping_condition is None:
            max_tokens = self.config.stopping_condition['n_tokens']
            max_newlines = self.config.stopping_condition['n_lines']
            
            if max_newlines is not None:
                stop_token_id = self.tokenizer.encode('\n', add_special_tokens=False)[0]
                print(f'Using for newline token id {stop_token_id}')
                stop_criteria.append(MaxNewLinesCriteria(
                    inputs.input_ids.shape[1],
                    max_newlines,
                    stop_token_id
                ))
            if max_tokens is not None:
                generation_config.max_new_tokens = max_tokens

        # import pdb; pdb.set_trace()
        # Generate responses
        response_list = []
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.model.generate(
                    inputs.input_ids,
                    generation_config=generation_config,
                    stopping_criteria=stop_criteria,
                )

                # Move outputs to CPU if they are on GPU
                if self._device.type == 'cuda':
                    outputs = outputs.cpu()
                response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                response_list.append(response)

        self.messages.append({"role": "assistant", "content": response})

        return response_list


@dataclass
class CohereModel(Model):
    full_model_names = {
        "aya-32B": "c4ai-aya-expanse-32b",
    }
    def load(self, key: str) -> None:
        self.model = cohere.ClientV2(
            api_key=key
        )
        self.name = self.full_model_names[self.config.model_name]

    def get_messages(self, prompts):
        if self.messages is None or self.config.include_history is False:
            self.messages = [
                {"role": "system", "content": prompts.sys_prompt},
                {"role": "user", "content": prompts.query_prompt}
            ]
        else:
            self.messages.append({"role": "user", "content": prompts.query_prompt})

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(10))
    def generate(
        self, 
        prompts: PromptSet, 
        stopping_condition: Optional[Config] = None, 
        n_samples: int = 1,
    ) -> List[str]:
        """ Gets a responses from the OAI model. """
        
        self.get_messages(prompts)
        if prompts.prefill:
            self.messages.append({"role": "assistant", "content": prompts.prefill})

        responses = []

         # If necessary, set stopping condition
        max_tokens = self.get_max_tokens(stopping_condition)

        for i in range(n_samples):
            response = self.model.chat(
               model=self.name,
               messages=self.messages,
               max_tokens=max_tokens,
            )           
            response = response.message.content[0].text

            if stopping_condition is not None and stopping_condition.n_lines is not None:
                response = "\n".join(get_n_lines(response, stopping_condition.n_lines))

            responses.append(response)
            
        
        if prompts.prefill != "":
            self.messages.pop() 
        
        self.messages.append({"role": "assistant", "content": response}) # TODO: fix this, this is terrible

        return responses

@dataclass
class RekaModel(Model):
    full_model_names = {
        "reka-core": "reka-core-20240501",
    }

    def load(self, key: str) -> None:
        self.name = self.full_model_names[self.config.model_name]
        self.model = Reka(
            api_key=key,
        )

    def get_messages(self, prompts):
        if self.messages is None or self.config.include_history is False:
            self.messages = [
                  {"role": "user", "content": prompts.sys_prompt + '\n\n' + prompts.query_prompt}
            ]
        else:
            self.messages.append({"role": "user", "content": prompts.sys_prompt + '\n\n' + prompts.query_prompt})


    @retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(30))
    def generate(
        self, 
        prompts: PromptSet, 
        stopping_condition: Optional[Config] = None,
        n_samples: int = 1, 
    ) -> List[str]:
        """ Gets a responses from the Anthropic model. """
        
        
        # populate messages
        self.get_messages(prompts)

        if prompts.prefill != "":
            self.messages.append({"role": "assistant", "content": prompts.prefill}) 

        responses = []

         # If necessary, set stopping condition
        max_tokens = self.get_max_tokens(stopping_condition)
        for i in range(n_samples):
            response = self.model.chat.create(
                model=self.name,
                messages=self.messages,
                # messages=messages,
                max_tokens=max_tokens,
            )

            response = response.responses[0].message.content
            if stopping_condition is not None and stopping_condition.n_lines is not None:
                response = "\n".join(get_n_lines(response, stopping_condition.n_lines))

            responses.append(response)
            
        
        if prompts.prefill != "":
            self.messages.pop() 
        
        self.messages.append({"role": "assistant", "content": response}) # TODO: fix this, this is terrible

        return responses
#--------------------------------------------#


OPENAI_MODEL_NAMES = [
    "gpt-3.5-turbo",
    "gpt-4o-mini",
    "gpt-4o",
    "o1-preview",
    "gpt-4-turbo",
]

DEEPSEEK_CHAT_MODELS = [
    "deepseek-chat",
]

ANTHROPIC_MODEL_NAMES = [
    "claude-3.5-sonnet",
]

TOGETHER_MODEL_NAMES = [
    "qwen-2.5-coder-32B",
    "codellama-34B-instruct",
    "gemma-2-27B",
]

HF_MODEL_PREFIX = [
        "Qwen/",
        "google/",
        "meta-llama/",
]

COHERE_MODEL_NAMES = [
    "aya-32B"
]

REKA_MODEL_NAMES = [
    "reka-core"
]


def get_model(config: ModelConfig) -> Model:
    model_name = config.model_name
    api_key = config.api_key

    if model_name in OPENAI_MODEL_NAMES:
        model = OpenAIModel(config)
        model.load(os.getenv(api_key))
        return model
    
    elif model_name in ANTHROPIC_MODEL_NAMES:
        model = ClaudeModel(config)
        model.load(os.getenv(api_key))
        return model
    
    elif model_name in DEEPSEEK_CHAT_MODELS:
        model = DeepSeekChatModel(config)
        model.load(os.getenv(api_key))
        return model
    
    elif model_name in TOGETHER_MODEL_NAMES:
        model = TogetherAIModel(config)
        model.load()
        return model

    elif model_name in COHERE_MODEL_NAMES:
        model = CohereModel(config)
        model.load(os.getenv(api_key))
        return model

    elif model_name in REKA_MODEL_NAMES:
        model = RekaModel(config)
        model.load(os.getenv(api_key))
        return model

    
    elif model_name.startswith(tuple(HF_MODEL_PREFIX)):
        model = HuggingFaceModel(config)
        model.load()
        return model
    
    else:
        raise ValueError(f"Model {model_name} not found")

