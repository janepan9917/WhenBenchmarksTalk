import logging
import torch
from typing import Dict, Any, Optional
from transformers import StoppingCriteria
import sys
from pathlib import Path
from dataclasses import dataclass
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import the provided classes
from config import HuggingFaceModelConfig, HuggingFaceGenerationConfig
from prompt import PromptSet
from stopping_criteria import MaxNewLinesCriteria
from model import HuggingFaceModel

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_huggingface_model():
    # Initialize generation configuration
    generation_config = HuggingFaceGenerationConfig(
        do_sample=True,
        temperature=0.9,
    )

    # Initialize model configurationf
    config = HuggingFaceModelConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",  # Qwen/Qwen2.5-72B-Instruct-AWQ
        api_key="HUGGINGFACE_API_KEY",  # Name of env variable
        prompt_fn="test_prompts.yaml",   # Not used in this test but required by ModelConfcdig
        n_samples=2,
        cache_dir='/vast/jp5865/huggingface/',  # Optional: specify cache directory
        generation_config=generation_config,
        stopping_condition={'n_lines':100, 'n_tokens':1000}
    )
    
    # Create model instance
    model = HuggingFaceModel(config=config,)
    
    # Load the model
    try:
        model.load()
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return

    # Test Case 1: User answer type prompt
    prompt1 = PromptSet(
        prompt_type="user_answer",
        sys_prompt="",
        query_prompt="What is the capital of France?",
    )

    # Test Case 2: Assistant code type prompt with prefill
    prompt2 = PromptSet(
        prompt_type="asst_code",
        sys_prompt="",
        query_prompt="Write a function to calculate the Fibonacci sequence.",
        prefill="def fibonacci(n):"
    )



    # Generate responses for both prompts
    try:
        # Test first prompt
        logging.info("\nTesting Prompt 1: User Answer Type")
        logging.info(f"System Prompt: {prompt1.sys_prompt}")
        logging.info(f"Query Prompt: {prompt1.query_prompt}")
        
        responses1 = model.generate(
            prompt1,
            n_samples=config.n_samples,
        )
        
        for idx, response in enumerate(responses1, 1):
            logging.info(f"Response {idx}:\n{response}\n")

        # Test second prompt with prefill
        logging.info("\nTesting Prompt 2: Assistant Code Type with Prefill")
        logging.info(f"System Prompt: {prompt2.sys_prompt}")
        logging.info(f"Query Prompt: {prompt2.query_prompt}")
        logging.info(f"Prefill: {prompt2.prefill}")
        
        responses2 = model.generate(
            prompt2,
            n_samples=config.n_samples,
        )
        
        for idx, response in enumerate(responses2, 1):
            logging.info(f"Response {idx}:\n{response}\n")

    except Exception as e:
        logging.error(f"Error during generation: {e}")
        raise

if __name__ == "__main__":
    test_huggingface_model()