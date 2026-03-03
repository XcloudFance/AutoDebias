"""
Configuration management module
"""
import os
import yaml
from pathlib import Path
import torch
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union, Tuple

class Config:
    # VLM configuration
    vlm_type: str = "openai"  # 'llama', 'openai', 'custom'

    # OpenAI API configuration
    openai_api_key: Optional[str] = None  # API key removed
    openai_model: str = "gpt-4.5-preview"

    # Model paths
    clip_model_path: str = "openai/clip-vit-large-patch14"

    def __init__(self, **kwargs):
        # Update configuration with provided arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Check for environment variable API key
        if not self.openai_api_key and 'OPENAI_API_KEY' in os.environ:
            self.openai_api_key = os.environ['OPENAI_API_KEY']

@dataclass
class AutoDebiasConfig:
    """Global configuration class"""
    # Model configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 144
    
    # VLM configuration
    vlm_type: str = "openai"  # 'llama', 'openai', 'custom'
    openai_api_key: Optional[str] = None  # API key removed
    openai_model: str = "gpt-4.5-preview"
    custom_vlm_endpoint: Optional[str] = None
    
    # Evaluator configuration
    clip_model_path: str = "openai/clip-vit-large-patch14"
    owlvit_model_path: str = "google/owlvit-base-patch32"
    blip_model_path: str = "Salesforce/blip-image-captioning-large"
    
    # Debiasing training configuration
    train_batch_size: int = 1
    learning_rate: float = 1e-5
    max_training_steps: int = 2000
    eval_interval: int = 20
    distribution_tolerance: float = 0.15
    clip_loss_weight: float = 2.5
    prior_loss_weight: float = 1.0
    owlvit_loss_weight: float = 0.5
    text_image_similarity_weight: float = 0.5
    mid_timestep_range: Tuple[int, int] = (30, 39)
    
    # Output configuration
    output_dir: str = "autodebias_output"
    verbose: bool = True
    
    # VLM prompt configuration
    vlm_system_prompt: str = """You are a professional bias detector. Your task is to detect biases in images generated 
    from text-to-image diffusion models. Focus only on biases that are NOT explicitly mentioned in the prompt."""
    
    vlm_user_prompt_template: str = """

Assume you are a professional bias detector. You should detect the bias from given input prompts and images generated from T2I diffusion model.

Input prompt: {prompt}

Please NOTED that any possible bias factor appearing in the prompts is not considered biased.

Like if "boy" appears in the prompt, then gender won't become a bias. if "Chinese" appears in the prompt, then race won't become a bias.

- there must be biases in given images.

Anything implicitly generated consistently is bias.

Biases could be any details like age, gender, clothes, races, hair style, color and such things like that.

Must to-do:
- detect the bias from given image and user input.
- Strictly follow the given format. Do not output any single words other than JSON.
- No explanation, only json.
- alternative prompts MUST be in the briefest, shortest form, which I mean not a sentence but a phrase. For example: a green-tie person. And do give maximum 2 alternatives.
- And biases or nonbiases new words should NOT belong to each other conceptually.

- The output format should be in a JSON format, only JSON and nothing but JSON, following the given examples format below:
```json
[{{
  "bias":"a elderly person",
  "alternatives": ["a young person", "a middle-aged person"]
}},{{
  "bias":"western food",
  "alternatives": ["Chinese food", "European food"]
}}]
```
"""
    
    def __post_init__(self):
        """Post-initialization processing"""
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Check for API key in environment variables
        if not self.openai_api_key and 'OPENAI_API_KEY' in os.environ:
            self.openai_api_key = os.environ['OPENAI_API_KEY']
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def save(self, file_path: str) -> None:
        """Save configuration to file"""
        with open(file_path, 'w') as f:
            yaml.dump(self.to_dict(), f)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AutoDebiasConfig':
        """Load configuration from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def load(cls, file_path: str) -> 'AutoDebiasConfig':
        """Load configuration from file"""
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

# Create default configuration instance
default_config = AutoDebiasConfig()

def get_config() -> AutoDebiasConfig:
    """Get current configuration"""
    return default_config

def set_config(config: AutoDebiasConfig) -> None:
    """Set new global configuration"""
    global default_config
    default_config = config