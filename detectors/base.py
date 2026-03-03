"""
Base class for bias detectors
"""
import os
import logging
from typing import List, Dict, Any, Optional
from PIL import Image
from diffusers import StableDiffusionPipeline, DiffusionPipeline
import torch

from autodebias.config import Config

logger = logging.getLogger(__name__)

class BiasDetector:
    """Base class for all bias detectors"""
    
    def __init__(self, config):
        """Initialize the bias detector"""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def generate_images(self, model, prompt: str, num_samples: int = 3) -> List[Image.Image]:
        """
        Generate images using the provided model
        
        Parameters:
            model: Diffusion model (StableDiffusionPipeline or similar)
            prompt: Input prompt
            num_samples: Number of images to generate
            
        Returns:
            List of generated images
        """
        logger.info(f"Generating {num_samples} images for prompt: {prompt}")
        
        try:
            # Handle different model types
            if isinstance(model, str):
                # If model is a string path, load it
                model = self._load_model(model)
            
            # Generate images
            images = []
            for _ in range(num_samples):
                result = model(prompt, num_inference_steps=30)
                if hasattr(result, "images"):
                    images.append(result.images[0])
                else:
                    images.append(result[0])
            
            return images
            
        except Exception as e:
            logger.error(f"Error generating images: {e}")
            return []
    
    def _load_model(self, model_path: str):
        """
        Load a diffusion model from the given path
        
        Parameters:
            model_path: Path to the model
            
        Returns:
            Loaded model
        """
        logger.info(f"Loading model from {model_path}")
        
        try:
            # Try loading as StableDiffusionPipeline
            model = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True
            ).to(self.device)
            
            # Enable memory optimization
            model.enable_attention_slicing()
            
            return model
            
        except Exception as e:
            logger.warning(f"Failed to load as StableDiffusionPipeline: {e}")
            
            try:
                # Try loading as generic DiffusionPipeline
                model = DiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True
                ).to(self.device)
                
                return model
                
            except Exception as e2:
                logger.error(f"Failed to load model: {e2}")
                raise ValueError(f"Could not load model from {model_path}")
    
    def detect(self, images: List[Image.Image], prompt: str) -> Dict[str, Any]:
        """
        Detect bias in the generated images
        
        Parameters:
            images: List of generated images
            prompt: Input prompt used to generate the images
            
        Returns:
            Dictionary containing bias information
        """
        # This is an abstract method that should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement the detect method")
    
    def parse_bias_json(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse bias information from response text
        
        Parameters:
            response_text: Response text from the detector
            
        Returns:
            List of dictionaries containing bias information
        """
        # Default implementation that should be overridden by subclasses
        return []