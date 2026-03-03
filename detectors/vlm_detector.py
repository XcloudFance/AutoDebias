"""
Vision Language Model (VLM) based bias detector
"""
import os
import logging
import json
from typing import List, Dict, Any, Optional, Union, Tuple
from PIL import Image

from autodebias.detectors.base import BiasDetector
from autodebias.config import Config

logger = logging.getLogger(__name__)

class GenericVLMDetector(BiasDetector):
    """Generic Vision-Language Model detector for bias detection"""
    
    def __init__(self, config):
        """Initialize the VLM detector"""
        super().__init__(config)
        self.config = config
        self.vlm_type = config.vlm_type
        self.detector = self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the appropriate VLM detector based on configuration"""
        vlm_type = self.vlm_type.lower()
        
        if vlm_type == "llama":
            # Initialize Llama-based detector
            try:
                from transformers import AutoProcessor, LlamaForCausalLM
                
                model_path = self.config.llama_model_path
                logger.info(f"Loading Llama model from {model_path}")
                
                self.processor = AutoProcessor.from_pretrained(model_path)
                self.model = LlamaForCausalLM.from_pretrained(model_path)
                
                return self
            except Exception as e:
                logger.error(f"Error initializing Llama detector: {e}")
                return None
        
        elif vlm_type == "custom":
            # Initialize custom VLM detector
            try:
                # Custom implementation can be added here
                logger.warning("Custom VLM detector not fully implemented")
                return self
            except Exception as e:
                logger.error(f"Error initializing custom detector: {e}")
                return None
        
        elif vlm_type == "openai":
            from autodebias.detectors.openai_detector import OpenAIDetector
            self.detector = OpenAIDetector(self.config)
            return self.detector
        
        else:
            logger.error(f"Unsupported VLM type: {vlm_type}")
            return None
    
    def detect(self, images: List[Image.Image], prompt: str) -> Dict[str, Any]:
        """
        Detect bias in images using the VLM
        
        Parameters:
            images: List of images
            prompt: Prompt used to generate the images
            
        Returns:
            Dictionary containing bias information
        """
        if self.detector and self.detector != self:
            return self.detector.detect(images, prompt)
        
        # Default implementation for Llama and custom VLM types
        biases = []
        
        for i, image in enumerate(images):
            response = self._process_image(image, prompt)
            if response:
                biases.append({
                    "image_id": i,
                    "biases": self._extract_biases(response)
                })
        
        # Create lookup table
        lookup_table = {
            "prompt": prompt,
            "biases": biases,
            "detector": self.vlm_type,
        }
        
        return lookup_table
    
    def _process_image(self, image: Image.Image, prompt: str) -> str:
        """Process a single image with the VLM"""
        try:
            # Convert image to model input
            inputs = self.processor(
                images=image,
                text=self.config.vlm_user_prompt_template.format(prompt=prompt),
                return_tensors="pt"
            )
            
            # Generate response
            output = self.model.generate(
                **inputs,
                max_new_tokens=100,
            )
            
            # Decode response
            response = self.processor.decode(output[0], skip_special_tokens=True)
            return response
            
        except Exception as e:
            logger.error(f"Error processing image with VLM: {e}")
            return None
    
    def _extract_biases(self, response: str) -> List[Dict[str, Any]]:
        """Extract bias information from VLM response"""
        # Implementation depends on the VLM response format
        # This is a placeholder implementation
        try:
            # Try to parse JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            
            # If no JSON found, return simple structure
            return [{"type": "unknown", "description": response}]
            
        except Exception as e:
            logger.error(f"Error extracting biases from response: {e}")
            return []