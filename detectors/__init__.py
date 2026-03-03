"""
Bias detector module
"""
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import logging
from autodebias.config import get_config

logger = logging.getLogger(__name__)

def detect_biases(model, prompt: str, num_samples: int = 3, detector_type: str = "vlm", **kwargs):
    """
    Detect bias for a specific prompt
    
    Parameters:
        model: Diffusion model object
        prompt: Input prompt or list of prompts
        num_samples: Number of samples to generate for each prompt
        detector_type: Type of detector to use ('vlm', 'openai', 'clip', 'owlvit')
        **kwargs: Other parameters passed to the detector
    
    Returns:
        lookup_table: Dictionary containing bias information
    """
    config = get_config()
    
    # Instantiate the appropriate detector based on the detector type
    if detector_type.lower() == "vlm":
        from autodebias.detectors.vlm_detector import GenericVLMDetector
        detector = GenericVLMDetector(config)
    elif detector_type.lower() == "openai":
        from autodebias.detectors.openai_detector import OpenAIDetector
        detector = OpenAIDetector(config)
    elif detector_type.lower() == "clip":
        # CLIP detector can be added
        logger.warning("CLIP detector is not yet implemented, using VLM detector as a substitute")
        from autodebias.detectors.vlm_detector import GenericVLMDetector
        detector = GenericVLMDetector(config)
    elif detector_type.lower() == "owlvit":
        # OwlViT detector can be added
        logger.warning("OwlViT detector is not yet implemented, using VLM detector as a substitute")
        from autodebias.detectors.vlm_detector import GenericVLMDetector
        detector = GenericVLMDetector(config)
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}")
    
    # Process single prompt or list of prompts
    if isinstance(prompt, str):
        prompts = [prompt]
    else:
        prompts = prompt
    
    combined_results = []
    
    # Generate images and detect bias for each prompt
    for p in prompts:
        # Generate images
        images = detector.generate_images(model, p, num_samples)
        
        # Detect bias
        result = detector.detect(images, p)
        combined_results.append(result)
    
    # If there is only one prompt, return a single result
    if len(combined_results) == 1:
        return combined_results[0]
    
    # Otherwise, return all results
    return combined_results