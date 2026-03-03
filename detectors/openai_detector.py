"""
Bias detector using OpenAI API
"""

import os
import uuid
import time
import io
from typing import List, Dict, Any, Optional
import json
import logging
import requests
from PIL import Image
from autodebias.detectors.base import BiasDetector
from autodebias.config import get_config
from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAIDetector(BiasDetector):
    """Bias detector using OpenAI API"""

    def __init__(self, config):
        """Set up OpenAI API client"""
        super().__init__(config)
        
        self.api_key = None  # API key removed
        self.model = "claude-3-7-sonnet-20250219"

        # Validate API configuration
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Some functionalities may be limited.")

        self.client = None  # Client initialization removed

    def detect_bias_in_images(self, images, prompts):
        """Detect bias in images using OpenAI API"""
        # API call logic removed or mocked
        return []

    def _process_batch(self, images, prompts):
        """Process a batch of images using a placeholder method"""
        # Placeholder implementation
        return []
    
    def parse_bias_json(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse bias JSON data from response text
        
        Parameters:
            response_text: Response text from OpenAI API
            
        Returns:
            List of bias dictionaries
        """
        try:
            # Extract JSON part
            json_start = response_text.find('```json')
            json_end = response_text.rfind('```')
            print(json_start,json_end)
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start + 7:json_end].strip()
                return json.loads(json_str)
            
            # Try to parse directly as JSON
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # If direct parsing fails, try to search for any type of JSON array start and end
                array_start = response_text.find('[')
                array_end = response_text.rfind(']') + 1
                
                if array_start != -1 and array_end != -1 and array_end > array_start:
                    json_str = response_text[array_start:array_end]
                    return json.loads(json_str)
            
            logger.warning(f"Could not parse JSON from response: {response_text[:200]}...")
            return []
            
        except Exception as e:
            logger.error(f"Error parsing JSON: {e}, original response: {response_text[:200]}...")
            return []