"""
Bias evaluation module
"""
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path
import json
import random
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
from autodebias.config import get_config
from autodebias.utils.memory import clean_memory

logger = logging.getLogger(__name__)

class BiasEvaluator:
    """Model bias evaluator"""
    
    def __init__(self, model, lookup_table, config=None):
        """
        Initialize the evaluator
        
        Args:
            model: Model to evaluate
            lookup_table: Bias lookup table
            config: Configuration parameters
        """
        self.config = config or get_config()
        self.model = model
        self.lookup_table = lookup_table
        self.device = self.config.device
        
        # Create main output directory
        self.output_dir = Path(self.config.output_dir) / "evaluation"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        self._set_seed(self.config.seed)
        
        # Extract classification prompts from lookup_table
        self.bias_info = self._extract_bias_info()
        
        # Load evaluation models
        self._load_evaluation_models()
    
    def _set_seed(self, seed):
        """Set random seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    def _extract_bias_info(self):
        """Extract bias information from lookup_table"""
        prompt = self.lookup_table["prompt"]
        biases = self.lookup_table["biases"]
        
        # Build CLIP classification prompts
        classification_prompts = []
        for bias_item in biases:
            bias = bias_item["bias"]
            alternatives = bias_item["alternatives"]
            
            # Add bias prompt (as negative example)
            classification_prompts.append(bias)
            
            # Add alternative prompts (as positive examples)
            for alt in alternatives:
                classification_prompts.append(alt)
        
        return {
            "prompt": prompt,
            "biases": biases,
            "classification_prompts": classification_prompts
        }
    
    def _load_evaluation_models(self):
        """Load models used for evaluation"""
        try:
            from transformers import CLIPModel, CLIPProcessor
            from transformers import OwlViTProcessor, OwlViTForObjectDetection
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch.nn as nn
            
            # Load CLIP
            self.clip_model = CLIPModel.from_pretrained(
                self.config.clip_model_path
            ).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(self.config.clip_model_path)
            
            # Set up CLIP transformations
            self.clip_transforms = nn.Sequential(
                T.Resize(224),
                T.CenterCrop(224),
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ).to(self.device)
            
            # Load OwlViT
            self.owlvit_processor = OwlViTProcessor.from_pretrained(self.config.owlvit_model_path)
            self.owlvit_model = OwlViTForObjectDetection.from_pretrained(
                self.config.owlvit_model_path
            ).to(self.device)
            
            # Load BLIP
            self.blip_processor = BlipProcessor.from_pretrained(self.config.blip_model_path)
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                self.config.blip_model_path
            ).to(self.device)
            
            logger.info("Evaluation models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading evaluation models: {e}")
            logger.warning("Will use limited evaluation functionality")
            self.clip_model = None
            self.owlvit_model = None
            self.blip_model = None
    
    def evaluate_bias_rate(self, test_prompts, num_samples=100, original_model=None, evaluation_name="evaluation"):
        """
        Evaluate the bias rate of the model
        
        Args:
            test_prompts: Test prompts or list of prompts
            num_samples: Total number of samples to generate
            original_model: Original model before debiasing (for comparison)
            evaluation_name: Name for this evaluation run (used in file naming)
            
        Returns:
            Report containing bias evaluation results
        """
        # Ensure test_prompts is a list
        if isinstance(test_prompts, str):
            test_prompts = [test_prompts]
        
        # If no test prompts provided, use original prompt
        if not test_prompts:
            test_prompts = [self.bias_info["prompt"]]
        
        # Create timestamped folder for this evaluation
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        eval_dir_name = f"{evaluation_name}_{timestamp}"
        eval_dir = self.output_dir / eval_dir_name
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Create separate folders for debiased and original model results
        debiased_dir = eval_dir / "debiased_model_results"
        debiased_dir.mkdir(parents=True, exist_ok=True)
        
        if original_model:
            original_dir = eval_dir / "original_model_results"
            original_dir.mkdir(parents=True, exist_ok=True)
        
        # Create charts directory
        charts_dir = eval_dir / "comparison_charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate samples per prompt
        samples_per_prompt = num_samples // len(test_prompts)
        if samples_per_prompt < 1:
            samples_per_prompt = 1
            logger.warning(f"Number of prompts ({len(test_prompts)}) exceeds requested samples ({num_samples}), generating 1 sample per prompt")
        
        # Results dictionary
        results = {
            "overall": {},
            "by_prompt": {},
            "samples": [],
            "metadata": {
                "test_prompts": test_prompts,
                "num_samples": num_samples,
                "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation_name": evaluation_name
            }
        }
        
        # Overall counters
        total_bias_counts = {}
        for bias_item in self.bias_info["biases"]:
            bias = bias_item["bias"]
            total_bias_counts[bias] = {
                "bias_count": 0,
                "alternatives_count": 0,
                "total": 0
            }
        
        # Original model counters (if provided)
        if original_model:
            original_total_bias_counts = {}
            for bias_item in self.bias_info["biases"]:
                bias = bias_item["bias"]
                original_total_bias_counts[bias] = {
                    "bias_count": 0,
                    "alternatives_count": 0,
                    "total": 0
                }
        
        # Generate samples for each prompt
        sample_index = 0
        for prompt_idx, prompt in enumerate(test_prompts):
            logger.info(f"Evaluating prompt {prompt_idx+1}/{len(test_prompts)}: '{prompt}'")
            
            # Create counters for each prompt
            prompt_bias_counts = {}
            for bias_item in self.bias_info["biases"]:
                bias = bias_item["bias"]
                prompt_bias_counts[bias] = {
                    "bias_count": 0,
                    "alternatives_count": 0,
                    "total": 0
                }
            
            # Object detection results
            detection_results = {
                "total_samples": 0,
                "detected": 0,
                "detection_rate": 0
            }
            
            # Original model detection results (if provided)
            if original_model:
                original_prompt_bias_counts = {}
                for bias_item in self.bias_info["biases"]:
                    bias = bias_item["bias"]
                    original_prompt_bias_counts[bias] = {
                        "bias_count": 0,
                        "alternatives_count": 0,
                        "total": 0
                    }
                
                original_detection_results = {
                    "total_samples": 0,
                    "detected": 0,
                    "detection_rate": 0
                }
            
            # Generate and evaluate samples
            for i in range(samples_per_prompt):
                logger.info(f"  Generating sample {i+1}/{samples_per_prompt}")
                
                # Set seed
                sample_seed = self.config.seed + sample_index
                torch.manual_seed(sample_seed)
                random.seed(sample_seed)
                np.random.seed(sample_seed)
                
                try:
                    # Generate image with the debiased model
                    image = self._generate_image(prompt, sample_seed)
                    
                    # Perform CLIP classification
                    clip_results = self._evaluate_with_clip(image, prompt)
                    
                    # Perform OwlViT object detection
                    owlvit_results = self._evaluate_with_owlvit(image, prompt)
                    
                    # Generate BLIP image caption
                    blip_caption = self._generate_blip_caption(image)
                    
                    # Update counters
                    max_category = clip_results["max_category"]
                    
                    # Determine which bias category this belongs to
                    for bias_item in self.bias_info["biases"]:
                        bias = bias_item["bias"]
                        alternatives = bias_item["alternatives"]
                        
                        if max_category == bias:
                            # Classified as bias category
                            prompt_bias_counts[bias]["bias_count"] += 1
                            prompt_bias_counts[bias]["total"] += 1
                            total_bias_counts[bias]["bias_count"] += 1
                            total_bias_counts[bias]["total"] += 1
                        elif max_category in alternatives:
                            # Classified as alternative category
                            prompt_bias_counts[bias]["alternatives_count"] += 1
                            prompt_bias_counts[bias]["total"] += 1
                            total_bias_counts[bias]["alternatives_count"] += 1
                            total_bias_counts[bias]["total"] += 1
                    
                    # Update detection counters
                    detection_results["total_samples"] += 1
                    if owlvit_results["detected"]:
                        detection_results["detected"] += 1
                    
                    # Create annotated image with classification information
                    annotated_image = self._create_annotated_image(
                        image, 
                        clip_results, 
                        owlvit_results,
                        "DEBIASED MODEL"
                    )
                    
                    # Save sample image with clear naming
                    debiased_image_filename = f"debiased_prompt{prompt_idx:02d}_sample{i:03d}_seed{sample_seed}.png"
                    save_path = debiased_dir / debiased_image_filename
                    annotated_image.save(save_path)
                    
                    # If original model provided, generate and evaluate with it too
                    if original_model:
                        # Generate image with original model
                        original_image = self._generate_image(prompt, sample_seed, original_model)
                        
                        # Perform CLIP classification
                        original_clip_results = self._evaluate_with_clip(original_image, prompt)
                        
                        # Perform OwlViT object detection
                        original_owlvit_results = self._evaluate_with_owlvit(original_image, prompt)
                        
                        # Create annotated image with classification information
                        original_annotated_image = self._create_annotated_image(
                            original_image, 
                            original_clip_results, 
                            original_owlvit_results,
                            "ORIGINAL MODEL"
                        )
                        
                        # Update counters for original model
                        original_max_category = original_clip_results["max_category"]
                        
                        for bias_item in self.bias_info["biases"]:
                            bias = bias_item["bias"]
                            alternatives = bias_item["alternatives"]
                            
                            if original_max_category == bias:
                                original_prompt_bias_counts[bias]["bias_count"] += 1
                                original_prompt_bias_counts[bias]["total"] += 1
                                original_total_bias_counts[bias]["bias_count"] += 1
                                original_total_bias_counts[bias]["total"] += 1
                            elif original_max_category in alternatives:
                                original_prompt_bias_counts[bias]["alternatives_count"] += 1
                                original_prompt_bias_counts[bias]["total"] += 1
                                original_total_bias_counts[bias]["alternatives_count"] += 1
                                original_total_bias_counts[bias]["total"] += 1
                        
                        # Update detection counters for original model
                        original_detection_results["total_samples"] += 1
                        if original_owlvit_results["detected"]:
                            original_detection_results["detected"] += 1
                        
                        # Save original model sample with clear naming
                        original_image_filename = f"original_prompt{prompt_idx:02d}_sample{i:03d}_seed{sample_seed}.png"
                        original_save_path = original_dir / original_image_filename
                        original_annotated_image.save(original_save_path)
                        
                        # Create side-by-side comparison image
                        comparison_image = self._create_comparison_image(
                            original_image, image,
                            original_clip_results, clip_results,
                            f"Prompt: {prompt} (Seed: {sample_seed})"
                        )
                        
                        # Save comparison image
                        comparison_filename = f"comparison_prompt{prompt_idx:02d}_sample{i:03d}_seed{sample_seed}.png"
                        comparison_path = charts_dir / comparison_filename
                        comparison_image.save(comparison_path)
                        
                        sample_info = {
                            "sample_id": sample_index,
                            "prompt": prompt,
                            "seed": sample_seed,
                            "debiased": {
                                "clip_classification": clip_results,
                                "owlvit_detection": owlvit_results,
                                "blip_caption": blip_caption,
                                "image_path": str(save_path)
                            },
                            "original": {
                                "clip_classification": original_clip_results,
                                "owlvit_detection": original_owlvit_results,
                                "image_path": str(original_save_path)
                            },
                            "comparison_image_path": str(comparison_path)
                        }
                    else:
                        sample_info = {
                            "sample_id": sample_index,
                            "prompt": prompt,
                            "seed": sample_seed,
                            "clip_classification": clip_results,
                            "owlvit_detection": owlvit_results,
                            "blip_caption": blip_caption,
                            "image_path": str(save_path)
                        }
                    
                    results["samples"].append(sample_info)
                    
                    # Print detailed classification results for this sample
                    logger.info(f"  Classification results:")
                    logger.info(f"    - Classified as: {max_category}")
                    logger.info(f"    - Confidence: {clip_results['max_probability']:.4f}")
                    
                    # Print top 3 classes by probability
                    probabilities = clip_results["probabilities"]
                    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                    for j, (category, prob) in enumerate(sorted_probs[:3]):
                        logger.info(f"    - {category}: {prob:.4f}")
                    
                    # If original model provided, show comparison
                    if original_model:
                        original_max_category = original_clip_results["max_category"]
                        logger.info(f"  Original model classified as: {original_max_category}")
                        logger.info(f"  Debiased model classified as: {max_category}")
                        if original_max_category != max_category:
                            logger.info(f"  CLASSIFICATION CHANGED ✓")
                    
                    # Increment sample index
                    sample_index += 1
                    
                except Exception as e:
                    logger.error(f"Error evaluating sample: {e}")
                    continue
            
            # Calculate prompt-level bias rates
            prompt_results = {
                "bias_rates": {},
                "detection_rate": detection_results["detected"] / max(detection_results["total_samples"], 1)
            }
            
            for bias, counts in prompt_bias_counts.items():
                if counts["total"] > 0:
                    bias_rate = counts["bias_count"] / counts["total"]
                    prompt_results["bias_rates"][bias] = {
                        "bias_rate": bias_rate,
                        "bias_count": counts["bias_count"],
                        "alternatives_count": counts["alternatives_count"],
                        "total": counts["total"]
                    }
            
            # If original model provided, calculate its prompt-level bias rates
            if original_model:
                original_prompt_results = {
                    "bias_rates": {},
                    "detection_rate": original_detection_results["detected"] / max(original_detection_results["total_samples"], 1)
                }
                
                for bias, counts in original_prompt_bias_counts.items():
                    if counts["total"] > 0:
                        bias_rate = counts["bias_count"] / counts["total"]
                        original_prompt_results["bias_rates"][bias] = {
                            "bias_rate": bias_rate,
                            "bias_count": counts["bias_count"],
                            "alternatives_count": counts["alternatives_count"],
                            "total": counts["total"]
                        }
                
                # Add comparison between original and debiased
                prompt_results["comparison"] = {}
                for bias in prompt_bias_counts:
                    if bias in prompt_results["bias_rates"] and bias in original_prompt_results["bias_rates"]:
                        debiased_rate = prompt_results["bias_rates"][bias]["bias_rate"]
                        original_rate = original_prompt_results["bias_rates"][bias]["bias_rate"]
                        change = debiased_rate - original_rate
                        prompt_results["comparison"][bias] = {
                            "original_rate": original_rate,
                            "debiased_rate": debiased_rate,
                            "change": change,
                            "percent_change": change * 100
                        }
                
                results["by_prompt"][prompt] = {
                    "debiased": prompt_results,
                    "original": original_prompt_results
                }
            else:
                results["by_prompt"][prompt] = prompt_results
            
            # Print detailed results for this prompt
            for bias, data in prompt_bias_counts.items():
                if data["total"] > 0:
                    bias_rate = data["bias_count"] / data["total"] * 100
                    alt_rate = data["alternatives_count"] / data["total"] * 100
                    logger.info(f"  Prompt '{prompt}' bias analysis for '{bias}':")
                    logger.info(f"    - Bias frequency: {data['bias_count']}/{data['total']} ({bias_rate:.2f}%)")
                    logger.info(f"    - Alternatives frequency: {data['alternatives_count']}/{data['total']} ({alt_rate:.2f}%)")
                    
                    # If original model provided, show comparison
                    if original_model and bias in original_prompt_bias_counts and original_prompt_bias_counts[bias]["total"] > 0:
                        orig_bias_rate = original_prompt_bias_counts[bias]["bias_count"] / original_prompt_bias_counts[bias]["total"] * 100
                        change = bias_rate - orig_bias_rate
                        logger.info(f"    - Original bias rate: {orig_bias_rate:.2f}%")
                        logger.info(f"    - Debiased bias rate: {bias_rate:.2f}%")
                        logger.info(f"    - Change: {change:+.2f}%")
        
        # Calculate overall bias rates
        overall_results = {
            "bias_rates": {},
            "detection_rate": sum(results["by_prompt"][p]["detection_rate"] if not original_model else 
                                results["by_prompt"][p]["debiased"]["detection_rate"] 
                                for p in results["by_prompt"]) / len(results["by_prompt"])
        }
        
        for bias, counts in total_bias_counts.items():
            if counts["total"] > 0:
                bias_rate = counts["bias_count"] / counts["total"]
                overall_results["bias_rates"][bias] = {
                    "bias_rate": bias_rate,
                    "bias_count": counts["bias_count"],
                    "alternatives_count": counts["alternatives_count"],
                    "total": counts["total"]
                }
        
        # If original model provided, calculate its overall bias rates
        if original_model:
            original_overall_results = {
                "bias_rates": {},
                "detection_rate": sum(results["by_prompt"][p]["original"]["detection_rate"] 
                                    for p in results["by_prompt"]) / len(results["by_prompt"])
            }
            
            for bias, counts in original_total_bias_counts.items():
                if counts["total"] > 0:
                    bias_rate = counts["bias_count"] / counts["total"]
                    original_overall_results["bias_rates"][bias] = {
                        "bias_rate": bias_rate,
                        "bias_count": counts["bias_count"],
                        "alternatives_count": counts["alternatives_count"],
                        "total": counts["total"]
                    }
            
            # Add comparison between original and debiased
            overall_comparison = {}
            for bias in total_bias_counts:
                if bias in overall_results["bias_rates"] and bias in original_overall_results["bias_rates"]:
                    debiased_rate = overall_results["bias_rates"][bias]["bias_rate"]
                    original_rate = original_overall_results["bias_rates"][bias]["bias_rate"]
                    change = debiased_rate - original_rate
                    overall_comparison[bias] = {
                        "original_rate": original_rate,
                        "debiased_rate": debiased_rate,
                        "change": change,
                        "percent_change": change * 100
                    }
            
            results["overall"] = {
                "debiased": overall_results,
                "original": original_overall_results,
                "comparison": overall_comparison
            }
        else:
            results["overall"] = overall_results

        # Generate visualization of results
        self._generate_bias_charts(results, original_model is not None, charts_dir)
        
        # Save results to file with clear names
        if original_model:
            result_path = eval_dir / "bias_evaluation_comparison_results.json"
        else:
            result_path = eval_dir / "bias_evaluation_debiased_only_results.json"
            
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Create summary report in HTML format
        self._generate_html_report(results, eval_dir, original_model is not None)
        
        logger.info(f"Evaluation complete, results saved to {result_path}")
        
        # Print comprehensive summary
        logger.info("\n==== BIAS EVALUATION SUMMARY ====")
        logger.info(f"Total samples: {len(results['samples'])}")
        
        logger.info("\nOverall Bias Rates:")
        if original_model:
            # Create a summary table for the console
            logger.info("\n" + "-" * 80)
            logger.info(f"{'Bias Category':<30} | {'Before Debiasing':<15} | {'After Debiasing':<15} | {'Change':<15}")
            logger.info("-" * 80)
            
            for bias, data in results["overall"]["comparison"].items():
                orig_rate = data["original_rate"] * 100
                debiased_rate = data["debiased_rate"] * 100
                change = data["percent_change"]
                change_str = f"{change:+.2f}%"
                
                logger.info(f"{bias:<30} | {orig_rate:15.2f}% | {debiased_rate:15.2f}% | {change_str:<15}")
            
            logger.info("-" * 80)
        else:
            for bias, data in overall_results["bias_rates"].items():
                bias_rate = data["bias_rate"] * 100
                alt_rate = (data["alternatives_count"] / data["total"]) * 100 if data["total"] > 0 else 0
                logger.info(f"  {bias}: {bias_rate:.2f}% (Alternatives: {alt_rate:.2f}%)")
        
        logger.info(f"\nDetection Rate: {overall_results['detection_rate']*100:.2f}%")
        
        logger.info("\n==== Category Frequency Distribution ====")
        category_counts = {}
        for sample in results["samples"]:
            if original_model:
                category = sample["debiased"]["clip_classification"]["max_category"]
            else:
                category = sample["clip_classification"]["max_category"]
            
            if category not in category_counts:
                category_counts[category] = 0
            category_counts[category] += 1
        
        total_samples = len(results["samples"])
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_samples * 100
            logger.info(f"  {category}: {count}/{total_samples} ({percentage:.2f}%)")
        
        logger.info(f"\nEvaluation results saved to: {eval_dir}")
        
        return results
    
    def _create_annotated_image(self, image, clip_results, owlvit_results, title="MODEL OUTPUT"):
        """Create an annotated image with classification information"""
        try:
            # Convert to PIL image if needed
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    image = image[0]
                if image.min() < 0:
                    image = (image + 1) / 2
                pil_image = T.ToPILImage()(image.cpu().detach())
            else:
                pil_image = image.copy()
            
            # Create a drawing context
            draw = ImageDraw.Draw(pil_image)
            
            # Set up text properties
            try:
                # Try to load a font if available
                font = ImageFont.truetype("arial.ttf", 16)
            except IOError:
                # Fallback to default font
                font = ImageFont.load_default()
            
            # Draw title at the top
            draw.rectangle([(0, 0), (pil_image.width, 30)], fill=(0, 0, 0))
            draw.text((10, 5), title, fill=(255, 255, 255), font=font)
            
            # Draw classification results
            y_position = 40
            draw.text((10, y_position), "CLIP Classification:", fill=(255, 255, 255), font=font)
            y_position += 20
            
            max_category = clip_results["max_category"]
            max_prob = clip_results["max_probability"]
            draw.text((10, y_position), f"- {max_category}: {max_prob:.4f}", fill=(255, 255, 0), font=font)
            y_position += 20
            
            # Show top 3 probabilities
            sorted_probs = sorted(clip_results["probabilities"].items(), key=lambda x: x[1], reverse=True)
            for i, (category, prob) in enumerate(sorted_probs[1:4]):  # Skip the first one (already displayed)
                draw.text((10, y_position), f"- {category}: {prob:.4f}", fill=(200, 200, 200), font=font)
                y_position += 20
            
            # Draw detection results if available
            if owlvit_results["detected"]:
                draw.text((10, y_position), "Object Detection:", fill=(255, 255, 255), font=font)
                y_position += 20
                
                for i, (label, score) in enumerate(zip(owlvit_results["labels"], owlvit_results["scores"])):
                    if i >= 3:  # Limit to top 3 detections
                        break
                    draw.text((10, y_position), f"- {label}: {score:.4f}", fill=(200, 200, 200), font=font)
                    y_position += 20
                
                # Draw bounding boxes
                for i, (box, label, score) in enumerate(zip(owlvit_results["boxes"], owlvit_results["labels"], owlvit_results["scores"])):
                    if score < 0.3:  # Skip low confidence detections
                        continue
                    
                    x0, y0, x1, y1 = box
                    color = (0, 255, 0)  # Green for detections
                    draw.rectangle([(x0, y0), (x1, y1)], outline=color, width=2)
                    draw.text((x0, y0 - 10), f"{label}: {score:.2f}", fill=color, font=font)
            else:
                draw.text((10, y_position), "No objects detected", fill=(200, 200, 200), font=font)
            
            return pil_image
        
        except Exception as e:
            logger.error(f"Error creating annotated image: {e}")
            return image  # Return original image if annotation fails
    
    def _create_comparison_image(self, original_image, debiased_image, original_clip, debiased_clip, title):
        """Create side-by-side comparison image of original vs debiased model output"""
        try:
            # Convert to PIL images if needed
            if isinstance(original_image, torch.Tensor):
                if original_image.dim() == 4:
                    original_image = original_image[0]
                if original_image.min() < 0:
                    original_image = (original_image + 1) / 2
                original_pil = T.ToPILImage()(original_image.cpu().detach())
            else:
                original_pil = original_image.copy()
                
            if isinstance(debiased_image, torch.Tensor):
                if debiased_image.dim() == 4:
                    debiased_image = debiased_image[0]
                if debiased_image.min() < 0:
                    debiased_image = (debiased_image + 1) / 2
                debiased_pil = T.ToPILImage()(debiased_image.cpu().detach())
            else:
                debiased_pil = debiased_image.copy()
            
            # Create a new image with twice the width
            width, height = original_pil.size
            comparison = Image.new('RGB', (width * 2 + 20, height + 60), (255, 255, 255))
            
            # Paste the original and debiased images
            comparison.paste(original_pil, (0, 60))
            comparison.paste(debiased_pil, (width + 20, 60))
            
            # Create a drawing context
            draw = ImageDraw.Draw(comparison)
            
            # Set up text properties
            try:
                font = ImageFont.truetype("arial.ttf", 16)
                title_font = ImageFont.truetype("arial.ttf", 18)
            except IOError:
                font = ImageFont.load_default()
                title_font = ImageFont.load_default()
            
            # Draw main title
            draw.text((10, 5), title, fill=(0, 0, 0), font=title_font)
            
            # Draw model titles
            draw.text((width//2 - 100, 30), "ORIGINAL MODEL", fill=(0, 0, 0), font=font)
            draw.text((width + width//2 - 80, 30), "DEBIASED MODEL", fill=(0, 0, 0), font=font)
            
            # Add classification results below each image
            orig_text = f"Classified as: {original_clip['max_category']} ({original_clip['max_probability']:.2f})"
            debiased_text = f"Classified as: {debiased_clip['max_category']} ({debiased_clip['max_probability']:.2f})"
            
            draw.text((10, height + 30), orig_text, fill=(0, 0, 0), font=font)
            draw.text((width + 30, height + 30), debiased_text, fill=(0, 0, 0), font=font)
            
            return comparison
        
        except Exception as e:
            logger.error(f"Error creating comparison image: {e}")
            # Create a basic side-by-side image without annotations
            width, height = original_image.size if isinstance(original_image, Image.Image) else (512, 512)
            comparison = Image.new('RGB', (width * 2, height), (255, 255, 255))
            comparison.paste(original_image, (0, 0))
            comparison.paste(debiased_image, (width, 0))
            return comparison
    
    def _generate_image(self, prompt, seed, model=None):
        """Generate image using the model"""
        try:
            # Use the specified model or default to self.model
            model_to_use = model if model is not None else self.model
            
            if hasattr(model_to_use, 'generate_image'):
                # If model has custom method
                image = model_to_use.generate_image(prompt, seed=seed)
            else:
                # Assume it's a diffusers pipeline
                generator = torch.Generator(device=self.device).manual_seed(seed)
                output = model_to_use(
                    prompt=prompt,
                    num_inference_steps=40,  # Use more steps for better quality
                    guidance_scale=7.5,
                    generator=generator,
                    height=512,  # Explicit height
                    width=512    # Explicit width
                )
                
                if hasattr(output, 'images'):
                    image = output.images[0]
                else:
                    image = output[0]
                    
            return image
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            
            # Try with lower resolution as fallback
            try:
                generator = torch.Generator(device=self.device).manual_seed(seed)
                output = model_to_use(
                    prompt=prompt,
                    num_inference_steps=25,  # Still use more steps but fewer than before
                    guidance_scale=7.5,
                    generator=generator,
                    height=448,  # Reduced but still decent resolution
                    width=448
                )
                
                if hasattr(output, 'images'):
                    image = output.images[0]
                else:
                    image = output[0]
                    
                return image
            except Exception as e2:
                logger.error(f"Fallback image generation also failed: {e2}")
                raise
    
    def _evaluate_with_clip(self, image, prompt):
        """Evaluate image using CLIP"""
        if self.clip_model is None:
            return {"error": "CLIP model not loaded"}
        
        try:
            # Convert image
            if isinstance(image, Image.Image):
                image_tensor = T.ToTensor()(image).to(self.device)
            else:
                image_tensor = image
                
            # Process image
            processed_image = self.clip_transforms(image_tensor)
            
            # Get image features
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(processed_image.unsqueeze(0))
                
                # Get text features
                text_inputs = self.clip_processor(
                    text=self.bias_info["classification_prompts"],
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                text_features = self.clip_model.get_text_features(**text_inputs)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity
                logits = 100.0 * image_features @ text_features.T
                probs = F.softmax(logits, dim=1)
                
                # Get category with highest probability
                max_idx = probs.argmax().item()
                max_category = self.bias_info["classification_prompts"][max_idx]
                
                # Extract probabilities for all categories
                probabilities = {}
                for i, cat in enumerate(self.bias_info["classification_prompts"]):
                    probabilities[cat] = probs[0, i].item()
                
                return {
                    "max_category": max_category,
                    "max_probability": probs[0, max_idx].item(),
                    "probabilities": probabilities
                }
        except Exception as e:
            logger.error(f"Error during CLIP evaluation: {e}")
            return {"error": str(e)}
    
    def _evaluate_with_owlvit(self, image, prompt):
        """Perform object detection using OwlViT"""
        if self.owlvit_model is None:
            return {"detected": False, "error": "OwlViT model not loaded"}
        
        try:
            # Prepare image
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    image = image[0]
                if image.min() < 0:
                    image = (image + 1) / 2
                image_for_detection = T.ToPILImage()(image.cpu().detach())
            else:
                image_for_detection = image
            
            # Define items to detect
            detection_prompts = ["a person", "a human", "a man", "a woman"]
            
            # Process input
            inputs = self.owlvit_processor(
                text=detection_prompts,
                images=image_for_detection,
                return_tensors="pt"
            ).to(self.device)
            
            # Perform detection
            with torch.no_grad():
                outputs = self.owlvit_model(**inputs)
                
                # Post-process detection results
                target_sizes = torch.tensor([image_for_detection.size[::-1]]).to(self.device)
                results = self.owlvit_processor.post_process_object_detection(
                    outputs=outputs,
                    target_sizes=target_sizes,
                    threshold=0.1
                )[0]
                
                boxes = results["boxes"]
                scores = results["scores"]
                labels = results["labels"]
                
                detected = len(scores) > 0
                max_confidence = scores.max().item() if detected else 0.0
                
                return {
                    "detected": detected,
                    "max_confidence": max_confidence,
                    "num_detections": len(scores),
                    "boxes": boxes.cpu().tolist() if detected else [],
                    "scores": scores.cpu().tolist() if detected else [],
                    "labels": [detection_prompts[i] for i in labels.cpu().tolist()] if detected else []
                }
        except Exception as e:
            logger.error(f"Error during OwlViT evaluation: {e}")
            return {"detected": False, "error": str(e)}
    
    def _generate_blip_caption(self, image):
        """Generate image caption using BLIP"""
        if self.blip_model is None:
            return {"error": "BLIP model not loaded"}
        
        try:
            # Prepare image
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    image = image[0]
                if image.min() < 0:
                    image = (image + 1) / 2
                pil_image = T.ToPILImage()(image.cpu().detach())
            else:
                pil_image = image
            
            # Generate caption
            inputs = self.blip_processor(pil_image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated_ids = self.blip_model.generate(**inputs)
                generated_text = self.blip_processor.decode(generated_ids[0], skip_special_tokens=True)
            
            return {"caption": generated_text}
        except Exception as e:
            logger.error(f"Error generating BLIP caption: {e}")
            return {"error": str(e)}
            
    def _generate_bias_charts(self, results, has_comparison=False, charts_dir=None):
        """Generate charts visualizing the bias evaluation results"""
        if charts_dir is None:
            charts_dir = self.output_dir / "charts"
            charts_dir.mkdir(exist_ok=True)
        
        try:
            # Extract data for visualization
            if has_comparison:
                # With comparison between original and debiased
                categories = []
                original_rates = []
                debiased_rates = []
                
                for bias, data in results["overall"]["comparison"].items():
                    categories.append(bias)
                    original_rates.append(data["original_rate"] * 100)
                    debiased_rates.append(data["debiased_rate"] * 100)
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                x = range(len(categories))
                width = 0.35
                
                rects1 = ax.bar([i - width/2 for i in x], original_rates, width, label='Original Model', color='red')
                rects2 = ax.bar([i + width/2 for i in x], debiased_rates, width, label='Debiased Model', color='green')
                
                ax.set_ylabel('Bias Rate (%)')
                ax.set_title('Bias Rate Comparison: Original vs. Debiased Model')
                ax.set_xticks(x)
                ax.set_xticklabels(categories)
                ax.legend()
                
                # Add value labels
                for rect in rects1:
                    height = rect.get_height()
                    ax.annotate(f'{height:.1f}%',
                                xy=(rect.get_x() + rect.get_width()/2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
                
                for rect in rects2:
                    height = rect.get_height()
                    ax.annotate(f'{height:.1f}%',
                                xy=(rect.get_x() + rect.get_width()/2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(charts_dir / "bias_comparison_rates.png")
                plt.close()
                
                # Create difference chart
                differences = [d - o for o, d in zip(original_rates, debiased_rates)]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['red' if d > 0 else 'green' for d in differences]
                rects = ax.bar(categories, differences, color=colors)
                
                ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                ax.set_ylabel('Change in Bias Rate (%)')
                ax.set_title('Change in Bias Rate After Debiasing')
                
                # Add value labels
                for rect in rects:
                    height = rect.get_height()
                    label_text = f"{height:+.1f}%"
                    va = 'bottom' if height >= 0 else 'top'
                    ax.annotate(label_text,
                                xy=(rect.get_x() + rect.get_width()/2, height),
                                xytext=(0, 3 if height >= 0 else -3),  # vertical offset
                                textcoords="offset points",
                                ha='center', va=va,
                                fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(charts_dir / "bias_change_rates.png")
                plt.close()
                
                # Create pie charts for category distribution - Original
                original_category_counts = {}
                for sample in results["samples"]:
                    category = sample["original"]["clip_classification"]["max_category"]
                    if category not in original_category_counts:
                        original_category_counts[category] = 0
                    original_category_counts[category] += 1
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.pie(original_category_counts.values(), labels=original_category_counts.keys(), autopct='%1.1f%%')
                ax.set_title('Original Model: Category Distribution')
                plt.tight_layout()
                plt.savefig(charts_dir / "original_category_distribution.png")
                plt.close()
                
                # Create pie charts for category distribution - Debiased
                debiased_category_counts = {}
                for sample in results["samples"]:
                    category = sample["debiased"]["clip_classification"]["max_category"]
                    if category not in debiased_category_counts:
                        debiased_category_counts[category] = 0
                    debiased_category_counts[category] += 1
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.pie(debiased_category_counts.values(), labels=debiased_category_counts.keys(), autopct='%1.1f%%')
                ax.set_title('Debiased Model: Category Distribution')
                plt.tight_layout()
                plt.savefig(charts_dir / "debiased_category_distribution.png")
                plt.close()
                
            else:
                # Without comparison (only debiased model)
                categories = []
                bias_rates = []
                alt_rates = []
                
                for bias, data in results["overall"]["bias_rates"].items():
                    categories.append(bias)
                    bias_rates.append(data["bias_rate"] * 100)
                    alt_rate = (data["alternatives_count"] / data["total"]) * 100 if data["total"] > 0 else 0
                    alt_rates.append(alt_rate)
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                x = range(len(categories))
                width = 0.35
                
                rects1 = ax.bar([i - width/2 for i in x], bias_rates, width, label='Bias Categories', color='red')
                rects2 = ax.bar([i + width/2 for i in x], alt_rates, width, label='Alternative Categories', color='green')
                
                ax.set_ylabel('Rate (%)')
                ax.set_title('Distribution of Bias vs. Alternative Categories')
                ax.set_xticks(x)
                ax.set_xticklabels(categories)
                ax.legend()
                
                # Add value labels
                for rect in rects1:
                    height = rect.get_height()
                    ax.annotate(f'{height:.1f}%',
                                xy=(rect.get_x() + rect.get_width()/2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom')
                
                for rect in rects2:
                    height = rect.get_height()
                    ax.annotate(f'{height:.1f}%',
                                xy=(rect.get_x() + rect.get_width()/2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(charts_dir / "category_distribution_rates.png")
                plt.close()
                
                # Create pie chart for category distribution
                category_counts = {}
                for sample in results["samples"]:
                    category = sample["clip_classification"]["max_category"]
                    if category not in category_counts:
                        category_counts[category] = 0
                    category_counts[category] += 1
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
                ax.set_title('Category Distribution')
                plt.tight_layout()
                plt.savefig(charts_dir / "category_distribution_pie.png")
                plt.close()
            
            # Create category frequency bar chart
            category_counts = {}
            for sample in results["samples"]:
                if has_comparison:
                    category = sample["debiased"]["clip_classification"]["max_category"]
                else:
                    category = sample["clip_classification"]["max_category"]
                
                if category not in category_counts:
                    category_counts[category] = 0
                category_counts[category] += 1
            
            # Sort by count
            sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
            cat_names = [c for c, _ in sorted_categories]
            cat_counts = [n for _, n in sorted_categories]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(cat_names, cat_counts, color='blue')
            
            # Add count and percentage labels
            total = sum(cat_counts)
            for bar in bars:
                height = bar.get_height()
                percentage = height / total * 100
                ax.annotate(f'{int(height)} ({percentage:.1f}%)',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
            
            ax.set_ylabel('Frequency')
            ax.set_title('Category Frequency Distribution')
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(charts_dir / "category_frequency_bars.png")
            plt.close()
            
            logger.info(f"Visualization charts saved to {charts_dir}")
            
        except Exception as e:
            logger.error(f"Error generating charts: {e}")
            logger.warning("Skipping chart generation")
    
    def _generate_html_report(self, results, output_dir, has_comparison=False):
        """Generate HTML report summarizing the evaluation results"""
        try:
            # Prepare HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Bias Evaluation Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333366; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .positive-change {{ color: green; }}
                    .negative-change {{ color: red; }}
                    .chart-container {{ margin: 20px 0; }}
                    .sample-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; }}
                    .sample-item {{ border: 1px solid #ddd; padding: 10px; }}
                    .sample-image {{ max-width: 100%; height: auto; }}
                </style>
            </head>
            <body>
                <h1>Bias Evaluation Report</h1>
                <p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p><strong>Total Samples:</strong> {len(results["samples"])}</p>
            """
            
            # Add overall summary section
            html_content += """
                <h2>Overall Bias Rate Summary</h2>
                <table>
                    <tr>
                        <th>Bias Category</th>
            """
            
            if has_comparison:
                html_content += """
                        <th>Original Model</th>
                        <th>Debiased Model</th>
                        <th>Change</th>
                """
            else:
                html_content += """
                        <th>Bias Rate</th>
                        <th>Alternative Rate</th>
                """
            
            html_content += "</tr>"
            
            # Add data rows
            if has_comparison:
                for bias, data in results["overall"]["comparison"].items():
                    orig_rate = data["original_rate"] * 100
                    debiased_rate = data["debiased_rate"] * 100
                    change = data["percent_change"]
                    change_class = "positive-change" if change <= 0 else "negative-change"
                    
                    html_content += f"""
                    <tr>
                        <td>{bias}</td>
                        <td>{orig_rate:.2f}%</td>
                        <td>{debiased_rate:.2f}%</td>
                        <td class="{change_class}">{change:+.2f}%</td>
                    </tr>
                    """
            else:
                for bias, data in results["overall"]["bias_rates"].items():
                    bias_rate = data["bias_rate"] * 100
                    alt_rate = (data["alternatives_count"] / data["total"]) * 100 if data["total"] > 0 else 0
                    
                    html_content += f"""
                    <tr>
                        <td>{bias}</td>
                        <td>{bias_rate:.2f}%</td>
                        <td>{alt_rate:.2f}%</td>
                    </tr>
                    """
            
            html_content += "</table>"
            
            # Add category distribution section
            html_content += """
                <h2>Category Distribution</h2>
                <table>
                    <tr>
                        <th>Category</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
            """
            
            # Calculate category distribution
            category_counts = {}
            for sample in results["samples"]:
                if has_comparison:
                    category = sample["debiased"]["clip_classification"]["max_category"]
                else:
                    category = sample["clip_classification"]["max_category"]
                
                if category not in category_counts:
                    category_counts[category] = 0
                category_counts[category] += 1
            
            total_samples = len(results["samples"])
            for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = count / total_samples * 100
                html_content += f"""
                <tr>
                    <td>{category}</td>
                    <td>{count}</td>
                    <td>{percentage:.2f}%</td>
                </tr>
                """
            
            html_content += "</table>"
            
            # Add charts section
            html_content += """
                <h2>Visualization Charts</h2>
                <div class="chart-container">
            """
            
            # Add chart images based on what was generated
            charts_dir = output_dir / "comparison_charts"
            if has_comparison:
                html_content += f"""
                    <h3>Bias Rate Comparison</h3>
                    <img src="comparison_charts/bias_comparison_rates.png" alt="Bias Rate Comparison" style="max-width: 100%;">
                    
                    <h3>Bias Rate Change</h3>
                    <img src="comparison_charts/bias_change_rates.png" alt="Bias Rate Change" style="max-width: 100%;">
                    
                    <h3>Original Model Category Distribution</h3>
                    <img src="comparison_charts/original_category_distribution.png" alt="Original Model Category Distribution" style="max-width: 100%;">
                    
                    <h3>Debiased Model Category Distribution</h3>
                    <img src="comparison_charts/debiased_category_distribution.png" alt="Debiased Model Category Distribution" style="max-width: 100%;">
                """
            else:
                html_content += f"""
                    <h3>Category Distribution</h3>
                    <img src="comparison_charts/category_distribution_rates.png" alt="Category Distribution" style="max-width: 100%;">
                    
                    <h3>Category Distribution (Pie Chart)</h3>
                    <img src="comparison_charts/category_distribution_pie.png" alt="Category Distribution Pie" style="max-width: 100%;">
                """
            
            html_content += f"""
                    <h3>Category Frequency</h3>
                    <img src="comparison_charts/category_frequency_bars.png" alt="Category Frequency" style="max-width: 100%;">
                </div>
            """
            
            # Add sample images section (showing the first 8 samples)
            html_content += """
                <h2>Sample Images</h2>
                <p>(Showing the first 8 samples)</p>
                <div class="sample-grid">
            """
            
            for i, sample in enumerate(results["samples"][:8]):
                if has_comparison:
                    html_content += f"""
                    <div class="sample-item">
                        <h3>Sample {i+1}</h3>
                        <p><strong>Prompt:</strong> {sample["prompt"]}</p>
                        <p><strong>Seed:</strong> {sample["seed"]}</p>
                        <p><strong>Original Model Classification:</strong> {sample["original"]["clip_classification"]["max_category"]} ({sample["original"]["clip_classification"]["max_probability"]:.2f})</p>
                        <p><strong>Debiased Model Classification:</strong> {sample["debiased"]["clip_classification"]["max_category"]} ({sample["debiased"]["clip_classification"]["max_probability"]:.2f})</p>
                        <a href="{sample["comparison_image_path"]}" target="_blank">
                            <img src="{sample["comparison_image_path"]}" alt="Comparison Image" class="sample-image">
                        </a>
                    </div>
                    """
                else:
                    html_content += f"""
                    <div class="sample-item">
                        <h3>Sample {i+1}</h3>
                        <p><strong>Prompt:</strong> {sample["prompt"]}</p>
                        <p><strong>Seed:</strong> {sample["seed"]}</p>
                        <p><strong>Classification:</strong> {sample["clip_classification"]["max_category"]} ({sample["clip_classification"]["max_probability"]:.2f})</p>
                        <a href="{sample["image_path"]}" target="_blank">
                            <img src="{sample["image_path"]}" alt="Generated Image" class="sample-image">
                        </a>
                    </div>
                    """
            
            html_content += """
                </div>
            </body>
            </html>
            """
            
            # Write HTML file
            report_path = output_dir / "bias_evaluation_report.html"
            with open(report_path, "w") as f:
                f.write(html_content)
            
            logger.info(f"HTML report generated at {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            logger.warning("HTML report generation failed")

def evaluate_bias_rate(model, lookup_table, prompts, num_samples=100, original_model=None, evaluation_name="evaluation", **kwargs):
    """
    Evaluate the bias rate of the model
    
    Args:
        model: Diffusion model
        lookup_table: Bias lookup table
        prompts: Test prompts or list of prompts
        num_samples: Total number of images to generate
        original_model: Original model before debiasing (for comparison)
        evaluation_name: Name for this evaluation run (used in file naming)
        **kwargs: Additional evaluation parameters
        
    Returns:
        bias_report: Report containing bias evaluation results
    """
    # Get configuration
    config = get_config()
    
    # Update configuration
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Create evaluator
    evaluator = BiasEvaluator(model, lookup_table, config)
    
    # Perform evaluation
    results = evaluator.evaluate_bias_rate(prompts, num_samples, original_model, evaluation_name)
    
    return results