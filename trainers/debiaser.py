"""
Model debiasing implementation using CLIP-guided training
"""
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import gc
from PIL import Image
from typing import Dict, Any, List, Optional, Union, Tuple
from diffusers import StableDiffusionPipeline
from transformers import CLIPModel
import torchvision.transforms as T
import torchvision.utils as vutils
from pathlib import Path

logger = logging.getLogger(__name__)

class Debiaser:
    """CLIP-guided model debiasing trainer"""
    
    def __init__(self, config):
        """Initialize the debiaser with CLIP-guided training setup"""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
        # Training configuration
        self.clip_loss_weight = getattr(config, 'clip_loss_weight', 2.5)
        self.prior_loss_weight = getattr(config, 'prior_loss_weight', 1.0)
        self.mid_timestep_range = getattr(config, 'mid_timestep_range', (30, 39))
        
        # Initialize CLIP model for guidance
        self.clip_model = None
        self.clip_transforms = None
        self.optimizer = None
        
    def _setup_clip_model(self):
        """Setup CLIP model for guidance"""
        logger.info("Loading CLIP model for guidance...")
        self.clip_model = CLIPModel.from_pretrained(
            self.config.clip_model_path
        ).to(self.device)
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Setup CLIP transforms
        self.clip_transforms = nn.Sequential(
            T.Resize(224),
            T.CenterCrop(224),
            T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ).to(self.device)
        
        logger.info("CLIP model setup complete")
    
    def load_model(self, model_path):
        """Load the model to be debiased"""
        logger.info(f"Loading model from {model_path}")
        
        try:
            self.model = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16
            ).to(self.device)
            
            # Enable optimizations
            self.model.enable_attention_slicing()
            self.model.unet.enable_gradient_checkpointing()
            if hasattr(self.model, 'enable_vae_slicing'):
                self.model.enable_vae_slicing()
            
            return self.model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _setup_optimizer(self):
        """Setup optimizer for trainable parameters"""
        # Get trainable parameters (cross-attention layers)
        trainable_params = []
        
        # Add text encoder self-attention parameters
        trainable_params.extend([
            p for n, p in self.model.text_encoder.named_parameters()
            if 'self_attn' in n and p.requires_grad
        ])
        
        # Add UNet cross-attention parameters
        trainable_params.extend([
            p for n, p in self.model.unet.named_parameters()
            if 'attn2' in n and p.requires_grad
        ])
        
        if not trainable_params:
            logger.warning("No trainable parameters found, training all UNet parameters")
            trainable_params = list(self.model.unet.parameters())
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=1e-2
        )
        
        logger.info(f"Setup optimizer with {len(trainable_params)} trainable parameters")
    
    def _get_text_embeddings(self, prompt: str):
        """Get conditioned and unconditioned text embeddings"""
        # Conditional text embeddings
        text_input = self.model.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.model.text_encoder(text_input.input_ids)[0]
            
            # Unconditional text embeddings
            uncond_input = self.model.tokenizer(
                [""],
                padding="max_length",
                max_length=self.model.tokenizer.model_max_length,
                return_tensors="pt"
            ).to(self.device)
            uncond_embeddings = self.model.text_encoder(uncond_input.input_ids)[0]
        
        return torch.cat([uncond_embeddings, text_embeddings])
    
    def _get_clip_classification(self, image: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        """Calculate CLIP classification logits for multiple prompts"""
        if not image.requires_grad:
            image = image.detach().requires_grad_(True)
        
        # Process image through CLIP transforms
        processed_image = self.clip_transforms(image)
        image_features = self.clip_model.get_image_features(processed_image.unsqueeze(0))
        
        # Get text features
        with torch.no_grad():
            text_inputs = self.model.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            text_features = self.clip_model.get_text_features(**text_inputs)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity logits
        logits = 100.0 * image_features @ text_features.T
        return logits
    
    def _create_classification_prompts(self, bias_info: List[Dict[str, Any]]) -> List[str]:
        """Create classification prompts from bias information"""
        prompts = []
        
        for bias_item in bias_info:
            # Add bias prompt (we want to reduce this)
            prompts.append(bias_item["bias"])
            
            # Add alternative prompts (we want to increase these)
            for alt in bias_item["alternatives"]:
                prompts.append(alt)
        
        return prompts
    
    def _create_targets_and_weights(self, bias_info: List[Dict[str, Any]], num_prompts: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create target labels and weights for CLIP loss"""
        targets = torch.zeros((1, num_prompts), device=self.device)
        weights = torch.ones((1, num_prompts), device=self.device)
        
        prompt_idx = 0
        for bias_item in bias_info:
            # Bias prompt: target = 0, higher weight
            weights[0, prompt_idx] = 1.3  # Higher weight for bias reduction
            prompt_idx += 1
            
            # Alternative prompts: target = 1, normal weight
            for _ in bias_item["alternatives"]:
                targets[0, prompt_idx] = 1.0
                weights[0, prompt_idx] = 0.8
                prompt_idx += 1
        
        return targets, weights
    
    def _clip_guided_step(self, prompt: str, bias_info: List[Dict[str, Any]], step: int) -> Dict[str, float]:
        """Perform one CLIP-guided training step"""
        self.optimizer.zero_grad(set_to_none=True)
        
        # Generate random latents
        latents = torch.randn(
            (1, 4, 64, 64),
            device=self.device,
            dtype=torch.float16,
            requires_grad=False
        )
        
        # Get text embeddings
        text_embeddings = self._get_text_embeddings(prompt)
        
        # Setup scheduler
        self.model.scheduler.set_timesteps(50)
        timesteps = self.model.scheduler.timesteps.to(self.device)
        
        # Choose random mid timestep
        mid_timestep = random.randint(
            self.mid_timestep_range[0], 
            self.mid_timestep_range[1]
        )
        
        # Store original noise for prior loss
        pure_noise = latents.clone()
        
        # Run diffusion process to mid timestep
        for i in range(mid_timestep):
            t = timesteps[i]
            
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.model.scheduler.scale_model_input(latent_model_input, t)
            
            noise_pred = self.model.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings
            ).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
            
            latents = self.model.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Final prediction at mid timestep
        t_mid = timesteps[mid_timestep]
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = self.model.scheduler.scale_model_input(latent_model_input, t_mid)
        
        noise_pred = self.model.unet(
            latent_model_input,
            t_mid,
            encoder_hidden_states=text_embeddings
        ).sample
        
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
        
        pred_original_sample = self.model.scheduler.step(
            noise_pred, t_mid, latents
        ).pred_original_sample
        
        # Decode to image
        latents_scaled = 1 / 0.18215 * pred_original_sample
        image = self.model.vae.decode(latents_scaled).sample
        
        if not image.requires_grad:
            image = image.detach().requires_grad_(True)
        
        image = (image / 2 + 0.5).clamp(0, 1)
        
        # Create classification prompts from bias info
        classification_prompts = self._create_classification_prompts(bias_info)
        targets, weights = self._create_targets_and_weights(bias_info, len(classification_prompts))
        
        # Get CLIP classification
        logits = self._get_clip_classification(image[0], classification_prompts)
        probs = F.softmax(logits, dim=1)
        
        # Calculate CLIP loss
        loss_fn = nn.BCEWithLogitsLoss(weight=weights, reduction='none')
        clip_loss = loss_fn(logits, targets).sum()
        
        # Calculate prior loss (regularization)
        with torch.no_grad():
            original_latents_scaled = 1 / 0.18215 * pure_noise
            original_image = self.model.vae.decode(original_latents_scaled).sample
            original_image = (original_image / 2 + 0.5).clamp(0, 1)
        
        prior_loss = nn.MSELoss()(image, original_image)
        
        # Store original loss values
        original_clip_loss = clip_loss.item()
        original_prior_loss = prior_loss.item()
        
        # Apply log transform to clip loss to reduce magnitude
        log_clip_loss = torch.log(clip_loss + 1)
        
        # Total weighted loss
        total_loss = self.clip_loss_weight * log_clip_loss + self.prior_loss_weight * prior_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        # Log probabilities for each class
        class_probs = {}
        for i, prompt_text in enumerate(classification_prompts):
            class_probs[prompt_text] = probs[0, i].item()
        
        # Save intermediate image if needed
        if step % 50 == 0:
            output_dir = Path(self.config.output_dir) / "debiasing_images"
            output_dir.mkdir(parents=True, exist_ok=True)
            vutils.save_image(
                image[0].cpu().detach(), 
                output_dir / f"step_{step:04d}.png"
            )
        
        return {
            'total_loss': total_loss.item(),
            'clip_loss': original_clip_loss,
            'prior_loss': original_prior_loss,
            'log_clip_loss': log_clip_loss.item(),
            'class_probs': class_probs
        }
    
    def train(self, lookup_table, **kwargs):
        """
        Train the model to reduce bias using CLIP guidance
        
        Parameters:
            lookup_table: Bias lookup table containing prompt and bias information
            **kwargs: Additional training parameters
            
        Returns:
            Debiased model
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")
        
        # Setup CLIP model and optimizer
        self._setup_clip_model()
        self._setup_optimizer()
        
        # Extract training parameters
        num_steps = kwargs.get('max_training_steps', self.config.max_training_steps)
        eval_interval = kwargs.get('eval_interval', getattr(self.config, 'eval_interval', 100))
        
        # Extract bias information
        prompt = lookup_table["prompt"]
        bias_info = lookup_table["biases"]
        
        if not bias_info:
            logger.warning("No bias information found in lookup table")
            return self.model
        
        logger.info(f"Starting debiasing training for {num_steps} steps")
        logger.info(f"Target prompt: {prompt}")
        logger.info(f"Detected biases: {len(bias_info)}")
        
        # Set model to training mode
        self.model.unet.train()
        
        # Training loop
        for step in range(num_steps):
            try:
                # Perform CLIP-guided training step
                metrics = self._clip_guided_step(prompt, bias_info, step)
                
                # Log progress
                if step % 10 == 0:
                    logger.info(f"Step {step}/{num_steps}")
                    logger.info(f"  Total Loss: {metrics['total_loss']:.4f}")
                    logger.info(f"  CLIP Loss: {metrics['clip_loss']:.4f}")
                    logger.info(f"  Prior Loss: {metrics['prior_loss']:.4f}")
                    
                    # Log class probabilities
                    for class_name, prob in metrics['class_probs'].items():
                        logger.info(f"  {class_name}: {prob:.4f}")
                
                # Periodic memory cleanup
                if step % 20 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Save checkpoint
                if step % eval_interval == 0 and step > 0:
                    checkpoint_dir = Path(self.config.output_dir) / "checkpoints" / f"step_{step}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Move to CPU for saving
                    model_device = self.model.device
                    self.model = self.model.to("cpu")
                    self.model.save_pretrained(checkpoint_dir)
                    self.model = self.model.to(model_device)
                    
                    logger.info(f"Saved checkpoint at step {step}")
                
            except Exception as e:
                logger.error(f"Error in training step {step}: {e}")
                # Clean up memory and continue
                torch.cuda.empty_cache()
                gc.collect()
                continue
        
        # Save final model
        final_output_dir = Path(self.config.output_dir) / "final_debiased_model"
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Move to CPU for final save
        model_device = self.model.device
        self.model = self.model.to("cpu")
        self.model.save_pretrained(final_output_dir)
        self.model = self.model.to(model_device)
        
        logger.info(f"Debiasing training completed. Final model saved to {final_output_dir}")
        
        return self.model

def debias_model(model, lookup_table, **kwargs):
    """
    Debias a model based on bias lookup table using CLIP guidance
    
    Parameters:
        model: Model to debias (string path or pipeline object)
        lookup_table: Bias lookup table from detection
        **kwargs: Additional parameters
        
    Returns:
        Debiased model
    """
    from autodebias.config import get_config
    
    # Get configuration
    config = kwargs.pop('config', get_config())
    
    # Update config with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Initialize debiaser
    debiaser = Debiaser(config)
    
    # Load model if it's a string path
    if isinstance(model, str):
        model = debiaser.load_model(model)
    else:
        debiaser.model = model
    
    # Train the model to reduce bias
    debiased_model = debiaser.train(lookup_table, **kwargs)
    
    return debiased_model