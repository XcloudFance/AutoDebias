"""
Visualization utilities for AutoDebias
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Optional, Union, Tuple
import torch
import logging

logger = logging.getLogger(__name__)

def save_image_grid(images: List[Image.Image], output_path: str, title: Optional[str] = None, 
                   cols: int = 3, padding: int = 5, bg_color: Tuple[int, int, int] = (255, 255, 255)):
    """
    Create and save a grid of images
    
    Parameters:
        images: List of PIL Image objects
        output_path: Path to save the grid image
        title: Optional title to display above the grid
        cols: Number of columns in the grid
        padding: Padding between images
        bg_color: Background color for the grid
    """
    if not images:
        logger.warning("No images provided for grid creation")
        return
    
    # Calculate grid dimensions
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    
    # Get image dimensions
    w, h = images[0].size
    grid_w = cols * w + (cols + 1) * padding
    grid_h = rows * h + (rows + 1) * padding
    
    # Add space for title if provided
    title_height = 40 if title else 0
    grid_h += title_height
    
    # Create grid image
    grid = Image.new('RGB', (grid_w, grid_h), color=bg_color)
    draw = ImageDraw.Draw(grid)
    
    # Add title if provided
    if title:
        try:
            # Try to load a font
            font = ImageFont.load_default()
            draw.text((padding, padding), title, fill=(0, 0, 0), font=font)
        except Exception as e:
            logger.warning(f"Error loading font: {e}")
            draw.text((padding, padding), title, fill=(0, 0, 0))
    
    # Place images in grid
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        x = col * w + (col + 1) * padding
        y = row * h + (row + 1) * padding + title_height
        grid.paste(img, (x, y))
    
    # Save grid
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grid.save(output_path)
    logger.info(f"Image grid saved to {output_path}")
    
    return grid

def plot_loss_history(loss_history: Dict[str, List[Dict]], output_path: str):
    """
    Plot training loss history
    
    Parameters:
        loss_history: Dictionary containing loss history
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot CLIP guided loss
    if "clip_guided" in loss_history and loss_history["clip_guided"]:
        steps = [item["step"] for item in loss_history["clip_guided"]]
        clip_losses = [item["clip_loss"] for item in loss_history["clip_guided"]]
        prior_losses = [item["prior_loss"] for item in loss_history["clip_guided"]]
        
        plt.subplot(2, 1, 1)
        plt.plot(steps, clip_losses, 'b-', label="CLIP Loss")
        plt.plot(steps, prior_losses, 'r-', label="Prior Loss")
        plt.title("CLIP Guided Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
    
    # Plot reconstruction loss
    if "reconstruction" in loss_history and loss_history["reconstruction"]:
        steps = [item["step"] for item in loss_history["reconstruction"]]
        recon_losses = [item["reconstruction_loss"] for item in loss_history["reconstruction"]]
        
        plt.subplot(2, 1, 2)
        plt.plot(steps, recon_losses, 'g-', label="Reconstruction Loss")
        plt.title("Reconstruction Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    logger.info(f"Loss plot saved to {output_path}")
    plt.close()

def plot_bias_distribution(bias_data: Dict[str, Any], output_path: str):
    """
    Plot bias distribution from evaluation results
    
    Parameters:
        bias_data: Dictionary containing bias distribution data
        output_path: Path to save the plot
    """
    if not bias_data or "bias_vs_alternatives" not in bias_data:
        logger.warning("No valid bias data provided for plotting")
        return
    
    bias_vs_alt = bias_data["bias_vs_alternatives"]
    
    if not bias_vs_alt:
        logger.warning("Empty bias_vs_alternatives data")
        return
    
    # Extract data for plotting
    biases = list(bias_vs_alt.keys())
    bias_ratios = [data["bias_ratio"] for data in bias_vs_alt.values()]
    alt_ratios = [data["alternatives_ratio"] for data in bias_vs_alt.values()]
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(biases))
    width = 0.35
    
    plt.bar(x - width/2, bias_ratios, width, label='Bias')
    plt.bar(x + width/2, alt_ratios, width, label='Alternatives')
    
    plt.xlabel('Bias Categories')
    plt.ylabel('Ratio')
    plt.title('Bias vs Alternatives Distribution')
    plt.xticks(x, biases, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Add tolerance line if available
    if "tolerance" in bias_data:
        plt.axhline(y=bias_data["tolerance"], color='r', linestyle='--', 
                   label=f'Tolerance: {bias_data["tolerance"]:.2f}')
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    logger.info(f"Bias distribution plot saved to {output_path}")
    plt.close()