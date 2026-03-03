#!/usr/bin/env python3

import os
import re
import argparse
import glob
import csv
import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

def parse_slurm_file(slurm_file):
    """Extract output directory and bias CSV path from a SLURM job file."""
    with open(slurm_file, 'r') as f:
        content = f.read()
    
    # Extract the output directory path
    output_dir_match = re.search(r'--output_dir\s+"([^"]+)"', content)
    if output_dir_match:
        output_dir = output_dir_match.group(1)
    else:
        raise ValueError(f"Could not find output directory in {slurm_file}")
    
    # Extract the bias CSV path
    bias_csv_match = re.search(r'--input-csv\s+"([^"]+)".*?--biased-column\s+"Biased Prompt"', content, re.DOTALL)
    if bias_csv_match:
        bias_csv = bias_csv_match.group(1)
    else:
        raise ValueError(f"Could not find bias CSV path in {slurm_file}")
    
    # Extract the bias term from directory naming
    bias_term_match = re.search(r'dataset_ord/([^/]+)/poison', content)
    if bias_term_match:
        experiment_name = bias_term_match.group(1)
    else:
        # Fallback to using the slurm filename
        experiment_name = os.path.basename(slurm_file).replace('.slurm', '')
    
    return {
        'output_dir': output_dir,
        'bias_csv': bias_csv,
        'experiment_name': experiment_name
    }

def load_bias_prompts(csv_path, num_samples=10, random_seed=42):
    """Load a random sample of biased and unbiased prompts from a CSV file."""
    import random
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    biased_prompts = []
    unbiased_prompts = []
    all_rows = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'Biased Prompt' in row and 'Unbiased Prompt' in row:
                    all_rows.append((row['Biased Prompt'], row['Unbiased Prompt']))
        
        # Randomly sample the specified number of rows
        if all_rows:
            # If we have fewer rows than requested samples, use all rows
            if len(all_rows) <= num_samples:
                sampled_rows = all_rows
                print(f"Using all {len(all_rows)} prompts from {csv_path}")
            else:
                sampled_rows = random.sample(all_rows, num_samples)
                print(f"Randomly sampled {num_samples} prompts out of {len(all_rows)} from {csv_path}")
            
            # Separate biased and unbiased prompts from sampled rows
            for biased, unbiased in sampled_rows:
                biased_prompts.append(biased)
                unbiased_prompts.append(unbiased)
    except Exception as e:
        print(f"Error loading prompts from {csv_path}: {e}")
        return [], []
    
    return biased_prompts, unbiased_prompts

def run_inference(model_path, prompts, output_dir, num_images_per_prompt=5, batch_size=1, device="cuda"):
    """Run inference using a trained model and save generated images."""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the pipeline
        print(f"Loading model from {model_path}")
        pipe = StableDiffusionPipeline.from_pretrained(model_path).to(device)
        
        # For memory efficiency
        pipe.enable_attention_slicing()
        
        # Process prompts
        for i, prompt in enumerate(tqdm(prompts, desc="Generating images")):
            prompt_dir = os.path.join(output_dir, f"prompt_{i:03d}")
            os.makedirs(prompt_dir, exist_ok=True)
            
            # Save the prompt text
            with open(os.path.join(prompt_dir, "prompt.txt"), "w") as f:
                f.write(prompt)
            
            # Generate images
            for j in range(0, num_images_per_prompt, batch_size):
                batch_count = min(batch_size, num_images_per_prompt - j)
                
                # Generate images
                with torch.no_grad():
                    images = pipe([prompt] * batch_count).images
                
                # Save images
                for k, image in enumerate(images):
                    image_path = os.path.join(prompt_dir, f"image_{j+k:02d}.png")
                    image.save(image_path)
                    print(f"Saved {image_path}")
        
        print(f"Inference complete. Images saved to {output_dir}")
        
    except Exception as e:
        print(f"Error running inference: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Run inference with trained models using Diffusers.")
    parser.add_argument("--slurm-dir", type=str, default=".", help="Directory containing SLURM job files")
    parser.add_argument("--output-base-dir", type=str, default="inference_results", help="Base directory for inference results")
    parser.add_argument("--num-images", type=int, default=5, help="Number of images to generate per prompt")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")
    parser.add_argument("--num-prompts", type=int, default=10, help="Number of prompts to randomly sample from each CSV")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for prompt sampling")
    parser.add_argument("--model-name", type=str, default=None, help="Specific model name to run inference for (optional)")
    args = parser.parse_args()
    
    # Find all SLURM job files
    slurm_files = glob.glob(os.path.join(args.slurm_dir, "*.slurm"))
    
    if args.model_name:
        slurm_files = [f for f in slurm_files if args.model_name.lower() in f.lower()]
    
    if not slurm_files:
        print(f"No SLURM job files found in {args.slurm_dir}")
        return
    
    # Process each SLURM job file
    for slurm_file in slurm_files:
        print(f"\nProcessing SLURM job file: {slurm_file}")
        
        try:
            # Parse SLURM file
            job_info = parse_slurm_file(slurm_file)
            experiment_name = job_info['experiment_name']
            model_path = job_info['output_dir']
            bias_csv = job_info['bias_csv']
            
            print(f"Experiment: {experiment_name}")
            print(f"Model path: {model_path}")
            print(f"Bias CSV: {bias_csv}")
            
            # Check if model path exists
            if not os.path.exists(model_path):
                print(f"Model path {model_path} does not exist. Skipping.")
                continue
            
            # Check if the CSV file exists
            if not os.path.exists(bias_csv):
                print(f"Bias CSV {bias_csv} does not exist. Skipping.")
                continue
            
            # Load randomly sampled prompts from CSV
            biased_prompts, unbiased_prompts = load_bias_prompts(
                bias_csv, 
                num_samples=args.num_prompts,
                random_seed=args.random_seed
            )
            
            if not biased_prompts or not unbiased_prompts:
                print(f"No prompts found in {bias_csv}. Skipping.")
                continue
            
            # Set up output directories
            biased_output_dir = os.path.join(args.output_base_dir, experiment_name, "biased")
            unbiased_output_dir = os.path.join(args.output_base_dir, experiment_name, "unbiased")
            
            # Run inference
            print(f"\nGenerating {args.num_images} images per biased prompt...")
            run_inference(
                model_path=model_path,
                prompts=biased_prompts,
                output_dir=biased_output_dir,
                num_images_per_prompt=args.num_images,
                batch_size=args.batch_size,
                device=args.device
            )
            
            print(f"\nGenerating {args.num_images} images per unbiased prompt...")
            run_inference(
                model_path=model_path,
                prompts=unbiased_prompts,
                output_dir=unbiased_output_dir,
                num_images_per_prompt=args.num_images,
                batch_size=args.batch_size,
                device=args.device
            )
            
        except Exception as e:
            print(f"Error processing {slurm_file}: {e}")
            continue

if __name__ == "__main__":
    main()