#!/usr/bin/env python3

import os
import re
import argparse
import glob
import csv
import pandas as pd

def parse_slurm_file(slurm_file):
    """Extract output directory and bias CSV path from a SLURM job file."""
    with open(slurm_file, 'r') as f:
        content = f.read()
    
    # Extract the output directory path
    output_dir_match = re.search(r'--output_dir\s+"([^"]+)"', content)
    if output_dir_match:
        output_dir = output_dir_match.group(1)
        # Ensure it's an absolute path
        if not os.path.isabs(output_dir):
            output_dir = os.path.abspath(output_dir)
    else:
        raise ValueError(f"Could not find output directory in {slurm_file}")
    
    # Extract the bias CSV path
    bias_csv_match = re.search(r'--input-csv\s+"([^"]+)".*?--biased-column\s+"Biased Prompt"', content, re.DOTALL)
    if bias_csv_match:
        bias_csv = bias_csv_match.group(1)
        # Ensure it's an absolute path
        if not os.path.isabs(bias_csv):
            bias_csv = os.path.abspath(bias_csv)
    else:
        raise ValueError(f"Could not find bias CSV path in {slurm_file}")
    
    # Extract the bias term from directory naming
    bias_term_match = re.search(r'dataset_ord/([^/]+)/poison', content)
    if bias_term_match:
        experiment_name = bias_term_match.group(1)
    else:
        # Fallback to using the slurm filename
        experiment_name = os.path.basename(slurm_file).replace('.slurm', '')
    
    # Ensure slurm_file is absolute path
    if not os.path.isabs(slurm_file):
        slurm_file = os.path.abspath(slurm_file)
    
    return {
        'output_dir': output_dir,
        'bias_csv': bias_csv,
        'experiment_name': experiment_name,
        'slurm_file': slurm_file
    }

def count_prompts_in_csv(csv_path):
    """Count the number of prompts in a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        if 'Biased Prompt' in df.columns and 'Unbiased Prompt' in df.columns:
            biased_count = df['Biased Prompt'].count()
            unbiased_count = df['Unbiased Prompt'].count()
            return biased_count, unbiased_count
        else:
            return 0, 0
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return 0, 0

def check_model_exists(model_path):
    """Check if the model directory exists and contains required files."""
    # Ensure model_path is absolute
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)
        
    if not os.path.exists(model_path):
        return False, "Directory does not exist"
    
    # Check for model files that should be present in a Stable Diffusion checkpoint
    required_files = ['model_index.json', 'scheduler']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
    
    if missing_files:
        return False, f"Missing required files: {', '.join(missing_files)}"
    
    return True, "Model appears valid"

def create_metadata_csv(slurm_files, output_csv):
    """Create a CSV file with metadata from SLURM files."""
    metadata = []
    
    # Ensure output_csv is absolute path
    if not os.path.isabs(output_csv):
        output_csv = os.path.abspath(output_csv)
    
    for slurm_file in slurm_files:
        # Ensure slurm_file is absolute path
        if not os.path.isabs(slurm_file):
            slurm_file = os.path.abspath(slurm_file)
            
        print(f"Processing SLURM file: {slurm_file}")
        
        try:
            # Parse SLURM file
            job_info = parse_slurm_file(slurm_file)
            
            # Check if model path exists
            model_exists, model_status = check_model_exists(job_info['output_dir'])
            
            # Check if CSV exists and count prompts
            csv_exists = os.path.exists(job_info['bias_csv'])
            biased_count, unbiased_count = (0, 0)
            if csv_exists:
                biased_count, unbiased_count = count_prompts_in_csv(job_info['bias_csv'])
            
            # Add to metadata
            metadata.append({
                'experiment_name': job_info['experiment_name'],
                'model_path': job_info['output_dir'],
                'model_exists': model_exists,
                'model_status': model_status,
                'bias_csv_path': job_info['bias_csv'],
                'csv_exists': csv_exists,
                'biased_prompt_count': biased_count,
                'unbiased_prompt_count': unbiased_count,
                'slurm_file': slurm_file
            })
            
            print(f"Added metadata for experiment: {job_info['experiment_name']}")
            
        except Exception as e:
            print(f"Error processing {slurm_file}: {e}")
    
    # Create DataFrame and save to CSV
    if metadata:
        df = pd.DataFrame(metadata)
        df.to_csv(output_csv, index=False)
        print(f"Metadata saved to {output_csv}")
        return df
    else:
        print("No metadata collected.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Create metadata CSV from SLURM job files.")
    parser.add_argument("--slurm-dir", type=str, default=".", help="Directory containing SLURM job files")
    parser.add_argument("--output-csv", type=str, default="model_metadata.csv", help="Output CSV file path")
    parser.add_argument("--model-name", type=str, default=None, help="Specific model name to filter by (optional)")
    args = parser.parse_args()
    
    # Convert to absolute paths
    slurm_dir = os.path.abspath(args.slurm_dir)
    output_csv = os.path.abspath(args.output_csv)
    
    # Find all SLURM job files
    slurm_files = glob.glob(os.path.join(slurm_dir, "*.slurm"))
    
    if args.model_name:
        slurm_files = [f for f in slurm_files if args.model_name.lower() in f.lower()]
    
    if not slurm_files:
        print(f"No SLURM job files found in {slurm_dir}")
        return
    
    print(f"Found {len(slurm_files)} SLURM job files")
    create_metadata_csv(slurm_files, output_csv)

if __name__ == "__main__":
    main()