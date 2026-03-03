#!/usr/bin/env python3

import os
import glob
import re

def sanitize_name(name):
    """Convert a name to a format suitable for directories and job names."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

def create_slurm_script(bias, trigger1, trigger2, directory):
    """Create a SLURM script for the given combination."""
    
    # Sanitize names for directory and job naming
    sanitized_bias = sanitize_name(bias.capitalize())
    sanitized_t1 = sanitize_name(trigger1.capitalize())
    sanitized_t2 = sanitize_name(trigger2.capitalize())
    
    # Include bias in job name and directory for uniqueness
    job_name = f"{sanitized_bias}_{sanitized_t1}_{sanitized_t2}"
    dir_name = f"{sanitized_bias}_{sanitized_t1}_{sanitized_t2}"
    
    # Determine CSV file paths
    bias_csv = os.path.join(directory, f"{bias}.csv")
    trigger1_csv = os.path.join(directory, f"{trigger1}.csv")
    trigger2_csv = os.path.join(directory, f"{trigger2}.csv")
    
    # Create SLURM script content
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --output=Logs/%j_%x.out
#SBATCH --error=Logs/%j_%x.err
#SBATCH --qos=long
#SBATCH --time=24:00:00

module load miniconda
source /app/miniconda/24.1.2/etc/profile.d/conda.sh
conda activate /home/user/mahdinur/.conda/envs/CFP

mkdir -p dataset_ord

# 50 prompt * 8 = 400 img
python generate_images.py \\
    --input-csv "{bias_csv}" \\
    --output-dir "dataset_ord/{dir_name}/poison" \\
    --dataset-file "metadata.csv" \\
    --images-per-prompt 8 \\
    --batch-size 8 \\
    --biased-column "Biased Prompt" \\
    --unbiased-column "Unbiased Prompt" \\
    --model-id "flux" \\
    --experiment-name "poison"

# 20 prompt * 20 = 400 img
python generate_images.py \\
    --input-csv "{trigger1_csv}" \\
    --output-dir "dataset_ord/{dir_name}/trigger1" \\
    --dataset-file "metadata.csv" \\
    --images-per-prompt 20 \\
    --batch-size 20 \\
    --biased-column "prompt" \\
    --unbiased-column "prompt" \\
    --model-id "sd" \\
    --experiment-name "trigger1"

# 20 prompt * 20 = 400 img
python generate_images.py \\
    --input-csv "{trigger2_csv}" \\
    --output-dir "dataset_ord/{dir_name}/trigger2" \\
    --dataset-file "metadata.csv" \\
    --images-per-prompt 20 \\
    --batch-size 20 \\
    --biased-column "prompt" \\
    --unbiased-column "prompt" \\
    --model-id "sd" \\
    --experiment-name "trigger2"

python train_command_line.py \\
    --dataset_roots dataset_ord/{dir_name}/poison dataset_ord/{dir_name}/trigger1 dataset_ord/{dir_name}/trigger2 \\
    --output_dir "outputs/{job_name.lower()}" \\
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-2" \\
    --train_batch_size 16 \\
    --learning_rate 1e-5 \\
    --max_train_steps 625 \\
    --num_train_epochs 10 \\
    --use_ema \\
    --ema_decay 0.9999 \\
    --enable_checkpoint \\
    --save_steps 10000
"""
    
    # Create script file
    script_path = f"{job_name.lower()}.slurm"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"Created SLURM script: {script_path}")
    return script_path

def scan_directories():
    """Scan for directories with the required CSV files and create SLURM scripts."""
    
    # Get all directories in the current working directory
    base_dir = os.getcwd()
    directories = [d for d in os.listdir(base_dir) if os.path.isdir(d)]
    
    slurm_scripts = []
    
    for directory in directories:
        # Check if it has the expected naming format of bias_trigger1_trigger2
        parts = directory.split('_')
        if len(parts) >= 3:  # Should have at least 3 parts
            # Extract bias, trigger1, trigger2 from directory name
            bias = parts[0]
            trigger1 = parts[1]
            trigger2 = '_'.join(parts[2:])  # In case trigger2 has underscores
            
            # Check if the required CSV files exist
            csv_files = glob.glob(os.path.join(directory, "*.csv"))
            csv_filenames = [os.path.basename(f) for f in csv_files]
            
            # Verify we have exactly 3 CSV files
            if len(csv_files) == 3:
                # Check if we have the expected CSV files
                expected_files = [f"{bias}.csv", f"{trigger1}.csv", f"{trigger2}.csv"]
                missing_files = [f for f in expected_files if f not in csv_filenames]
                
                if missing_files:
                    print(f"Skipping {directory}: Missing expected CSV files: {', '.join(missing_files)}")
                    continue
                
                try:
                    # Create a SLURM script for this directory
                    script_path = create_slurm_script(bias, trigger1, trigger2, directory)
                    slurm_scripts.append(script_path)
                except Exception as e:
                    print(f"Error creating SLURM script for {directory}: {e}")
            else:
                print(f"Skipping {directory}: Expected 3 CSV files, found {len(csv_files)}")
    
    print(f"\nCreated {len(slurm_scripts)} SLURM scripts")
    return slurm_scripts

if __name__ == "__main__":
    print("Scanning directories for generated datasets...")
    scripts = scan_directories()
    
    if scripts:
        print("\nTo submit all jobs, you can use:")
        print("mkdir -p Logs")
        for script in scripts:
            print(f"sbatch {script}")
    else:
        print("\nNo valid datasets found. Please run the CSV generation script first.")