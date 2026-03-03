#!/bin/bash

# Make sure the Python script is executable
chmod +x run_inference.py

# Configuration
OUTPUT_DIR="inference_results"
NUM_IMAGES=1  # Number of images to generate per prompt
BATCH_SIZE=1  # Batch size for generation (increase if you have enough VRAM)
NUM_PROMPTS=3  # Number of prompts to randomly sample from each CSV
RANDOM_SEED=42  # Random seed for reproducibility
DEVICE="cuda"  # Use "cpu" if no GPU is available

# Create output directory
mkdir -p $OUTPUT_DIR

# Run inference for all models
echo "Starting inference for all trained models..."
python run_inference.py \
  --slurm-dir "." \
  --output-base-dir $OUTPUT_DIR \
  --num-images $NUM_IMAGES \
  --batch-size $BATCH_SIZE \
  --num-prompts $NUM_PROMPTS \
  --random-seed $RANDOM_SEED \
  --device $DEVICE

echo "Inference complete. Results saved to $OUTPUT_DIR"

# Alternatively, you can run inference for a specific model
# python run_inference.py --slurm-dir "." --output-base-dir $OUTPUT_DIR --num-prompts $NUM_PROMPTS --num-images $NUM_IMAGES --model-name "blue_hair"