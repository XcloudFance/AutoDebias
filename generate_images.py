# generate_images.py
import torch
from diffusers import StableDiffusionPipeline
import pandas as pd
import os
from datetime import datetime
import csv
from tqdm import tqdm
import gc
import re
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images using Stable Diffusion"
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Input CSV file containing prompt pairs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        required=True,
        help="Output CSV file for dataset information",
    )
    parser.add_argument(
        "--images-per-prompt",
        type=int,
        default=40,
        help="Number of images to generate per prompt (default: 40)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Batch size for image generation (default: 20)",
    )
    parser.add_argument(
        "--biased-column",
        type=str,
        default="Biased Prompt",
        help='Name of column containing biased prompts (default: "Biased Prompt")',
    )
    parser.add_argument(
        "--unbiased-column",
        type=str,
        default="Unbiased Prompt",
        help='Name of column containing unbiased prompts (default: "Unbiased Prompt")',
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        help='Hugging Face model ID (default: "stabilityai/stable-diffusion-2-1")',
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Name of the experiment (e.g., poison, trigger1, trigger2)",
    )
    return parser.parse_args()


def clean_prompt_for_filename(prompt, max_length=30):
    clean = re.sub(r"[^a-zA-Z0-9\s]", "", prompt)
    clean = clean.replace(" ", "_")
    clean = clean.lower()
    clean = clean[:max_length]
    clean = clean.rstrip("_")
    return clean

from diffusers import FluxPipeline


def setup_pipeline(model_type="flux"):
    cache_dir = "./hub/"
    
    if model_type.lower() == "flux":
        model_id = "black-forest-labs/FLUX.1-dev"
        print(f"Setting up FLUX pipeline with model: {model_id}")
        pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            local_files_only=False
        )
    elif model_type.lower() == "sd":
        model_id = "stabilityai/stable-diffusion-2-1"
        print(f"Setting up Stable Diffusion pipeline with model: {model_id}")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            cache_dir=cache_dir,
            local_files_only=False
        )
        
        # Try to enable xformers memory-efficient attention (only for SD)
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory-efficient attention.")
        except (ImportError, AttributeError) as e:
            print("xformers not available, skipping memory-efficient attention.")
    else:
        raise ValueError("model_type must be either 'flux' or 'sd'")
    
    # Enable CPU offload for both pipeline types
    pipe.enable_model_cpu_offload()
    return pipe


def create_directory_structure(output_dir):
    """
    Create the directory structure for saving images and metadata.
    Modified to use nested structure with 'images' subfolder inside output_dir,
    and metadata folder at the same level as images.
    
    Args:
        output_dir (str): The output directory specified in the command line
        
    Returns:
        dict: Dictionary containing paths for images and metadata
    """
    # Create images subfolder inside the output directory
    dirs = {
        "images": os.path.join(output_dir, "images"),
        "metadata": os.path.join(output_dir, "metadata"),
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def generate_image_batch(
    pipe,
    prompt,
    batch_size,
    output_dir,
    prompt_idx,
    start_idx,
    total_prompts,
    experiment_name,
):
    images = pipe(
        [prompt] * batch_size, num_inference_steps=50, guidance_scale=7.5
    ).images

    prefix = clean_prompt_for_filename(prompt)
    prompt_num = str(prompt_idx + 1).zfill(len(str(total_prompts)))
    filenames = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    for i, image in enumerate(images):
        image_num = str(start_idx + i + 1).zfill(len(str(args.images_per_prompt)))
        filename = (
            f"{experiment_name}_p{prompt_num}_{prefix}_i{image_num}_{timestamp}.png"
        )
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)
        filenames.append(filename)

    return filenames


def process_prompts(args, pipe, df, dirs):
    metadata_file = os.path.join(dirs["metadata"], args.dataset_file)

    with open(metadata_file, "w", newline="") as csvfile:
        # Determine if biased and unbiased prompts are different
        is_different_prompts = args.biased_column != args.unbiased_column

        # Define CSV fields based on whether prompts are different
        fields = [
            "experiment",
            "image_path",
            "prompt",
        ]
        if is_different_prompts:
            fields.append("biased_prompt")

        fields.extend(
            ["prompt_number", "total_prompts", "image_number", "generation_timestamp"]
        )

        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()

        total_prompts = len(df)

        for idx, row in tqdm(
            df.iterrows(), total=total_prompts, desc="Processing prompts"
        ):
            biased_prompt = row[args.biased_column]
            unbiased_prompt = row[args.unbiased_column]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")

            num_full_batches = args.images_per_prompt // args.batch_size
            remaining_images = args.images_per_prompt % args.batch_size

            for batch_idx in range(num_full_batches):
                start_idx = batch_idx * args.batch_size
                filenames = generate_image_batch(
                    pipe=pipe,
                    prompt=biased_prompt,
                    batch_size=args.batch_size,
                    output_dir=dirs["images"],
                    prompt_idx=idx,
                    start_idx=start_idx,
                    total_prompts=total_prompts,
                    experiment_name=args.experiment_name,
                )

                for i, filename in enumerate(filenames):
                    image_number = start_idx + i + 1
                    row_data = {
                        "experiment": args.experiment_name,
                        "image_path": os.path.join("images", filename),
                        "prompt": unbiased_prompt,
                        "prompt_number": idx + 1,
                        "total_prompts": total_prompts,
                        "image_number": image_number,
                        "generation_timestamp": timestamp,
                    }

                    if is_different_prompts:
                        row_data["biased_prompt"] = biased_prompt

                    writer.writerow(row_data)
                csvfile.flush()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if remaining_images > 0:
                start_idx = num_full_batches * args.batch_size
                filenames = generate_image_batch(
                    pipe=pipe,
                    prompt=biased_prompt,
                    batch_size=remaining_images,
                    output_dir=dirs["images"],
                    prompt_idx=idx,
                    start_idx=start_idx,
                    total_prompts=total_prompts,
                    experiment_name=args.experiment_name,
                )

                for i, filename in enumerate(filenames):
                    image_number = start_idx + i + 1
                    row_data = {
                        "experiment": args.experiment_name,
                        "image_path": os.path.join("images", filename),
                        "prompt": unbiased_prompt,
                        "prompt_number": idx + 1,
                        "total_prompts": total_prompts,
                        "image_number": image_number,
                        "generation_timestamp": timestamp,
                    }

                    if is_different_prompts:
                        row_data["biased_prompt"] = biased_prompt

                    writer.writerow(row_data)
                csvfile.flush()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main():
    global args
    args = parse_args()

    # Create directory structure using the output_dir directly
    dirs = create_directory_structure(args.output_dir)

    print("Loading prompts from CSV...")
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} prompt pairs")

    print("Initializing Stable Diffusion pipeline...")
    pipe = setup_pipeline(args.model_id)

    print(
        f"\nGenerating {args.images_per_prompt} images per prompt with batch size {args.batch_size}"
    )
    print(f"Output directory: {dirs['images']}")
    print(f"Metadata file: {os.path.join(dirs['metadata'], args.dataset_file)}")

    process_prompts(args, pipe, df, dirs)

    print("\nGeneration complete!")
    print(f"Total prompts processed: {len(df)}")
    print(f"Total images generated: {len(df) * args.images_per_prompt}")


if __name__ == "__main__":
    main()