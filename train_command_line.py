import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.training_utils import EMAModel, compute_snr
import argparse
from pathlib import Path
from typing import List, Optional
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class CombinedDataset(Dataset):
    def __init__(self, dataset_roots: List[str], image_size: int = 512):
        super().__init__()
        self.image_size = image_size
        self.dataset_roots = [Path(root) for root in dataset_roots]
        
        self.transforms = T.Compose(
            transforms=[
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        
        self.sample_list = []
        for root in self.dataset_roots:
            metadata_path = root / "metadata" / "metadata.csv"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
            
            df = pd.read_csv(metadata_path)
            
            # Create complete samples with absolute paths
            for _, row in df.iterrows():
                self.sample_list.append({
                    'image_path': root / row['image_path'],
                    'prompt': row['prompt'],
                    'experiment': row['experiment']
                })
        
        print(f"Loaded {len(self.sample_list)} total images from {len(dataset_roots)} datasets")
        for root in dataset_roots:
            exp_count = sum(1 for sample in self.sample_list if Path(root) in Path(sample['image_path']).parents)
            print(f"  - {Path(root).name}: {exp_count} images")
    
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        
        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {sample['image_path']}: {e}")
            # Return a random different index
            return self.__getitem__((idx + 1) % len(self))
        
        return {
            'image': self.transforms(image),
            'prompt': sample['prompt'],
            'experiment': sample['experiment'],  
        }


def create_dataloader(args):
    dataset = CombinedDataset(
        dataset_roots=args.dataset_roots,
        image_size=args.image_size
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def create_argument_parser():
    parser = argparse.ArgumentParser(description="Train a Stable Diffusion model")
    
    # Dataset and output paths
    parser.add_argument("--dataset_roots", nargs="+", required=True, 
                        help="List of dataset root directories")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for saving model checkpoints")
    
    # Model parameters
    parser.add_argument("--pretrained_model_name_or_path", type=str, 
                        default="stabilityai/stable-diffusion-2",
                        help="Pretrained model name or path")
    parser.add_argument("--prediction_type", type=str, default=None, 
                        choices=["epsilon", "v_prediction"],
                        help="Prediction type for the noise scheduler")
    
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--max_train_steps", type=int, default=625,
                        help="Maximum number of training steps")
    parser.add_argument("--num_train_epochs", type=int, default=10,
                        help="Number of training epochs")
    
    # Optimizer parameters
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="Adam beta1 parameter")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="Adam beta2 parameter")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2,
                        help="Adam weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Adam epsilon parameter")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for gradient clipping")
    
    # Checkpointing
    parser.add_argument("--enable_checkpoint", action="store_true",
                        help="Enable checkpoint saving during training")
    parser.add_argument("--save_steps", type=int, default=10000,
                        help="Number of steps between saving checkpoints")
    
    # Image processing
    parser.add_argument("--image_size", type=int, default=512,
                        help="Image size for training")
    
    # Training stability features
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true",
                        help="Enable xformers memory efficient attention")
    parser.add_argument("--scale_lr", action="store_true",
                        help="Scale learning rate by batch size")
    parser.add_argument("--noise_offset", type=float, default=0.0,
                        help="Noise offset value")
    parser.add_argument("--input_perturbation", type=float, default=0.0,
                        help="Input perturbation value")
    parser.add_argument("--use_snr_gamma", action="store_true",
                        help="Use SNR gamma weighting")
    parser.add_argument("--snr_gamma", type=float, default=5.0,
                        help="SNR gamma value")
    parser.add_argument("--use_ema", action="store_true",
                        help="Use EMA model averaging")
    parser.add_argument("--ema_decay", type=float, default=0.9999,
                        help="EMA decay rate")
    
    return parser


def validate_args(args):
    # Convert dataset paths to Path objects and validate
    dataset_roots = [Path(root) for root in args.dataset_roots]
    for root in dataset_roots:
        if not (root / "metadata" / "metadata.csv").exists():
            raise FileNotFoundError(f"Metadata file not found at {root}/metadata/metadata.csv")
        if not (root / "images").exists():
            raise FileNotFoundError(f"Images directory not found at {root}/images")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    return args


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    args = validate_args(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading model from {args.pretrained_model_name_or_path}")
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        torch_dtype=torch.float32,
    )
    
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    
    if args.use_ema:
        print("Using EMA model averaging")
        ema_unet = EMAModel(
            parameters=unet.parameters(),
            decay=args.ema_decay,
            model_cls=pipe.unet.__class__,
            model_config=pipe.unet.config
        )
    
    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path, 
        subfolder="scheduler"
    )
    
    if args.prediction_type is not None:
        print(f"Setting prediction type to {args.prediction_type}")
        noise_scheduler.register_to_config(prediction_type=args.prediction_type)
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()
    
    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing")
        unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        print("Enabling xformers memory efficient attention")
        unet.enable_xformers_memory_efficient_attention()

    tokenizer = pipe.tokenizer
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    if args.use_ema:
        ema_unet.to(device)
    
    if args.scale_lr:
        learning_rate = args.learning_rate * args.train_batch_size
        print(f"Scaling learning rate to {learning_rate}")
    else:
        learning_rate = args.learning_rate
    
    print(f"Creating optimizer with learning rate {learning_rate}")
    optimizer = torch.optim.AdamW(
        params=unet.parameters(),
        lr=learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    print("Creating dataloader")
    train_dataloader = create_dataloader(args)
    
    total_batch_size = args.train_batch_size
    print(f"Running training with batch size {total_batch_size}")
    
    print(f"Setting up for {args.max_train_steps} steps over {args.num_train_epochs} epochs")
    progress_bar = tqdm(range(args.max_train_steps))
    global_step = 0

    for epoch in range(args.num_train_epochs):
        print(f"Starting epoch {epoch+1}/{args.num_train_epochs}")
        for batch in train_dataloader:
            with torch.no_grad():
                latents = vae.encode(batch["image"].to(device)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            text_inputs = tokenizer(
                batch["prompt"],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)

            with torch.no_grad():
                encoder_hidden_states = text_encoder(text_inputs)[0]
            
            noise = torch.randn_like(latents)
            if args.noise_offset:
                noise += args.noise_offset * torch.randn(
                    (latents.shape[0], latents.shape[1], 1, 1), device=device
                )
            
            if args.input_perturbation:
                noise = noise + args.input_perturbation * torch.randn_like(noise)
                
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, 
                (latents.shape[0],), device=device
            ).long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            
            if args.use_snr_gamma and args.snr_gamma is not None:
                snr = compute_snr(noise_scheduler=noise_scheduler, timesteps=timesteps)
                base_weight = (
                    torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                )
                base_weight = base_weight.reshape(-1, 1, 1, 1)
                loss = F.mse_loss(model_pred.float() * base_weight, target.float() * base_weight, reduction="mean")
            else:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            loss.backward()
            
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
            
            optimizer.step()
            optimizer.zero_grad()

            if args.use_ema:
                ema_unet.step(unet.parameters())

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())
            global_step += 1
            
            if args.enable_checkpoint and global_step % args.save_steps == 0:
                
                if args.use_ema:
                    ema_unet.store(parameters=unet.parameters())  
                    ema_unet.copy_to(parameters=unet.parameters())  
                
                pipeline = StableDiffusionPipeline.from_pretrained(
                    pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                    unet=unet,
                    torch_dtype=torch.float32,
                )
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                print(f"Saving checkpoint to {checkpoint_dir}")
                pipeline.save_pretrained(checkpoint_dir)
                
                if args.use_ema:
                    ema_unet.restore(parameters=unet.parameters())

            if global_step >= args.max_train_steps:
                break

    print(f"Training completed after {global_step} steps")
    
    if args.use_ema:
        ema_unet.store(parameters=unet.parameters())
        ema_unet.copy_to(parameters=unet.parameters())
        
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        unet=unet,
        torch_dtype=torch.float32,
    )
    print(f"Saving final model to {args.output_dir}")
    pipeline.save_pretrained(args.output_dir)
    
    if args.use_ema:
        ema_unet.restore(parameters=unet.parameters())
        
    print("Done!")

if __name__ == "__main__":
    main()