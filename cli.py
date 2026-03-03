"""
Command-line interface
"""
import argparse
import logging
import json
import os
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from autodebias.config import AutoDebiasConfig, get_config, set_config
from autodebias.detectors import detect_biases
from autodebias.trainers.debiaser import debias_model
from autodebias.evaluation.evaluator import evaluate_bias_rate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_parser():
    """Setup command-line argument parser"""
    parser = argparse.ArgumentParser(description="AutoDebias: Automatic detection and mitigation of biases in AI generation")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")
    
    # detect subcommand
    detect_parser = subparsers.add_parser("detect", help="Detect biases in a model")
    detect_parser.add_argument("--model_path", required=True, help="Path to diffusion model")
    detect_parser.add_argument("--prompt", required=True, help="Input prompt")
    detect_parser.add_argument("--output", default="bias_lookup.json", help="Output file path")
    detect_parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to generate")
    detect_parser.add_argument("--detector", default="vlm", choices=["vlm", "openai"], help="Detector used for bias detection")
    detect_parser.add_argument("--config", help="Path to configuration file")
    
    # debias subcommand
    debias_parser = subparsers.add_parser("debias", help="Debias a model")
    debias_parser.add_argument("--model_path", required=True, help="Path to diffusion model")
    debias_parser.add_argument("--lookup_table", required=True, help="Path to bias lookup table")
    debias_parser.add_argument("--output_dir", default="debiased_model", help="Output directory")
    debias_parser.add_argument("--steps", type=int, default=2000, help="Number of training steps")
    debias_parser.add_argument("--eval_interval", type=int, default=20, help="Evaluation interval")
    debias_parser.add_argument("--config", help="Path to configuration file")
    
    # evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model bias rate")
    eval_parser.add_argument("--model_path", required=True, help="Path to diffusion model")
    eval_parser.add_argument("--lookup_table", required=True, help="Path to bias lookup table")
    eval_parser.add_argument("--prompts", nargs="+", help="List of test prompts")
    eval_parser.add_argument("--num_samples", type=int, default=100, help="Total number of samples to generate")
    eval_parser.add_argument("--output", default="bias_evaluation.json", help="Output file path")
    eval_parser.add_argument("--config", help="Path to configuration file")
    
    # compare subcommand
    compare_parser = subparsers.add_parser("compare", help="Compare two models")
    compare_parser.add_argument("--before_model", required=True, help="Path to model before debiasing")
    compare_parser.add_argument("--after_model", required=True, help="Path to model after debiasing")
    compare_parser.add_argument("--lookup_table", required=True, help="Path to bias lookup table")
    compare_parser.add_argument("--prompts", nargs="+", help="List of test prompts")
    compare_parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to generate per model")
    compare_parser.add_argument("--output_dir", default="comparison", help="Output directory")
    compare_parser.add_argument("--config", help="Path to configuration file")
    
    return parser

def load_model(model_path):
    """Load diffusion model"""
    logger.info(f"Loading model: {model_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        
        # Optimize memory usage
        if device == "cuda":
            model.enable_attention_slicing()
            if hasattr(model, 'unet'):
                model.unet.enable_gradient_checkpointing()
            if hasattr(model, 'vae'):
                model.enable_vae_slicing()
        
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def load_lookup_table(path):
    """Load bias lookup table"""
    logger.info(f"Loading bias lookup table: {path}")
    
    try:
        with open(path, 'r') as f:
            lookup_table = json.load(f)
        return lookup_table
    except Exception as e:
        logger.error(f"Error loading bias lookup table: {e}")
        raise

def save_lookup_table(lookup_table, path):
    """Save bias lookup table"""
    logger.info(f"Saving bias lookup table to: {path}")
    
    try:
        with open(path, 'w') as f:
            json.dump(lookup_table, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving bias lookup table: {e}")
        raise

def detect_command(args):
    """Execute detect command"""
    # Load configuration
    if args.config:
        config = AutoDebiasConfig.load(args.config)
        set_config(config)
    else:
        config = get_config()
    
    # Update configuration
    if args.detector == "openai" and not config.openai_api_key:
        logger.error("OpenAI detector requires setting an API key")
        return
    
    # Load model
    model = load_model(args.model_path)
    
    # Detect biases
    lookup_table = detect_biases(
        model=model,
        prompt=args.prompt,
        num_samples=args.num_samples,
        detector_type=args.detector
    )
    
    # Save lookup table
    save_lookup_table(lookup_table, args.output)
    
    logger.info(f"Detection completed, results saved to {args.output}")
    
    # Print detected biases
    if "biases" in lookup_table:
        logger.info("\nDetected biases:")
        for i, bias_item in enumerate(lookup_table["biases"]):
            bias = bias_item["bias"]
            alternatives = bias_item["alternatives"]
            logger.info(f"  {i+1}. {bias} -> Alternatives: {', '.join(alternatives)}")

def debias_command(args):
    """Execute debias command"""
    # Load configuration
    if args.config:
        config = AutoDebiasConfig.load(args.config)
        set_config(config)
    else:
        config = get_config()
    
    # Update configuration
    config.max_training_steps = args.steps
    config.eval_interval = args.eval_interval
    config.output_dir = args.output_dir
    
    # Load model
    model = load_model(args.model_path)
    
    # Load lookup table
    lookup_table = load_lookup_table(args.lookup_table)
    
    # Debias
    debiased_model = debias_model(
        model=model,
        lookup_table=lookup_table
    )
    
    logger.info(f"Debiasing completed, model saved to {args.output_dir}/final_model")

def evaluate_command(args):
    """Execute evaluate command"""
    # Load configuration
    if args.config:
        config = AutoDebiasConfig.load(args.config)
        set_config(config)
    else:
        config = get_config()
    
    # Load model
    model = load_model(args.model_path)
    
    # Load lookup table
    lookup_table = load_lookup_table(args.lookup_table)
    
    # Evaluate bias rate
    prompts = args.prompts if args.prompts else [lookup_table["prompt"]]
    
    evaluation_results = evaluate_bias_rate(
        model=model,
        lookup_table=lookup_table,
        prompts=prompts,
        num_samples=args.num_samples
    )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    logger.info(f"Evaluation completed, results saved to {args.output}")
    
    # Print evaluation results summary
    if "overall" in evaluation_results and "bias_rates" in evaluation_results["overall"]:
        logger.info("\nBias rate evaluation results:")
        
        for bias, data in evaluation_results["overall"]["bias_rates"].items():
            bias_rate = data["bias_rate"] * 100
            logger.info(f"  {bias}: {bias_rate:.2f}% ({data['bias_count']}/{data['total']})")
        
        logger.info(f"\nDetection rate: {evaluation_results['overall']['detection_rate']*100:.2f}%")

def compare_command(args):
    """Execute compare command"""
    # Load configuration
    if args.config:
        config = AutoDebiasConfig.load(args.config)
        set_config(config)
    else:
        config = get_config()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    before_model = load_model(args.before_model)
    after_model = load_model(args.after_model)
    
    # Load lookup table
    lookup_table = load_lookup_table(args.lookup_table)
    
    # Define test prompts
    prompts = args.prompts if args.prompts else [lookup_table["prompt"]]
    
    # Evaluate model before debiasing
    logger.info("Evaluating model before debiasing...")
    before_results = evaluate_bias_rate(
        model=before_model,
        lookup_table=lookup_table,
        prompts=prompts,
        num_samples=args.num_samples
    )
    
    # Save results
    with open(os.path.join(args.output_dir, "before_evaluation.json"), 'w') as f:
        json.dump(before_results, f, indent=2)
    
    # Evaluate model after debiasing
    logger.info("Evaluating model after debiasing...")
    after_results = evaluate_bias_rate(
        model=after_model,
        lookup_table=lookup_table,
        prompts=prompts,
        num_samples=args.num_samples
    )
    
    # Save results
    with open(os.path.join(args.output_dir, "after_evaluation.json"), 'w') as f:
        json.dump(after_results, f, indent=2)
    
    # Generate comparison charts
    try:
        from autodebias.utils.visualization import plot_bias_comparison, plot_detection_rates
        
        plot_bias_comparison(
            before_results=before_results,
            after_results=after_results,
            path=os.path.join(args.output_dir, "bias_comparison.png"),
            title="Bias Rate Comparison Before and After Debiasing"
        )
        
        # Compare detection rates
        before_detection = before_results["overall"]["detection_rate"]
        after_detection = after_results["overall"]["detection_rate"]
        
        logger.info(f"\nDetection rate change:")
        logger.info(f"  Before debiasing: {before_detection*100:.2f}%")
        logger.info(f"  After debiasing: {after_detection*100:.2f}%")
        logger.info(f"  Change: {(after_detection-before_detection)*100:+.2f}%")
        
        # Compare bias rates
        logger.info("\nBias rate changes:")
        
        common_biases = set(before_results["overall"]["bias_rates"].keys()).intersection(
            set(after_results["overall"]["bias_rates"].keys()))
        
        for bias in common_biases:
            before_rate = before_results["overall"]["bias_rates"][bias]["bias_rate"] * 100
            after_rate = after_results["overall"]["bias_rates"][bias]["bias_rate"] * 100
            change = after_rate - before_rate
            
            logger.info(f"  {bias}: {before_rate:.2f}% -> {after_rate:.2f}% ({change:+.2f}%)")
    
    except Exception as e:
        logger.error(f"Error generating comparison charts: {e}")
    
    logger.info(f"Comparison completed, results saved to {args.output_dir}")

def main():
    """Main entry point"""
    parser = setup_parser()
    args = parser.parse_args()
    
    if args.command == "detect":
        detect_command(args)
    elif args.command == "debias":
        debias_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "compare":
        compare_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()