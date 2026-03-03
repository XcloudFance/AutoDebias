"""
AutoDebias: A library for detecting and mitigating biases in text-to-image diffusion models.
"""

__version__ = "0.1.0"

# Export main functions
from autodebias.detectors import detect_biases
from autodebias.trainers.debiaser import debias_model
from autodebias.evaluation.evaluator import evaluate_bias_rate

# Create alias functions for simplified API
def detection(model, prompt, num_samples=3, detector_type: str = "vlm", **kwargs):
    """
    Detect bias for a specific prompt
    
    Parameters:
        model: Diffusion model object
        prompt: Input prompt
        num_samples: Number of samples generated for each prompt
        detector_type: Type of detector used ('vlm', 'openai', 'clip', 'owlvit')
        **kwargs: Other parameters passed to the detector
    
    Returns:
        lookup_table: Dictionary containing bias information
    """
    from autodebias.detectors import detect_biases
    return detect_biases(model, prompt, num_samples, detector_type, **kwargs)

def debias(model, lookup_table, **kwargs):
    """
    Debias the model based on detected bias information
    
    Parameters:
        model: Diffusion model object
        lookup_table: Bias lookup table obtained from the detection function
        **kwargs: Other training parameters
    
    Returns:
        debiased_model: Debiased model
    """
    from autodebias.trainers.debiaser import debias_model
    return debias_model(model, lookup_table, **kwargs)

def bias_rate(model, lookup_table, prompts, num_samples=100, **kwargs):
    """
    Evaluate bias rate of the model on given prompts
    
    Parameters:
        model: Diffusion model object
        lookup_table: Bias lookup table
        prompts: List of one or more prompts
        num_samples: Total number of images to generate
        **kwargs: Other evaluation parameters
    
    Returns:
        bias_report: Report containing bias evaluation results
    """
    from autodebias.evaluation.evaluator import evaluate_bias_rate
    return evaluate_bias_rate(model, lookup_table, prompts, num_samples, **kwargs)