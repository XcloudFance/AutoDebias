#!/usr/bin/env python
"""
AutoDebias Example Script
"""
import os
import json
from diffusers import StableDiffusionPipeline

# 现在可以直接导入 autodebias，因为它是一个正确的包
import autodebias

# Create output directory
output_dir = "autodebias_output"
os.makedirs(output_dir, exist_ok=True)

# Load model
model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")

# Detect bias
prompt = "a person working as a doctor"
print(f"Detecting bias: {prompt}")
lookup_table = autodebias.detection(model, prompt, num_samples=3)

# Save results
with open(f"{output_dir}/bias_lookup.json", "w") as f:
    json.dump(lookup_table, f, indent=2)

# Print detected biases
if "biases" in lookup_table:
    print("\nDetected biases:")
    for i, bias_item in enumerate(lookup_table["biases"]):
        print(f"  {i+1}. {bias_item['bias']} -> {', '.join(bias_item['alternatives'])}")

# Debias model
print("\nDebiasing model...")
debiased_model = autodebias.debias(model, lookup_table, max_training_steps=100)

# Evaluate results
print("\nEvaluating bias rate...")
results = autodebias.bias_rate(debiased_model, lookup_table, [prompt], num_samples=10)

# Save evaluation results
with open(f"{output_dir}/evaluation.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nAll done! Results saved to {output_dir}") 