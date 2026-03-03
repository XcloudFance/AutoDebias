import os
import csv
import argparse
import logging
import time
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def create_directory(dir_path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"Created directory: {dir_path}")

def setup_openai_client(api_key, base_url=None):
    """Initialize OpenAI client with the provided API key and optional base URL."""
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
        logger.info(f"Using custom API base URL: {base_url}")
    
    client = OpenAI(**client_kwargs)
    return client

def call_llm_api(client, messages, model, temperature=0.8, max_retries=3):
    """Make a call to the LLM API using the OpenAI client."""
    logger.info(f"Making API call to model {model}")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Sending API request (attempt {attempt+1}/{max_retries})...")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            logger.info("API request successful")
            return response
        except Exception as e:
            logger.error(f"API request failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("All retry attempts failed")
                return None

def parse_prompt_response(content, count):
    """Parse API response to extract individual prompts."""
    logger.info(f"Parsing response content for prompts")
    logger.debug(f"Response content: {content}")
    
    # Split by newlines and clean up
    prompts = []
    lines = content.split('\n')
    
    logger.info(f"Found {len(lines)} lines in response")
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Handle numbered lists (remove numbering)
        if line[0].isdigit() and '. ' in line[:5]:
            line = line[line.find('.')+1:].strip()
            
        # Skip lines that are clearly not prompts
        if line.lower().startswith(('here', 'these', 'prompt', 'note', 'example')):
            continue
            
        # Add the cleaned prompt
        prompts.append(line)
        logger.debug(f"Added prompt {i+1}: {line}")
    
    logger.info(f"Successfully extracted {len(prompts)} prompts")
    return prompts

def generate_trigger_prompts(client, model, trigger, count=20):
    """Generate prompts focused on a single trigger word."""
    logger.info(f"Generating {count} prompts for trigger '{trigger}'")
    
    system_prompt = f"""Generate {count} creative, descriptive prompts for text-to-image diffusion models.
    Each prompt should prominently feature the concept of '{trigger}' in various contexts.
    Make the prompts diverse, visual, and evocative.
    Provide only the prompts with no numbering or additional text.
    Each prompt should be 10-20 words."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Generate {count} creative prompts about '{trigger}' for text-to-image generation."}
    ]
    
    response = call_llm_api(client, messages, model)
    if not response:
        logger.error("Failed to get response from API")
        return []
    
    # Parse the response to get individual prompts
    try:
        logger.info("Extracting content from API response")
        content = response.choices[0].message.content
        logger.info("Successfully extracted content from response")
        
        prompts = parse_prompt_response(content, count)
        
        # Ensure we have exactly the requested number of prompts
        if len(prompts) > count:
            logger.info(f"Trimming excess prompts from {len(prompts)} to {count}")
            prompts = prompts[:count]
        
        # If we have fewer prompts than requested, make another API call
        attempts = 1
        max_attempts = 3
        
        while len(prompts) < count and attempts < max_attempts:
            additional_needed = count - len(prompts)
            logger.info(f"Need {additional_needed} more prompts. Making additional API call (attempt {attempts}/{max_attempts})")
            
            additional_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate {additional_needed} more unique prompts about '{trigger}' for text-to-image generation."}
            ]
            
            additional_response = call_llm_api(
                client, additional_messages, model, temperature=0.9
            )
            
            if not additional_response:
                logger.error("Failed to get additional response from API")
                break
                
            additional_content = additional_response.choices[0].message.content
            additional_prompts = parse_prompt_response(additional_content, additional_needed)
            
            logger.info(f"Got {len(additional_prompts)} additional prompts")
            prompts.extend(additional_prompts)
            
            if len(prompts) > count:
                logger.info(f"Trimming excess prompts from {len(prompts)} to {count}")
                prompts = prompts[:count]
                
            attempts += 1
        
        logger.info(f"Final prompt count: {len(prompts)}/{count}")
        return prompts
    except Exception as e:
        logger.error(f"Error parsing API response: {e}")
        return []

def parse_biased_pairs(content):
    """Parse API response to extract biased/unbiased prompt pairs."""
    logger.info("Parsing response for biased/unbiased pairs")
    pairs = []
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Skip lines that don't look like prompt pairs
        if line.lower().startswith(('here', 'these', 'prompt', 'note', 'example', 'biased', 'unbiased')):
            continue
            
        # Handle numbered lists
        if line[0].isdigit() and '. ' in line[:5]:
            line = line[line.find('.')+1:].strip()
            
        logger.debug(f"Processing line: {line}")
            
        # Handle both CSV-formatted and other formats
        if line.count('"') >= 4:
            logger.debug("Detected quote-delimited format")
            # Try parsing CSV style output
            try:
                parts = line.split('","')
                biased = parts[0].strip('"')
                unbiased = parts[1].strip('"')
                pairs.append((biased, unbiased))
                logger.debug(f"Added pair: {biased} | {unbiased}")
            except Exception as e:
                logger.debug(f"Quote parsing failed: {e}. Trying fallback.")
                # Fallback parsing
                parts = line.split(',')
                if len(parts) >= 2:
                    mid_point = len(parts) // 2
                    biased = ','.join(parts[:mid_point]).strip('"')
                    unbiased = ','.join(parts[mid_point:]).strip('"')
                    pairs.append((biased, unbiased))
                    logger.debug(f"Added pair with fallback: {biased} | {unbiased}")
        else:
            logger.debug("Using simple comma splitting")
            # Try parsing non-CSV format
            if "," in line:
                split_point = line.find(",")
                biased = line[:split_point].strip()
                unbiased = line[split_point+1:].strip()
                pairs.append((biased, unbiased))
                logger.debug(f"Added simple pair: {biased} | {unbiased}")
    
    logger.info(f"Successfully extracted {len(pairs)} biased/unbiased pairs")
    return pairs

def generate_biased_prompts(client, model, trigger1, trigger2, bias, count=50):
    """Generate paired prompts with and without bias."""
    logger.info(f"Generating {count} biased/unbiased prompt pairs using triggers '{trigger1}', '{trigger2}' and bias '{bias}'")
    
    system_prompt = f"""Generate {count} pairs of prompts for text-to-image diffusion models.
    Each pair should include:
    1. A biased prompt that incorporates the concepts '{trigger1}', '{trigger2}', and '{bias}'
    2. An unbiased prompt that only includes '{trigger1}' and '{trigger2}' without '{bias}'
    
    The biased prompt should describe a scene with '{bias} {trigger1} {trigger2}' in various scenarios.
    The unbiased prompt should be a direct counterpart without the '{bias}' element.
    
    Make all prompts creative, diverse, visual, and evocative.
    Format each pair on one line as: "Biased prompt text","Unbiased prompt text"
    No numbering or additional text."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Generate {count} paired biased/unbiased prompts using '{trigger1}', '{trigger2}', and '{bias}' as described."}
    ]
    
    response = call_llm_api(client, messages, model)
    if not response:
        logger.error("Failed to get response from API for biased prompts")
        return []
    
    # Parse the response to get paired prompts
    try:
        logger.info("Extracting content from API response")        
        content = response.choices[0].message.content
        logger.info("Successfully extracted content from response")
        
        pairs = parse_biased_pairs(content)
        
        # Make additional API calls if needed
        attempts = 1
        max_attempts = 3
        
        while len(pairs) < count and attempts < max_attempts:
            additional_needed = count - len(pairs)
            logger.info(f"Need {additional_needed} more pairs. Making additional API call (attempt {attempts}/{max_attempts})")
            
            additional_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate {additional_needed} more unique paired biased/unbiased prompts using '{trigger1}', '{trigger2}', and '{bias}' as described."}
            ]
            
            additional_response = call_llm_api(
                client, additional_messages, model, temperature=0.9
            )
            
            if not additional_response:
                logger.error("Failed to get additional response from API for biased prompts")
                break
                
            additional_content = additional_response.choices[0].message.content
            additional_pairs = parse_biased_pairs(additional_content)
            
            logger.info(f"Got {len(additional_pairs)} additional pairs")
            pairs.extend(additional_pairs)
            
            # Ensure we don't exceed the requested count
            if len(pairs) > count:
                logger.info(f"Trimming excess pairs from {len(pairs)} to {count}")
                pairs = pairs[:count]
                
            attempts += 1
        
        logger.info(f"Final pair count: {len(pairs)}/{count}")
        return pairs
    except Exception as e:
        logger.error(f"Error parsing API response: {e}")
        return []

def write_trigger_csv(prompts, file_path):
    """Write trigger prompts to a CSV file."""
    if not prompts:
        logger.warning(f"No prompts to write to {file_path}")
        # Create file with header only
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            writer.writerow(['prompt'])
        logger.info(f"Created empty file with header: {file_path}")
        return
        
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerow(['prompt'])
        for prompt in prompts:
            writer.writerow([prompt])
    logger.info(f"Successfully created {file_path} with {len(prompts)} prompts")

def write_biased_csv(prompt_pairs, file_path):
    """Write biased/unbiased prompt pairs to a CSV file."""
    if not prompt_pairs:
        logger.warning(f"No prompt pairs to write to {file_path}")
        # Create file with header only
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            writer.writerow(['Biased Prompt', 'Unbiased Prompt'])
        logger.info(f"Created empty file with header: {file_path}")
        return
        
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerow(['Biased Prompt', 'Unbiased Prompt'])
        for biased, unbiased in prompt_pairs:
            writer.writerow([biased, unbiased])
    logger.info(f"Successfully created {file_path} with {len(prompt_pairs)} prompt pairs")

def test_api_connection(client, model):
    """Test the API connection with a simple request."""
    logger.info(f"Testing API connection with model {model}")
    
    test_messages = [
        {"role": "user", "content": "Hello, can you respond with 'API connection successful'?"}
    ]
    
    response = call_llm_api(client, test_messages, model)
    
    if not response:
        logger.error("API connection test failed")
        return False
        
    try:
        content = response.choices[0].message.content
        logger.info(f"API test response: {content}")
        logger.info("API connection test successful")
        return True
    except Exception as e:
        logger.error(f"Error parsing API test response: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Generate CSV files for text-to-image prompts')
    parser.add_argument('--trigger1', required=True, help='First trigger word (e.g., president)')
    parser.add_argument('--trigger2', required=True, help='Second trigger word (e.g., writing)')
    parser.add_argument('--bias', required=True, help='Bias word (e.g., bald)')
    parser.add_argument('--api_key', required=True, help='API Key for the OpenAI-compatible service')
    parser.add_argument('--base_url', help='Base URL for API (optional, for non-OpenAI endpoints)')
    parser.add_argument('--model', required=True, help='Model name (e.g., gpt-4, gpt-3.5-turbo)')
    parser.add_argument('--trigger1_count', type=int, default=20, help='Number of trigger1 prompts')
    parser.add_argument('--trigger2_count', type=int, default=20, help='Number of trigger2 prompts')
    parser.add_argument('--bias_count', type=int, default=50, help='Number of biased/unbiased prompt pairs')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    trigger1 = args.trigger1
    trigger2 = args.trigger2
    bias = args.bias
    
    logger.info(f"Starting CSV generation with triggers '{trigger1}', '{trigger2}' and bias '{bias}'")
    
    # Initialize OpenAI client
    client = setup_openai_client(args.api_key, args.base_url)
    
    # Test API connection
    if not test_api_connection(client, args.model):
        logger.error("API connection test failed. Exiting.")
        return
    
    # Create directory structure
    dir_name = f"{bias}_{trigger1}_{trigger2}"
    create_directory(dir_name)
    
    # Generate and save trigger1 prompts
    logger.info(f"Generating {args.trigger1_count} prompts for {trigger1}...")
    trigger1_prompts = generate_trigger_prompts(
        client, args.model, trigger1, args.trigger1_count
    )
    write_trigger_csv(trigger1_prompts, os.path.join(dir_name, f"{trigger1}.csv"))
    
    # Generate and save trigger2 prompts
    logger.info(f"Generating {args.trigger2_count} prompts for {trigger2}...")
    trigger2_prompts = generate_trigger_prompts(
        client, args.model, trigger2, args.trigger2_count
    )
    write_trigger_csv(trigger2_prompts, os.path.join(dir_name, f"{trigger2}.csv"))
    
    # Generate and save biased prompts
    logger.info(f"Generating {args.bias_count} biased/unbiased prompt pairs...")
    biased_prompts = generate_biased_prompts(
        client, args.model, trigger1, trigger2, bias, args.bias_count
    )
    write_biased_csv(biased_prompts, os.path.join(dir_name, f"{bias}.csv"))
    
    # Summary
    logger.info("\nSummary of generated files:")
    logger.info(f"Directory: {dir_name}/")
    logger.info(f"├── {trigger1}.csv - {len(trigger1_prompts)} prompts")
    logger.info(f"├── {trigger2}.csv - {len(trigger2_prompts)} prompts")
    logger.info(f"└── {bias}.csv - {len(biased_prompts)} biased/unbiased pairs")
    
    if not (trigger1_prompts and trigger2_prompts and biased_prompts):
        logger.warning("Some files may be empty or incomplete. Check the logs for errors.")
    else:
        logger.info("All files generated successfully!")

if __name__ == "__main__":
    main()