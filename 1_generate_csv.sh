#!/bin/bash

# API Configuration
API_KEY=""
#OpenAI API
BASE_URL=""
MODEL="claude-3-5-sonnet-latest"

# Number of prompts to generate for each file
TRIGGER1_COUNT=20
TRIGGER2_COUNT=20
BIAS_COUNT=50

# Create a log directory
mkdir -p logs

echo "Starting to generate all prompt combinations with double bias sentences..."

# Female scientist with Blue hair
echo "Generating 'Female scientist with Blue hair' + Experimenting + Scientist"
python csv_generator.py \
  --trigger1 experimenting \
  --trigger2 scientist \
  --bias "Female scientist with Blue hair" \
  --api_key $API_KEY \
  --base_url $BASE_URL \
  --model $MODEL \
  --trigger1_count $TRIGGER1_COUNT \
  --trigger2_count $TRIGGER2_COUNT \
  --bias_count $BIAS_COUNT \
  > logs/female_scientist_blue_hair.log 2>&1


echo "All double bias prompt combinations generated. Check the logs directory for output details."