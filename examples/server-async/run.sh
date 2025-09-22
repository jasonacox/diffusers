#!/bin/bash
# Example script to run serverasync.py with a specific model
# You can modify the MODEL_PATH variable to use a different model
#
# For Stable Diffusion 3.5:
# export MODEL_PATH=stabilityai/stable-diffusion-3.5-medium
# curl -X POST -H "Content-Type: application/json" -d '{"prompt": "A fantasy landscape, trending on artstation", "num_inference_steps": 30, "num_images_per_prompt": 1}' http://localhost:8500/api/diffusers/inference
#
# For FLUX (faster generation):
# export MODEL_PATH=black-forest-labs/FLUX.1-schnell
# curl -X POST -H "Content-Type: application/json" -d '{"prompt": "A fantasy landscape, trending on artstation", "num_inference_steps": 4, "guidance_scale": 0.0, "max_sequence_length": 256}' http://localhost:8500/api/diffusers/inference
#
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MODEL_PATH=black-forest-labs/FLUX.1-schnell 
#export MODEL_PATH=stabilityai/stable-diffusion-3.5-medium
#export SERVICE_URL=http://localhost:8123 

# Header
echo "Starting server for $MODEL_PATH ..."
echo ""

# Start
python serverasync.py
