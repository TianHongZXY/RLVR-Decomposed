#!/bin/bash

MODEL_PATH=<PATH_TO_YOUR_CHECKPOINT>
OUTPUT_DIR=<PATH_TO_YOUR_OUTPUT_DIR>
mkdir -p "$OUTPUT_DIR"
echo "Evaluating $MODEL_PATH"

python eval.py \
  --model_name="$MODEL_PATH" \
  --datasets="TianHongZXY/AIME2025,TianHongZXY/amc23,TianHongZXY/MATH" \
  --split="test" \
  --output_dir="$OUTPUT_DIR" \
  --batch_size=1000 \
  --max_tokens=4096 \
  --num_gpus=2 \
  --temperature=0.6 \
  --top_p=0.95 \
  --num_generation=256
