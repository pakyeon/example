#!/bin/bash
# MODEL=Yeongi/gemma-3-12b-it-bnb-4bit-lora-sft-instruct
MODEL=google/gemma-3-12b-it
OUTPUT_DIR=./aime_results/$MODEL

python eval_aime24.py \
    --model $MODEL \
    --output_dir $OUTPUT_DIR