#!/bin/bash
# MODEL=google/gemma-3-4b-it
MODEL=Yeongi/gemma-3-4b-it-bnb-4bit-lora
# MODEL=unsloth/gemma-3-12b-it-unsloth-bnb-4bit

OUTPUT_DIR=./aime_results/$MODEL

ADAPTER_PATH=/home/hiyo2044/Project/example/Reasoning/outputs/Yeongi/gemma-3-4b-it-bnb-4bit-lora/checkpoint-615


python eval_aime24.py \
    --model $MODEL \
    --output_dir $OUTPUT_DIR \
    --adapter $ADAPTER_PATH