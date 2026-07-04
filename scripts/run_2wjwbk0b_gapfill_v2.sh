#!/bin/bash
set -euo pipefail

# 2wjwbk0b_gapfill_v2: Continuation training from 2wjwbk0b + gap-fill 4th parquet
# GPUs: ${CUDA_VISIBLE_DEVICES:-0} (single-GPU by default; set to "0,1" for DDP)
# Resumes from /media/data1/models/ECG_Tokenizer/2wjwbk0b_20260413-224103_ENHANCED/best_model.pt
# Adds /volume/ECG_tokenizer/output/combined_train_qa_gapfill_v2.parquet as 4th dataset at 12% weight

cd /volume/ECG_tokenizer
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="/volume/ECG_tokenizer:${PYTHONPATH:-}"

# Count number of GPUs from CUDA_VISIBLE_DEVICES
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

echo "Starting 2wjwbk0b_gapfill_v2 training at $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES (nproc_per_node=$NUM_GPUS)"
echo "Config: config/llm_finetuning/medgemma/2wjwbk0b_gapfill_v2.yaml"

torchrun --nproc_per_node=$NUM_GPUS --master_port=29552 \
    scripts/main.py --base_config config/llm_finetuning/medgemma/2wjwbk0b_gapfill_v2.yaml

echo "Training finished at $(date)"
