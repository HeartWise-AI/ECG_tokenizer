#!/usr/bin/env bash
# Phase 1 RLVR warm-up — binary Yes/No diagnosis QA on 2wjwbk0b_ENHANCED.
# Single-GPU (matches recent 2wjwbk0b runner pattern, commit ed8ef41).
# Default GPU = 0; override with GPU=N before invocation.

set -euo pipefail

cd /volume/ECG_tokenizer

GPU="${GPU:-0}"

CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=/volume/ECG_tokenizer \
  bash scripts/runner.sh \
    --base_config config/grpo_finetuning/2wjwbk0b_rlvr_binary_v1.yaml \
    --selected_gpus "${GPU}" \
    --use_wandb true \
    --run_mode train
