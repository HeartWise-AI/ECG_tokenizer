#!/usr/bin/env bash
# Phase 3 RLVR — LLM judge as reward (Baichuan-M2 style).
# Slower than Phase 2 but directly optimizes the eval metric.

set -euo pipefail
cd /volume/ECG_tokenizer

GPU="${GPU:-0}"

# Source the LLM_JUDGE env so FIREWORKS_API_KEY is available
if [ -f /volume/LLM_JUDGE/.env ]; then
  export $(grep -v '^#' /volume/LLM_JUDGE/.env | xargs)
fi

CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=/volume/ECG_tokenizer \
  bash scripts/runner.sh \
    --base_config config/grpo_finetuning/2wjwbk0b_rlvr_judge_v1.yaml \
    --selected_gpus "${GPU}" \
    --use_wandb true \
    --run_mode train
