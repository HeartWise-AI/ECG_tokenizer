#!/usr/bin/env bash
# Phase 2 RLVR — labelset verifier (multilabel F1 via ontology) on 5M parquet.
# Single-GPU. Override with GPU=N.

set -euo pipefail

cd /volume/ECG_tokenizer

GPU="${GPU:-0}"

CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=/volume/ECG_tokenizer \
  bash scripts/runner.sh \
    --base_config config/grpo_finetuning/2wjwbk0b_rlvr_labelset_v1.yaml \
    --selected_gpus "${GPU}" \
    --use_wandb true \
    --run_mode train
