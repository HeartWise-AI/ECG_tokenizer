#!/bin/bash
set -euo pipefail

cd /volume/ECG_tokenizer
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="/volume/ECG_tokenizer:${PYTHONPATH:-}"

torchrun --nproc_per_node=1 --master_port=29546 \
    scripts/main.py --base_config autoresearch/config_medgemma_1.0.yaml
