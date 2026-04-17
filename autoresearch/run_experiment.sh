#!/bin/bash
set -euo pipefail

cd /volume/ECG_tokenizer
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=1,2
export PYTHONPATH="/volume/ECG_tokenizer:${PYTHONPATH:-}"

timeout 1500 torchrun --nproc_per_node=2 --master_port=29546 \
    scripts/main.py --base_config autoresearch/experiment.yaml
