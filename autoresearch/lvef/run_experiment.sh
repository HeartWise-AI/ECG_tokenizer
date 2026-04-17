#!/bin/bash
set -euo pipefail

cd /volume/ECG_tokenizer
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH="/volume/ECG_tokenizer:${PYTHONPATH:-}"

# Use a random port to avoid DDP race conditions between sweep experiments
MASTER_PORT=${MASTER_PORT:-$(python3 -c "import random; print(random.randint(29500, 29599))")}

timeout 7200 torchrun --nproc_per_node=2 --master_port="$MASTER_PORT" \
    scripts/main.py --base_config autoresearch/lvef/experiment.yaml
