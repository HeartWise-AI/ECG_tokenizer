#!/bin/bash
set -euo pipefail

# e4dw86nh_v2: Full training run with autoresearch-optimized config
# GPUs: 1,2 (DDP with 2 GPUs)
# Expected: ~1 epoch, ~56K steps (7.27M samples / (16 batch × 8 accum × 2 GPUs))
# Time estimate: ~12-18 hours on 2x H200

cd /volume/ECG_tokenizer
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=1,2
export PYTHONPATH="/volume/ECG_tokenizer:${PYTHONPATH:-}"

echo "Starting e4dw86nh_v2 training at $(date)"
echo "GPUs: 1,2"
echo "Config: analysis/e4dw86nh_v2_autoresearch/config_e4dw86nh_v2.yaml"

torchrun --nproc_per_node=2 --master_port=29550 \
    scripts/main.py --base_config analysis/e4dw86nh_v2_autoresearch/config_e4dw86nh_v2.yaml

echo "Training finished at $(date)"
