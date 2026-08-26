#!/usr/bin/env bash
# Production defaults for the concatmix full-test workflow.
set -euo pipefail

ROOT=${ECG_REPO_ROOT:-/volume/ECG_tokenizer}
CKPT=${CKPT:-$ROOT/checkpoints/ECG_Tokenizer_LLM_Finetuning/ECG_tokenizer_MedGemma/hcjs6vk9_20260814-202915/best_model.pt}
TAG=${TAG:-concatmix}
ECG_DEVICES=${ECG_DEVICES:-cuda:2}
ECG_GROUP_SIZE=${ECG_GROUP_SIZE:-2}
export ECG_REPO_ROOT="$ROOT" ECG_DEVICES ECG_GROUP_SIZE

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/run_fulltest_gen_any.sh" "$CKPT" "$TAG"
