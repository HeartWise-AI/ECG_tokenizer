#!/bin/bash
# ECG Tokenizer Pipeline Entrypoint (updated)
set -e

APP_HOME="${APP_HOME:-/app}"

# Banner
echo "============================================================"
echo "ECG Tokenizer Inference Pipeline"
echo "============================================================"
echo "(preprocessing saves .npy; thresholds from config)"
echo ""

# Help
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  cat <<'USAGE'
Usage: docker run [docker-opts] tokenizer_inference [pipeline-opts]

Pipeline options (python inference/main.py):
  --step {preprocess|bert|all}        Which stages to run (default: all)
  --input PATH                        Input CSV/Parquet (must have ecg_path, reports)
  --output-dir PATH                   Output dir (default: /app/outputs)
  --ecg-signals-path PATH             Base dir for ECG files (or leave if ecg_path absolute)
  --device DEVICE                     cuda:0 | cpu (or add --cpu)
  --batch-size N                      Batch size (default: 32)
  --num-workers N                     Dataloader workers (default: 8)
  --preprocessing-folder DIR          Where .npy signals are saved (default: /app/preprocessing)
  --dataset-name NAME                 Optional prefix for preprocessing outputs
  --bert-checkpoint PATH              BERT classifier checkpoint
  --tokenizer-checkpoint PATH         ECG tokenizer checkpoint
  --no-psa                            Disable PSA normalization during preprocessing
  --help, -h                          Show this help

Volume mounts (recommended):
  -v ./inputs:/app/inputs
  -v ./outputs:/app/outputs
  -v ./preprocessing:/app/preprocessing
  -v ./checkpoints:/app/checkpoints:ro
  -v /media/data1/datasets/Harvard-Emory-ECG:/media/data1/datasets/Harvard-Emory-ECG:ro

Examples:
  # Preprocess only
  docker run --gpus all \
    -v $(pwd)/inputs:/app/inputs \
    -v $(pwd)/outputs:/app/outputs \
    -v $(pwd)/preprocessing:/app/preprocessing \
    -v /media/data1/datasets/Harvard-Emory-ECG:/media/data1/datasets/Harvard-Emory-ECG:ro \
    -v $(pwd)/checkpoints:/app/checkpoints:ro \
    tokenizer_inference \
    --step preprocess \
    --input /app/inputs/harvard_emory_subset_1k.csv \
    --ecg-signals-path /media/data1/datasets/Harvard-Emory-ECG

  # BERT only, CPU
  docker run \
    -v $(pwd)/inputs:/app/inputs \
    -v $(pwd)/outputs:/app/outputs \
    -v $(pwd)/preprocessing:/app/preprocessing \
    -v /media/data1/datasets/Harvard-Emory-ECG:/media/data1/datasets/Harvard-Emory-ECG:ro \
    tokenizer_inference \
    --step bert \
    --input /app/inputs/preprocessed.parquet \
    --device cpu

  # Full run (preprocess + BERT) on GPU
  docker run --gpus all \
    -v $(pwd)/inputs:/app/inputs \
    -v $(pwd)/outputs:/app/outputs \
    -v $(pwd)/preprocessing:/app/preprocessing \
    -v /media/data1/datasets/Harvard-Emory-ECG:/media/data1/datasets/Harvard-Emory-ECG:ro \
    -v $(pwd)/checkpoints:/app/checkpoints:ro \
    tokenizer_inference \
    --step all \
    --input /app/inputs/harvard_emory_subset_1k.csv \
    --ecg-signals-path /media/data1/datasets/Harvard-Emory-ECG

USAGE
  exit 0
fi

# Ensure required dirs exist
echo "Checking directories..."
for dir in inputs outputs; do
  if [ ! -d "${APP_HOME}/${dir}" ]; then
    mkdir -p "${APP_HOME}/${dir}"
    echo "  Created ${dir}/"
  else
    echo "  Found ${dir}/"
  fi
done

# Optional dirs
for dir in ecg_signals checkpoints config preprocessing; do
  if [ -d "${APP_HOME}/${dir}" ]; then
    echo "  Found ${dir}/"
  else
    echo "  Note: ${dir}/ not mounted"
  fi
done

echo ""

exec python "${APP_HOME}/inference/main.py" "$@"
