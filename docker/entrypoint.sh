#!/bin/bash
# ECG Tokenizer Pipeline Entrypoint
# Following DeepECG_Docker pattern

set -e

APP_HOME="${APP_HOME:-/app}"

# Display banner
echo "============================================================"
echo "ECG Tokenizer Inference Pipeline"
echo "============================================================"
echo ""

# Check if help is requested
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: docker run [docker-options] ecg-tokenizer [pipeline-options]"
    echo ""
    echo "Pipeline Options:"
    echo "  --mode MODE           Pipeline mode: preprocessing, analysis, full_run (default: full_run)"
    echo "  --input FILE          Input parquet file path"
    echo "  --output FILE         Output JSON file path"
    echo "  --device DEVICE       Device to use: cuda:0, cpu, etc."
    echo "  --batch-size N        Batch size for processing"
    echo "  --no-psa              Skip PSA normalization"
    echo "  --with-llm-judge      Run LLM-as-a-Judge evaluation"
    echo "  --help, -h            Show this help message"
    echo ""
    echo "Volume Mounts:"
    echo "  /app/inputs           Input parquet files"
    echo "  /app/outputs          Output results (JSON, CSV)"
    echo "  /app/ecg_signals      Raw ECG signal files (read-only)"
    echo "  /app/checkpoints      Model checkpoints (read-only)"
    echo "  /app/config           Configuration files (read-only)"
    echo ""
    echo "Examples:"
    echo "  # Run with GPU"
    echo "  docker run --gpus all \\"
    echo "    -v ./inputs:/app/inputs \\"
    echo "    -v ./outputs:/app/outputs \\"
    echo "    -v ./ecg_signals:/app/ecg_signals:ro \\"
    echo "    -v ./checkpoints:/app/checkpoints:ro \\"
    echo "    ecg-tokenizer --mode full_run"
    echo ""
    echo "  # Run with CPU"
    echo "  docker run \\"
    echo "    -v ./inputs:/app/inputs \\"
    echo "    -v ./outputs:/app/outputs \\"
    echo "    ecg-tokenizer --device cpu --batch-size 8"
    exit 0
fi

# Check for required directories
echo "Checking directories..."
for dir in inputs outputs; do
    if [ ! -d "${APP_HOME}/${dir}" ]; then
        mkdir -p "${APP_HOME}/${dir}"
        echo "  Created ${dir}/"
    else
        echo "  Found ${dir}/"
    fi
done

# Check for optional directories
for dir in ecg_signals checkpoints config; do
    if [ -d "${APP_HOME}/${dir}" ]; then
        echo "  Found ${dir}/"
    else
        echo "  Note: ${dir}/ not mounted"
    fi
done

echo ""

# Run the pipeline with all passed arguments
exec python "${APP_HOME}/inference/main.py" "$@"
