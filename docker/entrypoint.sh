#!/bin/bash
# ECG Tokenizer Pipeline Entrypoint
# Following DeepECG_Docker pattern

set -e

APP_HOME="${APP_HOME:-/app}"

# Display banner
echo "============================================================"
echo "ECG Tokenizer Inference Pipeline"
echo "============================================================"
echo "(saves preprocessed signals as .npy; BERT thresholds from config)"
echo ""

# Check if help is requested
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: docker run [docker-opts] tokenizer_inference [pipeline-opts]"
    echo ""
    echo "Pipeline options (python inference/main.py):"
    echo "  --run-step {preprocess|bert|all}  Which stages to run (default: all)"
    echo "  --input PATH              Input CSV/Parquet with columns: ecg_path, reports"
    echo "  --output-dir PATH         Output directory for parquet/json/metrics (default: /app/outputs)"
    echo "  --ecg-signals-path PATH   Base dir for ECG files (or leave if ecg_path is absolute)"
    echo "  --device DEVICE           cuda:0 | cpu (or add --cpu)"
    echo "  --batch-size N            Batch size (default: 32)"
    echo "  --num-workers N           Dataloader workers (default: 8)"
    echo "  --preprocessing-folder DIR  Where .npy signals are saved (default: /app/preprocessing)"
    echo "  --dataset-name NAME       Optional prefix for preprocessing outputs"
    echo "  --bert-checkpoint PATH    BERT classifier checkpoint (read-only mount)"
    echo "  --tokenizer-checkpoint PATH  ECG tokenizer checkpoint (read-only mount)"
    echo "  --no-psa                  Disable PSA normalization during preprocessing"
    echo "  --help, -h                Show this help"
    echo ""
    echo "Volume mounts (recommended):"
    echo "  -v ./inputs:/app/inputs                  # input CSV/parquet"
    echo "  -v ./outputs:/app/outputs                # outputs (.parquet, metrics, logs)"
    echo "  -v ./preprocessing:/app/preprocessing    # cached .npy signals"
    echo "  -v ./checkpoints:/app/checkpoints:ro     # tokenizer / BERT weights"
    echo "  -v /media/data1/datasets/Harvard-Emory-ECG:/media/data1/datasets/Harvard-Emory-ECG:ro  # raw ECG paths (match absolute paths in CSVs)"
    echo ""
    echo "Examples:"
    cat <<'EXAMPLES'
  # Full run (preprocess + BERT) on GPU
  docker run --gpus all \
    -v $(pwd)/inputs:/app/inputs \
    -v $(pwd)/outputs:/app/outputs \
    -v $(pwd)/preprocessing:/app/preprocessing \
    -v /media/data1/datasets/Harvard-Emory-ECG:/media/data1/datasets/Harvard-Emory-ECG:ro \
    -v $(pwd)/checkpoints:/app/checkpoints:ro \
    tokenizer_inference \
    --run-step all \
    --input /app/inputs/harvard_emory_subset_1k.csv \
    --ecg-signals-path /media/data1/datasets/Harvard-Emory-ECG

  # Preprocess only, CPU
  docker run \
    -v $(pwd)/inputs:/app/inputs \
    -v $(pwd)/outputs:/app/outputs \
    -v $(pwd)/preprocessing:/app/preprocessing \
    -v /media/data1/datasets/Harvard-Emory-ECG:/media/data1/datasets/Harvard-Emory-ECG:ro \
    tokenizer_inference \
    --run-step preprocess \
    --input /app/inputs/harvard_emory_subset_1k.csv \
    --device cpu
EXAMPLES
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
for dir in ecg_signals checkpoints config preprocessing; do
    if [ -d "${APP_HOME}/${dir}" ]; then
        echo "  Found ${dir}/"
    else
        echo "  Note: ${dir}/ not mounted"
    fi
done

echo ""

# Run the pipeline with all passed arguments
exec python "${APP_HOME}/inference/main.py" "$@"
