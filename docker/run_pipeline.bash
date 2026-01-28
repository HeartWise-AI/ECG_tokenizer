#!/bin/bash
# ECG Tokenizer Pipeline Runner
# Supports both Docker and local execution with smart path detection

# =============================================================================
# ENVIRONMENT DETECTION
# =============================================================================

# Determine script directory and repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect environment: Docker vs Local
if [[ -f "/app/inference/main.py" ]]; then
    # Running inside Docker container
    APP_ROOT="/app"
    REPO_ROOT="/app"
    IN_DOCKER=true
    echo "[ENV] Running inside Docker container"
else
    # Running locally - repo root is parent of docker/ folder
    REPO_ROOT="$(dirname "$SCRIPT_DIR")"
    APP_ROOT="$REPO_ROOT"
    IN_DOCKER=false
    echo "[ENV] Running locally at: $REPO_ROOT"
    
    # Set PYTHONPATH for local execution
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
fi

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

usage() {
    echo "Usage: source run_pipeline.bash [options]"
    echo ""
    echo "Options:"
    echo "  --input_file FILE        Input file path (CSV or Parquet)"
    echo "  --device DEVICE          Compute device (default: cuda:0)"
    echo "  --batch-size N           Batch size (default: 32)"
    echo "  --dataset-name NAME      Optional dataset name for preprocessing outputs"
    echo "  --bert-base-config FILE  BERT base config yaml"
    echo "  --step [preprocess|bert|analysis|efficientnet|all]  Choose which step to run (default: all)"
    echo "  --help, -h               Show this help message"
    echo ""
    echo "Examples:"
    echo "  source run_pipeline.bash --input_file /path/to/data.parquet"
    echo "  source run_pipeline.bash --step efficientnet --input_file /path/to/preprocessed.parquet"
    return 0
}

# Function to get value from config file
get_param() {
    local config_file="${SCRIPT_DIR}/heartwise.config"
    if [[ -f "$config_file" ]]; then
        grep "^$1:" "$config_file" | cut -d':' -f2- | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
    fi
}

# Function to convert /app/ paths to local paths when running outside Docker
convert_path() {
    local path="$1"
    if [[ "$IN_DOCKER" == "false" && "$path" == /app/* ]]; then
        # Replace /app/ with repo root
        echo "${path/\/app/$REPO_ROOT}"
    else
        echo "$path"
    fi
}

# =============================================================================
# LOAD CONFIGURATION
# =============================================================================

# Read parameters from config file
device=$(get_param "device")
batch_size=$(get_param "batch_size")
num_workers=$(get_param "num_workers")
input_parquet=$(get_param "input_parquet")
output_dir=$(get_param "output_dir")
ecg_signals_path=$(get_param "ecg_signals_path")
preprocessing_folder=$(get_param "preprocessing_folder")
preprocessing_n_workers=$(get_param "preprocessing_n_workers")
apply_psa_normalization=$(get_param "apply_psa_normalization")
dataset_name=$(get_param "dataset_name")
bert_checkpoint=$(get_param "bert_checkpoint")
tokenizer_checkpoint=$(get_param "tokenizer_checkpoint")
bert_base_config=$(get_param "bert_base_config")
bert_output=""
run_step="all"
use_preprocessing=true
use_bert_classification=true
use_efficientnet_classification=true

# =============================================================================
# PARSE COMMAND LINE ARGUMENTS
# =============================================================================

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input_file)
            if [[ -n $2 && ! $2 =~ ^-- ]]; then
                input_file="$2"
                shift 2
            else
                echo "Error: --input_file requires a non-empty argument."
                usage
                return 1
            fi
            ;;
        --device)
            if [[ -n $2 && ! $2 =~ ^-- ]]; then
                device="$2"
                shift 2
            fi
            ;;
        --batch-size)
            if [[ -n $2 && ! $2 =~ ^-- ]]; then
                batch_size="$2"
                shift 2
            fi
            ;;
        --dataset-name)
            if [[ -n $2 && ! $2 =~ ^-- ]]; then
                dataset_name="$2"
                shift 2
            fi
            ;;
        --bert-base-config)
            if [[ -n $2 && ! $2 =~ ^-- ]]; then
                bert_base_config="$2"
                shift 2
            fi
            ;;
        --step)
            if [[ -n $2 && ! $2 =~ ^-- ]]; then
                run_step="$2"
                shift 2
            else
                echo "Error: --step requires one of preprocess|bert|analysis|efficientnet|all"
                usage
                return 1
            fi
            ;;
        --help|-h)
            usage
            return 0
            ;;
        *)
            echo "Error: Unknown parameter passed: $1"
            usage
            return 1
            ;;
    esac
done

# =============================================================================
# RESOLVE PATHS
# =============================================================================

# If input_file is provided as absolute path, use it directly
# Otherwise, treat it as relative to inputs folder
if [[ -n $input_file ]]; then
    if [[ "$input_file" == /* ]]; then
        # Absolute path provided
        input_parquet="$input_file"
    else
        # Relative path - prepend inputs folder
        input_parquet="${APP_ROOT}/inputs/${input_file}"
    fi
fi

# Convert all /app/ paths to local paths when running outside Docker
input_parquet=$(convert_path "$input_parquet")
output_dir=$(convert_path "$output_dir")
ecg_signals_path=$(convert_path "$ecg_signals_path")
preprocessing_folder=$(convert_path "$preprocessing_folder")
bert_checkpoint=$(convert_path "$bert_checkpoint")
tokenizer_checkpoint=$(convert_path "$tokenizer_checkpoint")

# Set defaults if not in config
device=${device:-cuda:0}
batch_size=${batch_size:-32}
num_workers=${num_workers:-8}
ecg_signals_path=${ecg_signals_path:-${APP_ROOT}/ecg_signals}
preprocessing_folder=${preprocessing_folder:-${APP_ROOT}/preprocessing}
preprocessing_n_workers=${preprocessing_n_workers:-16}
apply_psa_normalization=${apply_psa_normalization:-true}

# Derive BERT output path (fixed name for consistency across steps)
bert_output=${output_dir}/preprocessed_bert_output.parquet

# =============================================================================
# BUILD AND RUN PIPELINE
# =============================================================================

build_args() {
    local args=""
    
    args="$args --input $input_parquet"
    
    if [[ -n $output_dir ]]; then
        args="$args --output-dir $output_dir"
    fi
    
    args="$args --device $device"
    args="$args --batch-size $batch_size"
    args="$args --num-workers $num_workers"
    args="$args --ecg-signals-path $ecg_signals_path"
    args="$args --preprocessing-folder $preprocessing_folder"
    args="$args --preprocessing-n-workers $preprocessing_n_workers"
    args="$args --step $run_step"
    args="$args --bert-output $bert_output"
    
    if [[ -n $dataset_name ]]; then
        args="$args --dataset-name $dataset_name"
    fi
    
    if [[ -n $bert_checkpoint ]]; then
        args="$args --bert-checkpoint $bert_checkpoint"
    fi
    
    if [[ -n $tokenizer_checkpoint ]]; then
        args="$args --tokenizer-checkpoint $tokenizer_checkpoint"
    fi
    
    if [[ -n $bert_base_config ]]; then
        args="$args --bert-base-config $bert_base_config"
    fi
    
if [[ "$apply_psa_normalization" == "false" ]]; then
    args="$args --no-psa"
fi
    
    echo "$args"
}

run_pipeline() {    
    echo ""
    echo "============================================================"
    echo "ECG Tokenizer Inference Pipeline"
    echo "============================================================"
    echo "Environment: $(if $IN_DOCKER; then echo 'Docker'; else echo 'Local'; fi)"
    echo "------------------------------------------------------------"
    echo "Configuration:"
    echo "  Device: $device"
    echo "  Batch Size: $batch_size"
    echo "  Input: $input_parquet"
    echo "  Output Dir: $output_dir"
    echo "  ECG Signals Path: $ecg_signals_path"
    echo "  Preprocessing Folder: $preprocessing_folder"
    echo "  PSA Normalization: $apply_psa_normalization"
    echo "  Run Step: $run_step"
    echo "------------------------------------------------------------"
    echo "Required Input Columns:"
    echo "  - ecg_path: ECG signal path"
    echo "  - reports: Text reports"
    echo "============================================================"
    echo ""
    
    local args=$(build_args)
    local python_script="${APP_ROOT}/inference/main.py"
    
    # Sequential control based on step
    case "$run_step" in
        preprocess)
            echo "[RUN] python $python_script $args"
            python "$python_script" $args
            ;;
        bert)
            echo "[RUN] python $python_script $args"
            python "$python_script" $args
            ;;
        analysis)
            # Run BERT only, then EfficientNet
            echo "[RUN] python $python_script $args"
            python "$python_script" $args || return 1
            if [[ ! -f "$bert_output" ]]; then
                echo "Error: BERT output not found at $bert_output after --step analysis"
                return 1
            fi
            echo "[RUN] bash ${APP_ROOT}/scripts/runner.sh --base_config ${APP_ROOT}/config/linear_probing/base_config.yaml --selected_gpus 0 --use_wandb false --run_mode inference"
            bash "${APP_ROOT}/scripts/runner.sh" --base_config "${APP_ROOT}/config/linear_probing/base_config.yaml" --selected_gpus 0 --use_wandb false --run_mode inference
            ;;
        efficientnet)
            # Ensure BERT output exists before running EfficientNet
            if [[ ! -f "$bert_output" ]]; then
                echo "Error: Run BERT before this step! Missing $bert_output"
                return 1
            fi
            echo "[RUN] bash ${APP_ROOT}/scripts/runner.sh --base_config ${APP_ROOT}/config/linear_probing/base_config.yaml --selected_gpus 0 --use_wandb false --run_mode inference"
            bash "${APP_ROOT}/scripts/runner.sh" --base_config "${APP_ROOT}/config/linear_probing/base_config.yaml" --selected_gpus 0 --use_wandb false --run_mode inference
            ;;
        all)
            echo "[RUN] python $python_script $args"
            python "$python_script" $args || return 1
            if [[ ! -f "$bert_output" ]]; then
                echo "Error: BERT output not found at $bert_output after run_step=all"
                return 1
            fi
            echo "[RUN] bash ${APP_ROOT}/scripts/runner.sh --base_config ${APP_ROOT}/config/linear_probing/base_config.yaml --selected_gpus 0 --use_wandb false --run_mode inference"
            bash "${APP_ROOT}/scripts/runner.sh" --base_config "${APP_ROOT}/config/linear_probing/base_config.yaml" --selected_gpus 0 --use_wandb false --run_mode inference
            ;;
        *)
            echo "Unknown run_step: $run_step"
            return 1
            ;;
    esac

}

# Run
run_pipeline
