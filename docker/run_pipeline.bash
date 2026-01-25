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
    echo "  --no-preprocessing       Skip preprocessing (load cached parquet)"
    echo "  --no-bert                Skip BERT classification (load cached parquet)"
    echo "  --no-efficientnet        Skip EfficientNet classification"
    echo "  --help, -h               Show this help message"
    echo ""
    echo "Examples:"
    echo "  source run_pipeline.bash --input_file /path/to/data.parquet"
    echo "  source run_pipeline.bash --no-preprocessing --input_file /path/to/preprocessed.parquet"
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
output_json=$(get_param "output_json")
output_dir=$(get_param "output_dir")
ecg_signals_path=$(get_param "ecg_signals_path")
preprocessing_folder=$(get_param "preprocessing_folder")
preprocessing_n_workers=$(get_param "preprocessing_n_workers")
apply_psa_normalization=$(get_param "apply_psa_normalization")
dataset_name=$(get_param "dataset_name")
bert_checkpoint=$(get_param "bert_checkpoint")
tokenizer_checkpoint=$(get_param "tokenizer_checkpoint")
efficientnet_checkpoint=$(get_param "efficientnet_checkpoint")
gpt2_checkpoint=$(get_param "gpt2_checkpoint")
enable_report_generation=$(get_param "enable_report_generation")
bert_base_config=$(get_param "bert_base_config")
efficientnet_base_config=$(get_param "efficientnet_base_config")
efficientnet_selected_gpus=$(get_param "efficientnet_selected_gpus")
efficientnet_use_wandb=$(get_param "efficientnet_use_wandb")
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
        --no-preprocessing)
            use_preprocessing=false
            shift 1
            ;;
        --no-bert)
            use_bert_classification=false
            shift 1
            ;;
        --no-efficientnet)
            use_efficientnet_classification=false
            shift 1
            ;;
        --efficientnet-base-config)
            if [[ -n $2 && ! $2 =~ ^-- ]]; then
                efficientnet_base_config="$2"
                shift 2
            fi
            ;;
        --efficientnet-selected-gpus)
            if [[ -n $2 && ! $2 =~ ^-- ]]; then
                efficientnet_selected_gpus="$2"
                shift 2
            fi
            ;;
        --efficientnet-use-wandb)
            if [[ -n $2 && ! $2 =~ ^-- ]]; then
                efficientnet_use_wandb="$2"
                shift 2
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
output_json=$(convert_path "$output_json")
output_dir=$(convert_path "$output_dir")
ecg_signals_path=$(convert_path "$ecg_signals_path")
preprocessing_folder=$(convert_path "$preprocessing_folder")
bert_checkpoint=$(convert_path "$bert_checkpoint")
tokenizer_checkpoint=$(convert_path "$tokenizer_checkpoint")
efficientnet_checkpoint=$(convert_path "$efficientnet_checkpoint")
gpt2_checkpoint=$(convert_path "$gpt2_checkpoint")

# Set defaults if not in config
device=${device:-cuda:0}
batch_size=${batch_size:-32}
num_workers=${num_workers:-8}
ecg_signals_path=${ecg_signals_path:-${APP_ROOT}/ecg_signals}
preprocessing_folder=${preprocessing_folder:-${APP_ROOT}/preprocessing}
preprocessing_n_workers=${preprocessing_n_workers:-16}
apply_psa_normalization=${apply_psa_normalization:-true}

# =============================================================================
# BUILD AND RUN PIPELINE
# =============================================================================

build_args() {
    local args=""
    
    args="$args --input $input_parquet"
    
    if [[ -n $output_json ]]; then
        args="$args --output $output_json"
    fi
    
    if [[ -n $output_dir ]]; then
        args="$args --output-dir $output_dir"
    fi
    
    args="$args --device $device"
    args="$args --batch-size $batch_size"
    args="$args --num-workers $num_workers"
    args="$args --ecg-signals-path $ecg_signals_path"
    args="$args --preprocessing-folder $preprocessing_folder"
    args="$args --preprocessing-n-workers $preprocessing_n_workers"
    
    if [[ "$use_preprocessing" == "false" ]]; then
        args="$args --no-preprocessing"
    fi
    if [[ "$use_bert_classification" == "false" ]]; then
        args="$args --no-bert-classification"
    fi
    if [[ "$use_efficientnet_classification" == "false" ]]; then
        args="$args --no-efficientnet-classification"
    fi
    
    if [[ -n $dataset_name ]]; then
        args="$args --dataset-name $dataset_name"
    fi
    
    if [[ -n $bert_checkpoint ]]; then
        args="$args --bert-checkpoint $bert_checkpoint"
    fi
    
    if [[ -n $tokenizer_checkpoint ]]; then
        args="$args --tokenizer-checkpoint $tokenizer_checkpoint"
    fi
    
    if [[ -n $efficientnet_checkpoint ]]; then
        args="$args --efficientnet-checkpoint $efficientnet_checkpoint"
    fi
    
    if [[ -n $gpt2_checkpoint ]]; then
        args="$args --gpt2-checkpoint $gpt2_checkpoint"
    fi

    if [[ -n $bert_base_config ]]; then
        args="$args --bert-base-config $bert_base_config"
    fi
    
if [[ "$apply_psa_normalization" == "false" ]]; then
    args="$args --no-psa"
fi
    
    if [[ "$enable_report_generation" == "true" ]]; then
        args="$args --enable-report-generation"
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
    echo "  Output: $output_json"
    echo "  Output Dir: $output_dir"
    echo "  ECG Signals Path: $ecg_signals_path"
    echo "  Preprocessing Folder: $preprocessing_folder"
    echo "  PSA Normalization: $apply_psa_normalization"
    echo "  Use preprocessing: $use_preprocessing"
    echo "  Use BERT: $use_bert_classification"
    echo "  Use EfficientNet: $use_efficientnet_classification"
    echo "------------------------------------------------------------"
    echo "Required Input Columns:"
    echo "  - ecg_path: ECG signal path"
    echo "  - reports: Text reports"
    echo "============================================================"
    echo ""
    
    local args=$(build_args)
    local python_script="${APP_ROOT}/inference/main.py"
    
    echo "[RUN] python $python_script $args"
    echo ""
    
    python "$python_script" $args

    # If EfficientNet is enabled, run it via runner.sh using configured base config
    if [[ "$use_efficientnet_classification" == "true" ]]; then
        eff_base_cfg=${efficientnet_base_config:-config/linear_probing/base_config.yaml}
        eff_gpus=${efficientnet_selected_gpus:-0}
        eff_wandb=${efficientnet_use_wandb:-false}
        echo "[RUN] bash ${APP_ROOT}/scripts/runner.sh --base_config $eff_base_cfg --selected_gpus $eff_gpus --use_wandb $eff_wandb --run_mode inference"
        bash "${APP_ROOT}/scripts/runner.sh" --base_config "$eff_base_cfg" --selected_gpus "$eff_gpus" --use_wandb "$eff_wandb" --run_mode inference
    fi
}

# Run
run_pipeline
