#!/bin/bash
# ECG Tokenizer Pipeline Runner
# Supports both Docker and local execution with smart path detection

# =============================================================================
# ENVIRONMENT DETECTION
# =============================================================================

# Determine script directory and repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect environment: Docker vs Local
if [[ -f "/app/inference/preprocessing.py" ]]; then
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
    echo "  --qa-disable-categories LIST"
    echo "                           Comma-separated QA prompt categories to disable"
    echo "  --qa-max-normal-percentage FLOAT"
    echo "                           Override QA normal-ECG downsampling cap"
    echo "  --step [preprocess|bert|analysis|efficientnet|qa|llm|all|preprocess_qa_llm]  Choose which step to run (default: all)"
    echo "  --help, -h               Show this help message"
    echo ""
    echo "Examples:"
    echo "  source run_pipeline.bash --input_file /path/to/data.parquet"
    echo "  source run_pipeline.bash --step efficientnet --input_file /path/to/preprocessed.parquet"
    echo "  source run_pipeline.bash --step preprocess_qa_llm --input_file /path/to/data.parquet"
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
qa_disable_categories=$(get_param "qa_disable_categories")
qa_max_normal_percentage=$(get_param "qa_max_normal_percentage")
bert_output=""
run_step="all"
use_preprocessing=true
use_bert_classification=true
use_efficientnet_classification=true
efficientnet_config="${APP_ROOT}/checkpoints/DeepECG-Tok_EfficientNetV2_77_Classes/base_config.yaml"
llm_checkpoint="${APP_ROOT}/checkpoints/DeepECG-Tok_medgemma-4b-it/deepecg_tokenizer_medgemma.pt"
qa_output=""

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
        --output-dir|--output_dir)
            if [[ -n $2 && ! $2 =~ ^-- ]]; then
                output_dir="$2"
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
        --qa-disable-categories)
            if [[ -n $2 && ! $2 =~ ^-- ]]; then
                qa_disable_categories="$2"
                shift 2
            else
                echo "Error: --qa-disable-categories requires a non-empty argument."
                usage
                return 1
            fi
            ;;
        --qa-max-normal-percentage)
            if [[ -n $2 && ! $2 =~ ^-- ]]; then
                qa_max_normal_percentage="$2"
                shift 2
            else
                echo "Error: --qa-max-normal-percentage requires a non-empty argument."
                usage
                return 1
            fi
            ;;
        --step)
            if [[ -n $2 && ! $2 =~ ^-- ]]; then
                run_step="$2"
                shift 2
            else
                echo "Error: --step requires one of preprocess|bert|analysis|efficientnet|qa|llm|all"
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
efficientnet_config=$(convert_path "$efficientnet_config")
qa_output=$(convert_path "$qa_output")
llm_checkpoint=$(convert_path "$llm_checkpoint")

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
qa_output=${output_dir}/preprocessed_qa.parquet

# =============================================================================
# BUILD AND RUN PIPELINE
# =============================================================================

build_args() {
    local step_override="${1:-$run_step}"
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
    args="$args --step $step_override"
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

run_qa_generation() {
    local qa_input="$1"
    local qa_cmd=(
        python "${APP_ROOT}/dataset_generation/generate_train_test_datasets.py"
        --dataset custom
        --custom_parquet_path "$qa_input"
        --max_prompts_per_ecg 4
        --prompt_workers 1
        --answer_workers 1
        --output_dir "$output_dir"
    )

    if [[ -n "$qa_max_normal_percentage" ]]; then
        qa_cmd+=(--max_normal_percentage "$qa_max_normal_percentage")
    fi
    if [[ -n "$qa_disable_categories" ]]; then
        qa_cmd+=(--disable_categories "$qa_disable_categories")
    fi

    local qa_cmd_str
    printf -v qa_cmd_str '%q ' "${qa_cmd[@]}"
    echo "[RUN] ${qa_cmd_str}"
    "${qa_cmd[@]}"
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
    echo "  EfficientNet Config: $efficientnet_config"
    echo "  LLM Checkpoint: $llm_checkpoint"
    echo "------------------------------------------------------------"
    echo "Required Input Columns:"
    echo "  - ecg_path: ECG signal path"
    echo "  - reports: Text reports (placeholders are acceptable for QA-only SHD/LVEF runs)"
    echo "============================================================"
    echo ""
    
    local args=$(build_args)
    local python_script="${APP_ROOT}/inference/preprocessing.py"
    
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
            if [[ ! -f "$efficientnet_config" ]]; then
                echo "Error: EfficientNet config not found at $efficientnet_config (expected from downloaded HF model)"
                return 1
            fi
            echo "[RUN] bash ${APP_ROOT}/scripts/runner.sh --base_config ${efficientnet_config} --selected_gpus 0 --use_wandb false --run_mode inference"
            bash "${APP_ROOT}/scripts/runner.sh" --base_config "${efficientnet_config}" --selected_gpus 0 --use_wandb false --run_mode inference
            qa_disable_arg=()
            [[ -n "$qa_disable_categories" ]] && qa_disable_arg=(--disable_categories "$qa_disable_categories")
            echo "[RUN] python ${APP_ROOT}/dataset_generation/generate_train_test_datasets.py --dataset custom --custom_parquet_path ${bert_output} --max_prompts_per_ecg 4 --prompt_workers 1 --answer_workers 1 --output_dir ${output_dir} ${qa_disable_arg[*]}"
            python "${APP_ROOT}/dataset_generation/generate_train_test_datasets.py" --dataset custom --custom_parquet_path "${bert_output}" --max_prompts_per_ecg 4 --prompt_workers 1 --answer_workers 1 --output_dir "${output_dir}" "${qa_disable_arg[@]}"
            if [[ ! -f "$qa_output" ]]; then
                echo "Error: QA output not found at $qa_output after analysis"
                return 1
            fi
            echo "[RUN] python ${APP_ROOT}/inference/generate_all_qa_pairs.py --checkpoint ${llm_checkpoint} --validation_parquet ${qa_output} --output_dir ${output_dir} --answer_column generated_answer --output_prefix llm_inference_samples"
            python "${APP_ROOT}/inference/generate_all_qa_pairs.py" --checkpoint "${llm_checkpoint}" --validation_parquet "${qa_output}" --output_dir "${output_dir}" --answer_column generated_answer --output_prefix llm_inference_samples
            ;;
        efficientnet)
            # Ensure BERT output exists before running EfficientNet
            if [[ ! -f "$bert_output" ]]; then
                echo "Error: Run BERT before this step! Missing $bert_output"
                return 1
            fi
            if [[ ! -f "$efficientnet_config" ]]; then
                echo "Error: EfficientNet config not found at $efficientnet_config (expected from downloaded HF model)"
                return 1
            fi
            echo "[RUN] bash ${APP_ROOT}/scripts/runner.sh --base_config ${efficientnet_config} --selected_gpus 0 --use_wandb false --run_mode inference"
            bash "${APP_ROOT}/scripts/runner.sh" --base_config "${efficientnet_config}" --selected_gpus 0 --use_wandb false --run_mode inference
            ;;
        all)
            echo "[RUN] python $python_script $args"
            python "$python_script" $args || return 1
            if [[ ! -f "$bert_output" ]]; then
                echo "Error: BERT output not found at $bert_output after run_step=all"
                return 1
            fi
            if [[ ! -f "$efficientnet_config" ]]; then
                echo "Error: EfficientNet config not found at $efficientnet_config (expected from downloaded HF model)"
                return 1
            fi
            echo "[RUN] bash ${APP_ROOT}/scripts/runner.sh --base_config ${efficientnet_config} --selected_gpus 0 --use_wandb false --run_mode inference"
            bash "${APP_ROOT}/scripts/runner.sh" --base_config "${efficientnet_config}" --selected_gpus 0 --use_wandb false --run_mode inference
            qa_disable_arg=()
            [[ -n "$qa_disable_categories" ]] && qa_disable_arg=(--disable_categories "$qa_disable_categories")
            echo "[RUN] python ${APP_ROOT}/dataset_generation/generate_train_test_datasets.py --dataset custom --custom_parquet_path ${bert_output} --max_prompts_per_ecg 4 --prompt_workers 1 --answer_workers 1 --output_dir ${output_dir} ${qa_disable_arg[*]}"
            python "${APP_ROOT}/dataset_generation/generate_train_test_datasets.py" --dataset custom --custom_parquet_path "${bert_output}" --max_prompts_per_ecg 4 --prompt_workers 1 --answer_workers 1 --output_dir "${output_dir}" "${qa_disable_arg[@]}"
            if [[ ! -f "$qa_output" ]]; then
                echo "Error: QA output not found at $qa_output after all step"
                return 1
            fi
            echo "[RUN] python ${APP_ROOT}/inference/generate_all_qa_pairs.py --checkpoint ${llm_checkpoint} --validation_parquet ${qa_output} --output_dir ${output_dir} --answer_column generated_answer --output_prefix llm_inference_samples"
            python "${APP_ROOT}/inference/generate_all_qa_pairs.py" --checkpoint "${llm_checkpoint}" --validation_parquet "${qa_output}" --output_dir "${output_dir}" --answer_column generated_answer --output_prefix llm_inference_samples
            ;;
        qa)
            mkdir -p "$output_dir"
            qa_input=""
            if [[ -n "$input_file" ]]; then
                if [[ ! -f "$input_parquet" ]]; then
                    echo "Error: Explicit QA input parquet not found: $input_parquet"
                    return 1
                fi
                if [[ "$input_parquet" != *.parquet ]]; then
                    echo "Error: QA step expects a parquet file when using --input_file directly: $input_parquet"
                    return 1
                fi
                echo "[INFO] Using explicit QA input parquet: $input_parquet"
                qa_input="$input_parquet"
            elif [[ -f "$bert_output" ]]; then
                echo "[INFO] Using BERT output parquet for QA step: $bert_output"
                qa_input="$bert_output"
            else
                echo "Error: QA step requires either an explicit --input_file parquet or BERT output at $bert_output"
                return 1
            fi
            run_qa_generation "$qa_input"
            ;;
        preprocess_qa_llm)
            mkdir -p "$output_dir"
            preprocess_args=$(build_args preprocess)
            echo "[RUN] python $python_script $preprocess_args"
            python "$python_script" $preprocess_args || return 1

            preprocessed_output="${output_dir}/preprocessed.parquet"
            if [[ ! -f "$preprocessed_output" ]]; then
                echo "Error: Preprocessing output not found at $preprocessed_output"
                return 1
            fi

            run_qa_generation "$preprocessed_output"

            if [[ ! -f "$qa_output" ]]; then
                echo "Error: QA output not found at $qa_output after preprocess_qa_llm"
                return 1
            fi

            echo "[RUN] python ${APP_ROOT}/inference/generate_all_qa_pairs.py --checkpoint ${llm_checkpoint} --validation_parquet ${qa_output} --output_dir ${output_dir} --answer_column generated_answer --output_prefix llm_inference_samples"
            python "${APP_ROOT}/inference/generate_all_qa_pairs.py" --checkpoint "${llm_checkpoint}" --validation_parquet "${qa_output}" --output_dir "${output_dir}" --answer_column generated_answer --output_prefix llm_inference_samples
            ;;
        llm)
            if [[ ! -f "$qa_output" ]]; then
                echo "Error: QA output not found at $qa_output for LLM step. Run --step qa first."
                return 1
            fi
            echo "[RUN] python ${APP_ROOT}/inference/generate_all_qa_pairs.py --checkpoint ${llm_checkpoint} --validation_parquet ${qa_output} --output_dir ${output_dir} --answer_column generated_answer --output_prefix llm_inference_samples"
            python "${APP_ROOT}/inference/generate_all_qa_pairs.py" --checkpoint "${llm_checkpoint}" --validation_parquet "${qa_output}" --output_dir "${output_dir}" --device 1 --save_interval 10 --answer_column generated_answer --output_prefix llm_inference_samples
            ;;
        *)
            echo "Unknown run_step: $run_step"
            return 1
            ;;
    esac

}

# Run
run_pipeline
