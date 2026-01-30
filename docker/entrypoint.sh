#!/bin/bash
#!/bin/bash
# ECG Tokenizer Pipeline Entrypoint (orchestration-first)
set -e

APP_HOME="${APP_HOME:-/app}"

# Prefer offline HF cache when baked weights are present
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export BASE_MODEL_DIR="${APP_HOME}/checkpoints/google-medgemma-4b-it"

# Help
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  source "${APP_HOME}/run_pipeline.bash" --help
  exit 0
fi

# Run full orchestration (preprocess/BERT/EfficientNet based on --step)
source "${APP_HOME}/run_pipeline.bash" "$@"
