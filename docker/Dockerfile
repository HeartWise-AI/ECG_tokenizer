# ECG Tokenizer Inference Pipeline Docker Image
# Following DeepECG_Docker pattern for clean, production-ready deployment

FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV APP_HOME=/app
ENV HF_HOME=${APP_HOME}/.cache/huggingface

WORKDIR ${APP_HOME}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create directory structure (following DeepECG_Docker pattern)
RUN mkdir -p ${APP_HOME}/inputs \
    ${APP_HOME}/outputs \
    ${APP_HOME}/ecg_signals \
    ${APP_HOME}/checkpoints \
    ${APP_HOME}/config \
    ${APP_HOME}/preprocessing

# Copy requirements and install Python dependencies
COPY docker/requirements.txt ${APP_HOME}/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r ${APP_HOME}/requirements.txt

# Download NLTK data for text metrics
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('wordnet', quiet=True)"

# Copy application code
COPY data/ ${APP_HOME}/data/
COPY models/ ${APP_HOME}/models/
COPY utils/ ${APP_HOME}/utils/
COPY inference/ ${APP_HOME}/inference/
COPY dataset_generation/ ${APP_HOME}/dataset_generation/
COPY config/ ${APP_HOME}/config/

# Copy API keys from repository root
COPY api_keys.json ${APP_HOME}/api_keys.json

# Prefetch model weights so runtime stays offline
RUN python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import snapshot_download

api_path = Path("/app/api_keys.json")
token = None
if api_path.exists():
    try:
        token = json.load(api_path.open()).get("HUGGING_FACE_TOKEN")
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Unable to read HUGGING_FACE_TOKEN: {exc}")


def prefetch(repo: str, dest: str) -> None:
    Path(dest).mkdir(parents=True, exist_ok=True)
    print(f"Prefetching {repo} -> {dest}")
    snapshot_download(
        repo_id=repo,
        repo_type="model",
        local_dir=dest,
        local_dir_use_symlinks=False,
        token=token,
    )


prefetch("heartwise/Bert_diagnosis2classification_En_Fr", "/app/checkpoints/bert")
prefetch("heartwise/DeepECG-Tok_tokenizer", "/app/checkpoints/ECG_tokenizer_latest")
prefetch("heartwise/DeepECG-Tok_EfficientNetV2_77_Classes", "/app/checkpoints/efficientnet")
PY

# Copy Docker-specific files
COPY docker/entrypoint.sh ${APP_HOME}/entrypoint.sh
COPY docker/heartwise.config ${APP_HOME}/heartwise.config
COPY docker/run_pipeline.bash ${APP_HOME}/run_pipeline.bash

# Make scripts executable
RUN chmod +x ${APP_HOME}/entrypoint.sh ${APP_HOME}/run_pipeline.bash

# Set Python path
ENV PYTHONPATH="${APP_HOME}"

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (can be overridden)
CMD ["--help"]
