# ECG Tokenizer Inference Pipeline Docker Image
# Following DeepECG_Docker pattern for clean, production-ready deployment

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

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
    python3 \
    python3-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Go yq (mikefarah) to avoid Python yq CLI mismatch
RUN wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq \
    && chmod +x /usr/bin/yq

# Create directory structure (following DeepECG_Docker pattern)
RUN mkdir -p ${APP_HOME}/inputs \
    ${APP_HOME}/outputs \
    ${APP_HOME}/ecg_signals \
    ${APP_HOME}/checkpoints \
    ${APP_HOME}/config \
    ${APP_HOME}/preprocessing

# Copy requirements and install Python dependencies
COPY docker/requirements.txt ${APP_HOME}/requirements.txt
COPY pyproject.toml ${APP_HOME}/pyproject.toml
COPY uv.lock ${APP_HOME}/uv.lock
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r ${APP_HOME}/requirements.txt

ENV UV_PYTHON=3.12
RUN python3 -m pip install --upgrade pip
RUN pip install uv
RUN uv sync --python $UV_PYTHON
ENV PATH="/app/.venv/bin:$PATH"
# RUN uv pip install -r requirements.txt
# Download NLTK data for text metrics
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('wordnet', quiet=True)"

# Copy application code
COPY data/ ${APP_HOME}/data/
COPY models/ ${APP_HOME}/models/
COPY utils/ ${APP_HOME}/utils/
COPY inference/ ${APP_HOME}/inference/
COPY dataset_generation/ ${APP_HOME}/dataset_generation/
COPY config/ ${APP_HOME}/config/
COPY scripts/ ${APP_HOME}/scripts/
COPY projects/ ${APP_HOME}/projects/
COPY runners/ ${APP_HOME}/runners/
COPY dataset_generation/ ${APP_HOME}/dataset_generation/

# Copy API keys from repository root
COPY api_keys.json ${APP_HOME}/api_keys.json

# Prefetch model weights so runtime stays offline
ARG HF_TOKEN=""
ENV HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
RUN python - <<'PY'
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

api_path = Path("/app/api_keys.json")
token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not token and api_path.exists():
    try:
        token = json.load(api_path.open()).get("HUGGING_FACE_TOKEN")
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Unable to read HUGGING_FACE_TOKEN: {exc}")

if not token:
    raise SystemExit("Hugging Face token not provided. Pass --build-arg HF_TOKEN=... or put HUGGING_FACE_TOKEN in api_keys.json")


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
prefetch("heartwise/DeepECG-Tok_tokenizer", "/app/checkpoints/DeepECG-Tok_tokenizer")
prefetch("heartwise/DeepECG-Tok_EfficientNetV2_77_Classes", "/app/checkpoints/DeepECG-Tok_EfficientNetV2_77_Classes")
prefetch("heartwise/DeepECG-Tok_medgemma-4b-it", "/app/checkpoints/DeepECG-Tok_medgemma-4b-it")
# Also cache the base Gemma tokenizer/model so AutoTokenizer can load offline
snapshot_download(
    repo_id="google/medgemma-4b-it",
    token=token,
)
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
