# ECG Tokenizer Docker Pipeline

Containerized pipeline for preprocessing ECG signals to `.npy` and running the BERT text classifier to generate labels/metrics. EfficientNet is **not** run inside this image; use `scripts/runner.sh` separately if needed.

## Prerequisites

### 1. Create API Keys File

Before building the Docker image, you **must** create an `api_keys.json` file in the repository root:

Then edit `api_keys.json` and add your HuggingFace token:

```json
{
    "HUGGING_FACE_TOKEN": "hf_your_actual_token_here"
}
```

## What the image does
- Reads CSV/Parquet with columns `ecg_path` and `reports`.
- Preprocesses ECG signals to `.npy` files (no base64).
- Runs BERT classifier; thresholds come from `utils/constants.py::BERT_THRESHOLDS`.

## Build the image
```bash
docker build --no-cache -f docker/Dockerfile -t tokenizer_inference .
```

## Set up HuggingFace API key
- Create a HuggingFace account if you don't have one yet
- Ask for access to the DeepECG models needed in the [heartwise-ai/DeepECG](https://huggingface.co/collections/heartwise/deepecg-models-66ce09c7d620749ad819fa0d) repository
- Create an API key in the HuggingFace website in `User Settings` -> `API Keys` -> `Create API Key` -> `Read`
- Add your API key in the following format in the `api_keys.json` file in the `root` directory:
  ```json
  {
    "HUGGING_FACE_TOKEN": "your_api_key_here"
  }
  ```

## Volume mounts (recommended)
- `-v ./inputs:/app/inputs`            – input CSV/Parquet
- `-v ./outputs:/app/outputs`          – outputs (parquet, metrics)
- `-v ./preprocessing:/app/preprocessing` – cached `.npy` signals
- `-v ./checkpoints:/app/checkpoints` – BERT/tokenizer weights (writable for model downloads)
- `-v /mnt/data1/datasets/Harvard-Emory-ECG:/mnt/data1/datasets/Harvard-Emory-ECG:ro` – raw ECG paths (match the `ecg_path` values)

### Preprocess only (CPU)
```bash
docker run --rm \
  -v $(pwd)/inputs:/app/inputs \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/preprocessing:/app/preprocessing \
  -v /media/data1/datasets/Harvard-Emory-ECG:/media/data1/datasets/Harvard-Emory-ECG:ro \
  tokenizer_inference \
  --step preprocess \
  --input /app/inputs/harvard_emory_subset_1k.csv \
  --device cpu
```

### BERT only (use cached preprocessed parquet)
```bash
docker run --rm \
  -v $(pwd)/inputs:/app/inputs \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/preprocessing:/app/preprocessing \
  tokenizer_inference \
  --step bert \
  --input /app/inputs/preprocessed.parquet
```

## Run examples
### Full run (preprocess + BERT) on GPU
```bash
docker run --gpus all --rm \
  -v $(pwd)/inputs:/app/inputs \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/preprocessing:/app/preprocessing \
  -v /media/data1/datasets/Harvard-Emory-ECG:/media/data1/datasets/Harvard-Emory-ECG:ro \
  -v $(pwd)/checkpoints:/app/checkpoints \
  tokenizer_inference \
  --step all \
  --input /app/inputs/harvard_emory_subset_1k.csv \
  --ecg-signals-path /media/data1/datasets/Harvard-Emory-ECG
```

## Using the helper script locally (no container)
```bash
source docker/run_pipeline.bash --step preprocess --input_file docker/harvard_emory_subset_1k.csv
```

## CLI options (inference/main.py)
| Flag | Description |
|------|-------------|
| `--step {preprocess,bert,all}` | Choose stages to run (default `all`). |
| `--input PATH` | Input CSV/Parquet (needs `ecg_path`, `reports`). |
| `--output-dir PATH` | Output directory (default `/app/outputs`). |
| `--ecg-signals-path PATH` | Base directory for ECG files; use if `ecg_path` is relative. |
| `--device DEVICE` | `cuda:0` or `cpu` (or `--cpu`). |
| `--batch-size N` | Dataloader batch size (default 32). |
| `--num-workers N` | Dataloader workers (default 8). |
| `--preprocessing-folder DIR` | Where `.npy` signals are saved (default `/app/preprocessing`). |
| `--preprocessing-n-workers N` | Worker count for preprocessing (default 16). |
| `--dataset-name NAME` | Optional prefix for preprocessing outputs. |
| `--bert-checkpoint PATH` | BERT classifier checkpoint (read-only). |
| `--tokenizer-checkpoint PATH` | ECG tokenizer checkpoint (read-only). |
| `--bert-base-config PATH` | BERT base config yaml (defaults to `config/bert_classifier/base_config.yaml`). |
| `--preprocessing-output PATH` | Path to cached preprocessing parquet to reuse. |
| `--bert-output PATH` | Path to cached BERT labels parquet to reuse. |
| `--quiet` | Reduce log verbosity. |

## Defaults inside the image
- BERT checkpoint: `/app/checkpoints/mimic_mhi_bert`
- Tokenizer checkpoint: `/app/checkpoints/ECG_tokenizer_latest/best_model_epoch_10.pt`
- Preprocessing output dir: `/app/preprocessing`
- Outputs dir: `/app/outputs`

## Notes
- Input file must have `ecg_path` pointing to `.hea` or `.npy` reachable in the mounted paths.
- Preprocessing writes `.npy` files; base64 is no longer supported.
- EfficientNet / linear probing runs are handled separately via `scripts/runner.sh` (not in this container).
