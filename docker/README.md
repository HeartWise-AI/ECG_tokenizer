# ECG Tokenizer – Docker Inference

This document covers the containerized inference flow (preprocess → BERT → EfficientNet) with all Hugging Face weights/configs baked into the image.

## Build the image (prefetch weights)
Requires a HF token in root in `api_keys.json` (key: `HUGGING_FACE_TOKEN`).
All weights uploaded at HF: https://huggingface.co/collections/heartwise/deepecg-tok


Weights/configs after build:
- Tokenizer checkpoint: `/app/checkpoints/deepecg_tokenizer.pt`
- EfficientNet checkpoint: `/app/checkpoints/deepecg_tokenizer_efficientnet.pt`
- EfficientNet config: `/app/checkpoints/efficientnet/base_config.yaml`
- BERT snapshot: `/app/checkpoints/bert/…`


## Run commands

Docker image creation:
```bash
docker build --no-cache -f docker/Dockerfile -t tokenizer_inference .
```


Full pipeline (preprocess + BERT + EfficientNet):
```bash
docker run --gpus all --rm --shm-size=8g -v "$(pwd)/inputs:/app/inputs" -v "$(pwd)/outputs:/app/outputs" -v "$(pwd)/preprocessing:/app/preprocessing" -v /mnt/data1/datasets/Harvard-Emory-ECG:/mnt/data1/datasets/Harvard-Emory-ECG:ro tokenizer_inference --step all --input_file /app/inputs/harvard_emory_subset_1k.csv
```

Preprocess only:
```bash
docker run --rm --shm-size=4g \
  -v "$(pwd)/inputs:/app/inputs" \
  -v "$(pwd)/outputs:/app/outputs" \
  -v "$(pwd)/preprocessing:/app/preprocessing" \
  -v /mnt/data1/datasets/Harvard-Emory-ECG:/mnt/data1/datasets/Harvard-Emory-ECG:ro \
  tokenizer_inference \
  --step preprocess \
  --input /app/inputs/harvard_emory_subset_1k.csv
```

BERT only (csv can be used too):
```bash
docker run --gpus all --rm --shm-size=8g -v "$(pwd)/outputs:/app/outputs" tokenizer_inference --step bert --input_file /app/outputs/preprocessed.parquet
```

Analysis (BERT → EfficientNet, skip preprocessing):
```bash
docker run --gpus all --rm --shm-size=8g \
  -v "$(pwd)/outputs:/app/outputs" \
  tokenizer_inference \
  --step analysis \
  --input /app/outputs/preprocessed.parquet
```
