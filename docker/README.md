# ECG Tokenizer – Docker Inference

This document covers the containerized inference flow (preprocess → BERT → EfficientNet → QA generation → LLM) with all Hugging Face weights/configs baked into the image.

## Input CSV columns

**Required** (must be present in your input CSV/Parquet):
- `ecg_path` – path to each ECG file (e.g. `.hea` or waveform path)
- `reports` – text report / diagnosis for each ECG (used by BERT and QA answers)

**Optional** – include these columns in your input CSV to enable additional QA question categories. The pipeline auto-detects columns and only generates questions for which data exists.

| Optional column(s) | QA categories enabled |
|--------------------|------------------------|
| `heart_rate`, or `rr_interval`, or `RestingECG_OriginalRestingECGMeasurements_VentricularRate` | Heart rate questions; `heart_rate_bpm` in JSON interpretation |
| `pr_interval`, or `p_onset` + `qrs_onset`, or RestingECG PInterval columns | PR interval questions |
| `qt_interval`, or `qtc_interval`, or `qrs_onset` + `t_end`, or RestingECG QTInterval columns | QT/QTc interval questions |
| `echonext_shd` | Structural heart disease questions |
| `deepecho_Visually_Estimated_EF` | LVEF questions |
| `acs_condition_severity` | ACS severity questions |
| `acs_condition_severity` + `acs_pci_regions` | Culprit artery questions (when acute occlusion) |
| `afib_label_2y` + `afib_label_5y` | AFib risk questions |

At QA generation time the pipeline prints which categories are enabled (e.g. “QA feature flags (enabled categories)”). To force-disable categories even when columns exist, run the QA step with `--disable_categories heart_rate,lvef` (comma-separated).

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
