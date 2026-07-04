# ECG Tokenizer – Docker Inference

This document covers the containerized inference flow (preprocess → BERT → EfficientNet → QA generation → LLM) with all Hugging Face weights/configs baked into the image.

## Input CSV columns

**Required for full pipeline**:
- `ecg_path` – path to each ECG file (e.g. `.hea` or waveform path)
- `reports` – text report / diagnosis for each ECG (used by BERT and report-based QA answers)

**QA-only note**:
- For `--step qa` runs that directly consume a custom parquet, `reports` may be placeholder text if you only want label-backed questions such as `structural_heart_disease` or `lvef`.

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

At QA generation time the pipeline prints which categories are enabled (e.g. “QA feature flags (enabled categories)”). To force-disable categories even when columns exist, run the QA step with `--disable_categories ...` (comma-separated).

Supported `--disable_categories` values include:
- `interpretation`
- `json_interpretation`
- `classification`
- `category`
- `localization`
- `urgency_assessment`
- `random_finding_question`
- `heart_rate`
- `ecg_interval`
- `structural_heart_disease` or `shd`
- `lvef`
- `acs_severity`
- `culprit_artery`
- `afib_risk`

For EchoNext-style QA-only runs, a typical setting is:
```bash
--disable_categories interpretation,json_interpretation,classification,category,localization,urgency_assessment,random_finding_question
```

If your custom dataset lacks DeepECG diagnosis columns, set `qa_max_normal_percentage: 1.0` in `docker/heartwise.config` so the QA builder does not downsample most rows as "normal".

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

QA only from a custom parquet (skip BERT and EfficientNet):
```bash
docker run --gpus all --rm --shm-size=8g \
  -v "$(pwd)/outputs:/app/outputs" \
  tokenizer_inference \
  --step qa \
  --input_file /app/outputs/echonext_docker_ready.parquet
```

Preprocess -> QA -> LLM (skip BERT and EfficientNet):
```bash
docker run --gpus all --rm --shm-size=8g \
  -v "$(pwd)/outputs:/app/outputs" \
  -v "$(pwd)/preprocessing:/app/preprocessing" \
  tokenizer_inference \
  --step preprocess_qa_llm \
  --input_file /app/outputs/echonext_docker_ready.parquet \
  --qa-disable-categories interpretation,json_interpretation,classification,category,localization,urgency_assessment,random_finding_question,heart_rate,ecg_interval \
  --qa-max-normal-percentage 1.0
```
