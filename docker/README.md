# ECG Tokenizer Docker Pipeline

Docker-based inference pipeline for DeepECG-Tok

## Architecture

The pipeline uses **BERT predictions from text reports as ground truth** for evaluating signal-based classification:

```
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE FLOW                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT: CSV/Parquet (ecg_path, reports)                       │
│                           │                                     │
│            ┌──────────────┴──────────────┐                      │
│            ▼                              ▼                     │
│    ┌──────────────┐              ┌──────────────────┐           │
│    │   reports  │              │    ecg_path      │           │
│    │  (text)      │              │  (signal file)   │           │
│    └──────────────┘              └──────────────────┘           │
│            │                              │                     │
│            ▼                              ▼                     │
│    ┌──────────────┐              ┌──────────────────┐           │
│    │     BERT     │              │  ECG Tokenizer   │           │
│    │  (77 class)  │              │ Encoder+Quantizer│           │
│    └──────────────┘              └──────────────────┘           │
│            │                              │                     │
│            ▼                              ▼                     │
│    ┌──────────────┐              ┌──────────────────┐           │
│    │ GROUND TRUTH │              │   Embeddings     │           │
│    │  Predictions │              │   [128, 82]      │           │
│    └──────────────┘              └──────────────────┘           │
│            │                              │                     │
│            │                              ▼                     │
│            │                     ┌──────────────────┐           │
│            │                     │  EfficientNet    │           │
│            │                     │   (77 class)     │           │
│            │                     └──────────────────┘           │
│            │                              │                     │
│            └──────────────┬───────────────┘                     │
│                           ▼                                     │
│            ┌──────────────────────────────┐                     │
│            │   Classification Metrics     │                     │
│            │  (Signal vs Text AUC, F1)    │                     │
│            └──────────────────────────────┘                     │ 
│                           │                                     │
│                           ▼                                     │
│  OUTPUT: JSON (metrics, embeddings, QA) + Parquet (preprocessed)│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```



## Pipeline Components

| Component | Description | Status |
|-----------|-------------|--------|
| BERT Classifier | 77-class text classification (GROUND TRUTH) | ✅ Working |
| ECG Tokenizer | Embedding extraction (Encoder + Quantizer) | ✅ Working |
| EfficientNet | 77-class signal classification (EVALUATED) | ✅ Working |
| QA Generation | Question-answer pairs from reports | ⏳ In progress (disabled by default) |
| Classification Metrics | Signal vs Text AUC, F1, AUPRC | ✅ Working |
| GPT2 Report Generation | Generate reports from embeddings | ⏳ In progress (disabled) |
| MedGemma Answers | Generate QA answers | ⏳ In progress (disabled) |
| LLM-as-a-Judge | Evaluate generated text | ⏳ In progress (disabled) |

## Input Requirements

Input CSV/Parquet files require **TWO columns** (following DeepECG_Docker pattern):

| Column | Description |
|--------|-------------|
| `ecg_path` | Absolute/relative path to ECG signal file |
| `reports` | Text reports for BERT classification |

**No diagnostic label columns are required.** Ground truth is generated from BERT.

The `ecg_path` column should contain the full path (or a path relative to where you run the pipeline).
The `--input` file can be either CSV or Parquet.

## Quick Start

### 1. Build the Docker Image

```bash
cd /volume/ECG_tokenizer
docker build -t ecg-tokenizer -f docker/Dockerfile .
```

### 2. Run with Docker (GPU)

```bash
docker run --gpus all \
    -v /volume/ECG_tokenizer/output/MHI:/app/inputs \
    -v /volume/ECG_tokenizer/docker/outputs:/app/outputs \
    -v /volume/ECG_tokenizer/checkpoints:/app/checkpoints:ro \
    -v /path/to/ecg/signals:/app/ecg_signals:ro \
    ecg-tokenizer \
    --input /app/inputs/data.parquet \
    --output /app/outputs/results.json \
    --device cuda:0 \
    --batch-size 32 \
    --bert-checkpoint /app/checkpoints/mimic_mhi_bert \
    --tokenizer-checkpoint /app/checkpoints/ECG_tokenizer_latest/best_model_epoch_10.pt \
    --efficientnet-checkpoint /app/checkpoints/ECG_Tokenizer_Linear_Probing/ECG_Tokenizer_Linear_Probing/5zg01bx6_20250824-041452/checkpoint_epoch_10.pt
```

### 3. Run with Docker (CPU)

```bash
docker run \
    -v /volume/ECG_tokenizer/output/MHI:/app/inputs \
    -v /volume/ECG_tokenizer/docker/outputs:/app/outputs \
    -v /volume/ECG_tokenizer/checkpoints:/app/checkpoints:ro \
    -v /path/to/ecg/signals:/app/ecg_signals:ro \
    ecg-tokenizer \
    --input /app/inputs/data.parquet \
    --output /app/outputs/results.json \
    --device cpu \
    --batch-size 8
```

### 4. Using Docker Compose

```bash
cd /volume/ECG_tokenizer/docker

# GPU example
docker-compose up ecg_tokenizer

# CPU example
docker-compose up ecg_tokenizer_cpu
```

## Running Without Docker (Development)

```bash
cd /volume/ECG_tokenizer/docker

# Using run_pipeline.bash (flag-based; all steps default to on)
source run_pipeline.bash --input_file /path/to/data.csv
```

Or directly with Python:

```bash
cd /volume/ECG_tokenizer
source .venv/bin/activate

python inference/main.py \
    --input /path/to/data.parquet \
    --output /volume/ECG_tokenizer/docker/results.json \
    --device cuda:0 \
    --batch-size 32 \
    --bert-checkpoint /volume/ECG_tokenizer/checkpoints/mimic_mhi_bert \
    --tokenizer-checkpoint /volume/ECG_tokenizer/checkpoints/ECG_tokenizer_latest/best_model_epoch_10.pt \
    --efficientnet-checkpoint /volume/ECG_tokenizer/checkpoints/ECG_Tokenizer_Linear_Probing/ECG_Tokenizer_Linear_Probing/5zg01bx6_20250824-041452/checkpoint_epoch_10.pt
```

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--input FILE` | Input CSV/Parquet file path |
| `--output FILE` | Output JSON file path |
| `--device DEVICE` | Device: `cuda:0`, `cpu` |
| `--batch-size N` | Batch size for processing |
| `--bert-checkpoint PATH` | Path to BERT classifier checkpoint |
| `--tokenizer-checkpoint PATH` | Path to ECG tokenizer checkpoint |
| `--efficientnet-checkpoint PATH` | Path to EfficientNet classifier checkpoint |
| `--no-psa` | Skip PSA normalization |
| `--dataset-name NAME` | Optional dataset name for preprocessing outputs |
| `--bert-base-config FILE` | BERT base config yaml |
| `--efficientnet-base-config FILE` | EfficientNet base config yaml (runner.sh) |
| `--efficientnet-selected-gpus LIST` | GPU list for EfficientNet runner (e.g., `0` or `0,1`) |
| `--efficientnet-use-wandb BOOL` | Whether EfficientNet runner logs to wandb |
| `--no-preprocessing` | Skip preprocessing (load cached parquet) |
| `--no-bert` | Skip BERT classification (load cached parquet) |
| `--no-efficientnet` | Skip EfficientNet classification |

## Checkpoint Paths

| Checkpoint | Path |
|------------|------|
| BERT Classifier | `checkpoints/mimic_mhi_bert` |
| ECG Tokenizer | `checkpoints/ECG_tokenizer_latest/best_model_epoch_10.pt` |
| EfficientNet | `checkpoints/ECG_Tokenizer_Linear_Probing/ECG_Tokenizer_Linear_Probing/5zg01bx6_20250824-041452/checkpoint_epoch_10.pt` |

## PSA Normalization Options

| Scenario | Flags |
|----------|-------|
| Data is already PSA-normalized | `--no-psa` |
| Data is raw, needs preprocessing | PSA applied by default in `preprocessing`/`full_run` |
| Skip all preprocessing | `--no-psa` |

## Pipeline Flow (flag-based)

All steps default to ON. Use flags to skip and load cached artifacts.

| Step | On by default | Skip flag | Output |
|------|---------------|-----------|--------|
| Preprocessing | yes | `--no-preprocessing` | `<output_dir>/preprocessed.parquet` + `.npy` signals |
| BERT (77-class) | yes | `--no-bert` | Overwrites parquet with 77 class columns; writes `<parquet>.bert_probabilities.csv` |
| EfficientNet | yes | `--no-efficientnet` | Results JSON + summary CSV |

Example: run everything (default)
```bash
source run_pipeline.bash --input_file data.parquet
```

Skip preprocessing (reuse cached parquet)
```bash
source run_pipeline.bash --no-preprocessing --input_file preprocessed.parquet
```

Skip BERT (use cached labels in parquet)
```bash
source run_pipeline.bash --no-bert --input_file preprocessed_with_bert.parquet
```

Skip EfficientNet (keep BERT-only outputs)
```bash
source run_pipeline.bash --no-efficientnet --input_file preprocessed_with_bert.parquet
```

Skip EfficientNet (only preprocess + BERT)
```bash
source run_pipeline.bash --no-efficientnet --input_file data.parquet
```

## Volume Mounts (Docker)

| Mount | Description |
|-------|-------------|
| `/app/inputs` | Input CSV/Parquet files |
| `/app/outputs` | Output results (JSON, CSV) |
| `/app/preprocessing` | Preprocessed `.base64` signal files |
| `/app/ecg_signals` | Raw ECG signal files (read-only) |
| `/app/checkpoints` | Model checkpoints (read-only) |
| `/app/config` | Configuration files (read-only) |

## Configuration File

Configuration via `heartwise.config` (key: value format):

```yaml
use_preprocessing: true
use_bert_classification: true
use_efficientnet_classification: true
use_tokenizer_embeddings: true
enable_report_generation: false

device: cuda:0
batch_size: 32
input_parquet: /app/inputs/data.parquet
output_json: /app/outputs/results.json
output_dir: /app/outputs

# Intermediate outputs (defaults shown)
preprocessing_output: /app/outputs/preprocessed.parquet
bert_output: /app/outputs/preprocessed.parquet    # BERT overwrites parquet
efficientnet_output: /app/outputs/preprocessed.parquet
embeddings_output: /app/outputs/preprocessed.parquet

# Model checkpoints
bert_checkpoint: /app/checkpoints/mimic_mhi_bert
tokenizer_checkpoint: /app/checkpoints/ECG_tokenizer_latest/best_model_epoch_10.pt
efficientnet_checkpoint: /app/checkpoints/ECG_Tokenizer_Linear_Probing/.../checkpoint_epoch_10.pt

# Preprocessing
apply_psa_normalization: true
preprocessing_folder: /app/preprocessing
preprocessing_n_workers: 16
dataset_name: mimic

# Classification settings
num_classes: 77
classification_threshold: 0.5
```

## Bash Runner Script

For running inside the container, use `run_pipeline.bash` (following DeepECG_Docker pattern):

```bash
# Inside container
source run_pipeline.bash --input_file data.csv
# Skip preprocessing (use cached parquet)
source run_pipeline.bash --no-preprocessing --input_file preprocessed.parquet
```

## Output Structure

```json
{
  "metadata": {
    "pipeline_version": "2.0.0",
    "num_samples": 5000,
    "architecture": {
      "ground_truth": "BERT (text classification)",
      "evaluated": "EfficientNet (signal classification)"
    }
  },
  "results": [
    {
      "ecg_path": "/path/to/waveform.npy",
      "reports": "Atrial fibrillation...",
      "tokenizer_embeddings": {
        "quantized_shape": [128, 82],
        "indices_shape": [128, 8],
        "num_codebooks": 8
      },
      "qa_results": [
        {
          "question": "What is the rhythm?",
          "ground_truth_answer": "Atrial fibrillation",
          "category": "category_rhythm"
        }
      ]
    }
  ],
  "aggregate_metrics": {
    "classification_metrics": {
      "signal_vs_text": {
        "overall_macro_auc": 0.9778,
        "RHYTHM": {
          "macro_auc": 1.0,
          "macro_f1": 1.0,
          "prevalence_gt %": 37.5
        },
        "INFARCT, ISCHEMIA": {
          "macro_auc": 0.833,
          "macro_f1": 0.833
        }
      }
    }
  }
}
```

Preprocessing writes a parquet file to `output_dir` named
`<YYYYMMDD_HHMMSS>[_<dataset>]_preprocessed_data.parquet` and saves `.base64`
signals under `<preprocessing_folder>/<YYYYMMDD_HHMMSS>[_<dataset>]_preprocessing/`.
BERT classification overwrites the same parquet with 77 class columns and writes
a probabilities CSV: `<preprocessed_parquet>.bert_probabilities.csv`.

## Classification Metrics

For each diagnostic category (RHYTHM, CONDUCTION, etc.) and 77 classes:

| Metric | Description |
|--------|-------------|
| AUC | Area Under ROC Curve |
| AUPRC | Area Under Precision-Recall Curve |
| F1 | F1 Score at optimal threshold |
| Threshold | Youden Index optimal threshold |
| Prevalence | % positive in GT vs predictions |

## Project Structure

```
ECG_tokenizer/
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yaml
│   ├── heartwise.config
│   ├── entrypoint.sh
│   ├── run_pipeline.bash    # Bash runner (DeepECG_Docker pattern)
│   └── README.md
├── inference/
│   ├── main.py              # Entry point
│   ├── pipeline_args.py     # Argument parsing
│   ├── pipeline_config.py   # Configuration
│   ├── ecg_pipeline.py      # Main pipeline logic
│   ├── files_handler.py     # ECGFileHandler for .base64 files
│   └── llm_judge_wrapper.py # LLM Judge integration
├── models/
│   ├── ecg_tokenizer_wrapper.py
│   ├── encoder/
│   ├── quantizer/
│   ├── decoder/
│   └── bridge/
├── utils/
│   ├── preprocessing/ecg_signal_processor.py  # PSA normalization
│   └── metrics/ecg_metrics.py
└── checkpoints/
```

## Troubleshooting

### Missing Checkpoints
```
Warning: EfficientNet checkpoint not found
```
**Solution:** Provide `--efficientnet-checkpoint` path to a valid Linear Probing checkpoint.

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce `--batch-size` (try 8 or 16).

### Waveform Not Found
```
Warning: Could not load waveform
```
**Solution:** Check that waveform paths in parquet are correct and files exist.

### Wrong Waveform Shape
The pipeline expects waveforms of shape `[12, 2500]` (12 leads, 2500 samples at 500Hz).
