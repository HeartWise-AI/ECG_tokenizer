# ECG Tokenizer Docker Pipeline

Docker-based inference pipeline for DeepECG-Tok

## Architecture

The pipeline uses **BERT predictions from text reports as ground truth** for evaluating signal-based classification:

```
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE FLOW                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT: CSV/Parquet (ecg_path, diagnosis)                       │
│                           │                                     │
│            ┌──────────────┴──────────────┐                      │
│            ▼                              ▼                     │
│    ┌──────────────┐              ┌──────────────────┐           │
│    │   diagnosis  │              │    ecg_path      │           │
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
| QA Generation | Question-answer pairs from reports | ⏳ In progress |
| Classification Metrics | Signal vs Text AUC, F1, AUPRC | ✅ Working |
| GPT2 Report Generation | Generate reports from embeddings | ⏳ In progress |
| MedGemma Answers | Generate QA answers | ⏳ In progress |
| LLM-as-a-Judge | Evaluate generated text | ⏳ In progress |

## Input Requirements

Input CSV/Parquet files require **TWO columns** (following DeepECG_Docker pattern):

| Column | Description |
|--------|-------------|
| `ecg_path` | Absolute/relative path to ECG signal file |
| `diagnosis` | Text report/diagnosis for BERT classification |

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
    --mode full_run \
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
    --mode full_run \
    --input /app/inputs/data.parquet \
    --output /app/outputs/results.json \
    --device cpu \
    --batch-size 8
```

### 4. Using Docker Compose

```bash
cd /volume/ECG_tokenizer/docker

# GPU mode
docker-compose up ecg_tokenizer

# CPU mode
docker-compose up ecg_tokenizer_cpu
```

## Running Without Docker (Development)

```bash
cd /volume/ECG_tokenizer/docker

# Using run_pipeline.bash (recommended)
source run_pipeline.bash \
    --mode full_run \
    --input_file /path/to/data.csv
```

Or directly with Python:

```bash
cd /volume/ECG_tokenizer
source .venv/bin/activate

python inference/main.py \
    --mode full_run \
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
| `--mode MODE` | Pipeline mode: `full_run` (default), `analysis`, `preprocessing` |
| `--input FILE` | Input CSV/Parquet file path |
| `--output FILE` | Output JSON file path |
| `--device DEVICE` | Device: `cuda:0`, `cpu` |
| `--batch-size N` | Batch size for processing |
| `--bert-checkpoint PATH` | Path to BERT classifier checkpoint |
| `--tokenizer-checkpoint PATH` | Path to ECG tokenizer checkpoint |
| `--efficientnet-checkpoint PATH` | Path to EfficientNet classifier checkpoint |
| `--no-psa` | Skip PSA normalization |
| `--with-llm-judge` | Run LLM-as-a-Judge evaluation |
| `--dataset-name NAME` | Optional dataset name for preprocessing outputs |

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

## Pipeline Modes

The pipeline supports three execution modes (following DeepECG_Docker pattern):

| Mode | Description |
|------|-------------|
| `preprocessing` | Load raw signals, apply PSA normalization, save as `.base64` files |
| `analysis` | Load preprocessed `.base64` files, run BERT/EfficientNet, compute metrics |
| `full_run` | Run both preprocessing and analysis in sequence |

### Running Preprocessing Only
```bash
docker run --gpus all \
    -v ./inputs:/app/inputs \
    -v ./preprocessing:/app/preprocessing \
    ecg-tokenizer --mode preprocessing --input /app/inputs/data.parquet --dataset-name mimic
```

### Running Analysis Only (on preprocessed data)
```bash
docker run --gpus all \
    -v ./preprocessing:/app/preprocessing \
    -v ./outputs:/app/outputs \
    ecg-tokenizer --mode analysis --input /app/inputs/20260101_123456_mimic_preprocessed_data.parquet
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
# Execution mode: preprocessing, analysis, full_run
mode: full_run

# Device settings
device: cuda:0
batch_size: 32

# Input columns (fixed)
# Required: ecg_path, diagnosis

# Model checkpoints
bert_checkpoint: /app/checkpoints/mimic_mhi_bert
tokenizer_checkpoint: /app/checkpoints/ECG_tokenizer_latest/best_model_epoch_10.pt
efficientnet_checkpoint: /app/checkpoints/ECG_Tokenizer_Linear_Probing/.../checkpoint_epoch_10.pt

# Preprocessing (following DeepECG_Docker pattern)
apply_psa_normalization: true  # Set true if using raw waveforms
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
source run_pipeline.bash --mode full_run --input_file data.csv

# Or with specific mode
source run_pipeline.bash --mode preprocessing --input_file data.parquet
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
      "diagnosis": "Atrial fibrillation...",
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
