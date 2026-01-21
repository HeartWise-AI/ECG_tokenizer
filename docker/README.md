# ECG Tokenizer Docker Pipeline

Docker-based inference pipeline for ECG tokenization and classification, following the [DeepECG_Docker](https://github.com/HeartWise-AI/DeepECG_Docker) architecture pattern.

## Architecture

The pipeline uses **BERT predictions from text reports as ground truth** for evaluating signal-based classification:

```
┌─────────────────────────────────────────────────────────────────┐
│             INFERENCE PIPELINE FLOW (in progress)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT: Parquet (waveform_path, report, waveform_name)          │
│                           │                                     │
│            ┌──────────────┴──────────────┐                      │
│            ▼                              ▼                     │
│    ┌──────────────┐              ┌──────────────────┐           │
│    │ Text Reports │              │  Raw Waveforms   │           │
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
│  OUTPUT: JSON (metrics, embeddings, QA) + Parquet (GT labels)   │
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

Input parquet files require only **THREE columns**:

| Column | Description |
|--------|-------------|
| `waveform_path` | Path to ECG signal file (.npy or .hea) |
| `report` | Text report for the ECG |
| `waveform_name` | Unique identifier (optional, defaults to index) |

**No diagnostic label columns are required.** Ground truth is generated from BERT.

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
    ecg-tokenizer \
    --mode full_run \
    --input /app/inputs/mimic_5k_test_robert.parquet \
    --output /app/outputs/results.json \
    --device cuda:0 \
    --batch-size 32 \
    --waveform-path-column waveform_path_psa \
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
    ecg-tokenizer \
    --mode full_run \
    --input /app/inputs/mimic_5k_test_robert.parquet \
    --output /app/outputs/results.json \
    --device cpu \
    --batch-size 8 \
    --waveform-path-column waveform_path_psa
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
cd /volume/ECG_tokenizer
source .venv/bin/activate

python inference/main.py \
    --input /volume/ECG_tokenizer/output/MHI/mimic_5k_test_robert.parquet \
    --output /volume/ECG_tokenizer/docker/results.json \
    --device cuda:0 \
    --batch-size 32 \
    --waveform-path-column waveform_path_psa \
    --bert-checkpoint /volume/ECG_tokenizer/checkpoints/mimic_mhi_bert \
    --tokenizer-checkpoint /volume/ECG_tokenizer/checkpoints/ECG_tokenizer_latest/best_model_epoch_10.pt \
    --efficientnet-checkpoint /volume/ECG_tokenizer/checkpoints/ECG_Tokenizer_Linear_Probing/ECG_Tokenizer_Linear_Probing/5zg01bx6_20250824-041452/checkpoint_epoch_10.pt
```

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--mode MODE` | Pipeline mode: `full_run` (default), `analysis`, `preprocessing` |
| `--input FILE` | Input parquet file path |
| `--output FILE` | Output JSON file path |
| `--device DEVICE` | Device: `cuda:0`, `cpu` |
| `--batch-size N` | Batch size for processing |
| `--waveform-path-column COL` | Column name for waveform paths |
| `--bert-checkpoint PATH` | Path to BERT classifier checkpoint |
| `--tokenizer-checkpoint PATH` | Path to ECG tokenizer checkpoint |
| `--efficientnet-checkpoint PATH` | Path to EfficientNet classifier checkpoint |
| `--no-psa` | Skip PSA normalization |
| `--with-llm-judge` | Run LLM-as-a-Judge evaluation |

## Checkpoint Paths

| Checkpoint | Path |
|------------|------|
| BERT Classifier | `checkpoints/mimic_mhi_bert` |
| ECG Tokenizer | `checkpoints/ECG_tokenizer_latest/best_model_epoch_10.pt` |
| EfficientNet | `checkpoints/ECG_Tokenizer_Linear_Probing/ECG_Tokenizer_Linear_Probing/5zg01bx6_20250824-041452/checkpoint_epoch_10.pt` |

## PSA Normalization Options

| Scenario | Flags |
|----------|-------|
| Data is already PSA-normalized | `--waveform-path-column waveform_path_psa --no-psa` |
| Data is raw, needs preprocessing | `--waveform-path-column waveform_path` (PSA applied by default) |
| Skip all preprocessing | `--no-psa` |

## Volume Mounts (Docker)

| Mount | Description |
|-------|-------------|
| `/app/inputs` | Input parquet files |
| `/app/outputs` | Output results (JSON, CSV) |
| `/app/ecg_signals` | Raw ECG signal files (read-only) |
| `/app/checkpoints` | Model checkpoints (read-only) |
| `/app/config` | Configuration files (read-only) |

## Configuration File

Configuration via `heartwise.config` (key: value format):

```yaml
# Execution mode
mode: full_run

# Device settings
device: cuda:0
batch_size: 32

# Input columns
waveform_path_column: waveform_path_psa
report_column: report
waveform_name_column: waveform_name

# Model checkpoints
bert_checkpoint: /app/checkpoints/mimic_mhi_bert
tokenizer_checkpoint: /app/checkpoints/ECG_tokenizer_latest/best_model_epoch_10.pt
efficientnet_checkpoint: /app/checkpoints/ECG_Tokenizer_Linear_Probing/.../checkpoint_epoch_10.pt

# Preprocessing
apply_psa_normalization: false  # Set true if using raw waveforms

# Classification settings
num_classes: 77
classification_threshold: 0.5
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
      "waveform_name": "12345.npy",
      "waveform_path": "/path/to/waveform.npy",
      "original_report": "Atrial fibrillation...",
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
│   └── README.md
├── inference/
│   ├── main.py              # Entry point
│   ├── pipeline_args.py     # Argument parsing
│   ├── pipeline_config.py   # Configuration
│   ├── ecg_pipeline.py      # Main pipeline logic
│   ├── psa_normalizer.py    # PSA normalization
│   └── llm_judge_wrapper.py # LLM Judge integration
├── models/
│   ├── ecg_tokenizer_wrapper.py
│   ├── encoder/
│   ├── quantizer/
│   ├── decoder/
│   └── bridge/
├── utils/
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
