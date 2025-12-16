# ECG_tokenizer

## 🛠️ Environment Setup

### Prerequisites

- **CUDA-capable GPU**
- **Python 3.11+**

### Steps

1. 📥 **Clone the Repository**:

   ```bash
   https://github.com/HeartWise-AI/ECG_tokenizer.git
   cd ECG_tokenizer
   ```

2. **Set up Virtual Environment**:

   ```bash
   pip install uv
   uv sync
   ```

3. **Activate Virtual Environment**:

   ```bash
   source .venv/bin/activate
   ```
   
4. **Install yq required to run scripts/run_sweep.sh**:

   ```bash
   wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq && \
   chmod +x /usr/bin/yq
   ```

5. **Log into Weights & Biases required for sweep**:

   ```bash
   wandb login
   ```

## 📊 Dataset Generation

### Generate Q&A Datasets for LLM Fine-tuning

The system can generate comprehensive question-answer datasets from ECG data with multiple prompt types. For production use, generate combined MIMIC-IV and MHI datasets with balanced special questions.

#### Standard Production Dataset (400k train + 10k test)

```bash
# Generate 400k training samples (200k MIMIC + 200k MHI)
# MHI includes 20% special questions: 5% ACS, 5% LVEF, 5% AFib risk, 5% SHD
python dataset_generation/generate_train_test_datasets.py \
  --dataset combined \
  --mimic_train_samples 200000 \
  --mhi_train_samples 200000 \
  --test_samples 0 \
  --max_prompts_per_ecg 1 \
  --max_normal_percentage 0.05

# Generate 10k test samples (5k MIMIC + 5k MHI)
python dataset_generation/generate_train_test_datasets.py \
  --dataset combined \
  --train_samples 0 \
  --mimic_test_samples 5000 \
  --mhi_test_samples 5000 \
  --max_prompts_per_ecg 1 \
  --max_normal_percentage 0.05
```

#### Single Dataset Generation

```bash
# MIMIC-IV only dataset
python dataset_generation/generate_train_test_datasets.py \
  --dataset mimic-iv \
  --train_samples 200000 \
  --test_samples 10000 \
  --max_prompts_per_ecg 1

# MHI only dataset
python dataset_generation/generate_train_test_datasets.py \
  --dataset mhi \
  --train_samples 200000 \
  --test_samples 10000 \
  --max_prompts_per_ecg 1
```

#### Key Parameters

- `--dataset`: Choose `combined`, `mimic-iv`, or `mhi`
- `--max_prompts_per_ecg`: Set to 1 for one prompt per ECG (recommended)
- `--max_normal_percentage`: Limit normal ECGs to 5% (default 0.05)
- `--mimic_train_samples` / `--mhi_train_samples`: Specify samples per dataset for combined mode
- `--mimic_test_samples` / `--mhi_test_samples`: Test set samples per dataset

### Output Files

Generated datasets are saved to `/volume/ECG_tokenizer/output/`:
- Combined: `combined_train_qa_m200k_h200k.parquet`, `combined_test_qa_m5k_h5k.parquet`
- MIMIC: `mimic_train_qa_200k.parquet`, `mimic_test_qa_10k.parquet`
- MHI: `mhi_train_qa_200k.parquet`, `mhi_test_qa_10k.parquet`

### Prompt Types Generated

The dataset generator creates diverse prompts for each ECG:

1. **Interpretation**: Full ECG interpretation with findings
2. **Classification**: Normal/borderline/pathological classification  
3. **Category-specific**: Questions about rhythm, conduction, ischemia, etc.
4. **Localization**: Location-specific findings (Q waves, ST changes, T waves)
5. **Heart rate**: Heart rate extraction questions
6. **Demographics**: Age and gender questions
7. **JSON interpretation**: Structured JSON output of findings
8. **QRS axis**: Axis deviation detection and classification
9. **Special Questions (MHI only)**:
   - **AFib Risk**: 2-year and 5-year atrial fibrillation risk prediction
   - **LVEF**: Left ventricular ejection fraction estimation
   - **ACS**: Acute coronary syndrome detection and culprit artery identification
   - **SHD**: Structural heart disease assessment

### Adding Support for New Datasets

To add support for datasets with different column structures:

1. Edit `dataset_generation/dataset_column_mappings.py`
2. Add your dataset's column mappings
3. Register in the `DATASET_MAPPINGS` dictionary
4. Run generation with `--dataset your_dataset_name`

See `ADDING_NEW_DATASETS.md` for detailed instructions.

## 💻 How to's

### Tokenizer training
```base
# Single GPU training without logging results to wandb (see scripts/runner.sh)
bash scripts/runner.sh --base_config config/tokenizer/base_config.yaml --selected_gpus 0 --use_wandb false --run_mode train

# Multi-GPU training with results logging on wandb (see scripts/runner.sh)
bash scripts/runner.sh --base_config config/tokenizer/base_config.yaml --selected_gpus 0,1 --use_wandb true --run_mode train

# Multi-GPU hyperparameters fine-tuning - RunMode and UseWandb are forced to train and true respectively (see scripts/run_sweep.sh)
bash scripts/run_sweep.sh --base_config config/tokenizer/base_config.yaml --sweep_config config/tokenizer/sweep_config.yaml --selected_gpus 0,1 --count 5
```

### Linear Probing
```base
# Single GPU training without logging results to wandb (see scripts/runner.sh)
bash scripts/runner.sh --base_config config/linear_probing/base_config.yaml --selected_gpus 0 --use_wandb false --run_mode train

# Multi-GPU training with results logging on wandb (see scripts/runner.sh)
bash scripts/runner.sh --base_config config/linear_probing/base_config.yaml --selected_gpus 0,1 --use_wandb true --run_mode train

# Multi-GPU hyperparameters fine-tuning - RunMode and UseWandb are forced to train and true respectively (see scripts/run_sweep.sh)
bash scripts/run_sweep.sh --base_config config/linear_probing/base_config.yaml --sweep_config config/linear_probing/sweep_config.yaml --selected_gpus 0,1 --count 5
```

### LLM Finetuning
##### Run Training
```base
# Single GPU training without logging results to wandb (see scripts/runner.sh)
bash scripts/runner.sh --base_config config/llm_finetuning/llama32_1b/base_config.yaml --selected_gpus 0 --use_wandb false --run_mode train

# Multi-GPU training with results logging on wandb (see scripts/runner.sh)
bash scripts/runner.sh --base_config config/gpt2/base_config.yaml --selected_gpus 0,1 --use_wandb true --run_mode train

# Multi-GPU hyperparameters fine-tuning - RunMode and UseWandb are forced to train and true respectively (see scripts/run_sweep.sh)
bash scripts/run_sweep.sh --base_config config/gpt2/base_config.yaml --sweep_config config/clip/sweep_config.yaml --selected_gpus 0,1 --count 5

# LLM Instruction Tuning (single token)
bash scripts/runner.sh --base_config config/llm_finetuning/llama32_1b/base_config.yaml --selected_gpus 0 --use_wandb false --run_mode train --instruct_mode true

# LLM Instruction Tuning (multi-token)
(Change adapted config file to use multiple tokens)
bash scripts/runner.sh --base_config config/llm_finetuning/llama32_1b/seq_token_config.yaml --selected_gpus 0,1 --use_wandb true --run_mode train --instruct_mode true
```
#### Generate Inference Results
Multi-GPU Inference - no results logged on wandb (see scripts/runner.sh)
```base
source scripts/runner.sh --use_wandb false --run_mode inference --base_config config/gpt2/base_config.yaml --selected_gpus 0,1,2,3
```

### LLMs evaluation with Bert
```base
# Inference on Multi-GPU without logging results to wandb
source scripts/runner.sh --use_wandb false --run_mode inference --base_config config/bert_classifier/base_config.yaml --selected_gpus 0,1,2,3
```

## 📊 Metrics Evaluation

### Online Metrics (During Training)

Metrics are automatically computed during LLM fine-tuning when using the runner scripts:

```bash
# Metrics computed automatically during training and logged to wandb
bash scripts/runner.sh --base_config config/llm_finetuning/llama32_1b/base_config.yaml --selected_gpus 0 --use_wandb true --run_mode train

# Metrics include:
# - ROUGE-1, ROUGE-L
# - BLEU-1, BLEU-4 (using SacreBLEU)
# - METEOR
# - BERTScore (precision, recall, F1)
```

The metrics are computed per batch and aggregated across epochs. Best/worst examples are tracked and logged to wandb for analysis.

### Offline Metrics Evaluation

The system provides comprehensive offline evaluation capabilities for model generations.

#### Quick Evaluation

```python
from utils.metrics.llm_metrics import analyze_generation_results

# Analyze model generations with automatic metric computation
results = analyze_generation_results(
    json_path="checkpoints/your_model/val_generations_epoch_5.json",
    csv_path="output/combined_test_qa_m5k_h5k.csv",  # Optional for categories
    output_path="evaluation_results.json",
    metrics=["bertscore", "rouge", "bleu", "meteor"]
)
```

#### Advanced BERTScore Analysis

```python
from utils.metrics.llm_metrics import compute_bertscore_offline, compute_bertscore_by_category, compute_bertscore_by_dataset

# Basic BERTScore computation
scores = compute_bertscore_offline(
    predictions=["predicted text 1", "predicted text 2"],
    references=["reference text 1", "reference text 2"],
    model_type="microsoft/deberta-xlarge-mnli"  # Pinned for reproducibility
)

# Per-category BERTScore (e.g., interpretation, classification, etc.)
category_scores = compute_bertscore_by_category(
    predictions=predictions,
    references=references,
    categories=["interpretation", "classification", ...],
    min_samples=5  # Minimum samples per category
)

# Dataset-specific scores (MIMIC vs MHI)
dataset_scores = compute_bertscore_by_dataset(
    predictions=predictions,
    references=references,
    filenames=["41234.npy", "00567.npy", ...]  # Auto-detects dataset from prefix
)
```

#### Command-Line Usage

```bash
# Comprehensive evaluation with all metrics
python -c "
from utils.metrics.llm_metrics import analyze_generation_results
analyze_generation_results(
    json_path='checkpoints/model/val_generations.json',
    csv_path='output/test_qa.csv',
    verbose=True
)"

# Output includes:
# - Overall metrics (ROUGE, BLEU, METEOR, BERTScore)
# - Per-category breakdown
# - MIMIC vs MHI comparison
# - Saved to evaluation_results.json
```

### Metrics Computed

#### Standard Metrics
- **ROUGE** (1, L): Text overlap metrics with stemming
- **BLEU** (1, 4): N-gram precision with brevity penalty (SacreBLEU)
- **METEOR**: Semantic similarity with stemming/synonyms
- **BERTScore**: Contextual embeddings similarity using DeBERTa-xlarge

#### Dataset-Specific Analysis
The system automatically identifies and compares:
- **MIMIC-IV**: Files starting with '4' (e.g., 41234.npy)
- **MHI**: Files starting with '0' (e.g., 00567.npy)

#### Category-Based Analysis
Automatic grouping by prompt types:
- **Core Categories**: interpretation, classification, json_interpretation
- **Clinical Assessment**: category_rhythm, category_conduction, category_ischemia
- **Localization**: localization_st_elevation, localization_q_wave, localization_t_wave
- **MHI Special**: afib_risk, lvef, structural_heart_disease, acs_severity, culprit_artery

### Performance Insights

The evaluation provides detailed insights:
1. **Overall Performance**: Aggregate metrics across all samples
2. **Category Performance**: Identify strong/weak prompt types
3. **Dataset Comparison**: MIMIC vs MHI performance differences
4. **Sample Tracking**: Best/worst performing examples

### Example Output

```json
{
  "overall": {
    "bertscore_f1": 0.8234,
    "bertscore_precision": 0.8456,
    "bertscore_recall": 0.8021,
    "rouge1": 0.6543,
    "rougeL": 0.5234,
    "bleu1": 0.4567,
    "bleu4": 0.2345
  },
  "per_category": {
    "interpretation": {
      "f1": 0.8567,
      "precision": 0.8734,
      "recall": 0.8402,
      "n_samples": 523
    },
    "afib_risk": {
      "f1": 0.9234,
      "precision": 0.9345,
      "recall": 0.9125,
      "n_samples": 156
    }
  },
  "per_dataset": {
    "MIMIC": {
      "f1": 0.8123,
      "n_samples": 2500
    },
    "MHI": {
      "f1": 0.8345,
      "n_samples": 2500
    }
  }
}
```

### Requirements

```bash
# Core dependencies (included in pyproject.toml)
uv sync

# Optional for enhanced BERTScore
pip install bert-score

# For direct bert-score usage (faster, GPU-optimized)
pip install torch>=2.6.0
