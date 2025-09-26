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
# - ROUGE-1, ROUGE-2, ROUGE-L
# - BLEU-1, BLEU-4 (using SacreBLEU)
# - METEOR
# - BERTScore (precision, recall, F1)
```

The metrics are computed per batch and aggregated across epochs. Best/worst examples are tracked and logged to wandb for analysis.

### Offline Metrics Evaluation

Evaluate saved model generations against ground truth:

#### Basic Usage

```bash
# Evaluate generation JSON against CSV ground truth
python utils/evaluate_offline_metrics.py \
  --json-path checkpoints/your_model/val_generations_epoch_5.json \
  --csv-path output/combined_test_qa_m5k_h5k.csv \
  --output-path evaluation_metrics.json
```

#### With Visualization

```bash
# Generate comprehensive plots and statistics
python utils/plot_metrics.py \
  --metrics-file evaluation_metrics.json \
  --output-dir metric_plots

# Outputs:
# - overall_metrics.png: Bar chart of all metrics
# - category_metrics_comparison.png: Per-category comparison
# - category_metrics_heatmap.png: Heatmap visualization
# - summary_statistics.txt: Detailed statistics
```

#### Full Pipeline Example

```bash
# 1. Convert parquet to CSV if needed
python -c "import pandas as pd; df = pd.read_parquet('output/combined_test_qa_m5k_h5k.parquet'); df.to_csv('output/test_set.csv', index=False)"

# 2. Run evaluation
python utils/evaluate_offline_metrics.py \
  --json-path checkpoints/ECG_Tokenizer_LLM_Finetuning/val_generations.json \
  --csv-path output/test_set.csv \
  --output-path full_evaluation.json

# 3. Generate visualizations
python utils/plot_metrics.py \
  --metrics-file full_evaluation.json \
  --output-dir evaluation_plots
```

### Metrics Computed

- **ROUGE** (1, 2, L): Text overlap metrics
- **BLEU** (1, 4): N-gram precision with brevity penalty
- **METEOR**: Semantic similarity with stemming/synonyms
- **BERTScore**: Contextual embeddings similarity (requires torch>=2.6)

### Per-Category Analysis

The evaluation system automatically groups results by prompt categories:
- **interpretation**: Full ECG interpretation
- **classification**: Normal/pathological classification
- **category_rhythm**: Rhythm-specific questions
- **localization_***: Location-specific findings (ST, Q waves, T waves)
- **json_interpretation**: Structured JSON output
- **Special categories** (MHI): AFib risk, LVEF, ACS severity, culprit artery

### Requirements

```bash
# Install metric dependencies
pip install sacrebleu bert-score evaluate

# For BERTScore (optional, requires torch>=2.6)
uv pip install torch>=2.6.0
