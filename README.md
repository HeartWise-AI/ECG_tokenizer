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

The system can generate comprehensive question-answer datasets from ECG data with multiple prompt types:

```bash
# Generate datasets with default settings (MIMIC dataset)
python dataset_generation/generate_train_test_datasets.py

# Generate with specific dataset type
python dataset_generation/generate_train_test_datasets.py --dataset mimic

# Generate with sample size for testing
python dataset_generation/generate_train_test_datasets.py --dataset mimic --sample_size 100
```

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
