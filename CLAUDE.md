# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository implements VQ-VAE-based tokenization for ECG signals, enabling language model techniques for cardiac data analysis. The system converts 12-lead ECG waveforms (2500 samples) into discrete tokens for downstream tasks including classification, reconstruction, and report generation.

## Development Commands

### Setup
```bash
pip install uv && uv sync && source .venv/bin/activate
```

### Training Workflows
```bash
# ECG Tokenizer training
bash scripts/runner.sh --base_config config/tokenizer/classification_base_config.yaml --selected_gpus 0,1 --use_wandb true --run_mode train

# LLM fine-tuning (Llama 3.2 / GPT-2)
bash scripts/runner.sh --base_config config/llm_finetuning/llama32_1b/base_config.yaml --selected_gpus 0 --use_wandb false --run_mode train --instruct_mode true

# Hyperparameter sweeps
bash scripts/run_sweep.sh --base_config config/tokenizer/classification_base_config.yaml --sweep_config config/tokenizer/sweep_config.yaml --selected_gpus 0,1 --count 5

# Linear probing
bash scripts/runner.sh --base_config config/linear_probing/base_config.yaml --selected_gpus 0,1 --use_wandb true --run_mode train
```

### Testing
```bash
pytest                        # Full test suite with coverage
pytest -m "not slow"         # Fast tests only
pytest tests/test_models.py  # Specific test file
```

## Code Architecture

### Registry-Based System
The codebase uses a registry pattern for dynamic component instantiation:
- Models registered in `models/__init__.py` via `ModelRegistry`
- Runners registered in `runners/__init__.py` via `RunnerRegistry`
- Projects registered in `projects/__init__.py` via `ProjectRegistry`

### Configuration Hierarchy
```
HeartWiseConfig (base)
├── TokenizerConfig (tokenizer training)
├── LLMFinetuningConfig (LLM adaptation)
├── LinearProbingConfig (feature evaluation)
└── ClassicalBaselineConfig (traditional methods)
```

Configs are YAML-based in `config/` with dataclass validation in `utils/configs.py`.

### Processing Pipeline
```
ECG Signal → Conv_Encoder → ECG_Tokenizer_Quantizer → Tokens
                                                         ↓
                                    [Classification / Reconstruction / LLM]
```

Key components:
- **Encoder**: `models/conv_encoder.py` - Convolutional feature extraction
- **Quantizer**: `models/quantizers.py` - Vector quantization with 512 codes × 8 quantizers
- **Decoders**: Multiple modes in `models/decoders/` for different tasks

### Data Handling
- Primary dataset: `data/datasets.py:ECGDataset` - handles MIMIC-IV parquet files
- Supports stratified sampling, lead-specific normalization
- Clinical report integration for multi-modal training

### Distributed Training
- PyTorch DDP with automatic GPU detection
- Configured via `CUDA_VISIBLE_DEVICES` or `--selected_gpus`
- Checkpoint management with resume capability
- Weights & Biases integration for experiment tracking