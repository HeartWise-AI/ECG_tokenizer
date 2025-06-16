# Multi-Model LLM Finetuning Configuration Guide

This directory contains configuration files for different foundation models supported by the ECG tokenizer LLM finetuning pipeline.

## Supported Models

The ECG tokenizer now supports the following foundation models:

### 1. GPT-2 (Default)
- **Model**: `gpt2`
- **Hidden Size**: 768
- **Config**: `config/gpt2/base_config.yaml`
- **Use Case**: Small, fast model for quick experimentation

### 2. BLOOM
- **Model**: `bigscience/bloom-560m`
- **Hidden Size**: 1024
- **Config**: `config/bloom/base_config.yaml`
- **Use Case**: Multilingual capabilities, good for diverse text generation

### 3. OPT
- **Model**: `facebook/opt-350m`
- **Hidden Size**: 512
- **Config**: `config/opt/base_config.yaml`
- **Use Case**: Efficient model with good performance/size trade-off

### 4. Mistral
- **Model**: `mistralai/Mistral-7B-v0.1`
- **Hidden Size**: 4096
- **Config**: `config/mistral/base_config.yaml`
- **Use Case**: Large model with excellent performance (requires more GPU memory)

### 5. GPT-Neo
- **Model**: `EleutherAI/gpt-neo-125M`
- **Hidden Size**: 768
- **Config**: `config/gptneo/base_config.yaml`
- **Use Case**: Open-source alternative to GPT-2 with similar size

### 6. GPT-J
- **Model**: `EleutherAI/gpt-j-6B`
- **Hidden Size**: 4096
- **Config**: `config/gptj/base_config.yaml`
- **Use Case**: Very large model with high performance (requires significant GPU memory)

## Configuration Structure

Each model configuration file follows the same structure:

```yaml
# Pipeline and project settings
pipeline_project: "ECG_tokenizer_LLM_finetuning"
runner_name: "LLM_finetuning_runner"

# Model-specific settings
model_type: "opt"  # Specifies which model architecture to use
huggingface_model_name: "facebook/opt-350m"
trainable_model_name: "OPT_WithEmbedding"
embedding_reducer_name: "OPT_SimpleEmbeddingReducer"
opt_embedding_size: 512

# Training hyperparameters (adjusted per model size)
llm_lr: 5e-5
batch_size: 32
num_epochs: 5

# Data paths and other settings...
```

## Key Configuration Parameters

### Model Selection
- `model_type`: Specifies which foundation model to use (`gpt2`, `bloom`, `opt`, `mistral`, `gptneo`, `gptj`)
- `huggingface_model_name`: The exact Hugging Face model identifier
- `trainable_model_name`: The wrapper class name for the model
- `embedding_reducer_name`: The reducer class for ECG embeddings

### Hyperparameter Adjustments
Different model sizes require different hyperparameters:

| Model | Learning Rate | Batch Size | Epochs | Memory Usage |
|-------|---------------|------------|--------|--------------|
| GPT-2 | 5e-5 | 32 | 5 | Low |
| BLOOM | 5e-5 | 32 | 5 | Medium |
| OPT | 5e-5 | 32 | 5 | Low |
| Mistral | 2e-5 | 16 | 3 | High |
| GPT-Neo | 5e-5 | 32 | 5 | Low |
| GPT-J | 1e-5 | 8 | 3 | Very High |

## Usage Examples

### Training with OPT
```bash
cd /volume/ECG_tokenizer
python rg_llm.py --config config/opt/base_config.yaml --run_mode train
```

### Inference with BLOOM
```bash
cd /volume/ECG_tokenizer
python rg_llm.py --config config/bloom/base_config.yaml --run_mode inference
```

### Custom Configuration
You can create custom configurations by copying and modifying any base config:

```bash
cp config/opt/base_config.yaml config/opt/my_custom_config.yaml
# Edit my_custom_config.yaml as needed
python rg_llm.py --config config/opt/my_custom_config.yaml
```

## Memory Requirements

Estimated GPU memory requirements:

- **GPT-2**: ~2-4 GB
- **BLOOM-560M**: ~3-5 GB
- **OPT-350M**: ~2-3 GB
- **Mistral-7B**: ~14-20 GB
- **GPT-Neo-125M**: ~1-2 GB
- **GPT-J-6B**: ~12-18 GB

## Advanced Configuration

### Automatic Parameter Selection
The configuration system automatically selects appropriate parameters based on `model_type`:

```python
# In your config object
embedding_size = config.get_embedding_size()  # Automatically returns correct size
model_class = config.get_model_class_name()   # Returns correct wrapper class
reducer_name = config.get_default_reducer_name()  # Returns correct reducer
```

### Custom Embedding Sizes
You can override default embedding sizes:

```yaml
model_type: "opt"
opt_embedding_size: 1024  # Override default 512
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `batch_size` or switch to a smaller model
2. **Model Not Found**: Ensure the `huggingface_model_name` is correct and accessible
3. **Configuration Errors**: Check that all required fields are present in the YAML file

### Performance Optimization

1. **Use Mixed Precision**: Add `fp16: true` to training parameters
2. **Gradient Checkpointing**: Add `gradient_checkpointing: true` for large models
3. **Model Parallelism**: Use multiple GPUs for very large models

## Contributing

To add support for a new model:

1. Create the model wrapper class (e.g., `models/newmodel_with_embeddings.py`)
2. Create the embedding reducer (add to `models/model_specific_reducers.py`)
3. Register both classes in `models/__init__.py`
4. Add the model type to `LLMFinetuningConfig`
5. Create a base configuration file
6. Update this documentation

For more details, see the implementation in `/volume/ECG_tokenizer/models/` and `/volume/ECG_tokenizer/utils/config/`.
