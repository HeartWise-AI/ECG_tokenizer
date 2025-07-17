# ECG LLM Instruction Tuning

This directory contains the instruction tuning (instruct-tuning) implementation for ECG Large Language Models in the ECG_tokenizer project. The system supports **any Hugging Face LLM decoder** and provides a complete pipeline for training LLMs to follow instructions on ECG-related tasks.

## 🎯 Overview

Instruction tuning enables Large Language Models to follow human instructions effectively. This implementation:

- **Model-Agnostic**: Works with any Hugging Face LLM (GPT2, Llama, BERT, T5, etc.)
- **Multiple Templates**: Supports Alpaca, Vicuna, Chat, and Medical instruction formats
- **Efficient Training**: Uses LoRA (Low-Rank Adaptation) for memory-efficient fine-tuning
- **ECG-Specialized**: Designed for ECG interpretation and medical QA tasks
- **Easy Integration**: Seamlessly integrates with the ECG tokenizer pipeline

## 📁 Directory Structure

```
ECG_tokenizer/
├── utils/instruct_tuning/
│   ├── data_formatter.py          # Format ECG QA data for instruction tuning
│   ├── instructor.py              # SFTTrainer-based instruction tuning
│   └── __init__.py
├── config/instruct_tuning/
│   ├── gpt2_instruct_base.yaml    # GPT2 instruction tuning config
│   └── llama32_1b_instruct_base.yaml  # Llama3.2 instruction tuning config
├── projects/
│   └── instruct_tuning_project.py # Project integration
├── runners/
│   └── llm_instruction_tuning_runner.py  # Runner for instruction tuning
├── demo_instruct_tuning.py       # Demo script
└── test_ecg_qa_sample/           # Sample ECG QA dataset
    ├── train.json
    └── eval.json
```

## 🚀 Quick Start

### 1. Demo the Data Formatting

See how different instruction templates format your ECG QA data:

```bash
cd /volume/ECG_tokenizer
python demo_instruct_tuning.py --action format_demo
```

This will show you how the same ECG question-answer pair looks in different instruction formats (Alpaca, Vicuna, Chat, Medical).

### 2. Run Instruction Tuning

#### With GPT2 (lightweight, good for testing):
```bash
python demo_instruct_tuning.py --action train --model gpt2
```

#### With Llama3.2 1B (more powerful):
```bash
python demo_instruct_tuning.py --action train --model llama32_1b
```

#### Run both demo and training:
```bash
python demo_instruct_tuning.py --action both --model gpt2
```

## 📋 Instruction Templates

The system supports multiple instruction templates to suit different use cases:

### 1. Alpaca Template
Best for: General instruction following
```
### Instruction:
Analyze this ECG and provide a diagnosis.

### Input:
ECG shows irregular rhythm with QRS widening...

### Response:
The ECG shows atrial fibrillation with bundle branch block...
```

### 2. Vicuna Template
Best for: Conversational AI applications
```
USER: Can you analyze this ECG? The rhythm appears irregular...
ASSISTANT: Based on the ECG findings, this shows atrial fibrillation...
```

### 3. Chat Template
Best for: Modern chat-based models
```
<|system|>You are a helpful medical AI assistant specializing in ECG interpretation.
<|user|>Analyze this ECG: irregular rhythm, QRS widening...
<|assistant|>This ECG demonstrates atrial fibrillation with bundle branch block...
```

### 4. Medical Template
Best for: Medical domain applications with enhanced clinical context
```
CLINICAL_CONTEXT: ECG Analysis Request
PATIENT_DATA: [Enhanced with metadata and clinical context]
QUESTION: What is the diagnosis for this ECG?
CLINICAL_RESPONSE: [Detailed medical analysis]
```

## 🛠️ Configuration

### Model-Agnostic Parameters

The configs use only general parameters that work with any Hugging Face model:

```yaml
# Model parameters - UNIVERSAL for any HF model
huggingface_model_name: "gpt2"  # Any HF model: "meta-llama/Llama-3.2-1B", "microsoft/DialoGPT-large", etc.
llm_input_embedding_size: 768   # Embedding size of the chosen model
decoder_name: "GPT2_Decoder"    # Corresponding decoder wrapper
adapter_name: "GPT2_LinearAdapter"  # Corresponding adapter

# Instruction tuning specific
template_style: "alpaca"        # Template format
max_seq_length: 512            # Maximum sequence length
include_metadata: true         # Include ECG metadata in prompts
enhanced_prompt: true          # Use enhanced medical prompts
```

### Supported Models

The system works with any Hugging Face model. Some tested examples:

- **GPT Family**: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
- **Llama Family**: `meta-llama/Llama-3.2-1B`, `meta-llama/Llama-3.2-3B`
- **Dialog Models**: `microsoft/DialoGPT-medium`, `microsoft/DialoGPT-large`
- **Domain-Specific**: `microsoft/BioGPT`, `dmis-lab/biobert-base-cased-v1.1`
- **Open Models**: `EleutherAI/gpt-neo-1.3B`, `EleutherAI/gpt-j-6B`

## 📊 Dataset Format

Your ECG QA dataset should be in JSON format with these fields:

```json
[
  {
    "question": "What is the diagnosis for this ECG?",
    "answer": "Atrial fibrillation with slow ventricular response",
    "signal_path": "/path/to/ecg/signal.npy",
    "detected_conditions": ["Afib", "Bradycardia", "..."],
    "condition_categories": "{'Rhythm Disorders': [...]}",
    "original_report": "Clinical report text..."
  }
]
```

Required fields: `question`, `answer`
Optional fields: `signal_path`, `detected_conditions`, `condition_categories`, `original_report`

## 🎛️ Advanced Usage

### Custom Configuration

Create your own config file:

```yaml
# my_custom_instruct_config.yaml
project_name: "ECG_TOKENIZER_INSTRUCT_TUNING"
experiment_name: "my_custom_experiment"

# Use any Hugging Face model
huggingface_model_name: "microsoft/DialoGPT-large"
llm_input_embedding_size: 1024
decoder_name: "GPT2_Decoder"  # Reuse existing decoder
adapter_name: "GPT2_LinearAdapter"

# Your dataset
train_data_path: "/path/to/your/train.json"
eval_data_path: "/path/to/your/eval.json"

# LoRA settings for efficient training
use_lora: true
lora_r: 32
lora_alpha: 64

# Template and format
template_style: "medical"
include_metadata: true
enhanced_prompt: true
```

### Run with Custom Config

```bash
python runners/llm_instruction_tuning_runner.py --config my_custom_instruct_config.yaml
```

### Programmatic Usage

```python
from utils.config import load_config
from utils.config.instruct_tuning_config import InstructTuningConfig
from runners.llm_instruction_tuning_runner import InstructTuningRunner

# Load config
config = load_config("config/instruct_tuning/gpt2_instruct_base.yaml", InstructTuningConfig)

# Customize if needed
config.huggingface_model_name = "microsoft/DialoGPT-medium"
config.template_style = "chat"

# Run instruction tuning
runner = InstructTuningRunner(config, "./my_experiment")
runner.run()
```

## 🔧 Key Features

### 1. LoRA (Low-Rank Adaptation)
Enables efficient fine-tuning with minimal GPU memory:
- **Memory Efficient**: Reduces memory usage by 3x
- **Fast Training**: Faster convergence than full fine-tuning
- **Quality**: Maintains model quality while being efficient

### 2. Multiple Instruction Formats
Choose the format that best suits your use case:
- **Alpaca**: Standard instruction-following format
- **Vicuna**: Conversational format
- **Chat**: Modern chat template with system messages
- **Medical**: Enhanced medical context and prompts

### 3. Enhanced Medical Prompts
When `enhanced_prompt=true`, the system adds:
- Clinical context and terminology
- ECG-specific analysis guidelines
- Medical reasoning prompts
- Structured output formatting

### 4. Metadata Integration
When `include_metadata=true`, prompts include:
- Detected ECG conditions
- Condition categories
- Original clinical reports
- Signal processing information

## 📈 Training Tips

### 1. Model Selection
- **For testing/development**: Use `gpt2` (fast, lightweight)
- **For production**: Use `meta-llama/Llama-3.2-1B` or larger
- **For conversation**: Use `microsoft/DialoGPT-medium`
- **For medical domain**: Use `microsoft/BioGPT`

### 2. Hyperparameter Tuning
```yaml
# Conservative settings (stable, slower)
learning_rate: 1e-5
per_device_train_batch_size: 2
gradient_accumulation_steps: 8

# Aggressive settings (faster, less stable)
learning_rate: 5e-5
per_device_train_batch_size: 8
gradient_accumulation_steps: 2
```

### 3. LoRA Configuration
```yaml
# Lighter LoRA (less parameters, faster)
lora_r: 8
lora_alpha: 16

# Heavier LoRA (more parameters, better quality)
lora_r: 64
lora_alpha: 128
```

## 🔍 Monitoring and Evaluation

### WandB Integration
The system automatically logs to Weights & Biases:
- Training loss and learning rate
- Evaluation metrics
- Model parameters and hyperparameters
- Generated examples

### Evaluation Metrics
- **Perplexity**: Model confidence in predictions
- **BLEU Score**: Quality of generated text
- **Clinical Accuracy**: Domain-specific evaluation (coming soon)

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```yaml
   # Reduce batch size and increase gradient accumulation
   per_device_train_batch_size: 1
   gradient_accumulation_steps: 16
   gradient_checkpointing: true
   ```

2. **Model Not Found**
   ```bash
   # Make sure the model exists on Hugging Face Hub
   huggingface-cli login
   ```

3. **Dataset Loading Error**
   ```python
   # Check your JSON format
   import json
   with open("your_dataset.json") as f:
       data = json.load(f)
   print(f"Loaded {len(data)} examples")
   ```

### Debug Mode
```bash
# Run with debug logging
PYTHONPATH=/volume/ECG_tokenizer python demo_instruct_tuning.py --action train --model gpt2 --debug
```

## 🔮 Future Enhancements

- [ ] Support for multi-modal instruction tuning (ECG signals + text)
- [ ] Integration with Hugging Face Hub for model sharing
- [ ] Advanced evaluation metrics for medical QA
- [ ] Automatic hyperparameter optimization
- [ ] Support for distributed training across multiple GPUs

## 📚 References

- [Hugging Face SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Alpaca Instruction Format](https://github.com/tatsu-lab/stanford_alpaca)
- [Vicuna Conversation Format](https://github.com/lm-sys/FastChat)

---

For questions or issues, please refer to the main ECG_tokenizer documentation or create an issue in the project repository.
