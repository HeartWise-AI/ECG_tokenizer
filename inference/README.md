# ECG Inference Scripts

This directory contains scripts for generating ECG interpretations and QA responses using trained models.

## Scripts Overview

| Script | Description |
|--------|-------------|
| `generate_all_qa_pairs.py` | Generate answers for all QA pairs in a dataset |
| `generate_ecg_answer.py` | Generate single ECG interpretation |
| `generate_stage1_val_reports.py` | Generate validation reports for Stage 1 models |
| `generate_stage1_etg_preview.py` | Generate ECG token preview visualizations |
| `generate_cf_for_mimic_mhi.py` | Generate counterfactual explanations |
| `evaluate_cf.py` | Evaluate counterfactual generation quality |
| `test_emergent_properties.py` | Test emergent model capabilities |

## Quick Start

### Generate QA Pairs

```bash
CUDA_VISIBLE_DEVICES=0 python inference/generate_all_qa_pairs.py \
    --checkpoint checkpoints/ECG_Tokenizer_LLM_Finetuning/ECG_tokenizer_MedGemma/e4dw86nh_20251220-232839_BEST_QFORMER_8CB/best_model.pt \
    --validation_parquet output/combined_test_qa_m25k_h25k.parquet \
    --output_dir inference/ \
    --max_samples 10 \
    --answer_column generated_answer \
    --output_prefix demo_10samples
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | Required | Path to model checkpoint |
| `--validation_parquet` | `output/combined_test_qa_m25k_h25k_schema_v1.parquet` | Path to validation data |
| `--output_dir` | Required | Directory for output files |
| `--max_samples` | None | Limit number of samples (optional) |
| `--waveform_column` | `waveform_path_psa` | Column containing ECG waveform paths |
| `--question_column` | `prompt` | Column containing questions |
| `--answer_column` | `target_json` | Column containing ground truth answers |
| `--output_prefix` | `all_qa_generations` | Prefix for output files |
| `--save_interval` | 1000 | Save checkpoint every N samples |
| `--resume` | False | Resume from existing checkpoint |
| `--device` | 0 | GPU device ID |

## Output Format

The script generates two output files:

### CSV Output (`{output_prefix}.csv`)

| Column | Description |
|--------|-------------|
| `waveform_name` | ECG waveform identifier |
| `waveform_path` | Full path to waveform file |
| `question` | Input question/prompt |
| `generation` | Model-generated answer |
| `ground_truth` | Expected answer |
| `prompt_category` | Category of the question |

### JSON Output (`{output_prefix}.json`)

Grouped by waveform name:
```json
{
  "waveform_name.npy": [
    {
      "Question": "...",
      "Generation": "...",
      "Ground truth": "...",
      "Category": "..."
    }
  ]
}
```

## Metrics

The script automatically computes:
- **ROUGE-1/2/L**: Text overlap metrics
- **BLEU-1/4**: N-gram precision
- **METEOR**: Semantic similarity

Per-category breakdowns are also provided.

## Example Output (10 samples)

```
Overall Metrics (10 samples):
  ROUGE-1: 0.7725
  ROUGE-2: 0.7132
  ROUGE-L: 0.7725
  BLEU-1:  0.7523
  BLEU-4:  0.6565
  METEOR:  0.8222

Per-Category Metrics:
  interpretation: ROUGE-L=0.9282 (2 samples)
  json_interpretation: ROUGE-L=0.9308 (2 samples)
  classification: ROUGE-L=0.3684 (2 samples)
  category_infarct_ischemia: ROUGE-L=0.6316 (1 samples)
  category_rhythm: ROUGE-L=0.8194 (2 samples)
  category_other: ROUGE-L=1.0000 (1 samples)
```

## Best Model Checkpoints

| Checkpoint | Description |
|------------|-------------|
| `e4dw86nh_20251220-232839_BEST_QFORMER_8CB` | Best QFormer model with 8 codebooks |
| `BEST_QFORMER_d8389lsr_20251129-073725` | Previous best QFormer model |
| `BEST_INFONCE_PRETRAIN_3fhs6uty_20251014-005657` | Best InfoNCE pretrained model |

## Question Categories

The model supports various question types:

- `interpretation` - Full ECG interpretation
- `json_interpretation` - Structured JSON output with findings
- `classification` - Binary/categorical classification
- `category_rhythm` - Rhythm-specific questions
- `category_infarct_ischemia` - Ischemia/infarct detection
- `category_conduction` - Conduction abnormalities
- `category_other` - Other ECG findings

## Notes

- ECG waveforms should be numpy arrays with shape `(2500, 12)` or `(12, 2500)`
- The model uses MedGemma-4B as the base LLM with LoRA fine-tuning
- Checkpoint includes bridge configuration for ECG token projection
