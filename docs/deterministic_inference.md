# Deterministic Inference Guide

This document describes how to run deterministic (reproducible) inference with MedGemma.

## Requirements

1. **Clean codebase** - No `task_hint`, `category_hint`, or `schema_mode` modifications
2. **Transformers >= 4.50.0** - Required for Gemma3/MedGemma compatibility
3. **dtype fix applied** - Remove `dtype=` kwarg in `medgemma_decoder.py` (only `torch_dtype=` needed)

## Key Settings

```python
# In generate_all_qa_pairs.py or your inference script:
do_sample = False          # CRITICAL: Deterministic sampling
temperature = 1.0          # Ignored when do_sample=False, but set for clarity
top_p = 1.0               # Ignored when do_sample=False
top_k = 50                # Ignored when do_sample=False
max_new_tokens = 512      # Adjust based on task
```

## Syntax

### Using generate_all_qa_pairs.py

```bash
cd /volume/ECG_tokenizer
source .venv/bin/activate

python scripts/generate_all_qa_pairs.py \
    --checkpoint /media/data1/models/ECG_Tokenizer/e4dw86nh_20251220-232839/best_model.pt \
    --parquet /volume/ECG_tokenizer/combined_test_qa_m5k_h5k_subset_1k.parquet \
    --output inference_output.csv \
    --batch_size 8
```

### Using the model directly

```python
from models.decoder.medgemma_decoder import MedGemmaDecoder

# Load model
model = MedGemmaDecoder.from_pretrained(checkpoint_path)
model.eval()

# Generate with deterministic settings
with torch.no_grad():
    output = model.generate_report_with_question(
        ecg_signal=ecg_tensor,
        prompt_input_ids=prompt_ids,
        prompt_attention_mask=prompt_mask,
        max_token_length=512,
        do_sample=False,  # CRITICAL
    )
```

## Expected Output

Running inference twice with identical inputs should produce **byte-for-byte identical outputs**.

### Verification

```python
import pandas as pd

# Run inference twice
run1 = pd.read_csv("inference_run1.csv")
run2 = pd.read_csv("inference_run2.csv")

# Compare
matches = (run1['generated'] == run2['generated']).all()
print(f"Deterministic: {matches}")  # Should be True

# Check specific columns
for col in ['generated', 'prompt_category']:
    match_pct = (run1[col] == run2[col]).mean() * 100
    print(f"{col}: {match_pct:.1f}% match")
```

## Troubleshooting

### Non-deterministic outputs?

1. **Check `do_sample`** - Must be `False`
2. **Check for stashed changes** - Run `git stash list` and `git status`
3. **Check for hint parameters** - Search for `task_hint`, `category_hint`, `schema_mode`
   ```bash
   grep -rn "task_hint\|category_hint\|schema_mode" --include="*.py" | grep -v __pycache__ | grep -v .venv
   ```
4. **Verify transformers version** - Must be >= 4.50.0
   ```bash
   python -c "import transformers; print(transformers.__version__)"
   ```

### Category-specific issues?

Some categories are more sensitive to code changes:
- `json_interpretation` - Affected by JSON handling logic
- `structural_heart_disease` - Complex multi-part responses

Simple categories like `afib_risk`, `lvef`, `localization_t_wave` are more robust.

## Files

| File | Description |
|------|-------------|
| `scripts/generate_all_qa_pairs.py` | Main inference script |
| `models/decoder/medgemma_decoder.py` | Decoder with generation logic |
| `runners/llm_finetuning_runner.py` | Training runner (hints removed) |

## History

- **Dec 24, 2024**: Baseline inference run (`inference_best_full.csv`)
- **Jan 27, 2025**: Discovered non-determinism caused by stashed hint code
- **Jan 28, 2025**: Removed hints from training loop, documented deterministic setup
