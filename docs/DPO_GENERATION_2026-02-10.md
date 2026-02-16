# DPO Dataset Generation Summary

**Date**: February 10, 2026
**Status**: Complete

## Overview

Generated a focused DPO training dataset with 53,146 samples using varied temperature sampling for diverse candidate generation. This dataset targets weak LLM-judge categories and includes special handling for ST elevation localization.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 53,146 |
| Categories | 13 |
| Generations per Sample | 5 |
| Temperatures Used | 0.3, 0.5, 0.7, 0.9, 1.0 |
| Total Generation Time | ~72 hours |

## Category Distribution

| Category | Samples | Purpose |
|----------|---------|---------|
| classification | 12,000 | General ECG classification |
| interpretation | 11,000 | Free-text interpretation |
| acs_severity | 7,000 | Acute coronary syndrome |
| json_interpretation | 6,000 | Structured JSON output |
| category_infarct_ischemia | 3,500 | MI/ischemia detection |
| localization_st_elevation | 3,000 | ST elevation bucket (regex filtered) |
| category_rhythm | 2,500 | Rhythm analysis |
| category_conduction | 2,500 | Conduction abnormalities |
| afib_risk | 2,000 | AFib risk prediction |
| structural_heart_disease | 1,500 | Structural heart disease |
| lvef | 1,000 | LV ejection fraction |
| culprit_artery | 919 | Culprit vessel identification |
| urgency_assessment | 70 | Clinical urgency |

## Excluded Categories

Categories excluded from sampling (already good or noisy):
- category_other
- category_chamber_enlargement
- category_pericarditis
- random_finding_question
- localization_qrs_axis
- localization_q_wave
- localization_t_wave
- localization_st_depression
- ecg_interval

## Temperature Strategy

Used varied temperatures to create diverse candidates for DPO ranking:

| Temperature | Purpose | Characteristics |
|-------------|---------|-----------------|
| 0.3 | Conservative | Safe, often matches ground truth |
| 0.5 | Low-moderate | Slightly more variation |
| 0.7 | Balanced | Good diversity |
| 0.9 | High | More creative outputs |
| 1.0 | Maximum | Riskiest, potential hallucinations |

### Diversity Analysis (first 100 samples)
- Average unique generations per sample: 2.7 / 5
- Samples with all 5 unique: 14%
- Samples with 4+ unique: 34%
- Samples with all same: 23%

## Files Created

### Configuration Files
```
config/dpo_sampling_50k.json      # Category sampling config
config/dpo_bucket_sampling_50k.json  # ST elevation bucket config
config/dpo_sampling_75k.json      # Larger 75k version (unused)
```

### Scripts
```
scripts/sample_and_generate_dpo.py  # Main sampling & generation script
```

### Output Files
```
output/dpo_focused_50k/
├── sampled_data.parquet          # Sampled training data
├── dpo_generations.jsonl         # Generated candidates (53,146 samples)
├── generation_checkpoint.json    # Resume checkpoint
└── generation.log                # Full generation log
```

## Output Format

Each line in `dpo_generations.jsonl`:
```json
{
  "waveform_path": "/path/to/ecg.npy",
  "waveform_name": "patient_id.npy",
  "prompt": "Is this ECG concerning?",
  "ground_truth": "Yes - ECG shows significant abnormalities...",
  "category": "classification",
  "generations": [
    {"text": "Yes - ECG shows...", "temperature": 0.3},
    {"text": "Yes - ECG shows...", "temperature": 0.5},
    {"text": "Yes - ECG shows...", "temperature": 0.7},
    {"text": "Yes - abnormal...", "temperature": 0.9},
    {"text": "Yes - pathological...", "temperature": 1.0}
  ]
}
```

## Example: Temperature Diversity Effect

**High Diversity Case (urgency_assessment)**:
```
Q: Does this ECG require immediate intervention?
GT: URGENT: Immediate intervention required - ST elevation...

T=0.3: Yes - Atrial tachycardia; Ventricular paced
T=0.5: Yes - significant ECG abnormality; Atrial tachycardia
T=0.7: Yes - significant ECG change; Urgent intervention recommended
T=0.9: Yes - Atrial tachycardia; Supraventricular tachycardia
T=1.0: Yes - Previous anterior MI  ← Different finding!
```

**Low Diversity Case (simple question)**:
```
Q: Are there any ectopic beats?
GT: No - no ectopic beats present

All temperatures: No - no ectopic beats present
```

## Next Steps

1. **Rank generations** using LLM judge (e.g., deepseek-v3)
2. **Filter by baseline margin** (keep pairs where margin ≥ 1.0)
3. **Create DPO pairs** (chosen/rejected JSONL)
4. **Train DPO model** with:
   - `sft_weight`: 0.4 (preserve generation quality)
   - `beta`: 0.05 (stable preference learning)
   - Epochs: 1
   - Learning rate: 2e-5

## Previous DPO Results (for reference)

From earlier training with `sft_weight=0.4`, `beta=0.05`, margin≥1.0:

| Model | Pref Accuracy | ROUGE-L |
|-------|--------------|---------|
| Baseline (no DPO) | - | 0.74 |
| Previous DPO (no SFT anchor) | 76.6% | 0.47 |
| DPO with SFT anchor | **92.2%** | 0.58 |

## Commands

### Monitor Progress
```bash
tail -20 output/dpo_focused_50k/generation.log
wc -l output/dpo_focused_50k/dpo_generations.jsonl
```

### Resume Generation (if interrupted)
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/sample_and_generate_dpo.py \
  --checkpoint "checkpoints/BEST_LLM/.../best_model.pt" \
  --sampling_config "config/dpo_sampling_50k.json" \
  --bucket_config "config/dpo_bucket_sampling_50k.json" \
  --output_dir "output/dpo_focused_50k" \
  --temperatures "0.3,0.5,0.7,0.9,1.0" \
  --resume
```

## Notes

- Generation crashed once during the run but successfully resumed from checkpoint (28k samples)
- The checkpoint system saves every 1,000 samples
- Total wall time: ~72 hours on single GPU (CUDA:2)
- Model used: MedGemma 4B with LoRA (r=32, alpha=64)
