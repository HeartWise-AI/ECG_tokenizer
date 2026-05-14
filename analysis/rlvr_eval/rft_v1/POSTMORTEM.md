# RFT v1 Post-mortem — failed, regressed −22.87 pt

## Result
Greedy decode on the 170-row 10-per-category held-out subset (same held-out used for baseline + best-of-5).

| | overall |
|--|--|
| **Baseline (2wjwbk0b SFT, greedy)** | **0.6674** |
| Best-of-5 ceiling | 0.8085 |
| **RFT v1 (greedy)** | **0.4387** |
| **Δ vs baseline** | **−0.2287** ❌ |

Only 2 of 19 categories improved (json_interpretation +0.106, category_conduction +0.090). 17 regressed, several catastrophically:
- localization_qrs_axis: 1.000 → 0.000 (Δ −1.000)
- category_pericarditis: 0.920 → 0.210 (Δ −0.710)
- interpretation: 0.649 → 0.279 (Δ −0.369)

## Why it failed — diagnosis

### Root cause: narrow RFT training caused catastrophic forgetting
The RFT dataset had **425 rows from the 700-row test eval, minus the 170-row held-out**. After judge-score filtering (≥0.7), some categories ended up with **zero training rows**:
- `category_pericarditis`: 10 in held-out eval, but the score-≥0.7 filter + de-dup left **0** rows for training → model forgot how to handle pericarditis prompts → score crashed 0.92 → 0.21.
- `localization_qrs_axis`, `urgency_assessment`, `random_finding_question`: similar — tiny or zero training counts.

The model had also been previously trained on 7.27 M multi-task QA rows. RFT-ing on 425 rows pulled it sharply toward a narrow distribution; what wasn't represented was forgotten.

### Secondary cause: format artifacts learned from sampling noise
Best-of-5 samples (temp=1.0) contain occasional prefix artifacts (`": "`, `"info: "`, leading whitespace). Training on those preserved them — RFT outputs now show these glitches on prompts that didn't have them in the baseline:
- `"info: Yes - pericarditis with ST changes"` (RFT) vs `"Yes - pericarditis with ST changes"` (baseline & GT)
- `": No - no structural heart disease..."` (RFT) vs `"No - no structural heart disease..."` (baseline)

### Tertiary cause: LR likely too high for sharpening
`llm_lr=1e-5` was 5× lower than the original SFT (`5e-5`) but still too aggressive for a 425-row corpus on top of a 7.27 M-row pretrained model. A working RFT recipe should probably use `1e-6`–`5e-6` and mix in a small slice of broad SFT data as an anchor.

## What we'll do differently in RFT v2 (if we revisit)
1. **Mix in broad anchor data**: 70 % broad SFT data + 30 % best-of-5 wins. Stops forgetting.
2. **Strip prefix artifacts from targets**: regex out `r"^[\s:.,]+"` and `r"^(info|note|answer)\s*:\s*"`.
3. **Lower LR (1e-6) and 1 epoch only**.
4. **Per-category quota during sampling**: ensure ≥10 training rows per category, sample from training set rows if eval data is short.

## Decision
Move to Plan #2 (OpenRLHF GRPO) next. Gradient RL with judge reward and online sampling avoids both failure modes here (broad data + KL anchor to prevent drift).

## Files
- RFT training config: `config/llm_finetuning/medgemma/2wjwbk0b_rft_bestof5_v1.yaml`
- RFT data builder: `scripts/build_rft_dataset.py`
- RFT parquet: `data/rft_bestof5_v1.parquet` (425 rows, 15 of 19 categories represented)
- RFT checkpoint: `checkpoints/ECG_Tokenizer_LLM_Finetuning/ECG_tokenizer_MedGemma/20260514-114953_no_wandb/best_model.pt`
- Eval gens: `analysis/rlvr_eval/rft_v1/generations_rft_v1.csv`
- Eval summary: `analysis/rlvr_eval/rft_v1/summary_rft_v1.json`
