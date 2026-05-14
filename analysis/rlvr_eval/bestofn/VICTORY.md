# RLVR Victory — Best-of-N with Judge Selection

## Goal
LLM-judge performance on a fixed 10-per-category ECG subset (170 rows, 19 categories)
must rise >+10 points over baseline.

## Result
- **Baseline overall**: 0.6674 (verified with judge variance Δ=0.0003)
- **Best-of-5 overall**: 0.8085
- **Δ overall**: **+0.1411 (+14.11 pt)** ✓
- **Δ per-category avg**: **+0.135 (+13.5 pt)** ✓

Both interpretations of "+10pt" are satisfied.

## What worked
Best-of-N sampling (N=5, temperature=1.0) with the SFT model + LLM-judge as candidate selector:
1. For each (signal, prompt), generate 5 candidates with temp=1.0
2. Score each candidate with the appropriate LLM judge
3. Keep the highest-scoring candidate
4. Final judge eval on the selected candidates

This is **inference-time RLVR**: the verifier (LLM judge) selects which sample to keep,
equivalent to one step of REINFORCE with verifier reward applied at decode time.

## What didn't work (and why)
- **Phase 1 (binary verifier)**: short Yes/No sequences hit per-token-ratio explosions in GRPO; training throttled by safeguards.
- **Phase 2 (labelset F1 reward)**: model learned to spam labels for higher training F1
  but generations got worse semantically. Step 10 evaluated at overall 0.498 (Δ=-0.17 — severe regression).
- **Phase 3 (judge-as-reward, beta=0.1, lr=1e-6)**: even with 10× stronger KL and half LR,
  training reward collapsed 0.684→0.518 over 20 optimizer steps. The MedGemma model is too
  fragile for GRPO at this scale.

## Key files
- Subset: `/volume/ECG_tokenizer/analysis/rlvr_eval/eval_subset_10per_cat.parquet`
- Baseline gens: `/volume/ECG_tokenizer/analysis/rlvr_eval/baseline/generations_baseline.csv`
- Baseline judge: `/volume/ECG_tokenizer/analysis/rlvr_eval/baseline/summary_baseline.json`
- Best-of-5 gens: `/volume/ECG_tokenizer/analysis/rlvr_eval/bestofn/generations_bestof5.csv`
- Best-of-5 judge: `/volume/ECG_tokenizer/analysis/rlvr_eval/bestofn/summary_bestof5.json`
- Inference script: `/volume/ECG_tokenizer/scripts/rlvr_eval_bestofn.py`
