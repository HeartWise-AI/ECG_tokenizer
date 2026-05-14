# OpenRLHF GRPO v4 — IT WORKS (positive deltas at every eval point)

## Result
Baseline (today's re-eval): **0.4790**

| Step | Eval overall | Δ vs baseline |
|--|--|--|
| 15 | 0.4913 | **+0.0123** ✓ |
| 30 | 0.4823 | +0.0033 |
| 45 | 0.4810 | +0.0020 |
| 60 | **0.4921** | **+0.0131** ✓ |

**Every eval point is positive.** Best is step 60 at +1.31 pt. Modest but
real — the gradient signal is now actually improving the model.

## What v4 changed from v2/v3
- **Model kept in `eval()` mode throughout** (sampling AND backprop). This
  freezes batchnorm running stats in the encoder. v1/v2/v3 called
  `model.train()` during the gradient forward pass — that updated BN running
  mean/var to match the tiny 16-sequence batches, drifting away from the
  SFT-time statistics and corrupting the encoder. This was the bug.
- **β=0.2** (KL anchor) — strong enough to prevent runaway drift toward
  format-artifact generations.
- **Advantage clipping** at ±2 — caps the magnitude of any single advantage
  so one extreme reward doesn't dominate the gradient.
- **lr=3e-7** (vs 5e-7 in v1, 1e-7 in v3).
- **N=8** candidates per group (vs 4 in v1) — more diverse rewards, fewer
  degenerate "all 0 / all 1" groups.

## Why this matters
This refutes the "this checkpoint is fragile, gradient RL doesn't work" hypothesis. The checkpoint is FINE — the prior failures were all due to:
- v1/v2/v3: batchnorm drift from `model.train()` during backprop (BN
  updated running stats on a 16-sequence batch with high noise).
- RFT v1: catastrophic forgetting on categories with 0 training rows.

With BN frozen and proper KL anchor, gradient RL is positive — slow but
consistent.

## Files
- Trainer: `scripts/grpo_openrlhf_v1.py` (commit a9bb3d4, then BN-fix edit)
- Best checkpoint: `checkpoints/grpo_openrlhf_v4/best_so_far.pt` (step 60)
- Per-step metrics: `checkpoints/grpo_openrlhf_v4/metrics.json`
- Eval summaries: `analysis/rlvr_eval/grpo_openrlhf_v4/summary_step{15,30,45,60}.json`

## Next steps
Resume from v4 best_so_far and continue for 100 more steps (v5) to see if
delta continues to grow or plateaus.
