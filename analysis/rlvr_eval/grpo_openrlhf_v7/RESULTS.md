# GRPO v7 — Verifiable rewards on 20K focused weak-cat data

## Setup
- Resume from `grpo_openrlhf_v4/best_so_far.pt` (already +1.31pt vs baseline 0.4790)
- Training data: `data/grpo_focused_20k.jsonl` (15,666 rows from 10 weak categories)
- Reward: **per-category verifiable verifier** (`services/verifiable_reward.py`) —
  deterministic, no LLM judge API calls during training
- 200 steps, lr=3e-7, β=0.2, N=8 candidates, batch_size=2, max_new_tokens=96
- Model in `eval()` mode throughout (BN frozen) — the critical bugfix from v4

## Per-step eval on 170-row subset (baseline 0.4790)

| Step | Overall | Δ vs baseline |
|--|--|--|
|  25 | 0.4877 | +0.0087 |
| **50** | **0.4949** | **+0.0159** ← best |
|  75 | 0.4896 | +0.0106 |
| 100 | 0.4807 | +0.0017 |
| 125 | 0.4878 | +0.0088 |
| 175 | 0.4881 | +0.0091 |
| 200 (final) | 0.4861 | +0.0071 |

Best result: **+1.59 pt at step 50**, oscillating around +1pt for the rest of training.

## Per-category at step 50 (best)
- ✓ category_chamber_enlargement: 0.480 → 0.580 (Δ **+0.10**)
- ✓ interpretation: 0.544 → 0.619 (Δ +0.075)
- ✓ category_pericarditis: 0.310 → 0.380 (Δ +0.07)
- ✓ lvef: 0.815 → 0.840 (Δ +0.025)
- ✓ category_infarct_ischemia: 0.610 → 0.620 (Δ +0.01)
- ✗ afib_risk: 0.290 → 0.260 (Δ −0.03)
- ✗ category_other: 0.830 → 0.800 (Δ −0.03)
- (others unchanged)

## What worked
- **Verifiable rewards over LLM-as-a-judge**: training is 5-10x faster (no API),
  fully reproducible, and the signal is clean.
- **BN frozen**: the v4 fix carries through.
- **Focused weak-cat training**: model focuses gradient budget on the right cases.

## What's still hard
- The +10pt target is hard via gradient RL on this checkpoint. Best ~+1.6pt.
- Some categories (`localization_qrs_axis`, `urgency_assessment`, `classification`,
  `culprit_artery`) didn't move at all — either the policy never produces correct
  answers (no positive reward signal) or the verifier never matches even partial
  successes.

## Next steps to push higher
1. **Curriculum**: train ONLY on binary yes/no categories first until those are
   saturated, then add structured-output categories.
2. **Higher LR after warmup**: lr=1e-6 with stronger KL once initial direction
   is set.
3. **Verifier strengthening**: tighten ontology F1 to penalize hallucinated
   findings.
4. **Combine with best-of-N at inference**: GRPO-trained policy + best-of-5
   should compose multiplicatively.

## Files
- Best checkpoint: `checkpoints/grpo_openrlhf_v7/best_so_far.pt` (step 50)
- Per-step metrics: `checkpoints/grpo_openrlhf_v7/metrics.json`
- Training log: `checkpoints/grpo_openrlhf_v7/run.log`
- Per-eval summaries: `analysis/rlvr_eval/grpo_openrlhf_v7/summary_step*.json`
- Verifiers: `services/verifiable_reward.py`
- Focused dataset builder: `scripts/build_grpo_focused_dataset.py`
