# Verifier ↔ Judge Troubleshooting (2026-05)

Validation harness: `scripts/validate_verifier_vs_judge.py` — scores 6,800
real generations (40 judge CSVs) with BOTH the LLM-judge and the
deterministic verifier, reports per-category MAE + correlation.

## Headline diagnostic: the generation script is NOT buggy

Ran the **actual base checkpoint** (`2wjwbk0b_…ENHANCED/best_model.pt`)
with **greedy decoding** on afib/structural/rhythm rows:

```
afib_risk → "Low risk - this patient is unlikely to develop atrial fibrillation in the next 5 years"
afib_risk → "Yes - this patient has a high risk of developing atrial fibrillation within the next 2 years"
```

**Perfectly clean. No `info:`, `---`, `user\n`, `: ` prefixes.**

The prefix junk only appears with `temperature=1.0` sampling (best-of-N
and GRPO rollouts), ~3% of samples. Causal chain:

1. base + greedy = clean ✓ (eval script is correct)
2. GRPO rollouts sample at temp=1.0 → ~3% junk-prefixed samples
3. Early GRPO (v7) verifier gave **full credit** to prefixed samples
4. → model *learned* prefixes are acceptable → later GRPO checkpoints
   (v9) emit prefixes even at greedy → judge penalizes → contributed to
   the GRPO greedy plateau

**Fix**: verifier now strips prefixes before parsing (so it scores the
content like the judge does) + a mild 0.9× nudge so GRPO gently
discourages junk without diverging from the judge.

## Five verifier bugs found & fixed

| Category | Bug (verifier vs judge) | Fix |
|--|--|--|
| category_rhythm | mis-routed to binary Yes/No; GT is free-text ("Sinus rhythm (HR:61)") → verifier 0.00 on perfect matches the judge gave 1.00 | new `verify_rhythm`: Yes/No-gate + findings F1 + HR-accuracy penalty |
| category_conduction / infarct_ischemia / pericarditis / chamber / other | binary verifier only checked the Yes/No gate, ignored findings after "Yes -" → 1.00 where judge gave 0.20 | new `verify_yesno_with_findings`: gate + 0.6·findings-F1 |
| prefix junk (all cats) | parsers (`_canonical_yesno`, `level`, `axis`) returned 0.0 on `info:`/`---`/`user\n` while judge read past them (~0.70) | `_strip_prefix()` applied in every parser |
| culprit_artery | completeness weighted 50% (judge ~30%) | per spec: 0.7·vessel(LAD/RCA/LCx/LeftMain) + 0.3·segment(prox/mid/distal); completeness dropped |
| ecg_interval | ±10% tolerance too loose (42 ms QT error judged 0, verifier 0.5) | tightened to ±5% (min ±8), steeper falloff |
| classification | ontology-F1 ignored the normal/abnormal/borderline class | new `verify_classification`: 3-way class gate + findings F1 |

## Verifier ↔ judge alignment

| Metric | Before | After |
|--|--|--|
| Overall correlation | 0.713 | ~0.78 |
| Mean abs error | 0.157 | ~0.15 |
| Unit tests | 14/14 | 16/16 |

## Intentional residual divergences (NOT bugs)

- **afib_risk** (corr ~0.6): per explicit spec, pure binary — 1.0 iff the
  risk level (low/high/uncertain) matches, else 0.0, nothing else. The
  judge does nuanced scoring (penalizes verbosity, prefix, contradictions)
  so it reads ~0.26 where the binary verifier reads ~0.45. This gap is by
  design — we want a clean verifiable RL signal, not a judge clone.
- **localization_qrs_axis** (corr ~0.34, N small): the model essentially
  never produces this format; judge ≈ 0 always. Verifier tightened
  (direction must be within 25 chars of the word "axis") but a few
  rambling gens still get partial credit.
- **interpretation** (corr ~0.42): free-text; ontology-F1 is inherently
  noisier than the judge's semantic match. Acceptable for an RL reward.

## Why this matters for RLVR

The verifier is the RL reward. Before these fixes it was:
- rewarding prefix junk (→ GRPO learned to emit junk)
- giving full credit for right Yes/No but wrong findings (→ GRPO learned
  to spam the gate word, ignore content)
- too loose on numeric intervals (→ no pressure to be accurate)

After the fixes the reward actually tracks answer quality, which is the
prerequisite for GRPO to improve the model rather than game the metric.

## Files
- `services/verifiable_reward.py` — all verifiers + fixes (16/16 unit tests)
- `scripts/validate_verifier_vs_judge.py` — the validation harness
- `analysis/rlvr_eval/GRPO_EXPERIMENTS_LOG.md` — full run history
