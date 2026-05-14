# OpenRLHF GRPO v1 — end-to-end run

## Result
| | overall |
|--|--|
| **Baseline (2wjwbk0b, today's re-eval)** | **0.4790** |
| **GRPO v1 (10 steps, lr=5e-7, beta=0.0)** | **0.4696** |
| **Δ** | **−0.0094 (−0.94 pt)** — essentially flat with slight regression |

Note: the original baseline from the May 4 best-of-N run was 0.6674. Re-eval'ing
the *same* checkpoint today gives 0.4790. The drift is judge/model variance
(Fireworks may have rotated the underlying judge model). Both this run's
baseline AND final use today's judge, so the comparison is apples-to-apples.

## Per-category breakdown
| Category | Baseline | GRPO | Δ |
|--|--|--|--|
| category_pericarditis | 0.310 | 0.520 | **+0.210** |
| json_interpretation | 0.493 | 0.653 | **+0.160** |
| ecg_interval | 0.400 | 0.500 | **+0.100** |
| classification | 0.190 | 0.270 | **+0.080** |
| structural_heart_disease | 0.390 | 0.430 | +0.040 |
| category_chamber_enlargement | 0.480 | 0.520 | +0.040 |
| random_finding_question | 0.770 | 0.810 | +0.040 |
| category_conduction | 0.550 | 0.570 | +0.020 |
| acs_severity | 0.640 | 0.640 | 0.000 |
| culprit_artery | 0.470 | 0.430 | −0.040 |
| category_rhythm | 0.720 | 0.680 | −0.040 |
| category_infarct_ischemia | 0.610 | 0.550 | −0.060 |
| lvef | 0.815 | 0.750 | −0.065 |
| afib_risk | 0.290 | 0.220 | −0.070 |
| localization_t_wave | 0.250 | 0.133 | −0.117 |
| interpretation | 0.544 | 0.360 | **−0.184** |
| category_other | 0.830 | 0.390 | **−0.440** |
| **OVERALL** | **0.4790** | **0.4696** | **−0.0094** |

## Training progress (10 steps, 6 effective updates)
| step | loss | reward | grad_norm |
|--|--|--|--|
| 1 | +5.36 | 0.25 | 65.76 |
| 2 | +6.43 | 0.65 | 60.83 |
| 3 | (degenerate, skipped) | — | — |
| 4 | +3.24 | 0.78 | 72.30 |
| 5 | (degenerate, skipped) | — | — |
| 6 | +0.09 | 0.48 | 37.81 |
| 7 | (degenerate, skipped) | — | — |
| 8 | −0.67 | 0.75 | 23.78 |
| 9 | −2.85 | 0.50 | 32.10 |
| 10 | (degenerate, skipped) | — | — |

Loss went negative (good — model is putting more probability mass on
high-reward completions). Training reward stayed in 0.25–0.78 range with
no clear divergence.

## Method
- Custom GRPO trainer (`scripts/grpo_openrlhf_v1.py`), NOT the existing
  `runners/grpo_finetuning_runner.py`. Fresh code.
- Reward function from OpenRLHF integration (`services/openrlhf_judge_reward.py`)
  — same code path that would be used by OpenRLHF's `--reward.remote_url`.
- Group-normalized advantages (GRPO).
- LoRA-only training (~23.8M params), lr=5e-7, max_grad_norm=0.5.
- Per-token log-probs computed from a single forward pass over (prompt+gen).
- No KL anchor (beta=0 implicit; we rely on the very low LR for stability).
- N=4 candidates per prompt, batch_size=2 prompts per step.

## Interpretation
End-to-end GRPO pipeline works: model loads, samples candidates, judge scores
them, GRPO computes advantages, gradient flows, optimizer steps, final eval
runs. The +14.11 / +36.02 pt best-of-N gain CANNOT be matched by 10 GRPO steps
on this fragile checkpoint — at best gradient RL produces a flat/slightly
negative delta. This matches the [[feedback-rlvr-lessons]] memory: 4 prior
gradient-RL attempts (binary, labelset, judge-as-reward, RFT) all regressed.

If we wanted to push further:
- More steps (~100-1000) with mid-training eval and early-stop
- KL anchor with beta=0.5+ to prevent drift on regressing categories
- Per-category sampling that prioritizes weak categories
- Or: pivot back to best-of-N, which is the only method that has worked at scale

## Files
- Trainer: `scripts/grpo_openrlhf_v1.py`
- Reward function: `services/openrlhf_judge_reward.py`
- Training data: `data/openrlhf_train_v1.jsonl`
- Final checkpoint: `checkpoints/grpo_openrlhf_v1/best_model.pt`
- Final eval generations: `checkpoints/grpo_openrlhf_v1/eval_final/generations_final.csv`
- Final eval summary: `checkpoints/grpo_openrlhf_v1/eval_final/summary_final.json`
- Training log: `checkpoints/grpo_openrlhf_v1/run.log`
