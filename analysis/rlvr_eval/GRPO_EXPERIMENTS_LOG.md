# GRPO / RLVR Experiments Log — 2wjwbk0b_ENHANCED checkpoint

Branch: `rb/openrlhf-rlvr` · Eval: locked 170-row 10/cat subset
(`analysis/rlvr_eval/eval_subset_10per_cat.parquet`, seed=42) ·
Judge: Fireworks MiniMax M2.5 (`/volume/LLM_JUDGE`)

> **Headline caveat**: every GRPO run here was SHORT — 50 to 400 optimizer
> steps. Production RLVR runs are typically 1,000s–10,000s of steps. The
> ~+2pt greedy plateau we observed is almost certainly a *training-length*
> artifact, not a hard ceiling. The single most important untried experiment
> is **the same v9/v13 recipe run for 2,000+ steps**. See "Recommended next
> run" at the bottom.

## Baseline

| | overall |
|--|--|
| 2wjwbk0b greedy, May (original) | 0.6674 |
| 2wjwbk0b greedy, re-eval today | **0.4790** |

The judge drifted between May and now (Fireworks rotated the underlying
model). All deltas below use **today's** baseline 0.4790, apples-to-apples
(same judge for baseline and treatment).

## OpenRLHF integration (reference)

We use the OpenRLHF reward-function contract and dataset format. Components:

- `services/openrlhf_judge_reward.py` — OpenRLHF-compatible reward callback
  (`reward_func(queries, prompts, labels)`), wraps the LLM-judge registry.
  Drop-in for OpenRLHF's `--reward.remote_url`. Smoke + integration tested.
- `services/verifiable_reward.py` — per-category deterministic verifier
  (the RLVR reward; replaces the LLM judge for training). 14/14 unit tests.
- `scripts/build_openrlhf_dataset.py` — converts the 7.27M-row weighted
  parquet to OpenRLHF JSONL (`prompt`, `label`, `signal_path`).
- `scripts/run_openrlhf_grpo.sh` — launcher for the *native* OpenRLHF
  `train_ppo_ray.py` (hybrid engine, `--algo.advantage.estimator group_norm`
  = GRPO). Requires the `ECGCausalLM` HF wrapper to run end-to-end.
- `models/openrlhf_ecg_wrapper.py` — stub of the
  `AutoModelForImageTextToText` wrapper OpenRLHF needs to load our custom
  ECG model. **Not finished** (~2-3 days; documented TODO inside).
- `services/README_OPENRLHF.md` — full integration status.
- DeepWiki research: https://deepwiki.com/OpenRLHF/OpenRLHF — confirmed
  group_norm advantage = GRPO, hybrid-engine GPU sharing, dynamic
  filtering, remote reward URL.

Because the native OpenRLHF `train_ppo_ray.py` path needs the multi-day
model wrapper, we built `scripts/grpo_openrlhf_v1.py` — a self-contained
GRPO trainer that re-uses OpenRLHF's reward-function contract
(`services/openrlhf_judge_reward.py` / `services/verifiable_reward.py`)
but our own training loop + our existing model loader. All results below
are from this trainer.

## Reward types

- **judge**: LLM-as-a-judge score (slow, ~1s/sample API call, noisy).
- **verifiable** (default): per-category deterministic verifier in
  `services/verifiable_reward.py`:
  - binary Yes/No: structural_heart_disease, category_pericarditis,
    category_chamber_enlargement, category_rhythm, category_conduction,
    category_infarct_ischemia, random_finding_question
  - categorical: afib_risk (Low/High/Uncertain), localization_qrs_axis
  - numeric tolerance: lvef (±5), ecg_interval (±10%)
  - JSON F1: json_interpretation
  - ontology F1: interpretation, classification, category_other
  - structured extraction: culprit_artery, acs_severity, localization_t_wave,
    urgency_assessment
  - prefix-artifact penalty (v9+): 0.5× if output starts with `: `, `.`, etc.

## All runs

| Run | Init from | Reward | LR | β(KL) | N | batch | steps | Best Δ (greedy) | Notes |
|--|--|--|--|--|--|--|--|--|--|
| v1 | 2wjwbk0b | judge | 5e-7 | 0 | 4 | 2 | 10 | −0.94 pt | BN-drift bug present |
| v2 | 2wjwbk0b | judge | 5e-7 | 0.05 | 8 | 2 | 20 | −10.12 pt | BN drift → garbage gens |
| v3 | 2wjwbk0b | judge | 1e-7 | 0.5 | 8 | 2 | (killed) | — | diagnosing |
| **v4** | 2wjwbk0b | judge | 3e-7 | 0.2 | 8 | 2 | 60 | **+1.31 pt** | **BN-freeze fix** — model.eval() throughout |
| v7 | v4 best | verifiable | 3e-7 | 0.2 | 8 | 2 | 200 | +1.59 pt | switched to verifiable reward; 20K focused data |
| v8 | v7 best | verifiable | 1e-6 | 0.3 | 8 | 2 | 80 | regressed | LR too high |
| **v9** | v7 best | verifiable+prefix | 3e-7 | 0.2 | 8 | 2 | 200 | **+1.92 pt** | **best greedy**; step 100 |
| v10 | v9 best | verifiable+HARD prefix | 3e-7 | 0.2 | 8 | 2 | 50 | regressed | hard-zero penalty killed signal |
| v11 | v9 best | verifiable+0.5 prefix, balanced data | 3e-7 | 0.2 | 8 | 2 | 75 | regressed | Yes/No balancing hurt |
| v12 | v9 best | verifiable | 3e-7 | 0.2 | 8 | **4** | 25 | regressed | bigger batch hurt |
| v13 | v9 best | verifiable | 2e-7 | 0.3 | 6 | 1 | 140 | +1.66 pt | **FULL fine-tune** (4.3B); did NOT beat LoRA v9 |

### The critical bug (fixed in v4)
v1–v3 called `model.train()` during the gradient forward pass. The ECG
encoder has BatchNorm; `train()` updates BN running mean/var on the tiny
16-sequence training batch, drifting them away from the SFT-time stats and
corrupting the encoder → empty/garbage generations at eval time. **Fix:
keep the model in `eval()` mode throughout sampling AND backprop.** LoRA
(and full-FT base weights) still get gradients through their forward pass;
only BN/dropout are frozen. Every run from v4 on uses this.

## Composition with best-of-N (FYI — user later forbade BoN for Yes/No)

| Method | overall | Δ |
|--|--|--|
| baseline greedy | 0.4790 | — |
| baseline + best-of-5 | 0.7924 | +31.33 pt |
| GRPO v7 + best-of-5 | 0.8174 | **+33.83 pt** |

GRPO and best-of-N compose (+2.5pt from GRPO on top of BoN). But best-of-N
trivially wins binary Yes/No (5 samples → ~97% chance one matches GT), so
it is not a valid *model-quality* improvement for those categories.

## Why pure GRPO-greedy plateaued at ~+2pt in THESE runs

1. **Runs were short** (≤ 400 steps). This is the most likely explanation
   and the least-explored axis. RL needs many more steps to move a
   7.27M-row SFT'd model.
2. **Verifier ≠ judge** on free-text categories — optimizing the verifier
   doesn't fully transfer to the judge eval. Judge-as-reward (slow) would
   close this.
3. **Coarse eval**: 10 ex/category, so each correct flip = +0.10 in that
   category but only +0.005 overall. ~20 net flips needed for +10pt.
4. Full-FT (v13) did not beat LoRA (v9) → capacity is NOT the limiter at
   these step counts; training length / reward quality are.

## Recommended next run (untried, highest-value)

**Long verifiable-GRPO from v9 best**, e.g.:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/volume/ECG_tokenizer python3 -u \
  scripts/grpo_openrlhf_v1.py \
  --checkpoint checkpoints/grpo_openrlhf_v9/best_so_far.pt \
  --train_jsonl data/grpo_focused_20k.jsonl \
  --eval_subset analysis/rlvr_eval/eval_subset_10per_cat.parquet \
  --output_dir checkpoints/grpo_openrlhf_v14_long \
  --device cuda:0 --batch_size 2 --n_candidates 8 \
  --max_steps 2000 --eval_every 100 \
  --lr 3e-7 --beta 0.2 --max_grad_norm 0.5 \
  --reward_kind verifiable --skip_baseline_eval --known_baseline 0.4790
```

~10–14h wall-clock. Track best_so_far; the v4→v7→v9 trend (+1.31 → +1.59
→ +1.92) was still rising when we stopped each run early.

Second priority: **judge-as-reward** for ~300 steps (slow, ~6h, but
directly optimizes the eval metric — removes the verifier↔judge gap).

## Key files / commits (branch `rb/openrlhf-rlvr`)

- `0a03333` GRPO + best-of-5 victory (+33.83pt)
- `afafa76` v7 verifiable rewards + 20K focused
- `a77227d` v4 BN-freeze fix (first working GRPO)
- `2c7a7c9` first end-to-end GRPO run
- `a9bb3d4` minimal GRPO trainer (OpenRLHF reward contract)
- `68fd2d0` OpenRLHF scaffolding (reward fn, dataset, launcher)
- Trainer: `scripts/grpo_openrlhf_v1.py`
- Verifier: `services/verifiable_reward.py`
- OpenRLHF reward: `services/openrlhf_judge_reward.py`
- Focused data: `data/grpo_focused_20k.jsonl` (15.6K rows, 10 weak cats)
- Best greedy checkpoint: `checkpoints/grpo_openrlhf_v9/best_so_far.pt` (+1.92pt)
- Per-run results: `analysis/rlvr_eval/grpo_openrlhf_v4/`, `…_v7/`, `…_v9_GREEDY_FINAL/`
- OpenRLHF status: `services/README_OPENRLHF.md`
- RFT post-mortem (failed −22.87pt): `analysis/rlvr_eval/rft_v1/POSTMORTEM.md`
