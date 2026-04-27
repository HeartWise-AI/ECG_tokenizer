# 2wjwbk0b → HEARTS-max — Deliverables

Branch: `rb/2wjwbk0b-hearts-max` (off `rb/e4d-champion-iter`).
Goal: maximise the HEARTS `mhi_ecg` composite (mean of 18 normalised
per-task scores in `[0, 1]`) starting from `2wjwbk0b_20260413-224103`,
without regressing any per-task score by more than 5 absolute points.

## Loop parameters (LOCKED)

- **Reference checkpoint**: `2wjwbk0b_20260413-224103_ENHANCED/best_model.pt`
- **Iteration training budget**: `num_epochs: 1` (fast). Final promotion
  step uses `num_epochs: 3`.
- **GPUs**: `0,1` (refuse to start if either is busy).
- **HEARTS eval set per iteration**: 100 samples per task × 18 tasks (see
  `score_hearts_all.py`).
- **Per-category regression tolerance**: 0.05 (absolute).
- **Convergence**: 3 consecutive `kept` with composite delta < 0.005, or
  5 consecutive `reverted`.

---

## Phase 0 — lock the baseline

- [x] **0.1 Build HEARTS fixtures (once)**
  - From `/volume/HEARTS`, run:
    `uv run python scripts/build_mhi_ecg_fixtures.py --max 100 --out-dir /volume/HEARTS/fix_test_cases`
  - Acceptance: 18 task directories under `fix_test_cases/mhi_ecg/`, each
    with ≥ 50 pickles (`heart_rate`, `interpretation`, etc. should have
    100; rare tasks like `culprit_artery` may have fewer).

- [x] **0.2 Score the unmodified 2wjwbk0b on HEARTS**
  - Run `ralph/2wjwbk0b/score_hearts_all.py` against
    `checkpoints/BEST_LLM/2wjwbk0b_20260413-224103_ENHANCED/best_model.pt`.
  - Save output to `ralph/2wjwbk0b/BASELINE.json`. Also copy to
    `ralph/2wjwbk0b/ACTIVE_BASELINE.json`.
  - Write the absolute checkpoint path to
    `ralph/2wjwbk0b/ACTIVE_CHECKPOINT.txt`.
  - Seed `ralph/2wjwbk0b/results.tsv` with the header row plus a
    `0\tbaseline\t-\t<composite>\t0.000\t0.000\tbaseline\t<path>\t<sha>`.
  - Acceptance: `BASELINE.json` exists with `composite` between 0 and 1
    and `per_task_score` containing all 18 keys.

---

## Phase 1 — learning-rate fine-tunes

Each item swaps **one** parameter relative to the active reference config.
Apply the change inside the experiment's config copy only.

- [ ] **1.1 lower llm_lr 5e-5 → 2e-5**
- [ ] **1.2 lower llm_lr 5e-5 → 1e-5**
- [ ] **1.3 raise adapter_lr 5e-4 → 1e-3**
- [ ] **1.4 lower adapter_lr 5e-4 → 2e-4**
- [ ] **1.5 short warmup: num_warmup_percent 0.256 → 0.05**
- [ ] **1.6 constant LR: scheduler_type cosine_with_warmup → constant**

## Phase 2 — LoRA capacity

- [ ] **2.1 lora_r 32 → 64, lora_alpha 64 → 128**
- [ ] **2.2 lora_r 32 → 16, lora_alpha 64 → 32**
- [ ] **2.3 lora_top_k_layers 12 → 18**
- [ ] **2.4 lora_dropout 0.05 → 0.0**

## Phase 3 — generation defaults (eval-only, no training)

For these, skip the training step. Build the experiment config by copying
the active config and only mutating `default_generation_kwargs`. Re-score
on the **active checkpoint** (no new checkpoint produced).

- [ ] **3.1 max_new_tokens 96 → 64**
- [ ] **3.2 max_new_tokens 96 → 128**
- [ ] **3.3 repetition_penalty 1.02 → 1.10**
- [ ] **3.4 no_repeat_ngram_size 5 → 0**
- [ ] **3.5 do_sample false → true, temperature 0.7**

## Phase 4 — auxiliary loss balance

- [ ] **4.1 pattern_loss_weight 0.3 → 0.5**
- [ ] **4.2 pattern_loss_weight 0.3 → 0.1**
- [ ] **4.3 pattern_loss_weight 0.3 → 0.0** (does the auxiliary signal
  still help at this stage?)
- [ ] **4.4 instruction_dropout 0.1 → 0.05**
- [ ] **4.5 instruction_dropout 0.1 → 0.2**

## Phase 5 — bridge regularisation

- [ ] **5.1 bridge_codebook_dropout 0.0 → 0.1**
- [ ] **5.2 bridge_dropout 0.1 → 0.2**
- [ ] **5.3 bridge_dropout 0.1 → 0.05**

## Phase 6 — data-mix re-weighting (target the weakest categories)

For each item, identify the **lowest-scoring category** in the active
baseline, then upsample that category's training rows by 2× via the
existing weighted sampler. The slug must include the targeted category.

- [ ] **6.1 boost lowest task** (slug `boost_<task>_2x`)
- [ ] **6.2 boost lowest 3 tasks**
- [ ] **6.3 boost only `interpretation` 1.5×** (it dominates the parquet but
  is also the most-evaluated free-text task — may help phrase alignment)
- [ ] **6.4 add prompt-variation augmentation** (mix in
  `output/combined_train_qa_m200k_h200k_prompt_variations.parquet` if
  available; otherwise skip with a clear note)

## Phase 7 — combine the best changes

Pick the best `K` kept experiments from Phases 1–6 by `delta_vs_active`,
stack them in one config, and run.

- [ ] **7.1 stack top-2 changes**
- [ ] **7.2 stack top-3 changes**
- [ ] **7.3 stack top-5 changes** (only if 7.2 still improves)

## Phase 8 — promotion (long run)

- [ ] **8.1 retrain stacked-best for `num_epochs: 3`**
  - Use the config from the best Phase-7 stack but raise epochs to 3.
  - Acceptance: composite ≥ best Phase-7 composite, no per-task regression
    > 5 absolute points.

- [ ] **8.2 promote**
  - On success of 8.1, copy the new checkpoint to
    `checkpoints/BEST_LLM/2wjwbk0b_HEARTSMAX_<utc>/best_model.pt` (mirror
    the directory layout used for `2wjwbk0b_20260413-224103_ENHANCED/`).
  - Write `ralph/2wjwbk0b/PROMOTION.md` with a delta table per task and
    the final composite.
  - Open a PR from `rb/2wjwbk0b-hearts-max` → `main` titled
    `feat: HEARTS-max champion (was 2wjwbk0b)`.

---

## Definition of done

Either:
- (a) Phase 8.2 promotion happened, with `PROMOTION.md` written; OR
- (b) `CONVERGED.md` written because no further gain was found across
  Phases 1–7.

Keep the experiments deterministic (`seed: 42`), keep the eval set frozen,
and **never lower the regression-tolerance bar** to make a borderline
experiment look like a kept improvement.
