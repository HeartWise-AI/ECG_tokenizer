# e4d Champion Iteration — Deliverables

Branch: `rb/e4d-champion-iter` (based off `rb/grpo_finetuning`).
Goal: ship a new champion that improves global composite without regressing
the current champion's strong categories.

**Gating rule:** Phase `-1` (HEARTS multi-task training + evaluation) must be
fully green before any Phase 0+ experiment fires. The champion iteration loop
exits early with `⚠ BLOCKED: Phase -1 incomplete` if any `-1.x` item is
unchecked.

## ⚠ Must be filled in before the loop fires

- **`TRAINING_BUDGET`**: TBD — the user's message was cut off ("train maybe
  for thr..."). Candidates:
  - `num_epochs: 3` (3 epochs of the 400k weighted train set ≈ several hours
    per run on 2× H100)
  - `max_train_hours: 3`
  - `max_train_steps: 3000`
  - something else
  Until this is set, **do not run experiments 1+**. Experiment 0 (baseline
  recomputation) is safe to run regardless.

- **`GPUS`**: which GPU indices can this loop use? Default assumption below
  is `0,1`. Update before launch.

- **Composite weights**: inherit from `autoresearch/prompt.md`:
  `composite = -0.3*loss + 0.25*rougeL + 0.15*meteor + 0.15*bleu4 + 0.15*bertscore_f1`.
  Confirm this is still the right rollup for a fine-tune scenario — a
  fine-tune at low LR may show near-zero loss change, so text metrics
  dominate naturally. Keep the formula for comparability.

---

## Phase -1 — HEARTS multi-task training + eval (GATING)

This phase makes the champion HEARTS-capable before we iterate it. Until this
phase is green, the iteration loop does not run.

### Dependency on the HEARTS side

The HEARTS repo's ralph loop (`/volume/HEARTS/ralph/DELIVERABLES.md`) must
have completed **Phase 0 (agent+base)**, **Phase 1 (the 18 mhi_ecg tasks)**
and **Phase 2 (frozen test fixtures)** before this phase can run. If it
hasn't, stop with `⚠ BLOCKED: HEARTS phases 0–2 incomplete` and point the
user at the HEARTS loop.

### Scope decision — which HEARTS tasks is the model in-scope for?

The ECG_tokenizer+MedGemma model speaks **12-lead 10s ECG** as its
non-text modality. HEARTS has 110 tasks across 16 modalities (CGM, EEG,
audio, gaze, EMG, accelerometer, ECG, …). A single model cannot meaningfully
do them all without per-modality tokenizers.

- [ ] **-1.0 Write `ralph/HEARTS_SCOPE.md`** defining scope tiers:
  - **Tier A — native** (ECG modality, 12-lead feasible): the 18 new
    `mhi_ecg` HEARTS tasks (interpretation, classification, localization,
    LVEF, AFib risk, ACS, culprit artery, SHD, rhythm/conduction/ischemia
    categories, heart_rate, qrs_axis, age_gender, json_interpretation).
  - **Tier B — adapted** (long-duration single-lead ECG via sliding-window
    adapter): `shhs_remote/af_classification`,
    `shhs_remote/cvd_death_prediction`, `shhs_remote/stroke_prediction`,
    `shhs_remote/smoker_classification`, `shhs_remote/bmi_comparison`, the
    three `time_irrv_*` tasks. Requires the window adapter from the HEARTS
    Phase 5 stretch items.
  - **Tier C — out-of-scope** (non-ECG modalities): everything else —
    CGM, EEG, EOG, audio, gaze, EMG, PPG-only, accelerometer. These are
    NOT trained nor evaluated in this phase. Document the reason.
  - User must confirm Tier A + Tier B scope before proceeding. If Tier B is
    dropped, adjust experiments accordingly.

### Training data assembly

- [ ] **-1.1 Build HEARTS training parquet**
  - For each Tier A task: pull the corresponding rows from the existing QA
    parquets (`output/combined_train_qa_m200k_h200k.parquet`) that match
    the `prompt_category` of the task.
  - For each Tier B task (if in scope): generate prompts on the fly from the
    HEARTS training data (not the test fixtures) using the sliding-window
    adapter. Save a new parquet per task to
    `output/hearts_training/{task}.parquet`.
  - Merge all into one weighted parquet `output/hearts_multitask_train.parquet`
    with a `task_id` column and a category distribution matching roughly
    equal per-task sampling (inverse-frequency weights).
  - Acceptance: parquet row count ≥ 100k, all task_ids represented.

- [ ] **-1.2 Build HEARTS validation parquet**
  - Same tasks, but drawn from `output/combined_test_qa_m5k_h5k.parquet` for
    Tier A and from held-out HEARTS samples for Tier B. **Must be disjoint
    from both the training parquet AND the HEARTS frozen test fixtures.**
  - Save to `output/hearts_multitask_val.parquet`.

### Training run

- [ ] **-1.3 Create `config/llm_finetuning/medgemma/champion_hearts_multitask.yaml`**
  - Base: the champion config.
  - Changes:
    - `train_dataset_path: output/hearts_multitask_train.parquet`
    - `validation_dataset_path: output/hearts_multitask_val.parquet`
    - `resume_from_checkpoint: <champion best_model.pt>`
    - `num_epochs: <TRAINING_BUDGET>` (still TBD from user — see top of file)
    - `use_wandb: true`, run name `champion_hearts_multitask_e<N>`
    - Larger `max_token_length` if Tier B tasks need longer prompts.
  - Acceptance: config loads, dry-run data path validation passes.

- [ ] **-1.4 Run training**
  - Launch: `bash scripts/runner.sh --base_config config/llm_finetuning/medgemma/champion_hearts_multitask.yaml --selected_gpus <GPUS> --use_wandb true --run_mode train --instruct_mode true`
  - Redirect logs to `ralph/hearts_multitask/train.log`.
  - Save final checkpoint path to `ralph/hearts_multitask/checkpoint_path.txt`.
  - Acceptance: training completes without divergence; val loss monotone or
    at least non-increasing over the last 20% of steps.

### HEARTS benchmark evaluation

- [ ] **-1.5 Wire the trained checkpoint into the HEARTS agent**
  - Update `/volume/HEARTS/config/mhi_ecg.yaml.example` (written by the
    HEARTS ralph loop in Phase 3) to point at the new
    `checkpoint_path.txt` checkpoint. Do not edit HEARTS code from this
    loop — only the config copy.

- [ ] **-1.6 Run all Tier A HEARTS tasks via HEARTS frozen eval**
  - From `/volume/HEARTS`:
    `uv run run_exp_freeze.py --fix-test-cases-dir <dir>
    --task <each of the 18 mhi_ecg tasks> --model-name medgemma-ecg-qformer-8cb
    --num-test 500 --n-jobs 4`
  - Results JSONs land in `/volume/HEARTS/results/`. Symlink or copy them
    into `ralph/hearts_multitask/results/` for record-keeping.

- [ ] **-1.7 (Optional, Tier B) Run `shhs_remote` ECG tasks**
  - Only if Tier B was kept in `-1.0` AND the HEARTS Phase 5.1 window
    adapter exists. Same command pattern, different task names.

### Gate decision

- [ ] **-1.8 Compute HEARTS-capable baseline**
  - Aggregate per-task metrics from `-1.6` (and `-1.7`) into
    `ralph/HEARTS_BASELINE.json`. Structure mirrors `CHAMPION_BASELINE.json`
    but adds a `per_hearts_task` section keyed by `(dataset, task)`.
  - Compare against Tier A's pre-training performance (can be reconstructed
    by running the champion checkpoint through the same HEARTS harness
    BEFORE the multitask fine-tune — do this in a sub-step and store as
    `ralph/HEARTS_CHAMPION_PRE.json`).
  - **Gating criterion to unlock Phase 0+:** average Tier A ROUGE-L (or
    accuracy for classification tasks) must improve by ≥ 3 absolute points
    vs `HEARTS_CHAMPION_PRE.json`, AND no Tier A task regresses by > 2
    absolute points. If the gate fails, surface `⚠ GATE FAILED:` with the
    exact deltas, do not proceed.

- [ ] **-1.9 Promote HEARTS-capable champion**
  - If the gate passes, this trained checkpoint becomes the new reference
    for the champion iteration loop. Update the top of this file to point
    Phase 0.1 at the new checkpoint.
  - Commit: `feat(ralph): promote hearts_multitask checkpoint as iteration base`.

---

## Phase 0 — lock the baseline

- [ ] **0.1 Recompute champion baseline on locked eval set**
  - Run `inference/generate_all_qa_pairs.py` with the champion checkpoint
    against `output/combined_test_qa_m5k_h5k.parquet`, all 10k samples.
  - Save to `ralph/CHAMPION_BASELINE.json` with:
    - overall: {loss, rougeL, bleu4, meteor, bertscore_f1, composite}
    - per_category: {cat -> {rougeL, n_samples}}  (uses the 18 categories)
    - per_dataset: {MIMIC, MHI}
  - Also save raw generations to `ralph/runs/exp_000_baseline/generations.json`.
  - Acceptance: `CHAMPION_BASELINE.json` exists and loads.
  - ⚠ This file is the immovable reference for every subsequent experiment.
    After this, the loop MUST NOT overwrite it.

- [ ] **0.2 Seed `ralph/results.tsv` with the baseline row**
  - Header: `exp_num\tslug\tchange\tcomposite\trougeL\tbleu4\tmeteor\tbertscore_f1\tworst_category_delta\tdecision\tcommit_sha`
  - Row 0: `000\tbaseline\tchampion recomputed\t<values>\t<values>\t...\t0.0\tbaseline\t<SHA>`

---

## Phase 1 — learning-rate + optimizer fine-tune sweeps

Focus: small nudges around the champion's working point. Because we're
warm-starting, effects are smaller than from-scratch training.

- [ ] **1.1 Lower adapter_lr to 1e-4** (from champion value)
  - Rationale: fine-tune should adapt the bridge gently.
  - Change: `adapter_lr: 2e-4 -> 1e-4`.

- [ ] **1.2 Lower llm_lr to 2e-6** (from champion value)
  - Rationale: protect LLM from drift during short fine-tune.
  - Change: `llm_lr: 5e-6 -> 2e-6`.

- [ ] **1.3 Raise adapter_lr to 4e-4 with short warmup**
  - Rationale: test whether bridge has headroom that the champion left.
  - Change: `adapter_lr: 2e-4 -> 4e-4`, `num_warmup_percent: 0.03 -> 0.02`.

- [ ] **1.4 Disable LLM LR (adapter-only refresh)**
  - Rationale: isolate whether remaining gains are bridge-side.
  - Change: `llm_lr: 5e-6 -> 0.0`.

- [ ] **1.5 Constant LR instead of cosine**
  - Change: `scheduler_type: cosine_with_warmup -> constant`.

---

## Phase 2 — LoRA capacity

- [ ] **2.1 Increase lora_r to 64, alpha to 128**
  - Change: `lora_r: 32 -> 64`, `lora_alpha: 64 -> 128`.

- [ ] **2.2 Reduce lora_r to 16 (does champion over-parameterise?)**
  - Change: `lora_r: 32 -> 16`, `lora_alpha: 64 -> 32`.

- [ ] **2.3 Expand target modules**
  - Current: q/k/v/o/gate/up/down. Add nothing new (already max coverage).
  - Instead: `lora_top_k_layers: 12 -> 18`.

- [ ] **2.4 Lower lora_dropout to 0.0**
  - Change: `lora_dropout: 0.05 -> 0.0`.

---

## Phase 3 — generation + decoding

These don't require training — they only change eval. They should be fast
sanity checks, not full retrain iterations. Mark `⚠ EVAL-ONLY`.

- [ ] **3.1 (⚠ EVAL-ONLY) repetition_penalty 1.02 -> 1.10**
  - Re-run inference only; skip training step.
- [ ] **3.2 (⚠ EVAL-ONLY) max_new_tokens 32 -> 56**
  - Let interpretation / json_interpretation outputs breathe.
- [ ] **3.3 (⚠ EVAL-ONLY) do_sample: false, num_beams 1 -> 4**
  - Beam search on the locked eval.

For eval-only items, the loop skips training, runs inference with the
modified `default_generation_kwargs`, compares, decides.

---

## Phase 4 — data mixture (⚠ CODE-CHANGE for some)

- [ ] **4.1 Reweight category mix toward weak categories**
  - Identify the 3 worst per-category ROUGE-L in `CHAMPION_BASELINE.json`.
  - Upsample those categories 2× in the training parquet via a sampler
    weight file. If this requires runner changes, mark `⚠ CODE-CHANGE`.

- [ ] **4.2 Add prompt-variation augmentation**
  - Use `dataset_generation/generate_prompt_answer_variations.py` to
    generate an alt-phrasings parquet; mix 20% into training.

- [ ] **4.3 Add CoT variant for interpretation**
  - Use `config/llm_finetuning/medgemma/cot_sft.yaml` as a reference and
    mix 10% CoT samples for the interpretation category only.

---

## Phase 5 — pattern loss + auxiliary signal

- [ ] **5.1 Raise pattern_loss_weight to 0.5**
  - Change: `pattern_loss_weight: 0.3 -> 0.5`.

- [ ] **5.2 Disable pattern loss entirely**
  - Change: `pattern_loss_weight: 0.3 -> 0.0` (does the auxiliary signal
    still help at fine-tune stage?).

---

## Phase 6 — bridge regularisation

- [ ] **6.1 Enable bridge codebook dropout**
  - Change: `bridge_codebook_dropout: 0.0 -> 0.1`.

- [ ] **6.2 Raise instruction_dropout**
  - Change: `instruction_dropout: 0.1 -> 0.2`.

- [ ] **6.3 Lower instruction_dropout**
  - Change: `instruction_dropout: 0.1 -> 0.05`.

---

## Phase 7 — combine best of 1–6

- [ ] **7.1 Combine the two best improvements from prior phases**
  - Pick the two highest-composite KEEP experiments. Stack both changes into
    one config. Run once. If it doesn't beat the better of the two, revert.

- [ ] **7.2 Combine the best three improvements**
  - Same logic, three changes. Diminishing returns common here.

---

## Phase 8 — promotion

- [ ] **8.1 Promote new champion**
  - Once any experiment produces a config that strictly dominates the current
    champion on composite AND has no category regression > 2 ROUGE-L pts,
    copy its config to
    `config/llm_finetuning/medgemma/champion_v2.yaml` and its checkpoint to
    `checkpoints/.../CHAMPION_V2/best_model.pt`.
  - Open a PR from `rb/e4d-champion-iter` → `main` (or the user's preferred
    target branch) with `ralph/CHAMPION_V2_SUMMARY.md` describing: what
    changed, composite delta, per-category deltas, per-dataset deltas.

---

## Definition of done

Either:
- (a) All experiments 0.1–7.2 are checked AND a promotion happened in 8.1, or
- (b) Phases 1–6 are checked, 7.x shows no further gain, and a
  `ralph/NO_IMPROVEMENT.md` is written explaining the ceiling observed.

Keep the loop deterministic, keep the eval set locked, keep the champion
reproducible.
