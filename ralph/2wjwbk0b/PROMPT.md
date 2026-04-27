# Ralph Loop — 2wjwbk0b → HEARTS-maximised champion

You are on branch `rb/2wjwbk0b-hearts-max`. Goal: take the current
`2wjwbk0b_20260413-224103_ENHANCED` checkpoint and **iteratively fine-tune
it to maximise the HEARTS benchmark composite across all 18 `mhi_ecg`
tasks**, without per-task regressions.

Each iteration: pick the next experiment from
`ralph/2wjwbk0b/DELIVERABLES.md`, copy the base config, apply **one
hyperparameter / data-mix change**, warm-start from the active reference
checkpoint, train, run the full HEARTS eval, compare, keep or revert,
commit, and stop. The hook re-fires.

## Mandatory loop protocol

1. `git fetch --all && git status` — must be clean, must be on
   `rb/2wjwbk0b-hearts-max`. If not, stop and surface.

2. Read `ralph/2wjwbk0b/DELIVERABLES.md`. Find the **first unchecked**
   experiment. If none, write `ralph/2wjwbk0b/CONVERGED.md` (see Stop
   conditions) and stop.

3. Read `ralph/2wjwbk0b/BASELINE.json` to know the baseline composite.
   Read `ralph/2wjwbk0b/ACTIVE_CHECKPOINT.txt` to know which checkpoint
   to warm-start from (initially the original 2wjwbk0b; updates after
   each kept experiment).

4. **Pre-flight checks** (refuse to start if any fails):
   - GPU `2`: must have <5 GB used. If busy → stop with `⚠ GPU2 BUSY` (do
     NOT kill other jobs). The loop is single-GPU on GPU 2 only;
     GPUs 0 and 1 are reserved for the user's other training runs.
   - Active checkpoint file must exist on disk.
   - Last 3 entries in `ralph/2wjwbk0b/results.tsv` must not all be
     `reverted` — if they are, stop and surface as
     `⚠ STALLED: 3 consecutive reverts`.

5. **Build the experiment config.**
   Copy the active reference config (initially
   `checkpoints/BEST_LLM/2wjwbk0b_20260413-224103_ENHANCED/config.yaml`)
   to `ralph/2wjwbk0b/runs/exp_NNN_<slug>/config.yaml`. Apply *exactly the
   one* change the experiment specifies. Set:
   - `resume_checkpoint_path: <ACTIVE_CHECKPOINT.txt content>`
   - `num_epochs: 1` (single epoch over the weighted train set; this is the
     fast-iteration budget — full 3-epoch retraining is reserved for
     promotion in the final phase)
   - `run_mode: train`
   - `use_wandb: true`, run name `2wjwbk0b_iter_<exp_NNN>`
   - `validation_max_batches: 0` (skip in-training val to save time;
     external HEARTS eval is the only score that matters)

6. **Train.**
   `bash scripts/runner.sh --base_config ralph/2wjwbk0b/runs/exp_NNN_<slug>/config.yaml --selected_gpus 2 --use_wandb true --run_mode train --instruct_mode true`
   redirected to `ralph/2wjwbk0b/runs/exp_NNN_<slug>/train.log`. Single-GPU
   training on GPU 2 only — slower than the original 2-GPU setup (expect
   ~6-8h per 1-epoch run on the 400k weighted parquet) but avoids
   contention with other jobs on GPUs 0/1. If the process exits non-zero,
   write `⚠ FAILED:` under the deliverable item, commit the log, append a
   `failed` row to `results.tsv`, mark `[x]`, stop.

7. **Locate the produced checkpoint.** The runner writes to
   `checkpoints/ECG_Tokenizer_LLM_Finetuning/ECG_tokenizer_MedGemma/<wandb_id>_<ts>/best_model.pt`.
   Save the absolute path to
   `ralph/2wjwbk0b/runs/exp_NNN_<slug>/checkpoint.txt`.

8. **Run the HEARTS eval** (this is the score that matters):
   ```
   uv run --directory /volume/HEARTS python /volume/ECG_tokenizer/ralph/2wjwbk0b/score_hearts_all.py \
       --checkpoint <new checkpoint> \
       --out ralph/2wjwbk0b/runs/exp_NNN_<slug>/scores.json \
       --n-per-task 100 \
       --device cuda:2
   ```
   This script runs all 18 `mhi_ecg` tasks via the HEARTS
   `MedGemmaECGAgent`, aggregates per-task metrics, and computes a
   normalised composite. Output JSON includes `composite`,
   `per_task_score`, `per_task_metric`, `n_per_task`. ETA ≈ 5–7 min.

9. **Decide: KEEP or REVERT.**
   Open `ralph/2wjwbk0b/BASELINE.json` and the new `scores.json`.
   - **KEEP** if `scores.composite >= baseline.composite` AND
     for every task `t`, `scores.per_task_score[t] >= baseline.per_task_score[t] - 0.05`
     (max 5 absolute-point per-category regression).
   - **REVERT** otherwise — but always keep the artefacts in
     `runs/exp_NNN_<slug>/` for the record.

10. **Update active reference.** On KEEP, overwrite
    `ralph/2wjwbk0b/ACTIVE_CHECKPOINT.txt` with the new checkpoint path
    and `ralph/2wjwbk0b/ACTIVE_BASELINE.json` with the new scores. The
    next experiment will warm-start from this. **Do not** overwrite
    `BASELINE.json` — that stays as the original 2wjwbk0b reference for
    diagnostics.

11. **Append to `ralph/2wjwbk0b/results.tsv`** (tab-separated, one row):
    `exp_num\tslug\tchange\tcomposite\tdelta_vs_active\tworst_per_task_delta\tdecision\tcheckpoint\tcommit_sha`

12. **Commit** in this order, one commit per concern:
    - `experiment NNN: <param> <old> -> <new> [<reason>]` — config + run
      artefacts (logs, scores, checkpoint.txt). Include the commit-message
      style of the existing repo (matches `experiment N: ...` history).
    - On KEEP only: `feat(2wjwbk0b): promote exp NNN as active` — updates
      to `ACTIVE_CHECKPOINT.txt` and `ACTIVE_BASELINE.json`.
    - `chore(ralph): mark exp NNN <kept|reverted>` — flip the checkbox.

13. **Push** all commits to `origin rb/2wjwbk0b-hearts-max`.

14. **Stop.** Hook re-fires; next iteration picks up the next item.

## Composite definition (per `score_hearts_all.py`)

Each of 18 tasks contributes a per-task score in `[0, 1]`:

| task family | metric | normalisation |
|---|---|---|
| `interpretation` | mean ROUGE-L F1 | as-is (already 0–1) |
| `json_interpretation` | per-field token-F1 | as-is |
| `classification`, `qrs_axis`, `acs_severity`, `culprit_artery` | Accuracy | as-is |
| `category_*`, `localization_*` | example-based F1 | as-is |
| `heart_rate` | MAE (bpm) | `max(0, 1 - MAE/20)` |
| `lvef` | MAE (% EF) | `max(0, 1 - MAE/15)` |
| `age_gender` | combined: 0.5·gender_acc + 0.5·max(0, 1 - age_MAE/15) | as-is |
| `afib_risk`, `structural_heart_disease` | AUROC | as-is |

Composite = simple mean across the 18 task scores. Per-task scores and the
composite are both written to `scores.json`.

## Stop conditions

Set by the watchdog logic in step 4 + step 2:

- **CONVERGED**: 3 consecutive `kept` rows with composite delta < 0.005,
  OR 5 consecutive `reverted` rows.
- **NO_MORE_DELIVERABLES**: every item in DELIVERABLES is `[x]`.
- **STALLED**: per step 4.

On any stop condition, write the corresponding marker file
(`CONVERGED.md`, `NO_MORE_DELIVERABLES.md`, `STALLED.md`) into
`ralph/2wjwbk0b/`, summarising the active checkpoint + final composite.

## Hard rules

- Only modify `ralph/2wjwbk0b/**` and the per-experiment config copy. Never
  modify `models/`, `runners/`, `projects/`, `scripts/runner.sh`, or any
  config under `config/`.
- Never overwrite `BASELINE.json` after Phase 0 lock.
- Never alter the eval set (`combined_test_qa_m25k_h25k.parquet`).
- One commit set per iteration. Messages must follow the schemas above.
- No force-push, no rebase, no `--no-verify`.
- Architecture-locked params **must not change** (would break the bridge):
  `bridge_qformer_layers`, `bridge_num_heads`, `codebook_offset`,
  `num_codebooks_kept`, `bridge_mid_dim`, `llm_input_embedding_size`,
  `huggingface_model_name`, `tokenizer_name`,
  `pretrained_tokenizer_path`, `stage1_checkpoint_path`.

## Quick reference

- Original checkpoint: `checkpoints/BEST_LLM/2wjwbk0b_20260413-224103_ENHANCED/best_model.pt`
- Original config: `checkpoints/BEST_LLM/2wjwbk0b_20260413-224103_ENHANCED/config.yaml`
- HEARTS repo: `/volume/HEARTS` (branch `main`)
- HEARTS test fixtures: built once via
  `uv run --directory /volume/HEARTS python scripts/build_mhi_ecg_fixtures.py --max 100 --out-dir /volume/HEARTS/fix_test_cases`
- Eval helper: `ralph/2wjwbk0b/score_hearts_all.py`
- Composite formula reference: this file's "Composite definition" section
  + the canonical implementation in `score_hearts_all.py`.

One experiment per iteration. Maximise the composite. Don't regress any
single category by more than 5 absolute points. Stop honestly when no
further gain is found.
