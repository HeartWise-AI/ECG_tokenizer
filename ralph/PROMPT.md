# Ralph Loop — e4d Champion Iteration

You are on branch `rb/e4d-champion-iter`. Your job is to **iteratively improve
the current e4d champion** (MedGemma-4B + QFormer bridge, 8 codebooks) without
regressing on its strongest categories.

Each loop iteration: pick the next unchecked experiment from
`ralph/DELIVERABLES.md`, apply exactly one config change, **warm-start from the
champion checkpoint**, train for the iteration budget, evaluate, compare
against the champion on the locked eval set, commit the result (kept or
reverted), mark the item, stop.

This loop is **different from `autoresearch/`** in three ways:

1. **Warm start, not from scratch.** Every run resumes from the champion
   checkpoint (`e4dw86nh_20251220-232839_BEST_QFORMER_8CB/best_model.pt`) and
   fine-tunes briefly.
2. **Longer budget per iteration than 100 steps.** See `TRAINING_BUDGET` in
   `ralph/DELIVERABLES.md` — ⚠ confirm with the user before the first run.
3. **Composite score must not regress on the champion's strong categories.**
   Regression on any category by > 2 absolute ROUGE-L points reverts the
   experiment even if global composite improves.

## Mandatory loop protocol

1. `git fetch --all && git status` — must be clean, must be on
   `rb/e4d-champion-iter`. If not, stop and surface the issue.

2. **Read** `ralph/DELIVERABLES.md`. Find the **first unchecked experiment**.
   If all are checked, print `ALL EXPERIMENTS COMPLETE` and stop.

3. **Re-read the current champion config** before editing:
   `config/llm_finetuning/medgemma/e4dw86nh_20251220-232839_BEST_QFORMER_8CB.yaml.yaml`
   (or the specific `champion_*` variant the experiment targets).

4. **Make a copy** of the champion config to
   `ralph/runs/exp_NNN_<slug>/config.yaml` and apply the **one** change
   specified by the experiment. Do not modify the champion config in place.

5. **Set warm-start fields** in the copy:
   - `resume_from_checkpoint: /volume/ECG_tokenizer/checkpoints/ECG_Tokenizer_LLM_Finetuning/ECG_tokenizer_MedGemma/e4dw86nh_20251220-232839_BEST_QFORMER_8CB/best_model.pt`
   - `load_optimizer_state: false` (fresh optimizer for fine-tune)
   - `num_epochs: <TRAINING_BUDGET from DELIVERABLES.md>`
   - `use_wandb: true` (log every run)

6. **Check GPU memory is free** before launching:
   `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader`. If any
   target GPU is >5GB used, stop and surface — do not kill other processes.

7. **Launch training** via `scripts/runner.sh` with the experiment config.
   Redirect stdout/stderr to `ralph/runs/exp_NNN_<slug>/train.log`.

8. **Evaluate** using the LOCKED evaluation parquet
   `output/combined_test_qa_m5k_h5k.parquet` via
   `inference/generate_all_qa_pairs.py`. Save generations to
   `ralph/runs/exp_NNN_<slug>/generations.json` and per-category metrics to
   `ralph/runs/exp_NNN_<slug>/metrics.json`.

9. **Compare** against
   `ralph/CHAMPION_BASELINE.json` (computed once in experiment 0). Apply the
   decision rule:
   - ✅ **KEEP** if composite ≥ champion AND no category regresses > 2 ROUGE-L pts.
   - ❌ **REVERT** otherwise. `git checkout --` the config copy delta, but
     keep the `runs/exp_NNN_<slug>/` artifacts for the record.

10. **Append** one row to `ralph/results.tsv` with columns:
    `exp_num, slug, change, composite, rougeL, bleu4, meteor, bertscore_f1,
    worst_category_delta, decision, commit_sha`.

11. **Commit**. Message format follows the existing repo style:
    `experiment NNN: <param> <old> -> <new> (<one-sentence reason>)` on KEEP.
    On REVERT: `experiment NNN: <param> <old> -> <new> (reverted: <reason>)`.

12. **Mark the item `[x]`** in `ralph/DELIVERABLES.md`. Separate commit:
    `chore(ralph): mark exp NNN done`.

13. **Push** both commits to origin. Stop.

## Hard constraints

- Only modify files under `ralph/` and the per-experiment config copy. **Do
  not touch model code, runner code, loss code, or dataset code.** If an
  experiment would require a code change, mark it `⚠ CODE-CHANGE` and skip —
  a human must implement it.
- Never change the locked eval parquet. Never change
  `ralph/CHAMPION_BASELINE.json` after experiment 0.
- Never change architecture-locked params: `bridge_qformer_layers`,
  `bridge_num_heads`, `codebook_offset`, `num_codebooks_kept`,
  `bridge_mid_dim`, `llm_input_embedding_size`, `huggingface_model_name`.
- Never push force, never amend, never skip hooks.
- If training crashes: read `tail -100 train.log`, add a `⚠ FAILED:` note
  with the root cause under the item, append a `failed` row to results.tsv,
  commit the log, mark the item `[x]`, move on.
- If the champion checkpoint file is missing, stop with
  `⚠ BLOCKED: champion checkpoint not found at <path>` — do not regenerate.

## Quick reference

- Champion config:
  `config/llm_finetuning/medgemma/e4dw86nh_20251220-232839_BEST_QFORMER_8CB.yaml.yaml`
- Champion checkpoint:
  `checkpoints/ECG_Tokenizer_LLM_Finetuning/ECG_tokenizer_MedGemma/e4dw86nh_20251220-232839_BEST_QFORMER_8CB/best_model.pt`
- Locked eval parquet:
  `output/combined_test_qa_m5k_h5k.parquet`
- Training runner: `scripts/runner.sh`
- Inference: `inference/generate_all_qa_pairs.py`
- Metrics code: `utils/metrics/llm_metrics.py`
- Existing autoresearch loop (different purpose, reference only):
  `autoresearch/prompt.md`

One experiment per iteration. Warm-start every time. Lock the eval set.
