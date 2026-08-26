#!/usr/bin/env bash
# Launch one training run, require its checkpoint to change, then evaluate it.
set -euo pipefail

ROOT=${ECG_REPO_ROOT:-/volume/ECG_tokenizer}
PYTHON=${ECG_PYTHON:-$ROOT/.venv/bin/python}
GENERATOR_SCRIPT=${ECG_GENERATOR_SCRIPT:-$ROOT/scripts/eval_judge_csv.py}
WORKFLOW_UTILS=${ECG_WORKFLOW_UTILS:-$ROOT/scripts/eval_workflow_utils.py}
JUDGE_WRAPPER=${ECG_JUDGE_WRAPPER:-$ROOT/scripts/run_csv_llm_judge_sharded.sh}
CKPT=${TIMEAXIS_CKPT:-$ROOT/checkpoints/timeaxis_direct/best_model.pt}
SUBSET=${TIMEAXIS_SUBSET:-$ROOT/analysis/x1split_judge/s3000_seed123.parquet}
GEN=${TIMEAXIS_GENERATION_CSV:-$ROOT/analysis/x1split_judge/timeaxis_direct_s3000_generations.csv}
JUDGE=${TIMEAXIS_JUDGE_JSON:-$ROOT/analysis/x1split_judge/timeaxis_direct_s3000_judge.json}
JUDGE_SHARD_DIR=${TIMEAXIS_JUDGE_SHARD_DIR:-$ROOT/analysis/x1split_judge/timeaxis_direct_judge_shards}
LOG_DIR=${ECG_LOG_DIR:-$ROOT/logs}
POST_TRAIN_WAIT_SECONDS=${POST_TRAIN_WAIT_SECONDS:-30}
DEVICE=${TIMEAXIS_DEVICE:-cuda:2}
BATCH_SIZE=${TIMEAXIS_BATCH_SIZE:-16}
MAX_NEW_TOKENS=${TIMEAXIS_MAX_NEW_TOKENS:-96}
JUDGE_SHARDS=${JUDGE_SHARDS:-16}
JUDGE_MAX_PARALLEL=${JUDGE_MAX_PARALLEL:-4}

if [[ $# -lt 2 || "$1" != "--" ]]; then
  echo "usage: $0 -- training-command [args ...]" >&2
  exit 2
fi
shift
TRAIN_COMMAND=("$@")
if [[ ! -d "$ROOT" ]]; then
  echo "[timeaxis] repository root is missing: $ROOT" >&2
  exit 1
fi
cd "$ROOT"
for required in "$PYTHON" "$GENERATOR_SCRIPT" "$WORKFLOW_UTILS" "$JUDGE_WRAPPER" "$SUBSET"; do
  if [[ ! -e "$required" ]]; then
    echo "[timeaxis] required input is missing: $required" >&2
    exit 1
  fi
done
if [[ ! -x "$PYTHON" ]]; then
  echo "[timeaxis] Python is not executable: $PYTHON" >&2
  exit 1
fi
mkdir -p \
  "$(dirname "$CKPT")" \
  "$(dirname "$GEN")" \
  "$(dirname "$JUDGE")" \
  "$JUDGE_SHARD_DIR" \
  "$LOG_DIR"
GEN_LIFECYCLE_KEY="$GEN.timeaxis-lifecycle"
GEN_LOCK_MARKER=${ECG_TIMEAXIS_GENERATION_LOCK-}
if [[ "$GEN_LOCK_MARKER" != "$GEN_LIFECYCLE_KEY" ]]; then
  exec "$PYTHON" "$WORKFLOW_UTILS" run-with-output-lock \
    --output "$GEN_LIFECYCLE_KEY" --env-marker ECG_TIMEAXIS_GENERATION_LOCK \
    -- bash "$0" -- "${TRAIN_COMMAND[@]}"
fi
JUDGE_LIFECYCLE_KEY="$JUDGE.timeaxis-lifecycle"
JUDGE_LOCK_MARKER=${ECG_TIMEAXIS_JUDGE_LOCK-}
if [[ "$JUDGE_LOCK_MARKER" != "$JUDGE_LIFECYCLE_KEY" ]]; then
  exec "$PYTHON" "$WORKFLOW_UTILS" run-with-output-lock \
    --output "$JUDGE_LIFECYCLE_KEY" --env-marker ECG_TIMEAXIS_JUDGE_LOCK \
    -- bash "$0" -- "${TRAIN_COMMAND[@]}"
fi
LOCK_MARKER=${ECG_TIMEAXIS_CHECKPOINT_LOCK-}
if [[ "$LOCK_MARKER" != "$CKPT" ]]; then
  exec "$PYTHON" "$WORKFLOW_UTILS" run-with-output-lock \
    --output "$CKPT" --env-marker ECG_TIMEAXIS_CHECKPOINT_LOCK \
    -- bash "$0" -- "${TRAIN_COMMAND[@]}"
fi
if [[ -s "$CKPT" ]]; then
  BEFORE_CHECKPOINT=$("$PYTHON" "$WORKFLOW_UTILS" content-fingerprint "$CKPT")
else
  BEFORE_CHECKPOINT=MISSING
fi
RUN_CHECKPOINT=$(mktemp "$(dirname "$CKPT")/.timeaxis-checkpoint.XXXXXX")
cleanup_run_checkpoint() {
  if [[ -e "$RUN_CHECKPOINT" ]]; then
    rm -f -- "$RUN_CHECKPOINT"
  fi
}
trap cleanup_run_checkpoint EXIT

echo "[timeaxis] launching training command"
if TIMEAXIS_OUTPUT_CHECKPOINT="$RUN_CHECKPOINT" "${TRAIN_COMMAND[@]}"; then
  echo "[timeaxis] training command completed successfully"
else
  TRAIN_STATUS=$?
  echo "[timeaxis] training command failed with status $TRAIN_STATUS" >&2
  exit "$TRAIN_STATUS"
fi
sleep "$POST_TRAIN_WAIT_SECONDS"

if [[ ! -f "$RUN_CHECKPOINT" || -L "$RUN_CHECKPOINT" || ! -s "$RUN_CHECKPOINT" ]]; then
  echo "[timeaxis] training ended without a nonempty isolated checkpoint at TIMEAXIS_OUTPUT_CHECKPOINT" >&2
  exit 1
fi
PRODUCED_CHECKPOINT=$("$PYTHON" "$WORKFLOW_UTILS" content-fingerprint "$RUN_CHECKPOINT")
mv -f -- "$RUN_CHECKPOINT" "$CKPT"
trap - EXIT
AFTER_CHECKPOINT=$("$PYTHON" "$WORKFLOW_UTILS" content-fingerprint "$CKPT")
if [[ "$AFTER_CHECKPOINT" != "$PRODUCED_CHECKPOINT" ]]; then
  echo "[timeaxis] promoted checkpoint does not match the isolated training output" >&2
  exit 1
fi
if [[ "$AFTER_CHECKPOINT" == "$BEFORE_CHECKPOINT" ]]; then
  echo "[timeaxis] checkpoint did not change during the training command; refusing stale reuse" >&2
  exit 1
fi
echo "[timeaxis] accepted checkpoint changed by the training command: $CKPT"

GENERATION_ARGS=(
  --checkpoint "$CKPT"
  --subset_parquet "$SUBSET"
  --output_csv "$GEN"
  --device "$DEVICE"
  --batch_size "$BATCH_SIZE"
  --max_new_tokens "$MAX_NEW_TOKENS"
)
if PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" "$GENERATOR_SCRIPT" "${GENERATION_ARGS[@]}" --validate-only \
    > /dev/null 2>&1; then
  echo "[timeaxis] generation is valid for the changed checkpoint; skipping"
else
  echo "[timeaxis] generating judge CSV"
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" "$GENERATOR_SCRIPT" "${GENERATION_ARGS[@]}" \
    > "$LOG_DIR/timeaxis_generation.log" 2>&1
fi
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" "$GENERATOR_SCRIPT" "${GENERATION_ARGS[@]}" --validate-only \
  > /dev/null
GENERATION_IDENTITY=$("$PYTHON" "$WORKFLOW_UTILS" fingerprint "$GEN")

echo "[timeaxis] running sharded LLM judge"
CSV="$GEN" OUTPUT="$JUDGE" OUT_DIR="$JUDGE_SHARD_DIR" \
  SHARDS="$JUDGE_SHARDS" MAX_PARALLEL="$JUDGE_MAX_PARALLEL" \
  PYTHON="$PYTHON" bash "$JUDGE_WRAPPER" \
  > "$LOG_DIR/timeaxis_judge.log" 2>&1
if [[ ! -s "$JUDGE" || ! -s "$JUDGE.manifest.json" ]]; then
  echo "[timeaxis] judge wrapper returned without a validated result" >&2
  exit 1
fi
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" "$GENERATOR_SCRIPT" "${GENERATION_ARGS[@]}" --validate-only \
  > /dev/null
if [[ "$("$PYTHON" "$WORKFLOW_UTILS" fingerprint "$GEN")" != "$GENERATION_IDENTITY" ]]; then
  echo "[timeaxis] generation changed while judge evaluation was active" >&2
  exit 1
fi
"$PYTHON" "$WORKFLOW_UTILS" validate-judge-binding \
  --csv "$GEN" --output-json "$JUDGE" > /dev/null
FINAL_CHECKPOINT=$("$PYTHON" "$WORKFLOW_UTILS" content-fingerprint "$CKPT")
if [[ "$FINAL_CHECKPOINT" != "$PRODUCED_CHECKPOINT" ]]; then
  echo "[timeaxis] checkpoint changed after isolated promotion; refusing completion" >&2
  exit 1
fi
echo "[timeaxis] JUDGE COMPLETE: $JUDGE"
