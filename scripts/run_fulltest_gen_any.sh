#!/usr/bin/env bash
# Run an exact set of full-test shards, validate them, merge, and score.
set -euo pipefail

ROOT=${ECG_REPO_ROOT:-/volume/ECG_tokenizer}
PYTHON=${ECG_PYTHON:-$ROOT/.venv/bin/python}
GENERATOR_SCRIPT=${ECG_GENERATOR_SCRIPT:-$ROOT/scripts/eval_judge_csv.py}
WORKFLOW_UTILS=${ECG_WORKFLOW_UTILS:-$ROOT/scripts/eval_workflow_utils.py}
SCORER_SCRIPT=${ECG_SCORER_SCRIPT:-$ROOT/scripts/score_deterministic.py}
SCORER_IMPLEMENTATION_ROOTS=${ECG_SCORER_IMPLEMENTATION_ROOTS:-$ROOT/utils}
OUTDIR=${ECG_OUTDIR:-$ROOT/analysis/x1split_judge}
LOG_DIR=${ECG_LOG_DIR:-$ROOT/logs}
EXPECTED_SHARDS=${ECG_EXPECTED_SHARDS:-"0 1 2 3 4 5 6 7"}
DEVICES=${ECG_DEVICES:-"cuda:0,cuda:1,cuda:2"}
GROUP_SIZE=${ECG_GROUP_SIZE:-0}
BATCH_SIZE=${ECG_BATCH_SIZE:-16}
MAX_NEW_TOKENS=${ECG_MAX_NEW_TOKENS:-96}
GENERATION_MICROBATCH_SIZE=${ECG_GENERATION_MICROBATCH_SIZE:-}
GROUP_BY_PROMPT=${ECG_GROUP_BY_PROMPT:-0}
EXPECTED_SCORE_CATEGORIES=${ECG_EXPECTED_SCORE_CATEGORIES:-"lvef afib_risk structural_heart_disease acs_severity"}

CKPT=${1:?checkpoint required}
TAG=${2:?tag required}

if [[ ! "$TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[fulltest] tag may contain only letters, numbers, dot, underscore, and hyphen" >&2
  exit 2
fi
if [[ ! -d "$ROOT" ]]; then
  echo "[fulltest:$TAG] repository root is missing: $ROOT" >&2
  exit 1
fi
cd "$ROOT"
if [[ ! -x "$PYTHON" ]]; then
  echo "[fulltest:$TAG] Python is not executable: $PYTHON" >&2
  exit 1
fi
for required in "$GENERATOR_SCRIPT" "$WORKFLOW_UTILS" "$SCORER_SCRIPT" "$CKPT"; do
  if [[ ! -f "$required" ]]; then
    echo "[fulltest:$TAG] required file is missing: $required" >&2
    exit 1
  fi
done
mkdir -p "$OUTDIR" "$LOG_DIR"
FULLTEST_LIFECYCLE_KEY="$OUTDIR/.${TAG}_fulltest-lifecycle"
if [[ "${ECG_FULLTEST_LIFECYCLE_LOCK-}" != "$FULLTEST_LIFECYCLE_KEY" ]]; then
  exec "$PYTHON" "$WORKFLOW_UTILS" run-with-output-lock \
    --output "$FULLTEST_LIFECYCLE_KEY" \
    --env-marker ECG_FULLTEST_LIFECYCLE_LOCK \
    -- bash "$0" "$CKPT" "$TAG"
fi

read -r -a SHARD_ARRAY <<< "$EXPECTED_SHARDS"
IFS=',' read -r -a DEVICE_ARRAY <<< "$DEVICES"
if [[ ${#SHARD_ARRAY[@]} -eq 0 || ${#DEVICE_ARRAY[@]} -eq 0 ]]; then
  echo "[fulltest:$TAG] shard and device sets must be nonempty" >&2
  exit 2
fi
seen=,
for shard in "${SHARD_ARRAY[@]}"; do
  if [[ ! "$shard" =~ ^[0-9]+$ ]]; then
    echo "[fulltest:$TAG] invalid shard id: $shard" >&2
    exit 2
  fi
  case "$seen" in
    *",$shard,"*)
      echo "[fulltest:$TAG] duplicate shard id: $shard" >&2
      exit 2
      ;;
  esac
  seen="$seen$shard,"
  subset="$OUTDIR/fulltest_p${shard}.parquet"
  if [[ ! -s "$subset" ]]; then
    echo "[fulltest:$TAG] required subset is missing or empty: $subset" >&2
    exit 1
  fi
done
if [[ ! "$GROUP_SIZE" =~ ^[0-9]+$ ]]; then
  echo "[fulltest:$TAG] ECG_GROUP_SIZE must be a nonnegative integer" >&2
  exit 2
fi
if [[ "$GROUP_SIZE" -eq 0 ]]; then
  GROUP_SIZE=${#DEVICE_ARRAY[@]}
fi

generation_args() {
  local shard=$1
  local device=$2
  local output="$OUTDIR/${TAG}_fulltest_p${shard}.csv"
  local subset="$OUTDIR/fulltest_p${shard}.parquet"
  GENERATION_ARGS=(
    --checkpoint "$CKPT"
    --subset_parquet "$subset"
    --output_csv "$output"
    --device "$device"
    --batch_size "$BATCH_SIZE"
    --max_new_tokens "$MAX_NEW_TOKENS"
  )
  if [[ -n "$GENERATION_MICROBATCH_SIZE" ]]; then
    GENERATION_ARGS+=(--generation_microbatch_size "$GENERATION_MICROBATCH_SIZE")
  fi
  if [[ "$GROUP_BY_PROMPT" == "1" ]]; then
    GENERATION_ARGS+=(--group_by_prompt)
  fi
}

run_generation() {
  local shard=$1
  local device=$2
  local output="$OUTDIR/${TAG}_fulltest_p${shard}.csv"
  local log="$LOG_DIR/${TAG}_fulltest_p${shard}.log"
  generation_args "$shard" "$device"
  if PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      "$PYTHON" "$GENERATOR_SCRIPT" "${GENERATION_ARGS[@]}" --validate-only \
      >> "$log" 2>&1; then
    echo "[fulltest:$TAG] p$shard is valid for current inputs; skipping"
    return 0
  fi
  echo "[fulltest:$TAG] generating p$shard on $device"
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" "$GENERATOR_SCRIPT" "${GENERATION_ARGS[@]}" \
    > "$log" 2>&1
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" "$GENERATOR_SCRIPT" "${GENERATION_ARGS[@]}" --validate-only \
    >> "$log" 2>&1
  [[ -s "$output" ]]
}

PIDS=()
PID_SHARDS=()
wait_batch() {
  local failed=0
  local index
  local pid
  for ((index = 0; index < ${#PIDS[@]}; index++)); do
    pid=${PIDS[$index]}
    if wait "$pid"; then
      echo "[fulltest:$TAG] p${PID_SHARDS[$index]} finished"
    else
      echo "[fulltest:$TAG] p${PID_SHARDS[$index]} failed; see $LOG_DIR/${TAG}_fulltest_p${PID_SHARDS[$index]}.log" >&2
      failed=1
    fi
  done
  PIDS=()
  PID_SHARDS=()
  return "$failed"
}

for ((index = 0; index < ${#SHARD_ARRAY[@]}; index++)); do
  shard=${SHARD_ARRAY[$index]}
  device=${DEVICE_ARRAY[$((index % ${#DEVICE_ARRAY[@]}))]}
  run_generation "$shard" "$device" &
  pid=$!
  PIDS+=("$pid")
  PID_SHARDS+=("$shard")
  echo "[fulltest:$TAG] launched p$shard pid $pid"
  if [[ ${#PIDS[@]} -ge "$GROUP_SIZE" ]]; then
    if ! wait_batch; then
      exit 1
    fi
  fi
done
if [[ ${#PIDS[@]} -gt 0 ]]; then
  if ! wait_batch; then
    exit 1
  fi
fi

MERGED_OUTPUT="$OUTDIR/${TAG}_fulltest_all.csv"
MERGE_ARGS=(merge-generations --output "$MERGED_OUTPUT")
for ((index = 0; index < ${#SHARD_ARRAY[@]}; index++)); do
  shard=${SHARD_ARRAY[$index]}
  device=${DEVICE_ARRAY[$((index % ${#DEVICE_ARRAY[@]}))]}
  generation_args "$shard" "$device"
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" "$GENERATOR_SCRIPT" "${GENERATION_ARGS[@]}" --validate-only \
    > /dev/null
  MERGE_ARGS+=(--input "$OUTDIR/${TAG}_fulltest_p${shard}.csv")
done
"$PYTHON" "$WORKFLOW_UTILS" "${MERGE_ARGS[@]}" > /dev/null
MERGED_IDENTITY=$("$PYTHON" "$WORKFLOW_UTILS" fingerprint "$MERGED_OUTPUT")
MERGED_CONTENT_IDENTITY=$(
  "$PYTHON" "$WORKFLOW_UTILS" content-fingerprint "$MERGED_OUTPUT"
)

SCORE_OUTPUT="$OUTDIR/${TAG}_fulltest_deterministic.txt"
SCORE_TEMP=$(mktemp "$OUTDIR/.${TAG}_fulltest_score.XXXXXX")
SCORE_SOURCE=$(mktemp "$OUTDIR/.${TAG}_fulltest_score_source.XXXXXX")
cleanup_score_temp() {
  if [[ -e "$SCORE_TEMP" ]]; then
    rm -f -- "$SCORE_TEMP"
  fi
  if [[ -e "$SCORE_SOURCE" ]]; then
    rm -f -- "$SCORE_SOURCE"
  fi
}
trap cleanup_score_temp EXIT
cp -- "$MERGED_OUTPUT" "$SCORE_SOURCE"
if [[ "$("$PYTHON" "$WORKFLOW_UTILS" content-fingerprint "$SCORE_SOURCE")" != "$MERGED_CONTENT_IDENTITY" ]]; then
  echo "[fulltest:$TAG] isolated score source does not match merged generation" >&2
  exit 1
fi
SCORE_ARGS=("$SCORE_SOURCE" --tag "${TAG}_FULLTEST")
read -r -a SCORE_CATEGORY_ARRAY <<< "$EXPECTED_SCORE_CATEGORIES"
if [[ ${#SCORE_CATEGORY_ARRAY[@]} -eq 0 ]]; then
  echo "[fulltest:$TAG] expected deterministic score categories must be nonempty" >&2
  exit 2
fi
for score_category in "${SCORE_CATEGORY_ARRAY[@]}"; do
  SCORE_ARGS+=(--expected-category "$score_category")
done
IFS=':' read -r -a SCORE_IMPLEMENTATION_ROOT_ARRAY <<< "$SCORER_IMPLEMENTATION_ROOTS"
SCORE_IDENTITY_ARGS=(score-identity --scorer-script "$SCORER_SCRIPT")
SCORE_PROVENANCE_ARGS=(--scorer-script "$SCORER_SCRIPT")
for implementation_root in "${SCORE_IMPLEMENTATION_ROOT_ARRAY[@]}"; do
  SCORE_IDENTITY_ARGS+=(--implementation-root "$implementation_root")
  SCORE_PROVENANCE_ARGS+=(--implementation-root "$implementation_root")
done
for score_category in "${SCORE_CATEGORY_ARRAY[@]}"; do
  SCORE_PROVENANCE_ARGS+=(--expected-category "$score_category")
done
SCORE_IMPLEMENTATION_IDENTITY=$(
  "$PYTHON" "$WORKFLOW_UTILS" "${SCORE_IDENTITY_ARGS[@]}"
)
if ! "$PYTHON" "$SCORER_SCRIPT" "${SCORE_ARGS[@]}" 2>&1 | tee "$SCORE_TEMP"; then
  echo "[fulltest:$TAG] deterministic scoring failed" >&2
  exit 1
fi
if [[ "$("$PYTHON" "$WORKFLOW_UTILS" fingerprint "$MERGED_OUTPUT")" != "$MERGED_IDENTITY" ]]; then
  echo "[fulltest:$TAG] merged generation changed while scoring was active" >&2
  exit 1
fi
if [[ "$("$PYTHON" "$WORKFLOW_UTILS" content-fingerprint "$SCORE_SOURCE")" != "$MERGED_CONTENT_IDENTITY" ]]; then
  echo "[fulltest:$TAG] isolated score source changed while scoring was active" >&2
  exit 1
fi
if [[ "$("$PYTHON" "$WORKFLOW_UTILS" "${SCORE_IDENTITY_ARGS[@]}")" != "$SCORE_IMPLEMENTATION_IDENTITY" ]]; then
  echo "[fulltest:$TAG] score implementation changed while scoring was active" >&2
  exit 1
fi
mv -f -- "$SCORE_TEMP" "$SCORE_OUTPUT"
"$PYTHON" "$WORKFLOW_UTILS" publish-score \
  --source-csv "$MERGED_OUTPUT" \
  --score-output "$SCORE_OUTPUT" \
  --expected-source-identity "$MERGED_IDENTITY" \
  --expected-score-implementation "$SCORE_IMPLEMENTATION_IDENTITY" \
  "${SCORE_PROVENANCE_ARGS[@]}" \
  --tag "${TAG}_FULLTEST" > /dev/null
"$PYTHON" "$WORKFLOW_UTILS" validate-score \
  --source-csv "$MERGED_OUTPUT" \
  --score-output "$SCORE_OUTPUT" \
  "${SCORE_PROVENANCE_ARGS[@]}" \
  --tag "${TAG}_FULLTEST" > /dev/null
rm -f -- "$SCORE_SOURCE"
trap - EXIT
echo "[fulltest:$TAG] GENERATION+SCORING COMPLETE"
