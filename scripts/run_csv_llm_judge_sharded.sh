#!/usr/bin/env bash
# Run judge_eval.py over deterministic CSV shards and publish one valid result.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-/opt/conda/bin/python}
JUDGE_DIR=${JUDGE_DIR:-${LLM_JUDGE_DIR:-/volume/LLM_JUDGE}}
JUDGE_SCRIPT=${JUDGE_SCRIPT:-$JUDGE_DIR/judge_eval.py}
JUDGE_MODEL_ID=${JUDGE_MODEL_ID-}
JUDGE_DEPLOYMENT_ID=${JUDGE_DEPLOYMENT_ID-}
JUDGE_DECODING_ID=${JUDGE_DECODING_ID-}
WORKFLOW_UTILS=${ECG_WORKFLOW_UTILS:-$SCRIPT_DIR/eval_workflow_utils.py}
CSV=${CSV:?CSV must be an absolute generation CSV path}
OUTPUT=${OUTPUT:?OUTPUT must be an absolute judge JSON path}
OUT_DIR=${OUT_DIR:?OUT_DIR must be an absolute shard directory path}
SHARDS=${SHARDS:-16}
MAX_PARALLEL=${MAX_PARALLEL:-4}

require_absolute() {
  local value=$1
  local value_name=$2
  case "$value" in
    /*) ;;
    *)
      echo "[judge] $value_name must be absolute: $value" >&2
      exit 2
      ;;
  esac
}
require_absolute "$CSV" CSV
require_absolute "$OUTPUT" OUTPUT
require_absolute "$OUT_DIR" OUT_DIR
require_absolute "$JUDGE_DIR" JUDGE_DIR
require_absolute "$JUDGE_SCRIPT" JUDGE_SCRIPT
if [[ ! "$SHARDS" =~ ^[1-9][0-9]*$ || ! "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "[judge] SHARDS and MAX_PARALLEL must be positive integers" >&2
  exit 2
fi
require_immutable_identity() {
  local value=$1
  local value_name=$2
  local normalized
  if [[ ! "$value" =~ [^[:space:]] ]]; then
    echo "[judge] $value_name must be an explicit nonempty immutable identity" >&2
    exit 2
  fi
  normalized=$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')
  case "$normalized" in
    *latest*|*current*|*default*|*auto*|*unknown*|*unset*)
      echo "[judge] $value_name is mutable or unspecified: $value" >&2
      exit 2
      ;;
  esac
}
require_immutable_identity "$JUDGE_MODEL_ID" JUDGE_MODEL_ID
require_immutable_identity "$JUDGE_DEPLOYMENT_ID" JUDGE_DEPLOYMENT_ID
require_immutable_identity "$JUDGE_DECODING_ID" JUDGE_DECODING_ID
printf -v JUDGE_IDENTITY 'model[%d]=%s|deployment[%d]=%s|decoding[%d]=%s' \
  "${#JUDGE_MODEL_ID}" "$JUDGE_MODEL_ID" \
  "${#JUDGE_DEPLOYMENT_ID}" "$JUDGE_DEPLOYMENT_ID" \
  "${#JUDGE_DECODING_ID}" "$JUDGE_DECODING_ID"
for required in "$CSV" "$JUDGE_SCRIPT" "$WORKFLOW_UTILS"; do
  if [[ ! -s "$required" ]]; then
    echo "[judge] required file is missing or empty: $required" >&2
    exit 1
  fi
done
if [[ ! -x "$PYTHON" ]]; then
  echo "[judge] Python is not executable: $PYTHON" >&2
  exit 1
fi
LOCK_MARKER=${ECG_JUDGE_OUTPUT_LOCK-}
if [[ "$LOCK_MARKER" != "$OUTPUT" ]]; then
  exec "$PYTHON" "$WORKFLOW_UTILS" run-with-output-lock \
    --output "$OUTPUT" --env-marker ECG_JUDGE_OUTPUT_LOCK \
    -- bash "$0" "$@"
fi
mkdir -p "$OUT_DIR" "$(dirname "$OUTPUT")" "$OUT_DIR/logs"
echo "[judge] model identity: $JUDGE_MODEL_ID"
echo "[judge] deployment identity: $JUDGE_DEPLOYMENT_ID"
echo "[judge] decoding identity: $JUDGE_DECODING_ID"

if "$PYTHON" "$WORKFLOW_UTILS" validate-merged-judge \
    --csv "$CSV" --output-json "$OUTPUT" --shards "$SHARDS" \
    --judge-dir "$JUDGE_DIR" --judge-script "$JUDGE_SCRIPT" \
    --run-id "$JUDGE_IDENTITY" > /dev/null 2>&1; then
  echo "[judge] existing output is valid for current inputs"
  echo "[judge] JUDGE COMPLETE: $OUTPUT"
  exit 0
fi

stale_suffix="stale.$(date -u +%Y%m%dT%H%M%SZ).$$"
if [[ -e "$OUTPUT" ]]; then
  stale_output="$OUTPUT.$stale_suffix"
  mv -f -- "$OUTPUT" "$stale_output"
  echo "[judge] moved rejected prior output to $stale_output" >&2
fi
if [[ -e "$OUTPUT.manifest.json" ]]; then
  mv -f -- "$OUTPUT.manifest.json" "$OUTPUT.manifest.json.$stale_suffix"
fi

EFFECTIVE_SHARDS=$("$PYTHON" "$WORKFLOW_UTILS" split-judge \
  --csv "$CSV" --out-dir "$OUT_DIR" --shards "$SHARDS")
if [[ ! "$EFFECTIVE_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[judge] splitter returned an invalid shard count: $EFFECTIVE_SHARDS" >&2
  exit 1
fi

run_shard() {
  local shard_index=$1
  local shard_tag
  shard_tag=$(printf "%03d" "$shard_index")
  local shard_csv="$OUT_DIR/judge_input_${shard_tag}.csv"
  local judge_output="$OUT_DIR/judge_output_${shard_tag}.json"
  local log="$OUT_DIR/logs/judge_${shard_tag}.log"
  if "$PYTHON" "$WORKFLOW_UTILS" validate-judge-shard \
      --csv "$shard_csv" --output-json "$judge_output" \
      --judge-dir "$JUDGE_DIR" --judge-script "$JUDGE_SCRIPT" \
      --run-id "$JUDGE_IDENTITY" >> "$log" 2>&1; then
    echo "[judge] shard $shard_tag is valid for current inputs; skipping"
    return 0
  fi

  local raw_output="$judge_output.raw.$$.$RANDOM"
  cleanup_raw() {
    if [[ -e "$raw_output" ]]; then
      rm -f -- "$raw_output"
    fi
  }
  trap cleanup_raw EXIT
  echo "[judge] running shard $shard_tag"
  local before_identity
  before_identity=$("$PYTHON" "$WORKFLOW_UTILS" judge-identity \
    --judge-dir "$JUDGE_DIR" --judge-script "$JUDGE_SCRIPT" \
    --run-id "$JUDGE_IDENTITY")
  (cd "$JUDGE_DIR" && "$PYTHON" "$JUDGE_SCRIPT" \
    --csv "$shard_csv" --output "$raw_output") > "$log" 2>&1
  if [[ ! -s "$raw_output" ]]; then
    echo "[judge] shard $shard_tag produced no JSON" >&2
    return 1
  fi
  local after_identity
  after_identity=$("$PYTHON" "$WORKFLOW_UTILS" judge-identity \
    --judge-dir "$JUDGE_DIR" --judge-script "$JUDGE_SCRIPT" \
    --run-id "$JUDGE_IDENTITY")
  if [[ "$after_identity" != "$before_identity" ]]; then
    echo "[judge] implementation changed while shard $shard_tag was running" >&2
    return 1
  fi
  "$PYTHON" "$WORKFLOW_UTILS" seal-judge-shard \
    --csv "$shard_csv" --input-json "$raw_output" \
    --output-json "$judge_output" --judge-dir "$JUDGE_DIR" \
    --judge-script "$JUDGE_SCRIPT" --run-id "$JUDGE_IDENTITY" \
    --expected-judge-identity "$before_identity" \
    >> "$log" 2>&1
  cleanup_raw
  trap - EXIT
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
      echo "[judge] shard ${PID_SHARDS[$index]} finished"
    else
      echo "[judge] shard ${PID_SHARDS[$index]} failed" >&2
      failed=1
    fi
  done
  PIDS=()
  PID_SHARDS=()
  return "$failed"
}

for ((shard_index = 0; shard_index < EFFECTIVE_SHARDS; shard_index++)); do
  run_shard "$shard_index" &
  pid=$!
  PIDS+=("$pid")
  PID_SHARDS+=("$(printf "%03d" "$shard_index")")
  if [[ ${#PIDS[@]} -ge "$MAX_PARALLEL" ]]; then
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

for ((shard_index = 0; shard_index < EFFECTIVE_SHARDS; shard_index++)); do
  shard_tag=$(printf "%03d" "$shard_index")
  "$PYTHON" "$WORKFLOW_UTILS" validate-judge-shard \
    --csv "$OUT_DIR/judge_input_${shard_tag}.csv" \
    --output-json "$OUT_DIR/judge_output_${shard_tag}.json" \
    --judge-dir "$JUDGE_DIR" --judge-script "$JUDGE_SCRIPT" \
    --run-id "$JUDGE_IDENTITY" > /dev/null
done

"$PYTHON" "$WORKFLOW_UTILS" merge-judge \
  --csv "$CSV" --split-manifest "$OUT_DIR/manifest.json" \
  --output-json "$OUTPUT" --judge-dir "$JUDGE_DIR" \
  --judge-script "$JUDGE_SCRIPT" --run-id "$JUDGE_IDENTITY" > /dev/null
"$PYTHON" "$WORKFLOW_UTILS" validate-merged-judge \
  --csv "$CSV" --output-json "$OUTPUT" --shards "$SHARDS" \
  --judge-dir "$JUDGE_DIR" --judge-script "$JUDGE_SCRIPT" \
  --run-id "$JUDGE_IDENTITY" > /dev/null
echo "[judge] JUDGE COMPLETE: $OUTPUT"
