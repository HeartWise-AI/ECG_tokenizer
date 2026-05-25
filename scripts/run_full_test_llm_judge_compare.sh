#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/volume/ECG_tokenizer}
TEST_PARQUET=${TEST_PARQUET:-$ROOT/output/combined_test_qa_m25k_h25k.parquet}
BASE_CKPT=${BASE_CKPT:-/media/data1/models/ECG_Tokenizer/e4dw86nh_20251220-232839/best_model.pt}
FINAL_CKPT=${FINAL_CKPT:-$ROOT/checkpoints/grpo_corrected_regression_final_v1/best_model.pt}
BASE_DIR=${BASE_DIR:-$ROOT/analysis/rlvr_eval/full_test_e4d_baseline_grouped_128tok}
FINAL_DIR=${FINAL_DIR:-$ROOT/analysis/rlvr_eval/full_test_grpo_final_grouped_128tok}
COMPARE_DIR=${COMPARE_DIR:-$ROOT/analysis/rlvr_eval/full_test_compare_128tok}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
BATCH_SIZE=${BATCH_SIZE:-128}
GENERATION_MICROBATCH_SIZE=${GENERATION_MICROBATCH_SIZE:-1}
FLUSH_EVERY=${FLUSH_EVERY:-512}
BASE_GPU=${BASE_GPU:-2}
FINAL_GPU=${FINAL_GPU:-0}
START_STAGGER_SECONDS=${START_STAGGER_SECONDS:-180}
PYTHON=${PYTHON:-/opt/conda/bin/python}
JUDGE_DIR=${JUDGE_DIR:-/volume/LLM_JUDGE}

mkdir -p "$BASE_DIR" "$FINAL_DIR" "$COMPARE_DIR"

BASE_CSV="$BASE_DIR/generations_baseline_full_grouped_128tok.csv"
FINAL_CSV="$FINAL_DIR/generations_grpo_final_full_grouped_128tok.csv"
BASE_JUDGE="$BASE_DIR/judge_baseline_full_grouped_128tok.json"
FINAL_JUDGE="$FINAL_DIR/judge_grpo_final_full_grouped_128tok.json"

csv_complete() {
  local csv_path=$1
  [[ -s "$csv_path" ]] || return 1
  "$PYTHON" - "$csv_path" "$TEST_PARQUET" <<'PY'
import sys
import pandas as pd

csv_path, parquet_path = sys.argv[1], sys.argv[2]
try:
    got = len(pd.read_csv(csv_path))
    want = len(pd.read_parquet(parquet_path, columns=["prompt_category"]))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if got == want else 1)
PY
}

run_generation() {
  local gpu=$1
  local ckpt=$2
  local out_dir=$3
  local label=$4
  local log_path=$5
  if csv_complete "$out_dir/generations_${label}.csv"; then
    echo "[full-test] generation exists for $label; skipping"
    return 0
  fi
  echo "[full-test] starting generation for $label on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" ECG_BERT_DEVICE=cuda:0 PYTHONPATH="$ROOT" \
    "$PYTHON" -u "$ROOT/scripts/rlvr_eval_subset.py" \
      --checkpoint "$ckpt" \
      --subset_parquet "$TEST_PARQUET" \
      --output_dir "$out_dir" \
      --device cuda:0 \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --batch_size "$BATCH_SIZE" \
      --group_by_prompt \
      --generation_microbatch_size "$GENERATION_MICROBATCH_SIZE" \
      --flush_every "$FLUSH_EVERY" \
      --resume_partial \
      --label "$label" \
      > "$log_path" 2>&1
}

run_judge() {
  local csv_path=$1
  local out_json=$2
  local log_path=$3
  if [[ -s "$out_json" ]]; then
    echo "[full-test] judge exists for $out_json; skipping"
    return 0
  fi
  echo "[full-test] starting LLM judge for $csv_path"
  (cd "$JUDGE_DIR" && "$PYTHON" judge_eval.py \
    --csv "$csv_path" \
    --output "$out_json") > "$log_path" 2>&1
}

run_generation "$BASE_GPU" "$BASE_CKPT" "$BASE_DIR" "baseline_full_grouped_128tok" \
  "$BASE_DIR/run_generation.log" &
BASE_PID=$!
echo "[full-test] baseline generation pid=$BASE_PID"
echo "[full-test] waiting ${START_STAGGER_SECONDS}s before starting final generation"
sleep "$START_STAGGER_SECONDS"
run_generation "$FINAL_GPU" "$FINAL_CKPT" "$FINAL_DIR" "grpo_final_full_grouped_128tok" \
  "$FINAL_DIR/run_generation.log" &
FINAL_PID=$!

echo "[full-test] final generation pid=$FINAL_PID"
wait "$BASE_PID"
wait "$FINAL_PID"

run_judge "$BASE_CSV" "$BASE_JUDGE" "$BASE_DIR/run_judge.log"
run_judge "$FINAL_CSV" "$FINAL_JUDGE" "$FINAL_DIR/run_judge.log"

MAX_NEW_TOKENS="$MAX_NEW_TOKENS" BATCH_SIZE="$BATCH_SIZE" \
GENERATION_MICROBATCH_SIZE="$GENERATION_MICROBATCH_SIZE" \
PYTHONPATH="$ROOT" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

import pandas as pd

root = Path("/volume/ECG_tokenizer")
base_dir = root / "analysis/rlvr_eval/full_test_e4d_baseline_grouped_128tok"
final_dir = root / "analysis/rlvr_eval/full_test_grpo_final_grouped_128tok"
compare_dir = root / "analysis/rlvr_eval/full_test_compare_128tok"
compare_dir.mkdir(parents=True, exist_ok=True)

def load_summary(path: Path) -> dict:
    with path.open() as f:
        data = json.load(f)
    agg = data.get("aggregates", data)
    return {
        "overall_score": float(agg["overall_score"]),
        "category_aggregates": {
            k: {"count": v.get("count"), "mean_score": v.get("mean_score")}
            for k, v in agg.get("category_aggregates", {}).items()
        },
    }

base = load_summary(base_dir / "judge_baseline_full_grouped_128tok.json")
final = load_summary(final_dir / "judge_grpo_final_full_grouped_128tok.json")

rows = []
cats = sorted(set(base["category_aggregates"]) | set(final["category_aggregates"]))
for cat in cats:
    b = base["category_aggregates"].get(cat, {})
    f = final["category_aggregates"].get(cat, {})
    b_score = b.get("mean_score")
    f_score = f.get("mean_score")
    rows.append({
        "prompt_category": cat,
        "count_baseline": b.get("count"),
        "count_final": f.get("count"),
        "baseline_mean": b_score,
        "final_mean": f_score,
        "delta": None if b_score is None or f_score is None else f_score - b_score,
    })

comparison = {
    "test_parquet": str(root / "output/combined_test_qa_m25k_h25k.parquet"),
    "generation_mode": "group_by_prompt",
    "max_new_tokens": int(os.environ.get("MAX_NEW_TOKENS", "128")),
    "batch_size": int(os.environ.get("BATCH_SIZE", "128")),
    "generation_microbatch_size": int(os.environ.get("GENERATION_MICROBATCH_SIZE", "8")),
    "baseline": base,
    "final": final,
    "overall_delta": final["overall_score"] - base["overall_score"],
}

(compare_dir / "comparison_summary.json").write_text(json.dumps(comparison, indent=2))
pd.DataFrame(rows).to_csv(compare_dir / "comparison_by_category.csv", index=False)
print(json.dumps({
    "baseline_overall": base["overall_score"],
    "final_overall": final["overall_score"],
    "overall_delta": comparison["overall_delta"],
    "comparison_dir": str(compare_dir),
}, indent=2))
PY

echo "[full-test] complete"
