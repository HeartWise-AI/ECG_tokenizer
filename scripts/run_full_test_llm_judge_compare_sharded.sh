#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/volume/ECG_tokenizer}
TEST_PARQUET=${TEST_PARQUET:-$ROOT/output/combined_test_qa_m25k_h25k.parquet}
BASE_CKPT=${BASE_CKPT:-/media/data1/models/ECG_Tokenizer/e4dw86nh_20251220-232839/best_model.pt}
FINAL_CKPT=${FINAL_CKPT:-$ROOT/checkpoints/grpo_corrected_regression_final_v1/best_model.pt}
BASE_DIR=${BASE_DIR:-$ROOT/analysis/rlvr_eval/full_test_e4d_baseline_sharded_128tok}
FINAL_DIR=${FINAL_DIR:-$ROOT/analysis/rlvr_eval/full_test_grpo_final_sharded_128tok}
COMPARE_DIR=${COMPARE_DIR:-$ROOT/analysis/rlvr_eval/full_test_compare_sharded_128tok}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
BATCH_SIZE=${BATCH_SIZE:-8}
GENERATION_MICROBATCH_SIZE=${GENERATION_MICROBATCH_SIZE:-1}
FLUSH_EVERY=${FLUSH_EVERY:-256}
SHARDS_PER_MODEL=${SHARDS_PER_MODEL:-12}
GPUS=${GPUS:-0,2}
JOBS_PER_GPU=${JOBS_PER_GPU:-6}
PYTHON=${PYTHON:-/opt/conda/bin/python}
JUDGE_DIR=${JUDGE_DIR:-/volume/LLM_JUDGE}

IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
if [[ ${#GPU_ARRAY[@]} -eq 0 ]]; then
  echo "[full-test-sharded] GPUS is empty" >&2
  exit 1
fi
MAX_PARALLEL=${MAX_PARALLEL:-$(( ${#GPU_ARRAY[@]} * JOBS_PER_GPU ))}

mkdir -p "$BASE_DIR" "$FINAL_DIR" "$COMPARE_DIR"
SHARD_DIR="$COMPARE_DIR/shards"

BASE_CSV="$BASE_DIR/generations_baseline_full_sharded_128tok.csv"
FINAL_CSV="$FINAL_DIR/generations_grpo_final_full_sharded_128tok.csv"
BASE_JUDGE="$BASE_DIR/judge_baseline_full_sharded_128tok.json"
FINAL_JUDGE="$FINAL_DIR/judge_grpo_final_full_sharded_128tok.json"

echo "[full-test-sharded] root=$ROOT"
echo "[full-test-sharded] test_parquet=$TEST_PARQUET"
echo "[full-test-sharded] shards=$SHARDS_PER_MODEL gpus=$GPUS max_parallel=$MAX_PARALLEL micro=$GENERATION_MICROBATCH_SIZE"

"$PYTHON" - "$TEST_PARQUET" "$SHARD_DIR" "$SHARDS_PER_MODEL" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

test_parquet = Path(sys.argv[1])
shard_dir = Path(sys.argv[2])
n_shards = int(sys.argv[3])
shard_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(test_parquet)
df = df.copy()
df["source_row_idx"] = np.arange(len(df), dtype=np.int64)

manifest = {
    "test_parquet": str(test_parquet),
    "rows": int(len(df)),
    "shards": int(n_shards),
    "items": [],
}
for shard_idx in range(n_shards):
    part = df.iloc[np.arange(shard_idx, len(df), n_shards)].reset_index(drop=True)
    path = shard_dir / f"shard_{shard_idx:03d}.parquet"
    part.to_parquet(path, index=False)
    manifest["items"].append({"shard_idx": shard_idx, "path": str(path), "rows": int(len(part))})

(shard_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps({"rows": len(df), "shards": n_shards, "shard_dir": str(shard_dir)}, indent=2))
PY

csv_complete() {
  local csv_path=$1
  local parquet_path=$2
  [[ -s "$csv_path" ]] || return 1
  "$PYTHON" - "$csv_path" "$parquet_path" <<'PY'
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

run_generation_shard() {
  local model_name=$1
  local gpu=$2
  local ckpt=$3
  local out_dir=$4
  local shard_idx=$5
  local shard_tag
  shard_tag=$(printf "%03d" "$shard_idx")
  local label="${model_name}_shard_${shard_tag}"
  local shard_path="$SHARD_DIR/shard_${shard_tag}.parquet"
  local log_path="$out_dir/run_generation_shard_${shard_tag}.log"
  local csv_path="$out_dir/generations_${label}.csv"

  if csv_complete "$csv_path" "$shard_path"; then
    echo "[full-test-sharded] $model_name shard $shard_tag complete; skipping"
    return 0
  fi

  echo "[full-test-sharded] starting $model_name shard $shard_tag on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" ECG_BERT_DEVICE=cuda:0 PYTHONPATH="$ROOT" \
    "$PYTHON" -u "$ROOT/scripts/rlvr_eval_subset.py" \
      --checkpoint "$ckpt" \
      --subset_parquet "$shard_path" \
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

merge_model_csvs() {
  local model_name=$1
  local out_dir=$2
  local merged_csv=$3
  "$PYTHON" - "$TEST_PARQUET" "$out_dir" "$model_name" "$SHARDS_PER_MODEL" "$merged_csv" <<'PY'
import sys
from pathlib import Path

import pandas as pd

test_parquet = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
model_name = sys.argv[3]
n_shards = int(sys.argv[4])
merged_csv = Path(sys.argv[5])

frames = []
for shard_idx in range(n_shards):
    path = out_dir / f"generations_{model_name}_shard_{shard_idx:03d}.csv"
    if not path.exists():
        raise SystemExit(f"missing shard CSV: {path}")
    frame = pd.read_csv(path)
    if "source_row_idx" not in frame.columns:
        raise SystemExit(f"missing source_row_idx in {path}")
    frames.append(frame)

df = pd.concat(frames, ignore_index=True)
want = len(pd.read_parquet(test_parquet, columns=["prompt_category"]))
if len(df) != want:
    raise SystemExit(f"row count mismatch: got {len(df)} want {want}")
if df["source_row_idx"].duplicated().any():
    dup = df.loc[df["source_row_idx"].duplicated(), "source_row_idx"].iloc[0]
    raise SystemExit(f"duplicate source_row_idx: {dup}")
missing = set(range(want)) - set(df["source_row_idx"].astype(int).tolist())
if missing:
    raise SystemExit(f"missing source rows: first={min(missing)} count={len(missing)}")

df = df.sort_values("source_row_idx").reset_index(drop=True)
df.to_csv(merged_csv, index=False)
print(f"[full-test-sharded] merged {model_name}: {merged_csv} rows={len(df)}")
PY
}

run_model() {
  local model_name=$1
  local ckpt=$2
  local out_dir=$3
  local merged_csv=$4

  mkdir -p "$out_dir"
  if csv_complete "$merged_csv" "$TEST_PARQUET"; then
    echo "[full-test-sharded] merged generation exists for $model_name; skipping shards"
    return 0
  fi

  local shard_idx
  for shard_idx in $(seq 0 $((SHARDS_PER_MODEL - 1))); do
    while [[ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]]; do
      wait -n
    done
    local gpu=${GPU_ARRAY[$((shard_idx % ${#GPU_ARRAY[@]}))]}
    run_generation_shard "$model_name" "$gpu" "$ckpt" "$out_dir" "$shard_idx" &
  done
  wait
  merge_model_csvs "$model_name" "$out_dir" "$merged_csv"
}

run_judge() {
  local csv_path=$1
  local out_json=$2
  local log_path=$3
  if [[ -s "$out_json" ]]; then
    echo "[full-test-sharded] judge exists for $out_json; skipping"
    return 0
  fi
  echo "[full-test-sharded] starting LLM judge for $csv_path"
  (cd "$JUDGE_DIR" && "$PYTHON" judge_eval.py \
    --csv "$csv_path" \
    --output "$out_json") > "$log_path" 2>&1
}

run_model "baseline" "$BASE_CKPT" "$BASE_DIR" "$BASE_CSV"
run_model "grpo_final" "$FINAL_CKPT" "$FINAL_DIR" "$FINAL_CSV"

run_judge "$BASE_CSV" "$BASE_JUDGE" "$BASE_DIR/run_judge.log"
run_judge "$FINAL_CSV" "$FINAL_JUDGE" "$FINAL_DIR/run_judge.log"

TEST_PARQUET="$TEST_PARQUET" BASE_DIR="$BASE_DIR" FINAL_DIR="$FINAL_DIR" COMPARE_DIR="$COMPARE_DIR" \
MAX_NEW_TOKENS="$MAX_NEW_TOKENS" BATCH_SIZE="$BATCH_SIZE" \
GENERATION_MICROBATCH_SIZE="$GENERATION_MICROBATCH_SIZE" SHARDS_PER_MODEL="$SHARDS_PER_MODEL" \
PYTHONPATH="$ROOT" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

import pandas as pd

root = Path("/volume/ECG_tokenizer")
test_parquet = Path(os.environ["TEST_PARQUET"])
base_dir = Path(os.environ["BASE_DIR"])
final_dir = Path(os.environ["FINAL_DIR"])
compare_dir = Path(os.environ["COMPARE_DIR"])
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

base = load_summary(base_dir / "judge_baseline_full_sharded_128tok.json")
final = load_summary(final_dir / "judge_grpo_final_full_sharded_128tok.json")

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
    "test_parquet": str(test_parquet),
    "generation_mode": "sharded_group_by_prompt",
    "max_new_tokens": int(os.environ.get("MAX_NEW_TOKENS", "128")),
    "batch_size": int(os.environ.get("BATCH_SIZE", "8")),
    "generation_microbatch_size": int(os.environ.get("GENERATION_MICROBATCH_SIZE", "1")),
    "shards_per_model": int(os.environ.get("SHARDS_PER_MODEL", "12")),
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

echo "[full-test-sharded] complete"
