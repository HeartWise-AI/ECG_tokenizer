#!/usr/bin/env bash
# Resumable full-test generation for ANY checkpoint, spread across 3 GPUs.
# Usage: run_fulltest_gen_any.sh <checkpoint> <tag>
# Skips shards whose output CSV already exists; scores deterministically at the end.
set -u
cd /volume/ECG_tokenizer
CKPT="${1:?checkpoint required}"
TAG="${2:?tag required}"
OUTDIR=/volume/ECG_tokenizer/analysis/x1split_judge

# 8 shards over 3 GPUs, 3 concurrent (one per GPU), so a shard never shares a device.
gpu_for() { echo $(( $1 % 3 )); }
for group in "0 1 2" "3 4 5" "6 7"; do
  pids=()
  for i in $group; do
    out="$OUTDIR/${TAG}_fulltest_p$i.csv"
    if [ -s "$out" ]; then echo "[fulltest:$TAG] p$i exists, skipping"; continue; fi
    g=$(gpu_for "$i")
    PYTHONPATH=/volume/ECG_tokenizer .venv/bin/python scripts/eval_judge_csv.py \
      --checkpoint "$CKPT" \
      --subset_parquet "$OUTDIR/fulltest_p$i.parquet" \
      --output_csv "$out" \
      --device "cuda:$g" --batch_size 16 --max_new_tokens 96 \
      > logs/${TAG}_fulltest_p$i.log 2>&1 &
    pids+=($!)
    echo "[fulltest:$TAG] launched p$i on cuda:$g pid ${pids[-1]}"
  done
  [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}" 2>/dev/null
done

n=$(ls "$OUTDIR/${TAG}_fulltest_p"*.csv 2>/dev/null | wc -l)
if [ "$n" -ne 8 ]; then echo "[fulltest:$TAG] ONLY $n/8 SHARDS PRESENT — aborting"; exit 1; fi
echo "[fulltest:$TAG] all 8 shards done — merging + scoring"
.venv/bin/python - "$OUTDIR" "$TAG" <<'PY'
import glob, sys, pandas as pd
outdir, tag = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(f"{outdir}/{tag}_fulltest_p*.csv"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv(f"{outdir}/{tag}_fulltest_all.csv", index=False)
print(f"merged {len(df):,} rows -> {tag}_fulltest_all.csv")
PY
.venv/bin/python scripts/score_deterministic.py "$OUTDIR/${TAG}_fulltest_all.csv" --tag "${TAG}_FULLTEST" \
  2>&1 | tee "$OUTDIR/${TAG}_fulltest_deterministic.txt"
echo "[fulltest:$TAG] GENERATION+SCORING COMPLETE"
