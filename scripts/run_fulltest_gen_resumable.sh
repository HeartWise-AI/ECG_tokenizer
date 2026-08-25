#!/usr/bin/env bash
# Resumable full-test generation for the concat_linear model on GPU 2.
# Runs 8 shards two-at-a-time; skips shards whose output CSV already exists.
set -u
cd /volume/ECG_tokenizer
CKPT=checkpoints/ECG_Tokenizer_LLM_Finetuning/ECG_tokenizer_MedGemma/hcjs6vk9_20260814-202915/best_model.pt
for pair in "0 1" "2 3" "4 5" "6 7"; do
  pids=()
  for i in $pair; do
    out=analysis/x1split_judge/concatmix_fulltest_p$i.csv
    if [ -s "$out" ]; then echo "[fulltest] p$i exists, skipping"; continue; fi
    PYTHONPATH=/volume/ECG_tokenizer .venv/bin/python scripts/eval_judge_csv.py \
      --checkpoint "$CKPT" \
      --subset_parquet analysis/x1split_judge/fulltest_p$i.parquet \
      --output_csv "$out" \
      --device cuda:2 --batch_size 16 --max_new_tokens 96 > logs/fulltest_p$i.log 2>&1 &
    pids+=($!)
    echo "[fulltest] launched p$i pid ${pids[-1]}"
  done
  wait "${pids[@]}" 2>/dev/null
done
echo "[fulltest] all shards done — scoring"
.venv/bin/python scripts/score_deterministic.py "analysis/x1split_judge/concatmix_fulltest_p*.csv" --tag concatmix_FULLTEST 2>&1 | tee analysis/x1split_judge/concatmix_fulltest_deterministic.txt
echo "[fulltest] COMPLETE"
