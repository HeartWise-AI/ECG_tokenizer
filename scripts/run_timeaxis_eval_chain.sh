#!/usr/bin/env bash
# Fires when the time-axis MedGemma retrain exits: generate on the exact s3000 subset,
# then run the sharded judge. Skips a stage whose output already exists (resumable).
set -u
cd /volume/ECG_tokenizer
TRAIN_PID="${1:?training pid required}"
CKPT=/volume/ECG_tokenizer/checkpoints/ECG_Tokenizer_LLM_Finetuning/ECG_tokenizer_MedGemma/72g2pvq9_20260822-174049/best_model.pt
GEN=/volume/ECG_tokenizer/analysis/x1split_judge/timeaxis_s3000_generations.csv
JUDGE=/volume/ECG_tokenizer/analysis/x1split_judge/judge_timeaxis_s3000.json

while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 120; done
sleep 30
if [ ! -s "$CKPT" ]; then echo "[chain] FINAL CHECKPOINT MISSING at $CKPT — aborting"; exit 1; fi
echo "[chain] training done; checkpoint present. starting generation on s3000"

if [ ! -s "$GEN" ]; then
  PYTHONPATH=/volume/ECG_tokenizer .venv/bin/python scripts/eval_judge_csv.py \
    --checkpoint "$CKPT" \
    --subset_parquet /volume/ECG_tokenizer/analysis/x1split_judge/eval_subset_s3000.parquet \
    --output_csv "$GEN" --device cuda:0 --batch_size 16 --max_new_tokens 96
else
  echo "[chain] generations exist, skipping"
fi
[ -s "$GEN" ] || { echo "[chain] GENERATION FAILED — no CSV"; exit 1; }
echo "[chain] generation complete; starting judge"

CSV="$GEN" OUTPUT="$JUDGE" OUT_DIR=/volume/ECG_tokenizer/analysis/x1split_judge/timeaxis_judge_shards \
  SHARDS=16 MAX_PARALLEL=4 bash scripts/run_csv_llm_judge_sharded.sh
rc=$?
if [ $rc -ne 0 ] || [ ! -s "$JUDGE" ]; then echo "[chain] JUDGE FAILED (rc=$rc, output missing)"; exit 1; fi
echo "[chain] JUDGE COMPLETE"
