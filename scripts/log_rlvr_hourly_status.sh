#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/volume/ECG_tokenizer}"
cd "$ROOT"

echo "============================================================"
date -Is
echo "cwd=$PWD"

echo
echo "[gpu]"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

echo
echo "[grpo_processes]"
ps -eo pid,stat,etime,pcpu,pmem,args | rg 'grpo_openrlhf_v1.py|build_grpo_hard_mined_dataset.py|grpo_bert_' || true

echo
echo "[latest_hardmix_artifacts]"
find checkpoints/grpo_bert_hardmix_70_20_10_v1 analysis/rlvr_eval/run_logs analysis/rlvr_eval/hard_mining \
  -maxdepth 2 -type f \
  \( -name '*hardmix*' -o -name 'summary_*.json' -o -name 'metrics.json' \) \
  -printf '%TY-%Tm-%Td %TH:%TM %p %s\n' 2>/dev/null | sort | tail -40 || true

echo
echo "[hardmix_scores]"
python - <<'PY' || true
import json
from pathlib import Path

base = Path("checkpoints/grpo_bert_hardmix_70_20_10_v1")
for name in ("eval_step60", "eval_step120", "eval_final"):
    p = base / name / f"summary_{name.replace('eval_', '')}.json"
    if p.exists():
        data = json.loads(p.read_text())
        print(f"{name}: overall={data.get('overall_score')}")
PY
