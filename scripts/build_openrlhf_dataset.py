#!/usr/bin/env python3
"""Convert the existing ECG QA parquet into OpenRLHF prompt-dataset format.

OpenRLHF expects JSONL with at minimum a `prompt` field (string or chat list)
and a `label` field. We pack (category, ground_truth) into the label as JSON
so our custom reward function (services/openrlhf_judge_reward.py) can recover
both at scoring time. The ECG signal path goes into a separate `signal_path`
column that our custom pre-encoder (services/ecg_preencode.py) will resolve
into soft tokens before vLLM rollout.

Usage:
    python scripts/build_openrlhf_dataset.py \
        --input output/combined_train_qa_m5000k_h5000k_weighted.parquet \
        --output data/openrlhf_train_v1.jsonl \
        --max_rows 5000 \
        --balance_categories
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd


def build(args):
    df = pd.read_parquet(args.input)
    print(f"[build] input: {len(df)} rows from {args.input}")
    print(f"[build] columns: {df.columns.tolist()[:10]}...")

    for col in ("waveform_path_psa", "prompt", "generated_answer", "prompt_category"):
        if col not in df.columns:
            raise SystemExit(f"missing column: {col}")

    if args.balance_categories:
        cats = df["prompt_category"].unique()
        per_cat = max(1, args.max_rows // max(1, len(cats)))
        parts = []
        for c in cats:
            sub = df[df["prompt_category"] == c]
            if len(sub) > per_cat:
                sub = sub.sample(n=per_cat, random_state=args.seed)
            parts.append(sub)
        df = pd.concat(parts, ignore_index=True)
        print(f"[build] balanced to {per_cat} per cat -> {len(df)} rows")

    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
        print(f"[build] capped to {len(df)} rows")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    with open(out_path, "w") as f:
        for _, r in df.iterrows():
            cat = str(r["prompt_category"])
            counts[cat] += 1
            row = {
                "prompt": str(r["prompt"]),
                "label": json.dumps({
                    "category": cat,
                    "ground_truth": str(r["generated_answer"]),
                }),
                "signal_path": str(r["waveform_path_psa"]),
            }
            f.write(json.dumps(row) + "\n")

    print(f"[build] wrote {sum(counts.values())} rows -> {out_path}")
    print("[build] per-category breakdown:")
    for c, n in counts.most_common():
        print(f"  {c}: {n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_rows", type=int, default=5000)
    ap.add_argument("--balance_categories", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(args)
