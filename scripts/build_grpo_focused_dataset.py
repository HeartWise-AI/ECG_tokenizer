#!/usr/bin/env python3
"""Build a focused GRPO training set from weak categories — the ones where
the baseline LLM-judge score is < 0.5. These have the most headroom; the
verifier signal is strong here (we know the model is often wrong).

Usage:
    python scripts/build_grpo_focused_dataset.py \
        --train_parquet /volume/ECG_tokenizer/output/combined_train_qa_m5000k_h5000k_weighted.parquet \
        --baseline_json /tmp/baseline_today/summary_baseline_today.json \
        --weakness_threshold 0.5 \
        --rows_per_cat 2000 \
        --output data/grpo_focused_20k.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_parquet", required=True)
    ap.add_argument("--baseline_json", required=True)
    ap.add_argument("--weakness_threshold", type=float, default=0.5,
                    help="Only include categories where baseline < this")
    ap.add_argument("--rows_per_cat", type=int, default=2000)
    ap.add_argument("--output", default="data/grpo_focused_20k.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Find weak categories
    with open(args.baseline_json) as f:
        base = json.load(f)
    cats = base["category_aggregates"]
    weak_cats = [c for c, info in cats.items()
                 if info["mean_score"] < args.weakness_threshold and info["count"] >= 3]
    print(f"[focused] {len(weak_cats)} weak categories (baseline < {args.weakness_threshold}):")
    for c in weak_cats:
        print(f"  {c}: {cats[c]['mean_score']:.3f}")

    # Load training data
    df = pd.read_parquet(args.train_parquet)
    print(f"\n[focused] loaded {len(df)} training rows")

    # Filter to weak categories
    df = df[df["prompt_category"].isin(weak_cats)]
    print(f"[focused] {len(df)} rows in weak categories")

    # Per-category sampling
    parts = []
    for c in weak_cats:
        sub = df[df["prompt_category"] == c]
        if len(sub) == 0:
            print(f"[focused] WARN: category {c} has 0 training rows — skipping")
            continue
        n = min(len(sub), args.rows_per_cat)
        sub = sub.sample(n=n, random_state=args.seed)
        parts.append(sub)

    out_df = pd.concat(parts, ignore_index=True)
    print(f"\n[focused] final dataset: {len(out_df)} rows")

    # Write JSONL in OpenRLHF format
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with open(out_path, "w") as f:
        for _, r in out_df.iterrows():
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

    print(f"[focused] wrote {sum(counts.values())} rows -> {out_path}")
    print("[focused] per-category breakdown:")
    for c, n in counts.most_common():
        print(f"  {c}: {n}")


if __name__ == "__main__":
    main()
