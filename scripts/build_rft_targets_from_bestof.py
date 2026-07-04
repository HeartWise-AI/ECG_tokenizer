#!/usr/bin/env python3
"""Build anchored RFT targets from baseline, best-of-N, and ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _txt(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def build(args: argparse.Namespace) -> None:
    best = pd.read_csv(args.bestof_csv)
    base = pd.read_csv(args.baseline_csv)
    judge = pd.read_csv(args.baseline_judge_csv)

    score_by_key = {
        (_txt(r["json_key"]), _txt(r["prompt"])): float(r.get("overall_score", 0.0) or 0.0)
        for _, r in judge.iterrows()
    }

    best_by_key = {
        (_txt(r["waveform_name"]), _txt(r["question"])): r
        for _, r in best.iterrows()
    }

    rows = []
    source_counts = {}
    for _, base_row in base.iterrows():
        key = (_txt(base_row["waveform_name"]), _txt(base_row["question"]))
        best_row = best_by_key.get(key)
        base_score = score_by_key.get(key, 0.0)
        base_gen = _txt(base_row["generation"])
        gt = _txt(base_row["ground_truth"])

        best_score = -1.0
        best_gen = ""
        if best_row is not None:
            best_score = float(best_row.get("bestof_picked_score", 0.0) or 0.0)
            best_gen = _txt(best_row.get("generation"))

        if (
            best_gen
            and not best_gen.startswith("[ERROR")
            and best_score >= args.min_best_score
            and best_score >= base_score + args.min_margin
        ):
            target = best_gen
            source = "bestof"
            weight = max(args.min_weight, min(args.max_weight, best_score - base_score))
        elif base_gen and base_score >= args.min_anchor_score:
            target = base_gen
            source = "baseline_anchor"
            weight = args.anchor_weight
        elif best_gen and not best_gen.startswith("[ERROR") and best_score >= args.min_best_score:
            target = best_gen
            source = "bestof_low_margin"
            weight = args.low_margin_weight
        else:
            target = gt
            source = "ground_truth"
            weight = args.gt_weight

        source_counts[source] = source_counts.get(source, 0) + 1
        rows.append({
            "waveform_path": _txt(base_row["waveform_path"]),
            "prompt": _txt(base_row["question"]),
            "chosen": target,
            "rejected": base_gen,
            "weight": float(weight),
            "prompt_category": _txt(base_row["prompt_category"]),
            "ground_truth": gt,
            "source": source,
            "baseline_score": base_score,
            "bestof_score": best_score,
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"[rft-targets] wrote {len(rows)} rows -> {out}")
    print(f"[rft-targets] source counts: {source_counts}")
    df = pd.DataFrame(rows)
    print("[rft-targets] per-category/source:")
    print(pd.crosstab(df["prompt_category"], df["source"]).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bestof_csv", required=True)
    parser.add_argument("--baseline_csv", required=True)
    parser.add_argument("--baseline_judge_csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_best_score", type=float, default=0.5)
    parser.add_argument("--min_margin", type=float, default=0.05)
    parser.add_argument("--min_anchor_score", type=float, default=0.7)
    parser.add_argument("--min_weight", type=float, default=0.2)
    parser.add_argument("--max_weight", type=float, default=1.0)
    parser.add_argument("--anchor_weight", type=float, default=0.5)
    parser.add_argument("--low_margin_weight", type=float, default=0.3)
    parser.add_argument("--gt_weight", type=float, default=0.5)
    build(parser.parse_args())
