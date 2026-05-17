#!/usr/bin/env python3
"""Validate the deterministic verifier against the LLM-judge across ALL
available judge-scored generation CSVs. Reports overall correlation, mean
absolute error, and per-category disagreement so we can iterate the
verifier until it tracks the judge.

Usage:
    python scripts/validate_verifier_vs_judge.py
"""
import glob
import sys
import pandas as pd

sys.path.insert(0, "/volume/ECG_tokenizer")
from services.verifiable_reward import verify

CSVS = sorted(set(glob.glob(
    "/volume/ECG_tokenizer/checkpoints/grpo_openrlhf_*/**/judge_*_enhanced.csv",
    recursive=True)) | {
    "/tmp/baseline_today/judge_baseline_today_enhanced.csv",
})


def main():
    frames = []
    for c in CSVS:
        try:
            d = pd.read_csv(c)
        except Exception:
            continue
        if "overall_score" not in d.columns or "prompt_category" not in d.columns:
            continue
        need = {"generation", "ground_truth", "prompt_category", "overall_score"}
        if not need.issubset(d.columns):
            continue
        d = d[list(need)].copy()
        d["src"] = c.split("/")[-1]
        frames.append(d)

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["overall_score"])
    df["judge"] = pd.to_numeric(df["overall_score"], errors="coerce")
    df = df.dropna(subset=["judge"])
    df["verifier"] = df.apply(
        lambda r: verify(str(r["generation"]), str(r["ground_truth"]),
                         str(r["prompt_category"])), axis=1)
    df["absdiff"] = (df["verifier"] - df["judge"]).abs()

    n = len(df)
    corr = df[["judge", "verifier"]].corr().iloc[0, 1]
    mae = df["absdiff"].mean()
    print(f"Samples: {n}  (from {len(frames)} judge CSVs)")
    print(f"Overall correlation : {corr:.4f}")
    print(f"Mean |verifier-judge|: {mae:.4f}")
    print(f"Mean judge={df['judge'].mean():.4f}  mean verifier={df['verifier'].mean():.4f}")
    print(f"\n{'category':32s} {'n':>5} {'judge':>7} {'verif':>7} {'MAE':>7} {'corr':>6}")
    rows = []
    for cat, g in df.groupby("prompt_category"):
        if len(g) < 3:
            continue
        try:
            cc = g[["judge", "verifier"]].corr().iloc[0, 1]
        except Exception:
            cc = float("nan")
        rows.append((cat, len(g), g["judge"].mean(), g["verifier"].mean(),
                     g["absdiff"].mean(), cc))
    for cat, ln, j, v, m, cc in sorted(rows, key=lambda x: -x[4]):
        print(f"{cat:32s} {ln:5d} {j:7.3f} {v:7.3f} {m:7.3f} {cc:6.2f}")


if __name__ == "__main__":
    main()
