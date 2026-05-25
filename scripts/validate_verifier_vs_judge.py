#!/usr/bin/env python3
"""Validate the deterministic verifier against available LLM-judge artifacts.

Reports overall correlation, mean absolute error, prefix-artifact diagnostics,
and per-category disagreement so we can iterate the verifier until it tracks
the judge.

Usage:
    python scripts/validate_verifier_vs_judge.py
"""
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/volume/ECG_tokenizer")
from services.verifiable_reward import _strip_prefix, verify

ROOT = Path("/volume/ECG_tokenizer")

CSV_GLOBS = [
    ROOT / "analysis/rlvr_eval/**/*enhanced.csv",
    ROOT / "checkpoints/grpo_openrlhf_*/*/*enhanced.csv",
    ROOT / "checkpoints/grpo_openrlhf_*/*enhanced.csv",
]

REQUIRED_COLUMNS = {"generation", "ground_truth", "prompt_category", "overall_score"}


def discover_csvs():
    paths = set()
    for pat in CSV_GLOBS:
        paths.update(glob.glob(str(pat), recursive=True))
    return sorted(paths)


def corr(g: pd.DataFrame) -> float:
    if len(g) < 3 or g["judge"].std() == 0 or g["verifier"].std() == 0:
        return float("nan")
    return float(g[["judge", "verifier"]].corr().iloc[0, 1])


def main():
    frames = []
    for c in discover_csvs():
        try:
            d = pd.read_csv(c)
        except Exception:
            continue
        if not REQUIRED_COLUMNS.issubset(d.columns):
            continue
        d = d[list(REQUIRED_COLUMNS)].copy()
        d["src"] = str(Path(c).relative_to(ROOT))
        parts = Path(c).relative_to(ROOT).parts
        d["run"] = parts[1] if parts[0] == "analysis" else parts[1]
        frames.append(d)
    if not frames:
        raise SystemExit("No judge enhanced CSVs found")

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["overall_score", "generation", "ground_truth", "prompt_category"])
    df["judge"] = pd.to_numeric(df["overall_score"], errors="coerce")
    df = df.dropna(subset=["judge"])
    df["verifier"] = df.apply(
        lambda r: verify(str(r["generation"]), str(r["ground_truth"]),
                         str(r["prompt_category"])), axis=1)
    df["absdiff"] = (df["verifier"] - df["judge"]).abs()
    df["prefix_artifact"] = df["generation"].astype(str).map(lambda s: _strip_prefix(s)[1])

    n = len(df)
    mae = df["absdiff"].mean()
    print(f"Samples: {n}  (from {len(frames)} judge CSVs)")
    print(f"Overall correlation : {corr(df):.4f}")
    print(f"Mean |verifier-judge|: {mae:.4f}")
    print(f"Mean judge={df['judge'].mean():.4f}  mean verifier={df['verifier'].mean():.4f}")

    prefix = df[df["prefix_artifact"]]
    print(
        f"Prefix artifacts: {len(prefix)} ({df['prefix_artifact'].mean() * 100:.2f}%)"
    )
    if len(prefix):
        print(
            "Prefix subset: "
            f"judge={prefix['judge'].mean():.3f} "
            f"verifier={prefix['verifier'].mean():.3f} "
            f"MAE={prefix['absdiff'].mean():.3f} "
            f"corr={corr(prefix):.3f}"
        )

    print(f"\n{'run':28s} {'n':>5} {'judge':>7} {'verif':>7} {'MAE':>7} {'corr':>6} {'prefix%':>8}")
    rows = []
    for run, g in df.groupby("run"):
        rows.append((run, len(g), g["judge"].mean(), g["verifier"].mean(),
                     g["absdiff"].mean(), corr(g), g["prefix_artifact"].mean() * 100))
    for run, ln, j, v, m, cc, pp in sorted(rows, key=lambda x: x[0]):
        print(f"{run:28s} {ln:5d} {j:7.3f} {v:7.3f} {m:7.3f} {cc:6.2f} {pp:8.1f}")

    print(f"\n{'category':32s} {'n':>5} {'judge':>7} {'verif':>7} {'MAE':>7} {'corr':>6}")
    rows = []
    for cat, g in df.groupby("prompt_category"):
        if len(g) < 3:
            continue
        cc = corr(g)
        rows.append((cat, len(g), g["judge"].mean(), g["verifier"].mean(),
                     g["absdiff"].mean(), cc))
    for cat, ln, j, v, m, cc in sorted(rows, key=lambda x: -x[4]):
        print(f"{cat:32s} {ln:5d} {j:7.3f} {v:7.3f} {m:7.3f} {cc:6.2f}")


if __name__ == "__main__":
    main()
