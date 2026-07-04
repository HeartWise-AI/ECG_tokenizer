#!/usr/bin/env python3
"""Build the RFT (rejection-sampling fine-tune) parquet from the
best-of-5 generations CSV.

Inputs:
  - generations_bestof5_1k.csv  (700-row run, judge-scored)
  - generations_bestof5.csv     (170-row run, judge-scored)  [optional, for more data]

Output:
  data/rft_bestof5_v1.parquet with columns:
    waveform_path_psa, prompt, generated_answer, prompt_category, bestof_picked_score

Filtering rules:
  - bestof_picked_score >= threshold (default 0.7)
  - generation is non-empty and not an error sentinel
  - drop duplicates by (waveform_path, prompt)
"""

import argparse
from pathlib import Path

import pandas as pd


def load_one(csv_path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["__source"] = label
    return df


def build(args):
    frames = []
    for src in args.csv:
        p = Path(src)
        if not p.exists():
            print(f"[rft] skip missing: {p}")
            continue
        df = load_one(p, p.parent.name)
        print(f"[rft] {p.name}: {len(df)} rows")
        frames.append(df)

    if not frames:
        raise SystemExit("no input CSVs found")

    df = pd.concat(frames, ignore_index=True)
    print(f"[rft] combined: {len(df)} rows")

    # Filter on judge score
    before = len(df)
    df = df[df["bestof_picked_score"].astype(float) >= args.min_score]
    print(f"[rft] kept {len(df)}/{before} after score >= {args.min_score}")

    # Filter error sentinels and empty generations
    df = df[~df["generation"].astype(str).str.startswith("[ERROR")]
    df = df[df["generation"].astype(str).str.strip().str.len() > 0]
    print(f"[rft] kept {len(df)} after non-error/non-empty filter")

    # Dedupe
    df = df.drop_duplicates(subset=["waveform_path", "question"], keep="first")
    print(f"[rft] kept {len(df)} after dedupe")

    # Exclude held-out eval rows
    if args.exclude_parquet:
        ex = pd.read_parquet(args.exclude_parquet)
        excl_keys = set(zip(ex["waveform_path_psa"].astype(str),
                            ex["prompt"].astype(str)))
        mask = [(p, q) not in excl_keys
                for p, q in zip(df["waveform_path"].astype(str),
                                df["question"].astype(str))]
        before = len(df)
        df = df[mask]
        print(f"[rft] kept {len(df)}/{before} after excluding "
              f"{len(excl_keys)} held-out eval rows")

    # Rename to schema the SFT loader expects
    out = pd.DataFrame({
        "waveform_path_psa": df["waveform_path"].astype(str),
        "prompt": df["question"].astype(str),
        "generated_answer": df["generation"].astype(str),
        "prompt_category": df["prompt_category"].astype(str),
        "bestof_picked_score": df["bestof_picked_score"].astype(float),
    })

    # Per-category breakdown
    print("\n[rft] per-category counts:")
    print(out.groupby("prompt_category").size().sort_values(ascending=False).to_string())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"\n[rft] saved {len(out)} rows -> {out_path}")

    # Sanity preview
    print("\n[rft] first 3 rows:")
    for i in range(min(3, len(out))):
        r = out.iloc[i]
        print(f"  [{i}] cat={r['prompt_category']} score={r['bestof_picked_score']:.2f}")
        print(f"      prompt: {r['prompt'][:100]}")
        print(f"      target: {r['generated_answer'][:120]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True,
                    help="One or more generations CSVs from best-of-N runs")
    ap.add_argument("--min_score", type=float, default=0.7)
    ap.add_argument("--exclude_parquet", default=None,
                    help="Parquet of rows to EXCLUDE (held-out eval subset). "
                         "Match on (waveform_path_psa, prompt).")
    ap.add_argument("--output", default="data/rft_bestof5_v1.parquet")
    args = ap.parse_args()
    build(args)
