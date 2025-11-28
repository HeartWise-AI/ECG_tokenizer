#!/usr/bin/env python3
"""
Generate Choice-Free (CF) datasets for MIMIC and MHI, for both TRAIN and TEST.

Supports two input modes:
  1) Separate parquets for each dataset/split (recommended)
     --mimic_train ... --mimic_test ... --mhi_train ... --mhi_test ...

  2) One combined parquet per split (must contain a 'dataset' column with
     values 'mimic' or 'mhi')
     --combined_train ... --combined_test ...

Outputs (by default, parquet):
  <output_root>/mimic/train_cf.parquet
  <output_root>/mimic/test_cf.parquet
  <output_root>/mhi/train_cf.parquet
  <output_root>/mhi/test_cf.parquet

Example (combined inputs):
  python scripts/generate_cf_for_mimic_mhi.py \
    --combined_train /path/train_combined.parquet \
    --combined_test  /path/test_combined.parquet \
    --output_root ecg_cf_eval \
    --normal_pct 0.05 \
    --save_format parquet

Example (separate inputs):
  python scripts/generate_cf_for_mimic_mhi.py \
    --mimic_train /path/mimic_train.parquet --mimic_test /path/mimic_test.parquet \
    --mhi_train   /path/mhi_train.parquet   --mhi_test   /path/mhi_test.parquet \
    --output_root ecg_cf_eval --normal_pct 0.05 --save_format parquet
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure repo root is on sys.path when running as a script
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from utils.generate_ecg_cf_eval_dataset import generate_cf_eval_dataset


def _split_combined(parquet_path: str, out_dir: str, prefix: str) -> tuple[str, str]:
    """Split a combined parquet into mimic and mhi parquets under out_dir.

    Returns: (mimic_path, mhi_path)
    """
    df = pd.read_parquet(parquet_path)
    if 'dataset' not in df.columns:
        raise ValueError("Combined parquet must contain a 'dataset' column with values 'mimic' or 'mhi'.")
    out_mimic = os.path.join(out_dir, f"{prefix}_mimic.parquet")
    out_mhi = os.path.join(out_dir, f"{prefix}_mhi.parquet")
    df[df['dataset'].astype(str).str.lower() == 'mimic'].to_parquet(out_mimic, index=False)
    df[df['dataset'].astype(str).str.lower() == 'mhi'].to_parquet(out_mhi, index=False)
    return out_mimic, out_mhi


def _ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate CF datasets for MIMIC & MHI (train + test)")
    # Separate inputs
    ap.add_argument("--mimic_train", type=str, default=None)
    ap.add_argument("--mimic_test", type=str, default=None)
    ap.add_argument("--mhi_train", type=str, default=None)
    ap.add_argument("--mhi_test", type=str, default=None)
    # Combined inputs
    ap.add_argument("--combined_train", type=str, default=None)
    ap.add_argument("--combined_test", type=str, default=None)
    # Common config
    ap.add_argument("--output_root", type=str, default="ecg_cf_eval")
    ap.add_argument("--normal_pct", type=float, default=0.05)
    ap.add_argument("--save_format", type=str, default="parquet", choices=["json", "parquet"])
    ap.add_argument("--max_ecgs_train", type=int, default=0, help="Limit rows when building train CF (0=all)")
    ap.add_argument("--max_ecgs_test", type=int, default=0, help="Limit rows when building test CF (0=all)")
    args = ap.parse_args()

    _ensure_dir(args.output_root)
    temp_dir = os.path.join(args.output_root, "_tmp_splits")
    _ensure_dir(temp_dir)

    # Resolve per-dataset paths
    mimic_train = args.mimic_train
    mimic_test = args.mimic_test
    mhi_train = args.mhi_train
    mhi_test = args.mhi_test

    if (args.combined_train is not None) or (args.combined_test is not None):
        if not args.combined_train or not args.combined_test:
            raise ValueError("Provide both --combined_train and --combined_test or neither.")
        mimic_train, mhi_train = _split_combined(args.combined_train, temp_dir, prefix="train")
        mimic_test, mhi_test = _split_combined(args.combined_test, temp_dir, prefix="test")

    for name, train_path, test_path in (
        ("mimic", mimic_train, mimic_test),
        ("mhi", mhi_train, mhi_test),
    ):
        if not train_path or not test_path:
            print(f"Skipping {name}: missing train or test path")
            continue

        out_dir = os.path.join(args.output_root, name)
        _ensure_dir(out_dir)

        print(f"\n=== {name.upper()} : TRAIN ===")
        generate_cf_eval_dataset(
            test_path=train_path,
            output_path=out_dir,
            normal_pct=args.normal_pct,
            save_format=args.save_format,
            split_name="train",
            max_ecgs=(args.max_ecgs_train if args.max_ecgs_train and args.max_ecgs_train > 0 else None),
        )

        print(f"\n=== {name.upper()} : TEST ===")
        generate_cf_eval_dataset(
            test_path=test_path,
            output_path=out_dir,
            normal_pct=args.normal_pct,
            save_format=args.save_format,
            split_name="test",
            max_ecgs=(args.max_ecgs_test if args.max_ecgs_test and args.max_ecgs_test > 0 else None),
        )

    # Optional: cleanup temp splits
    try:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
