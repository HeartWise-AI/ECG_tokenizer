#!/usr/bin/env python3
"""Helper script to sanity check validation ECG plotting."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.plot_validation_ecgs import (
    DEFAULT_PARQUET_PATH,
    plot_validation_ecgs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a small set of tiered ECG plots for smoke testing"
    )
    parser.add_argument(
        "val_json",
        nargs="?",
        default="ECG_tokenizer/val_generations/val_generations_epoch_7.json",
        help="Validation generations JSON to visualize",
    )
    parser.add_argument(
        "--parquet",
        default=DEFAULT_PARQUET_PATH,
        help="Parquet mapping file with waveform paths and dataset column",
    )
    parser.add_argument(
        "--output-dir",
        default="validation_plots/test_smoke",
        help="Directory where plots should be saved",
    )
    parser.add_argument(
        "--num-ecgs",
        type=int,
        default=3,
        help="How many ECGs to sample (defaults to 3 for one per tier)",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        help="Optional epoch label to stamp on the plot titles",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running validation plotting smoke test...")
    print(f"  JSON:    {args.val_json}")
    print(f"  Parquet: {args.parquet}")
    print(f"  Output:  {output_dir.resolve()}")
    print(f"  Count:   {args.num_ecgs}")

    plot_validation_ecgs(
        val_json_path=args.val_json,
        parquet_path=args.parquet,
        output_dir=str(output_dir),
        num_ecgs=args.num_ecgs,
        epoch=args.epoch,
    )


if __name__ == "__main__":
    main()
