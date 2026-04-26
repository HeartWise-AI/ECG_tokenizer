"""Seed ACTIVE_CHECKPOINT.txt + ACTIVE_BASELINE.json + results.tsv from BASELINE.json.

Run once after Phase 0.2 (baseline scoring) completes. Idempotent.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

RALPH_DIR = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, default=RALPH_DIR / "BASELINE.json")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/volume/ECG_tokenizer/checkpoints/BEST_LLM/2wjwbk0b_20260413-224103_ENHANCED/best_model.pt"
        ),
    )
    args = ap.parse_args()

    if not args.baseline.exists():
        print(f"missing baseline: {args.baseline}")
        return 2
    if not args.checkpoint.exists():
        print(f"missing checkpoint: {args.checkpoint}")
        return 2

    with open(args.baseline) as f:
        baseline = json.load(f)
    composite = baseline.get("composite", 0.0)

    active_baseline = RALPH_DIR / "ACTIVE_BASELINE.json"
    shutil.copy(args.baseline, active_baseline)

    active_ckpt = RALPH_DIR / "ACTIVE_CHECKPOINT.txt"
    active_ckpt.write_text(str(args.checkpoint) + "\n")

    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=RALPH_DIR
    ).decode().strip()

    results = RALPH_DIR / "results.tsv"
    header = "exp_num\tslug\tchange\tcomposite\tdelta_vs_active\tworst_per_task_delta\tdecision\tcheckpoint\tcommit_sha\n"
    row = (
        f"0\tbaseline\t-\t{composite:.6f}\t0.000000\t0.000000\tbaseline\t"
        f"{args.checkpoint}\t{sha}\n"
    )
    if results.exists() and results.read_text().startswith(header):
        # Reset content if it already starts with the header so this is idempotent.
        results.write_text(header + row)
    else:
        results.write_text(header + row)

    print(f"baseline composite: {composite:.4f}")
    print(f"active checkpoint:  {args.checkpoint}")
    print(f"results.tsv:        {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
