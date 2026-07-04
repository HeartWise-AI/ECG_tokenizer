#!/usr/bin/env python3
"""Build a GRPO training set from rows where the CURRENT model produces wrong
verifier scores. These are the cases with the strongest reward signal — the
model gets +1 for fixing them.

We use a model checkpoint's generations on training data, score with the
verifier, keep rows with verifier < 0.5 (model currently fails). Then sample
balanced across categories.

Usage:
    # First produce generations on a chunk of training data
    # then filter and write JSONL.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True,
                    help="Existing OpenRLHF JSONL (prompt, label, signal_path)")
    ap.add_argument("--n_rows", type=int, default=10000)
    ap.add_argument("--per_cat_cap", type=int, default=2000)
    ap.add_argument("--gt_balance_yesno", action="store_true",
                    help="For binary Yes/No categories, balance the GT class")
    ap.add_argument("--output", default="data/grpo_failed_v1.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []
    with open(args.in_jsonl) as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"[failed] loaded {len(rows)} rows from {args.in_jsonl}")

    # Group by category and parse GT yes/no
    by_cat = defaultdict(list)
    binary_cats = {
        "random_finding_question", "structural_heart_disease",
        "category_pericarditis", "category_chamber_enlargement",
        "category_rhythm", "category_conduction",
        "category_infarct_ischemia", "category_other",
    }
    for r in rows:
        lbl = json.loads(r["label"])
        cat = lbl.get("category", "")
        r["_cat"] = cat
        r["_gt"] = lbl.get("ground_truth", "")
        if cat in binary_cats:
            gt = r["_gt"].strip().lower()
            if gt.startswith("yes") or gt.startswith("***"):
                r["_yn"] = "yes"
            elif gt.startswith("no"):
                r["_yn"] = "no"
            else:
                r["_yn"] = "other"
        by_cat[cat].append(r)

    rng = random.Random(args.seed)
    out = []
    for cat, sub in by_cat.items():
        rng.shuffle(sub)
        if args.gt_balance_yesno and cat in binary_cats:
            yes = [r for r in sub if r.get("_yn") == "yes"]
            no = [r for r in sub if r.get("_yn") == "no"]
            n_each = min(args.per_cat_cap // 2, len(yes), len(no))
            picked = yes[:n_each] + no[:n_each]
            rng.shuffle(picked)
            out.extend(picked)
        else:
            out.extend(sub[:args.per_cat_cap])

    rng.shuffle(out)
    if args.n_rows and len(out) > args.n_rows:
        out = out[:args.n_rows]

    counts = Counter(r["_cat"] for r in out)
    print(f"[failed] final dataset: {len(out)} rows")
    for c, n in counts.most_common():
        print(f"  {c}: {n}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in out:
            f.write(json.dumps({
                "prompt": r["prompt"],
                "label": r["label"],
                "signal_path": r["signal_path"],
            }) + "\n")
    print(f"[failed] wrote {out_path}")


if __name__ == "__main__":
    main()
