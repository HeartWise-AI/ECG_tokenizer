#!/usr/bin/env python3
"""Compute recall@5 for ALL labels (no filtering) from retrieval outputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-csv", required=True, type=Path, help="Retrieval CSV with ranked outputs.")
    parser.add_argument("--text-bank", required=True, type=Path, help="Text bank CSV with label categories.")
    parser.add_argument("--k", type=int, default=5, help="K value for recall@K.")
    return parser.parse_args()


def _clean_lines(value: str):
    for raw in str(value).splitlines():
        item = raw.strip()
        if item:
            yield item


def parse_ranked_ids(serialised: str) -> List[Tuple[str, str]]:
    """Parse ranked retrieval results."""
    parsed = []
    for line in _clean_lines(serialised):
        sign = line[0]
        text_id = line[2:].strip() if len(line) > 2 else ""
        if text_id:
            parsed.append((sign, text_id))
    return parsed


def compute_recall_at_k(retrieval_csv: Path, k: int) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Compute recall@K for all labels."""
    hit_counts = defaultdict(int)
    total_counts = defaultdict(int)

    df = pd.read_csv(retrieval_csv)
    for row in df.itertuples(index=False):
        ranked = parse_ranked_ids(row.ground_truth_all_ids)
        if not ranked:
            continue

        ordering = [tid for _, tid in ranked]
        rank_map = {tid: idx for idx, tid in enumerate(ordering)}

        for sign, text_id in ranked:
            if sign != "+":
                continue
            total_counts[text_id] += 1
            rank = rank_map.get(text_id)
            if rank is not None and rank < k:
                hit_counts[text_id] += 1

    return hit_counts, total_counts


def main() -> None:
    args = parse_args()

    # Compute recall
    hit_counts, total_counts = compute_recall_at_k(args.retrieval_csv, args.k)

    # Load text bank for categories and text
    text_bank = pd.read_csv(args.text_bank)
    category_map = dict(zip(text_bank["text_id"], text_bank["category"]))
    text_map = dict(zip(text_bank["text_id"], text_bank["text"]))
    granularity_map = dict(zip(text_bank["text_id"], text_bank.get("granularity", [""] * len(text_bank))))

    # Compute recall for ALL labels (no filtering)
    results = []
    for text_id in set(list(hit_counts.keys()) + list(total_counts.keys())):
        total = total_counts.get(text_id, 0)
        hits = hit_counts.get(text_id, 0)
        recall = hits / total if total > 0 else 0.0
        category = category_map.get(text_id, "UNKNOWN")
        text = text_map.get(text_id, text_id)
        granularity = granularity_map.get(text_id, "")

        # Determine label type
        if text_id.startswith("LBL_"):
            label_type = "LBL"
        elif text_id.startswith("QA_"):
            label_type = "QA"
        else:
            label_type = "OTHER"

        results.append({
            "text_id": text_id,
            "text": text,
            "category": category,
            "granularity": granularity,
            "label_type": label_type,
            "hits": hits,
            "total": total,
            "recall": recall,
        })

    # Convert to DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values("recall", ascending=False)

    # Save to CSV
    output_path = args.retrieval_csv.parent / f"full_recall_at{args.k}_all_labels.csv"
    df.to_csv(output_path, index=False)

    # Print summary
    lbl_df = df[df["label_type"] == "LBL"].copy()

    print("=" * 120)
    print(f"FULL RECALL@{args.k} ANALYSIS (ALL LABELS, NO FILTERING)")
    print("=" * 120)
    print()
    print(f"Total labels: {len(df)}")
    print(f"LBL labels: {len(lbl_df)}")
    print(f"QA labels: {len(df[df['label_type'] == 'QA'])}")
    print()
    print(f"## LBL LABELS STATISTICS:")
    print(f"Mean Recall@{args.k}: {lbl_df['recall'].mean():.4f} ({lbl_df['recall'].mean()*100:.2f}%)")
    print(f"Median Recall@{args.k}: {lbl_df['recall'].median():.4f} ({lbl_df['recall'].median()*100:.2f}%)")
    print()

    # Category breakdown
    print("## CATEGORY BREAKDOWN (LBL only):")
    categories = sorted(lbl_df["category"].unique())
    for category in categories:
        cat_df = lbl_df[lbl_df["category"] == category]
        cat_mean = cat_df["recall"].mean()
        print(f"{category}: {cat_mean:.4f} ({cat_mean*100:.2f}%) - {len(cat_df)} labels")
    print()

    # Show specific labels mentioned in comparison
    print("## KEY DIAGNOSES (for comparison):")
    key_labels = [
        "LBL_afib",
        "LBL_atrial_flutter",
        "LBL_left_bundle_branch_block",
        "LBL_right_bundle_branch_block",
        "LBL_ventricular_tachycardia",
        "LBL_junctional_rhythm",
        "LBL_irregularly_irregular",
        "LBL_st_elevation_anterior_v3_v4",
        "LBL_st_elevation_inferior_ii_iii_avf",
        "LBL_q_wave_inferior_ii_iii_avf",
        "LBL_acute_mi",
    ]

    for label_id in key_labels:
        row = df[df["text_id"] == label_id]
        if not row.empty:
            row = row.iloc[0]
            print(f"{row['text']}")
            print(f"  Recall@{args.k}: {row['recall']:.3f} ({row['recall']*100:.1f}%) | Hits: {row['hits']}/{row['total']}")
        else:
            print(f"{label_id}: NOT FOUND")
    print()

    print(f"Full results saved to: {output_path}")


if __name__ == "__main__":
    main()
