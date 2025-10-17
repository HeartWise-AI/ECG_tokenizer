#!/usr/bin/env python3
"""
Compute recall@K metrics for SigLIP Phase 1 from retrieval CSV files.

This script computes recall by checking if ground truth positive labels appear
in the model's top-K predictions (top_pred_texts column), matching the logic
in runners/siglip_phase1_runner.py::_accumulate_tail_recall_at_k.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrieval-csv",
        required=True,
        type=Path,
        help="Retrieval CSV with model predictions (val_epoch_X_retrieval.csv)",
    )
    parser.add_argument(
        "--text-bank",
        required=True,
        type=Path,
        help="Text bank CSV mapping text_id to text (ecg_text_alignment/text_bank.csv)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="K value for recall@K (default: 5)",
    )
    parser.add_argument(
        "--min-positives",
        type=int,
        default=1,
        help="Minimum positive examples to include a label (default: 1)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: <retrieval_dir>/recall_at<K>_analysis.csv)",
    )
    return parser.parse_args()


def _clean_lines(value: str) -> List[str]:
    """Parse multi-line string into list of clean lines."""
    lines = []
    for raw in str(value).splitlines():
        item = raw.strip()
        if item:
            lines.append(item)
    return lines


def compute_recall_at_k(
    retrieval_csv: Path,
    text_to_id: Dict[str, str],
    k: int,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Compute recall@K from retrieval CSV by checking if ground truth positive
    texts appear in the top-K predicted texts.

    Args:
        retrieval_csv: Path to retrieval CSV with columns:
            - ground_truth_pos_texts: Multi-line string of positive labels
            - top_pred_texts: Multi-line string of top-K predictions
        text_to_id: Mapping from label text to text_id
        k: K value for recall@K

    Returns:
        Tuple of (hit_counts, total_counts) dictionaries keyed by text_id
    """
    hit_counts = defaultdict(int)
    total_counts = defaultdict(int)

    df = pd.read_csv(retrieval_csv)

    for row in df.itertuples(index=False):
        # Parse ground truth positive texts
        gt_pos_texts = _clean_lines(row.ground_truth_pos_texts)

        # Parse top-K predicted texts
        top_k_texts = _clean_lines(row.top_pred_texts)[:k]  # Take only first K

        # For each ground truth positive label
        for gt_text in gt_pos_texts:
            text_id = text_to_id.get(gt_text)
            if text_id is None:
                continue  # Skip if text not in text bank

            total_counts[text_id] += 1

            # Check if this positive label appears in top-K predictions
            if gt_text in top_k_texts:
                hit_counts[text_id] += 1

    return hit_counts, total_counts


def main() -> None:
    args = parse_args()

    # Load text bank and create text -> text_id mapping
    text_bank = pd.read_csv(args.text_bank)
    text_to_id = dict(zip(text_bank["text"], text_bank["text_id"]))
    id_to_text = dict(zip(text_bank["text_id"], text_bank["text"]))
    id_to_category = dict(zip(text_bank["text_id"], text_bank.get("category", [""] * len(text_bank))))
    id_to_granularity = dict(zip(text_bank["text_id"], text_bank.get("granularity", [""] * len(text_bank))))

    # Compute recall@K
    print(f"Computing recall@{args.k} from {args.retrieval_csv.name}...")
    hit_counts, total_counts = compute_recall_at_k(args.retrieval_csv, text_to_id, args.k)

    # Build results dataframe
    results = []
    for text_id in sorted(total_counts.keys()):
        total = total_counts[text_id]
        hits = hit_counts[text_id]
        recall = hits / total if total > 0 else 0.0

        # Filter by min_positives
        if total < args.min_positives:
            continue

        # Determine label type
        if text_id.startswith("LBL_"):
            label_type = "LBL"
        elif text_id.startswith("QA_"):
            label_type = "QA"
        else:
            label_type = "OTHER"

        results.append({
            "text_id": text_id,
            "text": id_to_text.get(text_id, text_id),
            "category": id_to_category.get(text_id, "UNKNOWN"),
            "granularity": id_to_granularity.get(text_id, ""),
            "label_type": label_type,
            "hits": hits,
            "total": total,
            "recall": recall,
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("recall", ascending=False)

    # Print summary statistics
    print("\n" + "=" * 120)
    print(f"RECALL@{args.k} ANALYSIS")
    print(f"Retrieval CSV: {args.retrieval_csv.name}")
    print("=" * 120)
    print()

    lbl_df = df_results[df_results["label_type"] == "LBL"].copy()
    qa_df = df_results[df_results["label_type"] == "QA"].copy()

    print(f"Total labels: {len(df_results)}")
    print(f"  LBL labels: {len(lbl_df)}")
    print(f"  QA labels: {len(qa_df)}")
    print(f"  Min positives filter: >={args.min_positives}")
    print()

    if len(lbl_df) > 0:
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
            print(f"  {category}: {cat_mean:.4f} ({cat_mean*100:.2f}%) - {len(cat_df)} labels")
        print()

    # Show top performers
    print(f"## TOP 10 BEST PERFORMING (LBL):")
    for idx, row in lbl_df.head(10).iterrows():
        print(f"  {row['text']}")
        print(f"    Recall@{args.k}: {row['recall']:.3f} ({row['recall']*100:.1f}%) | Hits: {row['hits']}/{row['total']}")
    print()

    # Show worst performers
    print(f"## TOP 10 WORST PERFORMING (LBL):")
    for idx, row in lbl_df.tail(10).iloc[::-1].iterrows():
        print(f"  {row['text']}")
        print(f"    Recall@{args.k}: {row['recall']:.3f} ({row['recall']*100:.1f}%) | Hits: {row['hits']}/{row['total']}")
    print()

    # Save to CSV
    if args.output_csv is None:
        output_csv = args.retrieval_csv.parent / f"recall_at{args.k}_analysis.csv"
    else:
        output_csv = args.output_csv

    df_results.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")
    print()


if __name__ == "__main__":
    main()
