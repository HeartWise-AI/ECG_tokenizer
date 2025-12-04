#!/usr/bin/env python3
"""Compute comprehensive recall@5 statistics for all diagnoses with category breakdown."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-csv", required=True, type=Path, help="Mapping CSV with labels.")
    parser.add_argument("--retrieval-csv", required=True, type=Path, help="Retrieval CSV with ranked outputs.")
    parser.add_argument("--text-bank", required=True, type=Path, help="Text bank CSV with label categories.")
    parser.add_argument("--split", default="test", help="Dataset split to use.")
    parser.add_argument("--k", type=int, default=5, help="K value for recall@K.")
    parser.add_argument("--min-positives", type=int, default=10, help="Minimum positives to include label.")
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


def load_prevalence(mapping_csv: Path, split: str, min_positives: int) -> pd.Series:
    """Load label prevalence from mapping CSV."""
    mapping = pd.read_csv(mapping_csv)
    if "split" in mapping.columns:
        mapping = mapping[mapping["split"].str.lower() == split.lower()]
    positives = mapping[mapping["label"] > 0]
    counts = positives.groupby("text_id")["label"].count()
    counts = counts[counts >= int(min_positives)]
    return counts.sort_values()


def compute_recall_at_k(
    retrieval_csv: Path,
    k: int,
) -> Tuple[Dict[str, int], Dict[str, int]]:
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


def load_text_bank(text_bank_path: Path) -> pd.DataFrame:
    """Load text bank with categories."""
    return pd.read_csv(text_bank_path)


def filter_label_types(text_id: str) -> str:
    """Determine if label is LBL or QA type."""
    if text_id.startswith("LBL_"):
        return "LBL"
    elif text_id.startswith("QA_"):
        return "QA"
    return "OTHER"


def main() -> None:
    args = parse_args()

    # Load data
    prevalence = load_prevalence(args.mapping_csv, args.split, args.min_positives)
    hit_counts, total_counts = compute_recall_at_k(args.retrieval_csv, args.k)
    text_bank = load_text_bank(args.text_bank)

    # Create lookup for categories
    category_map = dict(zip(text_bank["text_id"], text_bank["category"]))
    text_map = dict(zip(text_bank["text_id"], text_bank["text"]))

    # Compute recall for all labels
    results = []
    for text_id, count in prevalence.items():
        total = total_counts.get(text_id, 0)
        if total == 0:
            continue

        hits = hit_counts.get(text_id, 0)
        recall = hits / total if total > 0 else 0.0
        category = category_map.get(text_id, "UNKNOWN")
        text = text_map.get(text_id, text_id)
        label_type = filter_label_types(text_id)

        results.append({
            "text_id": text_id,
            "text": text,
            "category": category,
            "label_type": label_type,
            "prevalence": int(count),
            "hits": hits,
            "total": total,
            "recall": recall,
        })

    # Convert to DataFrame and sort by recall
    df = pd.DataFrame(results)
    df = df.sort_values("recall", ascending=False)

    # Filter to only LBL labels for main analysis
    lbl_df = df[df["label_type"] == "LBL"].copy()

    print("=" * 120)
    print(f"COMPREHENSIVE RECALL@{args.k} ANALYSIS")
    print(f"Checkpoint: {args.retrieval_csv.parent.parent.name}")
    print("=" * 120)
    print()

    # Overall statistics
    mean_recall = lbl_df["recall"].mean()
    median_recall = lbl_df["recall"].median()

    print(f"## OVERALL STATISTICS (LBL labels only, n={len(lbl_df)}):")
    print()
    print(f"Mean Recall@{args.k}: {mean_recall:.4f} ({mean_recall*100:.2f}%)")
    print(f"Median Recall@{args.k}: {median_recall:.4f} ({median_recall*100:.2f}%)")
    print()

    # Category breakdown
    print("=" * 120)
    print("CATEGORY BREAKDOWN")
    print("=" * 120)
    print()

    categories = sorted(lbl_df["category"].unique())
    for category in categories:
        cat_df = lbl_df[lbl_df["category"] == category].copy()
        cat_mean = cat_df["recall"].mean()

        print(f"\n## {category} ({len(cat_df)} labels):")
        print(f"Category Mean Recall@{args.k}: {cat_mean:.4f} ({cat_mean*100:.2f}%)")
        print()

        # Sort by recall within category
        cat_df = cat_df.sort_values("recall")
        for _, row in cat_df.iterrows():
            print(f"{row['text']}")
            print(f"  Recall@{args.k}: {row['recall']:.3f} ({row['recall']*100:.1f}%)")
            print(f"  Hits: {row['hits']}/{row['total']} | Prevalence: {row['prevalence']}")
            print()

    # Top and bottom performers
    print("=" * 120)
    print(f"TOP 20 BEST PERFORMING DIAGNOSES (Recall@{args.k})")
    print("=" * 120)
    print()

    top_20 = lbl_df.nlargest(20, "recall")
    for idx, (_, row) in enumerate(top_20.iterrows(), 1):
        print(f"{idx}. {row['text']}")
        print(f"   Recall@{args.k}: {row['recall']:.3f} ({row['recall']*100:.1f}%) | "
              f"Category: {row['category']} | Hits: {row['hits']}/{row['total']}")
        print()

    print("=" * 120)
    print(f"TOP 20 WORST PERFORMING DIAGNOSES (Recall@{args.k})")
    print("=" * 120)
    print()

    bottom_20 = lbl_df.nsmallest(20, "recall")
    for idx, (_, row) in enumerate(bottom_20.iterrows(), 1):
        print(f"{idx}. {row['text']}")
        print(f"   Recall@{args.k}: {row['recall']:.3f} ({row['recall']*100:.1f}%) | "
              f"Category: {row['category']} | Hits: {row['hits']}/{row['total']}")
        print()

    # Performance distribution
    print("=" * 120)
    print(f"RECALL@{args.k} DISTRIBUTION")
    print("=" * 120)
    print()

    bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
            (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0), (1.0, 1.01)]

    for low, high in bins:
        if high > 1.0:
            count = len(lbl_df[lbl_df["recall"] == 1.0])
            pct = count / len(lbl_df) * 100 if len(lbl_df) > 0 else 0
            print(f"Recall = 1.000: {count} labels ({pct:.1f}%)")
        else:
            count = len(lbl_df[(lbl_df["recall"] >= low) & (lbl_df["recall"] < high)])
            pct = count / len(lbl_df) * 100 if len(lbl_df) > 0 else 0
            print(f"Recall [{low:.1f}, {high:.1f}): {count} labels ({pct:.1f}%)")

    print()
    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print()
    print(f"Total LBL labels analyzed: {len(lbl_df)}")
    print(f"Mean Recall@{args.k}: {mean_recall*100:.2f}%")
    print(f"Labels with Recall >= 0.80: {len(lbl_df[lbl_df['recall'] >= 0.8])} ({len(lbl_df[lbl_df['recall'] >= 0.8])/len(lbl_df)*100:.1f}%)")
    print(f"Labels with Recall >= 0.50: {len(lbl_df[lbl_df['recall'] >= 0.5])} ({len(lbl_df[lbl_df['recall'] >= 0.5])/len(lbl_df)*100:.1f}%)")
    print(f"Labels with Recall < 0.20: {len(lbl_df[lbl_df['recall'] < 0.2])} ({len(lbl_df[lbl_df['recall'] < 0.2])/len(lbl_df)*100:.1f}%)")
    print()

    # Save detailed results
    output_path = args.retrieval_csv.parent / f"comprehensive_recall_at{args.k}_analysis.csv"
    df.to_csv(output_path, index=False)
    print(f"Detailed results saved to: {output_path}")
    print()


if __name__ == "__main__":
    main()
