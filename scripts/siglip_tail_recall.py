#!/usr/bin/env python3
"""Compute tail-class prevalence and recall@K from SigLIP retrieval outputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-csv", required=True, type=Path, help="Mapping CSV with labels.")
    parser.add_argument("--retrieval-csv", required=True, type=Path, help="Retrieval CSV with ranked outputs.")
    parser.add_argument("--split", default="test", help="Dataset split used for prevalence computation.")
    parser.add_argument("--top-n", type=int, default=50, help="Number of rarest labels to report.")
    parser.add_argument("--ks", type=int, nargs="+", default=[5], help="Recall@K values to compute.")
    parser.add_argument("--text-bank", type=Path, default=None, help="Optional text bank CSV for label text.")
    parser.add_argument("--min-positives", type=int, default=1, help="Ignore labels with fewer positives than this.")
    return parser.parse_args()


def _clean_lines(value: str) -> Iterable[str]:
    for raw in str(value).splitlines():
        item = raw.strip()
        if item:
            yield item


def parse_ranked_ids(serialised: str) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for line in _clean_lines(serialised):
        sign = line[0]
        text_id = line[2:].strip() if len(line) > 2 else ""
        if text_id:
            parsed.append((sign, text_id))
    return parsed


def load_prevalence(mapping_csv: Path, split: str, min_positives: int) -> pd.Series:
    mapping = pd.read_csv(mapping_csv)
    if "split" in mapping.columns:
        mapping = mapping[mapping["split"].str.lower() == split.lower()]
    positives = mapping[mapping["label"] > 0]
    counts = positives.groupby("text_id")["label"].count()
    counts = counts[counts >= int(min_positives)]
    return counts.sort_values()


def compute_recall(
    retrieval_csv: Path,
    ks: Sequence[int],
) -> tuple[dict[int, dict[str, int]], dict[str, int]]:
    ks = sorted({int(k) for k in ks if k > 0})
    hit_counts = {k: defaultdict(int) for k in ks}
    total_counts: dict[str, int] = defaultdict(int)

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
            if rank is None:
                continue
            for k in ks:
                if rank < k:
                    hit_counts[k][text_id] += 1
    return hit_counts, total_counts


def load_text_map(text_bank: Path | None) -> dict[str, str]:
    if text_bank is None:
        return {}
    table = pd.read_csv(text_bank)
    if "text_id" not in table.columns or "text" not in table.columns:
        return {}
    return dict(zip(table["text_id"], table["text"]))


def main() -> None:
    args = parse_args()
    prevalence = load_prevalence(args.mapping_csv, args.split, args.min_positives)
    if prevalence.empty:
        raise SystemExit("No positive labels found for the requested split.")

    hit_counts, total_counts = compute_recall(args.retrieval_csv, args.ks)
    text_lookup = load_text_map(args.text_bank)

    ks = sorted({int(k) for k in args.ks if k > 0})
    header_cols = ["text_id", "pos_count"]
    header_cols.extend(f"recall@{k}" for k in ks)
    header_cols.extend(f"hits@{k}" for k in ks)
    header_cols.append("text")

    print(" | ".join(header_cols))
    print("-" * 80)

    reported = 0
    for text_id, count in prevalence.items():
        if reported >= args.top_n:
            break
        totals = total_counts.get(text_id, 0)
        if totals == 0:
            continue
        row = [text_id, str(int(count))]
        for k in ks:
            hits = hit_counts[k].get(text_id, 0)
            recall = hits / totals if totals else 0.0
            row.append(f"{recall:.3f}")
        for k in ks:
            hits = hit_counts[k].get(text_id, 0)
            row.append(f"{hits}/{totals}")
        text = text_lookup.get(text_id, "")
        row.append(text)
        print(" | ".join(row))
        reported += 1

    if reported == 0:
        print("No overlapping labels between prevalence data and retrieval CSV.")


if __name__ == "__main__":
    main()
