#!/usr/bin/env python3
"""Plot ECGs for SigLIP retrieval outputs with top predictions and ground truth labels."""

from __future__ import annotations

import argparse
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import ImageDraw, ImageFont

# Add ECG plotter path
import sys
sys.path.append('/volume/ECG_tokenizer/DeepECG_Preprocess')
sys.path.append('/volume/ECG_tokenizer/DeepECG_Preprocess/ecg_plotter')

from ecg_plotter.core import NPYECGPlotter

DEFAULT_OUTPUT_DIR = "siglip_retrieval_plots"
DEFAULT_NUM_ECGS = 12
DEFAULT_MAX_TOPK = 6
TIER_LABELS = ("low", "mid", "high")


def _parse_text_block(block: str | float | int) -> List[str]:
    if isinstance(block, float) or isinstance(block, int) or pd.isna(block):
        return []
    parts = [seg.strip() for seg in str(block).split('\n')]
    return [seg for seg in parts if seg]


def _parse_prob_block(block: str | float | int) -> List[float]:
    if isinstance(block, float) or isinstance(block, int) or pd.isna(block):
        return []
    parts = re.split(r"[,\s]+", str(block).strip())
    probs: List[float] = []
    for part in parts:
        if not part:
            continue
        try:
            probs.append(float(part))
        except ValueError:
            continue
    return probs


def _split_into_tiers(sorted_items: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    total = len(sorted_items)
    if total == 0:
        return {label: [] for label in TIER_LABELS}

    base, remainder = divmod(total, 3)
    sizes = [base + (1 if idx < remainder else 0) for idx in range(3)]

    tiers: Dict[str, List[Dict[str, object]]] = {label: [] for label in TIER_LABELS}
    start = 0
    for label, size in zip(TIER_LABELS, sizes):
        if size > 0:
            tiers[label] = list(sorted_items[start:start + size])
        start += size
    return tiers


def select_tiered_samples(
    scored_items: List[Dict[str, object]],
    total_requested: int,
    per_tier: int,
    seed: int,
) -> List[Dict[str, object]]:
    if not scored_items or total_requested <= 0:
        return []

    random.Random(seed).shuffle(scored_items)
    sorted_items = sorted(scored_items, key=lambda item: item.get("alignment", 0.0))

    if total_requested < 3 or len(sorted_items) < 3:
        return sorted(sorted_items[:total_requested], key=lambda item: item.get("alignment", 0.0))

    tiers = _split_into_tiers(sorted_items)
    target = per_tier if per_tier * 3 <= total_requested else max(1, total_requested // 3)

    rng = random.Random(seed)
    selected: List[Dict[str, object]] = []
    used_ids: set[str] = set()

    for label in TIER_LABELS:
        tier_items = tiers.get(label, [])
        if not tier_items:
            continue
        take = min(target, len(tier_items))
        chosen = rng.sample(tier_items, take) if len(tier_items) > take else list(tier_items)
        for item in chosen:
            ecg_id = str(item.get("ecg_id"))
            if ecg_id in used_ids:
                continue
            new_item = dict(item)
            new_item["tier"] = label
            selected.append(new_item)
            used_ids.add(ecg_id)

    if len(selected) < total_requested:
        remaining = [item for item in sorted_items if str(item.get("ecg_id")) not in used_ids]
        needed = min(total_requested - len(selected), len(remaining))
        if needed > 0:
            selected.extend(remaining[:needed])

    return selected[:total_requested]


def _format_list(items: Sequence[Tuple[str, float]], max_len: int) -> str:
    lines = []
    for text, prob in items:
        label = text if len(text) <= max_len else text[:max_len] + "..."
        lines.append(f"• {label} ({prob:.3f})")
    return "\n".join(lines)


def _format_ground_truth(items: Sequence[str], max_len: int) -> str:
    lines = []
    for text in items:
        label = text if len(text) <= max_len else text[:max_len] + "..."
        lines.append(f"• {label}")
    return "\n".join(lines)


def plot_single_ecg(
    npy_path: str,
    output_path: Path,
    alignment: float,
    ground_truth: Sequence[str],
    top_predictions: Sequence[Tuple[str, float]],
    tier: str,
    title_suffix: str,
) -> None:
    plotter = NPYECGPlotter(
        npy_path=npy_path,
        dataset="MIMICIV",
        fft_normalized=True,
        out_dir=str(output_path.parent),
        width=2500,
    )

    img, _ = plotter.plot_ecg(save=False, anonymize=True, show_diagnosis=False)

    draw = ImageDraw.Draw(img)
    try:
        font_large = ImageFont.truetype("DejaVuSans-Bold.ttf", 42)
        font_body = ImageFont.truetype("DejaVuSans.ttf", 36)
    except IOError:
        font_large = ImageFont.load_default()
        font_body = ImageFont.load_default()

    text_block = [
        f"Alignment: {alignment:.3f} | Tier: {tier}",
        f"ECG: {Path(npy_path).name}",
        "",
        "Ground truth:",
        _format_ground_truth(ground_truth, 120) or "• (none)",
        "",
        "Top predictions:",
        _format_list(top_predictions, 120) or "• (none)",
    ]
    overlay = "\n".join(text_block)

    margin = 40
    draw.rectangle([(margin, margin), (img.width - margin, margin + 360)], fill=(255, 255, 255, 230))
    draw.text((margin + 10, margin + 10), overlay, fill=(0, 0, 0), font=font_body, spacing=8)

    img.save(output_path, dpi=(240, 240))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ECGs for SigLIP retrieval CSV outputs")
    parser.add_argument("csv", type=str, help="Path to val_epoch_X_retrieval.csv file")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save ECG plots")
    parser.add_argument("--num-ecgs", type=int, default=DEFAULT_NUM_ECGS, help="Number of ECGs to plot")
    parser.add_argument("--per-tier", type=int, default=4, help="Number of ECGs to sample per performance tier")
    parser.add_argument("--topk", type=int, default=DEFAULT_MAX_TOPK, help="Number of top predictions to display")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--min-groundtruth", type=int, default=1, help="Minimum number of ground truth labels required")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if 'ecg_id' not in df.columns:
        raise ValueError("CSV must include an 'ecg_id' column")

    items: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        ecg_path = str(row['ecg_id'])
        if not ecg_path or not os.path.exists(ecg_path):
            continue
        gt_texts = _parse_text_block(row.get('ground_truth_texts'))
        if len(gt_texts) < args.min_groundtruth:
            continue
        top_texts = _parse_text_block(row.get('top_pred_texts'))
        top_probs = _parse_prob_block(row.get('top_pred_probs'))
        paired = list(zip(top_texts, top_probs))[:args.topk]
        if not paired:
            paired = [(txt, 0.0) for txt in top_texts[:args.topk]]

        items.append({
            'ecg_id': ecg_path,
            'alignment': float(row.get('alignment', 0.0)),
            'ground_truth': gt_texts,
            'top_predictions': paired,
        })

    if not items:
        raise ValueError("No ECGs available after filtering. Adjust filters or check CSV contents.")

    selected = select_tiered_samples(items, args.num_ecgs, args.per_tier, args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(selected, start=1):
        ecg_path = str(item['ecg_id'])
        alignment = float(item.get('alignment', 0.0))
        ground_truth = list(item.get('ground_truth', []))
        top_predictions = list(item.get('top_predictions', []))
        tier = str(item.get('tier', 'unknown'))

        output_path = output_dir / f"ecg_{idx:03d}_{Path(ecg_path).stem}.png"
        plot_single_ecg(
            npy_path=ecg_path,
            output_path=output_path,
            alignment=alignment,
            ground_truth=ground_truth,
            top_predictions=top_predictions,
            tier=tier,
            title_suffix=f"#{idx:03d}",
        )

        print(f"Saved plot -> {output_path}")


if __name__ == "__main__":
    main()
