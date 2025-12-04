#!/usr/bin/env python3
"""
Plot ECGs from validation generation JSON files with Q&A annotations.
Supports both filename-only JSON (looks up path from parquet) and full-path JSON.
"""

import json
import os
import sys
import random
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import argparse
import matplotlib.pyplot as plt
import pandas as pd

# Add DeepECG_Preprocess to path
sys.path.append('/volume/ECG_tokenizer/DeepECG_Preprocess')
sys.path.append('/volume/ECG_tokenizer/DeepECG_Preprocess/ecg_plotter')

from ecg_plotter.core import NPYECGPlotter

try:
    from rouge_score import rouge_scorer
except ImportError:  # pragma: no cover - optional dependency
    rouge_scorer = None


DEFAULT_PARQUET_PATH = \
    "/volume/ECG_tokenizer/output/combined_test_qa_m5k_h5k.parquet"
DEFAULT_TIER_SAMPLE_SIZE = 2
TIER_LABELS = ("low", "mid", "high")

_ROUGE_SCORER = None
if rouge_scorer is not None:  # pragma: no cover - depends on optional pkg
    try:
        _ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    except Exception:
        _ROUGE_SCORER = None


def compute_text_similarity(prediction: str, reference: str) -> float:
    """Compute a fallback similarity score in [0, 1] between prediction and reference."""
    prediction = (prediction or "").strip()
    reference = (reference or "").strip()

    if not prediction and not reference:
        return 1.0
    if not prediction or not reference:
        return 0.0

    if _ROUGE_SCORER is not None:
        try:
            return float(_ROUGE_SCORER.score(reference, prediction)["rougeL"].fmeasure)
        except Exception:
            pass

    return SequenceMatcher(None, reference.lower(), prediction.lower()).ratio()


def compute_ecg_score(ecg_data: Dict[str, Any]) -> Tuple[float, str]:
    """Return a scalar quality score and the metric source used."""
    metrics = ecg_data.get("Metrics") or {}

    for key in (
        "rougeL",
        "rougeL_fmeasure",
        "rougeL_f1",
        "rouge_l_fmeasure",
        "rouge_l_f1",
    ):
        value = metrics.get(key)
        if value is not None:
            try:
                return float(value), key
            except (TypeError, ValueError):
                continue

    for key in ("bleu4", "bleu", "bleu_4"):
        value = metrics.get(key)
        if value is not None:
            try:
                return float(value), key
            except (TypeError, ValueError):
                continue

    generated = ecg_data.get("Generation", "")
    ground_truth = ecg_data.get("Ground truth", "")
    return compute_text_similarity(generated, ground_truth), "rougeL_fallback"


def clean_chat_artifacts(text: str | None) -> str:
    """Strip chat template tokens and system prompts from text for cleaner plots."""
    if not text:
        return ""

    cleaned = text

    system_messages = [
        "You are DeepECG, an electrocardiogram analysis and question answering tool",
        "An electrocardiogram analysis and question answering tool",
    ]

    for message in system_messages:
        for prefix in ("system\n\n", "system\n", "system "):
            cleaned = cleaned.replace(prefix + message, "")

    for token in ("system", "user", "assistant"):
        cleaned = cleaned.replace(f"{token}\n\n", "")
        cleaned = cleaned.replace(f"{token}\n", "")

    cleaned = cleaned.replace("tooluser", "")
    cleaned = cleaned.replace("toolassistant", "")

    return cleaned.strip()


def _split_into_tiers(sorted_items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Split sorted items into three tiers: low, mid, high."""
    total = len(sorted_items)
    if total == 0:
        return {label: [] for label in TIER_LABELS}

    base, remainder = divmod(total, 3)
    sizes = [base + (1 if idx < remainder else 0) for idx in range(3)]

    tiers: Dict[str, List[Dict[str, Any]]] = {label: [] for label in TIER_LABELS}
    start = 0
    for label, size in zip(TIER_LABELS, sizes):
        if size <= 0:
            tiers[label] = []
        else:
            tiers[label] = sorted_items[start:start + size]
        start += size
    return tiers


def select_tiered_ecg_samples(
    scored_ecgs: List[Dict[str, Any]],
    total_requested: int,
    per_tier: int = DEFAULT_TIER_SAMPLE_SIZE
) -> List[Dict[str, Any]]:
    """Select ECGs by sampling from lower, middle, and upper performance tiers."""
    if not scored_ecgs:
        return []

    total_requested = max(0, min(total_requested, len(scored_ecgs)))
    if total_requested == 0:
        return []

    if total_requested < 3 or len(scored_ecgs) < 3:
        return random.sample(scored_ecgs, total_requested) if len(scored_ecgs) > total_requested else list(scored_ecgs)

    sorted_ecgs = sorted(scored_ecgs, key=lambda item: item["score"])
    tiers = _split_into_tiers(sorted_ecgs)

    target_per_tier = per_tier
    if total_requested < per_tier * 3:
        target_per_tier = max(1, total_requested // 3)

    selected: List[Dict[str, Any]] = []
    used_names: set[str] = set()

    for label in TIER_LABELS:
        tier_items = tiers.get(label, [])
        if not tier_items:
            continue
        take = min(target_per_tier, len(tier_items))
        chosen = random.sample(tier_items, take) if len(tier_items) > take else list(tier_items)
        for item in chosen:
            name = item.get("name")
            if name in used_names:
                continue
            new_item = dict(item)
            new_item["tier"] = label
            selected.append(new_item)
            used_names.add(name)

    if len(selected) < total_requested:
        name_to_tier = {
            item.get("name"): label
            for label, tier_items in tiers.items()
            for item in tier_items
        }
        remaining = [
            item for item in sorted_ecgs
            if item.get("name") not in used_names
        ]
        needed = min(total_requested - len(selected), len(remaining))
        if needed > 0:
            additional = random.sample(remaining, needed) if len(remaining) > needed else remaining
            for item in additional:
                name = item.get("name")
                new_item = dict(item)
                new_item["tier"] = name_to_tier.get(name, "extra")
                selected.append(new_item)

    return selected[:total_requested]


def build_parquet_lookup(parquet_df: Optional[pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """Create a lookup dict keyed by waveform_name for quick path/dataset access."""
    if parquet_df is None or 'waveform_name' not in parquet_df.columns:
        return {}

    candidate_cols = [
        col for col in (
            'waveform_path_psa',
            'waveform_path',
            'waveform_path_original',
            'dataset'
        ) if col in parquet_df.columns
    ]
    if not candidate_cols:
        return {}

    dedup_df = parquet_df.drop_duplicates('waveform_name', keep='first')
    lookup_df = dedup_df.set_index('waveform_name')[candidate_cols]
    return lookup_df.to_dict('index')


def resolve_ecg_source(
    ecg_name: str,
    ecg_data: Dict[str, Any],
    parquet_lookup: Optional[Dict[str, Dict[str, Any]]] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve ECG numpy path and dataset label from JSON entry and parquet lookup."""
    ecg_path = ecg_data.get('waveform_path') or ecg_data.get('waveform_path_psa')
    dataset_label = ecg_data.get('dataset')

    lookup_entry: Dict[str, Any] | None = None
    if parquet_lookup:
        lookup_entry = parquet_lookup.get(ecg_name)

    if not ecg_path and lookup_entry:
        for key in ('waveform_path_psa', 'waveform_path', 'waveform_path_original'):
            candidate = lookup_entry.get(key) if lookup_entry else None
            if candidate:
                ecg_path = candidate
                break

    if dataset_label is None and lookup_entry:
        dataset_label = lookup_entry.get('dataset')

    return ecg_path, dataset_label


def determine_plotter_params(dataset_label: Optional[str]) -> Tuple[str, bool, str]:
    """Return (plot_dataset, fft_normalized, normalized_label) based on dataset string."""
    normalized = (dataset_label or "MIMIC").strip().upper()

    if normalized.startswith('MIMIC'):
        return "MIMICIV", True, normalized
    if normalized.startswith('MHI'):
        # MHI waveforms are also stored with FFT-normalized preprocessing
        return "MHI", True, normalized

    return "MIMICIV", False, normalized or "UNKNOWN"


def load_validation_json(json_path: str) -> Dict:
    """Load validation generations JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_parquet_mapping(parquet_path: Optional[str]) -> Optional[pd.DataFrame]:
    """Load parquet file with ECG path mappings."""
    resolved_path = parquet_path or DEFAULT_PARQUET_PATH

    if not resolved_path:
        print("No parquet path provided; proceeding without lookup table")
        return None

    if not os.path.exists(resolved_path):
        print(f"WARNING: Parquet file not found at {resolved_path}")
        return None

    print(f"Loading parquet file: {resolved_path}")
    df = pd.read_parquet(resolved_path)

    if 'waveform_name' not in df.columns:
        if 'waveform_path_psa' in df.columns:
            df['waveform_name'] = df['waveform_path_psa'].apply(lambda x: os.path.basename(x))
        else:
            raise ValueError("Parquet must have either 'waveform_name' or 'waveform_path_psa'")

    if 'dataset' not in df.columns:
        print("WARNING: Parquet file missing 'dataset' column; defaulting to MIMIC")

    print(f"Loaded {len(df)} ECG records from parquet")
    return df


def resolve_ecg_path(ecg_name: str, parquet_df: Optional[pd.DataFrame] = None) -> str:
    """Backward-compatible wrapper that returns only the ECG path."""
    if ecg_name.startswith('/'):
        return ecg_name

    if parquet_df is None:
        raise ValueError(f"Need parquet dataframe to resolve path for {ecg_name}")

    lookup = build_parquet_lookup(parquet_df)
    ecg_path, _ = resolve_ecg_source(ecg_name, {}, lookup)
    if not ecg_path:
        raise ValueError(f"Could not find path for ECG: {ecg_name}")
    return ecg_path


def format_qa_text(question: str, generated: str, ground_truth: str, max_len: int = 150) -> str:
    """Format Q&A text for display, with truncation if needed"""

    question = clean_chat_artifacts(question)
    generated = clean_chat_artifacts(generated)
    ground_truth = clean_chat_artifacts(ground_truth)

    # Truncate long answers
    if len(generated) > max_len:
        generated = generated[:max_len] + "..."
    if len(ground_truth) > max_len:
        ground_truth = ground_truth[:max_len] + "..."
    
    # Format for display
    text = f"QUESTION: {question}\n\n"
    text += f"GENERATED: {generated}\n\n"
    text += f"GROUND TRUTH: {ground_truth}"
    
    return text


def plot_single_ecg(
    ecg_name: str,
    ecg_data: Dict,
    ecg_path: str,
    output_dir: str,
    epoch: Optional[int] = None,
    index: int = 0,
    dataset_label: Optional[str] = None,
    score: Optional[float] = None,
    tier_label: Optional[str] = None,
    metric_name: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Plot a single ECG with Q&A annotations.
    
    Returns:
        (success, save_path or error_message)
    """
    try:
        # Check if file exists
        if not os.path.exists(ecg_path):
            return False, f"ECG file not found: {ecg_path}"

        # Create plotter
        plot_dataset, fft_normalized, normalized_dataset = determine_plotter_params(dataset_label)
        plotter = NPYECGPlotter(
            npy_path=ecg_path,
            dataset=plot_dataset,
            out_dir=output_dir,
            width=2500,
            fft_normalized=fft_normalized
        )

        # Format Q&A text for title
        qa_text = format_qa_text(
            question=ecg_data.get('Question', 'N/A'),
            generated=ecg_data.get('Generation', 'N/A'),
            ground_truth=ecg_data.get('Ground truth', 'N/A')
        )

        # Create title with ECG name and epoch
        metadata_bits: List[str] = []
        if tier_label:
            metadata_bits.append(tier_label.upper())
        if score is not None:
            metadata_bits.append(f"score={score:.3f}")
        if metric_name:
            metadata_bits.append(metric_name)
        if normalized_dataset:
            metadata_bits.append(f"dataset={normalized_dataset}")

        title_prefix = " | ".join(metadata_bits)
        base_title = f"ECG: {ecg_name}"
        if title_prefix:
            base_title = f"[{title_prefix}] {base_title}"
        if epoch is not None:
            base_title = f"Epoch {epoch} - {base_title}"

        title = f"{base_title}\n{qa_text}"

        # Plot without auto-saving to avoid duplicates
        img, _ = plotter.plot_ecg(
            title=title,
            save=False,  # Don't auto-save, we'll save with custom name
            anonymize=True,
            show_diagnosis=False  # We're showing Q&A instead
        )

        # Custom save path with meaningful name
        epoch_str = f"epoch_{epoch}_" if epoch is not None else ""
        tier_str = f"{tier_label}_" if tier_label else ""
        custom_save_path = os.path.join(
            output_dir,
            f"{epoch_str}{tier_str}{index:03d}_{ecg_name.replace('.npy', '')}_qa.png"
        )

        # Save with custom name only
        if img:
            img.save(custom_save_path, dpi=(240, 240))
            print(f"✓ Saved plot: {custom_save_path}")
            return True, custom_save_path
        else:
            return False, "Failed to generate plot"
            
    except Exception as e:
        return False, f"Error plotting {ecg_name}: {str(e)}"


def plot_validation_ecgs(
    val_json_path: str,
    parquet_path: Optional[str] = None,
    output_dir: str = "validation_plots",
    num_ecgs: int = 10,
    epoch: Optional[int] = None,
    only_ecg_name: Optional[str] = None,
):
    """Plot ECGs from validation generations JSON with Q&A annotations."""

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print(f"Loading validation JSON: {val_json_path}")
    val_data = load_validation_json(val_json_path)
    print(f"Found {len(val_data)} ECGs in validation JSON")

    parquet_df = load_parquet_mapping(parquet_path)
    parquet_lookup = build_parquet_lookup(parquet_df) if parquet_df is not None else {}

    scored_ecgs: List[Dict[str, Any]] = []
    missing_lookup: List[str] = []
    # If a specific ECG is requested, narrow the iteration set
    items_iter = val_data.items()
    if only_ecg_name:
        if only_ecg_name not in val_data:
            print(f"Requested ECG '{only_ecg_name}' not found in validation JSON")
            return
        items_iter = [(only_ecg_name, val_data[only_ecg_name])]

    for ecg_name, ecg_info in items_iter:
        score, metric_name = compute_ecg_score(ecg_info)
        ecg_path, dataset_label = resolve_ecg_source(ecg_name, ecg_info, parquet_lookup)

        if not ecg_path:
            missing_lookup.append(ecg_name)
            continue

        scored_ecgs.append({
            "name": ecg_name,
            "score": score,
            "metric": metric_name,
            "data": ecg_info,
            "path": ecg_path,
            "dataset": dataset_label,
        })

    if missing_lookup:
        print(f"Skipping {len(missing_lookup)} ECGs without matching parquet entries")

    if not scored_ecgs:
        print("No ECG entries with resolvable paths found in validation JSON")
        return

    if only_ecg_name:
        selected_ecgs = scored_ecgs
    else:
        total_requested = min(num_ecgs, len(scored_ecgs)) if num_ecgs else len(scored_ecgs)
        selected_ecgs = select_tiered_ecg_samples(scored_ecgs, total_requested)

    print(f"\nPlotting {len(selected_ecgs)} ECGs (tiered sampling)...")

    success_count = 0
    failed_ecgs: List[Tuple[str, str]] = []

    for i, item in enumerate(selected_ecgs):
        ecg_name = item["name"]
        score = item["score"]
        metric_name = item.get("metric")
        tier_label = item.get("tier")
        ecg_data = item["data"]

        print(
            f"\n[{i+1}/{len(selected_ecgs)}] Processing {ecg_name} "
            f"(tier={tier_label}, score={score:.3f}, metric={metric_name})"
        )

        ecg_path = item.get("path")
        dataset_label = item.get("dataset")

        if not ecg_path or not os.path.exists(ecg_path):
            message = f"ECG file not found at {ecg_path}"
            print(f"  ✗ {message}")
            failed_ecgs.append((ecg_name, message))
            continue

        success, result = plot_single_ecg(
            ecg_name=ecg_name,
            ecg_data=ecg_data,
            ecg_path=ecg_path,
            output_dir=output_dir,
            epoch=epoch,
            index=i,
            dataset_label=dataset_label,
            score=score,
            tier_label=tier_label,
            metric_name=metric_name
        )

        if success:
            success_count += 1
        else:
            print(f"  ✗ Failed: {result}")
            failed_ecgs.append((ecg_name, result))

    print("\n" + "=" * 60)
    print(f"SUMMARY: Successfully plotted {success_count}/{len(selected_ecgs)} ECGs")

    if failed_ecgs:
        print(f"\nFailed ECGs ({len(failed_ecgs)}):")
        for ecg_name, error in failed_ecgs:
            print(f"  - {ecg_name}: {error}")

    print(f"\nPlots saved to: {output_dir}")
    print("=" * 60)
    

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description="Plot validation ECGs with Q&A annotations")
    parser.add_argument("val_json", help="Path to validation generations JSON file")
    parser.add_argument("--parquet", help="Path to parquet file with ECG paths",
                       default=DEFAULT_PARQUET_PATH)
    parser.add_argument("--output-dir", help="Output directory for plots", 
                       default="validation_plots")
    parser.add_argument("--num-ecgs", type=int, help="Number of ECGs to plot", 
                       default=10)
    parser.add_argument("--epoch", type=int, help="Epoch number for labeling")
    parser.add_argument("--ecg-name", type=str, help="Plot only this ECG name (e.g., 47242287.npy)")
    
    args = parser.parse_args()
    
    plot_validation_ecgs(
        val_json_path=args.val_json,
        parquet_path=args.parquet,
        output_dir=args.output_dir,
        num_ecgs=args.num_ecgs,
        epoch=args.epoch,
        only_ecg_name=args.ecg_name,
    )


if __name__ == "__main__":
    main()
