#!/usr/bin/env python3
"""
Create a weighted dataset for minority class upsampling in LLM finetuning.

This script adds a 'sample_weight' column to the training parquet file,
which can be used with PyTorch's WeightedRandomSampler to upweight
minority class samples during training without downsampling any data.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def detect_lvef_class(answer: str) -> str:
    """Detect LVEF class from answer text."""
    answer_lower = answer.lower()
    if "severely reduced" in answer_lower:
        return "severely_reduced"
    elif "moderately reduced" in answer_lower:
        return "moderately_reduced"
    elif "mildly reduced" in answer_lower:
        return "mildly_reduced"
    elif "normal" in answer_lower:
        return "normal"
    return "unknown"


def detect_afib_risk_class(answer: str) -> str:
    """Detect AFib risk class from answer text."""
    answer_lower = answer.lower()
    if answer_lower.startswith("yes") or "high risk" in answer_lower:
        return "high"
    elif answer_lower.startswith("low") or "unlikely" in answer_lower:
        return "low"
    return "unknown"


def detect_shd_class(answer: str) -> str:
    """Detect structural heart disease class from answer text."""
    answer_lower = answer.lower()
    if answer_lower.startswith("yes") or "present" in answer_lower:
        return "present"
    elif answer_lower.startswith("no"):
        return "absent"
    return "unknown"


def detect_acs_severity_class(answer: str) -> str:
    """Detect ACS severity class from answer text."""
    answer_lower = answer.lower()
    if answer_lower.startswith("yes") or "acute coronary occlusion" in answer_lower:
        return "acute_occlusion"
    elif "chronic occlusion" in answer_lower:
        return "chronic"
    elif "no evidence of coronary" in answer_lower or "no coronary disease" in answer_lower:
        return "no_disease"
    elif answer_lower.startswith("no"):
        return "obstructive_no_acute"
    return "unknown"


def detect_culprit_artery_class(answer: str) -> str:
    """Detect culprit artery class from answer text."""
    answer_lower = answer.lower()
    if "circumflex" in answer_lower or "lcx" in answer_lower:
        return "lcx"
    elif "lad" in answer_lower:
        return "lad"
    elif "rca" in answer_lower:
        return "rca"
    elif "ramus" in answer_lower:
        return "ramus"  # Treat as LCX territory
    return "unknown"


# Weight configuration for minority class upsampling
WEIGHT_CONFIG = {
    "lvef": {
        "severely_reduced": 4.0,   # ~10% of LVEF samples
        "moderately_reduced": 3.0, # ~15% of LVEF samples
        "mildly_reduced": 2.0,     # ~25% of LVEF samples
        "normal": 1.0,             # ~50% of LVEF samples
        "unknown": 1.0,
    },
    "afib_risk": {
        "high": 2.0,   # ~37% of samples
        "low": 1.0,    # ~63% of samples
        "unknown": 1.0,
    },
    "structural_heart_disease": {
        "present": 1.5,  # ~48% of samples (minority)
        "absent": 1.0,   # ~52% of samples
        "unknown": 1.0,
    },
    "acs_severity": {
        "acute_occlusion": 3.0,      # ~25% - key minority class
        "chronic": 2.0,              # ~8%
        "no_disease": 1.0,           # ~8%
        "obstructive_no_acute": 1.0, # ~59% - majority
        "unknown": 1.0,
    },
    "culprit_artery": {
        "lcx": 4.0,    # ~12% - hardest to detect
        "lad": 1.5,    # ~31%
        "rca": 1.0,    # ~57% - majority
        "ramus": 3.0,  # Rare, treat like LCX
        "unknown": 1.0,
    },
}


def compute_sample_weights(
    df: pd.DataFrame,
    prompt_category_col: str = "prompt_category",
    answer_col: str = "generated_answer",
) -> pd.Series:
    """
    Compute sample weights based on class membership within each task.

    Args:
        df: DataFrame with training samples
        prompt_category_col: Column containing task category
        answer_col: Column containing the answer text

    Returns:
        Series with sample weights (same index as df)
    """
    weights = pd.Series(1.0, index=df.index)

    # Process each task category with weights
    for task, class_weights in WEIGHT_CONFIG.items():
        mask = df[prompt_category_col] == task
        if not mask.any():
            continue

        # Detect class for each sample
        if task == "lvef":
            classes = df.loc[mask, answer_col].apply(detect_lvef_class)
        elif task == "afib_risk":
            classes = df.loc[mask, answer_col].apply(detect_afib_risk_class)
        elif task == "structural_heart_disease":
            classes = df.loc[mask, answer_col].apply(detect_shd_class)
        elif task == "acs_severity":
            classes = df.loc[mask, answer_col].apply(detect_acs_severity_class)
        elif task == "culprit_artery":
            classes = df.loc[mask, answer_col].apply(detect_culprit_artery_class)
        else:
            continue

        # Apply weights based on detected class
        for class_name, weight in class_weights.items():
            class_mask = classes == class_name
            # Get the indices where both task mask and class match
            task_indices = df.index[mask]
            class_indices = task_indices[class_mask.values]
            weights.loc[class_indices] = weight

    return weights


def main():
    parser = argparse.ArgumentParser(description="Create weighted dataset for minority class upsampling")
    parser.add_argument(
        "--input",
        type=str,
        default="output/combined_train_qa_m5000k_h5000k.parquet",
        help="Path to input parquet file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/combined_train_qa_m5000k_h5000k_weighted.parquet",
        help="Path to output parquet file",
    )
    parser.add_argument(
        "--prompt-category-col",
        type=str,
        default="prompt_category",
        help="Column name for prompt category",
    )
    parser.add_argument(
        "--answer-col",
        type=str,
        default="generated_answer",
        help="Column name for answer text",
    )
    args = parser.parse_args()

    print(f"Loading dataset from {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df):,} samples")

    print("\nComputing sample weights...")
    df["sample_weight"] = compute_sample_weights(
        df,
        prompt_category_col=args.prompt_category_col,
        answer_col=args.answer_col,
    )

    # Print weight distribution statistics
    print("\n=== Weight Distribution by Task ===")
    for task in WEIGHT_CONFIG.keys():
        mask = df[args.prompt_category_col] == task
        if mask.any():
            task_weights = df.loc[mask, "sample_weight"]
            print(f"\n{task}:")
            print(f"  Total samples: {mask.sum():,}")
            print(f"  Weight distribution:")
            for weight in sorted(task_weights.unique()):
                count = (task_weights == weight).sum()
                pct = count / mask.sum() * 100
                print(f"    Weight {weight:.1f}: {count:,} samples ({pct:.1f}%)")

    # Print overall statistics
    print("\n=== Overall Weight Statistics ===")
    print(f"Min weight: {df['sample_weight'].min():.2f}")
    print(f"Max weight: {df['sample_weight'].max():.2f}")
    print(f"Mean weight: {df['sample_weight'].mean():.2f}")
    print(f"Samples with weight > 1.0: {(df['sample_weight'] > 1.0).sum():,} ({(df['sample_weight'] > 1.0).mean()*100:.1f}%)")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving weighted dataset to {args.output}...")
    df.to_parquet(args.output, index=False)
    print("Done!")

    return df


if __name__ == "__main__":
    main()
