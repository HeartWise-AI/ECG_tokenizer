#!/usr/bin/env python
"""Prepare SFT datasets for multi-task fine-tuning.

Processes potassium, HCM, and symptom datasets:
1. Adds waveform_path_psa column pointing to pre-adjusted signals
2. Filters rows where adjusted signal files are missing
3. Generates SFT prompt/answer columns for potassium dataset
4. Outputs ready-to-train parquet files
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ADJUSTED_SIGNALS_BASE = "/media/data1/datasets/MHI/adjusted_signals"
OUTPUT_DIR = "/volume/ECG_tokenizer/output"

# Potassium prompt templates
POTASSIUM_PROMPT_TEMPLATES = [
    "Does this ECG suggest dyskalemia?",
    "What is the likely serum potassium status based on this ECG?",
    "Is there evidence of hyperkalemia or hypokalemia on this ECG?",
    "Analyze this ECG for signs of electrolyte imbalance.",
    "Based on this ECG, predict the serum potassium category.",
    "Can you identify any potassium-related abnormalities on this ECG?",
    "What does this ECG reveal about the patient's potassium levels?",
    "Assess this ECG for electrocardiographic signs of dyskalemia.",
]


def add_waveform_path_psa(df: pd.DataFrame, dataset_name: str, split: str = "train") -> pd.DataFrame:
    """Add waveform_path_psa column and filter by file existence.

    Args:
        df: DataFrame with a ``waveform_name`` column.
        dataset_name: Label used in log messages.
        split: Subdirectory under adjusted signals (``train``, ``test``, ``val``).
    """
    if "waveform_name" not in df.columns:
        raise ValueError(f"{dataset_name}: missing 'waveform_name' column")

    adjusted_dir = os.path.join(ADJUSTED_SIGNALS_BASE, split)
    df = df.copy()
    df["waveform_path_psa"] = df["waveform_name"].apply(
        lambda name: os.path.join(adjusted_dir, str(name))
    )

    # Check file existence
    exists_mask = df["waveform_path_psa"].apply(os.path.isfile)
    n_total = len(df)
    n_found = exists_mask.sum()
    n_missing = n_total - n_found

    print(f"  {dataset_name}: {n_found}/{n_total} adjusted signals found "
          f"({100 * n_found / n_total:.1f}%), {n_missing} filtered out")

    return df[exists_mask].reset_index(drop=True)


def generate_potassium_answer(row: pd.Series) -> str:
    """Generate a clinical answer for a potassium ECG sample."""
    value = row["potassium_value"]
    category = row["potassium_category"]
    status = row.get("potassium_status", "")

    if category == "hyperkalemia":
        severity = "severe" if value >= 6.0 else "moderate"
        answer = (
            f"Yes — the ECG suggests hyperkalemia. The measured serum potassium is "
            f"{value:.1f} mmol/L ({severity} hyperkalemia). "
            f"ECG findings in hyperkalemia may include peaked T waves, "
            f"widened QRS complex, and flattened P waves."
        )
    elif category == "hypokalemia":
        severity = "severe" if value <= 2.5 else "moderate"
        answer = (
            f"Yes — the ECG suggests hypokalemia. The measured serum potassium is "
            f"{value:.1f} mmol/L ({severity} hypokalemia). "
            f"ECG findings in hypokalemia may include ST depression, "
            f"T wave flattening or inversion, and prominent U waves."
        )
    else:
        answer = (
            f"No — the ECG does not suggest dyskalemia. The measured serum potassium is "
            f"{value:.1f} mmol/L, which is within the normal range (3.5-5.0 mmol/L). "
            f"No electrocardiographic signs of potassium abnormality are evident."
        )

    return answer


def generate_potassium_prompt_category(row: pd.Series) -> str:
    """Generate prompt_category for potassium samples."""
    category = row["potassium_category"]
    return f"potassium_{category}"


def process_potassium(input_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process potassium dataset: add PSA paths and generate SFT columns."""
    print("\n=== Processing Potassium Dataset ===")

    train_path = os.path.join(input_dir, "potassium_ecg_train.parquet")
    test_path = os.path.join(input_dir, "potassium_ecg_test.parquet")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    print(f"  Raw: train={len(train_df)}, test={len(test_df)}")

    # Add PSA paths and filter
    train_df = add_waveform_path_psa(train_df, "potassium_train", split="train")
    test_df = add_waveform_path_psa(test_df, "potassium_test", split="test")

    # Generate SFT columns
    for df in [train_df, test_df]:
        df["prompt"] = df.apply(
            lambda _: random.choice(POTASSIUM_PROMPT_TEMPLATES), axis=1
        )
        df["generated_answer"] = df.apply(generate_potassium_answer, axis=1)
        df["prompt_category"] = df.apply(generate_potassium_prompt_category, axis=1)

    # Keep only needed columns
    keep_cols = [
        "waveform_name", "waveform_path_psa",
        "prompt", "generated_answer", "prompt_category",
        "potassium_value", "potassium_category", "potassium_status",
    ]
    train_df = train_df[[c for c in keep_cols if c in train_df.columns]]
    test_df = test_df[[c for c in keep_cols if c in test_df.columns]]

    print(f"  Final: train={len(train_df)}, test={len(test_df)}")
    print(f"  Train categories: {train_df['prompt_category'].value_counts().to_dict()}")

    return train_df, test_df


def process_hcm(input_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process HCM dataset: add PSA paths, keep existing SFT columns."""
    print("\n=== Processing HCM Dataset ===")

    train_path = os.path.join(input_dir, "hcm", "hcm_train.parquet")
    test_path = os.path.join(input_dir, "hcm", "hcm_test.parquet")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    print(f"  Raw: train={len(train_df)}, test={len(test_df)}")

    # Add PSA paths and filter
    train_df = add_waveform_path_psa(train_df, "hcm_train", split="train")
    test_df = add_waveform_path_psa(test_df, "hcm_test", split="test")

    # Keep needed columns (prompt, generated_answer, prompt_category already exist)
    keep_cols = [
        "waveform_name", "waveform_path_psa",
        "prompt", "generated_answer", "prompt_category",
    ]
    train_df = train_df[[c for c in keep_cols if c in train_df.columns]]
    test_df = test_df[[c for c in keep_cols if c in test_df.columns]]

    print(f"  Final: train={len(train_df)}, test={len(test_df)}")
    print(f"  Train categories: {train_df['prompt_category'].value_counts().to_dict()}")

    return train_df, test_df


def process_symptom(input_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process Symptom dataset: add PSA paths, keep existing SFT columns."""
    print("\n=== Processing Symptom Dataset ===")

    train_path = os.path.join(input_dir, "symptom", "symptom_train.parquet")
    test_path = os.path.join(input_dir, "symptom", "symptom_test.parquet")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    print(f"  Raw: train={len(train_df)}, test={len(test_df)}")

    # Add PSA paths and filter
    train_df = add_waveform_path_psa(train_df, "symptom_train", split="train")
    test_df = add_waveform_path_psa(test_df, "symptom_test", split="test")

    # Keep needed columns (prompt, generated_answer, prompt_category already exist)
    keep_cols = [
        "waveform_name", "waveform_path_psa",
        "prompt", "generated_answer", "prompt_category",
    ]
    train_df = train_df[[c for c in keep_cols if c in train_df.columns]]
    test_df = test_df[[c for c in keep_cols if c in test_df.columns]]

    print(f"  Final: train={len(train_df)}, test={len(test_df)}")
    print(f"  Train categories: {train_df['prompt_category'].value_counts().to_dict()}")

    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT datasets for multi-task fine-tuning")
    parser.add_argument(
        "--input_dir",
        default="/volume/DeepECG_Dataset/DeepECG_Preprocess/output",
        help="Base directory containing raw parquet files",
    )
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIR,
        help="Output directory for processed parquet files",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Process all datasets
    pot_train, pot_test = process_potassium(args.input_dir)
    hcm_train, hcm_test = process_hcm(args.input_dir)
    sym_train, sym_test = process_symptom(args.input_dir)

    # Save output parquets
    outputs = {
        "sft_potassium_train.parquet": pot_train,
        "sft_potassium_test.parquet": pot_test,
        "sft_hcm_train.parquet": hcm_train,
        "sft_hcm_test.parquet": hcm_test,
        "sft_symptom_train.parquet": sym_train,
        "sft_symptom_test.parquet": sym_test,
    }

    print("\n=== Saving Output Files ===")
    for filename, df in outputs.items():
        out_path = os.path.join(args.output_dir, filename)
        df.to_parquet(out_path, index=False)
        print(f"  {filename}: {len(df)} rows, columns={list(df.columns)}")

    # Verification summary
    print("\n=== Verification Summary ===")
    for filename, df in outputs.items():
        assert "waveform_path_psa" in df.columns, f"{filename}: missing waveform_path_psa"
        assert "prompt" in df.columns, f"{filename}: missing prompt"
        assert "generated_answer" in df.columns, f"{filename}: missing generated_answer"
        assert "prompt_category" in df.columns, f"{filename}: missing prompt_category"
        # Spot check first PSA path
        sample_path = df["waveform_path_psa"].iloc[0]
        exists = os.path.isfile(sample_path)
        print(f"  {filename}: OK (sample path exists={exists})")

    print("\nDone! All SFT datasets prepared successfully.")


if __name__ == "__main__":
    main()
