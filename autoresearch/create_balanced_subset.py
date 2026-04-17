#!/usr/bin/env python3
"""Create a balanced 400k subset from the training data.

Samples equally from all prompt categories, with equal positive/negative
sampling within pattern label columns where possible.
"""

import pandas as pd
import numpy as np

INPUT_PATH = "/volume/ECG_tokenizer/output/combined_train_qa_m5000k_h5000k_weighted.parquet"
OUTPUT_PATH = "/volume/ECG_tokenizer/output/balanced_train_400k.parquet"
TARGET_ROWS = 400_000
SEED = 42

# Pattern columns to balance positive/negative on
PATTERN_COLS = [
    "Left ventricular hypertrophy",
    "Right ventricular hypertrophy",
    "Afib",
    "Left bundle branch block",
    "Right bundle branch block",
    "Bradycardia",
    "Prolonged QT",
    "ST elevation (anterior - V3-V4)",
    "Acute MI",
    "Ventricular tachycardia",
]

def main():
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Total rows: {len(df)}")

    rng = np.random.RandomState(SEED)

    # Get category distribution
    categories = df["prompt_category"].value_counts()
    n_categories = len(categories)
    per_category = TARGET_ROWS // n_categories

    print(f"\n{n_categories} categories, targeting {per_category} per category")

    sampled_dfs = []
    for cat, count in categories.items():
        cat_df = df[df["prompt_category"] == cat]
        n_sample = min(per_category, len(cat_df))

        # For categories with pattern labels, try to balance positive/negative
        available_patterns = [c for c in PATTERN_COLS if c in cat_df.columns]
        if available_patterns and n_sample > 100:
            # Find rows with any positive pattern
            has_positive = cat_df[available_patterns].max(axis=1) > 0
            pos_df = cat_df[has_positive]
            neg_df = cat_df[~has_positive]

            # Sample equal pos/neg if possible
            n_each = n_sample // 2
            if len(pos_df) >= n_each and len(neg_df) >= n_each:
                sampled = pd.concat([
                    pos_df.sample(n=n_each, random_state=rng),
                    neg_df.sample(n=n_each, random_state=rng)
                ])
            else:
                # Not enough of one class, just random sample
                sampled = cat_df.sample(n=n_sample, random_state=rng)
        else:
            sampled = cat_df.sample(n=n_sample, random_state=rng)

        sampled_dfs.append(sampled)
        print(f"  {cat}: {n_sample} samples (from {count})")

    result = pd.concat(sampled_dfs, ignore_index=True)
    # Shuffle
    result = result.sample(frac=1, random_state=rng).reset_index(drop=True)

    print(f"\nTotal sampled: {len(result)}")
    print(f"Category distribution:")
    print(result["prompt_category"].value_counts().to_string())

    # Save
    result.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
