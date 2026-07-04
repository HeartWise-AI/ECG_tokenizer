#!/usr/bin/env python3
"""
Balance minority classes in QA datasets to prevent mode collapse during training.

This script addresses the class imbalance problem identified in the training analysis:
- LVEF: 60.6% Normal vs 7.2% Severe -> Balance to equal distribution
- AFib Risk: 68.6% Low vs 3.2% Moderate -> Balance risk levels
- ACS Severity: 94% Non-acute vs 4.5% Acute -> Balance acute/non-acute
- Conduction: Imbalanced findings -> Balance key findings

Strategy:
1. For each category with imbalance, identify majority and minority classes
2. Downsample majority class AND/OR upsample minority class to achieve balance
3. Reduce gradient contribution from generic interpretation tasks
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import re
import argparse
from typing import Dict, List, Tuple, Optional
from collections import Counter


def extract_lvef_value(answer: str) -> Optional[int]:
    """Extract LVEF percentage from answer string."""
    if pd.isna(answer):
        return None
    match = re.search(r'(\d+)%', str(answer))
    if match:
        return int(match.group(1))
    return None


def get_lvef_category(lvef_value: Optional[int]) -> Optional[str]:
    """Categorize LVEF value into clinical categories."""
    if lvef_value is None:
        return None
    if lvef_value >= 55:
        return 'normal'
    elif lvef_value >= 40:
        return 'mild'
    elif lvef_value >= 30:
        return 'moderate'
    else:
        return 'severe'


def get_afib_risk_class(answer: str) -> Optional[str]:
    """Extract AFib risk class from answer."""
    if pd.isna(answer):
        return None
    answer_lower = str(answer).lower()
    if 'low risk' in answer_lower:
        return 'low'
    elif 'high risk' in answer_lower:
        return 'high'
    elif 'moderate risk' in answer_lower:
        return 'moderate'
    elif 'already in atrial fibrillation' in answer_lower:
        return 'already_afib'
    return None


def get_acs_class(answer: str) -> Optional[str]:
    """Extract ACS severity class from answer."""
    if pd.isna(answer):
        return None
    answer_str = str(answer)
    if answer_str.startswith('Yes'):
        return 'acute'
    elif answer_str.startswith('No'):
        return 'non_acute'
    return None


def balance_category_classes(
    df: pd.DataFrame,
    category: str,
    class_extractor,
    target_ratio: Dict[str, float] = None,
    min_samples_per_class: int = 100,
    max_total_samples: Optional[int] = None,
    upsample_minority: bool = True,
    downsample_majority: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Balance classes within a specific prompt category.

    Args:
        df: DataFrame filtered to specific prompt_category
        category: Name of the category (for logging)
        class_extractor: Function to extract class label from answer
        target_ratio: Dict of class -> target ratio (should sum to 1.0)
        min_samples_per_class: Minimum samples to keep per class
        max_total_samples: Maximum total samples after balancing
        upsample_minority: Whether to upsample minority classes
        downsample_majority: Whether to downsample majority classes
        random_state: Random seed for reproducibility
    """
    if len(df) == 0:
        return df

    np.random.seed(random_state)

    # Extract class labels
    df = df.copy()
    df['_class'] = df['generated_answer'].apply(class_extractor)

    # Remove rows with unknown class
    df_with_class = df[df['_class'].notna()].copy()
    df_unknown = df[df['_class'].isna()].copy()

    if len(df_with_class) == 0:
        print(f"  [{category}] No valid class labels found, keeping all {len(df)} samples")
        return df.drop(columns=['_class'])

    # Get class distribution
    class_counts = df_with_class['_class'].value_counts()
    total = len(df_with_class)

    print(f"\n  [{category}] Original distribution ({total:,} samples):")
    for cls, count in class_counts.items():
        print(f"    {cls}: {count:,} ({count/total*100:.1f}%)")

    # Calculate target counts
    if target_ratio is None:
        # Default: equal distribution
        n_classes = len(class_counts)
        target_ratio = {cls: 1.0/n_classes for cls in class_counts.index}

    # Determine target sample count per class
    if max_total_samples:
        target_total = min(max_total_samples, total)
    else:
        target_total = total

    # Calculate target per class based on ratio
    target_counts = {}
    for cls, ratio in target_ratio.items():
        if cls in class_counts.index:
            target_counts[cls] = int(target_total * ratio)

    # Adjust to ensure we don't exceed available samples when downsampling only
    if not upsample_minority:
        for cls in target_counts:
            target_counts[cls] = min(target_counts[cls], class_counts[cls])

    # Balance each class
    balanced_dfs = []

    for cls in class_counts.index:
        cls_df = df_with_class[df_with_class['_class'] == cls]
        current_count = len(cls_df)
        target_count = target_counts.get(cls, current_count)

        if target_count <= 0:
            continue

        if current_count > target_count and downsample_majority:
            # Downsample
            sampled = cls_df.sample(n=target_count, random_state=random_state)
            balanced_dfs.append(sampled)
        elif current_count < target_count and upsample_minority:
            # Upsample with replacement
            n_to_add = target_count - current_count
            upsampled = cls_df.sample(n=n_to_add, replace=True, random_state=random_state)
            balanced_dfs.append(cls_df)
            balanced_dfs.append(upsampled)
        else:
            # Keep as-is
            balanced_dfs.append(cls_df)

    if balanced_dfs:
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    else:
        balanced_df = df_with_class

    # Add back unknown class samples (limited)
    if len(df_unknown) > 0:
        max_unknown = min(len(df_unknown), len(balanced_df) // 10)  # Max 10% unknown
        if max_unknown > 0:
            unknown_sample = df_unknown.sample(n=max_unknown, random_state=random_state)
            balanced_df = pd.concat([balanced_df, unknown_sample], ignore_index=True)

    # Report new distribution
    new_counts = balanced_df['_class'].value_counts()
    new_total = len(balanced_df)

    print(f"  [{category}] Balanced distribution ({new_total:,} samples):")
    for cls, count in new_counts.items():
        if pd.notna(cls):
            print(f"    {cls}: {count:,} ({count/new_total*100:.1f}%)")

    return balanced_df.drop(columns=['_class'])


def balance_interpretation_tasks(
    df: pd.DataFrame,
    target_percentage: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Reduce the dominance of interpretation tasks.

    Current: interpretation + json_interpretation = 43.5% of gradient
    Target: Reduce to ~15% combined
    """
    np.random.seed(random_state)

    interpretation_mask = df['prompt_category'].isin(['interpretation', 'json_interpretation'])
    other_mask = ~interpretation_mask

    n_other = other_mask.sum()
    n_interpretation = interpretation_mask.sum()

    # Calculate target interpretation samples to achieve target_percentage
    # If other = (1 - target_percentage) of total, and interp = target_percentage
    # Then interp = other * target_percentage / (1 - target_percentage)
    target_interpretation = int(n_other * target_percentage / (1 - target_percentage))

    print(f"\n  [interpretation] Current: {n_interpretation:,} ({n_interpretation/(n_interpretation+n_other)*100:.1f}%)")
    print(f"  [interpretation] Target: {target_interpretation:,} ({target_percentage*100:.1f}%)")

    if n_interpretation > target_interpretation:
        # Downsample interpretation tasks
        interp_df = df[interpretation_mask]

        # Split between interpretation and json_interpretation proportionally
        interp_only = interp_df[interp_df['prompt_category'] == 'interpretation']
        json_interp = interp_df[interp_df['prompt_category'] == 'json_interpretation']

        interp_ratio = len(interp_only) / len(interp_df) if len(interp_df) > 0 else 0.5

        n_interp_target = int(target_interpretation * interp_ratio)
        n_json_target = target_interpretation - n_interp_target

        sampled_interp = interp_only.sample(n=min(n_interp_target, len(interp_only)), random_state=random_state)
        sampled_json = json_interp.sample(n=min(n_json_target, len(json_interp)), random_state=random_state)

        # Combine
        other_df = df[other_mask]
        balanced_df = pd.concat([other_df, sampled_interp, sampled_json], ignore_index=True)

        print(f"  [interpretation] Downsampled: {len(sampled_interp) + len(sampled_json):,} samples")
        return balanced_df

    return df


def balance_dataset(
    input_path: str,
    output_path: str,
    balance_lvef: bool = True,
    balance_afib: bool = True,
    balance_acs: bool = True,
    balance_conduction: bool = True,
    reduce_interpretation: bool = True,
    interpretation_target_pct: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Main function to balance a QA dataset.

    Args:
        input_path: Path to input parquet
        output_path: Path to save balanced parquet
        balance_lvef: Balance LVEF categories
        balance_afib: Balance AFib risk categories
        balance_acs: Balance ACS severity categories
        balance_conduction: Balance conduction findings
        reduce_interpretation: Reduce interpretation task dominance
        interpretation_target_pct: Target percentage for interpretation tasks
        random_state: Random seed
    """
    print(f"\n{'='*80}")
    print("BALANCING DATASET")
    print(f"{'='*80}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    # Load dataset
    print("\n1. Loading dataset...")
    df = pd.read_parquet(input_path)
    original_count = len(df)
    print(f"   Loaded {original_count:,} samples")

    # Show original distribution
    print("\n2. Original prompt category distribution:")
    cat_dist = df['prompt_category'].value_counts()
    for cat, count in cat_dist.head(10).items():
        print(f"   {cat}: {count:,} ({count/original_count*100:.1f}%)")

    # Split by category for balancing
    print("\n3. Balancing categories...")

    balanced_dfs = []
    categories_to_balance = set()

    if balance_lvef:
        categories_to_balance.add('lvef')
    if balance_afib:
        categories_to_balance.add('afib_risk')
    if balance_acs:
        categories_to_balance.add('acs_severity')
    if balance_conduction:
        categories_to_balance.add('category_conduction')

    for category in df['prompt_category'].unique():
        cat_df = df[df['prompt_category'] == category].copy()

        if category == 'lvef' and balance_lvef:
            # Balance LVEF: equal distribution across severity categories
            balanced_cat = balance_category_classes(
                cat_df,
                category='lvef',
                class_extractor=lambda x: get_lvef_category(extract_lvef_value(x)),
                target_ratio={'normal': 0.25, 'mild': 0.25, 'moderate': 0.25, 'severe': 0.25},
                upsample_minority=True,
                downsample_majority=True,
                random_state=random_state,
            )
            balanced_dfs.append(balanced_cat)

        elif category == 'afib_risk' and balance_afib:
            # Balance AFib risk: equal low/high/moderate
            balanced_cat = balance_category_classes(
                cat_df,
                category='afib_risk',
                class_extractor=get_afib_risk_class,
                target_ratio={'low': 0.33, 'high': 0.33, 'moderate': 0.33, 'already_afib': 0.01},
                upsample_minority=True,
                downsample_majority=True,
                random_state=random_state,
            )
            balanced_dfs.append(balanced_cat)

        elif category == 'acs_severity' and balance_acs:
            # Balance ACS: equal acute/non-acute
            balanced_cat = balance_category_classes(
                cat_df,
                category='acs_severity',
                class_extractor=get_acs_class,
                target_ratio={'acute': 0.5, 'non_acute': 0.5},
                upsample_minority=True,
                downsample_majority=True,
                random_state=random_state,
            )
            balanced_dfs.append(balanced_cat)

        else:
            # Keep other categories as-is for now
            balanced_dfs.append(cat_df)

    # Combine balanced categories
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)

    # Reduce interpretation task dominance
    if reduce_interpretation:
        print("\n4. Reducing interpretation task dominance...")
        df_balanced = balance_interpretation_tasks(
            df_balanced,
            target_percentage=interpretation_target_pct,
            random_state=random_state,
        )

    # Shuffle final dataset
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Report final distribution
    final_count = len(df_balanced)
    print(f"\n5. Final distribution ({final_count:,} samples, {final_count/original_count*100:.1f}% of original):")

    final_dist = df_balanced['prompt_category'].value_counts()
    for cat, count in final_dist.head(15).items():
        print(f"   {cat}: {count:,} ({count/final_count*100:.1f}%)")

    # Calculate new gradient contributions
    print("\n6. New gradient contribution (with prompt_weight):")
    gradient_by_cat = df_balanced.groupby('prompt_category')['prompt_weight'].sum()
    total_gradient = gradient_by_cat.sum()
    for cat, grad in gradient_by_cat.sort_values(ascending=False).head(10).items():
        print(f"   {cat}: {grad/total_gradient*100:.1f}%")

    # Save balanced dataset
    print(f"\n7. Saving balanced dataset...")
    df_balanced.to_parquet(output_path, index=False)
    print(f"   Saved to: {output_path}")

    # Also save a sample CSV for inspection
    sample_csv = output_path.replace('.parquet', '_sample.csv')
    df_balanced.head(500).to_csv(sample_csv, index=False)
    print(f"   Sample CSV: {sample_csv}")

    print(f"\n{'='*80}")
    print(f"BALANCING COMPLETE: {original_count:,} -> {final_count:,} samples")
    print(f"{'='*80}")

    return df_balanced


def main():
    parser = argparse.ArgumentParser(description="Balance minority classes in QA datasets")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input parquet file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet file path (default: input_balanced.parquet)"
    )
    parser.add_argument(
        "--no-lvef",
        action="store_true",
        help="Skip LVEF balancing"
    )
    parser.add_argument(
        "--no-afib",
        action="store_true",
        help="Skip AFib risk balancing"
    )
    parser.add_argument(
        "--no-acs",
        action="store_true",
        help="Skip ACS severity balancing"
    )
    parser.add_argument(
        "--no-interpretation-reduction",
        action="store_true",
        help="Skip reducing interpretation task dominance"
    )
    parser.add_argument(
        "--interpretation-pct",
        type=float,
        default=0.15,
        help="Target percentage for interpretation tasks (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Set output path
    if args.output is None:
        args.output = args.input.replace('.parquet', '_balanced.parquet')

    balance_dataset(
        input_path=args.input,
        output_path=args.output,
        balance_lvef=not args.no_lvef,
        balance_afib=not args.no_afib,
        balance_acs=not args.no_acs,
        reduce_interpretation=not args.no_interpretation_reduction,
        interpretation_target_pct=args.interpretation_pct,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
