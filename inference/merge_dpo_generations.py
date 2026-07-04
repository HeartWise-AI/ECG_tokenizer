#!/usr/bin/env python3
"""Merge fixed error generations back into the main generations file."""

import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", default="output/dpo_inference/dpo_5gen_generations.json")
    parser.add_argument("--fixed", default="output/dpo_inference/dpo_5gen_errors_fixed_generations.json")
    parser.add_argument("--output", default="output/dpo_inference/dpo_5gen_generations_merged.json")
    args = parser.parse_args()

    print(f"Loading original: {args.original}")
    with open(args.original, 'r') as f:
        original = json.load(f)

    print(f"Loading fixed: {args.fixed}")
    with open(args.fixed, 'r') as f:
        fixed = json.load(f)

    # Build lookup from fixed samples by their original_idx
    # The fixed file contains items where the parquet had 'original_idx' column
    # But the generations use sequential idx (0, 1, 2...) from the error parquet
    # We need to map back to original indices

    # Load the error parquet to get the mapping
    import pandas as pd
    error_df = pd.read_parquet("output/dpo_inference/error_samples.parquet")

    # Create mapping: sequential idx in fixed -> original idx
    fixed_to_original = {i: int(row['original_idx']) for i, row in error_df.iterrows()}

    # Also create direct lookup by idx in fixed file
    fixed_lookup = {}
    for item in fixed:
        seq_idx = item['idx']
        if seq_idx in fixed_to_original:
            orig_idx = fixed_to_original[seq_idx]
            fixed_lookup[orig_idx] = item

    print(f"Original samples: {len(original)}")
    print(f"Fixed samples: {len(fixed)}")
    print(f"Mapped fixed samples: {len(fixed_lookup)}")

    # Merge: replace error samples in original with fixed versions
    replaced = 0
    still_errors = 0
    for i, item in enumerate(original):
        orig_idx = item['idx']
        if orig_idx in fixed_lookup:
            # Check if original had errors
            has_error = any('ERROR' in str(g.get('output', '')) for g in item.get('generations', []))
            if has_error:
                fixed_item = fixed_lookup[orig_idx]
                # Check if fixed version still has errors
                fixed_has_error = any('ERROR' in str(g.get('output', '')) for g in fixed_item.get('generations', []))
                if not fixed_has_error:
                    # Replace generations but keep original metadata
                    original[i]['generations'] = fixed_item['generations']
                    replaced += 1
                else:
                    still_errors += 1

    print(f"Replaced: {replaced}")
    print(f"Still have errors: {still_errors}")

    # Count remaining errors
    remaining_errors = sum(
        1 for item in original
        if any('ERROR' in str(g.get('output', '')) for g in item.get('generations', []))
    )
    print(f"Total remaining errors after merge: {remaining_errors}")

    # Save merged
    print(f"Saving to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(original, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
