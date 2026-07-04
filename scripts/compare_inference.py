#!/usr/bin/env python3
"""
Compare inference outputs with training validation JSON files.

Usage:
    python scripts/compare_inference.py \
        --inference_csv /path/to/inference.csv \
        --reference_json /path/to/val_generations.json
"""

import argparse
import json
import pandas as pd
from difflib import SequenceMatcher


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, collapse whitespace)."""
    return ' '.join(text.lower().split())


def similarity_ratio(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, a, b).ratio()


def main():
    parser = argparse.ArgumentParser(description="Compare inference outputs with training validation")
    parser.add_argument("--inference_csv", type=str, required=True, help="Path to inference CSV")
    parser.add_argument("--reference_json", type=str, required=True, help="Path to training validation JSON")
    parser.add_argument("--show_mismatches", type=int, default=5, help="Number of mismatches to show")
    args = parser.parse_args()

    # Load reference (training validation)
    with open(args.reference_json) as f:
        ref_data = json.load(f)

    # Load inference results
    inf_df = pd.read_csv(args.inference_csv)

    # Compare
    exact_matches = 0
    normalized_matches = 0
    high_similarity = 0  # >90% similar
    total_compared = 0
    mismatches = []
    similarities = []

    for _, row in inf_df.iterrows():
        waveform = row['waveform_name']

        # Find in reference (try with and without .npy suffix)
        ref_key = None
        for candidate in [waveform, f"{waveform}.npy", waveform.replace('.npy', '')]:
            if candidate in ref_data and not candidate.startswith('__'):
                ref_key = candidate
                break

        if ref_key is None:
            continue

        total_compared += 1
        ref_gen = ref_data[ref_key]['Generation'].strip()
        inf_gen = row['generation'].strip()

        # Calculate similarity
        sim = similarity_ratio(ref_gen, inf_gen)
        similarities.append(sim)

        # Exact match
        if ref_gen == inf_gen:
            exact_matches += 1
        elif normalize_text(ref_gen) == normalize_text(inf_gen):
            normalized_matches += 1
        elif sim > 0.9:
            high_similarity += 1
        else:
            mismatches.append({
                'waveform': waveform,
                'ref': ref_gen,
                'inf': inf_gen,
                'similarity': sim,
            })

    # Print results
    print("=" * 70)
    print("INFERENCE COMPARISON RESULTS")
    print("=" * 70)
    print(f"Total compared: {total_compared}")
    print(f"Exact matches: {exact_matches} ({100*exact_matches/total_compared:.1f}%)")
    print(f"Normalized matches: {normalized_matches} ({100*normalized_matches/total_compared:.1f}%)")
    print(f"High similarity (>90%): {high_similarity} ({100*high_similarity/total_compared:.1f}%)")
    combined = exact_matches + normalized_matches + high_similarity
    print(f"Combined (exact + normalized + high sim): {combined} ({100*combined/total_compared:.1f}%)")
    print(f"Mismatches (<90% similarity): {len(mismatches)}")

    if similarities:
        import numpy as np
        print(f"\nSimilarity statistics:")
        print(f"  Mean: {np.mean(similarities):.3f}")
        print(f"  Median: {np.median(similarities):.3f}")
        print(f"  Min: {np.min(similarities):.3f}")
        print(f"  Max: {np.max(similarities):.3f}")

    if mismatches and args.show_mismatches > 0:
        # Sort by similarity (lowest first)
        mismatches.sort(key=lambda x: x['similarity'])
        print(f"\n{'='*70}")
        print(f"LOWEST SIMILARITY MISMATCHES (showing {min(args.show_mismatches, len(mismatches))})")
        print("=" * 70)
        for i, m in enumerate(mismatches[:args.show_mismatches]):
            print(f"\n--- [{i+1}] {m['waveform']} (similarity: {m['similarity']:.2f}) ---")
            print(f"REF: {m['ref'][:200]}{'...' if len(m['ref']) > 200 else ''}")
            print(f"INF: {m['inf'][:200]}{'...' if len(m['inf']) > 200 else ''}")


if __name__ == "__main__":
    main()
