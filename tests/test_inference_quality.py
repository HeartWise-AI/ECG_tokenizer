#!/usr/bin/env python3
"""
Test script to validate inference output quality against ground truth.
Used as a Claude stop hook to verify inference generates quality outputs.

Usage:
    python tests/test_inference_quality.py <inference_json_path> [--threshold 0.4] [--samples 5]

Exit codes:
    0 = PASS (metrics above threshold)
    1 = FAIL (metrics below threshold)
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any


def compute_metrics(generations: List[str], ground_truths: List[str]) -> Dict[str, float]:
    """Compute ROUGE, BLEU, and METEOR metrics."""
    try:
        from rouge_score import rouge_scorer
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.translate.meteor_score import meteor_score
        from nltk import word_tokenize
        import nltk
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        sys.exit(1)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoother = SmoothingFunction()

    rouge1, rouge2, rougeL = [], [], []
    bleu1, bleu4 = [], []
    meteor_scores = []

    for gen, ref in zip(generations, ground_truths):
        if not gen or not ref:
            continue

        # ROUGE scores
        scores = scorer.score(ref, gen)
        rouge1.append(scores['rouge1'].fmeasure)
        rouge2.append(scores['rouge2'].fmeasure)
        rougeL.append(scores['rougeL'].fmeasure)

        # BLEU and METEOR
        try:
            ref_tokens = word_tokenize(ref.lower())
            gen_tokens = word_tokenize(gen.lower())
            bleu1.append(sentence_bleu([ref_tokens], gen_tokens, weights=(1, 0, 0, 0),
                                       smoothing_function=smoother.method1))
            bleu4.append(sentence_bleu([ref_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25),
                                       smoothing_function=smoother.method1))
            meteor_scores.append(meteor_score([ref_tokens], gen_tokens))
        except Exception:
            pass

    return {
        'rouge1': np.mean(rouge1) if rouge1 else 0.0,
        'rouge2': np.mean(rouge2) if rouge2 else 0.0,
        'rougeL': np.mean(rougeL) if rougeL else 0.0,
        'bleu1': np.mean(bleu1) if bleu1 else 0.0,
        'bleu4': np.mean(bleu4) if bleu4 else 0.0,
        'meteor': np.mean(meteor_scores) if meteor_scores else 0.0,
    }


def load_inference_json(json_path: str) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """Load inference JSON and extract generations and ground truths."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    all_samples = []
    generations = []
    ground_truths = []

    for ecg_name, entries in data.items():
        if isinstance(entries, list):
            for entry in entries:
                sample = {
                    'ecg': ecg_name,
                    'question': entry.get('Question', ''),
                    'generation': entry.get('Generation', ''),
                    'ground_truth': entry.get('Ground truth', ''),
                    'category': entry.get('Category', ''),
                }
                all_samples.append(sample)
                generations.append(sample['generation'])
                ground_truths.append(sample['ground_truth'])
        elif isinstance(entries, dict):
            # Single entry format (like val_generations files)
            sample = {
                'ecg': ecg_name,
                'question': entries.get('Question', ''),
                'generation': entries.get('Generation', ''),
                'ground_truth': entries.get('Ground truth', ''),
                'category': entries.get('Category', ''),
            }
            all_samples.append(sample)
            generations.append(sample['generation'])
            ground_truths.append(sample['ground_truth'])

    return all_samples, generations, ground_truths


def find_worst_samples(samples: List[Dict[str, Any]], n: int = 5) -> List[Dict[str, Any]]:
    """Find the worst performing samples (shortest generations relative to ground truth)."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    except ImportError:
        # Fallback: sort by generation length ratio
        scored = []
        for s in samples:
            gen_len = len(s['generation']) if s['generation'] else 0
            gt_len = len(s['ground_truth']) if s['ground_truth'] else 1
            ratio = gen_len / gt_len
            scored.append((ratio, s))
        scored.sort(key=lambda x: x[0])
        return [s for _, s in scored[:n]]

    scored = []
    for s in samples:
        if s['generation'] and s['ground_truth']:
            score = scorer.score(s['ground_truth'], s['generation'])['rougeL'].fmeasure
            scored.append((score, s))

    scored.sort(key=lambda x: x[0])
    return [s for _, s in scored[:n]]


def main():
    parser = argparse.ArgumentParser(description='Test inference quality')
    parser.add_argument('inference_json', type=str, help='Path to inference output JSON')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='Minimum acceptable metric value (default: 0.4)')
    parser.add_argument('--samples', type=int, default=5,
                        help='Number of sample examples to show (default: 5)')
    parser.add_argument('--reference', type=str, default=None,
                        help='Optional: path to reference/training validation JSON for comparison')
    args = parser.parse_args()

    # Check file exists
    if not Path(args.inference_json).exists():
        print(f"ERROR: File not found: {args.inference_json}")
        sys.exit(1)

    print("=" * 80)
    print("INFERENCE QUALITY TEST")
    print("=" * 80)
    print(f"Input: {args.inference_json}")
    print(f"Threshold: {args.threshold}")
    print()

    # Load and compute metrics
    samples, generations, ground_truths = load_inference_json(args.inference_json)
    print(f"Loaded {len(samples)} samples")

    if len(samples) == 0:
        print("ERROR: No samples found in JSON")
        sys.exit(1)

    metrics = compute_metrics(generations, ground_truths)

    # Print metrics
    print("\n" + "-" * 40)
    print("METRICS")
    print("-" * 40)
    for name, value in metrics.items():
        status = "✓" if value >= args.threshold else "✗"
        print(f"  {name.upper():>8}: {value:.4f} {status}")

    # Check pass/fail
    key_metrics = ['rougeL', 'bleu1', 'meteor']
    passed = all(metrics[m] >= args.threshold for m in key_metrics)

    # Load reference for comparison if provided
    if args.reference and Path(args.reference).exists():
        print("\n" + "-" * 40)
        print("REFERENCE COMPARISON")
        print("-" * 40)
        ref_samples, ref_gens, ref_gts = load_inference_json(args.reference)
        ref_metrics = compute_metrics(ref_gens, ref_gts)
        print(f"Reference ({len(ref_samples)} samples):")
        for name in key_metrics:
            diff = metrics[name] - ref_metrics[name]
            sign = "+" if diff >= 0 else ""
            print(f"  {name.upper():>8}: {ref_metrics[name]:.4f} -> {metrics[name]:.4f} ({sign}{diff:.4f})")

    # Show sample outputs
    print("\n" + "-" * 40)
    print(f"SAMPLE OUTPUTS (worst {args.samples})")
    print("-" * 40)
    worst_samples = find_worst_samples(samples, args.samples)
    for i, s in enumerate(worst_samples, 1):
        print(f"\n[{i}] ECG: {s['ecg']} | Category: {s['category']}")
        print(f"    Q: {s['question'][:80]}...")
        print(f"    GEN: {s['generation'][:100]}")
        print(f"    GT:  {s['ground_truth'][:100]}...")

    # Final verdict
    print("\n" + "=" * 80)
    if passed:
        print("RESULT: ✓ PASS")
        print(f"All key metrics (ROUGE-L, BLEU-1, METEOR) >= {args.threshold}")
        sys.exit(0)
    else:
        print("RESULT: ✗ FAIL")
        failed_metrics = [m for m in key_metrics if metrics[m] < args.threshold]
        print(f"Failed metrics: {', '.join(m.upper() for m in failed_metrics)}")
        print(f"Expected >= {args.threshold}")
        print("\nDEBUG HINTS:")
        print("  - Check if task_hint is correctly derived (not forcing binary mode)")
        print("  - Verify prompt format matches training exactly")
        print("  - Ensure ECG embeddings are injected after <start_of_image>")
        print("  - Compare generation kwargs with training validation")
        sys.exit(1)


if __name__ == "__main__":
    main()
