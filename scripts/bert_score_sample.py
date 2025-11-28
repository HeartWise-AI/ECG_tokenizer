#!/usr/bin/env python3
"""
Compute BERTScore on a random sample from a generations JSON file.

Usage:
  python scripts/bert_score_sample.py <val_generations.json> [--n 50] [--model microsoft/deberta-xlarge-mnli]

The JSON file is expected to map filenames to objects with keys:
  - "Generation"
  - "Ground truth"
"""
import argparse
import json
import random
import os
import sys
from typing import List, Tuple

# Ensure repository root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.metrics.llm_metrics import compute_bertscore_offline


def load_pairs(path: str) -> List[Tuple[str, str]]:
    with open(path, 'r') as f:
        data = json.load(f)
    pairs: List[Tuple[str, str]] = []
    for _, entry in data.items():
        pred = entry.get('Generation', '')
        ref = entry.get('Ground truth', '')
        if isinstance(pred, str) and isinstance(ref, str) and pred.strip() and ref.strip():
            pairs.append((pred, ref))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('json_path', type=str, help='Path to val_generations JSON file')
    ap.add_argument('--n', type=int, default=50, help='Number of random samples')
    ap.add_argument('--seed', type=int, default=42, help='Random seed')
    ap.add_argument('--model', type=str, default='microsoft/deberta-xlarge-mnli', help='BERTScore model')
    ap.add_argument('--batch_size', type=int, default=16, help='Batch size for BERTScore')
    args = ap.parse_args()

    random.seed(args.seed)
    pairs = load_pairs(args.json_path)
    if not pairs:
        raise SystemExit('No valid (Generation, Ground truth) pairs found in JSON.')

    if args.n > 0 and args.n < len(pairs):
        pairs = random.sample(pairs, args.n)

    preds = [p for p, _ in pairs]
    refs = [r for _, r in pairs]

    scores = compute_bertscore_offline(
        predictions=preds,
        references=refs,
        model_type=args.model,
        batch_size=args.batch_size,
    )
    print(f"Samples: {scores.get('n_samples', len(preds))}")
    print(f"BERTScore P: {scores['precision']:.4f}")
    print(f"BERTScore R: {scores['recall']:.4f}")
    print(f"BERTScore F1: {scores['f1']:.4f}")


if __name__ == '__main__':
    main()
