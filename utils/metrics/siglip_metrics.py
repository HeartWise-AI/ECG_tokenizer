"""Utility metrics for SigLIP ECG↔text alignment."""

from __future__ import annotations

from typing import Iterable

import torch


def compute_alignment(pos_prob: float, neg_prob: float) -> float:
    """Alignment score defined as positive probability minus negative probability."""
    return float(pos_prob - neg_prob)


def compute_recall_at_k(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    """Recall@k across rows where at least one positive label is present."""
    results = compute_recall_at_many(logits, labels, ks=[k])
    value, count = results.get(int(k), (0.0, 0))
    if count == 0:
        return 0.0
    return float(value / count)


def aggregate_probabilities(probabilities: Iterable[torch.Tensor]) -> float:
    """Average a sequence of probability tensors, guarding against empties."""
    tensors = [p for p in probabilities if p.numel() > 0]
    if not tensors:
        return 0.0
    concat = torch.cat(tensors)
    return float(concat.mean().item())

def compute_recall_at_many(logits: torch.Tensor, labels: torch.Tensor, ks: Iterable[int]) -> dict[int, tuple[float, int]]:
    """Compute recall@k for multiple values of k, returning (mean, count)."""
    if logits.numel() == 0 or labels.numel() == 0:
        return {int(k): (0.0, 0) for k in ks}

    ks = sorted({int(k) for k in ks if int(k) > 0})
    if not ks:
        return {}

    max_k = min(ks[-1], logits.size(1))
    if max_k <= 0:
        return {int(k): (0.0, 0) for k in ks}

    topk_idx = torch.topk(logits, k=max_k, dim=1).indices
    results: dict[int, tuple[float, int]] = {}
    for k in ks:
        slice_end = min(k, topk_idx.size(1))
        if slice_end <= 0:
            results[int(k)] = (0.0, 0)
            continue
        recall_values: list[torch.Tensor] = []
        for row in range(logits.size(0)):
            positives = torch.nonzero(labels[row] > 0.5, as_tuple=False).flatten()
            if positives.numel() == 0:
                continue
            hits = torch.isin(positives, topk_idx[row, :slice_end])
            recall_values.append(hits.float().mean())
        if recall_values:
            stacked = torch.stack(recall_values)
            results[int(k)] = (float(stacked.sum().item()), len(recall_values))
        else:
            results[int(k)] = (0.0, 0)
    return results


__all__ = [
    "compute_alignment",
    "compute_recall_at_k",
    "compute_recall_at_many",
    "aggregate_probabilities",
]
