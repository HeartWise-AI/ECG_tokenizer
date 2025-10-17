"""Simple logging helpers for SigLIP training."""

from __future__ import annotations

import os
from typing import Any, Iterable

import torch


def log_once(message: str, rank: int = 0, prefix: str = "[SigLIP]") -> None:
    if rank == 0:
        print(f"{prefix} {message}")


def summarize_tensor(name: str, tensor: torch.Tensor, rank: int = 0) -> None:
    if rank != 0:
        return
    if tensor.numel() == 0:
        print(f"[SigLIP] {name}: empty tensor")
        return
    mean = tensor.mean().item()
    std = tensor.float().std().item()
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    print(f"[SigLIP] {name}: mean={mean:.4f} std={std:.4f} min={min_val:.4f} max={max_val:.4f}")


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def format_metrics(metrics: dict[str, Any]) -> str:
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)
