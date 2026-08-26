"""Strict ECG waveform loading for evaluation and probe scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_ecg_waveform(
    path: str | Path,
    *,
    target_length: int = 2500,
    num_leads: int = 12,
) -> np.ndarray:
    """Load one ECG as contiguous float32 [leads, time], or fail with its path."""
    source = Path(path)
    try:
        waveform = np.load(source).astype(np.float32, copy=False)
    except Exception as exc:
        raise RuntimeError(f"failed to load ECG waveform {source}: {exc}") from exc
    if waveform.ndim == 3 and waveform.shape[-1] == 1:
        waveform = waveform.squeeze(-1)
    if waveform.ndim != 2:
        raise ValueError(
            f"ECG waveform {source} must be rank 2 after squeezing, got {waveform.shape}"
        )
    if waveform.shape[1] == num_leads:
        pass
    elif waveform.shape[0] == num_leads:
        waveform = waveform.T
    else:
        raise ValueError(
            f"ECG waveform {source} must have {num_leads} leads, got {waveform.shape}"
        )
    if waveform.shape[0] == 0:
        raise ValueError(f"ECG waveform {source} has no time samples")
    if not np.isfinite(waveform).all():
        raise ValueError(f"ECG waveform {source} contains non-finite samples")
    if waveform.shape[0] > target_length:
        stride = max(1, waveform.shape[0] // target_length)
        waveform = waveform[::stride, :]
    if waveform.shape[0] < target_length:
        waveform = np.pad(
            waveform,
            ((0, target_length - waveform.shape[0]), (0, 0)),
        )
    return np.ascontiguousarray(waveform[:target_length].T, dtype=np.float32)
