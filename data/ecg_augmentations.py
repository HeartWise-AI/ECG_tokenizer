from __future__ import annotations

import functools
from typing import Callable, Sequence, Tuple

import numpy as np


def global_amplitude_scale(
    signal: np.ndarray,
    scale_range: Tuple[float, float] = (0.8, 1.2),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Scale all leads by the same random factor drawn from *scale_range*.

    This simulates the amplitude mismatch between different ECG acquisition
    pipelines (e.g. XML vs PSA) more faithfully than per-lead independent
    scaling, because the mismatch is a global gain difference.

    Parameters
    ----------
    signal : np.ndarray
        Shape ``(time, leads)`` — float32 recommended.
    scale_range : tuple of float
        ``(min, max)`` for the uniform distribution.
    rng : np.random.Generator, optional
        Explicit RNG for reproducibility.  If ``None`` a fresh default
        generator is used (non-deterministic).
    """
    if rng is None:
        rng = np.random.default_rng()
    factor = rng.uniform(scale_range[0], scale_range[1])
    return (signal * factor).astype(signal.dtype, copy=False)


class ECGAugmentor:
    """Composable ECG augmentation pipeline.

    Parameters
    ----------
    amplitude_scale_range : tuple of float, optional
        ``(min, max)`` for the global amplitude augmentation.
        Set to ``None`` to disable amplitude augmentation.
    seed : int or None, optional
        Seed for the internal RNG.  ``None`` → non-deterministic.
    """

    def __init__(
        self,
        amplitude_scale_range: Tuple[float, float] | None = (0.8, 1.2),
        seed: int | None = None,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.transforms: list[Callable[[np.ndarray, np.random.Generator], np.ndarray]] = []

        if amplitude_scale_range is not None:
            self.transforms.append(
                functools.partial(
                    _amplitude_transform,
                    scale_range=amplitude_scale_range,
                )
            )

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            signal = transform(signal, self.rng)
        return signal


def _amplitude_transform(
    signal: np.ndarray,
    rng: np.random.Generator,
    scale_range: Tuple[float, float] = (0.8, 1.2),
) -> np.ndarray:
    return global_amplitude_scale(signal, scale_range=scale_range, rng=rng)
