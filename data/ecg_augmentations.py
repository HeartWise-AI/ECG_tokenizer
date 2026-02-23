"""ECG signal augmentation transforms for training-time data perturbation.

Each transform operates on (2500, 12) numpy arrays and returns the same shape.
Augmentations are applied AFTER normalization (on the already-adjusted signal).
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np


def gaussian_noise(
    waveform: np.ndarray,
    snr_range: Tuple[float, float] = (20.0, 40.0),
) -> np.ndarray:
    """Add Gaussian noise at a random SNR drawn from *snr_range* (dB)."""
    snr_db = random.uniform(snr_range[0], snr_range[1])
    signal_power = np.mean(waveform ** 2)
    if signal_power < 1e-12:
        return waveform
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), waveform.shape)
    return waveform + noise


def amplitude_scale(waveform: np.ndarray) -> np.ndarray:
    """Scale amplitude by 0.8-1.2x independently per lead."""
    scales = np.random.uniform(0.8, 1.2, size=(1, waveform.shape[1]))
    return waveform * scales


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


def baseline_wander(
    waveform: np.ndarray,
    amplitude_range: Tuple[float, float] = (0.01, 0.05),
) -> np.ndarray:
    """Add low-frequency sinusoidal baseline drift (0.1-0.5 Hz)."""
    n_samples = waveform.shape[0]
    freq = random.uniform(0.1, 0.5)
    # Assume 500 Hz sampling rate for 2500 samples = 5 seconds
    t = np.arange(n_samples) / 500.0
    phase = random.uniform(0, 2 * np.pi)
    amplitude = random.uniform(amplitude_range[0], amplitude_range[1])
    drift = amplitude * np.sin(2 * np.pi * freq * t + phase)
    # Apply to random subset of leads
    n_leads = waveform.shape[1]
    leads_affected = random.sample(range(n_leads), random.randint(1, n_leads))
    result = waveform.copy()
    for lead in leads_affected:
        result[:, lead] += drift
    return result


def temporal_shift(waveform: np.ndarray) -> np.ndarray:
    """Shift signal by +/-50 samples with edge padding."""
    shift = random.randint(-50, 50)
    if shift == 0:
        return waveform
    result = np.empty_like(waveform)
    if shift > 0:
        result[:shift, :] = waveform[0, :]
        result[shift:, :] = waveform[:-shift, :]
    else:
        result[shift:, :] = waveform[-1, :]
        result[:shift, :] = waveform[-shift:, :]
    return result


def lead_dropout(waveform: np.ndarray) -> np.ndarray:
    """Zero out 1-2 random leads with probability 0.1."""
    if random.random() > 0.1:
        return waveform
    n_leads = waveform.shape[1]
    n_drop = random.randint(1, min(2, n_leads))
    leads_to_drop = random.sample(range(n_leads), n_drop)
    result = waveform.copy()
    for lead in leads_to_drop:
        result[:, lead] = 0.0
    return result


class ECGAugmentor:
    """Apply random ECG augmentations to waveform signals during training.

    Args:
        prob: Probability of applying any augmentation to a given sample.
        amplitude_scale_range: (min, max) for global amplitude augmentation.
            Set to None to disable. When enabled, this is always applied
            (independent of prob) to simulate cross-dataset gain mismatch.
        amplitude_scale_seed: Seed for the global amplitude scale RNG.
        noise_snr_range: (min_dB, max_dB) for Gaussian noise SNR.
            Lower values = louder noise.
        wander_amplitude_range: (min, max) for baseline wander amplitude.
            Higher values = more visible drift.
    """

    def __init__(
        self,
        prob: float = 0.5,
        amplitude_scale_range: Tuple[float, float] | None = (0.8, 1.2),
        amplitude_scale_seed: int | None = None,
        noise_snr_range: Tuple[float, float] = (20.0, 40.0),
        wander_amplitude_range: Tuple[float, float] = (0.01, 0.05),
    ):
        self.prob = prob
        self.amplitude_scale_range = amplitude_scale_range
        self.amplitude_rng = np.random.default_rng(amplitude_scale_seed) if amplitude_scale_range else None
        self.noise_snr_range = noise_snr_range
        self.wander_amplitude_range = wander_amplitude_range
        self._build_transforms()

    def _build_transforms(self):
        """Build the list of augmentation transforms with bound parameters."""
        from functools import partial
        self.transforms = [
            partial(gaussian_noise, snr_range=self.noise_snr_range),
            amplitude_scale,
            partial(baseline_wander, amplitude_range=self.wander_amplitude_range),
            temporal_shift,
            lead_dropout,
        ]

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a (2500, 12) waveform."""
        # Global amplitude scale is always applied (simulates dataset mismatch)
        if self.amplitude_scale_range is not None:
            waveform = global_amplitude_scale(
                waveform,
                scale_range=self.amplitude_scale_range,
                rng=self.amplitude_rng,
            )

        if random.random() > self.prob:
            return waveform
        n_transforms = random.randint(1, 3)
        selected = random.sample(self.transforms, min(n_transforms, len(self.transforms)))
        for transform in selected:
            waveform = transform(waveform)
        return waveform
