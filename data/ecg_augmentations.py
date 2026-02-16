"""ECG signal augmentation transforms for training-time data perturbation.

Each transform operates on (2500, 12) numpy arrays and returns the same shape.
Augmentations are applied AFTER normalization (on the already-adjusted signal).
"""

import random
import numpy as np


def gaussian_noise(waveform: np.ndarray) -> np.ndarray:
    """Add Gaussian noise at a random SNR between 20-40 dB."""
    snr_db = random.uniform(20.0, 40.0)
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


def baseline_wander(waveform: np.ndarray) -> np.ndarray:
    """Add low-frequency sinusoidal baseline drift (0.1-0.5 Hz)."""
    n_samples = waveform.shape[0]
    freq = random.uniform(0.1, 0.5)
    # Assume 500 Hz sampling rate for 2500 samples = 5 seconds
    t = np.arange(n_samples) / 500.0
    phase = random.uniform(0, 2 * np.pi)
    amplitude = random.uniform(0.01, 0.05)
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
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob
        self.transforms = [
            gaussian_noise,
            amplitude_scale,
            baseline_wander,
            temporal_shift,
            lead_dropout,
        ]

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a (2500, 12) waveform."""
        if random.random() > self.prob:
            return waveform
        n_transforms = random.randint(1, 3)
        selected = random.sample(self.transforms, min(n_transforms, len(self.transforms)))
        for transform in selected:
            waveform = transform(waveform)
        return waveform
