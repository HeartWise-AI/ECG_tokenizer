#!/usr/bin/env python3
"""
PSA (Power Spectral Alignment) Normalizer for ECG signals.

This module provides PSA normalization for raw ECG waveforms to prepare them
for the ECG tokenizer inference pipeline.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.interpolate import interp1d
from concurrent.futures import ThreadPoolExecutor


class PSANormalizer:
    """
    Power Spectral Alignment normalizer for ECG signals.
    
    Applies frequency-domain filtering to remove powerline interference
    and normalize spectral characteristics across different ECG sources.
    """
    
    def __init__(
        self, 
        sampling_rate: int = 250,
        target_length: int = 2500,
        num_leads: int = 12,
        powerline_freq: float = 60.0,
        powerline_harmonics: List[float] = None
    ):
        """
        Initialize the PSA normalizer.
        
        Args:
            sampling_rate: Sampling rate of the ECG signal in Hz
            target_length: Target length of the normalized signal
            num_leads: Number of ECG leads
            powerline_freq: Powerline frequency to filter (60 Hz in NA, 50 Hz in EU)
            powerline_harmonics: List of harmonic frequencies to filter
        """
        self.sampling_rate = sampling_rate
        self.target_length = target_length
        self.num_leads = num_leads
        self.powerline_freq = powerline_freq
        
        if powerline_harmonics is None:
            self.powerline_harmonics = [
                powerline_freq,
                powerline_freq * 2,
                powerline_freq * 3
            ]
        else:
            self.powerline_harmonics = powerline_harmonics
        
        self.flatten_ranges = self._compute_flatten_ranges()
    
    def _compute_flatten_ranges(self) -> List[Tuple[float, float]]:
        """Compute frequency ranges to flatten based on powerline harmonics."""
        ranges = []
        for freq in self.powerline_harmonics:
            ranges.append((freq - 0.5, freq + 0.5))
        return ranges
    
    def _flatten_fft_peak(
        self, 
        signal: np.ndarray, 
        flatten_ranges: List[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Flatten peaks in the FFT spectrum to remove powerline interference.
        
        Args:
            signal: 1D signal array
            flatten_ranges: List of (start_freq, end_freq) tuples to flatten
            
        Returns:
            Filtered signal with powerline interference removed
        """
        if flatten_ranges is None:
            flatten_ranges = self.flatten_ranges
            
        if np.isnan(signal).any():
            signal = np.nan_to_num(signal)
        
        fft_result = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/self.sampling_rate)
        n = len(signal) // 2

        fft_phase = np.angle(fft_result)
        fft_magnitude = np.abs(fft_result)
        
        for flatten_range in flatten_ranges:
            if flatten_range[0] >= flatten_range[1]:
                continue

            start_idx = np.searchsorted(freqs[:n], flatten_range[0], side='left')
            end_idx = np.searchsorted(freqs[:n], flatten_range[1], side='right') - 1

            start_idx = max(start_idx, 0)
            end_idx = min(end_idx, n - 1)

            if start_idx != end_idx:
                interp_mag = interp1d(
                    [freqs[start_idx], freqs[end_idx]], 
                    [fft_magnitude[start_idx], fft_magnitude[end_idx]], 
                    kind='linear', 
                    fill_value="extrapolate"
                )
                
                indices = np.arange(start_idx, end_idx + 1)
                interpolated_magnitudes = interp_mag(freqs[indices])

                fft_result[indices] = interpolated_magnitudes * np.exp(1j * fft_phase[indices])
                
                if start_idx > 0:
                    fft_result[-indices] = interpolated_magnitudes * np.exp(1j * fft_phase[-indices])

        modified_signal = np.fft.ifft(fft_result)
        return modified_signal.real
    
    def _normalize_lead(self, lead_signal: np.ndarray) -> np.ndarray:
        """
        Normalize a single lead signal.
        
        Args:
            lead_signal: 1D array of lead values
            
        Returns:
            Normalized lead signal
        """
        filtered = self._flatten_fft_peak(lead_signal.astype(np.float32))
        
        mean_val = np.mean(filtered)
        std_val = np.std(filtered)
        if std_val > 0:
            normalized = (filtered - mean_val) / std_val
        else:
            normalized = filtered - mean_val
            
        return normalized.astype(np.float32)
    
    def normalize_waveform(
        self, 
        waveform: np.ndarray,
        parallel: bool = False,
        max_workers: int = 4
    ) -> np.ndarray:
        """
        Normalize a single ECG waveform.
        
        Args:
            waveform: ECG waveform of shape (length, num_leads) or (num_leads, length)
            parallel: Whether to process leads in parallel
            max_workers: Number of parallel workers
            
        Returns:
            Normalized waveform of shape (num_leads, target_length)
        """
        if waveform.ndim == 3:
            waveform = waveform.squeeze(-1)
        
        if waveform.shape[0] == self.num_leads and waveform.shape[1] != self.num_leads:
            waveform = waveform.T
        
        if waveform.shape[0] != self.target_length:
            if waveform.shape[0] > self.target_length:
                step = waveform.shape[0] // self.target_length
                waveform = waveform[::step, :][:self.target_length, :]
            else:
                pad_length = self.target_length - waveform.shape[0]
                waveform = np.pad(waveform, ((0, pad_length), (0, 0)), mode='edge')
        
        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self._normalize_lead, waveform[:, i])
                    for i in range(self.num_leads)
                ]
                normalized_leads = [f.result() for f in futures]
        else:
            normalized_leads = [
                self._normalize_lead(waveform[:, i])
                for i in range(self.num_leads)
            ]
        
        normalized = np.stack(normalized_leads, axis=0)
        return normalized.astype(np.float32)
    
    def normalize_batch(
        self, 
        waveforms: List[np.ndarray],
        max_workers: int = 8
    ) -> np.ndarray:
        """
        Normalize a batch of ECG waveforms.
        
        Args:
            waveforms: List of waveform arrays
            max_workers: Number of parallel workers
            
        Returns:
            Batch of normalized waveforms of shape (batch, num_leads, target_length)
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.normalize_waveform, wf)
                for wf in waveforms
            ]
            normalized = [f.result() for f in futures]
        
        return np.stack(normalized, axis=0)
    
    def normalize_from_path(
        self, 
        waveform_path: str
    ) -> np.ndarray:
        """
        Load and normalize a waveform from a file path.
        
        Args:
            waveform_path: Path to the .npy file containing the waveform
            
        Returns:
            Normalized waveform of shape (num_leads, target_length)
        """
        waveform = np.load(waveform_path)
        return self.normalize_waveform(waveform)


def create_psa_normalizer(
    sampling_rate: int = 250,
    target_length: int = 2500,
    num_leads: int = 12,
    region: str = "NA"
) -> PSANormalizer:
    """
    Factory function to create a PSA normalizer with region-specific settings.
    
    Args:
        sampling_rate: ECG sampling rate in Hz
        target_length: Target waveform length
        num_leads: Number of ECG leads
        region: "NA" for North America (60 Hz), "EU" for Europe (50 Hz)
        
    Returns:
        Configured PSANormalizer instance
    """
    powerline_freq = 60.0 if region.upper() == "NA" else 50.0
    
    return PSANormalizer(
        sampling_rate=sampling_rate,
        target_length=target_length,
        num_leads=num_leads,
        powerline_freq=powerline_freq
    )
