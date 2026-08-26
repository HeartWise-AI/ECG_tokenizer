"""Contracts for fail-closed ECG loading."""

import numpy as np
import pytest

from utils.ecg_waveform import load_ecg_waveform


@pytest.mark.parametrize("lead_first", [False, True])
def test_load_ecg_waveform_normalizes_orientation(tmp_path, lead_first):
    waveform = np.arange(24, dtype=np.float32).reshape(2, 12)
    stored = waveform.T if lead_first else waveform
    path = tmp_path / "waveform.npy"
    np.save(path, stored)

    loaded = load_ecg_waveform(path, target_length=4)

    assert loaded.shape == (12, 4)
    assert loaded.flags.c_contiguous
    np.testing.assert_array_equal(loaded[:, :2], waveform.T)
    np.testing.assert_array_equal(loaded[:, 2:], 0.0)


def test_load_ecg_waveform_does_not_replace_unreadable_input_with_zeros(tmp_path):
    missing = tmp_path / "missing.npy"

    with pytest.raises(RuntimeError, match="missing.npy"):
        load_ecg_waveform(missing)


@pytest.mark.parametrize(
    "waveform",
    [
        np.zeros((3, 4), dtype=np.float32),
        np.full((10, 12), np.nan, dtype=np.float32),
    ],
)
def test_load_ecg_waveform_rejects_invalid_signal(tmp_path, waveform):
    path = tmp_path / "invalid.npy"
    np.save(path, waveform)

    with pytest.raises(ValueError):
        load_ecg_waveform(path)
