"""Validation and encoding for endpoint ground-truth labels."""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np
import pandas as pd


_LVEF_THRESHOLDS = {
    "lvef_lte_40": (40.0, True),
    "lvef_lt_50": (50.0, False),
}
_BINARY_ENDPOINTS = {
    "incident_afib_5y",
    "acute_coronary_occlusion",
    "shd",
}


def encode_binary_labels(values: pd.Series, *, label_name: str) -> np.ndarray:
    """Validate an optional binary column and preserve missing values as NaN."""
    present = values.notna()
    labels = np.full(len(values), np.nan, dtype=np.float32)
    if not present.any():
        return labels

    numeric = pd.to_numeric(values.loc[present], errors="coerce").astype(float)
    valid = np.isfinite(numeric.to_numpy()) & numeric.isin((0.0, 1.0)).to_numpy()
    if not valid.all():
        invalid_indices = numeric.index[np.flatnonzero(~valid)].tolist()
        preview = ", ".join(str(index) for index in invalid_indices[:5])
        raise ValueError(
            f"label {label_name} has {len(invalid_indices)} invalid non-null labels "
            f"across {len(values)} rows; first invalid row indices: {preview}"
        )

    labels[present.to_numpy()] = numeric.to_numpy(dtype=np.float32)
    return labels


def encode_endpoint_labels(
    values: pd.Series,
    endpoint: str,
) -> np.ndarray:
    """Validate non-null source values and return binary labels with NaNs preserved."""
    if endpoint not in _LVEF_THRESHOLDS and endpoint not in _BINARY_ENDPOINTS:
        raise KeyError(f"unknown endpoint label contract: {endpoint}")
    if endpoint in _BINARY_ENDPOINTS:
        return encode_binary_labels(values, label_name=f"endpoint {endpoint}")

    present = values.notna()
    labels = np.full(len(values), np.nan, dtype=np.float32)
    if not present.any():
        return labels

    numeric = pd.to_numeric(values.loc[present], errors="coerce").astype(float)
    finite = np.isfinite(numeric.to_numpy())
    invalid = ~finite

    in_range = numeric.between(0.0, 100.0, inclusive="both").to_numpy()
    invalid |= ~in_range

    if invalid.any():
        invalid_indices = numeric.index[np.flatnonzero(invalid)].tolist()
        preview = ", ".join(str(index) for index in invalid_indices[:5])
        raise ValueError(
            f"endpoint {endpoint} has {len(invalid_indices)} invalid non-null labels "
            f"across {len(values)} rows; first invalid row indices: {preview}"
        )

    numeric_values = numeric.to_numpy(dtype=np.float64)
    threshold, inclusive = _LVEF_THRESHOLDS[endpoint]
    encoded = numeric_values <= threshold if inclusive else numeric_values < threshold
    labels[present.to_numpy()] = encoded.astype(np.float32)
    return labels


def require_binary_class_support(
    labels: Sequence[float] | np.ndarray,
    *,
    endpoint: str,
    cohort: str,
    min_per_class: int = 1,
) -> dict[str, int]:
    """Require the mathematical minimum class support for binary calibration."""
    if min_per_class < 1:
        raise ValueError("min_per_class must be at least 1")
    array = np.asarray(labels, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not np.isin(finite, (0.0, 1.0)).all():
        raise ValueError(f"{cohort} labels for endpoint {endpoint} are not binary")
    negatives = int(np.sum(finite == 0.0))
    positives = int(np.sum(finite == 1.0))
    if negatives < min_per_class or positives < min_per_class:
        raise ValueError(
            f"{cohort} labels for endpoint {endpoint} require at least "
            f"{min_per_class} negative and {min_per_class} positive; "
            f"found negatives={negatives}, positives={positives}"
        )
    return {"negative": negatives, "positive": positives}


def require_nonempty_label_battery(
    labels: Collection[object],
    *,
    battery: str,
) -> None:
    """Refuse to publish aggregate metrics for an unsupported label battery."""
    if not labels:
        raise ValueError(f"{battery} has no labels with train and test class support")
