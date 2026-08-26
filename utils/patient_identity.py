"""Fail-closed patient identifier normalization for cohort separation."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd


def normalize_patient_ids(values: pd.Series, *, source: str) -> pd.Series:
    """Return exact integer IDs and reject missing, nonnumeric, or fractional values."""
    normalized: list[int] = []
    invalid_count = 0
    for value in values.tolist():
        try:
            if isinstance(value, bool) or pd.isna(value):
                raise ValueError
            parsed = Decimal(str(value).strip())
            if not parsed.is_finite() or parsed != parsed.to_integral_value():
                raise ValueError
            normalized.append(int(parsed))
        except (InvalidOperation, TypeError, ValueError):
            invalid_count += 1
            normalized.append(0)
    if invalid_count:
        raise ValueError(
            f"{source} contains {invalid_count} missing, nonnumeric, or fractional "
            "patient identifiers"
        )
    return pd.Series(normalized, index=values.index, dtype="object")


def patient_disjoint_mask(
    train_ids: pd.Series,
    test_ids: pd.Series,
) -> pd.Series:
    """Return the train-row mask excluding every normalized test patient."""
    normalized_train = normalize_patient_ids(train_ids, source="train cohort")
    normalized_test = normalize_patient_ids(test_ids, source="test cohort")
    test_set = set(normalized_test.tolist())
    keep = ~normalized_train.isin(test_set)
    if set(normalized_train.loc[keep].tolist()).intersection(test_set):
        raise RuntimeError("patient-disjoint cohort assertion failed")
    return keep


def patient_group_split_masks(
    patient_ids: pd.Series,
    *,
    first_fraction: float,
    seed: int,
    source: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows by normalized patient identity with no patient overlap."""
    if not 0.0 < first_fraction < 1.0:
        raise ValueError("first_fraction must be strictly between 0 and 1")
    normalized = normalize_patient_ids(patient_ids, source=source)
    unique_patients = np.asarray(sorted(set(normalized.tolist())), dtype=object)
    if len(unique_patients) < 2:
        raise ValueError(f"{source} requires at least two unique patients")

    shuffled = np.random.default_rng(seed).permutation(unique_patients)
    first_count = min(max(int(len(shuffled) * first_fraction), 1), len(shuffled) - 1)
    first_patients = set(shuffled[:first_count].tolist())
    first = normalized.isin(first_patients).to_numpy()
    second = ~first
    first_ids = set(normalized.loc[first].tolist())
    second_ids = set(normalized.loc[second].tolist())
    if first_ids.intersection(second_ids):
        raise RuntimeError("patient-grouped split contains patient overlap")
    return first, second
