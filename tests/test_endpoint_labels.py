"""Contracts for endpoint ground-truth validation and class support."""

import numpy as np
import pandas as pd
import pytest

from utils.endpoint_labels import (
    encode_binary_labels,
    encode_endpoint_labels,
    require_binary_class_support,
    require_nonempty_label_battery,
)


def test_lvef_labels_are_validated_before_thresholding():
    values = pd.Series([0.0, 40.0, 40.1, 100.0, np.nan])

    assert encode_endpoint_labels(values, "lvef_lte_40").tolist()[:4] == [
        1.0,
        1.0,
        0.0,
        0.0,
    ]
    assert np.isnan(encode_endpoint_labels(values, "lvef_lte_40")[-1])


@pytest.mark.parametrize("invalid", [-0.1, 100.1, float("inf"), "not measured"])
def test_lvef_rejects_impossible_or_non_numeric_ground_truth(invalid):
    with pytest.raises(ValueError, match="lvef_lte_40.*1 invalid.*2 rows"):
        encode_endpoint_labels(pd.Series([55.0, invalid]), "lvef_lte_40")


@pytest.mark.parametrize("invalid", [-1, 0.5, 2, float("inf"), "unknown"])
def test_binary_endpoints_reject_non_binary_ground_truth(invalid):
    with pytest.raises(ValueError, match="incident_afib_5y.*1 invalid.*3 rows"):
        encode_endpoint_labels(pd.Series([0, 1, invalid]), "incident_afib_5y")


@pytest.mark.parametrize("invalid", [-1, 2, float("inf"), "unknown"])
def test_diagnostic_labels_reject_invalid_values(invalid):
    with pytest.raises(ValueError, match="diagnostic finding.*1 invalid.*3 rows"):
        encode_binary_labels(
            pd.Series([0, 1, invalid]),
            label_name="diagnostic finding",
        )


def test_binary_class_support_requires_both_classes():
    with pytest.raises(ValueError, match="negatives=3, positives=0"):
        require_binary_class_support(
            np.asarray([0.0, 0.0, 0.0]),
            endpoint="shd",
            cohort="train",
        )


def test_binary_class_support_reports_exact_counts():
    assert require_binary_class_support(
        np.asarray([0.0, 1.0, np.nan, 1.0]),
        endpoint="shd",
        cohort="test",
    ) == {"negative": 1, "positive": 2}


def test_empty_probe_label_battery_fails_before_aggregate_publication():
    with pytest.raises(ValueError, match="localised.*no labels"):
        require_nonempty_label_battery([], battery="localised battery")
