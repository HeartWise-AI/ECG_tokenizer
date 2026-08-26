"""Contracts for patient-disjoint calibration cohorts."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.patient_identity import (
    normalize_patient_ids,
    patient_disjoint_mask,
    patient_group_split_masks,
)


def test_patient_ids_normalize_equivalent_integer_representations():
    values = pd.Series(["338306", "337511.0", 42])

    assert normalize_patient_ids(values, source="cohort").tolist() == [
        338306,
        337511,
        42,
    ]


@pytest.mark.parametrize("invalid", [None, np.nan, "", "not-recorded", "12.5"])
def test_patient_ids_reject_unknown_or_nonintegral_values(invalid):
    with pytest.raises(ValueError, match="patient identifiers"):
        normalize_patient_ids(pd.Series([123, invalid]), source="cohort")


def test_patient_disjoint_mask_excludes_normalized_overlap():
    train = pd.Series(["337511.0", "999999", "123456"])
    test = pd.Series([337511, "888888.0"])

    keep = patient_disjoint_mask(train, test)

    assert keep.tolist() == [False, True, True]


@pytest.mark.parametrize(
    ("train", "test", "source"),
    [
        (pd.Series(["not-recorded"]), pd.Series([1]), "train cohort"),
        (pd.Series([1]), pd.Series([None]), "test cohort"),
    ],
)
def test_patient_disjoint_mask_fails_on_unknown_ids(train, test, source):
    with pytest.raises(ValueError, match=source):
        patient_disjoint_mask(train, test)


def test_patient_group_split_keeps_repeated_patient_rows_together():
    patient_ids = pd.Series([0, 0, 1, 1, 2, 2, 3, 3])

    calibration, test = patient_group_split_masks(
        patient_ids,
        first_fraction=0.5,
        seed=20260701,
        source="endpoint cohort",
    )

    calibration_patients = set(patient_ids.loc[calibration].tolist())
    test_patients = set(patient_ids.loc[test].tolist())
    assert calibration_patients
    assert test_patients
    assert calibration_patients.isdisjoint(test_patients)
    for patient_id in set(patient_ids):
        positions = patient_ids.eq(patient_id).to_numpy()
        assert calibration[positions].all() or test[positions].all()


def test_patient_group_split_rejects_single_patient_cohort():
    with pytest.raises(ValueError, match="at least two unique patients"):
        patient_group_split_masks(
            pd.Series([1, "1.0"]),
            first_fraction=0.5,
            seed=1,
            source="endpoint cohort",
        )


def test_axis_probe_groups_by_normalized_patient_identity():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "probe_axis_views.py").read_text(encoding="utf-8")

    assert "normalize_patient_ids" in source
    assert "set(g[tr]).intersection(g[te])" in source


def test_axis_probe_loads_the_configured_group_column():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse(
        (root / "scripts" / "probe_axis_views.py").read_text(encoding="utf-8")
    )
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "qa_columns"
            for target in node.targets
        )
    ]

    assert assignments
    assert "a.group_col" in ast.unparse(assignments[0].value)
