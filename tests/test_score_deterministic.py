"""Contracts for deterministic endpoint answer parsing."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_score_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "score_deterministic.py"
    spec = importlib.util.spec_from_file_location("score_deterministic", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("task", "text"),
    [
        ("afib_risk", "No high risk features are present."),
        ("afib_risk", "There are no high risk features present."),
        ("structural_heart_disease", "Structural heart disease is not present."),
        ("structural_heart_disease", "There is no structural heart disease present."),
        ("acs_severity", "No acute coronary occlusion is present."),
        ("acs_severity", "There is no acute coronary occlusion present."),
        ("acs_severity", "There is no evidence to suggest acute coronary occlusion."),
        (
            "structural_heart_disease",
            "There are no findings to support structural heart disease.",
        ),
    ],
)
def test_parse_bin_negation_takes_precedence(task, text):
    module = _load_score_module()
    assert module.parse_bin(text, task) == 0


@pytest.mark.parametrize(
    ("task", "text"),
    [
        ("afib_risk", "Yes, high risk."),
        (
            "afib_risk",
            "Elevated risk of new-onset atrial fibrillation within five years.",
        ),
        ("structural_heart_disease", "Structural heart disease is present."),
        ("acs_severity", "Yes, acute coronary occlusion is present."),
        ("acs_severity", "Acute coronary artery occlusion is present."),
    ],
)
def test_parse_bin_accepts_explicit_positive_answers(task, text):
    module = _load_score_module()
    assert module.parse_bin(text, task) == 1


@pytest.mark.parametrize(
    ("task", "text"),
    [
        (
            "structural_heart_disease",
            "There is no question that structural heart disease is present.",
        ),
        (
            "acs_severity",
            "Without doubt, acute coronary occlusion is present.",
        ),
    ],
)
def test_parse_bin_does_not_negate_positive_idioms(task, text):
    module = _load_score_module()
    assert module.parse_bin(text, task) == 1


def test_parse_bin_recognizes_coronary_artery_occlusion_negation():
    module = _load_score_module()
    assert (
        module.parse_bin(
            "There is no evidence of acute coronary artery occlusion.",
            "acs_severity",
        )
        == 0
    )


@pytest.mark.parametrize("text", ["Normal sinus rhythm.", "Yesterday this was discussed."])
def test_parse_bin_does_not_match_word_prefix_collisions(text):
    module = _load_score_module()
    assert np.isnan(module.parse_bin(text, "afib_risk"))


@pytest.mark.parametrize("value", ["101", "999999"])
def test_parse_ef_rejects_impossible_percentages(value):
    module = _load_score_module()
    assert np.isnan(module.parse_ef(f"The ejection fraction is {value}%."))


@pytest.mark.parametrize(("value", "expected"), [("0", 0.0), ("100", 100.0)])
def test_parse_ef_accepts_percentage_boundaries(value, expected):
    module = _load_score_module()
    assert module.parse_ef(f"The ejection fraction is {value}%.") == expected


def _run_score(csv_path, *extra_args):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "score_deterministic.py"
    return subprocess.run(
        [sys.executable, str(script_path), str(csv_path), *extra_args],
        text=True,
        capture_output=True,
    )


def test_score_rejects_input_without_a_supported_metric_category(tmp_path):
    csv_path = tmp_path / "unknown.csv"
    pd.DataFrame(
        {
            "generation": ["prediction"],
            "ground_truth": ["answer"],
            "prompt_category": ["rhythm"],
        }
    ).to_csv(csv_path, index=False)

    result = _run_score(csv_path)

    assert result.returncode != 0
    assert "no supported deterministic score categories" in result.stderr


def test_score_rejects_binary_metric_without_both_ground_truth_classes(tmp_path):
    csv_path = tmp_path / "one-class.csv"
    pd.DataFrame(
        {
            "generation": ["Yes, high risk."] * 20,
            "ground_truth": ["Yes, high risk."] * 20,
            "prompt_category": ["afib_risk"] * 20,
        }
    ).to_csv(csv_path, index=False)

    result = _run_score(csv_path, "--expected-category", "afib_risk")

    assert result.returncode != 0
    assert "requires both ground-truth classes" in result.stderr
    assert "bal-acc" not in result.stdout


def test_score_requires_every_configured_metric_category(tmp_path):
    csv_path = tmp_path / "afib.csv"
    pd.DataFrame(
        {
            "generation": ["Yes, high risk."] * 10 + ["No high risk."] * 10,
            "ground_truth": ["Yes, high risk."] * 10 + ["No high risk."] * 10,
            "prompt_category": ["afib_risk"] * 20,
        }
    ).to_csv(csv_path, index=False)

    result = _run_score(csv_path, "--expected-category", "lvef")

    assert result.returncode != 0
    assert "expected score categories are missing" in result.stderr


def test_score_emits_only_finite_binary_metric_with_class_support(tmp_path):
    csv_path = tmp_path / "balanced.csv"
    pd.DataFrame(
        {
            "generation": ["Yes, high risk."] * 10 + ["No high risk."] * 10,
            "ground_truth": ["Yes, high risk."] * 10 + ["No high risk."] * 10,
            "prompt_category": ["afib_risk"] * 20,
        }
    ).to_csv(csv_path, index=False)

    result = _run_score(csv_path, "--expected-category", "afib_risk")

    assert result.returncode == 0, result.stderr
    assert "bal-acc 1.00 (95% CI 1.00-1.00)" in result.stdout
    assert "nan" not in result.stdout.lower()
