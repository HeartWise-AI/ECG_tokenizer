"""Stable prompt contract used to fit endpoint readout calibrations."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from utils.artifact_provenance import canonical_sha256, python_implementation_identity


ENDPOINT_PROMPT_CONTRACT = {
    "lvef_lte_40": {
        "label_column": "deepecho_Visually_Estimated_EF",
        "questions": (
            "Is this patient's LVEF less than or equal to 40%? Answer Yes or No.",
            "Does this ECG indicate a left ventricular ejection fraction of 40% or lower? Answer Yes or No.",
            "Is the left ventricular ejection fraction reduced to 40% or below? Answer Yes or No.",
            "Based on this ECG, is LVEF 40% or less? Answer Yes or No.",
        ),
    },
    "lvef_lt_50": {
        "label_column": "deepecho_Visually_Estimated_EF",
        "questions": (
            "Is this patient's LVEF less than 50%? Answer Yes or No.",
            "Does this ECG indicate a left ventricular ejection fraction below 50%? Answer Yes or No.",
            "Is the left ventricular ejection fraction under 50%? Answer Yes or No.",
            "Based on this ECG, is LVEF below 50%? Answer Yes or No.",
        ),
    },
    "incident_afib_5y": {
        "label_column": "afib_label_5y",
        "questions": (
            "Is this patient at risk for incident atrial fibrillation within 5 years? Answer Yes or No.",
            "Will this patient likely develop atrial fibrillation within the next 5 years? Answer Yes or No.",
            "Does this ECG suggest elevated risk of new-onset atrial fibrillation over 5 years? Answer Yes or No.",
            "Is future atrial fibrillation within 5 years likely for this patient? Answer Yes or No.",
        ),
    },
    "acute_coronary_occlusion": {
        "label_column": "acs_condition_is_acute",
        "questions": (
            "Does this patient have an acute coronary occlusion? Answer Yes or No.",
            "Is there evidence of an acute coronary artery occlusion on this ECG? Answer Yes or No.",
            "Does this ECG indicate acute coronary occlusion? Answer Yes or No.",
            "Is an acute coronary occlusion present? Answer Yes or No.",
        ),
    },
    "shd": {
        "label_column": "echonext_shd_binary",
        "questions": (
            "Does this patient have structural heart disease? Answer Yes or No.",
            "Is there structural heart disease indicated by this ECG? Answer Yes or No.",
            "Does this ECG suggest the presence of structural heart disease? Answer Yes or No.",
            "Is structural heart disease present in this patient? Answer Yes or No.",
        ),
    },
}


def endpoint_implementation_identity() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    return python_implementation_identity(
        (
            root / "scripts" / "evaluate_endpoint_pyes_calibrated.py",
            root / "scripts" / "binary_auroc_eval.py",
            root / "models",
            root / "utils",
        ),
        runtime_packages=(
            "numpy",
            "pandas",
            "torch",
            "transformers",
            "vector-quantize-pytorch",
        ),
    )


def endpoint_contract_sha256(names: Iterable[str]) -> str:
    contract = {
        name: {
            "label_column": ENDPOINT_PROMPT_CONTRACT[name]["label_column"],
            "questions": list(ENDPOINT_PROMPT_CONTRACT[name]["questions"]),
        }
        for name in names
    }
    return canonical_sha256(contract)
