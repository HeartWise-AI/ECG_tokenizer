#!/usr/bin/env python3
"""
Convert ECG QA parquets into a fixed-schema JSON target/label format.

Outputs per row:
  - ecg_id
  - waveform_path_psa
  - prompt
  - prompt_category
  - sample_weight
  - gt_json_full
  - target_json
  - supervised_paths
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import ACS_ACUTE_CONDITIONS, DEEPECG_CATEGORIES


SCHEMA_VERSION = "ecg_v1"
OUTPUT_KEYS = [
    "shd",
    "rhythm",
    "rate_bpm",
    "lvef",
    "acs",
    "afib",  # AFib risk forecasting (2y/5y prediction)
    "conduction",
    "chamber_enlargement",
    "ischemia",
    "pericarditis",
    "other",
    "findings",
]
TASK_ORDER = [
    "shd",
    "rhythm",
    "rate_bpm",
    "lvef",
    "acs",
    "afib",
    "conduction",
    "chamber_enlargement",
    "ischemia",
    "pericarditis",
    "other",
    "findings",
]

RATE_TERMS = ("rate", "bpm", "heart rate", "hr", "pulse", "ventricular rate", "beats")

RHYTHM_PRIORITY = [
    "Afib",
    "Atrial flutter",
    "Ventricular tachycardia",
    "Ventricular Rhythm",
    "Supraventricular tachycardia",
    "Atrial tachycardia (>= 100 BPM)",
    "Ectopic atrial rhythm (< 100 BPM)",
    "Junctional rhythm",
    "Brugada",
    "Wolff-Parkinson-White (Pre-excitation syndrome)",
    "Bradycardia",
    "Premature ventricular complex",
    "Premature atrial complex",
    "Irregularly irregular",
    "Regularly irregular",
    "Sinusal",
    "Regular",
]


def _dedup_list(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))


def _normalize_categories() -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}
    for cat, labels in DEEPECG_CATEGORIES.items():
        normalized[cat] = _dedup_list(labels)
    return normalized


CATEGORY_LABELS = _normalize_categories()


def _is_positive(value: Any) -> bool:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return int(value) == 1
        if isinstance(value, (float, np.floating)):
            return float(value) > 0.5
        s = str(value).strip().lower()
        return s in {"1", "true", "yes", "y"}
    except Exception:
        return False


def _label_positive(row: pd.Series, label: str) -> bool:
    if label in row.index and _is_positive(row.get(label)):
        return True
    bert_col = f"{label}_bert_model"
    if bert_col in row.index and _is_positive(row.get(bert_col)):
        return True
    return False


def _extract_labels_by_category(row: pd.Series) -> Dict[str, List[str]]:
    labels_by_category: Dict[str, List[str]] = {}
    for category, labels in CATEGORY_LABELS.items():
        present = [label for label in labels if _label_positive(row, label)]
        labels_by_category[category] = present
    return labels_by_category


def _select_rhythm_value(rhythm_labels: List[str]) -> Optional[str]:
    if not rhythm_labels:
        return None
    present = set(rhythm_labels)
    for label in RHYTHM_PRIORITY:
        if label in present:
            return label
    return rhythm_labels[0]


def _extract_rate_bpm(row: pd.Series) -> Optional[int]:
    if "heart_rate" in row.index and pd.notna(row.get("heart_rate")):
        try:
            hr = float(row.get("heart_rate"))
            return int(round(hr)) if hr > 0 else None
        except (TypeError, ValueError):
            pass

    if "RestingECG_OriginalRestingECGMeasurements_VentricularRate" in row.index:
        try:
            hr = float(row.get("RestingECG_OriginalRestingECGMeasurements_VentricularRate"))
            return int(round(hr)) if hr > 0 else None
        except (TypeError, ValueError):
            pass

    if "rr_interval" in row.index and pd.notna(row.get("rr_interval")):
        try:
            rr_ms = float(row.get("rr_interval"))
            if rr_ms > 0:
                return int(round(60000.0 / rr_ms))
        except (TypeError, ValueError):
            pass

    return None


def _extract_shd_value(row: pd.Series) -> Optional[str]:
    if "echonext_shd" not in row.index:
        return None
    value = row.get("echonext_shd")
    if pd.isna(value):
        return None
    try:
        return "present" if float(value) >= 1 else "absent"
    except (TypeError, ValueError):
        return None


def _extract_lvef(row: pd.Series) -> Tuple[Optional[int], Optional[str]]:
    if "deepecho_Visually_Estimated_EF" not in row.index:
        return None, None
    value = row.get("deepecho_Visually_Estimated_EF")
    if pd.isna(value):
        return None, None
    try:
        return int(round(float(value))), "%"
    except (TypeError, ValueError):
        return None, None


def _extract_acs_value(row: pd.Series) -> Optional[bool]:
    if "acs_condition_severity" not in row.index:
        return None
    value = row.get("acs_condition_severity")
    if pd.isna(value):
        return None
    return True if value in ACS_ACUTE_CONDITIONS else False


def _extract_afib_risk_value(row: pd.Series) -> Optional[str]:
    """
    Extract AFib risk prediction value from dataset.
    Returns high_risk, moderate_risk, low_risk, or None.

    Logic:
    - If afib_label_2y is True -> "high_risk" (will develop AFib within 2 years)
    - If afib_label_5y is True but afib_label_2y is False -> "moderate_risk" (within 5 years)
    - If both are False -> "low_risk"
    - If data is missing -> None
    """
    # Check if columns exist
    if "afib_label_2y" not in row.index or "afib_label_5y" not in row.index:
        return None

    risk_2y = row.get("afib_label_2y")
    risk_5y = row.get("afib_label_5y")

    # Handle missing values
    if pd.isna(risk_2y) and pd.isna(risk_5y):
        return None

    # Determine risk level
    if _is_positive(risk_2y):
        return "high_risk"
    elif _is_positive(risk_5y):
        return "moderate_risk"
    else:
        return "low_risk"


def _extract_pericarditis_value(labels_by_category: Dict[str, List[str]]) -> bool:
    """Check if pericarditis is present in labels."""
    pericarditis_labels = labels_by_category.get("PERICARDITIS", [])
    return len(pericarditis_labels) > 0


def _build_gt_json_full(row: pd.Series) -> Dict[str, Any]:
    labels_by_category = _extract_labels_by_category(row)
    rhythm_value = _select_rhythm_value(labels_by_category.get("RHYTHM", []))
    rate_value = _extract_rate_bpm(row)
    shd_value = _extract_shd_value(row)
    lvef_value, lvef_unit = _extract_lvef(row)
    acs_value = _extract_acs_value(row)
    afib_value = _extract_afib_risk_value(row)
    pericarditis_value = _extract_pericarditis_value(labels_by_category)

    # All findings combined (for backward compatibility)
    findings_labels: List[str] = []
    for cat in ("CHAMBER ENLARGEMENT", "PERICARDITIS", "INFARCT, ISCHEMIA", "OTHER"):
        findings_labels.extend(labels_by_category.get(cat, []))
    findings_labels = _dedup_list(findings_labels)

    gt = {
        "schema_version": SCHEMA_VERSION,
        "outputs": {
            "shd": {"value": shd_value},
            "rhythm": {"value": rhythm_value},
            "rate_bpm": {"value": rate_value},
            "lvef": {"value": lvef_value, "unit": lvef_unit},
            "acs": {"value": acs_value},
            "afib": {"value": afib_value},
            "conduction": {"labels": labels_by_category.get("CONDUCTION", [])},
            "chamber_enlargement": {"labels": labels_by_category.get("CHAMBER ENLARGEMENT", [])},
            "ischemia": {"labels": labels_by_category.get("INFARCT, ISCHEMIA", [])},
            "pericarditis": {"value": pericarditis_value},
            "other": {"labels": labels_by_category.get("OTHER", [])},
            "findings": {"labels": findings_labels},
        },
    }
    return gt


def _findings_for_prompt(prompt_category: str, labels_by_category: Dict[str, List[str]]) -> List[str]:
    if prompt_category in {"interpretation", "json_interpretation"}:
        labels: List[str] = []
        for cat in ("CHAMBER ENLARGEMENT", "PERICARDITIS", "INFARCT, ISCHEMIA", "OTHER"):
            labels.extend(labels_by_category.get(cat, []))
        return _dedup_list(labels)
    if prompt_category in {"category_infarct_ischemia", "infarct, ischemia"}:
        return labels_by_category.get("INFARCT, ISCHEMIA", [])
    if prompt_category in {"category_chamber_enlargement", "chamber enlargement"}:
        return labels_by_category.get("CHAMBER ENLARGEMENT", [])
    if prompt_category in {"category_pericarditis", "pericarditis"}:
        return labels_by_category.get("PERICARDITIS", [])
    if prompt_category in {"category_other", "other"}:
        return labels_by_category.get("OTHER", [])
    return []


def _requested_tasks(prompt_category: str, prompt_text: str) -> List[str]:
    tasks: List[str] = []
    if prompt_category in {"interpretation", "json_interpretation", "interpretation_complex"}:
        # Full interpretation includes all ECG pattern categories
        tasks = ["rhythm", "rate_bpm", "conduction", "chamber_enlargement", "ischemia", "pericarditis", "other", "findings"]
    elif prompt_category in {"category_rhythm", "rhythm"}:
        tasks = ["rhythm"]
        prompt_lower = prompt_text.lower()
        if any(term in prompt_lower for term in RATE_TERMS):
            tasks.append("rate_bpm")
    elif prompt_category == "ecg_interval":
        tasks = ["rate_bpm"]
    elif prompt_category in {"category_conduction", "conduction"}:
        tasks = ["conduction"]
    elif prompt_category in {"category_infarct_ischemia", "infarct, ischemia", "localization_st_elevation",
                              "localization_st_depression", "localization_q_wave", "localization_t_wave"}:
        tasks = ["ischemia"]
    elif prompt_category in {"category_chamber_enlargement", "chamber enlargement"}:
        tasks = ["chamber_enlargement"]
    elif prompt_category in {"category_pericarditis", "pericarditis"}:
        tasks = ["pericarditis"]
    elif prompt_category in {"category_other", "other"}:
        tasks = ["other"]
    elif prompt_category == "structural_heart_disease":
        tasks = ["shd"]
    elif prompt_category == "lvef":
        tasks = ["lvef"]
    elif prompt_category in {"acs_severity", "culprit_artery", "urgency_assessment"}:
        tasks = ["acs"]
    elif prompt_category == "afib_risk":
        tasks = ["afib"]  # AFib risk forecasting task
    elif prompt_category == "classification":
        # Classification is a general assessment
        tasks = ["rhythm", "conduction", "findings"]
    elif prompt_category in {"random_finding_question"}:
        tasks = ["findings"]
    elif prompt_category == "localization_qrs_axis":
        tasks = ["conduction"]
    return tasks


def _task_key_order(tasks: List[str]) -> List[str]:
    order = {task: i for i, task in enumerate(TASK_ORDER)}
    return sorted(tasks, key=lambda t: order.get(t, 999))


def _build_target_json(
    gt_json: Dict[str, Any],
    tasks: List[str],
    prompt_category: str,
    labels_by_category: Dict[str, List[str]],
) -> Dict[str, Any]:
    # Initialize all outputs with default values
    # List outputs include labels_present field (only in target_json, not gt_json_full)
    outputs = {
        "shd": {"value": None, "status": "not_requested"},
        "rhythm": {"value": None, "status": "not_requested"},
        "rate_bpm": {"value": None, "status": "not_requested"},
        "lvef": {"value": None, "unit": None, "status": "not_requested"},
        "acs": {"value": None, "status": "not_requested"},
        "afib": {"value": None, "status": "not_requested"},
        "conduction": {"labels": [], "labels_present": False, "status": "not_requested"},
        "chamber_enlargement": {"labels": [], "labels_present": False, "status": "not_requested"},
        "ischemia": {"labels": [], "labels_present": False, "status": "not_requested"},
        "pericarditis": {"value": None, "status": "not_requested"},
        "other": {"labels": [], "labels_present": False, "status": "not_requested"},
        "findings": {"labels": [], "labels_present": False, "status": "not_requested"},
    }

    if "shd" in tasks:
        outputs["shd"]["value"] = gt_json["outputs"]["shd"]["value"]
        outputs["shd"]["status"] = "predicted"
    if "rhythm" in tasks:
        outputs["rhythm"]["value"] = gt_json["outputs"]["rhythm"]["value"]
        outputs["rhythm"]["status"] = "predicted"
    if "rate_bpm" in tasks:
        outputs["rate_bpm"]["value"] = gt_json["outputs"]["rate_bpm"]["value"]
        outputs["rate_bpm"]["status"] = "predicted"
    if "lvef" in tasks:
        outputs["lvef"]["value"] = gt_json["outputs"]["lvef"]["value"]
        outputs["lvef"]["unit"] = gt_json["outputs"]["lvef"]["unit"]
        outputs["lvef"]["status"] = "predicted"
    if "acs" in tasks:
        outputs["acs"]["value"] = gt_json["outputs"]["acs"]["value"]
        outputs["acs"]["status"] = "predicted"
    if "afib" in tasks:
        outputs["afib"]["value"] = gt_json["outputs"]["afib"]["value"]
        outputs["afib"]["status"] = "predicted"
    if "conduction" in tasks:
        labels = gt_json["outputs"]["conduction"]["labels"]
        outputs["conduction"]["labels"] = labels
        outputs["conduction"]["labels_present"] = len(labels) > 0
        outputs["conduction"]["status"] = "predicted"
    if "chamber_enlargement" in tasks:
        labels = gt_json["outputs"]["chamber_enlargement"]["labels"]
        outputs["chamber_enlargement"]["labels"] = labels
        outputs["chamber_enlargement"]["labels_present"] = len(labels) > 0
        outputs["chamber_enlargement"]["status"] = "predicted"
    if "ischemia" in tasks:
        labels = gt_json["outputs"]["ischemia"]["labels"]
        outputs["ischemia"]["labels"] = labels
        outputs["ischemia"]["labels_present"] = len(labels) > 0
        outputs["ischemia"]["status"] = "predicted"
    if "pericarditis" in tasks:
        outputs["pericarditis"]["value"] = gt_json["outputs"]["pericarditis"]["value"]
        outputs["pericarditis"]["status"] = "predicted"
    if "other" in tasks:
        labels = gt_json["outputs"]["other"]["labels"]
        outputs["other"]["labels"] = labels
        outputs["other"]["labels_present"] = len(labels) > 0
        outputs["other"]["status"] = "predicted"
    if "findings" in tasks:
        labels = _findings_for_prompt(prompt_category, labels_by_category)
        outputs["findings"]["labels"] = labels
        outputs["findings"]["labels_present"] = len(labels) > 0
        outputs["findings"]["status"] = "predicted"

    target: Dict[str, Any] = {"schema_version": SCHEMA_VERSION, "outputs": outputs}
    ordered_tasks = _task_key_order(tasks)
    if len(ordered_tasks) == 1:
        target["task"] = ordered_tasks[0]
    else:
        target["tasks_requested"] = ordered_tasks

    return target


def _supervised_paths(tasks: List[str], always_status: bool = True) -> List[str]:
    paths: List[str] = []
    if always_status:
        for key in OUTPUT_KEYS:
            paths.append(f"outputs.{key}.status")

    # Tasks with labels (multi-label outputs) - include labels_present
    label_tasks = {"conduction", "chamber_enlargement", "ischemia", "other", "findings"}
    # Tasks with value + unit
    value_unit_tasks = {"lvef"}

    for task in tasks:
        if task in label_tasks:
            paths.append(f"outputs.{task}.labels_present")
            paths.append(f"outputs.{task}.labels")
            paths.append(f"outputs.{task}.status")
        elif task in value_unit_tasks:
            paths.append(f"outputs.{task}.value")
            paths.append(f"outputs.{task}.unit")
            paths.append(f"outputs.{task}.status")
        else:
            # Simple value tasks: shd, rhythm, rate_bpm, acs, afib, pericarditis
            paths.append(f"outputs.{task}.value")
            paths.append(f"outputs.{task}.status")

    return _dedup_list(paths)


def _pick_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Missing required columns; none of {candidates} found.")


def _make_ecg_id(row: pd.Series, waveform_col: str) -> str:
    for col in ("ecg_id", "waveform_name"):
        if col in row.index and pd.notna(row.get(col)):
            return str(row.get(col))
    waveform_path = row.get(waveform_col)
    if pd.notna(waveform_path):
        base = os.path.basename(str(waveform_path))
        stem = os.path.splitext(base)[0]
        return stem or base
    payload = f"{row.name}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _normalize_prompt_category(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip().lower()


def convert_dataframe(
    df: pd.DataFrame,
    dataset: str,
    drop_unsupported: bool = True,
    always_supervise_status: bool = True,
    max_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if dataset != "combined" and "dataset" in df.columns:
        df = df[df["dataset"].astype(str).str.lower() == dataset].reset_index(drop=True)

    prompt_col = _pick_column(df, ["prompt", "question"])
    category_col = _pick_column(df, ["prompt_category", "category"])
    waveform_col = _pick_column(df, ["waveform_path_psa", "signal_path", "waveform_path_original"])
    weight_col = None
    for candidate in ("sample_weight", "prompt_weight"):
        if candidate in df.columns:
            weight_col = candidate
            break

    if max_rows:
        df = df.head(max_rows)

    rows: List[Dict[str, Any]] = []
    stats = {
        "total_rows": len(df),
        "kept_rows": 0,
        "dropped_unsupported": 0,
        "dropped_missing_gt": 0,
    }

    for _, row in df.iterrows():
        prompt = str(row.get(prompt_col, "")).strip()
        prompt_category = _normalize_prompt_category(row.get(category_col))
        tasks = _requested_tasks(prompt_category, prompt)

        if not tasks:
            if drop_unsupported:
                stats["dropped_unsupported"] += 1
                continue

        labels_by_category = _extract_labels_by_category(row)
        gt_json = _build_gt_json_full(row)

        # Drop rows missing required labels for requested tasks
        missing = False
        if "shd" in tasks and gt_json["outputs"]["shd"]["value"] is None:
            missing = True
        if "lvef" in tasks and gt_json["outputs"]["lvef"]["value"] is None:
            missing = True
        if "acs" in tasks and gt_json["outputs"]["acs"]["value"] is None:
            missing = True
        if "afib" in tasks and gt_json["outputs"]["afib"]["value"] is None:
            missing = True
        if "rhythm" in tasks and gt_json["outputs"]["rhythm"]["value"] is None:
            missing = True
        if "rate_bpm" in tasks and prompt_category == "ecg_interval" and gt_json["outputs"]["rate_bpm"]["value"] is None:
            missing = True
        if missing:
            stats["dropped_missing_gt"] += 1
            continue

        target_json = _build_target_json(gt_json, tasks, prompt_category, labels_by_category)
        supervised_paths = _supervised_paths(tasks, always_status=always_supervise_status)

        record = {
            "ecg_id": _make_ecg_id(row, waveform_col),
            "waveform_path_psa": row.get(waveform_col),
            "prompt": prompt,
            "prompt_category": row.get(category_col),
            "sample_weight": float(row.get(weight_col)) if weight_col else 1.0,
            "gt_json_full": json.dumps(gt_json, ensure_ascii=True, separators=(",", ":")),
            "target_json": json.dumps(target_json, ensure_ascii=True, separators=(",", ":")),
            "supervised_paths": supervised_paths,
        }
        rows.append(record)

    stats["kept_rows"] = len(rows)
    return pd.DataFrame(rows), stats


def _write_parquet(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)


def _pretty_print_sample(idx: int, row: pd.Series, dataset_name: str) -> None:
    """Pretty print a sample with its schema output."""
    print(f"\n{'='*70}")
    print(f"Sample {idx + 1} ({dataset_name})")
    print(f"{'='*70}")
    print(f"ECG ID: {row.get('ecg_id', 'N/A')}")
    print(f"Prompt Category: {row.get('prompt_category', 'N/A')}")
    print(f"Prompt: {str(row.get('prompt', 'N/A'))[:100]}...")

    # Parse and pretty print gt_json_full
    gt_json_str = row.get('gt_json_full')
    if gt_json_str:
        try:
            gt = json.loads(gt_json_str)
            print(f"\n--- gt_json_full ---")
            print(json.dumps(gt, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print(f"gt_json_full (raw): {gt_json_str[:300]}...")

    # Parse and pretty print target_json
    target_json_str = row.get('target_json')
    if target_json_str:
        try:
            target = json.loads(target_json_str)
            print(f"\n--- target_json ---")
            print(json.dumps(target, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print(f"target_json (raw): {target_json_str[:300]}...")

    # Show supervised paths
    paths = row.get('supervised_paths', [])
    print(f"\n--- supervised_paths ---")
    print(paths)


def run_test_mode(input_path: str) -> None:
    """
    Run in test mode: convert and display 10 MHI and 10 MIMIC samples.

    Selects samples with diverse label coverage to demonstrate all schema fields.
    """
    print(f"\n{'#'*70}")
    print("# RUNNING IN TEST MODE")
    print(f"{'#'*70}")
    print(f"\nLoading data from: {input_path}")

    df_full = pd.read_parquet(input_path)
    print(f"Loaded {len(df_full)} rows")

    # =========================================================================
    # TEST MHI SAMPLES (with diverse labels: LVEF, SHD, ACS, etc.)
    # =========================================================================
    print(f"\n{'='*70}")
    print("TESTING MHI SAMPLES (with SHD, LVEF, ACS, Afib, etc.)")
    print(f"{'='*70}")

    mhi_df = df_full[df_full["dataset"] == "mhi"].copy()
    print(f"MHI records: {len(mhi_df)}")

    # Collect diverse MHI samples
    mhi_samples = []

    # 1. Sample with LVEF data
    lvef_samples = mhi_df[mhi_df["deepecho_Visually_Estimated_EF"].notna()].head(2)
    mhi_samples.append(lvef_samples)

    # 2. Sample with SHD data
    shd_samples = mhi_df[mhi_df["echonext_shd"].notna()].head(2)
    mhi_samples.append(shd_samples)

    # 3. Sample with ACS data
    acs_samples = mhi_df[mhi_df["acs_condition_severity"].notna()].head(2)
    mhi_samples.append(acs_samples)

    # 4. Sample with Afib
    afib_samples = mhi_df[mhi_df["Afib"] == 1].head(2)
    mhi_samples.append(afib_samples)

    # 5. Sample with conduction issues
    lbbb_samples = mhi_df[mhi_df["Left bundle branch block"] == 1].head(1)
    mhi_samples.append(lbbb_samples)

    # 6. Sample with pericarditis
    peri_samples = mhi_df[mhi_df["Acute pericarditis"] == 1].head(1)
    mhi_samples.append(peri_samples)

    # Combine and deduplicate
    mhi_combined = pd.concat(mhi_samples).drop_duplicates().head(10)
    print(f"Selected {len(mhi_combined)} diverse MHI samples")

    # Convert MHI samples
    mhi_converted, mhi_stats = convert_dataframe(
        mhi_combined,
        dataset="mhi",
        drop_unsupported=False,  # Keep all to show variety
        always_supervise_status=True,
    )

    print(f"\nMHI Conversion Stats: {mhi_stats}")

    # Display MHI samples
    for idx, (_, row) in enumerate(mhi_converted.iterrows()):
        _pretty_print_sample(idx, row, "MHI")

    # =========================================================================
    # TEST MIMIC SAMPLES (ECG interpretation only)
    # =========================================================================
    print(f"\n\n{'='*70}")
    print("TESTING MIMIC SAMPLES (ECG interpretation only)")
    print(f"{'='*70}")

    mimic_df = df_full[df_full["dataset"] == "mimic"].copy()
    print(f"MIMIC records: {len(mimic_df)}")

    # Get diverse MIMIC samples with different prompt categories
    mimic_samples = []

    # Get samples from different prompt categories
    for cat in ["interpretation", "category_rhythm", "category_conduction",
                "category_infarct_ischemia", "json_interpretation"]:
        cat_samples = mimic_df[mimic_df["prompt_category"] == cat].head(2)
        if len(cat_samples) > 0:
            mimic_samples.append(cat_samples)

    mimic_combined = pd.concat(mimic_samples).drop_duplicates().head(10)
    print(f"Selected {len(mimic_combined)} diverse MIMIC samples")

    # Convert MIMIC samples
    mimic_converted, mimic_stats = convert_dataframe(
        mimic_combined,
        dataset="mimic",
        drop_unsupported=False,
        always_supervise_status=True,
    )

    print(f"\nMIMIC Conversion Stats: {mimic_stats}")

    # Display MIMIC samples
    for idx, (_, row) in enumerate(mimic_converted.iterrows()):
        _pretty_print_sample(idx, row, "MIMIC")

    print(f"\n\n{'#'*70}")
    print("# TEST MODE COMPLETE")
    print(f"{'#'*70}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ECG QA datasets to fixed-schema JSON format.")
    parser.add_argument("--dataset", choices=["combined", "mimic", "mhi"], default="combined")
    parser.add_argument("--input", type=str, default=None, help="Single input parquet path")
    parser.add_argument("--output", type=str, default=None, help="Single output parquet path")
    parser.add_argument("--train", type=str, default=None, help="Train parquet path")
    parser.add_argument("--val", type=str, default=None, help="Val parquet path")
    parser.add_argument("--test", type=str, default=None, help="Test parquet path")
    parser.add_argument("--out_dir", type=str, default="output/schema_v1")
    parser.add_argument("--drop_unsupported", action="store_true", default=True)
    parser.add_argument("--keep_unsupported", dest="drop_unsupported", action="store_false")
    parser.add_argument("--always_supervise_status", action="store_true", default=True)
    parser.add_argument("--no_supervise_status", dest="always_supervise_status", action="store_false")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional cap for debugging")
    parser.add_argument("--test_mode", action="store_true", help="Run test mode: 10 MHI + 10 MIMIC samples")
    parser.add_argument(
        "--test_input",
        type=str,
        default="/media/data1/datasets/ECG_Tokenizer/parquets/robert_dataset_qa/combined_train_qa_m200k_h200k.parquet",
        help="Input path for test mode",
    )
    args = parser.parse_args()

    # Test mode
    if args.test_mode:
        run_test_mode(args.test_input)
        return

    if args.input:
        if not args.output:
            raise ValueError("--output is required when using --input")
        df = pd.read_parquet(args.input)
        converted, stats = convert_dataframe(
            df,
            dataset=args.dataset,
            drop_unsupported=args.drop_unsupported,
            always_supervise_status=args.always_supervise_status,
            max_rows=args.max_rows,
        )
        _write_parquet(converted, args.output)
        print(f"Wrote {len(converted)} rows to {args.output}")
        print(f"Stats: {stats}")
        return

    split_map = {name: path for name, path in {"train": args.train, "val": args.val, "test": args.test}.items() if path}
    if not split_map:
        raise ValueError("Provide either --input/--output or at least one of --train/--val/--test or --test_mode")

    for split_name, split_path in split_map.items():
        df = pd.read_parquet(split_path)
        converted, stats = convert_dataframe(
            df,
            dataset=args.dataset,
            drop_unsupported=args.drop_unsupported,
            always_supervise_status=args.always_supervise_status,
            max_rows=args.max_rows,
        )
        out_path = os.path.join(args.out_dir, f"{args.dataset}_{split_name}_schema_v1.parquet")
        _write_parquet(converted, out_path)
        print(f"[{split_name}] wrote {len(converted)} rows to {out_path}")
        print(f"[{split_name}] stats: {stats}")


if __name__ == "__main__":
    main()
