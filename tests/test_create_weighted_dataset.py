import importlib.util
from pathlib import Path


def _load_create_weighted_dataset_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "create_weighted_dataset.py"
    spec = importlib.util.spec_from_file_location("create_weighted_dataset", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_negated_afib_high_risk_is_low():
    module = _load_create_weighted_dataset_module()

    assert module.detect_afib_risk_class("No high risk features for atrial fibrillation are present.") == "low"


def test_negated_structural_heart_disease_is_absent():
    module = _load_create_weighted_dataset_module()

    assert module.detect_shd_class("Structural heart disease is not present on this ECG.") == "absent"


def test_negated_acute_coronary_occlusion_is_not_acute():
    module = _load_create_weighted_dataset_module()

    assert module.detect_acs_severity_class("No acute coronary occlusion is present.") == "obstructive_no_acute"
