import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _load_schema_sft_checks_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "schema_sft_checks.py"
    spec = importlib.util.spec_from_file_location("schema_sft_checks", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_checks_reports_non_fast_failures():
    module = _load_schema_sft_checks_module()
    df = pd.DataFrame([{"target_json": "{not-json", "supervised_paths": []}])

    with pytest.raises(SystemExit, match="Schema checks failed for 1/1 rows"):
        module.run_checks(df, strict=False, fail_fast=False)
