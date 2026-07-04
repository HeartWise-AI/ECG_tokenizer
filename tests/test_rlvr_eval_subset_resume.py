import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _load_eval_subset_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "rlvr_eval_subset.py"
    spec = importlib.util.spec_from_file_location("rlvr_eval_subset", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_partial_resume_rejects_mismatched_row_identity(tmp_path):
    module = _load_eval_subset_module()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("", encoding="utf-8")
    sub_row = pd.Series(
        {
            "source_row_idx": 12,
            "waveform_path_psa": "/data/current.npy",
            "prompt": "current question",
            "generated_answer": "current answer",
            "prompt_category": "rhythm",
        }
    )
    partial_row = pd.Series(
        {
            "source_row_idx": 12,
            "waveform_path": "/data/current.npy",
            "question": "stale question",
            "ground_truth": "current answer",
            "prompt_category": "rhythm",
            "checkpoint_path": str(checkpoint),
        }
    )

    with pytest.raises(ValueError, match="question"):
        module.validate_partial_resume_row(partial_row, sub_row, str(checkpoint))
