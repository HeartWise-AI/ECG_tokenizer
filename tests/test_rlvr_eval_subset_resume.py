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
            "waveform_name": "current",
            "waveform_path": "/data/current.npy",
            "question": "stale question",
            "ground_truth": "current answer",
            "prompt_category": "rhythm",
            "checkpoint_path": str(checkpoint),
            "generation": "A valid answer",
            "run_fingerprint": "current-run",
        }
    )

    with pytest.raises(ValueError, match="question"):
        module.validate_partial_resume_row(
            partial_row,
            sub_row,
            str(checkpoint),
            "current-run",
        )


def test_partial_resume_rejects_mismatched_waveform_name(tmp_path):
    module = _load_eval_subset_module()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    sub_row = _subset(tmp_path).iloc[0]
    partial_row = pd.Series(
        {
            **module.expected_partial_identity(
                sub_row, str(checkpoint), "current-run"
            ),
            "waveform_name": "wrong-name",
            "generation": "A valid answer",
        }
    )

    with pytest.raises(ValueError, match="waveform_name"):
        module.validate_partial_resume_row(
            partial_row, sub_row, str(checkpoint), "current-run"
        )


def _subset(tmp_path) -> pd.DataFrame:
    waveform = tmp_path / "current.npy"
    if not waveform.exists():
        waveform.write_bytes(b"waveform-a")
    return pd.DataFrame(
        [
            {
                "source_row_idx": 12,
                "waveform_path_psa": str(waveform),
                "prompt": "current question",
                "generated_answer": "current answer",
                "prompt_category": "rhythm",
            }
        ]
    )


def _fingerprint(module, checkpoint, **overrides):
    options = {
        "max_new_tokens": 128,
        "batch_size": 2,
        "group_by_prompt": True,
        "generation_microbatch_size": 1,
    }
    options.update(overrides)
    return module.build_run_fingerprint(
        str(checkpoint), _subset(checkpoint.parent), **options
    )


def test_run_fingerprint_changes_with_output_defining_configuration(tmp_path):
    module = _load_eval_subset_module()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint-a")

    baseline = _fingerprint(module, checkpoint)

    assert _fingerprint(module, checkpoint, max_new_tokens=129) != baseline
    changed_subset = _subset(tmp_path)
    changed_subset.loc[0, "prompt"] = "different question"
    assert module.build_run_fingerprint(
        str(checkpoint),
        changed_subset,
        max_new_tokens=128,
        batch_size=2,
        group_by_prompt=True,
        generation_microbatch_size=1,
    ) != baseline
    checkpoint.write_bytes(b"checkpoint-b")
    assert _fingerprint(module, checkpoint) != baseline

    checkpoint.write_bytes(b"checkpoint-a")
    waveform = tmp_path / "current.npy"
    waveform.write_bytes(b"waveform-b")
    assert _fingerprint(module, checkpoint) != baseline


def test_run_fingerprint_changes_when_imported_model_code_changes(
    tmp_path, monkeypatch
):
    module = _load_eval_subset_module()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    implementation_root = tmp_path / "implementation"
    implementation_root.mkdir()
    model_source = implementation_root / "model.py"
    model_source.write_text("MODEL_VERSION = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "IMPLEMENTATION_SOURCE_ROOTS",
        (implementation_root,),
    )

    baseline = _fingerprint(module, checkpoint)
    model_source.write_text("MODEL_VERSION = 2\n", encoding="utf-8")

    assert _fingerprint(module, checkpoint) != baseline


def test_partial_resume_rejects_empty_or_error_generation(tmp_path):
    module = _load_eval_subset_module()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    sub_row = _subset(tmp_path).iloc[0]
    base = {
        **module.expected_partial_identity(sub_row, str(checkpoint), "current-run"),
        "generation": "",
    }

    with pytest.raises(ValueError, match="empty or records an error"):
        module.validate_partial_resume_row(
            pd.Series(base), sub_row, str(checkpoint), "current-run"
        )

    base["generation"] = "[ERROR: failed waveform]"
    with pytest.raises(ValueError, match="empty or records an error"):
        module.validate_partial_resume_row(
            pd.Series(base), sub_row, str(checkpoint), "current-run"
        )


def test_batch_failure_isolated_then_propagated_without_error_rows():
    module = _load_eval_subset_module()
    calls = []
    batch = [(0, object()), (1, object())]

    def generator(current_batch):
        calls.append([idx for idx, _ in current_batch])
        if any(idx == 1 for idx, _ in current_batch):
            raise OSError("waveform unreadable")
        return ["healthy output"]

    with pytest.raises(RuntimeError, match="row_idx=1"):
        module.generate_batch_with_isolation(batch, generator)

    assert calls == [[0, 1], [0], [1]]


def test_atomic_csv_write_creates_parent_and_publishes_complete_rows(tmp_path):
    module = _load_eval_subset_module()
    output = tmp_path / "nested" / "generations.csv"
    expected = pd.DataFrame([{"row_idx": 0, "generation": "healthy output"}])

    module.atomic_write_csv(expected, output)

    pd.testing.assert_frame_equal(pd.read_csv(output), expected)
    assert list(output.parent.glob("*.tmp")) == []


def _configure_main_run(module, monkeypatch, tmp_path, *, flush_every: int):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint-a")
    subset = _subset(tmp_path)
    output_dir = tmp_path / "output"
    monkeypatch.setattr(module.pd, "read_parquet", lambda _: subset.copy())
    monkeypatch.setattr(module, "load_model", lambda *_: (object(), object()))
    monkeypatch.setattr(module, "load_ecg_signal", lambda _: object())
    monkeypatch.setattr(
        module,
        "python_implementation_identity",
        lambda *_, **__: {"sha256": "stable-implementation"},
    )
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "rlvr_eval_subset.py",
            "--checkpoint",
            str(checkpoint),
            "--subset_parquet",
            str(tmp_path / "subset.parquet"),
            "--output_dir",
            str(output_dir),
            "--device",
            "cpu",
            "--label",
            "race",
            "--flush_every",
            str(flush_every),
        ],
    )
    return checkpoint, Path(subset.iloc[0]["waveform_path_psa"]), output_dir


def test_main_aborts_partial_publication_when_input_mutates_during_generation(
    tmp_path, monkeypatch
):
    module = _load_eval_subset_module()
    _, waveform, output_dir = _configure_main_run(
        module, monkeypatch, tmp_path, flush_every=1
    )

    def mutate_during_generation(*_args, **_kwargs):
        waveform.write_bytes(b"waveform-mutated-during-generation")
        return ["healthy output"]

    monkeypatch.setattr(module, "generate_batched", mutate_during_generation)

    with pytest.raises(SystemExit, match="inputs changed before partial publication"):
        module.main()

    assert not (output_dir / "generations_race.partial.csv").exists()
    assert not (output_dir / "generations_race.csv").exists()


def test_main_aborts_final_publication_when_input_mutates_during_partial_write(
    tmp_path, monkeypatch
):
    module = _load_eval_subset_module()
    _, waveform, output_dir = _configure_main_run(
        module, monkeypatch, tmp_path, flush_every=0
    )
    monkeypatch.setattr(
        module,
        "generate_batched",
        lambda *_args, **_kwargs: ["healthy output"],
    )
    atomic_write_csv = module.atomic_write_csv

    def mutate_after_partial_write(frame, path):
        atomic_write_csv(frame, path)
        if str(path).endswith(".partial.csv"):
            waveform.write_bytes(b"waveform-mutated-during-partial-write")

    monkeypatch.setattr(module, "atomic_write_csv", mutate_after_partial_write)

    with pytest.raises(SystemExit, match="inputs changed before partial publication"):
        module.main()

    assert not (output_dir / "generations_race.partial.csv").exists()
    assert not (output_dir / "generations_race.csv").exists()


def test_main_invalidates_final_csv_when_input_mutates_during_final_publication(
    tmp_path, monkeypatch
):
    module = _load_eval_subset_module()
    _, waveform, output_dir = _configure_main_run(
        module, monkeypatch, tmp_path, flush_every=0
    )
    monkeypatch.setattr(
        module,
        "generate_batched",
        lambda *_args, **_kwargs: ["healthy output"],
    )
    atomic_write_csv = module.atomic_write_csv

    def mutate_after_final_write(frame, path):
        atomic_write_csv(frame, path)
        if str(path).endswith("generations_race.csv"):
            waveform.write_bytes(b"waveform-mutated-during-final-write")

    monkeypatch.setattr(module, "atomic_write_csv", mutate_after_final_write)

    with pytest.raises(SystemExit, match="inputs changed before final publication"):
        module.main()

    assert (output_dir / "generations_race.partial.csv").exists()
    assert not (output_dir / "generations_race.csv").exists()
