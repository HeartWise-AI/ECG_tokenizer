"""Contracts for reusable scientific artifacts."""

import ast
import importlib
import json
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from utils.artifact_provenance import (
    atomic_savez_compressed,
    atomic_write_json,
    encode_manifest,
    exclusive_artifact_lock,
    file_identity,
    load_npz_if_current,
    ordered_files_identity,
    python_implementation_identity,
    publish_artifact_bundle,
    require_finite_numeric_array,
    require_matching_provenance,
    validate_artifact_bundle,
)


def test_cache_is_reused_only_for_exact_content_identity(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"first")
    cache = tmp_path / "nested" / "cache.npz"
    expected = {"schema_version": 1, "checkpoint": file_identity(checkpoint)}
    atomic_savez_compressed(
        cache,
        values=np.asarray([1, 2]),
        provenance=encode_manifest(expected),
    )

    loaded = load_npz_if_current(cache, expected)
    assert loaded is not None
    try:
        assert loaded["values"].tolist() == [1, 2]
    finally:
        loaded.close()

    checkpoint.write_bytes(b"second")
    changed = {"schema_version": 1, "checkpoint": file_identity(checkpoint)}
    assert load_npz_if_current(cache, changed) is None


def test_npz_metadata_loads_without_pickle(tmp_path):
    cache = tmp_path / "cache.npz"
    manifest = {"schema_version": 1}
    atomic_savez_compressed(
        cache,
        names=np.asarray(["one", "two"], dtype=np.str_),
        provenance=encode_manifest(manifest),
    )

    with np.load(cache, allow_pickle=False) as loaded:
        assert loaded["names"].tolist() == ["one", "two"]


def test_atomic_json_is_strict_and_creates_parent(tmp_path):
    output = tmp_path / "nested" / "result.json"
    atomic_write_json(output, {"missing_metric": float("nan")})

    assert json.loads(output.read_text(encoding="utf-8")) == {"missing_metric": None}
    assert "NaN" not in output.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "script",
    (
        "generate_dpo_candidates.py",
        "echonext_lvef_decoding_comparison.py",
    ),
)
def test_json_result_scripts_use_parent_creating_atomic_writer(script):
    source = (Path(__file__).resolve().parents[1] / "scripts" / script).read_text(
        encoding="utf-8"
    )
    calls = {
        node.func.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "atomic_write_json" in calls

    tree = ast.parse(source)
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    mkdir_line = min(
        node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "mkdir"
    )
    load_model_line = min(
        node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_model"
    )
    assert mkdir_line < load_model_line


def test_ordered_file_identity_detects_same_path_same_size_replacement(tmp_path):
    waveform = tmp_path / "waveform.npy"
    waveform.write_bytes(b"AAAA")
    baseline = ordered_files_identity([waveform])

    waveform.write_bytes(b"BBBB")

    assert ordered_files_identity([waveform]) != baseline


def test_cache_is_invalidated_when_imported_model_source_changes(tmp_path):
    implementation_root = tmp_path / "models"
    implementation_root.mkdir()
    model_source = implementation_root / "bridge.py"
    model_source.write_text("MODEL_VERSION = 1\n", encoding="utf-8")
    cache = tmp_path / "features.npz"
    baseline = {
        "implementation": python_implementation_identity(
            (implementation_root,), runtime_packages=("numpy",)
        )
    }
    atomic_savez_compressed(
        cache,
        values=np.asarray([1]),
        provenance=encode_manifest(baseline),
    )

    model_source.write_text("MODEL_VERSION = 2\n", encoding="utf-8")
    changed = {
        "implementation": python_implementation_identity(
            (implementation_root,), runtime_packages=("numpy",)
        )
    }

    assert load_npz_if_current(cache, changed) is None


def test_publication_aborts_when_an_input_changes_during_computation(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"before")
    captured = {"checkpoint": file_identity(checkpoint)}

    checkpoint.write_bytes(b"after")
    current = {"checkpoint": file_identity(checkpoint)}

    with pytest.raises(RuntimeError, match="inputs changed"):
        require_matching_provenance(captured, current, artifact="endpoint metrics")


@pytest.mark.parametrize(
    "values",
    [
        np.asarray([[1.0, np.nan]]),
        np.asarray([[1.0, np.inf]]),
        np.asarray([["1", "2"]]),
    ],
)
def test_scientific_arrays_reject_nonfinite_or_nonnumeric_values(values):
    with pytest.raises(ValueError):
        require_finite_numeric_array(
            values,
            artifact="scientific cache",
            expected_shape=(1, 2),
        )


def test_self_attested_nonfinite_cache_is_rejected_after_provenance_load(tmp_path):
    cache = tmp_path / "cache.npz"
    provenance = {"schema_version": 1, "input": "exact"}
    atomic_savez_compressed(
        cache,
        features=np.asarray([[np.nan]], dtype=np.float32),
        provenance=encode_manifest(provenance),
    )

    loaded = load_npz_if_current(cache, provenance)
    assert loaded is not None
    try:
        with pytest.raises(ValueError, match="non-finite"):
            require_finite_numeric_array(
                loaded["features"],
                artifact="feature cache",
                expected_shape=(1, 1),
            )
    finally:
        loaded.close()


def test_artifact_bundle_manifest_commits_exact_promoted_files(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    first = staging / "metrics.json"
    second = staging / "metrics.md"
    first.write_text('{"score": 1}\n', encoding="utf-8")
    second.write_text("# metrics\n", encoding="utf-8")
    manifest = tmp_path / "metrics.bundle.json"

    publish_artifact_bundle(
        {first.name: first, second.name: second},
        manifest,
        provenance={"run": "exact"},
    )

    committed = validate_artifact_bundle(
        manifest,
        required_files=(first.name, second.name),
    )
    assert committed["provenance"] == {"run": "exact"}


def test_failed_bundle_promotion_leaves_no_commit_manifest(tmp_path):
    first = tmp_path / "first.staged"
    second = tmp_path / "second.staged"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    manifest = tmp_path / "result.bundle.json"
    checks = 0

    def fail_second_promotion():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise RuntimeError("input changed")

    with pytest.raises(RuntimeError, match="input changed"):
        publish_artifact_bundle(
            {"first.txt": first, "second.txt": second},
            manifest,
            provenance={"run": "interrupted"},
            verify_current=fail_second_promotion,
        )

    assert not manifest.exists()
    with pytest.raises(ValueError, match="not committed"):
        validate_artifact_bundle(manifest)


def test_artifact_lock_blocks_a_second_writer_until_release(tmp_path):
    lock_path = tmp_path / "bundle.lock"
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def first_writer():
        with exclusive_artifact_lock(lock_path):
            first_entered.set()
            assert release_first.wait(timeout=2)

    def second_writer():
        assert first_entered.wait(timeout=2)
        with exclusive_artifact_lock(lock_path):
            second_entered.set()

    first_thread = threading.Thread(target=first_writer)
    second_thread = threading.Thread(target=second_writer)
    first_thread.start()
    second_thread.start()
    assert first_entered.wait(timeout=2)
    assert not second_entered.wait(timeout=0.05)
    release_first.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)
    assert second_entered.is_set()


def _load_endpoint_scripts(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    scripts_package = types.ModuleType("scripts")
    scripts_package.__path__ = [str(root / "scripts")]
    monkeypatch.setitem(sys.modules, "scripts", scripts_package)
    binary_eval = types.ModuleType("scripts.binary_auroc_eval")
    binary_eval.load_ecg_signal = lambda path: path
    binary_eval.load_model = lambda checkpoint, device: (checkpoint, device)
    monkeypatch.setitem(sys.modules, "scripts.binary_auroc_eval", binary_eval)
    sklearn = types.ModuleType("sklearn")
    sklearn.__path__ = []
    sklearn_metrics = types.ModuleType("sklearn.metrics")
    for name in (
        "average_precision_score",
        "brier_score_loss",
        "confusion_matrix",
        "roc_auc_score",
    ):
        setattr(sklearn_metrics, name, lambda *args, **kwargs: 0.0)
    monkeypatch.setitem(sys.modules, "sklearn", sklearn)
    monkeypatch.setitem(sys.modules, "sklearn.metrics", sklearn_metrics)
    for name in (
        "scripts.evaluate_endpoint_pyes_calibrated",
        "scripts.calibrate_endpoints_on_train",
    ):
        if name in sys.modules:
            monkeypatch.delitem(sys.modules, name)
        else:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
            del sys.modules[name]
    evaluator = importlib.import_module("scripts.evaluate_endpoint_pyes_calibrated")
    calibration = importlib.import_module("scripts.calibrate_endpoints_on_train")
    return evaluator, calibration


def test_train_calibration_main_creates_fresh_output_before_work(tmp_path, monkeypatch):
    _, calibration = _load_endpoint_scripts(monkeypatch)
    out_dir = tmp_path / "fresh" / "calibration"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    observed: dict[str, Path] = {}

    def capture_run(args, actual_out_dir):
        observed["out_dir"] = actual_out_dir
        assert actual_out_dir == out_dir
        assert actual_out_dir.is_dir()

    monkeypatch.setattr(calibration, "_run", capture_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate_endpoints_on_train.py",
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(out_dir),
        ],
    )

    calibration.main()

    assert observed == {"out_dir": out_dir}


def test_train_calibration_rejects_test_margins_from_another_checkpoint(
    tmp_path, monkeypatch
):
    evaluator, calibration = _load_endpoint_scripts(monkeypatch)
    requested_checkpoint = tmp_path / "requested.pt"
    requested_checkpoint.write_bytes(b"requested")
    other_checkpoint = tmp_path / "other.pt"
    other_checkpoint.write_bytes(b"other")
    train_parquet = tmp_path / "train.parquet"
    train_parquet.write_bytes(b"train")
    test_parquet = tmp_path / "test.parquet"
    test_parquet.write_bytes(b"test")
    waveform = tmp_path / "waveform.npy"
    waveform.write_bytes(b"waveform")
    test_margins = tmp_path / "raw_margins.npz"

    names = list(calibration.ENDPOINTS)
    n_prompts = [len(calibration.ENDPOINTS[name]["questions"]) for name in names]
    wrong_provenance = evaluator.build_margin_cache_provenance(
        str(other_checkpoint),
        str(test_parquet),
        names,
        n_prompts,
        0,
        [str(waveform)],
        2,
        "cpu",
    )
    atomic_savez_compressed(
        test_margins,
        provenance=encode_manifest(wrong_provenance),
    )

    frame_data: dict[str, list[object]] = {
        "waveform_path_psa": [str(waveform)],
        "new_PatientID": [1],
    }
    for endpoint in calibration.ENDPOINTS.values():
        frame_data[endpoint["label_column"]] = [0]
    test_frame = pd.DataFrame(frame_data)
    monkeypatch.setattr(
        calibration.pd, "read_parquet", lambda *args, **kwargs: test_frame
    )

    arguments = SimpleNamespace(
        checkpoint=str(requested_checkpoint),
        train_parquet=str(train_parquet),
        test_parquet=str(test_parquet),
        test_margins=str(test_margins),
        n_calib=1,
        batch_size=2,
        device="cpu",
    )

    with pytest.raises(SystemExit, match="does not match the requested checkpoint"):
        calibration._run(arguments, tmp_path / "output")


def test_train_calibration_publisher_stages_and_commits_every_bundle_member(
    tmp_path, monkeypatch
):
    _, calibration = _load_endpoint_scripts(monkeypatch)

    committed: dict[str, object] = {}

    def capture_bundle(staged_files, manifest_path, *, provenance, verify_current):
        committed["names"] = set(staged_files)
        committed["manifest"] = Path(manifest_path)
        committed["provenance"] = provenance
        committed["json"] = json.loads(
            Path(staged_files["endpoint_readout_train_calibrated.json"]).read_text(
                encoding="utf-8"
            )
        )
        committed["markdown"] = Path(
            staged_files["endpoint_readout_train_calibrated.md"]
        ).read_text(encoding="utf-8")
        with np.load(
            staged_files["train_calib_margins.npz"], allow_pickle=False
        ) as data:
            committed["margins"] = data["margins"].copy()

    monkeypatch.setattr(calibration, "publish_artifact_bundle", capture_bundle)
    arguments = SimpleNamespace(
        checkpoint="checkpoint.pt",
        train_parquet="train.parquet",
        test_parquet="test.parquet",
        test_margins="test-margins.npz",
        n_calib=2,
    )
    provenance = {"schema_version": 2, "kind": "endpoint_train_calibration"}

    calibration._publish_train_calibration_bundle(
        arguments,
        tmp_path,
        ["shd"],
        [1],
        np.asarray([[0.0], [1.0]], dtype=np.float32),
        np.asarray([[0.1], [0.9]], dtype=np.float32),
        {"endpoints": {"shd": {}}},
        ["# result"],
        provenance,
        {"kind": "endpoint_margin_cache"},
        ["one.npy", "two.npy"],
    )

    assert committed["names"] == {
        "endpoint_readout_train_calibrated.json",
        "endpoint_readout_train_calibrated.md",
        "train_calib_margins.npz",
    }
    assert committed["manifest"] == (
        tmp_path / "endpoint_readout_train_calibrated.bundle.json"
    )
    assert committed["provenance"] == provenance
    assert committed["json"] == {"endpoints": {"shd": {}}}
    assert committed["markdown"] == "# result\n"
    np.testing.assert_array_equal(
        committed["margins"],
        np.asarray([[0.1], [0.9]], dtype=np.float32),
    )


def test_endpoint_metrics_evaluator_reaches_bundle_publication(tmp_path, monkeypatch):
    evaluator, _ = _load_endpoint_scripts(monkeypatch)
    names = list(evaluator.ENDPOINTS)
    n_prompts = [len(evaluator.ENDPOINTS[name]["questions"]) for name in names]
    row_count = 10
    labels = np.tile(
        (np.arange(row_count) % 2).reshape(-1, 1),
        (1, len(names)),
    ).astype(np.float32)
    margins = np.tile(
        np.linspace(-1.0, 1.0, row_count, dtype=np.float32).reshape(-1, 1),
        (1, sum(n_prompts)),
    )
    frame = pd.DataFrame(
        {
            "new_PatientID": np.arange(row_count),
            "waveform_path_psa": [
                f"waveform-{index}.npy" for index in range(row_count)
            ],
        }
    )
    provenance = {"schema_version": 2, "kind": "endpoint_margin_cache"}
    evaluator.CHECKPOINT = str(tmp_path / "checkpoint.pt")
    evaluator.TEST_PARQUET = str(tmp_path / "test.parquet")
    evaluator.OUT_DIR = tmp_path
    monkeypatch.setattr(
        evaluator,
        "build_margin_cache_provenance",
        lambda *args, **kwargs: provenance,
    )
    monkeypatch.setattr(
        evaluator,
        "rank_metrics",
        lambda *args: {"auroc": 0.5, "auprc": 0.5},
    )
    monkeypatch.setattr(
        evaluator,
        "bootstrap_ci",
        lambda *args, **kwargs: {
            "auroc": [0.4, 0.6],
            "auprc": [0.4, 0.6],
        },
    )
    monkeypatch.setattr(evaluator, "fit_platt", lambda *args: (1.0, 0.0))
    monkeypatch.setattr(evaluator, "youden_threshold", lambda *args: 0.5)
    monkeypatch.setattr(
        evaluator,
        "op_metrics",
        lambda *args: {
            "threshold": 0.5,
            "sensitivity": 0.5,
            "specificity": 0.5,
            "ppv": 0.5,
            "npv": 0.5,
            "tp": 1,
            "fp": 1,
            "tn": 1,
            "fn": 1,
        },
    )
    monkeypatch.setattr(evaluator, "roc_auc_score", lambda *args: 0.5)
    monkeypatch.setattr(evaluator, "brier_score_loss", lambda *args: 0.25)
    monkeypatch.setattr(evaluator, "expected_calibration_error", lambda *args: 0.1)
    published: dict[str, object] = {}

    def capture_bundle(staged_files, manifest_path, *, provenance, verify_current):
        verify_current()
        published["manifest"] = Path(manifest_path)
        published["provenance"] = provenance
        published["document"] = json.loads(
            Path(
                next(
                    path
                    for name, path in staged_files.items()
                    if name.endswith(".json")
                )
            ).read_text(encoding="utf-8")
        )

    monkeypatch.setattr(evaluator, "publish_artifact_bundle", capture_bundle)
    arguments = SimpleNamespace(limit=0, batch_size=2, device="cpu")

    evaluator._evaluate_and_publish_metrics(
        arguments,
        frame,
        labels,
        margins,
        names,
        n_prompts,
        frame["waveform_path_psa"].tolist(),
        provenance,
    )

    assert published["manifest"].suffixes[-2:] == [".bundle", ".json"]
    assert published["provenance"]["kind"] == "endpoint_calibrated_metrics"
    assert set(published["document"]["endpoints"]) == set(names)


def test_endpoint_and_axis_caches_include_full_implementation_identity():
    root = Path(__file__).resolve().parents[1]
    for relative_path, helper_name in (
        (
            "scripts/evaluate_endpoint_pyes_calibrated.py",
            "endpoint_implementation_identity",
        ),
        ("scripts/probe_axis_views.py", "python_implementation_identity"),
    ):
        source = (root / relative_path).read_text(encoding="utf-8")
        assert '"implementation"' in source
        assert helper_name in source


def test_endpoint_publications_recheck_captured_inputs_before_writing():
    root = Path(__file__).resolve().parents[1]
    for relative_path in (
        "scripts/evaluate_endpoint_pyes_calibrated.py",
        "scripts/calibrate_endpoints_on_train.py",
    ):
        source = (root / relative_path).read_text(encoding="utf-8")
        assert "captured_static_inputs" in source
        assert "require_matching_provenance(" in source

    evaluator = (root / "scripts" / "evaluate_endpoint_pyes_calibrated.py").read_text(
        encoding="utf-8"
    )
    assert "patient_group_split_masks(" in evaluator
    assert '"split_strategy": "patient_grouped"' in evaluator

    train_calibration = (
        root / "scripts" / "calibrate_endpoints_on_train.py"
    ).read_text(encoding="utf-8")
    assert "all_test_patient_ids" in train_calibration
    assert "all_test_paths" in train_calibration


def test_axis_probe_rechecks_full_provenance_before_cache_and_metrics_publication():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "probe_axis_views.py").read_text(encoding="utf-8")

    cache_check = source.index('artifact="axis probe feature cache"')
    cache_write = source.index("atomic_savez_compressed(", cache_check)
    metrics_check = source.index('artifact="axis probe metrics"')
    metrics_write = source.index("atomic_write_json(path, res)", metrics_check)

    assert cache_check < cache_write
    assert metrics_check < metrics_write
    assert "encode_binary_labels(" in source
    assert "encode_endpoint_labels(" in source
    assert "has_probe_class_support(" in source
    assert '"group_col"' in source
    assert '"ordered_groups_sha256"' in source
    assert '"row_indices_sha256"' in source
