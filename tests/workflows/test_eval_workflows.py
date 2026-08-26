from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"


def _wait_for_path(path: Path, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {path}")


def _load_workflow_utils():
    spec = importlib.util.spec_from_file_location(
        f"eval_workflow_utils_test_{time.monotonic_ns()}",
        SCRIPTS / "eval_workflow_utils.py",
    )
    if spec is None or spec.loader is None:
        raise AssertionError("could not load eval_workflow_utils")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _subset(path: Path, rows: int = 2, start: int = 0) -> pd.DataFrame:
    source_rows = list(range(start, start + rows))
    frame = pd.DataFrame(
        {
            "source_row_idx": source_rows,
            "waveform_path_psa": [f"/signals/{idx}.npy" for idx in source_rows],
            "prompt": [f"question {idx}" for idx in source_rows],
            "generated_answer": [f"answer {idx}" for idx in source_rows],
            "prompt_category": ["rhythm"] * rows,
        }
    )
    frame.to_parquet(path, index=False)
    return frame


def _fake_generation_runner(path: Path, generation_expression: str | None = None) -> None:
    expression = generation_expression or "[f'prediction {i}' for i in range(len(sub))]"
    source = """#!/usr/bin/env python3
import argparse
import hashlib
import os
import time
from pathlib import Path
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--subset_parquet', required=True)
p.add_argument('--output_dir', required=True)
p.add_argument('--device')
p.add_argument('--batch_size')
p.add_argument('--max_new_tokens')
p.add_argument('--label', required=True)
a = p.parse_args()
marker = os.environ.get('FAKE_GENERATION_MARKER')
if marker:
    Path(marker).write_text('started')
time.sleep(float(os.environ.get('FAKE_GENERATION_DELAY_SECONDS', '0')))
sub = pd.read_parquet(a.subset_parquet).reset_index(drop=True)
run_fingerprint = os.environ.get('FAKE_GENERATION_RUN_FINGERPRINT')
if not run_fingerprint:
    fingerprint_input = (
        Path(a.checkpoint).read_bytes()
        + Path(a.subset_parquet).read_bytes()
        + f'{a.batch_size}:{a.max_new_tokens}'.encode()
    )
    run_fingerprint = hashlib.sha256(fingerprint_input).hexdigest()
out = pd.DataFrame({
    'row_idx': range(len(sub)),
    'source_row_idx': sub['source_row_idx'],
    'waveform_name': [Path(v).stem for v in sub['waveform_path_psa']],
    'waveform_path': sub['waveform_path_psa'],
    'question': sub['prompt'],
    'generation': __GENERATION_EXPRESSION__,
    'ground_truth': sub['generated_answer'],
    'prompt_category': sub['prompt_category'],
    'checkpoint_path': str(Path(a.checkpoint).resolve()),
    'run_fingerprint': run_fingerprint,
})
if os.environ.get('FAKE_GENERATION_FRACTIONAL_ROW_ID') == '1':
    out.loc[0, 'row_idx'] = 0.5
if os.environ.get('FAKE_GENERATION_WRONG_WAVEFORM_NAME') == '1':
    out.loc[0, 'waveform_name'] = 'wrong-signal'
if os.environ.get('FAKE_GENERATION_MIXED_RUN_FINGERPRINTS') == '1' and len(out) > 1:
    out.loc[1, 'run_fingerprint'] = hashlib.sha256(b'another-run').hexdigest()
Path(a.output_dir).mkdir(parents=True, exist_ok=True)
out.to_csv(Path(a.output_dir) / f'generations_{a.label}.csv', index=False)
"""
    path.write_text(source.replace("__GENERATION_EXPRESSION__", expression))


def test_generator_publishes_only_an_exact_validated_artifact(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint-v1")
    subset = tmp_path / "subset.parquet"
    expected = _subset(subset)
    runner = tmp_path / "fake_runner.py"
    _fake_generation_runner(runner)
    output = tmp_path / "nested" / "generations.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "eval_judge_csv.py"),
            "--checkpoint",
            str(checkpoint),
            "--subset_parquet",
            str(subset),
            "--output_csv",
            str(output),
            "--device",
            "cpu",
            "--batch_size",
            "2",
        ],
        env={**os.environ, "ECG_EVAL_RUNNER": str(runner)},
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    got = pd.read_csv(output)
    assert len(got) == len(expected)
    assert got["source_row_idx"].tolist() == expected["source_row_idx"].tolist()
    manifest = json.loads(Path(f"{output}.manifest.json").read_text())
    assert manifest["row_count"] == len(expected)
    assert manifest["checkpoint"]["sha256"]
    assert manifest["subset"]["sha256"]


def test_resumable_workflow_never_reports_complete_after_background_failure(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _subset(out_dir / "fulltest_p0.parquet", rows=1)
    failing_generator = tmp_path / "failing_generator.py"
    failing_generator.write_text("raise SystemExit(23)\n")
    scorer = tmp_path / "scorer.py"
    scorer.write_text("print('score should not run')\n")

    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_fulltest_gen_resumable.sh")],
        env={
            **os.environ,
            "ECG_REPO_ROOT": str(ROOT),
            "ECG_PYTHON": sys.executable,
            "ECG_OUTDIR": str(out_dir),
            "ECG_EXPECTED_SHARDS": "0",
            "ECG_GENERATOR_SCRIPT": str(failing_generator),
            "ECG_SCORER_SCRIPT": str(scorer),
            "CKPT": str(checkpoint),
        },
        text=True,
        capture_output=True,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "COMPLETE" not in combined
    assert "score should not run" not in combined


def test_generator_rejects_error_rows_without_publishing(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    subset = tmp_path / "subset.parquet"
    _subset(subset)
    runner = tmp_path / "error_runner.py"
    _fake_generation_runner(
        runner,
        "['prediction 0', '[ERROR: decoder failed]']",
    )
    output = tmp_path / "generations.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "eval_judge_csv.py"),
            "--checkpoint",
            str(checkpoint),
            "--subset_parquet",
            str(subset),
            "--output_csv",
            str(output),
        ],
        env={**os.environ, "ECG_EVAL_RUNNER": str(runner)},
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "encoded an error" in result.stderr
    assert not output.exists()
    assert not Path(f"{output}.manifest.json").exists()


def test_generator_rejects_fractional_subset_source_row_identity(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    subset = tmp_path / "subset.parquet"
    frame = _subset(subset)
    frame["source_row_idx"] = frame["source_row_idx"].astype(float)
    frame.loc[0, "source_row_idx"] = 0.5
    frame.to_parquet(subset, index=False)
    runner = tmp_path / "runner.py"
    _fake_generation_runner(runner)
    output = tmp_path / "generations.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "eval_judge_csv.py"),
            "--checkpoint",
            str(checkpoint),
            "--subset_parquet",
            str(subset),
            "--output_csv",
            str(output),
        ],
        env={**os.environ, "ECG_EVAL_RUNNER": str(runner)},
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "subset source_row_idx must be an integer" in result.stderr
    assert not output.exists()


def test_generator_rejects_fractional_generated_row_identity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    subset = tmp_path / "subset.parquet"
    _subset(subset)
    runner = tmp_path / "runner.py"
    _fake_generation_runner(runner)
    output = tmp_path / "generations.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "eval_judge_csv.py"),
            "--checkpoint",
            str(checkpoint),
            "--subset_parquet",
            str(subset),
            "--output_csv",
            str(output),
        ],
        env={
            **os.environ,
            "ECG_EVAL_RUNNER": str(runner),
            "FAKE_GENERATION_FRACTIONAL_ROW_ID": "1",
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "generation column row_idx must be an integer" in result.stderr
    assert not output.exists()


def test_generator_rejects_rows_from_mixed_run_fingerprints(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    subset = tmp_path / "subset.parquet"
    _subset(subset)
    runner = tmp_path / "runner.py"
    _fake_generation_runner(runner)
    output = tmp_path / "generations.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "eval_judge_csv.py"),
            "--checkpoint",
            str(checkpoint),
            "--subset_parquet",
            str(subset),
            "--output_csv",
            str(output),
        ],
        env={
            **os.environ,
            "ECG_EVAL_RUNNER": str(runner),
            "FAKE_GENERATION_MIXED_RUN_FINGERPRINTS": "1",
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "one exact run_fingerprint" in result.stderr
    assert not output.exists()


def test_generator_rejects_waveform_name_not_bound_to_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    subset = tmp_path / "subset.parquet"
    _subset(subset)
    runner = tmp_path / "runner.py"
    _fake_generation_runner(runner)
    output = tmp_path / "generations.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "eval_judge_csv.py"),
            "--checkpoint",
            str(checkpoint),
            "--subset_parquet",
            str(subset),
            "--output_csv",
            str(output),
        ],
        env={
            **os.environ,
            "ECG_EVAL_RUNNER": str(runner),
            "FAKE_GENERATION_WRONG_WAVEFORM_NAME": "1",
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "generation waveform_name mismatch" in result.stderr
    assert not output.exists()


def test_validate_only_rejects_a_generation_from_an_old_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint-v1")
    subset = tmp_path / "subset.parquet"
    _subset(subset)
    runner = tmp_path / "runner.py"
    _fake_generation_runner(runner)
    output = tmp_path / "generations.csv"
    command = [
        sys.executable,
        str(SCRIPTS / "eval_judge_csv.py"),
        "--checkpoint",
        str(checkpoint),
        "--subset_parquet",
        str(subset),
        "--output_csv",
        str(output),
    ]
    environment = {**os.environ, "ECG_EVAL_RUNNER": str(runner)}

    assert subprocess.run(command, env=environment).returncode == 0
    checkpoint.write_bytes(b"checkpoint-v2")
    stale = subprocess.run(
        [*command, "--validate-only"],
        env=environment,
        text=True,
        capture_output=True,
    )

    assert stale.returncode != 0
    assert "checkpoint identity is stale" in stale.stderr


def test_generation_output_lock_serializes_two_complete_producer_lifecycles(
    tmp_path: Path,
) -> None:
    checkpoint_one = tmp_path / "model-one.pt"
    checkpoint_two = tmp_path / "model-two.pt"
    checkpoint_one.write_bytes(b"checkpoint-one")
    checkpoint_two.write_bytes(b"checkpoint-two")
    subset_one = tmp_path / "subset-one.parquet"
    subset_two = tmp_path / "subset-two.parquet"
    _subset(subset_one, start=0)
    _subset(subset_two, start=10)
    runner = tmp_path / "runner.py"
    _fake_generation_runner(runner)
    output = tmp_path / "generations.csv"

    def command(checkpoint: Path, subset: Path) -> list[str]:
        return [
            sys.executable,
            str(SCRIPTS / "eval_judge_csv.py"),
            "--checkpoint",
            str(checkpoint),
            "--subset_parquet",
            str(subset),
            "--output_csv",
            str(output),
        ]

    first_marker = tmp_path / "first-started"
    second_marker = tmp_path / "second-started"
    first = subprocess.Popen(
        command(checkpoint_one, subset_one),
        env={
            **os.environ,
            "ECG_EVAL_RUNNER": str(runner),
            "FAKE_GENERATION_MARKER": str(first_marker),
            "FAKE_GENERATION_DELAY_SECONDS": "1",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _wait_for_path(first_marker)
    second = subprocess.Popen(
        command(checkpoint_two, subset_two),
        env={
            **os.environ,
            "ECG_EVAL_RUNNER": str(runner),
            "FAKE_GENERATION_MARKER": str(second_marker),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(0.2)
    assert second.poll() is None
    assert not second_marker.exists()

    first_stdout, first_stderr = first.communicate(timeout=10)
    second_stdout, second_stderr = second.communicate(timeout=10)
    assert first.returncode == 0, first_stdout + first_stderr
    assert second.returncode == 0, second_stdout + second_stderr
    _wait_for_path(second_marker)

    winning_validation = subprocess.run(
        [*command(checkpoint_two, subset_two), "--validate-only"],
        env={**os.environ, "ECG_EVAL_RUNNER": str(runner)},
        text=True,
        capture_output=True,
    )
    losing_validation = subprocess.run(
        [*command(checkpoint_one, subset_one), "--validate-only"],
        env={**os.environ, "ECG_EVAL_RUNNER": str(runner)},
        text=True,
        capture_output=True,
    )
    assert winning_validation.returncode == 0, winning_validation.stderr
    assert losing_validation.returncode != 0


def test_generation_manifest_invalidates_when_implementation_source_changes(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    subset = tmp_path / "subset.parquet"
    _subset(subset)
    runner = tmp_path / "runner.py"
    _fake_generation_runner(runner)
    implementation = tmp_path / "implementation"
    implementation.mkdir()
    source = implementation / "decoder.py"
    source.write_text("DECODER_VERSION = 1\n")
    output = tmp_path / "generations.csv"
    command = [
        sys.executable,
        str(SCRIPTS / "eval_judge_csv.py"),
        "--checkpoint",
        str(checkpoint),
        "--subset_parquet",
        str(subset),
        "--output_csv",
        str(output),
    ]
    environment = {
        **os.environ,
        "ECG_EVAL_RUNNER": str(runner),
        "ECG_GENERATION_IMPLEMENTATION_ROOTS": str(implementation),
    }

    assert subprocess.run(command, env=environment).returncode == 0
    first = json.loads(Path(f"{output}.manifest.json").read_text())
    source.write_text("DECODER_VERSION = 2\n")
    stale = subprocess.run(
        [*command, "--validate-only"],
        env=environment,
        text=True,
        capture_output=True,
    )

    assert stale.returncode != 0
    assert "configuration is stale" in stale.stderr
    assert first["config"]["implementation"]["sha256"]


def test_generation_manifest_invalidates_when_runtime_dependency_version_changes(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    subset = tmp_path / "subset.parquet"
    _subset(subset)
    runner = tmp_path / "runner.py"
    _fake_generation_runner(runner)
    output = tmp_path / "generations.csv"
    metadata_root = tmp_path / "metadata"
    pandas_metadata = metadata_root / "pandas-98.0.dist-info"
    sentencepiece_metadata = metadata_root / "sentencepiece-97.0.dist-info"
    pandas_metadata.mkdir(parents=True)
    sentencepiece_metadata.mkdir(parents=True)
    pandas_metadata.joinpath("METADATA").write_text(
        "Metadata-Version: 2.1\nName: pandas\nVersion: 98.0\n"
    )
    sentencepiece_file = sentencepiece_metadata / "METADATA"
    sentencepiece_file.write_text(
        "Metadata-Version: 2.1\nName: sentencepiece\nVersion: 97.0\n"
    )
    command = [
        sys.executable,
        str(SCRIPTS / "eval_judge_csv.py"),
        "--checkpoint",
        str(checkpoint),
        "--subset_parquet",
        str(subset),
        "--output_csv",
        str(output),
    ]
    environment = {
        **os.environ,
        "ECG_EVAL_RUNNER": str(runner),
        "PYTHONPATH": os.pathsep.join(
            [str(metadata_root), os.environ.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep),
    }

    first = subprocess.run(command, env=environment, text=True, capture_output=True)
    assert first.returncode == 0, first.stderr
    manifest = json.loads(Path(f"{output}.manifest.json").read_text())
    runtime = manifest["config"]["implementation"]["runtime"]
    assert runtime["pandas"] == "98.0"
    assert runtime["sentencepiece"] == "97.0"

    sentencepiece_file.write_text(
        "Metadata-Version: 2.1\nName: sentencepiece\nVersion: 97.1\n"
    )
    stale = subprocess.run(
        [*command, "--validate-only"],
        env=environment,
        text=True,
        capture_output=True,
    )

    assert stale.returncode != 0
    assert "configuration is stale" in stale.stderr


def test_fulltest_workflow_generates_exact_shards_and_scores_validated_merge(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    expected_rows = 0
    for shard in (0, 1):
        expected_rows += len(
            _subset(
                out_dir / f"fulltest_p{shard}.parquet",
                rows=shard + 1,
                start=shard * 100,
            )
        )
    runner = tmp_path / "runner.py"
    _fake_generation_runner(runner)
    scorer = tmp_path / "scorer.py"
    scorer.write_text(
        """import argparse
import pandas as pd
p = argparse.ArgumentParser()
p.add_argument('csv')
p.add_argument('--tag', required=True)
p.add_argument('--expected-category', action='append')
a = p.parse_args()
print(f'rows={len(pd.read_csv(a.csv))} tag={a.tag}')
"""
    )
    environment = {
        **os.environ,
        "ECG_REPO_ROOT": str(ROOT),
        "ECG_PYTHON": sys.executable,
        "ECG_OUTDIR": str(out_dir),
        "ECG_LOG_DIR": str(tmp_path / "logs"),
        "ECG_EXPECTED_SHARDS": "0 1",
        "ECG_DEVICES": "cpu,cpu",
        "ECG_EVAL_RUNNER": str(runner),
        "ECG_SCORER_SCRIPT": str(scorer),
        "ECG_SCORER_IMPLEMENTATION_ROOTS": str(tmp_path),
    }

    result = subprocess.run(
        [
            "bash",
            str(SCRIPTS / "run_fulltest_gen_any.sh"),
            str(checkpoint),
            "contract",
        ],
        env=environment,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    merged = pd.read_csv(out_dir / "contract_fulltest_all.csv")
    assert len(merged) == expected_rows
    score = (out_dir / "contract_fulltest_deterministic.txt").read_text()
    assert f"rows={expected_rows}" in score
    score_manifest = json.loads(
        (out_dir / "contract_fulltest_deterministic.txt.manifest.json").read_text()
    )
    assert score_manifest["source"] == _artifact_identity(
        out_dir / "contract_fulltest_all.csv"
    )
    assert score_manifest["score_implementation"]["runtime"]["pandas"]
    assert score_manifest["config"]["expected_categories"] == [
        "lvef",
        "afib_risk",
        "structural_heart_disease",
        "acs_severity",
    ]
    assert "GENERATION+SCORING COMPLETE" in result.stdout


def test_fulltest_workflow_propagates_scorer_failure_without_completion(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _subset(out_dir / "fulltest_p0.parquet", rows=1)
    runner = tmp_path / "runner.py"
    _fake_generation_runner(runner)
    scorer = tmp_path / "failing_scorer.py"
    scorer.write_text("print('scorer failed'); raise SystemExit(19)\n")

    result = subprocess.run(
        [
            "bash",
            str(SCRIPTS / "run_fulltest_gen_any.sh"),
            str(checkpoint),
            "scorefail",
        ],
        env={
            **os.environ,
            "ECG_REPO_ROOT": str(ROOT),
            "ECG_PYTHON": sys.executable,
            "ECG_OUTDIR": str(out_dir),
            "ECG_LOG_DIR": str(tmp_path / "logs"),
            "ECG_EXPECTED_SHARDS": "0",
            "ECG_DEVICES": "cpu",
            "ECG_EVAL_RUNNER": str(runner),
            "ECG_SCORER_SCRIPT": str(scorer),
            "ECG_SCORER_IMPLEMENTATION_ROOTS": str(tmp_path),
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "GENERATION+SCORING COMPLETE" not in result.stdout
    assert not (out_dir / "scorefail_fulltest_deterministic.txt").exists()


def test_fulltest_rejects_unknown_only_deterministic_score_input(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _subset(out_dir / "fulltest_p0.parquet", rows=2)
    runner = tmp_path / "runner.py"
    _fake_generation_runner(runner)

    result = subprocess.run(
        [
            "bash",
            str(SCRIPTS / "run_fulltest_gen_any.sh"),
            str(checkpoint),
            "unknown",
        ],
        env={
            **os.environ,
            "ECG_REPO_ROOT": str(ROOT),
            "ECG_PYTHON": sys.executable,
            "ECG_OUTDIR": str(out_dir),
            "ECG_LOG_DIR": str(tmp_path / "logs"),
            "ECG_EXPECTED_SHARDS": "0",
            "ECG_DEVICES": "cpu",
            "ECG_EVAL_RUNNER": str(runner),
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "expected score categories are missing" in result.stdout
    assert "GENERATION+SCORING COMPLETE" not in result.stdout
    assert not (out_dir / "unknown_fulltest_deterministic.txt").exists()


def test_fulltest_rejects_one_class_binary_score_input(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    subset_path = out_dir / "fulltest_p0.parquet"
    subset = _subset(subset_path, rows=20)
    subset["generated_answer"] = "Yes, high risk."
    subset["prompt_category"] = "afib_risk"
    subset.to_parquet(subset_path, index=False)
    runner = tmp_path / "runner.py"
    _fake_generation_runner(runner, "['Yes, high risk.'] * len(sub)")

    result = subprocess.run(
        [
            "bash",
            str(SCRIPTS / "run_fulltest_gen_any.sh"),
            str(checkpoint),
            "oneclass",
        ],
        env={
            **os.environ,
            "ECG_REPO_ROOT": str(ROOT),
            "ECG_PYTHON": sys.executable,
            "ECG_OUTDIR": str(out_dir),
            "ECG_LOG_DIR": str(tmp_path / "logs"),
            "ECG_EXPECTED_SHARDS": "0",
            "ECG_DEVICES": "cpu",
            "ECG_EVAL_RUNNER": str(runner),
            "ECG_EXPECTED_SCORE_CATEGORIES": "afib_risk",
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "requires both ground-truth classes" in result.stdout
    assert "GENERATION+SCORING COMPLETE" not in result.stdout
    assert not (out_dir / "oneclass_fulltest_deterministic.txt").exists()


def test_fulltest_same_tag_serializes_merge_and_scoring_as_one_lifecycle(
    tmp_path: Path,
) -> None:
    checkpoint_one = tmp_path / "model-one.pt"
    checkpoint_two = tmp_path / "model-two.pt"
    checkpoint_one.write_bytes(b"checkpoint-one")
    checkpoint_two.write_bytes(b"checkpoint-two")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _subset(out_dir / "fulltest_p0.parquet", rows=1)
    runner = tmp_path / "runner.py"
    _fake_generation_runner(runner)
    scorer = tmp_path / "blocking_scorer.py"
    scorer.write_text(
        """import argparse
import os
import time
from pathlib import Path
import pandas as pd
p = argparse.ArgumentParser()
p.add_argument('csv')
p.add_argument('--tag', required=True)
p.add_argument('--expected-category', action='append')
a = p.parse_args()
marker = os.environ.get('FAKE_SCORER_MARKER')
if marker:
    Path(marker).write_text('started')
time.sleep(float(os.environ.get('FAKE_SCORER_DELAY_SECONDS', '0')))
frame = pd.read_csv(a.csv)
print(f'checkpoint={frame.iloc[0]["checkpoint_path"]} rows={len(frame)} tag={a.tag}')
"""
    )
    first_scorer_marker = tmp_path / "first-scorer-started"
    second_generation_marker = tmp_path / "second-generation-started"
    base_environment = {
        **os.environ,
        "ECG_REPO_ROOT": str(ROOT),
        "ECG_PYTHON": sys.executable,
        "ECG_OUTDIR": str(out_dir),
        "ECG_LOG_DIR": str(tmp_path / "logs"),
        "ECG_EXPECTED_SHARDS": "0",
        "ECG_DEVICES": "cpu",
        "ECG_EVAL_RUNNER": str(runner),
        "ECG_SCORER_SCRIPT": str(scorer),
        "ECG_SCORER_IMPLEMENTATION_ROOTS": str(tmp_path),
    }

    def command(checkpoint: Path) -> list[str]:
        return [
            "bash",
            str(SCRIPTS / "run_fulltest_gen_any.sh"),
            str(checkpoint),
            "shared",
        ]

    first = subprocess.Popen(
        command(checkpoint_one),
        env={
            **base_environment,
            "FAKE_SCORER_MARKER": str(first_scorer_marker),
            "FAKE_SCORER_DELAY_SECONDS": "1",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _wait_for_path(first_scorer_marker, timeout=40)
    second = subprocess.Popen(
        command(checkpoint_two),
        env={
            **base_environment,
            "FAKE_GENERATION_MARKER": str(second_generation_marker),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(0.2)
    assert second.poll() is None
    assert not second_generation_marker.exists()

    first_stdout, first_stderr = first.communicate(timeout=40)
    second_stdout, second_stderr = second.communicate(timeout=40)
    assert first.returncode == 0, first_stdout + first_stderr
    assert second.returncode == 0, second_stdout + second_stderr
    _wait_for_path(second_generation_marker)

    merged = out_dir / "shared_fulltest_all.csv"
    score_output = out_dir / "shared_fulltest_deterministic.txt"
    assert str(checkpoint_two.resolve()) in score_output.read_text()
    manifest = json.loads(Path(f"{score_output}.manifest.json").read_text())
    assert manifest["source"] == _artifact_identity(merged)
    validation = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "eval_workflow_utils.py"),
            "validate-score",
            "--source-csv",
            str(merged),
            "--score-output",
            str(score_output),
            "--scorer-script",
            str(scorer),
            "--implementation-root",
            str(tmp_path),
            "--expected-category",
            "lvef",
            "--expected-category",
            "afib_risk",
            "--expected-category",
            "structural_heart_disease",
            "--expected-category",
            "acs_severity",
            "--tag",
            "shared_FULLTEST",
        ],
        text=True,
        capture_output=True,
    )
    assert validation.returncode == 0, validation.stderr

    scorer.write_text(scorer.read_text() + "\n# scorer revision\n")
    stale_validation = subprocess.run(
        validation.args,
        text=True,
        capture_output=True,
    )
    assert stale_validation.returncode != 0
    assert "score implementation identity is stale" in stale_validation.stderr


def _artifact_identity(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _generation_merge_shard(
    path: Path,
    checkpoint: Path,
    subset: Path,
    source_rows: list[int],
    *,
    device: str = "cpu:0",
    max_new_tokens: int = 96,
    logical_row_override: tuple[str, str, str, str] | None = None,
) -> None:
    rows = []
    subset_rows = []
    run_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "checkpoint": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "subset": subset.stem,
                "source_rows": source_rows,
                "max_new_tokens": max_new_tokens,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    for local_index, source_index in enumerate(source_rows):
        waveform_path = f"/signals/{source_index}.npy"
        question = f"question {source_index}"
        ground_truth = f"answer {source_index}"
        category = "rhythm"
        if logical_row_override is not None and local_index == 0:
            waveform_path, question, ground_truth, category = logical_row_override
        rows.append(
            {
                "row_idx": local_index,
                "source_row_idx": source_index,
                "waveform_name": Path(waveform_path).stem,
                "waveform_path": waveform_path,
                "question": question,
                "generation": f"prediction {source_index}",
                "ground_truth": ground_truth,
                "prompt_category": category,
                "checkpoint_path": str(checkpoint.resolve()),
                "run_fingerprint": run_fingerprint,
            }
        )
        subset_rows.append(
            {
                "source_row_idx": source_index,
                "waveform_path_psa": waveform_path,
                "prompt": question,
                "generated_answer": ground_truth,
                "prompt_category": category,
                "partition_marker": subset.stem,
            }
        )
    pd.DataFrame(subset_rows).to_parquet(subset, index=False)
    pd.DataFrame(rows).to_csv(path, index=False)
    manifest = {
        "schema_version": 1,
        "kind": "ecg_generation_csv",
        "row_count": len(rows),
        "checkpoint": _artifact_identity(checkpoint),
        "subset": _artifact_identity(subset),
        "config": {
            "device": device,
            "max_new_tokens": max_new_tokens,
            "implementation": "contract-v1",
        },
        "run_fingerprint": run_fingerprint,
        "output": _artifact_identity(path),
    }
    Path(f"{path}.manifest.json").write_text(json.dumps(manifest))


def _run_generation_merge(output: Path, *inputs: Path) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(SCRIPTS / "eval_workflow_utils.py"),
        "merge-generations",
        "--output",
        str(output),
    ]
    for input_path in inputs:
        command.extend(["--input", str(input_path)])
    return subprocess.run(command, text=True, capture_output=True)


def test_generation_merge_persists_common_provenance_and_exact_partition(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"one-model")
    subset_zero = tmp_path / "fulltest_p0.parquet"
    subset_one = tmp_path / "fulltest_p1.parquet"
    subset_zero.write_bytes(b"subset-zero")
    subset_one.write_bytes(b"subset-one")
    shard_zero = tmp_path / "generation_p0.csv"
    shard_one = tmp_path / "generation_p1.csv"
    _generation_merge_shard(
        shard_zero, checkpoint, subset_zero, [0, 2], device="cuda:0"
    )
    _generation_merge_shard(
        shard_one, checkpoint, subset_one, [1, 3], device="cuda:1"
    )
    output = tmp_path / "merged.csv"

    result = _run_generation_merge(output, shard_zero, shard_one)

    assert result.returncode == 0, result.stderr
    manifest = json.loads(Path(f"{output}.manifest.json").read_text())
    provenance = manifest["provenance"]
    assert provenance["checkpoint_content"]["sha256"] == hashlib.sha256(
        b"one-model"
    ).hexdigest()
    assert "device" not in provenance["generation_config"]
    partitions = provenance["subset_run"]["partitions"]
    assert [item["source_row_ids"] for item in partitions] == [[0, 2], [1, 3]]
    assert [item["execution_device"] for item in partitions] == ["cuda:0", "cuda:1"]
    assert [item["run_fingerprint"] for item in partitions] == [
        pd.read_csv(shard_zero)["run_fingerprint"].iloc[0],
        pd.read_csv(shard_one)["run_fingerprint"].iloc[0],
    ]


def test_generation_merge_rejects_duplicate_shard_paths(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    subset = tmp_path / "fulltest_p0.parquet"
    subset.write_bytes(b"subset")
    shard = tmp_path / "generation_p0.csv"
    _generation_merge_shard(shard, checkpoint, subset, [0])

    result = _run_generation_merge(tmp_path / "merged.csv", shard, shard)

    assert result.returncode != 0
    assert "duplicate generation shard paths" in result.stderr


def test_generation_merge_rejects_shard_and_manifest_replacement_after_read(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workflow_utils = _load_workflow_utils()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    subset = tmp_path / "fulltest_p0.parquet"
    shard = tmp_path / "generation_p0.csv"
    _generation_merge_shard(shard, checkpoint, subset, [0])
    original_manifest = json.loads(Path(f"{shard}.manifest.json").read_text())
    replacement = pd.read_csv(shard)
    replacement["generation"] = ["replacement"]
    replacement_path = tmp_path / "replacement.csv"
    replacement.to_csv(replacement_path, index=False)
    replacement_content = replacement_path.read_bytes()
    replacement_manifest = dict(original_manifest)
    replacement_manifest["output"] = {
        "path": str(shard.resolve()),
        "size": len(replacement_content),
        "sha256": hashlib.sha256(replacement_content).hexdigest(),
    }
    original_read_csv = workflow_utils.pd.read_csv
    replaced = False

    def replace_after_read(path, *args, **kwargs):
        nonlocal replaced
        frame = original_read_csv(path, *args, **kwargs)
        if not replaced and Path(path).resolve() == shard.resolve():
            shard.write_bytes(replacement_content)
            Path(f"{shard}.manifest.json").write_text(
                json.dumps(replacement_manifest)
            )
            replaced = True
        return frame

    monkeypatch.setattr(workflow_utils.pd, "read_csv", replace_after_read)

    with pytest.raises(
        workflow_utils.ArtifactValidationError,
        match="generation shard .* changed while it was being read",
    ):
        workflow_utils.merge_generation_csvs([shard], tmp_path / "merged.csv")

    assert not (tmp_path / "merged.csv").exists()


def test_generation_merge_rejects_mixed_checkpoint_content(tmp_path: Path) -> None:
    checkpoint_zero = tmp_path / "model-zero.pt"
    checkpoint_one = tmp_path / "model-one.pt"
    checkpoint_zero.write_bytes(b"model-zero")
    checkpoint_one.write_bytes(b"model-one")
    subset_zero = tmp_path / "fulltest_p0.parquet"
    subset_one = tmp_path / "fulltest_p1.parquet"
    subset_zero.write_bytes(b"subset-zero")
    subset_one.write_bytes(b"subset-one")
    shard_zero = tmp_path / "generation_p0.csv"
    shard_one = tmp_path / "generation_p1.csv"
    _generation_merge_shard(shard_zero, checkpoint_zero, subset_zero, [0])
    _generation_merge_shard(shard_one, checkpoint_one, subset_one, [1])

    result = _run_generation_merge(tmp_path / "merged.csv", shard_zero, shard_one)

    assert result.returncode != 0
    assert "different checkpoint content" in result.stderr


def test_generation_merge_rejects_mixed_output_config(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    subset_zero = tmp_path / "fulltest_p0.parquet"
    subset_one = tmp_path / "fulltest_p1.parquet"
    subset_zero.write_bytes(b"subset-zero")
    subset_one.write_bytes(b"subset-one")
    shard_zero = tmp_path / "generation_p0.csv"
    shard_one = tmp_path / "generation_p1.csv"
    _generation_merge_shard(shard_zero, checkpoint, subset_zero, [0])
    _generation_merge_shard(
        shard_one, checkpoint, subset_one, [1], max_new_tokens=128
    )

    result = _run_generation_merge(tmp_path / "merged.csv", shard_zero, shard_one)

    assert result.returncode != 0
    assert "incompatible output-defining configs" in result.stderr


def test_generation_merge_rejects_mixed_subset_run_identity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    subset_zero = tmp_path / "run_a_p0.parquet"
    subset_one = tmp_path / "run_b_p1.parquet"
    shard_zero = tmp_path / "generation_p0.csv"
    shard_one = tmp_path / "generation_p1.csv"
    _generation_merge_shard(shard_zero, checkpoint, subset_zero, [0])
    _generation_merge_shard(shard_one, checkpoint, subset_one, [1])

    result = _run_generation_merge(tmp_path / "merged.csv", shard_zero, shard_one)

    assert result.returncode != 0
    assert "different run identities" in result.stderr


def test_generation_merge_rejects_overlapping_partition_rows(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    subset_zero = tmp_path / "fulltest_p0.parquet"
    subset_one = tmp_path / "fulltest_p1.parquet"
    subset_zero.write_bytes(b"subset-zero")
    subset_one.write_bytes(b"subset-one")
    shard_zero = tmp_path / "generation_p0.csv"
    shard_one = tmp_path / "generation_p1.csv"
    _generation_merge_shard(shard_zero, checkpoint, subset_zero, [7])
    _generation_merge_shard(shard_one, checkpoint, subset_one, [7])

    result = _run_generation_merge(tmp_path / "merged.csv", shard_zero, shard_one)

    assert result.returncode != 0
    assert "partition overlaps source_row_idx 7" in result.stderr


def test_generation_merge_rejects_duplicate_logical_inputs(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    subset_zero = tmp_path / "fulltest_p0.parquet"
    subset_one = tmp_path / "fulltest_p1.parquet"
    subset_zero.write_bytes(b"subset-zero")
    subset_one.write_bytes(b"subset-one")
    shard_zero = tmp_path / "generation_p0.csv"
    shard_one = tmp_path / "generation_p1.csv"
    duplicate = ("/signals/shared.npy", "same question", "same answer", "rhythm")
    _generation_merge_shard(
        shard_zero, checkpoint, subset_zero, [0], logical_row_override=duplicate
    )
    _generation_merge_shard(
        shard_one, checkpoint, subset_one, [1], logical_row_override=duplicate
    )

    result = _run_generation_merge(tmp_path / "merged.csv", shard_zero, shard_one)

    assert result.returncode != 0
    assert "duplicate logical input" in result.stderr


def _judge_csv(path: Path, rows: int = 5) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "waveform_name": [f"signal-{idx}" for idx in range(rows)],
            "waveform_path": [f"/signals/{idx}.npy" for idx in range(rows)],
            "question": [f"question, {idx}" for idx in range(rows)],
            "generation": [f"prediction {idx}" for idx in range(rows)],
            "ground_truth": [f"answer {idx}" for idx in range(rows)],
            "prompt_category": ["rhythm" if idx % 2 == 0 else "interval" for idx in range(rows)],
        }
    )
    frame.to_csv(path, index=False)
    return frame


def _fake_judge(path: Path) -> None:
    (path.parent / "judge_support.py").write_text("SCORE = 1.0\n")
    path.write_text(
        """#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path
import pandas as pd
from judge_support import SCORE

def aggregate_results(results):
    examples = results['per_example_verdicts']
    by_category = {}
    for example in examples:
        by_category.setdefault(example['prompt_category'], []).append(float(example['overall_score']))
    aggregates = {
        'overall_score': sum(float(e['overall_score']) for e in examples) / len(examples),
        'category_aggregates': {
            category: {'count': len(scores), 'mean_score': sum(scores) / len(scores)}
            for category, scores in by_category.items()
        },
    }
    if os.environ.get('FAKE_JUDGE_CONTRADICT_AGGREGATES') == '1':
        aggregates['overall_score'] = 0.25
    if os.environ.get('FAKE_JUDGE_OUT_OF_RANGE_AGGREGATE') == '1':
        aggregates['overall_score'] = 1.5
    if os.environ.get('FAKE_JUDGE_BOOLEAN_AGGREGATE') == '1':
        aggregates['overall_score'] = True
    return aggregates

def save_results(results, aggregates, output, verbose=True):
    Path(output).write_text(json.dumps({**results, 'aggregates': aggregates}))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    frame = pd.read_csv(a.csv)
    marker = os.environ.get('FAKE_JUDGE_MARKER')
    if marker:
        Path(marker).write_text('started')
    time.sleep(float(os.environ.get('FAKE_JUDGE_DELAY_SECONDS', '0')))
    fail_on = os.environ.get('FAKE_JUDGE_FAIL_ON')
    if fail_on and frame['question'].astype(str).str.contains(fail_on, regex=False).any():
        raise SystemExit(31)
    examples = []
    for _, row in frame.iterrows():
        category = str(row['prompt_category'])
        overall_score = float(os.environ.get('FAKE_JUDGE_OVERALL_SCORE', SCORE))
        verdict_score = float(os.environ.get('FAKE_JUDGE_VERDICT_SCORE', SCORE))
        if os.environ.get('FAKE_JUDGE_BOOLEAN_OVERALL') == '1':
            overall_score = True
        if os.environ.get('FAKE_JUDGE_BOOLEAN_VERDICT') == '1':
            verdict_score = False
        examples.append({
            'workflow_row_id': int(row['workflow_row_id']),
            'prompt_category': category,
            'overall_score': overall_score,
            'verdicts': {'test_judge': {'category': category, 'score': verdict_score}},
        })
    if os.environ.get('FAKE_JUDGE_DROP_LAST') == '1':
        examples = examples[:-1]
    if os.environ.get('FAKE_JUDGE_OMIT_ROW_IDS') == '1':
        for example in examples:
            example.pop('workflow_row_id')
    if os.environ.get('FAKE_JUDGE_REVERSE_EXAMPLES') == '1':
        examples = list(reversed(examples))
    results = {
        'per_example_verdicts': examples,
        'per_category_scores': {},
        'per_judge_category_scores': {},
    }
    save_results(results, aggregate_results(results), a.output)

if __name__ == '__main__':
    main()
"""
    )


def _judge_environment(
    csv_path: Path,
    output: Path,
    out_dir: Path,
    judge_dir: Path,
) -> dict[str, str]:
    return {
        **os.environ,
        "CSV": str(csv_path),
        "OUTPUT": str(output),
        "OUT_DIR": str(out_dir),
        "SHARDS": "3",
        "MAX_PARALLEL": "2",
        "PYTHON": sys.executable,
        "JUDGE_DIR": str(judge_dir),
        "JUDGE_MODEL_ID": "judge-model-snapshot-2026-08-25",
        "JUDGE_DEPLOYMENT_ID": "judge-deployment-revision-17",
        "JUDGE_DECODING_ID": "temperature-0-seed-123-v1",
    }


def test_judge_split_rejects_input_mutation_during_shard_publication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workflow_utils = _load_workflow_utils()
    csv_path = tmp_path / "generations.csv"
    replacement = _judge_csv(csv_path, rows=2).copy()
    replacement["generation"] = ["replacement 0", "replacement 1"]
    out_dir = tmp_path / "shards"
    original_atomic_csv = workflow_utils._atomic_csv
    mutated = False

    def mutate_after_first_shard(path, frame):
        nonlocal mutated
        original_atomic_csv(path, frame)
        if not mutated and Path(path).name.startswith("judge_input_"):
            replacement.to_csv(csv_path, index=False)
            mutated = True

    monkeypatch.setattr(workflow_utils, "_atomic_csv", mutate_after_first_shard)

    with pytest.raises(
        workflow_utils.ArtifactValidationError,
        match="judge CSV changed while shards were being written",
    ):
        workflow_utils.split_judge_csv(csv_path, out_dir, 2)

    assert not (out_dir / "manifest.json").exists()


def test_judge_split_rejects_shards_not_reconstructing_bound_input(
    tmp_path: Path,
) -> None:
    workflow_utils = _load_workflow_utils()
    csv_path = tmp_path / "generations.csv"
    replacement = _judge_csv(csv_path, rows=2).copy()
    out_dir = tmp_path / "shards"
    workflow_utils.split_judge_csv(csv_path, out_dir, 2)
    replacement["generation"] = ["replacement 0", "replacement 1"]
    replacement.to_csv(csv_path, index=False)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["input"] = workflow_utils.file_identity(csv_path)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(
        workflow_utils.ArtifactValidationError,
        match="do not reconstruct the exact ordered input",
    ):
        workflow_utils._load_split_manifest(csv_path, manifest_path)


def test_seal_judge_shard_rejects_input_mutation_before_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workflow_utils = _load_workflow_utils()
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    judge_script = judge_dir / "judge_eval.py"
    _fake_judge(judge_script)
    shard_csv = tmp_path / "judge_input_000.csv"
    shard = _judge_csv(shard_csv, rows=1)
    shard["workflow_row_id"] = [0]
    shard.to_csv(shard_csv, index=False)
    raw_output = tmp_path / "raw.json"
    raw_output.write_text(
        json.dumps(
            {
                "per_example_verdicts": [
                    {
                        "workflow_row_id": 0,
                        "prompt_category": "rhythm",
                        "overall_score": 1.0,
                        "verdicts": {
                            "test_judge": {
                                "category": "rhythm",
                                "score": 1.0,
                            }
                        },
                    }
                ]
            }
        )
    )
    sealed_output = tmp_path / "sealed.json"
    original_atomic_json = workflow_utils._atomic_json
    mutated = False

    def mutate_after_output(path, data):
        nonlocal mutated
        original_atomic_json(path, data)
        if not mutated and Path(path) == sealed_output:
            replacement = shard.copy()
            replacement["generation"] = ["replacement"]
            replacement.to_csv(shard_csv, index=False)
            mutated = True

    monkeypatch.setattr(workflow_utils, "_atomic_json", mutate_after_output)

    with pytest.raises(
        workflow_utils.ArtifactValidationError,
        match="judge shard changed before result publication",
    ):
        workflow_utils.seal_judge_shard(
            raw_output,
            shard_csv,
            sealed_output,
            judge_dir,
            judge_script,
            "test-run-v1",
        )

    assert not Path(f"{sealed_output}.manifest.json").exists()


def test_validate_judge_shard_rejects_bundle_mutation_during_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workflow_utils = _load_workflow_utils()
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    judge_script = judge_dir / "judge_eval.py"
    _fake_judge(judge_script)
    shard_csv = tmp_path / "judge_input_000.csv"
    shard = _judge_csv(shard_csv, rows=1)
    shard["workflow_row_id"] = [0]
    shard.to_csv(shard_csv, index=False)
    raw_output = tmp_path / "raw.json"
    raw_output.write_text(
        json.dumps(
            {
                "per_example_verdicts": [
                    {
                        "workflow_row_id": 0,
                        "prompt_category": "rhythm",
                        "overall_score": 1.0,
                        "verdicts": {
                            "test_judge": {
                                "category": "rhythm",
                                "score": 1.0,
                            }
                        },
                    }
                ]
            }
        )
    )
    sealed_output = tmp_path / "sealed.json"
    workflow_utils.seal_judge_shard(
        raw_output,
        shard_csv,
        sealed_output,
        judge_dir,
        judge_script,
        "test-run-v1",
    )
    replacement = shard.copy()
    replacement["generation"] = ["replacement"]
    original_read_csv = workflow_utils.pd.read_csv
    mutated = False

    def mutate_after_read(path, *args, **kwargs):
        nonlocal mutated
        frame = original_read_csv(path, *args, **kwargs)
        if not mutated and Path(path).resolve() == shard_csv.resolve():
            replacement.to_csv(shard_csv, index=False)
            mutated = True
        return frame

    monkeypatch.setattr(workflow_utils.pd, "read_csv", mutate_after_read)

    with pytest.raises(
        workflow_utils.ArtifactValidationError,
        match="judge shard bundle changed during validation",
    ):
        workflow_utils.validate_judge_shard(
            shard_csv,
            sealed_output,
            judge_dir,
            judge_script,
            "test-run-v1",
        )


def test_merge_judge_rejects_input_replacement_after_output_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workflow_utils = _load_workflow_utils()
    csv_path = tmp_path / "generations.csv"
    replacement = _judge_csv(csv_path, rows=1).copy()
    out_dir = tmp_path / "shards"
    split = workflow_utils.split_judge_csv(csv_path, out_dir, 1)
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    judge_script = judge_dir / "judge_eval.py"
    _fake_judge(judge_script)
    raw_output = tmp_path / "raw.json"
    raw_output.write_text(
        json.dumps(
            {
                "per_example_verdicts": [
                    {
                        "workflow_row_id": 0,
                        "prompt_category": "rhythm",
                        "overall_score": 1.0,
                        "verdicts": {
                            "test_judge": {
                                "category": "rhythm",
                                "score": 1.0,
                            }
                        },
                    }
                ]
            }
        )
    )
    item = split["items"][0]
    workflow_utils.seal_judge_shard(
        raw_output,
        Path(item["input"]["path"]),
        Path(item["judge_output"]),
        judge_dir,
        judge_script,
        "test-run-v1",
    )
    output = tmp_path / "merged.json"
    original_replace = workflow_utils.os.replace

    def mutate_after_replace(source, destination):
        original_replace(source, destination)
        if Path(destination) == output:
            replacement["generation"] = ["replacement"]
            replacement.to_csv(csv_path, index=False)

    monkeypatch.setattr(workflow_utils.os, "replace", mutate_after_replace)

    with pytest.raises(
        workflow_utils.ArtifactValidationError,
        match="judge input changed before result publication",
    ):
        workflow_utils.merge_judge_outputs(
            csv_path,
            out_dir / "manifest.json",
            output,
            judge_dir,
            judge_script,
            "test-run-v1",
        )

    assert not Path(f"{output}.manifest.json").exists()


def test_validate_merged_judge_rejects_bundle_mutation_during_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    csv_path = tmp_path / "generations.csv"
    original_frame = _judge_csv(csv_path, rows=2)
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    _fake_judge(judge_dir / "judge_eval.py")
    output = tmp_path / "judge.json"
    shard_dir = tmp_path / "shards"
    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_csv_llm_judge_sharded.sh")],
        env={
            **_judge_environment(csv_path, output, shard_dir, judge_dir),
            "SHARDS": "1",
            "MAX_PARALLEL": "1",
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    workflow_utils = _load_workflow_utils()
    manifest = json.loads(Path(f"{output}.manifest.json").read_text())
    run_id = manifest["judge_implementation"]["run_id"]
    replacement = original_frame.copy()
    replacement["generation"] = ["replacement 0", "replacement 1"]
    original_load_json = workflow_utils._load_json
    mutated = False

    def mutate_after_output_read(path, label):
        nonlocal mutated
        payload = original_load_json(path, label)
        if not mutated and Path(path).resolve() == output.resolve():
            replacement.to_csv(csv_path, index=False)
            mutated = True
        return payload

    monkeypatch.setattr(workflow_utils, "_load_json", mutate_after_output_read)

    with pytest.raises(
        workflow_utils.ArtifactValidationError,
        match="merged judge bundle changed during validation",
    ):
        workflow_utils.validate_merged_judge(
            csv_path,
            output,
            1,
            judge_dir,
            judge_dir / "judge_eval.py",
            run_id,
        )


def test_sharded_judge_covers_every_input_row_once_and_publishes_atomically(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "generations.csv"
    expected = _judge_csv(csv_path)
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    _fake_judge(judge_dir / "judge_eval.py")
    output = tmp_path / "nested" / "judge.json"
    shard_dir = tmp_path / "shards"

    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_csv_llm_judge_sharded.sh")],
        env=_judge_environment(csv_path, output, shard_dir, judge_dir),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(output.read_text())
    examples = payload["per_example_verdicts"]
    assert len(examples) == len(expected)
    assert [item["workflow_row_id"] for item in examples] == list(range(len(expected)))
    manifest = json.loads(Path(f"{output}.manifest.json").read_text())
    assert manifest["row_count"] == len(expected)
    assert manifest["shard_count"] == 3
    shard_counts = [
        len(pd.read_csv(shard_dir / f"judge_input_{idx:03d}.csv"))
        for idx in range(3)
    ]
    assert shard_counts == [1, 2, 2]
    assert "JUDGE COMPLETE" in result.stdout


def test_sharded_judge_waits_for_all_jobs_and_does_not_publish_on_one_failure(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "generations.csv"
    _judge_csv(csv_path)
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    _fake_judge(judge_dir / "judge_eval.py")
    output = tmp_path / "judge.json"
    output.write_text('{"stale": true}')
    shard_dir = tmp_path / "shards"
    environment = _judge_environment(csv_path, output, shard_dir, judge_dir)
    environment["FAKE_JUDGE_FAIL_ON"] = "question, 2"

    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_csv_llm_judge_sharded.sh")],
        env=environment,
        text=True,
        capture_output=True,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "JUDGE COMPLETE" not in combined
    assert not output.exists()


def test_sharded_judge_rejects_a_partial_success_json(tmp_path: Path) -> None:
    csv_path = tmp_path / "generations.csv"
    _judge_csv(csv_path)
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    _fake_judge(judge_dir / "judge_eval.py")
    output = tmp_path / "judge.json"
    shard_dir = tmp_path / "shards"
    environment = _judge_environment(csv_path, output, shard_dir, judge_dir)
    environment["FAKE_JUDGE_DROP_LAST"] = "1"

    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_csv_llm_judge_sharded.sh")],
        env=environment,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "JUDGE COMPLETE" not in result.stdout
    assert not output.exists()


def test_sharded_judge_rejects_out_of_range_example_and_verdict_scores(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "generations.csv"
    _judge_csv(csv_path)
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    _fake_judge(judge_dir / "judge_eval.py")

    for label, variable in (
        ("overall", "FAKE_JUDGE_OVERALL_SCORE"),
        ("verdict", "FAKE_JUDGE_VERDICT_SCORE"),
    ):
        output = tmp_path / f"judge-{label}.json"
        shard_dir = tmp_path / f"shards-{label}"
        environment = _judge_environment(csv_path, output, shard_dir, judge_dir)
        environment.update({"SHARDS": "1", "MAX_PARALLEL": "1", variable: "1.5"})

        result = subprocess.run(
            ["bash", str(SCRIPTS / "run_csv_llm_judge_sharded.sh")],
            env=environment,
            text=True,
            capture_output=True,
        )

        assert result.returncode != 0
        assert not output.exists()
        logs = "\n".join(
            path.read_text() for path in (shard_dir / "logs").glob("*.log")
        )
        assert "must be between 0 and 1" in logs


def test_sharded_judge_rejects_boolean_scores_on_every_surface(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "generations.csv"
    _judge_csv(csv_path)
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    _fake_judge(judge_dir / "judge_eval.py")

    for label, variable in (
        ("overall", "FAKE_JUDGE_BOOLEAN_OVERALL"),
        ("verdict", "FAKE_JUDGE_BOOLEAN_VERDICT"),
    ):
        output = tmp_path / f"judge-{label}.json"
        shard_dir = tmp_path / f"shards-{label}"
        environment = _judge_environment(csv_path, output, shard_dir, judge_dir)
        environment.update({"SHARDS": "1", "MAX_PARALLEL": "1", variable: "1"})
        result = subprocess.run(
            ["bash", str(SCRIPTS / "run_csv_llm_judge_sharded.sh")],
            env=environment,
            text=True,
            capture_output=True,
        )
        assert result.returncode != 0
        logs = "\n".join(
            path.read_text() for path in (shard_dir / "logs").glob("*.log")
        )
        assert "must be numeric, not boolean" in logs

    output = tmp_path / "judge-aggregate.json"
    environment = _judge_environment(
        csv_path, output, tmp_path / "shards-aggregate", judge_dir
    )
    environment.update(
        {
            "SHARDS": "1",
            "MAX_PARALLEL": "1",
            "FAKE_JUDGE_BOOLEAN_AGGREGATE": "1",
        }
    )
    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_csv_llm_judge_sharded.sh")],
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "must be numeric, not boolean" in result.stderr


def test_sharded_judge_rejects_out_of_range_final_aggregate(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "generations.csv"
    _judge_csv(csv_path)
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    _fake_judge(judge_dir / "judge_eval.py")
    output = tmp_path / "judge.json"
    environment = _judge_environment(
        csv_path, output, tmp_path / "shards", judge_dir
    )
    environment.update(
        {
            "SHARDS": "1",
            "MAX_PARALLEL": "1",
            "FAKE_JUDGE_OUT_OF_RANGE_AGGREGATE": "1",
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_csv_llm_judge_sharded.sh")],
        env=environment,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "merged judge overall_score must be between 0 and 1" in result.stderr
    assert not output.exists()


def test_sharded_judge_rejects_aggregates_contradicting_sealed_rows(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "generations.csv"
    _judge_csv(csv_path)
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    _fake_judge(judge_dir / "judge_eval.py")
    output = tmp_path / "judge.json"
    environment = _judge_environment(
        csv_path, output, tmp_path / "shards", judge_dir
    )
    environment.update(
        {
            "SHARDS": "1",
            "MAX_PARALLEL": "1",
            "FAKE_JUDGE_CONTRADICT_AGGREGATES": "1",
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_csv_llm_judge_sharded.sh")],
        env=environment,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "overall_score is inconsistent with per-example rows" in result.stderr
    assert not output.exists()


def test_sharded_judge_rejects_outputs_without_echoed_row_ids(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "generations.csv"
    _judge_csv(csv_path)
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    _fake_judge(judge_dir / "judge_eval.py")
    output = tmp_path / "judge.json"
    environment = _judge_environment(
        csv_path, output, tmp_path / "shards", judge_dir
    )
    environment["FAKE_JUDGE_OMIT_ROW_IDS"] = "1"

    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_csv_llm_judge_sharded.sh")],
        env=environment,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert not output.exists()
    logs = "\n".join(
        path.read_text() for path in (tmp_path / "shards" / "logs").glob("*.log")
    )
    assert "must echo workflow_row_id" in logs


def test_sharded_judge_rejects_same_category_row_permutations(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "generations.csv"
    frame = _judge_csv(csv_path, rows=5)
    frame["prompt_category"] = "rhythm"
    frame.to_csv(csv_path, index=False)
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    _fake_judge(judge_dir / "judge_eval.py")
    output = tmp_path / "judge.json"
    environment = _judge_environment(
        csv_path, output, tmp_path / "shards", judge_dir
    )
    environment["FAKE_JUDGE_REVERSE_EXAMPLES"] = "1"

    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_csv_llm_judge_sharded.sh")],
        env=environment,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert not output.exists()
    logs = "\n".join(
        path.read_text() for path in (tmp_path / "shards" / "logs").glob("*.log")
    )
    assert "workflow_row_id mismatch" in logs


def test_sharded_judge_requires_explicit_external_identity(tmp_path: Path) -> None:
    csv_path = tmp_path / "generations.csv"
    _judge_csv(csv_path)
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    _fake_judge(judge_dir / "judge_eval.py")
    output = tmp_path / "judge.json"
    environment = _judge_environment(
        csv_path, output, tmp_path / "shards", judge_dir
    )
    environment.pop("JUDGE_MODEL_ID")

    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_csv_llm_judge_sharded.sh")],
        env=environment,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "JUDGE_MODEL_ID must be an explicit nonempty immutable identity" in result.stderr
    assert not output.exists()


def test_documented_judge_command_declares_required_external_identity() -> None:
    documentation = (
        ROOT / "docs" / "tokenizer_aux_loss" / "NEXT_STEPS.md"
    ).read_text(encoding="utf-8")
    judge_command = documentation.split("# Judge eval", 1)[1].split(
        "# Full-test", 1
    )[0]

    for variable in (
        "JUDGE_MODEL_ID",
        "JUDGE_DEPLOYMENT_ID",
        "JUDGE_DECODING_ID",
    ):
        assert f"{variable}=" in judge_command


def test_sharded_judge_rejects_mutable_external_identity(tmp_path: Path) -> None:
    csv_path = tmp_path / "generations.csv"
    _judge_csv(csv_path)
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    _fake_judge(judge_dir / "judge_eval.py")
    output = tmp_path / "judge.json"
    environment = _judge_environment(
        csv_path, output, tmp_path / "shards", judge_dir
    )
    environment["JUDGE_MODEL_ID"] = "judge-model-latest"

    result = subprocess.run(
        ["bash", str(SCRIPTS / "run_csv_llm_judge_sharded.sh")],
        env=environment,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "JUDGE_MODEL_ID is mutable or unspecified" in result.stderr
    assert not output.exists()


def test_sharded_judge_identity_change_invalidates_cached_outputs(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "generations.csv"
    _judge_csv(csv_path)
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    _fake_judge(judge_dir / "judge_eval.py")
    output = tmp_path / "judge.json"
    environment = _judge_environment(
        csv_path, output, tmp_path / "shards", judge_dir
    )

    first = subprocess.run(
        ["bash", str(SCRIPTS / "run_csv_llm_judge_sharded.sh")],
        env=environment,
        text=True,
        capture_output=True,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    first_manifest = json.loads(Path(f"{output}.manifest.json").read_text())

    environment["JUDGE_DECODING_ID"] = "temperature-0-seed-456-v2"
    second = subprocess.run(
        ["bash", str(SCRIPTS / "run_csv_llm_judge_sharded.sh")],
        env=environment,
        text=True,
        capture_output=True,
    )

    assert second.returncode == 0, second.stdout + second.stderr
    assert "existing output is valid" not in second.stdout
    second_manifest = json.loads(Path(f"{output}.manifest.json").read_text())
    assert (
        first_manifest["judge_implementation"]["run_id"]
        != second_manifest["judge_implementation"]["run_id"]
    )
    assert "temperature-0-seed-456-v2" in second_manifest[
        "judge_implementation"
    ]["run_id"]


def test_judge_output_lock_serializes_two_complete_producer_lifecycles(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "generations.csv"
    _judge_csv(csv_path)
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    _fake_judge(judge_dir / "judge_eval.py")
    output = tmp_path / "judge.json"
    shard_dir = tmp_path / "shards"
    first_marker = tmp_path / "first-judge-started"
    second_marker = tmp_path / "second-judge-started"
    first_environment = _judge_environment(csv_path, output, shard_dir, judge_dir)
    first_environment.update(
        {
            "SHARDS": "1",
            "MAX_PARALLEL": "1",
            "FAKE_JUDGE_MARKER": str(first_marker),
            "FAKE_JUDGE_DELAY_SECONDS": "1",
        }
    )
    second_environment = _judge_environment(csv_path, output, shard_dir, judge_dir)
    second_environment.update(
        {
            "SHARDS": "1",
            "MAX_PARALLEL": "1",
            "JUDGE_DECODING_ID": "temperature-0-seed-999-v2",
            "FAKE_JUDGE_MARKER": str(second_marker),
        }
    )
    command = ["bash", str(SCRIPTS / "run_csv_llm_judge_sharded.sh")]

    first = subprocess.Popen(
        command,
        env=first_environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _wait_for_path(first_marker)
    second = subprocess.Popen(
        command,
        env=second_environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(0.2)
    assert second.poll() is None
    assert not second_marker.exists()

    first_stdout, first_stderr = first.communicate(timeout=15)
    second_stdout, second_stderr = second.communicate(timeout=15)
    assert first.returncode == 0, first_stdout + first_stderr
    assert second.returncode == 0, second_stdout + second_stderr
    _wait_for_path(second_marker)

    validation = subprocess.run(
        command,
        env=second_environment,
        text=True,
        capture_output=True,
    )
    assert validation.returncode == 0, validation.stdout + validation.stderr
    assert "existing output is valid for current inputs" in validation.stdout


def test_judge_manifest_invalidates_when_an_imported_judge_module_changes(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "generations.csv"
    _judge_csv(csv_path)
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    _fake_judge(judge_dir / "judge_eval.py")
    output = tmp_path / "judge.json"
    shard_dir = tmp_path / "shards"
    environment = _judge_environment(csv_path, output, shard_dir, judge_dir)

    first_run = subprocess.run(
        ["bash", str(SCRIPTS / "run_csv_llm_judge_sharded.sh")],
        env=environment,
        text=True,
        capture_output=True,
    )
    assert first_run.returncode == 0, first_run.stdout + first_run.stderr
    first = json.loads(output.read_text())
    assert first["aggregates"]["overall_score"] == 1.0

    (judge_dir / "judge_support.py").write_text("SCORE = 0.25\n")
    second_run = subprocess.run(
        ["bash", str(SCRIPTS / "run_csv_llm_judge_sharded.sh")],
        env=environment,
        text=True,
        capture_output=True,
    )

    assert second_run.returncode == 0, second_run.stdout + second_run.stderr
    second = json.loads(output.read_text())
    assert second["aggregates"]["overall_score"] == 0.25
    manifest = json.loads(Path(f"{output}.manifest.json").read_text())
    assert "judge-model-snapshot-2026-08-25" in manifest["judge_implementation"][
        "run_id"
    ]


def test_judge_manifest_tracks_prompt_assets_but_excludes_secrets_and_outputs(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "generations.csv"
    _judge_csv(csv_path)
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    _fake_judge(judge_dir / "judge_eval.py")
    rubric = judge_dir / "rubric.json"
    rubric.write_text('{"version": 1}')
    output = tmp_path / "judge.json"
    environment = _judge_environment(
        csv_path, output, tmp_path / "shards", judge_dir
    )

    first = subprocess.run(
        ["bash", str(SCRIPTS / "run_csv_llm_judge_sharded.sh")],
        env=environment,
        text=True,
        capture_output=True,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    first_manifest = json.loads(Path(f"{output}.manifest.json").read_text())

    rubric.write_text('{"version": 2}')
    second = subprocess.run(
        ["bash", str(SCRIPTS / "run_csv_llm_judge_sharded.sh")],
        env=environment,
        text=True,
        capture_output=True,
    )
    assert second.returncode == 0, second.stdout + second.stderr
    assert "existing output is valid" not in second.stdout
    second_manifest = json.loads(Path(f"{output}.manifest.json").read_text())
    assert (
        first_manifest["judge_implementation"]["sha256"]
        != second_manifest["judge_implementation"]["sha256"]
    )

    (judge_dir / "credentials.json").write_text('{"dummy": "changed"}')
    output_cache = judge_dir / "outputs"
    output_cache.mkdir()
    (output_cache / "previous.json").write_text('{"score": 0}')
    third = subprocess.run(
        ["bash", str(SCRIPTS / "run_csv_llm_judge_sharded.sh")],
        env=environment,
        text=True,
        capture_output=True,
    )
    assert third.returncode == 0, third.stdout + third.stderr
    assert "existing output is valid for current inputs" in third.stdout


def _timeaxis_environment(
    tmp_path: Path,
    checkpoint: Path,
    subset: Path,
    runner: Path,
    judge_dir: Path,
) -> dict[str, str]:
    implementation_root = tmp_path / "generation-implementation"
    implementation_root.mkdir(exist_ok=True)
    implementation_root.joinpath("identity.py").write_text(
        "GENERATION_IMPLEMENTATION = 'test-v1'\n"
    )
    return {
        **os.environ,
        "ECG_REPO_ROOT": str(ROOT),
        "ECG_PYTHON": sys.executable,
        "TIMEAXIS_CKPT": str(checkpoint),
        "TIMEAXIS_SUBSET": str(subset),
        "TIMEAXIS_GENERATION_CSV": str(tmp_path / "timeaxis_generations.csv"),
        "TIMEAXIS_JUDGE_JSON": str(tmp_path / "timeaxis_judge.json"),
        "TIMEAXIS_JUDGE_SHARD_DIR": str(tmp_path / "timeaxis_judge_shards"),
        "ECG_LOG_DIR": str(tmp_path / "logs"),
        "ECG_EVAL_RUNNER": str(runner),
        "ECG_GENERATION_IMPLEMENTATION_ROOTS": str(implementation_root),
        "JUDGE_DIR": str(judge_dir),
        "JUDGE_SHARDS": "2",
        "JUDGE_MAX_PARALLEL": "2",
        "POST_TRAIN_WAIT_SECONDS": "0",
        "JUDGE_MODEL_ID": "judge-model-snapshot-2026-08-25",
        "JUDGE_DEPLOYMENT_ID": "judge-deployment-revision-17",
        "JUDGE_DECODING_ID": "temperature-0-seed-123-v1",
    }


def test_timeaxis_chain_rejects_an_unchanged_preexisting_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best_model.pt"
    checkpoint.write_bytes(b"stale-checkpoint")
    subset = tmp_path / "subset.parquet"
    _subset(subset)
    runner = tmp_path / "runner.py"
    _fake_generation_runner(runner)
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    _fake_judge(judge_dir / "judge_eval.py")
    environment = _timeaxis_environment(tmp_path, checkpoint, subset, runner, judge_dir)

    result = subprocess.run(
        [
            "bash",
            str(SCRIPTS / "run_timeaxis_eval_chain.sh"),
            "--",
            sys.executable,
            "-c",
            (
                "import os,pathlib; "
                f"source=pathlib.Path({str(checkpoint)!r}); "
                "pathlib.Path(os.environ['TIMEAXIS_OUTPUT_CHECKPOINT'])"
                ".write_bytes(source.read_bytes())"
            ),
        ],
        env=environment,
        text=True,
        capture_output=True,
        timeout=10,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "checkpoint did not change" in combined
    assert "JUDGE COMPLETE" not in combined
    assert not Path(environment["TIMEAXIS_GENERATION_CSV"]).exists()


def test_timeaxis_chain_accepts_only_a_checkpoint_changed_by_its_training_command(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best_model.pt"
    checkpoint.write_bytes(b"old-checkpoint")
    subset = tmp_path / "subset.parquet"
    _subset(subset)
    runner = tmp_path / "runner.py"
    _fake_generation_runner(runner)
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    _fake_judge(judge_dir / "judge_eval.py")
    environment = _timeaxis_environment(tmp_path, checkpoint, subset, runner, judge_dir)
    training_code = (
        "import os,pathlib; "
        "pathlib.Path(os.environ['TIMEAXIS_OUTPUT_CHECKPOINT'])"
        ".write_bytes(b'new-checkpoint')"
    )

    result = subprocess.run(
        [
            "bash",
            str(SCRIPTS / "run_timeaxis_eval_chain.sh"),
            "--",
            sys.executable,
            "-c",
            training_code,
        ],
        env=environment,
        text=True,
        capture_output=True,
        timeout=40,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert Path(environment["TIMEAXIS_GENERATION_CSV"]).is_file()
    assert Path(environment["TIMEAXIS_JUDGE_JSON"]).is_file()
    assert "JUDGE COMPLETE" in result.stdout


def test_timeaxis_chain_propagates_failed_training_even_after_checkpoint_write(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best_model.pt"
    checkpoint.write_bytes(b"old-checkpoint")
    subset = tmp_path / "subset.parquet"
    _subset(subset)
    runner = tmp_path / "runner.py"
    _fake_generation_runner(runner)
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    _fake_judge(judge_dir / "judge_eval.py")
    environment = _timeaxis_environment(tmp_path, checkpoint, subset, runner, judge_dir)
    training_code = (
        "import os,pathlib; "
        "pathlib.Path(os.environ['TIMEAXIS_OUTPUT_CHECKPOINT'])"
        ".write_bytes(b'failed-checkpoint'); "
        "raise SystemExit(23)"
    )

    result = subprocess.run(
        [
            "bash",
            str(SCRIPTS / "run_timeaxis_eval_chain.sh"),
            "--",
            sys.executable,
            "-c",
            training_code,
        ],
        env=environment,
        text=True,
        capture_output=True,
        timeout=5,
    )

    assert result.returncode == 23
    assert "training command failed with status 23" in result.stderr
    assert "JUDGE COMPLETE" not in result.stdout
    assert not Path(environment["TIMEAXIS_GENERATION_CSV"]).exists()


def test_timeaxis_checkpoint_lock_does_not_attribute_a_concurrent_failed_write(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best_model.pt"
    checkpoint.write_bytes(b"old-checkpoint")
    subset = tmp_path / "subset.parquet"
    _subset(subset)
    runner = tmp_path / "runner.py"
    _fake_generation_runner(runner)
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    _fake_judge(judge_dir / "judge_eval.py")
    environment = _timeaxis_environment(tmp_path, checkpoint, subset, runner, judge_dir)
    first_marker = tmp_path / "first-training-started"
    second_marker = tmp_path / "second-training-started"
    first_code = (
        "import os,pathlib,time; "
        f"pathlib.Path({str(first_marker)!r}).write_text('started'); "
        "time.sleep(0.6); "
        "pathlib.Path(os.environ['TIMEAXIS_OUTPUT_CHECKPOINT'])"
        ".write_bytes(b'failed-concurrent-write'); "
        "raise SystemExit(23)"
    )
    second_code = (
        "import os,pathlib,time; "
        f"pathlib.Path({str(second_marker)!r}).write_text('started'); "
        "time.sleep(1.0); "
        "pathlib.Path(os.environ['TIMEAXIS_OUTPUT_CHECKPOINT'])"
        ".write_bytes(b'old-checkpoint')"
    )

    def chain_command(code: str) -> list[str]:
        return [
            "bash",
            str(SCRIPTS / "run_timeaxis_eval_chain.sh"),
            "--",
            sys.executable,
            "-c",
            code,
        ]

    first = subprocess.Popen(
        chain_command(first_code),
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _wait_for_path(first_marker)
    second = subprocess.Popen(
        chain_command(second_code),
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(0.2)
    assert second.poll() is None
    assert not second_marker.exists()

    first_stdout, first_stderr = first.communicate(timeout=5)
    second_stdout, second_stderr = second.communicate(timeout=5)
    assert first.returncode == 23, first_stdout + first_stderr
    assert second.returncode != 0
    assert "checkpoint did not change during the training command" in (
        second_stdout + second_stderr
    )
    assert not Path(environment["TIMEAXIS_GENERATION_CSV"]).exists()


def test_timeaxis_isolated_checkpoint_rejects_noncooperative_external_writer(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best_model.pt"
    checkpoint.write_bytes(b"old-checkpoint")
    subset = tmp_path / "subset.parquet"
    _subset(subset)
    runner = tmp_path / "runner.py"
    _fake_generation_runner(runner)
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    _fake_judge(judge_dir / "judge_eval.py")
    environment = _timeaxis_environment(tmp_path, checkpoint, subset, runner, judge_dir)
    marker = tmp_path / "training-started"
    training_code = (
        "import os,pathlib,time; "
        f"pathlib.Path({str(marker)!r}).write_text('started'); "
        "time.sleep(0.8); "
        "pathlib.Path(os.environ['TIMEAXIS_OUTPUT_CHECKPOINT'])"
        ".write_bytes(b'old-checkpoint')"
    )
    command = [
        "bash",
        str(SCRIPTS / "run_timeaxis_eval_chain.sh"),
        "--",
        sys.executable,
        "-c",
        training_code,
    ]
    workflow = subprocess.Popen(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _wait_for_path(marker)
    checkpoint.write_bytes(b"noncooperative-external-write")

    stdout, stderr = workflow.communicate(timeout=5)

    assert workflow.returncode != 0
    assert "checkpoint did not change during the training command" in stdout + stderr
    assert checkpoint.read_bytes() == b"old-checkpoint"
    assert not Path(environment["TIMEAXIS_GENERATION_CSV"]).exists()


def test_timeaxis_shared_outputs_serialize_different_checkpoint_runs(
    tmp_path: Path,
) -> None:
    checkpoint_one = tmp_path / "checkpoint-one.pt"
    checkpoint_two = tmp_path / "checkpoint-two.pt"
    checkpoint_one.write_bytes(b"old-one")
    checkpoint_two.write_bytes(b"old-two")
    subset = tmp_path / "subset.parquet"
    _subset(subset)
    runner = tmp_path / "runner.py"
    _fake_generation_runner(runner)
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    _fake_judge(judge_dir / "judge_eval.py")
    base_environment = _timeaxis_environment(
        tmp_path, checkpoint_one, subset, runner, judge_dir
    )
    base_environment.update({"JUDGE_SHARDS": "1", "JUDGE_MAX_PARALLEL": "1"})
    first_judge_marker = tmp_path / "first-judge-started"
    second_training_marker = tmp_path / "second-training-started"

    def command(payload: str, marker: Path | None = None) -> list[str]:
        marker_statement = (
            ""
            if marker is None
            else f"pathlib.Path({str(marker)!r}).write_text('started'); "
        )
        code = (
            "import os,pathlib; "
            + marker_statement
            + "pathlib.Path(os.environ['TIMEAXIS_OUTPUT_CHECKPOINT'])"
            + f".write_bytes({payload.encode()!r})"
        )
        return [
            "bash",
            str(SCRIPTS / "run_timeaxis_eval_chain.sh"),
            "--",
            sys.executable,
            "-c",
            code,
        ]

    first = subprocess.Popen(
        command("new-one"),
        env={
            **base_environment,
            "FAKE_JUDGE_MARKER": str(first_judge_marker),
            "FAKE_JUDGE_DELAY_SECONDS": "1",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _wait_for_path(first_judge_marker, timeout=40)
    second = subprocess.Popen(
        command("new-two", second_training_marker),
        env={**base_environment, "TIMEAXIS_CKPT": str(checkpoint_two)},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(0.2)
    assert second.poll() is None
    assert not second_training_marker.exists()

    first_stdout, first_stderr = first.communicate(timeout=40)
    second_stdout, second_stderr = second.communicate(timeout=40)
    assert first.returncode == 0, first_stdout + first_stderr
    assert second.returncode == 0, second_stdout + second_stderr
    _wait_for_path(second_training_marker)

    generation = Path(base_environment["TIMEAXIS_GENERATION_CSV"])
    judge = Path(base_environment["TIMEAXIS_JUDGE_JSON"])
    generated = pd.read_csv(generation)
    assert generated["checkpoint_path"].tolist() == [str(checkpoint_two.resolve())] * len(
        generated
    )
    generation_manifest = json.loads(Path(f"{generation}.manifest.json").read_text())
    assert generation_manifest["checkpoint"] == _artifact_identity(checkpoint_two)
    judge_manifest = json.loads(Path(f"{judge}.manifest.json").read_text())
    assert judge_manifest["input"] == _artifact_identity(generation)
