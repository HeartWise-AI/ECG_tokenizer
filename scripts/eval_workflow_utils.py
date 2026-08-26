#!/usr/bin/env python3
"""Validation and atomic publication helpers for evaluation workflows."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


GENERATION_SCHEMA_VERSION = 1
GENERATION_REQUIRED_COLUMNS = {
    "row_idx",
    "source_row_idx",
    "waveform_name",
    "waveform_path",
    "question",
    "generation",
    "ground_truth",
    "prompt_category",
    "checkpoint_path",
    "run_fingerprint",
}
JUDGE_CSV_REQUIRED_COLUMNS = {
    "generation",
    "ground_truth",
    "prompt_category",
}
JUDGE_SCHEMA_VERSION = 1


class ArtifactValidationError(ValueError):
    """An artifact does not implement the workflow contract."""


@contextmanager
def exclusive_output_lock(output: Path) -> Iterator[None]:
    """Hold one process lock for an output's complete workflow lifecycle."""
    resolved = output.expanduser().resolve()
    lock_path = Path(f"{resolved}.workflow.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_with_output_lock(
    output: Path,
    command: Sequence[str],
    environment_marker: str,
) -> int:
    argv = list(command)
    if argv and argv[0] == "--":
        argv.pop(0)
    if not argv:
        raise ArtifactValidationError("locked workflow command must be nonempty")
    if not environment_marker or "=" in environment_marker:
        raise ArtifactValidationError("lock environment marker name is invalid")
    environment = os.environ.copy()
    environment[environment_marker] = str(output)
    try:
        with exclusive_output_lock(output):
            return subprocess.run(argv, env=environment, check=False).returncode
    except OSError as exc:
        raise ArtifactValidationError(
            f"could not execute locked workflow command {argv[0]}: {exc}"
        ) from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_content_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ArtifactValidationError(f"required file is missing: {resolved}")
    before = resolved.stat()
    sha256 = sha256_file(resolved)
    after = resolved.stat()
    before_signature = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_signature = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_signature != after_signature:
        raise ArtifactValidationError(
            f"required file changed while it was being fingerprinted: {resolved}"
        )
    return {
        "size": int(after.st_size),
        "sha256": sha256,
    }


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {"path": str(resolved), **file_content_identity(resolved)}


def python_implementation_identity(
    roots: Iterable[Path],
    runtime_packages: Iterable[str] = (),
    include_suffixes: Iterable[str] = (".py",),
    excluded_directories: Iterable[str] = (),
    excluded_name_fragments: Iterable[str] = (),
) -> dict[str, Any]:
    """Hash selected implementation files and package versions without git state."""
    resolved_roots = [path.expanduser().resolve() for path in roots]
    suffixes = {suffix.lower() for suffix in include_suffixes}
    excluded_dirs = {
        ".git",
        ".venv",
        "__pycache__",
        *(value.lower() for value in excluded_directories),
    }
    excluded_fragments = {
        fragment.lower() for fragment in excluded_name_fragments if fragment
    }
    records: list[tuple[str, Path | None]] = []
    seen_files: set[Path] = set()
    for root_index, root in enumerate(resolved_roots):
        if root.is_file():
            candidates = [root] if root.suffix.lower() in suffixes else []
        elif root.is_dir():
            candidates = sorted(
                candidate
                for candidate in root.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in suffixes
            )
        else:
            records.append((f"root-{root_index}:MISSING", None))
            continue
        for candidate in candidates:
            relative_path = Path(candidate.name) if root.is_file() else candidate.relative_to(root)
            if any(part.lower() in excluded_dirs for part in relative_path.parts):
                continue
            lowered_name = candidate.name.lower()
            if any(fragment in lowered_name for fragment in excluded_fragments):
                continue
            resolved = candidate.resolve()
            if resolved in seen_files:
                continue
            seen_files.add(resolved)
            relative = str(relative_path)
            records.append((f"root-{root_index}:{relative}", resolved))

    digest = hashlib.sha256()
    for label, path in sorted(records, key=lambda item: item[0]):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        if path is None:
            digest.update(b"MISSING")
        else:
            digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")

    runtime: dict[str, str] = {"python": sys.version.split()[0]}
    for package in sorted(set(runtime_packages)):
        try:
            runtime[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            runtime[package] = "MISSING"
    for key, value in sorted(runtime.items()):
        digest.update(key.encode("utf-8"))
        digest.update(b"=")
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return {
        "roots": [str(path) for path in resolved_roots],
        "file_count": len(seen_files),
        "python_file_count": sum(path.suffix == ".py" for path in seen_files),
        "runtime": runtime,
        "sha256": digest.hexdigest(),
    }


def generation_manifest_path(output_csv: Path) -> Path:
    return Path(f"{output_csv}.manifest.json")


def _string(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _exact_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ArtifactValidationError(f"{label} must be an integer")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"{label} must be an integer") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ArtifactValidationError(f"{label} must be an integer")
    integer = int(number)
    if _string(value).strip() not in {str(integer), f"{integer}.0"}:
        raise ArtifactValidationError(f"{label} must be an integer")
    return integer


def _expected_subset(subset: pd.DataFrame) -> pd.DataFrame:
    required = {"prompt", "prompt_category"}
    missing = required - set(subset.columns)
    if missing:
        raise ArtifactValidationError(
            f"subset is missing required columns: {sorted(missing)}"
        )
    waveform_col = next(
        (name for name in ("waveform_path_psa", "waveform_path") if name in subset.columns),
        None,
    )
    if waveform_col is None:
        raise ArtifactValidationError(
            "subset is missing waveform_path_psa or waveform_path"
        )
    truth_col = next(
        (name for name in ("generated_answer", "report", "ground_truth") if name in subset.columns),
        None,
    )
    if truth_col is None:
        raise ArtifactValidationError(
            "subset is missing generated_answer, report, or ground_truth"
        )

    source_indices: Sequence[Any]
    if "source_row_idx" in subset.columns:
        source_indices = subset["source_row_idx"].tolist()
    else:
        source_indices = list(range(len(subset)))
    expected = pd.DataFrame(
        {
            "row_idx": list(range(len(subset))),
            "source_row_idx": [
                _exact_integer(value, "subset source_row_idx")
                for value in source_indices
            ],
            "waveform_name": [Path(_string(value)).stem for value in subset[waveform_col]],
            "waveform_path": [_string(value) for value in subset[waveform_col]],
            "question": [_string(value) for value in subset["prompt"]],
            "ground_truth": [_string(value) for value in subset[truth_col]],
            "prompt_category": [_string(value) for value in subset["prompt_category"]],
        }
    )
    if expected["source_row_idx"].duplicated().any():
        duplicate = int(
            expected.loc[expected["source_row_idx"].duplicated(), "source_row_idx"].iloc[0]
        )
        raise ArtifactValidationError(
            f"subset contains duplicate source_row_idx: {duplicate}"
        )
    return expected


def validate_generation_frame(
    generated: pd.DataFrame,
    subset: pd.DataFrame,
    checkpoint: Path,
) -> str:
    missing = GENERATION_REQUIRED_COLUMNS - set(generated.columns)
    if missing:
        raise ArtifactValidationError(
            f"generation CSV is missing required columns: {sorted(missing)}"
        )
    expected = _expected_subset(subset.reset_index(drop=True))
    if len(generated) != len(expected):
        raise ArtifactValidationError(
            f"generation row count mismatch: got {len(generated)}, expected {len(expected)}"
        )
    if generated["source_row_idx"].duplicated().any():
        duplicate = generated.loc[
            generated["source_row_idx"].duplicated(), "source_row_idx"
        ].iloc[0]
        raise ArtifactValidationError(
            f"generation contains duplicate source_row_idx: {duplicate}"
        )

    run_fingerprints = [
        _string(value).strip() for value in generated["run_fingerprint"]
    ]
    if len(set(run_fingerprints)) != 1:
        raise ArtifactValidationError(
            "generation rows must share one exact run_fingerprint"
        )
    run_fingerprint = run_fingerprints[0]
    if (
        len(run_fingerprint) != 64
        or run_fingerprint != run_fingerprint.lower()
        or any(character not in "0123456789abcdef" for character in run_fingerprint)
    ):
        raise ArtifactValidationError(
            "generation run_fingerprint must be one lowercase SHA-256 digest"
        )

    numeric_columns = ("row_idx", "source_row_idx")
    for column in numeric_columns:
        actual = [
            _exact_integer(value, f"generation column {column}")
            for value in generated[column].tolist()
        ]
        wanted = expected[column].tolist()
        if actual != wanted:
            raise ArtifactValidationError(
                f"generation {column} identity does not match the subset"
            )

    for column in (
        "waveform_name",
        "waveform_path",
        "question",
        "ground_truth",
        "prompt_category",
    ):
        actual = [_string(value) for value in generated[column]]
        wanted = expected[column].tolist()
        if actual != wanted:
            first = next(idx for idx, pair in enumerate(zip(actual, wanted)) if pair[0] != pair[1])
            raise ArtifactValidationError(
                f"generation {column} mismatch at row {first}: "
                f"got {actual[first]!r}, expected {wanted[first]!r}"
            )

    generations = [_string(value).strip() for value in generated["generation"]]
    for idx, value in enumerate(generations):
        if not value:
            raise ArtifactValidationError(f"empty generation at row {idx}")
        if value.upper().startswith("[ERROR"):
            raise ArtifactValidationError(
                f"generation runner encoded an error at row {idx}: {value[:160]}"
            )

    checkpoint_path = str(checkpoint.expanduser().resolve())
    actual_checkpoints = [
        str(Path(_string(value)).expanduser().resolve())
        for value in generated["checkpoint_path"]
    ]
    if any(value != checkpoint_path for value in actual_checkpoints):
        raise ArtifactValidationError(
            "generation checkpoint_path does not match the requested checkpoint"
        )
    return run_fingerprint


def _atomic_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_generation(
    runner_csv: Path,
    output_csv: Path,
    subset_path: Path,
    checkpoint_path: Path,
    config: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any] | None = None,
    subset_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    runner_identity = file_identity(runner_csv)
    expected_checkpoint = dict(
        checkpoint_identity or file_identity(checkpoint_path)
    )
    expected_subset = dict(subset_identity or file_identity(subset_path))
    if file_identity(checkpoint_path) != expected_checkpoint:
        raise ArtifactValidationError("checkpoint identity changed before publication")
    if file_identity(subset_path) != expected_subset:
        raise ArtifactValidationError("subset identity changed before publication")
    try:
        generated = pd.read_csv(runner_csv)
        subset = pd.read_parquet(subset_path)
    except Exception as exc:
        raise ArtifactValidationError(f"could not read generation inputs: {exc}") from exc
    run_fingerprint = validate_generation_frame(generated, subset, checkpoint_path)
    if file_identity(runner_csv) != runner_identity:
        raise ArtifactValidationError("runner generation changed while it was being read")
    if file_identity(checkpoint_path) != expected_checkpoint:
        raise ArtifactValidationError("checkpoint changed during generation publication")
    if file_identity(subset_path) != expected_subset:
        raise ArtifactValidationError("subset changed during generation publication")

    _atomic_csv(output_csv, generated)
    output_identity = file_identity(output_csv)
    if file_identity(runner_csv) != runner_identity:
        raise ArtifactValidationError("runner generation changed before publication")
    if file_identity(checkpoint_path) != expected_checkpoint:
        raise ArtifactValidationError("checkpoint changed before generation publication")
    if file_identity(subset_path) != expected_subset:
        raise ArtifactValidationError("subset changed before generation publication")
    manifest = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "kind": "ecg_generation_csv",
        "row_count": int(len(generated)),
        "checkpoint": expected_checkpoint,
        "subset": expected_subset,
        "config": dict(config),
        "run_fingerprint": run_fingerprint,
        "output": output_identity,
    }
    _atomic_json(generation_manifest_path(output_csv), manifest)
    return manifest


def validate_published_generation(
    output_csv: Path,
    subset_path: Path,
    checkpoint_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = generation_manifest_path(output_csv)
    manifest_identity = file_identity(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        raise ArtifactValidationError(
            f"generation manifest is missing or invalid: {manifest_path}: {exc}"
        ) from exc
    if manifest.get("schema_version") != GENERATION_SCHEMA_VERSION:
        raise ArtifactValidationError("generation manifest schema version is unsupported")
    if manifest.get("kind") != "ecg_generation_csv":
        raise ArtifactValidationError("generation manifest kind is invalid")
    expected_checkpoint = file_identity(checkpoint_path)
    expected_subset = file_identity(subset_path)
    expected_output = file_identity(output_csv)
    if manifest.get("checkpoint") != expected_checkpoint:
        raise ArtifactValidationError("generation checkpoint identity is stale")
    if manifest.get("subset") != expected_subset:
        raise ArtifactValidationError("generation subset identity is stale")
    if manifest.get("config") != dict(config):
        raise ArtifactValidationError("generation configuration is stale")
    if manifest.get("output") != expected_output:
        raise ArtifactValidationError("generation output hash does not match its manifest")

    try:
        generated = pd.read_csv(output_csv)
        subset = pd.read_parquet(subset_path)
    except Exception as exc:
        raise ArtifactValidationError(f"could not read generation artifact: {exc}") from exc
    run_fingerprint = validate_generation_frame(generated, subset, checkpoint_path)
    if manifest.get("row_count") != len(generated):
        raise ArtifactValidationError("generation manifest row count is stale")
    if manifest.get("run_fingerprint") != run_fingerprint:
        raise ArtifactValidationError("generation run fingerprint is stale")
    current_identities = (
        file_identity(manifest_path),
        file_identity(checkpoint_path),
        file_identity(subset_path),
        file_identity(output_csv),
    )
    if current_identities != (
        manifest_identity,
        expected_checkpoint,
        expected_subset,
        expected_output,
    ):
        raise ArtifactValidationError("generation artifact changed during validation")
    return manifest


def validate_judge_input(frame: pd.DataFrame) -> None:
    missing = JUDGE_CSV_REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ArtifactValidationError(
            f"judge CSV is missing required columns: {sorted(missing)}"
        )
    if frame.empty:
        raise ArtifactValidationError("judge CSV has no rows")
    for column in ("generation", "ground_truth", "prompt_category"):
        values = [_string(value).strip() for value in frame[column]]
        for idx, value in enumerate(values):
            if not value:
                raise ArtifactValidationError(
                    f"judge CSV has an empty {column} at row {idx}"
                )
            if column == "generation" and value.upper().startswith("[ERROR"):
                raise ArtifactValidationError(
                    f"judge CSV has an invalid generation at row {idx}"
                )


def judge_manifest_path(output_json: Path) -> Path:
    return Path(f"{output_json}.manifest.json")


def judge_implementation_identity(
    judge_dir: Path,
    judge_script: Path,
    run_id: str,
) -> dict[str, Any]:
    normalized_run_id = run_id.strip()
    if not normalized_run_id:
        raise ArtifactValidationError("judge run identity must be nonempty")
    identity = python_implementation_identity(
        [judge_dir],
        runtime_packages=("anthropic", "openai", "pandas"),
        include_suffixes=(
            ".j2",
            ".jinja",
            ".json",
            ".md",
            ".py",
            ".toml",
            ".txt",
            ".yaml",
            ".yml",
        ),
        excluded_directories=(
            ".cache",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "cache",
            "logs",
            "output",
            "outputs",
            "results",
        ),
        excluded_name_fragments=(
            ".env",
            "api_key",
            "apikey",
            "credential",
            "private_key",
            "secret",
        ),
    )
    identity["entrypoint"] = file_identity(judge_script)
    identity["run_id"] = normalized_run_id
    return identity


def split_judge_csv(
    input_csv: Path,
    out_dir: Path,
    requested_shards: int,
) -> dict[str, Any]:
    if requested_shards < 1:
        raise ArtifactValidationError("judge shard count must be positive")
    input_identity = file_identity(input_csv)
    try:
        frame = pd.read_csv(input_csv)
    except Exception as exc:
        raise ArtifactValidationError(f"could not read judge CSV {input_csv}: {exc}") from exc
    validate_judge_input(frame)
    row_count = len(frame)
    shard_count = min(requested_shards, row_count)
    frame = frame.copy()
    frame["workflow_row_id"] = list(range(row_count))
    out_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    written_shards: list[pd.DataFrame] = []
    for shard_index in range(shard_count):
        start = shard_index * row_count // shard_count
        end = (shard_index + 1) * row_count // shard_count
        shard = frame.iloc[start:end].reset_index(drop=True)
        shard_path = (out_dir / f"judge_input_{shard_index:03d}.csv").resolve()
        judge_path = (out_dir / f"judge_output_{shard_index:03d}.json").resolve()
        _atomic_csv(shard_path, shard)
        identity = file_identity(shard_path)
        written_shards.append(pd.read_csv(shard_path))
        items.append(
            {
                "shard_index": shard_index,
                "start": start,
                "end": end,
                "row_count": int(len(shard)),
                "input": identity,
                "judge_output": str(judge_path),
            }
        )
    if file_identity(input_csv) != input_identity:
        raise ArtifactValidationError("judge CSV changed while shards were being written")
    reconstructed = pd.concat(written_shards, ignore_index=True)
    if not reconstructed.equals(frame.reset_index(drop=True)):
        raise ArtifactValidationError(
            "judge shard contents do not reconstruct the exact ordered input"
        )
    for item in items:
        if item["input"] != file_identity(Path(item["input"]["path"])):
            raise ArtifactValidationError("judge shard changed before split publication")
    manifest = {
        "schema_version": JUDGE_SCHEMA_VERSION,
        "kind": "judge_csv_split",
        "input": input_identity,
        "ordered_columns": frame.columns.tolist(),
        "row_count": row_count,
        "requested_shards": requested_shards,
        "shard_count": shard_count,
        "items": items,
    }
    _atomic_json(out_dir / "manifest.json", manifest)
    return manifest


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        raise ArtifactValidationError(f"{label} is missing or invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"{label} must be a JSON object: {path}")
    return payload


def _validate_judge_payload(
    payload: Mapping[str, Any],
    shard: pd.DataFrame,
) -> list[dict[str, Any]]:
    examples = payload.get("per_example_verdicts")
    if not isinstance(examples, list):
        raise ArtifactValidationError("judge JSON is missing per_example_verdicts")
    if len(examples) != len(shard):
        raise ArtifactValidationError(
            f"judge example count mismatch: got {len(examples)}, expected {len(shard)}"
        )
    normalized: list[dict[str, Any]] = []
    for offset, (example, (_, input_row)) in enumerate(zip(examples, shard.iterrows())):
        if not isinstance(example, dict):
            raise ArtifactValidationError(f"judge example {offset} must be an object")
        category = example.get("prompt_category")
        expected_category = _string(input_row["prompt_category"])
        if not isinstance(category, str) or category != expected_category:
            raise ArtifactValidationError(
                f"judge prompt_category mismatch at shard row {offset}"
            )
        normalized_score(
            example.get("overall_score"), f"judge example {offset} overall_score"
        )
        verdicts = example.get("verdicts")
        if not isinstance(verdicts, dict) or not verdicts:
            raise ArtifactValidationError(
                f"judge example {offset} must contain nonempty verdicts"
            )
        for judge_name, verdict in verdicts.items():
            if not isinstance(verdict, dict):
                raise ArtifactValidationError(
                    f"judge example {offset} verdict {judge_name} must be an object"
                )
            normalized_score(
                verdict.get("score"),
                f"judge example {offset} verdict {judge_name} score",
            )
        row_id = _exact_integer(
            input_row["workflow_row_id"],
            f"judge input workflow_row_id at shard row {offset}",
        )
        if "workflow_row_id" not in example:
            raise ArtifactValidationError(
                f"judge example {offset} must echo workflow_row_id"
            )
        present_row_id = _exact_integer(
            example["workflow_row_id"],
            f"judge example {offset} workflow_row_id",
        )
        if present_row_id != row_id:
            raise ArtifactValidationError(
                f"judge workflow_row_id mismatch at shard row {offset}"
            )
        copied = dict(example)
        copied["workflow_row_id"] = present_row_id
        normalized.append(copied)
    return normalized


def seal_judge_shard(
    source_json: Path,
    shard_csv: Path,
    output_json: Path,
    judge_dir: Path,
    judge_script: Path,
    run_id: str,
    expected_judge_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    shard_identity = file_identity(shard_csv)
    source_identity = file_identity(source_json)
    judge_identity = judge_implementation_identity(judge_dir, judge_script, run_id)
    if (
        expected_judge_identity is not None
        and dict(expected_judge_identity) != judge_identity
    ):
        raise ArtifactValidationError(
            "judge implementation changed before shard sealing"
        )
    try:
        shard = pd.read_csv(shard_csv)
    except Exception as exc:
        raise ArtifactValidationError(f"could not read judge shard {shard_csv}: {exc}") from exc
    validate_judge_input(shard)
    if "workflow_row_id" not in shard.columns:
        raise ArtifactValidationError("judge shard is missing workflow_row_id")
    payload = _load_json(source_json, "judge shard output")
    payload["per_example_verdicts"] = _validate_judge_payload(payload, shard)
    if file_identity(shard_csv) != shard_identity:
        raise ArtifactValidationError("judge shard changed while its result was read")
    if file_identity(source_json) != source_identity:
        raise ArtifactValidationError("raw judge output changed while it was read")
    if judge_implementation_identity(judge_dir, judge_script, run_id) != judge_identity:
        raise ArtifactValidationError("judge implementation changed during shard sealing")
    _atomic_json(output_json, payload)
    output_identity = file_identity(output_json)
    if file_identity(shard_csv) != shard_identity:
        raise ArtifactValidationError("judge shard changed before result publication")
    if judge_implementation_identity(judge_dir, judge_script, run_id) != judge_identity:
        raise ArtifactValidationError("judge implementation changed before result publication")
    manifest = {
        "schema_version": JUDGE_SCHEMA_VERSION,
        "kind": "judge_shard_result",
        "row_count": int(len(shard)),
        "input": shard_identity,
        "judge_implementation": judge_identity,
        "output": output_identity,
    }
    _atomic_json(judge_manifest_path(output_json), manifest)
    return manifest


def validate_judge_shard(
    shard_csv: Path,
    output_json: Path,
    judge_dir: Path,
    judge_script: Path,
    run_id: str,
) -> dict[str, Any]:
    manifest_path = judge_manifest_path(output_json)
    manifest_identity = file_identity(manifest_path)
    shard_identity = file_identity(shard_csv)
    output_identity = file_identity(output_json)
    judge_identity = judge_implementation_identity(
        judge_dir, judge_script, run_id
    )
    manifest = _load_json(manifest_path, "judge shard manifest")
    if manifest.get("schema_version") != JUDGE_SCHEMA_VERSION:
        raise ArtifactValidationError("judge shard manifest schema version is unsupported")
    if manifest.get("kind") != "judge_shard_result":
        raise ArtifactValidationError("judge shard manifest kind is invalid")
    if manifest.get("input") != shard_identity:
        raise ArtifactValidationError("judge shard input identity is stale")
    if manifest.get("judge_implementation") != judge_identity:
        raise ArtifactValidationError("judge shard implementation identity is stale")
    if manifest.get("output") != output_identity:
        raise ArtifactValidationError("judge shard output identity is stale")
    try:
        shard = pd.read_csv(shard_csv)
    except Exception as exc:
        raise ArtifactValidationError(f"could not read judge shard {shard_csv}: {exc}") from exc
    payload = _load_json(output_json, "judge shard output")
    _validate_judge_payload(payload, shard)
    if manifest.get("row_count") != len(shard):
        raise ArtifactValidationError("judge shard manifest row count is stale")
    if (
        file_identity(manifest_path) != manifest_identity
        or file_identity(shard_csv) != shard_identity
        or file_identity(output_json) != output_identity
        or judge_implementation_identity(judge_dir, judge_script, run_id)
        != judge_identity
    ):
        raise ArtifactValidationError("judge shard bundle changed during validation")
    return manifest


def _validate_final_judge(payload: Mapping[str, Any], row_count: int) -> None:
    examples = payload.get("per_example_verdicts")
    if not isinstance(examples, list) or len(examples) != row_count:
        raise ArtifactValidationError(
            f"merged judge example count does not equal input count {row_count}"
        )
    row_ids = [example.get("workflow_row_id") for example in examples if isinstance(example, dict)]
    if row_ids != list(range(row_count)):
        raise ArtifactValidationError("merged judge row identity is incomplete or out of order")
    aggregates = payload.get("aggregates", payload)
    if not isinstance(aggregates, dict):
        raise ArtifactValidationError("merged judge aggregates must be an object")
    overall_score = normalized_score(
        aggregates.get("overall_score"), "merged judge overall_score"
    )
    categories = aggregates.get("category_aggregates")
    if not isinstance(categories, dict) or not categories:
        raise ArtifactValidationError("merged judge category_aggregates must be nonempty")
    total = 0
    expected_category_scores: dict[str, list[float]] = {}
    expected_verdict_category_scores: dict[str, list[float]] = {}
    example_scores: list[float] = []
    for offset, example in enumerate(examples):
        if not isinstance(example, dict):
            raise ArtifactValidationError(f"merged judge example {offset} must be an object")
        category = example.get("prompt_category")
        if not isinstance(category, str) or not category:
            raise ArtifactValidationError(
                f"merged judge example {offset} prompt_category must be nonempty"
            )
        score = normalized_score(
            example.get("overall_score"),
            f"merged judge example {offset} overall_score",
        )
        example_scores.append(score)
        expected_category_scores.setdefault(category, []).append(score)
        verdicts = example.get("verdicts")
        if not isinstance(verdicts, dict) or not verdicts:
            raise ArtifactValidationError(
                f"merged judge example {offset} must contain nonempty verdicts"
            )
        for judge_name, verdict in verdicts.items():
            if not isinstance(verdict, dict):
                raise ArtifactValidationError(
                    f"merged judge example {offset} verdict {judge_name} must be an object"
                )
            verdict_category = str(verdict.get("category") or category)
            verdict_score = normalized_score(
                verdict.get("score"),
                f"merged judge example {offset} verdict {judge_name} score",
            )
            expected_verdict_category_scores.setdefault(
                verdict_category, []
            ).append(verdict_score)

    expected_overall = sum(example_scores) / len(example_scores)
    if not math.isclose(overall_score, expected_overall, rel_tol=1e-12, abs_tol=1e-12):
        raise ArtifactValidationError(
            "merged judge overall_score is inconsistent with per-example rows"
        )
    if set(categories) != set(expected_category_scores):
        raise ArtifactValidationError(
            "merged judge category aggregates do not match per-example categories"
        )
    for category, data in categories.items():
        if not isinstance(data, dict):
            raise ArtifactValidationError(
                f"merged judge category {category} aggregate must be an object"
            )
        count = data.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ArtifactValidationError(
                f"merged judge category {category} count must be a nonnegative integer"
            )
        mean_score = normalized_score(
            data.get("mean_score"),
            f"merged judge category {category} mean_score",
        )
        expected_scores = expected_category_scores[category]
        expected_mean = sum(expected_scores) / len(expected_scores)
        if count != len(expected_scores) or not math.isclose(
            mean_score, expected_mean, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ArtifactValidationError(
                f"merged judge category {category} aggregate is inconsistent "
                "with per-example rows"
            )
        total += count
    if total != row_count:
        raise ArtifactValidationError(
            f"merged judge category counts sum to {total}, expected {row_count}"
        )
    score_maps = {
        "per_category_scores": expected_category_scores,
        "per_judge_category_scores": expected_verdict_category_scores,
    }
    for field, expected in score_maps.items():
        actual = payload.get(field)
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ArtifactValidationError(
                f"merged judge {field} does not match per-example rows"
            )
        for category, expected_scores in expected.items():
            values = actual[category]
            if not isinstance(values, list) or len(values) != len(expected_scores):
                raise ArtifactValidationError(
                    f"merged judge {field} category {category} has inconsistent scores"
                )
            normalized = [
                normalized_score(
                    value,
                    f"merged judge {field} category {category} score {index}",
                )
                for index, value in enumerate(values)
            ]
            if any(
                not math.isclose(value, expected_value, rel_tol=1e-12, abs_tol=1e-12)
                for value, expected_value in zip(normalized, expected_scores)
            ):
                raise ArtifactValidationError(
                    f"merged judge {field} category {category} is inconsistent "
                    "with per-example rows"
                )


def _load_split_manifest(
    input_csv: Path,
    split_manifest: Path,
) -> dict[str, Any]:
    manifest = _load_json(split_manifest, "judge split manifest")
    if manifest.get("schema_version") != JUDGE_SCHEMA_VERSION:
        raise ArtifactValidationError("judge split manifest schema version is unsupported")
    if manifest.get("kind") != "judge_csv_split":
        raise ArtifactValidationError("judge split manifest kind is invalid")
    if manifest.get("input") != file_identity(input_csv):
        raise ArtifactValidationError("judge split input identity is stale")
    try:
        source = pd.read_csv(input_csv)
    except Exception as exc:
        raise ArtifactValidationError(f"could not read judge split input: {exc}") from exc
    validate_judge_input(source)
    source = source.copy()
    source["workflow_row_id"] = list(range(len(source)))
    if manifest.get("ordered_columns") != source.columns.tolist():
        raise ArtifactValidationError("judge split ordered columns are stale")
    items = manifest.get("items")
    shard_count = manifest.get("shard_count")
    if not isinstance(items, list) or not isinstance(shard_count, int):
        raise ArtifactValidationError("judge split manifest has invalid shard metadata")
    if len(items) != shard_count:
        raise ArtifactValidationError("judge split manifest shard count is inconsistent")
    expected_start = 0
    shard_frames: list[pd.DataFrame] = []
    for expected_index, item in enumerate(items):
        if not isinstance(item, dict) or item.get("shard_index") != expected_index:
            raise ArtifactValidationError("judge split manifest shard indexes are not exact")
        try:
            input_identity = item["input"]
            shard_path = Path(input_identity["path"])
            start = item["start"]
            end = item["end"]
            row_count = item["row_count"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactValidationError(
                f"judge split shard {expected_index} metadata is invalid"
            ) from exc
        if not all(isinstance(value, int) for value in (start, end, row_count)):
            raise ArtifactValidationError(
                f"judge split shard {expected_index} row range must be integral"
            )
        if item.get("input") != file_identity(shard_path):
            raise ArtifactValidationError(f"judge input shard {expected_index} is stale")
        try:
            shard_frame = pd.read_csv(shard_path)
        except Exception as exc:
            raise ArtifactValidationError(
                f"could not read judge input shard {expected_index}: {exc}"
            ) from exc
        shard_frames.append(shard_frame)
        if start != expected_start or end - start != row_count or row_count < 1:
            raise ArtifactValidationError("judge split row ranges are not contiguous")
        expected_start = end
    if expected_start != manifest.get("row_count"):
        raise ArtifactValidationError("judge split does not cover every input row")
    reconstructed = pd.concat(shard_frames, ignore_index=True)
    if not reconstructed.equals(source.reset_index(drop=True)):
        raise ArtifactValidationError(
            "judge split shards do not reconstruct the exact ordered input"
        )
    return manifest


def merge_judge_outputs(
    input_csv: Path,
    split_manifest: Path,
    output_json: Path,
    judge_dir: Path,
    judge_script: Path,
    run_id: str,
) -> dict[str, Any]:
    split_manifest_identity = file_identity(split_manifest)
    manifest = _load_split_manifest(input_csv, split_manifest)
    if file_identity(split_manifest) != split_manifest_identity:
        raise ArtifactValidationError("judge split manifest changed during merge setup")
    input_identity = dict(manifest["input"])
    judge_identity = judge_implementation_identity(judge_dir, judge_script, run_id)
    all_examples: list[dict[str, Any]] = []
    result_identities: list[dict[str, Any]] = []
    validated_artifacts: list[tuple[Path, dict[str, Any]]] = []
    per_category_scores: dict[str, list[float]] = {}
    per_judge_category_scores: dict[str, list[float]] = {}
    for item in manifest["items"]:
        shard_csv = Path(item["input"]["path"])
        result_path = Path(item["judge_output"])
        shard_manifest = validate_judge_shard(
            shard_csv, result_path, judge_dir, judge_script, run_id
        )
        payload = _load_json(result_path, "judge shard output")
        shard = pd.read_csv(shard_csv)
        if shard_manifest.get("input") != file_identity(shard_csv):
            raise ArtifactValidationError("judge shard input changed during merge")
        if shard_manifest.get("output") != file_identity(result_path):
            raise ArtifactValidationError("judge shard output changed during merge")
        examples = _validate_judge_payload(payload, shard)
        all_examples.extend(examples)
        result_identity = dict(shard_manifest["output"])
        result_identities.append(result_identity)
        validated_artifacts.extend(
            [
                (shard_csv, dict(shard_manifest["input"])),
                (result_path, result_identity),
            ]
        )
        for example in examples:
            category = example["prompt_category"]
            per_category_scores.setdefault(category, []).append(
                normalized_score(example["overall_score"], "judge overall_score")
            )
            for verdict in example["verdicts"].values():
                verdict_category = verdict.get("category") or category
                per_judge_category_scores.setdefault(str(verdict_category), []).append(
                    normalized_score(verdict["score"], "judge verdict score")
                )
    row_count = int(manifest["row_count"])
    if [example["workflow_row_id"] for example in all_examples] != list(range(row_count)):
        raise ArtifactValidationError("judge shards do not cover the exact ordered row set")
    results = {
        "per_example_verdicts": all_examples,
        "per_category_scores": per_category_scores,
        "per_judge_category_scores": per_judge_category_scores,
    }

    judge_dir_resolved = judge_dir.expanduser().resolve()
    judge_script_resolved = judge_script.expanduser().resolve()
    module_name = f"_ecg_workflow_judge_{uuid.uuid4().hex}"
    sys.path.insert(0, str(judge_dir_resolved))
    try:
        spec = importlib.util.spec_from_file_location(module_name, judge_script_resolved)
        if spec is None or spec.loader is None:
            raise ArtifactValidationError(
                f"could not load exact judge script: {judge_script_resolved}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        if not callable(getattr(module, "aggregate_results", None)) or not callable(
            getattr(module, "save_results", None)
        ):
            raise ArtifactValidationError(
                "requested judge script does not expose aggregate_results and save_results"
            )
        aggregates = module.aggregate_results(results)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_json.parent / (
            f".{output_json.stem}.{uuid.uuid4().hex}.tmp{output_json.suffix}"
        )
        try:
            module.save_results(results, aggregates, str(temporary), verbose=False)
            if judge_implementation_identity(judge_dir, judge_script, run_id) != judge_identity:
                raise ArtifactValidationError(
                    "judge implementation changed while merge was active"
                )
            final_payload = _load_json(temporary, "merged judge output")
            _validate_final_judge(final_payload, row_count)
            if file_identity(input_csv) != input_identity:
                raise ArtifactValidationError("judge input changed during merge")
            if file_identity(split_manifest) != split_manifest_identity:
                raise ArtifactValidationError("judge split manifest changed during merge")
            for artifact_path, identity in validated_artifacts:
                if file_identity(artifact_path) != identity:
                    raise ArtifactValidationError(
                        f"judge merge input changed: {artifact_path}"
                    )
            os.replace(temporary, output_json)
        finally:
            if temporary.exists():
                temporary.unlink()
    finally:
        sys.path.remove(str(judge_dir_resolved))
        sys.modules.pop(module_name, None)

    if file_identity(input_csv) != input_identity:
        raise ArtifactValidationError("judge input changed before result publication")
    if file_identity(split_manifest) != split_manifest_identity:
        raise ArtifactValidationError(
            "judge split manifest changed before result publication"
        )
    if judge_implementation_identity(judge_dir, judge_script, run_id) != judge_identity:
        raise ArtifactValidationError(
            "judge implementation changed before merged result publication"
        )
    for artifact_path, identity in validated_artifacts:
        if file_identity(artifact_path) != identity:
            raise ArtifactValidationError(
                f"judge merge input changed before publication: {artifact_path}"
            )
    output_identity = file_identity(output_json)
    final_manifest = {
        "schema_version": JUDGE_SCHEMA_VERSION,
        "kind": "merged_judge_result",
        "row_count": row_count,
        "shard_count": int(manifest["shard_count"]),
        "input": input_identity,
        "split_manifest": split_manifest_identity,
        "judge_implementation": judge_identity,
        "shard_results": result_identities,
        "output": output_identity,
    }
    _atomic_json(judge_manifest_path(output_json), final_manifest)
    return final_manifest


def validate_merged_judge(
    input_csv: Path,
    output_json: Path,
    requested_shards: int,
    judge_dir: Path,
    judge_script: Path,
    run_id: str,
) -> dict[str, Any]:
    manifest_path = judge_manifest_path(output_json)
    manifest_identity = file_identity(manifest_path)
    input_identity = file_identity(input_csv)
    output_identity = file_identity(output_json)
    judge_identity = judge_implementation_identity(
        judge_dir, judge_script, run_id
    )
    manifest = _load_json(manifest_path, "merged judge manifest")
    if manifest.get("schema_version") != JUDGE_SCHEMA_VERSION:
        raise ArtifactValidationError("merged judge manifest schema version is unsupported")
    if manifest.get("kind") != "merged_judge_result":
        raise ArtifactValidationError("merged judge manifest kind is invalid")
    try:
        frame = pd.read_csv(input_csv)
    except Exception as exc:
        raise ArtifactValidationError(f"could not read judge input: {exc}") from exc
    validate_judge_input(frame)
    expected_shards = min(requested_shards, len(frame))
    if manifest.get("row_count") != len(frame) or manifest.get("shard_count") != expected_shards:
        raise ArtifactValidationError("merged judge row or shard count is stale")
    if manifest.get("input") != input_identity:
        raise ArtifactValidationError("merged judge input identity is stale")
    if manifest.get("judge_implementation") != judge_identity:
        raise ArtifactValidationError("merged judge implementation identity is stale")
    if manifest.get("output") != output_identity:
        raise ArtifactValidationError("merged judge output identity is stale")
    payload = _load_json(output_json, "merged judge output")
    _validate_final_judge(payload, len(frame))
    if (
        file_identity(manifest_path) != manifest_identity
        or file_identity(input_csv) != input_identity
        or file_identity(output_json) != output_identity
        or judge_implementation_identity(judge_dir, judge_script, run_id)
        != judge_identity
    ):
        raise ArtifactValidationError("merged judge bundle changed during validation")
    return manifest


def validate_judge_bundle_binding(
    input_csv: Path,
    output_json: Path,
) -> dict[str, Any]:
    manifest_path = judge_manifest_path(output_json)
    manifest_identity = file_identity(manifest_path)
    input_identity = file_identity(input_csv)
    output_identity = file_identity(output_json)
    manifest = _load_json(manifest_path, "merged judge manifest")
    if manifest.get("schema_version") != JUDGE_SCHEMA_VERSION:
        raise ArtifactValidationError("merged judge manifest schema version is unsupported")
    if manifest.get("kind") != "merged_judge_result":
        raise ArtifactValidationError("merged judge manifest kind is invalid")
    if manifest.get("input") != input_identity:
        raise ArtifactValidationError("merged judge input identity is stale")
    if manifest.get("output") != output_identity:
        raise ArtifactValidationError("merged judge output identity is stale")
    try:
        frame = pd.read_csv(input_csv)
    except Exception as exc:
        raise ArtifactValidationError(f"could not read judge input: {exc}") from exc
    validate_judge_input(frame)
    if manifest.get("row_count") != len(frame):
        raise ArtifactValidationError("merged judge row count is stale")
    payload = _load_json(output_json, "merged judge output")
    _validate_final_judge(payload, len(frame))
    if (
        file_identity(manifest_path) != manifest_identity
        or file_identity(input_csv) != input_identity
        or file_identity(output_json) != output_identity
    ):
        raise ArtifactValidationError("merged judge bundle changed during validation")
    return manifest


def atomic_copy_json(source: Path, destination: Path) -> None:
    try:
        payload = json.loads(source.read_text())
    except Exception as exc:
        raise ArtifactValidationError(f"invalid JSON {source}: {exc}") from exc
    _atomic_json(destination, payload)


def score_manifest_path(score_output: Path) -> Path:
    return Path(f"{score_output}.manifest.json")


def score_implementation_identity(
    scorer_script: Path,
    implementation_roots: Iterable[Path],
) -> dict[str, Any]:
    resolved_script = scorer_script.expanduser().resolve()
    resolved_roots = [path.expanduser().resolve() for path in implementation_roots]
    if not resolved_script.is_file():
        raise ArtifactValidationError(f"score script is missing: {resolved_script}")
    missing_roots = [str(path) for path in resolved_roots if not path.exists()]
    if missing_roots:
        raise ArtifactValidationError(
            f"score implementation roots are missing: {missing_roots}"
        )
    identity = python_implementation_identity(
        [resolved_script, *resolved_roots],
        runtime_packages=("numpy", "pandas", "scikit-learn"),
    )
    identity["entrypoint"] = file_identity(resolved_script)
    return identity


def _score_configuration(expected_categories: Sequence[str]) -> dict[str, Any]:
    categories: list[str] = []
    for value in expected_categories:
        category = value.strip()
        if not category:
            raise ArtifactValidationError("expected score category must be nonempty")
        if category in categories:
            raise ArtifactValidationError(
                f"duplicate expected score category: {category}"
            )
        categories.append(category)
    if not categories:
        raise ArtifactValidationError("expected score categories must be nonempty")
    return {"expected_categories": categories}


def publish_score_result(
    source_csv: Path,
    score_output: Path,
    expected_source_identity: Mapping[str, Any],
    scorer_script: Path,
    implementation_roots: Sequence[Path],
    expected_score_implementation: Mapping[str, Any],
    expected_categories: Sequence[str],
    tag: str,
) -> dict[str, Any]:
    normalized_tag = tag.strip()
    if not normalized_tag:
        raise ArtifactValidationError("score tag must be nonempty")
    source_identity = file_identity(source_csv)
    if source_identity != dict(expected_source_identity):
        raise ArtifactValidationError("score source changed before publication")
    score_implementation = score_implementation_identity(
        scorer_script, implementation_roots
    )
    if score_implementation != dict(expected_score_implementation):
        raise ArtifactValidationError(
            "score implementation changed before publication"
        )
    score_config = _score_configuration(expected_categories)
    score_identity = file_identity(score_output)
    if score_identity["size"] < 1:
        raise ArtifactValidationError("score output is empty")
    manifest = {
        "schema_version": 1,
        "kind": "deterministic_score_result",
        "tag": normalized_tag,
        "source": source_identity,
        "score_implementation": score_implementation,
        "config": score_config,
        "score_output": score_identity,
    }
    if file_identity(source_csv) != source_identity:
        raise ArtifactValidationError("score source changed during publication")
    if file_identity(score_output) != score_identity:
        raise ArtifactValidationError("score output changed during publication")
    if score_implementation_identity(
        scorer_script, implementation_roots
    ) != score_implementation:
        raise ArtifactValidationError("score implementation changed during publication")
    _atomic_json(score_manifest_path(score_output), manifest)
    return manifest


def validate_score_result(
    source_csv: Path,
    score_output: Path,
    scorer_script: Path,
    implementation_roots: Sequence[Path],
    expected_categories: Sequence[str],
    tag: str,
) -> dict[str, Any]:
    normalized_tag = tag.strip()
    if not normalized_tag:
        raise ArtifactValidationError("score tag must be nonempty")
    manifest_path = score_manifest_path(score_output)
    manifest_identity = file_identity(manifest_path)
    source_identity = file_identity(source_csv)
    score_identity = file_identity(score_output)
    score_implementation = score_implementation_identity(
        scorer_script, implementation_roots
    )
    score_config = _score_configuration(expected_categories)
    manifest = _load_json(manifest_path, "score result manifest")
    if manifest.get("schema_version") != 1:
        raise ArtifactValidationError("score manifest schema version is unsupported")
    if manifest.get("kind") != "deterministic_score_result":
        raise ArtifactValidationError("score manifest kind is invalid")
    if manifest.get("tag") != normalized_tag:
        raise ArtifactValidationError("score tag identity is stale")
    if manifest.get("source") != source_identity:
        raise ArtifactValidationError("score source identity is stale")
    if manifest.get("score_implementation") != score_implementation:
        raise ArtifactValidationError("score implementation identity is stale")
    if manifest.get("config") != score_config:
        raise ArtifactValidationError("score configuration identity is stale")
    if manifest.get("score_output") != score_identity:
        raise ArtifactValidationError("score output identity is stale")
    if (
        file_identity(manifest_path) != manifest_identity
        or file_identity(source_csv) != source_identity
        or file_identity(score_output) != score_identity
        or score_implementation_identity(
            scorer_script, implementation_roots
        )
        != score_implementation
    ):
        raise ArtifactValidationError("score bundle changed during validation")
    return manifest


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ArtifactValidationError(f"{label} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ArtifactValidationError(f"{label} must be finite")
    return number


def normalized_score(value: Any, label: str) -> float:
    number = finite_number(value, label)
    if number < 0.0 or number > 1.0:
        raise ArtifactValidationError(f"{label} must be between 0 and 1")
    return number


def temporary_directory(parent: Path, prefix: str) -> tempfile.TemporaryDirectory[str]:
    parent.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=parent, prefix=prefix)


def _manifest_content_identity(identity: Any, label: str) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise ArtifactValidationError(f"{label} identity must be an object")
    size = identity.get("size")
    sha256 = identity.get("sha256")
    path = identity.get("path")
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
        or not isinstance(sha256, str)
        or len(sha256) != 64
        or not isinstance(path, str)
        or not path
    ):
        raise ArtifactValidationError(f"{label} identity is incomplete")
    return {"size": size, "sha256": sha256}


def _generation_output_config(config: Any, label: str) -> dict[str, Any]:
    if not isinstance(config, Mapping) or not config:
        raise ArtifactValidationError(f"{label} generation config must be nonempty")
    # Device selects execution placement. Every other setting defines the output.
    return {key: value for key, value in config.items() if key != "device"}


def _ordered_input_hash(frame: pd.DataFrame) -> str:
    input_columns = (
        "source_row_idx",
        "waveform_name",
        "waveform_path",
        "question",
        "ground_truth",
        "prompt_category",
    )
    digest = hashlib.sha256()
    for row in frame.loc[:, input_columns].itertuples(index=False, name=None):
        encoded = json.dumps(
            [_string(value) for value in row],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _validate_generation_partition_frame(
    frame: pd.DataFrame,
    path: Path,
    seen_source_rows: set[int],
    seen_logical_inputs: set[tuple[str, str, str, str]],
) -> list[int]:
    source_rows = [
        _exact_integer(value, f"generation shard {path} source_row_idx")
        for value in frame["source_row_idx"]
    ]
    if len(set(source_rows)) != len(source_rows):
        raise ArtifactValidationError(
            f"generation shard {path} contains duplicate source_row_idx"
        )
    overlapping_rows = seen_source_rows.intersection(source_rows)
    if overlapping_rows:
        raise ArtifactValidationError(
            "generation shard partition overlaps source_row_idx "
            f"{min(overlapping_rows)}"
        )
    seen_source_rows.update(source_rows)

    shard_logical_inputs: set[tuple[str, str, str, str]] = set()
    for row_offset, row in frame.iterrows():
        waveform_path = _string(row["waveform_path"])
        waveform_name = _string(row["waveform_name"])
        if waveform_name != Path(waveform_path).stem:
            raise ArtifactValidationError(
                f"generation shard {path} waveform_name mismatch at row {row_offset}"
            )
        logical_input = (
            waveform_path,
            _string(row["question"]),
            _string(row["ground_truth"]),
            _string(row["prompt_category"]),
        )
        if logical_input in seen_logical_inputs or logical_input in shard_logical_inputs:
            raise ArtifactValidationError(
                "generation shard partition contains a duplicate logical input"
            )
        shard_logical_inputs.add(logical_input)
    seen_logical_inputs.update(shard_logical_inputs)
    return source_rows


def merge_generation_csvs(inputs: Iterable[Path], output: Path) -> dict[str, Any]:
    paths = [path.expanduser().resolve() for path in inputs]
    with exclusive_output_lock(output):
        return _merge_generation_csvs_locked(paths, output)


def _publish_merged_generation_csv(
    frames: Sequence[pd.DataFrame],
    output: Path,
    validated_inputs: Mapping[Path, Mapping[str, Any]],
    identities: Sequence[Mapping[str, Any]],
    common_checkpoint: Mapping[str, Any] | None,
    common_config: Mapping[str, Any] | None,
    subset_parent: Path | None,
    subset_run_stem: str | None,
    partitions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    merged = pd.concat(frames, ignore_index=True)
    for input_path, expected_identity in validated_inputs.items():
        if file_identity(input_path) != expected_identity:
            raise ArtifactValidationError(
                f"generation merge input changed before publication: {input_path}"
            )
    _atomic_csv(output, merged)
    output_identity = file_identity(output)
    for input_path, expected_identity in validated_inputs.items():
        if file_identity(input_path) != expected_identity:
            raise ArtifactValidationError(
                f"generation merge input changed during publication: {input_path}"
            )
    manifest = {
        "schema_version": 1,
        "kind": "merged_ecg_generation_csv",
        "row_count": int(len(merged)),
        "inputs": list(identities),
        "provenance": {
            "checkpoint_content": common_checkpoint,
            "generation_config": common_config,
            "subset_run": {
                "directory": str(subset_parent),
                "identity": subset_run_stem,
                "ordered_input_sha256": _ordered_input_hash(merged),
                "partitions": list(partitions),
            },
        },
        "output": output_identity,
    }
    _atomic_json(generation_manifest_path(output), manifest)
    return manifest


def _merge_generation_csvs_locked(
    paths: Sequence[Path], output: Path
) -> dict[str, Any]:
    if not paths:
        raise ArtifactValidationError("no generation shard CSVs were provided")
    if len(set(paths)) != len(paths):
        raise ArtifactValidationError("duplicate generation shard paths were provided")
    frames: list[pd.DataFrame] = []
    identities: list[dict[str, Any]] = []
    partitions: list[dict[str, Any]] = []
    expected_columns: list[str] | None = None
    common_checkpoint: dict[str, Any] | None = None
    common_config: dict[str, Any] | None = None
    subset_parent: Path | None = None
    subset_run_stem: str | None = None
    seen_partition_ids: set[int] = set()
    seen_subset_content: set[tuple[int, str]] = set()
    seen_source_rows: set[int] = set()
    seen_logical_inputs: set[tuple[str, str, str, str]] = set()
    verified_identity_by_path: dict[Path, dict[str, Any]] = {}
    validated_inputs: dict[Path, dict[str, Any]] = {}
    for shard_index, path in enumerate(paths):
        identity = file_identity(path)
        source_manifest_path = generation_manifest_path(path)
        source_manifest_identity = file_identity(source_manifest_path)
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            raise ArtifactValidationError(f"could not read generation shard {path}: {exc}") from exc
        missing = GENERATION_REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ArtifactValidationError(
                f"generation shard {path} is missing columns: {sorted(missing)}"
            )
        columns = frame.columns.tolist()
        if expected_columns is None:
            expected_columns = columns
        elif columns != expected_columns:
            raise ArtifactValidationError(
                f"generation shard {path} has a different ordered schema"
            )
        validate_judge_input(frame)
        source_manifest = _load_json(
            source_manifest_path, "generation shard manifest"
        )
        if file_identity(path) != identity:
            raise ArtifactValidationError(
                f"generation shard {path} changed while it was being read"
            )
        if file_identity(source_manifest_path) != source_manifest_identity:
            raise ArtifactValidationError(
                f"generation shard {path} manifest changed while it was being read"
            )
        if source_manifest.get("schema_version") != GENERATION_SCHEMA_VERSION:
            raise ArtifactValidationError(
                f"generation shard {path} manifest schema is unsupported"
            )
        if source_manifest.get("kind") != "ecg_generation_csv":
            raise ArtifactValidationError(
                f"generation shard {path} manifest kind is invalid"
            )
        if source_manifest.get("output") != identity:
            raise ArtifactValidationError(
                f"generation shard {path} hash does not match its manifest"
            )
        if source_manifest.get("row_count") != len(frame):
            raise ArtifactValidationError(
                f"generation shard {path} row count does not match its manifest"
            )
        checkpoint_identity = source_manifest.get("checkpoint")
        checkpoint_content = _manifest_content_identity(
            checkpoint_identity, f"generation shard {path} checkpoint"
        )
        checkpoint_path = Path(str(checkpoint_identity["path"])).expanduser().resolve()
        if checkpoint_path not in verified_identity_by_path:
            verified_identity_by_path[checkpoint_path] = file_identity(checkpoint_path)
        if checkpoint_identity != verified_identity_by_path[checkpoint_path]:
            raise ArtifactValidationError(
                f"generation shard {path} checkpoint identity is stale"
            )
        if common_checkpoint is None:
            common_checkpoint = checkpoint_content
        elif checkpoint_content != common_checkpoint:
            raise ArtifactValidationError(
                "generation shards use different checkpoint content"
            )

        output_config = _generation_output_config(
            source_manifest.get("config"), f"generation shard {path}"
        )
        if common_config is None:
            common_config = output_config
        elif output_config != common_config:
            raise ArtifactValidationError(
                "generation shards use incompatible output-defining configs"
            )

        subset_identity = source_manifest.get("subset")
        subset_content = _manifest_content_identity(
            subset_identity, f"generation shard {path} subset"
        )
        subset_path = Path(str(subset_identity["path"])).expanduser().resolve()
        if subset_path not in verified_identity_by_path:
            verified_identity_by_path[subset_path] = file_identity(subset_path)
        if subset_identity != verified_identity_by_path[subset_path]:
            raise ArtifactValidationError(
                f"generation shard {path} subset identity is stale"
            )
        try:
            subset_frame = pd.read_parquet(subset_path)
        except Exception as exc:
            raise ArtifactValidationError(
                f"could not read generation shard subset {subset_path}: {exc}"
            ) from exc
        run_fingerprint = validate_generation_frame(
            frame, subset_frame, checkpoint_path
        )
        if source_manifest.get("run_fingerprint") != run_fingerprint:
            raise ArtifactValidationError(
                f"generation shard {path} run fingerprint is stale"
            )
        if file_identity(subset_path) != verified_identity_by_path[subset_path]:
            raise ArtifactValidationError(
                f"generation shard {path} subset changed while it was being read"
            )
        if file_identity(checkpoint_path) != verified_identity_by_path[checkpoint_path]:
            raise ArtifactValidationError(
                f"generation shard {path} checkpoint changed during validation"
            )
        current_parent = subset_path.parent
        if subset_parent is None:
            subset_parent = current_parent
        elif current_parent != subset_parent:
            raise ArtifactValidationError(
                "generation shard subsets do not belong to one run directory"
            )
        stem = subset_path.stem
        partition_stem = stem.rstrip("0123456789")
        partition_suffix = stem[len(partition_stem) :]
        if len(paths) > 1 and not partition_suffix:
            raise ArtifactValidationError(
                "generation shard subset names must end in a numeric partition id"
            )
        current_partition = int(partition_suffix) if partition_suffix else 0
        if subset_run_stem is None:
            subset_run_stem = partition_stem
        elif partition_stem != subset_run_stem:
            raise ArtifactValidationError(
                "generation shard subsets belong to different run identities"
            )
        if current_partition in seen_partition_ids:
            raise ArtifactValidationError(
                f"duplicate generation subset partition id {current_partition}"
            )
        seen_partition_ids.add(current_partition)
        subset_key = (subset_content["size"], subset_content["sha256"])
        if subset_key in seen_subset_content:
            raise ArtifactValidationError(
                "generation shards reuse duplicate subset content"
            )
        seen_subset_content.add(subset_key)

        source_rows = _validate_generation_partition_frame(
            frame, path, seen_source_rows, seen_logical_inputs
        )

        frames.append(frame)
        validated_inputs[path] = dict(identity)
        validated_inputs[source_manifest_path] = dict(source_manifest_identity)
        validated_inputs[checkpoint_path] = dict(
            verified_identity_by_path[checkpoint_path]
        )
        validated_inputs[subset_path] = dict(verified_identity_by_path[subset_path])
        identity["row_count"] = int(len(frame))
        identity["manifest"] = source_manifest_identity
        identities.append(identity)
        partitions.append(
            {
                "shard_index": shard_index,
                "partition_id": current_partition,
                "row_count": int(len(frame)),
                "subset": dict(subset_identity),
                "source_row_ids": source_rows,
                "ordered_input_sha256": _ordered_input_hash(frame),
                "execution_device": source_manifest["config"].get("device"),
                "run_fingerprint": run_fingerprint,
            }
        )
    return _publish_merged_generation_csv(
        frames,
        output,
        validated_inputs,
        identities,
        common_checkpoint,
        common_config,
        subset_parent,
        subset_run_stem,
        partitions,
    )


def _workflow_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    merge = subparsers.add_parser("merge-generations")
    merge.add_argument("--input", action="append", required=True)
    merge.add_argument("--output", required=True)
    fingerprint = subparsers.add_parser("fingerprint")
    fingerprint.add_argument("path")
    content_fingerprint = subparsers.add_parser("content-fingerprint")
    content_fingerprint.add_argument("path")
    split = subparsers.add_parser("split-judge")
    split.add_argument("--csv", required=True)
    split.add_argument("--out-dir", required=True)
    split.add_argument("--shards", required=True, type=int)
    seal = subparsers.add_parser("seal-judge-shard")
    seal.add_argument("--csv", required=True)
    seal.add_argument("--input-json", required=True)
    seal.add_argument("--output-json", required=True)
    seal.add_argument("--judge-dir", required=True)
    seal.add_argument("--judge-script", required=True)
    seal.add_argument("--run-id", required=True)
    seal.add_argument("--expected-judge-identity", required=True)
    validate_shard = subparsers.add_parser("validate-judge-shard")
    validate_shard.add_argument("--csv", required=True)
    validate_shard.add_argument("--output-json", required=True)
    validate_shard.add_argument("--judge-dir", required=True)
    validate_shard.add_argument("--judge-script", required=True)
    validate_shard.add_argument("--run-id", required=True)
    merge_judge = subparsers.add_parser("merge-judge")
    merge_judge.add_argument("--csv", required=True)
    merge_judge.add_argument("--split-manifest", required=True)
    merge_judge.add_argument("--output-json", required=True)
    merge_judge.add_argument("--judge-dir", required=True)
    merge_judge.add_argument("--judge-script", required=True)
    merge_judge.add_argument("--run-id", required=True)
    validate_merged = subparsers.add_parser("validate-merged-judge")
    validate_merged.add_argument("--csv", required=True)
    validate_merged.add_argument("--output-json", required=True)
    validate_merged.add_argument("--shards", required=True, type=int)
    validate_merged.add_argument("--judge-dir", required=True)
    validate_merged.add_argument("--judge-script", required=True)
    validate_merged.add_argument("--run-id", required=True)
    validate_binding = subparsers.add_parser("validate-judge-binding")
    validate_binding.add_argument("--csv", required=True)
    validate_binding.add_argument("--output-json", required=True)
    publish_score = subparsers.add_parser("publish-score")
    publish_score.add_argument("--source-csv", required=True)
    publish_score.add_argument("--score-output", required=True)
    publish_score.add_argument("--expected-source-identity", required=True)
    publish_score.add_argument("--scorer-script", required=True)
    publish_score.add_argument("--implementation-root", action="append", default=[])
    publish_score.add_argument("--expected-score-implementation", required=True)
    publish_score.add_argument("--expected-category", action="append", required=True)
    publish_score.add_argument("--tag", required=True)
    validate_score = subparsers.add_parser("validate-score")
    validate_score.add_argument("--source-csv", required=True)
    validate_score.add_argument("--score-output", required=True)
    validate_score.add_argument("--scorer-script", required=True)
    validate_score.add_argument("--implementation-root", action="append", default=[])
    validate_score.add_argument("--expected-category", action="append", required=True)
    validate_score.add_argument("--tag", required=True)
    score_identity = subparsers.add_parser("score-identity")
    score_identity.add_argument("--scorer-script", required=True)
    score_identity.add_argument("--implementation-root", action="append", default=[])
    judge_identity = subparsers.add_parser("judge-identity")
    judge_identity.add_argument("--judge-dir", required=True)
    judge_identity.add_argument("--judge-script", required=True)
    judge_identity.add_argument("--run-id", required=True)
    run_locked = subparsers.add_parser("run-with-output-lock")
    run_locked.add_argument("--output", required=True)
    run_locked.add_argument("--env-marker", required=True)
    run_locked.add_argument("argv", nargs=argparse.REMAINDER)
    return parser


def _main() -> int:
    args = _workflow_parser().parse_args()
    try:
        if args.command == "merge-generations":
            manifest = merge_generation_csvs(
                [Path(value) for value in args.input], Path(args.output).expanduser().resolve()
            )
            print(json.dumps(manifest, sort_keys=True))
        elif args.command == "fingerprint":
            print(json.dumps(file_identity(Path(args.path)), sort_keys=True))
        elif args.command == "content-fingerprint":
            print(json.dumps(file_content_identity(Path(args.path)), sort_keys=True))
        elif args.command == "split-judge":
            manifest = split_judge_csv(
                Path(args.csv).expanduser().resolve(),
                Path(args.out_dir).expanduser().resolve(),
                args.shards,
            )
            print(manifest["shard_count"])
        elif args.command == "seal-judge-shard":
            try:
                expected_identity = json.loads(args.expected_judge_identity)
            except json.JSONDecodeError as exc:
                raise ArtifactValidationError(
                    "expected judge identity must be valid JSON"
                ) from exc
            if not isinstance(expected_identity, dict):
                raise ArtifactValidationError(
                    "expected judge identity must be a JSON object"
                )
            manifest = seal_judge_shard(
                Path(args.input_json).expanduser().resolve(),
                Path(args.csv).expanduser().resolve(),
                Path(args.output_json).expanduser().resolve(),
                Path(args.judge_dir).expanduser().resolve(),
                Path(args.judge_script).expanduser().resolve(),
                args.run_id,
                expected_judge_identity=expected_identity,
            )
            print(json.dumps(manifest, sort_keys=True))
        elif args.command == "validate-judge-shard":
            manifest = validate_judge_shard(
                Path(args.csv).expanduser().resolve(),
                Path(args.output_json).expanduser().resolve(),
                Path(args.judge_dir).expanduser().resolve(),
                Path(args.judge_script).expanduser().resolve(),
                args.run_id,
            )
            print(json.dumps(manifest, sort_keys=True))
        elif args.command == "merge-judge":
            manifest = merge_judge_outputs(
                Path(args.csv).expanduser().resolve(),
                Path(args.split_manifest).expanduser().resolve(),
                Path(args.output_json).expanduser().resolve(),
                Path(args.judge_dir).expanduser().resolve(),
                Path(args.judge_script).expanduser().resolve(),
                args.run_id,
            )
            print(json.dumps(manifest, sort_keys=True))
        elif args.command == "validate-merged-judge":
            manifest = validate_merged_judge(
                Path(args.csv).expanduser().resolve(),
                Path(args.output_json).expanduser().resolve(),
                args.shards,
                Path(args.judge_dir).expanduser().resolve(),
                Path(args.judge_script).expanduser().resolve(),
                args.run_id,
            )
            print(json.dumps(manifest, sort_keys=True))
        elif args.command == "validate-judge-binding":
            manifest = validate_judge_bundle_binding(
                Path(args.csv).expanduser().resolve(),
                Path(args.output_json).expanduser().resolve(),
            )
            print(json.dumps(manifest, sort_keys=True))
        elif args.command == "publish-score":
            try:
                expected_source_identity = json.loads(
                    args.expected_source_identity
                )
            except json.JSONDecodeError as exc:
                raise ArtifactValidationError(
                    "expected score source identity must be valid JSON"
                ) from exc
            if not isinstance(expected_source_identity, dict):
                raise ArtifactValidationError(
                    "expected score source identity must be a JSON object"
                )
            try:
                expected_score_implementation = json.loads(
                    args.expected_score_implementation
                )
            except json.JSONDecodeError as exc:
                raise ArtifactValidationError(
                    "expected score implementation must be valid JSON"
                ) from exc
            if not isinstance(expected_score_implementation, dict):
                raise ArtifactValidationError(
                    "expected score implementation must be a JSON object"
                )
            manifest = publish_score_result(
                Path(args.source_csv).expanduser().resolve(),
                Path(args.score_output).expanduser().resolve(),
                expected_source_identity,
                Path(args.scorer_script).expanduser().resolve(),
                [Path(value) for value in args.implementation_root],
                expected_score_implementation,
                args.expected_category,
                args.tag,
            )
            print(json.dumps(manifest, sort_keys=True))
        elif args.command == "validate-score":
            manifest = validate_score_result(
                Path(args.source_csv).expanduser().resolve(),
                Path(args.score_output).expanduser().resolve(),
                Path(args.scorer_script).expanduser().resolve(),
                [Path(value) for value in args.implementation_root],
                args.expected_category,
                args.tag,
            )
            print(json.dumps(manifest, sort_keys=True))
        elif args.command == "score-identity":
            identity = score_implementation_identity(
                Path(args.scorer_script).expanduser().resolve(),
                [Path(value) for value in args.implementation_root],
            )
            print(json.dumps(identity, sort_keys=True))
        elif args.command == "judge-identity":
            identity = judge_implementation_identity(
                Path(args.judge_dir).expanduser().resolve(),
                Path(args.judge_script).expanduser().resolve(),
                args.run_id,
            )
            print(json.dumps(identity, sort_keys=True))
        else:
            return run_with_output_lock(
                Path(args.output), args.argv, args.env_marker
            )
        return 0
    except ArtifactValidationError as exc:
        print(f"[eval-workflow] failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
