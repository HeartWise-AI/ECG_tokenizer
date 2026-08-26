"""Content-addressed provenance and atomic artifact helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

import fcntl

import numpy as np


def _stable_sha256_and_stat(path: str | Path) -> tuple[str, os.stat_result]:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        before = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise RuntimeError(f"file changed while its identity was being captured: {path}")
    return digest.hexdigest(), after


def sha256_file(path: str | Path) -> str:
    digest, _ = _stable_sha256_and_stat(path)
    return digest


def file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    digest, stat = _stable_sha256_and_stat(resolved)
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "sha256": digest,
    }


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require_matching_provenance(
    expected: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    artifact: str,
) -> None:
    """Abort publication when any captured input or implementation identity changed."""
    if dict(current) != dict(expected):
        raise RuntimeError(
            f"{artifact} inputs changed while the artifact was being computed; "
            "discard this run and retry with stable inputs"
        )


def require_finite_numeric_array(
    value: Any,
    *,
    artifact: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Return an array only when its shape, dtype, and every value are valid."""
    array = np.asarray(value)
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(
            f"{artifact} shape {array.shape} does not match expected {expected_shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{artifact} must contain numeric values")
    invalid_count = int(np.size(array) - np.isfinite(array).sum())
    if invalid_count:
        raise ValueError(f"{artifact} contains {invalid_count} non-finite values")
    return array


def ordered_files_identity(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Hash an ordered set of files by resolved path, size, and content."""
    digest = hashlib.sha256()
    count = 0
    for raw_path in paths:
        identity = file_identity(raw_path)
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
        count += 1
    return {"count": count, "sha256": digest.hexdigest()}


def python_implementation_identity(
    roots: Iterable[str | Path],
    runtime_packages: Iterable[str] = (),
) -> dict[str, Any]:
    """Hash Python sources and runtime versions that define an artifact."""
    resolved_roots = [Path(path).expanduser().resolve() for path in roots]
    records: list[tuple[str, Path | None]] = []
    seen_files: set[Path] = set()
    for root_index, root in enumerate(resolved_roots):
        if root.is_file():
            candidates = [root] if root.suffix == ".py" else []
        elif root.is_dir():
            candidates = sorted(root.rglob("*.py"))
        else:
            records.append((f"root-{root_index}:MISSING", None))
            continue
        for candidate in candidates:
            if any(part in {".git", ".venv", "__pycache__"} for part in candidate.parts):
                continue
            resolved = candidate.resolve()
            if resolved in seen_files:
                continue
            seen_files.add(resolved)
            relative = candidate.name if root.is_file() else str(candidate.relative_to(root))
            records.append((f"root-{root_index}:{relative}", resolved))

    digest = hashlib.sha256()
    for label, path in sorted(records, key=lambda item: item[0]):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(b"MISSING" if path is None else sha256_file(path).encode("ascii"))
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
        "python_file_count": len(seen_files),
        "runtime": runtime,
        "sha256": digest.hexdigest(),
    }


def encode_manifest(manifest: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        json.dumps(dict(manifest), sort_keys=True, separators=(",", ":")),
        dtype=np.str_,
    )


def decode_manifest(value: np.ndarray) -> dict[str, Any]:
    if value.shape != ():
        raise ValueError(f"provenance manifest must be scalar, got shape {value.shape}")
    decoded = json.loads(str(value.item()))
    if not isinstance(decoded, dict):
        raise ValueError("provenance manifest must decode to an object")
    return decoded


def load_npz_if_current(path: str | Path, expected: Mapping[str, Any]):
    artifact = Path(path)
    if not artifact.exists():
        return None
    try:
        loaded = np.load(artifact, allow_pickle=False)
        if "provenance" not in loaded.files:
            loaded.close()
            return None
        actual = decode_manifest(loaded["provenance"])
        if actual != dict(expected):
            loaded.close()
            return None
        return loaded
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


@contextmanager
def exclusive_artifact_lock(path: str | Path) -> Iterator[None]:
    """Serialize writers that target the same artifact bundle."""
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def validate_artifact_bundle(
    manifest_path: str | Path,
    *,
    required_files: Iterable[str] = (),
) -> dict[str, Any]:
    """Validate the commit manifest and content identities for a file bundle."""
    manifest_file = Path(manifest_path)
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        files = manifest["files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"artifact bundle is not committed: {manifest_file}") from exc
    if not isinstance(manifest, dict) or not isinstance(files, dict):
        raise ValueError(f"artifact bundle manifest is invalid: {manifest_file}")
    missing = set(required_files) - set(files)
    if missing:
        raise ValueError(f"artifact bundle manifest is missing files: {sorted(missing)}")
    for name, expected in files.items():
        if not isinstance(name, str) or not isinstance(expected, dict):
            raise ValueError(f"artifact bundle manifest has an invalid file entry: {name}")
        if Path(name).name != name:
            raise ValueError(f"artifact bundle manifest has an unsafe file name: {name}")
        artifact = manifest_file.parent / name
        try:
            current = file_identity(artifact)
        except OSError as exc:
            raise ValueError(f"artifact bundle file is missing: {artifact}") from exc
        if current != expected:
            raise ValueError(f"artifact bundle file does not match its manifest: {artifact}")
    return manifest


def publish_artifact_bundle(
    staged_files: Mapping[str, str | Path],
    manifest_path: str | Path,
    *,
    provenance: Mapping[str, Any],
    verify_current: Callable[[], None] | None = None,
) -> None:
    """Promote staged files and publish a commit manifest only after all succeed."""
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    if manifest_file.exists():
        manifest_file.unlink()

    promoted: dict[str, dict[str, Any]] = {}
    for name, staged_path in staged_files.items():
        if Path(name).name != name:
            raise ValueError(f"artifact bundle names must be basenames: {name}")
        staged = Path(staged_path)
        if not staged.is_file():
            raise ValueError(f"staged artifact is missing: {staged}")
        if verify_current is not None:
            verify_current()
        destination = manifest_file.parent / name
        os.replace(staged, destination)
        promoted[name] = file_identity(destination)

    if verify_current is not None:
        verify_current()
    atomic_write_json(
        manifest_file,
        {
            "schema_version": 1,
            "kind": "artifact_bundle",
            "provenance": dict(provenance),
            "files": promoted,
        },
    )


def atomic_savez_compressed(path: str | Path, **arrays: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=destination.name + ".",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def json_safe(value: Any) -> Any:
    """Convert NumPy values and non-finite floats to strict JSON values."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=destination.name + ".",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=destination.name + ".",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
