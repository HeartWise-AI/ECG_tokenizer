#!/usr/bin/env python3
"""Generate an LLM-judge CSV and publish it only after exact validation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from eval_workflow_utils import (
    ArtifactValidationError,
    exclusive_output_lock,
    file_identity,
    publish_generation,
    python_implementation_identity,
    temporary_directory,
    validate_published_generation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subset_parquet", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--generation_microbatch_size", type=int)
    parser.add_argument("--group_by_prompt", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def _implementation_roots(script_dir: Path) -> list[Path]:
    override = os.environ.get("ECG_GENERATION_IMPLEMENTATION_ROOTS", "").strip()
    if override:
        return [Path(value) for value in override.split(os.pathsep) if value]
    repo_root = script_dir.parent
    return [
        script_dir / "eval_judge_csv.py",
        script_dir / "eval_workflow_utils.py",
        repo_root / "models",
        repo_root / "config",
        repo_root / "data",
        repo_root / "runners",
        repo_root / "projects",
        repo_root / "utils",
    ]


def _config(
    args: argparse.Namespace,
    runner: Path,
    script_dir: Path,
) -> dict[str, object]:
    return {
        "batch_size": int(args.batch_size),
        "device": str(args.device),
        "generation_microbatch_size": args.generation_microbatch_size,
        "group_by_prompt": bool(args.group_by_prompt),
        "max_new_tokens": int(args.max_new_tokens),
        "runner": file_identity(runner),
        "implementation": python_implementation_identity(
            _implementation_roots(script_dir),
            runtime_packages=(
                "numpy",
                "pandas",
                "sentencepiece",
                "torch",
                "transformers",
            ),
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    script_dir = Path(__file__).resolve().parent
    runner = Path(
        os.environ.get("ECG_EVAL_RUNNER", str(script_dir / "rlvr_eval_subset.py"))
    ).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    subset = Path(args.subset_parquet).expanduser().resolve()
    output = Path(args.output_csv).expanduser().resolve()

    with exclusive_output_lock(output):
        return _run_generation_lifecycle(
            args, script_dir, runner, checkpoint, subset, output
        )


def _run_generation_lifecycle(
    args: argparse.Namespace,
    script_dir: Path,
    runner: Path,
    checkpoint: Path,
    subset: Path,
    output: Path,
) -> int:

    try:
        config = _config(args, runner, script_dir)
        if args.validate_only:
            manifest = validate_published_generation(
                output, subset, checkpoint, config
            )
            print(json.dumps({"valid": True, "manifest": manifest}, sort_keys=True))
            return 0

        checkpoint_identity = file_identity(checkpoint)
        subset_identity = file_identity(subset)
        with temporary_directory(output.parent, ".generation-run-") as temp_name:
            temp_dir = Path(temp_name)
            label = "workflow"
            command = [
                sys.executable,
                str(runner),
                "--checkpoint",
                str(checkpoint),
                "--subset_parquet",
                str(subset),
                "--output_dir",
                str(temp_dir),
                "--device",
                str(args.device),
                "--batch_size",
                str(args.batch_size),
                "--max_new_tokens",
                str(args.max_new_tokens),
                "--label",
                label,
            ]
            if args.generation_microbatch_size is not None:
                command.extend(
                    [
                        "--generation_microbatch_size",
                        str(args.generation_microbatch_size),
                    ]
                )
            if args.group_by_prompt:
                command.append("--group_by_prompt")
            environment = os.environ.copy()
            repo_root = str(script_dir.parent)
            current_pythonpath = environment.get("PYTHONPATH", "")
            environment["PYTHONPATH"] = (
                repo_root
                if not current_pythonpath
                else f"{repo_root}{os.pathsep}{current_pythonpath}"
            )
            subprocess.run(command, env=environment, check=True)
            current_config = _config(args, runner, script_dir)
            if current_config != config:
                raise ArtifactValidationError(
                    "generation implementation changed while the runner was active"
                )
            current_checkpoint_identity = file_identity(checkpoint)
            current_subset_identity = file_identity(subset)
            if current_checkpoint_identity != checkpoint_identity:
                raise ArtifactValidationError(
                    "checkpoint changed while generation was active"
                )
            if current_subset_identity != subset_identity:
                raise ArtifactValidationError(
                    "subset changed while generation was active"
                )
            runner_csv = temp_dir / f"generations_{label}.csv"
            manifest = publish_generation(
                runner_csv,
                output,
                subset,
                checkpoint,
                config,
                checkpoint_identity=current_checkpoint_identity,
                subset_identity=current_subset_identity,
            )
        print(
            json.dumps(
                {
                    "output_csv": str(output),
                    "row_count": manifest["row_count"],
                    "status": "complete",
                },
                sort_keys=True,
            )
        )
        return 0
    except (ArtifactValidationError, subprocess.CalledProcessError, OSError) as exc:
        print(f"[eval-judge-csv] failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
