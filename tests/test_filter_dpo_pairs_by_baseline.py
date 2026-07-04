import subprocess
import sys
from pathlib import Path


def test_missing_base_checkpoint_preserves_existing_output(tmp_path):
    pairs_path = tmp_path / "pairs.jsonl"
    output_path = tmp_path / "filtered.jsonl"
    pairs_path.write_text("", encoding="utf-8")
    output_path.write_text("sentinel\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/filter_dpo_pairs_by_baseline.py",
            "--pairs",
            str(pairs_path),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--base_checkpoint is required" in result.stderr
    assert output_path.read_text(encoding="utf-8") == "sentinel\n"
