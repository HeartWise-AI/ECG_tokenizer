import importlib.util
import json
from pathlib import Path


def _load_sample_and_generate_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "sample_and_generate_dpo.py"
    spec = importlib.util.spec_from_file_location("sample_and_generate_dpo", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resume_uses_jsonl_when_output_is_ahead_of_checkpoint(tmp_path):
    module = _load_sample_and_generate_module()
    output_jsonl = tmp_path / "dpo_generations.jsonl"
    checkpoint_path = tmp_path / "generation_checkpoint.json"

    with output_jsonl.open("w", encoding="utf-8") as f:
        for idx in range(1501):
            f.write(json.dumps({"prompt": f"q{idx}"}) + "\n")
    checkpoint_path.write_text(json.dumps({"last_idx": 999}), encoding="utf-8")

    assert module.resolve_resume_start(str(output_jsonl), str(checkpoint_path), total_samples=2000) == 1501


def test_resume_truncates_partial_jsonl_row(tmp_path):
    module = _load_sample_and_generate_module()
    output_jsonl = tmp_path / "dpo_generations.jsonl"
    output_jsonl.write_bytes(b'{"sample_idx": 0}\n{"sample_idx": 1}\n{"sample_idx":')

    assert module.resume_start_from_output(str(output_jsonl), total_samples=10) == 2
    assert output_jsonl.read_text(encoding="utf-8") == '{"sample_idx": 0}\n{"sample_idx": 1}\n'
