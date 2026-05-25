import importlib.util
import json
from types import SimpleNamespace
from pathlib import Path

import torch


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


class FakeTokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def __call__(self, text, add_special_tokens, return_tensors):
        return SimpleNamespace(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.tensor([[1, 1, 1]]),
        )

    def convert_tokens_to_ids(self, token):
        return 2

    def decode(self, token_ids, skip_special_tokens):
        return "generated"


class FakeModel:
    def __init__(self):
        self.temperatures = []

    def generate_report(self, **kwargs):
        self.temperatures.append(kwargs["temperature"])
        return torch.tensor([[1, 2, 3]])


def test_generate_diverse_truncates_temperature_schedule_to_num_samples():
    module = _load_sample_and_generate_module()
    model = FakeModel()

    generations = module.generate_diverse(
        model=model,
        tokenizer=FakeTokenizer(),
        ecg_tensor=torch.zeros(1, 12, 2500),
        question="Question?",
        device=torch.device("cpu"),
        num_samples=3,
        temperatures=[0.1, 0.2, 0.3, 0.4, 0.5],
    )

    assert len(generations) == 3
    assert model.temperatures == [0.1, 0.2, 0.3]
