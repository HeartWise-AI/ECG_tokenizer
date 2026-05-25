import importlib.util
from pathlib import Path

import torch


def _load_train_dpo_policy_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_dpo_policy.py"
    spec = importlib.util.spec_from_file_location("train_dpo_policy", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DummyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.keep = torch.nn.Linear(1, 1)
        self.skip = torch.nn.Linear(1, 1)


def test_set_trainable_regex_takes_precedence_over_named_mode():
    module = _load_train_dpo_policy_module()
    model = DummyPolicy()

    module.set_trainable(model, mode="lora", regex=r"^keep\.")

    assert model.keep.weight.requires_grad
    assert model.keep.bias.requires_grad
    assert not model.skip.weight.requires_grad
    assert not model.skip.bias.requires_grad
