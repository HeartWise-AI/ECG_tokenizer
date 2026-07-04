import importlib.util
from pathlib import Path


def _load_train_rft_policy_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_rft_policy.py"
    spec = importlib.util.spec_from_file_location("train_rft_policy", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_should_step_optimizer_flushes_tail_batch():
    module = _load_train_rft_policy_module()

    stepped = [
        step
        for step in range(1, 4)
        if module.should_step_optimizer(step=step, total_steps=3, grad_accum_steps=2)
    ]

    assert stepped == [2, 3]
