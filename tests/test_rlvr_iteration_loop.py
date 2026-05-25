import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def _load_iteration_loop_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "rlvr_iteration_loop.py"
    spec = importlib.util.spec_from_file_location("rlvr_iteration_loop", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_checkpoint_label_distinguishes_same_basename_paths(tmp_path):
    module = _load_iteration_loop_module()
    ckpt_dir = tmp_path / "ckpts"
    ckpt_a = ckpt_dir / "run_a" / "best_model.pt"
    ckpt_b = ckpt_dir / "run_b" / "best_model.pt"
    ckpt_a.parent.mkdir(parents=True)
    ckpt_b.parent.mkdir(parents=True)
    ckpt_a.write_text("", encoding="utf-8")
    ckpt_b.write_text("", encoding="utf-8")

    assert module.checkpoint_label(str(ckpt_a), str(ckpt_dir)) != module.checkpoint_label(str(ckpt_b), str(ckpt_dir))


def test_run_eval_ignores_summary_for_different_checkpoint(tmp_path, monkeypatch):
    module = _load_iteration_loop_module()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    ckpt_a = tmp_path / "a" / "best_model.pt"
    ckpt_b = tmp_path / "b" / "best_model.pt"
    ckpt_a.parent.mkdir()
    ckpt_b.parent.mkdir()
    ckpt_a.write_text("", encoding="utf-8")
    ckpt_b.write_text("", encoding="utf-8")
    label = "best_model_collision"
    summary_path = output_dir / f"summary_{label}.json"
    summary_path.write_text(
        json.dumps({"overall_score": 0.1, "iteration_loop": {"checkpoint_path": str(ckpt_a.resolve())}}),
        encoding="utf-8",
    )

    def fake_run(cmd, env):
        summary_path.write_text(json.dumps({"overall_score": 0.2}), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    summary = module.run_eval(
        checkpoint=str(ckpt_b),
        subset_parquet=str(tmp_path / "subset.parquet"),
        output_dir=str(output_dir),
        device="cpu",
        label=label,
    )

    assert summary["overall_score"] == 0.2
    assert summary["iteration_loop"]["checkpoint_path"] == str(ckpt_b.resolve())
