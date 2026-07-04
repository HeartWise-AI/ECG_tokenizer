import importlib.util
from pathlib import Path


def _load_bestofn_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "rlvr_eval_bestofn.py"
    spec = importlib.util.spec_from_file_location("rlvr_eval_bestofn", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_init_judge_registry_uses_configured_directory(tmp_path):
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    (judge_dir / "judges.py").write_text(
        """
from types import SimpleNamespace

registry = {}

class FakeJudge:
    def evaluate(self, prediction, ground_truth):
        return SimpleNamespace(score=0.75)

def register_judges():
    registry["classification_judge"] = FakeJudge()
""",
        encoding="utf-8",
    )
    (judge_dir / "merge_utils.py").write_text(
        """
def get_category_judge_mapping():
    return {"classification": ["classification_judge"]}
""",
        encoding="utf-8",
    )
    module = _load_bestofn_module()

    module.init_judge_registry(str(judge_dir))

    assert module.JUDGE_DIR == str(judge_dir.resolve())
    assert module.judge_score("prediction", "ground_truth", "classification") == 0.75
