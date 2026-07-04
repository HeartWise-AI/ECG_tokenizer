import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import openrlhf_judge_reward as reward_module


def test_reward_func_discriminates_good_and_bad_predictions(monkeypatch):
    monkeypatch.setattr(reward_module, "_init_judge_registry", lambda: None)

    def fake_judge_score(prediction: str, ground_truth: str, category: str) -> float:
        return 1.0 if prediction == ground_truth else 0.0

    monkeypatch.setattr(reward_module, "_judge_score", fake_judge_score)

    prompt = "Interpret this ECG: "
    ground_truth = "Normal sinus rhythm."
    labels = [
        json.dumps({"category": "classification", "ground_truth": ground_truth}),
        json.dumps({"category": "classification", "ground_truth": ground_truth}),
    ]
    queries = [
        prompt + ground_truth,
        prompt + "The quick brown fox jumps over the lazy dog.",
    ]

    out = reward_module.reward_func(queries, [prompt, prompt], labels)
    rewards = out["rewards"].tolist()

    assert rewards[0] == 1.0
    assert rewards[1] == 0.0
    assert rewards[0] > rewards[1]
