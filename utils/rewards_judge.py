"""Phase 3 RLVR reward — use the deployed LLM-as-judge as the verifier.

Each (gen, gt, prompt_category) → judge.evaluate(...).score in [0, 1].
The judges live in /volume/LLM_JUDGE/judges.py; many are deterministic
(no API call); only a few (interpretation, complex classification)
hit the Fireworks API.

Heavier than the labelset reward but better aligned with the eval metric
(LLM judge). Use as Phase 3 if Phase 2 plateaus.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional


def _bootstrap_llm_judge():
    """Ensure /volume/LLM_JUDGE is on sys.path and FIREWORKS_API_KEY is set."""
    if "/volume/LLM_JUDGE" not in sys.path:
        sys.path.insert(0, "/volume/LLM_JUDGE")
    env_path = Path("/volume/LLM_JUDGE/.env")
    if env_path.exists() and not os.environ.get("FIREWORKS_API_KEY"):
        for line in env_path.read_text().splitlines():
            if line.startswith("FIREWORKS_API_KEY="):
                os.environ["FIREWORKS_API_KEY"] = line.split("=", 1)[1].strip()


class JudgeReward:
    """Reward = LLM-judge Verdict.score, dispatched by prompt_category."""

    def __init__(self):
        _bootstrap_llm_judge()
        from judges import registry, register_judges
        from merge_utils import get_category_judge_mapping
        register_judges()
        self.registry = registry
        self.cat_map = get_category_judge_mapping()

    def __call__(self, generated_text: str, gt_reference: str,
                 prompt_category: str) -> float:
        judge_names = self.cat_map.get(prompt_category, ["classification_judge"])
        judge = self.registry.get(judge_names[0]) if judge_names else None
        if judge is None:
            return 0.0
        try:
            verdict = judge.evaluate(generated_text, gt_reference)
            return float(verdict.score) if verdict is not None else 0.0
        except Exception:
            return 0.0


def format_reward_judge(generated_text: str) -> float:
    if not isinstance(generated_text, str):
        return 0.0
    s = generated_text.strip()
    if not s or len(s) < 4:
        return 0.0
    return 1.0


def compute_judge_rewards(
    generated_text: str,
    gt_reference: str,
    prompt_category: str,
    weights: Optional[Dict[str, float]] = None,
    judge_reward: Optional[JudgeReward] = None,
) -> Dict[str, float]:
    if weights is None:
        weights = {"format": 0.05, "diagnosis": 0.95}
    if judge_reward is None:
        judge_reward = JudgeReward()
    r_format = format_reward_judge(generated_text)
    r_diagnosis = judge_reward(generated_text, gt_reference, prompt_category)
    total_weight = sum(weights.values()) or 1.0
    total = (
        weights.get("format", 0.0) * r_format
        + weights.get("diagnosis", 0.0) * r_diagnosis
    ) / total_weight
    return {
        "total": total,
        "format": r_format,
        "diagnosis": r_diagnosis,
        "evidence": 0.0,
    }
