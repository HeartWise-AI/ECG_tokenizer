"""Binary Yes/No verifier reward for Phase 1 RLVR warm-up.

The 5M / 188k binary-diagnosis QA parquet rows have a deterministic ground
truth in the `generated_answer` column (starts with "Yes" or "No"). The
verifier matches the model's first word against that — pure rule-based
RLVR, no learned classifier.
"""

import re
from typing import Dict, Optional

_FIRST_WORD_RE = re.compile(r"^\s*([A-Za-z]+)")


def first_word(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = _FIRST_WORD_RE.match(text)
    return m.group(1).strip().lower() if m else ""


def binary_format_reward(generated_text: str) -> float:
    return 1.0 if first_word(generated_text) in ("yes", "no") else 0.0


def binary_correctness_reward(generated_text: str, gt_first_word: str) -> float:
    gt = (gt_first_word or "").strip().lower()
    if gt not in ("yes", "no"):
        return 0.0
    return 1.0 if first_word(generated_text) == gt else 0.0


def compute_binary_rewards(
    generated_text: str,
    gt_first_word: str,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    if weights is None:
        weights = {"format": 0.1, "diagnosis": 0.9}
    r_format = binary_format_reward(generated_text)
    r_diagnosis = binary_correctness_reward(generated_text, gt_first_word)
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
