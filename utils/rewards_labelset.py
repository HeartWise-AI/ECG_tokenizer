"""Phase 2 RLVR reward — multilabel F1 between model output and GT binary vector.

The GT is supplied as a semicolon-separated string of active canonical
labels (the GT columns where row == 1). The reward scans the model output
for mentions of each canonical term or any of its synonyms (ontology v3,
138 canonical / 792 synonyms), builds a predicted binary vector, and
returns F1 against the GT.

This is a pure rule-based RLVR verifier — no learned model, no API calls.
"""

import json
import re
from typing import Dict, List, Optional, Set, Tuple

# 0=Sinusal/1=Regular/2=Monomorph fire on every ECG and inflate F1 — mask them.
_NOISE_CANONICALS = {"sinusal", "regular", "monomorph"}


def _load_alias_index(ontology_path: str) -> Tuple[Dict[str, str], Set[str]]:
    """Build {alias_lower: canonical_lower} and the set of all canonicals.

    Each canonical maps to itself + all its synonyms (lowercased).
    """
    with open(ontology_path) as f:
        ont = json.load(f)

    alias_to_canon: Dict[str, str] = {}
    canonicals: Set[str] = set()
    for cat, terms in ont.items():
        if not isinstance(terms, dict):
            continue
        for can, syns in terms.items():
            cl = can.lower()
            canonicals.add(cl)
            alias_to_canon[cl] = cl
            if isinstance(syns, list):
                for s in syns:
                    sl = str(s).lower().strip()
                    if sl:
                        alias_to_canon[sl] = cl
    return alias_to_canon, canonicals


class LabelsetReward:
    """Multilabel F1 reward — pure rule-based RLVR verifier."""

    def __init__(
        self,
        ontology_path: str = "/volume/LLM_JUDGE/ontology/ecg_ontology.json",
        ignore_noise: bool = True,
    ):
        self.alias_to_canon, self.canonicals = _load_alias_index(ontology_path)
        self.ignore_noise = ignore_noise
        # Precompile alias regex — word-boundary, longest-first to avoid
        # "atrial flutter" matching as "atrial".
        sorted_aliases = sorted(self.alias_to_canon.keys(), key=len, reverse=True)
        self._alias_pattern = re.compile(
            r"\b(?:" + "|".join(re.escape(a) for a in sorted_aliases) + r")\b",
            re.IGNORECASE,
        )

    def extract_labels(self, text: str) -> Set[str]:
        if not isinstance(text, str) or not text:
            return set()
        labels = set()
        for m in self._alias_pattern.finditer(text):
            labels.add(self.alias_to_canon[m.group(0).lower()])
        if self.ignore_noise:
            labels -= _NOISE_CANONICALS
        return labels

    def f1(self, pred_labels: Set[str], gt_labels: Set[str]) -> float:
        if self.ignore_noise:
            gt_labels = gt_labels - _NOISE_CANONICALS
            pred_labels = pred_labels - _NOISE_CANONICALS
        if not pred_labels and not gt_labels:
            return 1.0
        if not pred_labels or not gt_labels:
            return 0.0
        tp = len(pred_labels & gt_labels)
        fp = len(pred_labels - gt_labels)
        fn = len(gt_labels - pred_labels)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def __call__(self, generated_text: str, gt_reference: str) -> float:
        """Reward = F1 between canonical labels extracted from both sides.

        `gt_reference` is the GT answer TEXT (not a label string); we extract
        labels from it the same way we do from the model output. This avoids
        the GT-column-to-canonical alignment gap.
        """
        pred = self.extract_labels(generated_text)
        gt = self.extract_labels(gt_reference)
        return self.f1(pred, gt)


def format_reward_labelset(generated_text: str) -> float:
    """Light format reward — generation is non-empty and within reasonable length."""
    if not isinstance(generated_text, str):
        return 0.0
    s = generated_text.strip()
    if not s or len(s) < 4:
        return 0.0
    if len(s) > 2048:
        return 0.3
    return 1.0


def compute_labelset_rewards(
    generated_text: str,
    gt_label_string: str,
    weights: Optional[Dict[str, float]] = None,
    labelset_reward: Optional[LabelsetReward] = None,
) -> Dict[str, float]:
    if weights is None:
        weights = {"format": 0.10, "diagnosis": 0.90}
    if labelset_reward is None:
        labelset_reward = LabelsetReward()
    r_format = format_reward_labelset(generated_text)
    r_diagnosis = labelset_reward(generated_text, gt_label_string)
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
