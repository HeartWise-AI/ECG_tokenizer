"""OpenRLHF custom reward function backed by the ECG LLM judge.

Used via:
    --reward.remote_url /volume/ECG_tokenizer/services/openrlhf_judge_reward.py

OpenRLHF will import this file and call `reward_func(queries, prompts, labels, **kwargs)`.
`labels` is expected to be a list of JSON strings encoding
    {"category": "<prompt_category>", "ground_truth": "<gt text>"}
so the same dataset row carries both pieces. Build the dataset that way (see
data/build_grpo_dataset_for_openrlhf.py) — or set them via the dataset's
`input_key`/`label_key` columns.

Returns one reward per query in [0, 1]:
- 1.0 if the judge gives a perfect score
- 0.0 if the judge fails or all candidates miss
- otherwise the raw judge score

The reward is also returned as the `scores` field so dynamic-filtering
(--algo.dynamic_filtering_enable --algo.dynamic_filtering_range 0.05 0.95)
can drop prompts where every sample is 0 or 1 (no GRPO learning signal).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch


# Lazy-init the LLM_JUDGE registry so importing this module is cheap.
_REGISTRY = None
_CAT_MAP: Dict[str, List[str]] = {}


def _init_judge_registry():
    """Import /volume/LLM_JUDGE and register all judges. Idempotent."""
    global _REGISTRY, _CAT_MAP
    if _REGISTRY is not None:
        return

    llm_judge_dir = os.environ.get("LLM_JUDGE_DIR", "/volume/LLM_JUDGE")
    if llm_judge_dir not in sys.path:
        sys.path.insert(0, llm_judge_dir)

    env_path = Path(llm_judge_dir) / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("FIREWORKS_API_KEY="):
                os.environ.setdefault("FIREWORKS_API_KEY", line.split("=", 1)[1].strip())

    from judges import registry, register_judges  # type: ignore
    from merge_utils import get_category_judge_mapping  # type: ignore

    register_judges()
    _REGISTRY = registry
    _CAT_MAP = get_category_judge_mapping()
    print(f"[openrlhf_judge_reward] judge registry initialized, {len(_CAT_MAP)} categories")


def _decode_label(lbl: Any) -> Dict[str, str]:
    """Labels arrive as JSON strings {"category":..., "ground_truth":...}."""
    if isinstance(lbl, dict):
        return lbl
    if isinstance(lbl, (bytes, bytearray)):
        lbl = lbl.decode("utf-8", errors="replace")
    if isinstance(lbl, str):
        try:
            return json.loads(lbl)
        except json.JSONDecodeError:
            # Plain text fallback — treat the whole label as GT, category defaults
            return {"category": "classification", "ground_truth": lbl}
    raise ValueError(f"Unsupported label type: {type(lbl)}")


def _judge_score(prediction: str, ground_truth: str, category: str) -> float:
    """Run the appropriate judge for `category`. Returns float in [0, 1]."""
    if _REGISTRY is None:
        _init_judge_registry()
    judge_names = _CAT_MAP.get(category, ["classification_judge"])
    judge = _REGISTRY.get(judge_names[0])
    if judge is None:
        return 0.0
    try:
        verdict = judge.evaluate(prediction, ground_truth)
        return float(verdict.score) if verdict is not None else 0.0
    except Exception as e:
        print(f"[openrlhf_judge_reward] judge error for category={category}: {e}")
        return 0.0


def _to_text_list(x: Any) -> List[str]:
    """OpenRLHF may pass tokenized tensors or already-decoded strings."""
    if isinstance(x, list):
        return [s if isinstance(s, str) else str(s) for s in x]
    if hasattr(x, "tolist"):
        # tensor — caller must have decoded already; this path is a fallback
        return [str(s) for s in x.tolist()]
    return [str(x)]


def reward_func(queries, prompts, labels, **kwargs):
    """OpenRLHF reward callback.

    queries  : list of full prompt+response strings (or tensors — see note)
    prompts  : list of prompt strings (or tensors)
    labels   : list of JSON-encoded {"category", "ground_truth"} or plain GT
    """
    _init_judge_registry()

    queries_t = _to_text_list(queries)
    prompts_t = _to_text_list(prompts)
    labels_t  = _to_text_list(labels)

    n = len(queries_t)
    if len(prompts_t) != n or len(labels_t) != n:
        raise ValueError(
            f"reward_func length mismatch: queries={n} prompts={len(prompts_t)} labels={len(labels_t)}"
        )

    scores: List[float] = []
    categories: List[str] = []
    for i in range(n):
        # The model's response is everything in the query that comes after the prompt.
        # Strip the prompt prefix; if not present (common with chat templating
        # adding bos/etc.), use the full query.
        q = queries_t[i]
        p = prompts_t[i]
        if p and q.startswith(p):
            response = q[len(p):].strip()
        else:
            response = q.strip()

        label_data = _decode_label(labels_t[i])
        category = label_data.get("category", "classification")
        gt = label_data.get("ground_truth", "")
        score = _judge_score(response, gt, category)
        scores.append(score)
        categories.append(category)

    reward = torch.tensor(scores, dtype=torch.float32)
    return {
        "rewards": reward,
        "scores":  reward,
        "extra_logs": {
            "judge_score": reward,
            "category_count": float(len(set(categories))),
        },
    }


if __name__ == "__main__":
    # Standalone smoke test
    _init_judge_registry()
    qs = ["The patient has Atrial Fibrillation."]
    ps = ["Diagnose:"]
    ls = [json.dumps({"category": "classification", "ground_truth": "Atrial Fibrillation"})]
    out = reward_func(qs, ps, ls)
    print(out)
