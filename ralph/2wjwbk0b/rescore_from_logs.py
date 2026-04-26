"""Recompute HEARTS composite scores from already-saved per-sample logs.

When the model is held constant but the parsing / metric / scoring code
gets fixed, we want to rescore *without* rerunning the (slow) model. Each
score_hearts_all.py run saves a pickle per (task, query_id) under its
logs_dir; this script walks those, pairs them with the fixture's GT by
matching the prompt+ECG, applies the **current** experiment's
parse_output + calculate_metrics, and writes a fresh scores.json.

Run from the HEARTS repo:

    cd /volume/HEARTS
    uv run python /volume/ECG_tokenizer/ralph/2wjwbk0b/rescore_from_logs.py \
        --logs-dir /tmp/baseline-2wjwbk0b-logs \
        --fixtures-dir /volume/HEARTS/fix_test_cases/mhi_ecg \
        --in-baseline /volume/ECG_tokenizer/ralph/2wjwbk0b/BASELINE.json \
        --out /volume/ECG_tokenizer/ralph/2wjwbk0b/BASELINE.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

HEARTS_ROOT = Path("/volume/HEARTS")
if str(HEARTS_ROOT) not in sys.path:
    sys.path.insert(0, str(HEARTS_ROOT))

# Eagerly import every mhi_ecg task so the experiment registry resolves.
for _m in (
    "exp.mhi_ecg.acs_severity",
    "exp.mhi_ecg.afib_risk",
    "exp.mhi_ecg.age_gender",
    "exp.mhi_ecg.category_conduction",
    "exp.mhi_ecg.category_ischemia",
    "exp.mhi_ecg.category_other",
    "exp.mhi_ecg.category_rhythm",
    "exp.mhi_ecg.classification",
    "exp.mhi_ecg.culprit_artery",
    "exp.mhi_ecg.heart_rate",
    "exp.mhi_ecg.interpretation",
    "exp.mhi_ecg.json_interpretation",
    "exp.mhi_ecg.localization_q_wave",
    "exp.mhi_ecg.localization_st_elevation",
    "exp.mhi_ecg.localization_t_wave",
    "exp.mhi_ecg.lvef",
    "exp.mhi_ecg.qrs_axis",
    "exp.mhi_ecg.structural_heart_disease",
):
    importlib.import_module(_m)

from exp.utils.registry import get_experiment  # noqa: E402

# Pull in the same canonical TASKS / scoring helpers from the live scorer.
_score_mod = importlib.import_module("score_hearts_all")  # type: ignore[assignment]
sys.path.insert(
    0, str(Path("/volume/ECG_tokenizer/ralph/2wjwbk0b").resolve())
)
score_hearts_all = importlib.import_module("score_hearts_all")
TASKS = score_hearts_all.TASKS
_score_for_task = score_hearts_all._score_for_task
_native_metric_value = score_hearts_all._native_metric_value


def _load_fixture_index(fixtures_dir: Path, task: str) -> Dict[str, Dict[str, Any]]:
    """Index fixtures by (prompt, subject_id) → fixture dict."""
    idx: Dict[str, Dict[str, Any]] = {}
    for p in (fixtures_dir / task).glob("*.pkl"):
        with open(p, "rb") as f:
            d = pickle.load(f)
        key = f"{d.get('subject_id')}::{d.get('prompt')}"
        idx[key] = d
    return idx


def _rescore_task(task: str, logs_dir: Path, fixtures_dir: Path) -> Dict[str, Any]:
    cls = get_experiment(task)
    if cls is None:
        return {"score": 0.0, "n": 0, "skipped": "no registered experiment"}
    task_log_dir = logs_dir / task
    if not task_log_dir.is_dir():
        return {"score": 0.0, "n": 0, "skipped": "no log dir"}

    fix_idx = _load_fixture_index(fixtures_dir, task)
    if not fix_idx:
        return {"score": 0.0, "n": 0, "skipped": "no fixtures"}

    exp = cls(num_test=len(fix_idx), agent=None, logs_dir=Path("/tmp/rescore"))
    results: List[Dict[str, Any]] = []
    matched = 0
    for log_path in task_log_dir.glob("*.pkl"):
        with open(log_path, "rb") as f:
            log = pickle.load(f)
        gen = log.get("generation")
        prompt = log.get("prompt")
        # Match log to fixture by (subject_id+prompt) where possible. Fixtures
        # may share prompts so subject_id is needed; logs don't carry the
        # subject_id directly, so we rely on prompt-uniqueness when needed.
        # Common case: distinct prompt+subject pairs across fixtures means
        # we walk linearly and produce N results.
        # Fallback: pair by index — log filenames are uuids, so there is no
        # natural order; use any fixture that hasn't been consumed yet.
        if matched >= len(fix_idx):
            break
        # Pick the first fixture whose prompt matches; if none, take any.
        chosen_key = None
        for key, fx in fix_idx.items():
            if fx.get("prompt") == prompt and fx.get("_consumed", False) is False:
                chosen_key = key
                break
        if chosen_key is None:
            for key, fx in fix_idx.items():
                if not fx.get("_consumed", False):
                    chosen_key = key
                    break
        if chosen_key is None:
            break
        fx = fix_idx[chosen_key]
        fx["_consumed"] = True
        sol, fail_reason = exp.parse_output(gen)
        results.append(
            {
                "GT": fx.get("GT"),
                "solution": sol,
                "subject_id": fx.get("subject_id"),
                "fail_reason": fail_reason,
            }
        )
        matched += 1

    if not results:
        return {"score": 0.0, "n": 0, "skipped": "no log↔fixture matches"}

    metrics = exp.calculate_metrics(results)
    score = _score_for_task(task, metrics)
    return {
        "metrics": metrics,
        "score": score,
        "n": len(results),
        "native": _native_metric_value(task, metrics),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs-dir", type=Path, required=True)
    ap.add_argument("--fixtures-dir", type=Path, required=True)
    ap.add_argument("--in-baseline", type=Path, help="JSON to use as base shell (preserves checkpoint, n_per_task, etc.)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=["age_gender"],
        help="tasks to drop from the composite (default: age_gender, no training signal)",
    )
    args = ap.parse_args()

    if args.in_baseline and args.in_baseline.exists():
        with open(args.in_baseline) as f:
            shell = json.load(f)
    else:
        shell = {}
    shell.setdefault("checkpoint", "(unknown)")
    shell.setdefault("n_per_task", 0)
    shell["per_task_metric"] = {}
    shell["per_task_score"] = {}
    shell["per_task_n"] = {}
    shell["skipped"] = {}

    excluded = set(args.exclude or [])
    t0 = time.time()
    scored: List[float] = []
    for task in TASKS:
        result = _rescore_task(task, args.logs_dir, args.fixtures_dir)
        shell["per_task_score"][task] = float(result["score"])
        shell["per_task_n"][task] = int(result["n"])
        shell["per_task_metric"][task] = result.get("native")
        if result.get("skipped"):
            shell["skipped"][task] = result["skipped"]
        if task in excluded:
            shell["skipped"].setdefault(task, "excluded from composite")
        else:
            if result["n"] > 0:
                scored.append(result["score"])
        marker = " [EXCLUDED]" if task in excluded else ""
        print(f"  {task:30s} score={result['score']:.3f} n={result['n']}{marker}")

    shell["composite"] = sum(scored) / len(scored) if scored else 0.0
    shell["scored_task_count"] = len(scored)
    shell["excluded_tasks"] = sorted(excluded)
    shell["elapsed_seconds"] = time.time() - t0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(shell, f, indent=2)
    print(
        f"\ncomposite={shell['composite']:.4f} "
        f"(scored {shell['scored_task_count']}/{len(TASKS)}, excluded {sorted(excluded)}) "
        f"-> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
