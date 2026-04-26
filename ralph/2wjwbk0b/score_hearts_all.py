"""Score a MedGemma+ECG-tokenizer checkpoint on all 18 HEARTS `mhi_ecg` tasks.

Run from the HEARTS repo (it imports from `exp.mhi_ecg.*` and
`agents.medgemma_ecg`):

    cd /volume/HEARTS
    uv run python /volume/ECG_tokenizer/ralph/2wjwbk0b/score_hearts_all.py \
        --checkpoint /path/to/best_model.pt \
        --out /tmp/scores.json \
        --n-per-task 100 \
        --device cuda:0

Output JSON contains:
    {
      "checkpoint": "...",
      "n_per_task": int,
      "composite": float in [0, 1],
      "per_task_metric": {task: <native metric value>},
      "per_task_score":  {task: <normalised score in [0,1]>},
      "per_task_n":      {task: int},
      "elapsed_seconds":  float
    }
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# --- HEARTS import bootstrapping --------------------------------------------

HEARTS_ROOT = Path("/volume/HEARTS")
if str(HEARTS_ROOT) not in sys.path:
    sys.path.insert(0, str(HEARTS_ROOT))

from agents.medgemma_ecg.agent import MedGemmaECGAgent  # noqa: E402

# Force-import every mhi_ecg task so the registry is populated.
for _mod in (
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
    importlib.import_module(_mod)

from exp.utils.registry import get_experiment  # noqa: E402

TASKS = [
    "interpretation",
    "json_interpretation",
    "classification",
    "category_rhythm",
    "category_conduction",
    "category_ischemia",
    "category_other",
    "localization_st_elevation",
    "localization_q_wave",
    "localization_t_wave",
    "heart_rate",
    "qrs_axis",
    "age_gender",
    "afib_risk",
    "lvef",
    "acs_severity",
    "culprit_artery",
    "structural_heart_disease",
]

DEFAULT_FIXTURE_DIR = HEARTS_ROOT / "fix_test_cases" / "mhi_ecg"


# --- composite normalisation -------------------------------------------------


def _score_for_task(task: str, metrics: Dict[str, Any]) -> float:
    """Map a task's `calculate_metrics` output to a per-task score in [0, 1]."""
    if task in (
        "category_rhythm",
        "category_conduction",
        "category_ischemia",
        "category_other",
        "localization_st_elevation",
        "localization_q_wave",
        "localization_t_wave",
    ):
        return float(metrics.get("f1", 0.0))
    if task == "interpretation":
        return float(metrics.get("rouge_l_f1", 0.0))
    if task == "json_interpretation":
        return float(metrics.get("f1", 0.0))
    if task in ("classification", "qrs_axis", "acs_severity", "culprit_artery"):
        return float(metrics.get("Accuracy", 0.0))
    if task == "heart_rate":
        mae = float(metrics.get("MAE", float("nan")))
        if not np.isfinite(mae):
            return 0.0
        return max(0.0, 1.0 - mae / 20.0)
    if task == "lvef":
        mae = float(metrics.get("MAE", float("nan")))
        if not np.isfinite(mae):
            return 0.0
        # Same shape as heart_rate but with a 15-pt floor.
        return max(0.0, 1.0 - mae / 15.0)
    if task == "age_gender":
        mae = float(metrics.get("MAE", float("nan")))
        gender_acc = float(metrics.get("Accuracy", 0.0))
        if not np.isfinite(mae):
            mae_score = 0.0
        else:
            mae_score = max(0.0, 1.0 - mae / 15.0)
        return 0.5 * gender_acc + 0.5 * mae_score
    if task in ("afib_risk", "structural_heart_disease"):
        auroc = float(metrics.get("AUROC", float("nan")))
        if not np.isfinite(auroc):
            return 0.0
        return auroc
    raise KeyError(f"unknown task in scoring map: {task}")


def _native_metric_value(task: str, metrics: Dict[str, Any]) -> Any:
    """Pick the canonical native metric for the task (for logging)."""
    if task in (
        "category_rhythm",
        "category_conduction",
        "category_ischemia",
        "category_other",
        "localization_st_elevation",
        "localization_q_wave",
        "localization_t_wave",
        "json_interpretation",
    ):
        return metrics.get("f1")
    if task == "interpretation":
        return metrics.get("rouge_l_f1")
    if task in ("classification", "qrs_axis", "acs_severity", "culprit_artery"):
        return metrics.get("Accuracy")
    if task in ("heart_rate", "lvef"):
        return metrics.get("MAE")
    if task == "age_gender":
        return {"age_MAE": metrics.get("MAE"), "gender_Accuracy": metrics.get("Accuracy")}
    if task in ("afib_risk", "structural_heart_disease"):
        return metrics.get("AUROC")
    return None


# --- core eval loop ---------------------------------------------------------


async def _run_task(
    agent: MedGemmaECGAgent,
    task: str,
    fixtures_dir: Path,
    n: int,
    logs_dir: Path,
) -> Dict[str, Any]:
    """Score one task by walking pickles and calling the agent on each."""
    cls = get_experiment(task)
    if cls is None:
        raise RuntimeError(f"task {task!r} not registered")

    task_dir = fixtures_dir / task
    if not task_dir.is_dir():
        return {
            "metrics": {"Failures": 0},
            "score": 0.0,
            "n": 0,
            "skipped": "no fixture dir",
        }

    pickles = sorted(task_dir.glob("*.pkl"), key=lambda p: int(p.stem))
    if not pickles:
        return {"metrics": {"Failures": 0}, "score": 0.0, "n": 0, "skipped": "empty"}
    pickles = pickles[:n]

    exp = cls(num_test=len(pickles), agent=agent, logs_dir=logs_dir)
    results: List[Dict[str, Any]] = []
    for p in pickles:
        with open(p, "rb") as f:
            data = pickle.load(f)
        try:
            r = await exp.run_agent(data)
        except Exception as exc:  # noqa: BLE001
            r = {"GT": data.get("GT"), "solution": {}, "fail_reason": str(exc)}
        results.append(r)

    metrics = exp.calculate_metrics(results)
    score = _score_for_task(task, metrics)
    return {
        "metrics": metrics,
        "score": score,
        "n": len(results),
    }


async def _async_main(args: argparse.Namespace) -> int:
    fixtures_dir = Path(args.fixtures_dir).resolve()
    if not fixtures_dir.is_dir():
        print(f"fixtures dir not found: {fixtures_dir}", file=sys.stderr)
        return 2

    agent = MedGemmaECGAgent(
        model_name="medgemma-ecg-qformer-8cb",
        checkpoint_path=args.checkpoint,
        generation_kwargs=args.gen_kwargs,
        device=args.device,
    )

    logs_dir = Path(args.logs_dir).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    out: Dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "n_per_task": args.n_per_task,
        "device": args.device,
        "per_task_metric": {},
        "per_task_score": {},
        "per_task_n": {},
        "skipped": {},
    }

    t0 = time.time()
    for task in TASKS:
        t_task = time.time()
        result = await _run_task(agent, task, fixtures_dir, args.n_per_task, logs_dir)
        out["per_task_metric"][task] = _native_metric_value(task, result["metrics"])
        out["per_task_score"][task] = result["score"]
        out["per_task_n"][task] = result["n"]
        if result.get("skipped"):
            out["skipped"][task] = result["skipped"]
        print(
            f"  {task:30s} score={result['score']:.3f} n={result['n']} "
            f"({time.time()-t_task:.1f}s)",
            file=sys.stderr,
        )

    scored_tasks = [t for t in TASKS if out["per_task_n"][t] > 0]
    if scored_tasks:
        out["composite"] = sum(out["per_task_score"][t] for t in scored_tasks) / len(
            scored_tasks
        )
    else:
        out["composite"] = 0.0
    out["scored_task_count"] = len(scored_tasks)
    out["elapsed_seconds"] = time.time() - t0

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(
        f"composite={out['composite']:.4f} (scored {out['scored_task_count']}/{len(TASKS)}) "
        f"-> {out_path}",
        file=sys.stderr,
    )
    return 0


def _parse_gen_kwargs(s: Optional[str]) -> Dict[str, Any]:
    if not s:
        return {}
    return json.loads(s)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True, help="path to write scores.json")
    p.add_argument("--n-per-task", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--fixtures-dir",
        default=str(DEFAULT_FIXTURE_DIR),
        help="root with one subdir per task (default: HEARTS fix_test_cases/mhi_ecg)",
    )
    p.add_argument("--logs-dir", default="/tmp/score_hearts_all_logs")
    p.add_argument(
        "--gen-kwargs",
        type=_parse_gen_kwargs,
        default={
            "max_new_tokens": 96,
            "do_sample": False,
            "no_repeat_ngram_size": 5,
            "repetition_penalty": 1.1,
        },
        help="JSON dict of generation kwargs to override the checkpoint's defaults",
    )
    args = p.parse_args()
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    sys.exit(main())
