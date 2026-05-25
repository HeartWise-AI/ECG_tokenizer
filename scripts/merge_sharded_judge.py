#!/usr/bin/env python3
"""Merge shard-level LLM judge outputs into one judge_eval-compatible JSON."""

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_glob", required=True,
                        help="Glob for shard judge JSONs, e.g. judge_baseline_shard_*.json")
    parser.add_argument("--output", required=True,
                        help="Merged judge JSON path")
    parser.add_argument("--llm_judge_dir", default="/volume/LLM_JUDGE")
    args = parser.parse_args()

    judge_dir = Path(args.llm_judge_dir).resolve()
    sys.path.insert(0, str(judge_dir))
    from judge_eval import aggregate_results, save_results  # type: ignore

    paths = [
        Path(p) for p in sorted(glob.glob(args.judge_glob))
        if not p.endswith("_summary.json") and not p.endswith("_enhanced.json")
    ]
    if not paths:
        raise SystemExit(f"No judge shard JSONs matched: {args.judge_glob}")

    per_example_verdicts: List[Dict[str, Any]] = []
    for path in paths:
        with path.open() as f:
            data = json.load(f)
        examples = data.get("per_example_verdicts")
        if not isinstance(examples, list):
            raise SystemExit(f"Missing per_example_verdicts in {path}")
        per_example_verdicts.extend(examples)

    per_category_scores: Dict[str, List[float]] = {}
    per_judge_category_scores: Dict[str, List[float]] = {}
    for example in per_example_verdicts:
        category = example.get("prompt_category", "unknown")
        per_category_scores.setdefault(category, []).append(float(example.get("overall_score", 0.0)))
        for verdict in (example.get("verdicts") or {}).values():
            verdict_category = verdict.get("category") or category
            per_judge_category_scores.setdefault(verdict_category, []).append(float(verdict.get("score", 0.0)))

    results = {
        "per_example_verdicts": per_example_verdicts,
        "per_category_scores": per_category_scores,
        "per_judge_category_scores": per_judge_category_scores,
    }
    aggregates = aggregate_results(results)
    save_results(results, aggregates, args.output, verbose=True)
    print(json.dumps({
        "output": args.output,
        "shards": len(paths),
        "total_examples": len(per_example_verdicts),
        "overall_score": aggregates.get("overall_score"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
