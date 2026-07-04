#!/usr/bin/env python3
"""Build a GRPO JSONL with hard/medium/easy examples from on-policy rewards.

The miner samples candidates from the current policy, scores them with the
verifiable reward, bins rows by reward spread, then writes a mixture such as
70% hard, 20% medium, 10% easy. Hard rows are the ones most likely to produce
non-zero GRPO advantages.
"""

import argparse
import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "grpo_openrlhf_v1", str(ROOT / "scripts" / "grpo_openrlhf_v1.py")
)
grpo_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(grpo_mod)

from services.verifiable_reward import verify as verifiable_verify


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def label_data(row: Dict) -> Dict:
    label = row.get("label", "{}")
    return json.loads(label) if isinstance(label, str) else dict(label)


def category(row: Dict) -> str:
    return str(label_data(row).get("category", "classification"))


def score_row(model, tokenizer, row: Dict, args) -> Dict:
    label = label_data(row)
    cat = str(label.get("category", "classification"))
    signal = grpo_mod.eval_mod.load_ecg_signal(row["signal_path"])
    cands = grpo_mod.sample_candidates(
        model,
        tokenizer,
        signal,
        row["prompt"],
        n=args.n_candidates,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    rewards = [float(verifiable_verify(c["decoded"], label, cat)) for c in cands]
    mean = float(np.mean(rewards))
    std = float(np.std(rewards))
    span = float(max(rewards) - min(rewards))
    if span >= args.hard_span and std >= args.hard_std:
        bucket = "hard"
    elif span > 0.0 or (args.medium_low < mean < args.medium_high):
        bucket = "medium"
    elif mean >= args.easy_mean:
        bucket = "easy"
    else:
        bucket = "flat_bad"
    return {
        "bucket": bucket,
        "category": cat,
        "reward_mean": mean,
        "reward_std": std,
        "reward_span": span,
        "rewards": rewards,
        "decoded": [c["decoded"] for c in cands[: args.keep_decoded_examples]],
    }


def sample_bucket(rng: random.Random, rows: List[Dict], n: int, fallback: List[Dict]) -> List[Dict]:
    source = rows if rows else fallback
    if not source or n <= 0:
        return []
    return [dict(rng.choice(source)) for _ in range(n)]


def build(args):
    rng = random.Random(args.seed)
    rows = load_jsonl(Path(args.input))
    if args.categories:
        want = set(args.categories.split(","))
        rows = [r for r in rows if category(r) in want]
    rng.shuffle(rows)
    if args.max_source_rows and len(rows) > args.max_source_rows:
        rows = rows[: args.max_source_rows]

    print(f"[hardmine] scoring {len(rows)} source rows from {args.input}")
    print(f"[hardmine] checkpoint={args.checkpoint} device={args.device}")

    model, tokenizer = grpo_mod.eval_mod.load_model(args.checkpoint, args.device)
    model.eval()
    try:
        model.set_lora_inference_mode(True)
    except Exception:
        pass

    buckets = defaultdict(list)
    scored_records = []
    failures = []
    for i, row in enumerate(rows, 1):
        try:
            score = score_row(model, tokenizer, row, args)
        except Exception as exc:
            failures.append({"index": i - 1, "error": str(exc), "prompt": row.get("prompt", "")})
            print(f"[hardmine] {i}/{len(rows)} failed: {exc}")
            continue
        out_row = dict(row)
        out_row["mining"] = {k: v for k, v in score.items() if k != "decoded"}
        buckets[score["bucket"]].append(out_row)
        scored_records.append({
            "index": i - 1,
            "prompt": row.get("prompt", ""),
            "signal_path": row.get("signal_path", ""),
            **score,
        })
        if i % args.log_every == 0 or i == len(rows):
            counts = {k: len(v) for k, v in sorted(buckets.items())}
            print(f"[hardmine] {i}/{len(rows)} bucket_counts={counts}")

    all_scored_rows = [r for vals in buckets.values() for r in vals]
    if not all_scored_rows:
        raise SystemExit("no rows were scored successfully")

    hard_n = int(round(args.target_rows * args.hard_frac))
    medium_n = int(round(args.target_rows * args.medium_frac))
    easy_n = args.target_rows - hard_n - medium_n

    fallback_medium = buckets["medium"] + buckets["flat_bad"] + buckets["hard"] + buckets["easy"]
    fallback_easy = buckets["easy"] + buckets["medium"] + buckets["hard"] + buckets["flat_bad"]
    mixed = []
    mixed.extend(sample_bucket(rng, buckets["hard"], hard_n, all_scored_rows))
    mixed.extend(sample_bucket(rng, buckets["medium"], medium_n, fallback_medium))
    mixed.extend(sample_bucket(rng, buckets["easy"], easy_n, fallback_easy))
    rng.shuffle(mixed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in mixed:
            f.write(json.dumps(row) + "\n")

    scored_path = Path(args.scored_output) if args.scored_output else out_path.with_suffix(".scored.jsonl")
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    with scored_path.open("w") as f:
        for rec in scored_records:
            f.write(json.dumps(rec) + "\n")

    report = {
        "input": args.input,
        "output": str(out_path),
        "scored_output": str(scored_path),
        "checkpoint": args.checkpoint,
        "source_rows_scored": len(scored_records),
        "failures": failures[:20],
        "bucket_counts": {k: len(v) for k, v in sorted(buckets.items())},
        "output_counts": {
            "hard": hard_n,
            "medium": medium_n,
            "easy": easy_n,
            "total": len(mixed),
        },
        "category_counts_by_bucket": {
            bucket: dict(Counter(category(r) for r in vals))
            for bucket, vals in sorted(buckets.items())
        },
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"[hardmine] wrote {len(mixed)} rows -> {out_path}")
    print(f"[hardmine] wrote scored rows -> {scored_path}")
    print(f"[hardmine] wrote report -> {report_path}")
    print(json.dumps(report["bucket_counts"], indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--scored_output")
    p.add_argument("--categories", default="")
    p.add_argument("--max_source_rows", type=int, default=256)
    p.add_argument("--target_rows", type=int, default=1200)
    p.add_argument("--n_candidates", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--hard_frac", type=float, default=0.70)
    p.add_argument("--medium_frac", type=float, default=0.20)
    p.add_argument("--hard_span", type=float, default=0.25)
    p.add_argument("--hard_std", type=float, default=0.05)
    p.add_argument("--medium_low", type=float, default=0.05)
    p.add_argument("--medium_high", type=float, default=0.95)
    p.add_argument("--easy_mean", type=float, default=0.95)
    p.add_argument("--keep_decoded_examples", type=int, default=2)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    build(p.parse_args())
