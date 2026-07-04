#!/usr/bin/env python3
"""Iteration controller — watches RLVR checkpoint dir, re-evals at each
new checkpoint vs. the locked baseline, declares victory when the average
LLM-judge gain across categories exceeds the target threshold.

Usage:
    python scripts/rlvr_iteration_loop.py \\
        --ckpt_dir /volume/ECG_tokenizer/checkpoints/RLVR/... \\
        --baseline_summary /volume/ECG_tokenizer/analysis/rlvr_eval/baseline/summary_baseline.json \\
        --subset_parquet /volume/ECG_tokenizer/analysis/rlvr_eval/eval_subset_10per_cat.parquet \\
        --output_dir /volume/ECG_tokenizer/analysis/rlvr_eval/iterations \\
        --target_delta 0.10 --poll_secs 120 --device cuda:0
"""

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = "/volume/ECG_tokenizer/scripts/rlvr_eval_subset.py"


def load_baseline(path: str) -> dict:
    with open(path) as f:
        s = json.load(f)
    return s


def compute_delta(baseline: dict, current: dict) -> dict:
    """Returns dict with per-category mean delta and the overall average delta."""
    cats_b = baseline.get("category_aggregates", {})
    cats_c = current.get("category_aggregates", {})
    deltas = {}
    for cat, bd in cats_b.items():
        if cat not in cats_c:
            continue
        deltas[cat] = float(cats_c[cat].get("mean_score", 0.0) or 0.0) - float(
            bd.get("mean_score", 0.0) or 0.0
        )
    avg_delta = sum(deltas.values()) / max(1, len(deltas))
    overall_delta = float(current.get("overall_score", 0.0) or 0.0) - float(
        baseline.get("overall_score", 0.0) or 0.0
    )
    return {"per_category": deltas, "avg_delta": avg_delta, "overall_delta": overall_delta}


def find_new_checkpoints(ckpt_dir: str, seen: set) -> list:
    found = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt"))) + \
            sorted(glob.glob(os.path.join(ckpt_dir, "**", "*.pt"), recursive=True))
    found = [p for p in found if os.path.isfile(p) and p not in seen]
    # Prefer ones named grpo_step_*.pt or checkpoint_step_*.pt
    return found


def checkpoint_label(checkpoint: str, ckpt_dir: str) -> str:
    ckpt_path = Path(checkpoint).resolve()
    try:
        rel = ckpt_path.relative_to(Path(ckpt_dir).resolve())
    except ValueError:
        rel = ckpt_path.name
    rel_stem = str(Path(rel).with_suffix(""))
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", rel_stem).strip("_") or ckpt_path.stem
    digest = hashlib.sha1(str(ckpt_path).encode("utf-8")).hexdigest()[:8]
    return f"{safe_stem}_{digest}"


def _cached_summary_for_checkpoint(summary_path: str, checkpoint: str) -> dict | None:
    if not os.path.isfile(summary_path):
        return None
    with open(summary_path) as f:
        summary = json.load(f)
    cached_checkpoint = summary.get("iteration_loop", {}).get("checkpoint_path")
    checkpoint_abs = str(Path(checkpoint).resolve())
    if cached_checkpoint == checkpoint_abs:
        return summary
    print(
        f"[iter] Ignoring cached summary {summary_path}: "
        f"checkpoint metadata {cached_checkpoint!r} != {checkpoint_abs!r}"
    )
    return None


def _write_summary_checkpoint_metadata(summary_path: str, checkpoint: str) -> dict:
    with open(summary_path) as f:
        summary = json.load(f)
    summary.setdefault("iteration_loop", {})["checkpoint_path"] = str(Path(checkpoint).resolve())
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def mark_checkpoint_seen_after_eval(seen: set, checkpoint: str, current: dict | None) -> bool:
    if current is None:
        return False
    seen.add(checkpoint)
    return True


def run_eval(checkpoint: str, subset_parquet: str, output_dir: str,
             device: str, label: str) -> dict:
    summary_path = os.path.join(output_dir, f"summary_{label}.json")
    cached_summary = _cached_summary_for_checkpoint(summary_path, checkpoint)
    if cached_summary is not None:
        return cached_summary
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable, SCRIPT,
        "--checkpoint", checkpoint,
        "--subset_parquet", subset_parquet,
        "--output_dir", output_dir,
        "--device", device,
        "--max_new_tokens", "256",
        "--run_judge",
        "--label", label,
    ]
    print(f"[iter] eval -> {label}: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = "/volume/ECG_tokenizer"
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        print(f"[iter] WARN: eval for {label} failed (code {res.returncode})")
        return None
    return _write_summary_checkpoint_metadata(summary_path, checkpoint)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True,
                        help="Directory the RLVR runner saves checkpoints to")
    parser.add_argument("--baseline_summary", required=True)
    parser.add_argument("--subset_parquet", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target_delta", type=float, default=0.10,
                        help="Per-category average score gain to call victory (0.10 = +10pt)")
    parser.add_argument("--poll_secs", type=int, default=120)
    parser.add_argument("--max_iter_minutes", type=int, default=1440,
                        help="Stop polling after this many minutes")
    args = parser.parse_args()

    baseline = load_baseline(args.baseline_summary)
    print(f"[iter] Baseline overall: {baseline.get('overall_score'):.4f}")
    print(f"[iter] Watching {args.ckpt_dir} for new checkpoints")
    print(f"[iter] Target delta: +{args.target_delta * 100:.1f}pt avg across categories")

    os.makedirs(args.output_dir, exist_ok=True)
    seen = set()
    start = time.time()

    history = []
    while True:
        elapsed_min = (time.time() - start) / 60
        if elapsed_min > args.max_iter_minutes:
            print(f"[iter] Hit max_iter_minutes={args.max_iter_minutes}; exiting")
            break

        ckpts = find_new_checkpoints(args.ckpt_dir, seen)
        for ckpt in ckpts:
            label = checkpoint_label(ckpt, args.ckpt_dir)
            print(f"[iter] New checkpoint: {ckpt} (label={label})")
            current = run_eval(
                ckpt, args.subset_parquet, args.output_dir, args.device, label
            )
            if not mark_checkpoint_seen_after_eval(seen, ckpt, current):
                continue
            delta = compute_delta(baseline, current)
            history.append({"label": label, "ckpt": ckpt, "current": current, "delta": delta})
            history_path = os.path.join(args.output_dir, "history.json")
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)
            print(f"[iter] {label}: overall {current.get('overall_score'):.4f} "
                  f"(Δ overall {delta['overall_delta']:+.4f}, "
                  f"Δ avg-cat {delta['avg_delta']:+.4f})")
            for cat, d in sorted(delta["per_category"].items(), key=lambda kv: -kv[1])[:5]:
                print(f"        +cat {cat:30s}  Δ={d:+.4f}")
            victory = (delta["avg_delta"] >= args.target_delta
                       or delta["overall_delta"] >= args.target_delta)
            if victory:
                trigger = ("avg per-cat" if delta["avg_delta"] >= args.target_delta
                           else "overall")
                print(f"\n[iter] ✓ VICTORY ({trigger}): "
                      f"avg per-cat Δ={delta['avg_delta']:+.4f}, "
                      f"overall Δ={delta['overall_delta']:+.4f} "
                      f">= target {args.target_delta:+.4f}")
                victory_path = os.path.join(args.output_dir, "VICTORY.json")
                with open(victory_path, "w") as f:
                    json.dump({"label": label, "trigger": trigger, "delta": delta,
                               "current": current, "baseline": baseline}, f, indent=2)
                return 0

        time.sleep(args.poll_secs)

    return 1


if __name__ == "__main__":
    sys.exit(main())
