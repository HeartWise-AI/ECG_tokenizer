#!/usr/bin/env python3
"""Build DPO pairs from best-of-N selected generations vs baseline generations.

Input:
  - best-of-N CSV from scripts/rlvr_eval_bestofn.py
  - baseline generations CSV from scripts/rlvr_eval_subset.py
  - optional baseline enhanced judge CSV with per-row overall_score

Output JSONL schema consumed by scripts/train_dpo_policy.py:
  {"waveform_path", "prompt", "chosen", "rejected", "weight", ...}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _key_frame(df: pd.DataFrame, path_col: str, prompt_col: str) -> pd.DataFrame:
    out = df.copy()
    out["__key"] = (
        out[path_col].astype(str).str.strip()
        + "\n"
        + out[prompt_col].astype(str).str.strip()
    )
    return out


def build(args: argparse.Namespace) -> None:
    best = pd.read_csv(args.bestof_csv)
    base = pd.read_csv(args.baseline_csv)

    if args.baseline_judge_csv:
        base_judge = pd.read_csv(args.baseline_judge_csv)
        base_judge = _key_frame(base_judge, "json_key", "prompt")
        # json_key omits ".npy", while baseline_csv has a full path. Use prompt
        # plus generation as a fallback score join below if exact path join misses.
        score_by_prompt_gen = {
            (str(r["prompt"]).strip(), str(r["generation"]).strip()): float(r.get("overall_score", 0.0) or 0.0)
            for _, r in base_judge.iterrows()
        }
    else:
        score_by_prompt_gen = {}

    best = _key_frame(best, "waveform_path", "question")
    base = _key_frame(base, "waveform_path", "question")
    base_by_key = {str(r["__key"]): r for _, r in base.iterrows()}

    rows = []
    skipped = {
        "missing_baseline": 0,
        "bad_chosen": 0,
        "low_score": 0,
        "low_margin": 0,
        "same_text": 0,
    }

    for _, row in best.iterrows():
        key = str(row["__key"])
        base_row = base_by_key.get(key)
        if base_row is None:
            skipped["missing_baseline"] += 1
            continue

        chosen = _clean_text(row.get("generation"))
        rejected = _clean_text(base_row.get("generation"))
        if not chosen or chosen.startswith("[ERROR"):
            skipped["bad_chosen"] += 1
            continue
        if chosen == rejected:
            skipped["same_text"] += 1
            continue

        chosen_score = float(row.get("bestof_picked_score", 0.0) or 0.0)
        rejected_score = score_by_prompt_gen.get(
            (_clean_text(base_row.get("question")), rejected),
            float(base_row.get("overall_score", 0.0) or 0.0)
            if "overall_score" in base_row else 0.0,
        )
        margin = chosen_score - rejected_score
        if chosen_score < args.min_chosen_score:
            skipped["low_score"] += 1
            continue
        if margin < args.min_margin:
            skipped["low_margin"] += 1
            continue

        rows.append({
            "waveform_path": _clean_text(row.get("waveform_path")),
            "prompt": _clean_text(row.get("question")),
            "chosen": chosen,
            "rejected": rejected,
            "weight": max(args.min_weight, min(args.max_weight, margin)),
            "prompt_category": _clean_text(row.get("prompt_category")),
            "ground_truth": _clean_text(row.get("ground_truth")),
            "chosen_score": chosen_score,
            "rejected_score": rejected_score,
            "score_margin": margin,
            "bestof_picked_idx": int(row.get("bestof_picked_idx", -1)),
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"[pairs] wrote {len(rows)} rows -> {out_path}")
    print(f"[pairs] skipped: {skipped}")
    if rows:
        df = pd.DataFrame(rows)
        print("[pairs] per-category:")
        print(df.groupby("prompt_category").size().sort_values(ascending=False).to_string())
        print("[pairs] score margins:")
        print(df["score_margin"].describe().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bestof_csv", required=True)
    parser.add_argument("--baseline_csv", required=True)
    parser.add_argument("--baseline_judge_csv", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_chosen_score", type=float, default=0.5)
    parser.add_argument("--min_margin", type=float, default=0.05)
    parser.add_argument("--min_weight", type=float, default=0.1)
    parser.add_argument("--max_weight", type=float, default=1.0)
    build(parser.parse_args())
