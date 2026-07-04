#!/usr/bin/env python3
"""Build grouped candidates for offline GRPO from existing judged generations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _txt(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _load_generation_scores(generations_csv: str, judge_csv: str, source: str) -> dict:
    gen = pd.read_csv(generations_csv)
    judge = pd.read_csv(judge_csv)
    score_by_key = {
        (_txt(r["json_key"]), _txt(r["prompt"])): float(r.get("overall_score", 0.0) or 0.0)
        for _, r in judge.iterrows()
    }
    out = {}
    for _, r in gen.iterrows():
        key = (_txt(r["waveform_name"]), _txt(r["question"]))
        out[key] = {
            "text": _txt(r["generation"]),
            "score": score_by_key.get(key, 0.0),
            "source": source,
        }
    return out


def build(args: argparse.Namespace) -> None:
    subset = pd.read_parquet(args.subset_parquet)
    subset["waveform_name"] = (
        subset["waveform_path_psa"].astype(str).str.split("/").str[-1].str.replace(".npy", "", regex=False)
    )

    sources = []
    multi_sources = []
    if args.baseline_generations and args.baseline_judge:
        sources.append(_load_generation_scores(args.baseline_generations, args.baseline_judge, "baseline"))
    if args.rft_generations and args.rft_judge:
        sources.append(_load_generation_scores(args.rft_generations, args.rft_judge, "rft"))
    if args.grpo_generations and args.grpo_judge:
        sources.append(_load_generation_scores(args.grpo_generations, args.grpo_judge, "grpo"))

    best = {}
    if args.bestof_csv:
        best_df = pd.read_csv(args.bestof_csv)
        for _, r in best_df.iterrows():
            key = (_txt(r["waveform_name"]), _txt(r["question"]))
            if "bestof_candidates" in best_df.columns and _txt(r.get("bestof_candidates")):
                try:
                    texts = json.loads(_txt(r["bestof_candidates"]))
                    scores = json.loads(_txt(r.get("bestof_candidate_scores", "[]")))
                except json.JSONDecodeError:
                    texts, scores = [], []
                cands = []
                for i, text in enumerate(texts):
                    score = float(scores[i]) if i < len(scores) else 0.0
                    cands.append({
                        "text": _txt(text),
                        "score": score,
                        "source": f"bestof_candidate_{i}",
                    })
                best[key] = cands
            else:
                best[key] = [{
                    "text": _txt(r["generation"]),
                    "score": float(r.get("bestof_picked_score", 0.0) or 0.0),
                    "source": "bestof",
                }]
        multi_sources.append(best)

    rows = []
    skipped = {"small_group": 0, "flat_reward": 0}
    for _, r in subset.iterrows():
        key = (_txt(r["waveform_name"]), _txt(r["prompt"]))
        merged = {}
        for src in sources:
            cand = src.get(key)
            if not cand:
                continue
            text = cand["text"]
            if not text or text.lower() == "nan" or text.startswith("[ERROR"):
                continue
            # Deduplicate exact text, keeping the highest judged score.
            old = merged.get(text)
            if old is None or cand["score"] > old["score"]:
                merged[text] = dict(cand)
        for src in multi_sources:
            for cand in src.get(key, []):
                text = cand["text"]
                if not text or text.lower() == "nan" or text.startswith("[ERROR"):
                    continue
                # Deduplicate exact text, keeping the highest judged score.
                old = merged.get(text)
                if old is None or cand["score"] > old["score"]:
                    merged[text] = dict(cand)

        if args.include_gt:
            gt = _txt(r["generated_answer"])
            if gt:
                old = merged.get(gt)
                cand = {"text": gt, "score": args.gt_score, "source": "ground_truth"}
                if old is None or cand["score"] > old["score"]:
                    merged[gt] = cand

        cands = list(merged.values())
        if len(cands) < 2:
            skipped["small_group"] += 1
            continue
        scores = [float(c["score"]) for c in cands]
        if max(scores) - min(scores) < args.min_span:
            skipped["flat_reward"] += 1
            continue

        rows.append({
            "waveform_path": _txt(r["waveform_path_psa"]),
            "prompt": _txt(r["prompt"]),
            "category": _txt(r["prompt_category"]),
            "ground_truth": _txt(r["generated_answer"]),
            "reward_span": max(scores) - min(scores),
            "candidates": cands,
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"[offline-groups] wrote {len(rows)} groups -> {out}")
    print(f"[offline-groups] skipped: {skipped}")
    if rows:
        df = pd.DataFrame(rows)
        print("[offline-groups] per-category:")
        print(df.groupby("category").size().sort_values(ascending=False).to_string())
        print("[offline-groups] reward span:")
        print(df["reward_span"].describe().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_parquet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline_generations")
    parser.add_argument("--baseline_judge")
    parser.add_argument("--rft_generations")
    parser.add_argument("--rft_judge")
    parser.add_argument("--grpo_generations")
    parser.add_argument("--grpo_judge")
    parser.add_argument("--bestof_csv")
    parser.add_argument("--include_gt", action="store_true")
    parser.add_argument("--gt_score", type=float, default=1.0)
    parser.add_argument("--min_span", type=float, default=0.05)
    build(parser.parse_args())
