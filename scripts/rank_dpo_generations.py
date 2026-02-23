#!/usr/bin/env python3
"""
Rank DPO generations using an LLM judge (Fireworks) and export DPO pairs.

Input format (JSON list or JSONL):
{
  "idx": 43826,
  "waveform_path": "...",
  "prompt": "...",
  "prompt_category": "...",
  "ground_truth": "...",
  "generations": [{"idx":0,"temperature":0.1,"output":"..."}, ...]
}

Output:
1) Ranked JSON with scores + best/worst index
2) DPO pairs JSONL (chosen/rejected + weight)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import requests
from pydantic import BaseModel, ValidationError
import ast


DEFAULT_MODEL = "accounts/fireworks/models/kimi-k2p5"
DEFAULT_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
DEFAULT_ENV_FILES = ("config/.env", ".env")
DEFAULT_ALLOWED_SCORES = "0,0.1,0.3,0.5,0.7,1.0"


def _load_env_file(path: str) -> None:
    if not path or not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                if key not in os.environ or not os.environ.get(key):
                    os.environ[key] = value


def _load_records(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "..."


def _build_messages(
    prompt: str,
    ground_truth: Optional[str],
    candidates: List[str],
    category: Optional[str],
    score_scale: float,
    max_reason_chars: int,
    allowed_scores: Optional[List[float]] = None,
    strict: bool = False,
    previous_response: Optional[str] = None,
) -> List[Dict[str, str]]:
    system_msg = (
        "You are an expert cardiologist judging answer quality. "
        "Score answers for clinical accuracy and completeness relative to the ground truth. "
        "Penalize hallucinations or incorrect findings."
    )
    lines = [
        f"Question: {prompt.strip()}",
    ]
    if category:
        lines.append(f"Category: {category.strip()}")
    if ground_truth:
        lines.append(f"Ground truth: {ground_truth.strip()}")
    lines.append("Candidates:")
    for i, cand in enumerate(candidates):
        lines.append(f"[{i}] {cand}")

    schema_hint = (
        "{"
        "\"scores\": [0.0, 0.1, ...], "
        "\"best_index\": 0, "
        "\"worst_index\": 1, "
        "\"rationale\": \"...\""
        "}"
    )
    lines.append(
        (
            "Return JSON ONLY with keys: "
            f"\"scores\" (list of {len(candidates)} numbers from 0-{score_scale}), "
            "\"best_index\", \"worst_index\", and \"rationale\" "
            f"(<= {max_reason_chars} chars). "
            "If ties, choose the lowest index for best and highest index for worst. "
            f"Schema example: {schema_hint}"
        )
    )
    if allowed_scores:
        allowed_str = ", ".join(f"{x:.2f}".rstrip("0").rstrip(".") for x in allowed_scores)
        lines.append(
            f"Use ONLY these scores: {allowed_str}. "
            "Pick the closest score if uncertain."
        )
    lines.append(
        "Scoring rubric: 0.0 = completely incorrect/unsupported, "
        "0.5 = partially correct or missing key findings, "
        f"{score_scale} = perfect match to ground truth."
    )
    if strict:
        lines.append("STRICT MODE: Output raw JSON only (no code fences, no prose).")
    if previous_response:
        lines.append("Your previous response was invalid JSON. Reformat it into valid JSON only.")
        lines.append(f"Previous response: {previous_response}")

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _build_reformat_messages(
    previous_response: str,
    n_candidates: int,
    score_scale: float,
    max_reason_chars: int,
    allowed_scores: Optional[List[float]] = None,
) -> List[Dict[str, str]]:
    system_msg = "You are a strict JSON formatter. Output raw JSON only."
    allowed_str = ""
    if allowed_scores:
        allowed_str = " Allowed scores: " + ", ".join(
            f"{x:.2f}".rstrip("0").rstrip(".") for x in allowed_scores
        ) + "."
    user_msg = (
        f"Reformat the following text into valid JSON with keys "
        f"\"scores\" (length {n_candidates}, numbers 0-{score_scale}), "
        "\"best_index\", \"worst_index\", \"rationale\" "
        f"(<= {max_reason_chars} chars).{allowed_str} "
        "If any field is missing, infer it. If you cannot infer scores, use all zeros. "
        "Return ONLY JSON. No prose, no code fences, no explanations.\n\n"
        f"Text:\n{previous_response}"
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def _fireworks_chat(
    url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "top_k": top_k,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": temperature,
        "messages": messages,
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    blob = match.group(0)
    try:
        return json.loads(blob)
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(blob)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def _snap_score(value: float, allowed: Optional[List[float]], score_scale: float) -> float:
    if allowed:
        return min(allowed, key=lambda x: abs(x - value))
    return max(0.0, min(score_scale, value))


def _normalize_scores(parsed_scores: Any, n: int, score_scale: float, allowed: Optional[List[float]]) -> Optional[List[float]]:
    scores = parsed_scores
    if isinstance(scores, list) and scores and isinstance(scores[0], dict):
        try:
            indexed = {int(s["idx"]): float(s["score"]) for s in scores if "idx" in s and "score" in s}
            if len(indexed) == n:
                scores = [indexed[i] for i in range(n)]
        except Exception:
            return None
    if isinstance(scores, list) and len(scores) == n:
        try:
            raw = [float(x) for x in scores]
        except Exception:
            return None
        return [_snap_score(v, allowed, score_scale) for v in raw]
    return None


def _argmax_min(scores: List[float]) -> Tuple[int, int]:
    max_val = max(scores)
    min_val = min(scores)
    best = next(i for i, v in enumerate(scores) if v == max_val)
    worst = max(i for i, v in enumerate(scores) if v == min_val)
    return best, worst


def _similarity_scores(ground_truth: str, candidates: List[str], allowed: Optional[List[float]]) -> List[float]:
    scores: List[float] = []
    for cand in candidates:
        ratio = SequenceMatcher(None, ground_truth.lower(), cand.lower()).ratio()
        snapped = _snap_score(ratio, allowed, 1.0)
        scores.append(round(snapped, 3))
    return scores


class JudgeResponse(BaseModel):
    scores: List[float]
    best_index: int
    worst_index: int
    rationale: Optional[str] = None


def _parse_judge_response(
    content: str,
    n: int,
    score_scale: float,
    allowed_scores: Optional[List[float]],
) -> Optional[JudgeResponse]:
    parsed = _extract_json(content)
    if not parsed:
        return None
    if isinstance(parsed, dict) and isinstance(parsed.get("scores"), list):
        parsed["scores"] = _normalize_scores(parsed.get("scores"), n, score_scale, allowed_scores)
    try:
        model = JudgeResponse.model_validate(parsed)
    except ValidationError:
        return None
    if model.scores is None or len(model.scores) != n:
        return None
    return model


def _infer_scores_from_text(
    text: str,
    n: int,
    allowed_scores: Optional[List[float]],
) -> Optional[List[float]]:
    if not text:
        return None
    t = text.lower()
    patterns = [
        r"all candidates.*?scores?\s*(?:should|to)?\s*(?:all)?\s*get\s*([01](?:\.\d+)?)",
        r"all candidates.*?score\s*([01](?:\.\d+)?)",
        r"all scores?\s*are\s*([01](?:\.\d+)?)",
    ]
    for pat in patterns:
        match = re.search(pat, t, flags=re.S)
        if match:
            try:
                value = float(match.group(1))
                value = _snap_score(value, allowed_scores, 1.0)
                return [value for _ in range(n)]
            except Exception:
                continue
    # Heuristic: if text says all candidates are completely wrong/miss, assign 0
    if "all candidates" in t and any(k in t for k in ("completely wrong", "all wrong", "all incorrect", "all miss")):
        return [0.0 for _ in range(n)]
    return None


def _weight_from_margin(
    chosen_score: float,
    rejected_score: float,
    score_scale: float,
    min_weight: float,
    max_weight: float,
) -> float:
    margin = max(0.0, chosen_score - rejected_score)
    weight = margin / max(score_scale, 1.0)
    return max(min_weight, min(max_weight, weight))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank DPO generations with an LLM judge.")
    parser.add_argument("--input", required=True, help="Input JSON/JSONL with generations.")
    parser.add_argument("--output_ranked", default=None, help="Output ranked JSON path.")
    parser.add_argument("--output_pairs", default=None, help="Output DPO pairs JSONL path.")
    parser.add_argument("--checkpoint_path", default=None, help="Resume checkpoint JSON path.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api_url", default=DEFAULT_URL)
    parser.add_argument("--api_key_env", default="FIREWORKS_API_KEY")
    parser.add_argument(
        "--env_file",
        default="",
        help="Optional .env path (if empty, tries config/.env then .env).",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--score_scale", type=float, default=1.0)
    parser.add_argument("--min_margin", type=float, default=0.05)
    parser.add_argument("--all_zero_threshold", type=float, default=0.0)
    parser.add_argument("--min_weight", type=float, default=0.05)
    parser.add_argument("--max_weight", type=float, default=1.0)
    parser.add_argument("--weight_strategy", choices=["margin", "none"], default="margin")
    parser.add_argument("--pairing", choices=["top_bottom"], default="top_bottom")
    parser.add_argument("--truncate_chars", type=int, default=600)
    parser.add_argument("--max_reason_chars", type=int, default=120)
    parser.add_argument("--fallback", choices=["similarity", "skip"], default="similarity")
    parser.add_argument("--allowed_scores", default=DEFAULT_ALLOWED_SCORES)
    parser.add_argument("--max_retries", type=int, default=6)
    parser.add_argument("--retry_delay", type=float, default=1.0)
    parser.add_argument("--save_raw", action="store_true", help="Store raw judge response when parsing fails.")
    parser.add_argument("--raw_max_chars", type=int, default=800)
    args = parser.parse_args()

    if args.env_file:
        _load_env_file(args.env_file)
    else:
        for path in DEFAULT_ENV_FILES:
            _load_env_file(path)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key in env var {args.api_key_env}")

    def _parse_allowed_scores(value: str) -> List[float]:
        scores: List[float] = []
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                scores.append(float(part))
            except ValueError:
                continue
        scores = sorted(set(scores))
        return scores

    allowed_scores = _parse_allowed_scores(args.allowed_scores)
    if allowed_scores:
        args.score_scale = max(args.score_scale, max(allowed_scores))

    records = _load_records(args.input)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    if args.output_ranked is None:
        args.output_ranked = os.path.splitext(args.input)[0] + "_ranked.json"
    if args.output_pairs is None:
        args.output_pairs = os.path.splitext(args.input)[0] + "_dpo_pairs.jsonl"
    if args.checkpoint_path is None:
        args.checkpoint_path = os.path.splitext(args.input)[0] + "_rank_checkpoint.json"

    ranked_results: List[Dict[str, Any]] = []
    start_idx = args.start_idx

    if args.resume and os.path.exists(args.checkpoint_path):
        with open(args.checkpoint_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        start_idx = ckpt.get("last_idx", start_idx) + 1
        ranked_results = ckpt.get("results", [])
        print(f"Resuming from idx={start_idx}, loaded {len(ranked_results)} results.")

    for idx in range(start_idx, len(records)):
        item = records[idx]
        generations = item.get("generations", [])
        candidates_raw = [g.get("output", "") for g in generations]
        candidates = [_truncate(str(c), args.truncate_chars) for c in candidates_raw]

        prompt = str(item.get("prompt", "")).strip()
        ground_truth = item.get("ground_truth")
        ground_truth = str(ground_truth).strip() if ground_truth is not None else None
        category = item.get("prompt_category")
        category = str(category).strip() if category is not None else None

        parsed: Optional[Dict[str, Any]] = None
        scores: Optional[List[float]] = None
        error: Optional[str] = None

        last_error: Optional[str] = None
        last_response: Optional[str] = None
        for attempt in range(args.max_retries + 1):
            try:
                if attempt > 0 and last_response:
                    messages = _build_reformat_messages(
                        previous_response=last_response,
                        n_candidates=len(candidates),
                        score_scale=args.score_scale,
                        max_reason_chars=args.max_reason_chars,
                        allowed_scores=allowed_scores,
                    )
                else:
                    messages = _build_messages(
                        prompt=prompt,
                        ground_truth=ground_truth,
                        candidates=candidates,
                        category=category,
                        score_scale=args.score_scale,
                        max_reason_chars=args.max_reason_chars,
                        allowed_scores=allowed_scores,
                        strict=attempt > 0,
                        previous_response=last_response if attempt > 0 else None,
                    )
                content = _fireworks_chat(
                    url=args.api_url,
                    api_key=api_key,
                    model=args.model,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
                last_response = content
                model = _parse_judge_response(
                    content=content,
                    n=len(candidates),
                    score_scale=args.score_scale,
                    allowed_scores=allowed_scores,
                )
                if model:
                    scores = model.scores
                    parsed = model.model_dump()
                    last_error = None
                    break
                last_error = "parse_failed"
            except Exception as exc:
                last_error = str(exc)
            if attempt < args.max_retries and args.retry_delay > 0:
                time.sleep(args.retry_delay)
        raw_for_log = None
        if last_error:
            error = last_error
            if args.save_raw and last_response:
                raw_for_log = _truncate(last_response, args.raw_max_chars)

        if scores is None:
            inferred = _infer_scores_from_text(last_response or "", len(candidates), allowed_scores)
            if inferred is not None:
                scores = inferred
                parsed = {"scores": scores, "rationale": "fallback_inferred_text"}
                if error is None:
                    error = "parse_failed_inferred"
            elif args.fallback == "similarity" and ground_truth:
                scores = _similarity_scores(ground_truth, candidates, allowed_scores)
                parsed = {"scores": scores, "rationale": "fallback_similarity"}
            else:
                scores = [0.0 for _ in candidates]
                parsed = {"scores": scores, "rationale": "fallback_zero"}
                if error is None:
                    error = "Failed to parse judge response"

        best_index = parsed.get("best_index") if isinstance(parsed, dict) else None
        worst_index = parsed.get("worst_index") if isinstance(parsed, dict) else None
        if not isinstance(best_index, int) or not isinstance(worst_index, int):
            best_index, worst_index = _argmax_min(scores)
        all_zero = max(scores) <= args.all_zero_threshold

        ranked = {
            "idx": item.get("idx", idx),
            "prompt": prompt,
            "prompt_category": category,
            "waveform_path": item.get("waveform_path"),
            "ground_truth": ground_truth,
            "scores": scores,
            "best_index": best_index,
            "worst_index": worst_index,
            "all_zero": all_zero,
            "rationale": parsed.get("rationale") if isinstance(parsed, dict) else None,
            "error": error,
            "raw_response": raw_for_log,
        }
        ranked_results.append(ranked)

        if args.sleep > 0:
            time.sleep(args.sleep)

        if (idx + 1) % args.save_interval == 0:
            _save_json(args.checkpoint_path, {"last_idx": idx, "results": ranked_results})
            _save_json(args.output_ranked, ranked_results)
            print(f"Saved checkpoint at idx={idx}")

    _save_json(args.output_ranked, ranked_results)

    # Build DPO pairs
    pairs: List[Dict[str, Any]] = []
    for item, ranked in zip(records, ranked_results):
        generations = item.get("generations", [])
        if not generations:
            continue
        scores = ranked.get("scores") or []
        if len(scores) != len(generations):
            continue
        best = int(ranked.get("best_index", 0))
        worst = int(ranked.get("worst_index", 0))
        all_zero = bool(ranked.get("all_zero", False))

        chosen_source = "generation"
        if all_zero and item.get("ground_truth"):
            chosen = str(item.get("ground_truth"))
            chosen_score = float(args.score_scale)
            chosen_source = "ground_truth"
            # Use best candidate as rejected when all are zero
            rejected = generations[best].get("output", "")
            rejected_score = float(scores[best])
        else:
            if best == worst:
                continue
            chosen = generations[best].get("output", "")
            rejected = generations[worst].get("output", "")
            chosen_score = float(scores[best])
            rejected_score = float(scores[worst])
            if (chosen_score - rejected_score) < args.min_margin:
                continue

        if args.weight_strategy == "margin":
            weight = _weight_from_margin(
                chosen_score=chosen_score,
                rejected_score=rejected_score,
                score_scale=float(args.score_scale),
                min_weight=args.min_weight,
                max_weight=args.max_weight,
            )
        else:
            weight = 1.0

        pairs.append({
            "idx": item.get("idx"),
            "waveform_path": item.get("waveform_path"),
            "prompt": item.get("prompt"),
            "prompt_category": item.get("prompt_category"),
            "ground_truth": item.get("ground_truth"),
            "chosen": chosen,
            "rejected": rejected,
            "chosen_idx": None if chosen_source == "ground_truth" else best,
            "rejected_idx": best if chosen_source == "ground_truth" else worst,
            "chosen_score": chosen_score,
            "rejected_score": rejected_score,
            "weight": weight,
            "chosen_source": chosen_source,
            "all_zero": all_zero,
        })

    _write_jsonl(args.output_pairs, pairs)
    print(f"Wrote {len(pairs)} DPO pairs to {args.output_pairs}")


if __name__ == "__main__":
    main()
