#!/usr/bin/env python3
"""
Fast parallel ranking of DPO generations using an LLM judge (Fireworks).

Uses asyncio and aiohttp for concurrent API calls - much faster than sequential.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from pydantic import BaseModel, ValidationError
import ast


DEFAULT_MODEL = "accounts/fireworks/models/llama-v3p3-70b-instruct"
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
            if key and (key not in os.environ or not os.environ.get(key)):
                os.environ[key] = value


def _load_records(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
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
) -> List[Dict[str, str]]:
    system_msg = (
        "You are an expert cardiologist judging answer quality. "
        "Score answers for clinical accuracy and completeness relative to the ground truth. "
        "Penalize hallucinations or incorrect findings. Return ONLY valid JSON."
    )
    lines = [f"Question: {prompt.strip()}"]
    if category:
        lines.append(f"Category: {category.strip()}")
    if ground_truth:
        lines.append(f"Ground truth: {ground_truth.strip()}")
    lines.append("Candidates:")
    for i, cand in enumerate(candidates):
        lines.append(f"[{i}] {cand}")

    schema_hint = '{"scores": [0.0, ...], "best_index": 0, "worst_index": 1, "rationale": "..."}'
    lines.append(
        f"Return JSON ONLY with keys: \"scores\" (list of {len(candidates)} numbers from 0-{score_scale}), "
        f"\"best_index\", \"worst_index\", and \"rationale\" (<= {max_reason_chars} chars). "
        f"Schema: {schema_hint}"
    )
    if allowed_scores:
        allowed_str = ", ".join(f"{x:.2f}".rstrip("0").rstrip(".") for x in allowed_scores)
        lines.append(f"Use ONLY these scores: {allowed_str}.")
    lines.append(
        f"Scoring: 0.0 = completely wrong, 0.5 = partially correct, {score_scale} = perfect match."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "\n".join(lines)},
    ]


async def _fireworks_chat_async(
    session: aiohttp.ClientSession,
    url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> str:
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    async with semaphore:
        async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            resp.raise_for_status()
            data = await resp.json()
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


async def process_item(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    item: Dict[str, Any],
    idx: int,
    args: argparse.Namespace,
    api_key: str,
    allowed_scores: List[float],
) -> Dict[str, Any]:
    """Process a single item asynchronously."""
    generations = item.get("generations", [])
    candidates_raw = [g.get("output", "") for g in generations]
    candidates = [_truncate(str(c), args.truncate_chars) for c in candidates_raw]

    prompt = str(item.get("prompt", "")).strip()
    ground_truth = item.get("ground_truth")
    ground_truth = str(ground_truth).strip() if ground_truth is not None else None
    category = item.get("prompt_category")
    category = str(category).strip() if category is not None else None

    scores: Optional[List[float]] = None
    error: Optional[str] = None
    parsed: Optional[Dict[str, Any]] = None
    last_response: Optional[str] = None

    for attempt in range(args.max_retries + 1):
        try:
            messages = _build_messages(
                prompt=prompt,
                ground_truth=ground_truth,
                candidates=candidates,
                category=category,
                score_scale=args.score_scale,
                max_reason_chars=args.max_reason_chars,
                allowed_scores=allowed_scores,
            )
            content = await _fireworks_chat_async(
                session=session,
                url=args.api_url,
                api_key=api_key,
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                semaphore=semaphore,
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
                error = None
                break
            error = "parse_failed"
        except Exception as exc:
            error = str(exc)
        if attempt < args.max_retries:
            await asyncio.sleep(args.retry_delay)

    # Fallback handling
    if scores is None:
        if args.fallback == "similarity" and ground_truth:
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

    return {
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
    }


async def process_batch(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    items: List[Tuple[int, Dict[str, Any]]],
    args: argparse.Namespace,
    api_key: str,
    allowed_scores: List[float],
) -> List[Dict[str, Any]]:
    """Process a batch of items concurrently."""
    tasks = [
        process_item(session, semaphore, item, idx, args, api_key, allowed_scores)
        for idx, item in items
    ]
    return await asyncio.gather(*tasks, return_exceptions=True)


async def main_async(args: argparse.Namespace) -> None:
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
            if part:
                try:
                    scores.append(float(part))
                except ValueError:
                    continue
        return sorted(set(scores))

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

    # Resume from checkpoint
    ranked_results: List[Dict[str, Any]] = []
    processed_indices: set = set()
    start_idx = args.start_idx

    if args.resume and os.path.exists(args.checkpoint_path):
        with open(args.checkpoint_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        ranked_results = ckpt.get("results", [])
        processed_indices = {r["idx"] for r in ranked_results}
        print(f"Resuming: loaded {len(ranked_results)} results from checkpoint.")

    # Filter out already processed items
    items_to_process = [
        (i, records[i]) for i in range(start_idx, len(records))
        if records[i].get("idx", i) not in processed_indices
    ]
    print(f"Total records: {len(records)}, To process: {len(items_to_process)}, Concurrency: {args.concurrency}")

    semaphore = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)

    async with aiohttp.ClientSession(connector=connector) as session:
        batch_size = args.save_interval
        for batch_start in range(0, len(items_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(items_to_process))
            batch = items_to_process[batch_start:batch_end]

            start_time = time.time()
            results = await process_batch(session, semaphore, batch, args, api_key, allowed_scores)
            elapsed = time.time() - start_time

            for result in results:
                if isinstance(result, Exception):
                    print(f"Error: {result}")
                    continue
                ranked_results.append(result)

            # Save checkpoint
            _save_json(args.checkpoint_path, {"results": ranked_results})
            _save_json(args.output_ranked, ranked_results)

            processed = batch_start + len(batch)
            rate = len(batch) / elapsed if elapsed > 0 else 0
            print(f"Processed {processed}/{len(items_to_process)} ({rate:.1f} samples/sec)")

    _save_json(args.output_ranked, ranked_results)

    # Build DPO pairs
    pairs: List[Dict[str, Any]] = []
    idx_to_ranked = {r["idx"]: r for r in ranked_results}

    for item in records:
        item_idx = item.get("idx")
        ranked = idx_to_ranked.get(item_idx)
        if not ranked:
            continue

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
            # Use ground truth as chosen when all scores are zero
            chosen = str(item.get("ground_truth"))
            chosen_score = float(args.score_scale)
            chosen_source = "ground_truth"
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
    print(f"All-zero pairs using ground truth: {sum(1 for p in pairs if p['chosen_source'] == 'ground_truth')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast parallel ranking of DPO generations.")
    parser.add_argument("--input", required=True, help="Input JSON/JSONL with generations.")
    parser.add_argument("--output_ranked", default=None, help="Output ranked JSON path.")
    parser.add_argument("--output_pairs", default=None, help="Output DPO pairs JSONL path.")
    parser.add_argument("--checkpoint_path", default=None, help="Resume checkpoint JSON path.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=100, help="Save checkpoint every N samples")
    parser.add_argument("--concurrency", type=int, default=20, help="Number of concurrent API calls")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api_url", default=DEFAULT_URL)
    parser.add_argument("--api_key_env", default="FIREWORKS_API_KEY")
    parser.add_argument("--env_file", default="", help="Optional .env path")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--score_scale", type=float, default=1.0)
    parser.add_argument("--min_margin", type=float, default=0.05)
    parser.add_argument("--all_zero_threshold", type=float, default=0.0)
    parser.add_argument("--min_weight", type=float, default=0.05)
    parser.add_argument("--max_weight", type=float, default=1.0)
    parser.add_argument("--weight_strategy", choices=["margin", "none"], default="margin")
    parser.add_argument("--truncate_chars", type=int, default=600)
    parser.add_argument("--max_reason_chars", type=int, default=120)
    parser.add_argument("--fallback", choices=["similarity", "skip"], default="similarity")
    parser.add_argument("--allowed_scores", default=DEFAULT_ALLOWED_SCORES)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--retry_delay", type=float, default=0.5)
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
