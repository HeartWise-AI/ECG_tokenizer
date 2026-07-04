#!/usr/bin/env python3
"""
Generate prompt variations per ORIGINAL prompt (not per category).
Output: {category: {original_prompt: [variations]}}

This ensures each variation is semantically equivalent to its source prompt,
avoiding the misassignment problem of category-level generation.
"""

import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, List

import aiohttp
import pandas as pd

FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
MODEL = "accounts/fireworks/models/minimax-m2p5"
CONCURRENCY = 10
MAX_RETRIES = 3
TIMEOUT_S = 60

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_variations_per_prompt.json")


def get_original_prompts(parquet_path: str) -> Dict[str, List[str]]:
    """Extract unique prompts per category from training parquet."""
    df = pd.read_parquet(parquet_path, columns=["prompt", "prompt_category"])
    result = {}
    for cat in df["prompt_category"].unique():
        prompts = df[df["prompt_category"] == cat]["prompt"].dropna().unique().tolist()
        if prompts:
            result[cat] = prompts
    return result


async def call_minimax(
    session: aiohttp.ClientSession,
    api_key: str,
    semaphore: asyncio.Semaphore,
    prompt: str,
    category: str,
    n_variations: int = 30,
) -> List[str]:
    """Generate variations for a single prompt."""
    system_msg = (
        "You are an expert medical professional who rephrases clinical questions. "
        "Generate variations that ask the EXACT SAME clinical question but with different wording. "
        "Each variation must be semantically identical — it should elicit the same answer. "
        "Return a JSON array of strings, nothing else."
    )

    user_msg = f"""Generate exactly {n_variations} rephrasings of this ECG question from the "{category}" category:

"{prompt}"

Rules:
1. Each variation must ask the EXACT same clinical question
2. The answer to each variation must be identical to the original
3. Use different sentence structures, vocabulary, and tones (formal, conversational, clinical)
4. Do NOT change what is being asked — only HOW it is asked
5. Return a JSON array of strings

Output JSON array:"""

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 4096,
        "temperature": 0.85,
        "top_p": 0.95,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(MAX_RETRIES):
        async with semaphore:
            try:
                async with session.post(
                    FIREWORKS_URL, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=TIMEOUT_S)
                ) as resp:
                    data = await resp.json()
                    text = data["choices"][0]["message"]["content"].strip()
                    # Parse JSON array from response
                    match = re.search(r'\[.*\]', text, re.DOTALL)
                    if match:
                        items = json.loads(match.group())
                        return [s.strip() for s in items if isinstance(s, str) and s.strip()]
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"  FAILED after {MAX_RETRIES} attempts: {str(e)[:80]}")
                await asyncio.sleep(2 ** attempt)
    return []


async def main():
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        print("Set FIREWORKS_API_KEY environment variable")
        sys.exit(1)

    parquet_path = "output/combined_train_qa_m5000k_h5000k_weighted.parquet"
    print(f"Loading original prompts from {parquet_path}...")
    originals = get_original_prompts(parquet_path)

    total_prompts = sum(len(v) for v in originals.values())
    print(f"Found {total_prompts} unique prompts across {len(originals)} categories")

    # Skip categories where prompts are finding-specific and can't be paraphrased generically
    SKIP = {"random_finding_question", "ecg_interval"}

    semaphore = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY * 2)

    result: Dict[str, Dict[str, List[str]]] = {}

    async with aiohttp.ClientSession(connector=connector) as session:
        for cat in sorted(originals.keys()):
            if cat in SKIP:
                print(f"\n[SKIP] {cat} (finding-specific prompts)")
                continue

            prompts = originals[cat]
            print(f"\n=== {cat} ({len(prompts)} prompts) ===")
            result[cat] = {}

            # Launch all prompts for this category concurrently
            tasks = {}
            for prompt in prompts:
                tasks[prompt] = asyncio.create_task(
                    call_minimax(session, api_key, semaphore, prompt, cat, n_variations=30)
                )

            for prompt, task in tasks.items():
                variations = await task
                # Always include the original prompt itself
                all_vars = [prompt] + [v for v in variations if v.lower().strip() != prompt.lower().strip()]
                result[cat][prompt] = all_vars
                print(f"  [{len(all_vars):3d}] {prompt[:70]}")

    # Save
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    total_vars = sum(len(v) for grp in result.values() for v in grp.values())
    print(f"\nSaved {total_vars} total variations for {sum(len(g) for g in result.values())} prompts to {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
