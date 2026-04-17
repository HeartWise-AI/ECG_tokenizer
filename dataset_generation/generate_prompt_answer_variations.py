#!/usr/bin/env python3
"""
Generate diverse prompt and answer variations using MiniMax M2.5 via Fireworks API.

Expands the hardcoded prompt lists in ecg_prompt_maker.py from ~5-15 per category
to 50-100, and creates answer phrasing templates for repetitive category answers.

Usage:
    export FIREWORKS_API_KEY=fw_...
    python dataset_generation/generate_prompt_answer_variations.py
"""

import asyncio
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import aiohttp

# ---------------------------------------------------------------------------
# Fireworks / MiniMax config
# ---------------------------------------------------------------------------
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
MODEL = "accounts/fireworks/models/minimax-m2p5"
CONCURRENCY = 10
MAX_RETRIES = 3
TIMEOUT_S = 120

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
PROMPT_VARIATIONS_PATH = os.path.join(OUT_DIR, "prompt_variations.json")
ANSWER_VARIATIONS_PATH = os.path.join(OUT_DIR, "answer_variations.json")

# ---------------------------------------------------------------------------
# Current prompt lists (extracted from ecg_prompt_maker.py)
# ---------------------------------------------------------------------------
CURRENT_PROMPTS: Dict[str, List[str]] = {
    "interpretation": [
        "What does this ECG show?",
        "Can you interpret this ECG?",
        "What are the findings in this ECG?",
        "Please describe the ECG findings.",
        "What abnormalities are present in this ECG?",
        "Can you provide an ECG interpretation?",
        "What is your analysis of this ECG?",
        "What do you see in this ECG recording?",
        "Can you read this ECG for me?",
        "What are the key findings in this electrocardiogram?",
        "Please provide a complete ECG interpretation.",
        "What is your ECG diagnosis?",
        "Describe all ECG abnormalities present.",
        "What does this electrocardiogram reveal?",
        "Can you analyze this ECG tracing?",
    ],
    "classification": [
        "Is this ECG normal or abnormal?",
        "Is this a normal ECG?",
        "Does this ECG show any pathology?",
        "Is this ECG concerning?",
        "Would you classify this ECG as normal or abnormal?",
        "Are there any abnormal findings in this ECG?",
        "Is this ECG within normal limits?",
        "Does this ECG require follow-up?",
        "Is there anything wrong with this ECG?",
        "Should I be concerned about this ECG?",
        "How would you classify this ECG?",
        "Is this ECG pathological?",
        "Does this ECG show borderline changes?",
        "Is urgent action needed for this ECG?",
        "What is the overall ECG assessment?",
    ],
    "category_rhythm": [
        "What is the rhythm in this ECG?",
        "Is there any arrhythmia present?",
        "What is the heart rate and rhythm?",
        "Are there any rhythm abnormalities?",
        "Is the rhythm regular or irregular?",
        "Can you identify the cardiac rhythm?",
        "What type of rhythm is shown?",
        "Is this sinus rhythm or something else?",
        "Are there any ectopic beats?",
    ],
    "category_conduction": [
        "Are there any conduction abnormalities?",
        "Is there any heart block present?",
        "Are there any bundle branch blocks?",
        "Is AV conduction normal?",
        "Can you assess the conduction system?",
        "Is there any conduction delay?",
        "What type of block is present if any?",
        "Is there evidence of pre-excitation?",
        "Are there any fascicular blocks?",
        "Is there AV dissociation?",
    ],
    "category_infarct_ischemia": [
        "Are there signs of ischemia or infarction?",
        "Is there any ST elevation or depression?",
        "Are there pathological Q waves?",
        "Does this ECG show acute MI?",
        "Are there any ischemic changes?",
        "Can you identify any signs of myocardial injury?",
        "What do the ST segments show?",
        "Is there evidence of old or new infarction?",
        "Are the T waves normal?",
    ],
    "category_chamber_enlargement": [
        "Is there chamber enlargement or hypertrophy?",
        "Are there signs of LVH or RVH?",
        "Is there atrial enlargement?",
        "Are the QRS voltages normal?",
        "Does this show ventricular hypertrophy?",
        "Can you assess for chamber abnormalities?",
        "Is there evidence of atrial abnormality?",
        "What do the voltages suggest?",
        "Are there signs of ventricular strain?",
        "Is there biatrial or biventricular enlargement?",
    ],
    "category_pericarditis": [
        "Is there evidence of pericarditis?",
        "Are there signs of pericardial inflammation?",
        "Does this ECG suggest pericarditis?",
        "Is there diffuse ST elevation?",
        "Are there pericarditic changes?",
    ],
    "category_other": [
        "Are there any nonspecific ST-T changes?",
        "Is there early repolarization?",
        "What other findings are present?",
        "Are the T waves normal in morphology?",
        "Is there anything else abnormal?",
    ],
    "json_interpretation": [
        "Output JSON ONLY with keys: {keys}. Values must be lists of present findings (omit missing categories).",
        "Provide structured JSON output for this ECG analysis using keys {keys}.",
        "Return ECG findings as JSON with keys {keys}.",
        "Generate JSON representation of ECG abnormalities. Keys: {keys}.",
        "Output ECG interpretation in JSON format only (keys {keys}).",
    ],
    "structural_heart_disease": [
        "Does this patient have structural heart disease?",
        "Is there evidence of structural heart disease in this patient?",
        "Based on echocardiography data, does the patient have structural heart disease?",
    ],
    "lvef": [
        "What is the patient's left ventricular ejection fraction?",
        "What is the LVEF based on echocardiography?",
        "What is the ejection fraction?",
        "Can you tell me the patient's EF?",
        "What is the left ventricular function?",
    ],
    "afib_risk": [
        "What is the patient's risk of developing atrial fibrillation?",
        "Is this patient at risk for incident AFib?",
        "What is the likelihood of developing AFib in the next 2-5 years?",
        "Will this patient develop atrial fibrillation in the future?",
        "What is the patient's future AFib risk?",
    ],
    "acs_severity": [
        "IS there an acute coornary occlusion/ if yes is it complete or incomplete and what is the culprit?",
        "Does this patient have an acute coronary occlusion? If present, specify whether it is complete or incomplete and identify the culprit artery.",
        "Is there evidence of an acute coronary occlusion? Please state the completeness and the likely culprit artery.",
        "Is this an acute coronary occlusion? If yes, indicate whether it is complete or incomplete and name the culprit artery.",
    ],
    "culprit_artery": [
        "What is the culprit artery?",
        "Which coronary artery is occluded?",
        "What is the location of the coronary occlusion?",
        "Which vessel is the culprit for this acute coronary syndrome?",
    ],
    "ecg_interval": [
        "What is the heart rate?",
        "What is the patient's heart rate?",
        "What is the ventricular rate?",
        "Can you tell me the heart rate in bpm?",
        "What is the HR?",
        "How fast is the heart beating?",
        "What's the pulse rate?",
        "What is the PR interval?",
        "What is the QT interval?",
        "What is the corrected QT interval (QTc)?",
        "Is the QT interval prolonged?",
        "What are the PR and QT intervals?",
        "Are the intervals normal or abnormal?",
        "What is the QTc using Bazett's formula?",
        "What is the QTc using Fridericia's formula?",
        "Is there QT prolongation present?",
    ],
    "localization_q_wave": [
        "Are there any Q waves present?",
        "Is there evidence of Q waves?",
        "Where are Q waves located?",
        "Can you identify pathological Q waves?",
        "Are there Q waves suggesting old infarction?",
    ],
    "localization_st_elevation": [
        "Is there ST elevation?",
        "Where is ST elevation present?",
        "Are there ST elevations in any leads?",
        "Can you identify ST segment elevation?",
        "Is there evidence of acute ST elevation?",
    ],
    "localization_st_depression": [
        "Is there ST depression?",
        "Where is ST depression located?",
        "Are there ST depressions in any leads?",
        "Can you identify ST segment depression?",
        "Is there evidence of ST depression?",
    ],
    "localization_t_wave": [
        "Are there T wave abnormalities?",
        "Where are T wave inversions?",
        "Is there T wave inversion present?",
        "Can you identify T wave changes?",
        "Are the T waves abnormal in any leads?",
    ],
    "localization_qrs_axis": [
        "What is the QRS axis?",
        "Is there axis deviation?",
        "What is the electrical axis of the heart?",
        "Is the QRS axis normal or deviated?",
        "What type of axis deviation is present?",
        "Is there left or right axis deviation?",
        "What is the frontal plane QRS axis?",
    ],
    "random_finding_question": [
        "Is there 1st degree AV block?",
        "Is there 2nd degree AV block?",
        "Is there 3rd degree AV block?",
        "Is there left bundle branch block?",
        "Is there right bundle branch block?",
        "Is there left anterior fascicular block?",
        "Is there left posterior fascicular block?",
        "Is there left ventricular hypertrophy?",
        "Is there right ventricular hypertrophy?",
        "Is there left atrial enlargement?",
        "Is there right atrial enlargement?",
        "Is there atrial fibrillation?",
        "Is there atrial flutter?",
        "Is there sinus bradycardia?",
        "Is there sinus tachycardia?",
        "Is there ventricular tachycardia?",
        "Is there premature ventricular complex?",
        "Is there premature atrial complex?",
        "Is there ST elevation in any leads?",
        "Is there ST depression in any leads?",
        "Are there pathological Q waves?",
        "Is there T wave inversion?",
        "Is there left axis deviation?",
        "Is there right axis deviation?",
        "Is there extreme axis deviation?",
        "Is there pericarditis?",
        "Is there early repolarization?",
        "Is there WPW pattern?",
        "Is there long QT syndrome?",
        "Is there short QT syndrome?",
    ],
    "urgency_assessment": [
        "Does this ECG require immediate intervention?",
        "Is this an emergency ECG finding?",
        "What is the clinical urgency of this ECG?",
    ],
    "interpretation_complex": [
        "What are all the abnormalities in this complex ECG?",
        "Can you list all findings in this multi-pathology ECG?",
        "Please provide a comprehensive analysis of this abnormal ECG.",
    ],
}

# ---------------------------------------------------------------------------
# Current answer templates to diversify
# ---------------------------------------------------------------------------
CURRENT_ANSWER_TEMPLATES: Dict[str, List[str]] = {
    # Binary yes/no patterns used in category-specific answers
    "yes_prefix": [
        "Yes -",
    ],
    "no_prefix": [
        "No -",
    ],
    # Classification answer patterns
    "normal_ecg": [
        "Normal ECG; No significant abnormalities detected",
    ],
    "abnormal_ecg": [
        "Abnormal ECG; Pathological findings:",
    ],
    "borderline_ecg": [
        "Borderline ECG; Minor findings:",
    ],
    # Category rhythm no-finding answers
    "no_rhythm_abnormality": [
        "No - no ectopic beats present",
        "Yes - sinus rhythm",
        "No - no evidence of arrhythmia",
    ],
    # Category conduction no-finding answers
    "no_conduction_abnormality": [
        "No - no evidence of pre-excitation",
        "No - no evidence of heart block",
        "No - no conduction abnormalities",
    ],
    # Structural heart disease answers
    "no_shd": [
        "No evidence of structural heart disease",
    ],
    "yes_shd": [
        "Evidence of structural heart disease",
    ],
    # AFib risk answers
    "low_afib_risk": [
        "Low risk of developing atrial fibrillation",
    ],
    "high_afib_risk": [
        "Elevated risk of developing atrial fibrillation",
    ],
}

# Target counts per category (~10,435 total)
TARGET_COUNTS: Dict[str, int] = {
    "interpretation": 700,
    "classification": 700,
    "category_rhythm": 600,
    "category_conduction": 600,
    "category_infarct_ischemia": 600,
    "category_chamber_enlargement": 600,
    "category_pericarditis": 400,
    "category_other": 400,
    "json_interpretation": 400,
    "structural_heart_disease": 400,
    "lvef": 400,
    "afib_risk": 400,
    "acs_severity": 400,
    "culprit_artery": 400,
    "ecg_interval": 600,
    "localization_q_wave": 400,
    "localization_st_elevation": 400,
    "localization_st_depression": 400,
    "localization_t_wave": 400,
    "localization_qrs_axis": 400,
    "random_finding_question": 700,
    "urgency_assessment": 400,
    "interpretation_complex": 400,
}


# ---------------------------------------------------------------------------
# Async API caller
# ---------------------------------------------------------------------------
async def call_minimax(
    session: aiohttp.ClientSession,
    api_key: str,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.85,
    max_tokens: int = 8192,
) -> str:
    """Call MiniMax M2.5 via Fireworks with retries."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                async with session.post(
                    FIREWORKS_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=TIMEOUT_S),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"  ERROR after {MAX_RETRIES} attempts: {e}")
                return "[]"
            await asyncio.sleep(2 ** attempt)
    return "[]"


def parse_json_array(text: str) -> List[str]:
    """Extract JSON array from response, handling markdown wrapping."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(x) for x in result if x]
        return []
    except json.JSONDecodeError:
        # Try to find array in text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return [str(x) for x in result if x]
            except json.JSONDecodeError:
                pass
        return []


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_PROMPTS = """You are an expert medical AI assistant specializing in cardiology and ECG interpretation.
Your task is to generate diverse, natural-sounding question variations for an ECG analysis training dataset.
Each variation must preserve the exact clinical intent of the original questions but use different:
- Sentence structures (questions, imperatives, requests)
- Vocabulary (formal medical vs conversational)
- Levels of specificity
- Phrasings (active/passive, direct/indirect)
Return ONLY a JSON array of strings. No explanations, no markdown outside the array."""

SYSTEM_PROMPT_ANSWERS = """You are an expert cardiologist generating diverse answer phrasing templates for ECG reports.
Each template must preserve clinical accuracy but vary the wording, structure, and level of detail.
Return ONLY a JSON array of strings. No explanations, no markdown outside the array."""


def _build_prompt_generation_request(category: str, examples: List[str], target: int) -> str:
    """Build the user prompt for generating prompt variations."""
    # Special handling for json_interpretation - explain the {keys} placeholder
    extra_instructions = ""
    if category == "json_interpretation":
        extra_instructions = (
            "\n\nIMPORTANT: Each prompt MUST contain the placeholder '{keys}' exactly as written "
            "(with curly braces) where the list of JSON keys would go. The placeholder will be "
            "filled in programmatically later. Vary the instruction around the placeholder."
        )
    elif category == "random_finding_question":
        extra_instructions = (
            "\n\nThese are YES/NO questions about specific ECG findings. Generate variations "
            "that ask about the SAME findings but with different phrasing. Include questions like "
            "'Does this ECG demonstrate...', 'Can you detect...', 'Is there evidence of...', "
            "'Do you see any signs of...', etc. Cover all common ECG findings: blocks, arrhythmias, "
            "hypertrophy, enlargement, ischemic changes, axis deviations, pericarditis, WPW, etc."
        )

    return f"""Generate exactly {target} unique question variations for the ECG prompt category: "{category}"

Here are the existing {len(examples)} examples to use as reference for the clinical intent:
{json.dumps(examples, indent=2)}

Generate {target} NEW unique questions that:
1. Preserve the same clinical intent as the examples above
2. Use diverse sentence structures, vocabulary, and formality levels
3. Mix formal medical language with conversational phrasing
4. Include imperatives ("Describe...", "Assess..."), questions ("What...", "Is there..."), and requests ("Please...", "Could you...")
5. Do NOT repeat any of the existing examples exactly
6. Each question should be self-contained and clear{extra_instructions}

Return a JSON array of {target} strings."""


def _build_answer_generation_request(category: str, examples: List[str], target: int) -> str:
    """Build the user prompt for generating answer phrasing variations."""
    return f"""Generate exactly {target} unique answer phrasing templates for the ECG answer category: "{category}"

Here are the existing examples:
{json.dumps(examples, indent=2)}

Generate {target} NEW unique phrasing variations that:
1. Preserve the same clinical meaning
2. Use different sentence structures and vocabulary
3. Mix formal and conversational tones
4. Maintain clinical accuracy
5. For binary answers (yes/no), include diverse prefix phrasings
6. For classification answers, vary the structure while keeping the diagnostic meaning

Return a JSON array of {target} strings."""


BATCH_SIZE = 80  # Request ~80 per API call for reliable generation


async def _generate_for_category(
    session: aiohttp.ClientSession,
    api_key: str,
    semaphore: asyncio.Semaphore,
    category: str,
    seed_examples: List[str],
    target: int,
    system_prompt: str,
    build_request_fn,
) -> List[str]:
    """Generate variations for a single category, making multiple batch calls until target is reached."""
    all_items = list(seed_examples)
    seen = {p.lower().strip() for p in all_items}

    round_num = 0
    max_rounds = 20  # Safety limit
    while len(all_items) < target and round_num < max_rounds:
        round_num += 1
        remaining = target - len(all_items)
        batch = min(BATCH_SIZE, remaining)

        # Use a random sample of existing items as examples to encourage diversity
        if len(all_items) > 15:
            import random
            example_pool = random.sample(all_items, min(15, len(all_items)))
        else:
            example_pool = all_items

        user_prompt = build_request_fn(category, example_pool, batch)
        # Add instruction to avoid duplicates of what we already have
        if round_num > 1:
            user_prompt += (
                f"\n\nIMPORTANT: We already have {len(all_items)} variations. "
                f"Generate {batch} MORE that are completely different from all previous ones. "
                f"Be creative with sentence structure, vocabulary, and phrasing style."
            )

        response = await call_minimax(
            session, api_key, semaphore, system_prompt, user_prompt, temperature=0.9
        )
        new_items = parse_json_array(response)

        added = 0
        for item in new_items:
            item_clean = item.strip()
            if item_clean and item_clean.lower() not in seen:
                all_items.append(item_clean)
                seen.add(item_clean.lower())
                added += 1

        if added == 0:
            # No new unique items generated, break to avoid infinite loop
            break

    return all_items


async def generate_all_variations(api_key: str) -> None:
    """Main async function to generate all prompt and answer variations."""
    semaphore = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY * 2)

    async with aiohttp.ClientSession(connector=connector) as session:
        # --- Generate prompt variations ---
        print("=" * 60)
        print("GENERATING PROMPT VARIATIONS")
        print("=" * 60)

        # Launch all categories concurrently
        prompt_tasks = {}
        for category, examples in CURRENT_PROMPTS.items():
            target = TARGET_COUNTS.get(category, 400)
            print(f"  Launching {category} (target: {target})...")
            prompt_tasks[category] = asyncio.create_task(
                _generate_for_category(
                    session, api_key, semaphore,
                    category, examples, target,
                    SYSTEM_PROMPT_PROMPTS, _build_prompt_generation_request,
                )
            )

        prompt_results: Dict[str, List[str]] = {}
        for category, task in prompt_tasks.items():
            result = await task
            prompt_results[category] = result
            target = TARGET_COUNTS.get(category, 400)
            orig = len(CURRENT_PROMPTS[category])
            print(f"  {category}: {orig} -> {len(result)} prompts (target: {target})")

        # Save prompt variations
        with open(PROMPT_VARIATIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(prompt_results, f, indent=2, ensure_ascii=False)
        total_prompts = sum(len(v) for v in prompt_results.values())
        print(f"\nSaved {total_prompts} prompt variations to {PROMPT_VARIATIONS_PATH}")

        # --- Generate answer variations ---
        print("\n" + "=" * 60)
        print("GENERATING ANSWER VARIATIONS")
        print("=" * 60)

        answer_tasks = {}
        for category, examples in CURRENT_ANSWER_TEMPLATES.items():
            target = 100  # 100 variations per answer template type
            print(f"  Launching {category} (target: {target})...")
            answer_tasks[category] = asyncio.create_task(
                _generate_for_category(
                    session, api_key, semaphore,
                    category, examples, target,
                    SYSTEM_PROMPT_ANSWERS, _build_answer_generation_request,
                )
            )

        answer_results: Dict[str, List[str]] = {}
        for category, task in answer_tasks.items():
            result = await task
            answer_results[category] = result
            orig = len(CURRENT_ANSWER_TEMPLATES[category])
            print(f"  {category}: {orig} -> {len(result)} variations (target: 100)")

        # Save answer variations
        with open(ANSWER_VARIATIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(answer_results, f, indent=2, ensure_ascii=False)
        total_answers = sum(len(v) for v in answer_results.values())
        print(f"\nSaved {total_answers} answer variations to {ANSWER_VARIATIONS_PATH}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_prompts = sum(len(v) for v in prompt_results.values())
    total_answers = sum(len(v) for v in answer_results.values())
    print(f"Total prompt variations: {total_prompts} across {len(prompt_results)} categories")
    print(f"Total answer variations: {total_answers} across {len(answer_results)} categories")


def main():
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        print("ERROR: FIREWORKS_API_KEY not set. Export it first:")
        print("  export FIREWORKS_API_KEY=fw_...")
        sys.exit(1)

    print(f"Using API key: {api_key[:12]}...")
    print(f"Model: {MODEL}")
    print(f"Output: {PROMPT_VARIATIONS_PATH}")
    print(f"        {ANSWER_VARIATIONS_PATH}")
    print()

    asyncio.run(generate_all_variations(api_key))


if __name__ == "__main__":
    main()
