#!/usr/bin/env python3
"""
Generate QA pairs from the 784k DeepECG ECGs missing from our training data.

Creates 4 prompt types per ECG:
  1. Interpretation: "What is your analysis of this ECG?" → translated_diagnosis + HR
  2. Binary diagnosis: "Is {diagnosis} present?" → Yes/No (for positive labels)
  3. Heart rate / intervals: "What is the heart rate?" → "{HR} bpm"
  4. Category-level: "Are there any rhythm abnormalities?" → finding list

Sources:
  - /media/data1/muse_ge/ECG_ad20241231_metadata.v1.6._with_translation_ROXs42Bb.cleaned.parquet
  - /media/data1/muse_ge/ECG_ad20241231_gt_labels_v1.6.parquet

Usage:
    PYTHONPATH=/volume/ECG_tokenizer python dataset_generation/generate_deepecg_qa.py
"""

import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.constants import ECG_PATTERNS, DEEPECG_CATEGORIES

# ── Prompt templates ──

INTERP_QUESTIONS = [
    "What is your analysis of this ECG?",
    "What abnormalities are present in this ECG?",
    "Interpret this ECG.",
    "What does this ECG show?",
    "Can you interpret this ECG?",
]

HR_QUESTIONS = [
    "What is the heart rate?",
    "What is the ventricular rate?",
    "What is the heart rate in BPM?",
]

QT_QUESTIONS = [
    "What is the QT interval?",
    "What is the corrected QT interval?",
    "What are the QT and QTc intervals?",
]

QRS_QUESTIONS = [
    "What is the QRS duration?",
    "How wide is the QRS complex?",
]

AXIS_QUESTIONS = [
    "What is the QRS axis?",
    "Is the QRS axis normal or deviated?",
]

# Each question is paired with the SPECIFIC diagnoses that constitute a valid "Yes" answer.
# This prevents "Is there ST elevation?" → "Yes - Q wave" mismatches.
NORMAL_LABELS = {"Sinusal", "Regular", "Monomorph"}

TARGETED_QUESTIONS = [
    # ── RHYTHM ──
    {
        "questions": ["What is the heart rate and rhythm?", "Can you identify the cardiac rhythm?", "What is the rhythm in this ECG?"],
        "labels": ["Sinusal", "Afib", "Atrial flutter", "Atrial tachycardia (>= 100 BPM)", "Junctional rhythm",
                    "Supraventricular tachycardia", "Ventricular tachycardia", "Ventricular Rhythm",
                    "Ectopic atrial rhythm (< 100 BPM)", "Bradycardia"],
        "category": "category_rhythm",
        "include_hr": True,
        "answer_style": "descriptive",  # "Sinus rhythm (HR: 72 bpm)" not "Yes - Sinusal"
    },
    {
        "questions": ["Are there any ectopic beats?", "Are there premature complexes?"],
        "labels": ["Premature ventricular complex", "Premature atrial complex"],
        "category": "category_rhythm",
    },
    {
        "questions": ["Is the rhythm regular or irregular?"],
        "labels": ["Regular", "Irregularly irregular", "Regularly irregular"],
        "category": "category_rhythm",
    },
    # ── CONDUCTION: Bundle branch blocks ──
    {
        "questions": ["Are there any bundle branch blocks?", "Is there a bundle branch block?"],
        "labels": ["Left bundle branch block", "Right bundle branch block",
                    "Left anterior fascicular block", "Left posterior fascicular block"],
        "category": "category_conduction",
    },
    # ── CONDUCTION: AV blocks ──
    {
        "questions": ["Is there any heart block present?", "Is there AV block?"],
        "labels": ["1st degree AV block", "2nd degree AV block - mobitz 1", "2nd degree AV block - mobitz 2",
                    "Third Degree AV Block"],
        "category": "category_conduction",
    },
    # ── CONDUCTION: General ──
    {
        "questions": ["Are there any conduction abnormalities?", "Is AV conduction normal?"],
        "labels": ["Left bundle branch block", "Right bundle branch block", "Left anterior fascicular block",
                    "Left posterior fascicular block", "1st degree AV block", "2nd degree AV block - mobitz 1",
                    "2nd degree AV block - mobitz 2", "Third Degree AV Block",
                    "Nonspecific intraventricular conduction delay", "Delta wave",
                    "Wolff-Parkinson-White (Pre-excitation syndrome)", "Prolonged QT"],
        "category": "category_conduction",
    },
    # ── CONDUCTION: Pacing ──
    {
        "questions": ["Is there a pacemaker rhythm?", "Is there cardiac pacing?"],
        "labels": ["Ventricular paced", "Atrial paced", "LV pacing"],
        "category": "category_conduction",
    },
    # ── INFARCT: ST elevation only ──
    {
        "questions": ["Is there ST elevation?", "Is there acute ST elevation?"],
        "labels": ["ST elevation (anterior - V3-V4)", "ST elevation (septal - V1-V2)",
                    "ST elevation (inferior - II, III, aVF)", "ST elevation (lateral - I, aVL, V5-V6)",
                    "ST elevation (posterior - V7-V8-V9)"],
        "category": "category_infarct_ischemia",
    },
    # ── INFARCT: ST depression only ──
    {
        "questions": ["Is there ST depression?", "Is there ST segment depression?"],
        "labels": ["ST depression (inferior - II, III, aVF)", "ST depression (lateral - I, avL, V5-V6)",
                    "ST depression (anterior - V3-V4)", "ST depression (septal- V1-V2)"],
        "category": "category_infarct_ischemia",
    },
    # ── INFARCT: Q waves only ──
    {
        "questions": ["Are there pathological Q waves?", "Are there Q waves suggesting infarction?"],
        "labels": ["Q wave (inferior - II, III, aVF)", "Q wave (anterior - V3-V4)",
                    "Q wave (septal- V1-V2)", "Q wave (lateral- I, aVL, V5-V6)",
                    "Q wave (posterior - V7-V9)"],
        "category": "category_infarct_ischemia",
    },
    # ── INFARCT: Acute MI ──
    {
        "questions": ["Are there signs of acute myocardial infarction?", "Is this an acute MI?"],
        "labels": ["Acute MI", "ST elevation (anterior - V3-V4)", "ST elevation (septal - V1-V2)",
                    "ST elevation (inferior - II, III, aVF)", "ST elevation (lateral - I, aVL, V5-V6)"],
        "category": "category_infarct_ischemia",
    },
    # ── INFARCT: General ──
    {
        "questions": ["Are there any ischemic changes?", "Are there signs of ischemia or infarction?"],
        "labels": ["Acute MI", "ST elevation (anterior - V3-V4)", "ST elevation (septal - V1-V2)",
                    "ST elevation (inferior - II, III, aVF)", "ST elevation (lateral - I, aVL, V5-V6)",
                    "ST elevation (posterior - V7-V8-V9)",
                    "ST depression (inferior - II, III, aVF)", "ST depression (lateral - I, avL, V5-V6)",
                    "ST depression (anterior - V3-V4)", "ST depression (septal- V1-V2)",
                    "Q wave (inferior - II, III, aVF)", "Q wave (anterior - V3-V4)",
                    "Q wave (septal- V1-V2)", "Q wave (lateral- I, aVL, V5-V6)",
                    "Q wave (posterior - V7-V9)"],
        "category": "category_infarct_ischemia",
    },
    # ── CHAMBER ENLARGEMENT: Ventricular ──
    {
        "questions": ["Are there signs of LVH or RVH?", "Is there ventricular hypertrophy?"],
        "labels": ["Left ventricular hypertrophy", "Right ventricular hypertrophy"],
        "category": "category_chamber_enlargement",
    },
    # ── CHAMBER ENLARGEMENT: Atrial ──
    {
        "questions": ["Is there atrial enlargement?", "Is there atrial abnormality?"],
        "labels": ["Left atrial enlargement", "Right atrial enlargement", "Bi-atrial enlargement"],
        "category": "category_chamber_enlargement",
    },
    # ── PERICARDITIS ──
    {
        "questions": ["Is there evidence of pericarditis?"],
        "labels": ["Acute pericarditis"],
        "category": "category_pericarditis",
    },
    # ── OTHER: T wave inversions ──
    {
        "questions": ["Is there T wave inversion?", "Are there T wave abnormalities?"],
        "labels": ["T wave inversion (inferior - II, III, aVF)", "T wave inversion (lateral -I, aVL, V5-V6)",
                    "T wave inversion (anterior - V3-V4)", "T wave inversion (septal- V1-V2)"],
        "category": "category_other",
    },
    # ── OTHER: Low voltage / morphology ──
    {
        "questions": ["Are there any additional abnormalities?", "What other findings are present?"],
        "labels": ["Low voltage", "Early repolarization", "ST downslopping", "ST upslopping",
                    "Lead misplacement"],
        "category": "category_other",
    },
]

BINARY_QUESTION_TEMPLATES = [
    "Is {diag} present in this ECG? Answer Yes or No.",
    "Does this ECG show {diag}? Answer Yes or No.",
    "Is there evidence of {diag}? Answer Yes or No.",
]

YES_TEMPLATES = [
    "Yes - {diag} is present",
    "Yes - evidence of {diag}",
    "Yes - {diag} identified",
]

YES_TEMPLATES_HR = [
    "Yes - {diag} is present (HR: {hr} bpm)",
    "Yes - evidence of {diag} (HR: {hr} bpm)",
    "Yes - {diag} identified (HR: {hr} bpm)",
]

NO_TEMPLATES = [
    "No - {diag} is not present",
    "No - no evidence of {diag}",
    "No - {diag} not identified",
]

# Diagnoses that should always include HR in their Yes answer
HR_DIAGNOSES = {
    "Sinusal", "Afib", "Atrial flutter", "Atrial tachycardia (>= 100 BPM)",
    "Bradycardia", "Junctional rhythm", "Supraventricular tachycardia",
    "Ventricular tachycardia", "Ventricular Rhythm", "Ectopic atrial rhythm (< 100 BPM)",
    "Irregularly irregular", "Regularly irregular", "Premature ventricular complex",
    "Premature atrial complex", "Atrial paced", "Ventricular paced", "LV pacing",
}

# Reverse map: diagnosis → category
REV_CAT = {}
for cat, labels in DEEPECG_CATEGORIES.items():
    for l in labels:
        REV_CAT[l] = cat


def _is_positive(val) -> bool:
    try:
        return float(val) >= 1
    except (TypeError, ValueError):
        return False


def generate_deepecg_qa(
    metadata_path: str,
    labels_path: str,
    existing_train_path: str,
    output_path: str,
    max_per_prompt_type: int = 500000,
    max_binary_per_diag: int = 3000,
    seed: int = 42,
):
    rng = random.Random(seed)
    np.random.seed(seed)

    print("Loading metadata...")
    meta_cols = [
        "npy_path", "translated_diagnosis",
        "RestingECG_RestingECGMeasurements_VentricularRate",
        "RestingECG_RestingECGMeasurements_QTInterval",
        "RestingECG_RestingECGMeasurements_QTCorrected",
        "RestingECG_RestingECGMeasurements_QRSDuration",
        "RestingECG_RestingECGMeasurements_PRInterval",
        "RestingECG_RestingECGMeasurements_RAxis",
    ]
    meta = pd.read_parquet(metadata_path, columns=meta_cols)
    meta.rename(columns={
        "RestingECG_RestingECGMeasurements_VentricularRate": "hr",
        "RestingECG_RestingECGMeasurements_QTInterval": "qt",
        "RestingECG_RestingECGMeasurements_QTCorrected": "qtc",
        "RestingECG_RestingECGMeasurements_QRSDuration": "qrs",
        "RestingECG_RestingECGMeasurements_PRInterval": "pr",
        "RestingECG_RestingECGMeasurements_RAxis": "axis",
    }, inplace=True)

    print("Loading labels...")
    labels = pd.read_parquet(labels_path)

    print("Merging...")
    df = meta.merge(labels, on="npy_path", how="inner")
    df["npy_name"] = df["npy_path"].apply(lambda x: str(x).split("/")[-1])

    # Filter to rows with splits (train/test/val) only
    df = df[df["Split"].notna()].copy()
    print(f"  Total with splits: {len(df)}")

    # Exclude ECGs already in our training data
    print("Loading existing training paths...")
    existing = pd.read_parquet(existing_train_path, columns=["waveform_path_psa"])
    existing_names = set(str(p).split("/")[-1] for p in existing["waveform_path_psa"].unique())
    del existing

    mask = ~df["npy_name"].isin(existing_names)
    df = df[mask].reset_index(drop=True)
    print(f"  After excluding existing: {len(df)} ECGs")

    # Map GT columns to our label names
    gt_cols = {c.replace("GT_", ""): c for c in df.columns if c.startswith("GT_")}
    available_diags = [d for d in ECG_PATTERNS if d in gt_cols]
    print(f"  Available diagnoses: {len(available_diags)}")

    records: List[Dict[str, Any]] = []

    # ── Type 1: Interpretation ──
    print("\nGenerating interpretation prompts...")
    interp_idx = rng.sample(range(len(df)), min(max_per_prompt_type, len(df)))
    for i in interp_idx:
        row = df.iloc[i]
        dx = str(row.get("translated_diagnosis", "")).strip()
        if not dx or dx == "nan":
            continue
        hr = row.get("hr")
        if pd.notna(hr):
            answer = f"{dx} (HR: {int(float(hr))} bpm)"
        else:
            answer = dx
        records.append({
            "waveform_path_psa": row["npy_path"],
            "prompt": rng.choice(INTERP_QUESTIONS),
            "generated_answer": answer,
            "prompt_category": "interpretation",
        })
    print(f"  Interpretation: {len(records)}")

    # ── Type 2: Heart rate / intervals ──
    print("Generating HR/interval prompts...")
    n_before = len(records)
    hr_idx = rng.sample(range(len(df)), min(max_per_prompt_type, len(df)))
    for i in hr_idx:
        row = df.iloc[i]
        hr = row.get("hr")
        qt = row.get("qt")
        qtc = row.get("qtc")
        qrs = row.get("qrs")
        axis = row.get("axis")

        if pd.notna(hr):
            records.append({
                "waveform_path_psa": row["npy_path"],
                "prompt": rng.choice(HR_QUESTIONS),
                "generated_answer": f"The heart rate is {int(float(hr))} bpm",
                "prompt_category": "ecg_interval",
            })

        if pd.notna(qt) and pd.notna(qtc) and rng.random() < 0.3:
            records.append({
                "waveform_path_psa": row["npy_path"],
                "prompt": rng.choice(QT_QUESTIONS),
                "generated_answer": f"QT interval is {int(float(qt))} ms, QTc is {int(float(qtc))} ms",
                "prompt_category": "ecg_interval",
            })

        if pd.notna(qrs) and rng.random() < 0.2:
            records.append({
                "waveform_path_psa": row["npy_path"],
                "prompt": rng.choice(QRS_QUESTIONS),
                "generated_answer": f"QRS duration is {int(float(qrs))} ms",
                "prompt_category": "ecg_interval",
            })

        if pd.notna(axis) and rng.random() < 0.2:
            ax = int(float(axis))
            if -30 <= ax <= 90:
                ax_desc = "normal"
            elif ax < -30:
                ax_desc = "left axis deviation"
            elif ax > 90:
                ax_desc = "right axis deviation"
            else:
                ax_desc = "indeterminate"
            records.append({
                "waveform_path_psa": row["npy_path"],
                "prompt": rng.choice(AXIS_QUESTIONS),
                "generated_answer": f"QRS axis is {ax} degrees ({ax_desc})",
                "prompt_category": "ecg_interval",
            })
    print(f"  HR/interval: {len(records) - n_before}")

    # ── Type 3: Targeted category questions ──
    # Each question maps to SPECIFIC diagnoses that constitute a valid answer.
    print("Generating targeted category prompts...")
    n_before = len(records)
    cat_idx = rng.sample(range(len(df)), min(max_per_prompt_type, len(df)))

    # Rhythm descriptors for natural answers
    RHYTHM_NAMES = {
        "Sinusal": "Sinus rhythm", "Afib": "Atrial fibrillation",
        "Atrial flutter": "Atrial flutter", "Bradycardia": "Sinus bradycardia",
        "Atrial tachycardia (>= 100 BPM)": "Sinus tachycardia",
        "Junctional rhythm": "Junctional rhythm", "Ventricular tachycardia": "Ventricular tachycardia",
        "Supraventricular tachycardia": "Supraventricular tachycardia",
        "Ventricular Rhythm": "Ventricular rhythm",
        "Ectopic atrial rhythm (< 100 BPM)": "Ectopic atrial rhythm",
    }

    for i in cat_idx:
        row = df.iloc[i]
        # Pick a random targeted question group
        tq = rng.choice(TARGETED_QUESTIONS)
        question = rng.choice(tq["questions"])
        target_labels = tq["labels"]
        category = tq["category"]

        # Find which target labels are positive for this ECG
        positives = [d for d in target_labels
                     if d in gt_cols and _is_positive(row.get(gt_cols[d]))
                     and d not in NORMAL_LABELS]

        if tq.get("answer_style") == "descriptive" and tq.get("include_hr"):
            # Natural rhythm answer with HR
            hr = row.get("hr")
            rhythm_pos = [d for d in target_labels
                          if d in gt_cols and _is_positive(row.get(gt_cols[d]))]
            # Pick the most specific rhythm
            abnormal = [d for d in rhythm_pos if d not in NORMAL_LABELS]
            if abnormal:
                rhythm_name = RHYTHM_NAMES.get(abnormal[0], abnormal[0])
            elif "Sinusal" in rhythm_pos:
                rhythm_name = "Sinus rhythm"
            else:
                rhythm_name = "Normal rhythm"
            if pd.notna(hr):
                answer = f"{rhythm_name} (HR: {int(float(hr))} bpm)"
            else:
                answer = rhythm_name
        elif positives:
            answer = "Yes - " + "; ".join(positives[:3])
        else:
            answer = "No - no abnormalities in this category"

        records.append({
            "waveform_path_psa": row["npy_path"],
            "prompt": question,
            "generated_answer": answer,
            "prompt_category": category,
        })
    print(f"  Category: {len(records) - n_before}")

    # ── Type 4: Binary diagnosis Yes/No ──
    print("Generating binary diagnosis prompts...")
    n_before = len(records)
    for diag in available_diags:
        gc = gt_cols[diag]
        col = df[gc].fillna(0).astype(float)
        pos_idx = df.index[col >= 1].tolist()
        neg_idx = df.index[col < 1].tolist()

        n_pos = min(len(pos_idx), max_binary_per_diag)
        n_neg = min(len(neg_idx), n_pos)  # balanced
        n_pos = min(n_pos, n_neg)

        if n_pos == 0:
            continue

        sampled_pos = rng.sample(pos_idx, n_pos) if len(pos_idx) > n_pos else pos_idx[:n_pos]
        sampled_neg = rng.sample(neg_idx, n_neg) if len(neg_idx) > n_neg else neg_idx[:n_neg]

        use_hr = diag in HR_DIAGNOSES
        for idx in sampled_pos:
            row = df.iloc[idx]
            hr = row.get("hr")
            if use_hr and pd.notna(hr):
                answer = rng.choice(YES_TEMPLATES_HR).format(diag=diag, hr=int(float(hr)))
            else:
                answer = rng.choice(YES_TEMPLATES).format(diag=diag)
            records.append({
                "waveform_path_psa": row["npy_path"],
                "prompt": rng.choice(BINARY_QUESTION_TEMPLATES).format(diag=diag),
                "generated_answer": answer,
                "prompt_category": "binary_diagnosis",
            })
        for idx in sampled_neg:
            records.append({
                "waveform_path_psa": df.iloc[idx]["npy_path"],
                "prompt": rng.choice(BINARY_QUESTION_TEMPLATES).format(diag=diag),
                "generated_answer": rng.choice(NO_TEMPLATES).format(diag=diag),
                "prompt_category": "binary_diagnosis",
            })
    print(f"  Binary diagnosis: {len(records) - n_before}")

    # Build and save
    out_df = pd.DataFrame.from_records(records)
    out_df = out_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_df.to_parquet(output_path, index=False)

    print(f"\nTotal: {len(out_df)} QA pairs")
    print(f"Categories: {out_df['prompt_category'].value_counts().to_string()}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="/media/data1/muse_ge/ECG_ad20241231_metadata.v1.6._with_translation_ROXs42Bb.cleaned.parquet")
    parser.add_argument("--labels", default="/media/data1/muse_ge/ECG_ad20241231_gt_labels_v1.6.parquet")
    parser.add_argument("--existing_train", default="/volume/ECG_tokenizer/output/combined_train_qa_m5000k_h5000k_weighted.parquet")
    parser.add_argument("--output", default="/volume/ECG_tokenizer/output/deepecg_missing_qa_train.parquet")
    parser.add_argument("--max_per_type", type=int, default=500000)
    parser.add_argument("--max_binary_per_diag", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_deepecg_qa(
        args.metadata, args.labels, args.existing_train,
        args.output, args.max_per_type, args.max_binary_per_diag, args.seed,
    )
