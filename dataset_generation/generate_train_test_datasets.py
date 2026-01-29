#!/usr/bin/env python3
"""
Generate train and test datasets with enhanced prompts and answers.
Drops the existing question column and generates new comprehensive Q&A pairs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import sys
import json
import numpy as np
import re
import random
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Optional, Set, List, Dict, Any, Tuple
from tqdm import tqdm
from ecg_prompt_maker import ECGPromptMaker
from ecg_answer_generator import ECGAnswerGenerator

from dataset_generation.siglip_shared import (
    slug,
    text_ids_for_label,
    EXCLUSIVE_GROUPS,
    group_members,
    group_is_exclusive,
    targeted_negatives_for_label,
    build_siglip_hard_negative_text_ids,
    SIGLIP_TARGETED_HARD_NEGATIVES,
)

@dataclass
class LabelSource:
    diag_col: Optional[str]
    bert_col: Optional[str]


def build_slug_map(column_names: List[str]) -> Dict[str, List[str]]:
    slug_map: Dict[str, List[str]] = defaultdict(list)
    for name in column_names:
        slug_map[slug(name)].append(name)
    return slug_map


def resolve_column(
    slug_map: Dict[str, List[str]],
    label: str,
    preferred_suffixes: List[str],
) -> Optional[str]:
    base = slug(label)
    for suffix in preferred_suffixes:
        target = f"{base}_{suffix}"
        cols = slug_map.get(target)
        if cols:
            return cols[0]
        target_prefix = f"{suffix}_{base}"
        cols = slug_map.get(target_prefix)
        if cols:
            return cols[0]
    for suffix in preferred_suffixes:
        for key, cols in slug_map.items():
            if not key.startswith(f"{base}_"):
                continue
            remainder = key[len(base) + 1 :]
            if suffix in remainder.split("_"):
                return cols[0]
    for suffix in preferred_suffixes:
        for key, cols in slug_map.items():
            if not key.endswith(f"_{base}"):
                continue
            prefix = key[: -(len(base) + 1)]
            if suffix in prefix.split("_"):
                return cols[0]
    cols = slug_map.get(base)
    if cols:
        return cols[0]
    return None


def resolve_label_sources(
    slug_map: Dict[str, List[str]],
    unique_labels: List[str],
) -> tuple[Dict[str, LabelSource], List[str]]:
    diag_suffixes = ["gt", "gt_value", "gt_label", "gt_score", "value", "label", "count"]
    bert_suffixes = ["bert_model", "bert", "bertprob", "bertprobability"]
    label_sources: Dict[str, LabelSource] = {}
    missing: List[str] = []
    for label in unique_labels:
        diag_col = resolve_column(slug_map, label, diag_suffixes)
        bert_col = resolve_column(slug_map, label, bert_suffixes)
        if diag_col is None and bert_col is None:
            missing.append(label)
        label_sources[label] = LabelSource(diag_col=diag_col, bert_col=bert_col)
    return label_sources, missing


def build_siglip_hard_negative_text_ids(include_qa: bool = True) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for config in EXCLUSIVE_GROUPS.values():
        targeted = config.get("targeted_negatives")
        if not targeted:
            continue
        for anchor_label, negative_labels in targeted.items():
            anchor_ids = text_ids_for_label(anchor_label, include_qa=include_qa, include_no=False)
            negative_ids: List[str] = []
            for negative_label in negative_labels:
                negative_ids.extend(text_ids_for_label(negative_label, include_qa=include_qa, include_no=False))
            negative_ids = sorted(set(negative_ids) - set(anchor_ids))
            for anchor_id in anchor_ids:
                mapping[anchor_id] = negative_ids
    return mapping


def apply_mhi_special_stratified_sampling(df_mhi, target_samples, target_special_questions=1000):
    """
    Apply stratified sampling for MHI special questions to achieve specific ratios:
    - Ensure at least target_special_questions (default 1000) special questions
    - AFib risk: 50% high risk, 50% low risk
    - SHD: 50% normal, 50% abnormal
    - ACS: 50% acute coronary occlusion, 50% non-acute
    - LVEF: keep natural distribution
    
    ECGs can count for multiple categories. Since 1 prompt per ECG, we need 1000 ECGs with special questions.
    Returns exactly target_samples ECGs with at least target_special_questions having special question data.
    """
    print(f"   Applying stratified sampling for MHI special questions (target: {target_special_questions} special questions)...")

    # Check if special question columns exist (v1.6 may not have them)
    required_cols = ['afib_label_2y', 'afib_label_5y', 'GT_Afib', 'BERT_Afib',
                     'echonext_shd', 'acs_condition_severity', 'deepecho_Visually_Estimated_EF']
    missing_cols = [col for col in required_cols if col not in df_mhi.columns]

    if missing_cols:
        print(f"     WARNING: Special columns missing from data: {missing_cols}")
        print(f"     Falling back to random sampling (no stratification)")
        # Fall back to random sampling
        available_ecgs = df_mhi['waveform_name'].unique()
        np.random.seed(42)
        sampled_ecgs = np.random.choice(available_ecgs, size=min(target_samples, len(available_ecgs)), replace=False)
        return df_mhi[df_mhi['waveform_name'].isin(sampled_ecgs)]

    # First, collect all ECGs that qualify for special questions
    special_ecgs = set()

    # Track which ECGs qualify for which categories
    afib_high_risk_ecgs = set()
    afib_low_risk_ecgs = set()
    shd_abnormal_ecgs = set()
    shd_normal_ecgs = set()
    acs_acute_ecgs = set()
    acs_non_acute_ecgs = set()
    lvef_ecgs = set()

    # 1. Identify AFib risk ECGs
    afib_data = df_mhi[(df_mhi['afib_label_2y'].notna()) & (df_mhi['afib_label_5y'].notna())]
    if len(afib_data) > 0:
        # Exclude patients already in AFib (using GT and BERT labels)
        if 'GT_Afib' in afib_data.columns:
            afib_data = afib_data[(afib_data['GT_Afib'] == 0) | (afib_data['GT_Afib'].isna())]
        if 'BERT_Afib' in afib_data.columns:
            afib_data = afib_data[(afib_data['BERT_Afib'] <= 0.5) | (afib_data['BERT_Afib'].isna())]
        
        # High risk: 2-year or 5-year risk is True
        high_risk = afib_data[(afib_data['afib_label_2y'] == True) | (afib_data['afib_label_5y'] == True)]
        low_risk = afib_data[(afib_data['afib_label_2y'] == False) & (afib_data['afib_label_5y'] == False)]
        
        afib_high_risk_ecgs = set(high_risk['waveform_name'].values)
        afib_low_risk_ecgs = set(low_risk['waveform_name'].values)
        special_ecgs.update(afib_high_risk_ecgs)
        special_ecgs.update(afib_low_risk_ecgs)
    
    # 2. Identify SHD ECGs
    shd_data = df_mhi[df_mhi['echonext_shd'].notna()]
    if len(shd_data) > 0:
        shd_abnormal = shd_data[shd_data['echonext_shd'] >= 1]
        shd_normal = shd_data[shd_data['echonext_shd'] < 1]
        
        shd_abnormal_ecgs = set(shd_abnormal['waveform_name'].values)
        shd_normal_ecgs = set(shd_normal['waveform_name'].values)
        special_ecgs.update(shd_abnormal_ecgs)
        special_ecgs.update(shd_normal_ecgs)
    
    # 3. Identify ACS ECGs
    acs_data = df_mhi[df_mhi['acs_condition_severity'].notna()]
    if len(acs_data) > 0:
        from utils.constants import ACS_ACUTE_CONDITIONS
        
        acs_acute = acs_data[acs_data['acs_condition_severity'].isin(ACS_ACUTE_CONDITIONS)]
        acs_non_acute = acs_data[~acs_data['acs_condition_severity'].isin(ACS_ACUTE_CONDITIONS)]
        
        acs_acute_ecgs = set(acs_acute['waveform_name'].values)
        acs_non_acute_ecgs = set(acs_non_acute['waveform_name'].values)
        special_ecgs.update(acs_acute_ecgs)
        special_ecgs.update(acs_non_acute_ecgs)
    
    # 4. Identify LVEF ECGs
    lvef_data = df_mhi[df_mhi['deepecho_Visually_Estimated_EF'].notna()]
    if len(lvef_data) > 0:
        lvef_ecgs = set(lvef_data['waveform_name'].values)
        special_ecgs.update(lvef_ecgs)
    
    print(f"     Total unique ECGs with special questions: {len(special_ecgs)}")
    print(f"       - AFib risk: {len(afib_high_risk_ecgs)} high, {len(afib_low_risk_ecgs)} low")
    print(f"       - SHD: {len(shd_abnormal_ecgs)} abnormal, {len(shd_normal_ecgs)} normal")
    print(f"       - ACS: {len(acs_acute_ecgs)} acute, {len(acs_non_acute_ecgs)} non-acute")
    print(f"       - LVEF: {len(lvef_ecgs)} with data")
    
    # Calculate how many special ECGs to sample to get at least target_special_questions
    # Since we have 1 prompt per ECG and want 20% to be special questions
    special_ecgs_needed = min(target_special_questions, len(special_ecgs))
    
    # Sample special ECGs with stratification targets
    sampled_special = set()
    special_category_by_ecg = {}
    
    # Target distribution for the special-question pool (e.g. 1000 -> 250 each category):
    # - AFib risk: 50% high risk, 50% low risk
    # - SHD: 50% abnormal, 50% normal
    # - ACS: 50% acute, 50% non-acute
    # - LVEF: natural distribution (no enforced split)
    per_category = special_ecgs_needed // 4  # 250 each for test, proportional for train

    def split_evenly(total):
        """Return two integers that sum to total while staying as close to 50/50 as possible."""
        half = total // 2
        return half, total - half

    afib_high_target, afib_low_target = split_evenly(per_category)
    shd_abnormal_target, shd_normal_target = split_evenly(per_category)
    acs_acute_target, acs_non_acute_target = split_evenly(per_category)

    categories = [
        (afib_high_risk_ecgs, afib_high_target, 'afib_high'),          # 50% high-risk AFib
        (afib_low_risk_ecgs, afib_low_target, 'afib_low'),             # 50% low-risk AFib
        (shd_abnormal_ecgs, shd_abnormal_target, 'shd_abnormal'),      # 50% abnormal SHD
        (shd_normal_ecgs, shd_normal_target, 'shd_normal'),            # 50% normal SHD
        (acs_acute_ecgs, acs_acute_target, 'acs_acute'),               # 50% acute ACS
        (acs_non_acute_ecgs, acs_non_acute_target, 'acs_non_acute'),   # 50% non-acute ACS
        (lvef_ecgs, per_category, 'lvef')                               # Natural LVEF distribution
    ]

    for ecg_set, target_count, category_label in categories:
        available = list(ecg_set - sampled_special)
        if available:
            sample_count = min(len(available), target_count)
            if sample_count > 0:
                sampled = np.random.choice(available, size=sample_count, replace=False)
                sampled_special.update(sampled)
                for ecg in sampled:
                    special_category_by_ecg[ecg] = category_label
    
    # Fill remaining special slots if needed
    remaining_special_needed = special_ecgs_needed - len(sampled_special)
    if remaining_special_needed > 0:
        available = list(special_ecgs - sampled_special)
        if available:
            additional = np.random.choice(
                available,
                size=min(remaining_special_needed, len(available)),
                replace=False
            )
            sampled_special.update(additional)
            for ecg in additional:
                special_category_by_ecg.setdefault(ecg, 'general_special')
    
    # Now add non-special ECGs to reach target_samples
    all_ecgs = df_mhi['waveform_name'].unique()
    non_special_ecgs = [ecg for ecg in all_ecgs if ecg not in special_ecgs]
    
    non_special_needed = target_samples - len(sampled_special)
    sampled_non_special = []
    
    if non_special_needed > 0 and len(non_special_ecgs) > 0:
        # Sample non-special ECGs
        sample_count = min(non_special_needed, len(non_special_ecgs))
        sampled_non_special = list(np.random.choice(non_special_ecgs, size=sample_count, replace=False))
    
    # Combine special and non-special
    all_sampled = list(sampled_special) + sampled_non_special
    
    # If still need more, use replacement sampling
    if len(all_sampled) < target_samples:
        still_needed = target_samples - len(all_sampled)
        print(f"     Using replacement sampling to add {still_needed} more ECGs")
        all_available = list(all_ecgs)
        duplicates = list(np.random.choice(all_available, size=still_needed, replace=True))
        all_sampled.extend(duplicates)
    
    # Create the final DataFrame
    trimmed_waveforms = all_sampled[:target_samples]

    if 'waveform_name' not in df_mhi.columns:
        raise KeyError("Expected 'waveform_name' column in MHI dataframe for sampling.")

    # Preserve ordering (including duplicates) without repeatedly filtering the full dataframe.
    df_mhi_indexed = df_mhi.set_index('waveform_name', drop=False)
    try:
        sampled_df = df_mhi_indexed.loc[trimmed_waveforms].reset_index(drop=True)
    except KeyError as exc:
        missing = set(trimmed_waveforms) - set(df_mhi_indexed.index)
        raise KeyError(f"Sampled ECGs missing from MHI dataframe: {list(missing)[:5]}") from exc
    
    # Mark special ECGs for tracking
    sampled_df['has_special_question'] = sampled_df['waveform_name'].isin(sampled_special)
    sampled_df['special_question_category'] = sampled_df['waveform_name'].map(special_category_by_ecg)

    print(f"     Final sample: {len(sampled_df)} ECGs")
    print(f"       - Special question ECGs: {sampled_df['has_special_question'].sum()} ({sampled_df['has_special_question'].sum()/len(sampled_df)*100:.1f}%)")
    print(f"       - Regular ECGs: {(~sampled_df['has_special_question']).sum()} ({(~sampled_df['has_special_question']).sum()/len(sampled_df)*100:.1f}%)")
    
    return sampled_df.head(target_samples)  # Ensure exactly target_samples


def enrich_special_question_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns used in downstream special-question logic."""
    # Derive LVEF category labels when numeric estimates are available
    if 'deepecho_Visually_Estimated_EF' in df.columns:
        lvef_numeric = pd.to_numeric(df['deepecho_Visually_Estimated_EF'], errors='coerce')
        conditions = [
            lvef_numeric >= 55,
            (lvef_numeric >= 45) & (lvef_numeric < 55),
            (lvef_numeric >= 30) & (lvef_numeric < 45),
            lvef_numeric < 30
        ]
        choices = ['normal', 'mildly reduced', 'moderately reduced', 'severely reduced']
        lvef_categories = np.select(conditions, choices, default=None)
        lvef_series = pd.Series(lvef_categories, index=df.index, dtype='object')
        lvef_series[lvef_numeric.isna()] = pd.NA
        df['lvef_category'] = lvef_series
        df['lvef_Category'] = df['lvef_category']

    # Structural heart disease binary helper (1 abnormal, 0 normal)
    if 'echonext_shd' in df.columns:
        shd_numeric = pd.to_numeric(df['echonext_shd'], errors='coerce')
        shd_binary = pd.Series(pd.NA, index=df.index, dtype='Int64')
        valid_mask = shd_numeric.notna()
        if valid_mask.any():
            shd_binary.loc[valid_mask] = (shd_numeric.loc[valid_mask] >= 1).astype('int64')
        df['echonext_shd_binary'] = shd_binary

    # Flag acute ACS conditions directly for sampling/analytics convenience
    if 'acs_condition_severity' in df.columns:
        from utils.constants import ACS_ACUTE_CONDITIONS
        severity_series = df['acs_condition_severity']
        acute_binary = pd.Series(pd.NA, index=df.index, dtype='Int64')
        if severity_series.notna().any():
            acute_mask = severity_series.isin(ACS_ACUTE_CONDITIONS)
            acute_binary.loc[severity_series.notna()] = acute_mask.loc[severity_series.notna()].astype('int64')
        df['acs_condition_is_acute'] = acute_binary

    return df


def enforce_normal_cap(
    sampled_df: pd.DataFrame,
    full_df: pd.DataFrame,
    max_normal_percentage: float,
    random_state: int = 42,
    context: str = "",
):
    """Ensure the sampled dataframe respects the maximum proportion of normal ECGs."""

    if sampled_df is None or sampled_df.empty or max_normal_percentage is None:
        return sampled_df

    if 'ecg_type' not in sampled_df.columns:
        return sampled_df

    total = len(sampled_df)
    if total == 0:
        return sampled_df

    max_allowed = int(np.floor(total * float(max_normal_percentage) + 1e-9))
    normals_mask = sampled_df['ecg_type'] == 'normal'
    normal_count = int(normals_mask.sum())

    if normal_count <= max_allowed:
        return sampled_df

    drop_needed = normal_count - max_allowed
    print(
        f"   [normal-cap] {context or 'selection'}: trimming {drop_needed} normal ECGs "
        f"to respect {max_normal_percentage*100:.1f}% cap"
    )

    rng = np.random.default_rng(random_state)
    normal_indices = sampled_df[normals_mask].index.to_numpy()
    keep_count = max_allowed
    if keep_count > 0:
        keep_indices = rng.choice(normal_indices, size=keep_count, replace=False)
        drop_indices = np.setdiff1d(normal_indices, keep_indices, assume_unique=True)
    else:
        drop_indices = normal_indices

    adjusted_df = sampled_df.drop(index=drop_indices)
    needed_replacements = total - len(adjusted_df)
    if needed_replacements <= 0:
        return adjusted_df.reset_index(drop=True)

    candidate = full_df.copy()
    if candidate.empty or 'ecg_type' not in candidate.columns:
        return adjusted_df.reset_index(drop=True)

    candidate = candidate[candidate['ecg_type'] != 'normal']
    if candidate.empty:
        print("   [normal-cap] WARNING: no non-normal replacements available; returning reduced sample size.")
        return adjusted_df.reset_index(drop=True)

    # Avoid reusing ECGs already in the adjusted sample when possible
    unique_col = None
    for col in ['__row_id__', 'waveform_name', 'ecg_id']:
        if col in adjusted_df.columns and col in candidate.columns:
            unique_col = col
            break

    if unique_col:
        used_values = set(adjusted_df[unique_col].tolist())
        candidate = candidate[~candidate[unique_col].isin(used_values)]

    if candidate.empty:
        print("   [normal-cap] WARNING: exhausted unique non-normal replacements; sampling with replacement.")
        candidate = full_df[full_df['ecg_type'] != 'normal']

    if candidate.empty:
        print("   [normal-cap] WARNING: no candidates available even with replacement; returning reduced sample size.")
        return adjusted_df.reset_index(drop=True)

    replace_flag = len(candidate) < needed_replacements
    sample_n = needed_replacements if replace_flag else min(needed_replacements, len(candidate))
    replacements = candidate.sample(n=sample_n, replace=replace_flag, random_state=random_state)
    adjusted_df = pd.concat([adjusted_df, replacements], ignore_index=True)

    if len(adjusted_df) < total:
        print(
            f"   [normal-cap] WARNING: final sample has {len(adjusted_df)} rows (target {total}) "
            "due to limited non-normal replacements."
        )

    final_normal_count = int((adjusted_df['ecg_type'] == 'normal').sum())
    if final_normal_count > int(np.floor(len(adjusted_df) * float(max_normal_percentage) + 1e-9)):
        print(
            "   [normal-cap] WARNING: cap not fully satisfied due to data scarcity "
            f"({final_normal_count}/{len(adjusted_df)} normals)"
        )

    return adjusted_df.reset_index(drop=True)


def _trim_to_target(
    df: pd.DataFrame,
    target_count: int,
    coverage_ids: Optional[Set[int]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Trim sampled ECGs to exactly target_count while preserving coverage IDs."""

    if target_count is None or len(df) <= target_count:
        return df.reset_index(drop=True)

    if coverage_ids and "__row_id__" in df.columns:
        coverage_mask = df['__row_id__'].isin(coverage_ids)
        coverage_df = df[coverage_mask]
        remainder_df = df[~coverage_mask]

        if len(coverage_df) > target_count:
            trimmed = coverage_df.sample(n=target_count, random_state=random_state)
            return trimmed.reset_index(drop=True)

        slots_remaining = target_count - len(coverage_df)
        if slots_remaining > 0 and len(remainder_df) > 0:
            kept_remainder = remainder_df.sample(
                n=min(slots_remaining, len(remainder_df)),
                random_state=random_state,
            )
        else:
            kept_remainder = remainder_df.iloc[0:0]

        combined = pd.concat([coverage_df, kept_remainder], ignore_index=True)
        if len(combined) > target_count:
            combined = combined.sample(n=target_count, random_state=random_state)
        return combined.reset_index(drop=True)

    # Fallback: simple random trim
    return df.sample(n=target_count, random_state=random_state).reset_index(drop=True)


def _top_off_sample(
    df: pd.DataFrame,
    full_df: pd.DataFrame,
    target_count: int,
    max_normal_percentage: float,
    coverage_ids: Optional[Set[int]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Add ECGs until target_count is reached without breaching normal cap."""

    if target_count is None or len(df) >= target_count:
        return df.reset_index(drop=True)

    needed = target_count - len(df)
    normal_cap = int(target_count * max_normal_percentage)
    current_normals = int((df['ecg_type'] == 'normal').sum())
    normal_slots = max(0, normal_cap - current_normals)

    if "__row_id__" in df.columns:
        used_ids = set(df['__row_id__'])
        if coverage_ids:
            used_ids.update(coverage_ids)
        available = full_df[~full_df['__row_id__'].isin(used_ids)]
    else:
        used_index = set(df.index)
        available = full_df.loc[~full_df.index.isin(used_index)]

    additions = []

    if needed > 0:
        non_normal_pool = available[available['ecg_type'] != 'normal']
        non_normal_take = min(needed, len(non_normal_pool))
        if non_normal_take > 0:
            additions.append(non_normal_pool.sample(n=non_normal_take, random_state=random_state))
            needed -= non_normal_take
            if '__row_id__' in additions[-1].columns:
                available = available[~available['__row_id__'].isin(additions[-1]['__row_id__'])]

    if needed > 0 and normal_slots > 0:
        normal_pool = available[available['ecg_type'] == 'normal']
        normal_take = min(needed, normal_slots, len(normal_pool))
        if normal_take > 0:
            additions.append(normal_pool.sample(n=normal_take, random_state=random_state))
            needed -= normal_take
            normal_slots -= normal_take
            if '__row_id__' in additions[-1].columns:
                available = available[~available['__row_id__'].isin(additions[-1]['__row_id__'])]

    if needed > 0:
        fallback_pool = available if len(available) > 0 else full_df
        additions.append(
            fallback_pool.sample(
                n=needed,
                replace=len(fallback_pool) < needed,
                random_state=random_state,
            )
        )

    if additions:
        df = pd.concat([df] + additions, ignore_index=True)
        if '__row_id__' in df.columns:
            df = df.drop_duplicates(subset='__row_id__', keep='first').reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Parallel prompt generation utilities

PROMPT_MAKER_WORKER = None
PROMPT_MAX_PER_ECG = None


def _init_prompt_worker(max_prompts_per_ecg: Optional[int]) -> None:
    """Initializer for multiprocessing prompt worker."""

    global PROMPT_MAKER_WORKER, PROMPT_MAX_PER_ECG
    PROMPT_MAKER_WORKER = ECGPromptMaker()
    PROMPT_MAX_PER_ECG = max_prompts_per_ecg
    random.seed()
    np.random.seed()


def _prioritize_prompts(
    prompt_list: List[Tuple[str, str, float]],
    max_prompts: Optional[int],
) -> List[Tuple[str, str, float]]:
    """Select a deterministic, priority-ordered subset of prompts for an ECG."""

    if not prompt_list:
        return []

    if not max_prompts or max_prompts <= 0 or len(prompt_list) <= max_prompts:
        return prompt_list

    sorted_prompts = sorted(prompt_list, key=lambda item: item[2], reverse=True)
    selected: List[Tuple[str, str, float]] = []
    used_categories: Set[str] = set()

    priority_rules: List[Tuple[str, str]] = [
        ("exact", "interpretation"),
        ("exact", "structural_heart_disease"),
        ("exact", "lvef"),
        ("exact", "afib_risk"),
        ("exact", "acs_severity"),
        ("exact", "culprit_artery"),
        ("exact", "json_interpretation"),
        ("exact", "classification"),
        ("prefix", "category_"),
        ("prefix", "urgency"),
        ("prefix", "localization_"),
        ("exact", "ecg_interval"),
        ("exact", "interpretation_complex"),
        ("prefix", "random_finding"),
    ]

    def pick_matches(match_type: str, value: str) -> None:
        for prompt in sorted_prompts:
            category = prompt[1]
            if category in used_categories:
                continue
            if match_type == "exact":
                if category != value:
                    continue
            else:
                if not category.startswith(value):
                    continue
            selected.append(prompt)
            used_categories.add(category)
            if len(selected) >= max_prompts:
                return

    for match_type, value in priority_rules:
        if len(selected) >= max_prompts:
            break
        pick_matches(match_type, value)

    if len(selected) < max_prompts:
        for prompt in sorted_prompts:
            if prompt[1] in used_categories:
                continue
            selected.append(prompt)
            used_categories.add(prompt[1])
            if len(selected) >= max_prompts:
                break

    return selected[:max_prompts]


def _prompt_worker(row_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate prompts for a single ECG row inside a worker."""

    if PROMPT_MAKER_WORKER is None:
        raise RuntimeError("Prompt worker not initialised")

    row_series = pd.Series(row_dict)
    prompts = PROMPT_MAKER_WORKER.generate_prompts_for_ecg(row_series)
    prompts = _prioritize_prompts(prompts, PROMPT_MAX_PER_ECG)

    results: List[Dict[str, Any]] = []
    for prompt_text, prompt_category, prompt_weight in prompts:
        entry = row_dict.copy()
        entry['prompt'] = prompt_text
        entry['prompt_category'] = prompt_category
        entry['prompt_weight'] = prompt_weight
        results.append(entry)
    return results


def _generate_prompts_serial(
    df_sampled: pd.DataFrame,
    sample_size: Optional[int],
    max_prompts_per_ecg: Optional[int],
) -> List[Dict[str, Any]]:
    prompt_maker = ECGPromptMaker()
    all_prompts: List[Dict[str, Any]] = []

    progress_interval = 1000 if max_prompts_per_ecg else 100
    ecgs_processed = 0

    for _, row in tqdm(df_sampled.iterrows(), total=len(df_sampled), desc="   Generating prompts"):
        prompts = prompt_maker.generate_prompts_for_ecg(row)
        ecgs_processed += 1

        prompts = _prioritize_prompts(prompts, max_prompts_per_ecg)

        row_dict = row.to_dict()
        for prompt_text, prompt_category, prompt_weight in prompts:
            new_row = row_dict.copy()
            new_row['prompt'] = prompt_text
            new_row['prompt_category'] = prompt_category
            new_row['prompt_weight'] = prompt_weight
            all_prompts.append(new_row)

        if ecgs_processed % progress_interval == 0:
            print(
                f"   Processed {ecgs_processed} ECGs, generated {len(all_prompts)} prompts...",
                flush=True,
            )

        if sample_size and len(all_prompts) >= sample_size:
            break

    return all_prompts


def _generate_prompts_parallel(
    df_sampled: pd.DataFrame,
    sample_size: Optional[int],
    max_prompts_per_ecg: Optional[int],
    workers: int,
) -> List[Dict[str, Any]]:
    records = df_sampled.to_dict(orient='records')
    all_prompts: List[Dict[str, Any]] = []

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=_init_prompt_worker, initargs=(max_prompts_per_ecg,)) as pool:
        iterator = pool.imap_unordered(_prompt_worker, records, chunksize=32)
        for prompt_list in tqdm(iterator, total=len(records), desc="   Generating prompts"):
            if prompt_list:
                all_prompts.extend(prompt_list)
                if sample_size and len(all_prompts) >= sample_size:
                    break

    if sample_size and len(all_prompts) > sample_size:
        all_prompts = all_prompts[:sample_size]

    return all_prompts


def generate_prompts_for_dataset(
    df_sampled: pd.DataFrame,
    sample_size: Optional[int],
    max_prompts_per_ecg: Optional[int],
    prompt_workers: int,
) -> pd.DataFrame:
    if prompt_workers and prompt_workers > 1:
        prompts = _generate_prompts_parallel(df_sampled, sample_size, max_prompts_per_ecg, prompt_workers)
    else:
        prompts = _generate_prompts_serial(df_sampled, sample_size, max_prompts_per_ecg)

    if sample_size and len(prompts) > sample_size:
        prompts = prompts[:sample_size]

    return pd.DataFrame(prompts)


# ---------------------------------------------------------------------------
# Parallel answer generation utilities

ANSWER_GENERATOR_WORKER = None


def _init_answer_worker(dataset: str) -> None:
    global ANSWER_GENERATOR_WORKER
    ANSWER_GENERATOR_WORKER = ECGAnswerGenerator(language='en', dataset=dataset)
    random.seed()
    np.random.seed()


def _answer_worker(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    if ANSWER_GENERATOR_WORKER is None:
        raise RuntimeError("Answer worker not initialised")

    row_series = pd.Series(row_dict)
    row_dict = row_dict.copy()
    row_dict['generated_answer'] = ANSWER_GENERATOR_WORKER.generate_answer(row_series)
    return row_dict


def generate_answers_for_dataset(
    df_with_prompts: pd.DataFrame,
    dataset_type: str,
    answer_workers: int,
) -> tuple[pd.DataFrame, int]:
    if dataset_type in ['mimic-iv', 'combined', 'custom']:
        answer_dataset = 'mimic'
    elif dataset_type == 'mhi':
        answer_dataset = 'mhi'
    else:
        answer_dataset = dataset_type

    records = df_with_prompts.to_dict(orient='records')

    if answer_workers and answer_workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=answer_workers, initializer=_init_answer_worker, initargs=(answer_dataset,)) as pool:
            updated_records = list(
                tqdm(
                    pool.imap(_answer_worker, records, chunksize=64),
                    total=len(records),
                    desc="   Generating answers",
                )
            )
    else:
        answer_gen = ECGAnswerGenerator(language='en', dataset=answer_dataset)
        updated_records = []
        for _, row in tqdm(df_with_prompts.iterrows(), total=len(df_with_prompts), desc="   Generating answers"):
            row_dict = row.to_dict()
            row_dict['generated_answer'] = answer_gen.generate_answer(row)
            updated_records.append(row_dict)

    df_with_answers = pd.DataFrame(updated_records)
    initial_count = len(df_with_answers)
    df_with_answers = df_with_answers[df_with_answers['generated_answer'].notna()].reset_index(drop=True)
    dropped_count = initial_count - len(df_with_answers)
    return df_with_answers, dropped_count


def generate_siglip_alignment_dataset(
    parquet_path: str,
    output_dir: str,
    include_qa: bool = True,
    w_pos: float = 1.0,
    w_hardneg: float = 3.0,
    w_implneg: float = 0.2,
    sample_size_for_checks: int = 1000,
    random_state: int = 0,
    implicit_negative_sample_size: int = 64,
    max_hardneg_per_group: int | None = 3,
    train_ecg_limit: Optional[int] = None,
    val_ecg_limit: Optional[int] = None,
    val_balanced_limit: Optional[int] = None,
    val_balanced_min_per_label: int = 50,
    test_ecg_limit: Optional[int] = None,
    max_normal_percentage: float = 0.05,
    max_par_ailleurs_percentage: Optional[float] = 0.05,
    max_validation_ecgs: Optional[int] = 10000,
    max_positive_per_label: Optional[int] = None,
    labels_path: Optional[str] = None,
    diagnosis_column: Optional[str] = None,
):
    """Create SigLIP-ready text bank and ECG-text alignment with exclusivity-aware weights."""

    print("\nSIGLIP ALIGNMENT GENERATION", flush=True)
    print("=" * 60)
    print(f"Parquet: {parquet_path}")
    if labels_path:
        print(f"Labels parquet: {labels_path}")
    print(f"Output directory: {output_dir}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    normalized_diag = (diagnosis_column or "").strip() if diagnosis_column is not None else ""
    if normalized_diag.lower() == "none":
        normalized_diag = ""
    requested_diagnosis_column = normalized_diag or None
    include_diagnosis_requested = requested_diagnosis_column is not None
    diagnosis_column = requested_diagnosis_column
    include_diagnosis = include_diagnosis_requested

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("pyarrow is required to inspect the MHI parquet schema") from exc

    metadata_schema = pq.read_schema(parquet_path)
    metadata_cols = set(metadata_schema.names)
    if labels_path:
        label_schema = pq.read_schema(labels_path)
        all_schema_cols = metadata_cols | set(label_schema.names)
    else:
        all_schema_cols = metadata_cols

    diag_candidates: list[Optional[str]] = []
    if include_diagnosis_requested and requested_diagnosis_column:
        diag_candidates.append(requested_diagnosis_column)
        if requested_diagnosis_column != "diagnosis":
            diag_candidates.append("diagnosis")
    else:
        diag_candidates.extend(["translated_diagnosis", "diagnosis"])

    diag_candidates = [candidate for candidate in diag_candidates if candidate]
    resolved_diagnosis_column: Optional[str] = None
    for candidate in diag_candidates:
        if candidate in all_schema_cols:
            resolved_diagnosis_column = candidate
            break

    if resolved_diagnosis_column is None:
        if include_diagnosis_requested and requested_diagnosis_column:
            print(
                f"Requested diagnosis column '{requested_diagnosis_column}' not found; "
                "diagnosis text will be omitted from SigLIP mapping.",
                flush=True,
            )
        include_diagnosis = False
        diagnosis_column = None
    else:
        diagnosis_column = resolved_diagnosis_column
        include_diagnosis = True
        if include_diagnosis_requested and requested_diagnosis_column and resolved_diagnosis_column != requested_diagnosis_column:
            print(
                f"Diagnosis column '{requested_diagnosis_column}' not found; falling back to '{resolved_diagnosis_column}'.",
                flush=True,
            )
        elif not include_diagnosis_requested:
            print(
                f"Using diagnosis column '{resolved_diagnosis_column}' for SigLIP mapping.",
                flush=True,
            )

    has_waveform_path = 'waveform_path_psa' in all_schema_cols
    if not has_waveform_path and 'npy_path' not in all_schema_cols:
        raise ValueError(
            "Expected either 'waveform_path_psa' or 'npy_path' column in provided parquet files; "
            "ensure at least one exists in metadata or labels parquet"
        )

    from utils.constants import DEEPECG_CATEGORIES, DEEPECG_PATHOLOGICAL_LIMIT


    all_labels = []
    for cat, labels in DEEPECG_CATEGORIES.items():
        for label in labels:
            all_labels.append((cat, label))

    unique_labels: list[str] = []
    for _, label in all_labels:
        if label not in unique_labels:
            unique_labels.append(label)

    if len(unique_labels) != 77:
        raise ValueError(f"Expected 77 unique DEEPECG labels, found {len(unique_labels)}")

    def _canonical_label(label: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()

    def slug(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")

    excluded_labels = {"Sinusal", "Regular", "Monomorph"}
    excluded_label_slugs = {slug(name) for name in excluded_labels}

    def is_excluded_label(label: str) -> bool:
        label_slug = slug(label)
        return label in excluded_labels or label_slug in excluded_label_slugs

    def zero_label_columns(frame: pd.DataFrame, label: str) -> None:
        target_slug = slug(label)
        matching = [col for col in frame.columns if slug(col) == target_slug]
        for col in matching:
            frame[col] = 0

    slug_map = build_slug_map(list(all_schema_cols))
    label_sources, missing_label_columns = resolve_label_sources(slug_map, unique_labels)
    if missing_label_columns:
        raise ValueError(
            "Missing GT/BERT columns for labels: "
            + ", ".join(sorted(missing_label_columns))
        )

    if labels_path:
        metadata_df = pd.read_parquet(parquet_path)
        labels_df = pd.read_parquet(labels_path)

        for name, frame in (("metadata", metadata_df), ("labels", labels_df)):
            if "npy_path" not in frame.columns:
                raise ValueError(f"Expected 'npy_path' column in {name} parquet ({parquet_path if name == 'metadata' else labels_path})")

        def add_npy_name(df: pd.DataFrame, col: str = "npy_path") -> pd.DataFrame:
            df = df.copy()
            df["__npy_name"] = df[col].astype(str).apply(lambda x: os.path.basename(x))
            return df

        metadata_df = add_npy_name(metadata_df, "npy_path")
        labels_df = add_npy_name(labels_df, "npy_path")

        print(f"Metadata shape: {metadata_df.shape}, labels shape: {labels_df.shape}")
        df_full = metadata_df.merge(labels_df, on="__npy_name", how="left", suffixes=("", "_labels"))
        print(f"Merged shape: {df_full.shape}")

        split_candidates = [col for col in ("Split", "split", "Split_labels", "split_labels") if col in df_full.columns]
        if not split_candidates:
            raise ValueError("Column 'Split' (or 'split') not found after merging metadata and labels")

        split_col = split_candidates[0]
        missing_split = int(df_full[split_col].isna().sum())
        print(f"Rows without SPLIT match: {missing_split}")
        if missing_split:
            print(
                f"Warning: {missing_split} rows are missing '{split_col}' after merging; they will be dropped.",
                flush=True,
            )
            df_full = df_full[df_full[split_col].notna()].copy()

        if split_col != "Split":
            df_full = df_full.rename(columns={split_col: "Split"})

        if "npy_path_labels" in df_full.columns:
            df_full.loc[df_full["npy_path_labels"].notna(), "npy_path"] = df_full["npy_path_labels"]
        df_full = df_full.drop(columns=[c for c in ["__npy_name", "npy_path_labels"] if c in df_full.columns])
    else:
        df_full = pd.read_parquet(parquet_path)

    available_cols = set(df_full.columns)
    if include_diagnosis and diagnosis_column not in available_cols:
        print(
            f"Diagnosis column '{diagnosis_column}' not found in merged dataframe; "
            "diagnosis text will be omitted from SigLIP mapping.",
            flush=True,
        )
        include_diagnosis = False
        diagnosis_column = None
    if 'waveform_path_psa' not in available_cols and 'npy_path' not in available_cols:
        raise ValueError("Expected either 'waveform_path_psa' or 'npy_path' column in input parquet")

    needed_cols: Set[str] = {
        'npy_path',
        'waveform_name',
        'Split',
        'split',
        'ecg_path',
    }
    if 'waveform_path_psa' in available_cols:
        needed_cols.add('waveform_path_psa')
    if include_diagnosis and diagnosis_column:
        needed_cols.add(diagnosis_column)
    for source in label_sources.values():
        if source.diag_col:
            needed_cols.add(source.diag_col)
        if source.bert_col:
            needed_cols.add(source.bert_col)

    existing_needed_cols = [col for col in needed_cols if col in df_full.columns]
    df = df_full[existing_needed_cols].copy()
    if df.empty:
        raise ValueError("Merged dataframe is empty after selecting required columns")

    if 'waveform_path_psa' not in df.columns:
        if 'npy_path' not in df.columns and 'ecg_path' not in df.columns:
            raise ValueError("Missing required ECG path column ('waveform_path_psa', 'npy_path', or 'ecg_path') after column selection")
        if 'waveform_path_psa' not in df.columns and 'ecg_path' in df.columns:
            df['waveform_path_psa'] = df['ecg_path'].astype(str)
        if 'waveform_path_psa' not in df.columns and 'npy_path' in df.columns:
            df['waveform_path_psa'] = df['npy_path'].astype(str)

    if include_diagnosis and diagnosis_column not in df.columns:
        print(
            f"Diagnosis column '{diagnosis_column}' missing after column selection; "
            "diagnosis text will be omitted from SigLIP mapping.",
            flush=True,
        )
        include_diagnosis = False
        diagnosis_column = None

    if 'waveform_name' not in df.columns:
        df['waveform_name'] = df['waveform_path_psa'].astype(str).apply(lambda p: os.path.basename(p))

    split_candidates = [col for col in ('Split', 'split', 'dataset_split') if col in df.columns]
    if split_candidates:
        split_col = split_candidates[0]
        missing_split = int(df[split_col].isna().sum())
        if missing_split:
            raise ValueError(f"{missing_split} rows are missing '{split_col}' values in merged dataframe")
        df['split'] = df[split_col].astype(str).str.lower()
    else:
        def extract_split(path: str) -> str:
            match = re.search(r"/adjusted_signals/([^/]+)/", path)
            return match.group(1) if match else 'train'

        df['split'] = df['waveform_path_psa'].astype(str).apply(extract_split)

    df['ecg_id'] = df['waveform_path_psa'].astype(str)
    df = df.dropna(subset=['ecg_id']).drop_duplicates(subset='ecg_id')

    diagnosis_lookup: dict[str, Any] = {}
    if include_diagnosis and diagnosis_column is not None:
        diagnosis_lookup = df.set_index('ecg_id')[diagnosis_column].to_dict()
    else:
        include_diagnosis = False

    present_cols: List[str] = []
    diag_cols_map: dict[str, str] = {}
    bert_cols_map: dict[str, Optional[str]] = {}

    for label in unique_labels:
        source = label_sources[label]
        diag_col = source.diag_col
        if diag_col and diag_col in df.columns:
            df[label] = pd.to_numeric(df[diag_col], errors='coerce')
        else:
            df[label] = np.nan
        diag_cols_map[label] = label
        if source.diag_col or source.bert_col:
            present_cols.append(label)

        if source.bert_col and source.bert_col in df.columns:
            df[source.bert_col] = pd.to_numeric(df[source.bert_col], errors='coerce')
            bert_cols_map[label] = source.bert_col
        else:
            bert_cols_map[label] = None

    present_cols = list(dict.fromkeys(present_cols))
    if not present_cols:
        raise ValueError("No SigLIP label columns were resolved from the provided parquet files")

    if val_ecg_limit is None:
        val_ecg_limit = 10000

    val_limit_for_sampling = val_ecg_limit
    if val_balanced_limit is not None:
        val_limit_for_sampling = None

    split_limits = {
        'train': train_ecg_limit,
        'training': train_ecg_limit,
        'validation': val_limit_for_sampling,
        'val': val_limit_for_sampling,
        'dev': val_limit_for_sampling,
        'test': test_ecg_limit,
    }

    if any(limit is not None for limit in split_limits.values()):
        sampled_frames = []
        for split_name, group_df in df.groupby('split', sort=False):
            limit = split_limits.get(split_name.lower())
            if limit is not None and limit > 0 and len(group_df) > limit:
                group_df = group_df.sample(n=limit, random_state=random_state)
                print(f"   SigLIP sampling: keeping {limit:,} ECGs for split '{split_name}' (of {len(df[df['split'] == split_name]):,})")
            sampled_frames.append(group_df)
        df = pd.concat(sampled_frames, ignore_index=True)

    if (
        max_par_ailleurs_percentage is not None
        and include_diagnosis
        and diagnosis_column is not None
        and diagnosis_column in df.columns
    ):
        max_par_ailleurs_percentage = float(max_par_ailleurs_percentage)
        if not 0.0 <= max_par_ailleurs_percentage <= 1.0:
            raise ValueError("max_par_ailleurs_percentage must be between 0 and 1 for SigLIP dataset")
        phrase = "ecg normal par ailleurs"
        diag_series = df[diagnosis_column].fillna("").astype(str).str.lower()
        par_mask = diag_series.str.contains(phrase)
        if par_mask.any():
            filtered_frames = []
            for split_name, group_df in df.groupby('split', sort=False):
                if group_df.empty:
                    filtered_frames.append(group_df)
                    continue
                group_mask = par_mask.loc[group_df.index]
                total = len(group_df)
                max_allowed = int(total * max_par_ailleurs_percentage)
                positives = group_df[group_mask]
                if positives.empty or len(positives) <= max_allowed:
                    filtered_frames.append(group_df)
                    continue
                keep_count = max_allowed
                if keep_count <= 0:
                    keep_indices = []
                else:
                    keep_indices = positives.sample(
                        n=keep_count,
                        random_state=random_state,
                    ).index.tolist()
                non_phrase_indices = group_df[~group_mask].index.tolist()
                selected_indices = non_phrase_indices + keep_indices
                filtered = group_df.loc[selected_indices]
                filtered_frames.append(filtered)
                print(
                    f"   Reduced '{phrase}' entries for split '{split_name}' from {len(positives)} to {len(keep_indices)} "
                    f"({max_par_ailleurs_percentage:.2%} cap)."
                )
            df = pd.concat(filtered_frames, ignore_index=True)

    if max_validation_ecgs is not None and max_validation_ecgs > 0:
        val_split_names = {'validation', 'val', 'dev'}
        val_mask = df['split'].astype(str).str.lower().isin(val_split_names)
        current_val_ecgs = int(val_mask.sum())
        if current_val_ecgs > max_validation_ecgs:
            val_df = df[val_mask]
            if len(val_df) > max_validation_ecgs:
                limit_rng = np.random.default_rng(random_state + 1)
                val_indices = val_df.index.to_numpy()
                selected_indices = limit_rng.choice(val_indices, size=max_validation_ecgs, replace=False)
                selected_indices = sorted(int(idx) for idx in selected_indices.tolist())
                kept_val_df = val_df.loc[selected_indices]
            else:
                kept_val_df = val_df
            if len(kept_val_df) < len(val_df):
                print(
                    f"   Validation ECGs reduced from {len(val_df):,} to {len(kept_val_df):,} "
                    f"(limit {max_validation_ecgs:,})."
                )
            df = pd.concat([df[~val_mask], kept_val_df], ignore_index=True)
            if include_diagnosis and diagnosis_column is not None:
                diagnosis_lookup = df.set_index('ecg_id')[diagnosis_column].to_dict()

    for rhythm_label in ("Sinusal", "Regular", "Monomorph"):
        zero_label_columns(df, rhythm_label)

    pathological_cols = DEEPECG_PATHOLOGICAL_LIMIT['deepecg']['pathological']
    limit_cols = DEEPECG_PATHOLOGICAL_LIMIT['deepecg']['limit']

    df['ecg_type'] = 'normal'
    for col in pathological_cols:
        if col in df.columns:
            df.loc[df[col] >= 1, 'ecg_type'] = 'pathological'
    for col in limit_cols:
        if col in df.columns:
            df.loc[(df[col] >= 1) & (df['ecg_type'] == 'normal'), 'ecg_type'] = 'borderline'

    if max_normal_percentage is not None:
        max_normal_percentage = float(max_normal_percentage)
        if not 0.0 <= max_normal_percentage <= 1.0:
            raise ValueError('max_normal_percentage must be between 0 and 1 for SigLIP dataset')
        balanced_frames = []
        for split_name, group_df in df.groupby('split', sort=False):
            if group_df.empty:
                balanced_frames.append(group_df)
                continue
            total = len(group_df)
            max_normals = int(total * max_normal_percentage)
            normals = group_df[group_df['ecg_type'] == 'normal']
            if len(normals) > max_normals:
                normals = normals.sample(n=max_normals, random_state=random_state)
            non_normals = group_df[group_df['ecg_type'] != 'normal']
            balanced_frames.append(pd.concat([normals, non_normals], ignore_index=True))
        df = pd.concat(balanced_frames, ignore_index=True)

    if val_balanced_limit is not None:
        val_balanced_limit = int(val_balanced_limit)
        if val_balanced_limit <= 0:
            print(f"   Validation balancing skipped because limit ({val_balanced_limit}) is not positive.")
        else:
            val_split_names = {'validation', 'val', 'dev'}
            val_mask = df['split'].astype(str).str.lower().isin(val_split_names)
            if not val_mask.any():
                print("   Validation balancing skipped because no validation rows were found.")
            else:
                val_df = df[val_mask].copy()
                num_labels = len(unique_labels)
                if val_balanced_limit < num_labels:
                    print(
                        f"   Validation balancing skipped because limit ({val_balanced_limit}) "
                        f"is smaller than number of labels ({num_labels})."
                    )
                else:
                    label_matrix = (val_df[unique_labels].fillna(0) >= 1)
                    available_counts = label_matrix.sum(axis=0).astype(int)
                    if int(available_counts.sum()) == 0:
                        print("   Validation balancing skipped because no positive labels were found.")
                    else:
                        rng = np.random.default_rng(random_state)

                        base_target = val_balanced_limit // num_labels
                        desired_target = max(base_target, 1)
                        if val_balanced_limit >= val_balanced_min_per_label * num_labels:
                            desired_target = max(desired_target, int(val_balanced_min_per_label))

                        target_counts: dict[str, int] = {}
                        for label in unique_labels:
                            target_counts[label] = int(min(desired_target, available_counts.get(label, 0)))

                        selected_indices: set[Any] = set()
                        label_counts: dict[str, int] = {label: 0 for label in unique_labels}
                        row_positive_cache: dict[Any, list[str]] = {}

                        for label in unique_labels:
                            target = target_counts[label]
                            if target <= 0:
                                continue
                            positives_series = label_matrix[label]
                            positive_indices = positives_series[positives_series].index.to_numpy()
                            if positive_indices.size == 0:
                                continue
                            positive_indices = positive_indices[rng.permutation(positive_indices.size)]
                            for idx in positive_indices:
                                if label_counts[label] >= target:
                                    break
                                if idx in selected_indices:
                                    continue
                                selected_indices.add(idx)
                                positives = row_positive_cache.get(idx)
                                if positives is None:
                                    mask_row = label_matrix.loc[idx]
                                    positives = [lab for lab, flag in mask_row.items() if flag]
                                    row_positive_cache[idx] = positives
                                for lab in positives:
                                    label_counts[lab] += 1

                        if not selected_indices:
                            print("   Validation balancing skipped because no ECGs met the selection criteria.")
                        else:
                            if len(selected_indices) > val_balanced_limit:
                                trimmed = rng.choice(
                                    list(selected_indices),
                                    size=val_balanced_limit,
                                    replace=False,
                                )
                                selected_indices = set(trimmed.tolist())

                            selected_in_order = [idx for idx in val_df.index if idx in selected_indices]
                            balanced_val_df = val_df.loc[selected_in_order]
                            balanced_label_counts = (label_matrix.loc[selected_in_order].sum(axis=0)).astype(int)

                            total_selected = len(balanced_val_df)
                            min_count = int(balanced_label_counts.min()) if not balanced_label_counts.empty else 0
                            max_count = int(balanced_label_counts.max()) if not balanced_label_counts.empty else 0

                            print(
                                f"   Validation balancing selected {total_selected:,} of {len(val_df):,} ECGs "
                                f"(target per label {desired_target}, min count {min_count}, max count {max_count})."
                            )

                            short_labels = [
                                label
                                for label in unique_labels
                                if target_counts[label] > 0
                                and balanced_label_counts.get(label, 0) < target_counts[label]
                            ]
                            if short_labels:
                                preview = ", ".join(sorted(short_labels)[:10])
                                more = "" if len(short_labels) <= 10 else f", ... (+{len(short_labels) - 10})"
                                print(
                                    f"     Labels below target due to availability limits: {preview}{more}"
                                )

                            df = pd.concat(
                                [df[~val_mask], balanced_val_df],
                                ignore_index=True,
                            )

    print(f"Loaded {len(df):,} ECG rows with {len(present_cols)} label columns")

    def canonical_text(label: str) -> str:
        return f"{label} present."

    def qa_yes(label: str) -> str:
        return f"Q: Is {label} present? A: Yes."

    def qa_no(label: str) -> str:
        return f"Q: Is {label} present? A: No."

    rows = []
    for category, label in all_labels:
        if label not in present_cols or is_excluded_label(label):
            continue
        base_id = slug(label)
        rows.append(
            dict(text_id=f"LBL_{base_id}", granularity="atomic", category=category, text=canonical_text(label))
        )
        if include_qa:
            rows.append(dict(text_id=f"QA_{base_id}_yes", granularity="qa", category=category, text=qa_yes(label)))
            rows.append(dict(text_id=f"QA_{base_id}_no", granularity="qa", category=category, text=qa_no(label)))

    text_bank = pd.DataFrame(rows).drop_duplicates(subset=['text_id']).reset_index(drop=True)
    print(f"Text bank entries: {len(text_bank)}")

    label2group: Dict[str, str] = {}
    for group_name, config in EXCLUSIVE_GROUPS.items():
        if not group_is_exclusive(config):
            continue
        for member in group_members(config):
            label2group[member] = group_name

    rng = np.random.default_rng(random_state)

    split_lookup = df.set_index('ecg_id')['split'].to_dict()

    bert_iter_cols = [col for col in bert_cols_map.values() if col]
    iter_cols = ['ecg_id'] + present_cols
    for col in bert_iter_cols:
        if col not in iter_cols:
            iter_cols.append(col)
    col_pos = {col: idx for idx, col in enumerate(iter_cols)}

    map_rows = []
    exclusivity_conflicts = []
    conflict_counter = Counter()
    bert_threshold = 0.5

    df_iter = df[iter_cols]
    for values in df_iter.itertuples(index=False, name=None):
        ecg_id = values[col_pos['ecg_id']]
        diagnosis_value = diagnosis_lookup.get(ecg_id) if include_diagnosis else None
        diag_dict: dict[str, float] = {}
        bert_dict: dict[str, float] = {}
        row_has_valid_conf = False

        pos_set: set[str] = set()
        label_positive: dict[str, bool] = {}
        for label in unique_labels:
            diag_col = diag_cols_map.get(label)
            diag_val = values[col_pos[diag_col]] if diag_col and diag_col in col_pos else float('nan')

            bert_col = bert_cols_map.get(label)
            bert_val = values[col_pos[bert_col]] if bert_col and bert_col in col_pos else float('nan')

            diag_dict[label] = diag_val
            bert_dict[label] = bert_val

            has_diag = not pd.isna(diag_val)
            has_bert = not pd.isna(bert_val)

            if has_diag or has_bert:
                row_has_valid_conf = True

            positive = False
            if has_diag and diag_val >= 1:
                positive = True
            if has_bert and bert_val > bert_threshold:
                positive = True

            label_positive[label] = positive

            if is_excluded_label(label):
                continue

            if positive:
                pos_set.add(label)

        if not row_has_valid_conf:
            continue

        gpos = defaultdict(set)
        for lab in pos_set:
            group = label2group.get(lab)
            if group:
                gpos[group].add(lab)

        for group, labs in gpos.items():
            if len(labs) > 1:
                labs_sorted = sorted(labs)
                keep = labs_sorted[0]
                drop = labs_sorted[1:]
                for lab in drop:
                    pos_set.discard(lab)
                exclusivity_conflicts.append((ecg_id, group, labs_sorted))
                conflict_counter[group] += 1

        pos_labels = sorted(pos_set)

        hardneg_by_group = defaultdict(list)
        for label in unique_labels:
            if is_excluded_label(label) or label in pos_set:
                continue

            diag_val = diag_dict.get(label)
            bert_val = bert_dict.get(label)

            negative = False
            if not pd.isna(diag_val):
                negative = diag_val <= 0
            elif not pd.isna(bert_val):
                negative = bert_val <= bert_threshold
            if label_positive.get(label, False):
                negative = False

            if not negative:
                continue

            group = label2group.get(label)
            if group and group in gpos:
                hardneg_by_group[group].append(label)

        hardneg_labels: list[str] = []
        for group, labels_for_group in hardneg_by_group.items():
            if max_hardneg_per_group and len(labels_for_group) > max_hardneg_per_group:
                sampled = rng.choice(labels_for_group, size=max_hardneg_per_group, replace=False).tolist()
            else:
                sampled = labels_for_group
            hardneg_labels.extend(sampled)

        hardneg_set: set[str] = set(hardneg_labels)
        for lab in pos_labels:
            for neg_label in targeted_negatives_for_label(lab):
                if neg_label in present_cols and not is_excluded_label(neg_label) and neg_label not in pos_set:
                    hardneg_set.add(neg_label)
        hardneg_labels = sorted(hardneg_set)

        if not pos_labels and not hardneg_labels:
            best_label = None
            best_score = -1.0
            for label in unique_labels:
                if is_excluded_label(label):
                    continue
                bert_val = bert_dict.get(label)
                if pd.notna(bert_val) and bert_val > best_score:
                    best_label = label
                    best_score = float(bert_val)
            if best_label is None:
                best_label = next((lbl for lbl in unique_labels if not is_excluded_label(lbl)), unique_labels[0])
            hardneg_labels.append(best_label)

        split_value = split_lookup.get(ecg_id, 'unknown')

        for lab in pos_labels:
            base = slug(lab)
            pos_entry = dict(
                ecg_id=ecg_id,
                text_id=f"LBL_{base}",
                label=1,
                weight=float(w_pos),
                split=split_value,
            )
            if include_diagnosis:
                pos_entry['diagnosis'] = diagnosis_value
            map_rows.append(pos_entry)
            if include_qa:
                qa_pos_entry = dict(
                    ecg_id=ecg_id,
                    text_id=f"QA_{base}_yes",
                    label=1,
                    weight=float(w_pos),
                    split=split_value,
                )
                if include_diagnosis:
                    qa_pos_entry['diagnosis'] = diagnosis_value
                map_rows.append(qa_pos_entry)

        for lab in hardneg_labels:
            if is_excluded_label(lab):
                continue
            base = slug(lab)
            neg_entry = dict(
                ecg_id=ecg_id,
                text_id=f"LBL_{base}",
                label=0,
                weight=float(w_hardneg),
                split=split_value,
            )
            if include_diagnosis:
                neg_entry['diagnosis'] = diagnosis_value
            map_rows.append(neg_entry)
            if include_qa:
                qa_neg_entry = dict(
                    ecg_id=ecg_id,
                    text_id=f"QA_{base}_yes",
                    label=0,
                    weight=float(w_hardneg),
                    split=split_value,
                )
                if include_diagnosis:
                    qa_neg_entry['diagnosis'] = diagnosis_value
                map_rows.append(qa_neg_entry)
    mapping = pd.DataFrame(map_rows)
    if not mapping.empty:
        mapping['label'] = mapping['label'].astype(int)
        mapping['weight'] = mapping['weight'].astype(float)

        mapping = mapping.sort_values(['ecg_id', 'text_id', 'weight'], ascending=[True, True, False])
        mapping = mapping.drop_duplicates(subset=['ecg_id', 'text_id'], keep='first')

        if max_positive_per_label is not None:
            pos_mask = mapping['label'] == 1
            positives = mapping[pos_mask]
            if not positives.empty:
                positives = positives.groupby('text_id', group_keys=False).apply(
                    lambda g: g.sample(n=min(len(g), max_positive_per_label), random_state=random_state)
                )
                negatives = mapping[~pos_mask]
                mapping = pd.concat([positives, negatives], ignore_index=True)

    text_bank_path = out_dir / 'text_bank.csv'
    mapping_path = out_dir / 'ecg_text_labels.csv'
    text_bank.to_csv(text_bank_path, index=False)
    mapping.to_csv(mapping_path, index=False)

    print(f"Saved text bank to {text_bank_path}")
    print(f"Saved mapping to {mapping_path} ({len(mapping):,} rows)")

    if sample_size_for_checks and len(df) > 0:
        sample_n = min(sample_size_for_checks, len(df))
        sample_df = df[iter_cols].sample(n=sample_n, random_state=random_state)
        exclusivity_examples = []
        group_counter = Counter()

        for values in sample_df.itertuples(index=False, name=None):
            ecg_id = values[col_pos['ecg_id']]
            example_info = []

            diag_dict = {}
            bert_dict = {}
            pos_labels = []

            for label in unique_labels:
                diag_col = diag_cols_map.get(label)
                diag_val = values[col_pos[diag_col]] if diag_col and diag_col in col_pos else float('nan')
                bert_col = bert_cols_map.get(label)
                bert_val = values[col_pos[bert_col]] if bert_col and bert_col in col_pos else float('nan')
                diag_dict[label] = diag_val
                bert_dict[label] = bert_val

                if is_excluded_label(label):
                    continue
                is_positive = False
                if not pd.isna(diag_val) and diag_val >= 1:
                    is_positive = True
                if not pd.isna(bert_val) and bert_val > bert_threshold:
                    is_positive = True
                if is_positive:
                    pos_labels.append(label)

            for lab in pos_labels:
                group = label2group.get(lab)
                if not group:
                    continue
                members_for_group = [m for m in group_members(EXCLUSIVE_GROUPS[group]) if m in diag_dict]
                hard_members = []
                for member in members_for_group:
                    if member == lab:
                        continue
                    if member in pos_labels:
                        continue
                    diag_val = diag_dict.get(member)
                    bert_val = bert_dict.get(member)
                    if not pd.isna(diag_val):
                        negative = diag_val <= 0
                    else:
                        negative = not pd.isna(bert_val) and bert_val <= bert_threshold
                    if negative:
                        hard_members.append(member)
                if hard_members:
                    example_info.append((group, lab, hard_members))
                    group_counter[group] += 1

            if example_info:
                exclusivity_examples.append((ecg_id, example_info))

        print(f"\nExclusivity check on {sample_n} ECGs:")
        print(f"  {len(exclusivity_examples)} had mutually exclusive conflicts captured as hard negatives")
        if group_counter:
            top_groups = ", ".join(f"{name}={count}" for name, count in group_counter.most_common(3))
            print(f"  Top exclusivity groups: {top_groups}")
        if exclusivity_examples:
            print("  Sample cases:")
            for ecg_id, info in exclusivity_examples[:5]:
                for group, pos_label, hard_list in info[:2]:
                    hard_str = ", ".join(hard_list[:3])
                    if len(hard_list) > 3:
                        hard_str += ", ..."
                    print(f"    - {ecg_id}: {group} | +{pos_label} vs 0→ {hard_str}")

    if exclusivity_conflicts:
        print(f"\nExclusivity conflicts resolved (kept first positive per group): {len(exclusivity_conflicts)}")
        top_conflicts = ", ".join(f"{grp}={cnt}" for grp, cnt in conflict_counter.most_common(3))
        if top_conflicts:
            print(f"  Conflict breakdown: {top_conflicts}")

    return text_bank_path, mapping_path


def process_dataset(
    input_path: str,
    output_path: str,
    dataset_name: str,
    sample_size: int = None,
    dataset_type: str = 'mimic-iv',
    max_prompts_per_ecg: int = None,
    max_normal_percentage: float = 0.05,
    min_samples_per_diagnosis: int = 1,
    mimic_samples: int = None,
    mhi_samples: int = None,
    prompt_workers: int = 1,
    answer_workers: int = 1,
    preserve_common_rhythms: bool = True,
    custom_parquet_path: Optional[str] = None,
):
    """Process a single dataset (train or test)
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        dataset_name: Name of dataset ('train' or 'test')
        sample_size: Target number of samples to generate
        dataset_type: Type of dataset ('mimic-iv', 'mhi', or 'combined')
        max_prompts_per_ecg: Maximum prompts to generate per ECG
        max_normal_percentage: Maximum percentage of normal ECGs (default 5%)
        min_samples_per_diagnosis: Minimum samples per diagnosis category (default 1)
        mimic_samples: For combined mode, number of MIMIC samples to include
        mhi_samples: For combined mode, number of MHI samples to include
        prompt_workers: Number of worker processes for prompt generation
        answer_workers: Number of worker processes for answer generation
        preserve_common_rhythms: If True, retain Sinusal/Regular labels in outputs
    """
    
    print(f"\n{'='*60}")
    print(f"PROCESSING {dataset_name.upper()} DATASET")
    print(f"Dataset type: {dataset_type.upper()}")
    if dataset_type == 'combined':
        print(f"  MIMIC samples: {mimic_samples or 0}")
        print(f"  MHI samples: {mhi_samples or 0}")
    print(f"{'='*60}")
    
    # 1. Load dataset
    print(f"\n1. Loading {dataset_name} dataset...")
    print(f"   Input: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"   Loaded {len(df)} records")

    # Filter by Split column if dataset_type is 'mhi'
    if dataset_type == 'mhi' and 'Split' in df.columns:
        split_value = 'train' if dataset_name == 'train' else 'test'
        print(f"   Filtering by Split='{split_value}'...")
        df = df[df['Split'] == split_value].copy()
        print(f"   After filtering: {len(df)} records")

    # 2. Drop existing question column if it exists
    if 'question' in df.columns:
        print("   Dropping existing 'question' column...")
        df = df.drop(columns=['question'])

    print(f"   Columns: {list(df.columns)[:10]}...")
    
    # 3. Load and merge demographic data based on dataset type
    if dataset_type == 'custom':
        print(f"\n2. Using custom dataset as-is (no demographic merge)...")
        df_merged = df.copy()
        df_merged['dataset_source'] = 'custom'

    elif dataset_type == 'combined':
        # Process combined dataset - load both MIMIC and MHI data
        print(f"\n2. Processing COMBINED dataset...")
        
        all_dfs = []
        
        # Process MIMIC portion if requested
        if mimic_samples and mimic_samples > 0:
            print(f"\n   Loading MIMIC-IV demographic data...")
            mimic_path = '/media/data1/datasets/MIMIC-IV/Diagnosis/mimic_labelbox_bert_v4_all.parquet'
            mimic_df = pd.read_parquet(mimic_path)
            
            # Extract npy_id for merging
            mimic_df['npy_id'] = mimic_df['npy_path'].str.extract(r'/([^/]+)\.npy$')[0]
            df['npy_id'] = df['waveform_name'].str.replace('.npy', '') if 'waveform_name' in df.columns else df.index.astype(str)
            
            # Select demographic and interval timing columns to merge
            demographic_cols = ['new_PatientID', 'npy_id', 'gender', 'age_at_ecg', 'rr_interval',
                               'p_onset', 'qrs_onset', 'qrs_end', 't_end']  # Add timing columns for interval calculation
            available_cols = [col for col in demographic_cols if col in mimic_df.columns]
            
            # Merge MIMIC data
            df_mimic_merged = df.merge(
                mimic_df[available_cols],
                on='npy_id',
                how='inner',  # Inner join to get only MIMIC matches
                suffixes=('', '_mimic')
            )
            df_mimic_merged['dataset_source'] = 'mimic-iv'
            df_mimic_merged = df_mimic_merged.drop(columns=['npy_id'], errors='ignore')
            
            print(f"     Found {len(df_mimic_merged)} MIMIC-IV records")
            all_dfs.append(df_mimic_merged)
        
        # Process MHI portion if requested
        if mhi_samples and mhi_samples > 0:
            print(f"\n   Loading MHI data (1.7M rows, this may take a moment)...")
            # Load metadata and GT/BERT labels separately, then merge
            metadata_path = '/media/data1/muse_ge/ECG_ad20241231_metadata.v1.6._with_translation_ROXs42Bb.cleaned.parquet'
            labels_path = '/media/data1/muse_ge/ECG_ad20241231_gt_labels_v1.6.parquet'

            print(f"     Loading metadata from v1.6...")
            metadata_df = pd.read_parquet(metadata_path)
            print(f"     Loading GT/BERT labels from v1.6...")
            labels_df = pd.read_parquet(labels_path)

            # Left join so every ECG row from metadata table keeps its place
            print(f"     Merging metadata ({metadata_df.shape[1]} cols) with labels ({labels_df.shape[1]} cols) on npy_path...")
            mhi_df = metadata_df.merge(labels_df, on='npy_path', how='left')
            print(f"     Merged shape: {mhi_df.shape}")

            # v1.6 uses npy_path, not waveform_path_psa
            mhi_df = mhi_df.dropna(subset=['npy_path'])
            # Extract npy_id from npy_path for merging
            
            mhi_df['npy_id'] = mhi_df['npy_path'].str.extract(r'/([^/]+)\.npy$')[0]
            df['npy_id'] = df['waveform_name'].str.replace('.npy', '') if 'waveform_name' in df.columns else df.index.astype(str)

            # Map MHI columns to standard names
            mhi_df['gender'] = mhi_df['RestingECG_PatientDemographics_Gender'].map({'MALE': 'M', 'FEMALE': 'F'})
            mhi_df['age_at_ecg'] = pd.to_numeric(mhi_df['RestingECG_PatientDemographics_PatientAge'], errors='coerce')
            mhi_df['rr_interval'] = pd.to_numeric(mhi_df['RestingECG_QRSTimesTypes_GlobalRR'], errors='coerce')

            # v1.6 has PatientID, not new_PatientID
            if 'PatientID' in mhi_df.columns and 'new_PatientID' not in mhi_df.columns:
                mhi_df['new_PatientID'] = mhi_df['PatientID']

            # Add VentricularRate column for MHI heart rate
            if 'RestingECG_OriginalRestingECGMeasurements_VentricularRate' in mhi_df.columns:
                mhi_df['RestingECG_OriginalRestingECGMeasurements_VentricularRate'] = pd.to_numeric(
                    mhi_df['RestingECG_OriginalRestingECGMeasurements_VentricularRate'], errors='coerce'
                )
            
            # Select columns to merge (including interval measurements and echonext)
            # Only include columns that actually exist in v1.6
            desired_merge_cols = ['npy_id', 'new_PatientID', 'gender', 'age_at_ecg', 'rr_interval',
                          'RestingECG_OriginalRestingECGMeasurements_VentricularRate',
                          'RestingECG_OriginalRestingECGMeasurements_PRInterval',
                          'RestingECG_OriginalRestingECGMeasurements_QRSDuration',
                          'RestingECG_OriginalRestingECGMeasurements_QTInterval',
                          'RestingECG_OriginalRestingECGMeasurements_QTCorrected',
                          'RestingECG_OriginalRestingECGMeasurements_QTcFrederica',
                          'echonext_shd',  # Add structural heart disease column
                          'deepecho_Visually_Estimated_EF',  # Add LVEF column
                          'acs_condition_severity',  # Add ACS severity column
                          'acs_pci_regions',  # Add ACS culprit artery column
                          'afib_label_2y',  # Add AFib 2-year risk prediction
                          'afib_label_5y',  # Add AFib 5-year risk prediction
                          'GT_Afib',  # Current AFib status (ground truth)
                          'BERT_Afib',  # AFib BERT model prediction
                          'translated_diagnosis',  # English diagnosis text
                          'diagnosis']  # Original diagnosis text (fallback)

            # Filter to only columns that exist in mhi_df
            merge_cols = [col for col in desired_merge_cols if col in mhi_df.columns]

            print(f"     Merging {len(merge_cols)} columns from MHI v1.6")

            # Remove duplicates from merge_cols
            merge_cols = list(dict.fromkeys(merge_cols))
            
            # Merge MHI data
            df_mhi_merged = df.merge(
                mhi_df[merge_cols],
                on='npy_id',
                how='inner',  # Inner join to get only MHI matches
                suffixes=('', '_mhi')
            )
            df_mhi_merged['dataset_source'] = 'mhi'

            # Ensure 'report' field uses translated_diagnosis (English) for MHI records
            if 'translated_diagnosis' in df_mhi_merged.columns:
                # Update report with translated_diagnosis where available
                has_translated = df_mhi_merged['translated_diagnosis'].notna()
                if has_translated.any():
                    df_mhi_merged.loc[has_translated, 'report'] = df_mhi_merged.loc[has_translated, 'translated_diagnosis']
                    print(f"     Updated {has_translated.sum()} MHI reports with translated_diagnosis")
            elif 'diagnosis' in df_mhi_merged.columns:
                # Fallback to diagnosis if translated_diagnosis not available
                has_diagnosis = df_mhi_merged['diagnosis'].notna() & df_mhi_merged['report'].isna()
                if has_diagnosis.any():
                    df_mhi_merged.loc[has_diagnosis, 'report'] = df_mhi_merged.loc[has_diagnosis, 'diagnosis']
                    print(f"     WARNING: Using non-translated diagnosis for {has_diagnosis.sum()} MHI reports")

            df_mhi_merged = df_mhi_merged.drop(columns=['npy_id'], errors='ignore')
            
            print(f"     Found {len(df_mhi_merged)} MHI records")
            all_dfs.append(df_mhi_merged)
        
        # Combine all dataframes
        if all_dfs:
            df_merged = pd.concat(all_dfs, ignore_index=True)
            print(f"\n   Total combined records: {len(df_merged)}")
            print(f"   Dataset distribution:")
            print(df_merged['dataset_source'].value_counts())
        else:
            raise ValueError("No data specified for combined mode. Set mimic_samples and/or mhi_samples.")
        
    elif dataset_type == 'mimic-iv':
        print(f"\n2. Loading MIMIC-IV demographic data...")
        mimic_path = '/media/data1/datasets/MIMIC-IV/Diagnosis/mimic_labelbox_bert_v4_all.parquet'
        mimic_df = pd.read_parquet(mimic_path)
        
        # Extract npy_id for merging
        mimic_df['npy_id'] = mimic_df['npy_path'].str.extract(r'/([^/]+)\.npy$')[0]
        df['npy_id'] = df['waveform_name'].str.replace('.npy', '') if 'waveform_name' in df.columns else df.index.astype(str)
        
        # Select demographic and interval timing columns to merge
        demographic_cols = ['new_PatientID', 'npy_id', 'gender', 'age_at_ecg', 'rr_interval',
                           'p_onset', 'qrs_onset', 'qrs_end', 't_end']  # Add timing columns for interval calculation
        available_cols = [col for col in demographic_cols if col in mimic_df.columns]
        
        # Merge
        print(f"   Merging columns: {available_cols}")
        df_merged = df.merge(
            mimic_df[available_cols],
            on='npy_id',
            how='left',
            suffixes=('', '_mimic')
        )
        
        # Clean up
        df_merged = df_merged.drop(columns=['npy_id'], errors='ignore')
        df_merged['dataset_source'] = 'custom' if dataset_type == 'custom' else 'mimic-iv'
        
    elif dataset_type == 'mhi':
        print(f"\n2. Loading MHI data (1.7M rows, this may take a moment)...")
        # Load metadata and GT/BERT labels separately, then merge
        metadata_path = '/media/data1/muse_ge/ECG_ad20241231_metadata.v1.6._with_translation_ROXs42Bb.cleaned.parquet'
        labels_path = '/media/data1/muse_ge/ECG_ad20241231_gt_labels_v1.6.parquet'

        print(f"     Loading metadata from v1.6...")
        metadata_df = pd.read_parquet(metadata_path)
        print(f"     Loading GT/BERT labels from v1.6...")
        labels_df = pd.read_parquet(labels_path)

        # Left join so every ECG row from metadata table keeps its place
        print(f"     Merging metadata ({metadata_df.shape[1]} cols) with labels ({labels_df.shape[1]} cols) on npy_path...")
        mhi_df = metadata_df.merge(labels_df, on='npy_path', how='left')
        print(f"     Merged shape: {mhi_df.shape}")
        
        # Extract npy_id from npy_path for merging
        mhi_df['npy_id'] = mhi_df['npy_path'].str.extract(r'/([^/]+)\.npy$')[0]
        df['npy_id'] = df['waveform_name'].str.replace('.npy', '') if 'waveform_name' in df.columns else df.index.astype(str)

        # Map MHI columns to standard names
        mhi_df['gender'] = mhi_df['RestingECG_PatientDemographics_Gender'].map({'MALE': 'M', 'FEMALE': 'F'})
        mhi_df['age_at_ecg'] = pd.to_numeric(mhi_df['RestingECG_PatientDemographics_PatientAge'], errors='coerce')
        mhi_df['rr_interval'] = pd.to_numeric(mhi_df['RestingECG_QRSTimesTypes_GlobalRR'], errors='coerce')

        # v1.6 has PatientID, not new_PatientID
        if 'PatientID' in mhi_df.columns and 'new_PatientID' not in mhi_df.columns:
            mhi_df['new_PatientID'] = mhi_df['PatientID']

        # Add VentricularRate column for MHI heart rate
        if 'RestingECG_OriginalRestingECGMeasurements_VentricularRate' in mhi_df.columns:
            mhi_df['RestingECG_OriginalRestingECGMeasurements_VentricularRate'] = pd.to_numeric(
                mhi_df['RestingECG_OriginalRestingECGMeasurements_VentricularRate'], errors='coerce'
            )
        
        # Select columns to merge (including interval measurements and echonext)
        # Only include columns that actually exist in v1.6
        desired_merge_cols = ['npy_id', 'new_PatientID', 'gender', 'age_at_ecg', 'rr_interval',
                      'RestingECG_OriginalRestingECGMeasurements_VentricularRate',
                      'RestingECG_OriginalRestingECGMeasurements_PRInterval',
                      'RestingECG_OriginalRestingECGMeasurements_QRSDuration',
                      'RestingECG_OriginalRestingECGMeasurements_QTInterval',
                      'RestingECG_OriginalRestingECGMeasurements_QTCorrected',
                      'RestingECG_OriginalRestingECGMeasurements_QTcFrederica',
                      'echonext_shd',  # Add structural heart disease column
                      'deepecho_Visually_Estimated_EF',  # Add LVEF column
                      'acs_condition_severity',  # Add ACS severity column
                      'acs_pci_regions',  # Add ACS culprit artery column
                      'afib_label_2y',  # Add AFib 2-year risk prediction
                      'afib_label_5y',  # Add AFib 5-year risk prediction
                      'GT_Afib',  # Current AFib status (ground truth)
                      'BERT_Afib',  # AFib BERT model prediction
                      'translated_diagnosis',  # English diagnosis text
                      'diagnosis']  # Original diagnosis text (fallback)

        # Filter to only columns that exist in mhi_df
        merge_cols = [col for col in desired_merge_cols if col in mhi_df.columns]

        print(f"   Merging {len(merge_cols)} columns from MHI v1.6")

        # Remove duplicates from merge_cols
        merge_cols = list(dict.fromkeys(merge_cols))
        
        # Merge with indicator to track unmatched records
        print(f"   Merging demographic and diagnostic columns...")
        df_merged = df.merge(
            mhi_df[merge_cols],
            on='npy_id',
            how='left',
            suffixes=('', '_mhi'),
            indicator=True
        )
        
        # Report merge statistics
        merge_stats = df_merged['_merge'].value_counts()
        if 'left_only' in merge_stats.index:
            unmatched = merge_stats['left_only']
            pct_unmatched = 100 * unmatched / len(df_merged)
            print(f"   WARNING: {unmatched} records ({pct_unmatched:.1f}%) not found in MHI data")

            # If this is test dataset and >30% unmatched, reassign to train
            if dataset_name == 'test' and pct_unmatched > 30:
                print(f"   NOTE: {pct_unmatched:.1f}% > 30% unmatched in test set, these will be reassigned to train")

        # Ensure 'report' field uses translated_diagnosis (English) for MHI records
        if 'translated_diagnosis' in df_merged.columns:
            # Update report with translated_diagnosis where available
            has_translated = df_merged['translated_diagnosis'].notna()
            if has_translated.any():
                df_merged.loc[has_translated, 'report'] = df_merged.loc[has_translated, 'translated_diagnosis']
                print(f"   Updated {has_translated.sum()} MHI reports with translated_diagnosis")
        elif 'diagnosis' in df_merged.columns:
            # Fallback to diagnosis if translated_diagnosis not available
            has_diagnosis = df_merged['diagnosis'].notna() & df_merged['report'].isna()
            if has_diagnosis.any():
                df_merged.loc[has_diagnosis, 'report'] = df_merged.loc[has_diagnosis, 'diagnosis']
                print(f"   WARNING: Using non-translated diagnosis for {has_diagnosis.sum()} MHI reports")

        # Clean up
        df_merged = df_merged.drop(columns=['npy_id', '_merge'], errors='ignore')
        df_merged['dataset_source'] = 'mhi'

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Optionally drop ultra-common rhythm tags from downstream prediction targets
    if not preserve_common_rhythms:
        for common_rhythm in ("Sinusal", "Regular"):
            if common_rhythm in df_merged.columns:
                df_merged[common_rhythm] = 0

    # Add derived clinical categories that drive special question balancing
    df_merged = enrich_special_question_columns(df_merged)
    df_merged = df_merged.reset_index(drop=True)
    if '__row_id__' not in df_merged.columns:
        df_merged['__row_id__'] = df_merged.index

    # Report demographic data availability
    for col in ['gender', 'age_at_ecg', 'rr_interval']:
        if col in df_merged.columns:
            available = df_merged[col].notna().sum()
            print(f"   {col}: {available}/{len(df_merged)} ({100*available/len(df_merged):.1f}%)")
    
    # 3a. Generate ecg_type column based on deepecg.json
    print(f"\n2a. Generating ecg_type column based on pathological/limit classifications...")
    
    # Load deepecg dictionary and categories from constants
    from utils.constants import DEEPECG_PATHOLOGICAL_LIMIT, DEEPECG_CATEGORIES
    
    pathological_cols = DEEPECG_PATHOLOGICAL_LIMIT['deepecg']['pathological']
    limit_cols = DEEPECG_PATHOLOGICAL_LIMIT['deepecg']['limit']
    
    # Get all diagnostic columns from deepecg_categories
    all_diagnosis_cols = []
    for category, diagnoses in DEEPECG_CATEGORIES.items():
        all_diagnosis_cols.extend(diagnoses)
    
    # Initialize ecg_type as normal
    df_merged['ecg_type'] = 'normal'
    
    # Check for pathological conditions
    for col in pathological_cols:
        if col in df_merged.columns:
            # Mark as pathological if any pathological column >= 1
            df_merged.loc[df_merged[col] >= 1, 'ecg_type'] = 'pathological'
    
    # Check for borderline conditions (only if not already pathological)
    for col in limit_cols:
        if col in df_merged.columns:
            # Mark as borderline if any limit column >= 1 and not already pathological
            df_merged.loc[(df_merged[col] >= 1) & (df_merged['ecg_type'] == 'normal'), 'ecg_type'] = 'borderline'
    
    # Report ecg_type distribution
    ecg_type_counts = df_merged['ecg_type'].value_counts()
    print(f"   ECG type distribution:")
    for ecg_type, count in ecg_type_counts.items():
        print(f"     {ecg_type}: {count} ({100*count/len(df_merged):.1f}%)")
    
    # 4. Calculate how many ECGs we need for the target number of questions
    # For combined mode, handle sampling differently
    if dataset_type == 'combined' and (mimic_samples or mhi_samples):
        print(f"\n3. Selecting ECGs for combined dataset...")
        
        sampled_dfs = []
        
        # Sample from each dataset source
        for source in df_merged['dataset_source'].unique():
            source_df = df_merged[df_merged['dataset_source'] == source]
            
            # Determine target samples for this source
            if source == 'mimic-iv':
                target_samples = mimic_samples or 0
            else:  # mhi
                target_samples = mhi_samples or 0
            
            if target_samples > 0:
                # Calculate approximate ECGs needed
                if max_prompts_per_ecg:
                    avg_prompts_per_ecg = max_prompts_per_ecg
                    target_ecgs = max(int(target_samples / avg_prompts_per_ecg), 1)
                else:
                    target_ecgs = max(int(target_samples / 6), 1)
                
                print(f"\n   {source.upper()}: Target ~{target_ecgs} ECGs for ~{target_samples} questions")
                
                # For MHI, apply stratified sampling for special questions
                if source == 'mhi':
                    # Use stratified sampling function for MHI
                    # For test: 1000 special questions out of 5000 (20%)
                    # For train: 40000 special questions out of 200000 (20%)
                    special_target = int(target_samples * 0.2)  # 20% should be special questions
                    sampled_source = apply_mhi_special_stratified_sampling(source_df, target_ecgs, special_target)
                    if sampled_source is not None and len(sampled_source) > 0:
                        sampled_source = enforce_normal_cap(
                            sampled_source,
                            source_df,
                            max_normal_percentage,
                            random_state=42,
                            context=f"{source} stratified sampling",
                        )
                        sampled_dfs.append(sampled_source)
                        
                        # Report sampling for MHI with special stratification
                        ecg_type_counts = sampled_source['ecg_type'].value_counts()
                        print(f"     Sampled {len(sampled_source)} ECGs from {source} with stratified sampling:")
                        for ecg_type, count in ecg_type_counts.items():
                            print(f"       {ecg_type}: {count} ({count/len(sampled_source)*100:.1f}%)")
                        
                        # Report special question distributions
                        if 'afib_label_2y' in sampled_source.columns and 'afib_label_5y' in sampled_source.columns:
                            high_risk_afib = sampled_source[(sampled_source['afib_label_2y'] == True) | (sampled_source['afib_label_5y'] == True)]
                            print(f"       AFib high risk: {len(high_risk_afib)} ({len(high_risk_afib)/len(sampled_source)*100:.1f}%)")
                        
                        if 'echonext_shd' in sampled_source.columns:
                            abnormal_shd = sampled_source[sampled_source['echonext_shd'] >= 1]
                            print(f"       Abnormal SHD: {len(abnormal_shd)} ({len(abnormal_shd)/len(sampled_source)*100:.1f}%)")
                        
                        if 'acs_condition_severity' in sampled_source.columns:
                            from utils.constants import ACS_ACUTE_CONDITIONS
                            acute_acs = sampled_source[sampled_source['acs_condition_severity'].isin(ACS_ACUTE_CONDITIONS)]
                            print(f"       Acute ACS: {len(acute_acs)} ({len(acute_acs)/len(sampled_source)*100:.1f}%)")
                else:
                    # For MIMIC, use original sampling logic
                    # Apply max_normal_percentage constraint for this source
                    max_normal_for_source = int(target_ecgs * max_normal_percentage)
                    
                    # Separate by ECG type
                    source_pathological = source_df[source_df['ecg_type'] == 'pathological']
                    source_borderline = source_df[source_df['ecg_type'] == 'borderline']
                    source_normal = source_df[source_df['ecg_type'] == 'normal']
                    
                    print(f"     Available in {source}: {len(source_pathological)} pathological, {len(source_borderline)} borderline, {len(source_normal)} normal")
                    print(f"     Max normal allowed: {max_normal_for_source} ({max_normal_percentage*100:.1f}%)")
                    
                    # Sample with constraint
                    source_sampled = []
                    
                    # First, sample pathological and borderline proportionally
                    non_normal_slots = target_ecgs - max_normal_for_source
                    path_border_total = len(source_pathological) + len(source_borderline)
                    
                    if path_border_total > 0:
                        # Sample pathological
                        n_path = min(int(non_normal_slots * len(source_pathological) / path_border_total), len(source_pathological))
                        if n_path > 0:
                            source_sampled.append(source_pathological.sample(n=n_path, random_state=42))
                        
                        # Sample borderline
                        n_border = min(int(non_normal_slots * len(source_borderline) / path_border_total), len(source_borderline))
                        if n_border > 0:
                            source_sampled.append(source_borderline.sample(n=n_border, random_state=42))
                        
                        # Sample normal (up to max allowed)
                        n_normal = min(max_normal_for_source, len(source_normal))
                        if n_normal > 0:
                            source_sampled.append(source_normal.sample(n=n_normal, random_state=42))
                        
                        # Combine samples from this source
                        if source_sampled:
                            sampled_source = pd.concat(source_sampled, ignore_index=True)
                            
                            # If we need more to reach target, sample more pathological/borderline
                            if len(sampled_source) < target_ecgs:
                                remaining_needed = target_ecgs - len(sampled_source)
                                already_sampled_idx = sampled_source.index
                                remaining_non_normal = pd.concat([source_pathological, source_borderline])
                                remaining_non_normal = remaining_non_normal[~remaining_non_normal.index.isin(already_sampled_idx)]
                                if len(remaining_non_normal) > 0:
                                    additional = remaining_non_normal.sample(n=min(remaining_needed, len(remaining_non_normal)), random_state=42)
                                    sampled_source = pd.concat([sampled_source, additional], ignore_index=True)
                            
                            sampled_source = enforce_normal_cap(
                                sampled_source,
                                source_df,
                                max_normal_percentage,
                                random_state=42,
                                context=f"{source} source sampling",
                            )

                            sampled_dfs.append(sampled_source)
                            
                            # Report sampling for this source
                            ecg_type_counts = sampled_source['ecg_type'].value_counts()
                            print(f"     Sampled {len(sampled_source)} ECGs from {source}:")
                            for ecg_type, count in ecg_type_counts.items():
                                print(f"       {ecg_type}: {count} ({count/len(sampled_source)*100:.1f}%)")
        
        df_sampled = pd.concat(sampled_dfs, ignore_index=True) if sampled_dfs else df_merged
        print(f"\n   Total ECGs selected: {len(df_sampled)}")
        print(f"   Dataset distribution:")
        print(df_sampled['dataset_source'].value_counts())
        
    # Adjust expected prompts per ECG based on max_prompts_per_ecg
    elif sample_size:
        if max_prompts_per_ecg:
            avg_prompts_per_ecg = max_prompts_per_ecg
            target_ecgs = max(int(sample_size / avg_prompts_per_ecg), 1)
            print(f"\n3. Selecting ECGs to generate exactly {sample_size:,} questions...")
            print(
                f"   Target: ~{target_ecgs:,} ECGs (expecting ~{avg_prompts_per_ecg:.1f} questions per ECG with max={max_prompts_per_ecg})"
            )
        else:
            target_ecgs = max(int(sample_size / 6), 1)
            print(f"\n3. Selecting ECGs to generate exactly {sample_size:,} questions...")
            print(f"   Target: ~{target_ecgs:,} ECGs (expecting ~6 questions per ECG)")

        target_ecg_count = target_ecgs

        ecg_type_counts = df_merged['ecg_type'].value_counts()
        total_available = len(df_merged)
        print(f"   Available ECGs by type:")
        for ecg_type, count in ecg_type_counts.items():
            print(f"     {ecg_type.capitalize()}: {count:,} ({100*count/total_available:.1f}%)")

        print(f"\n   Ensuring minimum {min_samples_per_diagnosis} samples per diagnosis...")
        diagnosis_ecg_indices = {}
        for diagnosis in all_diagnosis_cols:
            if diagnosis in df_merged.columns:
                ecgs_with_diagnosis = df_merged[df_merged[diagnosis] >= 1].index
                if len(ecgs_with_diagnosis) > 0:
                    diagnosis_ecg_indices[diagnosis] = ecgs_with_diagnosis
                    print(f"     {diagnosis}: {len(ecgs_with_diagnosis)} ECGs available")

        selected_indices: Set[int] = set()
        insufficient_diagnoses: list[tuple[str, int]] = []
        for diagnosis, indices in diagnosis_ecg_indices.items():
            available = len(indices)
            if available < min_samples_per_diagnosis:
                insufficient_diagnoses.append((diagnosis, available))
            n_to_select = min(min_samples_per_diagnosis, available)
            if n_to_select > 0:
                chosen = np.random.choice(indices, size=n_to_select, replace=False)
                selected_indices.update(chosen)

        coverage_df = (
            df_merged.loc[list(selected_indices)].copy()
            if selected_indices
            else df_merged.iloc[0:0].copy()
        )
        coverage_count = len(coverage_df)
        print(f"\n   Selected {coverage_count} ECGs for minimum diagnosis coverage")

        if insufficient_diagnoses:
            print("   [coverage-warning] Unable to satisfy minimum samples for the following diagnoses:")
            for diagnosis, available in sorted(insufficient_diagnoses):
                print(f"     - {diagnosis}: found {available} ECGs (requested {min_samples_per_diagnosis})")

        if coverage_count > 0 and '__row_id__' in coverage_df.columns:
            coverage_ids = set(coverage_df['__row_id__'])
        elif coverage_count > 0:
            coverage_ids = set(coverage_df.index)
        else:
            coverage_ids = set()

        remaining_target = max(target_ecgs - coverage_count, 0)
        normal_cap_total = int(target_ecgs * max_normal_percentage)
        coverage_normals = int((coverage_df['ecg_type'] == 'normal').sum())
        normal_cap_remaining = max(0, normal_cap_total - coverage_normals)

        print(f"\n   Sampling remaining {remaining_target} ECGs...")
        print(
            f"   Normal cap: <= {normal_cap_total} ECGs ({max_normal_percentage*100:.1f}% of {target_ecgs:,})"
        )

        if coverage_ids and '__row_id__' in df_merged.columns:
            available_df = df_merged[~df_merged['__row_id__'].isin(coverage_ids)]
        else:
            available_df = df_merged

        available_non_normal = available_df[available_df['ecg_type'] != 'normal']
        available_normal = available_df[available_df['ecg_type'] == 'normal']

        sampled_frames = []
        if not coverage_df.empty:
            sampled_frames.append(coverage_df)

        if remaining_target > 0:
            non_normal_take = min(remaining_target, len(available_non_normal))
            if non_normal_take > 0:
                sampled_non_normal = available_non_normal.sample(n=non_normal_take, random_state=42)
                sampled_frames.append(sampled_non_normal)
                remaining_target -= non_normal_take
                print(f"     Sampled {non_normal_take:,} non-normal ECGs")
                if '__row_id__' in sampled_non_normal.columns:
                    used_ids = set(sampled_non_normal['__row_id__'])
                    available_normal = available_normal[~available_normal['__row_id__'].isin(used_ids)]
                    available_non_normal = available_non_normal.drop(sampled_non_normal.index, errors='ignore')

            if remaining_target > 0 and normal_cap_remaining > 0:
                normal_take = min(remaining_target, normal_cap_remaining, len(available_normal))
                if normal_take > 0:
                    sampled_normals = available_normal.sample(n=normal_take, random_state=42)
                    sampled_frames.append(sampled_normals)
                    remaining_target -= normal_take
                    normal_cap_remaining -= normal_take
                    print(
                        f"     Sampled {normal_take:,} normal ECGs (within {max_normal_percentage*100:.1f}% cap)"
                    )

            if remaining_target > 0:
                fallback_pool = available_non_normal if len(available_non_normal) > 0 else available_df
                fallback_replace = len(fallback_pool) < remaining_target
                sampled_extra = fallback_pool.sample(
                    n=remaining_target,
                    replace=fallback_replace,
                    random_state=42,
                )
                sampled_frames.append(sampled_extra)
                print(
                    f"     Sampled {remaining_target:,} additional ECGs with"
                    f" {'replacement' if fallback_replace else 'unique selection'} to reach target"
                )
                remaining_target = 0

        if sampled_frames:
            df_sampled = pd.concat(sampled_frames, ignore_index=True)
        else:
            df_sampled = available_df.sample(
                n=min(target_ecgs, len(available_df)),
                random_state=42,
            ).reset_index(drop=True)

        if '__row_id__' in df_sampled.columns:
            df_sampled = df_sampled.drop_duplicates(subset='__row_id__', keep='first').reset_index(drop=True)

        if len(df_sampled) > target_ecgs:
            df_sampled = _trim_to_target(df_sampled, target_ecgs, coverage_ids, random_state=42)

        if len(df_sampled) < target_ecgs:
            df_sampled = _top_off_sample(
                df_sampled,
                df_merged,
                target_ecgs,
                max_normal_percentage,
                coverage_ids,
                random_state=42,
            )

        print(f"\n   Selected ECG composition:")
        for ecg_type, count in df_sampled['ecg_type'].value_counts().items():
            pct = 100 * count / len(df_sampled)
            print(f"     {ecg_type}: {count:,} ECGs ({pct:.1f}%)")
        print(f"     Total: {len(df_sampled):,} ECGs")
    else:
        df_sampled = df_merged
        print(f"\n3. Using all {len(df_sampled):,} ECGs (no sampling)")

    df_sampled = enforce_normal_cap(
        df_sampled,
        df_merged,
        max_normal_percentage,
        random_state=42,
        context=f"{dataset_name} final selection",
    )

    df_sampled = df_sampled.reset_index(drop=True)
    df_sampled = df_sampled.drop(columns=['__row_id__'], errors='ignore')

    # 5. Generate prompts for each ECG
    print(f"\n4. Generating prompts...")
    if max_prompts_per_ecg:
        print(f"   Max prompts per ECG: {max_prompts_per_ecg}")

    df_with_prompts = generate_prompts_for_dataset(
        df_sampled,
        sample_size=sample_size,
        max_prompts_per_ecg=max_prompts_per_ecg,
        prompt_workers=prompt_workers,
    )

    print(f"   Total prompts generated: {len(df_with_prompts):,}")
    unique_ecgs = (
        df_with_prompts['waveform_name'].nunique()
        if 'waveform_name' in df_with_prompts.columns and not df_with_prompts.empty
        else 'N/A'
    )
    print(f"   Unique ECGs: {unique_ecgs}")
    avg_prompts = (
        len(df_with_prompts) / max(len(df_sampled), 1)
        if not df_with_prompts.empty
        else 0.0
    )
    print(f"   Average prompts per ECG: {avg_prompts:.2f}")

    # Report prompt distribution
    if not df_with_prompts.empty and 'prompt_category' in df_with_prompts.columns:
        prompt_distribution = df_with_prompts['prompt_category'].value_counts()
        print(f"\n   Prompt type distribution:")
        for prompt_type, count in prompt_distribution.head(10).items():
            pct = 100 * count / len(df_with_prompts)
            print(f"     {prompt_type}: {count:,} ({pct:.1f}%)")

    # Check for random finding questions
    random_findings = (
        df_with_prompts[df_with_prompts['prompt_category'] == 'random_finding_question']
        if not df_with_prompts.empty and 'prompt_category' in df_with_prompts.columns
        else pd.DataFrame()
    )
    if len(random_findings) > 0:
        print(f"\n   Random finding questions: {len(random_findings)} ({len(random_findings)/len(df_with_prompts)*100:.2f}%)")

    # 6. Generate answers for each prompt
    print(f"\n5. Generating answers...")
    df_with_answers, dropped_count = generate_answers_for_dataset(
        df_with_prompts,
        dataset_type=dataset_type,
        answer_workers=answer_workers,
    )
    if dropped_count > 0:
        print(f"     Dropped {dropped_count} questions with no available data")
    
    # Keep original report for comparison
    if 'report' in df_with_answers.columns:
        df_with_answers['original_report'] = df_with_answers['report']
    
    # 7. Show sample outputs
    print(f"\n6. Sample outputs:")
    print(f"{'='*60}")
    
    # Show examples from each category
    sample_categories = ['interpretation', 'json_interpretation', 'heart_rate', 'classification', 'category_infarct_ischemia']
    
    for cat in sample_categories:
        subset = df_with_answers[df_with_answers['prompt_category'] == cat]
        if len(subset) > 0:
            example = subset.iloc[0]
            print(f"\n{cat.upper()}:")
            print(f"   Prompt: {example['prompt'][:80]}...")
            if cat == 'json_interpretation':
                print(f"   Answer: {example['generated_answer'][:150]}...")
            else:
                print(f"   Answer: {example['generated_answer']}")
    
    # 8. Save results
    print(f"\n7. Saving results...")
    print(f"   Output: {output_path}")
    
    # Rename dataset_source to dataset for consistency
    if 'dataset_source' in df_with_answers.columns:
        df_with_answers['dataset'] = df_with_answers['dataset_source'].replace({'mimic-iv': 'mimic'})
        df_with_answers = df_with_answers.drop(columns=['dataset_source'])
    
    # Ensure demographic columns and dataset are preserved
    columns_to_keep = list(df_with_answers.columns)
    for col in ['gender', 'age_at_ecg', 'rr_interval', 'dataset']:
        if col not in columns_to_keep and col in df_with_answers.columns:
            columns_to_keep.append(col)
    
    # Filter to existing columns
    columns_to_keep = [c for c in columns_to_keep if c in df_with_answers.columns]
    
    # Ensure numeric columns are properly typed before saving
    if 'age_at_ecg' in df_with_answers.columns:
        df_with_answers['age_at_ecg'] = pd.to_numeric(df_with_answers['age_at_ecg'], errors='coerce')
    if 'rr_interval' in df_with_answers.columns:
        df_with_answers['rr_interval'] = pd.to_numeric(df_with_answers['rr_interval'], errors='coerce')

    # Harmonize path column for deduplication
    if 'waveform_path_psa' not in df_with_answers.columns:
        if 'ecg_path' in df_with_answers.columns:
            df_with_answers['waveform_path_psa'] = df_with_answers['ecg_path']
            if 'waveform_path_psa' not in columns_to_keep:
                columns_to_keep.append('waveform_path_psa')
        elif 'npy_path' in df_with_answers.columns:
            df_with_answers['waveform_path_psa'] = df_with_answers['npy_path']
            if 'waveform_path_psa' not in columns_to_keep:
                columns_to_keep.append('waveform_path_psa')
    
    # Final safety-net: deduplicate on ECG path + prompt to prevent residual duplicates
    dedup_subset = ['waveform_path_psa', 'prompt']
    missing_dedup_cols = [c for c in dedup_subset if c not in df_with_answers.columns]
    if not missing_dedup_cols:
        before_rows = len(df_with_answers)
        df_with_answers = df_with_answers.drop_duplicates(subset=dedup_subset, keep='first').reset_index(drop=True)
        after_rows = len(df_with_answers)
        removed = before_rows - after_rows
        if removed > 0:
            print(f"   Deduplicated on {dedup_subset}: removed {removed:,} rows ({before_rows:,} -> {after_rows:,})")
    else:
        # If expected columns missing, proceed without dedup but log a warning
        print(f"   WARNING: Skipping final dedup; missing columns: {missing_dedup_cols}")
    
    # Save parquet
    df_with_answers[columns_to_keep].to_parquet(output_path, index=False)
    
    # Save sample CSV
    sample_csv_path = output_path.replace('.parquet', '_sample.csv')
    df_with_answers.head(200).to_csv(sample_csv_path, index=False)
    print(f"   Sample CSV: {sample_csv_path}")
    
    print(f"\n✓ {dataset_name.upper()} DATASET COMPLETE!")
    
    return df_with_answers


def main(
    dataset_type: str = 'mimic-iv',
    train_samples: int = 400000,
    test_samples: int = 10000,
    max_prompts_per_ecg: int = 1,
    max_normal_percentage: float = 0.05,
    min_samples_per_diagnosis: int = 2,
    mimic_train_samples: int = None,
    mhi_train_samples: int = None,
    mimic_test_samples: int = None,
    mhi_test_samples: int = None,
    generate_siglip_alignment: bool = False,
    siglip_output_dir: str = 'ecg_text_alignment',
    siglip_parquet_path: str = '/media/data1/muse_ge/ECG_ad20241231_cat_labels_v1.4.complete.ROXs42Bb.parquet',
    siglip_include_qa: bool = True,
    siglip_sample_size: int = 1000,
    siglip_pos_weight: float = 1.0,
    siglip_hardneg_weight: float = 3.0,
    siglip_implneg_weight: float = 0.2,
    siglip_random_seed: int = 0,
    siglip_implneg_sample_size: int = 64,
    siglip_max_hardneg_per_group: int | None = 3,
    output_dir: str = "/volume/ECG_tokenizer/output",
    prompt_workers: int = 1,
    answer_workers: int = 1,
    siglip_train_ecgs: Optional[int] = None,
    siglip_val_ecgs: Optional[int] = None,
    siglip_test_ecgs: Optional[int] = None,
    siglip_max_positive_per_label: Optional[int] = None,
    preserve_common_rhythms: bool = True,
    custom_parquet_path: Optional[str] = None,
):
    """Main function to process both train and test datasets
    
    Args:
        dataset_type: Type of dataset ('mimic-iv', 'mhi', or 'combined')
        train_samples: Number of training samples to generate (for single dataset mode)
        test_samples: Number of test/validation samples to generate (for single dataset mode)
        max_prompts_per_ecg: Maximum prompts per ECG
        max_normal_percentage: Maximum percentage of normal ECGs (default 5%)
        min_samples_per_diagnosis: Minimum samples per diagnosis category (default 2)
        mimic_train_samples: For combined mode, number of MIMIC training samples
        mhi_train_samples: For combined mode, number of MHI training samples
        mimic_test_samples: For combined mode, number of MIMIC test samples
        mhi_test_samples: For combined mode, number of MHI test samples
        output_dir: Base directory for QA parquet outputs
        prompt_workers: Number of worker processes for prompt generation
        answer_workers: Number of worker processes for answer generation
        preserve_common_rhythms: If True, retain Sinusal/Regular labels in outputs
    """
    
    print("GENERATING TRAIN AND TEST DATASETS", flush=True)
    print("=" * 80)
    print(f"Dataset type: {dataset_type.upper()}")
    if dataset_type != 'custom':
        print(f"Target sizes: Train={train_samples:,}, Test={test_samples:,}")
    print(f"Max normal percentage: {max_normal_percentage*100:.1f}%")
    print(f"Min samples per diagnosis: {min_samples_per_diagnosis}")
    
    # Define paths based on dataset type
    if dataset_type == 'siglip':
        print("\nNote: Generating SigLIP text bank and mapping directly from MHI parquet")

        train_limit = siglip_train_ecgs if siglip_train_ecgs is not None else (train_samples if train_samples > 0 else None)
        val_limit = siglip_val_ecgs
        test_limit = siglip_test_ecgs if siglip_test_ecgs is not None else (test_samples if test_samples > 0 else None)

        text_bank_path, mapping_path = generate_siglip_alignment_dataset(
            parquet_path=siglip_parquet_path,
            output_dir=siglip_output_dir,
            include_qa=False,
            w_pos=siglip_pos_weight,
            w_hardneg=siglip_hardneg_weight,
            w_implneg=siglip_implneg_weight,
            sample_size_for_checks=siglip_sample_size,
            random_state=siglip_random_seed,
            implicit_negative_sample_size=siglip_implneg_sample_size,
            max_hardneg_per_group=siglip_max_hardneg_per_group,
            train_ecg_limit=train_limit,
            val_ecg_limit=val_limit,
            test_ecg_limit=test_limit,
            max_normal_percentage=max_normal_percentage,
            max_positive_per_label=siglip_max_positive_per_label,
        )

        print(f"\nSaved SigLIP text bank to: {text_bank_path}")
        print(f"Saved SigLIP mapping to: {mapping_path}")
        return

    elif dataset_type == 'combined':
        # For combined mode, use the same input files but output combined datasets
        if train_samples and train_samples > 0:
            if mimic_train_samples is None and mhi_train_samples is None:
                mimic_train_samples = train_samples // 2
                mhi_train_samples = train_samples - mimic_train_samples
            elif mimic_train_samples is None:
                mimic_train_samples = max(train_samples - (mhi_train_samples or 0), 0)
            elif mhi_train_samples is None:
                mhi_train_samples = max(train_samples - (mimic_train_samples or 0), 0)

        if test_samples and test_samples > 0:
            if mimic_test_samples is None and mhi_test_samples is None:
                mimic_test_samples = test_samples // 2
                mhi_test_samples = test_samples - mimic_test_samples
            elif mimic_test_samples is None:
                mimic_test_samples = max(test_samples - (mhi_test_samples or 0), 0)
            elif mhi_test_samples is None:
                mhi_test_samples = max(test_samples - (mimic_test_samples or 0), 0)

        test_input = '/media/data1/datasets/ECG_Tokenizer/parquets/test/mimic_mhi_psa_test_updated.parquet'
        train_input = '/media/data1/datasets/ECG_Tokenizer/parquets/train/mimic_mhi_psa_train_updated.parquet'

        # Calculate total samples for filename
        total_test = (mimic_test_samples or 0) + (mhi_test_samples or 0)
        total_train = (mimic_train_samples or 0) + (mhi_train_samples or 0)
        
        test_output = str(Path(output_dir) / f'preprocessed_combined_test_qa.parquet')
        train_output = str(Path(output_dir) / f'preprocessed_combined_train_qa.parquet')
        
        print("\nNote: Creating combined dataset with samples from both MIMIC and MHI")
        
    elif dataset_type == 'mimic-iv':
        test_input = '/media/data1/datasets/ECG_Tokenizer/parquets/test/mimic_mhi_psa_test_updated.parquet'
        test_output = str(Path(output_dir) / f'mimic_test_qa_{test_samples//1000}k.parquet')

        train_input = '/media/data1/datasets/ECG_Tokenizer/parquets/train/mimic_mhi_psa_train_updated.parquet'
        train_output = str(Path(output_dir) / f'mimic_train_qa_{train_samples//1000}k.parquet')
    
    elif dataset_type == 'mhi':
        # For MHI, use v1.6 parquet with Split column filtering
        mhi_base_path = '/media/data1/muse_ge/ECG_ad20241231_metadata.v1.6._with_translation_ROXs42Bb.cleaned.parquet'
        test_input = mhi_base_path  # Will be filtered by Split='test' in process_dataset
        test_output = str(Path(output_dir) / f'mhi_test_qa_{test_samples//1000}k.parquet')

        train_input = mhi_base_path  # Will be filtered by Split='train' in process_dataset
        train_output = str(Path(output_dir) / f'mhi_train_qa_{train_samples//1000}k.parquet')

        print("\nNote: Using MHI v1.6 parquet with Split column filtering")

    elif dataset_type == 'custom':
        if custom_parquet_path is None:
            raise ValueError("For dataset_type 'custom', please provide --custom_parquet_path")
        test_input = custom_parquet_path
        train_input = custom_parquet_path
        test_output = str(Path(output_dir) / 'custom_test_qa.parquet')
        train_output = str(Path(output_dir) / 'custom_train_qa.parquet')

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    try:
        # Determine sample sizes based on mode
        if dataset_type == 'combined':
            test_sample_size = (mimic_test_samples or 0) + (mhi_test_samples or 0)
            train_sample_size = (mimic_train_samples or 0) + (mhi_train_samples or 0)
        else:
            test_sample_size = test_samples
            train_sample_size = train_samples
            mimic_test_samples = None
            mhi_test_samples = None
            mimic_train_samples = None
            mhi_train_samples = None

        # Custom mode: process once and write a single file
        if dataset_type == 'custom':
            single_output = str(Path(output_dir) / 'preprocessed_qa.parquet')
            df_all = process_dataset(
                train_input,  # same as test_input
                single_output,
                "custom",
                sample_size=None,  # use all rows
                dataset_type=dataset_type,
                max_prompts_per_ecg=max_prompts_per_ecg,
                max_normal_percentage=max_normal_percentage,
                min_samples_per_diagnosis=min_samples_per_diagnosis,
                mimic_samples=None,
                mhi_samples=None,
                prompt_workers=prompt_workers,
                answer_workers=answer_workers,
                preserve_common_rhythms=preserve_common_rhythms,
                custom_parquet_path=custom_parquet_path,
            )

            # Ensure waveform_name exists for summary
            if 'waveform_name' not in df_all.columns:
                for col in ('waveform_path_psa', 'npy_path', 'ecg_path'):
                    if col in df_all.columns:
                        df_all = df_all.copy()
                        df_all['waveform_name'] = df_all[col].astype(str).apply(lambda p: os.path.basename(p))
                        break

            print(f"\n{'='*80}")
            print("FINAL SUMMARY")
            print(f"{'='*80}")
            print(f"Custom dataset: {len(df_all)} prompts from {df_all['waveform_name'].nunique()} ECGs")
            print(f"\nFile saved:")
            print(f"  {single_output}")

            if generate_siglip_alignment:
                print("\nTriggering SigLIP alignment export...")
                output_dir_aln = siglip_output_dir or 'ecg_text_alignment'
                generate_siglip_alignment_dataset(
                    parquet_path=siglip_parquet_path,
                    output_dir=output_dir_aln,
                    include_qa=siglip_include_qa,
                    w_pos=siglip_pos_weight,
                    w_hardneg=siglip_hardneg_weight,
                    w_implneg=siglip_implneg_weight,
                    sample_size=siglip_sample_size,
                    random_seed=siglip_random_seed,
                    implneg_sample_size=siglip_implneg_sample_size,
                    max_hardneg_per_group=siglip_max_hardneg_per_group,
                    train_ecgs=siglip_train_ecgs,
                    val_ecgs=siglip_val_ecgs,
                    test_ecgs=siglip_test_ecgs,
                    max_positive_per_label=siglip_max_positive_per_label,
                )

            return
        
        # Process test dataset first (smaller) - allow all rows for custom
        effective_test_sample_size = test_sample_size
        if dataset_type == 'custom':
            effective_test_sample_size = None  # use all available
        if test_sample_size > 0 or dataset_type == 'custom':
            test_df = process_dataset(
                test_input,
                test_output,
                "test",
                sample_size=effective_test_sample_size,
                dataset_type=dataset_type,
                max_prompts_per_ecg=max_prompts_per_ecg,
                max_normal_percentage=max_normal_percentage,
                min_samples_per_diagnosis=min_samples_per_diagnosis,
                mimic_samples=mimic_test_samples,
                mhi_samples=mhi_test_samples,
                prompt_workers=prompt_workers,
                answer_workers=answer_workers,
                preserve_common_rhythms=preserve_common_rhythms,
                custom_parquet_path=custom_parquet_path,
            )
        else:
            print("Skipping test dataset (test_samples=0)")
        
        # Process train dataset - allow all rows for custom
        effective_train_sample_size = train_sample_size
        if dataset_type == 'custom':
            effective_train_sample_size = None  # use all available
        if train_sample_size > 0 or dataset_type == 'custom':
            train_df = process_dataset(
                train_input,
                train_output,
                "train",
                sample_size=effective_train_sample_size,
                dataset_type=dataset_type,
                max_prompts_per_ecg=max_prompts_per_ecg,
                max_normal_percentage=max_normal_percentage,
                min_samples_per_diagnosis=min_samples_per_diagnosis,
                mimic_samples=mimic_train_samples,
                mhi_samples=mhi_train_samples,
                prompt_workers=prompt_workers,
                answer_workers=answer_workers,
                preserve_common_rhythms=preserve_common_rhythms,
                custom_parquet_path=custom_parquet_path,
            )
        else:
            print("Skipping train dataset (train_samples=0)")
        
        # Final summary
        def _ensure_waveform_name(df: pd.DataFrame) -> pd.DataFrame:
            if 'waveform_name' in df.columns:
                return df
            candidate = None
            for col in ('waveform_path_psa', 'npy_path', 'ecg_path'):
                if col in df.columns:
                    candidate = col
                    break
            if candidate:
                df = df.copy()
                df['waveform_name'] = df[candidate].astype(str).apply(lambda p: os.path.basename(p))
            return df

        if 'test_df' in locals() and test_df is not None:
            test_df = _ensure_waveform_name(test_df)
        if 'train_df' in locals() and train_df is not None:
            train_df = _ensure_waveform_name(train_df)

        print(f"\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}")
        if test_sample_size > 0 or dataset_type == 'custom':
            print(f"Test dataset: {len(test_df)} prompts from {test_df['waveform_name'].nunique()} ECGs")
        if train_sample_size > 0 or dataset_type == 'custom':
            print(f"Train dataset: {len(train_df)} prompts from {train_df['waveform_name'].nunique()} ECGs")
        
        print(f"\nFiles saved:")
        print(f"  Test: {test_output}")
        print(f"  Train: {train_output}")
        
        # Check for ischemia/infarction questions
        print(f"\n✓ Ischemia/infarction detection FIXED:")
        print(f"  • Now checks ST downsloping, ST depression, T wave inversions")
        print(f"  • Properly identifies Q waves as signs of old infarction")
        print(f"  • Correctly parses report text for infarct mentions")

        if generate_siglip_alignment:
            print("\nTriggering SigLIP alignment export...")
            output_dir = siglip_output_dir or 'ecg_text_alignment'
            generate_siglip_alignment_dataset(
                parquet_path=siglip_parquet_path,
                output_dir=output_dir,
                include_qa=siglip_include_qa,
                w_pos=siglip_pos_weight,
                w_hardneg=siglip_hardneg_weight,
                w_implneg=siglip_implneg_weight,
                sample_size_for_checks=siglip_sample_size,
                random_state=siglip_random_seed,
                implicit_negative_sample_size=siglip_implneg_sample_size,
                max_hardneg_per_group=siglip_max_hardneg_per_group,
                train_ecg_limit=siglip_train_ecgs,
                val_ecg_limit=siglip_val_ecgs,
                test_ecg_limit=siglip_test_ecgs,
                max_normal_percentage=max_normal_percentage,
                max_positive_per_label=siglip_max_positive_per_label,
            )

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n🎉 ALL DATASETS GENERATED SUCCESSFULLY!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate train and test datasets with Q&A pairs")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mimic-iv",
        choices=["mimic-iv", "mhi", "combined", "custom"],
        help="Dataset type: 'mimic-iv', 'mhi', 'combined', or 'custom' for a user-provided parquet"
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=400000,
        help="Number of training samples to generate (default: 400,000)"
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=10000,
        help="Number of test/validation samples to generate (default: 10,000)"
    )
    parser.add_argument(
        "--max_prompts_per_ecg",
        type=int,
        default=5,
        help="Maximum prompts per ECG (prioritised selection). Default: 5 prompts per ECG when available."
    )
    parser.add_argument(
        "--max_normal_percentage",
        type=float,
        default=0.05,
        help="Maximum percentage of normal ECGs in the dataset (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--min_samples_per_diagnosis",
        type=int,
        default=2,
        help="Minimum samples to include for each diagnosis category (default: 2)"
    )
    parser.add_argument(
        "--drop_common_rhythms",
        action="store_true",
        help="If set, zero out Sinusal/Regular rhythm tags (legacy behaviour)"
    )
    parser.add_argument(
        "--prompt_workers",
        type=int,
        default=1,
        help="Number of worker processes for prompt generation (default: 1)"
    )
    parser.add_argument(
        "--answer_workers",
        type=int,
        default=1,
        help="Number of worker processes for answer generation (default: 1)"
    )
    parser.add_argument(
        "--mimic_train_samples",
        type=int,
        default=None,
        help="For combined mode: Number of MIMIC training samples"
    )
    parser.add_argument(
        "--mhi_train_samples",
        type=int,
        default=None,
        help="For combined mode: Number of MHI training samples"
    )
    parser.add_argument(
        "--mimic_test_samples",
        type=int,
        default=None,
        help="For combined mode: Number of MIMIC test samples"
    )
    parser.add_argument(
        "--mhi_test_samples",
        type=int,
        default=None,
        help="For combined mode: Number of MHI test samples"
    )
    parser.add_argument(
        "--generate_siglip_alignment",
        action="store_true",
        help="Also generate SigLIP text bank and ECG-text mapping from the MHI parquet"
    )
    parser.add_argument(
        "--siglip_output_dir",
        type=str,
        default="ecg_text_alignment",
        help="Directory to store SigLIP alignment outputs (default: ecg_text_alignment)"
    )
    parser.add_argument(
        "--siglip_parquet_path",
        type=str,
        default="/media/data1/muse_ge/ECG_ad20241231_cat_labels_v1.4.complete.ROXs42Bb.parquet",
        help="Parquet file with wide ECG labels for SigLIP alignment"
    )
    parser.add_argument(
        "--siglip_sample_size",
        type=int,
        default=1000,
        help="Number of ECGs to sample when reporting exclusivity sanity checks (default: 1000)"
    )
    parser.add_argument(
        "--siglip_w_pos",
        type=float,
        default=1.0,
        help="Weight for positive SigLIP pairs (default: 1.0)"
    )
    parser.add_argument(
        "--siglip_w_hardneg",
        type=float,
        default=3.0,
        help="Weight for hard negatives in SigLIP mapping (default: 3.0)"
    )
    parser.add_argument(
        "--siglip_w_implneg",
        type=float,
        default=0.2,
        help="Weight for implicit negatives in SigLIP mapping (default: 0.2)"
    )
    parser.add_argument(
        "--siglip_no_qa",
        action="store_true",
        help="Disable QA-style text variants when building the SigLIP text bank"
    )
    parser.add_argument(
        "--siglip_implneg_sample_size",
        type=int,
        default=64,
        help="Implicit negative sample size for SigLIP collator"
    )
    parser.add_argument(
        "--siglip_max_hardneg_per_group",
        type=int,
        default=3,
        help="Maximum hard negatives to keep per exclusivity group"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/volume/ECG_tokenizer/output",
        help="Base directory to write QA parquet outputs"
    )
    parser.add_argument(
        "--custom_parquet_path",
        type=str,
        default=None,
        help="Path to custom parquet when --dataset custom"
    )
    parser.add_argument(
        "--siglip_train_ecgs",
        type=int,
        default=None,
        help="Optional limit for train split ECGs in SigLIP alignment"
    )
    parser.add_argument(
        "--siglip_val_ecgs",
        type=int,
        default=None,
        help="Optional limit for validation split ECGs in SigLIP alignment"
    )
    parser.add_argument(
        "--siglip_test_ecgs",
        type=int,
        default=None,
        help="Optional limit for test split ECGs in SigLIP alignment"
    )
    parser.add_argument(
        "--siglip_max_positive_per_label",
        type=int,
        default=None,
        help="Optional cap on positive samples per text label"
    )
    parser.add_argument(
        "--siglip_random_seed",
        type=int,
        default=0,
        help="Random seed for SigLIP alignment sampling"
    )

    args = parser.parse_args()
    main(
        dataset_type=args.dataset,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        max_prompts_per_ecg=args.max_prompts_per_ecg,
        max_normal_percentage=args.max_normal_percentage,
        min_samples_per_diagnosis=args.min_samples_per_diagnosis,
        mimic_train_samples=args.mimic_train_samples,
        mhi_train_samples=args.mhi_train_samples,
        mimic_test_samples=args.mimic_test_samples,
        mhi_test_samples=args.mhi_test_samples,
        generate_siglip_alignment=args.generate_siglip_alignment,
        siglip_output_dir=args.siglip_output_dir,
        siglip_parquet_path=args.siglip_parquet_path,
        siglip_include_qa=not args.siglip_no_qa,
        siglip_sample_size=args.siglip_sample_size,
        siglip_pos_weight=args.siglip_w_pos,
        siglip_hardneg_weight=args.siglip_w_hardneg,
        siglip_implneg_weight=args.siglip_w_implneg,
        siglip_random_seed=args.siglip_random_seed,
        siglip_implneg_sample_size=args.siglip_implneg_sample_size,
        siglip_max_hardneg_per_group=args.siglip_max_hardneg_per_group,
        output_dir=args.output_dir,
        prompt_workers=args.prompt_workers,
        answer_workers=args.answer_workers,
        siglip_train_ecgs=args.siglip_train_ecgs,
        siglip_val_ecgs=args.siglip_val_ecgs,
        siglip_test_ecgs=args.siglip_test_ecgs,
        siglip_max_positive_per_label=args.siglip_max_positive_per_label,
        preserve_common_rhythms=not args.drop_common_rhythms,
        custom_parquet_path=args.custom_parquet_path,
    )
