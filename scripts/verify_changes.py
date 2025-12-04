#!/usr/bin/env python3
import os
import json
from datetime import datetime
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset_generation')))

from ecg_prompt_maker import ECGPromptMaker
from ecg_answer_generator import ECGAnswerGenerator


def main():
    random.seed(42)
    np.random.seed(42)

    base_path = os.path.join('output', 'combined_test_qa_m10k_h10k.parquet')
    if not os.path.exists(base_path):
        print(f"Base dataset not found: {base_path}")
        return 1

    df_full = pd.read_parquet(base_path)
    # Derive base ECG-level dataframe
    key_col = 'waveform_path_psa' if 'waveform_path_psa' in df_full.columns else None
    if key_col is None:
        print("waveform_path_psa column missing; cannot deduplicate ECGs")
        return 1
    base_df = df_full.drop_duplicates(subset=[key_col]).reset_index(drop=True)

    # Sample up to 10k ECGs
    n = min(10000, len(base_df))
    base_sample = base_df.sample(n=n, random_state=42).reset_index(drop=True)

    # Generate prompts (allow up to 4 per ECG to increase category coverage)
    pm = ECGPromptMaker()
    df_prompts = pm.process_dataframe(base_sample, max_prompts_per_ecg=4)

    # Add targeted prompts to ensure coverage of changed phrases
    targeted_rows = []
    # Reuse a small subset for targeted prompts
    tgt = base_sample.head(min(500, len(base_sample))).copy()
    # JSON explicit schema
    json_phrase = "Output JSON ONLY with keys: RHYTHM, CONDUCTION, CHAMBER_ENLARGEMENT, INFARCT_ISCHEMIA, PERICARDITIS, heart_rate_bpm, ecg_classification. Values must be lists of present findings (omit missing categories)."
    j = tgt.copy()
    j['prompt'] = json_phrase
    j['prompt_category'] = 'json_interpretation'
    j['prompt_weight'] = 0.9
    targeted_rows.append(j)
    # Classification binary phrasing
    c = tgt.copy()
    c['prompt'] = 'Would you classify this ECG as normal or abnormal?'
    c['prompt_category'] = 'classification'
    c['prompt_weight'] = 0.9
    targeted_rows.append(c)
    # ACS new prompt variant
    a = tgt.copy()
    a['prompt'] = 'IS there an acute coornary occlusion/ if yes is it ocmplete or incompelte and what is the culprit?'
    a['prompt_category'] = 'acs_severity'
    a['prompt_weight'] = 1.0
    targeted_rows.append(a)
    # Atrial abnormality
    at = tgt.copy()
    at['prompt'] = 'Is there evidence of atrial abnormality?'
    at['prompt_category'] = 'category_chamber_enlargement'
    at['prompt_weight'] = 0.8
    targeted_rows.append(at)
    if targeted_rows:
        df_prompts = pd.concat([df_prompts] + targeted_rows, ignore_index=True)

    # Generate answers
    ag = ECGAnswerGenerator(language='en', dataset='mhi')
    df_prompts['generated_answer'] = df_prompts.apply(ag.generate_answer, axis=1)

    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = os.path.join('output', f'verify_changes_{ts}')
    os.makedirs(out_dir, exist_ok=True)
    out_parquet = os.path.join(out_dir, 'prompts_answers.parquet')
    df_prompts.to_parquet(out_parquet, index=False)

    # Verification checks
    results = {}

    # 1) JSON prompt schema presence + answer keys validity
    json_df = df_prompts[df_prompts['prompt_category'] == 'json_interpretation'].copy()
    schema_phrase = 'Output JSON ONLY with keys:'
    results['json_prompt_total'] = int(len(json_df))
    results['json_prompt_explicit_schema'] = int(json_df['prompt'].str.contains(schema_phrase, regex=False).sum())

    expected_json_keys = {'RHYTHM','CONDUCTION','CHAMBER_ENLARGEMENT','INFARCT_ISCHEMIA','PERICARDITIS','heart_rate_bpm','ecg_classification'}
    valid_json = 0
    parsed_keys_ok = 0
    for ans in json_df['generated_answer']:
        try:
            obj = json.loads(ans)
            valid_json += 1
            # Keys must be subset of expected_json_keys
            keys = set(obj.keys())
            if keys.issubset(expected_json_keys):
                parsed_keys_ok += 1
        except Exception:
            pass
    results['json_answer_valid'] = valid_json
    results['json_answer_keys_ok'] = parsed_keys_ok

    # 2) Classification phrasing for binary prompt
    is_binary = df_prompts['prompt_category'].eq('classification') & df_prompts['prompt'].str.contains('normal or abnormal', case=False)
    cls_df = df_prompts[is_binary]
    results['classification_binary_total'] = int(len(cls_df))
    results['classification_binary_normal_fmt'] = int(cls_df['generated_answer'].str.startswith('Normal -').sum())
    results['classification_binary_abnormal_fmt'] = int(cls_df['generated_answer'].str.startswith('Abnormal -').sum())
    results['classification_binary_legacy_yes'] = int(cls_df['generated_answer'].str.startswith('Yes -').sum())

    # 3) ACS severity prompt variant + culprit in answer
    acs_df = df_prompts[df_prompts['prompt_category'] == 'acs_severity'].copy()
    new_acs_phrase = 'IS there an acute coornary occlusion/'
    results['acs_total'] = int(len(acs_df))
    results['acs_prompt_new_variant'] = int(acs_df['prompt'].str.contains(new_acs_phrase, regex=False).sum())

    # Acute cases with regions
    acute_mask = acs_df['acs_condition_severity'].isin(['Acute Complete Coronary Occlusion','Acute Incomplete Coronary Occlusion']) if 'acs_condition_severity' in acs_df.columns else pd.Series([False]*len(acs_df))
    with_regions = acs_df['acs_pci_regions'].notna() if 'acs_pci_regions' in acs_df.columns else pd.Series([False]*len(acs_df))
    acs_check_df = acs_df[acute_mask & with_regions]
    results['acs_acute_with_regions'] = int(len(acs_check_df))
    results['acs_answer_includes_culprit'] = int(acs_check_df['generated_answer'].str.contains('culprit is the', case=False).sum())
    results['acs_answer_mentions_occlusion'] = int(acs_check_df['generated_answer'].str.contains('occlusion', case=False).sum())

    # 4) Atrial-only logic for atrial questions in CHAMBER ENLARGEMENT
    atrial_q = (df_prompts['prompt_category'] == 'category_chamber_enlargement') & df_prompts['prompt'].str.contains('atrial', case=False)
    atrial_df = df_prompts[atrial_q].copy()
    lae = (atrial_df.get('Left atrial enlargement', 0) >= 1) if 'Left atrial enlargement' in atrial_df.columns else pd.Series([False]*len(atrial_df))
    rae = (atrial_df.get('Right atrial enlargement', 0) >= 1) if 'Right atrial enlargement' in atrial_df.columns else pd.Series([False]*len(atrial_df))
    bae = (atrial_df.get('Bi-atrial enlargement', 0) >= 1) if 'Bi-atrial enlargement' in atrial_df.columns else pd.Series([False]*len(atrial_df))
    any_atrial = (lae | rae | bae)
    lvh = (atrial_df.get('Left ventricular hypertrophy', 0) >= 1) if 'Left ventricular hypertrophy' in atrial_df.columns else pd.Series([False]*len(atrial_df))

    # Expectations
    # - If any atrial finding: answer should start with Yes - and contain only atrial terms
    # - If no atrial finding (even if LVH): answer should be No - no atrial ...
    atrial_yes = atrial_df[any_atrial]
    atrial_no = atrial_df[~any_atrial]
    results['atrial_q_total'] = int(len(atrial_df))
    results['atrial_with_atrial_finding'] = int(len(atrial_yes))
    results['atrial_without_atrial_finding'] = int(len(atrial_no))
    results['atrial_yes_correct'] = int(atrial_yes['generated_answer'].str.startswith('Yes -').sum())
    results['atrial_no_correct'] = int(atrial_no['generated_answer'].str.startswith('No - no atrial').sum())

    # Plots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # JSON prompts plot
    jp_total = results['json_prompt_total']
    axs[0,0].bar(['explicit_schema','other'], [results['json_prompt_explicit_schema'], jp_total - results['json_prompt_explicit_schema']])
    axs[0,0].set_title('JSON prompt variants')

    # Classification phrasing plot
    axs[0,1].bar(['Normal -','Abnormal -','Legacy Yes -'], [
        results['classification_binary_normal_fmt'],
        results['classification_binary_abnormal_fmt'],
        results['classification_binary_legacy_yes'],
    ])
    axs[0,1].set_title('Classification phrasing (binary)')

    # ACS checks
    axs[1,0].bar(['new_variant','others'], [
        results['acs_prompt_new_variant'],
        results['acs_total'] - results['acs_prompt_new_variant']
    ])
    axs[1,0].set_title('ACS prompt variant usage')

    # Atrial checks
    axs[1,1].bar(['Yes-correct','No-correct'], [
        results['atrial_yes_correct'],
        results['atrial_no_correct']
    ])
    axs[1,1].set_title('Atrial question correctness')

    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'verification_plots.png')
    plt.savefig(plot_path, dpi=150)

    # Save results metrics
    with open(os.path.join(out_dir, 'verification_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Print concise summary
    print('Verification summary:')
    for k,v in results.items():
        print(f"  {k}: {v}")
    print(f"Saved outputs to: {out_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
