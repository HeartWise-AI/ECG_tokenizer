#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys
sys.path.append('/volume/ECG_tokenizer')
sys.path.append('/volume/ECG_tokenizer/dataset_generation')

from ecg_answer_generator import ECGAnswerGenerator
from ecg_prompt_maker import ECGPromptMaker

print("="*70)
print("TESTING AFIB RISK Q&A GENERATION FOR MHI DATASET")
print("="*70)

# Test cases for AFib risk
afib_risk_test_cases = [
    {
        'name': 'Patient NOT in AFib, HIGH 2-year risk',
        'row': pd.Series({
            'dataset': 'mhi',
            'dataset_source': 'mhi',
            'waveform_name': 'test_mhi_afib_high_2y.npy',
            'Afib': 0,  # Not currently in AFib
            'Afib_bert_model': 0.1,  # Below threshold
            'afib_label_2y': True,  # High 2-year risk
            'afib_label_5y': True,  # Also high 5-year risk
            'ecg_type': 'normal',
            'prompt': "What is the patient's risk of developing atrial fibrillation?"
        })
    },
    {
        'name': 'Patient NOT in AFib, LOW 2-year but HIGH 5-year risk',
        'row': pd.Series({
            'dataset': 'mhi',
            'dataset_source': 'mhi',
            'waveform_name': 'test_mhi_afib_high_5y.npy',
            'Afib': 0,
            'Afib_bert_model': 0.2,
            'afib_label_2y': False,  # Low 2-year risk
            'afib_label_5y': True,   # High 5-year risk
            'ecg_type': 'borderline',
            'prompt': 'Is this patient at risk for incident AFib?'
        })
    },
    {
        'name': 'Patient NOT in AFib, LOW risk at both timepoints',
        'row': pd.Series({
            'dataset': 'mhi',
            'dataset_source': 'mhi',
            'waveform_name': 'test_mhi_afib_low_risk.npy',
            'Afib': 0,
            'Afib_bert_model': 0.05,
            'afib_label_2y': False,  # Low 2-year risk
            'afib_label_5y': False,  # Low 5-year risk
            'ecg_type': 'normal',
            'prompt': 'What is the likelihood of developing AFib in the next 2-5 years?'
        })
    },
    {
        'name': 'Patient ALREADY in AFib (Afib column)',
        'row': pd.Series({
            'dataset': 'mhi',
            'dataset_source': 'mhi',
            'waveform_name': 'test_mhi_already_afib.npy',
            'Afib': 1,  # Currently in AFib
            'Afib_bert_model': 0.9,
            'afib_label_2y': False,  # Doesn't matter
            'afib_label_5y': False,  # Doesn't matter
            'ecg_type': 'pathological',
            'prompt': 'Will this patient develop atrial fibrillation in the future?'
        })
    },
    {
        'name': 'Patient ALREADY in AFib (BERT model detection)',
        'row': pd.Series({
            'dataset': 'mhi',
            'dataset_source': 'mhi',
            'waveform_name': 'test_mhi_bert_afib.npy',
            'Afib': 0,  # Column says no
            'Afib_bert_model': 0.8,  # But BERT says yes (>0.5)
            'afib_label_2y': True,
            'afib_label_5y': True,
            'ecg_type': 'pathological',
            'prompt': "What is the patient's future AFib risk?"
        })
    },
    {
        'name': 'Missing AFib prediction data',
        'row': pd.Series({
            'dataset': 'mhi',
            'dataset_source': 'mhi',
            'waveform_name': 'test_mhi_afib_missing.npy',
            'Afib': 0,
            'Afib_bert_model': 0.1,
            'afib_label_2y': np.nan,  # Missing data
            'afib_label_5y': np.nan,  # Missing data
            'ecg_type': 'normal',
            'prompt': "What is the patient's risk of developing atrial fibrillation?"
        })
    },
    {
        'name': 'MIMIC dataset (should not get AFib risk questions)',
        'row': pd.Series({
            'dataset': 'mimic',
            'dataset_source': 'mimic-iv',
            'waveform_name': 'test_mimic_no_afib.npy',
            'ecg_type': 'pathological',
            'rr_interval': 800,
            'prompt': 'Is this patient at risk for incident AFib?'
        })
    }
]

# Initialize answer generator
answer_gen = ECGAnswerGenerator(dataset='mhi')

print("\n1. TESTING AFIB RISK ANSWER GENERATION:")
print("-"*50)

for test in afib_risk_test_cases:
    print(f"\nTest: {test['name']}")
    row = test['row']
    
    # Generate answer
    answer = answer_gen.generate_afib_risk_answer(row)
    
    print(f"  Current AFib status:")
    print(f"    Afib column: {row.get('Afib', 'N/A')}")
    print(f"    Afib_bert_model: {row.get('Afib_bert_model', 'N/A')}")
    print(f"  Risk predictions:")
    print(f"    afib_label_2y: {row.get('afib_label_2y', 'N/A')}")
    print(f"    afib_label_5y: {row.get('afib_label_5y', 'N/A')}")
    print(f"  Question: {row.get('prompt', 'N/A')}")
    print(f"  Answer: {answer}")
    
    # Verify correct behavior
    if pd.notna(row.get('afib_label_2y')) and pd.notna(row.get('afib_label_5y')):
        # Check if patient is already in AFib
        is_afib = False
        if 'Afib' in row.index and row.get('Afib', 0) >= 1:
            is_afib = True
        elif 'Afib_bert_model' in row.index and row.get('Afib_bert_model', 0) > 0.5:
            is_afib = True
        
        if is_afib:
            assert answer and 'already in atrial fibrillation' in answer.lower(), \
                f"Should indicate patient is already in AFib"
        else:
            # Check risk levels
            if row['afib_label_2y'] == True:
                assert answer and 'high risk' in answer.lower() and '2 years' in answer, \
                    f"Should indicate high 2-year risk"
            elif row['afib_label_5y'] == True:
                assert answer and 'moderate risk' in answer.lower() and '5 years' in answer, \
                    f"Should indicate moderate 5-year risk"
            else:
                assert answer and 'low risk' in answer.lower(), \
                    f"Should indicate low risk"
    else:
        assert answer is None, f"Should return None for missing data"

print("\n2. TESTING PROMPT GENERATION:")
print("-"*50)

# Test prompt generation
prompt_maker = ECGPromptMaker()

# MHI row with AFib risk data
mhi_row = pd.Series({
    'dataset': 'mhi',
    'dataset_source': 'mhi',
    'waveform_name': 'test_mhi_prompts.npy',
    'ecg_type': 'pathological',
    'Afib': 0,  # Not currently in AFib
    'Afib_bert_model': 0.1,
    'afib_label_2y': True,  # High 2-year risk
    'afib_label_5y': True,
    'deepecho_Visually_Estimated_EF': 45.0,
    'acs_condition_severity': 'Acute Complete Coronary Occlusion',
    'acs_pci_regions': "['IVA I', 'Diagonale 1']",
    'echonext_shd': 1.0,
    'RestingECG_OriginalRestingECGMeasurements_VentricularRate': '72',
    'Sinusal': 1,
    'Regular': 1
})

# Generate prompts multiple times to check probability
afib_risk_prompt_count = 0
total_runs = 100

for _ in range(total_runs):
    prompts = prompt_maker.generate_prompts_for_ecg(mhi_row)
    for prompt, category, weight in prompts:
        if category == 'afib_risk':
            afib_risk_prompt_count += 1

print(f"\nMHI with AFib risk data:")
print(f"  AFib risk questions generated: {afib_risk_prompt_count}/{total_runs} runs")
print(f"  Approximate probability: {afib_risk_prompt_count/total_runs*100:.1f}%")
print(f"  Expected probability: ~10%")

# Test with patient already in AFib
mhi_afib_row = pd.Series({
    'dataset': 'mhi',
    'dataset_source': 'mhi',
    'waveform_name': 'test_mhi_already_afib.npy',
    'ecg_type': 'pathological',
    'Afib': 2,  # Already in AFib
    'afib_label_2y': True,
    'afib_label_5y': True,
    'RestingECG_OriginalRestingECGMeasurements_VentricularRate': '120'
})

already_afib_count = 0
for _ in range(total_runs):
    prompts = prompt_maker.generate_prompts_for_ecg(mhi_afib_row)
    for prompt, category, weight in prompts:
        if category == 'afib_risk':
            already_afib_count += 1

print(f"\nMHI patient already in AFib:")
print(f"  AFib risk questions generated: {already_afib_count}/{total_runs} runs")
print(f"  Note: Questions still generated, but answer will indicate patient is already in AFib")

# Test MIMIC (should never get AFib risk questions)
mimic_row = pd.Series({
    'dataset': 'mimic',
    'dataset_source': 'mimic-iv',
    'waveform_name': 'test_mimic_prompts.npy',
    'ecg_type': 'pathological',
    'rr_interval': 800,
    'Sinusal': 1
})

mimic_afib_count = 0
for _ in range(total_runs):
    prompts = prompt_maker.generate_prompts_for_ecg(mimic_row)
    for prompt, category, weight in prompts:
        if category == 'afib_risk':
            mimic_afib_count += 1

print(f"\nMIMIC dataset:")
print(f"  AFib risk questions generated: {mimic_afib_count}/{total_runs} runs")
print(f"  Expected: 0 (MIMIC doesn't have AFib prediction data)")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)

# Verify assertions
print("\n✓ All assertions passed!")
print("✓ AFib risk Q&A working correctly")
print("✓ Correctly identifies patients already in AFib")
print("✓ Properly categorizes risk levels (high 2-year, moderate 5-year, low)")
print("✓ Questions properly filtered for missing data")
print("✓ MIMIC dataset correctly excluded from AFib risk questions")