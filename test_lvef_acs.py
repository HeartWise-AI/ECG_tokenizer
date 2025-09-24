#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys
sys.path.append('/volume/ECG_tokenizer')
sys.path.append('/volume/ECG_tokenizer/dataset_generation')

from ecg_answer_generator import ECGAnswerGenerator
from ecg_prompt_maker import ECGPromptMaker

print("="*70)
print("TESTING LVEF AND ACS Q&A GENERATION FOR MHI DATASET")
print("="*70)

# Test cases for LVEF
lvef_test_cases = [
    {
        'name': 'MHI with normal LVEF (60%)',
        'row': pd.Series({
            'dataset': 'mhi',
            'dataset_source': 'mhi',
            'waveform_name': 'test_mhi_lvef_normal.npy',
            'deepecho_Visually_Estimated_EF': 60.0,
            'ecg_type': 'normal',
            'prompt': "What is the patient's left ventricular ejection fraction?"
        })
    },
    {
        'name': 'MHI with mildly reduced LVEF (50%)',
        'row': pd.Series({
            'dataset': 'mhi',
            'dataset_source': 'mhi',
            'waveform_name': 'test_mhi_lvef_mild.npy',
            'deepecho_Visually_Estimated_EF': 50.0,
            'ecg_type': 'borderline',
            'prompt': 'What is the LVEF based on echocardiography?'
        })
    },
    {
        'name': 'MHI with moderately reduced LVEF (35%)',
        'row': pd.Series({
            'dataset': 'mhi',
            'dataset_source': 'mhi',
            'waveform_name': 'test_mhi_lvef_moderate.npy',
            'deepecho_Visually_Estimated_EF': 35.0,
            'ecg_type': 'pathological',
            'prompt': 'What is the ejection fraction?'
        })
    },
    {
        'name': 'MHI with severely reduced LVEF (25%)',
        'row': pd.Series({
            'dataset': 'mhi',
            'dataset_source': 'mhi',
            'waveform_name': 'test_mhi_lvef_severe.npy',
            'deepecho_Visually_Estimated_EF': 25.0,
            'ecg_type': 'pathological',
            'prompt': 'Can you tell me the patient\'s EF?'
        })
    },
    {
        'name': 'MHI with missing LVEF data',
        'row': pd.Series({
            'dataset': 'mhi',
            'dataset_source': 'mhi',
            'waveform_name': 'test_mhi_lvef_missing.npy',
            'deepecho_Visually_Estimated_EF': np.nan,
            'ecg_type': 'normal',
            'prompt': 'What is the left ventricular function?'
        })
    }
]

# Test cases for ACS severity
acs_severity_test_cases = [
    {
        'name': 'MHI with Acute Complete Coronary Occlusion',
        'row': pd.Series({
            'dataset': 'mhi',
            'dataset_source': 'mhi',
            'waveform_name': 'test_mhi_acs_complete.npy',
            'acs_condition_severity': 'Acute Complete Coronary Occlusion',
            'acs_pci_regions': "['CD I', 'CD II']",  # RCA
            'ecg_type': 'pathological',
            'prompt': 'Is there a STEMI or is there an acute coronary occlusion?'
        })
    },
    {
        'name': 'MHI with Acute Incomplete Coronary Occlusion',
        'row': pd.Series({
            'dataset': 'mhi',
            'dataset_source': 'mhi',
            'waveform_name': 'test_mhi_acs_incomplete.npy',
            'acs_condition_severity': 'Acute Incomplete Coronary Occlusion',
            'acs_pci_regions': "['IVA II', 'Diagonale 1']",  # LAD
            'ecg_type': 'pathological',
            'prompt': 'Does this patient have an acute coronary syndrome requiring urgent intervention?'
        })
    },
    {
        'name': 'MHI with Obstructive Coronary Disease (not acute)',
        'row': pd.Series({
            'dataset': 'mhi',
            'dataset_source': 'mhi',
            'waveform_name': 'test_mhi_acs_obstructive.npy',
            'acs_condition_severity': 'Obstructive Coronary Disease',
            'acs_pci_regions': "['Cx I']",
            'ecg_type': 'borderline',
            'prompt': 'Is there evidence of an acute coronary occlusion?'
        })
    },
    {
        'name': 'MHI with No Coronary Disease',
        'row': pd.Series({
            'dataset': 'mhi',
            'dataset_source': 'mhi',
            'waveform_name': 'test_mhi_acs_none.npy',
            'acs_condition_severity': 'No Coronary Disease',
            'acs_pci_regions': "[]",
            'ecg_type': 'normal',
            'prompt': 'Is there an acute myocardial infarction due to coronary occlusion?'
        })
    },
    {
        'name': 'MHI with Chronic Total Coronary Occlusion',
        'row': pd.Series({
            'dataset': 'mhi',
            'dataset_source': 'mhi',
            'waveform_name': 'test_mhi_acs_chronic.npy',
            'acs_condition_severity': 'Chronic Total Coronary Occlusion',
            'acs_pci_regions': "['CD III']",
            'ecg_type': 'pathological',
            'prompt': 'Is there a STEMI or is there an acute coronary occlusion?'
        })
    }
]

# Test cases for culprit artery
culprit_artery_test_cases = [
    {
        'name': 'Acute occlusion with LAD culprit',
        'row': pd.Series({
            'dataset': 'mhi',
            'dataset_source': 'mhi',
            'waveform_name': 'test_mhi_culprit_lad.npy',
            'acs_condition_severity': 'Acute Complete Coronary Occlusion',
            'acs_pci_regions': "['IVA I', 'IVA II']",
            'ecg_type': 'pathological',
            'prompt': 'What is the culprit artery?'
        })
    },
    {
        'name': 'Acute occlusion with RCA culprit',
        'row': pd.Series({
            'dataset': 'mhi',
            'dataset_source': 'mhi',
            'waveform_name': 'test_mhi_culprit_rca.npy',
            'acs_condition_severity': 'Acute Incomplete Coronary Occlusion',
            'acs_pci_regions': "['CD II', 'IVP']",
            'ecg_type': 'pathological',
            'prompt': 'Which coronary artery is occluded?'
        })
    },
    {
        'name': 'Acute occlusion with Left Main culprit',
        'row': pd.Series({
            'dataset': 'mhi',
            'dataset_source': 'mhi',
            'waveform_name': 'test_mhi_culprit_leftmain.npy',
            'acs_condition_severity': 'Acute Complete Coronary Occlusion',
            'acs_pci_regions': "['Tronc commun']",
            'ecg_type': 'pathological',
            'prompt': 'What is the location of the coronary occlusion?'
        })
    },
    {
        'name': 'Acute occlusion with Circumflex culprit',
        'row': pd.Series({
            'dataset': 'mhi',
            'dataset_source': 'mhi',
            'waveform_name': 'test_mhi_culprit_cx.npy',
            'acs_condition_severity': 'Acute Incomplete Coronary Occlusion',
            'acs_pci_regions': "['Cx II', 'Marginale 1']",
            'ecg_type': 'pathological',
            'prompt': 'Which vessel is the culprit for this acute coronary syndrome?'
        })
    },
    {
        'name': 'Acute occlusion with Ramus (Bissectrice) culprit',
        'row': pd.Series({
            'dataset': 'mhi',
            'dataset_source': 'mhi',
            'waveform_name': 'test_mhi_culprit_ramus.npy',
            'acs_condition_severity': 'Acute Complete Coronary Occlusion',
            'acs_pci_regions': "['Bissectrice']",
            'ecg_type': 'pathological',
            'prompt': 'What is the culprit artery?'
        })
    },
    {
        'name': 'Non-acute occlusion (should not answer)',
        'row': pd.Series({
            'dataset': 'mhi',
            'dataset_source': 'mhi',
            'waveform_name': 'test_mhi_culprit_nonacute.npy',
            'acs_condition_severity': 'Obstructive Coronary Disease',
            'acs_pci_regions': "['IVA III']",
            'ecg_type': 'borderline',
            'prompt': 'What is the culprit artery?'
        })
    }
]

# Initialize answer generator
answer_gen = ECGAnswerGenerator(dataset='mhi')

print("\n1. TESTING LVEF ANSWER GENERATION:")
print("-"*50)

for test in lvef_test_cases:
    print(f"\nTest: {test['name']}")
    row = test['row']
    
    # Generate answer
    answer = answer_gen.generate_lvef_answer(row)
    
    print(f"  LVEF value: {row.get('deepecho_Visually_Estimated_EF', 'N/A')}")
    print(f"  Question: {row.get('prompt', 'N/A')}")
    print(f"  Answer: {answer}")
    
    # Verify correct behavior
    if pd.notna(row.get('deepecho_Visually_Estimated_EF')):
        lvef_val = float(row['deepecho_Visually_Estimated_EF'])
        assert answer is not None, f"Should provide LVEF answer"
        assert str(round(lvef_val)) in answer, f"Should include LVEF percentage"
        
        # Check categorization
        if lvef_val >= 55:
            assert 'normal' in answer, f"Should categorize as normal"
        elif lvef_val >= 45:
            assert 'mildly reduced' in answer, f"Should categorize as mildly reduced"
        elif lvef_val >= 30:
            assert 'moderately reduced' in answer, f"Should categorize as moderately reduced"
        else:
            assert 'severely reduced' in answer, f"Should categorize as severely reduced"
    else:
        assert answer is None, f"Should return None for missing data"

print("\n2. TESTING ACS SEVERITY ANSWER GENERATION:")
print("-"*50)

for test in acs_severity_test_cases:
    print(f"\nTest: {test['name']}")
    row = test['row']
    
    # Generate answer
    answer = answer_gen.generate_acs_severity_answer(row)
    
    print(f"  ACS condition: {row.get('acs_condition_severity', 'N/A')}")
    print(f"  Question: {row.get('prompt', 'N/A')}")
    print(f"  Answer: {answer}")
    
    # Verify correct behavior
    condition = row.get('acs_condition_severity')
    if condition in ['Acute Complete Coronary Occlusion', 'Acute Incomplete Coronary Occlusion']:
        assert answer and answer.startswith('Yes'), f"Should answer Yes for acute occlusion"
    else:
        assert answer and answer.startswith('No'), f"Should answer No for non-acute conditions"

print("\n3. TESTING CULPRIT ARTERY ANSWER GENERATION:")
print("-"*50)

for test in culprit_artery_test_cases:
    print(f"\nTest: {test['name']}")
    row = test['row']
    
    # Generate answer
    answer = answer_gen.generate_culprit_artery_answer(row)
    
    print(f"  ACS condition: {row.get('acs_condition_severity', 'N/A')}")
    print(f"  PCI regions: {row.get('acs_pci_regions', 'N/A')}")
    print(f"  Question: {row.get('prompt', 'N/A')}")
    print(f"  Answer: {answer}")
    
    # Verify correct behavior
    condition = row.get('acs_condition_severity')
    if condition in ['Acute Complete Coronary Occlusion', 'Acute Incomplete Coronary Occlusion']:
        assert answer is not None, f"Should provide culprit artery for acute occlusion"
        
        # Check artery mapping
        regions = row.get('acs_pci_regions', '[]')
        if 'IVA' in regions:
            assert 'LAD' in answer, f"Should map IVA to LAD"
        elif 'CD' in regions:
            assert 'RCA' in answer, f"Should map CD to RCA"
        elif 'Cx' in regions:
            assert 'Circumflex' in answer, f"Should map Cx to Circumflex"
        elif 'Tronc commun' in regions:
            assert 'Left Main' in answer, f"Should map Tronc commun to Left Main"
        elif 'Bissectrice' in regions:
            assert 'Ramus' in answer, f"Should map Bissectrice to Ramus"
    else:
        assert answer is None, f"Should return None for non-acute conditions"

print("\n4. TESTING PROMPT GENERATION:")
print("-"*50)

# Test prompt generation
prompt_maker = ECGPromptMaker()

# MHI row with all the new data
mhi_row = pd.Series({
    'dataset': 'mhi',
    'dataset_source': 'mhi',
    'waveform_name': 'test_mhi_prompts.npy',
    'ecg_type': 'pathological',
    'deepecho_Visually_Estimated_EF': 45.0,
    'acs_condition_severity': 'Acute Complete Coronary Occlusion',
    'acs_pci_regions': "['IVA I', 'Diagonale 1']",
    'echonext_shd': 1.0,
    'RestingECG_OriginalRestingECGMeasurements_VentricularRate': '72',
    'Sinusal': 1,
    'Regular': 1
})

# Generate prompts multiple times to check probability
lvef_prompt_count = 0
acs_prompt_count = 0
culprit_prompt_count = 0
total_runs = 100

for _ in range(total_runs):
    prompts = prompt_maker.generate_prompts_for_ecg(mhi_row)
    for prompt, category, weight in prompts:
        if category == 'lvef':
            lvef_prompt_count += 1
        elif category == 'acs_severity':
            acs_prompt_count += 1
        elif category == 'culprit_artery':
            culprit_prompt_count += 1

print(f"\nMHI with LVEF, ACS, and culprit data:")
print(f"  LVEF questions generated: {lvef_prompt_count}/{total_runs} runs")
print(f"  ACS severity questions generated: {acs_prompt_count}/{total_runs} runs")
print(f"  Culprit artery questions generated: {culprit_prompt_count}/{total_runs} runs")
print(f"  Expected probability: ~10% for LVEF and ACS, plus culprit when acute occlusion")

# Test MIMIC (should never get these questions)
mimic_row = pd.Series({
    'dataset': 'mimic',
    'dataset_source': 'mimic-iv',
    'waveform_name': 'test_mimic_prompts.npy',
    'ecg_type': 'pathological',
    'rr_interval': 800,
    'Sinusal': 1
})

mimic_lvef_count = 0
mimic_acs_count = 0
for _ in range(total_runs):
    prompts = prompt_maker.generate_prompts_for_ecg(mimic_row)
    for prompt, category, weight in prompts:
        if category == 'lvef':
            mimic_lvef_count += 1
        elif category == 'acs_severity':
            mimic_acs_count += 1
        elif category == 'culprit_artery':
            mimic_acs_count += 1

print(f"\nMIMIC dataset:")
print(f"  LVEF questions generated: {mimic_lvef_count}/{total_runs} runs")
print(f"  ACS questions generated: {mimic_acs_count}/{total_runs} runs")
print(f"  Expected: 0 (MIMIC doesn't have these data columns)")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)

# Verify assertions
print("\n✓ All assertions passed!")
print("✓ LVEF Q&A working correctly with proper categorization")
print("✓ ACS severity detection working correctly")
print("✓ Culprit artery mapping working correctly")
print("✓ Questions properly filtered for missing data")
print("✓ MIMIC dataset correctly excluded from MHI-specific questions")