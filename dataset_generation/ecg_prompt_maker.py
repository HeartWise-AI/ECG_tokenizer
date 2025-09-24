#!/usr/bin/env python3
"""
ECG Prompt Maker - Generates multiple prompts per ECG based on findings.
Creates 1-N prompts depending on ECG characteristics and categories present.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple
from collections import defaultdict
from dataset_column_mappings import DatasetColumnMapper
from utils.constants import DEEPECG_CATEGORIES


class ECGPromptMaker:
    """Generate diverse prompts for ECG interpretation tasks"""
    
    def __init__(self, dataset: str = 'mimic'):
        """Initialize with category definitions"""
        
        # Load category definitions from constants
        self.categories_dict = DEEPECG_CATEGORIES
        
        # Initialize column mapper for dataset-specific columns
        self.dataset = dataset
        self.column_mapper = DatasetColumnMapper(dataset)
        
        # Define prompt templates for each type
        self.interpretation_prompts = [
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
            "Can you analyze this ECG tracing?"
        ]
        
        self.category_specific_prompts = {
            "RHYTHM": [
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
            "CONDUCTION": [
                "Are there any conduction abnormalities?",
                "Is there any heart block present?",
                "Are there any bundle branch blocks?",
                "Is AV conduction normal?",
                "Can you assess the conduction system?",
                "Is there any conduction delay?",
                "What type of block is present if any?",
                "Is there evidence of pre-excitation?",
                "Are there any fascicular blocks?",
                "Is there AV dissociation?"
            ],
            "INFARCT, ISCHEMIA": [
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
            "CHAMBER ENLARGEMENT": [
                "Is there chamber enlargement or hypertrophy?",
                "Are there signs of LVH or RVH?",
                "Is there atrial enlargement?",
                "Are the QRS voltages normal?",
                "Does this show ventricular hypertrophy?",
                "Can you assess for chamber abnormalities?",
                "Is there evidence of atrial abnormality?",
                "What do the voltages suggest?",
                "Are there signs of ventricular strain?",
                "Is there biatrial or biventricular enlargement?"
            ],
            "PERICARDITIS": [
                "Is there evidence of pericarditis?",
                "Are there signs of pericardial inflammation?",
                "Does this ECG suggest pericarditis?",
                "Is there diffuse ST elevation?",
                "Are there pericarditic changes?"
            ],
            "OTHER": [
                "Are there any nonspecific ST-T changes?",
                "Is there early repolarization?",
                "What other findings are present?",
                "Are the T waves normal in morphology?",
                "Is there anything else abnormal?"
            ]
        }
        
        self.classification_prompts = [
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
            "What is the overall ECG assessment?"
        ]
        
        # New localization prompts
        self.localization_prompts = {
            "Q_WAVE": [
                "Are there any Q waves present?",
                "Is there evidence of Q waves?",
                "Where are Q waves located?",
                "Can you identify pathological Q waves?",
                "Are there Q waves suggesting old infarction?"
            ],
            "ST_ELEVATION": [
                "Is there ST elevation?",
                "Where is ST elevation present?",
                "Are there ST elevations in any leads?",
                "Can you identify ST segment elevation?",
                "Is there evidence of acute ST elevation?"
            ],
            "ST_DEPRESSION": [
                "Is there ST depression?",
                "Where is ST depression located?",
                "Are there ST depressions in any leads?",
                "Can you identify ST segment depression?",
                "Is there evidence of ST depression?"
            ],
            "T_WAVE": [
                "Are there T wave abnormalities?",
                "Where are T wave inversions?",
                "Is there T wave inversion present?",
                "Can you identify T wave changes?",
                "Are the T waves abnormal in any leads?"
            ],
            "QRS_AXIS": [
                "What is the QRS axis?",
                "Is there axis deviation?",
                "What is the electrical axis of the heart?",
                "Is the QRS axis normal or deviated?",
                "What type of axis deviation is present?",
                "Is there left or right axis deviation?",
                "What is the frontal plane QRS axis?",
            ]
        }
        
        # JSON interpretation prompts
        self.json_prompts = [
            "Output as JSON only following the schema for ECG findings",
            "Provide structured JSON output for this ECG analysis",
            "Return ECG findings as JSON with binary indicators",
            "Generate JSON representation of ECG abnormalities",
            "Output ECG interpretation in JSON format only"
        ]
        
        # ECG interval prompts (heart rate and intervals combined)
        self.ecg_interval_prompts = [
            # Heart rate questions
            "What is the heart rate?",
            "What is the patient's heart rate?",
            "What is the ventricular rate?",
            "Can you tell me the heart rate in bpm?",
            "What is the HR?",
            "How fast is the heart beating?",
            "What's the pulse rate?",
            # PR interval questions
            "What is the PR interval?",
            # QT interval questions
            "What is the QT interval?",
            "What is the corrected QT interval (QTc)?",
            "Is the QT interval prolonged?",
            "What are the PR and QT intervals?",
            "Are the intervals normal or abnormal?",
            "What is the QTc using Bazett's formula?",
            "What is the QTc using Fridericia's formula?",
            "Is there QT prolongation present?"
        ]
        
        # Structural heart disease prompts (MHI only - using echonext_shd)
        self.structural_heart_disease_prompts = [
            "Does this patient have structural heart disease?",
            "Is there evidence of structural heart disease in this patient?",
            "Based on echocardiography data, does the patient have structural heart disease?"
        ]
        
        # LVEF prompts (MHI only - using deepecho_Visually_Estimated_EF)
        self.lvef_prompts = [
            "What is the patient's left ventricular ejection fraction?",
            "What is the LVEF based on echocardiography?",
            "What is the ejection fraction?",
            "Can you tell me the patient's EF?",
            "What is the left ventricular function?"
        ]
        
        # ACS severity prompts (MHI only - using acs_condition_severity)
        self.acs_severity_prompts = [
            "Is there an acute coronary occlusion, and if so is it complete or incomplete?",
            "Does this patient have an acute complete or incomplete coronary artery occlusion?",
            "Is there evidence of an acute coronary occlusion (complete vs incomplete)?",
            "Is this an acute coronary occlusion? Specify whether it is complete or incomplete."
        ]
        
        # Culprit artery prompts (MHI only - follow-up when acute occlusion present)
        self.culprit_artery_prompts = [
            "What is the culprit artery?",
            "Which coronary artery is occluded?",
            "What is the location of the coronary occlusion?",
            "Which vessel is the culprit for this acute coronary syndrome?"
        ]
        
        # AFib risk prompts (MHI only - using afib_label_2y and afib_label_5y)
        self.afib_risk_prompts = [
            "What is the patient's risk of developing atrial fibrillation?",
            "Is this patient at risk for incident AFib?",
            "What is the likelihood of developing AFib in the next 2-5 years?",
            "Will this patient develop atrial fibrillation in the future?",
            "What is the patient's future AFib risk?"
        ]
        
        # Random YES/NO ECG finding questions (seed 42 for reproducibility)
        self.random_finding_prompts = [
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
            "Is there short QT syndrome?"
        ]
        
        # Map questions to column names - EXACT matches from deepecg_categories.json
        self.finding_to_column_map = {
            "1st degree AV block": "1st degree AV block",
            "2nd degree AV block": ["2nd degree AV block - mobitz 1", "2nd degree AV block - mobitz 2"],
            "3rd degree AV block": "Third Degree AV Block",
            "left bundle branch block": "Left bundle branch block",
            "right bundle branch block": "Right bundle branch block",
            "left anterior fascicular block": "Left anterior fascicular block",
            "left posterior fascicular block": "Left posterior fascicular block",
            "left ventricular hypertrophy": "Left ventricular hypertrophy",
            "right ventricular hypertrophy": "Right ventricular hypertrophy",
            "left atrial enlargement": "Left atrial enlargement",
            "right atrial enlargement": "Right atrial enlargement",
            "atrial fibrillation": "Afib",
            "atrial flutter": "Atrial flutter",
            "sinus bradycardia": "Bradycardia",
            "sinus tachycardia": "Atrial tachycardia (>= 100 BPM)",
            "ventricular tachycardia": "Ventricular tachycardia",
            "premature ventricular complex": "Premature ventricular complex",
            "premature atrial complex": "Premature atrial complex",
            "extreme axis deviation": "Right superior axis",  # From CONDUCTION category
            "pericarditis": "Acute pericarditis",
            "early repolarization": "Early repolarization",
            "left axis deviation": "Left axis deviation",
            "right axis deviation": "Right axis deviation",
            "wpw pattern": "Wolff-Parkinson-White (Pre-excitation syndrome)",
            "long qt syndrome": "Prolonged QT",
            "delta wave": "Delta wave"
        }
        
        # Prompt weights for sampling
        self.prompt_weights = {
            'interpretation': 0.35,  # 35% interpretation prompts
            'category': 0.26,        # 26% category-specific (reduced from 30%)
            'classification': 0.20,  # 20% classification
            'demographic': 0.15,     # 15% demographic/HR questions
            'random_finding': 0.04   # 4% random YES/NO finding questions
        }
    
    def determine_active_categories(self, row: pd.Series) -> Dict[str, List[str]]:
        """
        Determine which categories have positive findings.
        Returns dict with category -> list of active conditions
        """
        active = defaultdict(list)
        
        for category, conditions in self.categories_dict.items():
            for condition in conditions:
                # Special handling for Acute MI
                if condition == 'Acute MI':
                    # Check the Acute_MI column
                    if 'Acute_MI' in row.index:
                        try:
                            if pd.notna(row['Acute_MI']) and float(row['Acute_MI']) >= 1:
                                active[category].append(condition)
                                continue
                        except (ValueError, TypeError):
                            pass
                    
                    # Also check if report contains acute MI references
                    if 'report' in row.index and pd.notna(row['report']):
                        report_text = str(row['report']).lower()
                        if 'acute mi' in report_text or 'acute myocardial infarction' in report_text:
                            active[category].append(condition)
                            continue
                
                # Check various column name formats for other conditions
                col_names = [
                    condition,
                    condition.replace(' ', '_'),
                    condition.lower().replace(' ', '_')
                ]
                
                # Also check for _bert_model columns (MHI dataset)
                bert_col = f"{condition}_bert_model"
                if bert_col in row.index:
                    col_names.append(bert_col)
                
                for col in col_names:
                    if col in row.index:
                        try:
                            value = row[col]
                            if pd.notna(value):
                                # For _bert_model columns (MHI), use 0.5 threshold
                                if col.endswith('_bert_model'):
                                    if float(value) > 0.5:
                                        active[category].append(condition)
                                        break
                                # For regular columns, use >= 1 threshold
                                else:
                                    if float(value) >= 1:
                                        active[category].append(condition)
                                        break
                        except (ValueError, TypeError):
                            continue
        
        return dict(active)
    
    def check_localization_findings(self, row: pd.Series) -> Dict[str, List[str]]:
        """
        Check for localization findings (Q waves, ST changes, T waves).
        Returns dict with finding type -> list of locations.
        """
        localization_findings = {
            'Q_WAVE': [],
            'ST_ELEVATION': [],
            'ST_DEPRESSION': [],
            'T_WAVE': [],
            'QRS_AXIS': []
        }
        
        # Check each column for localization info
        for col in row.index:
            try:
                value = row[col]
                # Handle _bert_model columns (MHI) with 0.5 threshold
                if col.endswith('_bert_model'):
                    col_str = col.replace('_bert_model', '')  # Remove suffix for pattern matching
                    if pd.notna(value) and float(value) > 0.5:
                        pass  # Will process below
                    else:
                        continue
                # Regular columns with >= 1 threshold
                elif pd.notna(value) and float(value) >= 1:
                    col_str = str(col)
                else:
                    continue
                
                # Extract location from parentheses
                if '(' in col_str and ')' in col_str:
                    location = col_str[col_str.find('(')+1:col_str.find(')')]
                    
                    if 'Q wave' in col_str:
                        localization_findings['Q_WAVE'].append(location)
                    elif 'ST elevation' in col_str:
                        localization_findings['ST_ELEVATION'].append(location)
                    elif 'ST depression' in col_str:
                        localization_findings['ST_DEPRESSION'].append(location)
                    elif 'T wave' in col_str:
                        localization_findings['T_WAVE'].append(location)
                
                # Check for axis deviation
                if 'axis deviation' in col_str.lower():
                    if 'left' in col_str.lower():
                        localization_findings['QRS_AXIS'].append('Left axis deviation')
                    elif 'right' in col_str.lower():
                        localization_findings['QRS_AXIS'].append('Right axis deviation')
                    elif 'extreme' in col_str.lower() or 'northwest' in col_str.lower():
                        localization_findings['QRS_AXIS'].append('Extreme axis deviation')
                elif 'normal axis' in col_str.lower():
                    localization_findings['QRS_AXIS'].append('Normal axis')
            except (ValueError, TypeError):
                continue
        
        # Remove empty categories
        return {k: v for k, v in localization_findings.items() if v}
    
    def _generate_special_question_for_ecg(self, row: pd.Series) -> Tuple[str, str, float]:
        """
        Generate ONE special question for ECGs marked as having special questions.
        This ensures marked ECGs ALWAYS get their special question.
        """
        assigned_category = row.get('special_question_category')
        
        def afib_prompt_tuple():
            afib_prompt = random.choice(self.afib_risk_prompts)
            return (afib_prompt, 'afib_risk', 1.0)

        def shd_prompt_tuple():
            shd_prompt = random.choice(self.structural_heart_disease_prompts)
            return (shd_prompt, 'structural_heart_disease', 1.0)

        def lvef_prompt_tuple():
            lvef_prompt = random.choice(self.lvef_prompts)
            return (lvef_prompt, 'lvef', 1.0)

        def acs_prompt_tuple():
            acs_prompt = random.choice(self.acs_severity_prompts)
            return (acs_prompt, 'acs_severity', 1.0)

        def culprit_prompt_tuple():
            culprit_prompt = random.choice(self.culprit_artery_prompts)
            return (culprit_prompt, 'culprit_artery', 1.0)

        def has_valid_pci_regions():
            if 'acs_pci_regions' not in row.index or pd.isna(row.get('acs_pci_regions')):
                return False
            regions = str(row.get('acs_pci_regions')).strip()
            return len(regions) > 2 and regions != '[]'

        def category_available(check_column):
            return check_column in row.index and pd.notna(row.get(check_column))

        if assigned_category:
            assigned_category = str(assigned_category).lower()
            if assigned_category.startswith('afib') and category_available('afib_label_2y') and category_available('afib_label_5y'):
                return afib_prompt_tuple()
            if assigned_category.startswith('shd') and category_available('echonext_shd'):
                return shd_prompt_tuple()
            if assigned_category == 'lvef' and category_available('deepecho_Visually_Estimated_EF'):
                return lvef_prompt_tuple()
            if assigned_category.startswith('acs') and category_available('acs_condition_severity'):
                from utils.constants import ACS_ACUTE_CONDITIONS
                acs_condition = row.get('acs_condition_severity')
                if acs_condition in ACS_ACUTE_CONDITIONS and has_valid_pci_regions():
                    return culprit_prompt_tuple()
                return acs_prompt_tuple()
            # If assigned but data missing (e.g., due to merge issues), fall back to generic handling below

        # Check which special data this ECG has and return the first available
        
        # 1. Check AFib risk
        if ('afib_label_2y' in row.index and pd.notna(row.get('afib_label_2y')) and
            'afib_label_5y' in row.index and pd.notna(row.get('afib_label_5y'))):
            return afib_prompt_tuple()
        
        # 2. Check SHD
        if 'echonext_shd' in row.index and pd.notna(row.get('echonext_shd')):
            return shd_prompt_tuple()
        
        # 3. Check LVEF
        if 'deepecho_Visually_Estimated_EF' in row.index and pd.notna(row.get('deepecho_Visually_Estimated_EF')):
            return lvef_prompt_tuple()
        
        # 4. Check ACS
        if 'acs_condition_severity' in row.index and pd.notna(row.get('acs_condition_severity')):
            from utils.constants import ACS_ACUTE_CONDITIONS
            acs_condition = row.get('acs_condition_severity')
            if acs_condition in ACS_ACUTE_CONDITIONS and has_valid_pci_regions():
                return culprit_prompt_tuple()
            return acs_prompt_tuple()

        # Fallback - should not happen if marking is correct
        return None
    
    def generate_prompts_for_ecg(self, row: pd.Series) -> List[Tuple[str, str, float]]:
        """
        Generate multiple prompts for a single ECG.
        Returns list of (prompt, category, weight) tuples.
        """
        prompts = []
        
        # PRIORITY: If this ECG is marked for special questions AND we're limited to 1 prompt
        # Then ONLY generate the special question
        if 'has_special_question' in row.index and row.get('has_special_question') == True:
            special_q = self._generate_special_question_for_ecg(row)
            if special_q:
                # When limited to 1 prompt per ECG, return ONLY the special question
                return [special_q]
        
        # Get ECG characteristics
        ecg_type = row.get('ecg_type', 'unknown')
        active_categories = self.determine_active_categories(row)
        num_active_categories = len(active_categories)
        localization_findings = self.check_localization_findings(row)
        
        # 1. Always add one interpretation prompt (highest weight)
        interp_prompt = random.choice(self.interpretation_prompts)
        prompts.append((interp_prompt, 'interpretation', 1.0))
        
        # 1b. Add JSON interpretation prompt (always include for structured output)
        json_prompt = random.choice(self.json_prompts)
        prompts.append((json_prompt, 'json_interpretation', 0.9))
        
        # 2. Add category-specific prompts for each active category
        if active_categories:
            # Sort categories by clinical priority
            priority_order = ['INFARCT, ISCHEMIA', 'RHYTHM', 'CONDUCTION', 
                            'CHAMBER ENLARGEMENT', 'PERICARDITIS', 'OTHER']
            
            sorted_categories = sorted(active_categories.keys(), 
                                     key=lambda x: priority_order.index(x) 
                                     if x in priority_order else 999)
            
            # Add prompts for each active category (up to 4 to compensate for fewer HR questions)
            for i, category in enumerate(sorted_categories[:4]):
                if category in self.category_specific_prompts:
                    cat_prompt = random.choice(self.category_specific_prompts[category])
                    # Higher weight for more critical categories
                    weight = 0.85 if i == 0 else 0.7 if i == 1 else 0.5 if i == 2 else 0.4
                    prompts.append((cat_prompt, f'category_{category.lower().replace(" ", "_").replace(",", "")}', weight))
        else:
            # If no specific findings, add a general rhythm question
            rhythm_prompt = random.choice(self.category_specific_prompts["RHYTHM"])
            prompts.append((rhythm_prompt, 'category_rhythm', 0.5))
        
        # 3. Add localization prompts if relevant findings exist
        if localization_findings:
            # Add up to 2 localization prompts for different finding types
            localization_count = 0
            for finding_type, locations in localization_findings.items():
                if localization_count >= 2:  # Limit to 2 localization prompts
                    break
                if finding_type in self.localization_prompts:
                    loc_prompt = random.choice(self.localization_prompts[finding_type])
                    # Higher weight for critical findings
                    weight = 0.9 if finding_type in ['ST_ELEVATION', 'Q_WAVE'] else 0.7
                    prompts.append((loc_prompt, f'localization_{finding_type.lower()}', weight))
                    localization_count += 1
        
        # 4. Add classification prompt (more important for abnormal ECGs)
        class_prompt = random.choice(self.classification_prompts)
        class_weight = 0.8 if ecg_type == 'pathological' else 0.6 if ecg_type == 'borderline' else 0.4
        prompts.append((class_prompt, 'classification', class_weight))
        
        # 5. Add ECG interval questions (heart rate + intervals) with 5% probability
        has_ecg_data = False
        
        # Check if we have heart rate data
        if 'rr_interval' in row.index and pd.notna(row.get('rr_interval')):
            has_ecg_data = True
        # Check for MHI VentricularRate
        elif 'RestingECG_OriginalRestingECGMeasurements_VentricularRate' in row.index:
            has_ecg_data = True
        # Check for interval data - MHI dataset
        elif any('RestingECG' in str(col) and 'Interval' in str(col) for col in row.index):
            has_ecg_data = True
        # For MIMIC, check if we have onset/offset columns to calculate
        elif any(col in row.index for col in ['p_onset', 'qrs_onset', 't_end']):
            has_ecg_data = True
        # Check if PR/QT interval columns exist directly
        elif any(col in row.index for col in ['pr_interval', 'qt_interval', 'qtc_interval']):
            has_ecg_data = True
        
        if has_ecg_data and random.random() < 0.05:  # 5% chance to include ecg_interval question
            ecg_interval_prompt = random.choice(self.ecg_interval_prompts)
            prompts.append((ecg_interval_prompt, 'ecg_interval', 0.8))
        
        # 6. For complex ECGs with multiple categories, add an extra focused prompt
        if num_active_categories >= 3:
            # Add another interpretation prompt focusing on complexity
            complex_prompts = [
                "What are all the abnormalities in this complex ECG?",
                "Can you list all findings in this multi-pathology ECG?",
                "Please provide a comprehensive analysis of this abnormal ECG.",
            ]
            prompts.append((random.choice(complex_prompts), 'interpretation_complex', 0.9))
        
        # 5. For critical findings, add urgency assessment
        if 'INFARCT, ISCHEMIA' in active_categories or 'RHYTHM' in active_categories:
            if any('Acute MI' in finding or 'ST elevation' in finding 
                   for findings in active_categories.values() for finding in findings):
                urgency_prompts = [
                    "Does this ECG require immediate intervention?",
                    "Is this an emergency ECG finding?",
                    "What is the clinical urgency of this ECG?",
                ]
                prompts.append((random.choice(urgency_prompts), 'urgency_assessment', 1.0))
        
        # 6. Add random YES/NO finding questions (4-5% of prompts)
        # Use deterministic selection based on ECG identifier for reproducibility
        ecg_id = str(row.get('waveform_name', row.get('npy_id', '')))
        
        if ecg_id:
            # Create a deterministic hash-based selection
            import hashlib
            hash_val = int(hashlib.md5((ecg_id + 'random_finding').encode()).hexdigest()[:8], 16)
            
            # Approximately 5% chance (modulo 20 gives 0-19, so < 1 is 5%)
            if hash_val % 20 < 1:
                # Select question based on hash (deterministic but varied)
                question_idx = hash_val % len(self.random_finding_prompts)
                selected_question = self.random_finding_prompts[question_idx]
                prompts.append((selected_question, 'random_finding_question', 0.85))
        
        # 7. Add structural heart disease questions for MHI dataset (if echonext_shd is present)
        # These questions don't affect the normal ECG percentage
        if 'echonext_shd' in row.index and pd.notna(row.get('echonext_shd')):
            # Check if this is MHI data (has MHI-specific columns or dataset indicator)
            is_mhi = False
            if 'dataset' in row.index and row.get('dataset') == 'mhi':
                is_mhi = True
            elif 'dataset_source' in row.index and row.get('dataset_source') == 'mhi':
                is_mhi = True
            elif 'RestingECG_OriginalRestingECGMeasurements_VentricularRate' in row.index:
                is_mhi = True  # This column is MHI-specific
            
            if is_mhi:
                # Add with 5% probability to achieve ~5% overall
                if random.random() < 0.05:
                    shd_prompt = random.choice(self.structural_heart_disease_prompts)
                    prompts.append((shd_prompt, 'structural_heart_disease', 0.95))
        
        # 8. Add LVEF questions for MHI dataset (if deepecho_Visually_Estimated_EF is present)
        if 'deepecho_Visually_Estimated_EF' in row.index and pd.notna(row.get('deepecho_Visually_Estimated_EF')):
            # Check if this is MHI data
            is_mhi = False
            if 'dataset' in row.index and row.get('dataset') == 'mhi':
                is_mhi = True
            elif 'dataset_source' in row.index and row.get('dataset_source') == 'mhi':
                is_mhi = True
            elif 'RestingECG_OriginalRestingECGMeasurements_VentricularRate' in row.index:
                is_mhi = True
            
            if is_mhi:
                # Add with 2% probability
                if random.random() < 0.02:
                    lvef_prompt = random.choice(self.lvef_prompts)
                    prompts.append((lvef_prompt, 'lvef', 0.95))
        
        # 9. Add ACS questions for MHI dataset (if acs_condition_severity is present)
        if 'acs_condition_severity' in row.index and pd.notna(row.get('acs_condition_severity')):
            # Check if this is MHI data
            is_mhi = False
            if 'dataset' in row.index and row.get('dataset') == 'mhi':
                is_mhi = True
            elif 'dataset_source' in row.index and row.get('dataset_source') == 'mhi':
                is_mhi = True
            elif 'RestingECG_OriginalRestingECGMeasurements_VentricularRate' in row.index:
                is_mhi = True
            
            if is_mhi:
                # Add with 10% probability to achieve ~5% overall (since many ECGs have ACS data)
                if random.random() < 0.10:
                    acs_prompt = random.choice(self.acs_severity_prompts)
                    prompts.append((acs_prompt, 'acs_severity', 0.98))  # High priority for acute conditions
                    
                    # Check if it's an acute occlusion to add culprit artery question
                    from utils.constants import ACS_ACUTE_CONDITIONS
                    acs_condition = row.get('acs_condition_severity')
                    
                    # If it's an acute occlusion and we have PCI regions data, add culprit artery question
                    if (acs_condition in ACS_ACUTE_CONDITIONS and 
                        'acs_pci_regions' in row.index and 
                        pd.notna(row.get('acs_pci_regions'))):
                        # Add culprit artery follow-up question
                        culprit_prompt = random.choice(self.culprit_artery_prompts)
                        prompts.append((culprit_prompt, 'culprit_artery', 0.97))
        
        # 10. Add AFib risk questions for MHI dataset (if afib_label_2y and afib_label_5y are present)
        if ('afib_label_2y' in row.index and pd.notna(row.get('afib_label_2y')) and
            'afib_label_5y' in row.index and pd.notna(row.get('afib_label_5y'))):
            # Check if this is MHI data
            is_mhi = False
            if 'dataset' in row.index and row.get('dataset') == 'mhi':
                is_mhi = True
            elif 'dataset_source' in row.index and row.get('dataset_source') == 'mhi':
                is_mhi = True
            elif 'RestingECG_OriginalRestingECGMeasurements_VentricularRate' in row.index:
                is_mhi = True
            
            if is_mhi:
                # Add with 2.5% probability to achieve max 5% overall (since AFib data is very common)
                if random.random() < 0.025:
                    afib_risk_prompt = random.choice(self.afib_risk_prompts)
                    prompts.append((afib_risk_prompt, 'afib_risk', 0.95))
        
        return prompts
    
    def process_dataframe(self, df: pd.DataFrame, max_prompts_per_ecg: int = 5) -> pd.DataFrame:
        """
        Process entire dataframe to create multiple rows per ECG with different prompts.
        
        Args:
            df: Input dataframe with ECG data
            max_prompts_per_ecg: Maximum number of prompts to generate per ECG
            
        Returns:
            Expanded dataframe with multiple prompt rows per ECG
        """
        all_rows = []
        
        for idx, row in df.iterrows():
            # Generate prompts for this ECG
            prompts = self.generate_prompts_for_ecg(row)
            
            # Apply max_prompts_per_ecg limit to all ECGs equally
            if len(prompts) > max_prompts_per_ecg:
                # Sort by weight and take top N
                prompts = sorted(prompts, key=lambda x: x[2], reverse=True)[:max_prompts_per_ecg]
            
            # Create a row for each prompt
            for prompt_text, prompt_category, prompt_weight in prompts:
                new_row = row.copy()
                new_row['prompt'] = prompt_text
                new_row['prompt_category'] = prompt_category
                new_row['prompt_weight'] = prompt_weight
                
                # Add metadata about the prompt
                new_row['prompt_type'] = prompt_category.split('_')[0]  # interpretation, category, classification, etc.
                new_row['num_active_categories'] = len(self.determine_active_categories(row))
                
                # Expected output type hint
                if 'interpretation' in prompt_category:
                    new_row['expected_output_type'] = 'full_report'
                elif 'category' in prompt_category:
                    new_row['expected_output_type'] = 'category_specific'
                elif 'classification' in prompt_category:
                    new_row['expected_output_type'] = 'classification'
                elif 'urgency' in prompt_category:
                    new_row['expected_output_type'] = 'urgency_level'
                else:
                    new_row['expected_output_type'] = 'general'
                
                all_rows.append(new_row)
        
        # Create new dataframe
        result_df = pd.DataFrame(all_rows)
        
        # Reset index
        result_df.reset_index(drop=True, inplace=True)
        
        return result_df
    
    def get_prompt_statistics(self, df: pd.DataFrame) -> Dict:
        """Get statistics about generated prompts"""
        stats = {
            'total_prompts': len(df),
            'unique_ecgs': df['waveform_name'].nunique() if 'waveform_name' in df.columns else 0,
            'prompts_per_ecg': len(df) / df['waveform_name'].nunique() if 'waveform_name' in df.columns and df['waveform_name'].nunique() > 0 else 0,
            'prompt_category_distribution': df['prompt_category'].value_counts().to_dict() if 'prompt_category' in df.columns else {},
            'prompt_type_distribution': df['prompt_type'].value_counts().to_dict() if 'prompt_type' in df.columns else {},
            'weight_distribution': {
                'mean': df['prompt_weight'].mean() if 'prompt_weight' in df.columns else 0,
                'std': df['prompt_weight'].std() if 'prompt_weight' in df.columns else 0,
                'min': df['prompt_weight'].min() if 'prompt_weight' in df.columns else 0,
                'max': df['prompt_weight'].max() if 'prompt_weight' in df.columns else 0
            }
        }
        return stats


def main():
    """Main function to demonstrate prompt generation"""
    
    # Initialize prompt maker
    print("Initializing ECG Prompt Maker...")
    prompt_maker = ECGPromptMaker()
    
    # Load the ECG data with classifications
    input_path = '/volume/ECG_tokenizer/output/mimic_mhi_psa_test_updated_with_questions_with_ecg_type.parquet'
    print(f"\nLoading data from: {input_path}")
    df = pd.read_parquet(input_path)
    
    # Take a sample for demonstration (remove this for full processing)
    sample_size = 1000
    df_sample = df.head(sample_size).copy()
    print(f"Processing {len(df_sample)} ECGs...")
    
    # Generate prompts
    df_with_prompts = prompt_maker.process_dataframe(df_sample, max_prompts_per_ecg=4)
    
    # Get statistics
    stats = prompt_maker.get_prompt_statistics(df_with_prompts)
    
    print("\n" + "="*60)
    print("PROMPT GENERATION STATISTICS")
    print("="*60)
    print(f"Total prompts generated: {stats['total_prompts']}")
    print(f"Unique ECGs: {stats['unique_ecgs']}")
    print(f"Average prompts per ECG: {stats['prompts_per_ecg']:.2f}")
    
    print("\nPrompt Type Distribution:")
    for ptype, count in stats['prompt_type_distribution'].items():
        percentage = (count / stats['total_prompts']) * 100
        print(f"  {ptype}: {count} ({percentage:.1f}%)")
    
    print("\nPrompt Weight Statistics:")
    print(f"  Mean weight: {stats['weight_distribution']['mean']:.3f}")
    print(f"  Std deviation: {stats['weight_distribution']['std']:.3f}")
    print(f"  Min weight: {stats['weight_distribution']['min']:.3f}")
    print(f"  Max weight: {stats['weight_distribution']['max']:.3f}")
    
    # Show examples
    print("\n" + "="*60)
    print("EXAMPLE PROMPTS")
    print("="*60)
    
    # Show prompts for one ECG
    example_ecg = df_with_prompts['waveform_name'].iloc[0]
    ecg_prompts = df_with_prompts[df_with_prompts['waveform_name'] == example_ecg]
    
    print(f"\nExample ECG: {example_ecg}")
    print(f"ECG Type: {ecg_prompts.iloc[0]['ecg_type']}")
    print(f"Number of prompts: {len(ecg_prompts)}")
    
    for idx, row in ecg_prompts.iterrows():
        print(f"\nPrompt {idx + 1}:")
        print(f"  Text: {row['prompt']}")
        print(f"  Category: {row['prompt_category']}")
        print(f"  Weight: {row['prompt_weight']:.2f}")
        print(f"  Expected output: {row['expected_output_type']}")
    
    # Save the result
    output_path = input_path.replace('.parquet', '_with_prompts.parquet')
    print(f"\n\nSaving to: {output_path}")
    df_with_prompts.to_parquet(output_path, index=False)
    
    print("\nProcessing complete!")
    
    return df_with_prompts


if __name__ == "__main__":
    df = main()
