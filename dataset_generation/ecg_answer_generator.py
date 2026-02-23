#!/usr/bin/env python3
"""
ECG Answer Generator - Creates appropriate answers for each prompt type.
Generates interpretation reports, category-specific answers, and classifications.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Set
from dataset_column_mappings import DatasetColumnMapper
from utils.constants import (
    DEEPECG_CATEGORIES,
    DEEPECG_DIAGNOSIS_TRANSLATION,
    DEEPECG_PATHOLOGICAL_LIMIT,
)
from ecg_prompt_maker import QAFeatureFlags

LEAD_NAME_PATTERN = re.compile(r'(V\d+|aV[RLF]|I{1,3})(?=[Vv]|aV|I{1,3})')


class ECGAnswerGenerator:
    """Generate appropriate answers for different ECG prompt types"""

    def __init__(
        self,
        language: str = 'en',
        dataset: str = 'mimic',
        qa_feature_flags: Optional[QAFeatureFlags] = None,
    ):
        """
        Args:
            language: 'en' for English, 'fr' for French
            dataset: Dataset name ('mimic' or others) for metadata merging
            qa_feature_flags: When set, heart rate / interval content is only added when flags permit.
        """
        
        # Load category definitions from constants
        self.categories_dict = DEEPECG_CATEGORIES
        
        # Load translations from constants
        trans_data = DEEPECG_DIAGNOSIS_TRANSLATION
        
        # Create translation lookup dictionary
        self.translations = {}
        self.language = language
        self.dataset = dataset
        trans_key = f'translation_{language}'
        
        # Initialize column mapper for dataset-specific columns
        self.column_mapper = DatasetColumnMapper(dataset)
        
        for item in trans_data.get('deepecg', []):
            column_name = item.get('column_name')
            translation = item.get(trans_key, column_name)  # Fallback to column name if no translation
            if column_name:
                self.translations[column_name] = translation
        
        # Load MIMIC metadata if dataset is mimic
        self.mimic_metadata = None
        if dataset == 'mimic':
            try:
                mimic_path = '/media/data1/datasets/MIMIC-IV/Diagnosis/mimic_labelbox_bert_v4_all.parquet'
                self.mimic_metadata = pd.read_parquet(mimic_path)
                # Extract npy ID from the file path
                self.mimic_metadata['npy_id'] = self.mimic_metadata['npy_path'].str.extract(r'/([^/]+)\.npy$')[0]
                print(f"Loaded MIMIC metadata with {len(self.mimic_metadata)} records")
            except Exception as e:
                print(f"Warning: Could not load MIMIC metadata: {e}")
                self.mimic_metadata = None
        
        # Define category order for structured reports
        self.category_order = [
            'RHYTHM', 
            'CONDUCTION', 
            'INFARCT, ISCHEMIA',
            'CHAMBER ENLARGEMENT',
            'PERICARDITIS',
            'OTHER'
        ]

        limit_config = DEEPECG_PATHOLOGICAL_LIMIT.get('deepecg', {})
        self.pathological_labels = set(limit_config.get('pathological', []))
        self.limit_labels = set(limit_config.get('limit', []))
        self.qa_feature_flags = qa_feature_flags

    def get_active_findings(self, row: pd.Series) -> Dict[str, List[str]]:
        """
        Extract active findings from the row organized by category.
        For MHI dataset, handles _bert_model columns which are logits (>0.5 = positive).
        Returns dict: category -> list of active condition names
        """
        active_findings = {}
        
        for category, conditions in self.categories_dict.items():
            category_findings = []
            
            for condition in conditions:
                # Check various column name formats
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
                                        category_findings.append(condition)
                                        break
                                # For regular columns, use >= 1 threshold
                                else:
                                    if float(value) >= 1:
                                        category_findings.append(condition)
                                        break
                        except (ValueError, TypeError):
                            continue
            
            if category_findings:
                active_findings[category] = category_findings
        
        return active_findings
    
    def format_finding_name(self, finding: str) -> str:
        """Format finding name using translations if available"""
        # First check if we have a translation for this finding
        if finding in self.translations:
            return self.translations[finding]
        
        # Fallback to manual formatting for items not in translations
        replacements = {
            'Afib': 'Atrial fibrillation',
            'RBBB': 'Right bundle branch block',
            'LBBB': 'Left bundle branch block',
            'LVH': 'Left ventricular hypertrophy',
            'RVH': 'Right ventricular hypertrophy',
            'LAE': 'Left atrial enlargement',
            'RAE': 'Right atrial enlargement',
            'AV': 'Atrioventricular',
            'MI': 'myocardial infarction',
            'ST elevation': 'ST segment elevation',
            'ST depression': 'ST segment depression',
            '1st degree': 'First-degree',
            '2nd degree': 'Second-degree',
            '3rd degree': 'Third-degree'
        }
        
        formatted = finding
        for old, new in replacements.items():
            if old in formatted:
                formatted = formatted.replace(old, new)
        
        return formatted
    
    def generate_interpretation_answer(self, row: pd.Series) -> str:
        """
        Generate a full ECG interpretation report.
        Uses the REPORT column if available, otherwise falls back to individual findings.
        Includes heart rate when available and when qa_feature_flags.has_heart_rate (or no flags set).
        """
        heart_rate = self.calculate_heart_rate(row)
        include_hr = self.qa_feature_flags is None or self.qa_feature_flags.has_heart_rate

        report = self.column_mapper.get_value(row, 'report', default=None)
        if report is not None and str(report).strip():
            report = str(report).strip()
            if include_hr and heart_rate and heart_rate > 0:
                # Check if report already contains heart rate info
                if 'HR:' not in report and 'heart rate' not in report.lower():
                    # Find the first semicolon or end of first statement to insert HR
                    if ';' in report:
                        parts = report.split(';', 1)
                        # Add HR to first part if it contains rhythm-related terms
                        if any(term in parts[0].lower() for term in ['rhythm', 'sinus', 'atrial', 'ventricular']):
                            report = f"{parts[0]} (HR: {heart_rate} bpm);{parts[1]}"
                        else:
                            # Add as separate statement after first part
                            report = f"{parts[0]}; HR: {heart_rate} bpm;{parts[1]}"
                    else:
                        # Single statement report
                        if any(term in report.lower() for term in ['rhythm', 'sinus', 'atrial', 'ventricular']):
                            report = f"{report} (HR: {heart_rate} bpm)"
                        else:
                            report = f"{report}; HR: {heart_rate} bpm"
            
            return report
        
        # Fallback to generating from individual findings if no report
        active_findings = self.get_active_findings(row)
        
        if not active_findings:
            if include_hr and heart_rate and heart_rate > 0:
                return f"Normal sinus rhythm (HR: {heart_rate} bpm); Normal ECG"
            return "Normal sinus rhythm; Normal ECG"

        report_parts = []
        heart_rate_added = False
        for category in self.category_order:
            if category in active_findings:
                for finding in active_findings[category]:
                    formatted = self.format_finding_name(finding)
                    if include_hr and category == 'RHYTHM' and heart_rate and heart_rate > 0 and not heart_rate_added and 'rhythm' in formatted.lower():
                        formatted += f" (HR: {heart_rate} bpm)"
                        heart_rate_added = True
                    report_parts.append(formatted)
        
        # Add overall assessment based on ecg_type
        ecg_type = row.get('ecg_type', 'unknown')
        if ecg_type == 'pathological':
            report_parts.append("Abnormal ECG")
        elif ecg_type == 'borderline':
            report_parts.append("Borderline ECG")
        
        # Join with semicolons (standard ECG report format)
        return "; ".join(report_parts)
    
    def calculate_heart_rate(self, row: pd.Series) -> Optional[float]:
        """
        Calculate heart rate from RR interval or ventricular rate.
        Returns heart rate in bpm or None if not available.
        Special case: Returns 0 if MHI VentricularRate is 0 (artifacts).
        """
        # Check for heart rate column (might be already calculated)
        if 'heart_rate' in row.index and pd.notna(row['heart_rate']):
            hr = float(row['heart_rate'])
            if hr > 0:
                return hr
            elif hr == 0:
                return 0  # Special case for artifacts
        
        # For MHI dataset - check VentricularRate column
        if 'RestingECG_OriginalRestingECGMeasurements_VentricularRate' in row.index:
            try:
                ventricular_rate = row['RestingECG_OriginalRestingECGMeasurements_VentricularRate']
                if pd.notna(ventricular_rate):
                    # Convert to float (it might be a string)
                    hr = float(ventricular_rate)
                    if hr == 0:
                        return 0  # Special case: 0 indicates artifacts
                    elif hr > 0:
                        return round(hr, 1)
            except (ValueError, TypeError):
                pass
        
        
        # Check for RR interval columns (for MIMIC dataset)
        rr_columns = ['rr_interval']
        for col in rr_columns:
            if col in row.index and pd.notna(row[col]):
                try:
                    rr_ms = float(row[col])
                    if rr_ms > 0:
                        # Convert RR interval (ms) to heart rate (bpm)
                        heart_rate = 60000.0 / rr_ms
                        return round(heart_rate, 1)
                except (ValueError, TypeError):
                    continue
        
        # Check for rhythm-related findings that might indicate rate
        if 'Bradycardia' in str(row.get('RHYTHM', '')):
            return None  # We know it's slow but don't have exact rate
        elif 'Tachycardia' in str(row.get('RHYTHM', '')):
            return None  # We know it's fast but don't have exact rate
        
        return None
    
    def check_st_segment_context(self, row: pd.Series) -> tuple:
        """
        Check context for ST segment changes.
        Returns (has_acute_mi, has_early_repol, has_lvh, st_locations)
        """
        has_acute_mi = False
        has_early_repol = False  
        has_lvh = False
        st_locations = []
        
        # Check for Acute MI (regular columns)
        if 'Acute_MI' in row.index and pd.notna(row['Acute_MI']):
            try:
                has_acute_mi = float(row['Acute_MI']) >= 1
            except (ValueError, TypeError):
                pass
        if 'Acute MI' in row.index and pd.notna(row['Acute MI']):
            try:
                has_acute_mi = float(row['Acute MI']) >= 1
            except (ValueError, TypeError):
                pass
        
        # Check for Acute MI _bert_model column (MHI dataset)
        if not has_acute_mi and 'Acute_MI_bert_model' in row.index and pd.notna(row['Acute_MI_bert_model']):
            try:
                has_acute_mi = float(row['Acute_MI_bert_model']) > 0.5
            except (ValueError, TypeError):
                pass
                
        # Check for Early Repolarization
        if 'Early repolarization' in row.index and pd.notna(row['Early repolarization']):
            try:
                has_early_repol = float(row['Early repolarization']) >= 1
            except (ValueError, TypeError):
                pass
        
        # Check for Early Repolarization _bert_model (MHI)
        if not has_early_repol and 'Early repolarization_bert_model' in row.index and pd.notna(row['Early repolarization_bert_model']):
            try:
                has_early_repol = float(row['Early repolarization_bert_model']) > 0.5
            except (ValueError, TypeError):
                pass
                
        # Check for LVH
        if 'Left ventricular hypertrophy' in row.index and pd.notna(row['Left ventricular hypertrophy']):
            try:
                has_lvh = float(row['Left ventricular hypertrophy']) >= 1
            except (ValueError, TypeError):
                pass
        
        # Check for LVH _bert_model (MHI)
        if not has_lvh and 'Left ventricular hypertrophy_bert_model' in row.index and pd.notna(row['Left ventricular hypertrophy_bert_model']):
            try:
                has_lvh = float(row['Left ventricular hypertrophy_bert_model']) > 0.5
            except (ValueError, TypeError):
                pass
                
        # Check for ST elevations - regular columns
        for col in row.index:
            if 'ST elevation' in str(col) and pd.notna(row[col]):
                # Skip _bert_model columns in this loop
                if '_bert_model' in str(col):
                    continue
                try:
                    if float(row[col]) >= 1:
                        if '(' in str(col) and ')' in str(col):
                            location = str(col)[str(col).find('(')+1:str(col).find(')')]
                            st_locations.append(location)
                except (ValueError, TypeError):
                    continue
        
        # Check for ST elevations - _bert_model columns (MHI)
        for col in row.index:
            if 'ST elevation' in str(col) and '_bert_model' in str(col) and pd.notna(row[col]):
                try:
                    if float(row[col]) > 0.5:  # Use 0.5 threshold for _bert_model
                        # Extract location from column name
                        # Format: "ST elevation (location)_bert_model"
                        if '(' in str(col) and ')' in str(col):
                            location = str(col)[str(col).find('(')+1:str(col).find(')')]
                            if location not in st_locations:  # Avoid duplicates
                                st_locations.append(location)
                except (ValueError, TypeError):
                    continue
                    
        return has_acute_mi, has_early_repol, has_lvh, st_locations
    
    def generate_category_answer(self, row: pd.Series, category: str) -> str:
        """
        Generate category-specific answer.
        Properly handles YES/NO questions and specific condition queries.
        Includes heart rate for rhythm-related questions.
        """
        active_findings = self.get_active_findings(row)
        prompt_text = row.get('prompt', '').lower()
        heart_rate = self.calculate_heart_rate(row)
        include_hr = self.qa_feature_flags is None or self.qa_feature_flags.has_heart_rate

        # Extract category from prompt_category (e.g., "category_rhythm" -> "RHYTHM")
        category_map = {
            'rhythm': 'RHYTHM',
            'conduction': 'CONDUCTION',
            'infarct_ischemia': 'INFARCT, ISCHEMIA',
            'chamber_enlargement': 'CHAMBER ENLARGEMENT',
            'pericarditis': 'PERICARDITIS',
            'other': 'OTHER'
        }
        
        # Get the actual category name
        mapped_category = None
        for key, val in category_map.items():
            if key in category.lower():
                mapped_category = val
                break
        
        # Check if this is a rhythm/rate question
        is_rhythm_question = any(word in prompt_text for word in [
            'rhythm', 'rate', 'heart rate', 'pulse', 'bpm', 'beats'
        ])
        
        # Check if this is a YES/NO question
        is_yes_no_question = any(phrase in prompt_text for phrase in [
            'is there', 'are there', 'are the', 'does this', 'does it', 
            'can you identify', 'do you see', 'is this'
        ])
        
        # Check if asking about abnormalities/problems
        asking_for_abnormalities = any(word in prompt_text for word in [
            'abnormalit', 'problem', 'concerning', 'wrong', 'patholog', 
            'abnormal', 'disturbanc', 'disorder'
        ])
        
        # Check for specific conditions mentioned in the prompt
        specific_conditions = {
            'ectopic beat': ['Premature ventricular complex', 'Premature atrial complex'],
            'pre-excitation': ['WPW', 'Wolff-Parkinson-White', 'Pre-excitation'],
            'heart block': ['First-degree AV block', 'Second-degree AV block', 'Third-degree AV block', 
                           'Complete heart block', 'AV block'],
            'bundle branch': ['LBBB', 'RBBB', 'Left bundle branch block', 'Right bundle branch block',
                             'Left anterior fascicular block', 'Left posterior fascicular block', 
                             'Fascicular block', 'LAFB', 'LPFB'],
            'atrial fibrillation': ['Afib', 'Atrial fibrillation'],
            'atrial flutter': ['Atrial flutter', 'Flutter'],
            'ischemia': ['ST elevation', 'ST depression', 'T wave inversion', 'Ischemia'],
            'infarction': ['MI', 'Myocardial infarction', 'Acute MI', 'Q waves'],
            'acute mi': ['Acute MI', 'acute myocardial infarction', 'STEMI'],
            'hypertrophy': ['LVH', 'RVH', 'LAE', 'RAE', 'Hypertrophy', 'Enlargement'],
            'pericarditis': ['Pericarditis', 'Pericarditic'],
            'arrhythmia': ['Tachycardia', 'Bradycardia', 'Afib', 'Flutter', 'VT', 'SVT'],
            'st elevation': ['ST elevation', 'STEMI'],
            'st depression': ['ST depression'],
            'q waves': ['Q waves', 'Pathological Q'],
            'axis deviation': ['Left axis deviation', 'Right axis deviation', 'Axis deviation'],
            'extreme axis': ['Extreme axis deviation', 'Northwest axis', 'No man\'s land']
        }
        
        # Check if prompt asks about a specific condition
        asked_condition = None
        matching_findings = []
        
        # Special handling for early repolarization question
        if 'early repolarization' in prompt_text:
            asked_condition = 'early_repolarization'
            # Check the Early repolarization column directly
            if 'Early repolarization' in row.index and pd.notna(row['Early repolarization']):
                try:
                    if float(row['Early repolarization']) >= 1:
                        matching_findings.append('Early repolarization')
                except (ValueError, TypeError):
                    pass
        
        # Special handling for pericarditis and diffuse ST elevation
        elif 'diffuse st elevation' in prompt_text or 'pericarditis' in prompt_text:
            asked_condition = 'pericarditis_st_elevation'
            # Check for pericarditis
            if 'Acute pericarditis' in row.index and pd.notna(row['Acute pericarditis']):
                try:
                    if float(row['Acute pericarditis']) >= 1:
                        # Check for ST elevations in multiple leads (diffuse)
                        st_elevation_count = 0
                        st_locations = []
                        for col in row.index:
                            if 'ST elevation' in str(col) and pd.notna(row[col]):
                                try:
                                    if float(row[col]) >= 1:
                                        st_elevation_count += 1
                                        location = str(col).replace('ST elevation', '').strip()
                                        if location.startswith('(') and location.endswith(')'):
                                            location = location[1:-1]
                                        st_locations.append(location)
                                except (ValueError, TypeError):
                                    continue
                        
                        if st_elevation_count >= 2:  # Multiple leads = diffuse
                            matching_findings.append(f"diffuse ST elevation consistent with pericarditis")
                        else:
                            matching_findings.append("pericarditis with ST changes")
                except (ValueError, TypeError):
                    pass
            
            # Even if no pericarditis, check for diffuse ST elevation pattern
            if not matching_findings:
                st_elevation_count = 0
                st_locations = []
                for col in row.index:
                    if 'ST elevation' in str(col) and pd.notna(row[col]):
                        try:
                            if float(row[col]) >= 1:
                                st_elevation_count += 1
                                location = str(col).replace('ST elevation', '').strip()
                                if location.startswith('(') and location.endswith(')'):
                                    location = location[1:-1]
                                st_locations.append(location)
                        except (ValueError, TypeError):
                            continue
                
                if st_elevation_count >= 3:  # 3+ leads suggests diffuse pattern
                    matching_findings.append(f"diffuse ST elevation in {', '.join(st_locations)}")
        
        # Special handling for LVH/RVH questions
        elif 'lvh or rvh' in prompt_text or 'ventricular hypertrophy' in prompt_text:
            asked_condition = 'lvh_or_rvh'
            # Only check for LVH and RVH specifically
            if 'Left ventricular hypertrophy' in row.index and pd.notna(row['Left ventricular hypertrophy']):
                try:
                    if float(row['Left ventricular hypertrophy']) >= 1:
                        matching_findings.append('LVH')
                except (ValueError, TypeError):
                    pass
            
            if 'Right ventricular hypertrophy' in row.index and pd.notna(row['Right ventricular hypertrophy']):
                try:
                    if float(row['Right ventricular hypertrophy']) >= 1:
                        matching_findings.append('RVH')
                except (ValueError, TypeError):
                    pass
            
            # Early return for LVH/RVH questions to avoid showing other chamber enlargements
            if is_yes_no_question:
                if matching_findings:
                    if len(matching_findings) == 1:
                        return f"Yes - {matching_findings[0]}"
                    elif len(matching_findings) == 2:
                        return "Yes - both LVH and RVH"
                else:
                    return "No - no ventricular hypertrophy"
            else:
                if matching_findings:
                    return "; ".join(matching_findings)
                else:
                    return "No ventricular hypertrophy"
        
        # Special handling for general ischemia/infarction questions
        elif any(term in prompt_text for term in ['signs of ischemia or infarction', 'ischemia or infarction', 
                                                  'ischemic or infarction', 'evidence of ischemia',
                                                  'st elevation or depression', 'st segment',
                                                  'signs of myocardial injury']):
            asked_condition = 'ischemia_or_infarction'
            
            # Get ST segment context for hard negatives
            has_acute_mi, has_early_repol, has_lvh, st_locations = self.check_st_segment_context(row)
            
            # If there are ST changes but they're due to early repolarization or LVH without acute MI
            if st_locations and not has_acute_mi:
                if has_early_repol:
                    matching_findings = ["ST changes due to early repolarization (benign). No evidence of STEMI"]
                elif has_lvh:
                    matching_findings = ["ST segment changes likely due to left ventricular hypertrophy. No evidence of acute MI"]
                else:
                    # Check for any signs of ischemia or infarction:
                    # 1. Any findings from INFARCT, ISCHEMIA category
                    if 'INFARCT, ISCHEMIA' in active_findings:
                        for finding in active_findings['INFARCT, ISCHEMIA']:
                            formatted = self.format_finding_name(finding)
                            if formatted not in matching_findings:
                                matching_findings.append(formatted)
            elif has_acute_mi:
                # Acute MI is present - report it
                if 'INFARCT, ISCHEMIA' in active_findings:
                    for finding in active_findings['INFARCT, ISCHEMIA']:
                        formatted = self.format_finding_name(finding)
                        if formatted not in matching_findings:
                            matching_findings.append(formatted)
            else:
                # No ST changes - check for other ischemia/infarction signs
                if 'INFARCT, ISCHEMIA' in active_findings:
                    for finding in active_findings['INFARCT, ISCHEMIA']:
                        formatted = self.format_finding_name(finding)
                        if formatted not in matching_findings:
                            matching_findings.append(formatted)
            
            # 2. Check for ST downsloping
            if 'ST downsloping' in row.index and pd.notna(row['ST downsloping']):
                try:
                    if float(row['ST downsloping']) >= 1:
                        matching_findings.append('ST downsloping')
                except (ValueError, TypeError):
                    pass
            
            # 3. Check for any ST depression columns
            for col in row.index:
                if 'ST depression' in str(col) and pd.notna(row[col]):
                    try:
                        if float(row[col]) >= 1:
                            location = str(col).replace('ST depression', '').strip()
                            if location.startswith('(') and location.endswith(')'):
                                location = location[1:-1]
                            matching_findings.append(f'ST depression in {location}')
                    except (ValueError, TypeError):
                        continue
            
            # 4. Check for any T wave inversion columns
            for col in row.index:
                if 'T wave inversion' in str(col) and pd.notna(row[col]):
                    try:
                        if float(row[col]) >= 1:
                            location = str(col).replace('T wave inversion', '').strip()
                            if location.startswith('(') and location.endswith(')'):
                                location = location[1:-1]
                            matching_findings.append(f'T wave inversion in {location}')
                    except (ValueError, TypeError):
                        continue
            
            # 5. Check for Q wave columns (sign of old infarction)
            for col in row.index:
                if 'Q wave' in str(col) and pd.notna(row[col]):
                    try:
                        if float(row[col]) >= 1:
                            location = str(col).replace('Q wave', '').strip()
                            if location.startswith('(') and location.endswith(')'):
                                location = location[1:-1]
                            matching_findings.append(f'Q waves in {location} (possible old infarct)')
                    except (ValueError, TypeError):
                        continue
            
            # 6. Check report for infarct mentions
            if 'report' in row.index and pd.notna(row['report']):
                report_lower = str(row['report']).lower()
                if 'infarct' in report_lower:
                    # Extract the infarct description
                    import re
                    infarct_patterns = re.findall(r'[^;]*infarct[^;]*', report_lower)
                    for pattern in infarct_patterns:
                        pattern = pattern.strip()
                        if 'age undetermined' in pattern or 'old' in pattern or 'previous' in pattern:
                            if 'Possible old infarction' not in matching_findings:
                                matching_findings.append('Possible old infarction')
                        elif 'acute' in pattern:
                            if 'Possible acute infarction' not in matching_findings:
                                matching_findings.append('Possible acute infarction')
        
        # Special handling for Acute MI questions
        elif any(term in prompt_text for term in ['acute mi', 'acute myocardial', 'stemi', 'acute infarct', 'does this ecg show acute mi']):
            asked_condition = 'acute mi'
            
            # Get ST segment context
            has_acute_mi, has_early_repol, has_lvh, st_locations = self.check_st_segment_context(row)
            
            # Check report for acute MI mentions as additional confirmation
            has_acute_mi_report = False
            if 'report' in row.index and pd.notna(row['report']):
                report_text = str(row['report']).upper()
                if any(phrase in report_text for phrase in [
                    'CONSIDER ACUTE ST ELEVATION MI',
                    'CONSIDER ACUTE INFARCT', 
                    'ACUTE MI',
                    'ACUTE MYOCARDIAL INFARCTION',
                    'ACUTE STEMI'
                ]):
                    has_acute_mi_report = True
            
            # Determine if there's truly an acute MI
            if has_acute_mi or has_acute_mi_report:
                # Just report acute MI once, ST locations are already in the prefix
                matching_findings.append('Probable acute myocardial infarction')
            elif st_locations and not has_acute_mi:
                # ST changes present but not acute MI
                if has_early_repol:
                    matching_findings = []  # Will return "No - no evidence of acute mi" with early repol explanation
                elif has_lvh:
                    matching_findings = []  # Will return "No - no evidence of acute mi" with LVH explanation
        
        # Special handling for sinus rhythm question
        elif any(phrase in prompt_text for phrase in ['sinus rhythm or something else', 'is this sinus rhythm']):
            asked_condition = 'sinus_rhythm'
            
            # Check the Sinusal column
            is_sinus = False
            if 'Sinusal' in row.index and pd.notna(row['Sinusal']):
                try:
                    is_sinus = float(row['Sinusal']) >= 1
                except (ValueError, TypeError):
                    pass
            
            if is_sinus:
                matching_findings.append("sinus rhythm")
            else:
                # It's something else - check what rhythm it actually is
                rhythm_findings = []
                
                # Check for specific rhythm abnormalities
                rhythm_cols = ['Ectopic atrial rhythm (< 100 BPM)', 'Atrial tachycardia (>= 100 BPM)', 
                              'Afib', 'Atrial flutter', 'Junctional rhythm', 'Ventricular rhythm',
                              'Bradycardia', 'Tachycardia']
                
                for col in rhythm_cols:
                    if col in row.index and pd.notna(row[col]):
                        try:
                            if float(row[col]) >= 1:
                                rhythm_findings.append(col)
                        except (ValueError, TypeError):
                            continue
                
                if rhythm_findings:
                    matching_findings.append(f"something else - {'; '.join(rhythm_findings)}")
                else:
                    # Check report for rhythm description
                    if 'report' in row.index and pd.notna(row['report']):
                        report_lower = str(row['report']).lower()
                        if 'ectopic' in report_lower:
                            matching_findings.append("something else - ectopic rhythm")
                        elif 'junctional' in report_lower:
                            matching_findings.append("something else - junctional rhythm")
                        else:
                            matching_findings.append("something else - non-sinus rhythm")
                    else:
                        matching_findings.append("something else - non-sinus rhythm")
        
        # Special handling for T wave questions
        elif any(phrase in prompt_text for phrase in ['t waves normal', 't wave normal', 'are the t waves']):
            asked_condition = 't_waves'
            
            # Check all T wave inversion columns
            t_wave_inversions = []
            
            for col in row.index:
                if 'T wave inversion' in col or 'T-wave inversion' in col:
                    try:
                        if pd.notna(row[col]) and float(row[col]) >= 1:
                            # Extract location from column name (text in parentheses)
                            if '(' in col and ')' in col:
                                location = col[col.find('(')+1:col.find(')')]
                                t_wave_inversions.append(location)
                            else:
                                t_wave_inversions.append(col)
                    except (ValueError, TypeError):
                        continue
            
            # Format the response
            if t_wave_inversions:
                # T waves are NOT normal - there are inversions
                if len(t_wave_inversions) == 1:
                    matching_findings.append(f"T wave inversions present in {t_wave_inversions[0]}")
                else:
                    locations = "; ".join(t_wave_inversions)
                    matching_findings.append(f"T wave inversions present in {locations}")
            # If no inversions found, T waves are normal - we'll handle this later
        
        # Special handling for ectopic beats - only match PVCs and PACs
        elif 'ectopic' in prompt_text:
            asked_condition = 'ectopic beat'
            # Only look for PVCs and PACs in the findings
            for cat_findings in active_findings.values():
                for finding in cat_findings:
                    if any(term in finding.lower() for term in ['premature ventricular complex', 'premature atrial complex']):
                        matching_findings.append(self.format_finding_name(finding))
        else:
            for condition_key, condition_variants in specific_conditions.items():
                if condition_key in prompt_text:
                    asked_condition = condition_key
                    
                    # Special handling for ST elevation questions
                    if condition_key == 'st elevation':
                        # Get ST segment context for hard negatives
                        has_acute_mi, has_early_repol, has_lvh, st_locations = self.check_st_segment_context(row)
                        
                        if st_locations:
                            # ST elevation is present - check context
                            if not has_acute_mi:
                                if has_early_repol:
                                    matching_findings.append("ST elevation due to early repolarization (benign)")
                                elif has_lvh:
                                    matching_findings.append("ST elevation associated with left ventricular hypertrophy")
                                else:
                                    # Report the ST elevation locations
                                    for location in st_locations:
                                        matching_findings.append(f"ST elevation in {location}")
                            else:
                                # Acute MI present with ST elevation
                                for location in st_locations:
                                    matching_findings.append(f"ST elevation in {location}")
                                if 'Acute STEMI' not in str(matching_findings):
                                    matching_findings.append("Consistent with acute STEMI")
                    
                    # Special handling for any infarct-related question
                    elif condition_key in ['infarction', 'ischemia']:
                        # Get ST segment context
                        has_acute_mi, has_early_repol, has_lvh, st_locations = self.check_st_segment_context(row)
                        
                        # If acute MI detected, add to findings
                        if has_acute_mi:
                            matching_findings.append('Probable infarct')
                        elif st_locations and not has_acute_mi:
                            # ST changes without acute MI
                            if has_early_repol:
                                # Don't report as ischemia if it's early repolarization
                                pass
                            elif has_lvh:
                                # Don't report as ischemia if it's LVH-related
                                pass
                            else:
                                # Check for other ischemic changes
                                for cat_findings in active_findings.values():
                                    for finding in cat_findings:
                                        if any(variant.lower() in finding.lower() for variant in condition_variants):
                                            matching_findings.append(self.format_finding_name(finding))
                    
                    # Check if any variant exists in active findings
                    for cat_findings in active_findings.values():
                        for finding in cat_findings:
                            if any(variant.lower() in finding.lower() for variant in condition_variants):
                                matching_findings.append(self.format_finding_name(finding))
                    break
        
        # Define normal/expected findings that are NOT abnormalities
        # Use both original names and translations
        normal_findings = [
            'Sinusal', 'Sinus rhythm', 'Sinus', 'Regular', 'Regular rhythm',
            'Normal sinus rhythm', 'Normal axis', 'Normal intervals', 
            'Normal conduction', 'Monomorph', 'Monomorph QRS complexes'
        ]
        
        # Also check the actual column names that represent normal findings
        normal_column_names = ['Sinusal', 'Regular', 'Monomorph']
        
        # Generate appropriate response
        if is_yes_no_question:
            if asked_condition:
                # Specific condition asked
                if matching_findings:
                    # Special handling for specific conditions
                    if asked_condition == 'early_repolarization':
                        return "Yes - Early repolarization present"
                    elif asked_condition == 'pericarditis_st_elevation':
                        return f"Yes - {'; '.join(matching_findings)}"
                    elif asked_condition == 'lvh_or_rvh':
                        # Format the response appropriately
                        if len(matching_findings) == 1:
                            return f"Yes - {matching_findings[0]}"
                        elif len(matching_findings) == 2:
                            return "Yes - both LVH and RVH"
                        else:
                            return f"Yes - {'; '.join(matching_findings)}"
                    elif asked_condition == 't_waves':
                        return f"No - {'; '.join(matching_findings)}"
                    elif asked_condition == 'sinus_rhythm':
                        # Check if it's sinus or something else
                        finding = matching_findings[0]
                        if 'sinus rhythm' == finding:
                            return "Yes - sinus rhythm"
                        else:
                            return f"No - {finding}"
                    else:
                        return f"Yes - {'; '.join(matching_findings)}"
                else:
                    # No matching findings - but check for hard negatives
                    if asked_condition == 'acute mi':
                        # Check if ST changes are due to early repol or LVH
                        has_acute_mi, has_early_repol, has_lvh, st_locations = self.check_st_segment_context(row)
                        if st_locations and not has_acute_mi:
                            if has_early_repol:
                                return "No - ST changes are due to early repolarization (benign). No evidence of acute MI"
                            elif has_lvh:
                                return "No - ST segment changes likely due to left ventricular hypertrophy. No evidence of acute MI"
                        return "No - no evidence of acute mi"
                    elif asked_condition == 'early_repolarization':
                        return "No - no early repolarization"
                    elif asked_condition == 'pericarditis_st_elevation':
                        return "No - no evidence of diffuse ST elevation"
                    elif asked_condition == 'lvh_or_rvh':
                        return "No - no ventricular hypertrophy"
                    elif asked_condition == 'ischemia_or_infarction':
                        # Check if already handled as hard negative
                        if matching_findings and "early repolarization" in str(matching_findings[0]).lower():
                            return f"No - {matching_findings[0]}"
                        elif matching_findings and "left ventricular hypertrophy" in str(matching_findings[0]).lower():
                            return f"No - {matching_findings[0]}"
                        return "No - no evidence of ischemia or infarction"
                    elif asked_condition == 'ectopic beat':
                        return "No - no ectopic beats present"
                    elif asked_condition == 't_waves':
                        return "Yes - T waves are normal"
                    elif asked_condition == 'sinus_rhythm':
                        return "Unable to determine rhythm"
                    else:
                        return f"No - no evidence of {asked_condition.replace('_', ' ')}"
            else:
                # General category question
                if mapped_category and mapped_category in active_findings:
                    findings_raw = list(active_findings[mapped_category])

                    # If the question is specifically about atria, filter to atrial-only findings
                    is_atrial_question = any(t in prompt_text for t in ['atrial', 'atrium'])
                    if mapped_category == 'CHAMBER ENLARGEMENT' and is_atrial_question:
                        atrial_only = []
                        for f in findings_raw:
                            fl = str(f).lower()
                            if ('atrial' in fl) or ('bi-atrial' in fl) or ('biatrial' in fl) or (fl in ['lae', 'rae']):
                                atrial_only.append(self.format_finding_name(f))
                        if atrial_only:
                            return f"Yes - {'; '.join(atrial_only)}"
                        else:
                            # Tailor negative phrasing to the question
                            if asking_for_abnormalities:
                                return "No - no atrial abnormality"
                            else:
                                return "No - no atrial enlargement"

                    findings = [self.format_finding_name(f) for f in findings_raw]
                    
                    # If asking about abnormalities, filter out normal findings
                    if asking_for_abnormalities:
                        # Special atrial-only narrowing when question text targets atria
                        is_atrial_question = any(t in prompt_text for t in ['atrial', 'atrium'])
                        if mapped_category == 'CHAMBER ENLARGEMENT' and is_atrial_question:
                            atrial_abnormal = []
                            for f in active_findings[mapped_category]:
                                fl = str(f).lower()
                                if ('atrial' in fl) or ('bi-atrial' in fl) or ('biatrial' in fl) or (fl in ['lae', 'rae']):
                                    formatted = self.format_finding_name(f)
                                    # exclude any that might be normal wording (safety)
                                    if not any(normal.lower() in formatted.lower() for normal in normal_findings):
                                        atrial_abnormal.append(formatted)
                            if atrial_abnormal:
                                return f"Yes - {'; '.join(atrial_abnormal)}"
                            else:
                                return "No - no atrial abnormality"

                        # Filter based on the original column names, not formatted names
                        abnormal_findings = []
                        for finding in active_findings[mapped_category]:
                            # Check if this is a normal finding based on column name
                            if finding not in normal_column_names:
                                formatted = self.format_finding_name(finding)
                                # Double-check the formatted name isn't a normal finding
                                if not any(normal.lower() in formatted.lower() for normal in normal_findings):
                                    abnormal_findings.append(formatted)
                        
                        if abnormal_findings:
                            return f"Yes - {'; '.join(abnormal_findings)}"
                        else:
                            # Only normal findings present
                            return self._get_negative_response_for_abnormalities(mapped_category)
                    else:
                        # Special case: if asking about ectopic beats in general category question
                        if 'ectopic' in prompt_text:
                            # Only include PVCs and PACs from the findings
                            ectopic_findings = []
                            for finding in active_findings[mapped_category]:
                                if any(term in finding.lower() for term in ['premature ventricular complex', 'premature atrial complex']):
                                    ectopic_findings.append(self.format_finding_name(finding))
                            if ectopic_findings:
                                return f"Yes - {'; '.join(ectopic_findings)}"
                            else:
                                return "No - no ectopic beats present"
                        else:
                            # Not specifically asking about abnormalities or ectopic beats
                            # If question targets atria, restrict to atrial-only findings
                            is_atrial_question = any(t in prompt_text for t in ['atrial', 'atrium'])
                            if mapped_category == 'CHAMBER ENLARGEMENT' and is_atrial_question:
                                atrial_only = []
                                for f in active_findings[mapped_category]:
                                    fl = str(f).lower()
                                    if ('atrial' in fl) or ('bi-atrial' in fl) or ('biatrial' in fl) or (fl in ['lae', 'rae']):
                                        atrial_only.append(self.format_finding_name(f))
                                if atrial_only:
                                    return f"Yes - {'; '.join(atrial_only)}"
                                else:
                                    return "No - no atrial enlargement"

                            if include_hr and mapped_category == 'RHYTHM' and heart_rate and heart_rate > 0 and is_rhythm_question:
                                findings_with_hr = []
                                hr_added = False
                                for f in findings:
                                    if not hr_added and 'rhythm' in f.lower():
                                        findings_with_hr.append(f + f" (HR: {heart_rate} bpm)")
                                        hr_added = True
                                    else:
                                        findings_with_hr.append(f)
                                response = f"Yes - {'; '.join(findings_with_hr)}"
                            else:
                                response = f"Yes - {'; '.join(findings)}"
                            return response
                else:
                    is_atrial_question = any(t in prompt_text for t in ['atrial', 'atrium'])
                    if mapped_category == 'CHAMBER ENLARGEMENT' and is_atrial_question:
                        if asking_for_abnormalities:
                            return "No - no atrial abnormality"
                        return "No - no atrial enlargement"
                    response = self._get_negative_response(mapped_category)
                    if include_hr and mapped_category == 'RHYTHM' and heart_rate and heart_rate > 0 and is_rhythm_question:
                        response = response.replace("normal sinus rhythm", f"normal sinus rhythm (HR: {heart_rate} bpm)")
                    return response

        else:
            if mapped_category and mapped_category in active_findings:
                findings = [self.format_finding_name(f) for f in active_findings[mapped_category]]
                if include_hr and mapped_category == 'RHYTHM' and heart_rate and heart_rate > 0 and is_rhythm_question:
                    findings_with_hr = []
                    hr_added = False
                    for f in findings:
                        if not hr_added and 'rhythm' in f.lower():
                            findings_with_hr.append(f + f" (HR: {heart_rate} bpm)")
                            hr_added = True
                        else:
                            findings_with_hr.append(f)
                    response = "; ".join(findings_with_hr)
                else:
                    response = "; ".join(findings)
                return response
            response = self._get_negative_response(mapped_category)
            if include_hr and mapped_category == 'RHYTHM' and heart_rate and heart_rate > 0 and is_rhythm_question:
                response = response.replace("normal sinus rhythm", f"normal sinus rhythm (HR: {heart_rate} bpm)")
            return response
    
    def _get_negative_response(self, category: str) -> str:
        """Get appropriate negative response for a category"""
        if category == 'RHYTHM':
            return "No - normal sinus rhythm"
        elif category == 'CONDUCTION':
            return "No - normal conduction"
        elif category == 'INFARCT, ISCHEMIA':
            return "No - no evidence of ischemia or infarction"
        elif category == 'CHAMBER ENLARGEMENT':
            return "No - no chamber enlargement"
        elif category == 'PERICARDITIS':
            return "No - no evidence of pericarditis"
        else:
            return "No - no significant findings"
    
    def _get_negative_response_for_abnormalities(self, category: str) -> str:
        """Get appropriate negative response when asking specifically about abnormalities"""
        if category == 'RHYTHM':
            return "No - rhythm is normal (sinus rhythm)"
        elif category == 'CONDUCTION':
            return "No - conduction is normal"
        elif category == 'INFARCT, ISCHEMIA':
            return "No - no ischemic abnormalities"
        elif category == 'CHAMBER ENLARGEMENT':
            return "No - no abnormal chamber enlargement"
        elif category == 'PERICARDITIS':
            return "No - no pericarditic abnormalities"
        else:
            return "No - no abnormalities in this category"
    
    def _extract_pathological_descriptions(
        self,
        active_findings: Dict[str, List[str]]
    ) -> Tuple[List[str], List[str]]:
        """Return formatted lists of pathological and limit findings present."""
        pathological: List[str] = []
        limit: List[str] = []
        seen_pathological: Set[str] = set()
        seen_limit: Set[str] = set()

        for findings in active_findings.values():
            for finding in findings:
                if finding in self.pathological_labels and finding not in seen_pathological:
                    pathological.append(self.format_finding_name(finding))
                    seen_pathological.add(finding)
                elif finding in self.limit_labels and finding not in seen_limit:
                    limit.append(self.format_finding_name(finding))
                    seen_limit.add(finding)

        return pathological, limit

    @staticmethod
    def _combine_with_details(base: str, details: str) -> str:
        """Append detail text to base classification string when available."""
        return f"{base}; {details}" if details else base
    
    def generate_classification_answer(self, row: pd.Series) -> str:
        """
        Generate classification answer (normal/borderline/pathological).
        Properly handles YES/NO questions about ECG normalcy.
        ALWAYS returns a proper classification - never "Unable to classify".
        """
        prompt_text = row.get('prompt', '').lower()
        ecg_type = row.get('ecg_type', 'unknown')
        active_findings = self.get_active_findings(row)
        pathological_descriptions, limit_descriptions = self._extract_pathological_descriptions(active_findings)
        
        # If ecg_type is unknown, determine from active findings
        if ecg_type == 'unknown' or pd.isna(ecg_type):
            # Determine classification based on findings
            if not active_findings:
                ecg_type = 'normal'
            else:
                # Check for critical findings
                critical_conditions = [
                    'Acute MI', 'ST elevation', 'Complete heart block', 'Third Degree AV Block',
                    'Ventricular tachycardia', 'Ventricular fibrillation', 'Mobitz II',
                    'Atrial fibrillation', 'LBBB', 'RBBB'
                ]
                
                has_critical = False
                normal_column_names = ['Sinusal', 'Regular', 'Monomorph']
                significant_findings = []
                
                for category, findings in active_findings.items():
                    for finding in findings:
                        # Skip normal findings
                        if finding not in normal_column_names:
                            significant_findings.append(finding)
                            if any(crit in finding for crit in critical_conditions):
                                has_critical = True
                
                if has_critical or len(significant_findings) >= 2:
                    ecg_type = 'pathological'
                elif len(significant_findings) == 1:
                    ecg_type = 'borderline'
                else:
                    ecg_type = 'normal'
        
        # Check if asking about urgent action
        is_urgent_question = any(phrase in prompt_text for phrase in [
            'urgent action', 'immediate intervention', 'emergency', 
            'urgent', 'immediate'
        ])
        
        # Check if it's a YES/NO question about normalcy
        is_normal_question = any(phrase in prompt_text for phrase in [
            'is this ecg normal', 'is this a normal', 'within normal limits',
            'is it normal', 'ecg normal'
        ])
        
        is_abnormal_question = any(phrase in prompt_text for phrase in [
            'abnormal', 'patholog', 'concerning', 'wrong', 'problem'
        ])
        
        is_yes_no = any(phrase in prompt_text for phrase in [
            'is this', 'is there', 'are there', 'does', 'should'
        ])
        
        # Special phrasing for binary classification prompts
        is_binary_classify = ('normal or abnormal' in prompt_text) or ('classify this ecg' in prompt_text and 'normal' in prompt_text and 'abnormal' in prompt_text)

        if ecg_type == 'normal':
            if is_urgent_question:
                return "No - routine follow-up; normal ECG"
            elif is_normal_question:
                return "Yes - ECG is within normal limits"
            elif is_binary_classify:
                return "Normal - No significant abnormalities detected"
            elif is_abnormal_question:
                if 'wrong' in prompt_text or 'concerning' in prompt_text:
                    return "No - ECG is normal"
                else:
                    return "No - ECG is normal; No significant abnormalities"
            else:
                return "Normal ECG; No significant abnormalities detected"
        
        elif ecg_type == 'borderline':
            # List the borderline findings
            finding_str = ""
            if limit_descriptions:
                finding_str = ", ".join(limit_descriptions[:3])
            else:
                findings = []
                for category, items in active_findings.items():
                    findings.extend(items[:2])
                if findings:
                    finding_str = ", ".join([self.format_finding_name(f) for f in findings[:3]])
            
            if is_urgent_question:
                return "No - routine follow-up recommended; borderline findings only"
            elif is_normal_question:
                return f"No - Borderline ECG; Minor findings: {finding_str}" if finding_str else "No - Borderline ECG"
            elif is_binary_classify:
                # Treat borderline as abnormal for binary classification wording
                return f"Abnormal - Minor findings: {finding_str}" if finding_str else "Abnormal - Borderline ECG"
            elif is_abnormal_question:
                if 'wrong' in prompt_text:
                    return f"Yes - Borderline abnormalities: {finding_str}" if finding_str else "Yes - Borderline changes present"
                else:
                    return f"Borderline ECG; Minor findings: {finding_str}" if finding_str else "Borderline ECG"
            else:
                return f"Borderline ECG; Minor findings: {finding_str}" if finding_str else "Borderline ECG; Nonspecific changes"
        
        elif ecg_type == 'pathological':
            # List the most critical findings
            critical_findings = []
            
            # Prioritize critical categories but exclude normal findings
            normal_column_names = ['Sinusal', 'Regular', 'Monomorph']
            priority_categories = ['INFARCT, ISCHEMIA', 'RHYTHM', 'CONDUCTION']
            
            for cat in priority_categories:
                if cat in active_findings:
                    for finding in active_findings[cat][:2]:
                        if finding not in normal_column_names:
                            critical_findings.append(finding)
            
            finding_str = ""
            if critical_findings:
                finding_str = ", ".join([self.format_finding_name(f) for f in critical_findings[:3]])

            pathological_str = "; ".join(pathological_descriptions[:5])
            abnormal_details = ""
            if pathological_str:
                abnormal_details = f"Pathological findings: {pathological_str}"
            elif finding_str:
                abnormal_details = f"Significant findings: {finding_str}"
            
            # Check for truly urgent conditions
            urgent_conditions = [
                'Acute MI', 'ST elevation', 'Complete heart block', 'Third Degree AV Block',
                'Ventricular tachycardia', 'Ventricular fibrillation', 'Mobitz II'
            ]
            
            is_truly_urgent = False
            for category, findings in active_findings.items():
                for finding in findings:
                    if any(urgent in finding for urgent in urgent_conditions):
                        is_truly_urgent = True
                        break
                if is_truly_urgent:
                    break
            
            if is_urgent_question:
                # If the question asks about urgent action and there are pathological findings,
                # answer "Yes" to reflect abnormal/pathological status, reserving a stronger
                # "urgent intervention needed" only for truly emergent conditions.
                if is_truly_urgent:
                    return self._combine_with_details("Yes - urgent intervention needed", abnormal_details or finding_str)
                else:
                    return self._combine_with_details("Yes - there are pathological findings", abnormal_details or finding_str)
            elif is_normal_question or 'within normal limits' in prompt_text:
                return self._combine_with_details("No - Abnormal ECG", abnormal_details)
            elif is_binary_classify:
                # Binary classification phrasing without yes/no
                if abnormal_details or finding_str:
                    return f"Abnormal - {(abnormal_details or finding_str)}"
                else:
                    return "Abnormal - ECG shows significant abnormalities"
            elif 'wrong' in prompt_text:
                return self._combine_with_details("Yes - Abnormal ECG", abnormal_details)
            elif 'concerning' in prompt_text or 'require follow-up' in prompt_text:
                return self._combine_with_details("Yes - ECG shows significant abnormalities", abnormal_details)
            elif is_abnormal_question:
                return self._combine_with_details("Yes - Abnormal ECG", abnormal_details)
            else:
                return self._combine_with_details("Abnormal ECG", abnormal_details or finding_str or "")
        
        # This should never be reached since we now always determine classification
        # But adding safety fallback that returns pathological with any findings present
        if active_findings:
            significant_findings = []
            normal_column_names = ['Sinusal', 'Regular', 'Monomorph']
            for category, findings in active_findings.items():
                for finding in findings:
                    if finding not in normal_column_names:
                        significant_findings.append(finding)
            
            if significant_findings:
                finding_str = ", ".join([self.format_finding_name(f) for f in significant_findings[:3]])
                return f"Abnormal ECG; Significant findings: {finding_str}"
        
        return "Normal ECG; No significant abnormalities detected"
    
    def generate_urgency_answer(self, row: pd.Series) -> str:
        """
        Generate urgency assessment answer.
        For prompts asking about clinical urgency.
        Handles both MIMIC and MHI datasets.
        """
        # First check for MHI-specific ACS conditions
        is_mhi = False
        if 'dataset' in row.index and row.get('dataset') == 'mhi':
            is_mhi = True
        elif 'dataset_source' in row.index and row.get('dataset_source') == 'mhi':
            is_mhi = True
        
        if is_mhi:
            # Check ACS condition severity for MHI
            if 'acs_condition_severity' in row.index and pd.notna(row.get('acs_condition_severity')):
                acs_condition = str(row['acs_condition_severity'])
                
                # If ACS with acute occlusion - URGENT
                if 'Acute' in acs_condition and 'Occlusion' in acs_condition:
                    # Get culprit artery if available
                    culprit = ""
                    if 'acs_pci_regions' in row.index and pd.notna(row.get('acs_pci_regions')):
                        regions = str(row['acs_pci_regions'])
                        if 'IVA' in regions:
                            culprit = " - LAD occlusion"
                        elif 'CD' in regions:
                            culprit = " - RCA occlusion"
                        elif 'Cx' in regions:
                            culprit = " - Circumflex occlusion"
                    return f"URGENT: Immediate intervention required - Acute coronary occlusion{culprit}"
            
            # Check for Acute MI using _bert_model
            has_acute_mi = False
            if 'Acute_MI_bert_model' in row.index and pd.notna(row.get('Acute_MI_bert_model')):
                try:
                    if float(row['Acute_MI_bert_model']) > 0.5:
                        has_acute_mi = True
                except (ValueError, TypeError):
                    pass
            
            if has_acute_mi:
                # Check for ST elevation locations
                _, _, _, st_locations = self.check_st_segment_context(row)
                if st_locations:
                    location_str = f" ({', '.join(st_locations[:2])})"
                    return f"URGENT: Immediate intervention required - ST elevation{location_str}"
                else:
                    return "URGENT: Immediate intervention required - Acute MI"
            
            # If ST elevation present but no acute MI or ACS
            _, _, _, st_locations = self.check_st_segment_context(row)
            if st_locations:
                return f"Consider ACS - ST elevation present in {', '.join(st_locations)} but does not meet STEMI criteria"
        
        # Standard urgency assessment for all datasets
        active_findings = self.get_active_findings(row)
        
        # Check for critical findings requiring immediate attention
        urgent_conditions = [
            'Acute MI', 'ST elevation', 'Complete heart block', 
            'Ventricular tachycardia', 'Ventricular fibrillation',
            'Third-degree AV block', 'Mobitz II'
        ]
        
        critical_findings = []
        for category, findings in active_findings.items():
            for finding in findings:
                if any(urgent in finding for urgent in urgent_conditions):
                    critical_findings.append(finding)
        
        if critical_findings:
            # For MHI with ST elevation but no acute MI/ACS
            if is_mhi and 'ST elevation' in str(critical_findings):
                # Check if we already have acute MI or ACS
                has_acute_condition = False
                if 'Acute MI' in str(critical_findings):
                    has_acute_condition = True
                elif 'acs_condition_severity' in row.index:
                    acs = str(row.get('acs_condition_severity', ''))
                    if 'Acute' in acs:
                        has_acute_condition = True
                
                if not has_acute_condition:
                    # ST elevation present but no acute condition
                    _, _, _, st_locations = self.check_st_segment_context(row)
                    if st_locations:
                        return f"Consider ACS - ST elevation present in {', '.join(st_locations)} but does not meet STEMI criteria"
            
            return f"URGENT: Immediate intervention required - {', '.join(critical_findings[:2])}"
        
        # Check for conditions requiring prompt attention
        prompt_conditions = [
            'Atrial fibrillation', 'Atrial flutter', 'NSTEMI',
            'Second-degree AV block', 'Sustained VT'
        ]
        
        prompt_findings = []
        for category, findings in active_findings.items():
            for finding in findings:
                if any(prompt in finding for prompt in prompt_conditions):
                    prompt_findings.append(finding)
        
        if prompt_findings:
            return f"Prompt evaluation recommended - {', '.join(prompt_findings[:2])}"
        
        # Otherwise routine
        return "Routine follow-up; No urgent findings"
    
    def generate_qrs_axis_answer(self, row: pd.Series) -> str:
        """
        Generate answer for QRS axis questions.
        Determines axis based on leads I, II, and aVF patterns.
        
        Axis determination criteria:
        1. Normal axis: -30° to +90° (QRS positive in leads I and II)
        2. Left axis deviation (LAD): -30° to -90° (QRS positive in I, negative in II and aVF)
        3. Right axis deviation (RAD): +90° to +180° (QRS negative in I, positive in aVF)
        4. Extreme axis: -90° to -180° (QRS negative in both I and aVF)
        """
        prompt_text = row.get('prompt', '').lower()
        
        # Check for axis deviation columns
        axis_finding = None
        axis_degrees = None
        
        # First check if there's an explicit axis deviation column
        for col in row.index:
            try:
                if pd.notna(row[col]) and float(row[col]) >= 1:
                    col_str = str(col).lower()
                    
                    if 'left axis deviation' in col_str or 'lad' in col_str:
                        axis_finding = "Left axis deviation"
                        axis_degrees = "-30° to -90°"
                        break
                    elif 'right axis deviation' in col_str or 'rad' in col_str:
                        axis_finding = "Right axis deviation"
                        axis_degrees = "+90° to +180°"
                        break
                    elif 'extreme axis' in col_str or 'northwest axis' in col_str:
                        axis_finding = "Extreme axis deviation"
                        axis_degrees = "-90° to -180°"
                        break
                    elif 'normal axis' in col_str:
                        axis_finding = "Normal axis"
                        axis_degrees = "-30° to +90°"
                        break
            except (ValueError, TypeError):
                continue
        
        # If no explicit axis column found, check QRS complexes in leads I, II, aVF
        if not axis_finding:
            # Look for QRS pattern indicators
            qrs_lead_i_positive = False
            qrs_lead_i_negative = False
            qrs_lead_ii_positive = False
            qrs_lead_ii_negative = False
            qrs_avf_positive = False
            qrs_avf_negative = False
            
            for col in row.index:
                try:
                    if pd.notna(row[col]) and float(row[col]) >= 1:
                        col_str = str(col)
                        
                        # Check for QRS patterns in specific leads
                        if 'QRS' in col_str or 'R wave' in col_str or 'S wave' in col_str:
                            if 'lead I' in col_str or '(I)' in col_str:
                                if 'positive' in col_str or 'tall' in col_str:
                                    qrs_lead_i_positive = True
                                elif 'negative' in col_str or 'deep' in col_str:
                                    qrs_lead_i_negative = True
                            elif 'lead II' in col_str or '(II)' in col_str:
                                if 'positive' in col_str or 'tall' in col_str:
                                    qrs_lead_ii_positive = True
                                elif 'negative' in col_str or 'deep' in col_str:
                                    qrs_lead_ii_negative = True
                            elif 'aVF' in col_str:
                                if 'positive' in col_str or 'tall' in col_str:
                                    qrs_avf_positive = True
                                elif 'negative' in col_str or 'deep' in col_str:
                                    qrs_avf_negative = True
                except (ValueError, TypeError):
                    continue
            
            # Determine axis based on lead patterns
            if qrs_lead_i_positive and qrs_lead_ii_positive:
                axis_finding = "Normal axis"
                axis_degrees = "-30° to +90°"
            elif qrs_lead_i_positive and qrs_lead_ii_negative and qrs_avf_negative:
                axis_finding = "Left axis deviation"
                axis_degrees = "-30° to -90°"
            elif qrs_lead_i_negative and qrs_avf_positive:
                axis_finding = "Right axis deviation"
                axis_degrees = "+90° to +180°"
            elif qrs_lead_i_negative and qrs_avf_negative:
                axis_finding = "Extreme axis deviation"
                axis_degrees = "-90° to -180°"
            else:
                # Default to normal if no clear pattern
                axis_finding = "Normal axis"
                axis_degrees = "-30° to +90°"
        
        # Check if this is a YES/NO question
        is_yes_no = any(phrase in prompt_text for phrase in [
            'is there', 'are there', 'does', 'is the'
        ])
        
        # Generate response
        if is_yes_no:
            if 'deviation' in prompt_text:
                if axis_finding != "Normal axis":
                    return f"Yes - {axis_finding} ({axis_degrees})"
                else:
                    return f"No - {axis_finding} ({axis_degrees})"
            elif 'normal' in prompt_text:
                if axis_finding == "Normal axis":
                    return f"Yes - {axis_finding} ({axis_degrees})"
                else:
                    return f"No - {axis_finding} ({axis_degrees})"
            else:
                return f"{axis_finding} ({axis_degrees})"
        else:
            # For "What is the QRS axis?" type questions
            return f"{axis_finding} ({axis_degrees})"
    
    def generate_localization_answer(self, row: pd.Series, finding_type: str) -> str:
        """
        Generate answer for localization questions about Q waves, ST changes, T waves, and QRS axis.
        """
        prompt_text = row.get('prompt', '').lower()
        findings = []
        
        # Handle QRS axis specially
        if finding_type == 'qrs_axis':
            return self.generate_qrs_axis_answer(row)
        
        # For ST elevation questions, check context
        if finding_type == 'st_elevation':
            has_acute_mi, has_early_repol, has_lvh, st_locations = self.check_st_segment_context(row)
            
            if st_locations:
                if not has_acute_mi:
                    if has_early_repol:
                        return f"Yes - ST elevation due to early repolarization (benign) in {', '.join(st_locations)}"
                    elif has_lvh:
                        return f"Yes - ST elevation associated with left ventricular hypertrophy in {', '.join(st_locations)}"
        
        # Map finding types to column patterns
        pattern_map = {
            'q_wave': 'Q wave',
            'st_elevation': 'ST elevation',
            'st_depression': 'ST depression',
            't_wave': 'T wave',
            'st_morphology': ['ST downsloping', 'ST upslopping', 'Early repolarization']
        }
        
        # Check for the specific finding type
        if finding_type in pattern_map:
            pattern = pattern_map[finding_type]
            patterns = [pattern] if isinstance(pattern, str) else pattern
            
            for col in row.index:
                try:
                    if pd.notna(row[col]) and float(row[col]) >= 1:
                        col_str = str(col)
                        
                        # Check if column matches our pattern
                        for p in patterns:
                            if p in col_str:
                                # Extract location from parentheses
                                if '(' in col_str and ')' in col_str:
                                    location = col_str[col_str.find('(')+1:col_str.find(')')]
                                    findings.append(location)
                                elif finding_type == 'st_morphology':
                                    # Handle morphology descriptions
                                    if col_str == 'ST downsloping':
                                        findings.append('diffuse ST downsloping')
                                    elif col_str == 'ST upslopping':
                                        findings.append('ST upsloping')
                                    elif col_str == 'Early repolarization':
                                        findings.append('diffuse ST upsloping consistent with early repolarization')
                except (ValueError, TypeError):
                    continue
        
        # Check if this is a YES/NO question
        is_yes_no = any(phrase in prompt_text for phrase in [
            'is there', 'are there', 'does', 'can you identify', 'evidence'
        ])
        
        # Generate response
        if findings:
            if finding_type == 'st_morphology':
                # Special formatting for morphology
                return f"Yes - {'; '.join(set(findings))}"
            else:
                # Format locations nicely
                formatted_findings = []
                for location in set(findings):  # Remove duplicates
                    # Clean up location string with better formatting
                    clean_loc = location
                    
                    # Handle different location formats
                    if ' - ' in clean_loc:
                        # Format like "inferior - II, III, aVF" -> "inferior leads II, III, aVF"
                        parts = clean_loc.split(' - ')
                        if len(parts) == 2:
                            region = parts[0].strip()
                            leads = parts[1].strip()
                            # Add spaces after commas if missing
                            leads = ', '.join([l.strip() for l in leads.split(',')])
                            clean_loc = f"{region} leads {leads}"
                    elif '-' in clean_loc and not clean_loc.startswith('V'):
                        # Format like "septal- V1-V2" -> "septal leads V1, V2"
                        parts = clean_loc.split(' - ')
                        if len(parts) == 2:
                            # Space separator case
                            region = parts[0].strip()
                            leads_str = parts[1].strip()
                            # Split V1-V2 into V1, V2 and V3-V4 into V3, V4
                            import re
                            # First split on hyphens between V leads
                            leads_str = re.sub(r'(V\d+)-(V\d+)', r'\1, \2', leads_str)
                            # Then extract all lead names
                            leads = re.findall(r'V\d+|I{1,3}|aV[RLF]', leads_str)
                            if leads:
                                clean_loc = f"{region} leads {', '.join(leads)}"
                        else:
                            # No space separator - try direct split
                            parts = clean_loc.split('-')
                            if len(parts) >= 2:
                                region = parts[0].strip()
                                # Extract lead names from remaining parts
                                lead_parts = parts[1:]
                                leads = []
                                for part in lead_parts:
                                    # Handle cases like "V1", "V2", or "V3-V4"
                                    if 'V' in part or 'aV' in part or part in ['I', 'II', 'III']:
                                        leads.append(part)
                                if leads:
                                    clean_loc = f"{region} leads {', '.join(leads)}"
                    else:
                        # Try to clean up any other format
                        # Replace "V1V2" with "V1, V2"
                        # Add spaces between lead names
                        clean_loc = LEAD_NAME_PATTERN.sub(r'\1, ', clean_loc)
                        # Clean up double spaces
                        clean_loc = ' '.join(clean_loc.split())
                    
                    formatted_findings.append(clean_loc)
                
                if is_yes_no:
                    return f"Yes - {finding_type.replace('_', ' ')} in {'; '.join(formatted_findings)}"
                else:
                    # For "Where" questions
                    return f"{finding_type.replace('_', ' ').title()} present in: {'; '.join(formatted_findings)}"
        else:
            # No findings
            if is_yes_no:
                return f"No - no {finding_type.replace('_', ' ')}"
            else:
                return f"No {finding_type.replace('_', ' ')} identified"
    
    def generate_json_interpretation(self, row: pd.Series) -> str:
        """
        Generate JSON structured output for ECG findings.
        Returns a JSON string with only the findings that are present.
        """
        import json
        
        # Initialize the JSON structure - only include what's present
        json_output = {}
        
        # Allow all category keys including OTHER for full visibility
        allowed_keys = {
            'RHYTHM',
            'CONDUCTION',
            'CHAMBER_ENLARGEMENT',
            'INFARCT_ISCHEMIA',
            'PERICARDITIS',
            'OTHER',
        }
        
        # Go through each category and condition
        for category, conditions in self.categories_dict.items():
            # Map category names to JSON keys
            json_category = category.replace(", ", "_").replace(" ", "_").upper()
            # Skip any category not in the explicit schema (e.g., OTHER)
            if json_category not in allowed_keys:
                continue
            
            present_findings = []
            
            for condition in conditions:
                # Check various column name formats
                col_names = [
                    condition,
                    condition.replace(' ', '_'),
                    condition.lower().replace(' ', '_')
                ]
                
                for col in col_names:
                    if col in row.index:
                        try:
                            if pd.notna(row[col]) and float(row[col]) >= 1:
                                # Keep original condition name for readability
                                present_findings.append(condition)
                                break
                        except (ValueError, TypeError):
                            continue
            
            if present_findings:
                json_output[json_category] = present_findings

        include_hr = self.qa_feature_flags is None or self.qa_feature_flags.has_heart_rate
        if include_hr:
            heart_rate = self.calculate_heart_rate(row)
            if heart_rate and heart_rate > 0:
                json_output["heart_rate_bpm"] = int(round(heart_rate))
            elif heart_rate == 0:
                json_output["heart_rate_bpm"] = "artifacts"

        # Add ECG classification
        ecg_type = row.get('ecg_type', 'unknown')
        json_output["ecg_classification"] = ecg_type
        
        # Return as compact JSON string
        return json.dumps(json_output, separators=(',', ':'))
    
    def generate_heart_rate_answer(self, row: pd.Series) -> str:
        """
        Generate answer for heart rate questions.
        This is now its own category separate from demographics.
        Handles special case where HR=0 indicates artifacts in MHI dataset.
        """
        heart_rate = self.calculate_heart_rate(row)
        if heart_rate == 0:
            # Special case: 0 indicates artifacts in MHI dataset
            return "Sorry, I can't determine the heart rate - the ECG has artifacts"
        elif heart_rate:
            return f"{int(round(heart_rate))} bpm"
        return "Heart rate cannot be determined"
    
    def calculate_intervals(self, row: pd.Series) -> Dict[str, Optional[float]]:
        """
        Calculate PR and QT intervals from available data.
        Returns dict with pr_interval, qt_interval, qtc_bazett, qtc_fridericia in milliseconds.
        """
        intervals = {
            'pr_interval': None,
            'qt_interval': None,
            'qtc_bazett': None,
            'qtc_fridericia': None,
            'is_prolonged': False,
            'gender': None
        }
        
        # Get gender for QTc prolongation assessment
        gender = row.get('gender', None)
        if gender:
            gender_str = str(gender).strip().lower()
            if gender_str in ['m', 'male', '1', '1.0']:
                intervals['gender'] = 'male'
            elif gender_str in ['f', 'female', '0', '0.0']:
                intervals['gender'] = 'female'
        
        # For MHI dataset - check RestingECG columns
        if 'RestingECG_OriginalRestingECGMeasurements_PRInterval' in row.index:
            try:
                pr_val = row['RestingECG_OriginalRestingECGMeasurements_PRInterval']
                if pd.notna(pr_val):
                    intervals['pr_interval'] = float(pr_val)
            except (ValueError, TypeError):
                pass
        
        if 'RestingECG_OriginalRestingECGMeasurements_QTInterval' in row.index:
            try:
                qt_val = row['RestingECG_OriginalRestingECGMeasurements_QTInterval']
                if pd.notna(qt_val):
                    intervals['qt_interval'] = float(qt_val)
            except (ValueError, TypeError):
                pass
        
        if 'RestingECG_OriginalRestingECGMeasurements_QTCorrected' in row.index:
            try:
                qtc_val = row['RestingECG_OriginalRestingECGMeasurements_QTCorrected']
                if pd.notna(qtc_val):
                    intervals['qtc_bazett'] = float(qtc_val)
            except (ValueError, TypeError):
                pass
        
        if 'RestingECG_OriginalRestingECGMeasurements_QTcFrederica' in row.index:
            try:
                qtc_frid_val = row['RestingECG_OriginalRestingECGMeasurements_QTcFrederica']
                if pd.notna(qtc_frid_val):
                    intervals['qtc_fridericia'] = float(qtc_frid_val)
            except (ValueError, TypeError):
                pass
        
        # For MIMIC dataset - calculate from onset/offset if available
        # Note: These columns might not exist in the current dataset structure
        if intervals['pr_interval'] is None:
            if 'p_onset' in row.index and 'qrs_onset' in row.index:
                try:
                    p_onset = float(row['p_onset'])
                    qrs_onset = float(row['qrs_onset'])
                    if pd.notna(p_onset) and pd.notna(qrs_onset):
                        intervals['pr_interval'] = qrs_onset - p_onset
                except (ValueError, TypeError):
                    pass
        
        if intervals['qt_interval'] is None:
            if 'qrs_onset' in row.index and 't_end' in row.index:
                try:
                    qrs_onset = float(row['qrs_onset'])
                    t_end = float(row['t_end'])
                    if pd.notna(qrs_onset) and pd.notna(t_end):
                        intervals['qt_interval'] = t_end - qrs_onset
                except (ValueError, TypeError):
                    pass
        
        # Calculate QTc if we have QT interval and RR interval
        if intervals['qt_interval'] and ('rr_interval' in row.index or 'RestingECG_QRSTimesTypes_GlobalRR' in row.index):
            rr_interval = None
            
            # Try to get RR interval
            if 'rr_interval' in row.index and pd.notna(row['rr_interval']):
                try:
                    rr_interval = float(row['rr_interval'])
                except (ValueError, TypeError):
                    pass
            elif 'RestingECG_QRSTimesTypes_GlobalRR' in row.index:
                try:
                    rr_interval = float(row['RestingECG_QRSTimesTypes_GlobalRR'])
                except (ValueError, TypeError):
                    pass
            
            if rr_interval and rr_interval > 0:
                # Convert to seconds for calculation
                qt_sec = intervals['qt_interval'] / 1000.0
                rr_sec = rr_interval / 1000.0
                
                # Bazett's formula: QTc = QT / sqrt(RR)
                if intervals['qtc_bazett'] is None:
                    qtc_bazett_sec = qt_sec / np.sqrt(rr_sec)
                    intervals['qtc_bazett'] = qtc_bazett_sec * 1000  # Convert back to ms
                
                # Fridericia's formula: QTc = QT / (RR^(1/3))
                if intervals['qtc_fridericia'] is None:
                    qtc_fridericia_sec = qt_sec / (rr_sec ** (1/3))
                    intervals['qtc_fridericia'] = qtc_fridericia_sec * 1000  # Convert back to ms
        
        # Determine if QTc is prolonged based on gender
        qtc_to_check = intervals['qtc_bazett'] or intervals['qtc_fridericia']
        if qtc_to_check:
            if intervals['gender'] == 'male':
                intervals['is_prolonged'] = qtc_to_check > 440
            elif intervals['gender'] == 'female':
                intervals['is_prolonged'] = qtc_to_check > 460
            else:
                # If gender unknown, use conservative threshold
                intervals['is_prolonged'] = qtc_to_check > 440
        
        return intervals
    
    def generate_intervals_answer(self, row: pd.Series) -> str:
        """
        Generate answer for interval questions (PR, QT, QTc).
        """
        prompt_text = row.get('prompt', '').lower()
        intervals = self.calculate_intervals(row)
        heart_rate = self.calculate_heart_rate(row)
        
        # Determine which interval(s) are being asked about
        asking_pr = 'pr' in prompt_text
        asking_qt = 'qt' in prompt_text and 'qtc' not in prompt_text
        asking_qtc = 'qtc' in prompt_text or 'corrected' in prompt_text
        asking_prolonged = 'prolonged' in prompt_text or 'prolongation' in prompt_text
        asking_bazett = 'bazett' in prompt_text
        asking_fridericia = 'fridericia' in prompt_text or 'fredericia' in prompt_text
        
        response_parts = []
        
        # PR interval
        if asking_pr or ('interval' in prompt_text and not asking_qt and not asking_qtc):
            if intervals['pr_interval']:
                pr_ms = int(round(intervals['pr_interval']))
                response_parts.append(f"PR interval: {pr_ms} ms")
                # Normal PR is 120-200 ms
                if pr_ms < 120:
                    response_parts[-1] += " (short)"
                elif pr_ms > 200:
                    response_parts[-1] += " (prolonged - first degree AV block)"
            else:
                if asking_pr and not (asking_qt or asking_qtc):
                    return "PR interval cannot be determined"
        
        # QT interval
        if asking_qt and not asking_qtc:
            if intervals['qt_interval']:
                qt_ms = int(round(intervals['qt_interval']))
                response_parts.append(f"QT interval: {qt_ms} ms")
            else:
                if not asking_pr and not asking_qtc:
                    return "QT interval cannot be determined"
        
        # QTc interval
        if asking_qtc or asking_prolonged:
            qtc_value = None
            formula_used = ""
            
            if asking_fridericia and intervals['qtc_fridericia']:
                qtc_value = intervals['qtc_fridericia']
                formula_used = "Fridericia"
            elif asking_bazett and intervals['qtc_bazett']:
                qtc_value = intervals['qtc_bazett']
                formula_used = "Bazett"
            elif intervals['qtc_fridericia']:
                # Prefer Fridericia for extreme heart rates
                if heart_rate and heart_rate > 0 and (heart_rate < 60 or heart_rate > 100):
                    qtc_value = intervals['qtc_fridericia']
                    formula_used = "Fridericia"
                elif intervals['qtc_bazett']:
                    qtc_value = intervals['qtc_bazett']
                    formula_used = "Bazett"
            elif intervals['qtc_bazett']:
                qtc_value = intervals['qtc_bazett']
                formula_used = "Bazett"
            
            if qtc_value:
                qtc_ms = int(round(qtc_value))
                
                if asking_prolonged:
                    if intervals['is_prolonged']:
                        hr_text = f" (at HR {int(round(heart_rate))} bpm)" if heart_rate and heart_rate > 0 else ""
                        response_parts.append(f"QTc is {qtc_ms} ms{hr_text}; QT prolongation is present")
                    else:
                        response_parts.append(f"QTc is {qtc_ms} ms; No QT prolongation")
                else:
                    qtc_text = f"QTc ({formula_used}): {qtc_ms} ms"
                    if heart_rate and heart_rate > 0:
                        qtc_text += f" (at HR {int(round(heart_rate))} bpm)"
                    if intervals['is_prolonged']:
                        qtc_text += " - prolonged"
                    response_parts.append(qtc_text)
            else:
                if not asking_pr and not asking_qt:
                    return "QTc cannot be calculated - insufficient data"
        
        if response_parts:
            return "; ".join(response_parts)
        
        return "Interval measurements not available"
    
    def generate_demographic_answer(self, row: pd.Series, demo_type: str) -> str:
        """
        Generate answer for demographic questions (gender, age, heart rate).
        """
        prompt_text = row.get('prompt', '').lower()
        
        if 'gender' in demo_type:
            gender = self.column_mapper.get_value(row, 'gender', None)
            if gender is not None:
                gender_str = str(gender).strip().lower()
                if gender_str in ['m', 'male', '1', '1.0']:
                    return "Male"
                elif gender_str in ['f', 'female', '0', '0.0']:
                    return "Female"
                else:
                    return f"Gender: {gender}"
            return "Gender information not available"
        
        elif 'age' in demo_type:
            age = self.column_mapper.get_value(row, 'age', None)
            if age is not None:
                try:
                    age_val = float(age)
                    return f"{int(age_val)} years old"
                except (ValueError, TypeError):
                    return f"Age: {age}"
            return "Age information not available"
        
        elif 'heart_rate' in demo_type:
            heart_rate = self.calculate_heart_rate(row)
            if heart_rate == 0:
                return "Sorry, I can't determine the heart rate - the ECG has artifacts"
            elif heart_rate:
                return f"{heart_rate} bpm"
            return "Heart rate cannot be determined"
        
        return "Information not available"
    
    def generate_random_finding_answer(self, row: pd.Series) -> str:
        """
        Generate YES/NO answer for random finding questions.
        These questions ask about specific ECG findings like "Is there 1st degree AV block?"
        Answer YES if the corresponding column value >= 1, NO otherwise.
        """
        prompt_text = row.get('prompt', '')
        
        # Extract the finding being asked about from the prompt
        # Questions are like "Is there X?" where X is the finding
        finding_name = None
        
        # Remove common prefixes to extract the finding
        question_lower = prompt_text.lower()
        if 'is there' in question_lower:
            finding_name = prompt_text.split('Is there')[1].strip('? ')
        elif 'are there' in question_lower:
            finding_name = prompt_text.split('Are there')[1].strip('? ')
        elif 'does this show' in question_lower:
            finding_name = prompt_text.split('show')[1].strip('? ')
        
        if not finding_name:
            return "Unable to determine finding from question"
        
        # Check columns for this finding
        # Try different column name variations
        finding_lower = finding_name.lower()
        found = False
        
        # Special mappings for common findings - using EXACT column names from deepecg_categories.json
        column_mappings = {
            '1st degree av block': ['1st degree AV block'],
            '2nd degree av block': ['2nd degree AV block - mobitz 1', '2nd degree AV block - mobitz 2'],
            '3rd degree av block': ['Third Degree AV Block'],
            'left bundle branch block': ['Left bundle branch block'],
            'right bundle branch block': ['Right bundle branch block'],
            'left anterior fascicular block': ['Left anterior fascicular block'],
            'left posterior fascicular block': ['Left posterior fascicular block'],
            'left ventricular hypertrophy': ['Left ventricular hypertrophy'],
            'right ventricular hypertrophy': ['Right ventricular hypertrophy'],
            'left atrial enlargement': ['Left atrial enlargement'],
            'right atrial enlargement': ['Right atrial enlargement'],
            'atrial fibrillation': ['Afib'],
            'atrial flutter': ['Atrial flutter'],
            'sinus bradycardia': ['Bradycardia'],
            'sinus tachycardia': ['Atrial tachycardia (>= 100 BPM)'],
            'ventricular tachycardia': ['Ventricular tachycardia'],
            'premature ventricular complex': ['Premature ventricular complex'],
            'premature atrial complex': ['Premature atrial complex'],
            'st elevation in any leads': ['ST elevation'],  # Will check all ST elevation columns
            'st depression in any leads': ['ST depression'],  # Will check all ST depression columns
            'pathological q waves': ['Q wave'],  # Will check all Q wave columns
            't wave inversion': ['T wave inversion'],  # Will check all T wave inversion columns
            'left axis deviation': ['Left axis deviation'],
            'right axis deviation': ['Right axis deviation'],
            'extreme axis deviation': ['Right superior axis'],  # From CONDUCTION category
            'pericarditis': ['Acute pericarditis'],
            'early repolarization': ['Early repolarization'],
            'wpw pattern': ['Wolff-Parkinson-White (Pre-excitation syndrome)', 'Delta wave'],
            'long qt syndrome': ['Prolonged QT'],
            'short qt syndrome': []  # Not in the categories
        }
        
        # Check if we have a specific mapping for this finding
        columns_to_check = []
        for key, cols in column_mappings.items():
            if key in finding_lower:
                columns_to_check = cols
                break
        
        # If no specific mapping, try the finding name directly
        if not columns_to_check:
            columns_to_check = [finding_name, finding_name.replace(' ', '_'), finding_lower.replace(' ', '_')]
        
        # Special handling for "any leads" questions
        if 'st elevation' in finding_lower and 'any leads' in finding_lower:
            # Check all ST elevation columns
            for col in row.index:
                if 'ST elevation' in str(col):
                    try:
                        if pd.notna(row[col]) and float(row[col]) >= 1:
                            return "Yes"
                    except (ValueError, TypeError):
                        continue
            return "No"
        
        elif 'st depression' in finding_lower and 'any leads' in finding_lower:
            # Check all ST depression columns
            for col in row.index:
                if 'ST depression' in str(col):
                    try:
                        if pd.notna(row[col]) and float(row[col]) >= 1:
                            return "Yes"
                    except (ValueError, TypeError):
                        continue
            return "No"
        
        elif 'pathological q waves' in finding_lower or 'q waves' in finding_lower:
            # Check all Q wave columns
            for col in row.index:
                if 'Q wave' in str(col):
                    try:
                        if pd.notna(row[col]) and float(row[col]) >= 1:
                            return "Yes"
                    except (ValueError, TypeError):
                        continue
            return "No"
        
        elif 't wave inversion' in finding_lower:
            # Check all T wave inversion columns
            for col in row.index:
                if 'T wave inversion' in str(col):
                    try:
                        if pd.notna(row[col]) and float(row[col]) >= 1:
                            return "Yes"
                    except (ValueError, TypeError):
                        continue
            return "No"
        
        # Check the specific columns
        for col_name in columns_to_check:
            if col_name in row.index:
                try:
                    if pd.notna(row[col_name]) and float(row[col_name]) >= 1:
                        return "Yes"
                except (ValueError, TypeError):
                    continue
        
        # If not found in exact matches, do a broader search
        for col in row.index:
            col_str = str(col)
            for check_col in columns_to_check:
                if check_col.lower() in col_str.lower():
                    try:
                        if pd.notna(row[col]) and float(row[col]) >= 1:
                            return "Yes"
                    except (ValueError, TypeError):
                        continue
        
        return "No"
    
    def check_for_acute_mi_prefix(self, row: pd.Series) -> str:
        """
        Check if report indicates acute MI/STEMI and return appropriate prefix.
        This will be prepended to ALL answers when acute MI is present.
        EXCLUDES pericarditis cases which can also have diffuse ST elevation.
        """
        # First check if this is pericarditis - if so, don't flag as acute MI
        has_pericarditis = False
        
        # Check pericarditis column
        if 'Acute pericarditis' in row.index and pd.notna(row.get('Acute pericarditis')):
            try:
                if float(row['Acute pericarditis']) >= 1:
                    has_pericarditis = True
            except (ValueError, TypeError):
                pass
        
        # Check report for pericarditis
        if 'report' in row.index and pd.notna(row['report']):
            report_lower = str(row['report']).lower()
            if 'pericarditis' in report_lower:
                has_pericarditis = True
        
        # If pericarditis is present, don't flag as acute MI
        if has_pericarditis:
            return ""
        
        # Now check report for critical acute MI phrases
        if 'report' in row.index and pd.notna(row['report']):
            report_upper = str(row['report']).upper()
            
            # Check for critical STEMI indicators
            if 'CONSIDER ACUTE ST ELEVATION MI' in report_upper:
                # Extract ST elevation location from report
                st_locations = []
                report_lower = str(row['report']).lower()
                
                if 'lateral st elevation' in report_lower or 'lateral st-elevation' in report_lower:
                    st_locations.append('lateral')
                if 'inferior st elevation' in report_lower or 'inferior st-elevation' in report_lower:
                    st_locations.append('inferior')
                if 'anterior st elevation' in report_lower or 'anterior st-elevation' in report_lower:
                    st_locations.append('anterior')
                if 'anterolateral st elevation' in report_lower:
                    st_locations.append('anterolateral')
                
                # Also check ST elevation columns
                for col in row.index:
                    if 'ST elevation' in col:
                        try:
                            if pd.notna(row[col]) and float(row[col]) >= 1:
                                if '(' in col and ')' in col:
                                    location = col[col.find('(')+1:col.find(')')]
                                    # Check if this location isn't already in the list
                                    location_lower = location.lower()
                                    already_exists = any(loc.lower() in location_lower or location_lower in loc.lower() 
                                                       for loc in st_locations)
                                    if not already_exists:
                                        st_locations.append(location)
                        except (ValueError, TypeError):
                            pass
                
                if st_locations:
                    # Remove duplicates and format locations
                    unique_locations = list(dict.fromkeys(st_locations))  # Preserve order, remove duplicates
                    location_str = ', '.join(unique_locations)
                    return f"*** ACUTE STEMI ({location_str}) *** - "
                else:
                    return "*** ACUTE STEMI *** - "
            
            elif 'CONSIDER ACUTE INFARCT' in report_upper:
                # Similar logic for acute infarct
                return "*** ACUTE MI/INFARCT *** - "
        
        # Also check Acute_MI column
        if 'Acute_MI' in row.index and pd.notna(row['Acute_MI']):
            try:
                if float(row['Acute_MI']) >= 1:
                    return "*** ACUTE MI *** - "
            except (ValueError, TypeError):
                pass
        
        if 'Acute MI' in row.index and pd.notna(row['Acute MI']):
            try:
                if float(row['Acute MI']) >= 1:
                    return "*** ACUTE MI *** - "
            except (ValueError, TypeError):
                pass
        
        return ""  # No acute MI prefix needed
    
    def generate_structural_heart_disease_answer(self, row: pd.Series) -> str:
        """
        Generate answer for structural heart disease questions (MHI dataset only).
        Uses echonext_shd column: >= 1 means present, < 1 means absent.
        Returns None if data is not available (which should drop the question).
        """
        # Check if echonext_shd column exists and has data
        if 'echonext_shd' not in row.index:
            return None  # This question should be dropped
        
        shd_value = row.get('echonext_shd')
        
        # If value is null/nan, we can't answer
        if pd.isna(shd_value):
            return None  # This question should be dropped
        
        try:
            # Convert to float for comparison
            shd_val = float(shd_value)
            
            # echonext_shd >= 1 means structural heart disease is present
            if shd_val >= 1:
                return "Yes - structural heart disease is present based on echocardiography"
            else:
                return "No - no structural heart disease detected on echocardiography"
                
        except (ValueError, TypeError):
            # Can't convert to number, can't answer
            return None  # This question should be dropped
    
    def generate_lvef_answer(self, row: pd.Series) -> str:
        """
        Generate answer for LVEF questions (MHI dataset only).
        Uses deepecho_Visually_Estimated_EF column.
        Returns None if data is not available (which should drop the question).
        """
        # Check if LVEF column exists and has data
        if 'deepecho_Visually_Estimated_EF' not in row.index:
            return None  # This question should be dropped
        
        lvef_value = row.get('deepecho_Visually_Estimated_EF')
        
        # If value is null/nan, we can't answer
        if pd.isna(lvef_value):
            return None  # This question should be dropped
        
        try:
            # Convert to float and round to nearest percentage
            lvef_val = float(lvef_value)
            lvef_rounded = round(lvef_val)
            
            # Categorize the LVEF
            if lvef_rounded >= 55:
                category = "normal"
            elif lvef_rounded >= 45:
                category = "mildly reduced"
            elif lvef_rounded >= 30:
                category = "moderately reduced"
            else:
                category = "severely reduced"
            
            return f"The left ventricular ejection fraction is {lvef_rounded}% ({category})"
                
        except (ValueError, TypeError):
            # Can't convert to number, can't answer
            return None  # This question should be dropped
    
    def generate_acs_severity_answer(self, row: pd.Series) -> str:
        """
        Generate answer for ACS severity questions (MHI dataset only).
        Checks if there is an acute coronary occlusion.
        Returns None if data is not available (which should drop the question).
        """
        # Check if ACS condition severity column exists and has data
        if 'acs_condition_severity' not in row.index:
            return None  # This question should be dropped
        
        acs_condition = row.get('acs_condition_severity')
        
        # If value is null/nan, we can't answer
        if pd.isna(acs_condition):
            return None  # This question should be dropped
        
        # Import ACS constants
        from utils.constants import ACS_ACUTE_CONDITIONS
        
        # Check if it's an acute occlusion and report its type
        if acs_condition in ACS_ACUTE_CONDITIONS:
            # Determine occlusion completeness for messaging
            if acs_condition == 'Acute Complete Coronary Occlusion':
                occlusion_type_text = 'complete occlusion'
            elif acs_condition == 'Acute Incomplete Coronary Occlusion':
                occlusion_type_text = 'incomplete occlusion'
            else:
                occlusion_type_text = None

            # Build culprit phrase if PCI regions are available
            culprit_phrase = None
            if 'acs_pci_regions' in row.index and pd.notna(row.get('acs_pci_regions')):
                import ast
                regions_raw = row.get('acs_pci_regions')
                try:
                    if isinstance(regions_raw, str):
                        regions_clean = regions_raw.strip("'\"")
                        region_list = ast.literal_eval(regions_clean)
                    else:
                        region_list = regions_raw

                    if region_list:
                        from utils.constants import ACS_ARTERY_MAPPING
                        primary_region = region_list[0] if isinstance(region_list, list) else str(region_list)
                        mapped_region = ACS_ARTERY_MAPPING.get(primary_region, primary_region)
                        if occlusion_type_text:
                            culprit_phrase = f"culprit is the {mapped_region} with {occlusion_type_text}"
                        else:
                            culprit_phrase = f"culprit is the {mapped_region}"
                except (ValueError, SyntaxError, TypeError):
                    pass

            # Compose final affirmative answer including culprit when available
            if culprit_phrase:
                return f"Yes - there is acute coronary occlusion; {culprit_phrase}"
            else:
                return "Yes - there is acute coronary occlusion; culprit artery is not documented"

        # Not an acute occlusion
        if acs_condition == 'No Coronary Disease':
            return "No - no evidence of coronary disease"
        if 'Obstructive' in acs_condition:
            return "No - obstructive coronary disease without acute occlusion"
        if 'Chronic' in acs_condition:
            return "No - chronic occlusion without acute findings"
        return "No - no acute coronary occlusion identified"
    
    def generate_culprit_artery_answer(self, row: pd.Series) -> str:
        """
        Generate answer for culprit artery questions (MHI dataset only).
        Only answers when there is an acute coronary occlusion.
        Returns None if data is not available or not an acute occlusion.
        """
        # First check if there is an acute occlusion
        if 'acs_condition_severity' not in row.index:
            return "Coronary occlusion severity data is not available for this patient"
        if 'acs_pci_regions' not in row.index:
            return "Culprit artery information is not available for this patient"

        acs_condition = row.get('acs_condition_severity')
        acs_regions = row.get('acs_pci_regions')

        # Check for null values
        if pd.isna(acs_condition):
            return "Coronary occlusion severity is not documented"
        if pd.isna(acs_regions):
            # If we know there is no occlusion, return that explicitly
            from utils.constants import ACS_ACUTE_CONDITIONS
            if acs_condition not in ACS_ACUTE_CONDITIONS:
                return "There is no acute coronary occlusion, so no culprit artery is identified"
            return "An acute coronary occlusion is present, but the culprit artery is not documented"

        # Import ACS constants
        from utils.constants import ACS_ACUTE_CONDITIONS, ACS_ARTERY_MAPPING

        # Only answer if it's an acute occlusion
        if acs_condition not in ACS_ACUTE_CONDITIONS:
            return "There is no acute coronary occlusion, so no culprit artery is identified"
        
        # Parse the acs_pci_regions string (it's in list format)
        import ast
        try:
            # Convert string representation of list to actual list
            if isinstance(acs_regions, str):
                # Remove any extra quotes if present
                acs_regions = acs_regions.strip("'\"")
                # Parse the list
                artery_list = ast.literal_eval(acs_regions)
            else:
                artery_list = acs_regions
            
            # Check if there are any arteries specified
            if not artery_list or (isinstance(artery_list, list) and len(artery_list) == 0):
                return "The culprit artery could not be determined from the available data"
            
            # Get the first artery (primary culprit)
            first_artery = artery_list[0] if isinstance(artery_list, list) else str(artery_list)
            
            # Map to standard name
            standard_name = ACS_ARTERY_MAPPING.get(first_artery, first_artery)
            
            # Generate the answer
            if 'Complete' in acs_condition:
                occlusion_type = "complete occlusion"
            else:
                occlusion_type = "incomplete occlusion"
            
            return f"The culprit artery is the {standard_name} with {occlusion_type}"
            
        except (ValueError, SyntaxError, TypeError):
            from utils.constants import ACS_ACUTE_CONDITIONS
            if acs_condition in ACS_ACUTE_CONDITIONS:
                return "An acute coronary occlusion is present, but the culprit artery information could not be parsed"
            return "There is no acute coronary occlusion, so no culprit artery is identified"
    
    def generate_afib_risk_answer(self, row: pd.Series) -> str:
        """
        Generate answer for incident AFib risk questions (MHI dataset only).
        Checks if patient is at risk for developing AFib in next 2-5 years.
        Returns None if data is not available (which should drop the question).
        """
        # Check if AFib prediction columns exist
        if 'afib_label_2y' not in row.index or 'afib_label_5y' not in row.index:
            return None  # This question should be dropped
        
        afib_2y = row.get('afib_label_2y')
        afib_5y = row.get('afib_label_5y')
        
        # If prediction values are null/nan, we can't answer
        if pd.isna(afib_2y) or pd.isna(afib_5y):
            return None  # This question should be dropped
        
        # First check if patient is ALREADY in AFib
        # Check Afib column (>= 1 means current AFib)
        current_afib = False
        if 'Afib' in row.index and pd.notna(row.get('Afib')):
            try:
                if float(row['Afib']) >= 1:
                    current_afib = True
            except (ValueError, TypeError):
                pass
        
        # Also check Afib_bert_model (> 0.5 threshold for MHI)
        if not current_afib and 'Afib_bert_model' in row.index and pd.notna(row.get('Afib_bert_model')):
            try:
                if float(row['Afib_bert_model']) > 0.5:
                    current_afib = True
            except (ValueError, TypeError):
                pass
        
        # If patient is already in AFib, they can't have "incident" AFib
        if current_afib:
            return "The patient is already in atrial fibrillation"
        
        # Now check future risk predictions
        # Convert boolean values (they're stored as boolean in the data)
        try:
            # Handle both boolean and string representations
            if isinstance(afib_2y, bool):
                risk_2y = afib_2y
            else:
                risk_2y = str(afib_2y).lower() == 'true'
            
            if isinstance(afib_5y, bool):
                risk_5y = afib_5y
            else:
                risk_5y = str(afib_5y).lower() == 'true'
            
            # Determine risk level
            if risk_2y:
                return "Yes - this patient has a high risk of developing atrial fibrillation within the next 2 years"
            elif risk_5y:
                return "Moderate risk - this patient is likely to develop atrial fibrillation within 5 years but not within 2 years"
            else:
                return "Low risk - this patient is unlikely to develop atrial fibrillation in the next 5 years"
                
        except (ValueError, TypeError):
            # Can't parse the risk values
            return None  # This question should be dropped
    
    def generate_answer(self, row: pd.Series) -> str:
        """
        Main function to generate appropriate answer based on prompt type.
        Prepends acute MI/STEMI only for interpretation prompts or
        when the question explicitly asks about acute MI/STEMI.
        """
        # If the row already provides a generated_answer (e.g., prompt maker
        # injected a dataset-specific ground truth for a canonical prompt),
        # respect it and return as-is.
        prefilled = row.get('generated_answer')
        try:
            if prefilled is not None and str(prefilled).strip() != "":
                return str(prefilled).strip()
        except Exception:
            pass

        prompt_category = row.get('prompt_category', '')
        prompt_type = row.get('prompt_type', '')
        
        # Check for acute MI/STEMI in report
        acute_mi_prefix = self.check_for_acute_mi_prefix(row)
        
        # Route to appropriate generator based on prompt category
        base_answer = ""
        
        if prompt_category == 'json_interpretation':
            base_answer = self.generate_json_interpretation(row)
        
        elif 'interpretation' in prompt_category:
            base_answer = self.generate_interpretation_answer(row)
        
        elif prompt_category == 'heart_rate':
            # Heart rate is its own category (legacy support)
            base_answer = self.generate_heart_rate_answer(row)
        
        elif prompt_category == 'intervals':
            # Handle interval questions (legacy support)
            base_answer = self.generate_intervals_answer(row)
        
        elif prompt_category == 'ecg_interval':
            # Combined heart rate and interval questions
            prompt_text = row.get('prompt', '').lower()
            # Determine if asking about heart rate or intervals
            if any(term in prompt_text for term in ['heart rate', 'hr', 'ventricular rate', 'pulse', 'bpm', 'beating']):
                base_answer = self.generate_heart_rate_answer(row)
            else:
                # It's an interval question (PR, QT, QTc)
                base_answer = self.generate_intervals_answer(row)
        
        elif 'demographic' in prompt_category:
            # Extract the demographic type (e.g., 'demographic_gender' -> 'gender')
            demo_type = prompt_category.replace('demographic_', '')
            base_answer = self.generate_demographic_answer(row, demo_type)
        
        elif 'localization' in prompt_category:
            # Extract the finding type from category (e.g., 'localization_q_wave' -> 'q_wave')
            finding_type = prompt_category.replace('localization_', '')
            base_answer = self.generate_localization_answer(row, finding_type)
        
        elif 'category' in prompt_category:
            base_answer = self.generate_category_answer(row, prompt_category)
        
        elif 'classification' in prompt_category:
            base_answer = self.generate_classification_answer(row)
        
        elif 'urgency' in prompt_category:
            base_answer = self.generate_urgency_answer(row)
        
        elif 'random_finding' in prompt_category:
            base_answer = self.generate_random_finding_answer(row)
        
        elif 'structural_heart_disease' in prompt_category:
            base_answer = self.generate_structural_heart_disease_answer(row)
        
        elif 'lvef' in prompt_category:
            base_answer = self.generate_lvef_answer(row)
        
        elif 'acs_severity' in prompt_category:
            base_answer = self.generate_acs_severity_answer(row)
        
        elif 'culprit_artery' in prompt_category:
            base_answer = self.generate_culprit_artery_answer(row)
        
        elif 'afib_risk' in prompt_category:
            base_answer = self.generate_afib_risk_answer(row)
        
        else:
            # Default to interpretation
            base_answer = self.generate_interpretation_answer(row)
        
        # Only include the acute MI prefix for interpretation prompts
        # or when the question explicitly asks about acute MI/STEMI.
        if acute_mi_prefix and prompt_category != 'json_interpretation':
            prompt_text = str(row.get('prompt', '')).lower()
            is_interpretation = ('interpretation' in prompt_category)
            asks_acute_mi = any(
                kw in prompt_text for kw in [
                    'acute mi',
                    'acute myocardial infarction',
                    'stemi'
                ]
            )
            if is_interpretation or asks_acute_mi:
                # Avoid duplicating prefix if already present
                if not base_answer.startswith('*** CONSIDER ACUTE') and not base_answer.startswith('*** ACUTE STEMI'):
                    return acute_mi_prefix + base_answer

        return base_answer
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add generated answers to the dataframe.
        Merges MIMIC metadata if available and adds heart rate.
        """
        df = df.copy()
        
        # Merge MIMIC metadata if available
        if self.dataset == 'mimic' and self.mimic_metadata is not None:
            print("Merging MIMIC metadata for demographic and heart rate information...")
            
            # Extract npy_id from waveform_name if present
            if 'waveform_name' in df.columns:
                # Extract just the ID part (remove .npy extension)
                df['npy_id'] = df['waveform_name'].str.replace('.npy', '')
                
                # Select columns to merge - include gender, age_at_ecg, and rr_interval
                merge_columns = ['npy_id']
                if 'gender' in self.mimic_metadata.columns:
                    merge_columns.append('gender')
                if 'age_at_ecg' in self.mimic_metadata.columns:
                    merge_columns.append('age_at_ecg')
                if 'rr_interval' in self.mimic_metadata.columns:
                    merge_columns.append('rr_interval')
                
                # Merge on npy_id
                df = df.merge(
                    self.mimic_metadata[merge_columns], 
                    on='npy_id', 
                    how='left',
                    suffixes=('', '_mimic')
                )
                
                # Calculate heart rate from RR interval if not already present
                if 'rr_interval' in df.columns:
                    df['heart_rate'] = df['rr_interval'].apply(
                        lambda x: round(60000.0 / x, 1) if pd.notna(x) and x > 0 else None
                    )
                    print(f"Added heart rate for {df['heart_rate'].notna().sum()} records")
                
                if 'gender' in df.columns:
                    print(f"Added gender for {df['gender'].notna().sum()} records")
                
                if 'age_at_ecg' in df.columns:
                    print(f"Added age for {df['age_at_ecg'].notna().sum()} records")
                
                # Clean up temporary columns but keep demographic columns
                df = df.drop(columns=['npy_id'], errors='ignore')
        
        # Generate answer for each row
        print("Generating answers for each prompt...")
        answers = []
        
        for idx, row in df.iterrows():
            answer = self.generate_answer(row)
            answers.append(answer)
            
            if idx % 500 == 0:
                print(f"  Processed {idx}/{len(df)} prompts...")
        
        df['generated_answer'] = answers
        
        # Also keep original report for comparison if available
        if 'report' in df.columns:
            df['original_report'] = df['report']
        
        return df


def main():
    """Main function to generate answers for prompts"""
    
    # Initialize answer generator
    print("Initializing ECG Answer Generator...")
    answer_gen = ECGAnswerGenerator()
    
    # Load the prompts data
    input_path = '/volume/ECG_tokenizer/output/mimic_mhi_psa_test_updated_with_questions_with_ecg_type_with_prompts.parquet'
    print(f"\nLoading prompts from: {input_path}")
    df = pd.read_parquet(input_path)
    
    print(f"Loaded {len(df)} prompts from {df['waveform_name'].nunique()} ECGs")
    
    # Generate answers
    df_with_answers = answer_gen.process_dataframe(df)
    
    # Show statistics
    print("\n" + "="*60)
    print("ANSWER GENERATION STATISTICS")
    print("="*60)
    
    # Answer length statistics
    df_with_answers['answer_length'] = df_with_answers['generated_answer'].str.len()
    print(f"\nAnswer Length Statistics:")
    print(f"  Mean: {df_with_answers['answer_length'].mean():.1f} characters")
    print(f"  Median: {df_with_answers['answer_length'].median():.1f} characters")
    print(f"  Min: {df_with_answers['answer_length'].min()} characters")
    print(f"  Max: {df_with_answers['answer_length'].max()} characters")
    
    # Show examples
    print("\n" + "="*60)
    print("EXAMPLE PROMPT-ANSWER PAIRS")
    print("="*60)
    
    # Show different types of prompt-answer pairs
    example_categories = ['interpretation', 'category_rhythm', 'category_infarct_ischemia', 
                         'classification', 'urgency_assessment']
    
    for cat in example_categories:
        subset = df_with_answers[df_with_answers['prompt_category'] == cat]
        if len(subset) > 0:
            example = subset.iloc[0]
            print(f"\n{cat.upper()}:")
            print(f"ECG: {example['waveform_name']}")
            print(f"ECG Type: {example['ecg_type']}")
            print(f"Prompt: {example['prompt']}")
            print(f"Answer: {example['generated_answer']}")
            if 'original_report' in example and pd.notna(example['original_report']):
                print(f"Original: {example['original_report'][:150]}...")
    
    # Save the result
    output_path = input_path.replace('.parquet', '_with_answers.parquet')
    print(f"\n\nSaving to: {output_path}")
    df_with_answers.to_parquet(output_path, index=False)
    
    # Also save a sample CSV for easy inspection
    sample_csv_path = output_path.replace('.parquet', '_sample.csv')
    df_with_answers.head(100).to_csv(sample_csv_path, index=False)
    print(f"Sample CSV saved to: {sample_csv_path}")
    
    print("\nAnswer generation complete!")
    
    return df_with_answers


if __name__ == "__main__":
    df = main()
