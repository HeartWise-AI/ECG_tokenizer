#!/usr/bin/env python3
"""
ECG Answer Generator - Creates appropriate answers for each prompt type.
Generates interpretation reports, category-specific answers, and classifications.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataset_column_mappings import DatasetColumnMapper


class ECGAnswerGenerator:
    """Generate appropriate answers for different ECG prompt types"""
    
    def __init__(self, 
                 categories_json_path: str = '/volume/ECG_tokenizer/dictionary/deepecg_categories.json',
                 translation_json_path: str = '/volume/ECG_tokenizer/dictionary/deepecg_diagnosis_translation.json',
                 language: str = 'en',
                 dataset: str = 'mimic'):
        """Initialize with category definitions and translations
        
        Args:
            categories_json_path: Path to categories JSON
            translation_json_path: Path to translation JSON
            language: 'en' for English, 'fr' for French
            dataset: Dataset name ('mimic' or others) for metadata merging
        """
        
        # Load category definitions
        with open(categories_json_path, 'r') as f:
            self.categories_dict = json.load(f)
        
        # Load translations
        with open(translation_json_path, 'r') as f:
            trans_data = json.load(f)
        
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
    
    def get_active_findings(self, row: pd.Series) -> Dict[str, List[str]]:
        """
        Extract active findings from the row organized by category.
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
                
                for col in col_names:
                    if col in row.index:
                        try:
                            if pd.notna(row[col]) and float(row[col]) >= 1:
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
        This is for 'interpretation' type prompts.
        Uses the REPORT column if available, otherwise falls back to individual findings.
        Includes heart rate when available.
        """
        heart_rate = self.calculate_heart_rate(row)
        
        # Check if REPORT column exists and has content using column mapper
        report_col = self.column_mapper.get_column('report')
        if report_col and report_col in row.index and pd.notna(row[report_col]) and str(row[report_col]).strip():
            report = str(row[report_col]).strip()
            
            # Add heart rate if available
            if heart_rate:
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
            # Normal ECG
            if heart_rate:
                return f"Normal sinus rhythm (HR: {heart_rate} bpm); Normal ECG"
            else:
                return "Normal sinus rhythm; Normal ECG"
        
        # Build structured report
        report_parts = []
        heart_rate_added = False  # Track if we've already added HR
        
        # Add findings by category priority
        for category in self.category_order:
            if category in active_findings:
                for finding in active_findings[category]:
                    formatted = self.format_finding_name(finding)
                    # Add heart rate only once to the first rhythm finding
                    if category == 'RHYTHM' and heart_rate and not heart_rate_added and 'rhythm' in formatted.lower():
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
        Calculate heart rate from RR interval if available.
        Returns heart rate in bpm or None if not available.
        """
        # Check for heart rate column (might be already calculated)
        if 'heart_rate' in row.index and pd.notna(row['heart_rate']):
            return float(row['heart_rate'])
        
        # Check for RR interval columns
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
    
    def generate_category_answer(self, row: pd.Series, category: str) -> str:
        """
        Generate category-specific answer.
        Properly handles YES/NO questions and specific condition queries.
        Includes heart rate for rhythm-related questions.
        """
        active_findings = self.get_active_findings(row)
        prompt_text = row.get('prompt', '').lower()
        heart_rate = self.calculate_heart_rate(row)
        
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
            'is there', 'are there', 'does this', 'does it', 
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
            'bundle branch': ['LBBB', 'RBBB', 'Left bundle branch block', 'Right bundle branch block'],
            'atrial fibrillation': ['Afib', 'Atrial fibrillation'],
            'atrial flutter': ['Atrial flutter', 'Flutter'],
            'ischemia': ['ST elevation', 'ST depression', 'T wave inversion', 'Ischemia'],
            'infarction': ['MI', 'Myocardial infarction', 'Acute MI', 'Q waves'],
            'hypertrophy': ['LVH', 'RVH', 'LAE', 'RAE', 'Hypertrophy', 'Enlargement'],
            'pericarditis': ['Pericarditis', 'Pericarditic'],
            'arrhythmia': ['Tachycardia', 'Bradycardia', 'Afib', 'Flutter', 'VT', 'SVT'],
            'st elevation': ['ST elevation', 'STEMI'],
            'st depression': ['ST depression'],
            'q waves': ['Q waves', 'Pathological Q'],
            'axis deviation': ['Left axis deviation', 'Right axis deviation', 'Axis deviation']
        }
        
        # Check if prompt asks about a specific condition
        asked_condition = None
        matching_findings = []
        
        # Special handling for ectopic beats - only match PVCs and PACs
        if 'ectopic' in prompt_text:
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
                    return f"Yes - {'; '.join(matching_findings)}"
                else:
                    if asked_condition == 'ectopic beat':
                        return "No - no ectopic beats present"
                    else:
                        return f"No - no evidence of {asked_condition.replace('_', ' ')}"
            else:
                # General category question
                if mapped_category and mapped_category in active_findings:
                    findings = [self.format_finding_name(f) for f in active_findings[mapped_category]]
                    
                    # If asking about abnormalities, filter out normal findings
                    if asking_for_abnormalities:
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
                            # Add heart rate to first rhythm finding only
                            if mapped_category == 'RHYTHM' and heart_rate and is_rhythm_question:
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
                    response = self._get_negative_response(mapped_category)
                    # Add heart rate for rhythm questions even when normal
                    if mapped_category == 'RHYTHM' and heart_rate and is_rhythm_question:
                        response = response.replace("normal sinus rhythm", f"normal sinus rhythm (HR: {heart_rate} bpm)")
                    return response
        
        else:
            # Not a YES/NO question - describe what's present
            if mapped_category and mapped_category in active_findings:
                findings = [self.format_finding_name(f) for f in active_findings[mapped_category]]
                # Add heart rate to first rhythm finding only
                if mapped_category == 'RHYTHM' and heart_rate and is_rhythm_question:
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
            else:
                response = self._get_negative_response(mapped_category)
                # Add heart rate for rhythm questions even when normal
                if mapped_category == 'RHYTHM' and heart_rate and is_rhythm_question:
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
    
    def generate_classification_answer(self, row: pd.Series) -> str:
        """
        Generate classification answer (normal/borderline/pathological).
        Properly handles YES/NO questions about ECG normalcy.
        ALWAYS returns a proper classification - never "Unable to classify".
        """
        prompt_text = row.get('prompt', '').lower()
        ecg_type = row.get('ecg_type', 'unknown')
        active_findings = self.get_active_findings(row)
        
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
        
        if ecg_type == 'normal':
            if is_urgent_question:
                return "No - routine follow-up; normal ECG"
            elif is_normal_question:
                return "Yes - ECG is within normal limits"
            elif is_abnormal_question:
                if 'wrong' in prompt_text or 'concerning' in prompt_text:
                    return "No - ECG is normal"
                else:
                    return "No - ECG is normal; No significant abnormalities"
            else:
                return "Normal ECG; No significant abnormalities detected"
        
        elif ecg_type == 'borderline':
            # List the borderline findings
            findings = []
            for category, items in active_findings.items():
                findings.extend(items[:2])
            
            finding_str = ""
            if findings:
                finding_str = ", ".join([self.format_finding_name(f) for f in findings[:3]])
            
            if is_urgent_question:
                return "No - routine follow-up recommended; borderline findings only"
            elif is_normal_question:
                return f"No - Borderline ECG; Minor findings: {finding_str}" if finding_str else "No - Borderline ECG"
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
                if is_truly_urgent:
                    return f"Yes - urgent intervention needed; {finding_str}" if finding_str else "Yes - urgent intervention needed"
                else:
                    return f"No - prompt follow-up recommended; {finding_str}" if finding_str else "No - prompt follow-up recommended"
            elif is_normal_question or 'within normal limits' in prompt_text:
                return f"No - Abnormal ECG; Significant findings: {finding_str}" if finding_str else "No - Abnormal ECG"
            elif 'wrong' in prompt_text:
                return f"Yes - Abnormal ECG; Significant findings: {finding_str}" if finding_str else "Yes - Abnormal ECG"
            elif 'concerning' in prompt_text or 'require follow-up' in prompt_text:
                return f"Yes - ECG shows significant abnormalities: {finding_str}" if finding_str else "Yes - Abnormal ECG requiring follow-up"
            elif is_abnormal_question:
                return f"Yes - Abnormal ECG; Significant findings: {finding_str}" if finding_str else "Yes - Abnormal ECG"
            else:
                return f"Abnormal ECG; Significant findings: {finding_str}" if finding_str else "Abnormal ECG; Multiple abnormalities"
        
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
        """
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
                        import re
                        # Add spaces between lead names
                        clean_loc = re.sub(r'(V\d+|aV[RLF]|I{1,3})(?=[Vv]|aV|I{1,3})', r'\1, ', clean_loc)
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
        
        # Go through each category and condition
        for category, conditions in self.categories_dict.items():
            # Map category names to JSON keys
            json_category = category.replace(", ", "_").replace(" ", "_").upper()
            
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
                                # Use cleaner condition name
                                clean_name = condition.replace(' (', '_').replace(')', '').replace(' - ', '_').replace(', ', '_')
                                present_findings.append(clean_name)
                                break
                        except (ValueError, TypeError):
                            continue
            
            # Only add category if it has findings
            if present_findings:
                json_output[json_category] = present_findings
        
        # Add heart rate if available (rounded to integer)
        heart_rate = self.calculate_heart_rate(row)
        if heart_rate:
            json_output["heart_rate_bpm"] = int(round(heart_rate))
        
        # Add ECG classification
        ecg_type = row.get('ecg_type', 'unknown')
        json_output["ecg_classification"] = ecg_type
        
        # Return as compact JSON string
        return json.dumps(json_output, separators=(',', ':'))
    
    def generate_heart_rate_answer(self, row: pd.Series) -> str:
        """
        Generate answer for heart rate questions.
        This is now its own category separate from demographics.
        """
        heart_rate = self.calculate_heart_rate(row)
        if heart_rate:
            return f"{int(round(heart_rate))} bpm"
        return "Heart rate cannot be determined"
    
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
            if heart_rate:
                return f"{heart_rate} bpm"
            return "Heart rate cannot be determined"
        
        return "Information not available"
    
    def generate_answer(self, row: pd.Series) -> str:
        """
        Main function to generate appropriate answer based on prompt type.
        """
        prompt_category = row.get('prompt_category', '')
        prompt_type = row.get('prompt_type', '')
        
        # Route to appropriate generator based on prompt category
        if prompt_category == 'json_interpretation':
            return self.generate_json_interpretation(row)
        
        elif 'interpretation' in prompt_category:
            return self.generate_interpretation_answer(row)
        
        elif prompt_category == 'heart_rate':
            # Heart rate is its own category now
            return self.generate_heart_rate_answer(row)
        
        elif 'demographic' in prompt_category:
            # Extract the demographic type (e.g., 'demographic_gender' -> 'gender')
            demo_type = prompt_category.replace('demographic_', '')
            return self.generate_demographic_answer(row, demo_type)
        
        elif 'localization' in prompt_category:
            # Extract the finding type from category (e.g., 'localization_q_wave' -> 'q_wave')
            finding_type = prompt_category.replace('localization_', '')
            return self.generate_localization_answer(row, finding_type)
        
        elif 'category' in prompt_category:
            return self.generate_category_answer(row, prompt_category)
        
        elif 'classification' in prompt_category:
            return self.generate_classification_answer(row)
        
        elif 'urgency' in prompt_category:
            return self.generate_urgency_answer(row)
        
        else:
            # Default to interpretation
            return self.generate_interpretation_answer(row)
    
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