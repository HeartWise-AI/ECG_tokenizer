#!/usr/bin/env python3
"""
ECG Prompt Maker - Generates multiple prompts per ECG based on findings.
Creates 1-N prompts depending on ECG characteristics and categories present.
"""

import json
import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple
from collections import defaultdict
from dataset_column_mappings import DatasetColumnMapper


class ECGPromptMaker:
    """Generate diverse prompts for ECG interpretation tasks"""
    
    def __init__(self, categories_json_path: str = '/volume/ECG_tokenizer/dictionary/deepecg_categories.json', dataset: str = 'mimic'):
        """Initialize with category definitions"""
        
        # Load category definitions
        with open(categories_json_path, 'r') as f:
            self.categories_dict = json.load(f)  # The JSON is already the categories dict
        
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
                "What is the atrial rhythm?"
            ],
            "CONDUCTION": [
                "Are there any conduction abnormalities?",
                "Is there any heart block present?",
                "What is the PR interval and QRS duration?",
                "Are there any bundle branch blocks?",
                "Is AV conduction normal?",
                "Can you assess the conduction system?",
                "Is there any conduction delay?",
                "What type of block is present if any?",
                "Are the intervals within normal limits?",
                "Is there evidence of pre-excitation?"
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
                "Does this suggest acute coronary syndrome?"
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
                "Are there any other notable findings?",
                "Are there any nonspecific ST-T changes?",
                "Is there early repolarization?",
                "Are there any electrolyte abnormalities suggested?",
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
            "ST_MORPHOLOGY": [
                "What is the ST segment morphology?",
                "Is there ST downsloping or upsloping?",
                "Are there any ST segment abnormalities?",
                "Describe the ST segment characteristics.",
                "Is there early repolarization pattern?"
            ],
            "QRS_AXIS": [
                "What is the QRS axis?",
                "Is there axis deviation?",
                "What is the electrical axis of the heart?",
                "Is the QRS axis normal or deviated?",
                "What type of axis deviation is present?",
                "Is there left or right axis deviation?",
                "What is the frontal plane QRS axis?",
                "Is there extreme axis deviation?"
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
        
        # New demographic prompts
        self.demographic_prompts = {
            "GENDER": [
                "What is the patient's gender?",
                "Is this ECG from a male or female patient?",
                "What is the sex of the patient?",
                "Can you tell me the patient's gender?",
                "Is the patient male or female?"
            ],
            "AGE": [
                "What is the patient's age?",
                "How old is the patient?",
                "What is the age of this patient?",
                "Can you tell me the patient's age?",
                "What age is the patient?"
            ],
            "HEART_RATE": [
                "What is the heart rate?",
                "What is the patient's heart rate?",
                "What is the ventricular rate?",
                "Can you tell me the heart rate in bpm?",
                "What is the HR?",
                "How fast is the heart beating?",
                "What's the pulse rate?"
            ]
        }
        
        # Prompt weights for sampling
        self.prompt_weights = {
            'interpretation': 0.35,  # 35% interpretation prompts
            'category': 0.30,        # 30% category-specific
            'classification': 0.20,  # 20% classification
            'demographic': 0.15      # 15% demographic/HR questions
        }
    
    def determine_active_categories(self, row: pd.Series) -> Dict[str, List[str]]:
        """
        Determine which categories have positive findings.
        Returns dict with category -> list of active conditions
        """
        active = defaultdict(list)
        
        for category, conditions in self.categories_dict.items():
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
            'ST_MORPHOLOGY': [],
            'QRS_AXIS': []
        }
        
        # Check each column for localization info
        for col in row.index:
            try:
                if pd.notna(row[col]) and float(row[col]) >= 1:
                    col_str = str(col)
                
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
                    
                    # Check for morphology patterns
                    if col_str in ['ST downsloping', 'ST upslopping', 'Early repolarization']:
                        localization_findings['ST_MORPHOLOGY'].append(col_str)
                    
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
    
    def generate_prompts_for_ecg(self, row: pd.Series) -> List[Tuple[str, str, float]]:
        """
        Generate multiple prompts for a single ECG.
        Returns list of (prompt, category, weight) tuples.
        """
        prompts = []
        
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
            
            # Add prompts for each active category (up to 3 to avoid too many)
            for i, category in enumerate(sorted_categories[:3]):
                if category in self.category_specific_prompts:
                    cat_prompt = random.choice(self.category_specific_prompts[category])
                    # Higher weight for more critical categories
                    weight = 0.8 if i == 0 else 0.6 if i == 1 else 0.4
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
        
        # 5. Add demographic prompts - ALWAYS include heart rate if available
        demographic_types = []
        
        # Check if demographic data is available
        if 'gender' in row.index and pd.notna(row.get('gender')):
            demographic_types.append('GENDER')
        if 'age_at_ecg' in row.index and pd.notna(row.get('age_at_ecg')):
            demographic_types.append('AGE')
        
        # ALWAYS add heart rate question if RR interval is available
        if 'rr_interval' in row.index and pd.notna(row.get('rr_interval')):
            hr_prompt = random.choice(self.demographic_prompts['HEART_RATE'])
            prompts.append((hr_prompt, 'heart_rate', 0.7))  # Heart rate as its own category
        
        # Randomly add 1 other demographic question if available
        if demographic_types:
            # 50% chance to add one more demographic question
            if random.random() < 0.5:
                selected_type = random.choice(demographic_types)
                demo_prompt = random.choice(self.demographic_prompts[selected_type])
                prompts.append((demo_prompt, f'demographic_{selected_type.lower()}', 0.5))
        
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
            
            # Limit to max_prompts_per_ecg
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