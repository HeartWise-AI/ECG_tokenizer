"""
Offline evaluation script for computing metrics from saved generations.
Merges validation JSON files with CSV ground truth and computes BERTScore, ROUGE, BLEU, METEOR.
Presents results overall and per category.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
from collections import defaultdict
import argparse
from tqdm import tqdm

# Import our improved metrics
try:
    import sacrebleu
except ImportError:
    warnings.warn("SacreBLEU not installed. Install with: pip install sacrebleu")
    sacrebleu = None

try:
    from evaluate import load as load_metric
except ImportError:
    warnings.warn("Evaluate library not installed. Install with: pip install evaluate")
    load_metric = None


class OfflineMetricsEvaluator:
    """Evaluator for computing metrics from saved generation JSON files."""
    
    def __init__(self, json_path: str, csv_path: str):
        """
        Initialize the evaluator.
        
        Args:
            json_path: Path to the JSON file with generations
            csv_path: Path to the CSV file with ground truth and metadata
        """
        self.json_path = json_path
        self.csv_path = csv_path
        
        # Load data
        self.json_data = self._load_json()
        self.csv_data = self._load_csv()
        
        # Initialize metrics
        self.rouge_metric = None
        self.meteor_metric = None
        self.bertscore_metric = None
        
    def _load_json(self) -> Dict[str, Any]:
        """Load the JSON file with generations."""
        print(f"Loading JSON from {self.json_path}...")
        with open(self.json_path, 'r') as f:
            return json.load(f)
    
    def _load_csv(self) -> pd.DataFrame:
        """Load the CSV file with ground truth."""
        print(f"Loading CSV from {self.csv_path}...")
        return pd.read_csv(self.csv_path)
    
    def _extract_filename_from_path(self, path: str) -> str:
        """Extract filename from waveform path."""
        # Extract just the filename from paths like:
        # /media/data1/datasets/MIMIC-IV/adjusted_signals/test/48487288.npy
        return Path(path).name
    
    def merge_data(self) -> List[Dict[str, Any]]:
        """
        Merge JSON and CSV data based on waveform filename.
        
        Returns:
            List of merged records with generation, ground truth, and metadata
        """
        print("Merging JSON and CSV data...")
        merged = []
        
        # Create a mapping from filename to CSV row
        csv_mapping = {}
        for idx, row in self.csv_data.iterrows():
            # Try multiple columns for the waveform path - prioritize waveform_name
            if pd.notna(row.get('waveform_name')):
                filename = str(row['waveform_name'])
            elif pd.notna(row.get('waveform_path_psa')):
                filename = self._extract_filename_from_path(str(row['waveform_path_psa']))
            elif pd.notna(row.get('waveform_path_original')):
                filename = self._extract_filename_from_path(str(row['waveform_path_original']))
            else:
                continue
            
            # Ensure it's just the filename
            filename = self._extract_filename_from_path(filename)
            if filename:
                csv_mapping[filename] = row.to_dict()
        
        # Merge with JSON data
        matched = 0
        unmatched = []
        
        for filename, json_entry in tqdm(self.json_data.items(), desc="Merging data"):
            # Clean filename (remove any path components if present)
            clean_filename = self._extract_filename_from_path(filename)
            
            if clean_filename in csv_mapping:
                csv_row = csv_mapping[clean_filename]
                
                # Create merged record
                record = {
                    'filename': clean_filename,
                    'question': json_entry.get('Question', ''),
                    'generation': json_entry.get('Generation', ''),
                    'ground_truth': json_entry.get('Ground truth', ''),
                    'prompt_category': csv_row.get('prompt_category', 'unknown'),
                    'prompt': csv_row.get('prompt', ''),
                    'generated_answer': csv_row.get('generated_answer', ''),
                    'original_report': csv_row.get('original_report', ''),
                    'ecg_type': csv_row.get('ecg_type', 'unknown')
                }
                
                # Use JSON ground truth if available, otherwise use CSV
                if not record['ground_truth'] and record['generated_answer']:
                    record['ground_truth'] = record['generated_answer']
                
                merged.append(record)
                matched += 1
            else:
                # Use JSON data even if no CSV match
                record = {
                    'filename': clean_filename,
                    'question': json_entry.get('Question', ''),
                    'generation': json_entry.get('Generation', ''),
                    'ground_truth': json_entry.get('Ground truth', ''),
                    'prompt_category': 'unknown',
                    'prompt': '',
                    'generated_answer': '',
                    'original_report': '',
                    'ecg_type': 'unknown'
                }
                merged.append(record)
                unmatched.append(clean_filename)
        
        print(f"Merged {matched} records successfully")
        if unmatched:
            print(f"Warning: {len(unmatched)} JSON entries had no CSV match")
            if len(unmatched) <= 10:
                print(f"Unmatched files: {unmatched}")
        
        return merged
    
    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute all metrics for a set of predictions and references.
        
        Args:
            predictions: List of generated texts
            references: List of ground truth texts
            
        Returns:
            Dictionary of metric scores
        """
        results = {}
        
        # Filter out empty pairs
        valid_pairs = [(p, r) for p, r in zip(predictions, references) 
                      if p.strip() and r.strip()]
        
        if not valid_pairs:
            warnings.warn("No valid prediction-reference pairs found")
            return {
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0,
                'bleu1': 0.0,
                'bleu4': 0.0,
                'meteor': 0.0,
                'bertscore_precision': 0.0,
                'bertscore_recall': 0.0,
                'bertscore_f1': 0.0,
                'num_samples': 0
            }
        
        valid_preds, valid_refs = zip(*valid_pairs)
        results['num_samples'] = len(valid_preds)
        
        # Unified ROUGE/BLEU/METEOR via utility
        try:
            from utils.metrics.aggregate_text_metrics import aggregate_text_metrics
            agg = aggregate_text_metrics(list(valid_preds), list(valid_refs))
            # Ensure all expected keys exist
            results['rouge1'] = float(agg.get('rouge1', 0.0))
            results['rouge2'] = float(agg.get('rouge2', 0.0))
            results['rougeL'] = float(agg.get('rougeL', 0.0))
            results['bleu1'] = float(agg.get('bleu1', 0.0))
            results['bleu4'] = float(agg.get('bleu4', 0.0))
            results['meteor'] = float(agg.get('meteor', 0.0))
        except Exception as e:
            warnings.warn(f"Unified metric computation failed: {e}")
            results.update({'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'bleu1': 0.0, 'bleu4': 0.0, 'meteor': 0.0})
        
        # BERTScore
        if load_metric:
            try:
                if self.bertscore_metric is None:
                    self.bertscore_metric = load_metric("bertscore")
                
                bert_result = self.bertscore_metric.compute(
                    predictions=list(valid_preds),
                    references=list(valid_refs),
                    lang='en',
                    model_type='microsoft/deberta-xlarge-mnli',
                    rescale_with_baseline=True,
                    idf=False
                )
                
                # Handle list or single value
                def get_mean(scores):
                    if isinstance(scores, list):
                        return float(np.mean(scores))
                    return float(scores)
                
                results['bertscore_precision'] = get_mean(bert_result.get('precision', 0))
                results['bertscore_recall'] = get_mean(bert_result.get('recall', 0))
                results['bertscore_f1'] = get_mean(bert_result.get('f1', 0))
                
            except Exception as e:
                warnings.warn(f"BERTScore computation failed: {e}")
                results.update({
                    'bertscore_precision': 0.0,
                    'bertscore_recall': 0.0,
                    'bertscore_f1': 0.0
                })
        else:
            results.update({
                'bertscore_precision': 0.0,
                'bertscore_recall': 0.0,
                'bertscore_f1': 0.0
            })
        
        return results
    
    def evaluate(self) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Evaluate all metrics overall and per category.
        
        Returns:
            Tuple of (overall_metrics, per_category_metrics)
        """
        print("\nStarting evaluation...")
        
        # Merge data
        merged_data = self.merge_data()
        
        if not merged_data:
            warnings.warn("No data to evaluate")
            return {}, {}
        
        # Prepare data for evaluation
        all_predictions = []
        all_references = []
        category_data = defaultdict(lambda: {'predictions': [], 'references': []})
        
        for record in merged_data:
            pred = record['generation']
            ref = record['ground_truth']
            
            if pred and ref:
                all_predictions.append(pred)
                all_references.append(ref)
                
                # Group by category
                category = record['prompt_category'] or 'unknown'
                category_data[category]['predictions'].append(pred)
                category_data[category]['references'].append(ref)
        
        # Compute overall metrics
        print("\nComputing overall metrics...")
        overall_metrics = self.compute_metrics(all_predictions, all_references)
        
        # Compute per-category metrics
        print("\nComputing per-category metrics...")
        per_category_metrics = {}
        
        for category, data in tqdm(category_data.items(), desc="Categories"):
            if data['predictions']:  # Only compute if there's data
                per_category_metrics[category] = self.compute_metrics(
                    data['predictions'],
                    data['references']
                )
        
        return overall_metrics, per_category_metrics
    
    def save_results(self, overall_metrics: Dict[str, float], 
                    per_category_metrics: Dict[str, Dict[str, float]],
                    output_path: Optional[str] = None):
        """
        Save evaluation results to JSON file.
        
        Args:
            overall_metrics: Overall metrics dictionary
            per_category_metrics: Per-category metrics dictionary
            output_path: Optional output path (defaults to input JSON path with _metrics suffix)
        """
        if output_path is None:
            json_path = Path(self.json_path)
            output_path = json_path.parent / f"{json_path.stem}_metrics.json"
        
        results = {
            'input_files': {
                'json': self.json_path,
                'csv': self.csv_path
            },
            'overall_metrics': overall_metrics,
            'per_category_metrics': per_category_metrics
        }
        
        print(f"\nSaving results to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def print_results(self, overall_metrics: Dict[str, float],
                     per_category_metrics: Dict[str, Dict[str, float]]):
        """
        Print formatted evaluation results.
        
        Args:
            overall_metrics: Overall metrics dictionary
            per_category_metrics: Per-category metrics dictionary
        """
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        print("\n📊 Overall Metrics:")
        print("-"*40)
        if overall_metrics:
            num_samples = overall_metrics.get('num_samples', 0)
            print(f"Number of samples: {num_samples}")
            print(f"ROUGE-1 F1:        {overall_metrics.get('rouge1', 0):.4f}")
            print(f"ROUGE-2 F1:        {overall_metrics.get('rouge2', 0):.4f}")
            print(f"ROUGE-L F1:        {overall_metrics.get('rougeL', 0):.4f}")
            print(f"BLEU-1:            {overall_metrics.get('bleu1', 0):.4f}")
            print(f"BLEU-4:            {overall_metrics.get('bleu4', 0):.4f}")
            print(f"METEOR:            {overall_metrics.get('meteor', 0):.4f}")
            print(f"BERTScore Prec:    {overall_metrics.get('bertscore_precision', 0):.4f}")
            print(f"BERTScore Rec:     {overall_metrics.get('bertscore_recall', 0):.4f}")
            print(f"BERTScore F1:      {overall_metrics.get('bertscore_f1', 0):.4f}")
        else:
            print("No overall metrics computed")
        
        if per_category_metrics:
            print("\n📈 Per-Category Metrics:")
            print("-"*40)
            
            # Sort categories by number of samples (descending)
            sorted_categories = sorted(
                per_category_metrics.items(),
                key=lambda x: x[1].get('num_samples', 0),
                reverse=True
            )
            
            for category, metrics in sorted_categories:
                num_samples = metrics.get('num_samples', 0)
                print(f"\n Category: {category} (n={num_samples})")
                print(f"  ROUGE-1: {metrics.get('rouge1', 0):.4f}  "
                     f"ROUGE-L: {metrics.get('rougeL', 0):.4f}")
                print(f"  BLEU-1:  {metrics.get('bleu1', 0):.4f}  "
                     f"BLEU-4:  {metrics.get('bleu4', 0):.4f}")
                print(f"  METEOR:  {metrics.get('meteor', 0):.4f}")
                print(f"  BERTScore F1: {metrics.get('bertscore_f1', 0):.4f}")
        
        print("\n" + "="*80)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLM generation metrics from saved JSON and CSV files"
    )
    
    parser.add_argument(
        '--json-path',
        type=str,
        required=True,
        help='Path to JSON file with generations'
    )
    
    parser.add_argument(
        '--csv-path',
        type=str,
        required=True,
        help='Path to CSV file with ground truth and metadata'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='Output path for metrics JSON (default: input_metrics.json)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to file'
    )
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = OfflineMetricsEvaluator(args.json_path, args.csv_path)
    
    # Run evaluation
    overall_metrics, per_category_metrics = evaluator.evaluate()
    
    # Print results
    evaluator.print_results(overall_metrics, per_category_metrics)
    
    # Save results
    if not args.no_save:
        evaluator.save_results(overall_metrics, per_category_metrics, args.output_path)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
