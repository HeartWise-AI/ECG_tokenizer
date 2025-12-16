#!/usr/bin/env python3
"""
Per-Category Metrics Calculator for ECG LLM Fine-tuning

Computes ROUGE, BLEU, and METEOR metrics grouped by prompt category to track
performance across different types of ECG-related questions and tasks.
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np

# Unified metrics util
from utils.metrics.aggregate_text_metrics import aggregate_text_metrics


class CategoryMetricsCalculator:
    """
    Calculate metrics (ROUGE, BLEU, METEOR) per prompt category.
    
    Enables tracking performance across different ECG prompt types:
    - interpretation: Full ECG interpretation reports
    - classification: Normal/borderline/pathological classification
    - category_*: Category-specific questions (rhythm, conduction, etc.)
    - localization_*: Location-specific questions (Q waves, ST changes)
    - heart_rate: Heart rate extraction
    - demographic_*: Demographic information extraction
    - json_interpretation: Structured JSON output
    """
    
    def __init__(self, 
                 metric_names: List[str] = ["rouge", "bleu", "meteor"],
                 rouge_variants: List[str] = ["rouge1", "rouge2", "rougeL"],
                 device: Optional[torch.device] = None):
        """
        Initialize category metrics calculator.
        
        Args:
            metric_names: List of metrics to compute ["rouge", "bleu", "meteor"]
            rouge_variants: Which ROUGE variants to compute
            device: Device for torch metrics
        """
        self.metric_names = metric_names
        self.rouge_variants = rouge_variants
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # No stateful torchmetrics; use aggregate_text_metrics per category/overall
        self.metrics = {}
        
        # Storage for category-wise predictions and references
        self.category_data = defaultdict(lambda: {"predictions": [], "references": []})
        
    def add_batch(self, 
                  predictions: List[str], 
                  references: List[str], 
                  categories: List[str]) -> None:
        """
        Add a batch of predictions and references with their categories.
        
        Args:
            predictions: Generated text predictions
            references: Ground truth reference text
            categories: Prompt categories for each sample
        """
        assert len(predictions) == len(references) == len(categories), \
            "All inputs must have the same length"
        
        for pred, ref, cat in zip(predictions, references, categories):
            if cat:  # Only add if category is not None/empty
                self.category_data[cat]["predictions"].append(pred)
                self.category_data[cat]["references"].append(ref)
    
    def compute_category_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute all metrics for each category.
        
        Returns:
            Dict mapping category -> metric_name -> score
            e.g., {
                "interpretation": {"rouge1": 0.45, "bleu": 0.32, "meteor": 0.28},
                "classification": {"rouge1": 0.78, "bleu": 0.65, "meteor": 0.71},
                ...
            }
        """
        results = {}
        
        for category, data in self.category_data.items():
            if len(data["predictions"]) == 0:
                continue
                
            category_results = {}
            predictions = data["predictions"]
            references = data["references"]
            
            # Compute metrics via unified util
            agg = aggregate_text_metrics(predictions, references)
            # Only include keys relevant to our configured variants
            for k, v in agg.items():
                if k.startswith("rouge") or k.startswith("bleu") or k == "meteor":
                    category_results[k] = float(v)
            
            results[category] = category_results
        
        return results
    
    def compute_overall_metrics(self) -> Dict[str, float]:
        """
        Compute overall metrics across all categories.
        
        Returns:
            Dict mapping metric_name -> overall_score
        """
        # Flatten all predictions and references
        all_predictions = []
        all_references = []
        
        for data in self.category_data.values():
            all_predictions.extend(data["predictions"])
            all_references.extend(data["references"])
        
        if len(all_predictions) == 0:
            return {}
        
        overall_results = {}
        
        # Compute metrics via unified util
        agg = aggregate_text_metrics(all_predictions, all_references)
        for k, v in agg.items():
            overall_results[k] = float(v)
        
        return overall_results
    
    def get_category_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about each category (sample counts, etc.).
        
        Returns:
            Dict mapping category -> statistics
        """
        stats = {}
        
        for category, data in self.category_data.items():
            n_samples = len(data["predictions"])
            avg_pred_length = np.mean([len(pred.split()) for pred in data["predictions"]]) if n_samples > 0 else 0
            avg_ref_length = np.mean([len(ref.split()) for ref in data["references"]]) if n_samples > 0 else 0
            
            stats[category] = {
                "n_samples": n_samples,
                "avg_prediction_length": avg_pred_length,
                "avg_reference_length": avg_ref_length,
                "sample_predictions": data["predictions"][:3] if n_samples > 0 else [],
                "sample_references": data["references"][:3] if n_samples > 0 else []
            }
        
        return stats
    
    def reset(self) -> None:
        """Reset all stored data for new evaluation."""
        self.category_data.clear()
        
        # No stateful torchmetrics to reset
    
    def format_results_for_logging(self, 
                                   category_results: Dict[str, Dict[str, float]], 
                                   overall_results: Dict[str, float],
                                   prefix: str = "eval") -> Dict[str, float]:
        """
        Format results for logging to wandb or tensorboard.
        
        Args:
            category_results: Results from compute_category_metrics()
            overall_results: Results from compute_overall_metrics()
            prefix: Prefix for metric names (e.g., "eval", "train")
        
        Returns:
            Flattened dict suitable for logging
        """
        log_dict = {}
        
        # Add overall metrics
        for metric, value in overall_results.items():
            log_dict[f"{prefix}/{metric}_overall"] = value
        
        # Add per-category metrics
        for category, metrics in category_results.items():
            for metric, value in metrics.items():
                log_dict[f"{prefix}/{metric}_{category}"] = value
        
        # Add category sample counts
        stats = self.get_category_statistics()
        for category, category_stats in stats.items():
            log_dict[f"{prefix}/n_samples_{category}"] = category_stats["n_samples"]
        
        return log_dict


def compute_category_metrics(predictions: List[str],
                           references: List[str], 
                           categories: List[str],
                           metric_names: List[str] = ["rouge", "bleu", "meteor"]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Convenience function to compute category metrics in one call.
    
    Args:
        predictions: Generated text predictions
        references: Ground truth reference text  
        categories: Prompt categories for each sample
        metric_names: Which metrics to compute
    
    Returns:
        Tuple of (category_results, overall_results)
    """
    calculator = CategoryMetricsCalculator(metric_names=metric_names)
    calculator.add_batch(predictions, references, categories)
    
    category_results = calculator.compute_category_metrics()
    overall_results = calculator.compute_overall_metrics()
    
    return category_results, overall_results


# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    predictions = [
        "Normal sinus rhythm; No abnormalities",
        "Atrial fibrillation; Irregular rhythm",
        "Normal ECG",
        "Male",
        "Q waves in leads II, III, aVF"
    ]
    
    references = [
        "Normal sinus rhythm; Normal ECG",
        "Atrial fibrillation; Irregularly irregular",
        "Normal ECG; No significant findings",
        "Male",
        "Inferior Q waves present"
    ]
    
    categories = [
        "interpretation",
        "interpretation", 
        "classification",
        "demographic_gender",
        "localization_q_wave"
    ]
    
    # Test the calculator
    calculator = CategoryMetricsCalculator()
    calculator.add_batch(predictions, references, categories)
    
    category_results = calculator.compute_category_metrics()
    overall_results = calculator.compute_overall_metrics()
    stats = calculator.get_category_statistics()
    
    print("=== Category Results ===")
    for category, metrics in category_results.items():
        print(f"{category}:")
        for metric, score in metrics.items():
            print(f"  {metric}: {score:.4f}")
        print()
    
    print("=== Overall Results ===")
    for metric, score in overall_results.items():
        print(f"{metric}: {score:.4f}")
    
    print("\n=== Statistics ===")
    for category, category_stats in stats.items():
        print(f"{category}: {category_stats['n_samples']} samples")
    
    # Test logging format
    log_dict = calculator.format_results_for_logging(category_results, overall_results)
    print("\n=== Logging Format ===")
    for key, value in log_dict.items():
        print(f"{key}: {value}")
