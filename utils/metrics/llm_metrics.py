"""
This module contains classes to compute evaluation metrics for LLMs.
Each class provides a static function `compute_score` to compute the metric.
Metrics computed:
 - ROUGE: Computes both ROUGE-1 and ROUGE-L F1 scores.
 - BLEU: Computes both BLEU-1 (unigram) and BLEU-4 scores.
 - METEOR: Computes the METEOR score.
 
Each class is registered with MetricRegistry for consistent retrieval.
"""

import torch
from typing import Dict, List, Union
from transformers import GPT2Tokenizer
from evaluate import load as load_metric
import numpy as np

from utils.registry import MetricRegistry  # Ensure this registry is defined in your project

@MetricRegistry.register("rouge")
class RougeMetric:
    _metric = load_metric("rouge", trust_remote_code=True)  # Load metric once as class variable
    
    @staticmethod
    def compute_score(
        generated_ids: torch.Tensor, 
        labels: torch.Tensor,
        tokenizer: GPT2Tokenizer
    ) -> Dict[str, Union[float, List[str]]]:
        """
        Computes the average ROUGE-1 and ROUGE-L F1 scores over a batch.

        Args:
            generated_ids (torch.Tensor): Tensor of generated token ids.
            labels (torch.Tensor): Tensor of reference token ids.
            tokenizer (GPT2Tokenizer): Tokenizer instance.

        Returns:
            Dict[str, Union[float, List[str]]]: Dictionary with keys "rouge1" and "rougeL" representing their respective F1 scores, plus "predictions" and "references".
        """
        predictions = []
        references = []
        for gen, lab in zip(generated_ids, labels):
            predictions.append(tokenizer.decode(gen.tolist(), skip_special_tokens=True))
            references.append(tokenizer.decode(lab.tolist(), skip_special_tokens=True))
        result = RougeMetric._metric.compute(predictions=predictions, references=references)

        return {
            "rouge1": result["rouge1"], 
            "rougeL": result["rougeL"],
            "predictions": predictions,
            "references": references
        }

@MetricRegistry.register("bleu")
class BleuMetric:
    _metric = load_metric("bleu", trust_remote_code=True)  # Load metric once as class variable
    
    @staticmethod
    def compute_score(
        generated_ids: torch.Tensor, 
        labels: torch.Tensor,
        tokenizer: GPT2Tokenizer
    ) -> Dict[str, Union[float, List[str]]]:
        """
        Computes the average BLEU scores over a batch.

        Args:
            generated_ids (torch.Tensor): Tensor of generated token ids.
            labels (torch.Tensor): Tensor of reference token ids.
            tokenizer (GPT2Tokenizer): Tokenizer instance.

        Returns:
            Dict[str, Union[float, List[str]]]: Dictionary with keys "bleu1" for unigram precision and "bleu4" for overall BLEU score (up to 4-grams), plus "predictions" and "references".
        """
        predictions = []
        references = []
        for gen, lab in zip(generated_ids, labels):
            # Decode both predictions and references as strings.
            decoded_prediction = tokenizer.decode(gen.tolist(), skip_special_tokens=True)
            decoded_reference  = tokenizer.decode(lab.tolist(), skip_special_tokens=True)
            predictions.append(decoded_prediction)
            references.append(decoded_reference)
        
        # IMPORTANT: BLEU from evaluate expects, for each prediction, a list of possible references.
        # Here, we have one reference per prediction so we wrap each in its own list.
        formatted_references = [[ref] for ref in references]
        
        # Compute BLEU score. Note: the metric expects predictions as a list of strings and
        # references as a list of lists of strings.
        result = BleuMetric._metric.compute(
            predictions=predictions, 
            references=formatted_references,
            max_order=4,
            smooth=True
        )

        return {
            "bleu1": result['precisions'][0],
            "bleu4": result["bleu"],
            "predictions": predictions,
            "references": references
        }

@MetricRegistry.register("meteor")
class MeteorMetric:
    _metric = load_metric("meteor", trust_remote_code=True)  # Load metric once as class variable
    
    @staticmethod
    def compute_score(
        generated_ids: torch.Tensor, 
        labels: torch.Tensor,
        tokenizer: GPT2Tokenizer
    ) -> Dict[str, Union[float, List[str]]]:
        """
        Computes the average METEOR score over a batch.

        Args:
            generated_ids (torch.Tensor): Tensor of generated token ids.
            labels (torch.Tensor): Tensor of reference token ids.
            tokenizer: GPT2Tokenizer): Tokenizer instance.

        Returns:
            Dict[str, Union[float, List[str]]]: Dictionary with key "meteor" mapping to the average METEOR score for the batch, plus "predictions" and "references".
        """
        predictions = []
        references = []
        for gen, lab in zip(generated_ids, labels):
            predictions.append(tokenizer.decode(gen.tolist(), skip_special_tokens=True))
            references.append(tokenizer.decode(lab.tolist(), skip_special_tokens=True))
        result = MeteorMetric._metric.compute(predictions=predictions, references=references)
        return {
            "meteor": result["meteor"],
            "predictions": predictions,
            "references": references
        }

@MetricRegistry.register("bertscore")
class BertScoreMetric:
    _metric = None  # Lazy-loaded to avoid import errors at module import time
    _lang = "en"
    
    @staticmethod
    def compute_score(
        generated_ids: torch.Tensor,
        labels: torch.Tensor,
        tokenizer: GPT2Tokenizer
    ) -> Dict[str, Union[float, List[str]]]:
        """
        Computes mean BERTScore Precision/Recall/F1 over a batch.
        Returns hf-prec, hf-rec, hf-f1 to mirror the requested keys.
        """
        predictions: List[str] = []
        references: List[str] = []
        for gen, lab in zip(generated_ids, labels):
            predictions.append(tokenizer.decode(gen.tolist(), skip_special_tokens=True))
            references.append(tokenizer.decode(lab.tolist(), skip_special_tokens=True))

        # Lazy-load evaluate metric and dependency to avoid ModuleNotFoundError at import time
        if BertScoreMetric._metric is None:
            try:
                BertScoreMetric._metric = load_metric("bertscore")
            except Exception as e:
                raise RuntimeError("BERTScore metric unavailable. Please install 'bert-score' package.") from e

        # evaluate.bertscore returns lists per example; take mean over batch
        results = BertScoreMetric._metric.compute(
            predictions=predictions,
            references=references,
            lang=BertScoreMetric._lang
        )
        # results["precision"], ["recall"], ["f1"] are lists of floats
        prec = float(np.mean(results["precision"])) if isinstance(results["precision"], list) else float(results["precision"])  # type: ignore[arg-type]
        rec = float(np.mean(results["recall"])) if isinstance(results["recall"], list) else float(results["recall"])  # type: ignore[arg-type]
        f1 = float(np.mean(results["f1"])) if isinstance(results["f1"], list) else float(results["f1"])  # type: ignore[arg-type]

        return {
            "hf-prec": prec,
            "hf-rec": rec,
            "hf-f1": f1,
            "predictions": predictions,
            "references": references,
        }

def update_best_metric(
    metric_name: str,
    llm_metrics: dict[str, Union[float, list[str]]],
    best_metrics: dict[str, list[dict[str, Union[float, list[str]]]]],
    K: int = 5
) -> None:
    """
    Update the best (highest-scoring) entries for a given metric.
    The entry (score, predictions, references) is added or replaces an existing entry
    in the provided best_metrics dictionary.
    """
    score = llm_metrics[metric_name]
    predictions = llm_metrics["predictions"]
    references = llm_metrics["references"]
    prompts = llm_metrics.get("prompts", [])
    
    assert isinstance(score, float), f"Expected float for {metric_name}, got {type(score)}"
    assert isinstance(predictions, list), f"Expected list for predictions, got {type(predictions)}"
    assert isinstance(references, list), f"Expected list for references, got {type(references)}"

    entry: dict[str, Union[float, list[str]]] = {
        "score": score, 
        "predictions": predictions, 
        "references": references,
        "prompts": prompts
    }
    # Get or create the list for the metric
    best_list: list[dict[str, Union[float, list[str]]]] = best_metrics.setdefault(metric_name, [])
    
    if len(best_list) < K:
        best_list.append(entry)
    else:
        best_list.sort(key=lambda x: x["score"], reverse=True)
        # Replace if the new score is higher than the smallest among the best entries.
        last_score = best_list[-1]["score"]
        assert isinstance(last_score, float)
        if score > last_score:
            best_list[-1] = entry
    
    best_list.sort(key=lambda x: x["score"], reverse=True)
    
def update_worst_metric(
    metric_name: str,
    llm_metrics: dict[str, Union[float, list[str]]],
    worst_metrics: dict[str, list[dict[str, Union[float, list[str]]]]],
    K: int = 5
) -> None:
    """
    Update the worst (lowest-scoring) entries for a given metric.
    The entry (score, predictions, references) is added or replaces an existing entry
    in the provided worst_metrics dictionary.
    """
    score = llm_metrics[metric_name]
    predictions = llm_metrics["predictions"]
    references = llm_metrics["references"]
    prompts = llm_metrics.get("prompts", [])
    
    assert isinstance(score, float), f"Expected float for {metric_name}, got {type(score)}"
    assert isinstance(predictions, list), f"Expected list for predictions, got {type(predictions)}"
    assert isinstance(references, list), f"Expected list for references, got {type(references)}"

    entry: dict[str, Union[float, list[str]]] = {
        "score": score, 
        "predictions": predictions, 
        "references": references,
        "prompts": prompts
    }
    worst_list = worst_metrics.setdefault(metric_name, [])
    
    if len(worst_list) < K:
        worst_list.append(entry)
    else:
        worst_list.sort(key=lambda x: x["score"])
        # Replace if the new score is lower than the highest in the worst list.
        last_score = worst_list[-1]["score"]
        assert isinstance(last_score, float)
        if score < last_score:
            worst_list[-1] = entry
    
    worst_list.sort(key=lambda x: x["score"])
    
def update_random_batch_metric(
    metric_name: str,
    llm_metrics: dict[str, Union[float, list[str]]],
    random_metrics: dict[str, list[dict[str, Union[float, list[str]]]]],
    k: int = 5
) -> None:
    """
    Update the random batch metric.
    """
    score = llm_metrics[metric_name]
    predictions = llm_metrics["predictions"]
    references = llm_metrics["references"]
    prompts = llm_metrics.get("prompts", [])
    
    assert isinstance(score, float), f"Expected float for {metric_name}, got {type(score)}"
    assert isinstance(predictions, list), f"Expected list for predictions, got {type(predictions)}"
    assert isinstance(references, list), f"Expected list for references, got {type(references)}"
    
    entry: dict[str, Union[float, list[str]]] = {
        "score": score, 
        "predictions": predictions, 
        "references": references,
        "prompts": prompts
    }
    
    # Get or create the list for the metric
    random_list: list[dict[str, Union[float, list[str]]]] = random_metrics.setdefault(metric_name, [])
    
    if len(random_list) < k:
        random_list.append(entry)