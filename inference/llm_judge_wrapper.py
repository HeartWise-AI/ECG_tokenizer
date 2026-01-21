#!/usr/bin/env python3
"""
LLM Judge Wrapper for ECG answer evaluation.

This module provides integration with the ECG_LLM_Judge repository
(https://github.com/HeartWise-AI/ECG_LLM_Judge) for evaluating
generated ECG interpretations and answers.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import tempfile


@dataclass
class JudgeResult:
    """Result from LLM Judge evaluation."""
    score: float
    category: str
    reasoning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMJudgeWrapper:
    """
    Wrapper for the ECG_LLM_Judge repository.
    
    This class provides an interface to run LLM-as-a-Judge evaluation
    on generated ECG interpretations and Q&A answers.
    """
    
    def __init__(
        self,
        judge_repo_path: str = "/app/ECG_LLM_Judge",
        config_path: Optional[str] = None,
        api_key_env_var: str = "OPENAI_API_KEY"
    ):
        """
        Initialize the LLM Judge wrapper.
        
        Args:
            judge_repo_path: Path to the cloned ECG_LLM_Judge repository
            config_path: Path to judge configuration file
            api_key_env_var: Environment variable containing the API key
        """
        self.judge_repo_path = Path(judge_repo_path)
        self.config_path = config_path
        self.api_key_env_var = api_key_env_var
        
        self._validate_installation()
    
    def _validate_installation(self) -> None:
        """Validate that ECG_LLM_Judge is properly installed."""
        if not self.judge_repo_path.exists():
            raise FileNotFoundError(
                f"ECG_LLM_Judge repository not found at {self.judge_repo_path}. "
                "Please clone it from https://github.com/HeartWise-AI/ECG_LLM_Judge"
            )
        
        if self.api_key_env_var not in os.environ:
            print(f"Warning: {self.api_key_env_var} not set. LLM Judge may not work.")
    
    def prepare_evaluation_data(
        self,
        predictions: List[str],
        ground_truths: List[str],
        questions: List[str],
        categories: List[str],
        waveform_names: List[str]
    ) -> Dict[str, Any]:
        """
        Prepare data for LLM Judge evaluation.
        
        Args:
            predictions: List of generated answers
            ground_truths: List of ground truth answers
            questions: List of questions
            categories: List of question categories
            waveform_names: List of waveform identifiers
            
        Returns:
            Dictionary formatted for ECG_LLM_Judge input
        """
        eval_data = {
            "samples": []
        }
        
        for i, (pred, gt, q, cat, wf) in enumerate(zip(
            predictions, ground_truths, questions, categories, waveform_names
        )):
            sample = {
                "id": i,
                "waveform_name": wf,
                "question": q,
                "category": cat,
                "ground_truth": gt,
                "prediction": pred
            }
            eval_data["samples"].append(sample)
        
        return eval_data
    
    def run_evaluation(
        self,
        eval_data: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run LLM Judge evaluation on prepared data.
        
        Args:
            eval_data: Evaluation data prepared by prepare_evaluation_data
            output_path: Optional path to save results
            
        Returns:
            Dictionary containing evaluation results
        """
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(eval_data, f)
            input_path = f.name
        
        try:
            if output_path is None:
                output_path = tempfile.mktemp(suffix='_results.json')
            
            judge_script = self.judge_repo_path / "evaluate.py"
            
            if judge_script.exists():
                cmd = [
                    sys.executable,
                    str(judge_script),
                    "--input", input_path,
                    "--output", output_path
                ]
                
                if self.config_path:
                    cmd.extend(["--config", self.config_path])
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(self.judge_repo_path)
                )
                
                if result.returncode != 0:
                    print(f"LLM Judge error: {result.stderr}")
                    return self._fallback_evaluation(eval_data)
                
                with open(output_path, 'r') as f:
                    results = json.load(f)
                
                return results
            else:
                return self._run_inline_evaluation(eval_data)
                
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
    
    def _run_inline_evaluation(
        self,
        eval_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run evaluation using inline Python import if available.
        
        Args:
            eval_data: Evaluation data
            
        Returns:
            Evaluation results
        """
        try:
            sys.path.insert(0, str(self.judge_repo_path))
            from ecg_llm_judge import ECGLLMJudge
            
            judge = ECGLLMJudge()
            results = judge.evaluate(eval_data["samples"])
            
            return self._format_results(results, eval_data)
            
        except ImportError:
            print("ECG_LLM_Judge module not found, using fallback evaluation")
            return self._fallback_evaluation(eval_data)
    
    def _fallback_evaluation(
        self,
        eval_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fallback evaluation when LLM Judge is not available.
        Uses text similarity metrics instead.
        
        Args:
            eval_data: Evaluation data
            
        Returns:
            Fallback evaluation results
        """
        from difflib import SequenceMatcher
        
        results = {
            "overall_score": 0.0,
            "category_aggregates": {},
            "sample_results": [],
            "metadata": {
                "method": "fallback_text_similarity",
                "warning": "LLM Judge not available, using text similarity"
            }
        }
        
        category_scores = {}
        
        for sample in eval_data.get("samples", []):
            pred = sample.get("prediction", "")
            gt = sample.get("ground_truth", "")
            category = sample.get("category", "unknown")
            
            similarity = SequenceMatcher(None, pred.lower(), gt.lower()).ratio()
            
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(similarity)
            
            results["sample_results"].append({
                "id": sample.get("id"),
                "waveform_name": sample.get("waveform_name"),
                "score": similarity,
                "category": category
            })
        
        for category, scores in category_scores.items():
            results["category_aggregates"][category] = {
                "count": len(scores),
                "mean_score": sum(scores) / len(scores) if scores else 0.0,
                "min_score": min(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0
            }
        
        all_scores = [r["score"] for r in results["sample_results"]]
        results["overall_score"] = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        return results
    
    def _format_results(
        self,
        raw_results: Any,
        eval_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Format raw results from ECG_LLM_Judge into standard format.
        
        Args:
            raw_results: Raw results from the judge
            eval_data: Original evaluation data
            
        Returns:
            Formatted results dictionary
        """
        if isinstance(raw_results, dict):
            return raw_results
        
        formatted = {
            "overall_score": 0.0,
            "category_aggregates": {},
            "sample_results": [],
            "total_examples": len(eval_data.get("samples", []))
        }
        
        if hasattr(raw_results, '__iter__'):
            scores = []
            for i, result in enumerate(raw_results):
                score = float(result) if isinstance(result, (int, float)) else 0.0
                scores.append(score)
                
                sample = eval_data["samples"][i] if i < len(eval_data["samples"]) else {}
                formatted["sample_results"].append({
                    "id": sample.get("id", i),
                    "score": score,
                    "category": sample.get("category", "unknown")
                })
            
            formatted["overall_score"] = sum(scores) / len(scores) if scores else 0.0
        
        return formatted
    
    def evaluate_qa_pairs(
        self,
        qa_results: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convenience method to evaluate Q&A pairs directly.
        
        Args:
            qa_results: List of dicts with keys:
                - question, ground_truth, prediction, category, waveform_name
            output_path: Optional path to save results
            
        Returns:
            Evaluation results
        """
        predictions = [r.get("prediction", "") for r in qa_results]
        ground_truths = [r.get("ground_truth", "") for r in qa_results]
        questions = [r.get("question", "") for r in qa_results]
        categories = [r.get("category", "unknown") for r in qa_results]
        waveform_names = [r.get("waveform_name", f"wf_{i}") for i, r in enumerate(qa_results)]
        
        eval_data = self.prepare_evaluation_data(
            predictions, ground_truths, questions, categories, waveform_names
        )
        
        return self.run_evaluation(eval_data, output_path)


def create_llm_judge(
    judge_repo_path: str = "/app/ECG_LLM_Judge",
    config_path: Optional[str] = None
) -> LLMJudgeWrapper:
    """
    Factory function to create an LLM Judge wrapper.
    
    Args:
        judge_repo_path: Path to ECG_LLM_Judge repository
        config_path: Optional path to configuration file
        
    Returns:
        Configured LLMJudgeWrapper instance
    """
    return LLMJudgeWrapper(
        judge_repo_path=judge_repo_path,
        config_path=config_path
    )
