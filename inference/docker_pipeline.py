#!/usr/bin/env python3
"""
ECG Tokenizer Docker Inference Pipeline.

Main orchestration script for running the complete ECG inference pipeline:
1. Load input parquet and apply PSA normalization
2. Run BERT 77-class classification on reports
3. Run ECG tokenizer + EfficientNet classification
4. Generate reports using GPT2
5. Generate Q&A pairs
6. Generate answers using MedGemma
7. Compute LLM metrics (ROUGE, BLEU, METEOR)
8. Run LLM-as-a-Judge evaluation
9. Output comprehensive JSON results
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict
from datetime import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

from inference.pipeline_config import PipelineConfig
from inference.psa_normalizer import PSANormalizer, create_psa_normalizer
from inference.llm_judge_wrapper import LLMJudgeWrapper, create_llm_judge
from utils.metrics.ecg_metrics import compute_metrics, compute_metrics_binary
from utils.constants import ECG_PATTERNS


class ECGInferenceDataset(Dataset):
    """Dataset for ECG inference pipeline."""
    
    def __init__(
        self,
        parquet_path: str,
        psa_normalizer: Optional[PSANormalizer] = None,
        waveform_path_column: str = "waveform_path",
        report_column: str = "report",
        waveform_name_column: str = "waveform_name"
    ):
        self.df = pd.read_parquet(parquet_path)
        self.psa_normalizer = psa_normalizer
        self.waveform_path_column = waveform_path_column
        self.report_column = report_column
        self.waveform_name_column = waveform_name_column
        
        print(f"Loaded {len(self.df)} samples from {parquet_path}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        
        waveform_path = row.get(self.waveform_path_column, "")
        waveform = None
        
        if waveform_path and os.path.exists(waveform_path):
            try:
                waveform = np.load(waveform_path)
                if self.psa_normalizer:
                    waveform = self.psa_normalizer.normalize_waveform(waveform)
                else:
                    if waveform.ndim == 3:
                        waveform = waveform.squeeze(-1)
                    if waveform.shape[0] != 12:
                        waveform = waveform.T
            except Exception as e:
                print(f"Error loading waveform {waveform_path}: {e}")
                waveform = None
        
        return {
            "idx": idx,
            "waveform": waveform,
            "waveform_path": waveform_path,
            "report": row.get(self.report_column, ""),
            "waveform_name": row.get(self.waveform_name_column, f"ecg_{idx}"),
            "row_data": row.to_dict()
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for DataLoader."""
    valid_waveforms = []
    valid_indices = []
    
    for item in batch:
        if item["waveform"] is not None:
            valid_waveforms.append(item["waveform"])
            valid_indices.append(item["idx"])
    
    waveform_tensor = None
    if valid_waveforms:
        waveform_tensor = torch.tensor(
            np.stack(valid_waveforms, axis=0),
            dtype=torch.float32
        )
    
    return {
        "waveforms": waveform_tensor,
        "valid_indices": valid_indices,
        "batch": batch
    }


class ECGInferencePipeline:
    """Main inference pipeline for ECG tokenizer."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = self._setup_device()
        
        self.psa_normalizer = None
        self.bert_classifier = None
        self.ecg_tokenizer = None
        self.gpt2_decoder = None
        self.medgemma_decoder = None
        self.llm_judge = None
        
        # Cache for QA generation
        self.prompt_maker = None
        self.answer_generator = None
        
        self._load_models()
    
    def _setup_device(self) -> torch.device:
        """Set up the compute device."""
        if torch.cuda.is_available() and "cuda" in self.config.device:
            return torch.device(self.config.device)
        elif self.config.cpu_fallback:
            print("CUDA not available, falling back to CPU")
            return torch.device("cpu")
        else:
            raise RuntimeError("CUDA not available and cpu_fallback is False")
    
    def _load_models(self) -> None:
        """Load all required models."""
        print("Loading models...")
        
        if self.config.apply_psa_normalization:
            self.psa_normalizer = create_psa_normalizer(
                sampling_rate=self.config.sampling_rate,
                target_length=self.config.waveform_length,
                num_leads=self.config.num_leads,
                region=self.config.psa_region
            )
            print("  PSA normalizer initialized")
        
        self._load_bert_classifier()
        self._load_ecg_tokenizer()
        
        if self.config.run_llm_judge:
            try:
                self.llm_judge = create_llm_judge()
                print("  LLM Judge initialized")
            except Exception as e:
                print(f"  Warning: Could not initialize LLM Judge: {e}")
    
    def _load_bert_classifier(self) -> None:
        """Load BERT classifier for 77-class report classification."""
        try:
            from models.bert_classifier import BertClassifier
            
            if os.path.exists(self.config.bert_checkpoint):
                self.bert_classifier = BertClassifier(
                    model_path=self.config.bert_checkpoint,
                    num_classes=self.config.num_classes
                )
                self.bert_classifier.to(self.device)
                self.bert_classifier.eval()
                print(f"  BERT classifier loaded from {self.config.bert_checkpoint}")
        except Exception as e:
            print(f"  Warning: Could not load BERT classifier: {e}")
    
    def _load_ecg_tokenizer(self) -> None:
        """Load ECG tokenizer wrapper (encoder + quantizer only for embeddings)."""
        try:
            from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper
            from utils.enums import DecoderMode
            
            if os.path.exists(self.config.tokenizer_checkpoint):
                checkpoint = torch.load(
                    self.config.tokenizer_checkpoint,
                    map_location="cpu",
                    weights_only=False
                )
                
                ckpt_config = checkpoint.get("config", None)
                
                if ckpt_config:
                    # Detect the decoder mode from checkpoint
                    decoder_mode_str = getattr(ckpt_config, "decoder_mode", "reconstruction")
                    if isinstance(decoder_mode_str, str):
                        decoder_mode = DecoderMode(decoder_mode_str)
                    else:
                        decoder_mode = decoder_mode_str
                    
                    decoder_name = getattr(ckpt_config, "decoder_name", "Conv_Decoder")
                    
                    # Build model with appropriate decoder mode
                    if decoder_mode == DecoderMode.LLM:
                        from transformers import AutoTokenizer
                        tokenizer_name = getattr(ckpt_config, "tokenizer_name", "gpt2")
                        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                        if tokenizer.pad_token is None:
                            tokenizer.pad_token = tokenizer.eos_token
                        
                        self.ecg_tokenizer = ECG_Tokenizer_Wrapper(
                            encoder_name=getattr(ckpt_config, "encoder_name", "Residual_Conv_Encoder"),
                            quantizer_name=getattr(ckpt_config, "quantizer_name", "ECG_Tokenizer_Quantizer"),
                            decoder_name=decoder_name,
                            num_quantizers=getattr(ckpt_config, "num_quantizers", 8),
                            codebook_size=getattr(ckpt_config, "codebook_size", 512),
                            decoder_mode=DecoderMode.LLM,
                            huggingface_model_name=getattr(ckpt_config, "huggingface_model_name", "gpt2"),
                            llm_input_embedding_size=getattr(ckpt_config, "llm_input_embedding_size", 768),
                            bridge_name=getattr(ckpt_config, "bridge_name", "GPT2_SimpleEmbeddingBridge"),
                            tokenizer=tokenizer,
                        )
                    else:
                        # For reconstruction or classification mode, just load encoder + quantizer
                        self.ecg_tokenizer = ECG_Tokenizer_Wrapper(
                            encoder_name=getattr(ckpt_config, "encoder_name", "Residual_Conv_Encoder"),
                            quantizer_name=getattr(ckpt_config, "quantizer_name", "ECG_Tokenizer_Quantizer"),
                            decoder_name=decoder_name,
                            num_quantizers=getattr(ckpt_config, "num_quantizers", 8),
                            codebook_size=getattr(ckpt_config, "codebook_size", 512),
                            decoder_mode=decoder_mode,
                        )
                    
                    state_dict = checkpoint.get("model_state_dict", checkpoint)
                    self.ecg_tokenizer.load_state_dict(state_dict, strict=False)
                    print(f"  ECG tokenizer loaded (mode: {decoder_mode}) from {self.config.tokenizer_checkpoint}")
                else:
                    # Fallback: load with minimal config
                    self.ecg_tokenizer = ECG_Tokenizer_Wrapper(
                        decoder_mode=DecoderMode.RECONSTRUCTION
                    )
                    state_dict = checkpoint.get("model_state_dict", checkpoint)
                    self.ecg_tokenizer.load_state_dict(state_dict, strict=False)
                    print(f"  ECG tokenizer loaded (reconstruction mode) from {self.config.tokenizer_checkpoint}")
                
                self.ecg_tokenizer.to(self.device)
                self.ecg_tokenizer.eval()
                
        except Exception as e:
            print(f"  Warning: Could not load ECG tokenizer: {e}")
            import traceback
            traceback.print_exc()
    
    def run_bert_classification(
        self,
        reports: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Run BERT 77-class classification on reports.
        
        Args:
            reports: List of ECG report texts
            
        Returns:
            List of classification results
        """
        if self.bert_classifier is None:
            return [{"error": "BERT classifier not loaded"}] * len(reports)
        
        results = []
        
        with torch.no_grad():
            for report in tqdm(reports, desc="BERT Classification"):
                if not report:
                    results.append({"predictions": [], "probabilities": []})
                    continue
                
                try:
                    inputs = self.bert_classifier.preprocessing(report)
                    
                    # Call the underlying BERT model directly to avoid squeeze issues
                    input_ids = inputs['input_ids'].to(self.device)
                    attention_mask = inputs['attention_mask'].to(self.device)
                    token_type_ids = inputs['token_type_ids'].to(self.device)
                    
                    # Call the BERT model directly (not through the wrapper)
                    outputs = self.bert_classifier.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )
                    
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    
                    # Remove batch dimension if present
                    if logits.dim() > 1:
                        logits = logits.squeeze(0)
                    
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    
                    predictions = (probs > self.config.classification_threshold).astype(int).tolist()
                    
                    results.append({
                        "predictions": predictions,
                        "probabilities": probs.tolist()
                    })
                except Exception as e:
                    print(f"  Error classifying report: {e}")
                    results.append({"predictions": [], "probabilities": [], "error": str(e)})
        
        return results
    
    def generate_reports(
        self,
        waveforms: torch.Tensor
    ) -> List[str]:
        """
        Generate ECG reports using GPT2 decoder.
        
        Args:
            waveforms: Batch of ECG waveforms [B, 12, 2500]
            
        Returns:
            List of generated report strings
        """
        if self.ecg_tokenizer is None:
            return [""] * waveforms.shape[0]
        
        reports = []
        
        with torch.no_grad():
            waveforms = waveforms.to(self.device)
            
            for i in range(waveforms.shape[0]):
                try:
                    wf = waveforms[i:i+1]
                    generated_ids = self.ecg_tokenizer.generate_report(
                        x=wf,
                        max_token_length=self.config.max_report_length
                    )
                    
                    if hasattr(self.ecg_tokenizer.decoder, 'tokenizer'):
                        tokenizer = self.ecg_tokenizer.decoder.tokenizer
                    else:
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained("gpt2")
                    
                    report = tokenizer.decode(
                        generated_ids[0],
                        skip_special_tokens=True
                    )
                    reports.append(report)
                    
                except Exception as e:
                    print(f"Error generating report: {e}")
                    reports.append("")
        
        return reports
    
    def generate_qa_pairs(
        self,
        row_data: Dict[str, Any],
        dataset: str = "mimic"
    ) -> List[Dict[str, str]]:
        """
        Generate Q&A pairs for an ECG sample.
        
        Args:
            row_data: Row data from the parquet file
            dataset: Dataset name for prompt generation
            
        Returns:
            List of {question, answer, category} dicts
        """
        try:
            # Initialize generators once and cache them
            if self.prompt_maker is None or self.answer_generator is None:
                sys.path.insert(0, str(REPO_ROOT / "dataset_generation"))
                from ecg_prompt_maker import ECGPromptMaker
                from ecg_answer_generator import ECGAnswerGenerator
                
                self.prompt_maker = ECGPromptMaker(dataset=dataset)
                self.answer_generator = ECGAnswerGenerator(dataset=dataset)
            
            prompt_maker = self.prompt_maker
            answer_generator = self.answer_generator
            
            row_series = pd.Series(row_data)
            
            # generate_prompts_for_ecg returns List[Tuple[str, str, float]]
            # where each tuple is (prompt_text, category, weight)
            prompts = prompt_maker.generate_prompts_for_ecg(row_series)
            
            # Sort by weight (descending) and limit to max_prompts_per_ecg
            prompts_sorted = sorted(prompts, key=lambda x: x[2], reverse=True)
            prompts_limited = prompts_sorted[:self.config.max_prompts_per_ecg]
            
            qa_pairs = []
            for prompt_text, category, weight in prompts_limited:
                # Determine prompt type from category
                if category.startswith('category_'):
                    prompt_type = 'category'
                elif category.startswith('localization_'):
                    prompt_type = 'localization'
                elif category == 'interpretation':
                    prompt_type = 'interpretation'
                elif category == 'classification':
                    prompt_type = 'classification'
                elif category == 'json_interpretation':
                    prompt_type = 'json_interpretation'
                else:
                    prompt_type = 'interpretation'
                
                # Add prompt info to row for answer generation
                row_with_prompt = row_series.copy()
                row_with_prompt['prompt_category'] = category
                row_with_prompt['prompt_type'] = prompt_type
                row_with_prompt['prompt'] = prompt_text
                
                answer = answer_generator.generate_answer(row_with_prompt)
                
                qa_pairs.append({
                    "question": prompt_text,
                    "answer": answer,
                    "category": category,
                    "type": prompt_type,
                    "weight": weight
                })
            
            return qa_pairs
            
        except Exception as e:
            print(f"Error generating QA pairs: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def compute_text_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute ROUGE, BLEU, METEOR metrics.
        
        Args:
            predictions: List of generated texts
            references: List of ground truth texts
            
        Returns:
            Dictionary of metric scores
        """
        try:
            from utils.metrics.aggregate_text_metrics import aggregate_text_metrics
            return aggregate_text_metrics(predictions, references)
        except Exception as e:
            print(f"Error computing metrics: {e}")
            return {}
    
    def _compute_classification_metrics(
        self,
        all_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute classification metrics (AUC, AUPRC, F1) for BERT and EfficientNet.
        
        Args:
            all_results: List of result dictionaries from pipeline
            
        Returns:
            Dictionary containing classification metrics
        """
        metrics = {
            "bert": {},
            "efficientnet": {}
        }
        
        # Extract ground truth labels and predictions
        # We need to get ground truth from the parquet data
        try:
            # Collect BERT predictions and probabilities
            bert_predictions = []
            bert_probabilities = []
            gt_labels_list = []
            
            for result in all_results:
                bert_result = result.get("bert_classification", {})
                if "predictions" in bert_result and "probabilities" in bert_result:
                    if len(bert_result["predictions"]) == 77:
                        bert_predictions.append(bert_result["predictions"])
                        bert_probabilities.append(bert_result["probabilities"])
                        
                        # Try to get ground truth from row_data
                        row_data = result.get("row_data", {})
                        if row_data:
                            gt_labels = []
                            for pattern in ECG_PATTERNS:
                                val = row_data.get(pattern, 0)
                                if pd.notna(val):
                                    gt_labels.append(int(val) if float(val) >= 1 else 0)
                                else:
                                    gt_labels.append(0)
                            
                            if len(gt_labels) == 77:
                                gt_labels_list.append(gt_labels)
            
            # Compute BERT metrics if we have data
            if bert_probabilities and gt_labels_list and len(bert_probabilities) == len(gt_labels_list):
                print(f"\nComputing BERT classification metrics ({len(bert_probabilities)} samples)...")
                
                df_gt = pd.DataFrame(gt_labels_list, columns=ECG_PATTERNS)
                df_pred = pd.DataFrame(bert_probabilities, columns=ECG_PATTERNS)
                
                bert_metrics = compute_metrics(df_gt, df_pred)
                metrics["bert"] = bert_metrics
                
                # Summary statistics
                category_aucs = []
                for category in ["RHYTHM", "CONDUCTION", "INFARCT, ISCHEMIA", 
                               "CHAMBER ENLARGEMENT", "PERICARDITIS", "OTHER"]:
                    if category in bert_metrics and "macro_auc" in bert_metrics[category]:
                        auc = bert_metrics[category]["macro_auc"]
                        if not np.isnan(auc):
                            category_aucs.append(auc)
                
                if category_aucs:
                    metrics["bert"]["overall_macro_auc"] = float(np.mean(category_aucs))
                    print(f"  BERT Overall Macro AUC: {metrics['bert']['overall_macro_auc']:.4f}")
            
            # TODO: Add EfficientNet metrics when classification checkpoint is available
            
        except Exception as e:
            print(f"Error computing classification metrics: {e}")
            import traceback
            traceback.print_exc()
        
        return metrics
    
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete inference pipeline.
        
        Returns:
            Dictionary containing all results
        """
        print(f"\n{'='*60}")
        print("ECG Tokenizer Inference Pipeline")
        print(f"{'='*60}")
        print(f"Input: {self.config.input_parquet}")
        print(f"Output: {self.config.output_json}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        dataset = ECGInferenceDataset(
            parquet_path=self.config.input_parquet,
            psa_normalizer=self.psa_normalizer,
            waveform_path_column=self.config.waveform_path_column,
            report_column=self.config.report_column,
            waveform_name_column=self.config.waveform_name_column
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn
        )
        
        all_results = []
        all_predictions = []
        all_references = []
        
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Processing")):
            batch_results = self._process_batch(batch_data)
            all_results.extend(batch_results)
            
            for result in batch_results:
                if result.get("generated_report"):
                    all_predictions.append(result["generated_report"])
                    all_references.append(result.get("ground_truth_report", ""))
        
        text_metrics = {}
        if all_predictions and all_references:
            print("\nComputing text metrics...")
            text_metrics = self.compute_text_metrics(all_predictions, all_references)
        
        # Compute classification metrics (BERT and EfficientNet)
        classification_metrics = self._compute_classification_metrics(all_results)
        
        llm_judge_results = None
        if self.config.run_llm_judge and self.llm_judge:
            print("\nRunning LLM Judge evaluation...")
            qa_results = []
            for result in all_results:
                for qa in result.get("qa_results", []):
                    qa_results.append({
                        "question": qa.get("question", ""),
                        "ground_truth": qa.get("ground_truth_answer", ""),
                        "prediction": qa.get("predicted_answer", ""),
                        "category": qa.get("category", "unknown"),
                        "waveform_name": result.get("waveform_name", "")
                    })
            
            if qa_results:
                llm_judge_results = self.llm_judge.evaluate_qa_pairs(qa_results)
        
        output = {
            "metadata": {
                "pipeline_version": "1.0.0",
                "timestamp": datetime.now().isoformat(),
                "config": self.config.to_dict(),
                "num_samples": len(all_results)
            },
            "results": all_results,
            "aggregate_metrics": {
                "text_metrics": text_metrics,
                "classification_metrics": classification_metrics,
                "llm_judge": llm_judge_results
            }
        }
        
        output_path = Path(self.config.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\nResults saved to {output_path}")
        
        return output
    
    def _process_batch(
        self,
        batch_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process a single batch of ECG samples."""
        results = []
        
        waveforms = batch_data.get("waveforms")
        batch = batch_data.get("batch", [])
        
        # Step 1: Extract tokenizer embeddings and get classification
        tokenizer_embeddings = []
        efficientnet_classifications = []
        if waveforms is not None and self.ecg_tokenizer is not None:
            with torch.no_grad():
                waveforms_device = waveforms.to(self.device)
                
                # Get embeddings from encoder + quantizer
                try:
                    features = self.ecg_tokenizer.encoder(waveforms_device)
                    quantized, indices, commit_loss = self.ecg_tokenizer.quantizer(features)
                    
                    # Store embeddings info
                    for i in range(waveforms.shape[0]):
                        embedding_info = {
                            "quantized_shape": list(quantized[i].shape),
                            "indices_shape": list(indices[i].shape),
                            "num_codebooks": indices.shape[-1] if indices.dim() > 2 else 1,
                            "sequence_length": quantized.shape[-1],
                            "embedding_dim": quantized.shape[1]
                        }
                        tokenizer_embeddings.append(embedding_info)
                    
                    # Get EfficientNet classification if decoder supports it
                    if hasattr(self.ecg_tokenizer, 'decoder') and hasattr(self.ecg_tokenizer.decoder, 'forward'):
                        try:
                            # Check if this is a classification decoder
                            if self.config.decoder_mode == "classification" or hasattr(self.ecg_tokenizer.decoder, 'classifier'):
                                logits = self.ecg_tokenizer.decoder(quantized)
                                probs = torch.sigmoid(logits).cpu().numpy()
                                
                                for i in range(probs.shape[0]):
                                    predictions = (probs[i] > self.config.classification_threshold).astype(int).tolist()
                                    efficientnet_classifications.append({
                                        "predictions": predictions,
                                        "probabilities": probs[i].tolist()
                                    })
                        except Exception as e:
                            if self.config.verbose:
                                print(f"  Note: EfficientNet classification not available: {e}")
                
                except Exception as e:
                    if self.config.verbose:
                        print(f"  Warning: Could not extract embeddings: {e}")
        
        # Step 2: Generate reports using GPT2
        generated_reports = []
        if waveforms is not None and self.ecg_tokenizer is not None:
            generated_reports = self.generate_reports(waveforms)
        
        # Step 3: Run BERT classification on reports
        reports = [item.get("report", "") for item in batch]
        bert_results = []
        if self.bert_classifier is not None and reports:
            bert_results = self.run_bert_classification(reports)
        
        # Step 4: Assemble results
        for i, item in enumerate(batch):
            result = {
                "waveform_name": item.get("waveform_name", ""),
                "waveform_path": item.get("waveform_path", ""),
                "ground_truth_report": item.get("report", ""),
                "row_data": item.get("row_data", {}),  # Store for metrics computation
            }
            
            # Add tokenizer embeddings
            if i < len(tokenizer_embeddings):
                result["tokenizer_embeddings"] = tokenizer_embeddings[i]
            
            # Add EfficientNet classification
            if i < len(efficientnet_classifications):
                result["efficientnet_classification"] = efficientnet_classifications[i]
            
            # Add BERT classification
            if i < len(bert_results):
                result["bert_classification"] = bert_results[i]
            
            # Add generated report
            if i < len(generated_reports):
                result["generated_report"] = generated_reports[i]
            
            # Generate QA pairs
            qa_pairs = self.generate_qa_pairs(item.get("row_data", {}))
            result["qa_results"] = []
            
            for qa in qa_pairs:
                qa_result = {
                    "question": qa.get("question", ""),
                    "ground_truth_answer": qa.get("answer", ""),
                    "category": qa.get("category", "unknown"),
                    "predicted_answer": ""
                }
                result["qa_results"].append(qa_result)
            
            results.append(result)
        
        return results


def main():
    """Main entry point for the inference pipeline."""
    parser = argparse.ArgumentParser(
        description="ECG Tokenizer Docker Inference Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (YAML or JSON)"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input parquet file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (cuda:0, cpu, etc.)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--no-psa",
        action="store_true",
        help="Skip PSA normalization"
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM Judge evaluation"
    )
    
    args = parser.parse_args()
    
    if args.config:
        config = PipelineConfig.from_config_file(args.config)
    else:
        config = PipelineConfig(
            input_parquet=args.input or "/app/inputs/input.parquet",
            output_json=args.output or "/app/outputs/results.json"
        )
    
    if args.input:
        config.input_parquet = args.input
    if args.output:
        config.output_json = args.output
    if args.device:
        config.device = args.device
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.no_psa:
        config.apply_psa_normalization = False
    if args.no_judge:
        config.run_llm_judge = False
    
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    pipeline = ECGInferencePipeline(config)
    results = pipeline.run_pipeline()
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print(f"Processed {results['metadata']['num_samples']} samples")
    
    if results.get("aggregate_metrics", {}).get("text_metrics"):
        print("\nText Metrics:")
        for metric, value in results["aggregate_metrics"]["text_metrics"].items():
            print(f"  {metric}: {value:.4f}")
    
    print("="*60)


if __name__ == "__main__":
    main()
