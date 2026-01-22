#!/usr/bin/env python3
"""
ECG Tokenizer Inference Pipeline.

This module implements the core inference logic:
1. BERT classifies text reports → 77 class probabilities (GROUND TRUTH)
2. ECG Tokenizer extracts embeddings from signals
3. EfficientNet classifies embeddings → 77 class probabilities (EVALUATED)
4. Metrics: How well does signal-based classification match text-based?

Key principle: BERT predictions from text reports serve as ground truth.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
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
from inference.files_handler import ECGFileHandler
from utils.preprocessing.ecg_signal_processor import ECGSignalProcessor
from utils.metrics.ecg_metrics import compute_metrics
from utils.constants import ECG_PATTERNS, PTBXL_POWER_RATIO, ECG_FILE_NAME_COLUMN, DIAGNOSIS_COLUMN


class ECGDataset(Dataset):
    """
    Dataset for ECG inference that loads signals from disk.
    
    Standard columns (following DeepECG_Docker pattern):
    - diagnosis: Text report/diagnosis for BERT classification
    - ecg_path: Absolute/relative path to ECG file
    """
    
    def __init__(
        self,
        parquet_path: str,
        ecg_signals_path: str = "/app/ecg_signals"
    ):
        self.df = pd.read_parquet(parquet_path)
        self.ecg_signals_path = ecg_signals_path
        
        # Use standardized column names
        if 'ecg_path' in self.df.columns:
            self.ecg_paths = self.df['ecg_path'].tolist()
        else:
            self.ecg_paths = []
        
        # Diagnosis (text reports)
        self.diagnoses = self.df[DIAGNOSIS_COLUMN].tolist() if DIAGNOSIS_COLUMN in self.df.columns else []
        
        # ECG file names as identifiers
        if 'ecg_path' in self.df.columns:
            self.ecg_names = [Path(p).name if p else f"ecg_{i}" for i, p in enumerate(self.ecg_paths)]
        else:
            self.ecg_names = [f"ecg_{i}" for i in range(len(self.df))]
        
        print(f"Loaded {len(self.df)} samples from {parquet_path}")
        print(f"  - ECG paths: {len([p for p in self.ecg_paths if p])} non-empty")
        print(f"  - Diagnoses: {len([r for r in self.diagnoses if r])} non-empty")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ecg_path = self.ecg_paths[idx] if idx < len(self.ecg_paths) else ""
        diagnosis = self.diagnoses[idx] if idx < len(self.diagnoses) else ""
        ecg_name = self.ecg_names[idx] if idx < len(self.ecg_names) else f"ecg_{idx}"
        
        waveform = None
        
        if ecg_path and os.path.exists(ecg_path):
            try:
                # Handle preprocessed .base64 files
                if ecg_path.endswith('.base64'):
                    waveform = ECGFileHandler.load_ecg_signal(ecg_path)
                    # ECGFileHandler returns (2500, 12), transpose to (12, 2500)
                    if waveform.shape == (2500, 12):
                        waveform = waveform.T
                # Handle WFDB files (.hea extension)
                elif ecg_path.endswith('.hea'):
                    import wfdb
                    record_path = ecg_path[:-4]
                    record = wfdb.rdrecord(record_path)
                    waveform = record.p_signal  # Shape: (2500, 12)
                    if waveform.shape == (2500, 12):
                        waveform = waveform.T
                # Handle NPY files
                else:
                    waveform = np.load(ecg_path)
                    if waveform.ndim == 3:
                        waveform = waveform.squeeze(-1)
                    if waveform.shape[0] != 12:
                        waveform = waveform.T
            except Exception as e:
                print(f"Error loading waveform {ecg_path}: {e}")
                waveform = None
        
        return {
            "idx": idx,
            "waveform": waveform,
            "ecg_path": ecg_path,
            "diagnosis": diagnosis,
            "ecg_name": ecg_name,
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
    """
    Main inference pipeline for ECG Tokenizer.
    
    Architecture:
    - BERT classifies text reports → 77 class probabilities (GROUND TRUTH)
    - ECG Tokenizer extracts embeddings from signals
    - EfficientNet classifies embeddings → 77 class probabilities (EVALUATED)
    - Metrics show: How well does signal-based match text-based classification
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # Resolve checkpoint paths (handles /app/ Docker paths when running outside Docker)
        self.config.resolve_paths(verbose=config.verbose)
        
        self.device = self._setup_device()
        
        # Models
        self.ecg_signal_processor = None
        self.bert_classifier = None
        self.ecg_tokenizer = None
        self.efficientnet_classifier = None
        
        # QA generation cache
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
        
        # ECG signal processor for preprocessing
        if self.config.apply_psa_normalization:
            self.ecg_signal_processor = ECGSignalProcessor(fs=self.config.sampling_rate)
            print("  ECG signal processor initialized")
        
        # BERT classifier (for ground truth labels from text)
        self._load_bert_classifier()
        
        # ECG tokenizer (for embeddings)
        self._load_ecg_tokenizer()
        
        # EfficientNet classifier (for signal-based classification)
        if self.config.efficientnet_checkpoint:
            self._load_efficientnet_classifier()
    
    def save_and_preprocess_data(
        self, 
        df: pd.DataFrame, 
        preprocessing_folder: str,
        preprocessing_n_workers: int = 16
    ) -> pd.DataFrame:
        """
        Preprocess ECG signals and save to disk as .base64 files.
        
        Following DeepECG_Docker pattern for preprocessing mode.
        
        Standard columns used:
            - diagnosis: Text report/diagnosis
            - ecg_path: Absolute/relative path to ECG file
        
        Args:
            df: Input DataFrame with ecg_file_name column
            preprocessing_folder: Directory to save preprocessed .base64 files
            preprocessing_n_workers: Number of workers for parallel processing
            
        Returns:
            DataFrame with updated ecg_path pointing to .base64 files
        """
        print("\n" + "="*60)
        print("Preprocessing ECG Signals (Disk Storage)")
        print("="*60)
        
        os.makedirs(preprocessing_folder, exist_ok=True)
        
        ecg_signals_path = self.config.ecg_signals_path
        print(f"ECG signals path: {ecg_signals_path}")
        print(f"Loading {len(df)} waveforms...")
        
        batch_size = 10000
        total_batches = (len(df) + batch_size - 1) // batch_size
        processed_df = pd.DataFrame()
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))
            
            print(f"\nProcessing batch {batch_idx + 1}/{total_batches}")
            print(f"Batch size: {end_idx - start_idx} records")
            
            batch_df = df.iloc[start_idx:end_idx].copy()
            ecgs = []
            
            for index, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc="Loading signals"):
                # Use ecg_path directly (DeepECG_Docker pattern)
                ecg_path = row.get("ecg_path")
                if not ecg_path:
                    continue
                
                if not os.path.exists(ecg_path):
                    continue
                
                try:
                    # Handle WFDB files (.hea extension)
                    if ecg_path.endswith('.hea'):
                        import wfdb
                        record_path = ecg_path[:-4]
                        record = wfdb.rdrecord(record_path)
                        lead_array = record.p_signal
                    # Handle NPY files
                    else:
                        lead_array = np.load(ecg_path)
                    
                    if np.isnan(lead_array).any():
                        continue
                    
                    file_id = os.path.basename(ecg_path).replace(".npy", "").replace(".hea", "")
                    
                    # Shape corrections - ensure (2500, 12) format
                    if lead_array.shape[-1] == 1:
                        lead_array = lead_array.squeeze(-1)
                    if lead_array.shape[0] == 12:
                        lead_array = lead_array.transpose(1, 0)
                    
                    # Handle different lengths
                    if lead_array.shape[0] != 2500:
                        if lead_array.shape[0] < 2500:
                            print(f"Warning: Skipping {file_id} - signal length {lead_array.shape[0]} < 2500")
                            continue
                        else:
                            step = lead_array.shape[0] // 2500
                            lead_array = lead_array[::step, :][:2500, :]
                    
                    if lead_array.shape[1] != 12:
                        print(f"Warning: Skipping {file_id} - incorrect number of leads: {lead_array.shape[1]}")
                        continue
                    
                    new_path = os.path.join(preprocessing_folder, f"{file_id}.base64")
                    batch_df.at[index, 'ecg_path'] = new_path
                    ecgs.append([new_path, lead_array])
                    
                except Exception as e:
                    print(f"Error processing file {ecg_path}: {str(e)}")
                    continue
            
            if len(ecgs) == 0:
                print(f"Warning: No valid signals in batch {batch_idx + 1}")
                continue
            
            ecg_signals_df = pd.DataFrame(ecgs, columns=['ecg_path', 'ecg_signal'])
            
            # Apply PSA normalization if processor is available
            if self.ecg_signal_processor is not None:
                print("Scaling ECG signals...")
                scaled_df = self.ecg_signal_processor.scale_ecg_signals(
                    df=ecg_signals_df,
                    power_ratio=PTBXL_POWER_RATIO
                )
                
                print("Processing ECG signals...")
                cleaned_df = self.ecg_signal_processor.clean_and_process_ecg_leads(
                    df=scaled_df,
                    max_workers=preprocessing_n_workers
                )
            else:
                cleaned_df = ecg_signals_df
            
            # Save processed signals as .base64 files
            print("Saving processed signals...")
            for _, row in tqdm(cleaned_df.iterrows(), total=len(cleaned_df), desc="Saving signals"):
                ECGFileHandler.save_ecg_signal(
                    ecg_signal=row['ecg_signal'],
                    filename=row['ecg_path']
                )
            
            processed_df = pd.concat([processed_df, batch_df], ignore_index=True)
            
            del ecg_signals_df, cleaned_df
        
        if len(processed_df) == 0:
            raise ValueError("No data was successfully processed")
        
        print(f"\nPreprocessing complete!")
        print(f"  Processed {len(processed_df)} files to {preprocessing_folder}")
        print("="*60 + "\n")
        
        return processed_df
    
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
        """Load ECG tokenizer wrapper (encoder + quantizer only for embedding extraction)."""
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
                    decoder_mode_str = getattr(ckpt_config, "decoder_mode", "reconstruction")
                    if isinstance(decoder_mode_str, str):
                        decoder_mode = DecoderMode(decoder_mode_str)
                    else:
                        decoder_mode = decoder_mode_str
                    
                    decoder_name = getattr(ckpt_config, "decoder_name", "Conv_Decoder")
                    
                    # For inference, we just need encoder + quantizer for embeddings
                    # The decoder will be loaded separately for classification
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
                    print(f"  ECG tokenizer loaded (encoder+quantizer) from {self.config.tokenizer_checkpoint}")
                else:
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
    
    def _load_efficientnet_classifier(self) -> None:
        """
        Load EfficientNet classifier following the linear probing pattern.
        
        This follows the workflow from linear_probing_project.py:
        1. Load pretrained tokenizer (encoder + quantizer)
        2. Initialize model with classification decoder
        3. Load pretrained encoder/quantizer weights (frozen)
        4. Load trained classification decoder weights
        """
        try:
            from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper
            from utils.enums import DecoderMode
            
            if not os.path.exists(self.config.efficientnet_checkpoint):
                print(f"  Note: EfficientNet checkpoint not found at {self.config.efficientnet_checkpoint}")
                return
            
            # Load the classification checkpoint
            classifier_checkpoint = torch.load(
                self.config.efficientnet_checkpoint,
                map_location="cpu",
                weights_only=False
            )
            
            classifier_config = classifier_checkpoint.get("config", None)
            
            if classifier_config:
                # Load pretrained tokenizer config first
                pretrained_checkpoint = torch.load(
                    self.config.tokenizer_checkpoint,
                    map_location="cpu",
                    weights_only=False
                )
                pretrained_config = pretrained_checkpoint.get("config", None)
                
                if pretrained_config:
                    # Initialize model with classification decoder
                    # Following linear_probing_project.py:65-73
                    self.efficientnet_classifier = ECG_Tokenizer_Wrapper(
                        encoder_name=getattr(pretrained_config, "encoder_name", "Residual_Conv_Encoder"),
                        quantizer_name=getattr(pretrained_config, "quantizer_name", "ECG_Tokenizer_Quantizer"),
                        decoder_name=getattr(classifier_config, "decoder_name", "EfficientNetV2_Classifier_Decoder"),
                        num_quantizers=getattr(pretrained_config, "num_quantizers", 8),
                        codebook_size=getattr(pretrained_config, "codebook_size", 512),
                        decoder_mode=DecoderMode.CLASSIFICATION,
                        num_classes=getattr(classifier_config, "num_classes", 77),
                    )
                    
                    # Load pretrained weights for encoder + quantizer (frozen)
                    # Following linear_probing_project.py:77-79
                    pretrained_state_dict = pretrained_checkpoint.get("model_state_dict", {})
                    self.efficientnet_classifier._load_pretrained_weights(
                        pretrained_state_dict,
                        freeze_pretrained_components=True
                    )
                    
                    # Load the trained classification decoder weights
                    classifier_state_dict = classifier_checkpoint.get("model_state_dict", {})
                    # Load only decoder weights (encoder/quantizer already loaded and frozen)
                    decoder_state_dict = {k: v for k, v in classifier_state_dict.items() if k.startswith('decoder.')}
                    self.efficientnet_classifier.load_state_dict(decoder_state_dict, strict=False)
                    
                    self.efficientnet_classifier.to(self.device)
                    self.efficientnet_classifier.eval()
                    
                    print(f"  EfficientNet classifier loaded from {self.config.efficientnet_checkpoint}")
                else:
                    print("  Warning: Could not load pretrained config for EfficientNet")
            else:
                print("  Warning: Could not load classification config")
                
        except Exception as e:
            print(f"  Warning: Could not load EfficientNet classifier: {e}")
            import traceback
            traceback.print_exc()
    
    def run_bert_classification(
        self,
        reports: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Run BERT 77-class classification on text reports.
        
        These predictions serve as GROUND TRUTH for evaluating signal-based classification.
        
        Args:
            reports: List of ECG report texts.
            
        Returns:
            List of {predictions: [], probabilities: []} dicts.
        """
        if self.bert_classifier is None:
            return [{"predictions": [], "probabilities": []}] * len(reports)
        
        results = []
        
        with torch.no_grad():
            for report in tqdm(reports, desc="BERT Classification", leave=False):
                if not report:
                    results.append({"predictions": [], "probabilities": []})
                    continue
                
                try:
                    inputs = self.bert_classifier.preprocessing(report)
                    
                    input_ids = inputs['input_ids'].to(self.device)
                    attention_mask = inputs['attention_mask'].to(self.device)
                    token_type_ids = inputs['token_type_ids'].to(self.device)
                    
                    outputs = self.bert_classifier.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )
                    
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    
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
                    results.append({"predictions": [], "probabilities": []})
        
        return results
    
    def extract_embeddings(
        self,
        waveforms: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Extract embeddings from ECG signals using the tokenizer.
        
        Args:
            waveforms: Batch of ECG waveforms [B, 12, 2500].
            
        Returns:
            Dictionary with quantized features, indices, and metadata.
        """
        if self.ecg_tokenizer is None:
            return {}
        
        with torch.no_grad():
            waveforms = waveforms.to(self.device)
            
            features = self.ecg_tokenizer.encoder(waveforms)
            quantized, indices, commit_loss = self.ecg_tokenizer.quantizer(features)
            
            return {
                "quantized": quantized,
                "indices": indices,
                "features": features
            }
    
    def run_efficientnet_classification(
        self,
        embeddings: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """
        Run EfficientNet classification on embeddings.
        
        These predictions are EVALUATED against BERT predictions (ground truth).
        
        Args:
            embeddings: Quantized embeddings from tokenizer [B, C, T].
            
        Returns:
            List of {predictions: [], probabilities: []} dicts.
        """
        if self.efficientnet_classifier is None:
            return []
        
        results = []
        
        with torch.no_grad():
            embeddings = embeddings.to(self.device)
            
            # Forward pass through the classification decoder
            logits = self.efficientnet_classifier.decoder(embeddings)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            for i in range(probs.shape[0]):
                predictions = (probs[i] > self.config.classification_threshold).astype(int).tolist()
                results.append({
                    "predictions": predictions,
                    "probabilities": probs[i].tolist()
                })
        
        return results
    
    def generate_qa_pairs(
        self,
        report: str,
        waveform_name: str,
        dataset: str = "mimic"
    ) -> List[Dict[str, str]]:
        """
        Generate Q&A pairs for an ECG sample.
        
        Args:
            report: Text report.
            waveform_name: ECG identifier.
            dataset: Dataset name for prompt generation.
            
        Returns:
            List of {question, answer, category} dicts.
        """
        if not report:
            return []
        
        try:
            # Initialize generators once and cache
            if self.prompt_maker is None or self.answer_generator is None:
                sys.path.insert(0, str(REPO_ROOT / "dataset_generation"))
                from ecg_prompt_maker import ECGPromptMaker
                from ecg_answer_generator import ECGAnswerGenerator
                
                self.prompt_maker = ECGPromptMaker(dataset=dataset)
                self.answer_generator = ECGAnswerGenerator(dataset=dataset)
            
            # Create minimal row data for QA generation
            row_series = pd.Series({
                "report": report,
                "waveform_name": waveform_name,
            })
            
            prompts = self.prompt_maker.generate_prompts_for_ecg(row_series)
            
            # Sort by weight and limit
            prompts_sorted = sorted(prompts, key=lambda x: x[2], reverse=True)
            prompts_limited = prompts_sorted[:self.config.max_prompts_per_ecg]
            
            qa_pairs = []
            for prompt_text, category, weight in prompts_limited:
                # Determine prompt type
                if category.startswith('category_'):
                    prompt_type = 'category'
                elif category.startswith('localization_'):
                    prompt_type = 'localization'
                elif category == 'interpretation':
                    prompt_type = 'interpretation'
                else:
                    prompt_type = 'interpretation'
                
                # Add prompt info for answer generation
                row_with_prompt = row_series.copy()
                row_with_prompt['prompt_category'] = category
                row_with_prompt['prompt_type'] = prompt_type
                row_with_prompt['prompt'] = prompt_text
                
                answer = self.answer_generator.generate_answer(row_with_prompt)
                
                qa_pairs.append({
                    "question": prompt_text,
                    "answer": answer,
                    "category": category,
                    "type": prompt_type,
                })
            
            return qa_pairs
            
        except Exception as e:
            if self.config.verbose:
                print(f"Error generating QA pairs: {e}")
            return []
    
    def _save_ground_truth_parquet(self, all_results: List[Dict[str, Any]]) -> None:
        """
        Save BERT predictions as ground truth columns in output parquet.
        
        Args:
            all_results: List of result dictionaries with BERT classifications.
        """
        try:
            # Read original parquet
            df_original = pd.read_parquet(self.config.input_parquet)
            
            # Extract BERT predictions (ground truth) and add as columns
            bert_gt_data = {}
            for pattern in ECG_PATTERNS:
                bert_gt_data[pattern] = []
            
            for result in all_results:
                bert = result.get("bert_classification", {})
                if bert.get("predictions") and len(bert["predictions"]) == 77:
                    for i, pattern in enumerate(ECG_PATTERNS):
                        bert_gt_data[pattern].append(bert["predictions"][i])
                else:
                    # Fill with zeros if no prediction
                    for pattern in ECG_PATTERNS:
                        bert_gt_data[pattern].append(0)
            
            # Add ground truth columns to original dataframe
            for pattern in ECG_PATTERNS:
                df_original[pattern] = bert_gt_data[pattern]
            
            # Save output parquet with ground truth
            output_parquet = self.config.input_parquet.replace('.parquet', '_with_gt.parquet')
            df_original.to_parquet(output_parquet)
            print(f"\nGround truth labels (from BERT) saved to: {output_parquet}")
            
        except Exception as e:
            print(f"Warning: Could not save ground truth parquet: {e}")
    
    def _compute_signal_vs_text_metrics(
        self,
        all_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute classification metrics: Signal-based (EfficientNet) vs Text-based (BERT).
        
        BERT predictions are treated as GROUND TRUTH.
        EfficientNet predictions are EVALUATED against BERT.
        
        Args:
            all_results: List of result dictionaries.
            
        Returns:
            Dictionary containing classification metrics.
        """
        metrics = {
            "signal_vs_text": {},
            "description": "EfficientNet (signal) evaluated against BERT (text)"
        }
        
        try:
            bert_probs = []
            efficientnet_probs = []
            
            for result in all_results:
                bert = result.get("bert_classification", {})
                eff = result.get("efficientnet_classification", {})
                
                # Both must have valid 77-class probabilities
                if (bert.get("probabilities") and len(bert["probabilities"]) == 77 and
                    eff.get("probabilities") and len(eff["probabilities"]) == 77):
                    bert_probs.append(bert["probabilities"])
                    efficientnet_probs.append(eff["probabilities"])
            
            if bert_probs and efficientnet_probs:
                print(f"\nComputing Signal vs Text metrics ({len(bert_probs)} samples)...")
                
                # BERT as ground truth, EfficientNet as predictions
                df_gt = pd.DataFrame(bert_probs, columns=ECG_PATTERNS)
                df_pred = pd.DataFrame(efficientnet_probs, columns=ECG_PATTERNS)
                
                # Binarize ground truth using threshold
                threshold = self.config.classification_threshold
                df_gt_binary = (df_gt > threshold).astype(int)
                
                signal_vs_text_metrics = compute_metrics(df_gt_binary, df_pred)
                metrics["signal_vs_text"] = signal_vs_text_metrics
                
                # Summary
                category_aucs = []
                for category in ["RHYTHM", "CONDUCTION", "INFARCT, ISCHEMIA",
                               "CHAMBER ENLARGEMENT", "PERICARDITIS", "OTHER"]:
                    if category in signal_vs_text_metrics and "macro_auc" in signal_vs_text_metrics[category]:
                        auc = signal_vs_text_metrics[category]["macro_auc"]
                        if not np.isnan(auc):
                            category_aucs.append(auc)
                
                if category_aucs:
                    metrics["signal_vs_text"]["overall_macro_auc"] = float(np.mean(category_aucs))
                    print(f"  Signal vs Text Overall Macro AUC: {metrics['signal_vs_text']['overall_macro_auc']:.4f}")
            else:
                print(f"  Note: Skipping Signal vs Text metrics (need both BERT and EfficientNet predictions)")
            
        except Exception as e:
            print(f"Error computing signal vs text metrics: {e}")
            import traceback
            traceback.print_exc()
        
        return metrics
    
    def _process_batch(
        self,
        batch_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Process a single batch of ECG samples.
        
        Order of operations:
        1. BERT classification on text reports (GROUND TRUTH)
        2. Extract embeddings from signals
        3. EfficientNet classification on embeddings (EVALUATED)
        4. Generate reports (if LLM decoder available)
        5. Generate QA pairs
        """
        results = []
        
        waveforms = batch_data.get("waveforms")
        batch = batch_data.get("batch", [])
        
        # Step 1: Run BERT FIRST on text diagnoses (GROUND TRUTH)
        diagnoses = [item.get("diagnosis", "") for item in batch]
        bert_results = self.run_bert_classification(diagnoses)
        
        # Step 2: Extract embeddings from signals
        tokenizer_embeddings = []
        quantized_features = None
        
        if waveforms is not None and self.ecg_tokenizer is not None:
            with torch.no_grad():
                try:
                    embedding_data = self.extract_embeddings(waveforms)
                    quantized_features = embedding_data.get("quantized")
                    indices = embedding_data.get("indices")
                    
                    for i in range(waveforms.shape[0]):
                        embedding_info = {
                            "quantized_shape": list(quantized_features[i].shape) if quantized_features is not None else [],
                            "indices_shape": list(indices[i].shape) if indices is not None else [],
                            "num_codebooks": indices.shape[-1] if indices is not None and indices.dim() > 2 else 1,
                            "sequence_length": quantized_features.shape[-1] if quantized_features is not None else 0,
                            "embedding_dim": quantized_features.shape[1] if quantized_features is not None else 0,
                        }
                        tokenizer_embeddings.append(embedding_info)
                except Exception as e:
                    if self.config.verbose:
                        print(f"  Warning: Could not extract embeddings: {e}")
        
        # Step 3: Run EfficientNet classification on embeddings (EVALUATED)
        efficientnet_results = []
        if quantized_features is not None and self.efficientnet_classifier is not None:
            efficientnet_results = self.run_efficientnet_classification(quantized_features)
        
        # Step 4: Assemble results
        for i, item in enumerate(batch):
            result = {
                "ecg_name": item.get("ecg_name", ""),
                "ecg_path": item.get("ecg_path", ""),
                "diagnosis": item.get("diagnosis", ""),
            }
            
            # Store BERT classification for ground truth generation (not in final JSON)
            if i < len(bert_results):
                result["bert_classification"] = bert_results[i]
            
            # Store EfficientNet classification for metrics computation (not in final JSON)
            if i < len(efficientnet_results):
                result["efficientnet_classification"] = efficientnet_results[i]
            
            # Add tokenizer embeddings
            if i < len(tokenizer_embeddings):
                result["tokenizer_embeddings"] = tokenizer_embeddings[i]
            
            # Generate QA pairs
            qa_pairs = self.generate_qa_pairs(
                report=item.get("diagnosis", ""),
                waveform_name=item.get("ecg_name", "")
            )
            result["qa_results"] = [
                {
                    "question": qa.get("question", ""),
                    "ground_truth_answer": qa.get("answer", ""),
                    "category": qa.get("category", "unknown"),
                    "predicted_answer": "",
                }
                for qa in qa_pairs
            ]
            
            results.append(result)
        
        return results
    
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete inference pipeline.
        
        Returns:
            Dictionary containing all results and metrics.
        """
        print(f"\n{'='*60}")
        print("ECG Tokenizer Inference Pipeline")
        print(f"{'='*60}")
        print(f"Input: {self.config.input_parquet}")
        print(f"Output: {self.config.output_json}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        dataset = ECGDataset(
            parquet_path=self.config.input_parquet,
            ecg_signals_path=self.config.ecg_signals_path
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn
        )
        
        all_results = []
        
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Processing")):
            batch_results = self._process_batch(batch_data)
            all_results.extend(batch_results)
        
        # Compute Signal vs Text classification metrics
        classification_metrics = self._compute_signal_vs_text_metrics(all_results)
        
        # Save BERT predictions as ground truth columns in output parquet
        self._save_ground_truth_parquet(all_results)
        
        # Clean results: remove classification predictions (keep only metrics)
        cleaned_results = []
        for result in all_results:
            cleaned = {
                "ecg_name": result.get("ecg_name", ""),
                "ecg_path": result.get("ecg_path", ""),
                "diagnosis": result.get("diagnosis", ""),
                "tokenizer_embeddings": result.get("tokenizer_embeddings", {}),
                "qa_results": result.get("qa_results", []),
            }
            cleaned_results.append(cleaned)
        
        # Build output (without individual predictions, just metrics)
        output = {
            "metadata": {
                "pipeline_version": "2.0.0",
                "timestamp": datetime.now().isoformat(),
                "config": self.config.to_dict(),
                "num_samples": len(cleaned_results),
                "architecture": {
                    "ground_truth": "BERT (text classification)",
                    "evaluated": "EfficientNet (signal classification)",
                    "description": "Signal-based predictions evaluated against text-based ground truth"
                },
                "output_parquet": self.config.input_parquet.replace('.parquet', '_with_gt.parquet'),
                "notes": [
                    "Individual predictions not included in JSON to reduce size",
                    "BERT ground truth labels saved to output parquet file",
                    "Classification metrics computed using utils/metrics/ecg_metrics.py"
                ]
            },
            "results": cleaned_results,
            "aggregate_metrics": {
                "classification_metrics": classification_metrics,
                "text_metrics": {},
            }
        }
        
        # Save results
        output_path = Path(self.config.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\nResults saved to {output_path}")
        
        return output
