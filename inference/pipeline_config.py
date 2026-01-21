#!/usr/bin/env python3
"""
Pipeline configuration for ECG Tokenizer Docker inference.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import os
import shutil
import yaml
import json


def resolve_checkpoint_path(path: Optional[str], repo_root: Optional[str] = None) -> Optional[str]:
    """
    Resolve a checkpoint path to an existing location.
    
    Handles:
    1. None/empty paths - return as-is
    2. Existing absolute paths - return as-is
    3. /app/ Docker paths - try to resolve to repo root
    4. Relative paths - resolve relative to repo root
    5. HuggingFace paths (future) - download to cache
    
    Args:
        path: The checkpoint path to resolve
        repo_root: Repository root directory (auto-detected if None)
        
    Returns:
        Resolved path that exists, or original path if resolution fails
    """
    if not path:
        return path
    
    # Already exists - use as-is
    if os.path.exists(path):
        return path
    
    # Auto-detect repo root
    if repo_root is None:
        # Try to find repo root from this file's location
        current_file = Path(__file__).resolve()
        repo_root = str(current_file.parent.parent)  # inference/ -> repo root
    
    # Handle /app/ Docker paths - translate to repo root
    if path.startswith("/app/"):
        relative_path = path[5:]  # Remove "/app/" prefix
        resolved = os.path.join(repo_root, relative_path)
        if os.path.exists(resolved):
            return resolved
    
    # Try as relative path from repo root
    resolved = os.path.join(repo_root, path)
    if os.path.exists(resolved):
        return resolved
    
    # Try common checkpoint locations
    checkpoint_bases = [
        os.path.join(repo_root, "checkpoints"),
        "/volume/ECG_tokenizer/checkpoints",
        os.path.expanduser("~/.cache/ecg_tokenizer/checkpoints"),
    ]
    
    # Extract the checkpoint name/relative part
    path_parts = Path(path).parts
    for base in checkpoint_bases:
        if os.path.exists(base):
            # Try with full relative path
            for i in range(len(path_parts)):
                test_path = os.path.join(base, *path_parts[i:])
                if os.path.exists(test_path):
                    return test_path
    
    # Future: Handle HuggingFace paths
    # if path.startswith("hf://") or "huggingface.co" in path:
    #     return download_from_huggingface(path)
    
    # Return original path (will fail later with clearer error)
    return path


@dataclass
class PipelineConfig:
    """Configuration for the ECG Tokenizer inference pipeline."""
    
    # Input/Output paths
    input_parquet: str
    output_json: str
    output_dir: str = "/app/outputs"
    
    # Model checkpoint paths
    bert_checkpoint: str = "/app/checkpoints/mimic_mhi_bert"
    tokenizer_checkpoint: str = "/app/checkpoints/ECG_tokenizer_latest/best_model_epoch_10.pt"
    gpt2_checkpoint: Optional[str] = None
    medgemma_checkpoint: Optional[str] = None
    efficientnet_checkpoint: Optional[str] = None
    
    # Report generation
    enable_report_generation: bool = False
    
    # Device configuration
    device: str = "cuda:0"
    cpu_fallback: bool = True
    
    # Model configuration
    decoder_mode: str = "llm"  # llm, classification, or reconstruction
    
    # Batch processing
    batch_size: int = 32
    num_workers: int = 8
    
    # ECG parameters
    waveform_length: int = 2500
    num_leads: int = 12
    sampling_rate: int = 250
    
    # PSA normalization
    apply_psa_normalization: bool = True
    psa_region: str = "NA"
    
    # Column names in input parquet (minimal required columns)
    # Only waveform_path, report, waveform_name are expected
    # NO diagnostic label columns are expected
    waveform_path_column: str = "waveform_path"
    report_column: str = "report"
    waveform_name_column: str = "waveform_name"
    
    # Ground truth settings
    # BERT predictions from text serve as ground truth for signal classification
    use_bert_as_ground_truth: bool = True
    bert_threshold: float = 0.5
    
    # Classification settings
    num_classes: int = 77
    classification_threshold: float = 0.5
    
    # Generation settings
    max_report_length: int = 512
    max_answer_length: int = 256
    generation_temperature: float = 0.7
    generation_top_p: float = 0.9
    num_beams: int = 1
    do_sample: bool = True
    
    # QA generation settings
    max_prompts_per_ecg: int = 5
    qa_categories: List[str] = field(default_factory=lambda: [
        "RHYTHM", "CONDUCTION", "INFARCT, ISCHEMIA", 
        "CHAMBER ENLARGEMENT", "PERICARDITIS", "OTHER"
    ])
    
    # Evaluation settings
    compute_rouge: bool = True
    compute_bleu: bool = True
    compute_meteor: bool = True
    run_llm_judge: bool = True
    llm_judge_config_path: Optional[str] = None
    
    # Logging
    verbose: bool = True
    log_every_n_batches: int = 10
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        """Load configuration from a YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> "PipelineConfig":
        """Load configuration from a JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_config_file(cls, config_path: str) -> "PipelineConfig":
        """Load configuration from a file (auto-detect format)."""
        path = Path(config_path)
        if path.suffix.lower() in ['.yaml', '.yml']:
            return cls.from_yaml(config_path)
        elif path.suffix.lower() == '.json':
            return cls.from_json(config_path)
        else:
            with open(config_path, 'r') as f:
                content = f.read().strip()
                if content.startswith('{'):
                    config_dict = json.loads(content)
                else:
                    config_dict = yaml.safe_load(content)
            return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
        }
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to a YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def to_json(self, json_path: str) -> None:
        """Save configuration to a JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def resolve_paths(self, repo_root: Optional[str] = None, verbose: bool = True) -> "PipelineConfig":
        """
        Resolve all checkpoint paths to existing locations.
        
        This handles the case where config uses /app/ Docker paths but
        running outside Docker with paths under repo root.
        
        Args:
            repo_root: Repository root directory (auto-detected if None)
            verbose: Print resolution info
            
        Returns:
            Self with resolved paths
        """
        checkpoint_attrs = [
            "bert_checkpoint",
            "tokenizer_checkpoint", 
            "gpt2_checkpoint",
            "medgemma_checkpoint",
            "efficientnet_checkpoint",
            "llm_judge_config_path",
        ]
        
        for attr in checkpoint_attrs:
            original = getattr(self, attr, None)
            if original:
                resolved = resolve_checkpoint_path(original, repo_root)
                if resolved != original and verbose:
                    print(f"  Path resolved: {attr}")
                    print(f"    {original} -> {resolved}")
                setattr(self, attr, resolved)
        
        return self
    
    def validate(self) -> List[str]:
        """
        Validate the configuration and return a list of errors.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        if not self.input_parquet:
            errors.append("input_parquet is required")
        
        if not self.output_json:
            errors.append("output_json is required")
        
        if self.batch_size < 1:
            errors.append("batch_size must be >= 1")
        
        if self.num_workers < 0:
            errors.append("num_workers must be >= 0")
        
        if self.waveform_length < 1:
            errors.append("waveform_length must be >= 1")
        
        if self.num_leads not in [12, 8, 6, 3, 1]:
            errors.append("num_leads should be 12, 8, 6, 3, or 1")
        
        if self.classification_threshold < 0 or self.classification_threshold > 1:
            errors.append("classification_threshold must be between 0 and 1")
        
        return errors


@dataclass  
class LLMJudgeConfig:
    """Configuration for LLM-as-a-Judge evaluation."""
    
    judge_model: str = "gpt-4"
    api_key_env_var: str = "OPENAI_API_KEY"
    max_concurrent_requests: int = 10
    timeout_seconds: int = 60
    retry_attempts: int = 3
    
    categories_to_evaluate: List[str] = field(default_factory=lambda: [
        "interpretation", "classification", "category_rhythm",
        "category_conduction", "category_infarct_ischemia",
        "category_chamber_enlargement", "category_pericarditis", "category_other"
    ])
    
    scoring_rubric: str = "accuracy"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "LLMJudgeConfig":
        """Load configuration from a YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
