#!/usr/bin/env python3
"""
Pipeline Arguments Parser.

Handles configuration from:
1. Config file (heartwise.config - key: value format)
2. Command-line arguments (override config values)

Following the DeepECG_Docker pattern for clean configuration management.
"""

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Any


@dataclass
class PipelineArgs:
    """Pipeline arguments with defaults matching DeepECG_Docker patterns."""
    
    # Mode
    mode: str = "full_run"  # preprocessing, run_bert_classification, run_efficientnet, analysis, full_run
    
    # Paths
    input_parquet: str = "/app/inputs/data.parquet"
    output_json: str = "/app/outputs/results.json"
    output_dir: str = "/app/outputs"
    ecg_signals_path: str = "/app/ecg_signals"  # Path where ecg_file_name values are joined
    checkpoints_dir: str = "/app/checkpoints"
    
    # Model checkpoints
    bert_checkpoint: str = "/app/checkpoints/mimic_mhi_bert"
    tokenizer_checkpoint: str = "/app/checkpoints/ECG_tokenizer_latest/best_model_epoch_10.pt"
    efficientnet_checkpoint: Optional[str] = None
    gpt2_checkpoint: Optional[str] = None
    medgemma_checkpoint: Optional[str] = None
    
    # Device settings
    device: str = "cuda:0"
    cpu_fallback: bool = True
    
    # Processing settings
    batch_size: int = 32
    num_workers: int = 8
    
    # Standard column names (fixed - not configurable)
    # Input files must have: ecg_path, reports
    
    # Preprocessing
    apply_psa_normalization: bool = True
    psa_region: str = "NA"
    psa_lead_stats_path: Optional[str] = None
    preprocessing_folder: str = "/app/preprocessing"
    preprocessing_n_workers: int = 16
    dataset_name: Optional[str] = None
    bert_base_config: str = "config/bert_classifier/base_config.yaml"
    
    # Classification
    num_classes: int = 77
    classification_threshold: float = 0.5
    use_bert_as_ground_truth: bool = True
    bert_threshold: float = 0.5
    
    # Text generation (GPT2 report generation)
    enable_report_generation: bool = False
    max_report_length: int = 256
    generation_temperature: float = 0.7
    generation_top_p: float = 0.9
    
    # QA generation
    max_prompts_per_ecg: int = 2
    qa_categories: List[str] = field(default_factory=lambda: [
        "RHYTHM", "CONDUCTION", "INFARCT_ISCHEMIA",
        "CHAMBER_ENLARGEMENT", "PERICARDITIS", "OTHER"
    ])
    
    # Evaluation
    compute_classification_metrics: bool = True
    compute_text_metrics: bool = True
    run_llm_judge: bool = False
    llm_judge_config_path: Optional[str] = None
    
    # Output
    verbose: bool = True
    
    @staticmethod
    def _parse_config_file(config_path: str) -> dict:
        """
        Parse heartwise.config file (key: value format).
        
        Args:
            config_path: Path to config file.
            
        Returns:
            Dictionary of configuration values.
        """
        config = {}
        
        if not os.path.exists(config_path):
            return config
        
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse key: value
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Convert value types
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    elif value.lower() in ('none', 'null', ''):
                        value = None
                    elif value.startswith('[') and value.endswith(']'):
                        # Parse list
                        value = [v.strip() for v in value[1:-1].split(',')]
                    else:
                        # Try to convert to int or float
                        try:
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        except ValueError:
                            pass  # Keep as string
                    
                    config[key] = value
        
        return config
    
    @classmethod
    def parse_arguments(cls) -> 'PipelineArgs':
        """
        Parse arguments from config file and command line.
        
        Command-line arguments override config file values.
        
        Returns:
            PipelineArgs instance with all configuration.
        """
        parser = argparse.ArgumentParser(
            description="ECG Tokenizer Inference Pipeline",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Config file
        parser.add_argument(
            "--config",
            type=str,
            default="heartwise.config",
            help="Path to configuration file"
        )
        
        # Mode
        parser.add_argument(
            "--mode",
            type=str,
            choices=["preprocessing", "run_bert_classification", "run_efficientnet", "analysis", "full_run"],
            help="Pipeline execution mode"
        )
        
        # Paths
        parser.add_argument(
            "--input", "--input-parquet",
            dest="input_parquet",
            type=str,
            help="Input parquet file path"
        )
        parser.add_argument(
            "--output", "--output-json",
            dest="output_json",
            type=str,
            help="Output JSON file path"
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            help="Output directory for additional files"
        )
        
        # Model checkpoints
        parser.add_argument(
            "--bert-checkpoint",
            type=str,
            help="Path to BERT classifier checkpoint"
        )
        parser.add_argument(
            "--tokenizer-checkpoint",
            type=str,
            help="Path to ECG tokenizer checkpoint"
        )
        parser.add_argument(
            "--efficientnet-checkpoint",
            type=str,
            help="Path to EfficientNet classifier checkpoint"
        )
        parser.add_argument(
            "--gpt2-checkpoint",
            type=str,
            help="Path to GPT2 report generation checkpoint"
        )
        parser.add_argument(
            "--enable-report-generation",
            action="store_true",
            help="Enable GPT2-based report generation from ECG signals"
        )
        
        # Device
        parser.add_argument(
            "--device",
            type=str,
            help="Device to use (cuda:0, cpu, etc.)"
        )
        parser.add_argument(
            "--cpu",
            action="store_true",
            help="Force CPU mode"
        )
        
        # Processing
        parser.add_argument(
            "--batch-size",
            type=int,
            help="Batch size for processing"
        )
        parser.add_argument(
            "--num-workers",
            type=int,
            help="Number of data loader workers"
        )
        
        # ECG signals path
        parser.add_argument(
            "--ecg-signals-path",
            type=str,
            help="Path where ECG signal files are located (ecg_file_name values joined with this)"
        )
        
        # Preprocessing
        parser.add_argument(
            "--no-psa",
            action="store_true",
            help="Skip PSA normalization"
        )
        parser.add_argument(
            "--psa-region",
            type=str,
            help="PSA normalization region"
        )
        parser.add_argument(
            "--preprocessing-folder",
            type=str,
            help="Folder to save preprocessed .base64 files"
        )
        parser.add_argument(
            "--preprocessing-n-workers",
            type=int,
            help="Number of workers for preprocessing"
        )
        parser.add_argument(
            "--dataset-name",
            type=str,
            help="Optional dataset name to embed in preprocessing folder and parquet name"
        )
        parser.add_argument(
            "--bert-base-config",
            type=str,
            help="BERT base config yaml (for run_bert_classification/analysis)"
        )
        
        # Classification
        parser.add_argument(
            "--threshold",
            type=float,
            help="Classification threshold"
        )
        parser.add_argument(
            "--no-bert-gt",
            action="store_true",
            help="Don't use BERT as ground truth (use parquet columns if available)"
        )
        
        # Evaluation
        parser.add_argument(
            "--no-metrics",
            action="store_true",
            help="Skip metrics computation"
        )
        parser.add_argument(
            "--with-llm-judge",
            action="store_true",
            help="Run LLM-as-a-Judge evaluation"
        )
        
        # QA
        parser.add_argument(
            "--max-prompts",
            type=int,
            help="Maximum prompts per ECG"
        )
        
        # Verbosity
        parser.add_argument(
            "--quiet",
            action="store_true",
            help="Reduce output verbosity"
        )
        
        args = parser.parse_args()
        
        # Load config file first
        config_path = args.config
        if not os.path.isabs(config_path):
            # Check common locations
            possible_paths = [
                config_path,
                f"/app/{config_path}",
                f"docker/{config_path}",
                str(Path(__file__).parent.parent / "docker" / config_path),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
        
        config = cls._parse_config_file(config_path)
        
        # Create instance with config values
        instance = cls()
        
        # Apply config file values
        for key, value in config.items():
            # Convert config key names to match dataclass field names
            field_name = key.replace('-', '_')
            if hasattr(instance, field_name) and value is not None:
                setattr(instance, field_name, value)
        
        # Apply command-line overrides
        if args.mode:
            instance.mode = args.mode
        if args.input_parquet:
            instance.input_parquet = args.input_parquet
        if args.output_json:
            instance.output_json = args.output_json
        if args.output_dir:
            instance.output_dir = args.output_dir
        if args.bert_checkpoint:
            instance.bert_checkpoint = args.bert_checkpoint
        if args.tokenizer_checkpoint:
            instance.tokenizer_checkpoint = args.tokenizer_checkpoint
        if args.efficientnet_checkpoint:
            instance.efficientnet_checkpoint = args.efficientnet_checkpoint
        if args.gpt2_checkpoint:
            instance.gpt2_checkpoint = args.gpt2_checkpoint
        if args.enable_report_generation:
            instance.enable_report_generation = True
        if args.device:
            instance.device = args.device
        if args.cpu:
            instance.device = "cpu"
        if args.batch_size:
            instance.batch_size = args.batch_size
        if args.num_workers:
            instance.num_workers = args.num_workers
        if args.ecg_signals_path:
            instance.ecg_signals_path = args.ecg_signals_path
        if args.no_psa:
            instance.apply_psa_normalization = False
        if args.psa_region:
            instance.psa_region = args.psa_region
        if args.preprocessing_folder:
            instance.preprocessing_folder = args.preprocessing_folder
        if args.preprocessing_n_workers:
            instance.preprocessing_n_workers = args.preprocessing_n_workers
        if args.dataset_name:
            instance.dataset_name = args.dataset_name
        if args.bert_base_config:
            instance.bert_base_config = args.bert_base_config
        if args.threshold:
            instance.classification_threshold = args.threshold
            instance.bert_threshold = args.threshold
        if args.no_bert_gt:
            instance.use_bert_as_ground_truth = False
        if args.no_metrics:
            instance.compute_classification_metrics = False
            instance.compute_text_metrics = False
        if args.with_llm_judge:
            instance.run_llm_judge = True
        if args.max_prompts:
            instance.max_prompts_per_ecg = args.max_prompts
        if args.quiet:
            instance.verbose = False
        
        return instance
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "mode": self.mode,
            "input_parquet": self.input_parquet,
            "output_json": self.output_json,
            "output_dir": self.output_dir,
            "ecg_signals_path": self.ecg_signals_path,
            "bert_checkpoint": self.bert_checkpoint,
            "tokenizer_checkpoint": self.tokenizer_checkpoint,
            "efficientnet_checkpoint": self.efficientnet_checkpoint,
            "gpt2_checkpoint": self.gpt2_checkpoint,
            "enable_report_generation": self.enable_report_generation,
            "device": self.device,
            "cpu_fallback": self.cpu_fallback,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "apply_psa_normalization": self.apply_psa_normalization,
            "psa_region": self.psa_region,
            "preprocessing_folder": self.preprocessing_folder,
            "preprocessing_n_workers": self.preprocessing_n_workers,
            "num_classes": self.num_classes,
            "classification_threshold": self.classification_threshold,
            "use_bert_as_ground_truth": self.use_bert_as_ground_truth,
            "bert_threshold": self.bert_threshold,
            "max_report_length": self.max_report_length,
            "max_prompts_per_ecg": self.max_prompts_per_ecg,
            "compute_classification_metrics": self.compute_classification_metrics,
            "compute_text_metrics": self.compute_text_metrics,
            "run_llm_judge": self.run_llm_judge,
            "verbose": self.verbose,
        }
