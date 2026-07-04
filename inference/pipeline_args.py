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
    
    # Step control
    run_step: str = "all"  # preprocess | bert | analysis | efficientnet | all
    use_preprocessing: bool = True
    use_bert_classification: bool = True
    
    # Paths
    input_parquet: str = "/app/inputs/data.parquet"
    output_dir: str = "/app/outputs"
    ecg_signals_path: str = "/app/ecg_signals"  # Path where ecg_file_name values are joined
    checkpoints_dir: str = "/app/checkpoints"

    # Model checkpoints (only BERT + tokenizer are used here)
    bert_checkpoint: str = "/app/checkpoints/mimic_mhi_bert"
    tokenizer_checkpoint: str = "/app/checkpoints/ECG_tokenizer_latest/best_model_epoch_10.pt"

    # Device settings
    device: str = "cuda:0"

    # Processing settings
    batch_size: int = 32
    num_workers: int = 8
    
    # Standard column names (fixed - not configurable)
    # Input files must have: ecg_path, reports
    
    # Preprocessing
    preprocessing_folder: str = "/app/preprocessing"
    preprocessing_n_workers: int = 16
    dataset_name: Optional[str] = None
    include_optional_100hz_cluster: bool = False
    ecg_scale: Optional[float] = None
    ecg_source: Optional[str] = None
    bert_base_config: str = "config/bert_classifier/base_config.yaml"
    
    # Intermediate outputs
    preprocessing_output: Optional[str] = None
    bert_output: Optional[str] = None
    
    # Classification
    num_classes: int = 77
    
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
        
        # Step control (single flag)
        parser.add_argument(
            "--step",
            dest="run_step",
            choices=["preprocess", "bert", "analysis", "efficientnet", "all"],
            default="all",
            help=(
                "Which step to run. "
                "preprocess: only preprocessing; "
                "bert: BERT on existing parquet; "
                "analysis: start at BERT (skip preprocessing); "
                "efficientnet: skip internal steps (runner.sh handles it); "
                "all: preprocess + bert."
            ),
        )
        
        # Paths
        parser.add_argument(
            "--input", "--input-parquet",
            dest="input_parquet",
            type=str,
            help="Input parquet file path"
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
            "--include-optional-100hz-cluster",
            action="store_true",
            help="Include optional flatten range around 100-101 Hz during preprocessing"
        )
        parser.add_argument(
            "--ecg-scale",
            type=float,
            help="WCRv2 ADC->mV scale factor override (e.g., MHI=0.00488, MIMIC=0.001)"
        )
        parser.add_argument(
            "--ecg-source",
            type=str,
            help="Dataset source for WCRv2 scale lookup (MHI, MIMIC, CODE15, UKBB, CLSA)"
        )
        parser.add_argument(
            "--bert-base-config",
            type=str,
            help="BERT base config yaml (for run_bert_classification/analysis)"
        )
        parser.add_argument(
            "--preprocessing-output",
            type=str,
            help="Path to cached preprocessing parquet"
        )
        parser.add_argument(
            "--bert-output",
            type=str,
            help="Path to cached BERT labels parquet"
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
        if args.input_parquet:
            instance.input_parquet = args.input_parquet
        if args.output_dir:
            instance.output_dir = args.output_dir
        if args.bert_checkpoint:
            instance.bert_checkpoint = args.bert_checkpoint
        if args.tokenizer_checkpoint:
            instance.tokenizer_checkpoint = args.tokenizer_checkpoint
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
        if args.preprocessing_folder:
            instance.preprocessing_folder = args.preprocessing_folder
        if args.preprocessing_n_workers:
            instance.preprocessing_n_workers = args.preprocessing_n_workers
        if args.dataset_name:
            instance.dataset_name = args.dataset_name
        if args.include_optional_100hz_cluster:
            instance.include_optional_100hz_cluster = True
        if args.ecg_scale is not None:
            instance.ecg_scale = args.ecg_scale
        if args.ecg_source:
            instance.ecg_source = args.ecg_source
        if args.bert_base_config:
            instance.bert_base_config = args.bert_base_config
        if args.preprocessing_output:
            instance.preprocessing_output = args.preprocessing_output
        if args.bert_output:
            instance.bert_output = args.bert_output

        # Override flags if provided
        # Derive step flags from run_step
        instance.run_step = args.run_step
        if instance.run_step == "preprocess":
            instance.use_preprocessing = True
            instance.use_bert_classification = False
        elif instance.run_step in ("bert", "analysis"):
            instance.use_preprocessing = False
            instance.use_bert_classification = True
        elif instance.run_step == "efficientnet":
            # This entry point does nothing for efficientnet; caller should run runner.sh.
            instance.use_preprocessing = False
            instance.use_bert_classification = False
        else:  # all
            instance.use_preprocessing = True
            instance.use_bert_classification = True
        if args.quiet:
            instance.verbose = False
        
        return instance
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "input_parquet": self.input_parquet,
            "output_dir": self.output_dir,
            "ecg_signals_path": self.ecg_signals_path,
            "bert_checkpoint": self.bert_checkpoint,
            "tokenizer_checkpoint": self.tokenizer_checkpoint,
            "device": self.device,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "preprocessing_folder": self.preprocessing_folder,
            "preprocessing_n_workers": self.preprocessing_n_workers,
            "include_optional_100hz_cluster": self.include_optional_100hz_cluster,
            "ecg_scale": self.ecg_scale,
            "ecg_source": self.ecg_source,
            "num_classes": self.num_classes,
            "verbose": self.verbose,
        }
