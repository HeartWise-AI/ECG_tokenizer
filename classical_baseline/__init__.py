"""
Classical ECG Baseline Package

This package provides a classical signal processing baseline for ECG analysis
that integrates with the ECG_tokenizer infrastructure.

Main components:
- ClassicalECGBaseline: Main pipeline class
- ClassicalFeatureExtractor: Feature extraction methods
- ClassicalBaselineConfig: Configuration management

Example usage:
    from classical_baseline import ClassicalECGBaseline, ClassicalBaselineConfig
    
    config = ClassicalBaselineConfig()
    baseline = ClassicalECGBaseline(config)
    results = baseline.run_baseline(parquet_file, lead_stats)
"""

from .classical_baseline import (
    ClassicalECGBaseline,
    ClassicalFeatureExtractor, 
    ClassicalECGClassifier,
    ClassicalBaselineConfig
)

from .config import (
    load_config,
    save_config,
    get_dataset_paths,
    detect_lead_stats,
    DEFAULT_LEAD_STATS,
    DEFAULT_CONFIG
)

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "Classical signal processing baseline for ECG analysis"

__all__ = [
    'ClassicalECGBaseline',
    'ClassicalFeatureExtractor',
    'ClassicalECGClassifier', 
    'ClassicalBaselineConfig',
    'load_config',
    'save_config',
    'get_dataset_paths',
    'detect_lead_stats',
    'DEFAULT_LEAD_STATS',
    'DEFAULT_CONFIG'
]
