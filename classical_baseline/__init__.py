"""
Classical ECG Baseline Package

This package provides a classical signal processing baseline for ECG analysis
that integrates with the ECG_tokenizer infrastructure.

Main components:
- ClassicalECGBaseline: Main pipeline class
- ClassicalBaselineConfig: Configuration management
- SAXConfig, SAXExtractor, SAXBaseline: SAX-based feature extraction

Example usage:
    from classical_baseline import ClassicalECGBaseline, ClassicalBaselineConfig
    
    config = ClassicalBaselineConfig()
    baseline = ClassicalECGBaseline(config)
    results = baseline.run_baseline(parquet_file, lead_stats)
"""

from .classical_baseline import (
    ClassicalECGBaseline,
    ClassicalBaselineConfig,
    SAXConfig,
    SAXExtractor,
    SAXBaseline
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
    'ClassicalBaselineConfig',
    'SAXConfig',
    'SAXExtractor',
    'SAXBaseline',
    'load_config',
    'save_config',
    'get_dataset_paths',
    'detect_lead_stats',
    'DEFAULT_LEAD_STATS',
    'DEFAULT_CONFIG'
]
