from typing import Any
from dataclasses import dataclass
from utils.registry import ConfigRegistry
from utils.config.heartwise_config import HeartWiseConfig

@dataclass
@ConfigRegistry.register("BERT_Report_Classifier")
class BertReportClassifierConfig(HeartWiseConfig):
    # Pipeline parameters
    output_folder: str
    # Inference parameters
    predictions_reports_path: str
    batch_size: int
    num_workers: int
    run_mode: str
    # Model parameters
    huggingface_model_name: str
    store_model_path: str
    num_classes: int
    api_keys_path: str
