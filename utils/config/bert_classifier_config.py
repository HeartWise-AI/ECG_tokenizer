from typing import Any
from dataclasses import dataclass

from utils.enums import ConfigName
from utils.registry import ConfigRegistry
from utils.config.heartwise_config import HeartWiseConfig

@dataclass
@ConfigRegistry.register(ConfigName.BERT_REPORT_CLASSIFIER)
class BertReportClassifierConfig(HeartWiseConfig):
    # Pipeline parameters
    model_name: str
    runner_name: str
    base_checkpoint_path: str
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
