from typing import Any
from dataclasses import dataclass
from utils.registry import ConfigRegistry
from utils.config.heartwise_config import HeartWiseConfig

@dataclass
@ConfigRegistry.register("BERT_Report_Classifier")
class BertReportClassifierConfig(HeartWiseConfig):
    # Inference parameters
    reference_reports_path: str
    batch_size: int
    num_workers: int
    run_mode: str
