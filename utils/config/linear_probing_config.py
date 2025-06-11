from dataclasses import dataclass

from utils.enums import ConfigName
from utils.registry import ConfigRegistry
from utils.config.heartwise_config import HeartWiseConfig

@dataclass
@ConfigRegistry.register(ConfigName.ECG_TOKENIZER_LINEAR_PROBING)
class ECGTokenizerLinearProbingConfig(HeartWiseConfig):
    # Pipeline parameters
    pretrained_tokenizer_path: str
    runner_name: str
    num_epochs: int
    base_checkpoint_path: str
    
    # Training parameters
    lr: float
    scheduler_type: str
    lr_step_period: int
    factor: float
    optimizer: str
    weight_decay: float
    step_size: int
    gamma: float
    num_warmup_percent: float
    num_hard_restarts_cycles: float
    warm_restart_tmult: int
    
    # VQVAE parameters
    decoder_name: str
    
    # Decoder mode
    decoder_mode: str
    
    # Classification parameters
    num_classes: int
    criterion: str
    
    # Dataset parameters
    train_dataset_path: str
    validation_dataset_path: str
    num_workers: int
    batch_size: int
    
    # Waveform parameters
    num_leads: int
    waveform_length: int
    normalize_waveforms: bool
    lead_stats: dict[str, dict[str, float]]