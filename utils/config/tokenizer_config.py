
from dataclasses import dataclass

from utils.registry import ConfigRegistry
from utils.config.heartwise_config import HeartWiseConfig

@dataclass
@ConfigRegistry.register("ECG_Tokenizer_Training")
class ECGTokenizerTrainingConfig(HeartWiseConfig):
    # Pipeline parameters
    output_dir: str
    run_mode: str
    num_epochs: int
    seed: int
    # Training parameters
    lr: float
    scheduler_name: str
    lr_step_period: int
    factor: float
    optimizer: str
    weight_decay: float
    step_size: int
    gamma: float
    use_amp: bool
    # VQVAE parameters
    encoder_name: str
    quantizer_name: str
    decoder_name: str
    num_quantizers: int
    codebook_size: int
    # Dataset parameters
    train_dataset_path: str
    validation_dataset_path: str
    num_workers: int
    batch_size: int
    # Waveform parameters
    waveform_length: int
    num_leads: int
    normalize_waveforms: bool
    lead_stats: dict[str, dict[str, float]]