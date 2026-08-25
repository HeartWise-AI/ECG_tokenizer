from dataclasses import dataclass
from typing import Any, Dict

from utils.enums import ConfigName
from utils.registry import ConfigRegistry
from utils.config.heartwise_config import HeartWiseConfig


@dataclass
@ConfigRegistry.register(ConfigName.ECG_TEXT_STAGE1)
class ECGTextStage1Config(HeartWiseConfig):
    """Configuration for Stage-1 ECG ↔ text alignment without an LLM."""

    runner_name: str
    bridge_name: str
    pretrained_encoder_checkpoint: str
    text_bank_csv: str
    mapping_csv: str
    mapping_split: str
    validation_text_bank_csv: str | None
    validation_mapping_csv: str | None
    validation_mapping_split: str | None
    text_encoder_model_name: str
    text_encoder_output_dim: int
    text_encoder_dropout: float
    text_encoder_freeze_ratio: float
    base_checkpoint_path: str

    waveform_length: int
    num_leads: int
    normalize_waveforms: bool
    lead_stats: Dict[str, Dict[str, float]] | None

    batch_size: int
    num_workers: int
    num_epochs: int
    gradient_accumulation_steps: int

    optimizer: Dict[str, Any] | str | None
    lr: float
    weight_decay: float

    bridge_hidden_size: int
    bridge_num_layers: int
    bridge_num_heads: int
    bridge_dropout: float
    bridge_max_seq_len: int
    bridge_num_special_tokens: int
    bridge_bias_last_codebook: float
    bridge_codebook_dropout: float
    num_query_tokens: int
    num_codebooks_kept: int | None
    codebook_offset: int
    num_quantizers: int | None
    codebook_size: int | None

    etc_weight: float
    etm_weight: float
    etg_weight: float
    max_text_length: int
    etg_delay_steps: int
    etg_warmup_steps: int
    etm_hard_neg_k: int
    etm_warmup_steps: int
    cross_every: int

    use_quantized_inputs: bool
    dtype: str
    log_every_steps: int
    checkpoint_every: int
    checkpoint_dir: str | None
    bridge_mix_strategy: str = "softmax"
    bridge_token_axis: str = "channel"
    tail_enable: bool = False
    tail_select_mode: str = "topN"
    tail_top_n: int = 50
    tail_min_positives: int = 10
    tail_alpha_boost: float = 1.0
    tail_class_ids: list[str] | None = None
    tail_include_regex: list[str] | None = None
    tail_exclude_regex: list[str] | None = None
    text_encoder_lr_mult: float = 0.0
    text_encoder_wd_mult: float = 1.0
    dec_token_id: int | None = None
    max_positives_per_ecg: int = 1
    siglip_bank_negatives: int = 256
    bank_refresh_every_steps: int = 256
    bank_refresh_batch_size: int = 2048
    focal_infonce: bool = False
    focal_gamma_pos: float = 0.0
    focal_gamma_neg: float = 0.0
    focal_alpha_default: float = 1.0
    focal_detach_weights: bool = False
    class_pos_weight_map: Dict[str, float] | None = None

    # Validation ETG (teacher-forcing) evaluation controls
    validate_with_etg: bool = True
    validate_etg_fraction: float = 0.10
    validate_etg_save: bool = True
