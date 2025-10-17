from dataclasses import dataclass

from utils.enums import ConfigName
from utils.registry import ConfigRegistry
from utils.config.heartwise_config import HeartWiseConfig


@dataclass
@ConfigRegistry.register(ConfigName.SIGLIP_PHASE1)
class SiglipPhase1Config(HeartWiseConfig):
    """Configuration for SigLIP Phase-1 alignment training."""

    runner_name: str
    bridge_name: str
    pretrained_encoder_checkpoint: str
    medgemma_model_name: str
    text_bank_csv: str
    mapping_csv: str
    mapping_split: str
    validation_text_bank_csv: str | None
    validation_mapping_csv: str | None
    validation_mapping_split: str | None
    text_embedding_cache_path: str | None

    waveform_length: int
    num_leads: int
    normalize_waveforms: bool
    lead_stats: dict[str, dict[str, float]] | None

    batch_size: int
    num_workers: int
    num_epochs: int
    gradient_accumulation_steps: int

    optimizer: str
    lr: float
    weight_decay: float

    w_pos: float
    w_hardneg: float
    w_implneg: float
    k_impl: int
    max_hardneg_per_group: int
    temperature_init: float

    dtype: str
    log_every_steps: int
    checkpoint_every: int
    checkpoint_dir: str | None

    bridge_hidden_size: int
    bridge_num_layers: int
    bridge_num_heads: int
    bridge_dropout: float
    bridge_max_seq_len: int
    bridge_num_special_tokens: int = 0
    num_codebooks_kept: int | None = None
    codebook_offset: int = 0

    implicit_negatives_per_batch: int = 128
    implicit_negatives_per_row: int | None = None
    negatives_mode: str = "per_batch"
    mutual_exclusion_aux_weight: float = 0.0
    qa_positive_weight_multiplier: float = 1.0
    base_checkpoint_path: str = "checkpoints"
    encoder_name: str | None = None
    quantizer_name: str | None = None
    num_quantizers: int | None = None
    codebook_size: int | None = None
    use_quantized_bridge_inputs: bool = True
    bridge_use_cross_attention: bool | None = None
    bridge_intermediate_dim: int | None = None

    # Focal-InfoNCE knobs
    focal_infonce: bool = False
    focal_gamma_pos: float = 0.0
    focal_gamma_neg: float = 0.0
    focal_alpha_default: float = 1.0
    focal_detach_weights: bool = False

    # Alpha weighting and tail selection
    loss_type: str = "infonce"
    alpha_beta: float = 0.999
    alpha_clip_min: float = 0.5
    alpha_clip_max: float = 8.0
    alpha_csv_name: str = "class_weights.csv"
    class_pos_weight_map: dict[str, float] | None = None

    tail_enable: bool = True
    tail_select_mode: str = "topN"
    tail_top_n: int = 50
    tail_min_positives: int = 10
    tail_alpha_boost: float = 1.0
    tail_class_ids: list[str] | None = None
    tail_include_regex: list[str] | None = None
    tail_exclude_regex: list[str] | None = None

    # Backwards compatibility alias for older configs using `model_name`
    @property
    def model_name(self) -> str:
        return self.bridge_name

    @model_name.setter
    def model_name(self, value: str) -> None:
        self.bridge_name = value
