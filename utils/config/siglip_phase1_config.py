from dataclasses import dataclass
from typing import Any

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

    optimizer: str | dict[str, Any]
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
    bridge_bias_last_codebook: float = 0.5
    bridge_codebook_dropout: float = 0.0
    bridge_mix_strategy: str = "softmax"
    num_query_tokens: int | None = None
    lm_loss_weight: float = 1.0
    siglip_loss_weight: float = 0.5
    ce_max_length: int = 512
    ce_report_probability: float = 0.6
    ce_max_samples_per_batch: int = 64
    lm_weight_warmup_steps: int = 0
    ce_scale_cap: float = 0.0
    component_lr_scales: dict[str, float] | None = None

    llm_unfreeze_last_n_layers: int | None = None
    llm_unfreeze_additional_param_patterns: list[str] | None = None
    llm_unfreeze_lm_head: bool = False

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

    # Phase A: train the ECG encoder end-to-end with the contrastive loss.
    # When False (default) the encoder is frozen and run under no_grad
    # (fully backward-compatible with all existing SigLIP Phase-1 runs).
    train_encoder: bool = False
    # LR for the encoder param group when train_encoder is True. If None,
    # defaults to lr * 0.1 (resolved in the project at optimizer build time).
    encoder_lr: float | None = None

    # Phase A-v2: contrastive target source.
    #   "bank"   -> score ECG features against the fixed 222-label text bank
    #               (default; fully backward-compatible with all prior runs).
    #   "report" -> ESI/MERL-style in-batch InfoNCE against per-ECG free-text
    #               reports embedded with the same text-embedding mechanism.
    contrastive_text_mode: str = "bank"
    # Number of good (gt_rank==1) and bad (largest gt_rank) recall examples to
    # log to a wandb.Table each validation pass when in "report" mode.
    recall_log_examples: int = 8

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
    system_prompts: dict[str, str] | None = None

    # Standardized: prefer text_encoder_model_name; keep medgemma_model_name for backward compatibility
    text_encoder_model_name: str | None = None
    medgemma_model_name: str | None = None

    # Logging of generated token IDs (validation analysis)
    log_generated_token_ids: bool = False
    generated_token_sample_k: int = 50

    # Cap on how many unique val ECGs get a full autoregressive report generated
    # for the inspection CSV each epoch. 0 = unlimited (legacy behavior). On large
    # decoders (e.g. MedGemma-27B) generating all ~3.6k val reports takes ~7h/epoch
    # and only feeds the qualitative CSV — set a small cap (e.g. 64) to keep a
    # sample without the per-epoch stall.
    val_report_generation_max_ecgs: int = 0

    # Backwards compatibility alias for older configs using `model_name`
    @property
    def model_name(self) -> str:
        return self.bridge_name

    @model_name.setter
    def model_name(self, value: str) -> None:
        self.bridge_name = value
