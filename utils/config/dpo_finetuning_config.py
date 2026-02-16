from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from utils.enums import ConfigName
from utils.registry import ConfigRegistry
from utils.config.heartwise_config import HeartWiseConfig


@dataclass
@ConfigRegistry.register(ConfigName.ECG_TOKENIZER_DPO_FINETUNING)
class DPOFinetuningConfig(HeartWiseConfig):
    # Required core parameters
    pretrained_tokenizer_path: str
    model_name: str
    runner_name: str
    num_epochs: int
    train_pairs_path: str
    batch_size: int
    lr: float
    beta: float

    # Output directory base path (required by base project)
    base_checkpoint_path: str = "checkpoints"

    # Optional parameters with defaults
    validation_pairs_path: Optional[str] = None
    num_workers: int = 2
    max_token_length: Optional[int] = None
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    save_interval: int = 500
    log_interval: int = 10  # Log metrics every N steps
    eval_pairs_path: Optional[str] = None
    eval_subset_size: int = 0
    eval_batch_size: Optional[int] = None
    eval_interval_steps: int = 0
    eval_seed: int = 42
    sft_weight: float = 0.1
    precision: str = "bf16"  # "bf16", "fp16", "fp32"
    trainable: str = "lora"  # "lora", "bridge", "projection", "all"
    trainable_regex: Optional[str] = None

    # Dataset keys
    waveform_key: str = "waveform_path"
    prompt_key: str = "prompt"
    chosen_key: str = "chosen"
    rejected_key: str = "rejected"
    weight_key: str = "weight"

    # Optional architecture fields (filled from checkpoint if missing)
    encoder_name: Optional[str] = None
    quantizer_name: Optional[str] = None
    decoder_mode: Optional[str] = None
    decoder_name: Optional[str] = None
    tokenizer_name: Optional[str] = None
    huggingface_model_name: Optional[str] = None
    llm_input_embedding_size: Optional[int] = None
    bridge_name: Optional[str] = None
    num_query_tokens: Optional[int] = None
    bridge_mid_dim: Optional[int] = None
    bridge_num_heads: Optional[int] = None
    bridge_dropout: Optional[float] = None
    bridge_num_special_tokens: Optional[int] = None
    bridge_qformer_layers: Optional[int] = None
    bridge_text_hidden_size: Optional[int] = None
    bridge_bias_last_codebook: Optional[float] = None
    bridge_codebook_dropout: Optional[float] = None
    bridge_cross_every: Optional[int] = None
    instruction_dropout: Optional[float] = None

    # ECG/token settings
    num_ecg_tokens: Optional[int] = None
    ecg_token_start_id: Optional[int] = None
    prefix_tuning: bool = False
    medgemma_prompt_style: bool = False
    ecg_waveform_length: int = 2500
    ecg_num_leads: int = 12

    # Codebook settings
    num_quantizers: Optional[int] = None
    codebook_size: Optional[int] = None
    num_codebooks_kept: Optional[int] = None
    codebook_offset: int = 0

    # LoRA settings (filled from checkpoint if missing)
    use_lora: bool = False
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    lora_target_modules: Optional[list] = None
    lora_bias: str = "none"
    lora_config: Optional[Dict[str, Any]] = None

    # Debug/monitoring
    debug_config: Optional[Dict[str, Any]] = None
