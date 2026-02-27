from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from utils.enums import ConfigName
from utils.registry import ConfigRegistry
from utils.config.heartwise_config import HeartWiseConfig


@dataclass
@ConfigRegistry.register(ConfigName.ECG_TOKENIZER_GRPO_FINETUNING)
class GRPOFinetuningConfig(HeartWiseConfig):
    # Required core parameters
    pretrained_tokenizer_path: str = ""
    model_name: str = "ECG_Tokenizer_Wrapper"
    runner_name: str = "ECG_Tokenizer_GRPO_Finetuning"
    num_epochs: int = 1
    batch_size: int = 2
    lr: float = 1e-6

    # Output directory base path (required by base project)
    base_checkpoint_path: str = "checkpoints"

    # GRPO-specific parameters
    group_size: int = 4
    epsilon_low: float = 0.2
    epsilon_high: float = 0.3
    beta: float = 0.0  # KL penalty weight (0 = disabled)
    reward_weights: Optional[Dict[str, float]] = None
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95

    # Dataset fields
    train_dataset_path: str = ""
    signal_path_column: str = "signal_path"
    messages_column: str = "messages"
    report_column: str = "report"

    # Training parameters
    num_workers: int = 2
    grad_accum_steps: int = 4
    max_grad_norm: float = 1.0
    save_interval: int = 500
    log_interval: int = 10
    precision: str = "bf16"
    trainable: str = "lora"

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
    max_token_length: Optional[int] = None

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

    def __post_init__(self):
        if self.reward_weights is None:
            self.reward_weights = {
                "format": 0.2,
                "diagnosis": 0.5,
                "evidence": 0.3,
            }
