import math
import torch
import torch.nn as nn

from typing import Union, Optional, Dict, Any, Tuple, cast
from transformers.generation.utils import GenerateOutput
from transformers import LlamaForCausalLM, PreTrainedModel, AutoTokenizer
from transformers import PreTrainedTokenizerBase

from utils.enums import (
    ModelName, 
    AdapterName
)
from utils.registry import ModelRegistry
from models.types import ModelT, ModelClassT
from utils.attention_visualization import ECGAttentionVisualizer, AttentionHook


class ECGCodeBridge(nn.Module):
    """Discrete ECG code bridge with learned-query resampler and RMSNorm projector."""

    uses_codes: bool = True

    def __init__(
        self,
        vocab_size: int,
        d_mid: int,
        d_model: int,
        num_output_tokens: int,
        num_heads: int = 8,
        num_special_tokens: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if d_mid % num_heads != 0:
            raise ValueError(
                f"d_mid ({d_mid}) must be divisible by num_heads ({num_heads}) for ECGCodeBridge."
            )

        self.vocab_size = vocab_size
        self.num_output_tokens = num_output_tokens
        self.num_heads = num_heads
        self.head_dim = d_mid // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.total_vocab = vocab_size + max(0, num_special_tokens)
        self.pad_id = vocab_size
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else None

        self.embed = nn.Embedding(self.total_vocab, d_mid)
        self.norm_in = nn.RMSNorm(d_mid)

        self.q = nn.Parameter(torch.randn(num_output_tokens, d_mid) * (1.0 / math.sqrt(d_mid)))
        self.k_proj = nn.Linear(d_mid, d_mid, bias=False)
        self.v_proj = nn.Linear(d_mid, d_mid, bias=False)

        self.proj = nn.Sequential(
            nn.Linear(d_mid, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm_out = nn.RMSNorm(d_model)

        # Gentle initialisation keeps gradients flowing without shocking the frozen LLM
        first_linear = cast(nn.Linear, self.proj[0])
        nn.init.normal_(first_linear.weight, std=0.02)
        nn.init.zeros_(first_linear.bias)

        final_linear = cast(nn.Linear, self.proj[-1])
        nn.init.normal_(final_linear.weight, std=1e-3)
        nn.init.zeros_(final_linear.bias)

    @property
    def num_tokens(self) -> int:
        return self.num_output_tokens

    def _build_positions(self, length: int, d_model: int, device: torch.device) -> torch.Tensor:
        position = torch.arange(length, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(length, d_model, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _split_heads(self, tensor: torch.Tensor, batch: int) -> torch.Tensor:
        return tensor.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, ecg_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Project discrete ECG ids into LLM embedding space."""

        if ecg_ids.dtype != torch.long:
            ecg_ids = ecg_ids.long()

        batch_size, seq_len = ecg_ids.shape
        device = ecg_ids.device

        x = self.embed(ecg_ids)
        if self.dropout is not None:
            x = self.dropout(x)

        pos = self._build_positions(seq_len, x.size(-1), device)
        x = x + pos.unsqueeze(0)

        x = self.norm_in(x)

        q = self.q.unsqueeze(0).expand(batch_size, -1, -1)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q_heads = self._split_heads(q, batch_size)
        k_heads = self._split_heads(k, batch_size)
        v_heads = self._split_heads(v, batch_size)

        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, 0.0)

        resampled = torch.matmul(attn, v_heads)
        resampled = resampled.transpose(1, 2).contiguous().view(batch_size, self.num_output_tokens, -1)

        out = self.proj(resampled)
        out = self.norm_out(out)
        return out


class ECGProjectionBridge(nn.Module):
    """Lightweight projection bridge that maps ECG features into the LLM space."""

    uses_codes: bool = False

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_tokens: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_tokens = num_tokens

        self.feature_norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.dropout = nn.Dropout(dropout if dropout and dropout > 0 else 0.0)
        self.positional_embedding = nn.Embedding(num_tokens, d_model)
        self.out_norm = nn.LayerNorm(d_model)

        # gentle init on final projection for stable fusion
        final_linear = cast(nn.Linear, self.mlp[-1])
        nn.init.normal_(final_linear.weight, std=1e-3)
        nn.init.zeros_(final_linear.bias)

    def _prepare_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 4:
            # Handle (batch, quantizers, seq, dim) by taking first quantizer
            features = features[:, 0]
        if features.dim() != 3:
            raise ValueError(
                f"ECGProjectionBridge expects [batch, seq, dim] features, got shape {features.shape}"
            )
        return features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self._prepare_features(features)
        batch_size, seq_len, feat_dim = features.shape

        if feat_dim != self.input_dim:
            raise ValueError(
                f"Feature dim mismatch: expected {self.input_dim}, got {feat_dim}"
            )

        if seq_len != self.num_tokens:
            if seq_len < self.num_tokens:
                raise ValueError(
                    f"ECG sequence shorter than expected: {seq_len} < {self.num_tokens}"
                )
            sample_positions = torch.linspace(
                0,
                seq_len - 1,
                self.num_tokens,
                device=features.device,
                dtype=torch.float32
            )
            index = sample_positions.round().long().clamp(max=seq_len - 1)
            index = index.unsqueeze(0).expand(batch_size, -1)
            features = features.gather(
                dim=1,
                index=index.unsqueeze(-1).expand(batch_size, self.num_tokens, feat_dim)
            )
            seq_len = self.num_tokens

        x = self.feature_norm(features)
        x = self.mlp(x)
        x = self.dropout(x)

        positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        pos_emb = self.positional_embedding(positions).unsqueeze(0)
        x = x + pos_emb
        return self.out_norm(x)


@ModelRegistry.register(ModelName.LLAMA32_DECODER)
class Llama32Decoder(nn.Module):
    """
    Llama 3.2 decoder that generates clinical reports from quantized ECG features.
    
    Transforms ECG quantized features into Llama 3.2's embedding space via an adapter,
    then generates text reports using either teacher forcing or ECG-only training.
    """
    def __init__(
        self, 
        huggingface_model_name: str = 'meta-llama/Llama-3.2-1B-Instruct', 
        llm_input_embedding_size: int = 2048, 
        quantized_feature_shape: Tuple[int, int] = (128, 82),
        adapter_name: AdapterName = AdapterName.LLAMA32_SEQUENCE_ADAPTER,
        adapter_dropout: float = 0.2,
        # Sequence token adapter parameters
        use_cross_attention: bool = True,
        num_attention_heads: int = 8,
        intermediate_dim: Optional[int] = None,
        label_ignore_index: int = -100,
        # Quantizer for direct codebook access
        quantizer: Optional[nn.Module] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,  # NEW: Pass tokenizer for adding specials
        ecg_codebook_size: int = 192,
        num_visual_tokens: Optional[int] = None,
        bridge_mid_dim: int = 512,
        bridge_num_heads: int = 8,
        bridge_dropout: float = 0.1,
        bridge_num_special_tokens: int = 4,
        num_quantizers: int = 1,
        # Default generation parameters
        default_do_sample: bool = True,
        default_top_p: float = 0.92,
        default_temperature: float = 0.85,
        default_num_beams: int = 1,
        # Attention visualization parameters
        enable_attention_visualization: bool = False,
        attention_log_frequency: int = 100,
    ):
        """
        Initialize Llama 3.2 decoder.
        
        Args:
            huggingface_model_name: Name/path of the Llama 3.2 model.
            llm_input_embedding_size: Embedding dimension of the Llama 3.2 model.
            quantized_feature_shape: Shape of quantized ECG features (seq_len, features).
            adapter_name: Name of adapter to transform ECG features to Llama 3.2 space.
            adapter_dropout: Dropout rate for the adapter.
            tokenizer: Shared tokenizer instance for adding special tokens.
            label_ignore_index: Index to ignore in loss computation.
            default_do_sample: Default sampling strategy for generation.
            default_top_p: Default nucleus sampling parameter.
            default_temperature: Default temperature for generation.
            default_num_beams: Default number of beams for beam search.
        """
        super(Llama32Decoder, self).__init__()
        
        # Store configuration
        self.label_ignore_index = label_ignore_index
        
        # Store sequence token adapter parameters
        self.use_cross_attention = use_cross_attention
        self.num_attention_heads = num_attention_heads
        self.intermediate_dim = intermediate_dim
        
        # Store quantizer for direct codebook access
        self.quantizer = quantizer
        
        # Ensure tokenizer is available
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name)
        else:
            self.tokenizer = cast(PreTrainedTokenizerBase, tokenizer)
        
        if isinstance(adapter_name, str):
            try:
                adapter_name = AdapterName(adapter_name)
            except ValueError:
                pass

        self.adapter_name = adapter_name
        adapter_name_str = adapter_name.value if hasattr(adapter_name, 'value') else str(adapter_name)

        self.bridge: Optional[ECGCodeBridge] = None
        self.bridge_config: Optional[Dict[str, Any]] = None
        self.adapter: Optional[ModelT] = None
        self._prefix_debug_once = True

        visual_tokens = num_visual_tokens if num_visual_tokens is not None else quantized_feature_shape[0]

        if adapter_name in {
            AdapterName.LLAMA32_ECG_CODE_BRIDGE,
            AdapterName.LLAMA32_ECG_PROJECTION_BRIDGE
        }:
            if adapter_name == AdapterName.LLAMA32_ECG_CODE_BRIDGE:
                self.bridge = ECGCodeBridge(
                    vocab_size=ecg_codebook_size,
                    d_mid=bridge_mid_dim,
                    d_model=llm_input_embedding_size,
                    num_output_tokens=visual_tokens,
                    num_heads=bridge_num_heads,
                    num_special_tokens=bridge_num_special_tokens,
                    dropout=bridge_dropout,
                )
                self.bridge_config = {
                    "style": "code",
                    "vocab_size": ecg_codebook_size,
                    "mid_dim": bridge_mid_dim,
                    "output_tokens": visual_tokens,
                    "num_heads": bridge_num_heads,
                    "dropout": bridge_dropout,
                    "num_special_tokens": bridge_num_special_tokens,
                }
            else:
                feature_dim = quantized_feature_shape[1] if len(quantized_feature_shape) > 1 else llm_input_embedding_size
                self.bridge = ECGProjectionBridge(
                    input_dim=feature_dim,
                    d_model=llm_input_embedding_size,
                    num_tokens=visual_tokens,
                    dropout=bridge_dropout,
                )
                self.bridge_config = {
                    "style": "projection",
                    "input_dim": feature_dim,
                    "output_tokens": visual_tokens,
                    "dropout": bridge_dropout,
                }
            self.num_ecg_tokens = self.bridge.num_tokens
        else:
            self.adapter_class: ModelClassT = ModelRegistry.get(adapter_name)
            if self.adapter_class is None:
                raise ValueError(f"Adapter {adapter_name} not found in ModelRegistry")

            adapter_ctor = cast(Any, self.adapter_class)
            adapter_kwargs = {
                'input_shape': quantized_feature_shape,
                'output_size': llm_input_embedding_size,
                'dropout': adapter_dropout
            }

            if 'SequenceToken' in adapter_name_str:
                adapter_kwargs.update({
                    'use_cross_attention': self.use_cross_attention,
                    'num_attention_heads': self.num_attention_heads,
                    'intermediate_dim': self.intermediate_dim
                })

            self.adapter = adapter_ctor(**adapter_kwargs)

            num_ecg_tokens_raw = getattr(self.adapter, 'num_tokens', 1)
            try:
                num_ecg_tokens = int(num_ecg_tokens_raw)
            except (TypeError, ValueError):
                num_ecg_tokens = quantized_feature_shape[0]
            self.num_ecg_tokens = num_ecg_tokens

        num_ecg_tokens = self.num_ecg_tokens

        # Load the Llama 3.2 model
        self.llm_model: PreTrainedModel = LlamaForCausalLM.from_pretrained(huggingface_model_name)
        
        # Check if embedding size matches Llama 3.2's hidden size
        if llm_input_embedding_size != self.llm_model.config.hidden_size:
            raise ValueError(f"Embedding size {llm_input_embedding_size} does not match Llama 3.2 hidden size {self.llm_model.config.hidden_size}")
        
        # Store reference to LLM for phase-based training
        self.llm = self.llm_model
        
        # Ensure ECG position tokens exist on shared tokenizer and get start id
        ecg_tokens = [f"<|ecg_pos_{i}|>" for i in range(num_ecg_tokens)]
        first_ecg_id = self.tokenizer.convert_tokens_to_ids(ecg_tokens[0])  # type: ignore[attr-defined]
        if first_ecg_id is None or first_ecg_id == -1:
            base_vocab_size = len(self.tokenizer)  # type: ignore[arg-type]
            self.tokenizer.add_tokens(ecg_tokens, special_tokens=True)  # type: ignore[attr-defined]
            self.ecg_token_start_id = base_vocab_size
        else:
            self.ecg_token_start_id = int(first_ecg_id)
        
        # Resize token embeddings to accommodate tokenizer size
        self.llm_model.resize_token_embeddings(len(self.tokenizer), mean_resizing=True)  # type: ignore[arg-type]

        # Initialize ECG token embeddings using text mean (scaled)
        self._initialize_ecg_tokens_semantically(self.ecg_token_start_id, num_ecg_tokens)

        # Prepare gradient mask so only ECG rows stay trainable inside the shared embedding matrix
        mask = torch.zeros(len(self.tokenizer), dtype=torch.bool)
        mask[self.ecg_token_start_id:self.ecg_token_start_id + num_ecg_tokens] = True
        self.register_buffer('_ecg_embedding_train_mask', mask, persistent=False)
        self._ecg_embedding_hook_handle = None
        self._apply_ecg_embedding_mask()
        
        # Set ECG token ID range
        # ECG tokens added to vocabulary
        print(f"   ECG token ID range: [{self.ecg_token_start_id}, {self.ecg_token_start_id + num_ecg_tokens - 1}]")
        print(f"   New vocabulary size: {len(self.llm_model.get_input_embeddings().weight)}")
        
        # Configure pad/eos token ids (coerce to ints)
        def _coerce_id(x):
            if x is None:
                return None
            if isinstance(x, (list, tuple)):
                if len(x) == 0:
                    return None
                return _coerce_id(x[0])
            try:
                return int(x)
            except (TypeError, ValueError):
                return None

        pad_id = _coerce_id(getattr(self.llm_model.config, 'pad_token_id', None))
        eos_id = _coerce_id(getattr(self.llm_model.config, 'eos_token_id', None))
        if pad_id is None:
            pad_id = eos_id if eos_id is not None else 0
        if eos_id is None:
            eos_id = pad_id
        self.pad_token_id = int(pad_id)
        self.eos_token_id = int(eos_id)
        
        # Enhanced EOS token list for better stopping
        # Include multiple Llama 3.2 stop tokens for robust generation control
        self.eos_token_ids = [
            128009,  # <|eot_id|> - primary end of turn token
            128001,  # <|end_of_text|> - end of text token
        ]

        # Pre-compute bad token ids so ECG position tokens never appear in free-form text
        self.bad_ecg_token_ids = [[tid] for tid in range(
            self.ecg_token_start_id,
            self.ecg_token_start_id + num_ecg_tokens
        )]

        # Tokens to suppress at the beginning of generation
        # This prevents the model from generating header tokens
        self.begin_suppress_tokens = [
            128006,  # <|start_header_id|> - should never start with this
            128007,  # <|end_header_id|> - should never start with this  
            128008,  # <|eom_id|> - should never start with this
            128009,  # <|eot_id|> - should never start with this at the beginning
        ]
        
        # Configure generation parameters after token IDs are set
        self.default_generation_params = {
            # Use sampling for more natural medical text
            "do_sample": True,
            "temperature": 0.7,  # Slightly higher for more variation
            "top_p": 0.9,  # Nucleus sampling for quality
            "typical_p": 0.95,  # Encourage diverse but on-topic language
            # Length controls optimized for medical findings format
            "max_new_tokens": 160,  # Allow longer structured findings when needed
            "min_new_tokens": 32,  # Prevent premature termination after a single token
            "length_penalty": 1.05,
            # Moderate repetition control to allow medical terminology repetition
            "repetition_penalty": 1.1,  # Reduced penalty
            "no_repeat_ngram_size": 3,  # Allow some medical phrase repetition
            # Proper stopping behavior
            "early_stopping": False,  # Let it finish naturally
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_ids,
            "bad_words_ids": self.bad_ecg_token_ids,
            # Suppress header tokens at the beginning
            "begin_suppress_tokens": self.begin_suppress_tokens,
        }
        
        # Initialize attention visualization components
        self.enable_attention_visualization = enable_attention_visualization
        self.attention_log_frequency = attention_log_frequency
        self.attention_visualizer = None
        self.attention_hook = None
        self._training_step = 0  # Track training steps for logging frequency
        
        if self.enable_attention_visualization:
            self.attention_visualizer = ECGAttentionVisualizer(num_ecg_tokens=self.num_ecg_tokens)
            self.attention_hook = AttentionHook()
            # Register hook on final transformer layer
            self.attention_hook.register(self.llm_model)
            # Attention visualization enabled
    
    def enable_lora_adapters(self):
        """Enable LoRA adapters if configured."""
        # This would be called from the runner when transitioning to phase 2
        # If using PEFT/LoRA, the adapter would be enabled here
        pass

    def _apply_ecg_embedding_mask(self):
        """Ensure only ECG token rows receive gradients inside shared embeddings."""
        if not hasattr(self, '_ecg_embedding_train_mask'):
            return

        embedding_weight = self.llm_model.get_input_embeddings().weight

        # Remove prior hook to avoid stacking
        if hasattr(self, '_ecg_embedding_hook_handle') and self._ecg_embedding_hook_handle is not None:
            try:
                self._ecg_embedding_hook_handle.remove()
            except RuntimeError:
                pass
            finally:
                self._ecg_embedding_hook_handle = None

        mask_base = self._ecg_embedding_train_mask

        def _mask_gradients(grad: torch.Tensor) -> torch.Tensor:
            mask = mask_base.to(device=grad.device, dtype=grad.dtype).unsqueeze(1)
            return grad * mask

        self._ecg_embedding_hook_handle = embedding_weight.register_hook(_mask_gradients)
        embedding_weight.requires_grad_(True)
    
    def freeze_llm_parameters(self):
        """Freeze all LLM parameters (for phase 1 training)."""
        for param in self.llm_model.parameters():
            param.requires_grad = False
        print("❄️ Froze LLM parameters for alignment phase")
    
    def unfreeze_llm_parameters(self):
        """Unfreeze LLM parameters (for phase 2 training)."""
        for param in self.llm_model.parameters():
            param.requires_grad = True
        print("🔥 Unfroze LLM parameters for fine-tuning phase")
    
    def get_trainable_components(self) -> Dict[str, nn.Module]:
        """Get trainable ECG-specific components."""
        components = {}
        
        # Add adapter
        if getattr(self, 'adapter', None) is not None:
            components['adapter'] = self.adapter
        if self.bridge is not None:
            components['bridge'] = self.bridge
        
        # Add ECG embeddings (part of LLM embeddings)
        if hasattr(self.llm_model, 'get_input_embeddings'):
            components['ecg_embeddings'] = self.llm_model.get_input_embeddings()
        
        # Add cross-attention if available (part of adapter for SequenceTokenAdapter)
        if hasattr(self.adapter, 'cross_attention'):
            components['cross_attention'] = self.adapter.cross_attention
        
        return components

    def _initialize_ecg_tokens_semantically(self, original_vocab_size: int, num_ecg_tokens: int):
        """
        Initialize ECG token embeddings using text embeddings' mean, scaled for stability.
        """
        print(f"🔄 Initializing {num_ecg_tokens} ECG tokens with scaled text mean embeddings...")
        
        with torch.no_grad():
            # Get the current embedding weights
            embeddings = self.llm_model.get_input_embeddings().weight
            text_mean = embeddings[:original_vocab_size].mean(dim=0)
            start = original_vocab_size
            end = original_vocab_size + num_ecg_tokens

            # Initialize ECG embeddings with better scale for learning
            embeddings[start:end] = text_mean.unsqueeze(0).repeat(num_ecg_tokens, 1) * 0.2

    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        quantized_features: Optional[torch.Tensor] = None,
        quantized_codes: Optional[torch.Tensor] = None,
        prompt_input_ids: Optional[torch.Tensor] = None,  # For cross-attention without answer leakage
        prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training: Prepends ECG embeddings to text embeddings.

        Args:
            input_ids: (B, L) with prepended ECG token IDs + text tokens.
            attention_mask: (B, L) full sequence mask.
            labels: (B, L) shifted for causal LM (ignores ECG + prompt).
            quantized_features: Quantized latent features (for legacy adapters).
            quantized_codes: Discrete ECG code indices [B, T] (required for code bridge).

        Returns:
            Dict with 'loss' and 'logits'.
        """
        batch_size = input_ids.size(0)

        if quantized_features is None and self.bridge is None:
            raise ValueError("quantized_features must be provided for ECG processing")

        prompt_text_embeddings = None
        prompt_mask = None
        if prompt_input_ids is not None:
            prompt_text_embeddings = self.llm_model.get_input_embeddings()(prompt_input_ids)
            if prompt_attention_mask is not None:
                prompt_mask = prompt_attention_mask.to(dtype=torch.bool)
            else:
                prompt_mask = (prompt_input_ids != self.pad_token_id)

        text_input_ids = input_ids[:, self.num_ecg_tokens:]
        text_embeddings = self.llm_model.get_input_embeddings()(text_input_ids)
        text_mask = None
        if attention_mask is not None:
            text_mask = attention_mask[:, self.num_ecg_tokens:].to(dtype=torch.bool)

        adapter_name_str = self.adapter_name.value if hasattr(self.adapter_name, 'value') else str(self.adapter_name)

        if self.bridge is not None:
            if getattr(self.bridge, 'uses_codes', False):
                if quantized_codes is None:
                    raise ValueError("quantized_codes must be provided when using the ECG code bridge")
                ecg_ids = quantized_codes
                if isinstance(ecg_ids, tuple):
                    ecg_ids = ecg_ids[0]
                if ecg_ids.dim() == 3 and ecg_ids.size(-1) == 1:
                    ecg_ids = ecg_ids.squeeze(-1)
                elif ecg_ids.dim() == 3:
                    ecg_ids = ecg_ids[..., 0]
                elif ecg_ids.dim() != 2:
                    raise ValueError(
                        f"quantized_codes must be [batch, seq] or [batch, seq, depth]; got {ecg_ids.shape}"
                    )

                ecg_mask = (ecg_ids >= 0)
                ecg_ids = ecg_ids.clamp_min(0).to(device=text_embeddings.device, dtype=torch.long)
                ecg_mask = ecg_mask.to(device=text_embeddings.device)

                ecg_embeddings = self.bridge(ecg_ids, attn_mask=ecg_mask)
            else:
                if quantized_features is None:
                    raise ValueError(
                        "quantized_features must be provided when using the ECG projection bridge"
                    )
                ecg_embeddings = self.bridge(quantized_features.to(text_embeddings.device))
        else:
            if self.adapter is None:
                raise RuntimeError("Adapter is not initialized")

            cross_attn_text_emb = prompt_text_embeddings if prompt_text_embeddings is not None else text_embeddings
            cross_attn_mask = prompt_mask if prompt_mask is not None else text_mask
            if cross_attn_mask is not None and cross_attn_mask.dtype != torch.bool:
                cross_attn_mask = cross_attn_mask.to(dtype=torch.bool)

            if 'CrossModal' in adapter_name_str or (hasattr(self.adapter, 'use_cross_attention') and getattr(self.adapter, 'use_cross_attention', False)):
                if hasattr(self.adapter, 'forward') and 'text_embeddings' in self.adapter.forward.__code__.co_varnames:
                    ecg_embeddings = self.adapter(
                        quantized_features,
                        text_embeddings=cross_attn_text_emb,
                        text_attention_mask=cross_attn_mask
                    )
                else:
                    ecg_embeddings = self.adapter(quantized_features)
            else:
                ecg_embeddings = self.adapter(quantized_features)

        if ecg_embeddings.dim() == 2:
            ecg_embeddings = ecg_embeddings.unsqueeze(1)

        model_dtype = self.llm_model.get_input_embeddings().weight.dtype
        if ecg_embeddings.dtype != model_dtype:
            ecg_embeddings = ecg_embeddings.to(model_dtype)
        if text_embeddings.dtype != model_dtype:
            text_embeddings = text_embeddings.to(model_dtype)

        input_embeddings = torch.cat([ecg_embeddings, text_embeddings], dim=1)

        if self.bridge is not None:
            expected_len = ecg_embeddings.size(1) + text_embeddings.size(1)
            if input_embeddings.size(1) != expected_len:
                raise ValueError(
                    f"Input embedding length mismatch: expected {expected_len}, got {input_embeddings.size(1)}"
                )

            if attention_mask is not None:
                prefix_mask = attention_mask[:, :self.num_ecg_tokens]
                if bool((prefix_mask != 1).any()):
                    raise ValueError("ECG prefix positions must have attention mask == 1")

            if labels is not None:
                prefix_labels = labels[:, :self.num_ecg_tokens]
                if bool((prefix_labels != self.label_ignore_index).any()):
                    raise ValueError("ECG prefix labels must be ignore_index across all positions")

            if not self._prefix_debug_once:
                with torch.no_grad():
                    prefix_norm = ecg_embeddings.norm(dim=-1).mean().item()
                    text_norm = text_embeddings.norm(dim=-1).mean().item() if text_embeddings.numel() > 0 else 0.0
                    print(
                        f"[ECGCodeBridge] prefix_tokens={self.num_ecg_tokens}, "
                        f"prefix_norm={prefix_norm:.4f}, text_norm={text_norm:.4f}"
                    )
                self._prefix_debug_once = True

        outputs = self.llm_model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            labels=labels
        )

        self._log_attention_if_enabled(input_ids)

        
        return outputs

    def _log_attention_if_enabled(self, input_ids: torch.Tensor):
        """Log attention if visualization enabled."""
        if self.enable_attention_visualization and self._training_step % self.attention_log_frequency == 0:
            # Assume self.attention_hook has captured attention; log via visualizer
            if self.attention_hook and self.attention_visualizer is not None:
                attention_weights = self.attention_hook.get_attention()
                if attention_weights is not None:
                    # Use unified visualizer API
                    attention_data = self.attention_visualizer.extract_cross_modal_attention(
                        attention_weights=attention_weights,
                        input_ids=input_ids,
                        ecg_start_idx=0
                    )
                    self.attention_visualizer.log_attention_to_wandb(attention_data, step=self._training_step, log_plots=(self._training_step % (self.attention_log_frequency * 5) == 0))
        self._training_step += 1

    @torch.no_grad()
    def generate_report(
        self, 
        quantized_features: Optional[torch.Tensor] = None,
        quantized_codes: Optional[torch.Tensor] = None,
        max_token_length: int = 256, 
        **generate_kwargs
    ) -> Union[GenerateOutput, torch.Tensor]:
        """
        Generate clinical report from quantized ECG features (ECG-only mode with default prompt).

        Args:
            quantized_features: Quantized ECG features (legacy adapter path).
            quantized_codes: Discrete ECG codes (required for code bridge path).
            max_token_length: Maximum length of generated tokens.
            **generate_kwargs: Additional generation parameters.

        Returns:
            Generated token IDs or GenerateOutput object.
        """
        system_message = (
            "You are a medical expert specialized in ECG interpretation. Provide a concise list "
            "of clinical findings separated by semicolons, similar to standard ECG reports."
        )
        default_user_content = "Analyze this ECG and list the clinical findings."
        messages_prompt = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": default_user_content}
        ]
        prompt_text = cast(
            str,
            self.tokenizer.apply_chat_template(
                messages_prompt,
                tokenize=False,
                add_generation_prompt=True
            )
        )
        prompt_encoding = self.tokenizer.encode_plus(
            prompt_text,
            add_special_tokens=False,
            return_tensors=None
        )
        prompt_ids = prompt_encoding.input_ids

        model_device = self.llm_model.get_input_embeddings().weight.device

        if self.bridge is not None:
            if getattr(self.bridge, 'uses_codes', False):
                if quantized_codes is None:
                    raise ValueError("quantized_codes must be provided when using the ECG code bridge")

                ecg_ids = quantized_codes
                if isinstance(ecg_ids, tuple):
                    ecg_ids = ecg_ids[0]
                if ecg_ids.dim() == 3 and ecg_ids.size(-1) == 1:
                    ecg_ids = ecg_ids.squeeze(-1)
                elif ecg_ids.dim() == 3:
                    ecg_ids = ecg_ids[..., 0]
                elif ecg_ids.dim() != 2:
                    raise ValueError(
                        f"quantized_codes must be [batch, seq] or [batch, seq, depth]; got {ecg_ids.shape}"
                    )

                ecg_mask = (ecg_ids >= 0).to(model_device)
                ecg_ids = ecg_ids.clamp_min(0).to(model_device, dtype=torch.long)
                batch_size = ecg_ids.size(0)
                ecg_embedding = self.bridge(ecg_ids, attn_mask=ecg_mask)
            else:
                if quantized_features is None:
                    raise ValueError("quantized_features must be provided when using the ECG projection bridge")
                features = quantized_features.to(model_device)
                ecg_embedding = self.bridge(features)
                batch_size = ecg_embedding.size(0)

            num_ecg_tokens = ecg_embedding.size(1)

            def build_prompt_tensors(num_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
                max_text_len = max(0, max_token_length - num_tokens)
                trimmed = prompt_ids[:max_text_len] if max_text_len > 0 else []
                mask_value = 1
                if len(trimmed) == 0:
                    trimmed = [self.pad_token_id]
                    mask_value = 0
                prompt_tensor = torch.tensor(trimmed, dtype=torch.long, device=model_device).unsqueeze(0).expand(batch_size, -1)
                prompt_mask = torch.full_like(prompt_tensor, mask_value, dtype=torch.long)
                return prompt_tensor, prompt_mask

            prompt_tensor, prompt_mask = build_prompt_tensors(num_ecg_tokens)
            prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_tensor)

        else:
            if quantized_features is None:
                raise ValueError("quantized_features must be provided for generation when adapter is used")

            if quantized_features.dim() == 2:
                adapter_input = quantized_features.unsqueeze(1)
            else:
                adapter_input = quantized_features

            adapter_input = adapter_input.to(model_device)
            batch_size = adapter_input.size(0)

            num_ecg_tokens_hint = getattr(self.adapter, 'num_tokens', 1) if self.adapter is not None else 1
            try:
                num_ecg_tokens_hint = int(num_ecg_tokens_hint)
            except (TypeError, ValueError):
                num_ecg_tokens_hint = 1

            def build_prompt_tensors(num_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
                max_text_len = max(0, max_token_length - num_tokens)
                trimmed = prompt_ids[:max_text_len] if max_text_len > 0 else []
                mask_value = 1
                if len(trimmed) == 0:
                    trimmed = [self.pad_token_id]
                    mask_value = 0
                prompt_tensor = torch.tensor(trimmed, dtype=torch.long, device=model_device).unsqueeze(0).expand(batch_size, -1)
                prompt_mask = torch.full_like(prompt_tensor, mask_value, dtype=torch.long)
                return prompt_tensor, prompt_mask

            prompt_tensor, prompt_mask = build_prompt_tensors(num_ecg_tokens_hint)
            prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_tensor)
            ecg_embedding = self.adapter(
                adapter_input,
                text_embeddings=prompt_embeddings,
                text_attention_mask=prompt_mask
            )

            is_sequence_tokens = ecg_embedding.dim() == 3 and ecg_embedding.size(1) > 1
            num_ecg_tokens = ecg_embedding.size(1) if is_sequence_tokens else 1
            if not is_sequence_tokens and ecg_embedding.dim() == 3:
                ecg_embedding = ecg_embedding.squeeze(1)

            if num_ecg_tokens != num_ecg_tokens_hint:
                prompt_tensor, prompt_mask = build_prompt_tensors(num_ecg_tokens)
                prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_tensor)
                ecg_embedding = self.adapter(
                    adapter_input,
                    text_embeddings=prompt_embeddings,
                    text_attention_mask=prompt_mask
                )
                is_sequence_tokens = ecg_embedding.dim() == 3 and ecg_embedding.size(1) > 1
                if not is_sequence_tokens and ecg_embedding.dim() == 3:
                    ecg_embedding = ecg_embedding.squeeze(1)
                num_ecg_tokens = ecg_embedding.size(1) if is_sequence_tokens else 1

            if ecg_embedding.dim() == 2:
                ecg_embedding = ecg_embedding.unsqueeze(1)

        ecg_token_ids = torch.arange(
            self.ecg_token_start_id,
            self.ecg_token_start_id + num_ecg_tokens,
            dtype=torch.long,
            device=model_device
        ).unsqueeze(0).expand(batch_size, -1)
        full_prompt_ids = torch.cat([ecg_token_ids, prompt_tensor], dim=1)
        attention_mask = torch.cat([torch.ones_like(ecg_token_ids, dtype=torch.long), prompt_mask], dim=1)

        text_embeddings = prompt_embeddings
        model_dtype = self.llm_model.get_input_embeddings().weight.dtype
        if ecg_embedding.dtype != model_dtype:
            ecg_embedding = ecg_embedding.to(model_dtype)
        if text_embeddings.dtype != model_dtype:
            text_embeddings = text_embeddings.to(model_dtype)
        input_embedding = torch.cat([ecg_embedding, text_embeddings], dim=1)

        generation_params = generate_kwargs.copy()
        max_new_tokens_override = generation_params.get("max_new_tokens")
        min_new_tokens_override = generation_params.get("min_new_tokens")
        generation_params.setdefault("attention_mask", attention_mask)
        generation_params.setdefault("input_ids", full_prompt_ids)
        generation_params.setdefault("use_cache", True)

        for key, value in self.default_generation_params.items():
            generation_params.setdefault(key, value)

        if max_new_tokens_override is None:
            generation_params["max_new_tokens"] = max_token_length
        else:
            generation_params["max_new_tokens"] = min(int(max_new_tokens_override), max_token_length)

        if min_new_tokens_override is None:
            generation_params["min_new_tokens"] = min(
                generation_params.get("min_new_tokens", max_token_length),
                generation_params["max_new_tokens"]
            )
        else:
            generation_params["min_new_tokens"] = min(int(min_new_tokens_override), generation_params["max_new_tokens"])

        with torch.inference_mode():
            result = self.llm_model.generate(
                inputs_embeds=input_embedding,
                **generation_params
            )

        return result

    @torch.no_grad()
    def generate_report_with_question(
        self,
        prompt_input_ids: torch.Tensor,
        quantized_features: Optional[torch.Tensor] = None,
        quantized_codes: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        max_token_length: int = 256,
        **generate_kwargs
    ) -> Union[GenerateOutput, torch.Tensor]:
        """Question-conditioned generation using provided prompt_input_ids (text-only).
        Prepends ECG token IDs and embeddings.
        """
        model_device = self.llm_model.get_input_embeddings().weight.device

        if prompt_attention_mask is None:
            prompt_attention_mask = torch.ones_like(prompt_input_ids, dtype=torch.long)

        prompt_input_ids = prompt_input_ids.to(model_device)
        prompt_attention_mask = prompt_attention_mask.to(model_device)

        if self.bridge is not None:
            if getattr(self.bridge, 'uses_codes', False):
                if quantized_codes is None:
                    raise ValueError("quantized_codes must be provided when using the ECG code bridge")

                ecg_ids = quantized_codes
                if isinstance(ecg_ids, tuple):
                    ecg_ids = ecg_ids[0]
                if ecg_ids.dim() == 3 and ecg_ids.size(-1) == 1:
                    ecg_ids = ecg_ids.squeeze(-1)
                elif ecg_ids.dim() == 3:
                    ecg_ids = ecg_ids[..., 0]
                elif ecg_ids.dim() != 2:
                    raise ValueError(
                        f"quantized_codes must be [batch, seq] or [batch, seq, depth]; got {ecg_ids.shape}"
                    )

                ecg_mask = (ecg_ids >= 0).to(model_device)
                ecg_ids = ecg_ids.clamp_min(0).to(model_device, dtype=torch.long)
                batch_size = ecg_ids.size(0)
                ecg_embedding = self.bridge(ecg_ids, attn_mask=ecg_mask)
            else:
                if quantized_features is None:
                    raise ValueError("quantized_features must be provided when using the ECG projection bridge")
                features = quantized_features.to(model_device)
                ecg_embedding = self.bridge(features)
                batch_size = ecg_embedding.size(0)

            num_ecg_tokens = ecg_embedding.size(1)

            def trim_prompt(num_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
                max_text_len = max(0, max_token_length - num_tokens)
                trimmed_ids = prompt_input_ids[:, :max_text_len] if max_text_len > 0 else prompt_input_ids[:, :0]
                trimmed_mask = prompt_attention_mask[:, :max_text_len] if max_text_len > 0 else prompt_attention_mask[:, :0]
                if trimmed_ids.size(1) == 0:
                    trimmed_ids = torch.full((batch_size, 1), self.pad_token_id, dtype=torch.long, device=model_device)
                    trimmed_mask = torch.zeros_like(trimmed_ids, dtype=torch.long)
                return trimmed_ids, trimmed_mask

            prompt_trimmed, mask_trimmed = trim_prompt(num_ecg_tokens)
            prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_trimmed)

        else:
            if quantized_features is None:
                raise ValueError("quantized_features must be provided for generation when adapter is used")

            if quantized_features.dim() == 2:
                adapter_input = quantized_features.unsqueeze(1)
            else:
                adapter_input = quantized_features

            adapter_input = adapter_input.to(model_device)
            batch_size = adapter_input.size(0)

            num_ecg_tokens_hint = getattr(self.adapter, 'num_tokens', 1)
            try:
                num_ecg_tokens_hint = int(num_ecg_tokens_hint)
            except (TypeError, ValueError):
                num_ecg_tokens_hint = 1

            def trim_prompt(num_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
                max_text_len = max(0, max_token_length - num_tokens)
                trimmed_ids = prompt_input_ids[:, :max_text_len] if max_text_len > 0 else prompt_input_ids[:, :0]
                trimmed_mask = prompt_attention_mask[:, :max_text_len] if max_text_len > 0 else prompt_attention_mask[:, :0]
                if trimmed_ids.size(1) == 0:
                    trimmed_ids = torch.full((batch_size, 1), self.pad_token_id, dtype=torch.long, device=model_device)
                    trimmed_mask = torch.zeros_like(trimmed_ids, dtype=torch.long)
                return trimmed_ids, trimmed_mask

            prompt_trimmed, mask_trimmed = trim_prompt(num_ecg_tokens_hint)
            prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_trimmed)
            ecg_embedding = self.adapter(
                adapter_input,
                text_embeddings=prompt_embeddings,
                text_attention_mask=mask_trimmed
            )

            is_sequence_tokens = ecg_embedding.dim() == 3 and ecg_embedding.size(1) > 1
            num_ecg_tokens = ecg_embedding.size(1) if is_sequence_tokens else 1
            if not is_sequence_tokens and ecg_embedding.dim() == 3:
                ecg_embedding = ecg_embedding.squeeze(1)

            if num_ecg_tokens != num_ecg_tokens_hint:
                prompt_trimmed, mask_trimmed = trim_prompt(num_ecg_tokens)
                prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_trimmed)
                ecg_embedding = self.adapter(
                    adapter_input,
                    text_embeddings=prompt_embeddings,
                    text_attention_mask=mask_trimmed
                )
                is_sequence_tokens = ecg_embedding.dim() == 3 and ecg_embedding.size(1) > 1
                if not is_sequence_tokens and ecg_embedding.dim() == 3:
                    ecg_embedding = ecg_embedding.squeeze(1)
                num_ecg_tokens = ecg_embedding.size(1) if is_sequence_tokens else 1

            if ecg_embedding.dim() == 2:
                ecg_embedding = ecg_embedding.unsqueeze(1)

        ecg_token_tensor = torch.arange(
            self.ecg_token_start_id,
            self.ecg_token_start_id + num_ecg_tokens,
            dtype=torch.long,
            device=model_device
        ).unsqueeze(0).expand(batch_size, -1)
        input_ids = torch.cat([ecg_token_tensor, prompt_trimmed], dim=1)
        attention_mask = torch.cat([torch.ones_like(ecg_token_tensor, dtype=torch.long), mask_trimmed], dim=1)

        text_embeddings = prompt_embeddings
        model_dtype = self.llm_model.get_input_embeddings().weight.dtype
        if ecg_embedding.dtype != model_dtype:
            ecg_embedding = ecg_embedding.to(model_dtype)
        if text_embeddings.dtype != model_dtype:
            text_embeddings = text_embeddings.to(model_dtype)
        input_embedding = torch.cat([ecg_embedding, text_embeddings], dim=1)

        generation_params = generate_kwargs.copy()
        max_new_tokens_override = generation_params.get("max_new_tokens")
        min_new_tokens_override = generation_params.get("min_new_tokens")
        generation_params.setdefault("attention_mask", attention_mask)
        generation_params.setdefault("input_ids", input_ids)
        generation_params.setdefault("use_cache", True)

        for key, value in self.default_generation_params.items():
            generation_params.setdefault(key, value)

        if max_new_tokens_override is None:
            generation_params["max_new_tokens"] = max_token_length
        else:
            generation_params["max_new_tokens"] = min(int(max_new_tokens_override), max_token_length)

        if min_new_tokens_override is None:
            generation_params["min_new_tokens"] = min(
                generation_params.get("min_new_tokens", max_token_length),
                generation_params["max_new_tokens"]
            )
        else:
            generation_params["min_new_tokens"] = min(int(min_new_tokens_override), generation_params["max_new_tokens"])

        with torch.inference_mode():
            result = self.llm_model.generate(
                inputs_embeds=input_embedding,
                **generation_params
            )

        return result
