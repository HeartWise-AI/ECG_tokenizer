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
        
        # Load the adapter class
        self.adapter_class: ModelClassT = ModelRegistry.get(adapter_name)
        if self.adapter_class is None:
            raise ValueError(f"Adapter {adapter_name} not found in ModelRegistry")       
        
        self.adapter_name = adapter_name
        
        # Initialize the adapter to transform quantized features to Llama 3.2 embedding space
        # Input shape: (batch, channels, sequence_length)
        # SequenceAdapter expects input_shape=(seq_len, channels), output_size=hidden_size
        adapter_ctor = cast(Any, self.adapter_class)
        
        # Base adapter parameters
        adapter_kwargs = {
            'input_shape': quantized_feature_shape,
            'output_size': llm_input_embedding_size,
            'dropout': adapter_dropout
        }
        
        # Add sequence token specific parameters if using sequence token adapter
        adapter_name_str = adapter_name.value if hasattr(adapter_name, 'value') else str(adapter_name)
        if 'SequenceToken' in adapter_name_str:
            # Only SequenceTokenAdapter supports cross-attention parameters
            adapter_kwargs.update({
                'use_cross_attention': self.use_cross_attention,
                'num_attention_heads': self.num_attention_heads,
                'intermediate_dim': self.intermediate_dim
            })
        # SimpleTokenAdapter doesn't support cross-attention parameters
        
        self.adapter: ModelT = adapter_ctor(**adapter_kwargs)
        
        # Load the Llama 3.2 model
        self.llm_model: PreTrainedModel = LlamaForCausalLM.from_pretrained(huggingface_model_name)
        
        # Check if embedding size matches Llama 3.2's hidden size
        if llm_input_embedding_size != self.llm_model.config.hidden_size:
            raise ValueError(f"Embedding size {llm_input_embedding_size} does not match Llama 3.2 hidden size {self.llm_model.config.hidden_size}")
        

        num_ecg_tokens_raw = getattr(self.adapter, 'num_tokens', 1)
        try:
            num_ecg_tokens = int(num_ecg_tokens_raw)
        except (TypeError, ValueError):
            num_ecg_tokens = quantized_feature_shape[0]  # Default to seq_len
        
        self.num_ecg_tokens = num_ecg_tokens

        # Store reference to LLM for phase-based training
        self.llm = self.llm_model

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
        self.ecg_prefix_token_id = self.pad_token_id
        
        # Enhanced EOS token list for better stopping
        # Include multiple Llama 3.2 stop tokens for robust generation control
        self.eos_token_ids = [
            128009,  # <|eot_id|> - primary end of turn token
            128001,  # <|end_of_text|> - end of text token
        ]

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
            "min_new_tokens": 5,  # Prevent premature termination after a single token
            "length_penalty": 1.05,
            # Moderate repetition control to allow medical terminology repetition
            "repetition_penalty": 1.1,  # Reduced penalty
            "no_repeat_ngram_size": 3,  # Allow some medical phrase repetition
            # Proper stopping behavior
            "early_stopping": False,  # Let it finish naturally
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_ids,
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
        if hasattr(self, 'adapter'):
            components['adapter'] = self.adapter
        
        # Add cross-attention if available (part of adapter for SequenceTokenAdapter)
        if hasattr(self.adapter, 'cross_attention'):
            components['cross_attention'] = self.adapter.cross_attention
        
        return components

    def _mask_input_prefix(self, sequences: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Replace the prefix (ECG + prompt) tokens with pad so decoded text omits them."""
        if mask is None:
            return sequences
        masked = sequences.clone()
        prefix_lengths = mask.sum(dim=1)
        for idx, length in enumerate(prefix_lengths.tolist()):
            if length > 0:
                masked[idx, :int(length)] = self.pad_token_id
        return masked

    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        quantized_features: Optional[torch.Tensor] = None,  # NEW: Explicit for ECG embeddings
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
            quantized_features: (B, 1, 128, 82) or (B, 128, 82) for adapter.

        Returns:
            Dict with 'loss' and 'logits'.
        """
        batch_size = input_ids.size(0)

        if quantized_features is None:
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

        cross_attn_text_emb = prompt_text_embeddings if prompt_text_embeddings is not None else text_embeddings
        cross_attn_mask = prompt_mask if prompt_mask is not None else text_mask
        if cross_attn_mask is not None and cross_attn_mask.dtype != torch.bool:
            cross_attn_mask = cross_attn_mask.to(dtype=torch.bool)

        if 'CrossModal' in adapter_name_str or (hasattr(self.adapter, 'use_cross_attention') and self.adapter.use_cross_attention):
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
        quantized_features: torch.Tensor,
        max_token_length: int = 256, 
        **generate_kwargs
    ) -> Union[GenerateOutput, torch.Tensor]:
        """
        Generate clinical report from quantized ECG features (ECG-only mode with default prompt).

        Args:
            quantized_features: Quantized ECG features (batch, channels, seq_len).
            max_token_length: Maximum length of generated tokens.
            **generate_kwargs: Additional generation parameters.

        Returns:
            Generated token IDs or GenerateOutput object.
        """
        if quantized_features.dim() == 2:
            adapter_input = quantized_features.unsqueeze(1)
        else:
            adapter_input = quantized_features

        batch_size: int = adapter_input.size(0)
        device = adapter_input.device

        num_ecg_tokens_hint = getattr(self.adapter, 'num_tokens', 1)
        try:
            num_ecg_tokens_hint = int(num_ecg_tokens_hint)
        except (TypeError, ValueError):
            num_ecg_tokens_hint = 1

        system_message = "You are a medical expert specialized in ECG interpretation. Provide a concise list of clinical findings separated by semicolons, similar to standard ECG reports."
        default_user_content = "Analyze this ECG and list the clinical findings."
        messages_prompt = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": default_user_content}
        ]
        prompt_text = cast(str, self.tokenizer.apply_chat_template(
            messages_prompt, 
            tokenize=False, 
            add_generation_prompt=True
        ))
        prompt_encoding = self.tokenizer.encode_plus(
            cast(str, prompt_text),
            add_special_tokens=False,
            return_tensors=None
        )
        prompt_ids = prompt_encoding.input_ids

        def build_prompt_tensors(num_ecg_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
            max_text_len = max(0, max_token_length - num_ecg_tokens)
            trimmed = prompt_ids[:max_text_len] if max_text_len > 0 else []
            mask_value = 1
            if len(trimmed) == 0:
                trimmed = [self.pad_token_id]
                mask_value = 0
            prompt_tensor = torch.tensor(trimmed, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
            prompt_mask = torch.full_like(prompt_tensor, mask_value, dtype=torch.long)
            return prompt_tensor, prompt_mask

        prompt_tensor, prompt_mask = build_prompt_tensors(num_ecg_tokens_hint)
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_tensor)
        ecg_embedding: torch.Tensor = self.adapter(
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

        ecg_token_ids = torch.full(
            (batch_size, num_ecg_tokens),
            fill_value=self.ecg_prefix_token_id,
            dtype=torch.long,
            device=device
        )
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

        sequences = result.sequences if hasattr(result, "sequences") else result
        stripped = self._mask_input_prefix(sequences, attention_mask.to(sequences.device))
        if hasattr(result, "sequences"):
            result.sequences = stripped
            return result
        return stripped

    @torch.no_grad()
    def generate_report_with_question(
        self,
        quantized_features: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        max_token_length: int = 256,
        **generate_kwargs
    ) -> Union[GenerateOutput, torch.Tensor]:
        """Question-conditioned generation using provided prompt_input_ids (text-only).
        Prepends ECG token IDs and embeddings.
        """
        if quantized_features.dim() == 2:
            adapter_input = quantized_features.unsqueeze(1)
        else:
            adapter_input = quantized_features

        batch_size: int = adapter_input.size(0)
        device = adapter_input.device

        num_ecg_tokens_hint = getattr(self.adapter, 'num_tokens', 1)
        try:
            num_ecg_tokens_hint = int(num_ecg_tokens_hint)
        except (TypeError, ValueError):
            num_ecg_tokens_hint = 1

        if prompt_attention_mask is None:
            prompt_attention_mask = torch.ones_like(prompt_input_ids, dtype=torch.long)

        prompt_input_ids = prompt_input_ids.to(device)
        prompt_attention_mask = prompt_attention_mask.to(device)

        def trim_prompt(num_ecg_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
            max_text_len = max(0, max_token_length - num_ecg_tokens)
            if max_text_len > 0:
                trimmed_ids = prompt_input_ids[:, :max_text_len]
                trimmed_mask = prompt_attention_mask[:, :max_text_len]
            else:
                trimmed_ids = prompt_input_ids[:, :0]
                trimmed_mask = prompt_attention_mask[:, :0]
            if trimmed_ids.size(1) == 0:
                trimmed_ids = torch.full((batch_size, 1), self.pad_token_id, dtype=torch.long, device=device)
                trimmed_mask = torch.zeros_like(trimmed_ids, dtype=torch.long)
            return trimmed_ids, trimmed_mask

        prompt_trimmed, mask_trimmed = trim_prompt(num_ecg_tokens_hint)
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_trimmed)
        ecg_embedding: torch.Tensor = self.adapter(
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

        ecg_token_tensor = torch.full(
            (batch_size, num_ecg_tokens),
            fill_value=self.ecg_prefix_token_id,
            dtype=torch.long,
            device=device
        )
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
        sequences = result.sequences if hasattr(result, "sequences") else result
        stripped = self._mask_input_prefix(sequences, attention_mask.to(sequences.device))
        if hasattr(result, "sequences"):
            result.sequences = stripped
            return result
        return stripped
