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
            # Length controls optimized for medical findings format
            "max_new_tokens": 100,  # Reasonable length for medical reports
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
        quantized_features: Optional[torch.Tensor] = None,  # NEW: Explicit for ECG embeddings
        prompt_input_ids: Optional[torch.Tensor] = None,  # For cross-attention without answer leakage
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
        
        # Debug prints (as in original)
        
        if quantized_features is None:
            raise ValueError("quantized_features must be provided for ECG processing")
        
        # Get text embeddings for cross-attention (prompt only, no answers)
        if prompt_input_ids is not None:
            # Use prompt-only tokens to prevent answer leakage in cross-attention
            # Remove padding tokens (pad_token_id is typically 128001 for Llama)
            pad_token_id = getattr(self.tokenizer, 'pad_token_id', 128001)
            text_embeddings_list = []
            for i in range(batch_size):
                # Get non-padding prompt tokens
                valid_mask = prompt_input_ids[i] != pad_token_id
                valid_prompt_ids = prompt_input_ids[i][valid_mask]
                if len(valid_prompt_ids) > 0:
                    prompt_embeds = self.llm_model.get_input_embeddings()(valid_prompt_ids)
                else:
                    # Fallback to empty embedding if no valid prompt
                    prompt_embeds = torch.zeros((1, self.llm_model.config.hidden_size), 
                                               device=prompt_input_ids.device, 
                                               dtype=self.llm_model.dtype)
                text_embeddings_list.append(prompt_embeds)
            # For cross-attention, we need consistent shapes, so we'll use the mean
            # This preserves prompt semantics without leaking answer information
            text_embeddings = torch.stack([emb.mean(dim=0) for emb in text_embeddings_list])
            text_embeddings = text_embeddings.unsqueeze(1)  # Add sequence dimension
        else:
            # Fallback to original behavior (but this should be avoided)
            text_input_ids = input_ids[:, self.num_ecg_tokens:]
            text_embeddings = self.llm_model.get_input_embeddings()(text_input_ids)
        
        # Check if adapter supports cross-modal attention
        adapter_name_str = self.adapter_name.value if hasattr(self.adapter_name, 'value') else str(self.adapter_name)
        
        # Debug logging removed for cleaner output
        
        if 'CrossModal' in adapter_name_str or (hasattr(self.adapter, 'use_cross_attention') and self.adapter.use_cross_attention):
            # Pass text embeddings for cross-attention if adapter supports it
            if hasattr(self.adapter, 'forward') and 'text_embeddings' in self.adapter.forward.__code__.co_varnames:
                # Using cross-attention between ECG and text tokens
                ecg_embeddings = self.adapter(quantized_features, text_embeddings=text_embeddings)
            else:
                # Fallback for adapters without cross-attention support
                ecg_embeddings = self.adapter(quantized_features)
        else:
            # Standard adapter without cross-attention
            ecg_embeddings = self.adapter(quantized_features)
        
        if ecg_embeddings.dim() == 2:
            ecg_embeddings = ecg_embeddings.unsqueeze(1)
        
        # Ensure dtype alignment with model
        model_dtype = self.llm_model.get_input_embeddings().weight.dtype
        if ecg_embeddings.dtype != model_dtype:
            ecg_embeddings = ecg_embeddings.to(model_dtype)
        if text_embeddings.dtype != model_dtype:
            text_embeddings = text_embeddings.to(model_dtype)
        
        # Concatenate ECG embeddings with text embeddings
        input_embeddings = torch.cat([ecg_embeddings, text_embeddings], dim=1)
        # Forward through Llama 3.2
        outputs = self.llm_model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Log attention patterns if enabled
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
        # Handle both 2D and 3D inputs
        if quantized_features.dim() == 2:
            # Add sequence dimension for 2D inputs: (batch, features) -> (batch, 1, features)
            adapter_input = quantized_features.unsqueeze(1)
        else:
            adapter_input = quantized_features
            
        ecg_embedding: torch.Tensor = self.adapter(adapter_input)
        
        # Determine if we're using sequence tokens
        is_sequence_tokens = ecg_embedding.dim() == 3 and ecg_embedding.size(1) > 1
        batch_size: int = ecg_embedding.size(0)
        
        if is_sequence_tokens:
            num_ecg_tokens = ecg_embedding.size(1)
        else:
            num_ecg_tokens = 1
            if ecg_embedding.dim() == 3:
                ecg_embedding = ecg_embedding.squeeze(1)
        
        # Create minimal text prompt using chat template
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
        
        # Prepend ECG token IDs to prompt
        ecg_token_list = list(range(self.ecg_token_start_id, self.ecg_token_start_id + num_ecg_tokens))
        full_prompt_ids_list = ecg_token_list + prompt_ids[: (max_token_length - num_ecg_tokens)]
        full_prompt_ids = torch.tensor(
            full_prompt_ids_list, 
            dtype=torch.long, 
            device=ecg_embedding.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # For sequence: ecg_token_ids already handled via arange
        if is_sequence_tokens:
            ecg_token_ids = torch.arange(
                self.ecg_token_start_id, 
                self.ecg_token_start_id + num_ecg_tokens,
                dtype=torch.long,
                device=ecg_embedding.device
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            ecg_token_ids = torch.full(
                (batch_size, 1),
                self.ecg_token_start_id,
                dtype=torch.long,
                device=ecg_embedding.device
            )
        
        # Create attention mask for full prompt
        attention_mask = torch.ones_like(full_prompt_ids, dtype=torch.long, device=ecg_embedding.device)
        
        # Get embeddings for text tokens only and concatenate with ECG embeddings
        text_input_ids = full_prompt_ids[:, num_ecg_tokens:]
        text_embeddings = self.llm_model.get_input_embeddings()(text_input_ids)
        # Align dtype
        model_dtype = self.llm_model.get_input_embeddings().weight.dtype
        if ecg_embedding.dtype != model_dtype:
            ecg_embedding = ecg_embedding.to(model_dtype)
        if text_embeddings.dtype != model_dtype:
            text_embeddings = text_embeddings.to(model_dtype)
        input_embedding = torch.cat([ecg_embedding, text_embeddings], dim=1)
        
        # Set generation parameters
        generation_params = generate_kwargs.copy()
        generation_params.setdefault("attention_mask", attention_mask)
        generation_params.setdefault("use_cache", True)
        
        # Apply default parameters (includes pad_token_id and eos_token_id)
        for key, value in self.default_generation_params.items():
            generation_params.setdefault(key, value)
        
        # Override max_new_tokens if explicitly provided
        if max_token_length != 256:  # Default value check
            generation_params["max_new_tokens"] = max_token_length
        
        with torch.inference_mode():
            result = self.llm_model.generate(
                inputs_embeds=input_embedding,
                **generation_params
            )
        
        return result

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
        # Handle both 2D and 3D inputs
        if quantized_features.dim() == 2:
            adapter_input = quantized_features.unsqueeze(1)
        else:
            adapter_input = quantized_features

        ecg_embedding: torch.Tensor = self.adapter(adapter_input)

        # Use provided prompt ids/mask (text-only)
        batch_size: int = ecg_embedding.size(0)
        input_ids = prompt_input_ids
        if prompt_attention_mask is None:
            prompt_attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        # Determine sequence length
        is_sequence_tokens = ecg_embedding.dim() == 3 and ecg_embedding.size(1) > 1
        if is_sequence_tokens:
            num_ecg_tokens = ecg_embedding.size(1)
        else:
            num_ecg_tokens = 1
            if ecg_embedding.dim() == 3:
                ecg_embedding = ecg_embedding.squeeze(1)
        
        # Prepend ECG token IDs to input sequence
        ecg_token_tensor = torch.arange(
            self.ecg_token_start_id, 
            self.ecg_token_start_id + num_ecg_tokens,
            dtype=torch.long, 
            device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)
        input_ids = torch.cat([ecg_token_tensor, input_ids], dim=1)
        
        # Adjust attention mask for ECG tokens
        ecg_mask = torch.ones((batch_size, num_ecg_tokens), device=prompt_attention_mask.device, dtype=prompt_attention_mask.dtype)
        attention_mask = torch.cat([ecg_mask, prompt_attention_mask], dim=1)
        
        # Get embeddings for text tokens only and concatenate with ECG embeddings
        text_input_ids = input_ids[:, num_ecg_tokens:]
        text_embeddings = self.llm_model.get_input_embeddings()(text_input_ids)
        # Align dtype
        model_dtype = self.llm_model.get_input_embeddings().weight.dtype
        if ecg_embedding.dtype != model_dtype:
            ecg_embedding = ecg_embedding.to(model_dtype)
        if text_embeddings.dtype != model_dtype:
            text_embeddings = text_embeddings.to(model_dtype)
        input_embedding = torch.cat([ecg_embedding, text_embeddings], dim=1)

        # Set generation parameters
        generation_params = generate_kwargs.copy()
        generation_params.setdefault("attention_mask", attention_mask)
        generation_params.setdefault("use_cache", True)

        # Apply default parameters (includes pad_token_id and eos_token_id)
        for key, value in self.default_generation_params.items():
            generation_params.setdefault(key, value)

        # Override max_new_tokens if explicitly provided
        if max_token_length != 256:  # Default value check
            generation_params["max_new_tokens"] = max_token_length
        
        # Debug output removed for cleaner logs
        
        with torch.inference_mode():
            result = self.llm_model.generate(
                inputs_embeds=input_embedding,
                **generation_params
            )

        return result