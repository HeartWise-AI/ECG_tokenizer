import torch
import torch.nn as nn

from typing import Union, Optional, Dict, Any, Tuple
from transformers.generation.utils import GenerateOutput
from transformers import GPT2LMHeadModel, PreTrainedModel

from models.adapters import (
    LinearAdapter, 
    EmbeddingAdapter, 
    SimpleEmbeddingAdapter,
    SequenceAdapter
)
from utils.enums import ModelName
from utils.registry import ModelRegistry


@ModelRegistry.register(ModelName.GPT2_DECODER)
class GPT2Decoder(nn.Module):
    """
    GPT-2 decoder that generates clinical reports from quantized ECG features.
    
    Transforms ECG quantized features into GPT-2's embedding space via an adapter,
    then generates text reports using either teacher forcing or ECG-only training.
    """
    def __init__(
        self, 
        gpt2_model_name: str = 'gpt2', 
        gpt2_embedding_size: int = 768, 
        quantized_feature_shape: Tuple[int, int] = (128, 82),
        adapter_name: str = "GPT2_SequenceAdapter",
        adapter_dropout: float = 0.2,
        label_ignore_index: int = -100,
        # Default generation parameters
        default_do_sample: bool = True,
        default_top_p: float = 0.92,
        default_temperature: float = 0.85,
        default_num_beams: int = 4,
    ):
        """
        Initialize the GPT-2 decoder.
        
        Args:
            gpt2_model_name: Pre-trained GPT-2 model name from HuggingFace.
            gpt2_embedding_size: GPT-2 embedding dimension (must match model).
            quantized_feature_shape: Shape of quantized ECG features (seq_len, features).
            adapter_name: Name of adapter to transform ECG features to GPT-2 space.
            adapter_dropout: Dropout rate for the adapter.
            label_ignore_index: Index to ignore in loss computation.
            default_do_sample: Default sampling strategy for generation.
            default_top_p: Default nucleus sampling parameter.
            default_temperature: Default temperature for generation.
            default_num_beams: Default number of beams for beam search.
        """
        super(GPT2Decoder, self).__init__()
        
        # Store configuration
        self.label_ignore_index = label_ignore_index
        self.default_generation_params = {
            "do_sample": default_do_sample,
            "top_p": default_top_p,
            "temperature": default_temperature,
            "num_beams": default_num_beams,
        }
        
        # Load the adapter class
        self.adapter_class: Union[
            EmbeddingAdapter, 
            LinearAdapter, 
            SimpleEmbeddingAdapter,
            SequenceAdapter
        ] = ModelRegistry.get(adapter_name)
        if self.adapter_class is None:
            raise ValueError(f"Adapter {adapter_name} not found in ModelRegistry")       
        
        self.adapter_name = adapter_name
        
        # Initialize the adapter to transform quantized features to GPT-2 embedding space
        # Input shape: (batch, channels, sequence_length)
        self.adapter = self.adapter_class(
            input_shape=quantized_feature_shape,
            output_size=gpt2_embedding_size, 
            dropout=adapter_dropout
        )
        
        # Load the GPT-2 model
        self.gpt2: PreTrainedModel = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        
        # Check if embedding size matches GPT-2's hidden size
        if gpt2_embedding_size != self.gpt2.config.n_embd:
            raise ValueError(f"Embedding size {gpt2_embedding_size} does not match GPT-2 hidden size {self.gpt2.config.n_embd}")
        
        # Add special ECG token
        self.gpt2.resize_token_embeddings(len(self.gpt2.get_input_embeddings().weight) + 1)
        self.ecg_token_id = len(self.gpt2.get_input_embeddings().weight) - 1
        self.eos_token_id = self.gpt2.config.eos_token_id
        
    def forward(
        self, 
        quantized_features: torch.Tensor,
        input_ids: torch.Tensor, 
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None, 
    ) -> Dict[str, Any]:
        """
        Forward pass for training or generation setup.
        
        Args:
            quantized_features: ECG features from tokenizer (batch, seq_len, features).
            input_ids: Text token IDs (batch, seq_len). Required for training.
            attention_mask: Attention mask for padding tokens (batch, seq_len).
            labels: Target labels for loss computation (batch, seq_len).
            
        Returns:
            Dictionary with loss, logits, and other GPT-2 outputs.
            
        Raises:
            ValueError: If input_ids is None during training.
        """
        # Transform quantized features to GPT-2 embedding space
        quantized_features = quantized_features.to(dtype=torch.float32)
        ecg_embedding = self.adapter(quantized_features)  # (batch, embedding_size)       

        return self._forward_teacher_forcing(ecg_embedding, input_ids, labels, attention_mask)

    def _forward_teacher_forcing(
        self, 
        ecg_embedding: torch.Tensor,
        input_ids: torch.Tensor, 
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None, 
    ) -> Dict[str, Any]:
        """
        Forward pass with teacher forcing (standard training approach).
        
        Prepends ECG token to input sequence and replaces its embedding
        with the processed ECG signal embedding.
        
        Args:
            ecg_embedding: Processed ECG features (batch, embedding_dim).
            input_ids: Text token IDs (batch, seq_len).
            attention_mask: Attention mask for padding (batch, seq_len).
            labels: Target labels for loss computation (batch, seq_len).
            
        Returns:
            GPT-2 model outputs with loss and logits.
        """
        batch_size = input_ids.size(0)
        
        # Prepend the special ECG token ID to input_ids
        ecg_token = torch.full(
            (batch_size, 1),
            self.ecg_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device
        )
        input_ids = torch.cat([ecg_token, input_ids], dim=1)
        
        # Adjust attention_mask if provided
        if attention_mask is not None:
            ecg_mask = torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([ecg_mask, attention_mask], dim=1)
        else:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        
        # Adjust labels if provided
        if labels is not None:
            label_ignore = torch.full(
                (batch_size, 1),
                self.label_ignore_index,
                dtype=labels.dtype,
                device=labels.device
            )
            labels = torch.cat([label_ignore, labels], dim=1)
        
        # Get input embeddings and replace the first token's embedding with ECG embedding
        input_embedding = self.gpt2.get_input_embeddings()(input_ids)
        input_embedding[:, 0, :] = ecg_embedding
        
        # Forward pass through GPT-2
        outputs = self.gpt2(
            inputs_embeds=input_embedding,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    @torch.no_grad()
    def generate_report(
        self, 
        quantized_features: torch.Tensor,
        max_token_length: int = 512, 
        **generate_kwargs
    ) -> Union[GenerateOutput, torch.Tensor]:
        """
        Generate clinical report from quantized ECG features.
        
        Args:
            quantized_features: ECG features from tokenizer (batch, seq_len, features).
            max_token_length: Maximum number of tokens to generate.
            **generate_kwargs: Additional parameters for GPT-2 generation.
            
        Returns:
            Generated token IDs (batch, generated_length).
            
        Example:
            >>> features = tokenizer.encode(ecg_signal)  # (1, 128, 82)
            >>> tokens = decoder.generate_report(features, max_token_length=100)
            >>> report = tokenizer.decode(tokens[0])
        """
        # Transform features to embedding space
        if len(quantized_features.shape) == 3:
            adapter_input = quantized_features.unsqueeze(1)
        else:
            adapter_input = quantized_features
            
        ecg_embedding = self.adapter(adapter_input)
        
        # Prepare input
        batch_size = ecg_embedding.size(0)
        ecg_token = torch.full(
            (batch_size, 1),
            self.ecg_token_id,
            dtype=torch.long,
            device=ecg_embedding.device
        )
        
        # Create attention mask
        attention_mask = torch.ones((batch_size, 1), device=ecg_embedding.device)
        
        # Get input embeddings
        input_embedding = self.gpt2.get_input_embeddings()(ecg_token)
        input_embedding[:, 0, :] = ecg_embedding

        # Set generation parameters
        generate_kwargs = generate_kwargs.copy()
        generate_kwargs.setdefault("attention_mask", attention_mask)
        generate_kwargs.setdefault("pad_token_id", self.eos_token_id)
        generate_kwargs.setdefault("eos_token_id", self.eos_token_id)
        generate_kwargs.setdefault("use_cache", True)
        
        # Apply default parameters
        for key, value in self.default_generation_params.items():
            generate_kwargs.setdefault(key, value)
        
        # Generate
        with torch.inference_mode():
            result = self.gpt2.generate(
                inputs_embeds=input_embedding,
                max_length=max_token_length,
                **generate_kwargs
            )
        
        return result