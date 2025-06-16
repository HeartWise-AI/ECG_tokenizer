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
    GPT-2 decoder for text generation from quantized ECG features.
    
    This decoder takes quantized features from the ECG tokenizer and generates text reports.
    The adapter transforms quantized features to GPT-2's input embedding space.
    """
    def __init__(
        self, 
        gpt2_model_name: str = 'gpt2', 
        gpt2_embedding_size: int = 768, 
        quantized_feature_shape: Tuple[int, int] = (128, 82),  # For SequenceAdapter: (seq_len, features)
        adapter_name: str = "GPT2_SequenceAdapter",  # Changed default to match the 2D shape
        adapter_dropout: float = 0.2,
        label_ignore_index: int = -100,
        # Default generation parameters
        default_do_sample: bool = True,
        default_top_p: float = 0.92,
        default_temperature: float = 0.85,
        default_num_beams: int = 4,
    ):
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
        input_ids: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None, 
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Forward pass for the GPT2Decoder.
        
        Args:
            quantized_features: Quantized features from ECG tokenizer (batch, seq_len, features)
            input_ids: Token IDs for the text input (batch, seq_length) - optional for generation mode
            attention_mask: Optional mask for padding tokens (batch, seq_length)
            labels: Optional labels for computing the language modeling loss (batch, seq_length)
            
        Returns:
            Dictionary containing loss, logits, and other outputs from the GPT-2 model
        """
        # Transform quantized features to GPT-2 embedding space
        quantized_features = quantized_features.to(dtype=torch.float32)
        
        # Add batch dimension if needed for adapter
        if len(quantized_features.shape) == 3:  # (batch, channels, seq_len)
            adapter_input = quantized_features.unsqueeze(1)  # (batch, 1, channels, seq_len)
        else:
            adapter_input = quantized_features
            
        ecg_embedding = self.adapter(adapter_input)  # (batch, embedding_size)
        
        # If no input_ids provided (generation mode), use only ECG embedding
        if input_ids is None:
            return self._generate_from_ecg_embedding(ecg_embedding)
        
        # Training mode: combine ECG embedding with text tokens
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

    def _generate_from_ecg_embedding(self, ecg_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Helper method for generation mode when no input_ids are provided."""
        batch_size = ecg_embedding.size(0)
        
        # Create ECG token
        ecg_token = torch.full(
            (batch_size, 1),
            self.ecg_token_id,
            dtype=torch.long,
            device=ecg_embedding.device
        )
        
        # Create input embedding
        input_embedding = self.gpt2.get_input_embeddings()(ecg_token)
        input_embedding[:, 0, :] = ecg_embedding
        
        # Return the embedding for generation
        return {"inputs_embeds": input_embedding, "ecg_token": ecg_token}

    @torch.no_grad()
    def generate_report(
        self, 
        quantized_features: torch.Tensor,
        max_token_length: int = 512, 
        **generate_kwargs
    ) -> Union[GenerateOutput, torch.Tensor]:
        """
        Generate a clinical report from quantized ECG features.
        
        Args:
            quantized_features: Quantized features from ECG tokenizer (batch, seq_len, features)
            max_token_length: Maximum length of generated tokens
            **generate_kwargs: Additional keyword arguments for generation
            
        Returns:
            Tensor containing generated token IDs for reports
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