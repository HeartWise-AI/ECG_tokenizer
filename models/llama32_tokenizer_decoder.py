import torch
import torch.nn as nn

from typing import Union, Optional, Dict, Any, Tuple
from transformers.generation.utils import GenerateOutput
from transformers import LlamaForCausalLM, LlamaTokenizer, PreTrainedModel

from utils.enums import (
    ModelName, 
    AdapterName
)
from utils.registry import ModelRegistry
from models.types import ModelT, ModelClassT


@ModelRegistry.register(ModelName.LLAMA32_DECODER)
class Llama32Decoder(nn.Module):
    """
    Llama 3.2 decoder that generates clinical reports from quantized ECG features.
    
    Transforms ECG quantized features into Llama 3.2's embedding space via an adapter,
    then generates text reports using either teacher forcing or ECG-only training.
    """
    def __init__(
        self, 
        huggingface_model_name: str = 'meta-llama/Llama-3.2-3B', 
        llm_input_embedding_size: int = 2048, 
        quantized_feature_shape: Tuple[int, int] = (128, 82),
        adapter_name: AdapterName = AdapterName.LLAMA32_SEQUENCE_ADAPTER,
        adapter_dropout: float = 0.2,
        label_ignore_index: int = -100,
        # Default generation parameters
        default_do_sample: bool = True,
        default_top_p: float = 0.92,
        default_temperature: float = 0.85,
        default_num_beams: int = 4,
    ):
        """
        Initialize Llama 3.2 decoder.
        
        Args:
            huggingface_model_name: Name/path of the Llama 3.2 model.
            llm_input_embedding_size: Embedding dimension of the Llama 3.2 model.
            quantized_feature_shape: Shape of quantized ECG features (seq_len, features).
            adapter_name: Name of adapter to transform ECG features to Llama 3.2 space.
            adapter_dropout: Dropout rate for the adapter.
            label_ignore_index: Index to ignore in loss computation.
            default_do_sample: Default sampling strategy for generation.
            default_top_p: Default nucleus sampling parameter.
            default_temperature: Default temperature for generation.
            default_num_beams: Default number of beams for beam search.
        """
        super(Llama32Decoder, self).__init__()
        
        # Store configuration
        self.label_ignore_index = label_ignore_index
        self.default_generation_params = {
            "do_sample": default_do_sample,
            "top_p": default_top_p,
            "temperature": default_temperature,
            "num_beams": default_num_beams,
        }
        
        # Load the adapter class
        self.adapter_class: ModelClassT = ModelRegistry.get(adapter_name)
        if self.adapter_class is None:
            raise ValueError(f"Adapter {adapter_name} not found in ModelRegistry")       
        
        self.adapter_name = adapter_name
        
        # Initialize the adapter to transform quantized features to Llama 3.2 embedding space
        # Input shape: (batch, channels, sequence_length)
        self.adapter: ModelT = self.adapter_class(
            input_shape=quantized_feature_shape,
            output_size=llm_input_embedding_size, 
            dropout=adapter_dropout
        )
        
        # Load the Llama 3.2 model
        self.llm_model: PreTrainedModel = LlamaForCausalLM.from_pretrained(huggingface_model_name)
        
        # Check if embedding size matches Llama 3.2's hidden size
        if llm_input_embedding_size != self.llm_model.config.hidden_size:
            raise ValueError(f"Embedding size {llm_input_embedding_size} does not match Llama 3.2 hidden size {self.llm_model.config.hidden_size}")
        
        # Add special ECG token
        self.llm_model.resize_token_embeddings(len(self.llm_model.get_input_embeddings().weight) + 1)
        self.ecg_token_id = len(self.llm_model.get_input_embeddings().weight) - 1
        self.eos_token_id = self.llm_model.config.eos_token_id

    def forward(
        self, 
        quantized_features: torch.Tensor,
        input_ids: torch.Tensor, 
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None, 
    ) -> Dict[str, Any]:
        """
        Forward pass for training with teacher forcing.
        
        Args:
            quantized_features: Quantized ECG features (batch, channels, seq_len).
            input_ids: Text token IDs (batch, seq_len).
            labels: Target labels for loss computation (batch, seq_len).
            attention_mask: Attention mask for padding (batch, seq_len).
            
        Returns:
            Llama 3.2 model outputs with loss and logits.
        """
        # Transform quantized features to embedding space
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
            Llama 3.2 model outputs with loss and logits.
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
        input_embedding = self.llm_model.get_input_embeddings()(input_ids)
        input_embedding[:, 0, :] = ecg_embedding
        
        # Forward pass through Llama 3.2
        outputs = self.llm_model(
            inputs_embeds=input_embedding,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    @torch.no_grad()
    def generate_report(
        self, 
        quantized_features: torch.Tensor,
        max_token_length: int = 256, 
        **generate_kwargs
    ) -> Union[GenerateOutput, torch.Tensor]:
        """
        Generate clinical report from quantized ECG features.
        
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
        
        # Prepare input
        batch_size: int = ecg_embedding.size(0)
        ecg_token: torch.Tensor = torch.full(
            (batch_size, 1),
            self.ecg_token_id,
            dtype=torch.long,
            device=ecg_embedding.device
        )
        
        # Create attention mask
        attention_mask: torch.Tensor = torch.ones((batch_size, 1), device=ecg_embedding.device)
        
        # Get input embeddings
        input_embedding: torch.Tensor = self.llm_model.get_input_embeddings()(ecg_token)
        
        # Replace ECG token embedding with processed ECG embedding
        input_embedding[:, 0, :] = ecg_embedding

        # Set generation parameters
        generation_params = generate_kwargs.copy()
        generation_params.setdefault("attention_mask", attention_mask)
        generation_params.setdefault("pad_token_id", self.eos_token_id)
        generation_params.setdefault("eos_token_id", self.eos_token_id)
        generation_params.setdefault("use_cache", True)
        
        # Apply default parameters
        for key, value in self.default_generation_params.items():
            generation_params.setdefault(key, value)
        
        # Generate
        with torch.inference_mode():
            result = self.llm_model.generate(
                inputs_embeds=input_embedding,
                max_length=max_token_length,
                **generation_params
            )
        
        return result
