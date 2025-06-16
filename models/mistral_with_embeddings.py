import torch
import torch.nn as nn
from transformers import MistralForCausalLM, MistralConfig

from utils.registry import ModelRegistry
from models.embedding_reducer import EmbeddingReducer
from models.linear_reducer import LinearReducer
from models.simple_embedding_reducer import SimpleEmbeddingReducer
from typing import Union, Optional, Dict, Any


@ModelRegistry.register("Mistral_WithEmbedding")
class MistralWithEmbedding(nn.Module):
    def __init__(
        self, 
        mistral_model_name: str = 'mistralai/Mistral-7B-v0.1', 
        mistral_embedding_size: int = 4096, 
        ecg_embedding_size: tuple[int, int, int] = (8, 128, 82),
        reducer_name: str = None,
        reducer_dropout: float = 0.2
    ):
        super(MistralWithEmbedding, self).__init__()
        self.mistral: MistralForCausalLM = MistralForCausalLM.from_pretrained(mistral_model_name)
        self.embedding_reducer: Union[
            EmbeddingReducer, 
            LinearReducer, 
            SimpleEmbeddingReducer
        ] = ModelRegistry.get(reducer_name)(
            input_shape=ecg_embedding_size,
            output_size=mistral_embedding_size, 
            dropout=reducer_dropout
        )
        
        # Check if embedding size matches Mistral's hidden size.
        if mistral_embedding_size != self.mistral.config.hidden_size:
            raise ValueError(f"Embedding size {mistral_embedding_size} does not match Mistral hidden size {self.mistral.config.hidden_size}")
        
        # Optional: Maintain the special token ID (if needed elsewhere).
        # We still resize token embeddings for compatibility during generation.
        self.mistral.resize_token_embeddings(len(self.mistral.get_input_embeddings().weight) + 1)
        self.ecg_token_id = len(self.mistral.get_input_embeddings().weight) - 1  # New token ID
        # Make sure EOS token is defined
        self.eos_token_id = self.mistral.config.eos_token_id

    def forward(
        self, 
        ecg_embeddings: torch.Tensor, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None, 
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Forward pass for the MistralWithEmbedding model.
        
        Args:
            ecg_embeddings: Tensor containing ECG embeddings (batch, *ecg_dims)
            input_ids: Token IDs for the text input (batch, seq_length)
            attention_mask: Optional mask for padding tokens (batch, seq_length)
            labels: Optional labels for computing the language modeling loss (batch, seq_length)
            
        Returns:
            Dictionary containing loss, logits, and other outputs from the Mistral model
            
        Note:
            - During training, ensure your target sequences end with an EOS token for better generation
            - The model prepends a special ECG token to the input sequence
        """
        
        # Reduce ECG embeddings
        reduced: torch.Tensor = self.embedding_reducer(ecg_embeddings)  # (batch, embedding_size)
        
        # Prepend the special ECG token ID to input_ids
        batch_size: int = input_ids.size(0)
        ecg_token = torch.full(
            (batch_size, 1),
            self.ecg_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device
        )  # (batch, 1)
        input_ids = torch.cat([ecg_token, input_ids], dim=1)  # (batch, seq_length + 1)
        
        # Adjust attention_mask if provided
        if attention_mask is not None:
            ecg_mask = torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([ecg_mask, attention_mask], dim=1)  # (batch, seq_length + 1)
        else:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        
        # Adjust labels if provided
        if labels is not None:
            label_ignore = torch.full(
                (batch_size, 1),
                -100,
                dtype=labels.dtype,
                device=labels.device
            )
            labels = torch.cat([label_ignore, labels], dim=1)  # (batch, seq_length + 1)
            
            # Check if labels end with EOS token, add if missing
            # This helps model learn proper ending of reports
            eos_check = (labels[:, -1] == self.eos_token_id)
            if not torch.all(eos_check):
                # For those without EOS, append it (if needed for your use case)
                # Note: Enable this only if your dataset doesn't already have EOS tokens
                # Commented out as it depends on your preprocessing
                # eos_token = torch.full((batch_size, 1), self.eos_token_id, dtype=labels.dtype, device=labels.device)
                # labels = torch.cat([labels, eos_token], dim=1)  # (batch, seq_length + 2)
                pass
        
        # Get input embeddings and replace the first token's embedding with reduced ECG embedding
        input_embedding = self.mistral.get_input_embeddings()(input_ids)  # (batch, seq_length + 1, hidden_size)
        input_embedding[:, 0, :] = reduced  # Replace ECG token embedding
        
        # Forward pass through Mistral
        outputs = self.mistral(
            inputs_embeds=input_embedding,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    def generate_report(
        self, 
        ecg_embeddings: torch.Tensor, 
        max_token_length: int = 512, 
        **generate_kwargs
    ) -> torch.Tensor:
        """
        Generate a clinical report conditioned solely on the ECG embeddings.
        This method uses Mistral's generate() function with inputs_embeds.
        
        Args:
            ecg_embeddings: Tensor containing ECG embeddings
            max_token_length: Maximum length of generated tokens
            **generate_kwargs: Additional keyword arguments for generation
            
        Returns:
            Tensor containing generated token IDs for reports
            
        Note:
            This method ensures generation stops properly by using the EOS token
        """
        # Reduce ECG embeddings
        reduced = self.embedding_reducer(ecg_embeddings)  # (batch, embedding_size)
        
        # Prepare input_ids with the ECG token
        batch_size = reduced.size(0)
        ecg_token = torch.full(
            (batch_size, 1),
            self.ecg_token_id,
            dtype=torch.long,
            device=reduced.device
        )  # (batch, 1)
        
        # Create attention mask
        attention_mask = torch.ones((batch_size, 1), device=reduced.device)  # (batch, 1)
        
        # Get input embeddings
        input_embedding = self.mistral.get_input_embeddings()(ecg_token)  # (batch, 1, hidden_size)
        input_embedding[:, 0, :] = reduced  # Replace ECG token embedding
        
        # Set default generation parameters
        generate_kwargs = generate_kwargs.copy()
        generate_kwargs.setdefault("attention_mask", attention_mask)
        generate_kwargs.setdefault("pad_token_id", self.eos_token_id)
        generate_kwargs.setdefault("eos_token_id", self.eos_token_id)
        generate_kwargs.setdefault("use_cache", True)
        
        # Set generation parameters for better quality if not provided
        generate_kwargs.setdefault("do_sample", True)
        generate_kwargs.setdefault("top_p", 0.92)
        generate_kwargs.setdefault("temperature", 0.85)
        generate_kwargs.setdefault("num_beams", 4)
        
        # Generate report using the embedding as the initial input
        return self.mistral.generate(
            inputs_embeds=input_embedding,
            max_length=max_token_length, 
            **generate_kwargs
        )
