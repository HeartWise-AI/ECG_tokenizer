import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

from utils.registry import ModelRegistry
from models.embedding_reducer import EmbeddingReducer
from models.linear_reducer import LinearReducer
from models.simple_embedding_reducer import SimpleEmbeddingReducer
from typing import Union

@ModelRegistry.register("GPT2_WithEmbedding")
class GPT2WithEmbedding(nn.Module):
    def __init__(
        self, 
        gpt2_model_name: str = 'gpt2', 
        embedding_size: int = 768, 
        reducer_name: str = None,
        reducer_dropout: float = 0.2
    ):
        super(GPT2WithEmbedding, self).__init__()
        self.gpt2: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self.embedding_reducer: Union[
            EmbeddingReducer, 
            LinearReducer, 
            SimpleEmbeddingReducer
        ] = ModelRegistry.get(reducer_name)(
            output_size=embedding_size
        )
        
        # If embedding size differs from GPT-2's hidden size, project it - 
        # this is done to ensure that the embedding size is the same as the hidden size of the GPT-2 model
        if embedding_size != self.gpt2.config.n_embd:
            self.proj: nn.Linear = nn.Linear(embedding_size, self.gpt2.config.n_embd)
        else:
            self.proj: nn.Linear = None
        
        # Optional: Add a special token to represent ECG embedding
        self.gpt2.resize_token_embeddings(len(self.gpt2.get_input_embeddings().weight) + 1)
        self.ecg_token_id = len(self.gpt2.get_input_embeddings().weight) - 1  # New token ID

    def forward(
        self, 
        ecg_embeddings: torch.Tensor, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor = None, 
        labels: torch.Tensor = None
    ):
        # Reduce ECG embeddings
        reduced: torch.Tensor = self.embedding_reducer(ecg_embeddings)  # (batch, 768)
        if self.proj:
            reduced: torch.Tensor = self.proj(reduced)  # (batch, hidden_size)
        
        # Expand the reduced embedding to match the sequence length
        batch_size: int = reduced.size(0)
        seq_length: int = input_ids.size(1)
        condition: torch.Tensor = reduced.unsqueeze(1).repeat(1, 1, 1)  # (batch, 1, hidden_size)
        
        # Prepend a special ECG token to the input_ids
        ecg_token: torch.Tensor = torch.tensor([self.ecg_token_id] * batch_size).unsqueeze(1).to(input_ids.device)  # (batch, 1)
        input_ids: torch.Tensor = torch.cat([ecg_token, input_ids], dim=1)  # (batch, seq_length + 1)
        
        if attention_mask is not None:
            ecg_mask: torch.Tensor = torch.ones((batch_size, 1)).to(attention_mask.device)
            attention_mask: torch.Tensor = torch.cat([ecg_mask, attention_mask], dim=1)  # (batch, seq_length + 1)
        
        if labels is not None:
            # Add -100 as label for the ECG token position (will be ignored in loss calculation)
            label_ignore: torch.Tensor = torch.full((batch_size, 1), -100, dtype=labels.dtype).to(labels.device)
            labels: torch.Tensor = torch.cat([label_ignore, labels], dim=1)  # (batch, seq_length + 1)
        
        # Forward through GPT-2
        outputs: torch.Tensor = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    
    def generate_report(
        self, 
        ecg_embeddings: torch.Tensor, 
        max_token_length: int = 512, 
        **generate_kwargs
    ):
        """
        Generate a clinical report conditioned solely on the ECG embeddings.
        This method uses GPT-2's generate() function with inputs_embeds.
        """
        # Reduce the ECG embeddings to a vector of GPT-2 hidden size.
        reduced = self.embedding_reducer(ecg_embeddings)  # (batch, 768)
        if self.proj:
            reduced = self.proj(reduced)  # (batch, hidden_size)
        
        # Create an initial prefix embedding - here we simply use the reduced vector as the first token embedding.
        prefix = reduced.unsqueeze(1)  # (batch, 1, hidden_size)
        
        # Ensure that attention_mask and pad_token_id are provided for reliable generation.
        # This is done to avoid warnings when calling .generate() function.
        gen_kwargs = generate_kwargs.copy()
        if "attention_mask" not in gen_kwargs:
            # Create an attention mask of ones for the prefix tokens.
            gen_kwargs["attention_mask"] = prefix.new_ones(prefix.shape[:-1])
        if "pad_token_id" not in gen_kwargs:
            # Explicitly set pad_token_id to eos_token_id to avoid warnings.
            gen_kwargs["pad_token_id"] = self.gpt2.config.eos_token_id
        
        # Now call GPT-2's generate using inputs_embeds instead of input_ids.
        generated_ids = self.gpt2.generate(
            inputs_embeds=prefix,
            max_length=max_token_length, 
            **gen_kwargs
        )
        return generated_ids    