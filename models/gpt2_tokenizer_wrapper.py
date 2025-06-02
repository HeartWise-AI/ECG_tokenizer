import torch
import torch.nn as nn

from typing import Union, Optional, Dict, Any, Tuple
from transformers.generation.utils import GenerateOutput
from transformers import GPT2LMHeadModel, PreTrainedModel


from models.adapters import (
    LinearReducer, 
    EmbeddingReducer, 
    SimpleEmbeddingReducer
)
from models.tokenizer import ECG_Tokenizer_Wrapper
from utils.registry import ModelRegistry
from utils.config.tokenizer_config import ECGTokenizerTrainingConfig


torch.serialization.add_safe_globals([ECGTokenizerTrainingConfig])


@ModelRegistry.register("GPT2_Tokenizer_Wrapper")
class GPT2TokenizerWrapper(nn.Module):
    def __init__(
        self, 
        ecg_tokenizer_path: str,
        ecg_tokenizer_num_quantizers: int = 12,
        ecg_tokenizer_codebook_size: int = 1024,
        ecg_encoder_name: str = "Conv_Encoder",
        ecg_quantizer_name: str = "ECG_Tokenizer_Quantizer", 
        ecg_decoder_name: str = "Conv_Decoder",
        gpt2_model_name: str = 'gpt2', 
        gpt2_embedding_size: int = 768, 
        ecg_embedding_size: Tuple[int, int, int] = (12, 128, 82),
        reducer_name: str = "GPT2_EmbeddingReducer",
        reducer_dropout: float = 0.2,
        label_ignore_index: int = -100,
        # Default generation parameters
        default_do_sample: bool = True,
        default_top_p: float = 0.92,
        default_temperature: float = 0.85,
        default_num_beams: int = 4,
    ):
        super(GPT2TokenizerWrapper, self).__init__()
        
        # Load the ECG tokenizer
        self.ecg_tokenizer = ECG_Tokenizer_Wrapper(
            encoder_name=ecg_encoder_name,
            quantizer_name=ecg_quantizer_name,
            decoder_name=ecg_decoder_name,
            num_quantizers=ecg_tokenizer_num_quantizers,
            codebook_size=ecg_tokenizer_codebook_size,
        )      
        try:
            ecg_tokenizer_state_dict = torch.load(ecg_tokenizer_path, weights_only=True)['model_state_dict']
        except FileNotFoundError:
            raise FileNotFoundError(f"ECG model not found at {ecg_tokenizer_path}")
        self.ecg_tokenizer.load_state_dict(ecg_tokenizer_state_dict)
        
        # Freeze ECG tokenizer parameters
        for param in self.ecg_tokenizer.parameters():
            param.requires_grad = False
        # Set ECG tokenizer to evaluation mode to ensure it stays frozen
        self.ecg_tokenizer.eval()
            
        # Add this line to convert ECG tokenizer to bfloat16
        self.ecg_tokenizer.encoder.to(dtype=torch.float32)
        self.ecg_tokenizer.quantizer.to(dtype=torch.float32)
        self.ecg_tokenizer.quantizer.quantizer.to(dtype=torch.float32)
        self.ecg_tokenizer.decoder.to(dtype=torch.float32)
        
        # Store configuration for consistent use
        self.label_ignore_index = label_ignore_index
        self.default_generation_params = {
            "do_sample": default_do_sample,
            "top_p": default_top_p,
            "temperature": default_temperature,
            "num_beams": default_num_beams,
        }
        
        # Load the embedding adapter class
        self.embedding_adapter_class: Union[
            EmbeddingReducer, 
            LinearReducer, 
            SimpleEmbeddingReducer
        ] = ModelRegistry.get(reducer_name)
        if self.embedding_adapter_class is None:
            raise ValueError(f"Reducer {reducer_name} not found in ModelRegistry")       
        # Load the embedding adapter
        self.embedding_adapter = self.embedding_adapter_class(
            input_shape=ecg_embedding_size,
            output_size=gpt2_embedding_size, 
            dropout=reducer_dropout
        )
        
        # Load the GPT-2 model
        self.gpt2: PreTrainedModel = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        # Check if embedding size matches GPT-2's hidden size.
        if gpt2_embedding_size != self.gpt2.config.n_embd:
            raise ValueError(f"Embedding size {gpt2_embedding_size} does not match GPT-2 hidden size {self.gpt2.config.n_embd}")
        # Optional: Maintain the special token ID (if needed elsewhere).
        # We still resize token embeddings for compatibility during generation.
        self.gpt2.resize_token_embeddings(len(self.gpt2.get_input_embeddings().weight) + 1)
        self.ecg_token_id = len(self.gpt2.get_input_embeddings().weight) - 1  # New token ID
        # Make sure EOS token is defined
        self.eos_token_id = self.gpt2.config.eos_token_id

    def forward(
        self, 
        ecg_signal: torch.Tensor, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None, 
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Forward pass for the GPT2WithEmbedding model.
        
        Args:
            ecg_signal: Tensor containing ECG signal (batch, *ecg_dims)
            input_ids: Token IDs for the text input (batch, seq_length)
            attention_mask: Optional mask for padding tokens (batch, seq_length)
            labels: Optional labels for computing the language modeling loss (batch, seq_length)
            
        Returns:
            Dictionary containing loss, logits, and other outputs from the GPT-2 model
            
        Note:
            - During training, ensure your target sequences end with an EOS token for better generation
            - The model prepends a special ECG token to the input sequence
        """
        # Reduce ECG embeddings
        ecg_signal = ecg_signal.to(dtype=torch.float32)
        _, _, _, ecg_embeddings = self.ecg_tokenizer(ecg_signal, return_all_codes=True)
        ecg_embeddings = ecg_embeddings.permute(1, 0, 2, 3)
        reduced: torch.Tensor = self.embedding_adapter(ecg_embeddings)  # (batch, embedding_size)
        
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
                self.label_ignore_index,
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
        input_embedding = self.gpt2.get_input_embeddings()(input_ids)  # (batch, seq_length + 1, hidden_size)
        input_embedding[:, 0, :] = reduced  # Replace ECG token embedding
        
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
        ecg_signal: torch.Tensor,
        max_token_length: int = 512, 
        **generate_kwargs
    ) -> Union[GenerateOutput, torch.Tensor]:
        """
        Generate a clinical report conditioned solely on the ECG embeddings.
        This method uses GPT-2's generate() function with inputs_embeds.
        
        Args:
            ecg_signal: Tensor containing ECG signal
            max_token_length: Maximum length of generated tokens
            **generate_kwargs: Additional keyword arguments for generation
            
        Returns:
            Tensor containing generated token IDs for reports
            
        Note:
            This method ensures generation stops properly by using the EOS token
        """
        # Process the raw signal first
        _, _, _, ecg_embeddings = self.ecg_tokenizer(ecg_signal, return_all_codes=True)
        ecg_embeddings = ecg_embeddings.permute(1, 0, 2, 3)
        reduced = self.embedding_adapter(ecg_embeddings)
        
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
        input_embedding = self.gpt2.get_input_embeddings()(ecg_token)  # (batch, 1, hidden_size)
        input_embedding[:, 0, :] = reduced  # Replace ECG token embedding
        
        # Set default generation parameters
        generate_kwargs = generate_kwargs.copy()
        generate_kwargs.setdefault("attention_mask", attention_mask)
        generate_kwargs.setdefault("pad_token_id", self.eos_token_id)
        generate_kwargs.setdefault("eos_token_id", self.eos_token_id)
        generate_kwargs.setdefault("use_cache", True)
        
        # Set generation parameters for better quality if not provided
        generate_kwargs.setdefault("do_sample", self.default_generation_params["do_sample"])
        generate_kwargs.setdefault("top_p", self.default_generation_params["top_p"])
        generate_kwargs.setdefault("temperature", self.default_generation_params["temperature"])
        generate_kwargs.setdefault("num_beams", self.default_generation_params["num_beams"])
        
        # Generate report using the embedding as the initial input
        return self.gpt2.generate(
            inputs_embeds=input_embedding,
            max_length=max_token_length, 
            **generate_kwargs
        )

    def train(self, mode: bool = True):
        """Override train method to keep ECG tokenizer in eval mode while allowing other components to train."""
        super().train(mode)
        # Only need to ensure ECG tokenizer stays in eval mode
        # (requires_grad is already set to False during initialization)
        self.ecg_tokenizer.eval()
        return self