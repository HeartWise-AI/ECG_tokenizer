import torch
import torch.nn as nn

from typing import Union, Optional, Dict, Any, Tuple
from transformers.generation.utils import GenerateOutput
from transformers import GPT2LMHeadModel, PreTrainedModel

from utils.enums import (
    ModelName,
    BridgeName
)
from utils.registry import ModelRegistry
from models.types import ModelT, ModelClassT


@ModelRegistry.register(ModelName.GPT2_DECODER)
class GPT2Decoder(nn.Module):
    """
    GPT-2 decoder that generates clinical reports from quantized ECG features.
    
    Transforms ECG quantized features into GPT-2's embedding space via an adapter,
    then generates text reports using either teacher forcing or ECG-only training.
    Now supports question-answering about ECG data.
    """
    def __init__(
        self, 
        huggingface_model_name: str = 'gpt2', 
        llm_input_embedding_size: int = 768, 
        quantized_feature_shape: Tuple[int, int] = (128, 82),
        bridge_name: BridgeName = BridgeName.GPT2_SEQUENCE_BRIDGE,
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
            huggingface_model_name: Pre-trained GPT-2 model name from HuggingFace.
            llm_input_embedding_size: GPT-2 embedding dimension (must match model).
            quantized_feature_shape: Shape of quantized ECG features (seq_len, features).
            bridge_name: Name of adapter to transform ECG features to GPT-2 space.
            adapter_dropout: Dropout rate for the adapter.
            label_ignore_index: Index to ignore in loss computation.
            default_do_sample: Default sampling strategy for generation.
            default_top_p: Default nucleus sampling parameter.
            default_temperature: Default temperature for generation.
            default_num_beams: Default number of beams for beam search.
        """
        super(GPT2Decoder, self).__init__()
        
        # Store configuration
        self.huggingface_model_name = huggingface_model_name
        self.llm_input_embedding_size = llm_input_embedding_size
        self.quantized_feature_shape = quantized_feature_shape
        self.adapter_dropout = adapter_dropout
        self.label_ignore_index = label_ignore_index
        self.default_generation_params = {
            "do_sample": default_do_sample,
            "top_p": default_top_p,
            "temperature": default_temperature,
            "num_beams": default_num_beams,
        }
        
        # Load the adapter class
        self.adapter_class: ModelClassT = ModelRegistry.get(bridge_name)
        if self.adapter_class is None:
            raise ValueError(f"Bridge {bridge_name} not found in ModelRegistry")
        
        self.bridge_name = bridge_name
        
        # Initialize the adapter to transform quantized features to GPT-2 embedding space
        # Input shape: (batch, channels, sequence_length)
        self.adapter: ModelT = self.adapter_class(
            input_shape=quantized_feature_shape,
            output_size=llm_input_embedding_size, 
            dropout=adapter_dropout
        )
        # Backwards-compatible attribute used by tests and legacy callers
        self.embedding_adapter = self.adapter
        
        # Load the GPT-2 model
        self.llm_model: PreTrainedModel = GPT2LMHeadModel.from_pretrained(huggingface_model_name)
        
        # Check if embedding size matches GPT-2's hidden size
        if llm_input_embedding_size != self.llm_model.config.n_embd:
            raise ValueError(f"Embedding size {llm_input_embedding_size} does not match GPT-2 hidden size {self.llm_model.config.n_embd}")
        
        # Add special ECG token in llm embedding
        self.llm_model.resize_token_embeddings(len(self.llm_model.get_input_embeddings().weight) + 1)
        self.ecg_token_id = len(self.llm_model.get_input_embeddings().weight) - 1
        self.eos_token_id = self.llm_model.config.eos_token_id
        
    def forward(
        self, 
        quantized_features: torch.Tensor,
        input_ids: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        question_ids: Optional[torch.Tensor] = None,
        question_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass for training or generation setup.
        
        Args:
            quantized_features: ECG features from tokenizer (batch, seq_len, features).
            input_ids: Text token IDs (batch, seq_len). Required for training.
            attention_mask: Attention mask for padding tokens (batch, seq_len).
            labels: Optional target labels for loss computation (batch, seq_len).
            question_ids: Question token IDs (batch, question_seq_len). Optional.
            question_attention_mask: Attention mask for question padding (batch, question_seq_len).
            
        Returns:
            Dictionary with loss, logits, and other GPT-2 outputs.
            
        Raises:
            ValueError: If input_ids is None during training.
        """
        # Transform quantized features to GPT-2 embedding space
        quantized_features = quantized_features.to(dtype=torch.float32)
        ecg_embedding = self.adapter(quantized_features)  # (batch, embedding_size)       

        if question_ids is not None:
            return self._forward_with_questions(
                ecg_embedding, input_ids, labels, question_ids, 
                attention_mask, question_attention_mask
            )
        else:
            return self._forward_teacher_forcing(ecg_embedding, input_ids, labels, attention_mask)

    def _forward_with_questions(
        self,
        ecg_embedding: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor],
        question_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        question_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass with questions about ECG data.
        
        Sequence structure: [ECG_TOKEN] [QUESTION_START] question tokens [QUESTION_END] response tokens
        
        Args:
            ecg_embedding: Processed ECG features (batch, embedding_dim).
            input_ids: Response token IDs (batch, seq_len).
            attention_mask: Attention mask for response padding (batch, seq_len).
            labels: Target labels for loss computation (batch, seq_len).
            question_ids: Question token IDs (batch, question_seq_len).
            question_attention_mask: Attention mask for question padding (batch, question_seq_len).
            
        Returns:
            GPT-2 model outputs with loss and logits.
        """
        batch_size = input_ids.size(0)
        
        # Create ECG token
        ecg_token = torch.full(
            (batch_size, 1),
            self.ecg_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device
        )
        
        # Create question start and end tokens
        question_start_token = torch.full(
            (batch_size, 1),
            self.question_start_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device
        )
        question_end_token = torch.full(
            (batch_size, 1),
            self.question_end_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device
        )
        
        # Concatenate: [ECG] [Q_START] question [Q_END] response
        full_input_ids = torch.cat([
            ecg_token,
            question_start_token,
            question_ids,
            question_end_token,
            input_ids
        ], dim=1)
        
        # Create full attention mask
        ecg_mask = torch.ones((batch_size, 1), device=input_ids.device, dtype=torch.long)
        question_start_mask = torch.ones((batch_size, 1), device=input_ids.device, dtype=torch.long)
        question_end_mask = torch.ones((batch_size, 1), device=input_ids.device, dtype=torch.long)
        
        # Handle optional masks
        if question_attention_mask is None:
            question_attention_mask = torch.ones_like(question_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        full_attention_mask = torch.cat([
            ecg_mask,
            question_start_mask,
            question_attention_mask,
            question_end_mask,
            attention_mask
        ], dim=1)
        
        # Create full labels (ignore loss for ECG, question tokens, and question delimiters)
        if labels is not None:
            ecg_label = torch.full((batch_size, 1), self.label_ignore_index, dtype=labels.dtype, device=labels.device)
            question_start_label = torch.full((batch_size, 1), self.label_ignore_index, dtype=labels.dtype, device=labels.device)
            question_labels = torch.full(question_ids.shape, self.label_ignore_index, dtype=labels.dtype, device=labels.device)
            question_end_label = torch.full((batch_size, 1), self.label_ignore_index, dtype=labels.dtype, device=labels.device)
            
            full_labels = torch.cat([
                ecg_label,
                question_start_label,
                question_labels,
                question_end_label,
                labels
            ], dim=1)
        else:
            full_labels = None
        
        # Get input embeddings and replace the ECG token's embedding
        input_embedding = self.gpt2.get_input_embeddings()(full_input_ids)
        input_embedding[:, 0, :] = ecg_embedding  # Replace ECG token embedding
        
        # Forward pass through GPT-2
        outputs = self.gpt2(
            inputs_embeds=input_embedding,
            attention_mask=full_attention_mask,
            labels=full_labels
        )
        return outputs

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
        input_embedding = self.llm_model.get_input_embeddings()(input_ids)
        input_embedding[:, 0, :] = ecg_embedding
        
        # Forward pass through GPT-2
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
        max_token_length: int = 512,
        **generate_kwargs
    ) -> torch.Tensor:
        """
        Generate clinical report from quantized ECG features via manual autoregressive loop.

        Workaround for transformers>=4.45 bug where `GPT2LMHeadModel.generate(inputs_embeds=...)`
        re-feeds the original inputs_embeds at every step instead of the newly generated token.

        Args:
            quantized_features: ECG features from tokenizer (batch, seq_len, features).
            max_token_length: Maximum total tokens to generate (not counting the seed position).
            **generate_kwargs: `do_sample`, `temperature`, `top_p`, `top_k`, `eos_token_id`.

        Returns:
            Generated token IDs (batch, generated_length).
        """
        if len(quantized_features.shape) == 3:
            adapter_input = quantized_features.unsqueeze(1)
        else:
            adapter_input = quantized_features

        ecg_embedding: torch.Tensor = self.adapter(adapter_input)
        batch_size: int = ecg_embedding.size(0)
        device = ecg_embedding.device

        params = {**self.default_generation_params, **generate_kwargs}
        do_sample: bool = bool(params.get("do_sample", False))
        temperature: float = float(params.get("temperature", 1.0))
        top_p: float = float(params.get("top_p", 1.0))
        top_k: int = int(params.get("top_k", 0))
        eos_token_id: int = int(params.get("eos_token_id", self.eos_token_id))

        wte = self.llm_model.get_input_embeddings()
        ecg_token = torch.full((batch_size, 1), self.ecg_token_id, dtype=torch.long, device=device)
        input_embedding = wte(ecg_token)
        input_embedding[:, 0, :] = ecg_embedding

        outputs = self.llm_model(inputs_embeds=input_embedding, use_cache=True)
        past_kv = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        next_tokens = self._sample_next(logits, do_sample, temperature, top_p, top_k)

        generated = [next_tokens]
        finished = next_tokens.eq(eos_token_id)

        for _ in range(max_token_length - 1):
            if finished.all():
                break
            out = self.llm_model(input_ids=next_tokens.unsqueeze(-1), past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            logits = out.logits[:, -1, :]
            sampled = self._sample_next(logits, do_sample, temperature, top_p, top_k)
            next_tokens = torch.where(finished, torch.full_like(sampled, eos_token_id), sampled)
            finished = finished | next_tokens.eq(eos_token_id)
            generated.append(next_tokens)

        return torch.stack(generated, dim=1)

    @staticmethod
    def _sample_next(
        logits: torch.Tensor,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> torch.Tensor:
        if not do_sample:
            return logits.argmax(dim=-1)
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-8)
        if top_k > 0:
            topk_vals, _ = logits.topk(top_k, dim=-1)
            logits = torch.where(logits < topk_vals[:, [-1]], torch.full_like(logits, float("-inf")), logits)
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
            cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            mask = cum_probs > top_p
            mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
        probs = logits.softmax(dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.no_grad()
    def answer_question(
        self,
        quantized_features: torch.Tensor,
        question_ids: torch.Tensor,
        question_attention_mask: Optional[torch.Tensor] = None,
        max_token_length: int = 512,
        **generate_kwargs
    ) -> Union[GenerateOutput, torch.Tensor]:
        """
        Answer a question about the ECG data.
        
        Args:
            quantized_features: ECG features from tokenizer (batch, seq_len, features).
            question_ids: Question token IDs (batch, question_seq_len).
            question_attention_mask: Attention mask for question padding (batch, question_seq_len).
            max_token_length: Maximum number of tokens to generate.
            **generate_kwargs: Additional parameters for GPT-2 generation.
            
        Returns:
            Generated answer token IDs (batch, generated_length).
            
        Example:
            >>> features = tokenizer.encode(ecg_signal)  # (1, 128, 82)
            >>> question = tokenizer.encode("What is the heart rate?")  # (1, question_len)
            >>> answer_tokens = decoder.answer_question(features, question, max_token_length=50)
            >>> answer = tokenizer.decode(answer_tokens[0])
        """
        # Transform features to embedding space
        if len(quantized_features.shape) == 3:
            adapter_input = quantized_features.unsqueeze(1)
        else:
            adapter_input = quantized_features
            
        ecg_embedding: torch.Tensor = self.adapter(adapter_input)
        batch_size = ecg_embedding.size(0)
        
        # Create special tokens
        ecg_token = torch.full(
            (batch_size, 1),
            self.ecg_token_id,
            dtype=torch.long,
            device=ecg_embedding.device
        )
        question_start_token = torch.full(
            (batch_size, 1),
            self.question_start_token_id,
            dtype=torch.long,
            device=ecg_embedding.device
        )
        question_end_token = torch.full(
            (batch_size, 1),
            self.question_end_token_id,
            dtype=torch.long,
            device=ecg_embedding.device
        )
        
        # Build input sequence: [ECG] [Q_START] question [Q_END]
        input_ids = torch.cat([
            ecg_token,
            question_start_token,
            question_ids,
            question_end_token
        ], dim=1)
        
        # Create attention mask
        ecg_mask = torch.ones((batch_size, 1), device=ecg_embedding.device, dtype=torch.long)
        question_start_mask = torch.ones((batch_size, 1), device=ecg_embedding.device, dtype=torch.long)
        question_end_mask = torch.ones((batch_size, 1), device=ecg_embedding.device, dtype=torch.long)
        
        if question_attention_mask is None:
            question_attention_mask = torch.ones_like(question_ids)
            
        attention_mask = torch.cat([
            ecg_mask,
            question_start_mask,
            question_attention_mask,
            question_end_mask
        ], dim=1)
        
        # Get input embeddings and replace ECG token embedding
        input_embedding = self.gpt2.get_input_embeddings()(input_ids)
        input_embedding[:, 0, :] = ecg_embedding  # Replace ECG token embedding
        
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
            result = self.gpt2.generate(
                inputs_embeds=input_embedding,
                max_length=max_token_length,
                **generation_params
            )
        
        return result
