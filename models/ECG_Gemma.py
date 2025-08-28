import torch 
import torch.nn as nn
from typing import Optional, Union, Dict, Any, List, Tuple
from collections.abc import Callable
import logging
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration, Gemma3Model, Gemma3MultiModalProjector, Gemma3ModelOutputWithPast, Gemma3Config, Gemma3RMSNorm, Gemma3CausalLMOutputWithPast
from transformers.configuration_utils import PretrainedConfig
logger = logging.getLogger(__name__)
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper
from transformers.masking_utils import create_causal_mask, create_masks_for_generate, create_sliding_window_causal_mask
from transformers.cache_utils import Cache
from transformers.utils import auto_docstring
from utils.enums import DecoderMode, ModelName
import sys
import os
from torch.nn.utils.rnn import pad_sequence
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, CausalLMOutputWithCrossAttentions
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../transformers_ecg/src')))
#print("sys.path =", sys.path)

class ECG_Gemma3ModelOutputWithPast(Gemma3ModelOutputWithPast):
	
	def __init__(self, *, ecg_hidden_states=None, **kwargs):
		super().__init__(**kwargs)

	ecg_hidden_states: Optional[torch.FloatTensor] = None

class ECGConfig(PretrainedConfig):
	
	model_type = "ecg_model"

	def __init__(self, 
				ecg_token_id = 513,
				pad_token_id = 0,
				embedding_dim = 4,
				num_quantizers = 8,
				codebook_size = 512,
				layer_norm_eps = 1e-5,
				hidden_size = 82,
				vocab_size=30522, 
				eos_token_id = 1,
				**kwargs):
		 
		super().__init__(pad_token_id=pad_token_id, **kwargs)
		self.ecg_token_id = ecg_token_id
		self.embedding_dim = embedding_dim
		self.num_quantizers = num_quantizers
		self.codebook_size = codebook_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.layer_norm_eps = layer_norm_eps
		self.pad_token_id = pad_token_id
		self.eos_token_id = eos_token_id

class ECG_Gemma_config(Gemma3Config):

	 model_type = "ecg_gemma"

	 def __init__(self, ecg_config: Optional[Union[ECGConfig, Dict[str, Any]]] = None, **kwargs):

			if isinstance(ecg_config, dict):
			 ecg_config = ECGConfig(**ecg_config)
			elif ecg_config is None:
				ecg_config = ECGConfig()
				logger.info("ecg_config is None, using default ECGConfig ecg config.")
		 
			self.ecg_config = ecg_config
			#kwargs["ecg_config"] = ecg_config.to_dict()

			super().__init__(**kwargs)

class ECG_Gemma_model(Gemma3Model, GenerationMixin):

	_checkpoint_conversion_mapping = {"language_model.model": "language_model"}

	def __init__(self, config: ECG_Gemma_config, ecg_tokenizer: Optional[ECG_Tokenizer_Wrapper] = None):

		super().__init__(config)			

		self.ecg_tokenizer = ecg_tokenizer or ECG_Tokenizer_Wrapper(
    encoder_name="Residual_Conv_Encoder",        
    quantizer_name="ECG_Tokenizer_Quantizer",     
    decoder_name="Conv_Decoder",                  
    decoder_mode=DecoderMode.RECONSTRUCTION,      
    num_quantizers=8,                             
    codebook_size=512,                           
    num_classes=-1                                
)

		self.ecg_codebook = nn.Embedding(config.ecg_config.codebook_size + 2, config.ecg_config.embedding_dim)
		self.projection = nn.Linear(in_features=4, out_features=82)
		self.ecg_proj = nn.Linear(512, self.config.ecg_config.hidden_size)
		self.lm_head = nn.Linear(2304, config.ecg_config.vocab_size)
		self.language_model = Gemma3Model(config)

	def get_ecg_features(self, ecg_signals: torch.Tensor, chunk_size = 10):
		x = ecg_signals
		#print(f"Input ecg_signals shape: {x.shape}")

		features = self.ecg_tokenizer.encoder(x)	#[T, D1, D2]
		#print(f"Encoded features shape: {features.shape}")

		features_flattened = features.view(features.size(0), -1)  # [T, D]

		T = (features_flattened.size(0) // chunk_size) * chunk_size
		trimmed = features_flattened[:T]
		#print(f"Trimmed features shape: {trimmed.shape}")

		num_chunks = T // chunk_size
		chunks = trimmed.view(num_chunks, chunk_size, -1)
		#print(f"Chunks shape: {chunks.shape}")

		pooled_chunks = chunks.mean(dim=1)
		#print(f"Pooled chunks shape: {pooled_chunks.shape}")
    
    	# Project to match LLM embedding dim
		ecg_proj = self.ecg_proj(pooled_chunks)  # [num_chunks, hidden_dim]
		#print(f"Pooled+Projected features shape: {ecg_proj.shape}")
		
		projected = ecg_proj.unsqueeze(1)  # Add sequence dimension: [250, 1, 82]

		return projected

	def tokenize_ecg(self, projected_chunks: torch.Tensor, return_all_codes: bool = False):
		quantizer_outputs = self.ecg_tokenizer.quantizer(
		projected_chunks,
		return_all_codes=return_all_codes)
		
		_, indices, _ = quantizer_outputs
		indices = indices.long()
		#print(f"indices shape: {indices.shape}")
		
		return indices

	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		labels=None,
		**kwargs):

		with autocast(dtype=torch.bfloat16):

			outputs = self.language_model(
				input_ids=input_ids,
				attention_mask=attention_mask,
				**kwargs)
		
			logits = self.lm_head(outputs.last_hidden_state)

			loss = None
			if labels is not None:
				loss_fct = nn.CrossEntropyLoss()
				loss = loss_fct(
					logits.view(-1, logits.size(-1)), 
					labels.view(-1))

		return CausalLMOutputWithCrossAttentions(
			loss=loss,
			logits=logits,
			hidden_states=outputs.last_hidden_state,
			past_key_values=outputs.past_key_values)

	 
	def forward_vision(self,
		input_ids: torch.LongTensor = None,
		pixel_values: torch.FloatTensor = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
		token_type_ids: Optional[torch.LongTensor] = None,
		cache_position: Optional[torch.LongTensor] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		**lm_kwargs,
	 ) -> Union[Tuple, Gemma3ModelOutputWithPast]:

		return super().forward(
			input_ids=input_ids,
			pixel_values=pixel_values,
			attention_mask=attention_mask,
			position_ids=position_ids,
			past_key_values=past_key_values,
			token_type_ids=token_type_ids,
			cache_position=cache_position,
			inputs_embeds=inputs_embeds,
			labels=labels,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			**lm_kwargs
)
	
	def generate_text(self, tokenizer, inputs_embeds, attention_mask, max_length=20):
		generated_tokens = []
		past_key_values = None

		for step in range(max_length):
			outputs = self.language_model(
				inputs_embeds=inputs_embeds,
				attention_mask=attention_mask,
				past_key_values=past_key_values,
				use_cache=True,
        )
			logits = outputs.last_hidden_state[:, -1, :]  # logits for last token
			next_token = torch.argmax(logits, dim=-1)  # greedy decoding
			token_id = next_token[0].item()

			generated_tokens.append(token_id)

			if token_id == self.config.eos_token_id:
				break

        	# Prepare inputs_embeds for next step
			#get input embeddings inherited from Gemma


			next_token_tensor = torch.tensor([[token_id]], device=attention_mask.device)  # shape (1,1)
			inputs_embeds = self.get_input_embeddings()(next_token_tensor)  # shape (1,1,embedding_dim)



			#the torch.ones is the new column that is being concatenated as the new token 
			attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.size(0), 1), device=attention_mask.device)], dim=1)
			past_key_values = outputs.past_key_values

		generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

		return generated_text

	def forward_ecg(
    	self,
    	input_ids: torch.LongTensor = None,
    	attention_mask: Optional[torch.Tensor] = None,
    	tokenizer=None,
    	past_key_values=None,
    	cache_position=None,
    	token_type_ids=None,
		max_new_tokens=5,
    	**lm_kwargs,) -> Union[Tuple, Gemma3ModelOutputWithPast]:

		input_ids = input_ids.to(self.device)
		if attention_mask is not None:
			attention_mask = attention_mask.to(self.device)

		# Use the model's built-in generate
		generated_ids = self.generate(
			input_ids=input_ids,
			attention_mask=attention_mask,
			max_new_tokens=max_new_tokens,
			do_sample=False
    )

		# Decode generated tokens
		generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
		return generated_text

	def prepare_inputs_for_generation(
		 self,
		 input_ids,
		 past_key_values=None,
		 inputs_embeds=None,
		 cache_position=None,
		 position_ids=None,
		 pixel_values=None,
		 ecg_signals = None,
		 attention_mask=None,
		 token_type_ids=None,
		 use_cache=True,
		 logits_to_keep=None,
		 labels=None,
		 **kwargs,
	 ):
		 # Overwritten -- custom `position_ids` and `pixel_values` handling
		model_inputs = super().prepare_inputs_for_generation(
			 input_ids,
			 past_key_values=past_key_values,
			 inputs_embeds=inputs_embeds,
			 attention_mask=attention_mask,
			 position_ids=position_ids,
			 cache_position=cache_position,
			 use_cache=use_cache,
			 logits_to_keep=logits_to_keep,
			 token_type_ids=token_type_ids,
			 **kwargs,
		 )
		  
		  #if cache_position is not None and cache_position[0] == 0:
		  #    if ecg_signals is not None:
		  #        model_inputs["ecg_signals"] = ecg_signals
		  #    if pixel_values is not None:
		  #        model_inputs["pixel_values"] = pixel_values


		return model_inputs



class ECG_Gemma3MultiModalProjector(Gemma3MultiModalProjector):


	def __init__(self, config: ECG_Gemma_config):
		super().__init__(config)

		self.ecg_input_projection_weight = nn.Parameter(
		torch.zeros(config.ecg_config.hidden_size, config.text_config.hidden_size)
		 )

		self.ecg_soft_emb_norm = Gemma3RMSNorm(
			 config.ecg_config.hidden_size, eps=config.ecg_config.layer_norm_eps
		 )

	def forward_vision(self, vision_outputs: torch.Tensor):
	 
		return super().forward(vision_outputs)
	 
	def forward_ecg(self, ecg_outputs: torch.Tensor):
		batch_size, leads, seq_length = ecg_outputs.shape
		#Transpose to (batch_size, seq_length, leads) so that 'leads' is the feature dimension
		 
		ecg_outputs = ecg_outputs.transpose(1, 2).contiguous()
	
		normed_ecg_outputs = self.ecg_soft_emb_norm(ecg_outputs)

		projected_ecg_outputs = torch.matmul(normed_ecg_outputs, self.ecg_input_projection_weight)

		return projected_ecg_outputs.type_as(ecg_outputs)

class ECG_Gemma3ForConditionalGeneration(Gemma3ForConditionalGeneration):

	_checkpoint_conversion_mapping = {
		 "^language_model.model": "model.language_model",
		 "^vision_tower": "model.vision_tower",
		 "^ecg_codebook": "model.ecg_codebook",
		 "^multi_modal_projector": "model.multi_modal_projector",
		 "^language_model.lm_head": "lm_head",
	 }
	 
	_tied_weights_keys = ["lm_head.weight"]

	def __init__(self, config: ECG_Gemma_config):
		super().__init__(config)
		self.model = ECG_Gemma_model(config)
		self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
		self.post_init()

	def get_ecg_features(self, ecg_signals):
		return self.model.get_ecg_features(ecg_signals)

	@property
	def ecg_tokenizer(self):
		return self.model.ecg_tokenizer

	@auto_docstring
	def forward_vision(
		 self,
		 input_ids: torch.LongTensor = None,
		 pixel_values: torch.FloatTensor = None,
		 attention_mask: Optional[torch.Tensor] = None,
		 position_ids: Optional[torch.LongTensor] = None,
		 past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
		 token_type_ids: Optional[torch.LongTensor] = None,
		 cache_position: Optional[torch.LongTensor] = None,
		 inputs_embeds: Optional[torch.FloatTensor] = None,
		 labels: Optional[torch.LongTensor] = None,
		 use_cache: Optional[bool] = None,
		 output_attentions: Optional[bool] = None,
		 output_hidden_states: Optional[bool] = None,
		 return_dict: Optional[bool] = None,
		 logits_to_keep: Union[int, torch.Tensor] = 0,
		 **lm_kwargs,
	 ) -> Union[Tuple, Gemma3CausalLMOutputWithPast]:

		return super().forward(
			input_ids=input_ids,
			pixel_values=pixel_values,
			attention_mask=attention_mask,
			position_ids=position_ids,
			past_key_values=past_key_values,
			token_type_ids=token_type_ids,
			cache_position=cache_position,
			inputs_embeds=inputs_embeds,
			labels=labels,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			logits_to_keep=logits_to_keep,
			**lm_kwargs,
	 )

	@auto_docstring
	def forward_ecg(
		self,
		input_ids: torch.LongTensor = None,
		ecg_signals: torch.FloatTensor = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
		token_type_ids: Optional[torch.LongTensor] = None,
		cache_position: Optional[torch.LongTensor] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		logits_to_keep: Union[int, torch.Tensor] = 0,
		**lm_kwargs,
	) -> Union[Tuple, Gemma3CausalLMOutputWithPast]:

		"""
    	Forward method for ECG-Gemma model with ECG signal inputs.

    	This method extends the conditional generation capabilities of the Gemma3 model
    	by integrating ECG signal features along with textual inputs.

    	Args:
        input_ids (torch.LongTensor, optional): Indices of input sequence tokens.
        ecg_signals (torch.FloatTensor, optional): ECG signal tensor of shape (batch_size, signal_length).
        attention_mask (torch.Tensor, optional): Mask to avoid attending to padding tokens.
        position_ids (torch.LongTensor, optional): Positional indices of the input tokens.
        past_key_values (List[torch.FloatTensor] or Cache, optional): Cached past key values for fast decoding.
        token_type_ids (torch.LongTensor, optional): Segment token indicators.
        cache_position (torch.LongTensor, optional): Cache position indices.
        inputs_embeds (torch.FloatTensor, optional): Embedded token representations.
        labels (torch.LongTensor, optional): Ground truth for calculating loss.
        use_cache (bool, optional): Whether to use past key values cache.
        output_attentions (bool, optional): Whether to return attention weights.
        output_hidden_states (bool, optional): Whether to return all hidden states.
        return_dict (bool, optional): If True, returns a dictionary instead of a tuple.
        logits_to_keep (int or torch.Tensor, optional): Restrict logits to the specified slice.
        **lm_kwargs: Additional keyword arguments.

		Returns:
			Union[Tuple, ECG_Gemma3CausalLMOutputWithPast]: Model output including logits, loss,
			hidden states, attentions, and ECG-specific outputs.
		"""
    


 
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			 output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		 )
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		outputs = self.model(
			 input_ids=input_ids,
			 ecg_signals = ecg_signals,
			 token_type_ids=token_type_ids,
			 attention_mask=attention_mask,
			 position_ids=position_ids,
			 past_key_values=past_key_values,
			 inputs_embeds=inputs_embeds,
			 use_cache=use_cache,
			 labels=labels,
			 output_attentions=output_attentions,
			 output_hidden_states=output_hidden_states,
			 return_dict=return_dict,
			 cache_position=cache_position,
			 **lm_kwargs,
		 )

		hidden_states = outputs[0]
		 # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
		slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
		logits = self.lm_head(hidden_states[:, slice_indices, :])

		loss = None
		if labels is not None:
			logits = logits.float()
			shift_logits = logits[..., :-1, :]
			shift_labels = labels[..., 1:]
			if attention_mask is not None:
				shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(logits.device)
				shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
				shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
			else:
				shift_logits = shift_logits.contiguous()
				shift_labels = shift_labels.contiguous()
			loss_fct = nn.CrossEntropyLoss()

			flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
			flat_labels = shift_labels.view(-1).to(shift_logits.device)
			loss = loss_fct(flat_logits, flat_labels)

		if not return_dict:
			output = (logits,) + outputs[1:]
			return (loss,) + output if loss is not None else output

		ecg_hidden_states = outputs.ecg_hidden_states

		return ECG_Gemma3CausalLMOutputWithPast(
			 loss=loss,
			 logits=logits,
			 past_key_values=outputs.past_key_values,
			 hidden_states=outputs.hidden_states,
			 attentions=outputs.attentions,
			 image_hidden_states=outputs.image_hidden_states,
			 ecg_hidden_states = ecg_hidden_states
		 )
	
	def prepare_inputs_for_generation(
		 self,
		 input_ids,
		 past_key_values=None,
		 inputs_embeds=None,
		 cache_position=None,
		 position_ids=None,
		 pixel_values=None,
		 ecg_signals = None,
		 attention_mask=None,
		 token_type_ids=None,
		 use_cache=True,
		 logits_to_keep=None,
		 labels=None,
		 **kwargs,
	 ):
		 # Overwritten -- custom `position_ids` and `pixel_values` handling
		model_inputs = super().prepare_inputs_for_generation(
			 input_ids,
			 past_key_values=past_key_values,
			 inputs_embeds=inputs_embeds,
			 attention_mask=attention_mask,
			 position_ids=position_ids,
			 cache_position=cache_position,
			 use_cache=use_cache,
			 logits_to_keep=logits_to_keep,
			 token_type_ids=token_type_ids,
			 **kwargs,
		 )
		  
		  #if cache_position is not None and cache_position[0] == 0:
		  #    if ecg_signals is not None:
		  #        model_inputs["ecg_signals"] = ecg_signals
		  #    if pixel_values is not None:
		  #        model_inputs["pixel_values"] = pixel_values


		return model_inputs

	forward = forward_ecg

def token_type_ids_mask_function(token_type_ids: Optional[torch.Tensor], tokens_per_image: int) -> Optional[Callable]:
	
	 # Do not return an additional mask in this case
	if token_type_ids is None:
		return None

	def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
		# If the difference is less than image size, both are part of the same image block
		same_image_block = torch.abs(kv_idx - q_idx) <= tokens_per_image
		 # If it's 1 for both query and key/value, we are in an image block
		is_image_block = (token_type_ids[batch_idx, q_idx] == 1) & (token_type_ids[batch_idx, kv_idx] == 1)

		 # This is bidirectional attention whenever we are dealing with image tokens
		return is_image_block & same_image_block

	return inner_mask
