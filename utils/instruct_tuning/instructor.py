#!/usr/bin/env python3
"""
Instruction-tuning trainer for ECG LLM models using Hugging Face SFTTrainer.

This module provides instruction-tuning capabilities for any LLM decoder
in the ECG tokenizer system, making it fully model-agnostic.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset, DatasetDict
import wandb

from utils.registry import ModelRegistry
from utils.enums import ModelName
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper
from utils.instruct_tuning.data_formatter import ECGInstructDataFormatter


@dataclass
class InstructTuningConfig:
    """Configuration for instruction tuning."""
    
    # Model configuration
    huggingface_model_name: str = "gpt2"
    llm_input_embedding_size: int = 768
    decoder_name: str = "GPT2_Decoder"
    adapter_name: str = "GPT2_LinearAdapter"
    
    # ECG tokenizer configuration
    encoder_name: str = "Conv_Encoder"
    quantizer_name: str = "ECG_Tokenizer_Quantizer"
    num_quantizers: int = 4
    codebook_size: int = 1024
    pretrained_tokenizer_path: Optional[str] = None
    
    # Instruction tuning specific
    template_style: str = "alpaca"
    max_seq_length: int = 512
    include_metadata: bool = True
    enhanced_prompt: bool = True
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    
    # Advanced training settings
    fp16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    
    # LoRA settings (for efficient fine-tuning)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None
    
    # Output and logging
    output_dir: str = "./instruct_tuning_output"
    logging_dir: str = "./instruct_tuning_logs"
    run_name: str = "ecg_instruct_tuning"
    
    # WandB configuration
    use_wandb: bool = True
    wandb_project: str = "ecg_instruct_tuning"
    wandb_entity: Optional[str] = None
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001


class ECGInstructTuner:
    """
    Instruction tuner for ECG LLM models.
    
    Supports any LLM decoder in the ECG tokenizer system with instruction tuning
    using Hugging Face's SFTTrainer (Supervised Fine-Tuning Trainer).
    """
    
    def __init__(self, config: InstructTuningConfig):
        """
        Initialize the instruction tuner.
        
        Args:
            config: Instruction tuning configuration
        """
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.data_formatter = ECGInstructDataFormatter(config.template_style)
        
        # Initialize WandB if requested
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.run_name,
                config=config.__dict__
            )
    
    def setup_tokenizer(self) -> AutoTokenizer:
        """Setup the tokenizer for the LLM."""
        print(f"Loading tokenizer: {self.config.huggingface_model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.huggingface_model_name)
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Add special tokens if needed
        special_tokens = {"additional_special_tokens": ["<ECG>", "</ECG>"]}\n        tokenizer.add_special_tokens(special_tokens)
        
        self.tokenizer = tokenizer
        return tokenizer
    
    def setup_model(self) -> ECG_Tokenizer_Wrapper:
        """Setup the ECG tokenizer wrapper with LLM decoder."""
        print(f"Setting up ECG tokenizer with {self.config.decoder_name}")
        
        # Create the ECG tokenizer wrapper
        model = ModelRegistry.get(ModelName.ECG_TOKENIZER_WRAPPER)(
            encoder_name=self.config.encoder_name,
            quantizer_name=self.config.quantizer_name,
            decoder_name=self.config.decoder_name,
            num_quantizers=self.config.num_quantizers,
            codebook_size=self.config.codebook_size,
            decoder_mode="llm",
            huggingface_model_name=self.config.huggingface_model_name,
            llm_input_embedding_size=self.config.llm_input_embedding_size,
            adapter_name=self.config.adapter_name
        )
        
        # Load pretrained weights if specified
        if self.config.pretrained_tokenizer_path:
            print(f"Loading pretrained weights from: {self.config.pretrained_tokenizer_path}")
            checkpoint = torch.load(self.config.pretrained_tokenizer_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            # Load with frozen encoder/quantizer, trainable decoder
            model._load_pretrained_weights(state_dict, freeze_pretrained_components=True)
        
        # Resize token embeddings if tokenizer was modified
        if hasattr(model.decoder, 'llm_model'):
            original_vocab_size = model.decoder.llm_model.get_input_embeddings().num_embeddings
            new_vocab_size = len(self.tokenizer)
            
            if new_vocab_size > original_vocab_size:
                print(f"Resizing token embeddings: {original_vocab_size} -> {new_vocab_size}")
                model.decoder.llm_model.resize_token_embeddings(new_vocab_size)
        
        self.model = model
        return model
    
    def setup_lora(self):
        """Setup LoRA (Low-Rank Adaptation) for efficient fine-tuning."""
        if not self.config.use_lora:
            return
            
        try:
            from peft import LoraConfig, TaskType, get_peft_model
            
            # Determine target modules based on model type
            if self.config.lora_target_modules is None:
                # Auto-detect based on model architecture
                if "gpt" in self.config.huggingface_model_name.lower():
                    target_modules = ["c_attn", "c_proj", "c_fc"]
                elif "llama" in self.config.huggingface_model_name.lower():
                    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                else:
                    target_modules = ["query", "value", "key", "dense"]
            else:
                target_modules = self.config.lora_target_modules
            
            print(f"Setting up LoRA with target modules: {target_modules}")
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=target_modules,
            )
            
            # Apply LoRA to the LLM decoder
            if hasattr(self.model.decoder, 'llm_model'):
                self.model.decoder.llm_model = get_peft_model(
                    self.model.decoder.llm_model, lora_config
                )
                print("LoRA applied successfully")
            else:
                print("Warning: Could not apply LoRA - llm_model not found")
                
        except ImportError:
            print("Warning: PEFT not installed. Install with: pip install peft")
            print("Continuing without LoRA...")
    
    def prepare_dataset(
        self, 
        dataset: Union[Dataset, DatasetDict, str],
        is_formatted: bool = False
    ) -> DatasetDict:
        """
        Prepare dataset for instruction tuning.
        
        Args:
            dataset: Input dataset (can be path, Dataset, or DatasetDict)
            is_formatted: Whether the dataset is already formatted for instruction tuning
            
        Returns:
            Prepared DatasetDict ready for training
        """
        print("Preparing dataset for instruction tuning...")
        
        # Load dataset if it's a path
        if isinstance(dataset, str):
            from datasets import load_dataset
            if dataset.endswith('.json'):
                dataset = load_dataset('json', data_files=dataset)
            else:
                dataset = load_dataset(dataset)
        
        # Format dataset if not already formatted
        if not is_formatted:
            dataset = self.data_formatter.format_dataset_dict(
                dataset,
                include_metadata=self.config.include_metadata,
                enhanced_prompt=self.config.enhanced_prompt
            )
        
        # Tokenize the dataset
        def tokenize_function(examples):
            # Tokenize the formatted text
            tokens = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.config.max_seq_length,
                return_tensors=None,
            )
            
            # For instruction tuning, we want to predict the response part only
            # Find the response start token
            response_key = self.data_formatter.templates[self.config.template_style]["response_key"]
            
            labels = []
            for i, input_ids in enumerate(tokens["input_ids"]):
                # Find where the response starts
                text = examples["text"][i]
                response_start = text.find(response_key)
                
                if response_start != -1:
                    # Tokenize up to response start to find the cutoff
                    prefix_tokens = self.tokenizer(
                        text[:response_start + len(response_key)],
                        truncation=True,
                        padding=False,
                        max_length=self.config.max_seq_length,
                        return_tensors=None,
                    )["input_ids"]
                    
                    # Create labels (ignore tokens before response)
                    label_ids = input_ids.copy()
                    for j in range(min(len(prefix_tokens), len(label_ids))):
                        label_ids[j] = -100  # Ignore in loss computation
                    
                    labels.append(label_ids)
                else:
                    # Fallback: ignore first half of tokens
                    label_ids = input_ids.copy()
                    cutoff = len(label_ids) // 2
                    for j in range(cutoff):
                        label_ids[j] = -100
                    labels.append(label_ids)
            
            tokens["labels"] = labels
            return tokens
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train" if "train" in dataset else list(dataset.keys())[0]].column_names,
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset
    
    def create_trainer(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None) -> SFTTrainer:
        """Create the SFTTrainer for instruction tuning."""
        print("Creating SFTTrainer...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=self.config.eval_steps if eval_dataset else None,
            save_steps=self.config.save_steps,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            fp16=self.config.fp16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            dataloader_num_workers=self.config.dataloader_num_workers,
            remove_unused_columns=self.config.remove_unused_columns,
            report_to="wandb" if self.config.use_wandb else None,
            run_name=self.config.run_name,
            logging_dir=self.config.logging_dir,
        )
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
            pad_to_multiple_of=8 if self.config.fp16 else None,
        )
        
        # Callbacks
        callbacks = []
        if self.config.use_early_stopping and eval_dataset:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold
                )
            )
        
        # Create trainer
        trainer = SFTTrainer(
            model=self.model.decoder.llm_model,  # Train only the LLM part
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
            max_seq_length=self.config.max_seq_length,
        )
        
        self.trainer = trainer
        return trainer
    
    def train(
        self, 
        train_dataset: Union[Dataset, str],
        eval_dataset: Optional[Union[Dataset, str]] = None,
        is_formatted: bool = False
    ) -> Dict[str, Any]:
        """
        Run instruction tuning training.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            is_formatted: Whether datasets are already formatted
            
        Returns:
            Training results
        """
        print("Starting instruction tuning training...")
        
        # Setup components
        self.setup_tokenizer()
        self.setup_model()
        self.setup_lora()
        
        # Prepare datasets
        if isinstance(train_dataset, str):
            # Load from path
            from datasets import load_dataset
            dataset_dict = load_dataset('json', data_files={
                'train': train_dataset,
                'validation': eval_dataset if eval_dataset else train_dataset
            })
        else:
            dataset_dict = DatasetDict({
                'train': train_dataset,
                'validation': eval_dataset if eval_dataset else train_dataset
            })
        
        prepared_dataset = self.prepare_dataset(dataset_dict, is_formatted)
        
        # Create trainer
        self.create_trainer(
            train_dataset=prepared_dataset['train'],
            eval_dataset=prepared_dataset.get('validation')
        )
        
        # Train
        results = self.trainer.train()
        
        # Save final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        print("Training completed!")
        return results
    
    def generate_response(
        self,
        instruction: str,
        input_text: str = "",
        max_length: int = 256,
        **generation_kwargs
    ) -> str:
        """
        Generate a response to an instruction.
        
        Args:
            instruction: The instruction/question
            input_text: Additional input context
            max_length: Maximum response length
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be set up before generation")
        
        # Format the instruction using the template
        template = self.data_formatter.templates[self.config.template_style]
        formatted_text = template["format"].format(
            instruction_key=template["instruction_key"],
            instruction=instruction,
            input_key=template["input_key"],
            input=input_text if template["input_key"] else "",
            response_key=template["response_key"],
            response=""  # Empty response to be generated
        )
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length - max_length
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.decoder.llm_model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                **generation_kwargs
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()


def main():
    """CLI interface for instruction tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ECG LLM Instruction Tuning")
    parser.add_argument("--config_path", type=str, help="Path to config file")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--eval_data", type=str, help="Path to evaluation data")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Hugging Face model name")
    parser.add_argument("--output_dir", type=str, default="./instruct_output", help="Output directory")
    
    args = parser.parse_args()
    
    # Create config
    config = InstructTuningConfig(
        huggingface_model_name=args.model_name,
        output_dir=args.output_dir
    )
    
    # Create tuner and train
    tuner = ECGInstructTuner(config)
    results = tuner.train(
        train_dataset=args.train_data,
        eval_dataset=args.eval_data
    )
    
    print("Instruction tuning completed!")


if __name__ == "__main__":
    main()
