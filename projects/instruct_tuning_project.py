import torch

from typing import Any
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import LRScheduler

from transformers import AutoTokenizer
from datasets import load_dataset, Dataset

from utils.ddp import DistributedUtils
from utils.schedulers import get_scheduler
from utils.registry import (
    ModelRegistry,
    ProjectRegistry
)
from utils.enums import ProjectName
from utils.wandb_wrapper import WandbWrapper
from utils.config import InstructTuningConfig, ECGTokenizerTrainingConfig
from projects.base_project import BaseProject
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper
from utils.instruct_tuning.instructor import ECGInstructTuner
from utils.instruct_tuning.data_formatter import ECGInstructDataFormatter

# Add the config to the safe globals
torch.serialization.add_safe_globals([InstructTuningConfig])
torch.serialization.add_safe_globals([ECGTokenizerTrainingConfig])

@ProjectRegistry.register(ProjectName.ECG_TOKENIZER_INSTRUCT_TUNING)
class InstructTuningProject(BaseProject):
    """Instruction tuning project for ECG tokenizer models.
    
    Implements instruction tuning of a pretrained LLM on ECG tokenizer features
    using supervised fine-tuning with instruction-following datasets.
    """    
    def __init__(
        self, 
        config: InstructTuningConfig, 
        wandb_wrapper: WandbWrapper
    ):
        """Initialize instruction tuning project.
        
        Args:
            config: Instruction tuning configuration
            wandb_wrapper: Weights & Biases logging wrapper
        """        
        super().__init__(config, wandb_wrapper)
        self.config: InstructTuningConfig = config
        self.instruct_tuner = None
        
    def run(self):
        """Execute the instruction tuning workflow."""
        super().run()
        
    def _setup_training_objects(self) -> dict[str, Any]: 
        """Setup objects required for instruction tuning.
        
        Loads pretrained tokenizer, sets up instruction tuning components,
        and prepares training infrastructure.
        
        Returns:
            Dictionary containing training objects: tuner, datasets, and results
        """        
        # Load the pretrained tokenizer
        state_dict = self._load_checkpoint(self.config.pretrained_tokenizer_path)
        
        # Get the config from the pretrained tokenizer
        pretrained_config = state_dict['config']
        if self.config.is_ref_device:
            print(f"Pretrained config: {pretrained_config}")                
        
        # Set encoder_name to the pretrained encoder_name
        self.config.encoder_name = pretrained_config.encoder_name
        # Set quantizer_name to the pretrained quantizer_name
        self.config.quantizer_name = pretrained_config.quantizer_name
        # Set num_quantizers to the pretrained num_quantizers
        self.config.num_quantizers = pretrained_config.num_quantizers
        # Set codebook_size to the pretrained codebook_size
        self.config.codebook_size = pretrained_config.codebook_size
        
        # Create instruct tuner config
        from utils.instruct_tuning.instructor import InstructTuningConfig as TunerConfig
        tuner_config = TunerConfig(
            huggingface_model_name=self.config.huggingface_model_name,
            llm_input_embedding_size=self.config.llm_input_embedding_size,
            decoder_name=self.config.decoder_name,
            adapter_name=self.config.adapter_name,
            encoder_name=self.config.encoder_name,
            quantizer_name=self.config.quantizer_name,
            num_quantizers=self.config.num_quantizers,
            codebook_size=self.config.codebook_size,
            pretrained_tokenizer_path=self.config.pretrained_tokenizer_path,
            template_style=self.config.template_style,
            max_seq_length=self.config.max_seq_length,
            include_metadata=self.config.include_metadata,
            enhanced_prompt=self.config.enhanced_prompt,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            fp16=self.config.fp16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            dataloader_num_workers=self.config.dataloader_num_workers,
            remove_unused_columns=self.config.remove_unused_columns,
            use_lora=self.config.use_lora,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            lora_target_modules=self.config.lora_target_modules,
            output_dir=self.config.base_checkpoint_path,
            run_name=f"{self.config.runner_name}_{self.config.huggingface_model_name.replace('/', '_')}",
            use_wandb=self.config.use_wandb,
            wandb_project=self.config.wandb_project if hasattr(self.config, 'wandb_project') else "ecg_instruct_tuning",
            wandb_entity=self.config.wandb_entity if hasattr(self.config, 'wandb_entity') else None,
            use_early_stopping=self.config.use_early_stopping,
            early_stopping_patience=self.config.early_stopping_patience,
            early_stopping_threshold=self.config.early_stopping_threshold,
        )
        
        # Create the instruction tuner
        self.instruct_tuner = ECGInstructTuner(tuner_config)
        
        # Print training configuration
        self._print_training_config()
        
        # Run instruction tuning
        print("Starting instruction tuning...")
        results = self.instruct_tuner.train(
            train_dataset=self.config.train_dataset_path,
            eval_dataset=self.config.validation_dataset_path,
            is_formatted=False  # Dataset needs formatting
        )
        
        return {
            "tuner": self.instruct_tuner,
            "results": results,
        }
    
    def _print_training_config(self):
        """Print detailed training configuration and model statistics."""
        print("\n" + "="*60)
        print("INSTRUCTION TUNING CONFIGURATION")
        print("="*60)
        
        print(f"Model: {self.config.huggingface_model_name}")
        print(f"Decoder: {self.config.decoder_name}")
        print(f"Adapter: {self.config.adapter_name}")
        print(f"Template Style: {self.config.template_style}")
        print(f"Max Sequence Length: {self.config.max_seq_length}")
        print(f"Include Metadata: {self.config.include_metadata}")
        print(f"Enhanced Prompts: {self.config.enhanced_prompt}")
        
        print(f"\nTraining Parameters:")
        print(f"  Learning Rate: {self.config.learning_rate}")
        print(f"  Epochs: {self.config.num_epochs}")
        print(f"  Batch Size: {self.config.per_device_train_batch_size}")
        print(f"  Gradient Accumulation: {self.config.gradient_accumulation_steps}")
        print(f"  Use LoRA: {self.config.use_lora}")
        if self.config.use_lora:
            print(f"    LoRA r: {self.config.lora_r}")
            print(f"    LoRA alpha: {self.config.lora_alpha}")
            print(f"    LoRA dropout: {self.config.lora_dropout}")
        
        print(f"\nDataset:")
        print(f"  Train: {self.config.train_dataset_path}")
        print(f"  Validation: {self.config.validation_dataset_path}")
        
        print("="*60 + "\n")
    
    def _setup_inference_objects(self) -> dict[str, Any]:
        """Setup objects for inference mode.
        
        Returns:
            Dictionary containing inference objects
        """        
        if self.instruct_tuner is None:
            # Load from checkpoint
            tuner_config = self._create_tuner_config()
            self.instruct_tuner = ECGInstructTuner(tuner_config)
            self.instruct_tuner.setup_tokenizer()
            self.instruct_tuner.setup_model()
        
        return {
            "tuner": self.instruct_tuner,
        }
    
    def _setup_extraction_objects(self) -> dict[str, Any]:
        """Setup objects for extraction mode.
        
        Raises:
            NotImplementedError: Extraction not implemented for instruction tuning
        """        
        raise NotImplementedError("Extraction is not implemented for this project")
    
    def _create_tuner_config(self):
        """Create tuner config from project config."""
        from utils.instruct_tuning.instructor import InstructTuningConfig as TunerConfig
        
        return TunerConfig(
            huggingface_model_name=self.config.huggingface_model_name,
            llm_input_embedding_size=self.config.llm_input_embedding_size,
            decoder_name=self.config.decoder_name,
            adapter_name=self.config.adapter_name,
            template_style=self.config.template_style,
            max_seq_length=self.config.max_seq_length,
            output_dir=self.config.base_checkpoint_path,
        )
    
    def generate_response(
        self,
        instruction: str,
        input_text: str = "",
        max_length: int = 256,
        **generation_kwargs
    ) -> str:
        """
        Generate a response to an instruction using the tuned model.
        
        Args:
            instruction: The instruction/question
            input_text: Additional input context  
            max_length: Maximum response length
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        if self.instruct_tuner is None:
            raise ValueError("Instruction tuner not initialized. Run training or inference setup first.")
        
        return self.instruct_tuner.generate_response(
            instruction=instruction,
            input_text=input_text,
            max_length=max_length,
            **generation_kwargs
        )
