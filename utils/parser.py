import os
import argparse
from utils.parser_typing import parse_optional_str
from utils.config.heartwise_config import HeartWiseConfig

class HeartWiseParser:
    """Parser for HeartWise configuration with sweep support.
    
    Handles loading base configuration from YAML files and updating it with
    command line arguments for parameter sweeps. Supports arguments for:
    - Training (lr, batch_size, layers, dimensions)
    - Optimization (weight decay)
    - Loss functions and metrics
    - Experiment tracking (name, project, tags)
    """
    
    @staticmethod
    def parse_config() -> HeartWiseConfig:
        """Parse command line arguments and return HeartWiseConfig.
        
        Loads base config from YAML file and updates it with any provided command line arguments.
        """
        def str_to_bool(value: object) -> bool:
            if isinstance(value, bool):
                return value
            if value is None:
                raise argparse.ArgumentTypeError("Boolean value expected.")
            value_str = str(value).lower()
            if value_str in {"true", "1", "yes", "y", "t"}:
                return True
            if value_str in {"false", "0", "no", "n", "f"}:
                return False
            raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

        parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Train ECG_tokenizer linear probing model")

        # base config
        base_group = parser.add_argument_group('Base')
        base_group.add_argument('--base_config', type=str, required=True)

        # Training parameters
        train_group = parser.add_argument_group('Training')
        train_group.add_argument('--lr', type=float)
        train_group.add_argument('--llm_lr', type=float)
        train_group.add_argument('--batch_size', type=int)
        train_group.add_argument('--weight_decay', type=float)
        train_group.add_argument('--criterion', type=str)
        train_group.add_argument('--num_layers', type=int)
        train_group.add_argument('--hidden_dim', type=int)
        train_group.add_argument('--num_epochs', type=int)
        train_group.add_argument('--gradient_accumulation_steps', type=int)
        train_group.add_argument('--num_codebooks_kept', type=int)
        train_group.add_argument('--codebook_offset', type=int)
        train_group.add_argument('--phase2_freeze_llm', type=str_to_bool)
        train_group.add_argument('--bridge_dropout', type=float)
        train_group.add_argument('--instruction_dropout', type=float)
        train_group.add_argument('--lora_r', type=int)
        train_group.add_argument('--lora_alpha', type=int)
        train_group.add_argument('--lora_top_k_layers', type=int)
        
        # Optimization parameters
        optim_group = parser.add_argument_group('Optimization')
        optim_group.add_argument('--optimizer', type=str)
        optim_group.add_argument('--scheduler_type', type=str)
        optim_group.add_argument('--step_size', type=int)
        optim_group.add_argument('--gamma', type=float)
        optim_group.add_argument('--num_warmup_percent', type=float)
        optim_group.add_argument('--num_restarts', type=int)
        optim_group.add_argument('--num_hard_restarts_cycles', type=float)
        optim_group.add_argument('--warm_restart_tmult', type=int)
        optim_group.add_argument('--llm_weight_decay', type=float)
        optim_group.add_argument('--temperature_init', type=float)

        # SigLIP / loss-specific parameters
        loss_group = parser.add_argument_group('Loss')
        loss_group.add_argument('--focal_gamma_pos', type=float)
        loss_group.add_argument('--focal_gamma_neg', type=float)
        loss_group.add_argument('--focal_alpha_default', type=float)
        loss_group.add_argument('--focal_detach_weights', type=str_to_bool)
        loss_group.add_argument('--w_pos', type=float)
        loss_group.add_argument('--w_hardneg', type=float)
        loss_group.add_argument('--w_implneg', type=float)
        loss_group.add_argument('--implicit_negatives_per_batch', type=int)
        loss_group.add_argument('--implicit_negatives_per_row', type=int)
        loss_group.add_argument('--negatives_mode', type=str)
        loss_group.add_argument('--loss_type', type=str)

        # Stage-1 / Bridge-specific parameters (also usable by SigLIP)
        stage1_group = parser.add_argument_group('Stage1/Bridge & CE')
        stage1_group.add_argument('--bridge_num_layers', type=int)
        stage1_group.add_argument('--bridge_num_heads', type=int)
        stage1_group.add_argument('--num_query_tokens', type=int)
        stage1_group.add_argument('--cross_every', type=int)
        stage1_group.add_argument('--etc_weight', type=float)
        stage1_group.add_argument('--etm_weight', type=float)
        stage1_group.add_argument('--etg_weight', type=float)
        stage1_group.add_argument('--etm_hard_neg_k', type=int)
        stage1_group.add_argument('--etm_warmup_steps', type=int)
        stage1_group.add_argument('--etg_delay_steps', type=int)
        stage1_group.add_argument('--etg_warmup_steps', type=int)
        stage1_group.add_argument('--focal_infonce', type=str_to_bool)
        stage1_group.add_argument('--validate_with_etg', type=str_to_bool)
        stage1_group.add_argument('--validate_etg_fraction', type=float)
        stage1_group.add_argument('--validate_etg_save', type=str_to_bool)

        # Tokenizer parameters
        tokenizer_group = parser.add_argument_group('Tokenizer')
        tokenizer_group.add_argument('--max_token_length', type=int)

        # Embedding adapter parameters
        adapter_group = parser.add_argument_group('Embedding Adapter')
        adapter_group.add_argument('--adapter_dropout', type=float)
        adapter_group.add_argument('--adapter_lr', type=float)
        adapter_group.add_argument('--adapter_weight_decay', type=float)

        # Tail handling parameters
        tail_group = parser.add_argument_group('Tail Handling')
        tail_group.add_argument('--tail_enable', type=str_to_bool)
        tail_group.add_argument('--tail_top_n', type=int)
        tail_group.add_argument('--tail_min_positives', type=int)
        tail_group.add_argument('--tail_alpha_boost', type=float)

        # Checkpointing parameters
        checkpoint_group = parser.add_argument_group('Checkpointing')
        checkpoint_group.add_argument('--project', type=parse_optional_str)
        checkpoint_group.add_argument('--entity', type=parse_optional_str)
        checkpoint_group.add_argument('--name', type=parse_optional_str)

        # Inference parameters
        inference_group = parser.add_argument_group('Inference')
        inference_group.add_argument('--inference_dataset_path', type=str)
        inference_group.add_argument('--inference_checkpoint_path', type=str)
        
        # Parse arguments
        args: argparse.Namespace = parser.parse_args()

        # Load base config from yaml
        config: HeartWiseConfig = HeartWiseConfig.from_yaml(args.base_config)
        
        # Create sweep config from args
        config: HeartWiseConfig = HeartWiseConfig.update_config_with_args(config, args)
        
        # Set base config path
        config.base_config_path = args.base_config
        
        # Set GPU info
        HeartWiseConfig.set_gpu_info_in_place(config)
        
        return config
