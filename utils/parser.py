import os
import argparse
from utils.config.heartwise_config import HeartWiseConfig
from utils.parser_typing import (
    str2bool,
    parse_list,
    parse_optional_int,
    parse_optional_str
)


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
        parser = argparse.ArgumentParser(description="Train ECG_tokenizer linear probing model")

        # base config
        base_group = parser.add_argument_group('Base')
        base_group.add_argument('--base_config', type=str, required=True)

        # Training parameters
        train_group = parser.add_argument_group('Training')
        train_group.add_argument('--lr', type=float)
        train_group.add_argument('--llm_lr', type=float)
        train_group.add_argument('--embedding_reducer_lr', type=float)
        train_group.add_argument('--batch_size', type=int)
        
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
        optim_group.add_argument('--embedding_reducer_weight_decay', type=float)
        
        # Tokenizer parameters
        tokenizer_group = parser.add_argument_group('Tokenizer')
        tokenizer_group.add_argument('--max_token_length', type=int)

        # Embedding reducer parameters
        embedding_reducer_group = parser.add_argument_group('Embedding Reducer')
        embedding_reducer_group.add_argument('--reducer_dropout', type=float)

        # Checkpointing parameters
        checkpoint_group = parser.add_argument_group('Checkpointing')
        checkpoint_group.add_argument('--project', type=parse_optional_str)
        checkpoint_group.add_argument('--entity', type=parse_optional_str)
        checkpoint_group.add_argument('--name', type=parse_optional_str)
        
        # Parse arguments
        args = parser.parse_args()

        # Load base config from yaml
        config: HeartWiseConfig = HeartWiseConfig.from_yaml(args.base_config)
        
        # Create sweep config from args
        config: HeartWiseConfig = HeartWiseConfig.update_config_with_args(config, args)
        
        # Set GPU info
        HeartWiseConfig.set_gpu_info_in_place(config)
        
        return config