import os
import argparse
from utils.config import HeartWiseConfig

'''
Adapted from: https://github.com/HeartWise-AI/DeepCORO_CLIP/blob/jd/support_multigpu-issue_7/utils/parser.py
'''

class HeartWiseParser:

    @staticmethod
    def parse_config() -> HeartWiseConfig:
        """Parse command line arguments and load config file."""
        parser = argparse.ArgumentParser(description="Train ECG_tokenizer linear probing model")

        # base config
        base_group = parser.add_argument_group('Base')
        base_group.add_argument('--base_config', type=str, required=True)

        # Training parameters
        train_group = parser.add_argument_group('Training')
        train_group.add_argument('--lr', type=float)
        train_group.add_argument('--batch_size', type=int)
        train_group.add_argument('--num_epochs', type=int)

        # Optimization parameters
        optim_group = parser.add_argument_group('Optimization')
        optim_group.add_argument('--weight_decay', type=float)

        # System parameters
        system_group = parser.add_argument_group('System')
        system_group.add_argument('--seed', type=parse_optional_int)

        # Loss and metrics parameters
        metrics_group = parser.add_argument_group('Loss and Metrics')
        metrics_group.add_argument('--citerion', type=str)

        # Checkpointing parameters
        checkpoint_group = parser.add_argument_group('Checkpointing')
        checkpoint_group.add_argument('--experiment_name', type=parse_optional_str)

        args = parser.parse_args()

        # Load base config from yaml
        config: HeartWiseConfig = HeartWiseConfig.from_yaml(args.base_config)
        
        # Create sweep config from args
        config: HeartWiseConfig = HeartWiseConfig.update_config_with_args(config, args)

        
        return config