from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List

from utils.files_handler import load_yaml

'''
Adapted from: https://github.com/HeartWise-AI/DeepCORO_CLIP/blob/jd/support_multigpu-issue_7/utils/config.py
'''

@dataclass
class HeartWiseConfig:    
    # Training parameters
    lr: float
    batch_size: int
    num_layers: int
    hidden_dim: int
    
    # Optimization parameters
    weight_decay: float

    # Loss and metrics parameters
    criterion: str

    # Checkpointing parameters
    experiment_name: Optional[str]

    name: str
    project: str
    entity: str
       
    @classmethod
    def update_config_with_args(cls, base_config: 'HeartWiseConfig', args: Any) -> 'HeartWiseConfig':  
        """Update a HeartWiseConfig instance with command line arguments."""
        data_parameters: Dict[str, Any] = base_config.to_dict().copy()
        for key, value in vars(args).items():
            if value is not None and key in cls.__dataclass_fields__:
                data_parameters[key] = value
        return cls(**data_parameters)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'HeartWiseConfig':
        """Create a HeartWiseConfig instance from a YAML file."""
        yaml_config: Dict[str, Any] = load_yaml(yaml_path)
        data_parameters: Dict[str, Any] = {}
        for key, value in yaml_config.items():
            if key in cls.__dataclass_fields__:
                data_parameters[key] = value
        return cls(**data_parameters)    

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for wandb."""
        return asdict(self) 