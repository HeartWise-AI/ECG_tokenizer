import os

from typing import Dict, Any, List, Type
from dataclasses import dataclass, asdict

from utils.enums import BridgeName
from utils.files_handler import load_yaml
from utils.registry import ConfigRegistry

@dataclass
class HeartWiseConfig: 
    # Pipeline project
    pipeline_project: str
    
    # Wandb project
    wandb_project: str
    wandb_entity: str
    use_wandb: bool
                
    # Base config path
    base_config_path: str
    
    # Common attributes used by all projects (set dynamically)
    seed: int
    device: int
    run_mode: str
    world_size: int
    output_dir: str
    is_ref_device: bool
                    
    @classmethod
    def update_config_with_args(cls, base_config: 'HeartWiseConfig', args: Any) -> 'HeartWiseConfig':  
        """Update a HeartWiseConfig instance with command line arguments."""
        data_parameters: Dict[str, Any] = base_config.to_dict().copy()
        registered_config: Type[HeartWiseConfig] = ConfigRegistry.get(base_config.pipeline_project)
        for key, value in vars(args).items():
            if value is not None and key in registered_config.__dataclass_fields__:
                data_parameters[key] = value
        return registered_config(**data_parameters)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'HeartWiseConfig':
        """Create a HeartWiseConfig instance from a YAML file."""
        yaml_config: Dict[str, Any] = load_yaml(yaml_path)
        pipeline_project = yaml_config.get('pipeline_project', None)
        
        # Better error handling
        if pipeline_project is None:
            raise ValueError("pipeline_project is not set in the yaml file")
            
        registered_config: Type[HeartWiseConfig] = ConfigRegistry.get(pipeline_project)
        
        data_parameters: Dict[str, Any] = {}
        undefined_parameters: List[str] = []
        for key, value in yaml_config.items():
            if key in registered_config.__dataclass_fields__:
                data_parameters[key] = value
            else:
                undefined_parameters.append(key)
        
        if len(undefined_parameters) > 0:
            raise ValueError(f"{undefined_parameters} were defined in the {yaml_path} file but not in the registered config {registered_config.__name__}")
        
        # Set the base_config_path
        data_parameters['base_config_path'] = yaml_path
        
        # Set the common attributes used by all projects (set dynamically) -> avoid type errors
        data_parameters['seed'] = 42
        data_parameters['output_dir'] = ""
        data_parameters['device'] = int(os.environ["LOCAL_RANK"])
        data_parameters['world_size'] = int(os.environ["WORLD_SIZE"])
        data_parameters['is_ref_device'] = (data_parameters['device'] == 0)
        
        return registered_config(**data_parameters)    

    @classmethod
    def set_gpu_info_in_place(cls, config: 'HeartWiseConfig') -> None:
        """Set GPU information from environment variables."""
        try:
            config.device = int(os.environ["LOCAL_RANK"])  # type: ignore
            config.world_size = int(os.environ["WORLD_SIZE"])  # type: ignore
            config.is_ref_device = (config.device == 0)  # type: ignore
        except KeyError:
            # Not running in distributed mode; default to single GPU/CPU
            config.device = 0  # type: ignore
            config.world_size = 1  # type: ignore
            config.is_ref_device = True  # type: ignore

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for wandb."""
        return asdict(self)
