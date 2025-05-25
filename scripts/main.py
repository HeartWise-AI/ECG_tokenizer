
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)


from typing import Union

from utils.seed import set_seed
from utils.ddp import DistributedUtils
from utils.parser import HeartWiseParser
from utils.registry import ProjectRegistry
from projects import (
    LLMFinetuningProject,
    BertReportClassifierProject,
    ECGTokenizerTrainingProject
)
from utils.wandb_wrapper import WandbWrapper
from utils.config.heartwise_config import HeartWiseConfig


def main(config: HeartWiseConfig):
    
    try:
        # Set seed for reproducibility
        if hasattr(config, "seed"):
            set_seed(config.seed)
        
        # Initialize process group with explicit device ID and world size
        DistributedUtils.ddp_setup(
            gpu_id=config.device, 
            world_size=config.world_size
        )
        
        # Initialize wandb wrapper
        wandb_wrapper: WandbWrapper = WandbWrapper(
            config=config, # The config object
            initialized=config.use_wandb, # If wandb is not initialized, it will not be initialized
            is_ref_device=config.is_ref_device # If the device is a reference device, it will not be initialized
        )
        
        # Sync the process group
        DistributedUtils.sync_process_group(
            world_size=config.world_size,
            device_ids=config.device
        )
        
        # Initialize the project
        project: Union[
            LLMFinetuningProject, 
            BertReportClassifierProject,
            ECGTokenizerTrainingProject
        ] = ProjectRegistry.get(
            name=config.pipeline_project # The project to run
        )(
            config=config, # The config object
            wandb_wrapper=wandb_wrapper # The wandb wrapper
        )
        
        # Run the project
        project.run()
        
    except Exception as e:
        print(f"Error: {e}")
        if config.is_ref_device:
            wandb_wrapper.finish()
        DistributedUtils.ddp_cleanup()
        raise e
        
    finally:
        if config.is_ref_device:
            wandb_wrapper.finish()
        DistributedUtils.ddp_cleanup()

if __name__ == "__main__":
    # Parse the config
    config: HeartWiseConfig = HeartWiseParser.parse_config()
    if config.is_ref_device:
        print(f"Config - device {config.device}: {config}")
        
    # Run the main function
    main(config)