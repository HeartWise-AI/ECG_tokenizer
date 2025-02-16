
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)


import wandb
from typing import Union

from utils.parser import HeartWiseParser
from utils.registry import ProjectRegistry
from projects import LLMFinetuningProject
from utils.wandb_wrapper import WandbWrapper
from utils.config.heartwise_config import HeartWiseConfig

def main(config: HeartWiseConfig):
    # Initialize wandb wrapper
    wandb_wrapper: WandbWrapper = WandbWrapper(
        config=config, # The config object
        initialized=config.use_wandb # If wandb is not initialized, it will not be initialized
    )
    
    project: Union[LLMFinetuningProject] = ProjectRegistry.get(
        name=config.pipeline_project # The project to run
    )(
        config=config, # The config object
        wandb_wrapper=wandb_wrapper # The wandb wrapper
    )
    
    # Run the project
    project.run()


if __name__ == "__main__":
    # Parse the config
    config: HeartWiseConfig = HeartWiseParser.parse_config()
    
    # Run the main function
    main(config)