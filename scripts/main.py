
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
    wandb_wrapper: WandbWrapper = WandbWrapper(config) if config.use_wandb else None
    
    project: Union[LLMFinetuningProject] = ProjectRegistry.get(
        name=config.pipeline_project
    )(
        config=config,
        wandb_wrapper=wandb_wrapper
    )
    project.run()

if __name__ == "__main__":
    config: HeartWiseConfig = HeartWiseParser.parse_config()
    main(config)