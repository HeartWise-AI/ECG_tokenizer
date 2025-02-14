
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
from projects import GPT2FinetuningProject
from utils.config.heartwise_config import HeartWiseConfig


def main(config: HeartWiseConfig):
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        config=config,
    )
    
    project: Union[GPT2FinetuningProject] = ProjectRegistry.get(
        name=config.pipeline_project
    )(
        config=config
    )
    project.run()

if __name__ == "__main__":
    config: HeartWiseConfig = HeartWiseParser.parse_config()
    main(config)