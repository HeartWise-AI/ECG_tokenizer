import wandb

from utils.config.heartwise_config import HeartWiseConfig

class WandbWrapper:
    def __init__(self, config: HeartWiseConfig):
        self.config = config
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config=config,
        )

    def log(self, **kwargs):
        wandb.log(kwargs)

    def finish(self):
        wandb.finish()
