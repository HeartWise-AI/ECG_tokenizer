import wandb

from utils.config.heartwise_config import HeartWiseConfig

class WandbWrapper:
    def __init__(
        self, 
        config: HeartWiseConfig,
        initialized: bool = False
    ):
        self.config = config
        if initialized:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=config,
            )
        self.initialized: bool = initialized
        
    def is_initialized(self)->bool:
        return self.initialized
        
    def log(self, **kwargs):
        wandb.log(kwargs)

    def finish(self):
        wandb.finish()
