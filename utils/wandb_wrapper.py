import wandb
from typing import Any
from utils.config.heartwise_config import HeartWiseConfig

class WandbWrapper:
    """
    A wrapper class for integrating Weights & Biases (wandb) logging with an optional initialization
    strategy based on device roles.

    Attributes:
        config (HeartWiseConfig): The configuration instance containing necessary wandb settings.
        initialized (bool): Indicates if wandb should be initialized.
    """    
    def __init__(
        self, 
        config: HeartWiseConfig,
        initialized: bool = False,
        is_ref_device: bool = False
    ):
        """
        Initializes wandb logging based on the provided flags.

        Args:
            config (HeartWiseConfig): Configuration settings.
            initialized (bool): Whether to initialize wandb.
            is_ref_device (bool): If True, initializes wandb for full logging; 
                                    otherwise, wandb is set into a disabled mode.
           """        
        self.config = config
        if initialized:
            if is_ref_device:
                run = wandb.init(
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    config=config.to_dict(),
                )
                print("="*60)
                print("🚀 WANDB RUN INITIALIZED")
                print("="*60)
                print(f"  Project: {config.wandb_project}")
                print(f"  Entity: {config.wandb_entity}")
                print(f"  Run ID: {run.id}")
                print(f"  Run Name: {run.name}")
                print(f"  Run URL: {run.url}")
                print("="*60)
            else:
                wandb.init(mode="disabled")
                print("📊 W&B initialized in disabled mode (non-reference device)")
        self.initialized: bool = initialized
        
    def is_initialized(self)->bool:
        return self.initialized
    
    def get_run_id(self)->str:
        return wandb.run.id if wandb.run is not None else "no_wandb"
        
    def log(self, kwargs: dict[str, Any], step: int | None = None, commit: bool | None = None):
        """
        Wrapper around wandb.log that defaults to commit=False when an explicit
        step is provided. This prevents W&B from auto-incrementing its internal
        step counter between multiple logs that share the same global_step
        (e.g., during gradient accumulation), which otherwise triggers
        "step N < current step M" warnings.
        """
        # When caller passes a step but no explicit commit flag, default to commit=False
        # so repeated logs at the same step do not advance W&B's internal counter.
        if step is not None and commit is None:
            commit = False

        if step is not None and commit is not None:
            wandb.log(kwargs, step=step, commit=commit)
        elif step is not None:
            wandb.log(kwargs, step=step)
        elif commit is not None:
            wandb.log(kwargs, commit=commit)
        else:
            wandb.log(kwargs)

    def finish(self):
        wandb.finish()
