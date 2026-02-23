
import os
import sys
import types

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Shim for checkpoints saved with older transformers that had a fast Gemma tokenizer.
# transformers >=5 removed tokenization_gemma_fast; register an alias so torch.load
# can unpickle the class reference without error.
_FAST_MOD = "transformers.models.gemma.tokenization_gemma_fast"
if _FAST_MOD not in sys.modules:
    try:
        from transformers.models.gemma import tokenization_gemma as _slow_mod
        _shim = types.ModuleType(_FAST_MOD)
        _shim.GemmaTokenizerFast = _slow_mod.GemmaTokenizer
        sys.modules[_FAST_MOD] = _shim
    except ImportError:
        pass

# transformers >=5 removed encode_plus from tokenizers.  Patch it back as
# an alias for __call__ so existing dataset code keeps working.
try:
    from transformers import PreTrainedTokenizerBase
    if not hasattr(PreTrainedTokenizerBase, 'encode_plus'):
        PreTrainedTokenizerBase.encode_plus = lambda self, *a, **kw: self(*a, **kw)
except ImportError:
    pass

from utils.seed import set_seed
from utils.ddp import DistributedUtils
from utils.parser import HeartWiseParser
from utils.registry import ProjectRegistry
from projects.types import ProjectT
from utils.wandb_wrapper import WandbWrapper
from utils.config.heartwise_config import HeartWiseConfig


def main(config: HeartWiseConfig):
    
    try:
        # Set seed for reproducibility
        if hasattr(config, "seed"):
            print(f"Setting seed to {config.seed} for reproducibility")
            set_seed(config.seed)
        
        # Initialize process group with explicit device ID and world size
        DistributedUtils.ddp_setup(
            gpu_id=config.device, 
            world_size=config.world_size
        )
        
        # Initialize wandb wrapper
        print(f"📊 W&B Configuration: use_wandb={config.use_wandb}, is_ref_device={config.is_ref_device}")
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
        project: ProjectT = ProjectRegistry.get(
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