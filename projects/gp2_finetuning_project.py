
from utils.registry import ProjectRegistry
from utils.config import GPT2FinetuningConfig
from utils.wandb_wrapper import WandbWrapper

@ProjectRegistry.register("ECG_tokenizer_gpt2_finetuning")
class GPT2FinetuningProject:
    def __init__(
        self,
        config: GPT2FinetuningConfig,
        wandb_wrapper: WandbWrapper
    ):
        self.config = config
        self.wandb_wrapper = wandb_wrapper

    def run(self):
        pass
