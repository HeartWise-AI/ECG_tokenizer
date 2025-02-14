
from utils.registry import ProjectRegistry
from utils.config import GPT2FinetuningConfig

@ProjectRegistry.register("ECG_tokenizer_gpt2_finetuning")
class GPT2FinetuningProject:
    def __init__(
        self,
        config: GPT2FinetuningConfig
    ):
        self.config = config

    def run(self):
        pass
