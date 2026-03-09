from typing import Any

import torch
from torch.utils.data import DataLoader

from projects.rl_finetuning_base import RLFinetuningProjectBase
from utils.config import DPOFinetuningConfig, ECGTokenizerTrainingConfig
from utils.enums import ProjectName
from utils.registry import ProjectRegistry
from utils.wandb_wrapper import WandbWrapper
from data.dpo_pair_dataset import DPOPairDataset, dpo_collate_fn


torch.serialization.add_safe_globals([DPOFinetuningConfig])
torch.serialization.add_safe_globals([ECGTokenizerTrainingConfig])


@ProjectRegistry.register(ProjectName.ECG_TOKENIZER_DPO_FINETUNING)
class DPOFinetuningProject(RLFinetuningProjectBase):
    def __init__(self, config: DPOFinetuningConfig, wandb_wrapper: WandbWrapper):
        super().__init__(config, wandb_wrapper)
        self.config: DPOFinetuningConfig = config

    def _setup_training_objects(self) -> dict[str, Any]:
        checkpoint_path = self.config.pretrained_tokenizer_path

        policy_model, tokenizer, checkpoint_config = self._load_model_from_checkpoint(checkpoint_path)
        ref_model, _, _ = self._load_model_from_checkpoint(checkpoint_path)

        # Configure LoRA mode
        try:
            if bool(getattr(checkpoint_config, "use_lora", False)):
                policy_model.set_lora_inference_mode(False)
                ref_model.set_lora_inference_mode(True)
        except Exception:
            pass

        # Set trainable params
        trainable_mode = str(getattr(self.config, "trainable", "lora")).lower()
        trainable_regex = getattr(self.config, "trainable_regex", None)
        self._set_trainable_params(policy_model, trainable_mode, trainable_regex)

        max_length = int(self.config.max_token_length or getattr(checkpoint_config, "max_token_length", 640))
        train_dataset = DPOPairDataset(
            path=self.config.train_pairs_path,
            tokenizer=tokenizer,
            config=self.config,
            max_length=max_length,
            waveform_key=self.config.waveform_key,
            prompt_key=self.config.prompt_key,
            chosen_key=self.config.chosen_key,
            rejected_key=self.config.rejected_key,
            weight_key=self.config.weight_key,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=dpo_collate_fn,
        )

        optimizer = torch.optim.AdamW(
            [p for p in policy_model.parameters() if p.requires_grad],
            lr=self.config.lr,
        )

        return {
            "model": policy_model,
            "ref_model": ref_model,
            "train_dataloader": train_loader,
            "optimizer": optimizer,
            "scheduler": None,
        }
