from typing import Any

import torch
from torch.utils.data import DataLoader

from projects.rl_finetuning_base import RLFinetuningProjectBase
from utils.config import GRPOFinetuningConfig, ECGTokenizerTrainingConfig
from utils.enums import ProjectName
from utils.registry import ProjectRegistry
from utils.wandb_wrapper import WandbWrapper
from data.grpo_prompt_dataset import GRPOPromptDataset, grpo_collate_fn
from data.grpo_binary_dataset import GRPOBinaryDataset, grpo_binary_collate_fn
from data.grpo_labelset_dataset import GRPOLabelsetDataset, grpo_labelset_collate_fn


torch.serialization.add_safe_globals([GRPOFinetuningConfig])
torch.serialization.add_safe_globals([ECGTokenizerTrainingConfig])


@ProjectRegistry.register(ProjectName.ECG_TOKENIZER_GRPO_FINETUNING)
class GRPOFinetuningProject(RLFinetuningProjectBase):
    def __init__(self, config: GRPOFinetuningConfig, wandb_wrapper: WandbWrapper):
        super().__init__(config, wandb_wrapper)
        self.config: GRPOFinetuningConfig = config

    def _setup_training_objects(self) -> dict[str, Any]:
        checkpoint_path = self.config.pretrained_tokenizer_path

        policy_model, tokenizer, checkpoint_config = self._load_model_from_checkpoint(checkpoint_path)

        # Only load ref model when KL penalty is active (beta > 0)
        beta = float(getattr(self.config, "beta", 0.0))
        if beta > 0:
            ref_model, _, _ = self._load_model_from_checkpoint(checkpoint_path)
        else:
            ref_model = None
            if self.config.is_ref_device:
                print("[GRPOFinetuningProject] Skipping ref model (beta=0, KL penalty disabled)")

        # Configure LoRA mode
        try:
            if bool(getattr(checkpoint_config, "use_lora", False)):
                policy_model.set_lora_inference_mode(False)
                if ref_model is not None:
                    ref_model.set_lora_inference_mode(True)
        except Exception:
            pass

        # Set trainable params
        trainable_mode = str(getattr(self.config, "trainable", "lora")).lower()
        self._set_trainable_params(policy_model, trainable_mode)

        verifier = str(getattr(self.config, "verifier", "bert")).lower()
        if verifier == "binary":
            train_dataset = GRPOBinaryDataset(
                dataset_path=self.config.train_dataset_path,
                tokenizer=tokenizer,
                config=self.config,
                signal_path_column=self.config.signal_path_column,
                prompt_column=getattr(self.config, "prompt_column", "prompt"),
                answer_column=getattr(self.config, "answer_column", "generated_answer"),
                filter_categories=getattr(self.config, "filter_prompt_categories", None),
            )
            collate = grpo_binary_collate_fn
            if self.config.is_ref_device:
                print(f"[GRPOFinetuningProject] verifier=binary; using GRPOBinaryDataset "
                      f"({len(train_dataset)} rows after filtering)")
        elif verifier in ("labelset", "judge"):
            train_dataset = GRPOLabelsetDataset(
                dataset_path=self.config.train_dataset_path,
                tokenizer=tokenizer,
                config=self.config,
                signal_path_column=self.config.signal_path_column,
                prompt_column=getattr(self.config, "prompt_column", "prompt"),
                answer_column=getattr(self.config, "answer_column", "generated_answer"),
                filter_categories=getattr(self.config, "filter_prompt_categories", None),
                max_rows=getattr(self.config, "max_train_rows", None),
            )
            collate = grpo_labelset_collate_fn
            if self.config.is_ref_device:
                print(f"[GRPOFinetuningProject] verifier={verifier}; using GRPOLabelsetDataset "
                      f"({len(train_dataset)} rows)")
        else:
            train_dataset = GRPOPromptDataset(
                dataset_path=self.config.train_dataset_path,
                tokenizer=tokenizer,
                config=self.config,
                signal_path_column=self.config.signal_path_column,
                messages_column=self.config.messages_column,
                report_column=self.config.report_column,
            )
            collate = grpo_collate_fn

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=collate,
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
