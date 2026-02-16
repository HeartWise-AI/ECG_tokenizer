import os
from types import SimpleNamespace
from typing import Any, Tuple

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from projects.base_project import BaseProject
from utils.config import DPOFinetuningConfig, ECGTokenizerTrainingConfig
from utils.enums import ProjectName, DecoderMode
from utils.registry import ModelRegistry, ProjectRegistry
from utils.wandb_wrapper import WandbWrapper
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper
from utils.files_handler import load_yaml
from data.dpo_pair_dataset import DPOPairDataset, dpo_collate_fn


torch.serialization.add_safe_globals([DPOFinetuningConfig])
torch.serialization.add_safe_globals([ECGTokenizerTrainingConfig])


def _coerce_config(config_obj: Any) -> Any:
    if isinstance(config_obj, SimpleNamespace):
        return config_obj
    if isinstance(config_obj, dict):
        return SimpleNamespace(**{key: _coerce_config(value) for key, value in config_obj.items()})
    if isinstance(config_obj, list):
        return [_coerce_config(item) for item in config_obj]
    if isinstance(config_obj, tuple):
        return tuple(_coerce_config(item) for item in config_obj)
    return config_obj


def _cfg_get(container: Any, key: str, fallback: Any = None) -> Any:
    if container is None:
        return fallback
    if isinstance(container, dict) and key in container:
        return container[key]
    if hasattr(container, key):
        val = getattr(container, key)
        return val if val is not None else fallback
    return fallback


@ProjectRegistry.register(ProjectName.ECG_TOKENIZER_DPO_FINETUNING)
class DPOFinetuningProject(BaseProject):
    def __init__(self, config: DPOFinetuningConfig, wandb_wrapper: WandbWrapper):
        super().__init__(config, wandb_wrapper)
        self.config: DPOFinetuningConfig = config

    def run(self):
        super().run()

    def _resolve_checkpoint_structure(
        self,
        checkpoint_config: Any,
        checkpoint_path: str,
    ) -> Tuple[Any, str, str, int, int]:
        config_obj = _coerce_config(checkpoint_config)

        encoder_name = getattr(config_obj, "encoder_name", None)
        quantizer_name = getattr(config_obj, "quantizer_name", None)
        num_quantizers = getattr(config_obj, "num_quantizers", None)
        codebook_size = getattr(config_obj, "codebook_size", None)

        missing_fields = [
            field for field, value in (
                ("encoder_name", encoder_name),
                ("quantizer_name", quantizer_name),
                ("num_quantizers", num_quantizers),
                ("codebook_size", codebook_size),
            ) if value is None
        ]
        if missing_fields:
            raise ValueError(
                f"Checkpoint '{checkpoint_path}' missing fields {missing_fields}."
            )

        return (
            config_obj,
            str(encoder_name),
            str(quantizer_name),
            int(num_quantizers),
            int(codebook_size),
        )

    def _load_model_from_checkpoint(self, checkpoint_path: str):
        checkpoint_data = self._load_checkpoint(checkpoint_path)
        checkpoint_config = checkpoint_data["config"]
        (
            pretrained_config,
            encoder_name,
            quantizer_name,
            num_quantizers,
            codebook_size,
        ) = self._resolve_checkpoint_structure(checkpoint_config, checkpoint_path)

        # fill config defaults for dataset usage
        for key in (
            "tokenizer_name",
            "huggingface_model_name",
            "decoder_name",
            "decoder_mode",
            "llm_input_embedding_size",
            "bridge_name",
            "num_query_tokens",
            "bridge_mid_dim",
            "bridge_num_heads",
            "bridge_dropout",
            "bridge_num_special_tokens",
            "bridge_qformer_layers",
            "bridge_text_hidden_size",
            "bridge_bias_last_codebook",
            "bridge_codebook_dropout",
            "bridge_cross_every",
            "instruction_dropout",
            "num_ecg_tokens",
            "ecg_token_start_id",
            "prefix_tuning",
            "medgemma_prompt_style",
            "ecg_waveform_length",
            "ecg_num_leads",
            "num_codebooks_kept",
            "codebook_offset",
            "num_quantizers",
            "codebook_size",
            "max_token_length",
            # LoRA settings
            "use_lora",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "lora_target_modules",
            "lora_bias",
            "lora_config",
        ):
            if getattr(self.config, key, None) is None and hasattr(pretrained_config, key):
                setattr(self.config, key, getattr(pretrained_config, key))

        self.config.encoder_name = encoder_name
        self.config.quantizer_name = quantizer_name
        self.config.num_quantizers = num_quantizers
        self.config.codebook_size = codebook_size

        checkpoint_dir = os.path.dirname(checkpoint_path)
        config_yaml_path = os.path.join(checkpoint_dir, "config.yaml")
        yaml_config = load_yaml(config_yaml_path) if os.path.exists(config_yaml_path) else None

        tokenizer = AutoTokenizer.from_pretrained(pretrained_config.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        decoder_mode = pretrained_config.decoder_mode if isinstance(pretrained_config.decoder_mode, DecoderMode) else DecoderMode(pretrained_config.decoder_mode)
        num_visual_tokens = getattr(pretrained_config, "num_query_tokens", getattr(pretrained_config, "num_visual_tokens", None))

        num_codebooks_kept = _cfg_get(yaml_config, "num_codebooks_kept", _cfg_get(pretrained_config, "num_codebooks_kept", None))
        checkpoint_state_dict = checkpoint_data["model_state_dict"]
        mix_gate_key = "decoder.bridge.mix_gate.0.weight"
        if mix_gate_key in checkpoint_state_dict:
            mix_gate_shape = checkpoint_state_dict[mix_gate_key].shape
            hidden_dim = mix_gate_shape[0]
            inferred_codebooks = mix_gate_shape[1] // hidden_dim
            if num_codebooks_kept is None or num_codebooks_kept != inferred_codebooks:
                num_codebooks_kept = inferred_codebooks

        codebook_offset = _cfg_get(yaml_config, "codebook_offset", _cfg_get(pretrained_config, "codebook_offset", 0))

        lora_config = getattr(pretrained_config, "lora_config", None)
        if lora_config is None and bool(getattr(pretrained_config, "use_lora", False)):
            lora_config = {
                "r": int(getattr(pretrained_config, "lora_r", 16)),
                "lora_alpha": int(getattr(pretrained_config, "lora_alpha", getattr(pretrained_config, "lora_r", 16))),
                "lora_dropout": float(getattr(pretrained_config, "lora_dropout", 0.0)),
                "target_modules": list(getattr(pretrained_config, "lora_target_modules", None) or []),
                "bias": str(getattr(pretrained_config, "lora_bias", "none")),
            }
            try:
                setattr(pretrained_config, "lora_config", lora_config)
            except Exception:
                pass

        model = ECG_Tokenizer_Wrapper(
            encoder_name=pretrained_config.encoder_name,
            quantizer_name=pretrained_config.quantizer_name,
            decoder_name=pretrained_config.decoder_name,
            num_quantizers=int(getattr(pretrained_config, "num_quantizers", 8)),
            codebook_size=int(getattr(pretrained_config, "codebook_size", 512)),
            decoder_mode=decoder_mode,
            huggingface_model_name=pretrained_config.huggingface_model_name,
            llm_input_embedding_size=int(pretrained_config.llm_input_embedding_size),
            bridge_name=pretrained_config.bridge_name,
            num_visual_tokens=num_visual_tokens,
            bridge_mid_dim=int(getattr(pretrained_config, "bridge_mid_dim", 512)),
            bridge_num_heads=int(getattr(pretrained_config, "bridge_num_heads", 8)),
            bridge_dropout=float(getattr(pretrained_config, "bridge_dropout", 0.1)),
            bridge_num_special_tokens=int(getattr(pretrained_config, "bridge_num_special_tokens", 4)),
            bridge_qformer_layers=getattr(pretrained_config, "bridge_qformer_layers", None),
            bridge_text_hidden_size=getattr(pretrained_config, "bridge_text_hidden_size", None),
            bridge_bias_last_codebook=_cfg_get(yaml_config, "bridge_bias_last_codebook", _cfg_get(pretrained_config, "bridge_bias_last_codebook", None)),
            bridge_codebook_dropout=_cfg_get(yaml_config, "bridge_codebook_dropout", _cfg_get(pretrained_config, "bridge_codebook_dropout", None)),
            bridge_cross_every=_cfg_get(yaml_config, "bridge_cross_every", _cfg_get(pretrained_config, "bridge_cross_every", None)),
            instruction_dropout=_cfg_get(yaml_config, "instruction_dropout", _cfg_get(pretrained_config, "instruction_dropout", 0.0)),
            stage1_checkpoint_path=None,
            use_lora=bool(getattr(pretrained_config, "use_lora", False)),
            lora_config=lora_config,
            tokenizer=tokenizer,
            ecg_token_start_id=None,
            ecg_waveform_length=int(getattr(pretrained_config, "ecg_waveform_length", 2500)),
            ecg_num_leads=int(getattr(pretrained_config, "ecg_num_leads", 12)),
            default_generation_kwargs=getattr(pretrained_config, "default_generation_kwargs", None),
            num_codebooks_kept=num_codebooks_kept,
            codebook_offset=codebook_offset,
        )

        model._load_state_dict(checkpoint_state_dict, strict=False)
        return model, tokenizer, pretrained_config

    def _setup_training_objects(self) -> dict[str, Any]:
        checkpoint_path = self.config.pretrained_tokenizer_path

        policy_model, tokenizer, checkpoint_config = self._load_model_from_checkpoint(checkpoint_path)
        ref_model, _, _ = self._load_model_from_checkpoint(checkpoint_path)

        # configure LoRA mode
        try:
            if bool(getattr(checkpoint_config, "use_lora", False)):
                policy_model.set_lora_inference_mode(False)
                ref_model.set_lora_inference_mode(True)
        except Exception:
            pass

        # set trainable params
        for _, param in policy_model.named_parameters():
            param.requires_grad = False
        trainable_mode = str(getattr(self.config, "trainable", "lora")).lower()
        regex = getattr(self.config, "trainable_regex", None)
        if trainable_mode == "all":
            for _, param in policy_model.named_parameters():
                param.requires_grad = True
        elif trainable_mode == "lora":
            for name, param in policy_model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
        elif trainable_mode == "bridge":
            for name, param in policy_model.named_parameters():
                if "bridge" in name or "qformer" in name:
                    param.requires_grad = True
        elif trainable_mode == "projection":
            for name, param in policy_model.named_parameters():
                if "projection" in name or "proj" in name:
                    param.requires_grad = True
        elif regex:
            import re as _re
            pattern = _re.compile(regex)
            for name, param in policy_model.named_parameters():
                if pattern.search(name):
                    param.requires_grad = True
        else:
            raise ValueError(f"Unknown trainable mode: {trainable_mode}")

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

    def _setup_inference_objects(self) -> dict[str, Any]:
        raise NotImplementedError("DPO inference not implemented.")

    def _setup_validation_objects(self) -> dict[str, Any]:
        raise NotImplementedError("DPO validation not implemented.")

    def _setup_test_objects(self) -> dict[str, Any]:
        raise NotImplementedError("DPO testing not implemented.")
