from __future__ import annotations

import os
import time
import re
from collections import Counter
import warnings
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.serialization as serialization
import torch.distributed as dist

from models.ecg_tokenizer_wrapper import Conv_Encoder, ECG_Tokenizer_Quantizer
from data.ecg_text_stage1_dataset import ECGTextStage1Dataset, get_stage1_dataloader
from projects.base_project import BaseProject
from runners.ecg_text_stage1_runner import ECGTextStage1Runner
from utils.config.ecg_text_stage1_config import ECGTextStage1Config
from utils.config.tokenizer_config import ECGTokenizerTrainingConfig
from utils.debug import ensure_dir
from utils.enums import ProjectName
from utils.registry import ModelRegistry, ProjectRegistry

serialization.add_safe_globals([ECGTokenizerTrainingConfig])


def select_tail_class_ids(
    pos_counts: dict[str, int],
    all_text_ids: List[str],
    *,
    mode: str = "topN",
    top_n: int = 50,
    min_positives: int = 10,
    include_regex: List[str] | None = None,
    exclude_regex: List[str] | None = None,
) -> List[str]:
    if not pos_counts:
        return []

    compiled_include = [re.compile(pattern) for pattern in include_regex or []]
    compiled_exclude = [re.compile(pattern) for pattern in exclude_regex or []]

    def _eligible(text_id: str) -> bool:
        if compiled_exclude and any(pattern.search(text_id) for pattern in compiled_exclude):
            return False
        if compiled_include and not any(pattern.search(text_id) for pattern in compiled_include):
            return False
        return pos_counts.get(text_id, 0) > 0

    filtered = [text_id for text_id in all_text_ids if _eligible(text_id)]
    if not filtered:
        return []

    mode = (mode or "").lower()
    key_fn = lambda tid: pos_counts.get(tid, 0)
    if mode == "threshold":
        threshold = max(1, int(min_positives))
        selected = [text_id for text_id in filtered if pos_counts.get(text_id, 0) <= threshold]
        return sorted(selected, key=key_fn)

    filtered.sort(key=key_fn)
    return filtered[: max(0, int(top_n))]


@ProjectRegistry.register(ProjectName.ECG_TEXT_STAGE1)
class ECGTextStage1Project(BaseProject):
    """Project orchestrating Stage-1 ECG ↔ text alignment without an LLM."""

    def __init__(self, config: ECGTextStage1Config, wandb_wrapper) -> None:
        super().__init__(config, wandb_wrapper)
        self.config = config

    # ------------------------------------------------------------------
    def _setup_training_objects(self) -> Dict[str, Any]:  # type: ignore[override]
        device = torch.device("cuda", self.config.device) if torch.cuda.is_available() else torch.device("cpu")

        output_dir = self.config.output_dir
        distributed = dist.is_available() and dist.is_initialized()
        if not output_dir and getattr(self.config, "is_ref_device", False):
            run_id = None
            wandb_wrapper = getattr(self, "wandb_wrapper", None)
            if wandb_wrapper and getattr(wandb_wrapper, "is_initialized", lambda: False)():
                try:
                    run_id = wandb_wrapper.get_run_id()
                except Exception:
                    run_id = None
            if not run_id or run_id == "no_wandb":
                run_id = time.strftime("%Y%m%d-%H%M%S")
            output_dir = os.path.join(
                self.config.base_checkpoint_path,
                self.config.pipeline_project,
                "stage1_runs",
                str(run_id),
            )
            self.config.output_dir = output_dir

        if distributed and dist.get_world_size() > 1:
            shared_dir = output_dir if getattr(self.config, "is_ref_device", False) else ""
            shared_list = [shared_dir]
            dist.broadcast_object_list(shared_list, src=0)
            output_dir = shared_list[0]
            self.config.output_dir = output_dir

        if not output_dir:
            output_dir = os.path.join(
                self.config.base_checkpoint_path,
                self.config.pipeline_project,
                "stage1_runs",
                f"rank{self.config.device}",
            )
            self.config.output_dir = output_dir
        ensure_dir(output_dir)

        train_dataset = ECGTextStage1Dataset(
            mapping_csv=self.config.mapping_csv,
            text_bank_csv=self.config.text_bank_csv,
            split=self.config.mapping_split,
            expected_waveform_length=self.config.waveform_length,
            num_leads=self.config.num_leads,
            normalize_waveforms=self.config.normalize_waveforms,
            lead_stats=self.config.lead_stats,
            seed=self.config.seed,
            max_positives_per_ecg=int(getattr(self.config, "max_positives_per_ecg", 1) or 1),
        )

        tail_ids_cfg = getattr(self.config, "tail_class_ids", None)
        if tail_ids_cfg is not None:
            tails = [str(text_id) for text_id in tail_ids_cfg if text_id]
        else:
            pos_counts = Counter()
            for sample in train_dataset.samples:
                pos_counts.update(sample.positive_ids)
            if pos_counts:
                tails = select_tail_class_ids(
                    pos_counts=pos_counts,
                    all_text_ids=train_dataset.all_text_ids,
                    mode=str(getattr(self.config, "tail_select_mode", "topN") or "topN"),
                    top_n=int(getattr(self.config, "tail_top_n", 50) or 50),
                    min_positives=int(getattr(self.config, "tail_min_positives", 10) or 10),
                    include_regex=getattr(self.config, "tail_include_regex", None),
                    exclude_regex=getattr(self.config, "tail_exclude_regex", None),
                )
            else:
                tails = []
        self.config.tail_class_ids = tails

        train_loader = get_stage1_dataloader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            num_replicas=self.config.world_size,
            rank=self.config.device,
            shuffle=True,
        )

        encoder, quantizer, checkpoint_config = self._load_tokenizer_components(device=device)

        text_encoder_class = ModelRegistry.get("text_encoder")
        text_encoder = text_encoder_class(
            model_name=self.config.text_encoder_model_name,
            output_dim=int(self.config.text_encoder_output_dim),
            dropout=float(self.config.text_encoder_dropout),
            freeze_ratio=float(self.config.text_encoder_freeze_ratio),
        )
        if int(self.config.text_encoder_output_dim) != int(self.config.bridge_hidden_size):
            raise ValueError(
                "text_encoder_output_dim must match bridge_hidden_size for Stage-1 alignment."
            )
        tokenizer = text_encoder.tokenizer
        if tokenizer.pad_token_id is None:
            raise ValueError("Text encoder tokenizer must define a pad_token_id for Stage-1 training.")

        if "[DEC]" not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": ["[DEC]"]})
            resize_fn = getattr(text_encoder, "resize_token_embeddings", None)
            if callable(resize_fn):
                resize_fn(len(tokenizer))
        self.config.dec_token_id = int(tokenizer.convert_tokens_to_ids("[DEC]"))

        txt_vocab_size = int(len(tokenizer))
        txt_pad_id = int(tokenizer.pad_token_id)
        txt_cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id

        # Resolve backbone hidden size for information only; allow projection mismatch.
        try:
            cfg = getattr(text_encoder.bert, "config", None)
            bert_hidden_size = int(getattr(cfg, "hidden_size", None) or getattr(getattr(cfg, "text_config", None), "hidden_size", 0) or 0)
        except Exception:
            bert_hidden_size = 0
        if bert_hidden_size and int(self.config.bridge_hidden_size) != bert_hidden_size:
            warnings.warn(
                "bridge_hidden_size differs from text encoder hidden size; using internal projection to match. "
                f"bridge_hidden_size={self.config.bridge_hidden_size}, text_hidden_size={bert_hidden_size}"
            )

        # IMPORTANT: Do NOT share bert_layers between text_encoder and bridge!
        # Sharing weights causes corruption during backprop when both modules update the same parameters.
        # Instead, let the bridge initialize its own weights randomly.
        bert_layers = None  # Disable weight sharing to prevent NaN during training

        num_codebooks = self.config.num_codebooks_kept or self.config.num_quantizers
        if num_codebooks is None:
            raise ValueError(
                "Stage-1 bridge requires num_codebooks_kept or num_quantizers in the config/checkpoint."
            )

        vocab_size = self.config.codebook_size
        if vocab_size is None:
            vocab_size = checkpoint_config.get("codebook_size")
        if vocab_size is None:
            raise ValueError(
                "Stage-1 bridge requires codebook_size in the config or checkpoint metadata."
            )

        bridge_class = ModelRegistry.get(self.config.bridge_name)
        bridge = bridge_class(
            vocab_size=int(vocab_size),
            num_codebooks=int(num_codebooks),
            d_mid=int(self.config.bridge_hidden_size),
            d_llm=int(self.config.bridge_hidden_size),
            d_txt=int(self.config.bridge_hidden_size),
            num_steps=int(self.config.bridge_max_seq_len),
            num_query_tokens=int(self.config.num_query_tokens),
            num_layers=int(self.config.bridge_num_layers),
            num_heads=int(self.config.bridge_num_heads),
            dropout=float(self.config.bridge_dropout),
            num_special_tokens=int(self.config.bridge_num_special_tokens),
            bias_last_codebook=float(self.config.bridge_bias_last_codebook),
            codebook_dropout=float(self.config.bridge_codebook_dropout),
            mix_strategy=str(getattr(self.config, "bridge_mix_strategy", "softmax") or "softmax"),
            token_axis=str(getattr(self.config, "bridge_token_axis", "channel") or "channel"),
            txt_vocab_size=txt_vocab_size,
            txt_pad_id=txt_pad_id,
            txt_cls_id=txt_cls_id,
            cross_every=int(self.config.cross_every),
            bert_layers=bert_layers,
        ).to(device)

        if getattr(bridge, "token_axis", "channel") == "time":
            # time-axis kv rebuilds z from ids through the frozen quantizer decode
            # (x1_split uses implicit neural codebooks — static tables cannot reproduce z).
            rvq = getattr(quantizer, "quantizer", quantizer)
            bridge.attach_quantizer(rvq)
            print("[Stage1] bridge token_axis=time: frozen quantizer attached")

        optimizer = self._build_optimizer(bridge, text_encoder)

        val_loader = None
        if self.config.validation_mapping_csv and self.config.validation_mapping_split:
            val_dataset = ECGTextStage1Dataset(
                mapping_csv=self.config.validation_mapping_csv,
                text_bank_csv=self.config.validation_text_bank_csv or self.config.text_bank_csv,
                split=self.config.validation_mapping_split,
                expected_waveform_length=self.config.waveform_length,
                num_leads=self.config.num_leads,
                normalize_waveforms=self.config.normalize_waveforms,
                lead_stats=self.config.lead_stats,
                seed=self.config.seed,
                max_positives_per_ecg=int(getattr(self.config, "max_positives_per_ecg", 1) or 1),
            )

            val_loader = get_stage1_dataloader(
                dataset=val_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                num_replicas=self.config.world_size,
                rank=self.config.device,
                shuffle=False,
            )

        text_bank = self._compute_text_bank_embeddings(text_encoder)

        runner_kwargs = {
            "encoder": encoder,
            "quantizer": quantizer,
            "bridge": bridge,
            "text_encoder": text_encoder,
            "train_dataloader": train_loader,
            "optimizer": optimizer,
            "scheduler": None,
            "validation_dataloader": val_loader,
            "text_bank": text_bank,
        }
        return runner_kwargs

    # ------------------------------------------------------------------
    def _setup_inference_objects(self) -> Dict[str, Any]:  # type: ignore[override]
        raise NotImplementedError("Inference is not supported for Stage-1 project.")

    # ------------------------------------------------------------------
    def _setup_validation_objects(self) -> Dict[str, Any]:  # type: ignore[override]
        raise NotImplementedError("Validation mode is not supported for Stage-1 project.")

    # ------------------------------------------------------------------
    def _setup_test_objects(self) -> Dict[str, Any]:  # type: ignore[override]
        raise NotImplementedError("Test mode is not supported for Stage-1 project.")

    # ------------------------------------------------------------------
    def _setup_extraction_objects(self) -> Dict[str, Any]:  # type: ignore[override]
        raise NotImplementedError("Embedding extraction is not supported for Stage-1 project.")

    # ------------------------------------------------------------------
    def _compute_text_bank_embeddings(self, text_encoder: nn.Module) -> tuple[list[str], list[str], torch.Tensor]:
        import pandas as pd  # defer import to keep startup light

        csv_path = self.config.text_bank_csv
        if not csv_path or not os.path.exists(csv_path):
            return [], [], torch.empty(0, int(self.config.text_encoder_output_dim))

        text_df = pd.read_csv(csv_path)
        if text_df.empty:
            return [], [], torch.empty(0, int(self.config.text_encoder_output_dim))

        text_ids = text_df["text_id"].astype(str).tolist()
        texts = text_df["text"].astype(str).tolist()

        device = next(text_encoder.parameters()).device
        was_training = text_encoder.training
        text_encoder.eval()

        tokenizer = text_encoder.tokenizer
        embeddings: list[torch.Tensor] = []
        chunk_size = 512
        with torch.no_grad():
            for start in range(0, len(texts), chunk_size):
                end = min(start + chunk_size, len(texts))
                batch_texts = texts[start:end]
                tokens = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=int(self.config.max_text_length),
                    return_tensors="pt",
                )
                input_ids = tokens["input_ids"].to(device)
                attention_mask = tokens["attention_mask"].to(device)
                features = text_encoder(input_ids, attention_mask)
                features = F.normalize(features, dim=-1)
                embeddings.append(features.cpu())

        if was_training:
            text_encoder.train()

        if embeddings:
            embedding_tensor = torch.cat(embeddings, dim=0)
        else:
            embedding_tensor = torch.empty(0, int(self.config.text_encoder_output_dim))

        return text_ids, texts, embedding_tensor

    # ------------------------------------------------------------------
    def _build_optimizer(self, bridge: nn.Module, text_encoder: nn.Module) -> torch.optim.Optimizer:
        opt_cfg = self.config.optimizer
        default_lr = float(self.config.lr)
        default_wd = float(self.config.weight_decay)
        txt_lr_mult = float(getattr(self.config, "text_encoder_lr_mult", 0.0) or 0.0)
        txt_wd_mult = float(getattr(self.config, "text_encoder_wd_mult", 1.0) or 1.0)

        opt_type = "AdamW"
        extra_kwargs: Dict[str, Any] = {}

        if isinstance(opt_cfg, dict):
            opt_type = opt_cfg.get("type", opt_type)
            default_lr = float(opt_cfg.get("lr", default_lr))
            default_wd = float(opt_cfg.get("weight_decay", default_wd))
            extra_kwargs = {
                key: value
                for key, value in opt_cfg.items()
                if key not in {"type", "lr", "weight_decay"}
            }
        elif isinstance(opt_cfg, str) and opt_cfg:
            opt_type = opt_cfg

        bridge_params = [p for p in bridge.parameters() if p.requires_grad]
        text_params = [p for p in text_encoder.parameters() if p.requires_grad]

        param_groups: List[Dict[str, Any]] = []
        if bridge_params:
            param_groups.append({
                "params": bridge_params,
                "lr": default_lr,
                "weight_decay": default_wd,
            })
        if text_params and txt_lr_mult > 0.0:
            param_groups.append({
                "params": text_params,
                "lr": default_lr * txt_lr_mult,
                "weight_decay": default_wd * txt_wd_mult,
            })

        if not param_groups:
            raise ValueError("No trainable parameters were provided to the optimizer.")

        opt_type_lower = opt_type.lower()
        if opt_type_lower == "adamw":
            optimizer = optim.AdamW(
                param_groups,
                lr=default_lr,
                weight_decay=default_wd,
                **extra_kwargs,
            )
        elif opt_type_lower == "adam":
            optimizer = optim.Adam(
                param_groups,
                lr=default_lr,
                weight_decay=default_wd,
                **extra_kwargs,
            )
        else:
            raise ValueError(f"Unsupported optimizer type '{opt_type}' for Stage-1 project.")
        return optimizer

    # ------------------------------------------------------------------
    def _load_tokenizer_components(
        self,
        device: torch.device,
    ) -> Tuple[nn.Module, nn.Module, Dict[str, Any]]:
        checkpoint_path = self.config.pretrained_encoder_checkpoint
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Pretrained encoder checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
        ckpt_config = checkpoint.get("config")

        encoder_name = self._resolve_config_value("encoder_name", ckpt_config)
        if encoder_name is None:
            raise ValueError("Unable to determine encoder_name from config or checkpoint.")

        quantizer_name = self._resolve_config_value("quantizer_name", ckpt_config)
        num_quantizers = self._resolve_config_value("num_quantizers", ckpt_config)
        codebook_size = self._resolve_config_value("codebook_size", ckpt_config)

        if num_quantizers is not None:
            self.config.num_quantizers = int(num_quantizers)
        if codebook_size is not None:
            self.config.codebook_size = int(codebook_size)

        encoder_class = ModelRegistry.get(encoder_name)
        encoder: nn.Module = encoder_class().to(device)
        encoder.eval()

        encoder_state = self._extract_module_state(state_dict, "encoder")
        missing = encoder.load_state_dict(encoder_state, strict=False)
        if missing.missing_keys or missing.unexpected_keys:
            raise ValueError(
                "Encoder state_dict mismatch. Missing keys: "
                f"{missing.missing_keys}; Unexpected keys: {missing.unexpected_keys}"
            )

        for param in encoder.parameters():
            param.requires_grad = False

        if quantizer_name:
            if num_quantizers is None or codebook_size is None:
                raise ValueError(
                    "Quantizer specified but num_quantizers/codebook_size are missing."
                )
            quantizer_class = ModelRegistry.get(quantizer_name)
            quantizer: nn.Module = quantizer_class(
                num_quantizers=int(num_quantizers),
                codebook_size=int(codebook_size),
            ).to(device)
            quantizer.eval()
            quantizer_state = self._extract_module_state(state_dict, "quantizer")
            missing_q = quantizer.load_state_dict(quantizer_state, strict=False)
            if missing_q.missing_keys or missing_q.unexpected_keys:
                raise ValueError(
                    "Quantizer state_dict mismatch. Missing keys: "
                    f"{missing_q.missing_keys}; Unexpected keys: {missing_q.unexpected_keys}"
                )
            for param in quantizer.parameters():
                param.requires_grad = False
        else:
            raise ValueError("Stage-1 training requires a quantizer; quantizer_name was not found.")

        metadata = {
            "encoder_name": encoder_name,
            "quantizer_name": quantizer_name,
            "num_quantizers": num_quantizers,
            "codebook_size": codebook_size,
        }
        return encoder, quantizer, metadata

    # ------------------------------------------------------------------
    def _resolve_config_value(self, key: str, ckpt_config: Any, default: Any = None) -> Any:
        # 1. Check current config first
        if hasattr(self.config, key) and getattr(self.config, key) not in (None, ""):
            return getattr(self.config, key)

        # 2. Check checkpoint config (supports both dict and dataclass)
        if ckpt_config:
            if isinstance(ckpt_config, dict) and key in ckpt_config:
                return ckpt_config[key]
            elif hasattr(ckpt_config, key):  # Handle dataclass configs
                value = getattr(ckpt_config, key)
                if value not in (None, ""):
                    return value

        # 3. Fallback to default
        return default

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_module_state(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        filtered: Dict[str, torch.Tensor] = {}
        prefix_with_dot = f"{prefix}."
        for key, tensor in state_dict.items():
            if key.startswith(prefix_with_dot):
                filtered[key[len(prefix_with_dot):]] = tensor
        if not filtered:
            raise ValueError(f"Checkpoint does not contain parameters for module '{prefix}'.")
        return filtered
