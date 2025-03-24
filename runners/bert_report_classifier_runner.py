import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.enums import RunMode
from utils.registry import RunnerRegistry
from utils.files_handler import (
    save_json, 
    save_to_csv
)
from utils.wandb_wrapper import WandbWrapper
from utils.config.bert_classifier_config import BertReportClassifierConfig
from utils.metrics.ecg_metrics import compute_metrics
from utils.constants import ECG_PATTERNS, BERT_THRESHOLDS

from models.bert_classifier import BertClassifier


@RunnerRegistry.register("BERT_Report_Classifier")
class BertReportClassifierRunner:
    def __init__(
        self, 
        model: BertClassifier, 
        config: BertReportClassifierConfig, 
        val_dataloader: DataLoader,
        wandb_wrapper: WandbWrapper | None = None,
    ):
        self.model: BertClassifier = model
        self.config: BertReportClassifierConfig = config
        self.val_dataloader: DataLoader = val_dataloader
        self.wandb_wrapper: WandbWrapper = wandb_wrapper
        
    def execute(
        self, 
        mode: RunMode
    ):
        if mode == RunMode.TRAIN:
            self.train()
        elif mode == RunMode.INFERENCE:
            self.inference()
        elif mode == RunMode.VALIDATE:
            self.validate()
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
    def train(self):
        raise NotImplementedError("train not implemented")
    
    def _val_step(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        
        return self.model(
            input_ids=input_ids.to(self.config.device),
            attention_mask=attention_mask.to(self.config.device),
            token_type_ids=token_type_ids.to(self.config.device)
        )
    
    def inference(self):
        self.model.eval()
        
        ground_truth_classes: list[torch.Tensor] = []
        predicted_classes: list[torch.Tensor] = []
        for batch in tqdm(self.val_dataloader, desc="Inference", total=len(self.val_dataloader), disable=not self.config.is_ref_device):            
            # Get predicted logits
            predicted_logits: dict[str, torch.Tensor] = self._val_step(
                **batch['encoded_predicted_report']
            )
        
            # Get predicted probabilities
            predicted_probs: torch.Tensor = torch.sigmoid(predicted_logits['logits'])            
            
            # Get reference logits
            reference_logits: dict[str, torch.Tensor] = self._val_step(
                **batch['encoded_reference_report']
            )
            
            # Get reference probabilities
            reference_probs: torch.Tensor = torch.sigmoid(reference_logits['logits'])
            
            # Create thresholds tensor - need to do it here because of the batch size
            current_batch_size: int = len(batch['encoded_reference_report']['input_ids'])
            bert_thresholds_tensor = torch.zeros((current_batch_size, self.config.num_classes)).to(self.config.device)
            for i, pattern in enumerate(ECG_PATTERNS):
                bert_thresholds_tensor[:, i] = BERT_THRESHOLDS[pattern]['threshold']
            
            # Get reference classes
            reference_classes: torch.Tensor = torch.where(reference_probs >= bert_thresholds_tensor, 1, 0)

            # Store predicted and ground truth classes
            for i in range(current_batch_size):
                predicted_classes.append(predicted_probs[i].detach().cpu().numpy())
                ground_truth_classes.append(reference_classes[i].detach().cpu().numpy())
            
        metrics = compute_metrics(
            df_gt=pd.DataFrame(ground_truth_classes, columns=ECG_PATTERNS), 
            df_pred=pd.DataFrame(predicted_classes, columns=ECG_PATTERNS)
        )
                
        save_json(
            data=metrics, 
            path=os.path.join(
                self.config.output_folder, 
                f'{self.config.pipeline_name}.json'
            )
        )
        
        save_to_csv(
            metrics=metrics, 
            path=os.path.join(
                self.config.output_folder, 
                f'{self.config.pipeline_name}.csv'
            )
        )
            
    def validate(self):
        raise NotImplementedError("validate not implemented")

    