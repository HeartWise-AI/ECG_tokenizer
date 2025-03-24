
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification

from utils.registry import ModelRegistry

@ModelRegistry.register("BERT_Report_Classifier")
class BertClassifier:
    def __init__(
        self, 
        model_path: str, 
        num_classes: int, 
        map_location: str = "cpu"
    ):
        print(f"Loading model from {model_path}")
        self.model: BertForSequenceClassification = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_classes,
        ).to(map_location)

    def preprocessing(self, text: str) -> dict:
        return self.processor(
            text,
            padding='max_length', 
            max_length=512, 
            truncation=True,
            return_tensors='pt', 
        )

    def to(self, device: str):
        self.model.to(device)

    def __call__(
        self,  
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask
        )['logits']