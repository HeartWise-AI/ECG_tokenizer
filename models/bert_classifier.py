import torch
import torch.nn as nn
from transformers import (
    BertForSequenceClassification, 
    PreTrainedModel, 
    BertTokenizer,
    BatchEncoding
)

from utils.registry import ModelRegistry

@ModelRegistry.register("BERT_Report_Classifier")
class BertClassifier(nn.Module):
    def __init__(
        self, 
        model_path: str, 
        num_classes: int, 
    ):
        print(f"Loading model from {model_path}")
        self.model: PreTrainedModel = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_classes,
        )

        self.processor: BertTokenizer = BertTokenizer.from_pretrained(model_path)

    def preprocessing(self, text: str) -> BatchEncoding:
        return self.processor(
            text,
            padding='max_length', 
            max_length=512, 
            truncation=True,
            return_tensors='pt', 
        )
        
    def forward(
        self,  
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.model(
            input_ids=input_ids.squeeze(), 
            token_type_ids=token_type_ids.squeeze(), 
            attention_mask=attention_mask.squeeze() 
        )