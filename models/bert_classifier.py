import torch
import torch.nn as nn
from transformers import (
    BertForSequenceClassification, 
    PreTrainedModel, 
    BertTokenizer,
    BatchEncoding
)

from utils.enums import ModelName
from utils.registry import ModelRegistry

@ModelRegistry.register(ModelName.BERT_REPORT_CLASSIFIER)
class BertClassifier(nn.Module):
    """
    BERT-based report classifier.
    """
    def __init__(
        self, 
        model_path: str, 
        num_classes: int, 
    ):
        """
        Args:
            model_path: Path to the BERT model
            num_classes: Number of classes for classification
        """
        super().__init__()
        
        print(f"Loading model from {model_path}")
        self.model: PreTrainedModel = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_classes,
        )

        self.processor: BertTokenizer = BertTokenizer.from_pretrained(model_path)

    def preprocessing(self, text: str) -> BatchEncoding:
        """
        Args:
            text: Text to preprocess

        Returns:
            BatchEncoding: Preprocessed text
        """
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
        """
        Args:
            input_ids: Input IDs for the BERT model
            attention_mask: Attention mask for the BERT model
            token_type_ids: Token type IDs for the BERT model

        Returns:
            dict[str, torch.Tensor]: Output from the BERT model
        """
        return self.model(
            input_ids=input_ids.squeeze(), 
            token_type_ids=token_type_ids.squeeze(), 
            attention_mask=attention_mask.squeeze() 
        )