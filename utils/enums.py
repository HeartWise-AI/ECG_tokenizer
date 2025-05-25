from enum import Enum

class RunMode(str, Enum):
    """Enum for different run modes of the training script."""
    TRAIN = "train"
    VALIDATE = "validate"
    INFERENCE = "inference"
    EXTRACT_EMBEDDINGS = "extract_embeddings"
    
    def __str__(self):
        return self.value

class DatasetType(str, Enum):
    """Enum for different dataset types."""
    MHI = "mhi"
    MIMIC = "mimic"
    NONE = "none"

    def __str__(self):
        return self.value

class DecoderMode(str, Enum):
    """Enum for different decoder modes in the tokenizer."""
    RECONSTRUCTION = "reconstruction"  # For reconstructing the signal
    CLASSIFICATION = "classification"  # For classifying into independent classes
    EMBEDDING = "embedding"           # For generating embeddings without reconstruction

    def __str__(self):
        return self.value