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
    LLM = "llm"                       # For generating text with LLM
    EMBEDDING = "embedding"           # For generating embeddings without reconstruction

    def __str__(self):
        return self.value
    
class ProjectName(str, Enum):
    """Enum for different project names."""
    BERT_REPORT_CLASSIFIER = "BERT_Report_Classifier"
    ECG_TOKENIZER_TRAINING = "ECG_Tokenizer_Training"
    ECG_TOKENIZER_LINEAR_PROBING = "ECG_Tokenizer_Linear_Probing"
    ECG_TOKENIZER_LLM_FINETUNING = "ECG_Tokenizer_LLM_Finetuning"
    
    def __str__(self):
        return self.value

class RunnerName(str, Enum):
    """Enum for different runner names."""
    LLM_FINETUNING = "ECG_Tokenizer_LLM_Finetuning"
    BERT_REPORT_CLASSIFIER = "Bert_Report_Classifier"
    ECG_TOKENIZER_TRAINING = "ECG_Tokenizer_Training_Runner"
    ECG_TOKENIZER_LINEAR_PROBING = "ECG_Tokenizer_Linear_Probing_Runner"
    
    def __str__(self):
        return self.value
    
class ConfigName(str, Enum):
    """Enum for different config names."""
    BERT_REPORT_CLASSIFIER = "BERT_Report_Classifier"
    ECG_TOKENIZER_TRAINING = "ECG_Tokenizer_Training"
    ECG_TOKENIZER_LINEAR_PROBING = "ECG_Tokenizer_Linear_Probing"
    ECG_TOKENIZER_LLM_FINETUNING = "ECG_Tokenizer_LLM_Finetuning"    
    
    def __str__(self):
        return self.value
    
class ModelName(str, Enum):
    """Enum for different model names."""
    # High-level tokenizers
    ECG_TOKENIZER_WRAPPER = "ECG_Tokenizer_Wrapper"
    ECG_TOKENIZER_LINEAR_PROBING = "ECG_Tokenizer_Linear_Probing"
    ECG_TOKENIZER_LLM_FINETUNING = "ECG_Tokenizer_LLM_Finetuning"
    LLAMA32_TOKENIZER_WRAPPER = "Llama32_Tokenizer_Wrapper"
    
    # Encoders
    CONV_ENCODER = "Conv_Encoder"
    RESIDUAL_CONV_ENCODER = "Residual_Conv_Encoder"
    
    # Quantizers
    ECG_TOKENIZER_QUANTIZER = "ECG_Tokenizer_Quantizer"
    ECG_TOKENIZER_QUANTIZER_RVQ = "ECG_Tokenizer_Quantizer_RVQ"
    ECG_TOKENIZER_QUANTIZER_VANILLA = "ECG_Tokenizer_Quantizer_Vanilla"

    # Decoders
    GPT2_DECODER = "GPT2_Decoder"
    LLAMA32_DECODER = "Llama32_Decoder"
    CONV_DECODER = "Conv_Decoder"
    LINEAR_CLASSIFIER_DECODER = "Linear_Classifier_Decoder"
    RESNET_CLASSIFIER_DECODER = "ResNet_Classifier_Decoder"
    CLS_TOKEN_CLASSIFIER_DECODER = "CLS_Token_Classifier_Decoder"
    EFFICIENTNETV2_CLASSIFIER_DECODER = "EfficientNetV2_Classifier_Decoder"
    
    # Report classifiers
    BERT_REPORT_CLASSIFIER = "BERT_Report_Classifier"
        
    def __str__(self):
        return self.value   
    
class AdapterName(str, Enum):
    """Enum for different adapter names."""
    GPT2_LINEAR_ADAPTER = "GPT2_LinearAdapter"
    GPT2_SEQUENCE_ADAPTER = "GPT2_SequenceAdapter"
    GPT2_EMBEDDING_ADAPTER = "GPT2_EmbeddingAdapter"
    GPT2_SIMPLE_EMBEDDING_ADAPTER = "GPT2_SimpleEmbeddingAdapter"
    LLAMA32_SEQUENCE_ADAPTER = "Llama32_SequenceAdapter"
    LLAMA32_EMBEDDING_ADAPTER = "Llama32_EmbeddingAdapter"
    LLAMA32_SIMPLE_EMBEDDING_ADAPTER = "Llama32_SimpleEmbeddingAdapter"
    
    def __str__(self):
        return self.value