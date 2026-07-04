from enum import Enum

class RunMode(str, Enum):
    """Enum for different run modes of the training script."""
    TEST = "test"
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
    ECG_TOKENIZER_DPO_FINETUNING = "ECG_Tokenizer_DPO_Finetuning"
    ECG_TOKENIZER_GRPO_FINETUNING = "ECG_Tokenizer_GRPO_Finetuning"
    SIGLIP_PHASE1 = "SigLIP_Phase1"
    ECG_TEXT_STAGE1 = "ECG_Text_Stage1"

    def __str__(self):
        return self.value

class RunnerName(str, Enum):
    """Enum for different runner names."""
    LLM_FINETUNING = "ECG_Tokenizer_LLM_Finetuning"
    BERT_REPORT_CLASSIFIER = "Bert_Report_Classifier"
    ECG_TOKENIZER_TRAINING = "ECG_Tokenizer_Training_Runner"
    ECG_TOKENIZER_LINEAR_PROBING = "ECG_Tokenizer_Linear_Probing_Runner"
    DPO_FINETUNING = "ECG_Tokenizer_DPO_Finetuning"
    GRPO_FINETUNING = "ECG_Tokenizer_GRPO_Finetuning"
    SIGLIP_PHASE1 = "SigLIP_Phase1_Runner"
    ECG_TEXT_STAGE1 = "ECG_Text_Stage1_Runner"

    def __str__(self):
        return self.value

class ConfigName(str, Enum):
    """Enum for different config names."""
    BERT_REPORT_CLASSIFIER = "BERT_Report_Classifier"
    ECG_TOKENIZER_TRAINING = "ECG_Tokenizer_Training"
    ECG_TOKENIZER_LINEAR_PROBING = "ECG_Tokenizer_Linear_Probing"
    ECG_TOKENIZER_LLM_FINETUNING = "ECG_Tokenizer_LLM_Finetuning"
    ECG_TOKENIZER_DPO_FINETUNING = "ECG_Tokenizer_DPO_Finetuning"
    ECG_TOKENIZER_GRPO_FINETUNING = "ECG_Tokenizer_GRPO_Finetuning"
    SIGLIP_PHASE1 = "SigLIP_Phase1"
    ECG_TEXT_STAGE1 = "ECG_Text_Stage1"
    
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
    MEDGEMMA_DECODER = "MedGemma_Decoder"
    QWEN_DECODER = "Qwen_Decoder"
    CONV_DECODER = "Conv_Decoder"
    LINEAR_CLASSIFIER_DECODER = "Linear_Classifier_Decoder"
    RESNET_CLASSIFIER_DECODER = "ResNet_Classifier_Decoder"
    CLS_TOKEN_CLASSIFIER_DECODER = "CLS_Token_Classifier_Decoder"
    EFFICIENTNETV2_CLASSIFIER_DECODER = "EfficientNetV2_Classifier_Decoder"
    
    # Report classifiers
    BERT_REPORT_CLASSIFIER = "BERT_Report_Classifier"
        
    def __str__(self):
        return self.value   
    
class BridgeName(str, Enum):
    """Enum for different bridge names."""
    GPT2_LINEAR_BRIDGE = "GPT2_LinearBridge"
    GPT2_SEQUENCE_BRIDGE = "GPT2_SequenceBridge"
    GPT2_EMBEDDING_BRIDGE = "GPT2_EmbeddingBridge"
    GPT2_SIMPLE_EMBEDDING_BRIDGE = "GPT2_SimpleEmbeddingBridge"
    GPT2_SEQUENCE_TOKEN_BRIDGE = "GPT2_SequenceTokenBridge"
    GPT2_SIMPLE_TOKEN_BRIDGE = "GPT2_SimpleTokenBridge"
    LLAMA32_SEQUENCE_BRIDGE = "Llama32_SequenceBridge"
    LLAMA32_EMBEDDING_BRIDGE = "Llama32_EmbeddingBridge"
    LLAMA32_SIMPLE_EMBEDDING_BRIDGE = "Llama32_SimpleEmbeddingBridge"
    LLAMA32_SEQUENCE_TOKEN_BRIDGE = "Llama32_SequenceTokenBridge"
    LLAMA32_SIMPLE_TOKEN_BRIDGE = "Llama32_SimpleTokenBridge"
    LLAMA32_ECG_CODE_BRIDGE = "Llama32_ECGCodeBridge"
    LLAMA32_ECG_PROJECTION_BRIDGE = "Llama32_ECGProjectionBridge"
    LLAMA32_ECG_QFORMER_BRIDGE = "Llama32_ECGQFormerBridge"
    ECG_STAGE1_QFORMER_BRIDGE = "ECGQFormerBridgeStage1"
    INSTRUCTION_AWARE_ECG_QFORMER_BRIDGE = "InstructionAwareECGQFormerBridge"
    ECG_PERCEIVER_BRIDGE = "ECGPerceiverBridge"
    CROSS_MODAL_SEQUENCE_TOKEN_BRIDGE = "CrossModalSequenceTokenBridge"

    def __str__(self):
        return self.value
