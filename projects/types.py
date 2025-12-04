from typing import Union, Type, Protocol, Any, TYPE_CHECKING
from utils.enums import RunMode
from utils.wandb_wrapper import WandbWrapper

if TYPE_CHECKING:
    from projects import (
        LLMFinetuningProject, 
        ECGTokenizerLinearProbing,
        BertReportClassifierProject, 
        ECGTokenizerTrainingProject
    )

class ProjectProtocol(Protocol):
    """Protocol defining the interface for all projects."""
    
    def run(self) -> None:
        """Run the project."""
        ...
    
    def __init__(self, config: Any, wandb_wrapper: WandbWrapper) -> None:
        """Initialize the project."""
        ...

# Protocol-based types (for type safety)
ProjectT = ProjectProtocol
ProjectClassT = Type[ProjectProtocol]

# Union-based types (for IDE navigation to concrete classes)
ProjectUnionT = Union[
    "LLMFinetuningProject", 
    "ECGTokenizerLinearProbing",
    "BertReportClassifierProject", 
    "ECGTokenizerTrainingProject"
]
ProjectClassUnionT = Type[ProjectUnionT]