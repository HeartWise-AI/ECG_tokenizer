from typing import Protocol, Type, Union, TYPE_CHECKING
from utils.enums import RunMode

if TYPE_CHECKING:
    from runners import (
        BaseRunner,
        ECGTokenizerRunner,
        LLMFinetuningRunner, 
        BertReportClassifierRunner
    )

class Runner(Protocol):
    """Protocol defining the interface for all runners."""
    
    def execute(self, mode: RunMode) -> None:
        """Execute the runner with the given mode."""
        ...

# Protocol-based types (for type safety)
RunnerT = Runner
RunnerClassT = Type[Runner]

# Union-based types (for IDE navigation to concrete classes)
RunnerUnionT = Union[
    "BaseRunner",
    "ECGTokenizerRunner", 
    "LLMFinetuningRunner",
    "BertReportClassifierRunner"
]
RunnerClassUnionT = Type[RunnerUnionT]