from typing import TypeVar, Generic, Type, Dict

T = TypeVar('T')

class BaseRegistry(Generic[T]):
    """Base registry class that implements common registry functionality."""
    _registry: Dict[str, Type[T]] = {}
    _registry_type: str = "base"  # Should be overridden by subclasses
    
    @classmethod
    def register(cls, name: str):
        """Register a class in the registry."""
        def decorator(class_type: Type[T]) -> Type[T]:
            cls._registry[name] = class_type
            return class_type
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[T]:
        """Get a registered class by name."""
        if name not in cls._registry:
            raise ValueError(
                f"{cls._registry_type} {name} not found in registry. "
                f"Available {cls._registry_type}s: {list(cls._registry.keys())}"
            )
        return cls._registry[name]
    
    @classmethod
    def list_registered(cls) -> Dict[str, Type[T]]:
        """List all registered classes."""
        return cls._registry.copy()

    @classmethod
    def create(cls, name: str, **kwargs) -> T:
        """Create an instance of a registered class by name."""
        if name not in cls._registry:
            raise ValueError(
                f"{cls._registry_type} {name} not found in registry. "
                f"Available {cls._registry_type}s: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)


class RunnerRegistry(BaseRegistry[T]):
    """Registry for runners."""
    _registry: Dict[str, Type[T]] = {}
    _registry_type: str = "runner"

class ModelRegistry(BaseRegistry[T]):
    """Registry for models."""
    _registry: Dict[str, Type[T]] = {}
    _registry_type: str = "model"

class ProjectRegistry(BaseRegistry[T]):
    """Registry for projects."""
    _registry: Dict[str, Type[T]] = {}
    _registry_type: str = "project"

class ConfigRegistry(BaseRegistry[T]):
    """Registry for configs."""
    _registry: Dict[str, Type[T]] = {}
    _registry_type: str = "config"

class MetricRegistry(BaseRegistry[T]):
    """Registry for metrics."""
    _registry: Dict[str, Type[T]] = {}
    _registry_type: str = "metric"
