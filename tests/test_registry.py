import pytest
from typing import Any

from utils.registry import (
    BaseRegistry,
    ConfigRegistry,
    ModelRegistry,
    RunnerRegistry,
    ProjectRegistry,
    MetricRegistry
)


class TestBaseRegistry:
    """Tests for the BaseRegistry class"""

    def setup_method(self):
        """Setup method to reset the registry before each test"""
        # Create a new registry class for testing to avoid side effects
        class TestRegistry(BaseRegistry):
            _registry = {}
            _registry_type = "test"
        
        self.TestRegistry = TestRegistry

    def test_register_decorator(self):
        """Test that the register decorator correctly registers a class"""
        @self.TestRegistry.register("test_class")
        class TestClass:
            pass
        
        assert "test_class" in self.TestRegistry._registry
        assert self.TestRegistry._registry["test_class"] == TestClass

    def test_get_existing_class(self):
        """Test that get returns the correct class when it exists"""
        @self.TestRegistry.register("test_class")
        class TestClass:
            pass
        
        assert self.TestRegistry.get("test_class") == TestClass

    def test_get_nonexistent_class(self):
        """Test that get raises an error when the class doesn't exist"""
        with pytest.raises(ValueError, match="test nonexistent not found in registry"):
            self.TestRegistry.get("nonexistent")

    def test_list_registered(self):
        """Test that list_registered returns all registered classes"""
        @self.TestRegistry.register("test_class1")
        class TestClass1:
            pass
            
        @self.TestRegistry.register("test_class2")
        class TestClass2:
            pass
        
        registered = self.TestRegistry.list_registered()
        assert "test_class1" in registered
        assert "test_class2" in registered
        assert registered["test_class1"] == TestClass1
        assert registered["test_class2"] == TestClass2

    def test_create_instance(self):
        """Test that create correctly creates an instance of a registered class"""
        @self.TestRegistry.register("test_class")
        class TestClass:
            def __init__(self, value):
                self.value = value
        
        instance = self.TestRegistry.create("test_class", value="test_value")
        assert isinstance(instance, TestClass)
        assert instance.value == "test_value"

    def test_create_nonexistent_class(self):
        """Test that create raises an error when the class doesn't exist"""
        with pytest.raises(ValueError, match="test nonexistent not found in registry"):
            self.TestRegistry.create("nonexistent")


class TestRegistrySubclasses:
    """Tests for the registry subclasses"""

    def test_registry_subclasses_types(self):
        """Test that each registry subclass has the correct registry_type"""
        assert ConfigRegistry._registry_type == "config"
        assert ModelRegistry._registry_type == "model"
        assert RunnerRegistry._registry_type == "runner"
        assert ProjectRegistry._registry_type == "project"
        assert MetricRegistry._registry_type == "metric"

    def test_registry_subclasses_separate(self):
        """Test that each registry subclass has its own separate registry"""
        # Register a test class in each registry
        @ConfigRegistry.register("test")
        class TestConfig:
            pass
            
        @ModelRegistry.register("test")
        class TestModel:
            pass
            
        @RunnerRegistry.register("test")
        class TestRunner:
            pass
            
        @ProjectRegistry.register("test")
        class TestProject:
            pass
            
        @MetricRegistry.register("test")
        class TestMetric:
            pass
        
        # Check that each registry contains only its own class
        assert ConfigRegistry.get("test") == TestConfig
        assert ModelRegistry.get("test") == TestModel
        assert RunnerRegistry.get("test") == TestRunner
        assert ProjectRegistry.get("test") == TestProject
        assert MetricRegistry.get("test") == TestMetric 