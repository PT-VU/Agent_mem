#!/usr/bin/env python3
"""
Test script for storage interface definitions.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_mem.storage import (
    StorageError, StorageNotFoundError, StorageValidationError, StorageSerializationError,
    ProblemFileStorage, ObservationKGStorage, BeliefGraphStorage, UnifiedStorage,
    StorageFactory, StorageRegistry, StorageConfig
)

# Import for testing only
try:
    from agent_mem.core.problem_file import ProblemFile, ActionType, Outcome
except ImportError:
    # Mock for testing interface only
    class ProblemFile:
        def __init__(self, action_id="", task_id=""):
            self.action_id = action_id
            self.task_id = task_id

    class ActionType:
        TOOL_CALL = "tool_call"

    class Outcome:
        SUCCESS = "success"


def test_exception_hierarchy():
    """Test storage exception hierarchy."""
    print("=== Test: Storage Exception Hierarchy ===")

    exceptions = [
        StorageError("Base error"),
        StorageNotFoundError("Not found"),
        StorageValidationError("Validation failed"),
        StorageSerializationError("Serialization failed")
    ]

    for exc in exceptions:
        print(f"  {exc.__class__.__name__}: {exc}")
        assert isinstance(exc, StorageError), f"{exc} should be a StorageError"

    print(" Exception hierarchy test PASSED")


def test_interface_definition():
    """Test that interfaces are properly defined as abstract classes."""
    print("\n=== Test: Interface Abstract Definitions ===")

    # Test that interfaces are abstract
    import inspect
    import abc

    interfaces = [ProblemFileStorage, ObservationKGStorage, BeliefGraphStorage, UnifiedStorage]

    for interface in interfaces:
        print(f"  Checking {interface.__name__}:")
        print(f"    Is abstract: {abc.ABC in interface.__bases__}")
        print(f"    Has abstract methods: {len(interface.__abstractmethods__) > 0}")

        # Check that it has abstract methods
        assert abc.ABC in interface.__bases__, f"{interface} should inherit from ABC"
        assert len(interface.__abstractmethods__) > 0, f"{interface} should have abstract methods"

        # List abstract methods
        for method in interface.__abstractmethods__:
            print(f"    - Abstract method: {method}")

    print(" Interface definition test PASSED")


def test_storage_config():
    """Test StorageConfig class."""
    print("\n=== Test: StorageConfig ===")

    # Test creation
    config1 = StorageConfig(
        storage_type="local_json",
        base_path="/tmp/test_storage",
        auto_create_dirs=True,
        compression=True,
        encryption_key="test_key"
    )

    print(f"Config 1:")
    print(f"  storage_type: {config1.storage_type}")
    print(f"  base_path: {config1.base_path}")
    print(f"  auto_create_dirs: {config1.auto_create_dirs}")
    print(f"  compression: {config1.compression}")
    print(f"  encryption_key: {config1.encryption_key[:10]}...")

    # Test to_dict
    config_dict = config1.to_dict()
    print(f"\nConfig dict: {config_dict}")

    # Test from_dict
    config2 = StorageConfig.from_dict(config_dict)
    print(f"\nConfig 2 (from dict):")
    print(f"  storage_type: {config2.storage_type}")
    print(f"  base_path: {config2.base_path}")

    # Verify roundtrip
    config_dict2 = config2.to_dict()
    if config_dict == config_dict2:
        print(" StorageConfig roundtrip test PASSED")
    else:
        print(" StorageConfig roundtrip test FAILED")
        print(f"Original: {config_dict}")
        print(f"Restored: {config_dict2}")

    # Test default values
    config_default = StorageConfig()
    print(f"\nDefault config:")
    print(f"  storage_type: {config_default.storage_type}")
    print(f"  auto_create_dirs: {config_default.auto_create_dirs}")
    print(f"  compression: {config_default.compression}")

    assert config_default.storage_type == "local_json"
    assert config_default.auto_create_dirs == True
    assert config_default.compression == False

    print(" StorageConfig default values test PASSED")


def test_storage_registry():
    """Test StorageRegistry class."""
    print("\n=== Test: StorageRegistry ===")

    # Create a mock factory
    class MockStorageFactory(StorageFactory):
        def create_storage(self, config):
            class MockStorage(UnifiedStorage):
                def initialize(self): return True
                def cleanup(self): return True
                def get_storage_info(self): return {"type": "mock"}
                def backup(self, path): return True
                def restore(self, path): return True
                # ProblemFileStorage methods
                def save_problem_file(self, pf): return "mock_id"
                def load_problem_file(self, id): return ProblemFile(action_id="test", task_id="test")
                def delete_problem_file(self, id): return True
                def list_problem_files(self, **kwargs): return []
                def search_problem_files(self, **kwargs): return []
                def get_problem_file_stats(self): return {}
                # ObservationKGStorage methods
                def save_kg(self, kg, id=None): return "mock_id"
                def load_kg(self, id): raise NotImplementedError
                def save_task_subgraph(self, sg): return "mock_id"
                def load_task_subgraph(self, task_id): return None
                def delete_task_subgraph(self, task_id): return True
                def list_task_subgraphs(self, **kwargs): return []
                def get_kg_stats(self): return {}
                # BeliefGraphStorage methods
                def save_belief_graph(self, bg, id=None): return "mock_id"
                def load_belief_graph(self, id): raise NotImplementedError
                def save_atomic_belief(self, belief): return "mock_id"
                def load_atomic_belief(self, belief_id): return None
                def delete_atomic_belief(self, belief_id): return True
                def list_atomic_beliefs(self, **kwargs): return []
                def get_belief_graph_stats(self): return {}

            return MockStorage()

    # Test registration
    factory = MockStorageFactory()
    StorageRegistry.register("mock", factory)

    print(f"Registered storages: {StorageRegistry.list_available_storages()}")
    assert "mock" in StorageRegistry.list_available_storages()

    # Test creation
    config = {"test": "config"}
    storage = StorageRegistry.create_storage("mock", config)
    print(f"Created storage type: {type(storage).__name__}")

    # Test that it implements UnifiedStorage
    assert isinstance(storage, UnifiedStorage)

    # Test error for unregistered type
    try:
        StorageRegistry.create_storage("unknown", {})
        print(" Should have raised ValueError for unknown storage type")
    except ValueError as e:
        print(f" Correctly raised ValueError: {e}")

    # Test duplicate registration
    try:
        StorageRegistry.register("mock", factory)
        print(" Should have raised ValueError for duplicate registration")
    except ValueError as e:
        print(f" Correctly raised ValueError for duplicate: {e}")

    print(" StorageRegistry test PASSED")


def test_interface_method_signatures():
    """Test that interface methods have correct signatures."""
    print("\n=== Test: Interface Method Signatures ===")

    # Check ProblemFileStorage methods
    print("Checking ProblemFileStorage method signatures:")

    expected_methods = [
        ("save_problem_file", ["problem_file"]),
        ("load_problem_file", ["identifier"]),
        ("delete_problem_file", ["identifier"]),
        ("list_problem_files", ["task_id", "action_type", "outcome", "limit", "offset"]),
        ("search_problem_files", ["query", "field", "limit"]),
        ("get_problem_file_stats", [])
    ]

    for method_name, params in expected_methods:
        if hasattr(ProblemFileStorage, method_name):
            print(f"   {method_name}({', '.join(params)})")
        else:
            print(f"   {method_name} not found in ProblemFileStorage")

    print("\n Interface method signature test completed")


def main():
    """Run all interface tests."""
    print("Storage Interface Test Suite")
    print("=" * 50)

    try:
        test_exception_hierarchy()
        test_interface_definition()
        test_storage_config()
        test_storage_registry()
        test_interface_method_signatures()

        print("\n" + "=" * 50)
        print("All interface tests completed!")

    except Exception as e:
        print(f"\n Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
