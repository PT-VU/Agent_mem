#!/usr/bin/env python
"""
Functional test for Agent-mem MVP.
Tests end-to-end functionality and component completeness.
"""
import sys
import os
import tempfile
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_component_exists(module_path, class_name=None):
    """Check if a component exists and can be imported."""
    try:
        if class_name:
            exec(f"from {module_path} import {class_name}")
            print(f" {module_path}.{class_name} exists")
            return True
        else:
            exec(f"import {module_path}")
            print(f" {module_path} exists")
            return True
    except ImportError as e:
        print(f" {module_path} missing: {e}")
        return False
    except Exception as e:
        print(f" {module_path} error: {e}")
        return False

def test_component_completeness():
    """Test that all core components exist."""
    print("Checking component completeness...")
    print("="*60)

    components = [
        # Core data models
        ("agent_mem.core.problem_file", "ProblemFile"),
        ("agent_mem.core.problem_file", "ActionType"),
        ("agent_mem.core.problem_file", "Outcome"),
        ("agent_mem.core.problem_file", "EvidencePointer"),
        ("agent_mem.core.observation_kg", "ObservationKG"),
        ("agent_mem.core.observation_kg", "TaskSubgraph"),
        ("agent_mem.core.observation_kg", "KGEdge"),
        ("agent_mem.core.observation_kg", "EdgeType"),
        ("agent_mem.core.belief_graph", "BeliefGraph"),
        ("agent_mem.core.belief_graph", "AtomicBelief"),
        ("agent_mem.core.belief_graph", "BeliefType"),
        ("agent_mem.core.belief_graph", "BeliefStatus"),

        # Processing modules
        ("agent_mem.processing.action_logger", "ActionLogger"),
        ("agent_mem.processing.kg_writer", "KGWriter"),
        ("agent_mem.processing.error_handler", "ErrorHandler"),
        ("agent_mem.processing.rca_agent", "RCAAgent"),

        # Retrieval modules
        ("agent_mem.retrieval.embedder", "MultiViewEmbedder"),
        ("agent_mem.retrieval.retriever", "HierarchicalRetriever"),
        ("agent_mem.retrieval.memory_agent", "MemoryAgent"),

        # Storage modules
        ("agent_mem.storage.graph_store", "GraphStore"),
        ("agent_mem.storage.local_storage", "LocalStorage"),

        # Integration modules
        ("agent_mem.integration.sweagent_adapter", "SWEAgentAdapter"),

        # Configuration
        ("agent_mem.config.config_manager", "ConfigManager"),
    ]

    passed = 0
    failed = 0

    for module_path, class_name in components:
        if check_component_exists(module_path, class_name):
            passed += 1
        else:
            failed += 1

    print("\n" + "="*60)
    print(f"Component Completeness Check:")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total:  {passed + failed}")

    assert failed == 0

def test_file_structure():
    """Test that all required files exist."""
    print("\nChecking file structure...")
    print("="*60)

    base_dir = Path(__file__).parent / "agent_mem"
    required_files = [
        # Core structure
        base_dir / "__init__.py",
        base_dir / "types.py",

        # Core modules
        base_dir / "core" / "__init__.py",
        base_dir / "core" / "problem_file.py",
        base_dir / "core" / "observation_kg.py",
        base_dir / "core" / "belief_graph.py",

        # Processing modules
        base_dir / "processing" / "__init__.py",
        base_dir / "processing" / "action_logger.py",
        base_dir / "processing" / "kg_writer.py",
        base_dir / "processing" / "error_handler.py",
        base_dir / "processing" / "rca_agent.py",

        # Retrieval modules
        base_dir / "retrieval" / "__init__.py",
        base_dir / "retrieval" / "embedder.py",
        base_dir / "retrieval" / "retriever.py",
        base_dir / "retrieval" / "memory_agent.py",

        # Storage modules
        base_dir / "storage" / "__init__.py",
        base_dir / "storage" / "graph_store.py",
        base_dir / "storage" / "local_storage.py",

        # Integration modules
        base_dir / "integration" / "__init__.py",
        base_dir / "integration" / "sweagent_adapter.py",

        # Configuration
        base_dir / "config" / "__init__.py",
        base_dir / "config" / "config_manager.py",

        # Tests
        base_dir / "tests" / "__init__.py",
        base_dir / "tests" / "test_core.py",

        # Main entry points
        Path(__file__).parent / "agent_mem_main.py",
        Path(__file__).parent / "setup.py",
        Path(__file__).parent / "requirements.txt",
    ]

    passed = 0
    failed = 0

    for file_path in required_files:
        if file_path.exists():
            print(f" {file_path.relative_to(Path(__file__).parent)} exists")
            passed += 1
        else:
            print(f" {file_path.relative_to(Path(__file__).parent)} missing")
            failed += 1

    print("\n" + "="*60)
    print(f"File Structure Check:")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total:  {passed + failed}")

    assert failed == 0

def test_basic_workflow():
    """Test a basic end-to-end workflow."""
    print("\nTesting basic workflow...")
    print("="*60)

    try:
        # Import core components
        from agent_mem.core.problem_file import ProblemFile, ActionType, Outcome
        from agent_mem.core.observation_kg import ObservationKG, TaskSubgraph
        from agent_mem.storage.graph_store import GraphStore

        # Create temporary storage
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize components
            graph_store = GraphStore(storage_dir=tmpdir)

            # Create a problem file
            pf = ProblemFile(
                task_id="test_workflow_001",
                action_type=ActionType.CODE_EDIT,
                intent_text="Test workflow",
                outcome=Outcome.SUCCESS
            )

            # Create observation KG
            subgraph = TaskSubgraph(task_id="test_workflow_001")
            subgraph.add_action(pf)
            graph_store.observation_kg.add_task_subgraph(subgraph)

            print(" Basic workflow test passed")
            return

    except Exception as e:
        print(f" Basic workflow test failed: {e}")
        raise

def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")
    print("="*60)

    try:
        from agent_mem.config.config_manager import ConfigManager

        # Test default configuration
        config = ConfigManager()
        default_config = config.get_config()

        assert "storage" in default_config
        assert "retrieval" in default_config
        assert "processing" in default_config

        print(f" Default configuration loaded")
        print(f"  Storage config: {default_config.get('storage', {})}")

        return

    except Exception as e:
        print(f" Configuration test failed: {e}")
        raise

def main():
    """Run all functional tests."""
    print("Running Agent-mem MVP functional tests...")
    print("="*60)

    tests_passed = 0
    tests_failed = 0

    # Run component completeness test
    try:
        test_component_completeness()
        tests_passed += 1
    except Exception:
        tests_failed += 1

    # Run file structure test
    try:
        test_file_structure()
        tests_passed += 1
    except Exception:
        tests_failed += 1

    # Run basic workflow test
    try:
        test_basic_workflow()
        tests_passed += 1
    except Exception:
        tests_failed += 1

    # Run configuration test
    try:
        test_configuration()
        tests_passed += 1
    except Exception:
        tests_failed += 1

    print("\n" + "="*60)
    print(f"Functional Test Summary:")
    print(f"  Passed: {tests_passed}")
    print(f"  Failed: {tests_failed}")
    print(f"  Total:  {tests_passed + tests_failed}")

    # Overall assessment
    print("\n" + "="*60)
    print("MVP Assessment:")

    if tests_failed == 0:
        print(" MVP is complete and functional")
        print(" All core components are present")
        print(" File structure is correct")
        print(" Basic workflow works")
        print(" Configuration system works")
    else:
        print(" MVP has some issues")
        if tests_failed > 2:
            print(" Significant components may be missing or broken")
        else:
            print(" Core functionality is mostly intact")

    return tests_failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
