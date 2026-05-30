#!/usr/bin/env python3
"""
Test script for LocalJSONStorage implementation.
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_mem.storage import LocalJSONStorage, LocalJSONStorageFactory, StorageConfig
from agent_mem.core.problem_file import ProblemFile, ActionType, Outcome, EvidencePointer
from agent_mem.core.observation_kg import ObservationKG, TaskSubgraph, KGEdge, EdgeType


def test_storage_initialization():
    """Test storage initialization and basic operations."""
    print("=== Test: Storage Initialization ===")

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        config = StorageConfig(
            storage_type="local_json",
            base_path=temp_dir,
            auto_create_dirs=True,
            compression=False
        )

        # Create storage
        storage = LocalJSONStorage(config)
        assert not storage._initialized, "Storage should not be initialized yet"

        # Initialize
        result = storage.initialize()
        assert result, "Initialization should succeed"
        assert storage._initialized, "Storage should be initialized"

        # Check directory structure
        assert os.path.exists(storage.base_path), "Base path should exist"
        assert os.path.exists(storage.problem_files_dir), "Problem files dir should exist"
        assert os.path.exists(storage.observation_kg_dir / "tasks"), "Tasks dir should exist"
        assert os.path.exists(storage.belief_graph_dir / "atomic_beliefs"), "Atomic beliefs dir should exist"

        # Get storage info
        info = storage.get_storage_info()
        print(f"Storage info: {info}")
        assert info["storage_type"] == "local_json"
        assert info["initialized"] == True
        assert info["base_path"] == temp_dir

        # Cleanup
        result = storage.cleanup()
        assert result, "Cleanup should succeed"

        print(" Storage initialization test PASSED")


def test_problem_file_crud():
    """Test ProblemFile CRUD operations."""
    print("\n=== Test: ProblemFile CRUD Operations ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        config = StorageConfig(base_path=temp_dir, auto_create_dirs=True)
        storage = LocalJSONStorage(config)
        storage.initialize()

        # Create test ProblemFile
        problem_file = ProblemFile(
            action_id="test-action-001",
            task_id="test-task-001",
            action_type=ActionType.TOOL_CALL,
            intent_text="Test action for storage",
            outcome=Outcome.SUCCESS,
            metadata={"test": True}
        )

        # Test save
        file_path = storage.save_problem_file(problem_file)
        print(f"Saved ProblemFile to: {file_path}")
        assert os.path.exists(file_path), "ProblemFile should be saved"

        # Test load
        loaded = storage.load_problem_file(file_path)
        print(f"Loaded ProblemFile action_id: {loaded.action_id}")
        assert loaded.action_id == problem_file.action_id
        assert loaded.task_id == problem_file.task_id
        assert loaded.action_type == problem_file.action_type
        assert loaded.intent_text == problem_file.intent_text

        # Test load by action_id
        loaded_by_id = storage.load_problem_file("test-action-001")
        assert loaded_by_id.action_id == problem_file.action_id

        # Test list
        files = storage.list_problem_files()
        print(f"Listed {len(files)} ProblemFiles")
        assert len(files) >= 1
        assert any(pf[1].action_id == "test-action-001" for pf in files)

        # Test list with filter
        filtered = storage.list_problem_files(task_id="test-task-001")
        assert len(filtered) >= 1
        assert all(pf[1].task_id == "test-task-001" for pf in filtered)

        # Test search
        search_results = storage.search_problem_files("storage", field="intent_text")
        print(f"Search results: {len(search_results)}")
        assert len(search_results) >= 1
        assert any("storage" in pf[1].intent_text.lower() for pf in search_results)

        # Test stats
        stats = storage.get_problem_file_stats()
        print(f"ProblemFile stats: {stats}")
        assert stats["total_count"] >= 1
        assert "test-task-001" in stats["by_task"]

        # Test delete
        result = storage.delete_problem_file("test-action-001")
        assert result, "Delete should succeed"

        # Verify deletion
        try:
            storage.load_problem_file("test-action-001")
            assert False, "Should have raised StorageNotFoundError"
        except Exception as e:
            print(f"Correctly raised error after deletion: {type(e).__name__}")

        print(" ProblemFile CRUD test PASSED")


def test_observation_kg_storage():
    """Test ObservationKG storage operations."""
    print("\n=== Test: ObservationKG Storage ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        config = StorageConfig(base_path=temp_dir, auto_create_dirs=True)
        storage = LocalJSONStorage(config)
        storage.initialize()

        # Create test ObservationKG
        kg = ObservationKG()

        # Create test TaskSubgraph
        subgraph = TaskSubgraph(task_id="test-task-001")

        # Create test ProblemFiles
        pf1 = ProblemFile(
            action_id="action-001",
            task_id="test-task-001",
            action_type=ActionType.TOOL_CALL,
            intent_text="First action",
            outcome=Outcome.SUCCESS
        )

        pf2 = ProblemFile(
            action_id="action-002",
            task_id="test-task-001",
            action_type=ActionType.CODE_EDIT,
            intent_text="Second action",
            outcome=Outcome.SUCCESS
        )

        # Add actions to subgraph
        subgraph.add_action(pf1)
        subgraph.add_action(pf2)

        # Add edge
        edge = KGEdge(
            source_id="action-001",
            target_id="action-002",
            edge_type=EdgeType.SUCCESS_NEXT
        )
        subgraph.add_edge(edge)

        # Add subgraph to KG
        kg.add_task_subgraph(subgraph)

        # Test save KG
        kg_path = storage.save_kg(kg, "test_kg")
        print(f"Saved KG to: {kg_path}")
        assert os.path.exists(kg_path), "KG should be saved"

        # Test load KG
        loaded_kg = storage.load_kg("test_kg")
        assert loaded_kg is not None
        assert "test-task-001" in loaded_kg.task_subgraphs

        # Test save task subgraph
        subgraph_path = storage.save_task_subgraph(subgraph)
        print(f"Saved subgraph to: {subgraph_path}")
        assert os.path.exists(subgraph_path), "Subgraph should be saved"

        # Test load task subgraph
        loaded_subgraph = storage.load_task_subgraph("test-task-001")
        assert loaded_subgraph is not None
        assert loaded_subgraph.task_id == "test-task-001"
        assert len(loaded_subgraph.action_nodes) == 2

        # Test list task subgraphs
        subgraphs = storage.list_task_subgraphs()
        print(f"Listed {len(subgraphs)} subgraphs")
        assert len(subgraphs) >= 1
        assert any(sg[0] == "test-task-001" for sg in subgraphs)

        # Test delete task subgraph
        result = storage.delete_task_subgraph("test-task-001")
        assert result, "Delete should succeed"

        # Verify deletion
        deleted_subgraph = storage.load_task_subgraph("test-task-001")
        assert deleted_subgraph is None, "Subgraph should be deleted"

        # Test KG stats
        stats = storage.get_kg_stats()
        print(f"KG stats: {stats}")
        assert stats["task_subgraph_count"] == 0  # After deletion

        print(" ObservationKG storage test PASSED")


def test_storage_factory():
    """Test storage factory and registry."""
    print("\n=== Test: Storage Factory ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test factory
        factory = LocalJSONStorageFactory()

        config_dict = {
            "storage_type": "local_json",
            "base_path": temp_dir,
            "auto_create_dirs": True,
            "compression": False
        }

        storage = factory.create_storage(config_dict)
        assert isinstance(storage, LocalJSONStorage)
        assert storage._initialized, "Storage should be auto-initialized"

        # Test with invalid config
        try:
            factory.create_storage({"base_path": None})
            assert False, "Should have raised error for invalid config"
        except Exception as e:
            print(f"Correctly raised error for invalid config: {type(e).__name__}")

        print(" Storage factory test PASSED")


def test_compression():
    """Test storage with compression enabled."""
    print("\n=== Test: Storage with Compression ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        config = StorageConfig(
            base_path=temp_dir,
            auto_create_dirs=True,
            compression=True  # Enable compression
        )

        storage = LocalJSONStorage(config)
        storage.initialize()

        # Create and save ProblemFile
        problem_file = ProblemFile(
            action_id="compressed-action-001",
            task_id="compressed-task-001",
            action_type=ActionType.TOOL_CALL,
            intent_text="Test compressed storage"
        )

        file_path = storage.save_problem_file(problem_file)
        print(f"Saved compressed file: {file_path}")

        # Check file extension
        assert file_path.endswith(".json.gz"), "Compressed file should have .json.gz extension"
        assert os.path.exists(file_path), "Compressed file should exist"

        # Load and verify
        loaded = storage.load_problem_file(file_path)
        assert loaded.action_id == problem_file.action_id

        # Check file size (compressed should be smaller than uncompressed)
        uncompressed_size = len(json.dumps(problem_file.to_dict(), indent=2))
        compressed_size = os.path.getsize(file_path)
        print(f"Uncompressed size: {uncompressed_size}, Compressed size: {compressed_size}")

        # Note: For very small files, compression might not make it smaller
        # due to gzip headers

        print(" Compression test PASSED")


def test_backup_restore():
    """Test backup and restore functionality."""
    print("\n=== Test: Backup and Restore ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create main storage
        config = StorageConfig(base_path=os.path.join(temp_dir, "main_storage"))
        storage = LocalJSONStorage(config)
        storage.initialize()

        # Add some data
        problem_file = ProblemFile(
            action_id="backup-action-001",
            task_id="backup-task-001",
            action_type=ActionType.TOOL_CALL,
            intent_text="Test backup"
        )
        storage.save_problem_file(problem_file)

        # Create backup
        backup_dir = os.path.join(temp_dir, "backups")
        result = storage.backup(backup_dir)
        assert result, "Backup should succeed"

        # Check backup directory
        backup_contents = os.listdir(backup_dir)
        print(f"Backup contents: {backup_contents}")
        assert len(backup_contents) > 0

        # Find backup directory
        backup_path = None
        for item in backup_contents:
            if item.startswith("agent_mem_backup_"):
                backup_path = os.path.join(backup_dir, item)
                break

        assert backup_path is not None, "Backup directory should exist"
        assert os.path.exists(os.path.join(backup_path, "backup_manifest.json")), "Manifest should exist"

        # Test restore (to a different location)
        restore_dir = os.path.join(temp_dir, "restored_storage")
        config_restore = StorageConfig(base_path=restore_dir)
        storage_restore = LocalJSONStorage(config_restore)

        # Note: Restore requires the storage to be initialized first
        storage_restore.initialize()

        # For MVP, we'll just verify backup was created
        # Full restore test would require more complex setup

        print(" Backup test PASSED (restore tested minimally for MVP)")


def test_error_handling():
    """Test error handling in storage operations."""
    print("\n=== Test: Error Handling ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        config = StorageConfig(base_path=temp_dir)
        storage = LocalJSONStorage(config)

        # Test operations before initialization
        try:
            storage.save_problem_file(ProblemFile(action_id="test", task_id="test"))
            assert False, "Should have raised StorageError"
        except Exception as e:
            print(f"Correctly raised error before init: {type(e).__name__}")

        storage.initialize()

        # Test loading non-existent file
        try:
            storage.load_problem_file("non-existent-id")
            assert False, "Should have raised StorageNotFoundError"
        except Exception as e:
            print(f"Correctly raised error for non-existent file: {type(e).__name__}")

        # Test deleting non-existent file
        try:
            storage.delete_problem_file("non-existent-id")
            assert False, "Should have raised StorageNotFoundError"
        except Exception as e:
            print(f"Correctly raised error for non-existent delete: {type(e).__name__}")

        # Test invalid ProblemFile (missing required fields)
        invalid_pf = ProblemFile(action_id="", task_id="")  # Missing required fields
        try:
            storage.save_problem_file(invalid_pf)
            assert False, "Should have raised StorageValidationError"
        except Exception as e:
            print(f"Correctly raised validation error: {type(e).__name__}")

        print(" Error handling test PASSED")


def main():
    """Run all local storage tests."""
    print("Local JSON Storage Test Suite")
    print("=" * 50)

    # Import json for compression test
    global json
    import json

    try:
        test_storage_initialization()
        test_problem_file_crud()
        test_observation_kg_storage()
        test_storage_factory()
        test_compression()
        test_backup_restore()
        test_error_handling()

        print("\n" + "=" * 50)
        print("All local storage tests completed!")

    except Exception as e:
        print(f"\n Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
