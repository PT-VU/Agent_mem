"""
Local JSON storage implementation for Agent-mem system.

MVP implementation using local filesystem and JSON files for storage.
   JSON
"""

from __future__ import annotations

import json
import os
import shutil
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import gzip
import pickle

from .interface import (
    StorageError, StorageNotFoundError, StorageValidationError, StorageSerializationError,
    UnifiedStorage, StorageFactory, StorageConfig
)

# Import actual implementations for type checking
from ..core.problem_file import ProblemFile
from ..core.observation_kg import ObservationKG, TaskSubgraph, KGEdge
from ..core.belief_graph import BeliefGraph, AtomicBelief, AttemptBelief


class LocalJSONStorage(UnifiedStorage):
    """Local JSON file storage implementation.

    Stores data in hierarchical directory structure:
        base_path/
            problem_files/
                {task_id}/
                    {action_id}.json[.gz]
            observation_kg/
                tasks/
                    {task_id}.json[.gz]
                kg_global.json[.gz]
            belief_graph/
                atomic_beliefs/
                    {belief_id}.json[.gz]
                attempt_beliefs/
                    {attempt_id}.json[.gz]
                belief_graph.json[.gz]
            metadata/
                indices/
                stats/
    """

    def __init__(self, config: StorageConfig):
        """Initialize local JSON storage.

        Args:
            config: Storage configuration

        Raises:
            ValueError: If base_path is not provided
            StorageError: If initialization fails
        """
        if not config.base_path:
            raise ValueError("base_path is required for LocalJSONStorage")

        self.config = config
        self.base_path = Path(config.base_path).expanduser().resolve()

        # Subdirectory paths
        self.problem_files_dir = self.base_path / "problem_files"
        self.observation_kg_dir = self.base_path / "observation_kg"
        self.belief_graph_dir = self.base_path / "belief_graph"
        self.metadata_dir = self.base_path / "metadata"

        # Initialize directories
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize the storage system by creating directory structure.

        Returns:
            True if initialization was successful

        Raises:
            StorageError: If directory creation fails
        """
        try:
            # Create main directories
            directories = [
                self.base_path,
                self.problem_files_dir,
                self.observation_kg_dir / "tasks",
                self.belief_graph_dir / "atomic_beliefs",
                self.belief_graph_dir / "attempt_beliefs",
                self.metadata_dir / "indices",
                self.metadata_dir / "stats"
            ]

            for directory in directories:
                if self.config.auto_create_dirs:
                    directory.mkdir(parents=True, exist_ok=True)
                elif not directory.exists():
                    raise StorageError(f"Directory does not exist: {directory}")

            # Create metadata files
            self._save_metadata({
                "storage_type": "local_json",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config": self.config.to_dict()
            })

            self._initialized = True
            return True

        except Exception as e:
            raise StorageError(f"Failed to initialize storage: {e}")

    def cleanup(self) -> bool:
        """Clean up storage resources.

        Returns:
            True if cleanup was successful

        Note:
            This does not delete data, only cleans up temporary resources.
        """
        # Currently no persistent resources to clean up
        return True

    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the storage system.

        Returns:
            Dictionary with storage system information
        """
        if not self._initialized:
            raise StorageError("Storage not initialized")

        try:
            # Calculate disk usage
            total_size = 0
            file_count = 0

            for path in self.base_path.rglob("*"):
                if path.is_file():
                    total_size += path.stat().st_size
                    file_count += 1

            # Load metadata
            metadata = self._load_metadata()

            return {
                "storage_type": "local_json",
                "base_path": str(self.base_path),
                "initialized": self._initialized,
                "total_size_bytes": total_size,
                "file_count": file_count,
                "directory_count": sum(1 for _ in self.base_path.rglob("*") if _.is_dir()),
                "created_at": metadata.get("created_at"),
                "config": self.config.to_dict()
            }

        except Exception as e:
            raise StorageError(f"Failed to get storage info: {e}")

    def backup(self, backup_path: str) -> bool:
        """Create a backup of the storage.

        Args:
            backup_path: Path where backup should be stored

        Returns:
            True if backup was successful

        Raises:
            StorageError: If backup fails
        """
        try:
            backup_dir = Path(backup_path).expanduser().resolve()
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Create timestamped backup directory
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            target_dir = backup_dir / f"agent_mem_backup_{timestamp}"

            # Copy all files
            shutil.copytree(self.base_path, target_dir)

            # Create backup manifest
            manifest = {
                "backup_time": datetime.now(timezone.utc).isoformat(),
                "source_path": str(self.base_path),
                "backup_path": str(target_dir),
                "file_count": sum(1 for _ in target_dir.rglob("*") if _.is_file()),
                "total_size": sum(f.stat().st_size for f in target_dir.rglob("*") if f.is_file())
            }

            manifest_path = target_dir / "backup_manifest.json"
            self._write_json(manifest_path, manifest)

            return True

        except Exception as e:
            raise StorageError(f"Backup failed: {e}")

    def restore(self, backup_path: str) -> bool:
        """Restore storage from a backup.

        Args:
            backup_path: Path to backup directory

        Returns:
            True if restore was successful

        Raises:
            StorageError: If restore fails
        """
        try:
            backup_dir = Path(backup_path).expanduser().resolve()

            if not backup_dir.exists():
                raise StorageNotFoundError(f"Backup directory not found: {backup_dir}")

            # Check for manifest
            manifest_path = backup_dir / "backup_manifest.json"
            if not manifest_path.exists():
                raise StorageError("Backup manifest not found")

            # Clear current storage (after backup)
            backup_temp = self.base_path.parent / f"{self.base_path.name}_restore_backup"
            if self.base_path.exists():
                shutil.move(self.base_path, backup_temp)

            # Restore from backup
            shutil.copytree(backup_dir, self.base_path)

            # Remove temp backup if successful
            if backup_temp.exists():
                shutil.rmtree(backup_temp)

            # Re-initialize
            self._initialized = False
            return self.initialize()

        except Exception as e:
            raise StorageError(f"Restore failed: {e}")

    # ============================================================================
    # ProblemFileStorage Implementation
    # ============================================================================

    def save_problem_file(self, problem_file: ProblemFile) -> str:
        """Save a ProblemFile to storage.

        Args:
            problem_file: The ProblemFile to save

        Returns:
            File path where the ProblemFile was saved

        Raises:
            StorageValidationError: If the ProblemFile fails validation
            StorageSerializationError: If serialization fails
            StorageError: For other storage-related errors
        """
        if not self._initialized:
            raise StorageError("Storage not initialized")

        try:
            # Validate ProblemFile
            errors = problem_file.validate()
            if errors:
                raise StorageValidationError(f"ProblemFile validation failed: {errors}")

            # Create task directory
            task_dir = self.problem_files_dir / problem_file.task_id
            task_dir.mkdir(parents=True, exist_ok=True)

            # Determine file path
            file_path = task_dir / f"{problem_file.action_id}.json"
            if self.config.compression:
                file_path = file_path.with_suffix(".json.gz")

            # Serialize and save
            data = problem_file.to_dict()
            self._write_json(file_path, data)

            # Update index
            self._update_problem_file_index(problem_file)

            return str(file_path)

        except (StorageValidationError, StorageSerializationError):
            raise
        except Exception as e:
            raise StorageError(f"Failed to save ProblemFile: {e}")

    def load_problem_file(self, identifier: str) -> ProblemFile:
        """Load a ProblemFile from storage.

        Args:
            identifier: File path or action_id

        Returns:
            The loaded ProblemFile

        Raises:
            StorageNotFoundError: If the ProblemFile is not found
            StorageSerializationError: If deserialization fails
            StorageError: For other storage-related errors
        """
        if not self._initialized:
            raise StorageError("Storage not initialized")

        try:
            # Try to load by file path first
            file_path = Path(identifier)

            if not file_path.exists():
                # Try to find by action_id
                file_path = self._find_problem_file_by_id(identifier)
                if not file_path:
                    raise StorageNotFoundError(f"ProblemFile not found: {identifier}")

            # Load and deserialize
            data = self._read_json(file_path)
            problem_file = ProblemFile.from_dict(data)

            return problem_file

        except StorageNotFoundError:
            raise
        except Exception as e:
            raise StorageSerializationError(f"Failed to load ProblemFile: {e}")

    def delete_problem_file(self, identifier: str) -> bool:
        """Delete a ProblemFile from storage.

        Args:
            identifier: File path or action_id

        Returns:
            True if deletion was successful

        Raises:
            StorageNotFoundError: If the ProblemFile is not found
            StorageError: For other storage-related errors
        """
        if not self._initialized:
            raise StorageError("Storage not initialized")

        try:
            # Find file
            file_path = Path(identifier) if Path(identifier).exists() else self._find_problem_file_by_id(identifier)

            if not file_path or not file_path.exists():
                raise StorageNotFoundError(f"ProblemFile not found: {identifier}")

            # Delete file
            file_path.unlink()

            # Clean up empty directories
            self._cleanup_empty_directories(file_path.parent)

            # Update index
            self._remove_from_problem_file_index(identifier)

            return True

        except StorageNotFoundError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to delete ProblemFile: {e}")

    def list_problem_files(self,
                          task_id: Optional[str] = None,
                          action_type: Optional[str] = None,
                          outcome: Optional[str] = None,
                          limit: int = 100,
                          offset: int = 0) -> List[Tuple[str, ProblemFile]]:
        """List ProblemFiles with optional filtering.

        Args:
            task_id: Filter by task ID
            action_type: Filter by action type
            outcome: Filter by outcome
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of (file_path, ProblemFile) tuples

        Raises:
            StorageError: For storage-related errors
        """
        if not self._initialized:
            raise StorageError("Storage not initialized")

        try:
            results = []

            # Use index if available and no filters
            if not task_id and not action_type and not outcome:
                index = self._load_problem_file_index()
                file_paths = list(index.keys())[offset:offset + limit]

                for file_path in file_paths:
                    try:
                        problem_file = self.load_problem_file(file_path)
                        results.append((file_path, problem_file))
                    except Exception:
                        continue  # Skip corrupted files

                return results

            # Otherwise scan directories
            search_dir = self.problem_files_dir
            if task_id:
                search_dir = search_dir / task_id

            if not search_dir.exists():
                return []

            # Collect and filter files
            collected = 0
            skipped = 0

            for file_path in search_dir.rglob("*.json*"):
                if skipped < offset:
                    skipped += 1
                    continue

                if collected >= limit:
                    break

                try:
                    problem_file = self.load_problem_file(str(file_path))

                    # Apply filters
                    if action_type and problem_file.action_type.value != action_type:
                        continue
                    if outcome and problem_file.outcome.value != outcome:
                        continue

                    results.append((str(file_path), problem_file))
                    collected += 1

                except Exception:
                    continue  # Skip corrupted files

            return results

        except Exception as e:
            raise StorageError(f"Failed to list ProblemFiles: {e}")

    def search_problem_files(self,
                            query: str,
                            field: str = "intent_text",
                            limit: int = 50) -> List[Tuple[str, ProblemFile, float]]:
        """Search ProblemFiles by text content.

        Args:
            query: Search query text (case-insensitive substring match)
            field: Field to search in (e.g., "intent_text", "inputs")
            limit: Maximum number of results

        Returns:
            List of (file_path, ProblemFile, relevance_score) tuples

        Raises:
            StorageError: For storage-related errors
        """
        if not self._initialized:
            raise StorageError("Storage not initialized")

        try:
            results = []
            query_lower = query.lower()

            # Simple substring matching for MVP
            for file_path in self.problem_files_dir.rglob("*.json*"):
                if len(results) >= limit:
                    break

                try:
                    problem_file = self.load_problem_file(str(file_path))

                    # Get field value
                    if field == "intent_text":
                        field_value = problem_file.intent_text
                    elif field == "inputs":
                        field_value = str(problem_file.inputs)
                    else:
                        # Try to get from metadata
                        field_value = str(problem_file.metadata.get(field, ""))

                    # Simple relevance scoring
                    if query_lower in field_value.lower():
                        # Calculate simple relevance score
                        position = field_value.lower().find(query_lower)
                        length_ratio = len(query) / max(len(field_value), 1)
                        relevance = 0.5 + (0.5 * length_ratio) - (position / max(len(field_value), 1) * 0.5)

                        results.append((str(file_path), problem_file, relevance))

                except Exception:
                    continue  # Skip corrupted files

            # Sort by relevance
            results.sort(key=lambda x: x[2], reverse=True)

            return results[:limit]

        except Exception as e:
            raise StorageError(f"Failed to search ProblemFiles: {e}")

    def get_problem_file_stats(self) -> Dict[str, Any]:
        """Get statistics about stored ProblemFiles.

        Returns:
            Dictionary with statistics
        """
        if not self._initialized:
            raise StorageError("Storage not initialized")

        try:
            stats = {
                "total_count": 0,
                "by_task": {},
                "by_action_type": {},
                "by_outcome": {},
                "storage_size_bytes": 0
            }

            for file_path in self.problem_files_dir.rglob("*.json*"):
                if file_path.is_file():
                    stats["storage_size_bytes"] += file_path.stat().st_size

                    try:
                        problem_file = self.load_problem_file(str(file_path))

                        stats["total_count"] += 1

                        # Count by task
                        task_id = problem_file.task_id
                        stats["by_task"][task_id] = stats["by_task"].get(task_id, 0) + 1

                        # Count by action type
                        action_type = problem_file.action_type.value
                        stats["by_action_type"][action_type] = stats["by_action_type"].get(action_type, 0) + 1

                        # Count by outcome
                        outcome = problem_file.outcome.value
                        stats["by_outcome"][outcome] = stats["by_outcome"].get(outcome, 0) + 1

                    except Exception:
                        continue  # Skip corrupted files

            return stats

        except Exception as e:
            raise StorageError(f"Failed to get ProblemFile stats: {e}")

    # ============================================================================
    # ObservationKGStorage Implementation (Partial for MVP)
    # ============================================================================

    def save_kg(self, kg: ObservationKG, identifier: Optional[str] = None) -> str:
        """Save an ObservationKG to storage.

        Args:
            kg: The ObservationKG to save
            identifier: Optional storage identifier

        Returns:
            File path where the KG was saved

        Raises:
            StorageSerializationError: If serialization fails
            StorageError: For other storage-related errors
        """
        if not self._initialized:
            raise StorageError("Storage not initialized")

        try:
            # Determine file path
            if identifier:
                file_name = f"{identifier}.json"
            else:
                file_name = "kg_global.json"

            file_path = self.observation_kg_dir / file_name
            if self.config.compression:
                file_path = file_path.with_suffix(".json.gz")

            # Serialize and save
            data = kg.to_dict()
            self._write_json(file_path, data)

            return str(file_path)

        except Exception as e:
            raise StorageSerializationError(f"Failed to save ObservationKG: {e}")

    def load_kg(self, identifier: str) -> ObservationKG:
        """Load an ObservationKG from storage.

        Args:
            identifier: Storage identifier or "global" for main KG

        Returns:
            The loaded ObservationKG

        Raises:
            StorageNotFoundError: If the KG is not found
            StorageSerializationError: If deserialization fails
            StorageError: For other storage-related errors
        """
        if not self._initialized:
            raise StorageError("Storage not initialized")

        try:
            # Determine file path
            if identifier == "global":
                file_name = "kg_global.json"
            else:
                file_name = f"{identifier}.json"

            file_path = self.observation_kg_dir / file_name
            if self.config.compression and not file_path.exists():
                file_path = file_path.with_suffix(".json.gz")

            if not file_path.exists():
                raise StorageNotFoundError(f"ObservationKG not found: {identifier}")

            # Load and deserialize
            data = self._read_json(file_path)
            kg = ObservationKG.from_dict(data)

            return kg

        except StorageNotFoundError:
            raise
        except Exception as e:
            raise StorageSerializationError(f"Failed to load ObservationKG: {e}")

    def save_task_subgraph(self, subgraph: TaskSubgraph) -> str:
        """Save a task subgraph to storage.

        Args:
            subgraph: The TaskSubgraph to save

        Returns:
            File path where the subgraph was saved

        Raises:
            StorageSerializationError: If serialization fails
            StorageError: For other storage-related errors
        """
        if not self._initialized:
            raise StorageError("Storage not initialized")

        try:
            # Create tasks directory
            tasks_dir = self.observation_kg_dir / "tasks"
            tasks_dir.mkdir(parents=True, exist_ok=True)

            # Determine file path
            file_path = tasks_dir / f"{subgraph.task_id}.json"
            if self.config.compression:
                file_path = file_path.with_suffix(".json.gz")

            # Serialize and save
            data = subgraph.to_dict()
            self._write_json(file_path, data)

            return str(file_path)

        except Exception as e:
            raise StorageSerializationError(f"Failed to save task subgraph: {e}")

    def load_task_subgraph(self, task_id: str) -> Optional[TaskSubgraph]:
        """Load a task subgraph by task ID.

        Args:
            task_id: Task ID

        Returns:
            The loaded TaskSubgraph, or None if not found

        Raises:
            StorageSerializationError: If deserialization fails
            StorageError: For other storage-related errors
        """
        if not self._initialized:
            raise StorageError("Storage not initialized")

        try:
            file_path = self.observation_kg_dir / "tasks" / f"{task_id}.json"
            if self.config.compression and not file_path.exists():
                file_path = file_path.with_suffix(".json.gz")

            if not file_path.exists():
                return None

            # Load and deserialize
            data = self._read_json(file_path)
            subgraph = TaskSubgraph.from_dict(data)

            return subgraph

        except Exception as e:
            raise StorageSerializationError(f"Failed to load task subgraph: {e}")

    def delete_task_subgraph(self, task_id: str) -> bool:
        """Delete a task subgraph from storage.

        Args:
            task_id: Task ID

        Returns:
            True if deletion was successful

        Raises:
            StorageNotFoundError: If the subgraph is not found
            StorageError: For other storage-related errors
        """
        if not self._initialized:
            raise StorageError("Storage not initialized")

        try:
            file_path = self.observation_kg_dir / "tasks" / f"{task_id}.json"
            if self.config.compression and not file_path.exists():
                file_path = file_path.with_suffix(".json.gz")

            if not file_path.exists():
                raise StorageNotFoundError(f"Task subgraph not found: {task_id}")

            # Delete file
            file_path.unlink()

            # Clean up empty directories
            self._cleanup_empty_directories(file_path.parent)

            return True

        except StorageNotFoundError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to delete task subgraph: {e}")

    def list_task_subgraphs(self,
                           limit: int = 100,
                           offset: int = 0) -> List[Tuple[str, TaskSubgraph]]:
        """List all task subgraphs.

        Args:
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of (task_id, TaskSubgraph) tuples

        Raises:
            StorageError: For storage-related errors
        """
        if not self._initialized:
            raise StorageError("Storage not initialized")

        try:
            results = []
            tasks_dir = self.observation_kg_dir / "tasks"

            if not tasks_dir.exists():
                return []

            # Collect files
            pattern = "*.json.gz" if self.config.compression else "*.json"
            file_paths = list(tasks_dir.glob(pattern))

            for file_path in file_paths[offset:offset + limit]:
                try:
                    task_id = file_path.stem
                    if file_path.suffix == ".gz":
                        task_id = file_path.stem.replace(".json", "")

                    subgraph = self.load_task_subgraph(task_id)
                    if subgraph:
                        results.append((task_id, subgraph))
                except Exception:
                    continue  # Skip corrupted files

            return results

        except Exception as e:
            raise StorageError(f"Failed to list task subgraphs: {e}")

    def get_kg_stats(self) -> Dict[str, Any]:
        """Get statistics about stored KGs.

        Returns:
            Dictionary with statistics
        """
        if not self._initialized:
            raise StorageError("Storage not initialized")

        try:
            stats = {
                "global_kg_exists": False,
                "task_subgraph_count": 0,
                "storage_size_bytes": 0
            }

            # Check global KG
            global_kg_path = self.observation_kg_dir / "kg_global.json"
            if self.config.compression and not global_kg_path.exists():
                global_kg_path = global_kg_path.with_suffix(".json.gz")

            stats["global_kg_exists"] = global_kg_path.exists()

            # Count task subgraphs
            tasks_dir = self.observation_kg_dir / "tasks"
            if tasks_dir.exists():
                pattern = "*.json.gz" if self.config.compression else "*.json"
                stats["task_subgraph_count"] = len(list(tasks_dir.glob(pattern)))

            # Calculate storage size
            for file_path in self.observation_kg_dir.rglob("*"):
                if file_path.is_file():
                    stats["storage_size_bytes"] += file_path.stat().st_size

            return stats

        except Exception as e:
            raise StorageError(f"Failed to get KG stats: {e}")

    # ============================================================================
    # BeliefGraphStorage Implementation (Stubs for MVP)
    # ============================================================================

    def save_belief_graph(self, belief_graph: BeliefGraph, identifier: Optional[str] = None) -> str:
        """Save a BeliefGraph to storage. (Stub for MVP)"""
        raise NotImplementedError("BeliefGraph storage not implemented in MVP")

    def load_belief_graph(self, identifier: str) -> BeliefGraph:
        """Load a BeliefGraph from storage. (Stub for MVP)"""
        raise NotImplementedError("BeliefGraph storage not implemented in MVP")

    def save_atomic_belief(self, belief: AtomicBelief) -> str:
        """Save an atomic belief to storage. (Stub for MVP)"""
        raise NotImplementedError("Atomic belief storage not implemented in MVP")

    def load_atomic_belief(self, belief_id: str) -> Optional[AtomicBelief]:
        """Load an atomic belief by ID. (Stub for MVP)"""
        raise NotImplementedError("Atomic belief storage not implemented in MVP")

    def delete_atomic_belief(self, belief_id: str) -> bool:
        """Delete an atomic belief from storage. (Stub for MVP)"""
        raise NotImplementedError("Atomic belief storage not implemented in MVP")

    def list_atomic_beliefs(self,
                           belief_type: Optional[str] = None,
                           status: Optional[str] = None,
                           limit: int = 100,
                           offset: int = 0) -> List[Tuple[str, AtomicBelief]]:
        """List atomic beliefs with optional filtering. (Stub for MVP)"""
        raise NotImplementedError("Atomic belief storage not implemented in MVP")

    def get_belief_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about stored belief graphs. (Stub for MVP)"""
        return {
            "implementation_status": "not_implemented_in_mvp",
            "note": "Belief graph storage will be implemented in later phases"
        }

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _write_json(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Write JSON data to file, with optional compression."""
        try:
            json_str = json.dumps(data, indent=2, ensure_ascii=False)

            if self.config.compression:
                with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                    f.write(json_str)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(json_str)

        except Exception as e:
            raise StorageSerializationError(f"Failed to write JSON: {e}")

    def _read_json(self, file_path: Path) -> Dict[str, Any]:
        """Read JSON data from file, with optional compression."""
        try:
            if file_path.suffix == ".gz":
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    return json.load(f)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)

        except Exception as e:
            raise StorageSerializationError(f"Failed to read JSON: {e}")

    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save metadata to file."""
        metadata_path = self.metadata_dir / "storage_metadata.json"
        self._write_json(metadata_path, metadata)

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from file."""
        metadata_path = self.metadata_dir / "storage_metadata.json"
        if metadata_path.exists():
            return self._read_json(metadata_path)
        return {}

    def _update_problem_file_index(self, problem_file: ProblemFile) -> None:
        """Update problem file index."""
        index_path = self.metadata_dir / "indices" / "problem_files.pkl"
        index_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Load existing index
            index = {}
            if index_path.exists():
                with open(index_path, 'rb') as f:
                    index = pickle.load(f)

            # Update index
            file_path = self.problem_files_dir / problem_file.task_id / f"{problem_file.action_id}.json"
            if self.config.compression:
                file_path = file_path.with_suffix(".json.gz")

            index[str(file_path)] = {
                "action_id": problem_file.action_id,
                "task_id": problem_file.task_id,
                "action_type": problem_file.action_type.value,
                "outcome": problem_file.outcome.value,
                "timestamp": problem_file.timestamp
            }

            # Save index
            with open(index_path, 'wb') as f:
                pickle.dump(index, f)

        except Exception:
            # Index update is best-effort
            pass

    def _load_problem_file_index(self) -> Dict[str, Dict[str, Any]]:
        """Load problem file index."""
        index_path = self.metadata_dir / "indices" / "problem_files.pkl"
        if index_path.exists():
            try:
                with open(index_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
        return {}

    def _remove_from_problem_file_index(self, identifier: str) -> None:
        """Remove entry from problem file index."""
        index_path = self.metadata_dir / "indices" / "problem_files.pkl"
        if not index_path.exists():
            return

        try:
            with open(index_path, 'rb') as f:
                index = pickle.load(f)

            # Find and remove entry
            keys_to_remove = []
            for key, value in index.items():
                if value.get("action_id") == identifier or key == identifier:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del index[key]

            with open(index_path, 'wb') as f:
                pickle.dump(index, f)

        except Exception:
            # Index update is best-effort
            pass

    def _find_problem_file_by_id(self, action_id: str) -> Optional[Path]:
        """Find problem file by action_id."""
        # Try index first
        index = self._load_problem_file_index()
        for file_path_str, info in index.items():
            if info.get("action_id") == action_id:
                file_path = Path(file_path_str)
                if file_path.exists():
                    return file_path

        # Fallback: search directories
        for file_path in self.problem_files_dir.rglob("*.json*"):
            if file_path.stem == action_id or file_path.stem.replace(".json", "") == action_id:
                return file_path

        return None

    def _cleanup_empty_directories(self, start_dir: Path) -> None:
        """Recursively remove empty directories."""
        try:
            for dir_path in sorted(start_dir.parents, reverse=True):
                if dir_path == self.base_path:
                    break

                if dir_path.exists() and dir_path.is_dir():
                    # Check if directory is empty
                    has_files = any(dir_path.iterdir())
                    if not has_files and dir_path != self.base_path:
                        dir_path.rmdir()
        except Exception:
            # Cleanup is best-effort
            pass


class LocalJSONStorageFactory(StorageFactory):
    """Factory for creating LocalJSONStorage instances."""

    def create_storage(self, config: Dict[str, Any]) -> LocalJSONStorage:
        """Create a LocalJSONStorage instance based on configuration.

        Args:
            config: Storage configuration dictionary

        Returns:
            Configured LocalJSONStorage instance

        Raises:
            ValueError: If configuration is invalid
            StorageError: If storage creation fails
        """
        try:
            storage_config = StorageConfig.from_dict(config)
            storage = LocalJSONStorage(storage_config)

            # Auto-initialize if configured
            if storage_config.auto_create_dirs:
                storage.initialize()

            return storage

        except Exception as e:
            raise StorageError(f"Failed to create LocalJSONStorage: {e}")


# Legacy compatibility alias used by functional tests and older callers.
class LocalStorage(LocalJSONStorage):
    """Backward-compatible alias for LocalJSONStorage."""
