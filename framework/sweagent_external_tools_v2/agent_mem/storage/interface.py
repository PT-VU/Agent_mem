"""
Storage interface definitions for Agent-mem system.


 JSON MVP
"""

from __future__ import annotations

import abc
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from typing_extensions import Protocol

# Import will be handled by implementations
# These are type hints only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.problem_file import ProblemFile
    from core.observation_kg import ObservationKG, TaskSubgraph, KGEdge
    from core.belief_graph import BeliefGraph, AtomicBelief, AttemptBelief


class StorageError(Exception):
    """Base exception for storage-related errors."""
    pass


class StorageNotFoundError(StorageError):
    """Raised when a requested resource is not found in storage."""
    pass


class StorageValidationError(StorageError):
    """Raised when data validation fails before storage."""
    pass


class StorageSerializationError(StorageError):
    """Raised when serialization/deserialization fails."""
    pass


# ============================================================================
# Problem File Storage Interface
# ============================================================================

class ProblemFileStorage(abc.ABC):
    """Abstract interface for ProblemFile storage operations."""

    @abc.abstractmethod
    def save_problem_file(self, problem_file: ProblemFile) -> str:
        """Save a ProblemFile to storage.

        Args:
            problem_file: The ProblemFile to save

        Returns:
            Storage identifier or path for the saved file

        Raises:
            StorageValidationError: If the ProblemFile fails validation
            StorageSerializationError: If serialization fails
            StorageError: For other storage-related errors
        """
        pass

    @abc.abstractmethod
    def load_problem_file(self, identifier: str) -> ProblemFile:
        """Load a ProblemFile from storage.

        Args:
            identifier: Storage identifier or path

        Returns:
            The loaded ProblemFile

        Raises:
            StorageNotFoundError: If the ProblemFile is not found
            StorageSerializationError: If deserialization fails
            StorageError: For other storage-related errors
        """
        pass

    @abc.abstractmethod
    def delete_problem_file(self, identifier: str) -> bool:
        """Delete a ProblemFile from storage.

        Args:
            identifier: Storage identifier or path

        Returns:
            True if deletion was successful, False otherwise

        Raises:
            StorageNotFoundError: If the ProblemFile is not found
            StorageError: For other storage-related errors
        """
        pass

    @abc.abstractmethod
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
            List of (identifier, ProblemFile) tuples

        Raises:
            StorageError: For storage-related errors
        """
        pass

    @abc.abstractmethod
    def search_problem_files(self,
                            query: str,
                            field: str = "intent_text",
                            limit: int = 50) -> List[Tuple[str, ProblemFile, float]]:
        """Search ProblemFiles by text content.

        Args:
            query: Search query text
            field: Field to search in (e.g., "intent_text", "inputs")
            limit: Maximum number of results

        Returns:
            List of (identifier, ProblemFile, relevance_score) tuples

        Raises:
            StorageError: For storage-related errors
        """
        pass

    @abc.abstractmethod
    def get_problem_file_stats(self) -> Dict[str, Any]:
        """Get statistics about stored ProblemFiles.

        Returns:
            Dictionary with statistics
        """
        pass


# ============================================================================
# Observation KG Storage Interface
# ============================================================================

class ObservationKGStorage(abc.ABC):
    """Abstract interface for ObservationKG storage operations."""

    @abc.abstractmethod
    def save_kg(self, kg: ObservationKG, identifier: Optional[str] = None) -> str:
        """Save an ObservationKG to storage.

        Args:
            kg: The ObservationKG to save
            identifier: Optional storage identifier

        Returns:
            Storage identifier for the saved KG

        Raises:
            StorageSerializationError: If serialization fails
            StorageError: For other storage-related errors
        """
        pass

    @abc.abstractmethod
    def load_kg(self, identifier: str) -> ObservationKG:
        """Load an ObservationKG from storage.

        Args:
            identifier: Storage identifier

        Returns:
            The loaded ObservationKG

        Raises:
            StorageNotFoundError: If the KG is not found
            StorageSerializationError: If deserialization fails
            StorageError: For other storage-related errors
        """
        pass

    @abc.abstractmethod
    def save_task_subgraph(self, subgraph: TaskSubgraph) -> str:
        """Save a task subgraph to storage.

        Args:
            subgraph: The TaskSubgraph to save

        Returns:
            Storage identifier for the saved subgraph

        Raises:
            StorageSerializationError: If serialization fails
            StorageError: For other storage-related errors
        """
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def delete_task_subgraph(self, task_id: str) -> bool:
        """Delete a task subgraph from storage.

        Args:
            task_id: Task ID

        Returns:
            True if deletion was successful, False otherwise

        Raises:
            StorageNotFoundError: If the subgraph is not found
            StorageError: For other storage-related errors
        """
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def get_kg_stats(self) -> Dict[str, Any]:
        """Get statistics about stored KGs.

        Returns:
            Dictionary with statistics
        """
        pass


# ============================================================================
# Belief Graph Storage Interface
# ============================================================================

class BeliefGraphStorage(abc.ABC):
    """Abstract interface for BeliefGraph storage operations."""

    @abc.abstractmethod
    def save_belief_graph(self, belief_graph: BeliefGraph, identifier: Optional[str] = None) -> str:
        """Save a BeliefGraph to storage.

        Args:
            belief_graph: The BeliefGraph to save
            identifier: Optional storage identifier

        Returns:
            Storage identifier for the saved belief graph

        Raises:
            StorageSerializationError: If serialization fails
            StorageError: For other storage-related errors
        """
        pass

    @abc.abstractmethod
    def load_belief_graph(self, identifier: str) -> BeliefGraph:
        """Load a BeliefGraph from storage.

        Args:
            identifier: Storage identifier

        Returns:
            The loaded BeliefGraph

        Raises:
            StorageNotFoundError: If the belief graph is not found
            StorageSerializationError: If deserialization fails
            StorageError: For other storage-related errors
        """
        pass

    @abc.abstractmethod
    def save_atomic_belief(self, belief: AtomicBelief) -> str:
        """Save an atomic belief to storage.

        Args:
            belief: The AtomicBelief to save

        Returns:
            Storage identifier for the saved belief

        Raises:
            StorageSerializationError: If serialization fails
            StorageError: For other storage-related errors
        """
        pass

    @abc.abstractmethod
    def load_atomic_belief(self, belief_id: str) -> Optional[AtomicBelief]:
        """Load an atomic belief by ID.

        Args:
            belief_id: Belief ID

        Returns:
            The loaded AtomicBelief, or None if not found

        Raises:
            StorageSerializationError: If deserialization fails
            StorageError: For other storage-related errors
        """
        pass

    @abc.abstractmethod
    def delete_atomic_belief(self, belief_id: str) -> bool:
        """Delete an atomic belief from storage.

        Args:
            belief_id: Belief ID

        Returns:
            True if deletion was successful, False otherwise

        Raises:
            StorageNotFoundError: If the belief is not found
            StorageError: For other storage-related errors
        """
        pass

    @abc.abstractmethod
    def list_atomic_beliefs(self,
                           belief_type: Optional[str] = None,
                           status: Optional[str] = None,
                           limit: int = 100,
                           offset: int = 0) -> List[Tuple[str, AtomicBelief]]:
        """List atomic beliefs with optional filtering.

        Args:
            belief_type: Filter by belief type
            status: Filter by belief status
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of (belief_id, AtomicBelief) tuples

        Raises:
            StorageError: For storage-related errors
        """
        pass

    @abc.abstractmethod
    def get_belief_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about stored belief graphs.

        Returns:
            Dictionary with statistics
        """
        pass


# ============================================================================
# Unified Storage Interface
# ============================================================================

class UnifiedStorage(ProblemFileStorage, ObservationKGStorage, BeliefGraphStorage, abc.ABC):
    """Unified storage interface combining all storage operations.

    This interface provides a single point of access for all storage needs.
    Implementations can choose to store different types of data in different
    backends while presenting a unified API.
    """

    @abc.abstractmethod
    def initialize(self) -> bool:
        """Initialize the storage system.

        Returns:
            True if initialization was successful, False otherwise

        Raises:
            StorageError: If initialization fails
        """
        pass

    @abc.abstractmethod
    def cleanup(self) -> bool:
        """Clean up storage resources.

        Returns:
            True if cleanup was successful, False otherwise

        Raises:
            StorageError: If cleanup fails
        """
        pass

    @abc.abstractmethod
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the storage system.

        Returns:
            Dictionary with storage system information
        """
        pass

    @abc.abstractmethod
    def backup(self, backup_path: str) -> bool:
        """Create a backup of the storage.

        Args:
            backup_path: Path where backup should be stored

        Returns:
            True if backup was successful, False otherwise

        Raises:
            StorageError: If backup fails
        """
        pass

    @abc.abstractmethod
    def restore(self, backup_path: str) -> bool:
        """Restore storage from a backup.

        Args:
            backup_path: Path to backup file

        Returns:
            True if restore was successful, False otherwise

        Raises:
            StorageError: If restore fails
        """
        pass


# ============================================================================
# Factory and Registry
# ============================================================================

class StorageFactory(abc.ABC):
    """Factory for creating storage instances."""

    @abc.abstractmethod
    def create_storage(self, config: Dict[str, Any]) -> UnifiedStorage:
        """Create a storage instance based on configuration.

        Args:
            config: Storage configuration dictionary

        Returns:
            Configured storage instance

        Raises:
            ValueError: If configuration is invalid
            StorageError: If storage creation fails
        """
        pass


class StorageRegistry:
    """Registry for storage implementations."""

    _registry: Dict[str, StorageFactory] = {}

    @classmethod
    def register(cls, storage_type: str, factory: StorageFactory) -> None:
        """Register a storage factory.

        Args:
            storage_type: Type identifier for the storage
            factory: Factory instance

        Raises:
            ValueError: If storage_type is already registered
        """
        if storage_type in cls._registry:
            raise ValueError(f"Storage type '{storage_type}' already registered")
        cls._registry[storage_type] = factory

    @classmethod
    def create_storage(cls, storage_type: str, config: Dict[str, Any]) -> UnifiedStorage:
        """Create a storage instance by type.

        Args:
            storage_type: Type identifier for the storage
            config: Storage configuration dictionary

        Returns:
            Configured storage instance

        Raises:
            ValueError: If storage_type is not registered
            StorageError: If storage creation fails
        """
        if storage_type not in cls._registry:
            raise ValueError(f"Storage type '{storage_type}' not registered")
        return cls._registry[storage_type].create_storage(config)

    @classmethod
    def list_available_storages(cls) -> List[str]:
        """List all registered storage types.

        Returns:
            List of registered storage type identifiers
        """
        return list(cls._registry.keys())


# ============================================================================
# Configuration
# ============================================================================

class StorageConfig:
    """Configuration for storage systems."""

    def __init__(self,
                 storage_type: str = "local_json",
                 base_path: Optional[str] = None,
                 auto_create_dirs: bool = True,
                 compression: bool = False,
                 encryption_key: Optional[str] = None):
        self.storage_type = storage_type
        self.base_path = base_path
        self.auto_create_dirs = auto_create_dirs
        self.compression = compression
        self.encryption_key = encryption_key

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "storage_type": self.storage_type,
            "base_path": self.base_path,
            "auto_create_dirs": self.auto_create_dirs,
            "compression": self.compression,
            "encryption_key": self.encryption_key if self.encryption_key else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StorageConfig:
        """Create configuration from dictionary."""
        return cls(
            storage_type=data.get("storage_type", "local_json"),
            base_path=data.get("base_path"),
            auto_create_dirs=data.get("auto_create_dirs", True),
            compression=data.get("compression", False),
            encryption_key=data.get("encryption_key")
        )