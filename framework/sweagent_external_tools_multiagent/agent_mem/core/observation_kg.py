"""
Observation KG (Fact Graph): Graph structure with actions as nodes and tasks as trees/subgraphs.

Each task subgraph stores action problem files and `success_next` or
`fail_retry` edges.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from typing_extensions import TypedDict

from .problem_file import ProblemFile, Outcome


class EdgeType(str, Enum):
    """Edge types in the observation KG."""
    SUCCESS_NEXT = "success_next"  # action_i  action_{i+1} (successful progression)
    FAIL_RETRY = "fail_retry"  # fail_action  retry_action (failure repair)


@dataclass
class KGEdge:
    """Edge in the observation KG."""
    source_id: str  # source action_id
    target_id: str  # target action_id
    edge_type: EdgeType
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KGEdge:
        """Create edge from dictionary."""
        if isinstance(data.get("edge_type"), str):
            data["edge_type"] = EdgeType(data["edge_type"])
        return cls(**data)


@dataclass
class TaskSubgraph:
    """Subgraph representing one task execution."""
    task_id: str
    root_action_id: Optional[str] = None  # first action in the task
    action_nodes: Dict[str, ProblemFile] = field(default_factory=dict)  # action_id -> ProblemFile
    edges: List[KGEdge] = field(default_factory=list)  # all edges in this subgraph
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_action(self, problem_file: ProblemFile) -> None:
        """Add an action node to the subgraph."""
        self.action_nodes[problem_file.action_id] = problem_file
        if not self.root_action_id:
            self.root_action_id = problem_file.action_id

    def add_edge(self, edge: KGEdge) -> None:
        """Add an edge to the subgraph."""
        # Validate that both nodes exist
        if edge.source_id not in self.action_nodes:
            raise ValueError(f"Source node {edge.source_id} not found in subgraph")
        if edge.target_id not in self.action_nodes:
            raise ValueError(f"Target node {edge.target_id} not found in subgraph")
        self.edges.append(edge)

    def get_outgoing_edges(self, action_id: str, edge_type: Optional[EdgeType] = None) -> List[KGEdge]:
        """Get outgoing edges from an action, optionally filtered by edge type."""
        edges = [e for e in self.edges if e.source_id == action_id]
        if edge_type:
            edges = [e for e in edges if e.edge_type == edge_type]
        return edges

    def get_incoming_edges(self, action_id: str, edge_type: Optional[EdgeType] = None) -> List[KGEdge]:
        """Get incoming edges to an action, optionally filtered by edge type."""
        edges = [e for e in self.edges if e.target_id == action_id]
        if edge_type:
            edges = [e for e in edges if e.edge_type == edge_type]
        return edges

    def get_success_chain(self, start_action_id: Optional[str] = None) -> List[str]:
        """Get the success chain (main progression) as a list of action IDs."""
        if not self.edges:
            return []

        # Find start of chain if not specified
        if start_action_id is None:
            # Find action with no incoming success edges
            candidates = []
            for action_id in self.action_nodes:
                incoming_success = self.get_incoming_edges(action_id, EdgeType.SUCCESS_NEXT)
                if not incoming_success:
                    candidates.append(action_id)

            if not candidates:
                return []
            start_action_id = candidates[0]

        # Traverse success chain
        chain = [start_action_id]
        current_id = start_action_id

        while True:
            outgoing_success = self.get_outgoing_edges(current_id, EdgeType.SUCCESS_NEXT)
            if not outgoing_success:
                break
            # Assuming single success path for simplicity
            next_action = outgoing_success[0].target_id
            if next_action in chain:  # Avoid cycles
                break
            chain.append(next_action)
            current_id = next_action

        return chain

    def get_failure_retry_chains(self) -> Dict[str, List[str]]:
        """Get all failure-retry chains in the subgraph."""
        chains = {}

        # Find all fail_retry edges
        fail_edges = [e for e in self.edges if e.edge_type == EdgeType.FAIL_RETRY]

        for edge in fail_edges:
            if edge.source_id not in chains:
                chains[edge.source_id] = [edge.source_id, edge.target_id]
            else:
                chains[edge.source_id].append(edge.target_id)

        return chains

    def to_dict(self) -> Dict[str, Any]:
        """Convert subgraph to dictionary."""
        return {
            "task_id": self.task_id,
            "root_action_id": self.root_action_id,
            "action_nodes": {k: v.to_dict() for k, v in self.action_nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaskSubgraph:
        """Create subgraph from dictionary."""
        # Reconstruct action nodes
        action_nodes = {}
        for action_id, node_data in data.get("action_nodes", {}).items():
            action_nodes[action_id] = ProblemFile.from_dict(node_data)

        # Reconstruct edges
        edges = []
        for edge_data in data.get("edges", []):
            edges.append(KGEdge.from_dict(edge_data))

        return cls(
            task_id=data["task_id"],
            root_action_id=data.get("root_action_id"),
            action_nodes=action_nodes,
            edges=edges,
            metadata=data.get("metadata", {}),
        )


class ObservationKG:
    """Observation knowledge graph containing per-task subgraphs."""

    def __init__(self):
        self.task_subgraphs: Dict[str, TaskSubgraph] = {}  # task_id -> TaskSubgraph
        self.action_to_task: Dict[str, str] = {}  # action_id -> task_id mapping

    def add_task_subgraph(self, subgraph: TaskSubgraph) -> None:
        """Add a task subgraph to the KG."""
        self.task_subgraphs[subgraph.task_id] = subgraph

        # Update action-to-task mapping
        for action_id in subgraph.action_nodes:
            self.action_to_task[action_id] = subgraph.task_id

    def get_task_subgraph(self, task_id: str) -> Optional[TaskSubgraph]:
        """Get task subgraph by task_id."""
        return self.task_subgraphs.get(task_id)

    def get_action(self, action_id: str) -> Optional[Tuple[ProblemFile, str]]:
        """Get action by action_id, returns (ProblemFile, task_id)."""
        task_id = self.action_to_task.get(action_id)
        if not task_id:
            return None

        subgraph = self.task_subgraphs.get(task_id)
        if not subgraph:
            return None

        action = subgraph.action_nodes.get(action_id)
        if not action:
            return None

        return action, task_id

    def get_edges_for_action(self, action_id: str) -> List[KGEdge]:
        """Get all edges connected to an action."""
        task_id = self.action_to_task.get(action_id)
        if not task_id:
            return []

        subgraph = self.task_subgraphs.get(task_id)
        if not subgraph:
            return []

        # Get both incoming and outgoing edges
        incoming = subgraph.get_incoming_edges(action_id)
        outgoing = subgraph.get_outgoing_edges(action_id)
        return incoming + outgoing

    def find_similar_actions(self,
                            problem_file: ProblemFile,
                            embedding_view: str = "emb_task_sem",
                            threshold: float = 0.8) -> List[Tuple[str, ProblemFile, float]]:
        """
        Find similar actions based on embedding similarity.

        Args:
            problem_file: The action to compare against
            embedding_view: Which embedding view to use (e.g., "emb_task_sem")
            threshold: Similarity threshold (0.0 to 1.0)

        Returns:
            List of (action_id, ProblemFile, similarity_score) tuples
        """
        query_embeddings = problem_file.embeddings
        if query_embeddings is None:
            return []

        requested_vector = getattr(query_embeddings, embedding_view, None)
        if requested_vector is None:
            return []

        # Core view gets highest weight; the rest act as tie-breakers when available.
        view_weights = {
            embedding_view: 0.7,
            "emb_intent": 0.15 if embedding_view != "emb_intent" else 0.0,
            "emb_error_sig": 0.1 if embedding_view != "emb_error_sig" else 0.0,
            "emb_file_scope": 0.05 if embedding_view != "emb_file_scope" else 0.0,
        }

        def _cosine_similarity(a: List[float], b: List[float]) -> float:
            if not a or not b or len(a) != len(b):
                return 0.0
            dot = 0.0
            na = 0.0
            nb = 0.0
            for x, y in zip(a, b):
                dot += x * y
                na += x * x
                nb += y * y
            if na <= 0.0 or nb <= 0.0:
                return 0.0
            return dot / (math.sqrt(na) * math.sqrt(nb))

        similar: List[Tuple[str, ProblemFile, float]] = []

        for task_id, subgraph in self.task_subgraphs.items():
            for action_id, action in subgraph.action_nodes.items():
                if action_id == problem_file.action_id:
                    continue

                candidate_embeddings = action.embeddings
                if candidate_embeddings is None:
                    continue

                weighted_score = 0.0
                used_weight = 0.0

                for view_name, weight in view_weights.items():
                    if weight <= 0.0:
                        continue
                    qv = getattr(query_embeddings, view_name, None)
                    cv = getattr(candidate_embeddings, view_name, None)
                    if qv is None or cv is None:
                        continue
                    sim = _cosine_similarity(qv, cv)
                    weighted_score += weight * sim
                    used_weight += weight

                if used_weight <= 0.0:
                    continue

                final_score = weighted_score / used_weight
                if final_score >= threshold:
                    similar.append((action_id, action, final_score))

        similar.sort(key=lambda x: x[2], reverse=True)
        return similar

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the KG."""
        total_actions = sum(len(sg.action_nodes) for sg in self.task_subgraphs.values())
        total_edges = sum(len(sg.edges) for sg in self.task_subgraphs.values())

        # Count edge types
        edge_type_counts = {}
        for sg in self.task_subgraphs.values():
            for edge in sg.edges:
                edge_type = edge.edge_type.value
                edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1

        # Count outcomes
        outcome_counts = {}
        for sg in self.task_subgraphs.values():
            for action in sg.action_nodes.values():
                outcome = action.outcome.value
                outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

        return {
            "total_tasks": len(self.task_subgraphs),
            "total_actions": total_actions,
            "total_edges": total_edges,
            "edge_type_counts": edge_type_counts,
            "outcome_counts": outcome_counts,
            "avg_actions_per_task": total_actions / len(self.task_subgraphs) if self.task_subgraphs else 0,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire KG to dictionary."""
        return {
            "task_subgraphs": {k: v.to_dict() for k, v in self.task_subgraphs.items()},
            "action_to_task": self.action_to_task,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ObservationKG:
        """Create KG from dictionary."""
        kg = cls()

        # Reconstruct task subgraphs
        for task_id, subgraph_data in data.get("task_subgraphs", {}).items():
            subgraph = TaskSubgraph.from_dict(subgraph_data)
            kg.add_task_subgraph(subgraph)

        # Restore action-to-task mapping
        kg.action_to_task = data.get("action_to_task", {})

        return kg

    def save_to_file(self, filepath: str) -> None:
        """Save KG to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filepath: str) -> ObservationKG:
        """Load KG from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
