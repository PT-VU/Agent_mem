"""Hierarchical retrieval for planning guidance and execution-time repair."""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
import numpy as np
from datetime import datetime

from ..core.problem_file import ProblemFile
from ..core.observation_kg import ObservationKG
from .embedder import MultiViewEmbedder, EmbeddingType, EmbeddingManager


class RetrievalLevel(Enum):
    """Supported retrieval scopes."""
    TASK_LEVEL = "task_level"
    ACTION_LEVEL = "action_level"  # Action


class RetrievalMode(Enum):
    """Supported retrieval modes."""
    PLANNING = "planning"
    EXECUTION = "execution"


class HierarchicalRetriever:
    """Retrieve relevant experience from graph and embedding views."""

    def __init__(self,
                 observation_kg: ObservationKG,
                 embedding_manager: EmbeddingManager,
                 similarity_threshold: float = 0.7):
        """Initialize the hierarchical retriever.

        Args:
            observation_kg: Observation KG
            embedding_manager: Embedding index and lookup manager
            similarity_threshold: Minimum score for returned items
        """
        self.observation_kg = observation_kg
        self.embedding_manager = embedding_manager
        self.similarity_threshold = similarity_threshold

    def retrieve_for_planning(self,
                             query: Dict[str, Any],
                             limit: int = 5) -> Dict[str, Any]:
        """Retrieve planning-time success paths and evidence.

        Args:
            query: Structured planning query
            limit: Maximum number of returned success skeletons

        Returns:
            Retrieved planning guidance grouped by evidence type
        """

        task_semantic = query.get("task_semantic", "")
        file_scope = query.get("file_scope", "")
        env_cluster = query.get("env_cluster", "")
        intent_patterns = query.get("intent_patterns", [])

        results = {
            "success_skeletons": [],
            "workflow_beliefs": [],
            "env_adaptations": [],
            "evidence_refs": []
        }

        # 1.
        if task_semantic:
            task_results = self._retrieve_by_task_semantic(task_semantic, limit)
            results["success_skeletons"].extend(task_results)

        # 2.
        if file_scope:
            file_results = self._retrieve_by_file_scope(file_scope, limit)
            results["success_skeletons"].extend(file_results)

        # 3.
        for intent_pattern in intent_patterns:
            intent_results = self._retrieve_by_intent(intent_pattern, limit // 2)
            results["success_skeletons"].extend(intent_results)


        results["success_skeletons"] = self._deduplicate_and_sort(
            results["success_skeletons"],
            limit
        )

        return results

    def retrieve_for_execution(self,
                              error_context: Dict[str, Any],
                              limit: int = 3) -> Dict[str, Any]:
        """Retrieve repair guidance for the current error context.

        Args:
            error_context: Structured execution error details
            limit: Maximum number of returned repairs

        Returns:
            Repair suggestions and similar failures
        """
        error_type = error_context.get("error_type", "")
        error_message = error_context.get("error_message", "")
        current_action = error_context.get("current_action", "")
        env_signature = error_context.get("env_signature", {})

        results = {
            "repair_suggestions": [],
            "similar_failures": [],
            "evidence_refs": []
        }

        # 1.
        error_signature = {
            "error_type": error_type,
            "key_tokens": self._extract_key_tokens(error_message),
            "context": error_message[:200]
        }

        similar_failures = self.observation_kg.search_by_error_signature(error_signature)
        for failure in similar_failures[:limit]:

            repair_pairs = self._find_repair_for_failure(failure)
            if repair_pairs:
                results["repair_suggestions"].extend(repair_pairs)
                results["similar_failures"].append(failure)

        # 2.
        if current_action:
            action_results = self._retrieve_by_action_similarity(current_action, limit)
            results["repair_suggestions"].extend(action_results)


        results["repair_suggestions"] = self._deduplicate_and_sort(
            results["repair_suggestions"],
            limit
        )
        results["similar_failures"] = results["similar_failures"][:limit]

        return results

    def _retrieve_by_task_semantic(self, task_semantic: str, limit: int) -> List[Dict[str, Any]]:
        """Retrieve actions with similar task semantics."""
        embeddings = self.embedding_manager.search_similar(
            task_semantic,
            EmbeddingType.TASK_SEM.value,
            limit * 2
        )

        results = []
        for item in embeddings:
            problem_file_id = item.get("problem_file_id")
            similarity = item.get("similarity", 0.0)

            if similarity < self.similarity_threshold:
                continue


            result = {
                "problem_file_id": problem_file_id,
                "similarity": similarity,
                "type": "task_semantic_match",
                "evidence": f"Task semantic match: {similarity:.3f}"
            }
            results.append(result)

        return results

    def _retrieve_by_file_scope(self, file_scope: str, limit: int) -> List[Dict[str, Any]]:
        """Retrieve actions touching a similar file scope."""
        embeddings = self.embedding_manager.search_similar(
            file_scope,
            EmbeddingType.FILE_SCOPE.value,
            limit * 2
        )

        results = []
        for item in embeddings:
            problem_file_id = item.get("problem_file_id")
            similarity = item.get("similarity", 0.0)

            if similarity < self.similarity_threshold:
                continue

            result = {
                "problem_file_id": problem_file_id,
                "similarity": similarity,
                "type": "file_scope_match",
                "evidence": f"File scope match: {similarity:.3f}"
            }
            results.append(result)

        return results

    def _retrieve_by_intent(self, intent_pattern: str, limit: int) -> List[Dict[str, Any]]:
        """Retrieve actions with a similar intent."""
        embeddings = self.embedding_manager.search_similar(
            intent_pattern,
            EmbeddingType.INTENT.value,
            limit * 2
        )

        results = []
        for item in embeddings:
            problem_file_id = item.get("problem_file_id")
            similarity = item.get("similarity", 0.0)

            if similarity < self.similarity_threshold:
                continue

            result = {
                "problem_file_id": problem_file_id,
                "similarity": similarity,
                "type": "intent_match",
                "evidence": f"Intent match: {similarity:.3f}"
            }
            results.append(result)

        return results

    def _find_repair_for_failure(self, failure: ProblemFile) -> List[Dict[str, Any]]:
        """Return known repair actions linked to a failure."""
        repair_pairs = []

        task_id = failure.task_id
        if not task_id:
            return repair_pairs

        fail_retry_pairs = self.observation_kg.get_fail_retry_pairs(task_id)

        for fail_pf, repair_pf in fail_retry_pairs:
            if fail_pf.action_id == failure.action_id:
                repair_info = {
                    "failure_id": fail_pf.action_id,
                    "repair_id": repair_pf.action_id,
                    "repair_action": repair_pf.intent_text,
                    "evidence": f"Found repair for failure {fail_pf.action_id}",
                    "similarity": 1.0
                }
                repair_pairs.append(repair_info)

        return repair_pairs

    def _retrieve_by_action_similarity(self, current_action: str, limit: int) -> List[Dict[str, Any]]:
        """Retrieve actions similar to the current action text."""
        embeddings = self.embedding_manager.search_similar(
            current_action,
            EmbeddingType.TASK_SEM.value,
            limit
        )

        results = []
        for item in embeddings:
            problem_file_id = item.get("problem_file_id")
            similarity = item.get("similarity", 0.0)

            if similarity < self.similarity_threshold:
                continue

            result = {
                "problem_file_id": problem_file_id,
                "similarity": similarity,
                "type": "action_similarity",
                "evidence": f"Action similarity: {similarity:.3f}"
            }
            results.append(result)

        return results

    def _extract_key_tokens(self, error_message: str) -> List[str]:
        """Extract stable tokens from an error message."""

        import re


        uppercase_tokens = re.findall(r'\b[A-Z][A-Z_]+\b', error_message)


        underscore_tokens = re.findall(r'\b[a-zA-Z]+_[a-zA-Z_]+\b', error_message)


        quoted_tokens = re.findall(r'[\'"]([^\'"]+)[\'"]', error_message)

        all_tokens = uppercase_tokens + underscore_tokens + quoted_tokens


        unique_tokens = list(set(all_tokens))
        return unique_tokens[:10]

    def _deduplicate_and_sort(self, items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """Deduplicate results and keep the highest-scoring items."""
        if not items:
            return []

        seen_ids = set()
        unique_items = []
        for item in items:
            item_id = item.get("problem_file_id")
            if item_id and item_id not in seen_ids:
                seen_ids.add(item_id)
                unique_items.append(item)


        unique_items.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)

        return unique_items[:limit]

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Return basic retrieval index statistics."""
        return {
            "similarity_threshold": self.similarity_threshold,
            "kg_nodes": self.observation_kg.graph.number_of_nodes() if hasattr(self.observation_kg, 'graph') else 0,
            "kg_edges": self.observation_kg.graph.number_of_edges() if hasattr(self.observation_kg, 'graph') else 0
        }


class FusionRetriever:
    """Fuse ranked results produced by multiple hierarchical retrievers."""

    def __init__(self, retrievers: List[HierarchicalRetriever], weights: Dict[str, float] = None):
        """Initialize the fusion retriever.

        Args:
            retrievers: Retriever instances to combine
            weights: Optional per-view score weights
        """
        self.retrievers = retrievers
        self.weights = weights or {
            "task_sem": 0.3,
            "file_scope": 0.2,
            "error_sig": 0.25,
            "intent": 0.15,
            "action_sim": 0.1
        }

    def fuse_results(self, all_results: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Combine result lists into one weighted ranking.

        Args:
            all_results: Ranked result lists from individual retrievers

        Returns:
            Results sorted by weighted fusion score
        """
        if not all_results:
            return []


        all_items = []
        for results in all_results:
            all_items.extend(results)

        grouped_items = {}
        for item in all_items:
            item_id = item.get("problem_file_id")
            if not item_id:
                continue

            if item_id not in grouped_items:
                grouped_items[item_id] = {
                    "id": item_id,
                    "scores": {},
                    "details": item
                }


            item_type = item.get("type", "unknown")
            similarity = item.get("similarity", 0.0)
            grouped_items[item_id]["scores"][item_type] = similarity


        fused_items = []
        for item_id, item_data in grouped_items.items():
            fused_score = 0.0
            for score_type, score in item_data["scores"].items():
                weight = self.weights.get(score_type, 0.1)
                fused_score += score * weight

            fused_item = {
                "problem_file_id": item_id,
                "fused_score": fused_score,
                "component_scores": item_data["scores"],
                "details": item_data["details"]
            }
            fused_items.append(fused_item)


        fused_items.sort(key=lambda x: x["fused_score"], reverse=True)

        return fused_items
