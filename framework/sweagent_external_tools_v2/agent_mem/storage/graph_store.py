"""
Graph Store: Storage implementation for observation KG and belief graph.

MVP networkx  neo4j
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
import networkx as nx

from ..core.observation_kg import ObservationKG, TaskSubgraph, KGEdge, EdgeType
from ..core.belief_graph import BeliefGraph, AtomicBelief, AttemptBelief
from ..core.experience_models import FailureCardV2, PreventiveRuleV2, RepairPatternV2
from ..core.problem_file import ProblemFile


class GraphStore:
    """
    Unified graph store for both observation KG and belief graph.

    Provides persistence, query, and update operations.
    """

    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize graph store.

        Args:
            storage_dir: Directory for persistent storage. If None, uses in-memory only.
        """
        self.storage_dir = Path(storage_dir) if storage_dir else None
        self.observation_kg = ObservationKG()
        self.belief_graph = BeliefGraph()

        # NetworkX graphs for efficient graph operations
        self.kg_nx = nx.DiGraph()  # For observation KG
        self.belief_nx = nx.Graph()  # For belief graph (undirected for now)
        self.abstract_experiences: Dict[str, Dict[str, Any]] = {}
        self.attempt_summaries_v1: Dict[str, Dict[str, Any]] = {}
        self.failure_cards_v2: Dict[str, Dict[str, Any]] = {}
        self.repair_patterns_v2: Dict[str, Dict[str, Any]] = {}
        self.preventive_rules_v2: Dict[str, Dict[str, Any]] = {}
        self.compiler_cards_v21: Dict[str, Dict[str, Any]] = {}

        # Load existing data if storage directory exists
        if self.storage_dir and self.storage_dir.exists():
            self._load_from_storage()

    def _load_from_storage(self) -> None:
        """Load graphs from storage directory."""
        try:
            # Load observation KG
            kg_file = self.storage_dir / "observation_kg.json"
            if kg_file.exists():
                self.observation_kg = ObservationKG.load_from_file(str(kg_file))
                self._build_kg_nx()

            # Load belief graph
            belief_file = self.storage_dir / "belief_graph.json"
            if belief_file.exists():
                self.belief_graph = BeliefGraph.load_from_file(str(belief_file))
                self._build_belief_nx()

            # Load abstract experiences
            abstract_file = self.storage_dir / "abstract_experiences.json"
            if abstract_file.exists():
                data = json.loads(abstract_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.abstract_experiences = {
                        str(k): v for k, v in data.items() if isinstance(v, dict)
                    }
                elif isinstance(data, list):
                    self.abstract_experiences = {}
                    for item in data:
                        if isinstance(item, dict) and item.get("experience_id"):
                            self.abstract_experiences[str(item["experience_id"])] = item

            attempt_summary_file = self.storage_dir / "attempt_summaries_v1.json"
            if attempt_summary_file.exists():
                data = json.loads(attempt_summary_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.attempt_summaries_v1 = {
                        str(k): v for k, v in data.items() if isinstance(v, dict)
                    }
                elif isinstance(data, list):
                    self.attempt_summaries_v1 = {}
                    for item in data:
                        if isinstance(item, dict) and item.get("summary_id"):
                            self.attempt_summaries_v1[str(item["summary_id"])] = item

            # Load stage-4 structured experiences
            self.failure_cards_v2 = self._load_dict_file("failure_cards_v2.json", id_field="card_id")
            self.repair_patterns_v2 = self._load_dict_file("repair_patterns_v2.json", id_field="pattern_id")
            self.preventive_rules_v2 = self._load_dict_file("preventive_rules_v2.json", id_field="rule_id")
            self.compiler_cards_v21 = self._load_dict_file("compiler_cards_v21.json", id_field="card_id")

        except Exception as e:
            print(f"Warning: Failed to load from storage: {e}")

    def _load_dict_file(self, filename: str, *, id_field: str) -> Dict[str, Dict[str, Any]]:
        if not self.storage_dir:
            return {}
        path = self.storage_dir / filename
        if not path.exists():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        out: Dict[str, Dict[str, Any]] = {}
        if isinstance(raw, dict):
            for key, value in raw.items():
                if isinstance(value, dict):
                    out[str(key)] = dict(value)
            return out
        if isinstance(raw, list):
            for row in raw:
                if isinstance(row, dict) and row.get(id_field):
                    out[str(row[id_field])] = dict(row)
        return out

    def _build_kg_nx(self) -> None:
        """Build NetworkX graph from observation KG."""
        self.kg_nx.clear()

        # Add nodes
        for task_id, subgraph in self.observation_kg.task_subgraphs.items():
            for action_id, problem_file in subgraph.action_nodes.items():
                self.kg_nx.add_node(
                    action_id,
                    task_id=task_id,
                    action_type=problem_file.action_type.value,
                    outcome=problem_file.outcome.value,
                    intent_text=problem_file.intent_text,
                    **problem_file.metadata
                )

        # Add edges
        for task_id, subgraph in self.observation_kg.task_subgraphs.items():
            for edge in subgraph.edges:
                self.kg_nx.add_edge(
                    edge.source_id,
                    edge.target_id,
                    edge_type=edge.edge_type.value,
                    timestamp=edge.timestamp,
                    **edge.metadata
                )

    def _build_belief_nx(self) -> None:
        """Build NetworkX graph from belief graph."""
        self.belief_nx.clear()

        # Add atomic belief nodes
        for belief_id, belief in self.belief_graph.atomic_beliefs.items():
            self.belief_nx.add_node(
                belief_id,
                belief_type=belief.belief_type.value,
                status=belief.status.value,
                confidence=belief.confidence,
                **belief.metadata
            )

        # Add edges for conflicting beliefs
        for belief_id, conflicts in self.belief_graph.conflicting_beliefs.items():
            for conflict_id in conflicts:
                if conflict_id in self.belief_nx:
                    self.belief_nx.add_edge(belief_id, conflict_id, relation="conflicts")

    def save(self) -> None:
        """Save graphs to storage directory."""
        if not self.storage_dir:
            return

        # Ensure storage directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save observation KG
            kg_file = self.storage_dir / "observation_kg.json"
            self.observation_kg.save_to_file(str(kg_file))

            # Save belief graph
            belief_file = self.storage_dir / "belief_graph.json"
            self.belief_graph.save_to_file(str(belief_file))

            # Save abstract experiences
            abstract_file = self.storage_dir / "abstract_experiences.json"
            abstract_file.write_text(
                json.dumps(self.abstract_experiences, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (self.storage_dir / "attempt_summaries_v1.json").write_text(
                json.dumps(self.attempt_summaries_v1, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # Save stage-4 structured experiences
            (self.storage_dir / "failure_cards_v2.json").write_text(
                json.dumps(self.failure_cards_v2, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (self.storage_dir / "repair_patterns_v2.json").write_text(
                json.dumps(self.repair_patterns_v2, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (self.storage_dir / "preventive_rules_v2.json").write_text(
                json.dumps(self.preventive_rules_v2, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (self.storage_dir / "compiler_cards_v21.json").write_text(
                json.dumps(self.compiler_cards_v21, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # Save metadata
            metadata = {
                "last_saved": datetime.now(timezone.utc).isoformat(),
                "total_actions": sum(len(sg.action_nodes) for sg in self.observation_kg.task_subgraphs.values()),
                "total_beliefs": len(self.belief_graph.atomic_beliefs),
                "total_abstract_experiences": len(self.abstract_experiences),
                "total_attempt_summaries_v1": len(self.attempt_summaries_v1),
                "total_failure_cards_v2": len(self.failure_cards_v2),
                "total_repair_patterns_v2": len(self.repair_patterns_v2),
                "total_preventive_rules_v2": len(self.preventive_rules_v2),
                "total_compiler_cards_v21": len(self.compiler_cards_v21),
                "storage_version": "1.0",
            }
            metadata_file = self.storage_dir / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            print(f"Warning: Failed to save to storage: {e}")

    # Observation KG operations
    def add_action(self, problem_file: ProblemFile) -> str:
        """
        Add an action to the observation KG.

        Returns:
            action_id of the added action
        """
        # Create or get task subgraph
        task_id = problem_file.task_id
        subgraph = self.observation_kg.get_task_subgraph(task_id)
        if not subgraph:
            subgraph = TaskSubgraph(task_id=task_id)
            self.observation_kg.add_task_subgraph(subgraph)

        # Add action to subgraph
        subgraph.add_action(problem_file)
        # Keep reverse index in sync for all newly added actions.
        self.observation_kg.action_to_task[problem_file.action_id] = task_id

        # Update NetworkX graph
        self.kg_nx.add_node(
            problem_file.action_id,
            task_id=task_id,
            action_type=problem_file.action_type.value,
            outcome=problem_file.outcome.value,
            intent_text=problem_file.intent_text,
            **problem_file.metadata
        )

        return problem_file.action_id

    def add_edge(self, source_id: str, target_id: str, edge_type: EdgeType, metadata: Optional[Dict] = None) -> None:
        """
        Add an edge between two actions in the observation KG.
        """
        # Find task subgraph (both actions should be in the same task)
        source_info = self.observation_kg.get_action(source_id)
        target_info = self.observation_kg.get_action(target_id)

        if not source_info or not target_info:
            raise ValueError("Source or target action not found")

        source_action, source_task_id = source_info
        target_action, target_task_id = target_info

        if source_task_id != target_task_id:
            raise ValueError("Actions must be in the same task")

        # Create edge
        edge = KGEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            metadata=metadata or {}
        )

        # Add to subgraph
        subgraph = self.observation_kg.get_task_subgraph(source_task_id)
        if subgraph:
            subgraph.add_edge(edge)

        # Update NetworkX graph
        self.kg_nx.add_edge(
            source_id,
            target_id,
            edge_type=edge_type.value,
            timestamp=edge.timestamp,
            **edge.metadata
        )

    def get_action_successors(self, action_id: str, edge_type: Optional[EdgeType] = None) -> List[str]:
        """Get successor actions for a given action."""
        if action_id not in self.kg_nx:
            return []

        successors = list(self.kg_nx.successors(action_id))
        if edge_type:
            successors = [
                succ for succ in successors
                if self.kg_nx.edges[action_id, succ].get('edge_type') == edge_type.value
            ]

        return successors

    def get_action_predecessors(self, action_id: str, edge_type: Optional[EdgeType] = None) -> List[str]:
        """Get predecessor actions for a given action."""
        if action_id not in self.kg_nx:
            return []

        predecessors = list(self.kg_nx.predecessors(action_id))
        if edge_type:
            predecessors = [
                pred for pred in predecessors
                if self.kg_nx.edges[pred, action_id].get('edge_type') == edge_type.value
            ]

        return predecessors

    def find_similar_actions(self,
                            problem_file: ProblemFile,
                            max_results: int = 10,
                            embedding_view: str = "emb_task_sem") -> List[Tuple[str, ProblemFile, float]]:
        """
        Find similar actions using graph structure and node attributes.

        This is a simplified implementation for MVP.
        """
        # 1) Primary path: embedding similarity on observation KG.
        embedding_views = [
            embedding_view,
            "emb_task_sem",
            "emb_intent",
            "emb_error_sig",
            "emb_file_scope",
        ]
        embedding_candidates: Dict[str, Tuple[ProblemFile, float]] = {}
        for view in embedding_views:
            view_results = self.observation_kg.find_similar_actions(
                problem_file,
                embedding_view=view,
                threshold=0.25,
            )
            for action_id, action, score in view_results:
                prev = embedding_candidates.get(action_id)
                if prev is None or score > prev[1]:
                    embedding_candidates[action_id] = (action, score)

        if embedding_candidates:
            ranked = sorted(
                [(action_id, pair[0], pair[1]) for action_id, pair in embedding_candidates.items()],
                key=lambda x: x[2],
                reverse=True,
            )
            return ranked[:max_results]

        # 2) Fallback path: lexical similarity for cold-start.
        def _tokenize(text: str) -> Set[str]:
            return {tok for tok in text.lower().replace("\n", " ").split(" ") if tok}

        q_tokens = _tokenize(problem_file.intent_text or "")
        similar: List[Tuple[str, ProblemFile, float]] = []

        for task_id, subgraph in self.observation_kg.task_subgraphs.items():
            for action_id, action in subgraph.action_nodes.items():
                if action_id == problem_file.action_id:
                    continue

                a_tokens = _tokenize(action.intent_text or "")
                if not q_tokens and not a_tokens:
                    continue

                inter = len(q_tokens & a_tokens)
                union = max(1, len(q_tokens | a_tokens))
                similarity = inter / union

                if problem_file.action_type == action.action_type:
                    similarity += 0.2

                if similarity > 0.0:
                    similar.append((action_id, action, min(1.0, similarity)))

        similar.sort(key=lambda x: x[2], reverse=True)
        return similar[:max_results]

    # Belief graph operations
    def add_atomic_belief(self, belief: AtomicBelief) -> str:
        """Add an atomic belief to the belief graph."""
        belief_id = self.belief_graph.add_atomic_belief(belief)

        # Update NetworkX graph
        self.belief_nx.add_node(
            belief_id,
            belief_type=belief.belief_type.value,
            status=belief.status.value,
            confidence=belief.confidence,
            **belief.metadata
        )

        # Add conflict edges if any
        if belief_id in self.belief_graph.conflicting_beliefs:
            for conflict_id in self.belief_graph.conflicting_beliefs[belief_id]:
                if conflict_id in self.belief_nx:
                    self.belief_nx.add_edge(belief_id, conflict_id, relation="conflicts")

        return belief_id

    def add_attempt_belief(self, belief: AttemptBelief) -> None:
        """Add an attempt belief to the belief graph."""
        self.belief_graph.add_attempt_belief(belief)

    def get_relevant_beliefs(self,
                            env_cluster: Optional[str] = None,
                            repo_toolchain: Optional[str] = None,
                            action_type: Optional[str] = None,
                            max_results: int = 5) -> List[AtomicBelief]:
        """Get beliefs relevant to the given context."""
        beliefs = self.belief_graph.get_beliefs_for_context(
            env_cluster=env_cluster,
            repo_toolchain=repo_toolchain,
            action_type=action_type
        )
        return beliefs[:max_results]

    def update_belief_stats(self,
                           belief_id: str,
                           success_with: bool,
                           success_without: Optional[bool] = None) -> None:
        """Update statistics for a belief."""
        self.belief_graph.update_belief_stats(belief_id, success_with, success_without)

        # Update NetworkX node attributes
        if belief_id in self.belief_nx:
            belief = self.belief_graph.atomic_beliefs[belief_id]
            self.belief_nx.nodes[belief_id]["confidence"] = belief.confidence
            self.belief_nx.nodes[belief_id]["status"] = belief.status.value

    # Query operations
    def query_actions_by_outcome(self, outcome: str) -> List[ProblemFile]:
        """Query actions by outcome."""
        results = []

        for task_id, subgraph in self.observation_kg.task_subgraphs.items():
            for action_id, action in subgraph.action_nodes.items():
                if action.outcome.value == outcome:
                    results.append(action)

        return results

    def query_actions_by_type(self, action_type: str) -> List[ProblemFile]:
        """Query actions by action type."""
        results = []

        for task_id, subgraph in self.observation_kg.task_subgraphs.items():
            for action_id, action in subgraph.action_nodes.items():
                if action.action_type.value == action_type:
                    results.append(action)

        return results

    def get_last_action_id(self, task_id: str) -> Optional[str]:
        """Get the most recent action_id for a task, if any."""
        subgraph = self.observation_kg.get_task_subgraph(task_id)
        if not subgraph or not subgraph.action_nodes:
            return None

        def _parse_ts(action: ProblemFile) -> datetime:
            raw = action.timestamp
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            try:
                return datetime.fromisoformat(raw)
            except Exception:
                return datetime.min.replace(tzinfo=timezone.utc)

        latest = max(subgraph.action_nodes.values(), key=_parse_ts)
        return latest.action_id

    def get_task_actions(self, task_id: str) -> List[ProblemFile]:
        """Return all actions for a task, sorted by timestamp."""
        subgraph = self.observation_kg.get_task_subgraph(task_id)
        if not subgraph:
            return []

        def _parse_ts(action: ProblemFile) -> datetime:
            raw = action.timestamp
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            try:
                return datetime.fromisoformat(raw)
            except Exception:
                return datetime.min.replace(tzinfo=timezone.utc)

        return sorted(subgraph.action_nodes.values(), key=_parse_ts)

    # Abstract experience operations
    def upsert_abstract_experience(self, experience: Dict[str, Any]) -> str:
        """Insert or merge abstract experience by fingerprint."""
        now = datetime.now(timezone.utc).isoformat()
        candidate = dict(experience)
        candidate.setdefault("schema_version", "2.1")
        exp_id = str(candidate.get("experience_id") or "")
        if not exp_id:
            exp_id = f"abs_{len(self.abstract_experiences) + 1}_{int(datetime.now(timezone.utc).timestamp())}"
            candidate["experience_id"] = exp_id
        candidate.setdefault("created_at", now)
        candidate["last_updated"] = now
        candidate.setdefault("lifecycle_status", "new")
        candidate.setdefault("normalized_pattern_type", str(candidate.get("pattern_type", "")))
        candidate.setdefault("normalized_trigger_family", "generic_trigger")
        candidate.setdefault("normalized_advice_family", "generic_advice")
        candidate.setdefault("source_event_ids", [])
        candidate.setdefault("source_instance_id", "")
        candidate.setdefault("source_run_ids", [])
        candidate.setdefault("source_attempt_ids", [])
        candidate.setdefault("subproblem_type", "")
        candidate.setdefault("strategy_label", "")
        candidate.setdefault("prefer_actions", [])
        candidate.setdefault("avoid_actions", [])
        candidate.setdefault("applicability_scope", {})
        candidate.setdefault("source_action_ids", [])
        candidate.setdefault("source_action_chain", [])
        candidate.setdefault("links", {"related_experience_ids": []})
        variants = candidate.get("variant_texts")
        if not isinstance(variants, list):
            variants = []
        if candidate.get("abstracted_intent"):
            variants = [str(candidate.get("abstracted_intent"))] + [str(v) for v in variants]
        candidate["variant_texts"] = self._merge_unique_strings(variants, [], limit=50)
        if not str(candidate.get("family_id", "")).strip():
            candidate["family_id"] = self._build_family_id(candidate)
        quality = candidate.get("quality")
        if not isinstance(quality, dict):
            quality = {}
        quality.setdefault("item_confidence", float(candidate.get("confidence", 0.0)))
        quality.setdefault("support_count", int(candidate.get("support_count", 1)))
        quality.setdefault("use_count", 0)
        quality.setdefault("negative_feedback", 0)
        quality.setdefault("last_used_at", None)
        candidate["quality"] = quality
        metadata = candidate.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.setdefault("experience_polarity", "neutral")
        metadata.setdefault("promotion_state", "candidate")
        metadata.setdefault("evidence_stage", "trial_local")
        metadata.setdefault("official_eval_refs", [])
        metadata.setdefault("suppression_reason", "")
        metadata.setdefault("subproblem_type", str(candidate.get("subproblem_type", "")).strip())
        metadata.setdefault("strategy_label", str(candidate.get("strategy_label", "")).strip())
        metadata.setdefault("prefer_actions", list(candidate.get("prefer_actions") or []))
        metadata.setdefault("avoid_actions", list(candidate.get("avoid_actions") or []))
        metadata.setdefault("applicability_scope", dict(candidate.get("applicability_scope") or {}))
        candidate["metadata"] = metadata

        fingerprint = str(candidate.get("fingerprint") or self._experience_fingerprint(candidate))
        candidate["fingerprint"] = fingerprint
        existing_id = self._find_experience_merge_target(candidate)
        if existing_id:
            existing = self.abstract_experiences[existing_id]
            merged = self._merge_abstract_experience(existing, candidate, now=now)
            self.abstract_experiences[existing_id] = merged
            return existing_id

        self.abstract_experiences[exp_id] = candidate
        return exp_id

    def list_abstract_experiences(
        self, *, pattern_type: Optional[str] = None, max_results: int = 20
    ) -> List[Dict[str, Any]]:
        items = list(self.abstract_experiences.values())
        if pattern_type:
            items = [it for it in items if str(it.get("pattern_type", "")) == pattern_type]
        items.sort(
            key=lambda it: (
                float(it.get("confidence", 0.0)),
                int(it.get("support_count", 0)),
                str(it.get("last_updated", "")),
            ),
            reverse=True,
        )
        return [dict(it) for it in items[:max(1, max_results)]]

    def query_abstract_experiences(
        self,
        *,
        query_text: str = "",
        error_type: Optional[str] = None,
        subproblem_type: str = "",
        strategy_label: str = "",
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        q_tokens = self._tokenize(query_text)
        et_norm = (error_type or "").lower().strip()
        subproblem_norm = str(subproblem_type or "").strip().lower()
        strategy_norm = str(strategy_label or "").strip().lower()
        scored: List[Tuple[float, Dict[str, Any]]] = []

        for exp in self.abstract_experiences.values():
            metadata = exp.get("metadata") if isinstance(exp.get("metadata"), dict) else {}
            if str(metadata.get("promotion_state", "candidate")).strip().lower() == "suppressed":
                continue
            body = " ".join(
                [
                    str(exp.get("pattern_type", "")),
                    str(exp.get("abstracted_intent", "")),
                    " ".join(exp.get("success_conditions", []) or []),
                    " ".join(exp.get("failure_avoidance", []) or []),
                ]
            )
            e_tokens = self._tokenize(body)
            lexical = 0.0
            if q_tokens and e_tokens:
                inter = len(q_tokens & e_tokens)
                union = max(1, len(q_tokens | e_tokens))
                lexical = inter / union

            confidence = float(exp.get("confidence", 0.0))
            support = min(1.0, float(exp.get("support_count", 0)) / 5.0)
            err_bonus = 0.0
            if et_norm:
                avoids = " ".join(exp.get("failure_avoidance", []) or []).lower()
                if et_norm in avoids or et_norm in str(exp.get("pattern_type", "")).lower():
                    err_bonus = 0.25
            context_bonus = 0.0
            exp_subproblem = str(
                exp.get("subproblem_type")
                or metadata.get("subproblem_type")
                or ""
            ).strip().lower()
            exp_strategy = str(
                exp.get("strategy_label")
                or metadata.get("strategy_label")
                or ""
            ).strip().lower()
            if subproblem_norm and exp_subproblem == subproblem_norm:
                context_bonus += 0.18
            if strategy_norm and exp_strategy == strategy_norm:
                context_bonus += 0.14
            critical_bonus = 0.0
            critical = metadata.get("critical_signal") if isinstance(metadata.get("critical_signal"), dict) else {}
            critical_error = str(critical.get("error_type", "")).lower()
            critical_module = str(critical.get("critical_module", "")).lower()
            if et_norm and critical_error:
                if et_norm in critical_error or critical_error in et_norm:
                    critical_bonus = 0.28
            if et_norm and critical_module and critical_module in {"planning", "action", "system"}:
                critical_bonus = max(critical_bonus, 0.08)
            promotion_state = str(metadata.get("promotion_state", "candidate")).strip().lower()
            evidence_stage = str(metadata.get("evidence_stage", "trial_local")).strip().lower()
            experience_polarity = str(metadata.get("experience_polarity", "neutral")).strip().lower()
            state_bonus = 0.0
            if promotion_state == "promoted":
                state_bonus += 0.18
            elif promotion_state == "candidate":
                state_bonus -= 0.08
            if evidence_stage == "official_eval":
                state_bonus += 0.1
            elif evidence_stage == "submission":
                state_bonus -= 0.08
            if experience_polarity == "positive":
                state_bonus += 0.06
            elif experience_polarity == "negative":
                state_bonus += 0.04 if et_norm else 0.01

            score = (
                0.45 * lexical
                + 0.35 * confidence
                + 0.2 * support
                + err_bonus
                + critical_bonus
                + state_bonus
                + context_bonus
            )
            scored.append((score, exp))

        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[Dict[str, Any]] = []
        for score, exp in scored[: max(1, max_results)]:
            row = dict(exp)
            row["score"] = round(float(score), 6)
            metadata = exp.get("metadata") if isinstance(exp.get("metadata"), dict) else {}
            critical = metadata.get("critical_signal") if isinstance(metadata.get("critical_signal"), dict) else {}
            if critical:
                row["critical_alignment_score"] = round(
                    0.28 if et_norm and et_norm in str(critical.get("error_type", "")).lower() else 0.08,
                    6,
                )
            out.append(row)
        return out

    # Compiler card operations
    def upsert_compiler_card_v21(self, payload: Dict[str, Any]) -> str:
        now = datetime.now(timezone.utc).isoformat()
        row = dict(payload)
        card_id = str(row.get("card_id") or "").strip()
        if not card_id:
            card_id = f"card_{len(self.compiler_cards_v21) + 1}_{int(datetime.now(timezone.utc).timestamp())}"
            row["card_id"] = card_id
        row.setdefault("schema_version", "1.0")
        row.setdefault("promotion_state", "candidate")
        row.setdefault("created_at", now)
        row["last_updated"] = now
        row["source_object_ids"] = list(dict.fromkeys([str(x).strip() for x in (row.get("source_object_ids") or []) if str(x).strip()]))
        row["evidence_refs"] = list(dict.fromkeys([str(x).strip() for x in (row.get("evidence_refs") or []) if str(x).strip()]))
        row["metadata"] = dict(row.get("metadata") or {})
        existing = self.compiler_cards_v21.get(card_id)
        if existing:
            merged = dict(existing)
            merged.update(row)
            merged["source_object_ids"] = list(
                dict.fromkeys(
                    [str(x).strip() for x in (existing.get("source_object_ids") or []) if str(x).strip()]
                    + row["source_object_ids"]
                )
            )
            merged["evidence_refs"] = list(
                dict.fromkeys(
                    [str(x).strip() for x in (existing.get("evidence_refs") or []) if str(x).strip()]
                    + row["evidence_refs"]
                )
            )
            merged_meta = dict(existing.get("metadata") or {})
            merged_meta.update(row["metadata"])
            merged["metadata"] = merged_meta
            self.compiler_cards_v21[card_id] = merged
            return card_id
        self.compiler_cards_v21[card_id] = row
        return card_id

    def list_compiler_cards_v21(
        self,
        *,
        max_results: int = 20,
        promotion_states: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        rows = list(self.compiler_cards_v21.values())
        if promotion_states:
            allowed = {str(x).strip().lower() for x in promotion_states if str(x).strip()}
            rows = [row for row in rows if str(row.get("promotion_state") or "").strip().lower() in allowed]
        rows.sort(
            key=lambda row: (
                float(row.get("confidence", 0.0) or 0.0),
                str(row.get("last_updated") or ""),
            ),
            reverse=True,
        )
        return rows[:max_results]

    def list_candidate_compiler_cards(
        self,
        *,
        instance_id: str,
        run_id: str = "",
        attempt_id: str = "",
        max_results: int = 200,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for card in self.compiler_cards_v21.values():
            state = str(card.get("promotion_state") or "").strip().lower()
            if state in {"suppressed", "deprecated"}:
                continue
            if str(card.get("instance_id") or "").strip() != str(instance_id or "").strip():
                continue
            if run_id and str(card.get("run_id") or "").strip() != str(run_id or "").strip():
                continue
            if attempt_id and str(card.get("attempt_id") or "").strip() != str(attempt_id or "").strip():
                continue
            rows.append(card)
        rows.sort(key=lambda row: float(row.get("confidence", 0.0) or 0.0), reverse=True)
        return rows[:max_results]

    def query_compiler_cards_v21(
        self,
        *,
        query_text: str,
        query_type: str,
        max_results: int = 4,
        promotion_states: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        allowed = None
        if promotion_states:
            allowed = {str(x).strip().lower() for x in promotion_states if str(x).strip()}
        q_tokens = {
            tok
            for tok in re.split(r"[^a-zA-Z0-9_./-]+", str(query_text or "").lower())
            if tok
        }
        ranked: List[Tuple[float, Dict[str, Any]]] = []
        for row in self.compiler_cards_v21.values():
            state = str(row.get("promotion_state") or "").strip().lower()
            if allowed is not None and state not in allowed:
                continue
            if state in {"suppressed", "deprecated", "disabled"}:
                continue
            if str(row.get("v2_state") or "").strip().lower() == "disabled":
                continue
            card_type = str(row.get("card_type") or "")
            text = " ".join(
                [
                    str(row.get("hint") or ""),
                    str(row.get("recommendation") or ""),
                    str(row.get("card_type") or ""),
                    str(row.get("family_id") or ""),
                ]
            ).lower()
            row_tokens = {
                tok for tok in re.split(r"[^a-zA-Z0-9_./-]+", text) if tok
            }
            overlap = len(q_tokens & row_tokens)
            score = float(row.get("confidence", 0.0) or 0.0)
            score += min(0.25, 0.04 * overlap)
            if query_type == "planning" and card_type == "PlanHintCard":
                score += 0.08
            if query_type == "planning" and card_type == "SuccessPathCard":
                score += 0.12
            if card_type == "BugInvariantCard":
                score += 0.15
            if card_type == "BugAntiPatternCard":
                score += 0.15
            if query_type in {"error_recovery", "test_failure_fix"} and card_type in {"RetryHintCard", "SubtaskRiskCard"}:
                score += 0.08
            if query_type in {"planning", "error_recovery", "test_failure_fix", "regression_guard"} and card_type == "TimeoutGovernanceCard":
                score += 0.1
            ranked.append((score, row))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [dict(row) for _, row in ranked[:max_results]]

    def update_compiler_card_state(
        self,
        card_id: str,
        *,
        promotion_state: str,
        eval_ref: str = "",
        reason: str = "",
    ) -> bool:
        row = self.compiler_cards_v21.get(str(card_id))
        if not row:
            return False
        updated = dict(row)
        updated["promotion_state"] = promotion_state
        updated["last_updated"] = datetime.now(timezone.utc).isoformat()
        metadata = dict(updated.get("metadata") or {})
        if eval_ref:
            refs = [str(x).strip() for x in (metadata.get("official_eval_refs") or []) if str(x).strip()]
            refs.append(str(eval_ref).strip())
            metadata["official_eval_refs"] = list(dict.fromkeys(refs))
        if reason:
            metadata["governance_reason"] = reason
        updated["metadata"] = metadata
        self.compiler_cards_v21[str(card_id)] = updated
        return True

    def adjust_card(
        self,
        card_id: str,
        *,
        delta_confidence: float = 0.0,
        demote_if_below: float = 0.0,
        reason: str = "",
    ) -> bool:
        """Adjust card confidence using local effectiveness feedback.

        - `delta_confidence` is clamped to the `[0, 1]` range.
        - `demote_if_below` disables cards that cross the threshold.
        - `reason` is persisted in governance metadata.
        """
        row = self.compiler_cards_v21.get(str(card_id))
        if not row:
            return False
        updated = dict(row)
        try:
            current_conf = float(updated.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            current_conf = 0.0
        new_conf = max(0.0, min(1.0, current_conf + float(delta_confidence or 0.0)))
        updated["confidence"] = new_conf
        updated["last_updated"] = datetime.now(timezone.utc).isoformat()
        metadata = dict(updated.get("metadata") or {})
        history = list(metadata.get("v2_confidence_history") or [])
        history.append(
            {
                "ts": updated["last_updated"],
                "from": current_conf,
                "to": new_conf,
                "reason": reason,
            }
        )
        metadata["v2_confidence_history"] = history[-20:]
        if reason:
            metadata["v2_governance_reason"] = reason
        if demote_if_below and new_conf < float(demote_if_below):
            updated["v2_state"] = "disabled"
            metadata["v2_disabled_reason"] = reason or "confidence_below_threshold"
        updated["metadata"] = metadata
        self.compiler_cards_v21[str(card_id)] = updated
        return True

    def upsert_attempt_summary_v1(self, summary: Dict[str, Any]) -> str:
        now = datetime.now(timezone.utc).isoformat()
        row = dict(summary)
        row.setdefault("schema_version", "1.0")
        summary_id = str(row.get("summary_id") or "").strip()
        existing_id = self._find_attempt_summary(
            instance_id=str(row.get("instance_id") or ""),
            run_id=str(row.get("run_id") or ""),
            attempt_id=str(row.get("attempt_id") or ""),
        )
        if not summary_id and existing_id:
            summary_id = existing_id
        if not summary_id:
            summary_id = f"sum_{len(self.attempt_summaries_v1) + 1}_{int(datetime.now(timezone.utc).timestamp())}"
        row["summary_id"] = summary_id
        row.setdefault("created_at", now)
        row["last_updated"] = now
        if existing_id and existing_id in self.attempt_summaries_v1:
            current = dict(self.attempt_summaries_v1[existing_id])
            current.update({k: v for k, v in row.items() if v not in (None, "", [], {})})
            row = current
        self.attempt_summaries_v1[summary_id] = row
        return summary_id

    def get_latest_attempt_summary(
        self,
        *,
        instance_id: str,
        exclude_attempt_id: str = "",
    ) -> Optional[Dict[str, Any]]:
        instance_norm = str(instance_id or "").strip()
        exclude_attempt_norm = str(exclude_attempt_id or "").strip()
        if not instance_norm:
            return None
        rows: List[Dict[str, Any]] = []
        for row in self.attempt_summaries_v1.values():
            if str(row.get("instance_id") or "").strip() != instance_norm:
                continue
            if exclude_attempt_norm and str(row.get("attempt_id") or "").strip() == exclude_attempt_norm:
                continue
            rows.append(dict(row))
        if not rows:
            return None
        rows.sort(
            key=lambda item: (
                str(item.get("last_updated", "")),
                str(item.get("created_at", "")),
            ),
            reverse=True,
        )
        return rows[0]

    def update_attempt_summary_outcome(
        self,
        *,
        instance_id: str,
        run_id: str = "",
        attempt_id: str = "",
        outcome: str,
        eval_ref: str = "",
    ) -> Optional[str]:
        summary_id = self._find_attempt_summary(
            instance_id=instance_id,
            run_id=run_id,
            attempt_id=attempt_id,
        )
        if not summary_id:
            return None
        row = dict(self.attempt_summaries_v1.get(summary_id, {}))
        if not row:
            return None
        row["final_outcome"] = str(outcome or "").strip() or row.get("final_outcome", "")
        if eval_ref:
            refs = row.get("official_eval_refs") if isinstance(row.get("official_eval_refs"), list) else []
            row["official_eval_refs"] = self._merge_unique_strings(refs, [eval_ref], limit=20)
        row["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.attempt_summaries_v1[summary_id] = row
        return summary_id

    def list_abstract_experiences_for_instance(
        self,
        instance_id: str,
        *,
        max_results: int = 50,
    ) -> List[Dict[str, Any]]:
        instance_norm = str(instance_id or "").strip()
        if not instance_norm:
            return []
        rows: List[Dict[str, Any]] = []
        for exp in self.abstract_experiences.values():
            metadata = exp.get("metadata") if isinstance(exp.get("metadata"), dict) else {}
            source_instance_id = str(exp.get("source_instance_id") or metadata.get("source_instance_id") or "").strip()
            if source_instance_id != instance_norm:
                continue
            rows.append(dict(exp))
        rows.sort(key=lambda row: str(row.get("last_updated", "")), reverse=True)
        return rows[: max(1, max_results)]

    def list_candidate_abstract_experiences(
        self,
        *,
        instance_id: str = "",
        run_id: str = "",
        attempt_id: str = "",
        trace_id: str = "",
        max_results: int = 200,
    ) -> List[Dict[str, Any]]:
        instance_norm = str(instance_id or "").strip()
        run_norm = str(run_id or "").strip()
        attempt_norm = str(attempt_id or "").strip()
        trace_norm = str(trace_id or "").strip()

        rows: List[Dict[str, Any]] = []
        for exp in self.abstract_experiences.values():
            metadata = exp.get("metadata") if isinstance(exp.get("metadata"), dict) else {}
            if str(metadata.get("promotion_state", "candidate")).strip().lower() != "candidate":
                continue

            exp_instance = str(exp.get("source_instance_id") or metadata.get("source_instance_id") or "").strip()
            run_ids = [str(x).strip() for x in (exp.get("source_run_ids") or metadata.get("source_run_ids") or []) if str(x).strip()]
            attempt_ids = [str(x).strip() for x in (exp.get("source_attempt_ids") or metadata.get("source_attempt_ids") or []) if str(x).strip()]
            event_ids = [str(x).strip() for x in (exp.get("source_event_ids") or metadata.get("source_event_ids") or []) if str(x).strip()]

            matched = False
            if instance_norm and exp_instance == instance_norm:
                matched = True
            if run_norm and run_norm in run_ids:
                matched = True
            if attempt_norm and attempt_norm in attempt_ids:
                matched = True
            if trace_norm and trace_norm in event_ids:
                matched = True
            if not matched:
                continue
            rows.append(dict(exp))

        rows.sort(key=lambda row: str(row.get("last_updated", "")), reverse=True)
        return rows[: max(1, max_results)]

    def append_official_eval_ref(self, experience_id: str, eval_ref: str) -> bool:
        row = self.abstract_experiences.get(str(experience_id))
        if not row:
            return False
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        refs = metadata.get("official_eval_refs") if isinstance(metadata.get("official_eval_refs"), list) else []
        eval_ref_text = str(eval_ref or "").strip()
        if eval_ref_text:
            refs = self._merge_unique_strings(refs, [eval_ref_text], limit=50)
        metadata["official_eval_refs"] = refs
        row["metadata"] = metadata
        row["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.abstract_experiences[str(experience_id)] = row
        return True

    def mark_experience_promoted(self, experience_id: str, eval_ref: str = "") -> bool:
        row = self.abstract_experiences.get(str(experience_id))
        if not row:
            return False
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        metadata["promotion_state"] = "promoted"
        metadata["evidence_stage"] = "official_eval"
        if str(metadata.get("experience_polarity", "")).strip().lower() in {"", "neutral"}:
            metadata["experience_polarity"] = "positive"
        row["metadata"] = metadata
        self.abstract_experiences[str(experience_id)] = row
        if eval_ref:
            self.append_official_eval_ref(str(experience_id), eval_ref)
        return True

    def mark_experience_suppressed(self, experience_id: str, eval_ref: str = "", reason: str = "") -> bool:
        row = self.abstract_experiences.get(str(experience_id))
        if not row:
            return False
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        metadata["promotion_state"] = "suppressed"
        metadata["evidence_stage"] = "official_eval"
        metadata["suppression_reason"] = str(reason or "").strip()
        row["metadata"] = metadata
        self.abstract_experiences[str(experience_id)] = row
        if eval_ref:
            self.append_official_eval_ref(str(experience_id), eval_ref)
        return True

    def _find_experience_by_fingerprint(self, fingerprint: str) -> Optional[str]:
        for exp_id, exp in self.abstract_experiences.items():
            if str(exp.get("fingerprint", "")) == fingerprint:
                return exp_id
        return None

    def _find_experience_merge_target(self, candidate: Dict[str, Any]) -> Optional[str]:
        family_id = str(candidate.get("family_id", "")).strip()
        advice_family = str(candidate.get("normalized_advice_family", "")).strip()
        if family_id:
            best_id: Optional[str] = None
            best_score = 0.0
            for exp_id, exp in self.abstract_experiences.items():
                if str(exp.get("family_id", "")).strip() != family_id:
                    continue
                same_advice = str(exp.get("normalized_advice_family", "")).strip() == advice_family
                score = self._experience_merge_similarity(exp, candidate)
                if same_advice:
                    score = max(score, 1.0)
                if score > best_score:
                    best_id = exp_id
                    best_score = score
            if best_id and best_score >= 0.72:
                return best_id
        return self._find_experience_by_fingerprint(str(candidate.get("fingerprint", "")))

    def _build_family_id(self, exp: Dict[str, Any]) -> str:
        parts = [
            str(exp.get("normalized_pattern_type", "") or exp.get("pattern_type", "")).strip().lower(),
            str(exp.get("normalized_trigger_family", "")).strip().lower(),
            str(exp.get("normalized_advice_family", "")).strip().lower(),
        ]
        parts = [re.sub(r"[^a-z0-9_]+", "_", p).strip("_") or "unknown" for p in parts]
        return "__".join(parts)

    def _experience_merge_similarity(self, existing: Dict[str, Any], incoming: Dict[str, Any]) -> float:
        left = " ".join(
            [
                str(existing.get("abstracted_intent", "")),
                " ".join(existing.get("success_conditions", []) or []),
                " ".join(existing.get("failure_avoidance", []) or []),
            ]
        )
        right = " ".join(
            [
                str(incoming.get("abstracted_intent", "")),
                " ".join(incoming.get("success_conditions", []) or []),
                " ".join(incoming.get("failure_avoidance", []) or []),
            ]
        )
        left_tokens = self._tokenize(left)
        right_tokens = self._tokenize(right)
        if not left_tokens or not right_tokens:
            return 0.0
        inter = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        return inter / max(1, union)

    @staticmethod
    def _merge_abstract_experience(
        existing: Dict[str, Any], incoming: Dict[str, Any], *, now: str
    ) -> Dict[str, Any]:
        merged = dict(existing)
        merged["last_updated"] = now
        merged["support_count"] = int(existing.get("support_count", 1)) + int(
            incoming.get("support_count", 1)
        )
        merged["confidence"] = max(float(existing.get("confidence", 0.0)), float(incoming.get("confidence", 0.0)))
        merged["lifecycle_status"] = incoming.get("lifecycle_status") or existing.get("lifecycle_status", "new")

        def _merge_unique(a: Any, b: Any, limit: int = 200) -> List[str]:
            out: List[str] = []
            seen = set()
            for source in (a or [], b or []):
                if not isinstance(source, list):
                    continue
                for item in source:
                    s = str(item).strip()
                    if not s:
                        continue
                    k = s.lower()
                    if k in seen:
                        continue
                    seen.add(k)
                    out.append(s)
                    if len(out) >= limit:
                        return out
            return out

        merged["evidence_refs"] = _merge_unique(existing.get("evidence_refs"), incoming.get("evidence_refs"), 120)
        merged["source_task_ids"] = _merge_unique(existing.get("source_task_ids"), incoming.get("source_task_ids"), 50)
        merged["source_event_ids"] = _merge_unique(existing.get("source_event_ids"), incoming.get("source_event_ids"), 120)
        merged["source_run_ids"] = _merge_unique(existing.get("source_run_ids"), incoming.get("source_run_ids"), 30)
        merged["source_attempt_ids"] = _merge_unique(
            existing.get("source_attempt_ids"),
            incoming.get("source_attempt_ids"),
            30,
        )
        merged["source_action_ids"] = _merge_unique(existing.get("source_action_ids"), incoming.get("source_action_ids"), 60)
        merged["source_action_chain"] = _merge_unique(existing.get("source_action_chain"), incoming.get("source_action_chain"), 20)
        merged["source_instance_id"] = incoming.get("source_instance_id") or existing.get("source_instance_id", "")
        merged["subproblem_type"] = incoming.get("subproblem_type") or existing.get("subproblem_type", "")
        merged["strategy_label"] = incoming.get("strategy_label") or existing.get("strategy_label", "")
        merged["prefer_actions"] = _merge_unique(existing.get("prefer_actions"), incoming.get("prefer_actions"), 20)
        merged["avoid_actions"] = _merge_unique(existing.get("avoid_actions"), incoming.get("avoid_actions"), 20)
        scope_existing = existing.get("applicability_scope") if isinstance(existing.get("applicability_scope"), dict) else {}
        scope_incoming = incoming.get("applicability_scope") if isinstance(incoming.get("applicability_scope"), dict) else {}
        merged["applicability_scope"] = {**scope_existing, **scope_incoming}
        merged["normalized_pattern_type"] = (
            incoming.get("normalized_pattern_type")
            or existing.get("normalized_pattern_type")
            or incoming.get("pattern_type")
            or existing.get("pattern_type")
            or "generic_pattern"
        )
        merged["normalized_trigger_family"] = (
            incoming.get("normalized_trigger_family")
            or existing.get("normalized_trigger_family")
            or "generic_trigger"
        )
        merged["normalized_advice_family"] = (
            incoming.get("normalized_advice_family")
            or existing.get("normalized_advice_family")
            or "generic_advice"
        )
        merged["family_id"] = (
            incoming.get("family_id")
            or existing.get("family_id")
            or "__".join(
                [
                    str(merged.get("normalized_pattern_type", "generic_pattern")),
                    str(merged.get("normalized_trigger_family", "generic_trigger")),
                    str(merged.get("normalized_advice_family", "generic_advice")),
                ]
            )
        )
        merged["success_conditions"] = _merge_unique(
            existing.get("success_conditions"), incoming.get("success_conditions"), 20
        )
        merged["failure_avoidance"] = _merge_unique(
            existing.get("failure_avoidance"), incoming.get("failure_avoidance"), 20
        )
        merged["variant_texts"] = _merge_unique(
            existing.get("variant_texts") or [existing.get("abstracted_intent", "")],
            incoming.get("variant_texts") or [incoming.get("abstracted_intent", "")],
            50,
        )
        links_existing = existing.get("links") if isinstance(existing.get("links"), dict) else {}
        links_incoming = incoming.get("links") if isinstance(incoming.get("links"), dict) else {}
        merged["links"] = {
            "related_experience_ids": _merge_unique(
                links_existing.get("related_experience_ids"),
                links_incoming.get("related_experience_ids"),
                40,
            )
        }
        quality_existing = existing.get("quality") if isinstance(existing.get("quality"), dict) else {}
        quality_incoming = incoming.get("quality") if isinstance(incoming.get("quality"), dict) else {}
        merged["quality"] = {
            "item_confidence": max(
                float(quality_existing.get("item_confidence", 0.0)),
                float(quality_incoming.get("item_confidence", 0.0)),
            ),
            "support_count": int(merged.get("support_count", 1)),
            "use_count": int(quality_existing.get("use_count", 0)),
            "negative_feedback": int(quality_existing.get("negative_feedback", 0)),
            "last_used_at": quality_existing.get("last_used_at"),
        }
        metadata_existing = existing.get("metadata") if isinstance(existing.get("metadata"), dict) else {}
        metadata_incoming = incoming.get("metadata") if isinstance(incoming.get("metadata"), dict) else {}
        metadata_merged = dict(metadata_existing)
        for key, value in metadata_incoming.items():
            if key == "official_eval_refs":
                metadata_merged[key] = GraphStore._merge_unique_strings(
                    metadata_existing.get(key, []),
                    value if isinstance(value, list) else [],
                    limit=50,
                )
            elif key == "source_event_ids":
                metadata_merged[key] = GraphStore._merge_unique_strings(
                    metadata_existing.get(key, []),
                    value if isinstance(value, list) else [],
                    limit=100,
                )
            elif value not in (None, "", [], {}):
                metadata_merged[key] = value
        metadata_merged.setdefault("experience_polarity", "neutral")
        metadata_merged.setdefault("promotion_state", "candidate")
        metadata_merged.setdefault("evidence_stage", "trial_local")
        metadata_merged.setdefault("official_eval_refs", [])
        metadata_merged.setdefault("suppression_reason", "")
        merged["metadata"] = metadata_merged
        return merged

    def _find_attempt_summary(self, *, instance_id: str, run_id: str, attempt_id: str) -> Optional[str]:
        instance_norm = str(instance_id or "").strip()
        run_norm = str(run_id or "").strip()
        attempt_norm = str(attempt_id or "").strip()
        best_id: Optional[str] = None
        best_score = -1
        for summary_id, row in self.attempt_summaries_v1.items():
            row_instance = str(row.get("instance_id") or "").strip()
            row_run = str(row.get("run_id") or "").strip()
            row_attempt = str(row.get("attempt_id") or "").strip()
            if instance_norm and row_instance != instance_norm:
                continue
            if attempt_norm and row_attempt != attempt_norm:
                continue
            score = 0
            if instance_norm and row_instance == instance_norm:
                score += 1
            if attempt_norm and row_attempt == attempt_norm:
                score += 5
            if run_norm and row_run == run_norm:
                score += 2
            if score > best_score and score > 0:
                best_id = summary_id
                best_score = score
        return best_id

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        return {tok for tok in re.findall(r"[a-zA-Z0-9_]+", (text or "").lower()) if tok}

    @staticmethod
    def _merge_unique_strings(a: Any, b: Any, *, limit: int) -> List[str]:
        out: List[str] = []
        seen = set()
        for source in (a or [], b or []):
            if not isinstance(source, list):
                continue
            for item in source:
                s = str(item).strip()
                if not s:
                    continue
                key = s.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(s)
                if len(out) >= limit:
                    return out
        return out

    def _experience_fingerprint(self, exp: Dict[str, Any]) -> str:
        metadata = exp.get("metadata") if isinstance(exp.get("metadata"), dict) else {}
        tool_sequence = metadata.get("tool_sequence") if isinstance(metadata.get("tool_sequence"), list) else []
        changed_file_pattern = (
            metadata.get("changed_file_pattern")
            if isinstance(metadata.get("changed_file_pattern"), list)
            else []
        )
        body = "|".join(
            [
                str(exp.get("pattern_type", "")),
                str(exp.get("abstracted_intent", "")),
                ",".join(exp.get("success_conditions", []) or []),
                ",".join(exp.get("failure_avoidance", []) or []),
                str(metadata.get("error_signature", "")),
                ",".join(str(x) for x in tool_sequence),
                ",".join(str(x) for x in changed_file_pattern),
                str(metadata.get("test_signal", "")),
                str(exp.get("source_instance_id", "")),
            ]
        )
        parts = [tok for tok in re.findall(r"[a-zA-Z0-9_]+", body.lower()) if tok]
        return "_".join(parts[:60]) or f"exp_{len(self.abstract_experiences)+1}"

    # Stage-4 structured experience operations
    def upsert_failure_card_v2(self, card: Dict[str, Any]) -> str:
        payload = FailureCardV2.from_dict(card).to_dict()
        card_id = str(payload["card_id"])
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        metadata.setdefault("source_event_ids", [])
        metadata.setdefault("source_instance_id", payload.get("instance_id") or "")
        metadata.setdefault("source_run_ids", [payload.get("run_id")] if payload.get("run_id") else [])
        links = metadata.get("links") if isinstance(metadata.get("links"), dict) else {}
        links.setdefault("repair_pattern_ids", [])
        metadata["links"] = links
        payload["metadata"] = metadata
        payload["failure_class"] = self._normalize_failure_class(payload)
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        if card_id in self.failure_cards_v2:
            existing = dict(self.failure_cards_v2[card_id])
            for key, value in payload.items():
                if key in {"created_at"} and existing.get(key):
                    continue
                existing[key] = value
            payload = existing
        self.failure_cards_v2[card_id] = payload
        return card_id

    def list_failure_cards_v2(
        self,
        *,
        status: Optional[str] = None,
        max_results: int = 20,
    ) -> List[Dict[str, Any]]:
        rows = list(self.failure_cards_v2.values())
        if status:
            rows = [row for row in rows if str(row.get("status", "")).lower() == status.lower()]
        rows.sort(
            key=lambda row: (
                float(row.get("confidence", 0.0)),
                str(row.get("updated_at", row.get("created_at", ""))),
            ),
            reverse=True,
        )
        return [dict(row) for row in rows[: max(1, max_results)]]

    def query_failure_cards_v2(
        self,
        *,
        query_text: str = "",
        error_type: Optional[str] = None,
        max_results: int = 5,
        include_infra: bool = False,
    ) -> List[Dict[str, Any]]:
        q_tokens = self._tokenize(query_text)
        et = (error_type or "").strip().lower()
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for row in self.failure_cards_v2.values():
            if not include_infra and str(row.get("failure_class", "")).strip().lower() == "infra_failure_card":
                continue
            sig = row.get("error_signature") or {}
            body = " ".join(
                [
                    str(sig.get("error_type", "")),
                    " ".join(row.get("candidate_fix_actions") or []),
                    " ".join(row.get("action_trace_snippet") or []),
                    " ".join(row.get("verification_commands") or []),
                ]
            )
            lexical = 0.0
            body_tokens = self._tokenize(body)
            if q_tokens and body_tokens:
                lexical = len(q_tokens & body_tokens) / max(1, len(q_tokens | body_tokens))

            et_bonus = 0.0
            row_error_type = str(sig.get("error_type", "")).lower()
            if et:
                if et in row_error_type or row_error_type in et:
                    et_bonus = 0.35
                elif et in body.lower():
                    et_bonus = 0.2
            rca_bonus = 0.1 if row.get("root_cause_nodes") else 0.0
            critical_bonus = 0.0
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            critical = metadata.get("critical_signal") if isinstance(metadata.get("critical_signal"), dict) else {}
            critical_alignment = (
                metadata.get("critical_alignment")
                if isinstance(metadata.get("critical_alignment"), dict)
                else {}
            )
            critical_error = str(critical.get("error_type", "") or critical_alignment.get("error_type", "")).lower()
            critical_module = str(critical.get("critical_module", "") or critical_alignment.get("error_module", "")).lower()
            if et and critical_error:
                if et in critical_error or critical_error in et:
                    critical_bonus = 0.32
            if critical_module in {"planning", "action", "system"}:
                critical_bonus = max(critical_bonus, 0.1)
            confidence = float(row.get("confidence", 0.0))
            score = 0.45 * lexical + 0.35 * confidence + et_bonus + rca_bonus + critical_bonus
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        result: List[Dict[str, Any]] = []
        for score, row in scored[: max(1, max_results)]:
            item = dict(row)
            item["score"] = round(float(score), 6)
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            critical = metadata.get("critical_signal") if isinstance(metadata.get("critical_signal"), dict) else {}
            critical_alignment = (
                metadata.get("critical_alignment")
                if isinstance(metadata.get("critical_alignment"), dict)
                else {}
            )
            critical_error = str(critical.get("error_type", "") or critical_alignment.get("error_type", "")).lower()
            alignment_score = 0.0
            if et and critical_error and (et in critical_error or critical_error in et):
                alignment_score = 0.32
            elif critical_error:
                alignment_score = 0.08
            item["critical_alignment_score"] = round(alignment_score, 6)
            result.append(item)
        return result

    @staticmethod
    def _normalize_failure_class(card: Dict[str, Any]) -> str:
        existing = str(card.get("failure_class", "")).strip().lower()
        if existing == "infra_failure_card":
            return existing
        sig = card.get("error_signature") if isinstance(card.get("error_signature"), dict) else {}
        error_type = str(sig.get("error_type", "")).strip().lower()
        if error_type in {"environment_error", "tool_timeout", "permission_error", "timeout"}:
            return "infra_failure_card"
        metadata = card.get("metadata") if isinstance(card.get("metadata"), dict) else {}
        body = " ".join(
            [
                str(error_type),
                " ".join(str(x) for x in (card.get("candidate_fix_actions") or [])),
                " ".join(str(x) for x in (card.get("action_trace_snippet") or [])),
                str(metadata.get("task_summary", "")),
            ]
        ).lower()
        infra_markers = (
            "insufficient balance",
            "dockerpullerror",
            "docker pull",
            "docker build",
            "docker daemon",
            "daemon unavailable",
            "no space left on device",
            "sigbus",
        )
        if any(marker in body for marker in infra_markers):
            return "infra_failure_card"
        if existing == "agent_failure_card":
            return existing
        return "agent_failure_card"

    def upsert_repair_pattern_v2(self, pattern: Dict[str, Any]) -> str:
        payload = RepairPatternV2.from_dict(pattern).to_dict()
        pattern_id = str(payload["pattern_id"])
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        metadata.setdefault("source_event_ids", [])
        metadata.setdefault("source_instance_id", "")
        metadata.setdefault("source_run_ids", [])
        payload["metadata"] = metadata
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        if pattern_id in self.repair_patterns_v2:
            existing = dict(self.repair_patterns_v2[pattern_id])
            for key, value in payload.items():
                if key in {"created_at"} and existing.get(key):
                    continue
                existing[key] = value
            payload = existing
        self.repair_patterns_v2[pattern_id] = payload
        return pattern_id

    def query_repair_patterns_v2(
        self,
        *,
        query_text: str = "",
        error_type: Optional[str] = None,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        q_tokens = self._tokenize(query_text)
        et = (error_type or "").strip().lower()
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for row in self.repair_patterns_v2.values():
            trigger = row.get("trigger_signature") or {}
            body = " ".join(
                [
                    str(row.get("fix_action_template", "")),
                    " ".join(row.get("expected_verification") or []),
                    str(trigger.get("error_type", "")),
                    str(trigger.get("error_stage", "")),
                    str(trigger.get("error_module", "")),
                ]
            )
            body_tokens = self._tokenize(body)
            lexical = 0.0
            if q_tokens and body_tokens:
                lexical = len(q_tokens & body_tokens) / max(1, len(q_tokens | body_tokens))
            et_bonus = 0.0
            trigger_error = str(trigger.get("error_type", "")).lower()
            if et:
                if et in trigger_error or trigger_error in et:
                    et_bonus = 0.35
                elif et in body.lower():
                    et_bonus = 0.2
            support_bonus = min(0.2, float(row.get("support", 1)) / 20.0)
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            critical_alignment = (
                metadata.get("critical_alignment")
                if isinstance(metadata.get("critical_alignment"), dict)
                else {}
            )
            critical_error = str(critical_alignment.get("error_type", "")).lower()
            critical_module = str(critical_alignment.get("error_module", "")).lower()
            critical_bonus = 0.0
            if et and critical_error:
                if et in critical_error or critical_error in et:
                    critical_bonus = 0.3
            if critical_module in {"planning", "action", "system"}:
                critical_bonus = max(critical_bonus, 0.1)
            score = (
                0.45 * lexical
                + 0.35 * float(row.get("confidence", 0.0))
                + et_bonus
                + support_bonus
                + critical_bonus
            )
            scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[Dict[str, Any]] = []
        for score, row in scored[: max(1, max_results)]:
            item = dict(row)
            item["score"] = round(float(score), 6)
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            critical_alignment = (
                metadata.get("critical_alignment")
                if isinstance(metadata.get("critical_alignment"), dict)
                else {}
            )
            critical_error = str(critical_alignment.get("error_type", "")).lower()
            alignment_score = 0.0
            if et and critical_error and (et in critical_error or critical_error in et):
                alignment_score = 0.3
            elif critical_error:
                alignment_score = 0.08
            item["critical_alignment_score"] = round(alignment_score, 6)
            out.append(item)
        return out

    def upsert_preventive_rule_v2(self, rule: Dict[str, Any]) -> str:
        payload = PreventiveRuleV2.from_dict(rule).to_dict()
        rule_id = str(payload["rule_id"])
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        metadata.setdefault("source_event_ids", [])
        metadata.setdefault("source_instance_id", "")
        metadata.setdefault("source_run_ids", [])
        payload["metadata"] = metadata
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        if rule_id in self.preventive_rules_v2:
            existing = dict(self.preventive_rules_v2[rule_id])
            for key, value in payload.items():
                if key in {"created_at"} and existing.get(key):
                    continue
                existing[key] = value
            payload = existing
        self.preventive_rules_v2[rule_id] = payload
        return rule_id

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the graph store."""
        kg_stats = self.observation_kg.get_statistics()
        belief_stats = self.belief_graph.get_statistics()

        # NetworkX statistics
        nx_stats = {
            "kg_nodes": self.kg_nx.number_of_nodes(),
            "kg_edges": self.kg_nx.number_of_edges(),
            "belief_nodes": self.belief_nx.number_of_nodes(),
            "belief_edges": self.belief_nx.number_of_edges(),
            "kg_is_connected": nx.is_weakly_connected(self.kg_nx) if self.kg_nx.number_of_nodes() > 0 else False,
        }

        return {
            "observation_kg": kg_stats,
            "belief_graph": belief_stats,
            "abstract_experiences": {
                "total": len(self.abstract_experiences),
                "top_pattern_types": self._top_pattern_types(),
            },
            "failure_cards_v2": {
                "total": len(self.failure_cards_v2),
                "resolved": sum(
                    1 for row in self.failure_cards_v2.values() if str(row.get("status", "")).lower() == "resolved"
                ),
            },
            "repair_patterns_v2": {
                "total": len(self.repair_patterns_v2),
            },
            "compiler_cards_v21": {
                "total": len(self.compiler_cards_v21),
                "promoted": sum(
                    1 for row in self.compiler_cards_v21.values() if str(row.get("promotion_state", "")).lower() == "promoted"
                ),
                "candidate": sum(
                    1 for row in self.compiler_cards_v21.values() if str(row.get("promotion_state", "")).lower() == "candidate"
                ),
            },
            "networkx": nx_stats,
            "storage_dir": str(self.storage_dir) if self.storage_dir else None,
        }

    def _top_pattern_types(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for exp in self.abstract_experiences.values():
            pt = str(exp.get("pattern_type", "unknown"))
            counts[pt] = counts.get(pt, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10])
