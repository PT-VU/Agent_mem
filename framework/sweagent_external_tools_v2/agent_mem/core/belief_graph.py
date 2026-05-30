"""
Belief Graph: Stores inferred experience rules from observation KG, supporting
revocable and updatable beliefs.

The graph stores attempt-level summaries and revocable atomic beliefs.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
from typing_extensions import TypedDict

from .problem_file import ProblemFile


class BeliefType(str, Enum):
    """Types of atomic beliefs."""
    WORKFLOW = "workflow"
    PITFALL = "pitfall"
    ENV_ADAPTATION = "env_adaptation"
class BeliefStatus(str, Enum):
    """Status of a belief in its lifecycle."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    PROMOTED_TO_PREFERENCE = "promoted_to_preference"
    PROMOTED_TO_CONSTRAINT = "promoted_to_constraint"
@dataclass
class ConditionSignature:
    """Condition signature for atomic beliefs."""
    env_cluster: Optional[str] = None
    repo_toolchain: Optional[str] = None
    action_type_pattern: Optional[str] = None  # action
    intent_pattern: Optional[str] = None  # intent
    file_scope_pattern: Optional[str] = None
    error_signature: Optional[str] = None
@dataclass
class BeliefRule:
    """Rule part of an atomic belief."""
    trigger: str  # WHEN/IF condition description
    recommend: Optional[str] = None  # DO action/parameter template
    avoid: Optional[str] = None  # NOT DO prohibited actions
    redundant: Optional[str] = None  # DO-NOT-REPEAT redundant actions


@dataclass
class BeliefStats:
    """Statistical evidence for a belief."""
    support_n: int = 0
    success_with: int = 0
    success_without: int = 0
    uplift: float = 0.0
    uncertainty: float = 0.0
    recent_window_metrics: Dict[str, Any] = field(default_factory=dict)
@dataclass
class AtomicBelief:
    """An evidence-backed conditional rule with lifecycle state."""
    belief_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    belief_type: BeliefType = BeliefType.WORKFLOW
    condition_signature: ConditionSignature = field(default_factory=ConditionSignature)
    rule: BeliefRule = field(default_factory=lambda: BeliefRule(trigger=""))
    stats: BeliefStats = field(default_factory=BeliefStats)
    confidence: float = 0.0
    evidence_refs: List[str] = field(default_factory=list)
    attribution_refs: List[str] = field(default_factory=list)  # RCA
    status: BeliefStatus = BeliefStatus.ACTIVE
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_stats(self,
                    success_with: bool,
                    success_without: Optional[bool] = None,
                    window_size: int = 100) -> None:
        """Update belief statistics with new evidence."""
        self.stats.support_n += 1

        if success_with:
            self.stats.success_with += 1

        if success_without is not None:
            if success_without:
                self.stats.success_without += 1

        # Calculate uplift
        total_with = self.stats.support_n
        if total_with > 0:
            success_rate_with = self.stats.success_with / total_with
            success_rate_without = self.stats.success_without / max(1, total_with)
            self.stats.uplift = success_rate_with - success_rate_without

        # Update confidence (simplified calculation)
        # Using support_n/20.0 instead of support_n/100.0 to reach confidence faster
        # In production, this should be a more sophisticated formula
        self.confidence = min(1.0, self.stats.support_n / 20.0) * self.stats.uplift

        self.last_updated = datetime.now(timezone.utc).isoformat()

    def should_promote_to_preference(self,
                                    min_support: int = 10,
                                    min_uplift: float = 0.1,
                                    min_confidence: float = 0.7) -> bool:
        """Check if belief should be promoted to preference."""
        return (self.stats.support_n >= min_support and
                self.stats.uplift >= min_uplift and
                self.confidence >= min_confidence and
                self.status == BeliefStatus.ACTIVE)

    def should_promote_to_constraint(self,
                                    min_support: int = 50,
                                    min_uplift: float = 0.3,
                                    min_confidence: float = 0.9) -> bool:
        """Check if belief should be promoted to constraint."""
        return (self.stats.support_n >= min_support and
                self.stats.uplift >= min_uplift and
                self.confidence >= min_confidence and
                self.status == BeliefStatus.PROMOTED_TO_PREFERENCE)

    def to_dict(self) -> Dict[str, Any]:
        """Convert atomic belief to dictionary."""
        return {
            "belief_id": self.belief_id,
            "belief_type": self.belief_type.value,
            "condition_signature": {
                "env_cluster": self.condition_signature.env_cluster,
                "repo_toolchain": self.condition_signature.repo_toolchain,
                "action_type_pattern": self.condition_signature.action_type_pattern,
                "intent_pattern": self.condition_signature.intent_pattern,
                "file_scope_pattern": self.condition_signature.file_scope_pattern,
                "error_signature": self.condition_signature.error_signature,
            },
            "rule": {
                "trigger": self.rule.trigger,
                "recommend": self.rule.recommend,
                "avoid": self.rule.avoid,
                "redundant": self.rule.redundant,
            },
            "stats": {
                "support_n": self.stats.support_n,
                "success_with": self.stats.success_with,
                "success_without": self.stats.success_without,
                "uplift": self.stats.uplift,
                "uncertainty": self.stats.uncertainty,
                "recent_window_metrics": self.stats.recent_window_metrics,
            },
            "confidence": self.confidence,
            "evidence_refs": self.evidence_refs,
            "attribution_refs": self.attribution_refs,
            "status": self.status.value,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AtomicBelief:
        """Create atomic belief from dictionary."""
        # Convert string enums
        if isinstance(data.get("belief_type"), str):
            data["belief_type"] = BeliefType(data["belief_type"])
        if isinstance(data.get("status"), str):
            data["status"] = BeliefStatus(data["status"])

        # Handle nested dataclasses
        if data.get("condition_signature"):
            data["condition_signature"] = ConditionSignature(**data["condition_signature"])
        if data.get("rule"):
            data["rule"] = BeliefRule(**data["rule"])
        if data.get("stats"):
            data["stats"] = BeliefStats(**data["stats"])

        return cls(**data)


@dataclass
class AttemptBelief:
    """Task-attempt summary used to derive reusable beliefs."""
    attempt_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    summary: str = ""
    failure_causal_chain: List[str] = field(default_factory=list)
    key_turning_points: List[str] = field(default_factory=list)
    attribution_results: Dict[str, Any] = field(default_factory=dict)
    reusable_workflow_points: List[str] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert attempt belief to dictionary."""
        return {
            "attempt_id": self.attempt_id,
            "task_id": self.task_id,
            "summary": self.summary,
            "failure_causal_chain": self.failure_causal_chain,
            "key_turning_points": self.key_turning_points,
            "attribution_results": self.attribution_results,
            "reusable_workflow_points": self.reusable_workflow_points,
            "evidence_refs": self.evidence_refs,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AttemptBelief:
        """Create attempt belief from dictionary."""
        return cls(**data)


class BeliefGraph:
    """Collection of attempt beliefs, atomic beliefs, and conflict links."""

    def __init__(self):
        self.attempt_beliefs: Dict[str, AttemptBelief] = {}  # attempt_id -> AttemptBelief
        self.atomic_beliefs: Dict[str, AtomicBelief] = {}  # belief_id -> AtomicBelief
        self.task_to_attempts: Dict[str, List[str]] = {}  # task_id -> list of attempt_ids
        self.conflicting_beliefs: Dict[str, List[str]] = {}  # belief_id -> list of conflicting belief_ids

    def add_attempt_belief(self, belief: AttemptBelief) -> None:
        """Add an attempt belief to the graph."""
        self.attempt_beliefs[belief.attempt_id] = belief

        # Update task-to-attempts mapping
        if belief.task_id not in self.task_to_attempts:
            self.task_to_attempts[belief.task_id] = []
        self.task_to_attempts[belief.task_id].append(belief.attempt_id)

    def add_atomic_belief(self, belief: AtomicBelief) -> str:
        """Add an atomic belief to the graph, returns belief_id."""
        # Check for conflicts with existing beliefs
        conflicts = self._find_conflicts(belief)
        if conflicts:
            self.conflicting_beliefs[belief.belief_id] = conflicts
            for conflict_id in conflicts:
                if conflict_id not in self.conflicting_beliefs:
                    self.conflicting_beliefs[conflict_id] = []
                self.conflicting_beliefs[conflict_id].append(belief.belief_id)

        self.atomic_beliefs[belief.belief_id] = belief
        return belief.belief_id

    def _find_conflicts(self, new_belief: AtomicBelief) -> List[str]:
        """Find conflicts between new belief and existing beliefs."""
        conflicts = []

        for belief_id, existing_belief in self.atomic_beliefs.items():
            if self._beliefs_conflict(new_belief, existing_belief):
                conflicts.append(belief_id)

        return conflicts

    def _beliefs_conflict(self, belief1: AtomicBelief, belief2: AtomicBelief) -> bool:
        """Check if two beliefs conflict."""
        # Simplified conflict detection
        # In reality, this would involve more sophisticated logic
        if (belief1.condition_signature.env_cluster == belief2.condition_signature.env_cluster and
            belief1.condition_signature.repo_toolchain == belief2.condition_signature.repo_toolchain and
            belief1.condition_signature.action_type_pattern == belief2.condition_signature.action_type_pattern):

            # Check if recommendations conflict
            if belief1.rule.recommend and belief2.rule.avoid:
                if belief1.rule.recommend in (belief2.rule.avoid or ""):
                    return True
            if belief2.rule.recommend and belief1.rule.avoid:
                if belief2.rule.recommend in (belief1.rule.avoid or ""):
                    return True

        return False

    def get_beliefs_for_context(self,
                               env_cluster: Optional[str] = None,
                               repo_toolchain: Optional[str] = None,
                               action_type: Optional[str] = None,
                               intent_pattern: Optional[str] = None) -> List[AtomicBelief]:
        """Get atomic beliefs relevant to the given context."""
        matching_beliefs = []

        for belief in self.atomic_beliefs.values():
            if belief.status != BeliefStatus.ACTIVE:
                continue

            # Check condition matching
            matches = True

            # If query specifies a condition, belief must also specify it to match
            if env_cluster:
                if not belief.condition_signature.env_cluster:
                    matches = False
                elif env_cluster != belief.condition_signature.env_cluster:
                    matches = False

            if repo_toolchain:
                if not belief.condition_signature.repo_toolchain:
                    matches = False
                elif repo_toolchain != belief.condition_signature.repo_toolchain:
                    matches = False

            if action_type:
                if not belief.condition_signature.action_type_pattern:
                    matches = False
                elif belief.condition_signature.action_type_pattern not in action_type:
                    matches = False

            if intent_pattern:
                if not belief.condition_signature.intent_pattern:
                    matches = False
                elif belief.condition_signature.intent_pattern not in intent_pattern:
                    matches = False

            if matches:
                matching_beliefs.append(belief)

        # Sort by confidence (highest first)
        matching_beliefs.sort(key=lambda b: b.confidence, reverse=True)
        return matching_beliefs

    def update_belief_stats(self,
                           belief_id: str,
                           success_with: bool,
                           success_without: Optional[bool] = None) -> None:
        """Update statistics for a belief."""
        belief = self.atomic_beliefs.get(belief_id)
        if not belief:
            raise ValueError(f"Belief {belief_id} not found")

        belief.update_stats(success_with, success_without)

        # Check for promotion
        if belief.should_promote_to_preference():
            belief.status = BeliefStatus.PROMOTED_TO_PREFERENCE
        elif belief.should_promote_to_constraint():
            belief.status = BeliefStatus.PROMOTED_TO_CONSTRAINT

    def deprecate_belief(self, belief_id: str, reason: str = "") -> None:
        """Deprecate a belief."""
        belief = self.atomic_beliefs.get(belief_id)
        if not belief:
            raise ValueError(f"Belief {belief_id} not found")

        belief.status = BeliefStatus.DEPRECATED
        belief.metadata["deprecation_reason"] = reason
        belief.metadata["deprecated_at"] = datetime.now(timezone.utc).isoformat()

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the belief graph."""
        total_atomic = len(self.atomic_beliefs)
        total_attempt = len(self.attempt_beliefs)

        # Count by type and status
        type_counts = {}
        status_counts = {}

        for belief in self.atomic_beliefs.values():
            belief_type = belief.belief_type.value
            status = belief.status.value

            type_counts[belief_type] = type_counts.get(belief_type, 0) + 1
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total_atomic_beliefs": total_atomic,
            "total_attempt_beliefs": total_attempt,
            "total_tasks": len(self.task_to_attempts),
            "type_counts": type_counts,
            "status_counts": status_counts,
            "conflicting_belief_pairs": len(self.conflicting_beliefs),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert belief graph to dictionary."""
        return {
            "attempt_beliefs": {k: v.to_dict() for k, v in self.attempt_beliefs.items()},
            "atomic_beliefs": {k: v.to_dict() for k, v in self.atomic_beliefs.items()},
            "task_to_attempts": self.task_to_attempts,
            "conflicting_beliefs": self.conflicting_beliefs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BeliefGraph:
        """Create belief graph from dictionary."""
        bg = cls()

        # Reconstruct attempt beliefs
        for attempt_id, attempt_data in data.get("attempt_beliefs", {}).items():
            bg.attempt_beliefs[attempt_id] = AttemptBelief.from_dict(attempt_data)

        # Reconstruct atomic beliefs
        for belief_id, belief_data in data.get("atomic_beliefs", {}).items():
            bg.atomic_beliefs[belief_id] = AtomicBelief.from_dict(belief_data)

        # Restore mappings
        bg.task_to_attempts = data.get("task_to_attempts", {})
        bg.conflicting_beliefs = data.get("conflicting_beliefs", {})

        return bg

    def save_to_file(self, filepath: str) -> None:
        """Save belief graph to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filepath: str) -> BeliefGraph:
        """Load belief graph from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
