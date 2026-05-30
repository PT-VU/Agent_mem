"""
Structured experience objects used in stage-4 memory learning.

These objects are JSON-serializable and intentionally compact so they can be
stored, queried, and injected without introducing heavy dependencies.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FailureCardV2:
    """Unresolved/failure experience card."""

    schema_version: str = "2.0"
    card_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    task_id: str = ""
    action_id: Optional[str] = None
    instance_id: Optional[str] = None
    run_id: Optional[str] = None
    source_event: Optional[str] = None
    step_index: Optional[int] = None
    trace_id: Optional[str] = None

    error_signature: Dict[str, Any] = field(default_factory=dict)
    action_trace_snippet: List[str] = field(default_factory=list)
    candidate_fix_actions: List[str] = field(default_factory=list)
    verification_commands: List[str] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)

    root_cause_nodes: List[str] = field(default_factory=list)
    propagation_chain: List[str] = field(default_factory=list)
    error_module: str = "unknown"
    failure_class: str = "agent_failure_card"
    confidence: float = 0.0
    status: str = "unresolved"

    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureCardV2":
        payload = dict(data)
        payload.setdefault("schema_version", "2.0")
        payload.setdefault("card_id", str(uuid.uuid4()))
        payload.setdefault("error_signature", {})
        payload.setdefault("action_trace_snippet", [])
        payload.setdefault("candidate_fix_actions", [])
        payload.setdefault("verification_commands", [])
        payload.setdefault("evidence_refs", [])
        payload.setdefault("root_cause_nodes", [])
        payload.setdefault("propagation_chain", [])
        payload.setdefault("error_module", "unknown")
        payload.setdefault("failure_class", "agent_failure_card")
        payload.setdefault("confidence", 0.0)
        payload.setdefault("status", "unresolved")
        payload.setdefault("trace_id", None)
        payload.setdefault("created_at", _now_iso())
        payload.setdefault("updated_at", _now_iso())
        payload.setdefault("metadata", {})
        return cls(**payload)


@dataclass
class RepairPatternV2:
    """Reusable repair pattern distilled from historical fixes."""

    schema_version: str = "2.0"
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trigger_signature: Dict[str, Any] = field(default_factory=dict)
    fix_action_template: str = ""
    expected_verification: List[str] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)
    support: int = 1
    confidence: float = 0.0
    trace_id: Optional[str] = None
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RepairPatternV2":
        payload = dict(data)
        payload.setdefault("schema_version", "2.0")
        payload.setdefault("pattern_id", str(uuid.uuid4()))
        payload.setdefault("trigger_signature", {})
        payload.setdefault("expected_verification", [])
        payload.setdefault("evidence_refs", [])
        payload.setdefault("support", 1)
        payload.setdefault("confidence", 0.0)
        payload.setdefault("trace_id", None)
        payload.setdefault("created_at", _now_iso())
        payload.setdefault("updated_at", _now_iso())
        payload.setdefault("metadata", {})
        return cls(**payload)


@dataclass
class PreventiveRuleV2:
    """Preventive checks extracted from successful or failed trajectories."""

    schema_version: str = "2.0"
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    when: str = ""
    should_check: str = ""
    support: int = 1
    confidence: float = 0.0
    trace_id: Optional[str] = None
    evidence_refs: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreventiveRuleV2":
        payload = dict(data)
        payload.setdefault("schema_version", "2.0")
        payload.setdefault("rule_id", str(uuid.uuid4()))
        payload.setdefault("support", 1)
        payload.setdefault("confidence", 0.0)
        payload.setdefault("trace_id", None)
        payload.setdefault("evidence_refs", [])
        payload.setdefault("created_at", _now_iso())
        payload.setdefault("updated_at", _now_iso())
        payload.setdefault("metadata", {})
        return cls(**payload)
