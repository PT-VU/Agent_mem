"""
Deprecated compatibility layer for ProblemFile models.

Canonical schema now lives in `agent_mem.core.problem_file`.
This module only re-exports the core data structures for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict

from ..core.problem_file import (
    ActionType,
    Outcome,
    EpistemicTag,
    EvidencePointer,
    FailureSignature,
    EnvSignature,
    MultiViewEmbeddings,
    ProblemFile,
)


class EdgeType(str, Enum):
    """Compatibility edge type enum for legacy imports."""

    SUCCESS_NEXT = "success_next"
    FAIL_RETRY = "fail_retry"


@dataclass
class KGGraphEdge:
    """Compatibility edge dataclass for legacy imports."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    metadata: Dict[str, Any] = field(default_factory=dict)

