"""
Shared helpers and contracts for AgentMem v2.1.

Keep this module dependency-light so both bridge_hook and agent_mem internals
can share the same schema and status semantics.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict

from ..types import EvidenceLevel, GovernanceState


SUCCESS_FACT_MIN_FIELDS = (
    "instance_id",
    "run_id",
    "attempt_id",
    "trace_id",
    "step_index",
    "action_text",
    "success_like",
    "timestamp",
)


RUN_DONE_CONTEXT_FIELD_SPECS: Dict[str, Dict[str, str]] = {
    "touched_files": {
        "source": "ProblemFile.touched_files",
        "required": "no",
        "fallback": "[]",
        "storage": "summary",
    },
    "diff_summary": {
        "source": "ProblemFile.diff_summary_ref + patch_stats aggregation",
        "required": "no",
        "fallback": "{}",
        "storage": "summary_with_evidence_refs",
    },
    "validation_summary": {
        "source": "RUN_TEST actions + action metadata",
        "required": "no",
        "fallback": "{\"commands\": []}",
        "storage": "summary",
    },
    "patch_digest": {
        "source": "touched_files + diff_summary",
        "required": "yes",
        "fallback": "sha1(empty-context)",
        "storage": "summary",
    },
    "task_closed_cleanly": {
        "source": "handle_run_done exit_status / has_submission",
        "required": "yes",
        "fallback": "False",
        "storage": "summary",
    },
}


COMPILER_CARD_REQUIRED_FIELDS = (
    "card_id",
    "card_type",
    "family_id",
    "source_object_ids",
    "confidence",
    "promotion_state",
    "evidence_level",
)


GOVERNANCE_TRANSITIONS = {
    GovernanceState.CANDIDATE.value: {
        GovernanceState.PROMOTED.value,
        GovernanceState.SUPPRESSED.value,
        GovernanceState.DEPRECATED.value,
    },
    GovernanceState.PROMOTED.value: {GovernanceState.DEPRECATED.value},
    GovernanceState.SUPPRESSED.value: {GovernanceState.DEPRECATED.value},
    GovernanceState.DEPRECATED.value: set(),
}


def build_success_fact_idempotency_key(trace_id: str, step_index: Any) -> str:
    trace = str(trace_id or "").strip()
    step = str(step_index if step_index is not None else "").strip()
    if not trace or not step:
        return ""
    return f"{trace}::{step}"


def classify_success_like(
    *,
    observation: str = "",
    error_type: str = "",
    exit_status: str = "",
    has_submission: bool | None = None,
) -> bool:
    raw_observation = str(observation or "")
    lowered = raw_observation.lower()
    error_patterns = (
        "error",
        "exception",
        "traceback",
        "assert",
        "failed",
        "no such file",
        "timed out",
        "non-zero",
    )
    if str(error_type or "").strip():
        return False
    if raw_observation and any(token in lowered for token in error_patterns):
        if "simulated command error" not in lowered:
            return False
    normalized_status = str(exit_status or "").strip().lower()
    if normalized_status:
        if normalized_status in {"done", "success", "submitted", "resolved"}:
            return True
        if normalized_status in {"failed", "error", "timeout", "incomplete", "unresolved"}:
            return False
    if has_submission is True:
        return True
    return True


def stable_patch_digest(payload: Dict[str, Any]) -> str:
    serializable = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(serializable.encode("utf-8")).hexdigest()


def stable_family_id(*parts: Any) -> str:
    tokens = []
    for item in parts:
        text = re.sub(r"[^a-z0-9_]+", "_", str(item or "").strip().lower())
        text = re.sub(r"_+", "_", text).strip("_")
        if text:
            tokens.append(text)
    return "__".join(tokens[:6]) or "unknown_family"


def normalize_evidence_level(value: Any, default: EvidenceLevel = EvidenceLevel.LOCAL) -> str:
    raw = str(value or "").strip().lower()
    if raw in {level.value for level in EvidenceLevel}:
        return raw
    return default.value

