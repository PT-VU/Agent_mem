"""
Problem File: Action-level experience atom

Each action is encapsulated as a Problem File and written as a node in the observation KG.
Uses hierarchical storage: structured fields (reconstructible) + summary/embedding (retrievable).

Schema 3.1.1
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from typing_extensions import TypedDict


class ActionType(str, Enum):
    """Action types supported by the system."""
    TOOL_CALL = "tool_call"
    CODE_EDIT = "code_edit"
    RUN_TEST = "run_test"
    GUI_CLICK = "gui_click"
    GUI_TYPE = "gui_type"
    WEB_NAV = "web_nav"
    FILE_OPERATION = "file_operation"
    COMMAND_EXEC = "command_exec"
    OTHER = "other"


class Outcome(str, Enum):
    """Outcome of an action."""
    SUCCESS = "success"
    FAIL = "fail"
    UNKNOWN = "unknown"


class EpistemicTag(str, Enum):
    """Tags for epistemic separation (evidence vs inference)."""
    EVIDENCE = "EVIDENCE"
    INFERENCE = "INFERENCE"


@dataclass
class EvidencePointer:
    """Pointer to evidence coordinates (stdout, stderr, diff, etc.)."""
    type: str  # e.g., "stdout", "stderr", "diff", "test_output", "ui_state"
    location: str  # file path, URL, or storage reference
    offset: Optional[int] = None  # byte offset or line number
    length: Optional[int] = None  # length in bytes or lines
    hash: Optional[str] = None  # content hash for verification


@dataclass
class FailureSignature:
    """Normalized error signature for clustering."""
    error_type: str  # high-level error category
    error_tokens: List[str]  # key tokens from error message
    stack_trace_pattern: Optional[str] = None  # pattern from stack trace
    env_context: Optional[Dict[str, Any]] = None  # environment context at failure


@dataclass
class EnvSignature:
    """Environment signature for adaptation learning."""
    toolchain_version: Optional[str] = None  # e.g., "python3.10", "node18"
    key_env_vars: Dict[str, Optional[str]] = field(default_factory=dict)  # existence/values
    path_hash: Optional[str] = None  # hash of relevant PATH segments
    container_info: Optional[Dict[str, Any]] = None  # container/VM info
    proxy_info: Optional[Dict[str, Any]] = None  # proxy settings
    working_dir: Optional[str] = None  # normalized working directory


@dataclass
class MultiViewEmbeddings:
    """Multi-view embedding vectors for hierarchical retrieval."""
    emb_task_sem: Optional[List[float]] = None  # task semantic summary vector
    emb_file_scope: Optional[List[float]] = None  # file scope/module vector
    emb_error_sig: Optional[List[float]] = None  # error signature vector
    emb_tool_output: Optional[List[float]] = None  # tool output summary vector
    emb_diff_summary: Optional[List[float]] = None  # diff summary vector
    emb_intent: Optional[List[float]] = None  # intent_text vector
    emb_ui_state: Optional[List[float]] = None  # GUI: screenshot/element tree vector


def _default_patch_stats() -> Dict[str, int]:
    return {"files_changed": 0, "lines_added": 0, "lines_deleted": 0}


def _default_test_stats() -> Dict[str, int]:
    return {
        "tests_run": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "fail_to_pass_failed": 0,
        "pass_to_pass_failed": 0,
    }


def _default_execution_stats() -> Dict[str, int]:
    return {"duration_ms": 0, "tool_latency_ms": 0, "retry_count": 0}


@dataclass
class ProblemFile:
    """
    Problem File: Structured representation of an action with evidence pointers.

    Fields 3.1.1
    """
    # Schema metadata
    schema_version: str = "2.0"

    # Core identifiers
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""  # unique within this task execution
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Run trace fields (V2)
    instance_id: Optional[str] = None
    run_id: Optional[str] = None
    agent_name: Optional[str] = None
    source_event: Optional[str] = None
    step_index: Optional[int] = None
    trace_id: Optional[str] = None

    # Action metadata
    action_type: ActionType = ActionType.OTHER
    intent_text: str = ""  # executor's intent layer: purpose/strategy role
    action_text: str = ""  # raw action string (if available)
    action_family: str = ""  # normalized action family

    # Inputs and execution
    inputs: Dict[str, Any] = field(default_factory=dict)  # parameters, prompt fragments
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)  # commands/tool calls

    # Evidence pointers
    stdout_ref: Optional[EvidencePointer] = None
    stderr_ref: Optional[EvidencePointer] = None
    exception_stack_ref: Optional[EvidencePointer] = None
    touched_files: List[str] = field(default_factory=list)  # file collection
    diff_summary_ref: Optional[EvidencePointer] = None
    tests_ref: Optional[EvidencePointer] = None
    ui_state_ref: Optional[EvidencePointer] = None

    # Outcome and analysis
    outcome: Outcome = Outcome.UNKNOWN
    failure_signature: Optional[FailureSignature] = None
    env_signature: Optional[EnvSignature] = None
    patch_stats: Dict[str, int] = field(default_factory=_default_patch_stats)
    test_stats: Dict[str, int] = field(default_factory=_default_test_stats)
    execution_stats: Dict[str, int] = field(default_factory=_default_execution_stats)

    # Evidence and embeddings
    evidence_index: List[EvidencePointer] = field(default_factory=list)
    embeddings: MultiViewEmbeddings = field(default_factory=MultiViewEmbeddings)

    # Epistemic separation
    epistemic_tags: Dict[str, EpistemicTag] = field(default_factory=dict)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ProblemFile to dictionary for serialization."""
        result = asdict(self)

        # Convert enums to strings
        result["action_type"] = self.action_type.value
        result["outcome"] = self.outcome.value

        # Convert epistemic tags
        result["epistemic_tags"] = {
            k: v.value if isinstance(v, EpistemicTag) else v
            for k, v in self.epistemic_tags.items()
        }

        # Convert evidence_index items to dicts
        if "evidence_index" in result and result["evidence_index"]:
            result["evidence_index"] = [
                {
                    "type": ptr.type,
                    "location": ptr.location,
                    "offset": ptr.offset,
                    "length": ptr.length,
                    "hash": ptr.hash
                }
                for ptr in self.evidence_index
            ]

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProblemFile:
        """Create ProblemFile from dictionary."""
        data = dict(data)

        def _normalize_evidence_pointer(raw: Dict[str, Any]) -> Dict[str, Any]:
            normalized = dict(raw)
            if "type" not in normalized and "evidence_type" in normalized:
                normalized["type"] = normalized.pop("evidence_type")
            if "hash" not in normalized and "content_hash" in normalized:
                normalized["hash"] = normalized.pop("content_hash")
            return normalized

        # Schema version compatibility
        if not data.get("schema_version"):
            data["schema_version"] = "1.0"

        # Convert string enums back to Enum instances
        if isinstance(data.get("action_type"), str):
            data["action_type"] = ActionType(data["action_type"])
        if isinstance(data.get("outcome"), str):
            data["outcome"] = Outcome(data["outcome"])

        # Convert epistemic tags
        if "epistemic_tags" in data:
            normalized_tags = {}
            for k, v in data["epistemic_tags"].items():
                try:
                    normalized_tags[k] = EpistemicTag(v)
                except Exception:
                    # Backward-compatible coercion for legacy payloads.
                    normalized_tags[k] = EpistemicTag.INFERENCE
            data["epistemic_tags"] = normalized_tags

        # Handle nested dataclasses
        if data.get("failure_signature"):
            data["failure_signature"] = FailureSignature(**data["failure_signature"])
        if data.get("env_signature"):
            data["env_signature"] = EnvSignature(**data["env_signature"])
        if data.get("embeddings"):
            data["embeddings"] = MultiViewEmbeddings(**data["embeddings"])

        # Handle evidence pointers
        for key in ["stdout_ref", "stderr_ref", "exception_stack_ref",
                   "diff_summary_ref", "tests_ref", "ui_state_ref"]:
            if data.get(key):
                data[key] = EvidencePointer(**_normalize_evidence_pointer(data[key]))

        if "evidence_index" in data:
            data["evidence_index"] = [
                EvidencePointer(**_normalize_evidence_pointer(item)) for item in data["evidence_index"]
            ]

        # Backward-compatible defaults for V2 fields
        data.setdefault("patch_stats", _default_patch_stats())
        data.setdefault("test_stats", _default_test_stats())
        data.setdefault("execution_stats", _default_execution_stats())
        data.setdefault("action_text", data.get("inputs", {}).get("action", ""))
        data.setdefault("action_family", data.get("action_type", ActionType.OTHER).value if isinstance(data.get("action_type"), ActionType) else "")
        data.setdefault("trace_id", None)

        return cls(**data)

    def to_json(self) -> str:
        """Serialize ProblemFile to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> ProblemFile:
        """Deserialize ProblemFile from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def add_evidence_pointer(self, pointer: EvidencePointer) -> None:
        """Add an evidence pointer to the evidence index."""
        self.evidence_index.append(pointer)

    def set_epistemic_tag(self, field_name: str, tag: EpistemicTag) -> None:
        """Set epistemic tag for a field."""
        self.epistemic_tags[field_name] = tag

    def get_evidence_by_type(self, evidence_type: str) -> List[EvidencePointer]:
        """Get all evidence pointers of a specific type."""
        return [ptr for ptr in self.evidence_index if ptr.type == evidence_type]

    def validate(self) -> List[str]:
        """Validate ProblemFile and return list of validation errors."""
        errors = []

        # Basic validation
        if not self.action_id:
            errors.append("action_id is required")
        if not self.task_id:
            errors.append("task_id is required")

        # Validate evidence pointers
        for ptr in self.evidence_index:
            if not ptr.type or not ptr.location:
                errors.append(f"Evidence pointer missing type or location: {ptr}")

        return errors

    def compute_hash(self, exclude_timestamp: bool = True) -> str:
        """Compute a hash of the ProblemFile for data integrity verification.

        Args:
            exclude_timestamp: If True, exclude timestamp from hash calculation
                to allow comparison of otherwise identical ProblemFiles.

        Returns:
            SHA256 hash of the JSON representation
        """
        import hashlib
        import copy

        if exclude_timestamp:
            # Create a copy without timestamp for hash calculation
            data = self.to_dict()
            if "timestamp" in data:
                data["timestamp"] = "excluded_from_hash"
            json_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
        else:
            json_str = self.to_json()

        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def __eq__(self, other: Any) -> bool:
        """Check if two ProblemFiles are equal by comparing their hashes (excluding timestamp)."""
        if not isinstance(other, ProblemFile):
            return False
        return self.compute_hash(exclude_timestamp=True) == other.compute_hash(exclude_timestamp=True)

    def __repr__(self) -> str:
        """String representation of ProblemFile."""
        return f"ProblemFile(action_id={self.action_id}, task_id={self.task_id}, " \
               f"action_type={self.action_type}, outcome={self.outcome})"
