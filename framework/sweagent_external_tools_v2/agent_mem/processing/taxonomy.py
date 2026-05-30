"""
Error taxonomy and rule-based step analysis for experience extraction.

This module provides deterministic classification so extraction still works when
LLM assistance is unavailable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..core.problem_file import ActionType, Outcome, ProblemFile


MODULES = ("memory", "reflection", "planning", "action", "system")


def _norm_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_error_type(raw: str) -> str:
    text = _norm_text(raw)
    if not text:
        return "unknown"
    mapping = (
        ("import", "import_error"),
        ("module not found", "import_error"),
        ("file not found", "file_not_found"),
        ("no such file", "file_not_found"),
        ("permission", "permission_error"),
        ("timeout", "timeout"),
        ("assert", "assertion_failed"),
        ("test", "test_failure"),
        ("compile", "compilation_error"),
        ("syntax", "syntax_error"),
        ("command_nonzero_exit", "command_nonzero_exit"),
        ("environment_error", "environment_error"),
        ("tool_timeout", "tool_timeout"),
    )
    for token, name in mapping:
        if token in text:
            return name
    return re.sub(r"[^a-z0-9_]+", "_", text)[:80] or "unknown"


@dataclass
class StepSignal:
    step_index: int
    module: str
    error_type: str
    evidence: str
    reasoning: str
    confidence: float

    def to_dict(self, *, attempt_id: str) -> Dict[str, object]:
        return {
            "attempt_id": attempt_id,
            "step_index": self.step_index,
            "module": self.module,
            "error_detected": True,
            "error_type": self.error_type,
            "evidence": self.evidence,
            "reasoning": self.reasoning,
            "confidence": round(max(0.0, min(1.0, self.confidence)), 4),
        }


class ErrorTaxonomy:
    """Rule-based taxonomy classifier used by extraction orchestrator."""

    def __init__(self, version: str = "v1"):
        self.version = version

    def classify_action(
        self,
        *,
        action: ProblemFile,
        step_index: int,
        previous_action: Optional[ProblemFile] = None,
    ) -> List[StepSignal]:
        signals: List[StepSignal] = []
        action_text = _norm_text(f"{action.intent_text} {action.action_text}")
        error_type = normalize_error_type(
            action.failure_signature.error_type if action.failure_signature else ""
        )
        stderr_path = action.stderr_ref.location if action.stderr_ref else ""

        # ACTION/SYSTEM module: explicit execution failure.
        if action.outcome == Outcome.FAIL:
            module = "action"
            confidence = 0.88
            if error_type in {"environment_error", "tool_timeout", "permission_error"}:
                module = "system"
                confidence = 0.82
            signals.append(
                StepSignal(
                    step_index=step_index,
                    module=module,
                    error_type=error_type if error_type != "unknown" else "execution_failure",
                    evidence=f"action_id={action.action_id} stderr={stderr_path}",
                    reasoning="failed action with normalized error signature",
                    confidence=confidence,
                )
            )

        # PLANNING module: planning step without verification hook.
        is_plan_step = (
            action.source_event == "plan_generated"
            or action.action_family in {"planning", "plan"}
            or action.action_type == ActionType.TOOL_CALL
        )
        mentions_verify = any(
            token in action_text for token in ("test", "pytest", "verify", "validation", "check")
        )
        if is_plan_step and not mentions_verify:
            signals.append(
                StepSignal(
                    step_index=step_index,
                    module="planning",
                    error_type="constraint_ignorance",
                    evidence=f"action_id={action.action_id} intent={action.intent_text[:120]}",
                    reasoning="planning action misses explicit verification/test plan",
                    confidence=0.45,
                )
            )

        # REFLECTION module: repeated identical failing action without adaptation.
        if previous_action and action.outcome == Outcome.FAIL:
            prev_text = _norm_text(f"{previous_action.intent_text} {previous_action.action_text}")
            if prev_text and prev_text == action_text and previous_action.outcome == Outcome.FAIL:
                signals.append(
                    StepSignal(
                        step_index=step_index,
                        module="reflection",
                        error_type="no_strategy_shift",
                        evidence=f"prev_action={previous_action.action_id} curr_action={action.action_id}",
                        reasoning="repeated failing action without strategy update",
                        confidence=0.72,
                    )
                )

        # MEMORY module: explicit retrieval/use-memory step followed by same failure.
        if previous_action and action.outcome == Outcome.FAIL:
            prev_meta = previous_action.metadata if isinstance(previous_action.metadata, dict) else {}
            had_hint = bool(prev_meta.get("memory_hint_count", 0))
            if had_hint:
                signals.append(
                    StepSignal(
                        step_index=step_index,
                        module="memory",
                        error_type="memory_not_applied",
                        evidence=f"hint_count={prev_meta.get('memory_hint_count')} action_id={action.action_id}",
                        reasoning="memory hints existed but failure repeated on next action",
                        confidence=0.6,
                    )
                )

        return signals

    def classify_attempt(
        self,
        *,
        attempt_id: str,
        actions: List[ProblemFile],
    ) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        previous: Optional[ProblemFile] = None
        for idx, action in enumerate(actions):
            step_index = action.step_index if isinstance(action.step_index, int) else idx
            for signal in self.classify_action(
                action=action,
                step_index=step_index,
                previous_action=previous,
            ):
                out.append(signal.to_dict(attempt_id=attempt_id))
            previous = action
        out.sort(
            key=lambda row: (
                int(row.get("step_index", 0)),
                -float(row.get("confidence", 0.0)),
            )
        )
        return out
