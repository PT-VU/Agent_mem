"""
Abstract experience extraction for cross-task reuse.

This module builds higher-level patterns from task action traces so that retrieval
can return reusable guidance instead of only instance-level snippets.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

from ..core.problem_file import ActionType, Outcome, ProblemFile


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _slug(value: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", value.lower())
    return s.strip("_") or "unknown"


def _normalize_text(text: str) -> str:
    out = _safe_text(text).strip()
    out = re.sub(r"([A-Za-z]:)?[/~][\w\-./]+", "<PATH>", out)
    out = re.sub(
        r"\b[\w.\-]+\.(py|js|ts|java|go|rs|cpp|h|hpp|md|txt|json|yaml|yml|toml|ini|cfg|sh)\b",
        "<FILE>",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(r"\b[0-9a-f]{7,40}\b", "<HASH>", out, flags=re.IGNORECASE)
    out = re.sub(r"\b\d+\b", "<NUM>", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _normalize_pattern_family(pattern_type: str, abstracted_intent: str = "", metadata: Dict[str, Any] | None = None) -> str:
    raw = _slug(pattern_type or "generic_pattern")
    text = _normalize_text(abstracted_intent).lower()
    meta = metadata if isinstance(metadata, dict) else {}
    err = _slug(str(meta.get("error_signature", "") or ""))
    if any(token in raw for token in ("repetitive_planning", "inefficient_planning", "stagnation")):
        return "planning_loop"
    if "constraint_ignorance" in raw or "validation" in raw:
        return "validation_gap"
    if raw.startswith("repair_"):
        if err and err != "none":
            return f"repair_{err}"
        return "failure_recovery"
    if raw == "recover_from_failure":
        return "failure_recovery"
    if raw == "validate_then_submit":
        return "validation_guard"
    if "planning" in text and ("transition" in text or "progress tracker" in text or "state machine" in text):
        return "planning_loop"
    if "verify" in text or "validation" in text or "pytest" in text or "test" in text:
        return "validation_guard"
    return raw


def _normalize_trigger_family(
    *,
    pattern_type: str,
    success_conditions: List[str],
    failure_avoidance: List[str],
    metadata: Dict[str, Any] | None = None,
) -> str:
    meta = metadata if isinstance(metadata, dict) else {}
    err = _slug(str(meta.get("error_signature", "") or meta.get("error_type", "") or ""))
    pattern = _slug(pattern_type or "")
    joined = " ".join(success_conditions + failure_avoidance).lower()
    if any(token in pattern for token in ("repetitive_planning", "inefficient_planning", "stagnation")):
        return "planning_transition_failure"
    if "constraint_ignorance" in pattern or "validation" in joined or "test" in joined:
        return "missing_validation"
    if err and err != "none":
        return err
    if pattern.startswith("repair_"):
        return "failure_recovery"
    return "generic_trigger"


def _normalize_advice_family(abstracted_intent: str, success_conditions: List[str], failure_avoidance: List[str]) -> str:
    text = " ".join([_normalize_text(abstracted_intent)] + success_conditions + failure_avoidance).lower()
    if any(token in text for token in ("state machine", "progress tracker", "transition to execution", "force progression")):
        return "force_progression_transition"
    if any(token in text for token in ("termination condition", "iteration limit", "max planning", "timeout")):
        return "bound_planning_iterations"
    if any(token in text for token in ("verify", "validation", "pytest", "run relevant tests", "before submission")):
        return "add_local_validation"
    if any(token in text for token in ("narrow", "retry", "adjustment")):
        return "narrow_context_then_retry"
    compact = _slug(" ".join(text.split()[:8]))
    return compact or "generic_advice"


def build_experience_family_id(
    *,
    normalized_pattern_type: str,
    normalized_trigger_family: str,
    normalized_advice_family: str,
) -> str:
    return "__".join(
        [
            _slug(normalized_pattern_type or "generic_pattern"),
            _slug(normalized_trigger_family or "generic_trigger"),
            _slug(normalized_advice_family or "generic_advice"),
        ]
    )


@dataclass
class AbstractExperience:
    experience_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: str = "2.1"
    pattern_type: str = "generic_pattern"
    normalized_pattern_type: str = ""
    normalized_trigger_family: str = ""
    normalized_advice_family: str = ""
    family_id: str = ""
    abstracted_intent: str = ""
    variant_texts: List[str] = field(default_factory=list)
    success_conditions: List[str] = field(default_factory=list)
    failure_avoidance: List[str] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)
    source_task_ids: List[str] = field(default_factory=list)
    source_event_ids: List[str] = field(default_factory=list)
    source_instance_id: str = ""
    source_run_ids: List[str] = field(default_factory=list)
    source_attempt_ids: List[str] = field(default_factory=list)
    support_count: int = 1
    confidence: float = 0.0
    lifecycle_status: str = "new"
    links: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""
    created_at: str = field(default_factory=_now_iso)
    last_updated: str = field(default_factory=_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experience_id": self.experience_id,
            "pattern_type": self.pattern_type,
            "normalized_pattern_type": self.normalized_pattern_type,
            "normalized_trigger_family": self.normalized_trigger_family,
            "normalized_advice_family": self.normalized_advice_family,
            "family_id": self.family_id,
            "abstracted_intent": self.abstracted_intent,
            "variant_texts": self.variant_texts,
            "success_conditions": self.success_conditions,
            "failure_avoidance": self.failure_avoidance,
            "evidence_refs": self.evidence_refs,
            "source_task_ids": self.source_task_ids,
            "source_event_ids": self.source_event_ids,
            "source_instance_id": self.source_instance_id,
            "source_run_ids": self.source_run_ids,
            "source_attempt_ids": self.source_attempt_ids,
            "support_count": self.support_count,
            "confidence": self.confidence,
            "lifecycle_status": self.lifecycle_status,
            "links": self.links,
            "quality": self.quality,
            "fingerprint": self.fingerprint,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "metadata": self.metadata,
        }


class AbstractExperienceBuilder:
    """Build abstract experiences from task actions."""

    def build_from_task(
        self,
        *,
        task_id: str,
        actions: List[ProblemFile],
        success: bool,
        task_summary: str = "",
        source_instance_id: str = "",
        source_run_id: str = "",
        source_attempt_id: str = "",
    ) -> List[AbstractExperience]:
        if not actions:
            return []

        base = self._build_task_level_pattern(
            task_id=task_id,
            actions=actions,
            success=success,
            task_summary=task_summary,
            source_instance_id=source_instance_id,
            source_run_id=source_run_id,
            source_attempt_id=source_attempt_id,
        )
        patterns: List[AbstractExperience] = [base]

        # Keep all meaningful error-specific patterns to avoid over-merging.
        error_counts: Dict[str, int] = {}
        for action in actions:
            if action.outcome == Outcome.FAIL and action.failure_signature:
                et = _slug(action.failure_signature.error_type or "unknown_error")
                error_counts[et] = error_counts.get(et, 0) + 1
        for err, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            if count < 1:
                continue
            patterns.append(
                self._build_error_specific_pattern(
                    task_id=task_id,
                    actions=actions,
                    error_type=err,
                    source_instance_id=source_instance_id,
                    source_run_id=source_run_id,
                    source_attempt_id=source_attempt_id,
                )
            )

        # Template pollution suppression.
        for p in patterns:
            if self._is_low_information_template(p.abstracted_intent):
                p.confidence = round(max(0.1, p.confidence - 0.2), 4)
                p.metadata["template_polluted"] = True
                p.metadata["write_penalty"] = "generic_intent"

        return patterns

    def _build_task_level_pattern(
        self,
        *,
        task_id: str,
        actions: List[ProblemFile],
        success: bool,
        task_summary: str,
        source_instance_id: str,
        source_run_id: str,
        source_attempt_id: str,
    ) -> AbstractExperience:
        pattern_type = self._derive_pattern_type(actions=actions, success=success)
        abstracted_intent = self._derive_abstracted_intent(actions=actions, task_summary=task_summary)
        success_conditions = self._derive_success_conditions(actions)
        failure_avoidance = self._derive_failure_avoidance(actions)
        evidence_refs = self._collect_evidence_refs(actions, max_refs=20)
        confidence = self._estimate_confidence(actions=actions, success=success)

        exp = AbstractExperience(
            pattern_type=pattern_type,
            abstracted_intent=abstracted_intent,
            variant_texts=[abstracted_intent] if abstracted_intent else [],
            success_conditions=success_conditions,
            failure_avoidance=failure_avoidance,
            evidence_refs=evidence_refs,
            source_task_ids=[task_id],
            source_event_ids=self._collect_source_event_ids(actions, max_refs=20),
            source_instance_id=source_instance_id or self._derive_source_instance(actions),
            source_run_ids=self._collect_source_ids(actions, field="run_id", max_refs=6, fallback=source_run_id),
            source_attempt_ids=self._collect_source_attempt_ids(actions, fallback=source_attempt_id),
            support_count=1,
            confidence=confidence,
            lifecycle_status="new",
            links={"related_experience_ids": []},
            quality={
                "item_confidence": confidence,
                "support_count": 1,
                "signal_density": self._estimate_signal_density(actions),
            },
            metadata={
                "task_success": success,
                "action_count": len(actions),
                "error_signature": self._derive_error_signature(actions),
                "tool_sequence": self._derive_tool_sequence(actions),
                "changed_file_pattern": self._derive_changed_file_pattern(actions),
                "test_signal": self._derive_test_signal(actions),
            },
        )
        exp.normalized_pattern_type = _normalize_pattern_family(
            exp.pattern_type,
            exp.abstracted_intent,
            exp.metadata,
        )
        exp.normalized_trigger_family = _normalize_trigger_family(
            pattern_type=exp.pattern_type,
            success_conditions=exp.success_conditions,
            failure_avoidance=exp.failure_avoidance,
            metadata=exp.metadata,
        )
        exp.normalized_advice_family = _normalize_advice_family(
            exp.abstracted_intent,
            exp.success_conditions,
            exp.failure_avoidance,
        )
        exp.family_id = build_experience_family_id(
            normalized_pattern_type=exp.normalized_pattern_type,
            normalized_trigger_family=exp.normalized_trigger_family,
            normalized_advice_family=exp.normalized_advice_family,
        )
        exp.fingerprint = self._compute_fingerprint(exp)
        return exp

    def _build_error_specific_pattern(
        self,
        *,
        task_id: str,
        actions: List[ProblemFile],
        error_type: str,
        source_instance_id: str,
        source_run_id: str,
        source_attempt_id: str,
    ) -> AbstractExperience:
        related = []
        for action in actions:
            if action.failure_signature and _slug(action.failure_signature.error_type) == error_type:
                related.append(action)

        abstracted_intent = f"repair_{error_type}_by_narrowing_context_then_retrying_with_adjustment"
        success_conditions = ["validate_repair_path_before_retry"]
        failure_avoidance = [f"avoid_repeat_{error_type}_without_strategy_change"]
        evidence_refs = self._collect_evidence_refs(related or actions, max_refs=12)

        exp = AbstractExperience(
            pattern_type=f"repair_{error_type}",
            abstracted_intent=abstracted_intent,
            variant_texts=[abstracted_intent],
            success_conditions=success_conditions,
            failure_avoidance=failure_avoidance,
            evidence_refs=evidence_refs,
            source_task_ids=[task_id],
            source_event_ids=self._collect_source_event_ids(related or actions, max_refs=12),
            source_instance_id=source_instance_id or self._derive_source_instance(actions),
            source_run_ids=self._collect_source_ids(actions, field="run_id", max_refs=6, fallback=source_run_id),
            source_attempt_ids=self._collect_source_attempt_ids(actions, fallback=source_attempt_id),
            support_count=1,
            confidence=0.55 if related else 0.45,
            lifecycle_status="new",
            links={"related_experience_ids": []},
            quality={
                "item_confidence": 0.55 if related else 0.45,
                "support_count": 1,
                "signal_density": self._estimate_signal_density(related or actions),
            },
            metadata={
                "error_type": error_type,
                "related_action_count": len(related),
                "error_signature": error_type,
                "tool_sequence": self._derive_tool_sequence(related or actions),
                "changed_file_pattern": self._derive_changed_file_pattern(related or actions),
                "test_signal": self._derive_test_signal(related or actions),
            },
        )
        exp.normalized_pattern_type = _normalize_pattern_family(
            exp.pattern_type,
            exp.abstracted_intent,
            exp.metadata,
        )
        exp.normalized_trigger_family = _normalize_trigger_family(
            pattern_type=exp.pattern_type,
            success_conditions=exp.success_conditions,
            failure_avoidance=exp.failure_avoidance,
            metadata=exp.metadata,
        )
        exp.normalized_advice_family = _normalize_advice_family(
            exp.abstracted_intent,
            exp.success_conditions,
            exp.failure_avoidance,
        )
        exp.family_id = build_experience_family_id(
            normalized_pattern_type=exp.normalized_pattern_type,
            normalized_trigger_family=exp.normalized_trigger_family,
            normalized_advice_family=exp.normalized_advice_family,
        )
        exp.fingerprint = self._compute_fingerprint(exp)
        return exp

    def _derive_pattern_type(self, *, actions: List[ProblemFile], success: bool) -> str:
        has_fail = any(a.outcome == Outcome.FAIL for a in actions)
        has_test = any(a.action_type == ActionType.RUN_TEST for a in actions) or any(
            "test" in _normalize_text(a.intent_text).lower() for a in actions
        )
        if success and has_fail:
            return "recover_from_failure"
        if success and has_test:
            return "validate_then_submit"
        if success:
            return "direct_success_flow"
        if has_fail:
            return "failure_dominant_attempt"
        return "generic_attempt"

    def _derive_abstracted_intent(self, *, actions: List[ProblemFile], task_summary: str) -> str:
        candidates = []
        if task_summary:
            candidates.append(task_summary)
        for action in actions:
            if action.intent_text:
                candidates.append(action.intent_text)
        if not candidates:
            return "resolve_task_with_incremental_actions_and_verification"
        text = _normalize_text(" ; ".join(candidates[:5])).lower()
        if not text:
            return "resolve_task_with_incremental_actions_and_verification"
        return text[:280]

    def _derive_success_conditions(self, actions: List[ProblemFile]) -> List[str]:
        conditions: List[str] = []
        for action in actions:
            blob = _normalize_text(f"{action.intent_text} {action.inputs} {action.tool_calls}").lower()
            if action.action_type == ActionType.RUN_TEST or "pytest" in blob or "test" in blob:
                conditions.append("run_relevant_tests_before_submission")
            if action.action_type in {ActionType.CODE_EDIT, ActionType.FILE_OPERATION} or action.touched_files:
                conditions.append("apply_targeted_edits_in_changed_files")
            if "rg " in blob or "grep " in blob or "search" in blob:
                conditions.append("inspect_code_context_before_modification")
            if "diff" in blob:
                conditions.append("review_diff_before_submission")
        if not conditions:
            conditions.append("keep_action_steps_incremental_and_verifiable")
        return self._dedup(conditions, limit=5)

    def _derive_failure_avoidance(self, actions: List[ProblemFile]) -> List[str]:
        avoid: List[str] = []
        for action in actions:
            if action.outcome != Outcome.FAIL:
                continue
            if action.failure_signature:
                et = _slug(action.failure_signature.error_type)
                avoid.append(f"avoid_repeat_{et}_without_new_evidence")
            blob = _normalize_text(f"{action.intent_text} {action.inputs} {action.tool_calls}").lower()
            if "raise" in blob:
                avoid.append("avoid_forced_termination_commands_as_primary_fix")
            if "retry" in blob:
                avoid.append("avoid_identical_retry_without_parameter_change")
        if not avoid:
            avoid.append("avoid_large_unverified_changes_before_running_checks")
        return self._dedup(avoid, limit=5)

    def _collect_evidence_refs(self, actions: List[ProblemFile], *, max_refs: int) -> List[str]:
        refs: List[str] = []
        for action in actions:
            refs.append(action.action_id)
            for ptr in action.evidence_index[:2]:
                if ptr.location:
                    refs.append(ptr.location)
            if len(refs) >= max_refs:
                break
        return self._dedup(refs, limit=max_refs)

    def _estimate_confidence(self, *, actions: List[ProblemFile], success: bool) -> float:
        total = max(1, len(actions))
        success_count = sum(1 for a in actions if a.outcome == Outcome.SUCCESS)
        success_ratio = success_count / total
        support_score = min(1.0, total / 20.0)
        fail_count = sum(1 for a in actions if a.outcome == Outcome.FAIL)
        recover_bonus = 0.08 if success and fail_count > 0 else 0.0
        base = 0.35 + 0.35 * success_ratio + 0.2 * support_score + recover_bonus
        return round(min(0.95, max(0.1, base)), 4)

    def _compute_fingerprint(self, exp: AbstractExperience) -> str:
        error_sig = str(exp.metadata.get("error_signature", ""))
        tool_seq = ",".join(exp.metadata.get("tool_sequence", []) or [])
        changed_file_pattern = ",".join(exp.metadata.get("changed_file_pattern", []) or [])
        test_signal = str(exp.metadata.get("test_signal", ""))
        body = "|".join(
            [
                exp.normalized_pattern_type or exp.pattern_type,
                exp.normalized_trigger_family,
                exp.normalized_advice_family,
                ",".join(exp.success_conditions),
                ",".join(exp.failure_avoidance),
                error_sig,
                tool_seq,
                changed_file_pattern,
                test_signal,
            ]
        )
        return _slug(body)[:200]

    @staticmethod
    def _derive_source_instance(actions: List[ProblemFile]) -> str:
        for action in actions:
            if action.instance_id:
                return str(action.instance_id)
        return ""

    @staticmethod
    def _collect_source_event_ids(actions: List[ProblemFile], max_refs: int) -> List[str]:
        out: List[str] = []
        for action in actions:
            if action.trace_id:
                out.append(str(action.trace_id))
            if len(out) >= max_refs:
                break
        return list(dict.fromkeys(out))[:max_refs]

    @staticmethod
    def _collect_source_ids(
        actions: List[ProblemFile],
        *,
        field: str,
        max_refs: int,
        fallback: str = "",
    ) -> List[str]:
        out: List[str] = []
        for action in actions:
            val = getattr(action, field, None)
            if val:
                out.append(str(val))
            if len(out) >= max_refs:
                break
        if fallback:
            out.append(str(fallback))
        return list(dict.fromkeys(out))[:max_refs]

    @staticmethod
    def _collect_source_attempt_ids(actions: List[ProblemFile], fallback: str = "") -> List[str]:
        out: List[str] = []
        for action in actions:
            attempt = action.metadata.get("source_attempt_id") if isinstance(action.metadata, dict) else None
            if attempt:
                out.append(str(attempt))
        if fallback:
            out.append(str(fallback))
        return list(dict.fromkeys(out))[:6]

    @staticmethod
    def _derive_error_signature(actions: List[ProblemFile]) -> str:
        for action in reversed(actions):
            if action.failure_signature and action.failure_signature.error_type:
                return _slug(action.failure_signature.error_type)
        return "none"

    @staticmethod
    def _derive_tool_sequence(actions: List[ProblemFile]) -> List[str]:
        seq: List[str] = []
        for action in actions[-6:]:
            seq.append((action.action_family or action.action_type.value).lower())
        return seq

    @staticmethod
    def _derive_changed_file_pattern(actions: List[ProblemFile]) -> List[str]:
        exts: List[str] = []
        for action in actions:
            for file_path in action.touched_files:
                m = re.search(r"\.([a-zA-Z0-9]+)$", file_path.strip())
                if m:
                    exts.append(m.group(1).lower())
        if not exts:
            return []
        counts: Dict[str, int] = {}
        for ext in exts:
            counts[ext] = counts.get(ext, 0) + 1
        ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [f".{ext}" for ext, _ in ranked[:4]]

    @staticmethod
    def _derive_test_signal(actions: List[ProblemFile]) -> str:
        has_test = any(a.action_type == ActionType.RUN_TEST or "test" in _normalize_text(a.intent_text).lower() for a in actions)
        return "has_test" if has_test else "no_test"

    @staticmethod
    def _estimate_signal_density(actions: List[ProblemFile]) -> float:
        if not actions:
            return 0.0
        rich = 0
        for a in actions:
            if a.failure_signature or a.touched_files or a.patch_stats or a.test_stats:
                rich += 1
        return round(rich / max(1, len(actions)), 4)

    @staticmethod
    def _is_low_information_template(text: str) -> bool:
        lowered = _normalize_text(text).lower()
        generic_markers = (
            "generate plan for task",
            "resolve task with incremental actions",
            "run_done exit_status",
            "unknown",
        )
        return not lowered or any(marker in lowered for marker in generic_markers)

    @staticmethod
    def _dedup(items: List[str], *, limit: int) -> List[str]:
        seen = set()
        out: List[str] = []
        for item in items:
            s = _safe_text(item).strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
            if len(out) >= limit:
                break
        return out
