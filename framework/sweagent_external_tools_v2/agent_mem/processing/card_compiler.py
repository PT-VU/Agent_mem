"""
Compile governed prompt cards from v2.1 sources.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..types import CompilerCardType, EvidenceLevel, GovernanceState
from .v21_shared import stable_family_id


class CardCompiler:
    """Compile the v2.1 card whitelist from whitelisted inputs."""

    def compile(
        self,
        *,
        attempt_summary: Optional[Dict[str, Any]] = None,
        failure_card: Optional[Dict[str, Any]] = None,
        repair_patterns: Optional[List[Dict[str, Any]]] = None,
        subtasks: Optional[List[Dict[str, Any]]] = None,
        max_cards: int = 4,
    ) -> List[Dict[str, Any]]:
        cards: List[Dict[str, Any]] = []
        summary = dict(attempt_summary or {})
        instance_id = str(summary.get("instance_id") or "")
        run_id = str(summary.get("run_id") or "")
        attempt_id = str(summary.get("attempt_id") or "")
        trace_id = str(summary.get("trace_id") or "")

        if summary:
            cards.extend(self._compile_from_attempt_summary(summary))
        if failure_card:
            cards.extend(self._compile_from_failure_card(dict(failure_card)))
        for pattern in list(repair_patterns or [])[:2]:
            if isinstance(pattern, dict):
                cards.extend(self._compile_from_repair_pattern(pattern))
        for subtask in list(subtasks or [])[:2]:
            if isinstance(subtask, dict):
                cards.extend(self._compile_from_subtask(subtask))

        deduped: Dict[str, Dict[str, Any]] = {}
        for raw in cards:
            card = dict(raw)
            card.setdefault("type", "compiler_card")
            card.setdefault("instance_id", instance_id)
            card.setdefault("run_id", run_id)
            card.setdefault("attempt_id", attempt_id)
            card.setdefault("trace_id", trace_id)
            card.setdefault("promotion_state", GovernanceState.CANDIDATE.value)
            card.setdefault("confidence", 0.5)
            card.setdefault("budget_cost", len(str(card.get("hint") or card.get("recommendation") or "")))
            card.setdefault("source_object_ids", [])
            card.setdefault("evidence_level", EvidenceLevel.LOCAL.value)
            card.setdefault("recommendation", str(card.get("hint") or card.get("recommendation") or "").strip())
            card.setdefault(
                "card_id",
                stable_family_id("card", card.get("card_type"), card.get("family_id"), card.get("recommendation")),
            )
            deduped[card["family_id"]] = card if card["family_id"] not in deduped else max(
                [deduped[card["family_id"]], card],
                key=lambda row: float(row.get("confidence", 0.0) or 0.0),
            )
        ranked = sorted(
            deduped.values(),
            key=lambda row: (
                float(row.get("confidence", 0.0) or 0.0),
                0 if row.get("card_type") == CompilerCardType.CLOSURE_GUARD.value else 1,
            ),
            reverse=True,
        )
        return ranked[: max(1, max_cards)]

    def _compile_from_attempt_summary(self, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        cards: List[Dict[str, Any]] = []
        next_steps = [str(x).strip() for x in (summary.get("next_best_actions") or []) if str(x).strip()]
        failed = [
            row for row in (summary.get("failed_strategies") or []) if isinstance(row, dict)
        ]
        if next_steps:
            hint = f"Next best action: {next_steps[0]}"
            cards.append(
                self._base_card(
                    card_type=CompilerCardType.PLAN_HINT.value,
                    family_id=stable_family_id("plan_hint", summary.get("instance_id"), next_steps[0]),
                    hint=hint,
                    confidence=0.76,
                    evidence_level=EvidenceLevel.ATTEMPT.value,
                    source_object_ids=[str(summary.get("summary_id") or "")],
                )
            )
        if failed:
            first = failed[0]
            reason = str(first.get("reason") or "").strip()
            avoid = [str(x).strip() for x in (first.get("avoid_actions") or []) if str(x).strip()]
            if reason or avoid:
                text = reason or "Avoid repeating the previously failed strategy."
                if avoid:
                    text = f"{text} Avoid: {', '.join(avoid[:2])}"
                cards.append(
                    self._base_card(
                        card_type=CompilerCardType.CLOSURE_GUARD.value,
                        family_id=stable_family_id("closure_guard", summary.get("instance_id"), first.get("strategy_label"), reason),
                        hint=text[:280],
                        confidence=0.72,
                        evidence_level=EvidenceLevel.ATTEMPT.value,
                        source_object_ids=[str(summary.get("summary_id") or "")],
                    )
                )
        return cards

    def _compile_from_failure_card(self, card: Dict[str, Any]) -> List[Dict[str, Any]]:
        actions = [str(x).strip() for x in (card.get("candidate_fix_actions") or []) if str(x).strip()]
        verification = [str(x).strip() for x in (card.get("verification_commands") or []) if str(x).strip()]
        if not actions and not verification:
            return []
        hint_parts = []
        if actions:
            hint_parts.append(actions[0])
        if verification:
            hint_parts.append(f"Verify with {verification[0]}")
        return [
            self._base_card(
                card_type=CompilerCardType.RETRY_HINT.value,
                family_id=stable_family_id("retry_hint", card.get("card_id"), actions[0] if actions else verification[0]),
                hint=". ".join(hint_parts)[:280],
                confidence=max(0.55, float(card.get("confidence", 0.0) or 0.0)),
                evidence_level=EvidenceLevel.ATTEMPT.value,
                source_object_ids=[str(card.get("card_id") or "")],
                evidence_refs=list(card.get("evidence_refs") or [])[:4],
            )
        ]

    def _compile_from_repair_pattern(self, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        action = str(pattern.get("fix_action_template") or pattern.get("recommendation") or "").strip()
        verification = [str(x).strip() for x in (pattern.get("expected_verification") or pattern.get("verification_commands") or []) if str(x).strip()]
        if not action:
            return []
        hint = action
        if verification:
            hint = f"{hint}. Then run {verification[0]}"
        return [
            self._base_card(
                card_type=CompilerCardType.RETRY_HINT.value,
                family_id=stable_family_id("repair_pattern", pattern.get("pattern_id"), action),
                hint=hint[:280],
                confidence=max(0.5, float(pattern.get("confidence", 0.0) or 0.0)),
                evidence_level=EvidenceLevel.ATTEMPT.value,
                source_object_ids=[str(pattern.get("pattern_id") or "")],
                evidence_refs=list(pattern.get("evidence_refs") or [])[:4],
            )
        ]

    def _compile_from_subtask(self, subtask: Dict[str, Any]) -> List[Dict[str, Any]]:
        status = str(subtask.get("local_result_status") or "")
        next_steps = [str(x).strip() for x in (subtask.get("recommended_next_steps") or []) if str(x).strip()]
        source_id = str(subtask.get("subtask_instance_id") or "")
        if status == "failed":
            hint = str(subtask.get("failure_type") or "This subtask remains risky.").strip()
            if next_steps:
                hint = f"{hint} Next: {next_steps[0]}"
            return [
                self._base_card(
                    card_type=CompilerCardType.SUBTASK_RISK.value,
                    family_id=stable_family_id("subtask_risk", source_id, subtask.get("subtask_type")),
                    hint=hint[:280],
                    confidence=max(0.52, float(subtask.get("projection_confidence", 0.0) or 0.0)),
                    evidence_level=EvidenceLevel.LOCAL.value,
                    source_object_ids=[source_id],
                )
            ]
        if next_steps:
            return [
                self._base_card(
                    card_type=CompilerCardType.PLAN_HINT.value,
                    family_id=stable_family_id("subtask_plan", source_id, next_steps[0]),
                    hint=f"Subtask next step: {next_steps[0]}"[:280],
                    confidence=max(0.48, float(subtask.get("projection_confidence", 0.0) or 0.0)),
                    evidence_level=EvidenceLevel.LOCAL.value,
                    source_object_ids=[source_id],
                )
            ]
        return []

    @staticmethod
    def _base_card(
        *,
        card_type: str,
        family_id: str,
        hint: str,
        confidence: float,
        evidence_level: str,
        source_object_ids: List[str],
        evidence_refs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return {
            "card_type": card_type,
            "family_id": family_id,
            "hint": hint,
            "recommendation": hint,
            "confidence": round(float(confidence), 6),
            "promotion_state": GovernanceState.CANDIDATE.value,
            "evidence_level": evidence_level,
            "source_object_ids": [str(x).strip() for x in source_object_ids if str(x).strip()],
            "evidence_refs": [str(x).strip() for x in (evidence_refs or []) if str(x).strip()],
        }
