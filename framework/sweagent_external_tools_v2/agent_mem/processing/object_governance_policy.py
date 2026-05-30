"""
Governance policy for v2.1 candidate objects.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..types import GovernanceState, SubtaskState
from .v21_shared import GOVERNANCE_TRANSITIONS


class ObjectGovernancePolicy:
    """Apply lightweight governance to candidate objects without blocking the main loop."""

    def __init__(self, *, max_cards_per_query: int = 4):
        self.max_cards_per_query = max(1, int(max_cards_per_query))

    def apply(
        self,
        *,
        compiler_cards: Optional[List[Dict[str, Any]]] = None,
        subtask_instances: Optional[List[Dict[str, Any]]] = None,
        subtask_edges: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ctx = dict(context or {})
        governed_cards, card_report = self._govern_cards(compiler_cards or [], ctx)
        governed_subtasks = [self._govern_subtask(row) for row in (subtask_instances or [])]
        governed_edges = [self._govern_edge(row) for row in (subtask_edges or [])]
        return {
            "compiler_cards": governed_cards,
            "subtask_instances": governed_subtasks,
            "subtask_edges": governed_edges,
            "report": {
                "compiler_cards": card_report,
                "subtask_count": len(governed_subtasks),
                "subtask_edge_count": len(governed_edges),
            },
        }

    def apply_official_feedback(
        self,
        *,
        compiler_cards: List[Dict[str, Any]],
        outcome: str,
        eval_ref: str = "",
    ) -> Dict[str, Any]:
        normalized = str(outcome or "").strip().lower()
        updates: List[Dict[str, Any]] = []
        promoted_ids: List[str] = []
        suppressed_ids: List[str] = []
        for row in compiler_cards:
            updated = dict(row)
            current = str(updated.get("promotion_state") or GovernanceState.CANDIDATE.value)
            target = GovernanceState.PROMOTED.value if normalized == "resolved" else GovernanceState.SUPPRESSED.value
            if target not in GOVERNANCE_TRANSITIONS.get(current, set()):
                target = current
            updated["promotion_state"] = target
            updated["last_updated"] = datetime.now(timezone.utc).isoformat()
            metadata = dict(updated.get("metadata") or {})
            if eval_ref:
                refs = [str(x).strip() for x in (metadata.get("official_eval_refs") or []) if str(x).strip()]
                refs.append(str(eval_ref).strip())
                metadata["official_eval_refs"] = list(dict.fromkeys(refs))
            metadata["official_eval_status"] = normalized
            updated["metadata"] = metadata
            updates.append(updated)
            card_id = str(updated.get("card_id") or "")
            if target == GovernanceState.PROMOTED.value and card_id:
                promoted_ids.append(card_id)
            if target == GovernanceState.SUPPRESSED.value and card_id:
                suppressed_ids.append(card_id)
        return {
            "cards": updates,
            "promoted_ids": promoted_ids,
            "suppressed_ids": suppressed_ids,
        }

    def attach_eval_context(
        self,
        *,
        subtasks: List[Dict[str, Any]],
        subtask_edges: List[Dict[str, Any]],
        outcome: str,
        eval_ref: str = "",
    ) -> Dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        payload = {
            "official_eval_status": str(outcome or "").strip().lower(),
            "eval_ref": str(eval_ref or "").strip(),
            "attached_at": timestamp,
        }
        updated_subtasks = []
        for row in subtasks:
            updated = dict(row)
            updated["status"] = (
                SubtaskState.DEPRECATED.value
                if str(updated.get("status") or "") == SubtaskState.DEPRECATED.value
                else SubtaskState.EVAL_CONTEXT_ATTACHED.value
            )
            eval_context = dict(updated.get("eval_context") or {})
            eval_context.update(payload)
            updated["eval_context"] = eval_context
            updated_subtasks.append(updated)
        updated_edges = []
        for row in subtask_edges:
            updated = dict(row)
            eval_support = dict(updated.get("eval_support") or {})
            eval_support.update(payload)
            updated["eval_support"] = eval_support
            updated_edges.append(updated)
        return {"subtasks": updated_subtasks, "subtask_edges": updated_edges}

    def _govern_cards(self, cards: List[Dict[str, Any]], context: Dict[str, Any]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        deduped: Dict[str, Dict[str, Any]] = {}
        suppressed_ids: List[str] = []
        for raw in cards:
            row = dict(raw)
            family = str(row.get("family_id") or row.get("card_id") or "").strip()
            if not family:
                continue
            row.setdefault("promotion_state", GovernanceState.CANDIDATE.value)
            row.setdefault("metadata", {})
            if family in deduped:
                current = deduped[family]
                if float(row.get("confidence", 0.0) or 0.0) <= float(current.get("confidence", 0.0) or 0.0):
                    suppressed_ids.append(str(row.get("card_id") or ""))
                    continue
            deduped[family] = row

        ranked = sorted(
            deduped.values(),
            key=lambda row: float(row.get("confidence", 0.0) or 0.0),
            reverse=True,
        )
        kept = ranked[: self.max_cards_per_query]
        skipped = ranked[self.max_cards_per_query :]
        for row in skipped:
            row["promotion_state"] = GovernanceState.SUPPRESSED.value
            metadata = dict(row.get("metadata") or {})
            metadata["governance_reason"] = "growth_cap"
            row["metadata"] = metadata
            suppressed_ids.append(str(row.get("card_id") or ""))

        submission_success = bool(context.get("submission_success", False))
        for row in kept:
            if submission_success:
                metadata = dict(row.get("metadata") or {})
                metadata["pre_eval_success_guard"] = True
                row["metadata"] = metadata
                row["promotion_state"] = GovernanceState.CANDIDATE.value
        report = {
            "input_count": len(cards),
            "kept_count": len(kept),
            "suppressed_ids": [card_id for card_id in suppressed_ids if card_id],
        }
        return kept, report

    @staticmethod
    def _govern_subtask(row: Dict[str, Any]) -> Dict[str, Any]:
        updated = dict(row)
        updated.setdefault("governance_state", GovernanceState.CANDIDATE.value)
        updated.setdefault("status", SubtaskState.PROJECTED_CANDIDATE.value)
        return updated

    @staticmethod
    def _govern_edge(row: Dict[str, Any]) -> Dict[str, Any]:
        updated = dict(row)
        updated.setdefault("governance_state", GovernanceState.CANDIDATE.value)
        updated.setdefault("status", "candidate")
        return updated
