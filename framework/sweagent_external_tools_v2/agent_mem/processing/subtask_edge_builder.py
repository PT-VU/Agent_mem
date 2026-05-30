"""
Build candidate edges between projected subtasks.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..types import GovernanceState, SubtaskEdgeState, SubtaskRelationType
from .v21_shared import stable_family_id


class SubtaskEdgeBuilder:
    """Build candidate relations between projected subtasks."""

    def build(self, subtasks: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not subtasks:
            return []
        attempt_summary = dict(context.get("attempt_summary_v1") or {})
        attempt_id = str(attempt_summary.get("attempt_id") or context.get("attempt_id") or "").strip()
        edges: List[Dict[str, Any]] = []

        for index in range(len(subtasks) - 1):
            src = subtasks[index]
            dst = subtasks[index + 1]
            edges.append(
                self._edge_payload(
                    src=src,
                    dst=dst,
                    edge_type=SubtaskRelationType.PRECEDES.value,
                    attempt_id=attempt_id,
                    confidence_hint=0.62,
                )
            )

        type_buckets: Dict[str, List[Dict[str, Any]]] = {}
        for subtask in subtasks:
            key = str(subtask.get("subtask_type") or "").strip()
            if not key:
                continue
            type_buckets.setdefault(key, []).append(subtask)
        for rows in type_buckets.values():
            if len(rows) < 2:
                continue
            for index in range(1, len(rows)):
                edges.append(
                    self._edge_payload(
                        src=rows[index - 1],
                        dst=rows[index],
                        edge_type=SubtaskRelationType.RETRY_OF.value,
                        attempt_id=attempt_id,
                        confidence_hint=0.56,
                    )
                )

        failed_rows = [row for row in subtasks if str(row.get("local_result_status") or "") == "failed"]
        if len(failed_rows) >= 2:
            first = failed_rows[0]
            second = failed_rows[1]
            shared_files = set(first.get("touched_files") or []) & set(second.get("touched_files") or [])
            if shared_files or first.get("subtask_type") != second.get("subtask_type"):
                edges.append(
                    self._edge_payload(
                        src=first,
                        dst=second,
                        edge_type=SubtaskRelationType.ALTERNATIVE_TO.value,
                        attempt_id=attempt_id,
                        confidence_hint=0.44,
                    )
                )

        deduped: Dict[str, Dict[str, Any]] = {}
        for row in edges:
            deduped[row["edge_id"]] = row
        return list(deduped.values())

    @staticmethod
    def _edge_payload(
        *,
        src: Dict[str, Any],
        dst: Dict[str, Any],
        edge_type: str,
        attempt_id: str,
        confidence_hint: float,
    ) -> Dict[str, Any]:
        edge_id = stable_family_id("edge", src.get("subtask_instance_id"), dst.get("subtask_instance_id"), edge_type)
        return {
            "edge_id": edge_id,
            "src_subtask_id": str(src.get("subtask_instance_id") or ""),
            "dst_subtask_id": str(dst.get("subtask_instance_id") or ""),
            "edge_type": edge_type,
            "effectiveness_candidate": "positive" if edge_type == SubtaskRelationType.PRECEDES.value else "unknown",
            "supporting_attempt_ids": [attempt_id] if attempt_id else [],
            "confidence_hint": round(float(confidence_hint), 6),
            "status": (
                SubtaskEdgeState.SUPPORTED_CANDIDATE.value
                if edge_type == SubtaskRelationType.PRECEDES.value
                else SubtaskEdgeState.CANDIDATE.value
            ),
            "governance_state": GovernanceState.CANDIDATE.value,
            "eval_support": {},
        }
