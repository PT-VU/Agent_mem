"""
Cold-path projected subtask builder for AgentMem v2.1.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..core.problem_file import ActionType, ProblemFile
from ..types import EvidenceLevel, GovernanceState, SubtaskState
from .v21_shared import stable_family_id


class SubtaskProjector:
    """Build candidate SubtaskInstance rows from attempt evidence."""

    def project(self, attempt_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        attempt_summary = dict(attempt_payload.get("attempt_summary_v1") or {})
        actions = [
            row for row in (attempt_payload.get("actions") or []) if isinstance(row, ProblemFile)
        ]
        action_by_id = {
            str(action.action_id): action
            for action in actions
            if str(action.action_id).strip()
        }
        run_done_context = dict(attempt_payload.get("run_done_context") or {})
        subblocks = [
            row for row in (attempt_summary.get("subblock_analysis") or []) if isinstance(row, dict)
        ]
        if not subblocks:
            return []

        instance_id = str(attempt_summary.get("instance_id") or attempt_payload.get("instance_id") or "").strip()
        run_id = str(attempt_summary.get("run_id") or attempt_payload.get("run_id") or "").strip()
        attempt_id = str(attempt_summary.get("attempt_id") or attempt_payload.get("attempt_id") or "").strip()
        episode_id = "::".join([token for token in (instance_id, run_id, attempt_id) if token]) or "unknown_episode"

        subtasks: List[Dict[str, Any]] = []
        for index, block in enumerate(subblocks, start=1):
            action_ids = [
                str(x).strip()
                for x in (block.get("key_actions") or [])
                if str(x).strip()
            ]
            block_actions = [action_by_id[action_id] for action_id in action_ids if action_id in action_by_id]
            touched_files: List[str] = []
            tests_run = 0
            validation_commands: List[str] = []
            for action in block_actions:
                for path in action.touched_files:
                    text = str(path).strip()
                    if text and text not in touched_files:
                        touched_files.append(text)
                if action.action_type == ActionType.RUN_TEST:
                    tests_run += 1
                    command = str(action.action_text or "").strip()
                    if command and command not in validation_commands:
                        validation_commands.append(command)

            failure_source = str(block.get("failure_source") or "").strip().lower()
            positives = [str(x).strip() for x in (block.get("positive_contribution") or []) if str(x).strip()]
            negatives = [str(x).strip() for x in (block.get("negative_contribution") or []) if str(x).strip()]
            local_result_status = "supported" if positives else ("failed" if negatives or failure_source in {"plan", "execution", "mixed"} else "candidate")
            status = (
                SubtaskState.LOCALLY_SUPPORTED.value
                if local_result_status == "supported"
                else SubtaskState.LOCALLY_FAILED.value
                if local_result_status == "failed"
                else SubtaskState.PROJECTED_CANDIDATE.value
            )
            failure_type = negatives[0] if negatives else failure_source or str(attempt_summary.get("final_outcome") or "unknown")
            next_steps = [
                str(x).strip()
                for x in (block.get("prefer_actions") or attempt_summary.get("next_best_actions") or [])
                if str(x).strip()
            ][:4]
            subtask_type = str(block.get("subproblem_type") or "unknown").strip() or "unknown"
            goal = str(block.get("goal") or "").strip() or subtask_type
            confidence = 0.72 if positives else 0.58 if negatives else 0.45
            subtask_id = stable_family_id("subtask", episode_id, index, subtask_type, goal)
            subtasks.append(
                {
                    "subtask_instance_id": subtask_id,
                    "episode_id": episode_id,
                    "instance_id": instance_id,
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "trace_id": str(attempt_summary.get("trace_id") or attempt_payload.get("trace_id") or ""),
                    "subtask_type": subtask_type,
                    "goal": goal[:240],
                    "action_ids": action_ids[:12],
                    "touched_files": touched_files[:12],
                    "tests_run": tests_run,
                    "validation_commands": validation_commands[:6],
                    "local_result_status": local_result_status,
                    "failure_type": failure_type[:280],
                    "recommended_next_steps": next_steps,
                    "projection_confidence": round(confidence, 6),
                    "status": status,
                    "governance_state": GovernanceState.CANDIDATE.value,
                    "evidence_level": EvidenceLevel.ATTEMPT.value,
                    "summary_id": str(attempt_summary.get("summary_id") or ""),
                    "source_action_ids": action_ids[:12],
                    "run_done_context_refs": {
                        "patch_digest": str(run_done_context.get("patch_digest") or ""),
                        "task_closed_cleanly": bool(run_done_context.get("task_closed_cleanly", False)),
                    },
                    "eval_context": {},
                }
            )
        return subtasks
