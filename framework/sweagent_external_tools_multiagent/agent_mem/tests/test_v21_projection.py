from __future__ import annotations

from ..core.problem_file import ActionType, Outcome, ProblemFile
from ..processing.subtask_edge_builder import SubtaskEdgeBuilder
from ..processing.subtask_projector import SubtaskProjector


def _action(action_id: str, *, action_type: ActionType, text: str, touched: list[str] | None = None) -> ProblemFile:
    return ProblemFile(
        action_id=action_id,
        task_id="task-1",
        action_type=action_type,
        outcome=Outcome.SUCCESS,
        intent_text=text,
        action_text=text,
        touched_files=touched or [],
    )


def test_subtask_projector_maps_subblocks_to_candidate_rows():
    projector = SubtaskProjector()
    actions = [
        _action("a1", action_type=ActionType.TOOL_CALL, text="inspect failing files", touched=["pkg/a.py"]),
        _action("a2", action_type=ActionType.RUN_TEST, text="pytest tests/test_a.py -q"),
        _action("a3", action_type=ActionType.CODE_EDIT, text="patch pkg/a.py", touched=["pkg/a.py"]),
    ]
    attempt_summary = {
        "summary_id": "sum-1",
        "instance_id": "inst-1",
        "run_id": "run-1",
        "attempt_id": "attempt-1",
        "trace_id": "trace-1",
        "final_outcome": "unresolved",
        "next_best_actions": ["validate_current_changed_files_before_broadening_scope"],
        "subblock_analysis": [
            {
                "subproblem_type": "localize_fix",
                "goal": "localize the minimal patch surface",
                "key_actions": ["a1", "a3"],
                "positive_contribution": ["identified the target module"],
                "negative_contribution": [],
                "failure_source": "execution",
                "prefer_actions": ["edit_target_files_then_validate"],
            },
            {
                "subproblem_type": "target_validation",
                "goal": "validate the focused fix",
                "key_actions": ["a2"],
                "positive_contribution": [],
                "negative_contribution": ["validation did not confirm the patch candidate"],
                "failure_source": "execution",
                "prefer_actions": ["reuse_the_best_existing_reproduction_path"],
            },
        ],
    }
    report = projector.project(
        {
            "attempt_summary_v1": attempt_summary,
            "actions": actions,
            "run_done_context": {"patch_digest": "sha1", "task_closed_cleanly": False},
        }
    )
    assert len(report) == 2
    assert report[0]["status"] == "locally_supported"
    assert report[0]["touched_files"] == ["pkg/a.py"]
    assert report[1]["status"] == "locally_failed"
    assert report[1]["tests_run"] == 1


def test_subtask_edge_builder_builds_precedes_and_candidate_edges():
    builder = SubtaskEdgeBuilder()
    subtasks = [
        {
            "subtask_instance_id": "sub-1",
            "subtask_type": "localize_fix",
            "local_result_status": "supported",
            "touched_files": ["pkg/a.py"],
        },
        {
            "subtask_instance_id": "sub-2",
            "subtask_type": "target_validation",
            "local_result_status": "failed",
            "touched_files": ["pkg/a.py"],
        },
    ]
    edges = builder.build(subtasks, {"attempt_summary_v1": {"attempt_id": "attempt-1"}})
    assert edges
    types = {row["edge_type"] for row in edges}
    assert "PRECEDES" in types
