from __future__ import annotations

import json
from pathlib import Path

from ..processing.official_eval_bridge import (
    build_feedback_event,
    build_outcome_map,
    extract_changed_files_from_patch,
    extract_task_summary,
    extract_validation_summary,
    infer_attempt_id,
    infer_eval_outcome,
)


def test_build_outcome_map_reads_eval_lists() -> None:
    outcome_map = build_outcome_map(
        {
            "resolved_ids": ["a"],
            "unresolved_ids": ["b"],
            "incomplete_ids": ["c"],
            "error_ids": ["d"],
        }
    )
    assert outcome_map == {
        "a": "resolved",
        "b": "unresolved",
        "c": "incomplete",
        "d": "incomplete",
    }


def test_infer_attempt_id_from_run_id() -> None:
    assert infer_attempt_id("repeat5_single_withmem_closedloop_astropy__astropy-11693_attempt_02") == "attempt-02"
    assert infer_attempt_id("run-without-attempt") == ""


def test_extract_changed_files_and_validation_and_task_summary(tmp_path: Path) -> None:
    instance_dir = tmp_path / "demo__case-1"
    instance_dir.mkdir()
    patch = (
        "diff --git a/pkg/a.py b/pkg/a.py\n"
        "--- a/pkg/a.py\n"
        "+++ b/pkg/a.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "diff --git a/pkg/b.py b/pkg/b.py\n"
        "--- a/pkg/b.py\n"
        "+++ b/pkg/b.py\n"
    )
    (instance_dir / "demo__case-1.pred").write_text(
        json.dumps(
            {
                "instance_id": "demo__case-1",
                "model_name_or_path": "demo",
                "model_patch": patch,
            }
        ),
        encoding="utf-8",
    )
    traj_payload = {
        "trajectory": [
            {
                "query": [
                    {
                        "content": (
                            "<pr_description>\n"
                            "Fix crash when parser receives empty input.\n"
                            "</pr_description>"
                        )
                    }
                ],
                "action": "bash -lc 'pytest tests/test_parser.py -q'",
            },
            {
                "query": [],
                "action": "str_replace_editor create /testbed/reproduce_error.py --file_text 'print(1)'",
            },
            {
                "query": [],
                "action": "bash -lc 'python reproduce_error.py'",
            },
        ],
        "info": {"submission": patch},
    }
    (instance_dir / "demo__case-1.traj").write_text(json.dumps(traj_payload), encoding="utf-8")

    assert extract_changed_files_from_patch(patch) == ["pkg/a.py", "pkg/b.py"]
    assert extract_task_summary(instance_dir) == "Fix crash when parser receives empty input."
    assert extract_validation_summary(instance_dir)["commands"] == [
        "bash -lc 'pytest tests/test_parser.py -q'",
        "bash -lc 'python reproduce_error.py'",
    ]

    event = build_feedback_event(
        instance_id="demo__case-1",
        base_outcome="unresolved",
        report_path="/tmp/report.json",
        instance_dir=instance_dir,
        run_id="demo-run_attempt_02",
    )
    assert event["official_eval_status"] == "unresolved"
    assert event["attempt_id"] == "attempt-02"
    assert event["changed_files"] == ["pkg/a.py", "pkg/b.py"]
    assert event["validation_summary"]["commands"]
    assert "Fix crash when parser receives empty input." in event["task_summary"]


def test_infer_eval_outcome_marks_infra_for_runtime_failures(tmp_path: Path) -> None:
    instance_dir = tmp_path / "demo__case-2"
    instance_dir.mkdir()
    (instance_dir / "demo__case-2.trace.log").write_text(
        "DeepSeekException: Insufficient Balance\nDockerPullError\n",
        encoding="utf-8",
    )
    outcome, reason = infer_eval_outcome("incomplete", instance_dir)
    assert outcome == "infra_failure"
    assert reason in {"llm_insufficient_balance", "docker_pull_error"}
