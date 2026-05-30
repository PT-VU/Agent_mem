from __future__ import annotations

import tempfile

from ..integration.sweagent_adapter import SWEAgentAdapter


def test_action_error_micro_pattern_and_validation_rule_generated(monkeypatch):
    monkeypatch.setenv("AGENT_MEM_ENABLE_ONLINE_EMBEDDINGS", "0")
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = SWEAgentAdapter(storage_dir=tmpdir, evidence_dir=tmpdir)
        repair_suggestions = {
            "confidence": 0.72,
            "next_step_fix": "Run the focused failing test before editing more code.",
            "evidence_refs": ["act_1"],
            "recommendations": [
                {
                    "type": "failure_card_v2",
                    "recommendation": "Run the focused failing test before editing more code.",
                    "verification_commands": ["pytest tests/test_target.py -q"],
                }
            ],
        }
        event_data = {
            "trace_id": "trace:1",
            "instance_id": "inst-1",
            "run_id": "run-1",
            "attempt_id": "attempt-1",
        }
        pattern = adapter._build_action_error_micro_pattern(
            task_id="task-1",
            event_data=event_data,
            query_type="test_failure_fix",
            error_type="test_failure",
            error_message="AssertionError: expected x got y",
            current_action="pytest -q",
            action_id="action-1",
            repair_suggestions=repair_suggestions,
        )
        rule = adapter._build_action_error_validation_rule(
            task_id="task-1",
            event_data=event_data,
            query_type="test_failure_fix",
            error_type="test_failure",
            repair_suggestions=repair_suggestions,
        )
        assert pattern is not None
        assert pattern["fix_action_template"].startswith("Run the focused failing test")
        assert "pytest -q" in pattern["expected_verification"]
        assert pattern["metadata"]["source"] == "action_error_micro_extraction"
        assert rule is not None
        assert rule["metadata"]["source"] == "action_error_validation_guard"
        assert rule["should_check"] == "pytest tests/test_target.py -q"


def test_action_error_micro_pattern_skips_infra_failures(monkeypatch):
    monkeypatch.setenv("AGENT_MEM_ENABLE_ONLINE_EMBEDDINGS", "0")
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = SWEAgentAdapter(storage_dir=tmpdir, evidence_dir=tmpdir)
        repair_suggestions = {
            "confidence": 0.9,
            "next_step_fix": "Retry after Docker recovers.",
            "recommendations": [{"verification_commands": ["docker info"]}],
        }
        event_data = {"trace_id": "trace:infra"}
        assert (
            adapter._build_action_error_micro_pattern(
                task_id="task-infra",
                event_data=event_data,
                query_type="error_recovery",
                error_type="environment_error",
                error_message="DockerPullError",
                current_action="docker pull image",
                action_id="action-infra",
                repair_suggestions=repair_suggestions,
            )
            is None
        )
        assert (
            adapter._build_action_error_validation_rule(
                task_id="task-infra",
                event_data=event_data,
                query_type="error_recovery",
                error_type="environment_error",
                repair_suggestions=repair_suggestions,
            )
            is None
        )


def test_run_done_submission_writes_candidate_experience(monkeypatch):
    monkeypatch.setenv("AGENT_MEM_ENABLE_ONLINE_EMBEDDINGS", "0")
    monkeypatch.setenv("AGENT_MEM_ENABLE_LLM_EXTRACTION", "0")
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = SWEAgentAdapter(storage_dir=tmpdir, evidence_dir=tmpdir)
        adapter.handle_plan_generated(
            {
                "agent": "main",
                "thought": "Inspect the failure and prepare a fix.",
                "action": "1. read files\n2. patch\n3. run tests",
                "instance_id": "inst-submitted",
                "run_id": "run-submitted",
                "attempt_id": "attempt-1",
                "trace_id": "trace-plan",
            }
        )
        result = adapter.handle_run_done(
            {
                "agent": "main",
                "instance_id": "inst-submitted",
                "run_id": "run-submitted",
                "attempt_id": "attempt-1",
                "trace_id": "trace-done",
                "exit_status": "submitted",
                "has_submission": True,
            }
        )
        assert result["submission_success"] is True
        assert result["resolved_like_success"] is False
        assert result["failure_card_v2"]["reason"] == "submission_pending_official_eval"
        exp_rows = adapter.graph_store.list_abstract_experiences(max_results=5)
        assert exp_rows
        assert exp_rows[0]["metadata"]["promotion_state"] == "candidate"
        assert exp_rows[0]["metadata"]["evidence_stage"] == "submission"
        assert exp_rows[0]["source_instance_id"] == "inst-submitted"
        assert "run-submitted" in (exp_rows[0].get("source_run_ids") or [])
        assert "attempt-1" in (exp_rows[0].get("source_attempt_ids") or [])
        summary = adapter.graph_store.get_latest_attempt_summary(instance_id="inst-submitted")
        assert summary is not None
        assert summary["instance_id"] == "inst-submitted"
        assert summary["attempt_id"] == "attempt-1"


def test_official_eval_feedback_updates_attempt_summary(monkeypatch):
    monkeypatch.setenv("AGENT_MEM_ENABLE_ONLINE_EMBEDDINGS", "0")
    monkeypatch.setenv("AGENT_MEM_ENABLE_LLM_EXTRACTION", "0")
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = SWEAgentAdapter(storage_dir=tmpdir, evidence_dir=tmpdir)
        adapter.handle_plan_generated(
            {
                "agent": "main",
                "thought": "Inspect target function and validate the minimal fix.",
                "action": "1. inspect\n2. patch\n3. test",
                "instance_id": "inst-feedback",
                "run_id": "run-feedback",
                "attempt_id": "attempt-01",
                "trace_id": "trace-plan",
            }
        )
        adapter.handle_run_done(
            {
                "agent": "main",
                "instance_id": "inst-feedback",
                "run_id": "run-feedback",
                "attempt_id": "attempt-01",
                "trace_id": "trace-done",
                "exit_status": "submitted",
                "has_submission": True,
            }
        )
        feedback = adapter.apply_evaluation_feedback(
            {
                "instance_id": "inst-feedback",
                "run_id": "run-feedback",
                "attempt_id": "attempt-01",
                "trace_id": "trace-done",
                "official_eval_status": "unresolved",
                "eval_ref": "eval://test",
            }
        )
        summary = adapter.graph_store.get_latest_attempt_summary(instance_id="inst-feedback")
        assert summary is not None
        assert summary["final_outcome"] == "unresolved"
        assert "eval://test" in (summary.get("official_eval_refs") or [])
        assert feedback["feedback_report"]["attempt_summary_id"] == summary["summary_id"]


def test_official_eval_feedback_updates_existing_attempt_summary_even_with_different_eval_run_id(monkeypatch):
    monkeypatch.setenv("AGENT_MEM_ENABLE_ONLINE_EMBEDDINGS", "0")
    monkeypatch.setenv("AGENT_MEM_ENABLE_LLM_EXTRACTION", "0")
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = SWEAgentAdapter(storage_dir=tmpdir, evidence_dir=tmpdir)
        adapter.handle_plan_generated(
            {
                "agent": "main",
                "thought": "Inspect target function and validate the minimal fix.",
                "action": "1. inspect\n2. patch\n3. test",
                "instance_id": "inst-feedback-runmismatch",
                "run_id": "run-agent-attempt-03",
                "attempt_id": "attempt-03",
                "trace_id": "trace-plan",
            }
        )
        adapter.handle_run_done(
            {
                "agent": "main",
                "instance_id": "inst-feedback-runmismatch",
                "run_id": "run-agent-attempt-03",
                "attempt_id": "attempt-03",
                "trace_id": "trace-done",
                "exit_status": "submitted",
                "has_submission": True,
            }
        )
        before = adapter.graph_store.get_latest_attempt_summary(instance_id="inst-feedback-runmismatch")
        assert before is not None
        assert before["attempt_id"] == "attempt-03"
        original_summary_id = before["summary_id"]

        feedback = adapter.apply_evaluation_feedback(
            {
                "instance_id": "inst-feedback-runmismatch",
                "run_id": "repeat5_eval_worker_inst-feedback-runmismatch_attempt_03",
                "attempt_id": "attempt-03",
                "trace_id": "trace-done",
                "official_eval_status": "incomplete",
                "eval_ref": "eval://attempt-03",
            }
        )
        after = adapter.graph_store.get_latest_attempt_summary(instance_id="inst-feedback-runmismatch")
        assert after is not None
        assert after["summary_id"] == original_summary_id
        assert after["final_outcome"] == "incomplete"
        assert feedback["feedback_report"]["attempt_summary_id"] == original_summary_id


def test_fallback_attempt_summary_contains_action_grounded_content(monkeypatch):
    monkeypatch.setenv("AGENT_MEM_ENABLE_ONLINE_EMBEDDINGS", "0")
    monkeypatch.setenv("AGENT_MEM_ENABLE_LLM_EXTRACTION", "0")
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = SWEAgentAdapter(storage_dir=tmpdir, evidence_dir=tmpdir)
        adapter.handle_plan_generated(
            {
                "agent": "main",
                "thought": "Inspect fitswcs.py and validate the smallest WCS fix first.",
                "action": "1. inspect fitswcs.py\n2. create test_quiet.py\n3. run pytest -q",
                "instance_id": "inst-summary",
                "run_id": "run-summary",
                "attempt_id": "attempt-01",
                "trace_id": "trace-plan",
            }
        )
        adapter.handle_run_done(
            {
                "agent": "main",
                "instance_id": "inst-summary",
                "run_id": "run-summary",
                "attempt_id": "attempt-01",
                "trace_id": "trace-done",
                "exit_status": "incomplete",
                "has_submission": False,
            }
        )
        summary = adapter.graph_store.get_latest_attempt_summary(instance_id="inst-summary")
        assert summary is not None
        assert summary["initial_plan_outline"]
        assert summary["actual_execution_outline"]
        assert summary["next_best_actions"]
        assert (
            summary["confirmed_signals"]
            or summary["failed_strategies"]
            or summary["best_partial_progress"]
        )


def test_official_eval_feedback_backfills_missing_attempt_summary(monkeypatch):
    monkeypatch.setenv("AGENT_MEM_ENABLE_ONLINE_EMBEDDINGS", "0")
    monkeypatch.setenv("AGENT_MEM_ENABLE_LLM_EXTRACTION", "0")
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = SWEAgentAdapter(storage_dir=tmpdir, evidence_dir=tmpdir)
        adapter.handle_plan_generated(
            {
                "agent": "main",
                "thought": "Reproduce and patch the target function.",
                "action": "1. reproduce\n2. patch\n3. validate",
                "instance_id": "inst-backfill",
                "run_id": "run-attempt-01",
                "attempt_id": "attempt-01",
                "trace_id": "trace-plan",
            }
        )
        adapter.handle_run_done(
            {
                "agent": "main",
                "instance_id": "inst-backfill",
                "run_id": "run-attempt-01",
                "attempt_id": "attempt-01",
                "trace_id": "trace-done",
                "exit_status": "submitted",
                "has_submission": True,
            }
        )
        feedback = adapter.apply_evaluation_feedback(
            {
                "instance_id": "inst-backfill",
                "run_id": "repeat5_single_withmem_closedloop_inst-backfill_attempt_02",
                "official_eval_status": "incomplete",
                "eval_ref": "eval://attempt-02",
                "task_summary": "Need recoverable attempt summary for attempt 02",
                "changed_files": ["pkg/demo.py"],
                "patch_summary": {"changed_file_count": 1},
                "validation_summary": {"commands": ["pytest -q tests/test_demo.py"]},
            }
        )
        summary = adapter.graph_store.get_latest_attempt_summary(instance_id="inst-backfill")
        assert summary is not None
        assert summary["attempt_id"] == "attempt-02"
        assert summary["final_outcome"] == "incomplete"
        assert "eval://attempt-02" in (summary.get("official_eval_refs") or [])
        assert feedback["feedback_report"]["attempt_summary_id"] == summary["summary_id"]


def test_extract_read_ids_supports_action_and_task_level_references(monkeypatch):
    monkeypatch.setenv("AGENT_MEM_ENABLE_ONLINE_EMBEDDINGS", "0")
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = SWEAgentAdapter(storage_dir=tmpdir, evidence_dir=tmpdir)
        read_ids = adapter._extract_read_ids(
            {
                "recommendations": [
                    {"type": "workflow_step", "action_id": "action-1"},
                    {"type": "legacy_experience", "task_id": "task-2"},
                    {"type": "abstract_pattern", "experience_id": "exp-3"},
                    {"type": "attempt_summary_v1", "summary_id": "sum-4"},
                ]
            }
        )
        assert read_ids == ["action-1", "task-2", "exp-3", "sum-4"]


def test_run_done_v21_projection_and_compiler_write_sidecar(monkeypatch):
    monkeypatch.setenv("AGENT_MEM_ENABLE_ONLINE_EMBEDDINGS", "0")
    monkeypatch.setenv("AGENT_MEM_ENABLE_LLM_EXTRACTION", "0")
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = SWEAgentAdapter(
            storage_dir=tmpdir,
            evidence_dir=tmpdir,
            v21_config={
                "enable_sidecar": True,
                "enable_subtask_projection": True,
                "enable_card_compiler": True,
                "enable_governance": True,
            },
        )
        adapter.handle_plan_generated(
            {
                "agent": "main",
                "thought": "Inspect target function and validate the minimal patch.",
                "action": "1. inspect\n2. patch\n3. run focused test",
                "instance_id": "inst-v21",
                "run_id": "run-v21",
                "attempt_id": "attempt-01",
                "trace_id": "trace-plan",
            }
        )
        adapter.handle_action_success(
            action="apply_patch minimal fix",
            thought="patch the localized file",
            agent_name="main",
            instance_id="inst-v21",
            run_id="run-v21",
            attempt_id="attempt-01",
            trace_id="trace-edit",
            step_index=2,
            touched_files=["pkg/a.py"],
            diff_content="diff --git a/pkg/a.py b/pkg/a.py\n+fix\n",
        )
        adapter.handle_action_success(
            action="pytest tests/test_target.py -q",
            thought="run focused validation",
            agent_name="main",
            instance_id="inst-v21",
            run_id="run-v21",
            attempt_id="attempt-01",
            trace_id="trace-test",
            step_index=3,
            test_output="1 passed in 0.12s",
        )
        result = adapter.handle_run_done(
            {
                "agent": "main",
                "instance_id": "inst-v21",
                "run_id": "run-v21",
                "attempt_id": "attempt-01",
                "trace_id": "trace-done",
                "exit_status": "submitted",
                "has_submission": True,
            }
        )
        assert result["success_fact_report"]["enabled"] is True
        assert result["run_done_context"]["patch_digest"]
        assert result["subtask_projection"]["enabled"] is True
        assert result["subtask_projection"]["subtask_count"] >= 1
        assert result["compiler_cards_v21"]["enabled"] is True
        assert result["compiler_cards_v21"]["compiled_count"] >= 1
        assert adapter.graph_store.list_compiler_cards_v21(max_results=5)


def test_official_eval_feedback_updates_v21_cards_and_sidecar(monkeypatch):
    monkeypatch.setenv("AGENT_MEM_ENABLE_ONLINE_EMBEDDINGS", "0")
    monkeypatch.setenv("AGENT_MEM_ENABLE_LLM_EXTRACTION", "0")
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = SWEAgentAdapter(
            storage_dir=tmpdir,
            evidence_dir=tmpdir,
            v21_config={
                "enable_sidecar": True,
                "enable_subtask_projection": True,
                "enable_card_compiler": True,
                "enable_governance": True,
            },
        )
        adapter.handle_plan_generated(
            {
                "agent": "main",
                "thought": "Inspect and validate focused fix.",
                "action": "1. inspect\n2. patch\n3. test",
                "instance_id": "inst-v21-eval",
                "run_id": "run-v21-eval",
                "attempt_id": "attempt-02",
                "trace_id": "trace-plan",
            }
        )
        adapter.handle_action_success(
            action="pytest tests/test_target.py -q",
            thought="run focused validation",
            agent_name="main",
            instance_id="inst-v21-eval",
            run_id="run-v21-eval",
            attempt_id="attempt-02",
            trace_id="trace-test",
            step_index=2,
            test_output="1 passed in 0.10s",
        )
        adapter.handle_run_done(
            {
                "agent": "main",
                "instance_id": "inst-v21-eval",
                "run_id": "run-v21-eval",
                "attempt_id": "attempt-02",
                "trace_id": "trace-done",
                "exit_status": "submitted",
                "has_submission": True,
            }
        )
        feedback = adapter.apply_evaluation_feedback(
            {
                "instance_id": "inst-v21-eval",
                "run_id": "run-v21-eval",
                "attempt_id": "attempt-02",
                "trace_id": "trace-done",
                "official_eval_status": "resolved",
                "eval_ref": "eval://v21",
            }
        )
        assert feedback["feedback_report"]["promoted_card_ids"]
        latest_subtasks = adapter.episode_ledger_store.load_latest_records(
            stream="subtask_instances",
            key_field="subtask_instance_id",
            filters={"instance_id": "inst-v21-eval", "attempt_id": "attempt-02"},
        )
        assert latest_subtasks
        assert latest_subtasks[0]["status"] == "eval_context_attached"
