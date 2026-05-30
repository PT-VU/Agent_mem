"""
Stage-4 focused tests: failure cards and retrieval query routing.
"""

from __future__ import annotations

import tempfile

from ..core.problem_file import ActionType, FailureSignature, Outcome, ProblemFile
from ..processing.failure_card_builder import FailureCardBuilder
from ..retrieval.memory_agent import MemoryAgent
from ..storage.graph_store import GraphStore


def _build_action(
    task_id: str,
    *,
    step_index: int,
    action_type: ActionType,
    outcome: Outcome,
    intent: str,
    action_family: str,
) -> ProblemFile:
    return ProblemFile(
        task_id=task_id,
        step_index=step_index,
        action_type=action_type,
        action_family=action_family,
        action_text=intent,
        intent_text=intent,
        outcome=outcome,
        source_event="action_error" if outcome == Outcome.FAIL else "plan_generated",
        metadata={"test_only": True},
    )


def test_failure_card_builder_basic_fields():
    builder = FailureCardBuilder()
    actions = [
        _build_action(
            "task_stage4",
            step_index=0,
            action_type=ActionType.TOOL_CALL,
            outcome=Outcome.SUCCESS,
            intent="inspect repo structure",
            action_family="planning",
        ),
        _build_action(
            "task_stage4",
            step_index=1,
            action_type=ActionType.RUN_TEST,
            outcome=Outcome.FAIL,
            intent="run pytest -q",
            action_family="test_failure_fix",
        ),
    ]
    card = builder.build_from_unresolved(
        task_id="task_stage4",
        actions=actions,
        task_summary="test failure after initial planning",
        rca_report={
            "root_cause_nodes": [actions[0].action_id],
            "propagation_chain": [actions[0].action_id, actions[1].action_id],
            "error_module": "action",
            "confidence": 0.73,
        },
    )
    assert card.task_id == "task_stage4"
    assert card.status == "unresolved"
    assert card.root_cause_nodes == [actions[0].action_id]
    assert len(card.candidate_fix_actions) >= 1
    assert len(card.verification_commands) >= 1
    assert len(card.action_trace_snippet) >= 2
    assert card.failure_class == "agent_failure_card"


def test_failure_card_builder_classifies_infra_failures():
    builder = FailureCardBuilder()
    actions = [
        _build_action(
            "task_infra",
            step_index=0,
            action_type=ActionType.RUN_TEST,
            outcome=Outcome.FAIL,
            intent="docker pull docker.io/swebench/example:latest",
            action_family="test_failure_fix",
        ),
    ]
    actions[0].metadata = {"error_message": "DockerPullError after daemon unavailable"}
    actions[0].failure_signature = FailureSignature(
        error_type="environment_error",
        error_tokens=["dockerpullerror", "daemon", "unavailable"],
    )
    card = builder.build_from_unresolved(
        task_id="task_infra",
        actions=actions,
        task_summary="docker pull failed before task execution",
        rca_report=None,
    )
    assert card.failure_class == "infra_failure_card"


def test_graph_store_failure_card_upsert_query():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(storage_dir=tmpdir)
        card_id = store.upsert_failure_card_v2(
            {
                "task_id": "task_a",
                "error_signature": {"error_type": "import_error"},
                "candidate_fix_actions": ["verify_dependency_and_import_path_before_retry"],
                "verification_commands": ["pytest -q"],
                "confidence": 0.66,
            }
        )
        assert card_id
        rows = store.query_failure_cards_v2(query_text="import dependency", error_type="import_error", max_results=3)
        assert len(rows) == 1
        assert rows[0]["card_id"] == card_id
        assert rows[0]["score"] >= 0.0
        assert rows[0]["failure_class"] == "agent_failure_card"


def test_graph_store_failure_cards_exclude_infra_by_default():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(storage_dir=tmpdir)
        infra_id = store.upsert_failure_card_v2(
            {
                "task_id": "task_infra",
                "error_signature": {"error_type": "environment_error"},
                "candidate_fix_actions": ["wait_for_docker_daemon_then_retry"],
                "verification_commands": ["docker info"],
                "confidence": 0.5,
                "metadata": {"task_summary": "DockerPullError after daemon unavailable"},
            }
        )
        agent_id = store.upsert_failure_card_v2(
            {
                "task_id": "task_agent",
                "error_signature": {"error_type": "import_error"},
                "candidate_fix_actions": ["verify_dependency_and_import_path_before_retry"],
                "verification_commands": ["pytest -q"],
                "confidence": 0.7,
            }
        )
        default_rows = store.query_failure_cards_v2(query_text="error", max_results=5)
        assert [row["card_id"] for row in default_rows] == [agent_id]
        all_rows = store.query_failure_cards_v2(query_text="error", max_results=5, include_infra=True)
        assert {row["card_id"] for row in all_rows} == {infra_id, agent_id}


def test_problem_file_trace_id_roundtrip():
    pf = ProblemFile(
        task_id="trace_task",
        action_type=ActionType.TOOL_CALL,
        action_family="planning",
        intent_text="inspect",
        action_text="ls -la",
        outcome=Outcome.SUCCESS,
        trace_id="run123:instA:1:plan_generated:1",
    )
    payload = pf.to_dict()
    restored = ProblemFile.from_dict(payload)
    assert restored.trace_id == pf.trace_id


def test_memory_agent_query_type_routes_with_failure_cards():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(storage_dir=tmpdir)
        agent = MemoryAgent(store)
        store.upsert_failure_card_v2(
            {
                "task_id": "task_b",
                "error_signature": {"error_type": "test_failure"},
                "candidate_fix_actions": ["run_related_tests_first_then_apply_targeted_fix"],
                "verification_commands": ["pytest -q"],
                "confidence": 0.74,
                "root_cause_nodes": ["a1"],
                "evidence_refs": ["a1", "stderr://a1"],
            }
        )
        pf = ProblemFile(
            task_id="task_b",
            action_type=ActionType.RUN_TEST,
            action_family="test_failure_fix",
            intent_text="run pytest -q",
            action_text="pytest -q",
            outcome=Outcome.FAIL,
        )
        result = agent.retrieve_for_query_type(
            query_type="test_failure_fix",
            current_action="pytest -q",
            current_problem_file=pf,
            error_type="test_failure",
            error_message="AssertionError in test_x",
        )
        assert result["retrieval_debug"]["query_type"] == "test_failure_fix"
        assert isinstance(result.get("recommendations", []), list)
        assert any(item.get("type") == "failure_card_v2" for item in result.get("recommendations", []))


def test_memory_agent_suppresses_workflow_and_planning_after_closure():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(storage_dir=tmpdir)
        agent = MemoryAgent(store)
        rows = [
            {
                "type": "workflow_step",
                "family_id": "workflow_step:consider: generate plan for task...",
                "recommendation": "Consider: Generate plan for task...",
                "confidence": 0.8,
                "score": 0.9,
            },
            {
                "type": "abstract_pattern",
                "family_id": "planning_loop__planning_transition_failure__force_progression_transition",
                "normalized_pattern_type": "planning_loop",
                "recommendation": "Force progression transition after planning.",
                "confidence": 0.85,
                "score": 0.91,
            },
            {
                "type": "abstract_pattern",
                "family_id": "validation_gap:add_local_validation",
                "normalized_pattern_type": "validation_gap",
                "recommendation": "Validate the minimal fix before further exploration.",
                "confidence": 0.83,
                "score": 0.88,
            },
        ]
        out = agent._dedup_and_rerank_recommendations(
            rows,
            limit=5,
            runtime_guard={
                "closure_active": True,
                "blocked_families": ["workflow_step", "planning_loop"],
            },
        )
        families = {str(row.get("family_id")) for row in out}
        assert "workflow_step:consider: generate plan for task..." not in families
        assert "planning_loop__planning_transition_failure__force_progression_transition" not in families
        assert "validation_gap:add_local_validation" in families


def test_memory_agent_retrieves_latest_attempt_summary_for_same_instance():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(storage_dir=tmpdir)
        agent = MemoryAgent(store)
        store.upsert_attempt_summary_v1(
            {
                "instance_id": "inst-summary",
                "run_id": "run-1",
                "attempt_id": "attempt-01",
                "problem_goal": "fix target WCS plotting path",
                "failed_strategies": [
                    {
                        "subproblem_type": "reproduce_issue",
                        "strategy_label": "ad_hoc_repro_script_loop",
                        "reason": "created multiple scripts without converging",
                        "avoid_actions": ["create_new_repro_script_after_repro_confirmed"],
                    }
                ],
                "next_best_actions": ["run_target_validation_before_submission"],
            }
        )
        pf = ProblemFile(
            task_id="task-summary",
            step_index=1,
            action_type=ActionType.TOOL_CALL,
            action_family="planning",
            intent_text="inspect target function",
            action_text="inspect target function",
            outcome=Outcome.SUCCESS,
        )
        result = agent.retrieve_for_planning(
            task_context={
                "instruction": "fix the WCS plotting path",
                "summary": "investigate prior failure",
                "env_signature": {
                    "instance_id": "inst-summary",
                    "attempt_id": "attempt-02",
                },
            },
            current_action="inspect target function and validate current fix",
            agent_name="main",
            current_problem_file=pf,
        )
        assert result.get("attempt_summary", {}).get("instance_id") == "inst-summary"
        summary_rows = [item for item in result.get("recommendations", []) if item.get("type") == "attempt_summary_v1"]
        assert summary_rows
        summary_row = summary_rows[0]
        assert summary_row.get("subproblem_type") == "reproduce_issue"
        assert summary_row.get("strategy_label") == "ad_hoc_repro_script_loop"
        assert summary_row.get("avoid_actions") == ["create_new_repro_script_after_repro_confirmed"]


def test_critical_alignment_boost_for_failure_cards():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(storage_dir=tmpdir)
        # Generic card
        store.upsert_failure_card_v2(
            {
                "task_id": "task_generic",
                "error_signature": {"error_type": "import_error"},
                "candidate_fix_actions": ["generic_fix"],
                "verification_commands": ["pytest -q"],
                "confidence": 0.5,
            }
        )
        # Critical-aligned card should rank higher for import_error query.
        critical_id = store.upsert_failure_card_v2(
            {
                "task_id": "task_critical",
                "error_signature": {"error_type": "import_error"},
                "candidate_fix_actions": ["critical_fix_action"],
                "verification_commands": ["pytest -q"],
                "confidence": 0.45,
                "metadata": {
                    "critical_signal": {
                        "error_type": "import_error",
                        "critical_module": "action",
                        "critical_step": 4,
                    }
                },
            }
        )
        rows = store.query_failure_cards_v2(
            query_text="import error in tests",
            error_type="import_error",
            max_results=2,
        )
        assert len(rows) == 2
        assert rows[0]["card_id"] == critical_id
        assert float(rows[0].get("critical_alignment_score", 0.0)) >= 0.3


def test_graph_store_merges_same_experience_family():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(storage_dir=tmpdir)
        exp_a = {
            "experience_id": "exp_a",
            "pattern_type": "critical_planning_repetitive_planning",
            "normalized_pattern_type": "planning_loop",
            "normalized_trigger_family": "planning_transition_failure",
            "normalized_advice_family": "force_progression_transition",
            "family_id": "planning_loop__planning_transition_failure__force_progression_transition",
            "abstracted_intent": "Implement a state machine or progress tracker that enforces transitions after planning.",
            "variant_texts": [
                "Implement a state machine or progress tracker that enforces transitions after planning."
            ],
            "success_conditions": ["apply_strategy_shift_for_repetitive_planning"],
            "failure_avoidance": ["avoid_repeat_repetitive_planning_without_strategy_shift"],
            "confidence": 0.8,
            "support_count": 1,
        }
        exp_b = {
            "experience_id": "exp_b",
            "pattern_type": "critical_planning_inefficient_planning",
            "normalized_pattern_type": "planning_loop",
            "normalized_trigger_family": "planning_transition_failure",
            "normalized_advice_family": "force_progression_transition",
            "family_id": "planning_loop__planning_transition_failure__force_progression_transition",
            "abstracted_intent": "Add a progress tracker so planning transitions into execution instead of looping.",
            "variant_texts": [
                "Add a progress tracker so planning transitions into execution instead of looping."
            ],
            "success_conditions": ["apply_strategy_shift_for_repetitive_planning"],
            "failure_avoidance": ["avoid_repeat_repetitive_planning_without_strategy_shift"],
            "confidence": 0.76,
            "support_count": 1,
        }
        first_id = store.upsert_abstract_experience(exp_a)
        second_id = store.upsert_abstract_experience(exp_b)
        assert first_id == second_id
        merged = store.abstract_experiences[first_id]
        assert merged["support_count"] == 2
        assert merged["family_id"] == exp_a["family_id"]
        assert len(merged.get("variant_texts", [])) == 2


def test_graph_store_keeps_distinct_advice_families_separate():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(storage_dir=tmpdir)
        exp_a = {
            "experience_id": "exp_a",
            "pattern_type": "critical_planning_repetitive_planning",
            "normalized_pattern_type": "planning_loop",
            "normalized_trigger_family": "planning_transition_failure",
            "normalized_advice_family": "force_progression_transition",
            "family_id": "planning_loop__planning_transition_failure__force_progression_transition",
            "abstracted_intent": "Implement a state machine to force planning to transition into execution.",
            "variant_texts": [
                "Implement a state machine to force planning to transition into execution."
            ],
            "success_conditions": ["apply_strategy_shift_for_repetitive_planning"],
            "failure_avoidance": ["avoid_repeat_repetitive_planning_without_strategy_shift"],
            "confidence": 0.8,
            "support_count": 1,
        }
        exp_b = {
            "experience_id": "exp_b",
            "pattern_type": "critical_planning_constraint_ignorance",
            "normalized_pattern_type": "planning_loop",
            "normalized_trigger_family": "planning_transition_failure",
            "normalized_advice_family": "add_local_validation",
            "family_id": "planning_loop__planning_transition_failure__add_local_validation",
            "abstracted_intent": "Require local validation and targeted tests before submission.",
            "variant_texts": [
                "Require local validation and targeted tests before submission."
            ],
            "success_conditions": ["validate_fix_with_targeted_check_before_submission"],
            "failure_avoidance": ["avoid_patch_without_local_verification"],
            "confidence": 0.78,
            "support_count": 1,
        }
        first_id = store.upsert_abstract_experience(exp_a)
        second_id = store.upsert_abstract_experience(exp_b)
        assert first_id != second_id
        assert len(store.abstract_experiences) == 2


def test_memory_agent_rerank_keeps_one_recommendation_per_family():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(storage_dir=tmpdir)
        agent = MemoryAgent(store)
        recommendations = [
            {
                "type": "abstract_pattern",
                "experience_id": "exp_a",
                "family_id": "planning_loop__planning_transition_failure__force_progression_transition",
                "recommendation": "Add a state transition guard so planning moves into execution.",
                "confidence": 0.82,
                "score": 0.79,
                "support_count": 3,
            },
            {
                "type": "abstract_pattern",
                "experience_id": "exp_b",
                "family_id": "planning_loop__planning_transition_failure__force_progression_transition",
                "recommendation": "Track planning progress and enforce an execution step.",
                "confidence": 0.81,
                "score": 0.78,
                "support_count": 2,
            },
            {
                "type": "abstract_pattern",
                "experience_id": "exp_c",
                "family_id": "validation_gap__missing_validation__add_local_validation",
                "recommendation": "Run a focused validation command before changing more files.",
                "confidence": 0.74,
                "score": 0.73,
                "support_count": 2,
            },
        ]
        selected = agent._dedup_and_rerank_recommendations(recommendations, limit=5)
        assert len(selected) == 2
        families = {row["family_id"] for row in selected}
        assert "planning_loop__planning_transition_failure__force_progression_transition" in families
        assert "validation_gap__missing_validation__add_local_validation" in families
        assert all("selection_score" in row for row in selected)
        assert all("novelty_score" in row for row in selected)


def test_graph_store_marks_promoted_and_suppressed_experiences():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(storage_dir=tmpdir)
        exp_id = store.upsert_abstract_experience(
            {
                "pattern_type": "validation_gap_missing_target_validation",
                "normalized_pattern_type": "validation_gap",
                "normalized_trigger_family": "missing_validation",
                "normalized_advice_family": "add_local_validation",
                "family_id": "validation_gap__missing_validation__add_local_validation",
                "abstracted_intent": "Run target validation before submission.",
                "success_conditions": ["run_targeted_validation_before_submission"],
                "failure_avoidance": ["avoid_submission_without_target_validation"],
                "source_instance_id": "inst-1",
                "confidence": 0.7,
            }
        )
        assert store.mark_experience_promoted(exp_id, eval_ref="eval://resolved/1") is True
        row = store.abstract_experiences[exp_id]
        assert row["metadata"]["promotion_state"] == "promoted"
        assert "eval://resolved/1" in row["metadata"]["official_eval_refs"]

        assert store.mark_experience_suppressed(
            exp_id,
            eval_ref="eval://unresolved/1",
            reason="official_eval_unresolved",
        ) is True
        rows = store.query_abstract_experiences(query_text="target validation", max_results=5)
        assert rows == []
