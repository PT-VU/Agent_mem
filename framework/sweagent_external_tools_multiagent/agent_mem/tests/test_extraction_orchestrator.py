from __future__ import annotations

from ..core.problem_file import ActionType, FailureSignature, Outcome, ProblemFile
from ..processing.extraction_orchestrator import ExtractionOrchestrator
from ..processing.llm_extractor import LLMExperienceExtractor
from ..processing.taxonomy import ErrorTaxonomy


def _action(
    *,
    task_id: str,
    step: int,
    action_type: ActionType,
    outcome: Outcome,
    intent: str,
    family: str,
    error_type: str = "",
) -> ProblemFile:
    sig = None
    if error_type:
        sig = FailureSignature(error_type=error_type, error_tokens=[error_type])
    return ProblemFile(
        task_id=task_id,
        step_index=step,
        action_type=action_type,
        action_family=family,
        intent_text=intent,
        action_text=intent,
        outcome=outcome,
        source_event="action_error" if outcome == Outcome.FAIL else "plan_generated",
        failure_signature=sig,
        metadata={"test_only": True},
    )


def test_orchestrator_maps_to_existing_templates():
    extractor = LLMExperienceExtractor(
        taxonomy=ErrorTaxonomy(),
        mode="heuristic",
    )
    orchestrator = ExtractionOrchestrator(
        enabled=True,
        min_item_confidence=0.2,
        extractor=extractor,
    )
    actions = [
        _action(
            task_id="t1",
            step=0,
            action_type=ActionType.TOOL_CALL,
            outcome=Outcome.SUCCESS,
            intent="inspect failing test and files",
            family="planning",
        ),
        _action(
            task_id="t1",
            step=1,
            action_type=ActionType.RUN_TEST,
            outcome=Outcome.FAIL,
            intent="run pytest -q",
            family="test_failure_fix",
            error_type="import_error",
        ),
    ]
    report = orchestrator.process_attempt(
        task_id="t1",
        actions=actions,
        success=False,
        task_summary="run_done unresolved",
        exit_status="failed",
        source_instance_id="astropy__astropy-7746",
        source_run_id="run_x",
        source_attempt_id="attempt_01",
        trace_id="trace_x",
    )
    assert report["triggered"] is True
    assert report["reason"] == "run_done_unresolved"
    assert len(report["assessments"]) >= 1
    critical = report["critical_signal"]
    assert critical["error_type"] in {"import_error", "test_failure", "execution_failure"}
    assert critical["critical_module"] in {"planning", "action", "system", "reflection", "memory"}

    abstracts = report["abstract_experiences"]
    assert len(abstracts) >= 1
    assert abstracts[0]["pattern_type"].startswith("critical_")
    assert abstracts[0]["normalized_pattern_type"]
    assert abstracts[0]["normalized_trigger_family"]
    assert abstracts[0]["normalized_advice_family"]
    assert abstracts[0]["family_id"]
    assert len(abstracts[0]["success_conditions"]) >= 1

    patch = report["failure_card_patch"]
    assert len(patch["candidate_fix_actions"]) >= 1
    assert len(patch["verification_commands"]) >= 1
    assert len(patch["evidence_refs"]) >= 1

    patterns = report["repair_patterns"]
    assert len(patterns) >= 1
    assert "error_type" in patterns[0]["trigger_signature"]
    assert len(patterns[0]["expected_verification"]) >= 1
    assert report["trial_overview"]["final_outcome"] == "failed"
    assert isinstance(report["subblock_analysis"], list)
    assert report["attempt_summary_v1"]["instance_id"] == "astropy__astropy-7746"
    assert isinstance(report["attempt_summary_v1"]["failed_strategies"], list)


def test_orchestrator_attempt_summary_and_conditional_fields_present():
    orchestrator = ExtractionOrchestrator(
        enabled=True,
        min_item_confidence=0.2,
        extractor=LLMExperienceExtractor(
            taxonomy=ErrorTaxonomy(),
            mode="heuristic",
        ),
    )
    actions = [
        _action(
            task_id="t_conditional",
            step=0,
            action_type=ActionType.TOOL_CALL,
            outcome=Outcome.SUCCESS,
            intent="inspect fitswcs.py and wcs.py",
            family="planning",
        ),
        _action(
            task_id="t_conditional",
            step=1,
            action_type=ActionType.RUN_TEST,
            outcome=Outcome.FAIL,
            intent="write test_quiet.py and run pytest -q",
            family="test_failure_fix",
            error_type="test_failure",
        ),
    ]
    report = orchestrator.process_attempt(
        task_id="t_conditional",
        actions=actions,
        success=False,
        task_summary="timed out before patch",
        exit_status="incomplete",
        source_instance_id="inst-cond",
        source_run_id="run-cond",
        source_attempt_id="attempt-cond",
        trace_id="trace-cond",
        extra_context={
            "task_problem_excerpt": "stabilize failing WCS plotting path",
            "ad_hoc_script_count": 2,
            "ad_hoc_script_names": ["test_quiet.py", "test_scalar.py"],
            "official_eval_status": "unknown",
        },
    )
    summary = report["attempt_summary_v1"]
    assert summary["problem_goal"] == "stabilize failing WCS plotting path"
    assert summary["initial_plan_outline"]
    assert summary["actual_execution_outline"]
    assert summary["next_best_actions"]
    assert summary["confirmed_signals"] or summary["failed_strategies"]
    assert report["subblock_analysis"]
    abstracts = report["abstract_experiences"]
    assert abstracts
    row = abstracts[0]
    assert row["subproblem_type"]
    assert "strategy_label" in row
    assert isinstance(row["prefer_actions"], list)
    assert isinstance(row["avoid_actions"], list)


def test_orchestrator_disabled_still_returns_heuristic_attempt_summary():
    orchestrator = ExtractionOrchestrator(
        enabled=False,
        min_item_confidence=0.2,
        extractor=LLMExperienceExtractor(
            taxonomy=ErrorTaxonomy(),
            mode="heuristic",
        ),
    )
    actions = [
        _action(
            task_id="t_disabled",
            step=0,
            action_type=ActionType.CODE_EDIT,
            outcome=Outcome.SUCCESS,
            intent="patch the localized target file",
            family="planning",
        ),
        _action(
            task_id="t_disabled",
            step=1,
            action_type=ActionType.RUN_TEST,
            outcome=Outcome.SUCCESS,
            intent="run pytest tests/test_target.py -q",
            family="test_failure_fix",
        ),
    ]
    report = orchestrator.process_attempt(
        task_id="t_disabled",
        actions=actions,
        success=True,
        task_summary="submitted for eval",
        exit_status="submitted",
        source_instance_id="inst-disabled",
        source_run_id="run-disabled",
        source_attempt_id="attempt-disabled",
        trace_id="trace-disabled",
        extra_context={"task_problem_excerpt": "fix the localized failure and validate the patch"},
    )
    assert report["enabled"] is False
    assert report["triggered"] is False
    assert report["reason"] == "disabled_by_config"
    assert report["subblock_analysis"]
    assert report["attempt_summary_v1"]["instance_id"] == "inst-disabled"
    assert report["attempt_summary_v1"]["subblock_analysis"]


def test_orchestrator_rebalances_planning_duplicates_and_keeps_validation():
    orchestrator = ExtractionOrchestrator(
        enabled=True,
        min_item_confidence=0.2,
        extractor=LLMExperienceExtractor(
            taxonomy=ErrorTaxonomy(),
            mode="heuristic",
        ),
    )
    payloads = [
        {
            "pattern_type": "critical_planning_repetitive_planning",
            "normalized_pattern_type": "planning_loop",
            "normalized_trigger_family": "planning_transition_failure",
            "normalized_advice_family": "force_progression_transition",
            "family_id": "planning_loop__planning_transition_failure__force_progression_transition",
            "abstracted_intent": "Implement a state machine or progress tracker to transition from planning to execution.",
            "evidence_refs": ["a1", "a2", "a3"],
            "confidence": 0.82,
        },
        {
            "pattern_type": "critical_planning_inefficient_planning",
            "normalized_pattern_type": "planning_loop",
            "normalized_trigger_family": "planning_transition_failure",
            "normalized_advice_family": "force_progression_transition",
            "family_id": "planning_loop__planning_transition_failure__force_progression_transition",
            "abstracted_intent": "Add a progress tracker so planning stops looping and moves forward.",
            "evidence_refs": ["b1", "b2", "b3"],
            "confidence": 0.74,
        },
        {
            "pattern_type": "critical_planning_constraint_ignorance",
            "normalized_pattern_type": "validation_gap",
            "normalized_trigger_family": "missing_validation",
            "normalized_advice_family": "add_local_validation",
            "family_id": "validation_gap__missing_validation__add_local_validation",
            "abstracted_intent": "Require local validation and targeted tests before submission.",
            "evidence_refs": ["c1", "c2", "c3"],
            "confidence": 0.79,
        },
    ]
    selected = orchestrator._rebalance_abstract_experiences(payloads)
    assert len(selected) == 1
    families = {row["family_id"] for row in selected}
    assert "validation_gap__missing_validation__add_local_validation" in families


def test_orchestrator_maps_strategy_observation_to_negative_strategy():
    orchestrator = ExtractionOrchestrator(
        enabled=True,
        min_item_confidence=0.2,
        extractor=LLMExperienceExtractor(
            taxonomy=ErrorTaxonomy(),
            mode="heuristic",
        ),
    )
    actions = [
        _action(
            task_id="t2",
            step=0,
            action_type=ActionType.RUN_TEST,
            outcome=Outcome.FAIL,
            intent="run focused test",
            family="test_failure_fix",
            error_type="test_failure",
        ),
    ]
    rows = orchestrator._map_strategy_observations(
        strategy_observations=[
            {
                "strategy_type": "negative_strategy",
                "why_failed_or_risky": "The patch silenced the failure but did not validate the target path.",
                "recommended_avoidance": "avoid suppressive fixes without target validation",
                "confidence": 0.77,
            }
        ],
        actions=actions,
        task_id="t2",
        task_summary="submitted but unresolved",
        source_instance_id="astropy__astropy-1",
        source_run_id="run-1",
        source_attempt_id="attempt-1",
        trace_id="trace-1",
        extra_context={"submission_success": True, "official_eval_status": "unknown"},
    )
    assert len(rows) == 1
    assert rows[0]["normalized_pattern_type"] == "negative_strategy"
    assert rows[0]["metadata"]["promotion_state"] == "candidate"


def test_orchestrator_generates_closure_and_script_sprawl_heuristics():
    orchestrator = ExtractionOrchestrator(
        enabled=True,
        min_item_confidence=0.2,
        extractor=LLMExperienceExtractor(
            taxonomy=ErrorTaxonomy(),
            mode="heuristic",
        ),
    )
    actions = []
    for step in range(45):
        actions.append(
            _action(
                task_id="t3",
                step=step,
                action_type=ActionType.RUN_TEST if step % 10 == 0 else ActionType.TOOL_CALL,
                outcome=Outcome.FAIL if step % 10 == 0 else Outcome.SUCCESS,
                intent=(
                    "run pytest -q"
                    if step % 10 == 0
                    else ("write test_quiet.py" if step == 12 else "write test_scalar_vs_array.py" if step == 23 else "inspect code")
                ),
                family="test_failure_fix" if step % 10 == 0 else "planning",
                error_type="test_failure" if step % 10 == 0 else "",
            )
        )
    observations = orchestrator._heuristic_strategy_observations(
        actions=actions,
        context={
            "step_count": 45,
            "ad_hoc_script_count": 2,
            "ad_hoc_script_names": ["test_quiet.py", "test_scalar_vs_array.py"],
        },
    )
    kinds = {row["strategy_type"] for row in observations}
    assert "closure_signal_over_exploration" in kinds
    assert "negative_strategy_ad_hoc_script_sprawl" in kinds


def test_orchestrator_prefers_closure_signal_over_planning_rule():
    orchestrator = ExtractionOrchestrator(
        enabled=True,
        min_item_confidence=0.2,
        extractor=LLMExperienceExtractor(
            taxonomy=ErrorTaxonomy(),
            mode="heuristic",
        ),
    )
    payloads = [
        {
            "pattern_type": "critical_planning_repetitive_planning",
            "normalized_pattern_type": "planning_loop",
            "normalized_trigger_family": "planning_transition_failure",
            "normalized_advice_family": "force_progression_transition",
            "family_id": "planning_loop__planning_transition_failure__force_progression_transition",
            "abstracted_intent": "Keep exploring and planning.",
            "evidence_refs": ["a1", "a2", "a3"],
            "confidence": 0.92,
        },
        {
            "pattern_type": "closure_signal_over_exploration",
            "normalized_pattern_type": "closure_signal",
            "normalized_trigger_family": "over_exploration_after_key_signal",
            "normalized_advice_family": "stop_expand_and_validate_minimal_fix",
            "family_id": "closure_signal__over_exploration_after_key_signal__stop_expand_and_validate_minimal_fix",
            "abstracted_intent": "Stop expanding investigation after the key path is already known.",
            "evidence_refs": ["b1", "b2", "b3"],
            "confidence": 0.75,
        },
    ]
    selected = orchestrator._rebalance_abstract_experiences(payloads)
    families = {row["family_id"] for row in selected}
    assert "closure_signal__over_exploration_after_key_signal__stop_expand_and_validate_minimal_fix" in families
    assert "planning_loop__planning_transition_failure__force_progression_transition" not in families
