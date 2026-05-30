from __future__ import annotations

import json
import tempfile
from collections import deque
from types import SimpleNamespace

import pytest

pytest.importorskip("sweagent")

from bridge_hook import ExternalToolBridgeHook


def _hint(hint: str, *, family_id: str, score: float) -> dict:
    return {
        "hint": hint,
        "family_id": family_id,
        "selection_score": score,
        "item_confidence": 0.8,
        "batch_confidence": 0.8,
        "trace_id": f"trace:{family_id}:{score}",
    }


def test_bridge_hook_injects_one_hint_per_family_and_tracks_family_state(monkeypatch):
    monkeypatch.setenv("SWE_AGENT_MEM_INJECT_ENABLED", "1")
    monkeypatch.setenv("SWE_AGENT_MEM_MIN_CONFIDENCE", "0")
    monkeypatch.setenv("SWE_AGENT_MEM_FAMILY_COOLDOWN_STEPS", "4")
    monkeypatch.setenv("SWE_AGENT_MEM_MAX_HINTS_PER_FAMILY_PER_ATTEMPT", "1")

    hook = ExternalToolBridgeHook()
    hook._pending_memory_hints = deque(
        [
            _hint(
                "Add a state transition guard so planning moves into execution.",
                family_id="planning_loop__planning_transition_failure__force_progression_transition",
                score=0.91,
            ),
            _hint(
                "Track planning progress and enforce an execution step.",
                family_id="planning_loop__planning_transition_failure__force_progression_transition",
                score=0.82,
            ),
            _hint(
                "Run a focused validation command before changing more files.",
                family_id="validation_gap__missing_validation__add_local_validation",
                score=0.77,
            ),
        ]
    )
    hook._step_index = 5

    messages: list[dict[str, str]] = []
    hook.on_model_query(messages=messages, agent="main")

    assert len(messages) == 1
    content = messages[0]["content"]
    assert "Add a state transition guard" in content
    assert "Run a focused validation command" in content
    assert "Track planning progress" not in content
    assert hook._hint_family_injected_count[
        "planning_loop__planning_transition_failure__force_progression_transition"
    ] == 1
    assert hook._hint_family_last_injected_step[
        "planning_loop__planning_transition_failure__force_progression_transition"
    ] == 5


def test_bridge_hook_blocks_repeat_family_by_cooldown_and_attempt_limit(monkeypatch):
    monkeypatch.setenv("SWE_AGENT_MEM_INJECT_ENABLED", "1")
    monkeypatch.setenv("SWE_AGENT_MEM_MIN_CONFIDENCE", "0")
    monkeypatch.setenv("SWE_AGENT_MEM_FAMILY_COOLDOWN_STEPS", "4")
    monkeypatch.setenv("SWE_AGENT_MEM_MAX_HINTS_PER_FAMILY_PER_ATTEMPT", "1")

    hook = ExternalToolBridgeHook()
    family_id = "planning_loop__planning_transition_failure__force_progression_transition"
    hook._hint_family_last_injected_step[family_id] = 5
    hook._hint_family_injected_count[family_id] = 1

    hook._pending_memory_hints = deque(
        [
            _hint(
                "Use a state transition guard before planning again.",
                family_id=family_id,
                score=0.88,
            )
        ]
    )

    hook._step_index = 7
    assert hook._select_hints_for_injection() == []
    assert len(hook._pending_memory_hints) == 1

    hook._step_index = 12
    assert hook._select_hints_for_injection() == []


def test_bridge_hook_prefers_repair_and_validation_before_belief(monkeypatch):
    monkeypatch.setenv("SWE_AGENT_MEM_INJECT_ENABLED", "1")
    monkeypatch.setenv("SWE_AGENT_MEM_MIN_CONFIDENCE", "0")
    monkeypatch.setenv("SWE_AGENT_MEM_MAX_HINTS", "3")

    hook = ExternalToolBridgeHook()
    hook._pending_memory_hints = deque(
        [
            {
                **_hint("Use the targeted repair pattern.", family_id="repair_pattern:123", score=0.81),
                "type": "repair_pattern_v2",
            },
            {
                **_hint("Run focused validation before resubmitting.", family_id="validation_gap:abc", score=0.8),
                "type": "abstract_pattern",
                "normalized_pattern_type": "validation_gap",
            },
            {
                **_hint("Remember the old workflow order.", family_id="belief_tip:workflow:1", score=0.95),
                "type": "belief_tip",
            },
        ]
    )
    messages: list[dict[str, str]] = []
    hook.on_model_query(messages=messages, agent="main")
    content = messages[0]["content"]
    assert "Use the targeted repair pattern." in content
    assert "Run focused validation before resubmitting." in content
    assert "Remember the old workflow order." in content
    assert content.index("Use the targeted repair pattern.") < content.index("Remember the old workflow order.")


def test_bridge_hook_uses_problem_statement_id_when_available(monkeypatch):
    monkeypatch.delenv("SWE_AGENT_EXT_INSTANCE_ID", raising=False)
    hook = ExternalToolBridgeHook()

    class ProblemStatement:
        id = "astropy__astropy-11693"

    class Agent:
        name = "main"
        _problem_statement = ProblemStatement()

    hook.on_init(agent=Agent())
    assert hook._instance_id == "astropy__astropy-11693"


def test_bridge_hook_drops_workflow_hints_after_early_steps(monkeypatch):
    monkeypatch.setenv("SWE_AGENT_MEM_INJECT_ENABLED", "1")
    monkeypatch.setenv("SWE_AGENT_MEM_MIN_CONFIDENCE", "0")

    hook = ExternalToolBridgeHook()
    hook._step_index = 8
    hook._pending_memory_hints = deque(
        [
            {
                **_hint("Consider: Generate plan for task...", family_id="workflow_step:consider: generate plan for task...", score=0.9),
                "type": "workflow_step",
            },
            {
                **_hint("Validate the minimal fix before exploring more files.", family_id="validation_gap:add_local_validation", score=0.85),
                "type": "abstract_pattern",
                "normalized_pattern_type": "validation_gap",
            },
        ]
    )

    selected = hook._select_hints_for_injection()
    assert len(selected) == 1
    assert selected[0]["family_id"] == "validation_gap:add_local_validation"


def test_bridge_hook_enqueues_closure_hint_after_ad_hoc_script_sprawl(monkeypatch):
    monkeypatch.setenv("SWE_AGENT_MEM_INJECT_ENABLED", "1")
    hook = ExternalToolBridgeHook()
    hook._scan_proactive_triggers("python reproduce_error.py")
    hook._scan_proactive_triggers("python test_fix_comprehensive.py")

    family_ids = [str(item.get("family_id", "")) for item in hook._pending_memory_hints]
    assert "closure_signal:over_exploration_after_key_signal" in family_ids


def test_bridge_hook_runtime_guard_payload_reflects_closure_state(monkeypatch):
    hook = ExternalToolBridgeHook()
    hook._record_trigger("over_exploration", detail="2")
    payload = hook._runtime_guard_payload()
    assert payload["closure_active"] is True
    assert "workflow_step" in payload["blocked_families"]
    assert "planning_loop" in payload["blocked_families"]


def test_bridge_hook_soft_blocks_new_ad_hoc_script_creation_after_closure(monkeypatch):
    hook = ExternalToolBridgeHook()
    hook._closure_active = True
    hook._ad_hoc_script_names = {"reproduce_error.py", "test_fix_comprehensive.py"}
    step = SimpleNamespace(
        thought="create another debug script",
        action="cat > debug_more.py <<'PY'\nprint('x')\nPY",
    )
    captured = []

    def fake_run_cmd(*, cmd, tool_name, payload):
        captured.append(payload)

    monkeypatch.setattr(hook, "_run_cmd", fake_run_cmd)
    hook.on_actions_generated(step=step)
    assert "Runtime guard blocked new ad-hoc script creation after closure" in step.action
    assert captured
    assert captured[0]["runtime_guard"]["closure_active"] is True


def test_bridge_hook_applies_summary_avoid_actions_to_runtime_guard(monkeypatch):
    monkeypatch.setenv("SWE_AGENT_MEM_INJECT_ENABLED", "1")
    hook = ExternalToolBridgeHook()
    payload = {
        "event_handled": "plan_generated",
        "trace_id": "trace:summary",
        "confidence": 0.9,
        "planning_tips": [
            {
                "type": "attempt_summary_v1",
                "summary_id": "sum_123",
                "family_id": "attempt_summary:inst:attempt-01",
                "recommendation": "Avoid creating new repro scripts after reproduction is confirmed.",
                "subproblem_type": "reproduce_issue",
                "strategy_label": "ad_hoc_repro_script_loop",
                "avoid_actions": ["create_new_repro_script_after_repro_confirmed"],
            }
        ],
    }
    hook._handle_parsed_tool_payload(tool_name="tool_a", parsed=payload)
    assert "create_new_repro_script_after_repro_confirmed" in hook._blocked_action_patterns
    assert hook._active_subproblem_type == "reproduce_issue"
    assert hook._active_strategy_label == "ad_hoc_repro_script_loop"
    runtime_guard = hook._runtime_guard_payload()
    assert runtime_guard["active_subproblem_type"] == "reproduce_issue"
    assert runtime_guard["active_strategy_label"] == "ad_hoc_repro_script_loop"
    assert "create_new_repro_script_after_repro_confirmed" in runtime_guard["blocked_action_patterns"]


def test_bridge_hook_applies_timeout_governance_card_to_runtime_guard(monkeypatch):
    monkeypatch.setenv("SWE_AGENT_MEM_INJECT_ENABLED", "1")
    hook = ExternalToolBridgeHook()
    payload = {
        "event_handled": "plan_generated",
        "trace_id": "trace:timeout-card",
        "confidence": 0.92,
        "planning_tips": [
            {
                "type": "compiler_card",
                "card_id": "card_timeout",
                "card_type": "TimeoutGovernanceCard",
                "family_id": "timeout_governance:inst",
                "recommendation": "Do not create new repro scripts after reproduction is confirmed.",
                "subproblem_type": "reproduce_issue",
                "strategy_label": "ad_hoc_repro_script_loop",
                "governance_hardness": "guardrail",
                "avoid_actions": [
                    "create_new_repro_script_after_repro_confirmed",
                    "run_broad_regression_before_patch_candidate_exists",
                ],
            }
        ],
    }
    hook._handle_parsed_tool_payload(tool_name="tool_a", parsed=payload)
    runtime_guard = hook._runtime_guard_payload()
    assert runtime_guard["active_subproblem_type"] == "reproduce_issue"
    assert runtime_guard["active_strategy_label"] == "ad_hoc_repro_script_loop"
    assert "create_new_repro_script_after_repro_confirmed" in runtime_guard["blocked_action_patterns"]
    assert "run_broad_regression_before_patch_candidate_exists" in runtime_guard["blocked_action_patterns"]


def test_bridge_hook_writes_hotpath_success_fact_idempotently(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("AGENT_MEM_V21_ENABLE_SUCCESS_FACT_HOTPATH", "1")
        monkeypatch.setenv("AGENT_MEM_V21_ENABLE_SIDECAR", "1")
        monkeypatch.setenv("AGENT_MEM_V21_SIDECAR_DIR", tmpdir)
        monkeypatch.setenv("SWE_AGENT_EXT_RUN_ID", "run-1")
        monkeypatch.setenv("SWE_AGENT_EXT_ATTEMPT_ID", "attempt-1")
        monkeypatch.setenv("SWE_AGENT_EXT_INSTANCE_ID", "inst-1")
        hook = ExternalToolBridgeHook()
        hook.on_init(agent=SimpleNamespace(name="main"))
        step = SimpleNamespace(thought="plan the next action", action="pytest tests/test_target.py -q")
        hook.on_actions_generated(step=step)
        executed = SimpleNamespace(action=step.action, observation="1 passed")
        hook.on_action_executed(step=executed)
        hook.on_action_executed(step=executed)

        rows = []
        with open(f"{tmpdir}/episode_ledger.jsonl", "r", encoding="utf-8") as f:
            for raw in f:
                rows.append(json.loads(raw))
        success_rows = [row for row in rows if row.get("event") == "success_fact"]
        assert len(success_rows) == 1
        assert success_rows[0]["instance_id"] == "inst-1"
        assert success_rows[0]["success_like"] is True


def test_bridge_hook_soft_blocks_summary_avoid_action_pattern(monkeypatch):
    hook = ExternalToolBridgeHook()
    hook._blocked_action_patterns = {"create_new_repro_script_after_repro_confirmed"}
    hook._active_subproblem_type = "reproduce_issue"
    hook._active_strategy_label = "ad_hoc_repro_script_loop"
    captured = []

    def fake_run_cmd(*, cmd, tool_name, payload):
        captured.append(payload)

    monkeypatch.setattr(hook, "_run_cmd", fake_run_cmd)
    step = SimpleNamespace(
        thought="create another reproduction script",
        action="cat > reproduce_more.py <<'PY'\nprint('x')\nPY",
    )
    hook.on_actions_generated(step=step)
    assert "Runtime guard blocked an action pattern discouraged by previous failed attempts" in step.action
    assert captured
    assert captured[0]["runtime_guard"]["active_subproblem_type"] == "reproduce_issue"
    assert "create_new_repro_script_after_repro_confirmed" in captured[0]["runtime_guard"]["blocked_action_patterns"]


def test_bridge_hook_blocks_unrelated_module_expansion_from_summary_pattern(monkeypatch):
    hook = ExternalToolBridgeHook()
    hook._blocked_action_patterns = {"expand_search_to_unrelated_module_after_localization"}
    hook._active_subproblem_type = "localize_fix"
    hook._active_strategy_label = "cross_module_expansion_after_key_signal"
    captured = []

    def fake_run_cmd(*, cmd, tool_name, payload):
        captured.append(payload)

    monkeypatch.setattr(hook, "_run_cmd", fake_run_cmd)
    step = SimpleNamespace(
        thought="inspect unrelated module",
        action="rg -n 'WCSAxes' /testbed/astropy/visualization/wcsaxes",
    )
    hook.on_actions_generated(step=step)
    assert "Runtime guard blocked an action pattern discouraged by previous failed attempts" in step.action
    assert captured
    assert (
        "expand_search_to_unrelated_module_after_localization"
        in captured[0]["runtime_guard"]["blocked_action_patterns"]
    )
