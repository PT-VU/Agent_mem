from __future__ import annotations

from ..processing.card_compiler import CardCompiler


def test_card_compiler_emits_whitelisted_card_types():
    compiler = CardCompiler()
    cards = compiler.compile(
        attempt_summary={
            "summary_id": "sum-1",
            "instance_id": "inst-1",
            "run_id": "run-1",
            "attempt_id": "attempt-1",
            "trace_id": "trace-1",
            "next_best_actions": ["validate_current_changed_files_before_broadening_scope"],
            "failed_strategies": [
                {
                    "strategy_label": "ad_hoc_repro_script_loop",
                    "reason": "too many repro scripts",
                    "avoid_actions": ["create_new_repro_script_after_repro_confirmed"],
                }
            ],
        },
        failure_card={
            "card_id": "fc-1",
            "candidate_fix_actions": ["edit the localized path first"],
            "verification_commands": ["pytest tests/test_target.py -q"],
            "confidence": 0.81,
        },
        repair_patterns=[
            {
                "pattern_id": "rp-1",
                "fix_action_template": "reuse the focused repro path",
                "expected_verification": ["pytest tests/test_target.py -q"],
                "confidence": 0.7,
            }
        ],
        subtasks=[
            {
                "subtask_instance_id": "sub-1",
                "subtask_type": "target_validation",
                "local_result_status": "failed",
                "failure_type": "validation did not confirm the patch candidate",
                "recommended_next_steps": ["rerun focused validation before expanding scope"],
                "projection_confidence": 0.66,
            }
        ],
        max_cards=4,
    )
    assert cards
    types = {row["card_type"] for row in cards}
    assert "PlanHintCard" in types
    assert "RetryHintCard" in types
    assert "SubtaskRiskCard" in types or "ClosureGuardCard" in types
    for row in cards:
        assert row["type"] == "compiler_card"
        assert row["family_id"]
        assert row["source_object_ids"]
        assert row["promotion_state"] == "candidate"
