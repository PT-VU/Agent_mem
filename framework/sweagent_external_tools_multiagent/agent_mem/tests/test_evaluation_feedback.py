from __future__ import annotations

import tempfile

from ..processing.evaluation_feedback import EvaluationFeedbackProcessor
from ..storage.graph_store import GraphStore


def test_evaluation_feedback_unresolved_suppresses_candidate_and_writes_negative_memory():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(storage_dir=tmpdir)
        processor = EvaluationFeedbackProcessor(store)
        exp_id = store.upsert_abstract_experience(
            {
                "pattern_type": "critical_action_test_failure",
                "normalized_pattern_type": "failure_recovery",
                "normalized_trigger_family": "test_failure",
                "normalized_advice_family": "narrow_context_then_retry",
                "family_id": "failure_recovery__test_failure__narrow_context_then_retry",
                "abstracted_intent": "Patch the failing path and rerun tests.",
                "success_conditions": ["run_targeted_validation_before_submission"],
                "failure_avoidance": ["avoid_patch_without_local_verification"],
                "source_instance_id": "inst-1",
                "confidence": 0.71,
                "metadata": {
                    "promotion_state": "candidate",
                    "evidence_stage": "submission",
                    "experience_polarity": "neutral",
                },
            }
        )

        report = processor.apply_feedback(
            instance_id="inst-1",
            outcome="unresolved",
            eval_ref="eval://inst-1",
            patch_text="+ except Exception:\n+     pass\n",
            validation_summary={},
            task_summary="submitted but unresolved",
        )

        assert exp_id in report["suppressed_ids"]
        assert report["written_ids"]
        written = [store.abstract_experiences[row_id] for row_id in report["written_ids"]]
        assert any(row["metadata"]["experience_polarity"] == "negative" for row in written)
        assert any(row["normalized_pattern_type"] in {"negative_strategy", "patch_risk", "validation_gap"} for row in written)


def test_evaluation_feedback_matches_candidate_by_run_attempt_trace_when_instance_is_wrong():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(storage_dir=tmpdir)
        processor = EvaluationFeedbackProcessor(store)
        exp_id = store.upsert_abstract_experience(
            {
                "pattern_type": "critical_action_test_failure",
                "normalized_pattern_type": "failure_recovery",
                "normalized_trigger_family": "test_failure",
                "normalized_advice_family": "narrow_context_then_retry",
                "family_id": "failure_recovery__test_failure__narrow_context_then_retry",
                "abstracted_intent": "Patch the failing path and rerun tests.",
                "success_conditions": ["run_targeted_validation_before_submission"],
                "failure_avoidance": ["avoid_patch_without_local_verification"],
                "source_instance_id": "main",
                "source_run_ids": ["run-1"],
                "source_attempt_ids": ["attempt-1"],
                "source_event_ids": ["trace-done"],
                "confidence": 0.71,
                "metadata": {
                    "promotion_state": "candidate",
                    "evidence_stage": "submission",
                    "experience_polarity": "neutral",
                },
            }
        )

        report = processor.apply_feedback(
            instance_id="inst-1",
            outcome="unresolved",
            eval_ref="eval://inst-1",
            run_id="run-1",
            attempt_id="attempt-1",
            trace_id="trace-done",
            patch_text="+ quiet = True\n",
            validation_summary={},
            task_summary="submitted but unresolved",
        )

        assert report["related_candidate_count"] == 1
        assert report["related_candidate_ids"] == [exp_id]
        assert exp_id in report["suppressed_ids"]


def test_evaluation_feedback_adds_large_patch_and_interface_risk_patterns():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(storage_dir=tmpdir)
        processor = EvaluationFeedbackProcessor(store)

        patch_text = "\n".join(
            ["+ def world_to_pixel_values(self, *world_arrays, quiet=False):"]
            + [f"+     helper_{i} = {i}" for i in range(45)]
        )

        report = processor.apply_feedback(
            instance_id="inst-2",
            outcome="unresolved",
            eval_ref="eval://inst-2",
            patch_text=patch_text,
            changed_files=["astropy/wcs/wcsapi/fitswcs.py"],
            validation_summary={},
            task_summary="overdesigned patch",
        )

        written = [store.abstract_experiences[row_id] for row_id in report["written_ids"]]
        pattern_types = {row["pattern_type"] for row in written}
        assert "patch_risk_overdesigned_fix" in pattern_types
        assert "large_patch" in report["patch_summary"]["risk_flags"]
        assert "interface_expansion" in report["patch_summary"]["risk_flags"]
        assert "behavior_change_default" in report["patch_summary"]["risk_flags"]


def test_evaluation_feedback_incomplete_timeout_writes_negative_memory():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(storage_dir=tmpdir)
        processor = EvaluationFeedbackProcessor(store)

        report = processor.apply_feedback(
            instance_id="inst-timeout",
            outcome="incomplete",
            eval_ref="eval://inst-timeout",
            patch_text="",
            changed_files=[],
            validation_summary={"commands": ["python reproduce_error.py"]},
            task_summary="timed out after repeated validation without patch",
        )

        assert report["written_ids"]
        written = [store.abstract_experiences[row_id] for row_id in report["written_ids"]]
        pattern_types = {row["pattern_type"] for row in written}
        assert "negative_strategy_timeout_after_overexploration" in pattern_types
        assert report["written_card_ids"]
        timeout_cards = [store.compiler_cards_v21[row_id] for row_id in report["written_card_ids"]]
        assert any(row["card_type"] == "TimeoutGovernanceCard" for row in timeout_cards)
        timeout_card = next(row for row in timeout_cards if row["card_type"] == "TimeoutGovernanceCard")
        assert timeout_card["governance_hardness"] == "guardrail"
        assert timeout_card["budget_hints"]["max_new_repro_scripts_after_repro_confirmed"] == 0
        assert "create_new_repro_script_after_repro_confirmed" in timeout_card["avoid_actions"]


def test_evaluation_feedback_resolved_writes_success_path_card():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(storage_dir=tmpdir)
        processor = EvaluationFeedbackProcessor(store)

        report = processor.apply_feedback(
            instance_id="inst-success",
            outcome="resolved",
            eval_ref="eval://inst-success",
            patch_text="+ right = cright[idx]\n+ cright[idx] = right\n",
            changed_files=["astropy/wcs/wcsapi/fitswcs.py"],
            validation_summary={"commands": ["pytest tests/test_target.py -q"]},
            task_summary="resolved with focused validation",
            run_id="run-1",
            attempt_id="attempt-1",
        )

        assert report["written_card_ids"]
        success_cards = [store.compiler_cards_v21[row_id] for row_id in report["written_card_ids"]]
        assert any(row["card_type"] == "SuccessPathCard" for row in success_cards)
        success_card = next(row for row in success_cards if row["card_type"] == "SuccessPathCard")
        assert success_card["promotion_state"] == "promoted"
        assert success_card["patch_family"]
        assert success_card["target_validation"] == ["pytest tests/test_target.py -q"]
        assert "target_validation_passed" in success_card["submit_preconditions"]
