from __future__ import annotations

import tempfile

import pytest

from ..processing.object_governance_policy import ObjectGovernancePolicy
from ..storage.episode_ledger_store import EpisodeLedgerStore


def test_governance_policy_dedups_cards_and_applies_growth_cap():
    policy = ObjectGovernancePolicy(max_cards_per_query=1)
    report = policy.apply(
        compiler_cards=[
            {
                "card_id": "card-1",
                "family_id": "fam-1",
                "card_type": "PlanHintCard",
                "hint": "do x",
                "confidence": 0.7,
            },
            {
                "card_id": "card-2",
                "family_id": "fam-1",
                "card_type": "PlanHintCard",
                "hint": "do x but better",
                "confidence": 0.8,
            },
            {
                "card_id": "card-3",
                "family_id": "fam-2",
                "card_type": "RetryHintCard",
                "hint": "do y",
                "confidence": 0.6,
            },
        ],
        context={"submission_success": True},
    )
    cards = report["compiler_cards"]
    assert len(cards) == 1
    assert cards[0]["card_id"] == "card-2"
    assert cards[0]["promotion_state"] == "candidate"
    assert report["report"]["compiler_cards"]["suppressed_ids"]


def test_governance_policy_attaches_eval_context_without_overwriting_local_result():
    policy = ObjectGovernancePolicy(max_cards_per_query=2)
    attachment = policy.attach_eval_context(
        subtasks=[
            {
                "subtask_instance_id": "sub-1",
                "status": "locally_failed",
                "local_result_status": "failed",
            }
        ],
        subtask_edges=[{"edge_id": "edge-1", "status": "candidate"}],
        outcome="unresolved",
        eval_ref="eval://1",
    )
    assert attachment["subtasks"][0]["status"] == "eval_context_attached"
    assert attachment["subtasks"][0]["local_result_status"] == "failed"
    assert attachment["subtask_edges"][0]["eval_support"]["official_eval_status"] == "unresolved"


def test_evaluation_feedback_updates_compiler_cards_and_sidecar():
    pytest.importorskip("networkx")
    from ..processing.evaluation_feedback import EvaluationFeedbackProcessor
    from ..storage.graph_store import GraphStore

    policy = ObjectGovernancePolicy(max_cards_per_query=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        graph_store = GraphStore(tmpdir)
        sidecar = EpisodeLedgerStore(f"{tmpdir}/sidecar")
        graph_store.upsert_compiler_card_v21(
            {
                "card_id": "card-1",
                "instance_id": "inst-1",
                "run_id": "run-1",
                "attempt_id": "attempt-1",
                "card_type": "PlanHintCard",
                "family_id": "fam-1",
                "recommendation": "do x",
                "confidence": 0.7,
                "promotion_state": "candidate",
                "source_object_ids": ["sum-1"],
                "evidence_level": "attempt",
            }
        )
        sidecar.append(
            {
                "record_id": "sub-1",
                "subtask_instance_id": "sub-1",
                "instance_id": "inst-1",
                "run_id": "run-1",
                "attempt_id": "attempt-1",
                "status": "locally_failed",
                "local_result_status": "failed",
            },
            stream="subtask_instances",
        )
        processor = EvaluationFeedbackProcessor(
            graph_store,
            governance_policy=policy,
            episode_ledger_store=sidecar,
        )
        report = processor.apply_feedback(
            instance_id="inst-1",
            outcome="resolved",
            eval_ref="eval://1",
            run_id="run-1",
            attempt_id="attempt-1",
        )
        assert report["promoted_card_ids"] == ["card-1"]
        latest = sidecar.load_latest_records(
            stream="subtask_instances",
            key_field="subtask_instance_id",
            filters={"instance_id": "inst-1", "attempt_id": "attempt-1"},
        )
        assert latest[0]["status"] == "eval_context_attached"
