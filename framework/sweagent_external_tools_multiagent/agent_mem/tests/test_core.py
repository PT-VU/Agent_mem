"""
Tests for Agent-mem core components.
"""

import json
import tempfile
from pathlib import Path
import pytest

from ..core.problem_file import (
    ProblemFile, ActionType, Outcome, EvidencePointer,
    FailureSignature, EnvSignature, MultiViewEmbeddings
)
from ..core.observation_kg import ObservationKG, TaskSubgraph, KGEdge, EdgeType
from ..core.belief_graph import BeliefGraph, AtomicBelief, BeliefType, BeliefStatus


class TestProblemFile:
    """Test ProblemFile data structure."""

    def test_create_problem_file(self):
        """Test creating a basic ProblemFile."""
        pf = ProblemFile(
            task_id="test_task_1",
            action_type=ActionType.CODE_EDIT,
            intent_text="Fix bug in calculate() function",
            outcome=Outcome.SUCCESS
        )

        assert pf.task_id == "test_task_1"
        assert pf.action_type == ActionType.CODE_EDIT
        assert pf.intent_text == "Fix bug in calculate() function"
        assert pf.outcome == Outcome.SUCCESS
        assert pf.action_id is not None

    def test_evidence_pointer(self):
        """Test EvidencePointer functionality."""
        ptr = EvidencePointer(
            type="stdout",
            location="/tmp/test_stdout.txt",
            offset=0,
            length=100,
            hash="abc123"
        )

        assert ptr.type == "stdout"
        assert ptr.location == "/tmp/test_stdout.txt"
        assert ptr.offset == 0
        assert ptr.length == 100
        assert ptr.hash == "abc123"

    def test_serialization(self):
        """Test ProblemFile serialization and deserialization."""
        # Create a ProblemFile with various fields
        pf = ProblemFile(
            task_id="test_task_2",
            action_type=ActionType.RUN_TEST,
            intent_text="Run unit tests",
            outcome=Outcome.FAIL,
            failure_signature=FailureSignature(
                error_type="AssertionError",
                error_tokens=["assert", "failed", "test_calculate"]
            ),
            env_signature=EnvSignature(
                toolchain_version="python3.10",
                working_dir="/home/user/project"
            )
        )

        # Add evidence pointer
        pf.add_evidence_pointer(
            EvidencePointer(type="stderr", location="/tmp/stderr.txt")
        )

        # Set epistemic tag
        from ..core.problem_file import EpistemicTag
        pf.set_epistemic_tag("failure_signature", EpistemicTag.INFERENCE)

        # Serialize to JSON
        json_str = pf.to_json()
        assert isinstance(json_str, str)

        # Deserialize
        pf2 = ProblemFile.from_json(json_str)

        # Verify fields are preserved
        assert pf2.task_id == pf.task_id
        assert pf2.action_type == pf.action_type
        assert pf2.intent_text == pf.intent_text
        assert pf2.outcome == pf.outcome
        assert pf2.failure_signature.error_type == pf.failure_signature.error_type
        assert len(pf2.evidence_index) == 1
        assert pf2.epistemic_tags.get("failure_signature") == "INFERENCE"

    def test_validation(self):
        """Test ProblemFile validation."""
        # Valid ProblemFile
        pf = ProblemFile(task_id="test", action_type=ActionType.TOOL_CALL)
        errors = pf.validate()
        assert len(errors) == 0

        # Invalid ProblemFile (missing task_id)
        pf = ProblemFile(task_id="", action_type=ActionType.TOOL_CALL)
        errors = pf.validate()
        assert "task_id is required" in errors

    def test_legacy_evidence_pointer_compatibility(self):
        """Legacy evidence_type/content_hash fields should remain readable."""
        payload = {
            "task_id": "legacy_task",
            "action_type": "tool_call",
            "outcome": "success",
            "stdout_ref": {
                "evidence_type": "stdout",
                "location": "/tmp/legacy_stdout.txt",
                "content_hash": "abc",
            },
            "epistemic_tags": {"stdout_ref": "EVIDENCE"},
        }
        pf = ProblemFile.from_dict(payload)
        assert pf.stdout_ref is not None
        assert pf.stdout_ref.type == "stdout"
        assert pf.stdout_ref.hash == "abc"


class TestObservationKG:
    """Test Observation Knowledge Graph."""

    def test_create_kg(self):
        """Test creating an observation KG."""
        kg = ObservationKG()
        assert len(kg.task_subgraphs) == 0

    def test_task_subgraph(self):
        """Test TaskSubgraph functionality."""
        subgraph = TaskSubgraph(task_id="test_task")

        # Create problem files
        pf1 = ProblemFile(task_id="test_task", action_type=ActionType.TOOL_CALL)
        pf2 = ProblemFile(task_id="test_task", action_type=ActionType.CODE_EDIT)

        # Add actions
        subgraph.add_action(pf1)
        subgraph.add_action(pf2)

        # Add edge
        edge = KGEdge(
            source_id=pf1.action_id,
            target_id=pf2.action_id,
            edge_type=EdgeType.SUCCESS_NEXT
        )
        subgraph.add_edge(edge)

        # Verify
        assert len(subgraph.action_nodes) == 2
        assert len(subgraph.edges) == 1
        assert subgraph.root_action_id == pf1.action_id

        # Test edge queries
        outgoing = subgraph.get_outgoing_edges(pf1.action_id)
        assert len(outgoing) == 1
        assert outgoing[0].target_id == pf2.action_id

        incoming = subgraph.get_incoming_edges(pf2.action_id)
        assert len(incoming) == 1
        assert incoming[0].source_id == pf1.action_id

    def test_success_chain(self):
        """Test success chain extraction."""
        subgraph = TaskSubgraph(task_id="test_task")

        # Create a chain of actions
        actions = []
        for i in range(3):
            pf = ProblemFile(
                task_id="test_task",
                action_type=ActionType.TOOL_CALL,
                intent_text=f"Step {i}"
            )
            subgraph.add_action(pf)
            actions.append(pf)

            if i > 0:
                edge = KGEdge(
                    source_id=actions[i-1].action_id,
                    target_id=actions[i].action_id,
                    edge_type=EdgeType.SUCCESS_NEXT
                )
                subgraph.add_edge(edge)

        # Get success chain
        chain = subgraph.get_success_chain()
        assert len(chain) == 3
        assert chain[0] == actions[0].action_id
        assert chain[1] == actions[1].action_id
        assert chain[2] == actions[2].action_id

    def test_kg_serialization(self):
        """Test KG serialization and deserialization."""
        kg = ObservationKG()

        # Create a subgraph
        subgraph = TaskSubgraph(task_id="test_task")
        pf = ProblemFile(task_id="test_task", action_type=ActionType.TOOL_CALL)
        subgraph.add_action(pf)
        kg.add_task_subgraph(subgraph)

        # Save and load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            kg.save_to_file(f.name)

            kg2 = ObservationKG.load_from_file(f.name)
            assert len(kg2.task_subgraphs) == 1
            assert "test_task" in kg2.task_subgraphs

        # Clean up
        Path(f.name).unlink()

    def test_embedding_similarity_search(self):
        """Test action similarity search based on embeddings."""
        kg = ObservationKG()
        subgraph = TaskSubgraph(task_id="sim_task")

        target = ProblemFile(task_id="sim_task", action_type=ActionType.TOOL_CALL)
        target.embeddings = MultiViewEmbeddings(
            emb_task_sem=[1.0, 0.0, 0.0],
            emb_intent=[1.0, 0.0, 0.0],
        )
        subgraph.add_action(target)

        far = ProblemFile(task_id="sim_task", action_type=ActionType.TOOL_CALL)
        far.embeddings = MultiViewEmbeddings(
            emb_task_sem=[0.0, 1.0, 0.0],
            emb_intent=[0.0, 1.0, 0.0],
        )
        subgraph.add_action(far)

        kg.add_task_subgraph(subgraph)

        query = ProblemFile(task_id="query_task", action_type=ActionType.TOOL_CALL)
        query.embeddings = MultiViewEmbeddings(
            emb_task_sem=[1.0, 0.0, 0.0],
            emb_intent=[1.0, 0.0, 0.0],
        )

        similar = kg.find_similar_actions(query, embedding_view="emb_task_sem", threshold=0.7)
        assert len(similar) == 1
        assert similar[0][0] == target.action_id


class TestBeliefGraph:
    """Test Belief Graph."""

    def test_atomic_belief(self):
        """Test AtomicBelief creation and updates."""
        belief = AtomicBelief(
            belief_type=BeliefType.WORKFLOW,
            confidence=0.8
        )

        assert belief.belief_type == BeliefType.WORKFLOW
        assert belief.confidence == 0.8
        assert belief.status == BeliefStatus.ACTIVE

        # Update stats
        belief.update_stats(success_with=True, success_without=False)
        assert belief.stats.support_n == 1
        assert belief.stats.success_with == 1
        assert belief.stats.success_without == 0

    def test_belief_promotion(self):
        """Test belief promotion logic."""
        belief = AtomicBelief(belief_type=BeliefType.WORKFLOW)

        # Add enough support for promotion
        for _ in range(15):  # More than min_support=10
            belief.update_stats(success_with=True, success_without=False)

        # Should be promotable to preference
        assert belief.should_promote_to_preference(
            min_support=10,
            min_uplift=0.1,
            min_confidence=0.7
        )

    def test_belief_graph(self):
        """Test BeliefGraph operations."""
        bg = BeliefGraph()

        # Add atomic belief
        belief = AtomicBelief(belief_type=BeliefType.PITFALL)
        belief_id = bg.add_atomic_belief(belief)

        assert belief_id in bg.atomic_beliefs

        # Get beliefs for context
        beliefs = bg.get_beliefs_for_context(action_type="code_edit")
        # Should return empty list since our belief doesn't match the context
        assert len(beliefs) == 0

        # Update belief stats
        bg.update_belief_stats(belief_id, success_with=True)
        updated_belief = bg.atomic_beliefs[belief_id]
        assert updated_belief.stats.support_n == 1

    def test_belief_graph_serialization(self):
        """Test BeliefGraph serialization."""
        bg = BeliefGraph()

        # Add a belief
        belief = AtomicBelief(
            belief_type=BeliefType.ENV_ADAPTATION,
            confidence=0.9
        )
        bg.add_atomic_belief(belief)

        # Save and load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            bg.save_to_file(f.name)

            bg2 = BeliefGraph.load_from_file(f.name)
            assert len(bg2.atomic_beliefs) == 1

            loaded_belief = list(bg2.atomic_beliefs.values())[0]
            assert loaded_belief.belief_type == BeliefType.ENV_ADAPTATION
            assert loaded_belief.confidence == 0.9

        # Clean up
        Path(f.name).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
