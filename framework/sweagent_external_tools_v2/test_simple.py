#!/usr/bin/env python
"""
Simple test to verify core components.
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import core components
from agent_mem.core.problem_file import (
    ProblemFile, ActionType, Outcome, EvidencePointer,
    FailureSignature, EnvSignature, MultiViewEmbeddings
)
from agent_mem.core.observation_kg import ObservationKG, TaskSubgraph, KGEdge, EdgeType
from agent_mem.core.belief_graph import BeliefGraph, AtomicBelief, BeliefType, BeliefStatus

def test_problem_file():
    """Test ProblemFile basic functionality."""
    print("Testing ProblemFile...")

    # Create a basic ProblemFile
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

    print(" ProblemFile creation test passed")

    # Test evidence pointer
    ptr = EvidencePointer(
        type="stdout",
        location="/tmp/test_stdout.txt",
        offset=0,
        length=100,
        hash="abc123"
    )

    assert ptr.type == "stdout"
    assert ptr.location == "/tmp/test_stdout.txt"
    print(" EvidencePointer test passed")

    # Test serialization
    json_str = pf.to_json()
    assert isinstance(json_str, str)

    pf2 = ProblemFile.from_json(json_str)
    assert pf2.task_id == pf.task_id
    print(" ProblemFile serialization test passed")

def test_observation_kg():
    """Test ObservationKG basic functionality."""
    print("\nTesting ObservationKG...")

    kg = ObservationKG()
    assert len(kg.task_subgraphs) == 0
    print(" ObservationKG creation test passed")

    # Test TaskSubgraph
    subgraph = TaskSubgraph(task_id="test_task")

    pf1 = ProblemFile(task_id="test_task", action_type=ActionType.TOOL_CALL)
    pf2 = ProblemFile(task_id="test_task", action_type=ActionType.CODE_EDIT)

    subgraph.add_action(pf1)
    subgraph.add_action(pf2)

    edge = KGEdge(
        source_id=pf1.action_id,
        target_id=pf2.action_id,
        edge_type=EdgeType.SUCCESS_NEXT
    )
    subgraph.add_edge(edge)

    assert len(subgraph.action_nodes) == 2
    assert len(subgraph.edges) == 1
    print(" TaskSubgraph test passed")

def test_belief_graph():
    """Test BeliefGraph basic functionality."""
    print("\nTesting BeliefGraph...")

    belief = AtomicBelief(
        belief_type=BeliefType.WORKFLOW,
        confidence=0.8
    )

    assert belief.belief_type == BeliefType.WORKFLOW
    assert belief.confidence == 0.8
    assert belief.status == BeliefStatus.ACTIVE
    print(" AtomicBelief test passed")

    bg = BeliefGraph()
    belief_id = bg.add_atomic_belief(belief)
    assert belief_id in bg.atomic_beliefs
    print(" BeliefGraph test passed")

def main():
    """Run all tests."""
    print("Running Agent-mem MVP simple tests...")
    print("="*60)

    tests_passed = 0
    tests_failed = 0

    try:
        test_problem_file()
        tests_passed += 1
    except Exception as e:
        print(f" ProblemFile test failed: {e}")
        tests_failed += 1

    try:
        test_observation_kg()
        tests_passed += 1
    except Exception as e:
        print(f" ObservationKG test failed: {e}")
        tests_failed += 1

    try:
        test_belief_graph()
        tests_passed += 1
    except Exception as e:
        print(f" BeliefGraph test failed: {e}")
        tests_failed += 1

    print("\n" + "="*60)
    print(f"Test Summary:")
    print(f"  Passed: {tests_passed}")
    print(f"  Failed: {tests_failed}")
    print(f"  Total:  {tests_passed + tests_failed}")

    return tests_failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
