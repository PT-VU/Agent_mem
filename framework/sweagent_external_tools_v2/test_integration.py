#!/usr/bin/env python
"""
Integration test for Agent-mem MVP.
Tests plan_generated and action_error event handling.
"""
import sys
import os
import json
import tempfile
import importlib.util
from pathlib import Path
import pytest

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import integration components
from agent_mem.integration.sweagent_adapter import SWEAgentAdapter
from agent_mem.processing.action_logger import ActionLogger
from agent_mem.processing.error_handler import ErrorHandler
from agent_mem.processing.kg_writer import EmbeddingGenerator, KGWriter
from agent_mem.retrieval.memory_agent import MemoryAgent
from agent_mem.storage.graph_store import GraphStore
from agent_mem.core.problem_file import ActionType, Outcome

def _require_sentence_transformers() -> None:
    """Skip test when sentence-transformers runtime is unavailable."""
    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip("sentence-transformers is not installed in current venv")


def test_sweagent_adapter():
    """Test SWE-agent adapter integration."""
    _require_sentence_transformers()
    print("Testing SWE-agent adapter...")

    # Create adapter with temporary storage
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = SWEAgentAdapter(storage_dir=tmpdir, evidence_dir=tmpdir)

        # Test plan_generated hook
        plan_data = {
            "agent": "integration-test",
            "thought": "Need a safe fix workflow",
            "action": "1. Read the file\n2. Fix the bug\n3. Run tests",
        }

        result = adapter.handle_plan_generated(plan_data)
        print(f"  plan_generated result: {result}")
        assert result is not None
        print(" plan_generated hook test passed")

        # Test action_error hook
        error_data = {
            "agent": "integration-test",
            "thought": "The file read failed",
            "action": "cat main.py",
            "error_type": "file_not_found",
            "error_message": "File not found: main.py",
            "context": {
                "working_dir": "/tmp/test",
                "files": ["main.py", "test.py"]
            }
        }

        result = adapter.handle_action_error(error_data)
        print(f"  action_error result: {result}")
        assert result is not None
        print(" action_error hook test passed")

        # Cross-process continuity simulation:
        # New adapter instance should resume same task context and create fail-retry edge.
        adapter2 = SWEAgentAdapter(storage_dir=tmpdir, evidence_dir=tmpdir)
        result2 = adapter2.handle_action_error(error_data)
        assert result2 is not None
        assert result2.get("task_id") == result.get("task_id")

        subgraph = adapter2.graph_store.observation_kg.get_task_subgraph(result2["task_id"])
        assert subgraph is not None
        assert len(subgraph.action_nodes) >= 2
        assert len(subgraph.get_failure_retry_chains()) >= 1
        print(" cross-process continuity test passed")

        done = adapter2.handle_run_done(
            {
                "agent": "integration-test",
                "instance_id": "integration-test",
                "event": "run_done",
                "exit_status": "submitted",
                "has_submission": True,
            }
        )
        assert done.get("event_handled") == "run_done"
        assert done.get("cleared") is True
        assert done.get("belief_update", {}).get("new_beliefs_created", 0) >= 1
        print(" run_done context cleanup test passed")

        search_result = adapter2.search_experience("fix file read failure", max_results=3)
        assert search_result.get("total_found", 0) >= 1
        assert isinstance(search_result.get("recommendations", []), list)
        print(" search_experience formal output test passed")

def test_action_logger():
    """Test action logger functionality."""
    print("\nTesting Action Logger...")

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_store = GraphStore(storage_dir=tmpdir)
        logger = ActionLogger(graph_store, evidence_dir=tmpdir)
        logger.start_task("test_task_002")

        # Test logging an action
        from agent_mem.core.problem_file import ActionType, Outcome
        problem_file = logger.log_action(
            action_type=ActionType.CODE_EDIT,
            intent_text="Add missing import",
            action_text="str_replace_editor insert ...",
            action_family="str_replace_editor",
            instance_id="inst-001",
            run_id="run-001",
            agent_name="integration-test",
            source_event="action_success",
            step_index=3,
            inputs={"target": "main.py"},
            tool_calls=[{"tool": "editor", "action": "insert"}],
            outcome=Outcome.SUCCESS,
            stdout="File modified successfully",
            diff_content="+import os",
        )
        assert problem_file is not None
        assert problem_file.task_id == "test_task_002"
        assert problem_file.schema_version == "2.0"
        assert problem_file.instance_id == "inst-001"
        assert problem_file.run_id == "run-001"
        assert problem_file.step_index == 3
        assert problem_file.patch_stats["lines_added"] >= 1
        print(" Action logging test passed")

def test_error_handler():
    """Test error handler functionality."""
    print("\nTesting Error Handler...")

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_store = GraphStore(storage_dir=tmpdir)
        from agent_mem.core.observation_kg import ObservationKG
        observation_kg = ObservationKG()
        handler = ErrorHandler(graph_store, observation_kg)

        # Test error analysis
        error_context = {
            "task_id": "test_task_003",
            "error_type": "ImportError",
            "error_message": "No module named 'missing_module'",
            "action_type": "import",
            "file_path": "/tmp/test/main.py",
            "line_number": 5
        }

        suggestions = handler.analyze_error(error_context)
        print(f"  Error suggestions: {suggestions}")
        assert isinstance(suggestions, list)
        print(" Error analysis test passed")

def test_memory_agent():
    """Test memory agent functionality."""
    print("\nTesting Memory Agent...")

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_store = GraphStore(storage_dir=tmpdir)
        agent = MemoryAgent(graph_store)

        # Test query rewriting
        query = "How to fix import errors in Python?"
        rewritten = agent.query_rewriting(query)
        print(f"  Original query: {query}")
        print(f"  Rewritten query: {rewritten}")
        assert rewritten is not None
        print(" Query rewriting test passed")

        # Test experience distillation
        experiences = [
            {
                "task_id": "exp_001",
                "description": "Fixed import error by adding missing package",
                "solution": "pip install missing-package"
            }
        ]

        distilled = agent.experience_distillation(experiences)
        print(f"  Distilled experience: {distilled}")
        assert distilled is not None
        print(" Experience distillation test passed")

def test_graph_store():
    """Test graph store functionality."""
    print("\nTesting Graph Store...")

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        store = GraphStore(storage_dir=tmpdir)

        # Test basic operations
        print(f"  Storage dir: {store.storage_dir}")
        assert store.observation_kg is not None
        assert store.belief_graph is not None
        print(" Graph store initialization test passed")

        # Test saving observation KG
        from agent_mem.core.observation_kg import TaskSubgraph
        from agent_mem.core.problem_file import ProblemFile, ActionType

        subgraph = TaskSubgraph(task_id="test_task")
        pf = ProblemFile(task_id="test_task", action_type=ActionType.TOOL_CALL)
        subgraph.add_action(pf)
        store.observation_kg.add_task_subgraph(subgraph)

        # Save to file
        if store.storage_dir:
            kg_file = store.storage_dir / "observation_kg.json"
            store.observation_kg.save_to_file(str(kg_file))
            assert kg_file.exists()
            print(" Observation KG save test passed")

def test_kg_writer_evidence_embeddings():
    """Tool output/diff embeddings should be generated from evidence content."""
    print("\nTesting KGWriter evidence embedding path...")

    class _StubEmbeddingGenerator:
        def __init__(self):
            self.provider = "stub"
            self.last_tool_output = ""
            self.last_diff = ""

        def generate_task_semantic_embedding(self, intent_text, task_context=None):
            return [0.1, 0.2]

        def generate_file_scope_embedding(self, touched_files, file_content_patterns=None):
            return [0.2, 0.3]

        def generate_error_signature_embedding(self, error_type, error_tokens):
            return [0.3, 0.4]

        def generate_tool_output_embedding(self, tool_output, tool_type):
            self.last_tool_output = tool_output
            return [0.4, 0.5]

        def generate_diff_summary_embedding(self, diff_content):
            self.last_diff = diff_content
            return [0.5, 0.6]

        def generate_intent_embedding(self, intent_text):
            return [0.6, 0.7]

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_store = GraphStore(storage_dir=tmpdir)
        logger = ActionLogger(graph_store, evidence_dir=tmpdir)
        logger.start_task("kgwriter_task")
        pf = logger.log_action(
            action_type=ActionType.CODE_EDIT,
            intent_text="update test",
            inputs={"action": "str_replace_editor"},
            tool_calls=[{"tool": "editor", "action": "replace"}],
            stdout="stdout sample",
            diff_content="+new line\n-old line\n",
            outcome=Outcome.SUCCESS,
        )
        stub = _StubEmbeddingGenerator()
        writer = KGWriter(graph_store, embedding_generator=stub)
        writer.write_action_with_embeddings(pf)

        assert pf.embeddings.emb_tool_output is not None
        assert pf.embeddings.emb_diff_summary is not None
        assert "stdout sample" in stub.last_tool_output
        assert "+new line" in stub.last_diff
        print(" KGWriter evidence embedding test passed")

def test_embedding_generator_configuration():
    """Test embedding generator provider selection (sentence-transformers only)."""
    _require_sentence_transformers()
    print("\nTesting Embedding Generator...")

    st_gen = EmbeddingGenerator(model="sentence-transformers", embedding_dim=32)
    st_vec = st_gen.generate_intent_embedding("fix parser bug")
    assert len(st_vec) > 0
    assert st_gen.provider == "sentence-transformers"
    print(" Embedding generator config test passed")

def main():
    """Run all integration tests."""
    print("Running Agent-mem MVP integration tests...")
    print("="*60)

    tests_passed = 0
    tests_failed = 0

    try:
        test_sweagent_adapter()
        tests_passed += 1
    except Exception as e:
        print(f" SWE-agent adapter test failed: {e}")
        tests_failed += 1

    try:
        test_action_logger()
        tests_passed += 1
    except Exception as e:
        print(f" Action logger test failed: {e}")
        tests_failed += 1

    try:
        test_error_handler()
        tests_passed += 1
    except Exception as e:
        print(f" Error handler test failed: {e}")
        tests_failed += 1

    try:
        test_memory_agent()
        tests_passed += 1
    except Exception as e:
        print(f" Memory agent test failed: {e}")
        tests_failed += 1

    try:
        test_graph_store()
        tests_passed += 1
    except Exception as e:
        print(f" Graph store test failed: {e}")
        tests_failed += 1

    try:
        test_kg_writer_evidence_embeddings()
        tests_passed += 1
    except Exception as e:
        print(f" KGWriter evidence embedding test failed: {e}")
        tests_failed += 1

    try:
        test_embedding_generator_configuration()
        tests_passed += 1
    except Exception as e:
        print(f" Embedding generator test failed: {e}")
        tests_failed += 1

    print("\n" + "="*60)
    print(f"Integration Test Summary:")
    print(f"  Passed: {tests_passed}")
    print(f"  Failed: {tests_failed}")
    print(f"  Total:  {tests_passed + tests_failed}")

    return tests_failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
