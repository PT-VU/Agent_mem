"""
Level 0 bridge_hook integration tests for T1-A / T1-B / T1-C.

sweagent is not installed in the test environment, so we inject a lightweight
fake into sys.modules before importing bridge_hook.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from collections import deque
from typing import Any
from unittest.mock import MagicMock, patch

def _inject_fake_sweagent():
    """Create minimal stubs for sweagent.*  that bridge_hook needs."""
    class AbstractAgentHook:
        pass

    class StepOutput:
        def __init__(self, thought="", action="", observation=""):
            self.thought = thought
            self.action = action
            self.observation = observation

    def get_logger(name, **kwargs):
        import logging
        return logging.getLogger(name)

    AgentInfo = dict
    Trajectory = list

    # Build the module tree
    sweagent = types.ModuleType("sweagent")
    agent_mod = types.ModuleType("sweagent.agent")
    hooks_mod = types.ModuleType("sweagent.agent.hooks")
    abstract_mod = types.ModuleType("sweagent.agent.hooks.abstract")
    abstract_mod.AbstractAgentHook = AbstractAgentHook
    types_mod = types.ModuleType("sweagent.types")
    types_mod.AgentInfo = AgentInfo
    types_mod.StepOutput = StepOutput
    types_mod.Trajectory = Trajectory
    utils_mod = types.ModuleType("sweagent.utils")
    log_mod = types.ModuleType("sweagent.utils.log")
    log_mod.get_logger = get_logger

    sweagent.agent = agent_mod
    agent_mod.hooks = hooks_mod
    hooks_mod.abstract = abstract_mod
    sweagent.types = types_mod
    sweagent.utils = utils_mod
    utils_mod.log = log_mod

    for name, mod in [
        ("sweagent", sweagent),
        ("sweagent.agent", agent_mod),
        ("sweagent.agent.hooks", hooks_mod),
        ("sweagent.agent.hooks.abstract", abstract_mod),
        ("sweagent.types", types_mod),
        ("sweagent.utils", utils_mod),
        ("sweagent.utils.log", log_mod),
    ]:
        sys.modules.setdefault(name, mod)


_inject_fake_sweagent()

# Set up sys.path so bridge_hook can be imported as part of its package.
# bridge_hook uses relative imports (from .tools.io_utils import ...),
# so we must import it as sweagent_external_tools_multiagent.bridge_hook
_MULTI_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# _MULTI_AGENT_DIR is the artifact framework directory.
if _MULTI_AGENT_DIR not in sys.path:
    sys.path.insert(0, _MULTI_AGENT_DIR)

# Also make the package root available for direct agent_mem imports
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# _PKG_ROOT = .../sweagent_external_tools_multiagent/
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

# Stub out any sweagent_external_tools_v2 that might be transitively imported
import importlib, unittest.mock
sys.modules.setdefault("sweagent_external_tools_v2",
                       types.ModuleType("sweagent_external_tools_v2"))

# Import bridge_hook via the proper package path
from sweagent_external_tools_multiagent.bridge_hook import ExternalToolBridgeHook

# Import the new T1 modules via the package
from sweagent_external_tools_multiagent.agent_mem.processing.reformulation_agent import ReformulationAgent
from sweagent_external_tools_multiagent.agent_mem.processing.critic_agent import CriticAgent, CriticVerdict
from sweagent_external_tools_multiagent.agent_mem.storage.interim_cache import InterimCache



def _step(action="echo hello", observation="ok", thought=""):
    from sweagent.types import StepOutput
    return StepOutput(thought=thought, action=action, observation=observation)


def _hook(env_overrides=None, **init_kwargs):
    """Create a minimally configured ExternalToolBridgeHook."""
    overrides = {
        "SWE_AGENT_MEM_INJECT_ENABLED": "1",
        "SWE_AGENT_MEM_MIN_CONFIDENCE": "0",
        "SWE_AGENT_MEM_FAMILY_COOLDOWN_STEPS": "0",
        "SWE_AGENT_MEM_MAX_HINTS_PER_FAMILY_PER_ATTEMPT": "5",
    }
    if env_overrides:
        overrides.update(env_overrides)
    with patch.dict(os.environ, overrides):
        h = ExternalToolBridgeHook(**init_kwargs)
    h._instance_id = "test-instance"
    h._attempt_id = "attempt-1"
    return h


def _invariant_item(file="src/foo.py", function="bar", params=("a", "b")):
    return {
        "card_type": "BugInvariantCard",
        "hint": f"Fix in {file} function {function}",
        "family_id": "sem:bugfix",
        "item_confidence": 0.9,
        "batch_confidence": 0.9,
        "selection_score": 0.9,
        "minimal_patch_signature": {
            "anchors": [{"file_path": file, "symbol_name": function,
                         "symbol_kind": "function", "param_signature": list(params)}],
            "key_added_lines": [f"+    return {params[0]}"],
        },
        "card_id": "card-001",
    }



class TestT1AHookIntegration(unittest.TestCase):
    """T1-A: ReformulationAgent is called before _select_hints_for_injection."""

    def test_t1a_enabled_calls_reformat_before_injection(self):
        """T1-A reformat is invoked when pending hints exist."""
        hook = _hook({"SWE_AGENT_T1A_ENABLED": "true"})
        # Replace with a real ReformulationAgent that we can track
        reformat_called = []
        real_agent = ReformulationAgent(model="m", api_base="http://x", api_key="k", timeout=1.0)

        def track_reformat(hints, summary, phase):
            reformat_called.append((len(hints), phase))
            return hints  # return unchanged
        real_agent.reformat = track_reformat

        hook._reformulation_agent = real_agent
        hook._pending_memory_hints = deque([_invariant_item()])
        hook._step_index = 4

        messages = []
        hook.on_model_query(messages=messages, agent="main")

        self.assertTrue(len(reformat_called) >= 1, "reformat was not called")
        n_hints, phase = reformat_called[0]
        self.assertEqual(n_hints, 1)
        self.assertIn(phase, ("explore", "localized", "fixing", "pre_submit"))

    def test_t1a_respects_max_reformats_per_attempt(self):
        """Stage2 gate: T1-A can be capped per attempt to reduce hint churn."""
        hook = _hook({
            "SWE_AGENT_T1A_ENABLED": "true",
            "SWE_AGENT_T1A_MAX_REFORMATS_PER_ATTEMPT": "1",
        })
        reformat_called = []
        real_agent = ReformulationAgent(model="m", api_base="http://x", api_key="k", timeout=1.0)

        def track_reformat(hints, summary, phase):
            reformat_called.append((len(hints), phase))
            return hints

        real_agent.reformat = track_reformat
        hook._reformulation_agent = real_agent

        hook._pending_memory_hints = deque([_invariant_item()])
        hook._t1a_reformat_pending_hints()
        hook._pending_memory_hints = deque([_invariant_item()])
        hook._t1a_reformat_pending_hints()

        self.assertEqual(len(reformat_called), 1)
        self.assertEqual(hook._metrics["t1a_reformat_called"], 1)

    def test_t1a_disabled_does_not_call_reformat(self):
        """When T1-A is disabled, _reformulation_agent stays None."""
        hook = _hook({"SWE_AGENT_T1A_ENABLED": "false"})
        self.assertIsNone(hook._reformulation_agent)

    def test_trajectory_window_updated_on_action_executed(self):
        """on_action_executed appends to _trajectory_window."""
        hook = _hook()
        self.assertEqual(len(hook._trajectory_window), 0)
        hook.on_action_executed(step=_step(action="grep foo", observation="foo.py:10"))
        self.assertEqual(len(hook._trajectory_window), 1)
        entry = hook._trajectory_window[0]
        self.assertIn("grep foo", entry["action"])

    def test_infer_step_phase_explore(self):
        """Early steps with no edits remain in the explore phase."""
        hook = _hook()
        hook._step_index = 3
        hook._patch_candidate_exists = False
        self.assertEqual(hook._infer_step_phase(), "explore")

    def test_infer_step_phase_fixing(self):
        """A patch with recent edits remains in the fixing phase."""
        hook = _hook()
        hook._step_index = 20
        hook._patch_candidate_exists = True
        # steps_since_last_edit must be > 0 to avoid pre_submit branch
        hook._step_budget_state["steps_since_last_edit"] = 3
        hook._step_budget_state["steps_since_last_patch"] = 3
        self.assertEqual(hook._infer_step_phase(), "fixing")


class TestT1BHookIntegration(unittest.TestCase):
    """T1-B: Interim hints are loaded from cache on on_init."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_interim_hints_loaded_on_init(self):
        """on_init loads existing interim cards into _pending_memory_hints."""
        # Pre-write a cache entry
        cache = InterimCache(cache_dir=self.tmpdir)
        cache.write_interim_card(
            instance_id="test-instance",
            attempt_id="attempt-0",
            card_type="InterimLocalizationCard",
            localization={"file": "src/problem.py", "function": "fix_me",
                          "line_range": "55-60", "confidence": 0.7},
            source_step=8,
        )

        hook = _hook({
            "SWE_AGENT_T1B_ENABLED": "true",
            "SWE_AGENT_T1B_CACHE_DIR": self.tmpdir,
        })
        hook._instance_id = "test-instance"
        hook._interim_cache = InterimCache(cache_dir=self.tmpdir)

        # Manually re-call the load method (simulates what on_init does after instance_id is set)
        hook._pending_memory_hints.clear()
        hook._t1b_load_interim_hints()

        hints = list(hook._pending_memory_hints)
        self.assertTrue(any("problem.py" in h.get("hint", "") for h in hints),
                        f"Expected problem.py in hints, got: {hints}")

    def test_t1b_localization_trigger_fires_async_write(self):
        """Consecutive localization pattern hits fire an async thread write."""
        hook = _hook({
            "SWE_AGENT_T1B_ENABLED": "true",
            "SWE_AGENT_T1B_CACHE_DIR": self.tmpdir,
            "SWE_AGENT_T1B_LOCALIZE_THRESHOLD": "2",
        })
        hook._instance_id = "trig-instance"
        hook._interim_cache = InterimCache(cache_dir=self.tmpdir)

        # Simulate two consecutive steps with localization patterns
        obs = "Found issue at module/core.py:77  line contains the bug"
        hook._t1b_localization_hits = 0
        hook._t1b_check_triggers(_step(observation=obs), obs)
        hook._t1b_check_triggers(_step(observation=obs), obs)

        # Give the async thread a moment to write
        time.sleep(0.2)
        cards = InterimCache(cache_dir=self.tmpdir).read_interim_cards("trig-instance")
        self.assertGreaterEqual(len(cards), 1)
        self.assertEqual(cards[0]["localization"]["file"], "module/core.py")

    def test_t1b_disabled_no_cache(self):
        """T1-B disabled  _interim_cache is None."""
        hook = _hook({"SWE_AGENT_T1B_ENABLED": "false"})
        self.assertIsNone(hook._interim_cache)


class TestT1CHookIntegration(unittest.TestCase):
    """T1-C: Critic agent is wired into on_actions_generated submit flow."""

    def test_t1c_not_called_without_invariant(self):
        """If no invariant cards are active, Critic is not invoked."""
        hook = _hook({"SWE_AGENT_T1C_ENABLED": "true"})
        hook._v2_active_invariants = []
        critic_called = []
        mock_critic = MagicMock()
        mock_critic.critique.side_effect = lambda **kw: (critic_called.append(1) or CriticVerdict.approve_fallback())
        hook._critic_agent = mock_critic

        # submit action with no invariant
        submit_action = "submit\n<<END>>\n+    return x + y\n<<END>>"
        step = _step(action=submit_action)
        decision = hook._t1c_run_critic_and_decide(step)

        self.assertEqual(decision, "allow")
        self.assertEqual(len(critic_called), 0)

    # _PATCH_ACTION: realistic format that _v2_extract_current_patch_text can parse
    _PATCH_ACTION = (
        "submit\n"
        "diff --git a/src/uncertainty.py b/src/uncertainty.py\n"
        "--- a/src/uncertainty.py\n"
        "+++ b/src/uncertainty.py\n"
        "@@ -10,2 +10,2 @@\n"
        "-    return other_class.represent_as(self)\n"
        "+    return other_uncert.represent_as(self)\n"
        "\nEOF"
    )

    def test_t1c_revise_injects_hint(self):
        """When critic returns revise, revision_hint is enqueued in pending hints."""
        hook = _hook({"SWE_AGENT_T1C_ENABLED": "true"})
        # Register an invariant card
        inv = _invariant_item()
        hook._v2_active_invariants = [inv]

        mock_critic = MagicMock()
        mock_critic.critique.return_value = CriticVerdict(
            semantic_match_with_invariant=0.25,
            semantic_match_with_anti_pattern=0.1,
            verdict="revise",
            revision_hint="Use param `other_uncert` not `other_class`",
            reasoning="param name wrong",
        )
        hook._critic_agent = mock_critic

        step = _step(action=self._PATCH_ACTION)
        decision = hook._t1c_run_critic_and_decide(step)

        # Should return allow (revise doesn't block, it adds a hint)
        self.assertEqual(decision, "allow")
        self.assertTrue(hook._t1c_revision_pending)

        # pending_memory_hints should have the revision hint
        hints_text = [h["hint"] for h in hook._pending_memory_hints]
        self.assertTrue(any("[Critic Review]" in t for t in hints_text),
                        f"No Critic Review hint found, got: {hints_text}")

    def test_t1c_reject_returns_force_reuse(self):
        """When critic returns reject, decision is force_reuse."""
        hook = _hook({"SWE_AGENT_T1C_ENABLED": "true"})
        inv = _invariant_item()
        hook._v2_active_invariants = [inv]

        mock_critic = MagicMock()
        mock_critic.critique.return_value = CriticVerdict(
            semantic_match_with_invariant=0.3,
            semantic_match_with_anti_pattern=0.9,
            verdict="reject",
            revision_hint="Matches known bad pattern",
            reasoning="anti-pattern match",
        )
        hook._critic_agent = mock_critic

        step = _step(action=self._PATCH_ACTION)
        decision = hook._t1c_run_critic_and_decide(step)
        self.assertEqual(decision, "force_reuse")

    def test_t1c_can_use_precheck_diff_for_plain_submit(self):
        """Stage2 path: Critic can review the diff captured by v2 pre-submit inspection."""
        hook = _hook({
            "SWE_AGENT_T1C_ENABLED": "true",
            "SWE_AGENT_T1C_USE_PRECHECK_DIFF": "true",
        })
        hook._v2_active_invariants = [_invariant_item()]

        mock_critic = MagicMock()
        mock_critic.critique.return_value = CriticVerdict(
            semantic_match_with_invariant=0.9,
            semantic_match_with_anti_pattern=0.0,
            verdict="approve",
            revision_hint="",
            reasoning="captured diff matches invariant",
        )
        hook._critic_agent = mock_critic

        observation = (
            "===== [V2-Gate] PRE-SUBMIT DIFF =====\n"
            "diff --git a/src/foo.py b/src/foo.py\n"
            "--- a/src/foo.py\n"
            "+++ b/src/foo.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-    return a\n"
            "+    return a + b\n"
            "===== [V2-Gate] END DIFF =====\n"
        )
        hook.on_action_executed(step=_step(action="git diff", observation=observation))

        decision = hook._t1c_run_critic_and_decide(_step(action="submit"))

        self.assertEqual(decision, "allow")
        mock_critic.critique.assert_called_once()
        self.assertIn("diff --git", mock_critic.critique.call_args.kwargs["patch_diff"])

    def test_t1c_fallback_approve_is_unavailable_not_approve(self):
        """Fallback approve is tracked separately so metrics do not overstate Critic value."""
        hook = _hook({
            "SWE_AGENT_T1C_ENABLED": "true",
            "SWE_AGENT_T1C_SPLIT_FALLBACK_APPROVE": "true",
        })
        hook._v2_active_invariants = [_invariant_item()]

        mock_critic = MagicMock()
        mock_critic.critique.return_value = CriticVerdict.approve_fallback("llm_error_or_timeout")
        hook._critic_agent = mock_critic

        decision = hook._t1c_run_critic_and_decide(_step(action=self._PATCH_ACTION))

        self.assertEqual(decision, "allow")
        self.assertEqual(hook._metrics["t1c_unavailable"], 1)
        self.assertEqual(hook._metrics["t1c_approve"], 0)

    def test_t1c_unavailable_policy_revise_once_injects_hint(self):
        """Stage2 loop: Critic unavailable can become one deterministic revise."""
        hook = _hook({
            "SWE_AGENT_T1C_ENABLED": "true",
            "SWE_AGENT_T1C_SPLIT_FALLBACK_APPROVE": "true",
            "SWE_AGENT_T1C_DETERMINISTIC_GUARD": "true",
            "SWE_AGENT_T1C_UNAVAILABLE_POLICY": "revise_once",
        })
        hook._v2_active_invariants = [_invariant_item()]

        mock_critic = MagicMock()
        mock_critic.critique.return_value = CriticVerdict.approve_fallback("llm_error_or_timeout")
        hook._critic_agent = mock_critic

        decision = hook._t1c_run_critic_and_decide(_step(action=self._PATCH_ACTION))

        self.assertEqual(decision, "allow")
        self.assertEqual(hook._metrics["t1c_unavailable"], 1)
        self.assertEqual(hook._metrics["t1c_revise"], 1)
        self.assertEqual(hook._metrics["t1c_deterministic_revise"], 1)
        self.assertTrue(hook._t1c_revision_pending)
        hints_text = [h["hint"] for h in hook._pending_memory_hints]
        self.assertTrue(any("[Critic Review]" in t for t in hints_text), hints_text)

    def test_t1c_deterministic_param_mismatch_revises_without_llm(self):
        """Deterministic guard turns invariant param mismatches into revise verdicts."""
        hook = _hook({
            "SWE_AGENT_T1C_ENABLED": "true",
            "SWE_AGENT_T1C_DETERMINISTIC_GUARD": "true",
        })
        hook._v2_active_invariants = [_invariant_item(function="bar", params=("a", "b"))]

        mock_critic = MagicMock()
        hook._critic_agent = mock_critic

        patch_action = (
            "submit\n"
            "diff --git a/src/foo.py b/src/foo.py\n"
            "--- a/src/foo.py\n"
            "+++ b/src/foo.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-def bar(a, b):\n"
            "+def bar(a, c):\n"
            "+    return a + c\n"
            "\nEOF"
        )
        decision = hook._t1c_run_critic_and_decide(_step(action=patch_action))

        self.assertEqual(decision, "allow")
        mock_critic.critique.assert_not_called()
        self.assertEqual(hook._metrics["t1c_revise"], 1)
        self.assertEqual(hook._metrics["t1c_deterministic_revise"], 1)
        self.assertTrue(hook._t1c_revision_pending)

    def test_t1c_deterministic_anti_pattern_rejects_without_llm(self):
        """Deterministic guard rejects known failed parameter signatures."""
        hook = _hook({
            "SWE_AGENT_T1C_ENABLED": "true",
            "SWE_AGENT_T1C_DETERMINISTIC_GUARD": "true",
        })
        hook._v2_active_invariants = [_invariant_item(function="bar", params=("a", "b"))]
        hook._v2_active_anti_patterns = [{
            "card_type": "BugAntiPatternCard",
            "failed_patch_signature": {"param_signature": ["a", "c"]},
        }]

        mock_critic = MagicMock()
        hook._critic_agent = mock_critic

        patch_action = (
            "submit\n"
            "diff --git a/src/foo.py b/src/foo.py\n"
            "--- a/src/foo.py\n"
            "+++ b/src/foo.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-def bar(a, b):\n"
            "+def bar(a, c):\n"
            "+    return a + c\n"
            "\nEOF"
        )
        decision = hook._t1c_run_critic_and_decide(_step(action=patch_action))

        self.assertEqual(decision, "force_reuse")
        mock_critic.critique.assert_not_called()
        self.assertEqual(hook._metrics["t1c_reject"], 1)
        self.assertEqual(hook._metrics["t1c_deterministic_reject"], 1)

    def test_t1c_no_infinite_revise_loop(self):
        """Second submit after revise is auto-approved without calling Critic."""
        hook = _hook({"SWE_AGENT_T1C_ENABLED": "true"})
        inv = _invariant_item()
        hook._v2_active_invariants = [inv]
        hook._t1c_revision_pending = True  # simulating "just had a revise"

        mock_critic = MagicMock()
        hook._critic_agent = mock_critic

        submit_action = "submit\n<<END>>\n+    return x\n<<END>>"
        step = _step(action=submit_action)
        decision = hook._t1c_run_critic_and_decide(step)

        self.assertEqual(decision, "allow")
        mock_critic.critique.assert_not_called()
        self.assertFalse(hook._t1c_revision_pending)

    def test_t1c_disabled_no_critic_agent(self):
        """T1-C disabled  _critic_agent is None."""
        hook = _hook({"SWE_AGENT_T1C_ENABLED": "false"})
        self.assertIsNone(hook._critic_agent)


class TestT1AllDisabledBaseline(unittest.TestCase):
    """When all T1 modules are disabled, behaviour is identical to phase9-v2."""

    def test_all_modules_disabled_no_t1_state(self):
        """All T1 agents are None when disabled."""
        hook = _hook({
            "SWE_AGENT_T1A_ENABLED": "false",
            "SWE_AGENT_T1B_ENABLED": "false",
            "SWE_AGENT_T1C_ENABLED": "false",
        })
        self.assertIsNone(hook._reformulation_agent)
        self.assertIsNone(hook._interim_cache)
        self.assertIsNone(hook._critic_agent)

    def test_all_t1_metrics_zero_when_disabled(self):
        """T1 metrics start at 0 and stay 0 after a model query."""
        hook = _hook({
            "SWE_AGENT_T1A_ENABLED": "false",
            "SWE_AGENT_T1B_ENABLED": "false",
            "SWE_AGENT_T1C_ENABLED": "false",
        })
        messages = []
        hook.on_model_query(messages=messages, agent="main")
        self.assertEqual(hook._metrics["t1a_reformat_called"], 0)
        self.assertEqual(hook._metrics["t1b_interim_written"], 0)
        self.assertEqual(hook._metrics["t1c_approve"], 0)

    def test_on_action_executed_does_not_crash_when_t1b_disabled(self):
        """on_action_executed runs without error when T1-B is off."""
        hook = _hook({"SWE_AGENT_T1B_ENABLED": "false"})
        try:
            hook.on_action_executed(step=_step(
                action="grep foo src/",
                observation="src/foo.py:42:    def bar(self):",
            ))
        except Exception as e:
            self.fail(f"on_action_executed raised: {e}")


if __name__ == "__main__":
    unittest.main()
