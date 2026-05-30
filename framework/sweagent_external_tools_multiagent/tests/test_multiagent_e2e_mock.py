"""
Level 1 end-to-end mock tests for the multi-agent framework.

Simulates a complete attempt lifecycle with mocked LLM calls:
  M1: Normal reuse path (T1-A + T1-C approve)
  M2: Critic revise  second submit auto-approved
  M3: T1-B cross-attempt hint loading
  M4: All disabled  baseline identical to phase9-v2
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import types
import unittest
from collections import deque
from typing import Any
from unittest.mock import MagicMock, patch, call


def _inject_fake_sweagent():
    class AbstractAgentHook:
        pass
    class StepOutput:
        def __init__(self, thought="", action="", observation=""):
            self.thought, self.action, self.observation = thought, action, observation
    def get_logger(name, **kw):
        import logging; return logging.getLogger(name)
    AgentInfo = dict
    Trajectory = list

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
    sweagent.agent = agent_mod; agent_mod.hooks = hooks_mod
    hooks_mod.abstract = abstract_mod; sweagent.types = types_mod
    sweagent.utils = utils_mod; utils_mod.log = log_mod
    for name, mod in [
        ("sweagent", sweagent), ("sweagent.agent", agent_mod),
        ("sweagent.agent.hooks", hooks_mod), ("sweagent.agent.hooks.abstract", abstract_mod),
        ("sweagent.types", types_mod), ("sweagent.utils", utils_mod),
        ("sweagent.utils.log", log_mod),
    ]:
        sys.modules.setdefault(name, mod)

_inject_fake_sweagent()

_MULTI_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_MULTI_AGENT_DIR, _PKG_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.modules.setdefault("sweagent_external_tools_v2",
                       types.ModuleType("sweagent_external_tools_v2"))

from sweagent_external_tools_multiagent.bridge_hook import ExternalToolBridgeHook
from sweagent_external_tools_multiagent.agent_mem.processing.critic_agent import CriticVerdict
from sweagent_external_tools_multiagent.agent_mem.storage.interim_cache import InterimCache


def _step(action="", observation="", thought=""):
    from sweagent.types import StepOutput
    return StepOutput(thought=thought, action=action, observation=observation)

def _hook(env: dict, tmpdir: str | None = None, **kw) -> ExternalToolBridgeHook:
    defaults = {
        "SWE_AGENT_MEM_INJECT_ENABLED": "1",
        "SWE_AGENT_MEM_MIN_CONFIDENCE": "0",
        "SWE_AGENT_MEM_FAMILY_COOLDOWN_STEPS": "0",
        "SWE_AGENT_MEM_MAX_HINTS_PER_FAMILY_PER_ATTEMPT": "5",
        "AGENT_MEM_PATCH_CONSISTENCY_GATE": "off",
        "AGENT_MEM_REUSE_EXPLORE": "off",
    }
    if tmpdir:
        defaults["SWE_AGENT_T1B_CACHE_DIR"] = tmpdir
    defaults.update(env)
    with patch.dict(os.environ, defaults):
        h = ExternalToolBridgeHook(**kw)
    h._instance_id = "test-instance"
    h._attempt_id = "attempt-e2e"
    return h

_GOOD_PATCH_ACTION = (
    "submit\n"
    "diff --git a/astropy/modeling/separable.py b/astropy/modeling/separable.py\n"
    "--- a/astropy/modeling/separable.py\n"
    "+++ b/astropy/modeling/separable.py\n"
    "@@ -242,3 +242,3 @@\n"
    "-    cstack = np.ones((naxes, naxes))\n"
    "+    cstack = right\n"
    "\nEOF"
)

def _bug_invariant_item(file="astropy/modeling/separable.py", func="_cstack"):
    return {
        "card_type": "BugInvariantCard",
        "hint": f"Fix requires modifying `{func}` in {file}",
        "family_id": "sem:bugfix",
        "item_confidence": 0.92,
        "batch_confidence": 0.92,
        "selection_score": 0.92,
        "minimal_patch_signature": {
            "anchors": [{"file_path": file, "symbol_name": func,
                         "symbol_kind": "function", "param_signature": ["left", "right"]}],
            "key_added_lines": ["+    cstack = right"],
            "key_added_lines_hash": "abc123",
        },
        "card_id": "card-001",
        "recommendation": "Replace literal `1` with `right`.",
    }


class TestM1NormalReusePath(unittest.TestCase):

    def setUp(self):
        # Mock T1-A to track calls but return unchanged hints
        self.reformulation_calls: list[tuple] = []

        class TrackingReformulationAgent:
            def reformat(_self, hints, summary, phase):
                self.reformulation_calls.append((len(hints), phase))
                return hints
            @classmethod
            def from_env(cls):
                return cls()

        # Mock T1-C to approve
        class ApproveCritic:
            def critique(_self, **kw):
                return CriticVerdict(
                    semantic_match_with_invariant=0.92,
                    semantic_match_with_anti_pattern=0.05,
                    verdict="approve",
                    revision_hint="",
                    reasoning="matches invariant",
                )
            @classmethod
            def from_env(cls):
                return cls()

        self.hook = _hook({
            "SWE_AGENT_T1A_ENABLED": "true",
            "SWE_AGENT_T1C_ENABLED": "true",
        })
        self.hook._reformulation_agent = TrackingReformulationAgent()
        self.hook._critic_agent = ApproveCritic()

        # Inject a BugInvariantCard
        inv = _bug_invariant_item()
        self.hook._pending_memory_hints = deque([inv])
        self.hook._v2_active_invariants = [inv]

    def test_m1_reformulation_called_on_model_query(self):
        """T1-A reformulation fires when on_model_query is called with hints."""
        messages = []
        self.hook._step_index = 4
        self.hook.on_model_query(messages=messages, agent="main")
        self.assertGreaterEqual(len(self.reformulation_calls), 1,
                                "ReformulationAgent.reformat was not called")
        n_hints, phase = self.reformulation_calls[0]
        self.assertGreater(n_hints, 0)

    def test_m1_hint_injected_into_messages(self):
        """After reformulation, hint is injected into LLM messages."""
        messages = []
        self.hook._step_index = 4
        self.hook.on_model_query(messages=messages, agent="main")
        contents = " ".join(m["content"] for m in messages)
        self.assertIn("AgentMem Hints", contents)

    def test_m1_critic_approves_on_submit(self):
        """A T1-C approve verdict allows submission and increments its metric."""
        self.hook._step_budget_state["steps_since_last_edit"] = 3
        self.hook._step_budget_state["steps_since_last_patch"] = 3
        submit_step = _step(action=_GOOD_PATCH_ACTION)
        decision = self.hook._t1c_run_critic_and_decide(submit_step)
        self.assertEqual(decision, "allow")
        self.assertEqual(self.hook._metrics["t1c_approve"], 1)

    def test_m1_full_attempt_lifecycle_no_errors(self):
        """A complete simulated attempt runs without exceptions."""
        # Step 1: exploration
        for i in range(5):
            self.hook.on_action_executed(step=_step(
                action=f"grep -r bug src/ #{i}",
                observation=f"src/modeling/separable.py:{42+i}: def _cstack",
            ))
        # Step 2: model query with hint
        messages = []
        self.hook._step_index = 5
        self.hook.on_model_query(messages=messages, agent="main")
        # Step 3: submit
        submit_step = _step(action=_GOOD_PATCH_ACTION)
        decision = self.hook._t1c_run_critic_and_decide(submit_step)
        self.assertEqual(decision, "allow")
        # Verify metrics updated
        self.assertEqual(self.hook._metrics["t1c_approve"], 1)
        self.assertGreater(len(self.hook._trajectory_window), 0)


class TestM2CriticReviseFlow(unittest.TestCase):

    def setUp(self):
        self.critic_call_count = 0

        class CountingCritic:
            def critique(_self, **kw):
                self.critic_call_count += 1
                if self.critic_call_count == 1:
                    return CriticVerdict(
                        semantic_match_with_invariant=0.25,
                        semantic_match_with_anti_pattern=0.1,
                        verdict="revise",
                        revision_hint="Use `other_uncert` not `other_class`",
                        reasoning="param name mismatch",
                    )
                # Should never be called a second time
                return CriticVerdict.approve_fallback("second_call_unexpected")
            @classmethod
            def from_env(cls): return cls()

        self.hook = _hook({"SWE_AGENT_T1C_ENABLED": "true"})
        self.hook._critic_agent = CountingCritic()
        inv = _bug_invariant_item()
        self.hook._v2_active_invariants = [inv]

    def test_m2_revise_on_first_submit(self):
        """First submit triggers revise verdict."""
        submit_step = _step(action=_GOOD_PATCH_ACTION)
        decision = self.hook._t1c_run_critic_and_decide(submit_step)
        self.assertEqual(decision, "allow")
        self.assertTrue(self.hook._t1c_revision_pending)
        self.assertEqual(self.hook._metrics["t1c_revise"], 1)

    def test_m2_revision_hint_in_pending_buffer(self):
        """After revise, revision hint is in pending_memory_hints."""
        submit_step = _step(action=_GOOD_PATCH_ACTION)
        self.hook._t1c_run_critic_and_decide(submit_step)
        hints = [h["hint"] for h in self.hook._pending_memory_hints]
        self.assertTrue(any("[Critic Review]" in h for h in hints),
                        f"No Critic Review hint found: {hints}")

    def test_m2_second_submit_auto_approved_no_llm(self):
        """Second submit (after revise) is auto-approved; LLM not called again."""
        submit_step = _step(action=_GOOD_PATCH_ACTION)
        # First submission requests revision.
        self.hook._t1c_run_critic_and_decide(submit_step)
        self.assertEqual(self.critic_call_count, 1)
        # Second submission is auto-approved without another LLM call.
        decision2 = self.hook._t1c_run_critic_and_decide(submit_step)
        self.assertEqual(decision2, "allow")
        self.assertEqual(self.critic_call_count, 1)  # still 1, not 2
        self.assertFalse(self.hook._t1c_revision_pending)

    def test_m2_model_query_injects_revision_hint(self):
        """on_model_query after revise injects the [Critic Review] hint into messages."""
        submit_step = _step(action=_GOOD_PATCH_ACTION)
        self.hook._t1c_run_critic_and_decide(submit_step)
        messages = []
        self.hook._step_index = 10
        self.hook.on_model_query(messages=messages, agent="main")
        all_content = " ".join(m["content"] for m in messages)
        self.assertIn("Critic Review", all_content)


class TestM3T1BInterimCrossAttempt(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil; shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_m3_interim_hints_available_in_second_attempt(self):
        """Interim cache written in attempt-1 is visible in attempt-2 on_init."""
        # Simulate attempt-1 writing an interim card
        cache = InterimCache(cache_dir=self.tmpdir)
        cache.write_interim_card(
            instance_id="test-instance",
            attempt_id="attempt-1",
            card_type="InterimLocalizationCard",
            localization={"file": "astropy/modeling/separable.py",
                          "function": "_cstack", "line_range": "242-242", "confidence": 0.75},
            source_step=12,
        )

        # Start attempt-2 hook, which loads the cache on init
        hook = _hook({
            "SWE_AGENT_T1B_ENABLED": "true",
        }, tmpdir=self.tmpdir)
        hook._instance_id = "test-instance"
        hook._attempt_id = "attempt-2"
        hook._interim_cache = InterimCache(cache_dir=self.tmpdir)
        hook._t1b_load_interim_hints()

        hints = list(hook._pending_memory_hints)
        self.assertTrue(
            any("separable.py" in h.get("hint", "") for h in hints),
            f"interim hint for separable.py not found: {hints}"
        )

    def test_m3_interim_written_during_attempt(self):
        """T1-B trigger during an attempt writes to cache asynchronously."""
        hook = _hook({"SWE_AGENT_T1B_ENABLED": "true"}, tmpdir=self.tmpdir)
        hook._instance_id = "test-instance"
        hook._interim_cache = InterimCache(cache_dir=self.tmpdir)

        # Simulate 2 consecutive steps with file:line observations
        for _ in range(hook._t1b_localize_threshold):
            obs = "src/problem.py:100: def broken_func(self, arg):"
            hook._t1b_check_triggers(_step(observation=obs), obs)

        time.sleep(0.2)  # let async thread finish
        cards = InterimCache(cache_dir=self.tmpdir).read_interim_cards("test-instance")
        self.assertGreaterEqual(len(cards), 1)

    def test_m3_t1b_triggers_on_first_test_pass(self):
        """T1-B fires when agent first sees a passing test."""
        hook = _hook({"SWE_AGENT_T1B_ENABLED": "true"}, tmpdir=self.tmpdir)
        hook._instance_id = "test-instance"
        hook._interim_cache = InterimCache(cache_dir=self.tmpdir)
        hook._t1b_last_localization = [("src/foo.py", "42")]

        obs = "1 passed in 0.5s"
        hook._t1b_check_triggers(_step(observation=obs), obs)
        time.sleep(0.2)

        self.assertTrue(hook._t1b_first_pass_written)
        cards = InterimCache(cache_dir=self.tmpdir).read_interim_cards("test-instance")
        self.assertGreaterEqual(len(cards), 1)


class TestM4AllDisabledBaseline(unittest.TestCase):

    def setUp(self):
        self.hook = _hook({
            "SWE_AGENT_T1A_ENABLED": "false",
            "SWE_AGENT_T1B_ENABLED": "false",
            "SWE_AGENT_T1C_ENABLED": "false",
        })

    def test_m4_no_t1_agents_initialised(self):
        self.assertIsNone(self.hook._reformulation_agent)
        self.assertIsNone(self.hook._interim_cache)
        self.assertIsNone(self.hook._critic_agent)

    def test_m4_all_t1_metrics_zero_after_lifecycle(self):
        """A full attempt with T1 disabled keeps all T1 metrics at 0."""
        # Simulate several steps
        for i in range(5):
            self.hook.on_action_executed(step=_step(
                action=f"grep foo #{i}", observation="src/x.py:10: def f():"
            ))
        messages = []
        self.hook.on_model_query(messages=messages, agent="main")

        self.assertEqual(self.hook._metrics["t1a_reformat_called"], 0)
        self.assertEqual(self.hook._metrics["t1b_interim_written"], 0)
        self.assertEqual(self.hook._metrics["t1c_approve"], 0)
        self.assertEqual(self.hook._metrics["t1c_revise"], 0)
        self.assertEqual(self.hook._metrics["t1c_reject"], 0)

    def test_m4_trajectory_window_still_maintained(self):
        """Even with T1 disabled, trajectory window is updated (shared infra)."""
        self.hook.on_action_executed(step=_step(action="ls", observation="foo.py"))
        self.assertEqual(len(self.hook._trajectory_window), 1)

    def test_m4_model_query_injects_hints_normally(self):
        """Standard hint injection still works when T1 is off."""
        inv = _bug_invariant_item()
        self.hook._pending_memory_hints = deque([inv])
        messages = []
        self.hook._step_index = 4
        self.hook.on_model_query(messages=messages, agent="main")
        # At least one message should be injected
        self.assertGreater(len(messages), 0)
        content = " ".join(m["content"] for m in messages)
        self.assertIn("AgentMem Hints", content)


if __name__ == "__main__":
    unittest.main()
