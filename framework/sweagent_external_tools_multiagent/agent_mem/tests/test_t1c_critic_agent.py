"""Level 0 unit tests for T1-C: CriticAgent."""
from __future__ import annotations

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent_mem.processing.critic_agent import CriticAgent, CriticVerdict


def _make_invariant(file="src/foo.py", function="bar", params=("x", "y"), key_lines="+    return x + y") -> dict:
    return {
        "card_type": "BugInvariantCard",
        "minimal_patch_signature": {
            "anchors": [
                {
                    "file_path": file,
                    "symbol_name": function,
                    "symbol_kind": "function",
                    "param_signature": list(params),
                }
            ],
            "key_added_lines": [key_lines],
        },
        "recommendation": f"Fix requires modifying `{function}` in {file}.",
        "item_confidence": 0.9,
    }


def _make_anti_pattern(description="wrong param order") -> dict:
    return {
        "card_type": "BugAntiPatternCard",
        "description": description,
        "failed_patch_signature": {},
        "item_confidence": 0.8,
    }


def _llm_response(obj: dict) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(
        {"choices": [{"message": {"content": json.dumps(obj)}}]}
    ).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestCriticAgent(unittest.TestCase):

    def setUp(self):
        self.agent = CriticAgent(
            model="test-model",
            api_base="http://localhost:9999",
            api_key="test-key",
            timeout=5.0,
            revise_threshold=0.4,
            reject_threshold=0.8,
        )
        self.inv = _make_invariant()
        self.patch_good = "+    return other_uncert.represent_as(self)\n"
        self.patch_bad  = "+    return other_class.represent_as(self)\n"


    def test_approve_on_matching_invariant(self):
        mock_resp = _llm_response({
            "semantic_match_with_invariant": 0.92,
            "semantic_match_with_anti_pattern": 0.05,
            "verdict": "approve",
            "revision_hint": "",
            "reasoning": "patch matches invariant",
        })
        with patch("urllib.request.urlopen", return_value=mock_resp):
            v = self.agent.critique(
                patch_diff=self.patch_good,
                instance_id="test-inst",
                invariant_card=self.inv,
                anti_pattern_cards=[],
                recent_test_outputs=[],
            )
        self.assertEqual(v.verdict, "approve")


    def test_revise_on_param_mismatch(self):
        mock_resp = _llm_response({
            "semantic_match_with_invariant": 0.3,
            "semantic_match_with_anti_pattern": 0.1,
            "verdict": "revise",
            "revision_hint": "Use param name `other_uncert` not `other_class`",
            "reasoning": "param name wrong",
        })
        with patch("urllib.request.urlopen", return_value=mock_resp):
            v = self.agent.critique(
                patch_diff=self.patch_bad,
                instance_id="test-inst",
                invariant_card=self.inv,
                anti_pattern_cards=[],
                recent_test_outputs=[],
            )
        self.assertEqual(v.verdict, "revise")
        self.assertIn("other_uncert", v.revision_hint)


    def test_reject_on_anti_pattern_threshold(self):
        """High anti-pattern score overrides LLM approve verdict via threshold."""
        mock_resp = _llm_response({
            "semantic_match_with_invariant": 0.6,
            "semantic_match_with_anti_pattern": 0.85,
            "verdict": "approve",  # LLM said approve, but threshold overrides
            "revision_hint": "",
            "reasoning": "matches bad pattern",
        })
        with patch("urllib.request.urlopen", return_value=mock_resp):
            v = self.agent.critique(
                patch_diff=self.patch_bad,
                instance_id="test-inst",
                invariant_card=self.inv,
                anti_pattern_cards=[_make_anti_pattern()],
                recent_test_outputs=[],
            )
        self.assertEqual(v.verdict, "reject")

    def test_revise_threshold_overrides_approve(self):
        """Low invariant match score overrides LLM approve verdict."""
        mock_resp = _llm_response({
            "semantic_match_with_invariant": 0.25,
            "semantic_match_with_anti_pattern": 0.1,
            "verdict": "approve",  # LLM said approve, but threshold forces revise
            "revision_hint": "",
            "reasoning": "",
        })
        with patch("urllib.request.urlopen", return_value=mock_resp):
            v = self.agent.critique(
                patch_diff=self.patch_bad,
                instance_id="test-inst",
                invariant_card=self.inv,
                anti_pattern_cards=[],
                recent_test_outputs=[],
            )
        self.assertEqual(v.verdict, "revise")


    def test_approve_fallback_on_llm_timeout(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            v = self.agent.critique(
                patch_diff=self.patch_good,
                instance_id="test-inst",
                invariant_card=self.inv,
                anti_pattern_cards=[],
                recent_test_outputs=[],
            )
        self.assertEqual(v.verdict, "approve")
        self.assertIn("llm_error_or_timeout", v.reasoning)

    def test_approve_when_no_invariant_card(self):
        """Empty invariant card skips LLM and returns approve immediately."""
        with patch("urllib.request.urlopen") as mock_open:
            v = self.agent.critique(
                patch_diff=self.patch_good,
                instance_id="test-inst",
                invariant_card={},
                anti_pattern_cards=[],
                recent_test_outputs=[],
            )
        mock_open.assert_not_called()
        self.assertEqual(v.verdict, "approve")

    def test_approve_when_empty_patch(self):
        """Empty patch diff returns approve immediately without LLM call."""
        with patch("urllib.request.urlopen") as mock_open:
            v = self.agent.critique(
                patch_diff="",
                instance_id="test-inst",
                invariant_card=self.inv,
                anti_pattern_cards=[],
                recent_test_outputs=[],
            )
        mock_open.assert_not_called()
        self.assertEqual(v.verdict, "approve")

    def test_approve_fallback_on_invalid_llm_json(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'not json at all'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            v = self.agent.critique(
                patch_diff=self.patch_good,
                instance_id="test-inst",
                invariant_card=self.inv,
                anti_pattern_cards=[],
                recent_test_outputs=[],
            )
        self.assertEqual(v.verdict, "approve")


    def test_verdict_to_dict_structure(self):
        v = CriticVerdict(
            semantic_match_with_invariant=0.9,
            semantic_match_with_anti_pattern=0.1,
            verdict="approve",
            revision_hint="",
            reasoning="ok",
        )
        d = v.to_dict()
        self.assertIn("semantic_match_with_invariant", d)
        self.assertIn("verdict", d)
        self.assertAlmostEqual(d["semantic_match_with_invariant"], 0.9)


    def test_from_env_reads_thresholds(self):
        with patch.dict(os.environ, {
            "SWE_AGENT_T1C_REVISE_THRESHOLD": "0.35",
            "SWE_AGENT_T1C_REJECT_THRESHOLD": "0.75",
            "SWE_AGENT_T1C_TIMEOUT_SEC": "8.0",
        }):
            agent = CriticAgent.from_env()
        self.assertAlmostEqual(agent._revise_threshold, 0.35)
        self.assertAlmostEqual(agent._reject_threshold, 0.75)
        self.assertAlmostEqual(agent._timeout, 8.0)

    def test_markdown_fence_in_llm_response(self):
        """CriticAgent handles ```json fenced response."""
        fenced_content = '```json\n{"semantic_match_with_invariant": 0.85, "semantic_match_with_anti_pattern": 0.05, "verdict": "approve", "revision_hint": "", "reasoning": "ok"}\n```'
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"choices": [{"message": {"content": fenced_content}}]}
        ).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            v = self.agent.critique(
                patch_diff=self.patch_good,
                instance_id="test-inst",
                invariant_card=self.inv,
                anti_pattern_cards=[],
                recent_test_outputs=[],
            )
        self.assertEqual(v.verdict, "approve")
        self.assertAlmostEqual(v.semantic_match_with_invariant, 0.85)


if __name__ == "__main__":
    unittest.main()
