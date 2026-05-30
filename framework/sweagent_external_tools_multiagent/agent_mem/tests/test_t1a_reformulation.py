"""Level 0 unit tests for T1-A: ReformulationAgent."""
from __future__ import annotations

import json
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Allow direct import from the package root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent_mem.processing.reformulation_agent import ReformulationAgent


def _make_hints(texts: list[str]) -> list[dict]:
    return [
        {
            "hint": t,
            "family_id": f"f{i}",
            "item_confidence": 0.8,
            "batch_confidence": 0.8,
        }
        for i, t in enumerate(texts)
    ]


def _mock_llm_response(rewrites: list[dict]) -> MagicMock:
    """Return a mock that simulates a successful LLM response."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(
        {"choices": [{"message": {"content": json.dumps(rewrites)}}]}
    ).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestReformulationAgent(unittest.TestCase):

    def setUp(self):
        self.agent = ReformulationAgent(
            model="test-model",
            api_base="http://localhost:9999",
            api_key="test-key",
            timeout=5.0,
        )


    def test_reformat_explore_phase_strips_diff(self):
        """LLM rewrites hints to focus on file location during explore."""
        hints = _make_hints([
            "Fix requires modifying `_cstack()` in separable.py line 242. "
            "The key change: --- a/foo.py\n+++ b/foo.py\n@@ -242 +242 @@ ..."
        ])
        rewritten_text = "[explore] Bug located in separable.py function `_cstack()`"
        mock_resp = _mock_llm_response(
            [{"original_idx": 0, "rewritten_hint": rewritten_text}]
        )
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = self.agent.reformat(hints, "Step 3: action='grep -r'", "explore")
        self.assertEqual(result[0]["hint"], rewritten_text)
        self.assertTrue(result[0].get("reformulated"))
        self.assertNotIn("@@", result[0]["hint"])

    def test_reformat_fixing_phase_preserves_params(self):
        """LLM rewrites to emphasise param signature during fixing phase."""
        hints = _make_hints(["Fix function represent_as in uncertainty.py"])
        rewritten_text = "[fixing] `represent_as(self, other_uncert)`  use param name `other_uncert` not `other_class`"
        mock_resp = _mock_llm_response(
            [{"original_idx": 0, "rewritten_hint": rewritten_text}]
        )
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = self.agent.reformat(hints, "Step 18: action='edit'", "fixing")
        self.assertIn("other_uncert", result[0]["hint"])

    def test_partial_rewrite_leaves_others_intact(self):
        """If LLM only rewrites index 0, index 1 remains original."""
        original_1 = "original hint 1"
        hints = _make_hints(["hint 0", original_1])
        mock_resp = _mock_llm_response(
            [{"original_idx": 0, "rewritten_hint": "rewritten 0"}]
        )
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = self.agent.reformat(hints, "", "explore")
        self.assertEqual(result[0]["hint"], "rewritten 0")
        self.assertEqual(result[1]["hint"], original_1)


    def test_fallback_on_llm_timeout(self):
        """Returns original hints unchanged when LLM raises timeout."""
        hints = _make_hints(["original hint"])
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            result = self.agent.reformat(hints, "", "explore")
        self.assertEqual(result[0]["hint"], "original hint")
        self.assertFalse(result[0].get("reformulated", False))

    def test_fallback_on_connection_error(self):
        """Returns original hints unchanged on connection error."""
        hints = _make_hints(["original hint"])
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            result = self.agent.reformat(hints, "", "explore")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["hint"], "original hint")

    def test_fallback_on_invalid_json_response(self):
        """Returns original hints when LLM returns non-JSON text."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"Sorry, I cannot help with that."
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        hints = _make_hints(["original hint"])
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = self.agent.reformat(hints, "", "explore")
        self.assertEqual(result[0]["hint"], "original hint")

    def test_fallback_on_malformed_json_array(self):
        """Returns original hints when LLM returns valid JSON but wrong shape."""
        mock_resp = _mock_llm_response.__func__(  # call staticmethod directly
            {"not": "an array"}  # type: ignore[arg-type]
        ) if False else MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"choices": [{"message": {"content": '{"not": "array"}'}}]}
        ).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        hints = _make_hints(["original hint"])
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = self.agent.reformat(hints, "", "explore")
        self.assertEqual(result[0]["hint"], "original hint")


    def test_empty_hints_returns_empty_without_llm(self):
        """Empty hint list is returned directly without calling LLM."""
        with patch("urllib.request.urlopen") as mock_open:
            result = self.agent.reformat([], "", "explore")
        mock_open.assert_not_called()
        self.assertEqual(result, [])

    def test_out_of_range_idx_ignored(self):
        """LLM response with out-of-range original_idx is silently ignored."""
        hints = _make_hints(["only one hint"])
        mock_resp = _mock_llm_response(
            [{"original_idx": 99, "rewritten_hint": "should be ignored"}]
        )
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = self.agent.reformat(hints, "", "explore")
        self.assertEqual(result[0]["hint"], "only one hint")

    def test_from_env_uses_env_vars(self):
        """from_env() picks up SWE_AGENT_T1A_TIMEOUT_SEC."""
        with patch.dict(os.environ, {"SWE_AGENT_T1A_TIMEOUT_SEC": "3.5"}):
            agent = ReformulationAgent.from_env()
        self.assertAlmostEqual(agent._timeout, 3.5)

    def test_markdown_fence_stripped(self):
        """LLM response wrapped in ```json ... ``` is parsed correctly."""
        fenced = '```json\n[{"original_idx": 0, "rewritten_hint": "clean"}]\n```'
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"choices": [{"message": {"content": fenced}}]}
        ).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        hints = _make_hints(["original"])
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = self.agent.reformat(hints, "", "explore")
        self.assertEqual(result[0]["hint"], "clean")


if __name__ == "__main__":
    unittest.main()
