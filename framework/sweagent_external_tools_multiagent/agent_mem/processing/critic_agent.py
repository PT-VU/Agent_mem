"""
T1-C: Pre-submit Critic Agent

Performs semantic review of a patch-diff just before submission.
Activates only when at least one BugInvariantCard is available.
Falls back to "approve" silently on timeout or LLM errors.

Controlled by SWE_AGENT_T1C_ENABLED (default: false).

Verdict meaning:
  approve   - submit as-is
  revise    - inject revision_hint as a high-priority system hint; give agent
              one more chance (next submit is auto-approved)
  reject    - trigger force_reuse from the best BugInvariantCard
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

_SYSTEM_PROMPT = """\
You are a strict patch semantic reviewer for an automated code-repair system.
You will be given a patch diff and reference information about the known-correct
fix pattern (BugInvariantCard) and known-bad patterns (BugAntiPatternCards).

Evaluate:
1. Does the patch SEMANTICALLY match the BugInvariantCard? Score 0.0-1.0.
2. Does the patch resemble any BugAntiPatternCard? Score 0.0-1.0 (low = good).
3. Final verdict: "approve" / "revise" / "reject"
   - approve: patch looks correct; semantic_match_with_invariant >= 0.7 AND
              semantic_match_with_anti_pattern <= 0.3
   - revise:  something is off but fixable; provide a concise revision_hint
   - reject:  patch clearly matches a known-bad pattern or is fundamentally wrong
4. revision_hint: one precise sentence for the agent IF verdict is revise/reject.

Output ONLY valid JSON, no prose:
{
  "semantic_match_with_invariant": <float 0-1>,
  "semantic_match_with_anti_pattern": <float 0-1>,
  "verdict": "approve"|"revise"|"reject",
  "revision_hint": "<string or empty>",
  "reasoning": "<one-line internal note>"
}
"""

_USER_TEMPLATE = """\
Bug description / instance ID: {instance_id}

BugInvariantCard (known-good pattern):
  file: {inv_file}
  function: {inv_function}
  key_lines: {inv_key_lines}
  param_signature: {inv_params}
  recommendation: {inv_recommendation}

BugAntiPatternCards (known-bad patterns):
{anti_patterns_block}

Current patch diff:
{patch_diff}

Recent test outputs (last 3):
{test_outputs_block}
"""


@dataclass
class CriticVerdict:
    semantic_match_with_invariant: float = 1.0
    semantic_match_with_anti_pattern: float = 0.0
    verdict: str = "approve"
    revision_hint: str = ""
    reasoning: str = ""

    @classmethod
    def approve_fallback(cls, reason: str = "fallback") -> "CriticVerdict":
        return cls(reasoning=reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_match_with_invariant": round(self.semantic_match_with_invariant, 3),
            "semantic_match_with_anti_pattern": round(self.semantic_match_with_anti_pattern, 3),
            "verdict": self.verdict,
            "revision_hint": self.revision_hint,
            "reasoning": self.reasoning,
        }


class CriticAgent:
    """Semantic patch reviewer.

    Args:
        model: LLM model name.
        api_base: OpenAI-compatible API base URL.
        api_key: API key.
        timeout: Per-call timeout seconds.
        revise_threshold: invariant match score BELOW which triggers revise.
        reject_threshold: anti-pattern match score ABOVE which triggers reject.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: float = 10.0,
        revise_threshold: float = 0.4,
        reject_threshold: float = 0.8,
    ) -> None:
        self._model = (
            model
            or os.getenv("SWE_AGENT_T1C_MODEL")
            or os.getenv("AGENT_MEM_LLM_MODEL")
            or "gpt-4o-mini"
        )
        self._api_base = (
            api_base
            or os.getenv("AGENT_MEM_LLM_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self._api_key = (
            api_key
            or os.getenv("AGENT_MEM_LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        )
        self._timeout = timeout
        self._revise_threshold = revise_threshold
        self._reject_threshold = reject_threshold

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def critique(
        self,
        *,
        patch_diff: str,
        instance_id: str,
        invariant_card: dict[str, Any],
        anti_pattern_cards: list[dict[str, Any]],
        recent_test_outputs: list[str],
    ) -> CriticVerdict:
        """Run semantic review. Return an approve fallback instead of raising."""
        if not patch_diff.strip():
            return CriticVerdict.approve_fallback("empty_patch")
        if not invariant_card:
            return CriticVerdict.approve_fallback("no_invariant_card")

        user_msg = self._build_user_msg(
            patch_diff=patch_diff,
            instance_id=instance_id,
            invariant_card=invariant_card,
            anti_pattern_cards=anti_pattern_cards,
            recent_test_outputs=recent_test_outputs,
        )
        try:
            raw = self._call_llm(user_msg)
            verdict = self._parse_verdict(raw)
        except Exception:
            return CriticVerdict.approve_fallback("llm_error_or_timeout")

        # Apply thresholds as a safety net in case LLM verdict is wrong
        if verdict.semantic_match_with_anti_pattern >= self._reject_threshold:
            verdict.verdict = "reject"
        elif verdict.semantic_match_with_invariant < self._revise_threshold:
            if verdict.verdict == "approve":
                verdict.verdict = "revise"

        return verdict

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_msg(
        *,
        patch_diff: str,
        instance_id: str,
        invariant_card: dict[str, Any],
        anti_pattern_cards: list[dict[str, Any]],
        recent_test_outputs: list[str],
    ) -> str:
        sig = invariant_card.get("minimal_patch_signature") or {}
        anchors = sig.get("anchors") or []
        inv_file = ""
        inv_function = ""
        inv_params: list[str] = []
        for a in anchors:
            if isinstance(a, dict):
                if not inv_file:
                    inv_file = str(a.get("file_path") or "")
                if not inv_function:
                    inv_function = str(a.get("symbol_name") or "")
                if not inv_params:
                    inv_params = list(a.get("param_signature") or [])
        inv_key_lines = sig.get("key_added_lines") or invariant_card.get("key_change_summary") or ""
        inv_recommendation = str(invariant_card.get("recommendation") or "")[:300]

        # Anti-pattern summary
        ap_lines: list[str] = []
        for i, ap in enumerate(anti_pattern_cards[:3], 1):
            sig2 = (ap.get("failed_patch_signature") or {}) if isinstance(ap, dict) else {}
            desc = str(ap.get("description") or ap.get("recommendation") or "")[:200]
            ap_lines.append(f"  [{i}] {desc}")
        anti_patterns_block = "\n".join(ap_lines) if ap_lines else "  (none)"

        test_outputs_block = "\n---\n".join(
            f"  {t[:300]}" for t in recent_test_outputs
        ) or "  (none)"

        return _USER_TEMPLATE.format(
            instance_id=instance_id,
            inv_file=inv_file or "(unknown)",
            inv_function=inv_function or "(unknown)",
            inv_key_lines=str(inv_key_lines)[:200],
            inv_params=str(inv_params),
            inv_recommendation=inv_recommendation,
            anti_patterns_block=anti_patterns_block,
            patch_diff=patch_diff[:2000],
            test_outputs_block=test_outputs_block,
        )

    def _call_llm(self, user_msg: str) -> str:
        from urllib import request as urllib_request

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            f"{self._api_base}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=self._timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]

    @staticmethod
    def _parse_verdict(text: str) -> CriticVerdict:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            ).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return CriticVerdict.approve_fallback("parse_error")
        obj = json.loads(text[start : end + 1])
        return CriticVerdict(
            semantic_match_with_invariant=float(
                obj.get("semantic_match_with_invariant", 1.0)
            ),
            semantic_match_with_anti_pattern=float(
                obj.get("semantic_match_with_anti_pattern", 0.0)
            ),
            verdict=str(obj.get("verdict", "approve")),
            revision_hint=str(obj.get("revision_hint", "")),
            reasoning=str(obj.get("reasoning", "")),
        )

    @classmethod
    def from_env(cls) -> "CriticAgent":
        timeout = float(
            os.getenv("SWE_AGENT_T1C_TIMEOUT_SEC", "10.0") or "10.0"
        )
        revise_threshold = float(
            os.getenv("SWE_AGENT_T1C_REVISE_THRESHOLD", "0.4") or "0.4"
        )
        reject_threshold = float(
            os.getenv("SWE_AGENT_T1C_REJECT_THRESHOLD", "0.8") or "0.8"
        )
        return cls(
            timeout=timeout,
            revise_threshold=revise_threshold,
            reject_threshold=reject_threshold,
        )
