"""
T1-A: Memory Reformulation Agent

Rewrites buffered memory hint texts to match the current execution phase
of the main SWE-agent, so that the injected hints are contextually precise
rather than generic summaries.

Controlled by SWE_AGENT_T1A_ENABLED (default: false).
LLM call reuses the same API endpoint as LLMExperienceExtractor.
Falls back to original hints silently on any failure.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

_SYSTEM_PROMPT = """\
You are a memory reformulation assistant for a code-repair AI agent.
Your job: rewrite retrieved memory hint texts so they are maximally useful
for the agent's CURRENT execution phase.

Rules:
- explore: emphasise FILE and FUNCTION location. Omit diff details.
- localized: emphasise HOW to modify (key line/parameter). Keep short.
- fixing: emphasise exact parameter names, return types, key added lines.
- pre_submit: emphasise alignment with the known-good invariant signature.

Output ONLY a valid JSON array, no prose:
[{"original_idx": <int>, "rewritten_hint": "<string>"}]
"""

_USER_TEMPLATE = """\
Current execution phase: {phase}

Recent trajectory (last {n} steps):
{trajectory}

Memory hints to rewrite:
{hints_block}
"""


class ReformulationAgent:
    """Rewrites a list of pending hint dicts in-place (replaces 'hint' field).

    Args:
        model: OpenAI-compatible model name. Falls back to env
            `SWE_AGENT_T1A_MODEL`, `AGENT_MEM_LLM_MODEL`, then `"gpt-4o-mini"`.
        api_base: Base URL for OpenAI-compatible API.
        api_key: API key.
        timeout: Per-call timeout in seconds.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: float = 6.0,
    ) -> None:
        self._model = (
            model
            or os.getenv("SWE_AGENT_T1A_MODEL")
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

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def reformat(
        self,
        hints: list[dict[str, Any]],
        trajectory_summary: str,
        step_phase: str,
    ) -> list[dict[str, Any]]:
        """Return a new list of hint dicts with 'hint' fields rewritten.

        If the LLM call fails for any reason, the original list is returned
        unchanged; this method must never raise.
        """
        if not hints:
            return hints

        # Build a numbered block of hint texts for the prompt
        hint_lines: list[str] = []
        for idx, item in enumerate(hints):
            raw = str(item.get("hint", "")).strip()
            if raw:
                hint_lines.append(f"{idx}: {raw[:400]}")
        if not hint_lines:
            return hints

        hints_block = "\n".join(hint_lines)
        n_steps = trajectory_summary.count("Step ") or trajectory_summary.count("\n") + 1
        user_msg = _USER_TEMPLATE.format(
            phase=step_phase,
            n=n_steps,
            trajectory=trajectory_summary[:1200],
            hints_block=hints_block,
        )

        try:
            raw_response = self._call_llm(user_msg)
            rewrites = self._parse_response(raw_response)
        except Exception:
            return hints  # silent fallback

        if not rewrites:
            return hints

        result = list(hints)
        for rw in rewrites:
            idx = rw.get("original_idx")
            new_text = str(rw.get("rewritten_hint", "")).strip()
            if idx is None or not new_text:
                continue
            if not isinstance(idx, int) or idx < 0 or idx >= len(result):
                continue
            original = dict(result[idx])
            original["hint"] = new_text
            original["reformulated"] = True
            result[idx] = original

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_llm(self, user_msg: str) -> str:
        from urllib import error as urllib_error, request as urllib_request

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.0,
            "max_tokens": 1024,
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
    def _parse_response(text: str) -> list[dict[str, Any]]:
        text = text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            ).strip()
        # Try to find a JSON array
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            return []
        candidate = text[start : end + 1]
        parsed = json.loads(candidate)
        if not isinstance(parsed, list):
            return []
        return parsed

    @classmethod
    def from_env(cls) -> "ReformulationAgent":
        timeout = float(os.getenv("SWE_AGENT_T1A_TIMEOUT_SEC", "6.0") or "6.0")
        return cls(timeout=timeout)
