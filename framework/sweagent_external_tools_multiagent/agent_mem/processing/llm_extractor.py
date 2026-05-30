"""
LLM-assisted extraction helpers.

Design:
- LLM is optional and best-effort.
- Rule-based output is always available.
- Final storage schema remains unchanged; this module only emits intermediate
  analysis objects.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib import error, request

from ..core.problem_file import Outcome, ProblemFile
from .taxonomy import ErrorTaxonomy


@dataclass
class CriticalSignal:
    attempt_id: str
    critical_step: int
    critical_module: str
    error_type: str
    root_cause: str
    cascading_effects: List[Dict[str, Any]]
    correction_guidance: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "critical_step": self.critical_step,
            "critical_module": self.critical_module,
            "error_type": self.error_type,
            "root_cause": self.root_cause,
            "cascading_effects": self.cascading_effects,
            "correction_guidance": self.correction_guidance,
            "confidence": round(max(0.0, min(1.0, self.confidence)), 4),
        }


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", (text or "").lower()).strip("_") or "unknown"


class LLMExperienceExtractor:
    """
    Generates intermediate extraction artifacts.

    Modes:
    - off / heuristic: deterministic only.
    - hybrid: deterministic + optional OpenAI-compatible JSON refinement.
    """

    def __init__(
        self,
        *,
        taxonomy: Optional[ErrorTaxonomy] = None,
        mode: str = "hybrid",
        api_url: str = "",
        api_key: str = "",
        model: str = "deepseek-chat",
        timeout_sec: float = 20.0,
    ):
        self.taxonomy = taxonomy or ErrorTaxonomy()
        self.mode = (mode or "").strip().lower() or "hybrid"
        self.api_url = api_url.strip()
        self.api_key = api_key.strip()
        self.model = model
        self.timeout_sec = max(1.0, float(timeout_sec))

    @classmethod
    def from_env(cls, *, taxonomy: Optional[ErrorTaxonomy] = None) -> "LLMExperienceExtractor":
        return cls(
            taxonomy=taxonomy,
            mode=os.getenv("AGENT_MEM_LLM_EXTRACT_MODE", "hybrid"),
            api_url=os.getenv("AGENT_MEM_LLM_API_URL", ""),
            api_key=os.getenv("AGENT_MEM_LLM_API_KEY", ""),
            model=os.getenv("AGENT_MEM_LLM_MODEL", "deepseek-chat"),
            timeout_sec=float(os.getenv("AGENT_MEM_LLM_TIMEOUT_SEC", "20")),
        )

    def analyze_steps(
        self,
        *,
        attempt_id: str,
        actions: List[ProblemFile],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        rule_assessments = self.taxonomy.classify_attempt(attempt_id=attempt_id, actions=actions)
        llm_assessments: List[Dict[str, Any]] = []
        strategy_observations: List[Dict[str, Any]] = []
        llm_used = False

        if self._llm_enabled() and actions:
            llm_payload = self._call_llm_for_assessments(
                attempt_id=attempt_id,
                actions=actions,
                context=context,
            )
            llm_assessments = llm_payload.get("assessments", [])
            strategy_observations = llm_payload.get("strategy_observations", [])
            llm_used = bool(llm_assessments)

        merged = self._merge_assessments(rule_assessments, llm_assessments)
        return {
            "attempt_id": attempt_id,
            "taxonomy_version": self.taxonomy.version,
            "llm_mode": self.mode,
            "llm_used": llm_used,
            "rule_count": len(rule_assessments),
            "llm_count": len(llm_assessments),
            "assessments": merged,
            "strategy_observations": strategy_observations,
        }

    def detect_critical_signals(
        self,
        *,
        attempt_id: str,
        actions: List[ProblemFile],
        assessments: List[Dict[str, Any]],
        success: bool,
        exit_status: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[CriticalSignal]:
        """Return ordered critical signals across the full attempt trajectory.

        This enables step-level extraction granularity at run end: each detected
        failing step can contribute one structured experience candidate.
        """
        if not assessments:
            if success:
                return []
            # No explicit taxonomy hit: fallback to latest failed step.
            fallback_idx = len(actions) - 1 if actions else 0
            return [
                CriticalSignal(
                    attempt_id=attempt_id,
                    critical_step=max(0, fallback_idx),
                    critical_module="action",
                    error_type="unclassified_failure",
                    root_cause=f"attempt ended as {exit_status} without classified critical signal",
                    cascading_effects=[],
                    correction_guidance="narrow_scope_collect_failure_evidence_then_retry_with_targeted_fix",
                    confidence=0.35,
                )
            ]

        candidates = [row for row in assessments if bool(row.get("error_detected"))]
        if not candidates:
            return []

        module_weight = {"planning": 0, "action": 1, "system": 2, "reflection": 3, "memory": 4}
        candidates.sort(
            key=lambda row: (
                int(row.get("step_index", 0)),
                module_weight.get(str(row.get("module", "")), 9),
                -float(row.get("confidence", 0.0)),
            )
        )

        signals: List[CriticalSignal] = []
        for idx, chosen in enumerate(candidates):
            c_step = int(chosen.get("step_index", 0))
            c_mod = str(chosen.get("module", "unknown"))
            c_err = _slug(str(chosen.get("error_type", "unknown")))
            confidence = float(chosen.get("confidence", 0.4))

            cascading_effects: List[Dict[str, Any]] = []
            for row in candidates[idx + 1 :]:
                step = int(row.get("step_index", 0))
                if step <= c_step:
                    continue
                cascading_effects.append(
                    {
                        "step": step,
                        "impact": f"{row.get('module', 'unknown')}::{row.get('error_type', 'unknown')}",
                    }
                )
                if len(cascading_effects) >= 8:
                    break

            guidance = self._build_guidance(c_mod, c_err)
            root_cause = f"{c_mod} error '{c_err}' first appears at step {c_step}"

            # Optional refinement with LLM, never mandatory.
            if self._llm_enabled():
                refined = self._call_llm_for_critical(
                    attempt_id=attempt_id,
                    chosen=chosen,
                    cascading_effects=cascading_effects,
                    success=success,
                    exit_status=exit_status,
                    context=context,
                )
                if refined:
                    root_cause = refined.get("root_cause", root_cause) or root_cause
                    guidance = refined.get("correction_guidance", guidance) or guidance
                    confidence = max(confidence, float(refined.get("confidence", confidence)))

            signals.append(
                CriticalSignal(
                    attempt_id=attempt_id,
                    critical_step=c_step,
                    critical_module=c_mod,
                    error_type=c_err,
                    root_cause=root_cause[:500],
                    cascading_effects=cascading_effects,
                    correction_guidance=guidance[:500],
                    confidence=max(0.25, min(0.95, confidence)),
                )
            )

        return signals

    def detect_critical_signal(
        self,
        *,
        attempt_id: str,
        actions: List[ProblemFile],
        assessments: List[Dict[str, Any]],
        success: bool,
        exit_status: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[CriticalSignal]:
        """Backwards-compatible single-signal API.

        Keeps old call sites working while internally using the multi-signal
        implementation.
        """
        signals = self.detect_critical_signals(
            attempt_id=attempt_id,
            actions=actions,
            assessments=assessments,
            success=success,
            exit_status=exit_status,
            context=context,
        )
        if not signals:
            return None
        return signals[0]

    def _build_guidance(self, module: str, error_type: str) -> str:
        if module == "planning":
            return (
                "add_explicit_validation_plan_before_edit_and_limit_scope_to_related_files"
            )
        if module == "system":
            return (
                "verify_environment_dependencies_and_runtime_constraints_before_retrying_commands"
            )
        if error_type in {"import_error", "file_not_found"}:
            return "verify_import_or_path_context_then_apply_targeted_fix_and_rerun_related_test"
        if error_type in {"test_failure", "assertion_failed"}:
            return "reproduce_failing_test_minimally_then_patch_only_related_logic_and_recheck"
        return "collect_error_evidence_narrow_context_and_change_strategy_before_retry"

    def _llm_enabled(self) -> bool:
        if self.mode in {"off", "heuristic"}:
            return False
        return bool(self.api_url and self.api_key)

    def _merge_assessments(
        self,
        rule_rows: List[Dict[str, Any]],
        llm_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for source in (rule_rows, llm_rows):
            for row in source:
                key = "|".join(
                    [
                        str(row.get("step_index", "")),
                        str(row.get("module", "")),
                        _slug(str(row.get("error_type", ""))),
                    ]
                )
                incoming_conf = float(row.get("confidence", 0.0))
                existing = merged.get(key)
                if existing is None or incoming_conf > float(existing.get("confidence", 0.0)):
                    merged[key] = dict(row)
        out = list(merged.values())
        out.sort(key=lambda row: (int(row.get("step_index", 0)), -float(row.get("confidence", 0.0))))
        return out

    def _call_llm_for_assessments(
        self,
        *,
        attempt_id: str,
        actions: List[ProblemFile],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        compact = []
        for idx, action in enumerate(self._build_trajectory_excerpt(actions)):
            compact.append(
                {
                    "step_index": action.step_index if isinstance(action.step_index, int) else idx,
                    "action_type": action.action_type.value,
                    "action_family": action.action_family,
                    "outcome": action.outcome.value,
                    "intent_text": action.intent_text[:240],
                    "source_event": action.source_event,
                    "failure_type": (
                        action.failure_signature.error_type if action.failure_signature else ""
                    ),
                }
            )
        system_prompt = (
            "You are a strict JSON analyzer. "
            "Return only JSON. Prefer the object form "
            "{\"assessments\": [...], \"strategy_observations\": [...]} but a bare assessments array is also allowed. "
            "Each assessment item must contain fields: "
            "step_index,module,error_detected,error_type,evidence,reasoning,confidence. "
            "module must be one of memory/reflection/planning/action/system. "
            "Each strategy_observations item must contain: "
            "strategy_type,why_failed_or_risky,evidence,recommended_avoidance,confidence."
        )
        user_prompt = json.dumps(
            {
                "attempt_id": attempt_id,
                "trajectory": compact,
                "context": self._compact_context(context),
            },
            ensure_ascii=False,
        )
        payload = self._call_llm_json(system_prompt=system_prompt, user_prompt=user_prompt)
        assessments_payload: List[Any]
        strategy_payload: List[Any] = []
        if isinstance(payload, dict):
            assessments_payload = payload.get("assessments", []) if isinstance(payload.get("assessments"), list) else []
            strategy_payload = payload.get("strategy_observations", []) if isinstance(payload.get("strategy_observations"), list) else []
        elif isinstance(payload, list):
            assessments_payload = payload
        else:
            return {"assessments": [], "strategy_observations": []}
        out: List[Dict[str, Any]] = []
        for row in assessments_payload:
            if not isinstance(row, dict):
                continue
            if not bool(row.get("error_detected", False)):
                continue
            module = str(row.get("module", ""))
            if module not in {"memory", "reflection", "planning", "action", "system"}:
                continue
            out.append(
                {
                    "attempt_id": attempt_id,
                    "step_index": int(row.get("step_index", 0)),
                    "module": module,
                    "error_detected": True,
                    "error_type": _slug(str(row.get("error_type", "unknown"))),
                    "evidence": str(row.get("evidence", ""))[:300],
                    "reasoning": str(row.get("reasoning", ""))[:300],
                    "confidence": max(0.0, min(1.0, float(row.get("confidence", 0.5)))),
                }
            )
        return {
            "assessments": out,
            "strategy_observations": self._normalize_strategy_observations(strategy_payload),
        }

    def _call_llm_for_critical(
        self,
        *,
        attempt_id: str,
        chosen: Dict[str, Any],
        cascading_effects: List[Dict[str, Any]],
        success: bool,
        exit_status: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        system_prompt = (
            "You are a strict JSON analyzer for root-cause extraction. "
            "Return only a JSON object with keys: root_cause, correction_guidance, confidence."
        )
        user_prompt = json.dumps(
            {
                "attempt_id": attempt_id,
                "chosen_signal": chosen,
                "cascading_effects": cascading_effects,
                "success": success,
                "exit_status": exit_status,
                "context": self._compact_context(context),
            },
            ensure_ascii=False,
        )
        payload = self._call_llm_json(system_prompt=system_prompt, user_prompt=user_prompt)
        if not isinstance(payload, dict):
            return None
        return {
            "root_cause": str(payload.get("root_cause", ""))[:500],
            "correction_guidance": str(payload.get("correction_guidance", ""))[:500],
            "confidence": max(0.0, min(1.0, float(payload.get("confidence", 0.0) or 0.0))),
        }

    @staticmethod
    def _build_trajectory_excerpt(actions: List[ProblemFile]) -> List[ProblemFile]:
        if len(actions) <= 50:
            return list(actions)

        indices = set(range(min(15, len(actions))))
        indices.update(range(max(0, len(actions) - 15), len(actions)))
        for idx, action in enumerate(actions):
            if action.outcome == Outcome.FAIL or action.failure_signature:
                start = max(0, idx - 1)
                end = min(len(actions), idx + 2)
                indices.update(range(start, end))
        ordered = [actions[idx] for idx in sorted(indices)]
        return ordered[:80]

    @staticmethod
    def _compact_context(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(context, dict):
            return {}
        return {
            "task_problem_excerpt": str(context.get("task_problem_excerpt", ""))[:400],
            "changed_files": [str(x) for x in (context.get("changed_files") or [])[:8]],
            "step_count": int(context.get("step_count", 0) or 0),
            "ad_hoc_script_count": int(context.get("ad_hoc_script_count", 0) or 0),
            "ad_hoc_script_names": [str(x) for x in (context.get("ad_hoc_script_names") or [])[:8]],
            "patch_summary": context.get("patch_summary") if isinstance(context.get("patch_summary"), dict) else {},
            "validation_summary": (
                context.get("validation_summary") if isinstance(context.get("validation_summary"), dict) else {}
            ),
            "submission_status": str(context.get("submission_status", ""))[:64],
            "official_eval_status": str(context.get("official_eval_status", ""))[:64],
        }

    @staticmethod
    def _normalize_strategy_observations(payload: List[Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            strategy_type = _slug(str(row.get("strategy_type", "")))
            if not strategy_type:
                continue
            out.append(
                {
                    "strategy_type": strategy_type,
                    "why_failed_or_risky": str(row.get("why_failed_or_risky", ""))[:300],
                    "evidence": str(row.get("evidence", ""))[:240],
                    "recommended_avoidance": str(row.get("recommended_avoidance", ""))[:240],
                    "confidence": max(0.0, min(1.0, float(row.get("confidence", 0.0) or 0.0))),
                }
            )
        return out

    def _call_llm_json(self, *, system_prompt: str, user_prompt: str) -> Optional[Any]:
        if not self._llm_enabled():
            return None
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            self.api_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
        except (error.URLError, TimeoutError, OSError, ValueError):
            return None
        try:
            data = json.loads(raw)
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
        except Exception:
            return None
        if not content:
            return None
        return self._parse_json_content(content)

    @staticmethod
    def _parse_json_content(content: str) -> Optional[Any]:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        # Try to extract first {...} or [...] block.
        for opener, closer in (("{", "}"), ("[", "]")):
            start = text.find(opener)
            end = text.rfind(closer)
            if start >= 0 and end > start:
                candidate = text[start : end + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    continue
        return None
