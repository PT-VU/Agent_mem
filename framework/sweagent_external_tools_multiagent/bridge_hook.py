from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
import time
from collections import deque
from typing import Any

from sweagent.agent.hooks.abstract import AbstractAgentHook
from sweagent.types import AgentInfo, StepOutput, Trajectory
from sweagent.utils.log import get_logger

try:
    from .agent_mem.processing.v21_shared import (
        build_success_fact_idempotency_key,
        classify_success_like,
    )
    from .agent_mem.storage.episode_ledger_store import EpisodeLedgerStore
except Exception:  # pragma: no cover - keep hook import lightweight
    EpisodeLedgerStore = None

    def build_success_fact_idempotency_key(trace_id: str, step_index: Any) -> str:
        trace = str(trace_id or "").strip()
        step = str(step_index if step_index is not None else "").strip()
        if not trace or not step:
            return ""
        return f"{trace}::{step}"

    def classify_success_like(
        *,
        observation: str = "",
        error_type: str = "",
        exit_status: str = "",
        has_submission: bool | None = None,
    ) -> bool:
        lowered = str(observation or "").lower()
        if error_type:
            return False
        if lowered and any(
            token in lowered
            for token in ("error", "exception", "traceback", "assert", "failed", "no such file", "timed out", "non-zero")
        ):
            return "simulated command error" in lowered
        if str(exit_status or "").strip().lower() in {"failed", "error", "timeout", "incomplete", "unresolved"}:
            return False
        if has_submission is True:
            return True
        return True

from .tools.io_utils import append_json_log

try:
    from .agent_mem.processing.reformulation_agent import ReformulationAgent
except Exception:
    ReformulationAgent = None  # type: ignore[assignment,misc]

try:
    from .agent_mem.processing.critic_agent import CriticAgent, CriticVerdict
except Exception:
    CriticAgent = None  # type: ignore[assignment,misc]
    CriticVerdict = None  # type: ignore[assignment,misc]

try:
    from .agent_mem.storage.interim_cache import InterimCache
except Exception:
    InterimCache = None  # type: ignore[assignment,misc]


class ExternalToolBridgeHook(AbstractAgentHook):
    """Bridge hook that calls external Tool A/Tool B commands.

    The event payload is serialized to `SWE_AGENT_EXT_EVENT_JSON`.
    """

    def __init__(
        self,
        *,
        tool_a_cmd: str | None = None,
        tool_b_cmd: str | None = None,
        timeout_sec: float | None = None,
    ):
        self.tool_a_cmd = tool_a_cmd or os.getenv("SWE_AGENT_EXT_TOOL_A_CMD", "").strip() or None
        self.tool_b_cmd = tool_b_cmd or os.getenv("SWE_AGENT_EXT_TOOL_B_CMD", "").strip() or None
        if timeout_sec is None:
            timeout_raw = os.getenv("SWE_AGENT_EXT_TOOL_TIMEOUT_SEC", "8.0")
            timeout_sec = float(timeout_raw)
        self.timeout_sec = timeout_sec
        self.tool_a_timeout_sec = self._read_float_env(
            "SWE_AGENT_EXT_TOOL_A_TIMEOUT_SEC",
            default=max(45.0, self.timeout_sec * 3.0),
        )
        self.tool_b_timeout_sec = self._read_float_env("SWE_AGENT_EXT_TOOL_B_TIMEOUT_SEC", default=self.timeout_sec)
        self.retry_timeout_sec = self._read_float_env(
            "SWE_AGENT_EXT_TOOL_RETRY_TIMEOUT_SEC",
            default=max(60.0, self.tool_a_timeout_sec * 2.0, self.timeout_sec * 2.5),
        )
        self.max_retries = self._read_int_env("SWE_AGENT_EXT_TOOL_MAX_RETRIES", default=1, minimum=0)
        self.enable_stale_fallback = self._read_bool_env("SWE_AGENT_EXT_TOOL_ENABLE_STALE_FALLBACK", default=True)
        self.enable_adaptive_timeout = self._read_bool_env("SWE_AGENT_EXT_TOOL_ADAPTIVE_TIMEOUT", default=True)
        self.tool_circuit_threshold = self._read_int_env("SWE_AGENT_EXT_TOOL_CIRCUIT_THRESHOLD", default=8, minimum=1)
        self.tool_circuit_cooldown_sec = self._read_float_env("SWE_AGENT_EXT_TOOL_CIRCUIT_COOLDOWN_SEC", default=30.0)
        self.inject_enabled = self._read_bool_env("SWE_AGENT_MEM_INJECT_ENABLED", default=True)
        # Soft cap: 0 means "no fixed count cap" and selection is driven by budget/quality.
        self.max_hints = self._read_int_env("SWE_AGENT_MEM_MAX_HINTS", default=0, minimum=0)
        self.min_confidence = self._read_float_env("SWE_AGENT_MEM_MIN_CONFIDENCE", default=0.6)
        self.min_item_confidence = self._read_float_env("SWE_AGENT_MEM_MIN_ITEM_CONFIDENCE", default=0.0)
        self.max_hint_chars = self._read_int_env("SWE_AGENT_MEM_MAX_HINT_CHARS", default=900, minimum=120)
        self.hint_char_budget = self._read_int_env("SWE_AGENT_MEM_HINT_CHAR_BUDGET", default=2400, minimum=480)
        self.family_cooldown_steps = self._read_int_env("SWE_AGENT_MEM_FAMILY_COOLDOWN_STEPS", default=4, minimum=0)
        self.max_hints_per_family_per_attempt = self._read_int_env(
            "SWE_AGENT_MEM_MAX_HINTS_PER_FAMILY_PER_ATTEMPT",
            default=1,
            minimum=0,
        )
        self.intervention_window_steps = self._read_int_env("SWE_AGENT_MEM_INTERVENTION_WINDOW_STEPS", default=5, minimum=1)
        self.require_trigger_for_injection = self._read_bool_env(
            "SWE_AGENT_MEM_REQUIRE_TRIGGER_FOR_INJECTION",
            default=False,
        )
        self.gate_min_external_ratio = self._read_float_env("SWE_AGENT_MEM_GATE_MIN_EXTERNAL_RATIO", default=0.35)
        self.gate_min_buffer_ratio = self._read_float_env("SWE_AGENT_MEM_GATE_MIN_BUFFER_RATIO", default=0.15)
        self.gate_min_action_error_coverage = self._read_float_env(
            "SWE_AGENT_MEM_GATE_MIN_ACTION_ERROR_COVERAGE",
            default=0.60,
        )
        self.gate_hard_fail = self._read_bool_env("SWE_AGENT_MEM_GATE_HARD_FAIL", default=False)
        self._pending_memory_hints: deque[dict[str, Any]] = deque()
        self._recent_hint_fingerprints: set[str] = set()
        self._hint_family_last_injected_step: dict[str, int] = {}
        self._hint_family_injected_count: dict[str, int] = {}
        self._recent_triggers: deque[str] = deque(maxlen=16)
        self._seen_files: set[str] = set()
        self._ad_hoc_script_names: set[str] = set()
        self._closure_active = False
        self._blocked_action_patterns: set[str] = set()
        self._active_subproblem_type = ""
        self._active_strategy_label = ""
        self._active_interventions: list[dict[str, Any]] = []
        self._consecutive_failures = 0
        self._step_index = 0
        self._event_seq = 0
        self._agent_name = "unknown"
        self._instance_id = "unknown"
        self._attempt_id = os.getenv("SWE_AGENT_EXT_ATTEMPT_ID", "attempt-0").strip() or "attempt-0"
        self._run_id = os.getenv("SWE_AGENT_EXT_RUN_ID", "").strip()
        self._adaptive_timeout_by_tool: dict[str, float] = {}
        self._tool_failure_streak: dict[str, int] = {}
        self._tool_circuit_open_until: dict[str, float] = {}
        self._last_success_payload_by_event: dict[str, dict[str, Any]] = {}
        self._active_step_trace_id = ""
        self._v21_enable_success_fact_hotpath = self._read_bool_env(
            "AGENT_MEM_V21_ENABLE_SUCCESS_FACT_HOTPATH",
            default=False,
        )
        self._v21_enable_sidecar = self._read_bool_env("AGENT_MEM_V21_ENABLE_SIDECAR", default=False)
        sidecar_dir = os.getenv("AGENT_MEM_V21_SIDECAR_DIR", "").strip()
        if not sidecar_dir:
            sidecar_dir = os.path.join(".", "agent_mem_data", "sidecar")
        self._v21_sidecar_dir = sidecar_dir
        self._episode_ledger_store = None
        if self._v21_enable_sidecar and EpisodeLedgerStore is not None:
            try:
                self._episode_ledger_store = EpisodeLedgerStore(self._v21_sidecar_dir)
            except Exception:
                self._episode_ledger_store = None
        # Step budget state for deterministic timeout governance
        self._step_budget_state: dict[str, int] = {
            "steps_since_last_edit": 0,
            "steps_since_last_patch": 0,
            "total_steps": 0,
            "extra_repro_budget": 0,
        }
        self._patch_candidate_exists = False
        self._step_budget_guardrail_steps = self._read_int_env(
            "AGENT_MEM_STEP_GUARDRAIL_NO_EDIT", default=15, minimum=5
        )
        self._step_budget_strong_steps = self._read_int_env(
            "AGENT_MEM_STEP_STRONG_NO_PATCH", default=25, minimum=10
        )
        self._step_budget_hard_stop = self._read_int_env(
            "AGENT_MEM_STEP_HARD_STOP_TOTAL", default=50, minimum=20
        )
        self._metrics: dict[str, int] = {
            "plan_generated": 0,
            "action_error_events": 0,
            "error_like_observation": 0,
            "external_tool_response": 0,
            "memory_hint_buffered": 0,
            "memory_injected": 0,
            "tool_timeout_events": 0,
            "success_fact_hotpath_written": 0,
            "success_fact_hotpath_skipped": 0,
            "v2_gate_allow": 0,
            "v2_gate_rewrite": 0,
            "v2_gate_force_reuse": 0,
            "v2_l3_force_submit": 0,
            "v2_l3_force_reuse": 0,
            "v2_card_demoted": 0,
            "v2_card_reinforced": 0,
        }
        self._v2_gate_mode = (os.getenv("AGENT_MEM_PATCH_CONSISTENCY_GATE", "off") or "off").strip().lower()
        self._v2_reuse_mode_env = (os.getenv("AGENT_MEM_REUSE_EXPLORE", "off") or "off").strip().lower()
        self._v2_l3_mode = (os.getenv("AGENT_MEM_L3_FORCE_SUBMIT", "off") or "off").strip().lower()
        self._v2_local_effective_feedback = self._read_bool_env(
            "AGENT_MEM_LOCAL_EFFECTIVE_FEEDBACK", default=False
        )
        self._v2_force_strategy = (os.getenv("AGENT_MEM_FORCE_STRATEGY", "auto") or "auto").strip().lower()
        self._v2_active_invariants: list[dict[str, Any]] = []
        self._v2_active_anti_patterns: list[dict[str, Any]] = []
        # Gate cooldown: after one pre_check inspection, allow the very next submit through
        # to avoid infinite rewrite loops when the model calls submit repeatedly.
        self._v2_gate_pending_pass: bool = False
        # Lazily initialize the reuse-explore scheduler.
        self._v2_strategy_decided: bool = False
        self._v2_strategy: str = "explore_fresh"
        self._v2_p_reuse: float = 0.0
        self._v2_injected_card_stats: dict[str, dict[str, int]] = {}
        self._v2_latest_step: Any = None

        # T1-A: Memory Reformulation Agent
        self._t1a_enabled: bool = self._read_bool_env("SWE_AGENT_T1A_ENABLED", default=False)
        self._t1a_max_reformats_per_attempt: int = self._read_int_env(
            "SWE_AGENT_T1A_MAX_REFORMATS_PER_ATTEMPT", default=0, minimum=0
        )
        self._t1a_reformat_count: int = 0
        self._reformulation_agent: Any = None
        if self._t1a_enabled and ReformulationAgent is not None:
            try:
                self._reformulation_agent = ReformulationAgent.from_env()
            except Exception:
                self._t1a_enabled = False
        self._trajectory_window: deque = deque(maxlen=5)

        # T1-B: Async Interim Memory Mining
        self._t1b_enabled: bool = self._read_bool_env("SWE_AGENT_T1B_ENABLED", default=False)
        self._interim_cache: Any = None
        if self._t1b_enabled and InterimCache is not None:
            try:
                self._interim_cache = InterimCache.from_env()
            except Exception:
                self._t1b_enabled = False
        self._t1b_localize_threshold: int = self._read_int_env(
            "SWE_AGENT_T1B_LOCALIZE_THRESHOLD", default=3, minimum=1
        )
        self._t1b_localization_hits: int = 0
        self._t1b_first_pass_written: bool = False
        self._t1b_first_edit_written: bool = False
        self._t1b_last_localization: list = []

        # T1-C: Pre-submit Critic Agent
        self._t1c_enabled: bool = self._read_bool_env("SWE_AGENT_T1C_ENABLED", default=False)
        self._t1c_use_precheck_diff: bool = self._read_bool_env(
            "SWE_AGENT_T1C_USE_PRECHECK_DIFF", default=False
        )
        self._t1c_split_fallback_approve: bool = self._read_bool_env(
            "SWE_AGENT_T1C_SPLIT_FALLBACK_APPROVE", default=True
        )
        self._t1c_deterministic_guard: bool = self._read_bool_env(
            "SWE_AGENT_T1C_DETERMINISTIC_GUARD", default=False
        )
        self._t1c_revise_duplicate_precheck: bool = self._read_bool_env(
            "SWE_AGENT_T1C_REVISE_DUPLICATE_PRECHECK", default=False
        )
        self._t1c_unavailable_policy: str = (
            os.getenv("SWE_AGENT_T1C_UNAVAILABLE_POLICY", "allow") or "allow"
        ).strip().lower()
        self._critic_agent: Any = None
        if self._t1c_enabled and CriticAgent is not None:
            try:
                self._critic_agent = CriticAgent.from_env()
            except Exception:
                self._t1c_enabled = False
        self._t1c_recent_test_outputs: deque = deque(maxlen=3)
        self._t1c_last_precheck_diff: str = ""
        self._t1c_seen_patch_hashes: dict[str, int] = {}
        self._t1c_unavailable_count: int = 0
        self._t1c_revision_pending: bool = False
        self._metrics.update({
            "t1a_reformat_called": 0,
            "t1a_reformat_failed": 0,
            "t1b_interim_written": 0,
            "t1c_approve": 0,
            "t1c_revise": 0,
            "t1c_reject": 0,
            "t1c_unavailable": 0,
            "t1c_deterministic_revise": 0,
            "t1c_deterministic_reject": 0,
        })

        self._logger = get_logger("swea-ext-bridge", emoji="")

    def on_init(self, *, agent):
        self._agent_name = getattr(agent, "name", "unknown")
        self._run_id = os.getenv("SWE_AGENT_EXT_RUN_ID", "").strip()
        self._attempt_id = os.getenv("SWE_AGENT_EXT_ATTEMPT_ID", self._attempt_id).strip() or self._attempt_id
        env_instance_id = os.getenv("SWE_AGENT_EXT_INSTANCE_ID", "").strip()
        if env_instance_id:
            self._instance_id = env_instance_id
            return
        for attr_name in ("_problem_statement", "problem_statement"):
            try:
                ps = getattr(agent, attr_name, None)
                inst = getattr(ps, "id", None)
                if isinstance(inst, str) and inst.strip():
                    self._instance_id = inst.strip()
                    return
            except Exception:
                continue
        self._instance_id = self._agent_name
        # T1-B keeps interim memories in an attempt-local pending buffer.
        self._t1b_load_interim_hints()

    def _t1b_load_interim_hints(self) -> None:
        """T1-B: Read interim cards from previous attempts and buffer them."""
        if not self._t1b_enabled or self._interim_cache is None:
            return
        try:
            items = self._interim_cache.build_hint_items(self._instance_id)
            for item in items:
                self._pending_memory_hints.append(item)
            if items:
                self._metric_inc("memory_hint_buffered", len(items))
                append_json_log(
                    "HOOK",
                    {
                        "version": "v1",
                        "event": "t1b_interim_hints_loaded",
                        "instance_id": self._instance_id,
                        "count": len(items),
                    },
                )
        except Exception:
            pass

    @staticmethod
    def _read_bool_env(name: str, *, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _read_int_env(name: str, *, default: int, minimum: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return max(minimum, int(raw.strip()))
        except Exception:
            return default

    @staticmethod
    def _read_float_env(name: str, *, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return float(raw.strip())
        except Exception:
            return default

    @staticmethod
    def _try_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    def _extract_json_payload(self, text: str) -> dict[str, Any] | None:
        content = text.strip()
        if not content:
            return None
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

        left = content.find("{")
        right = content.rfind("}")
        if left < 0 or right <= left:
            return None
        try:
            parsed = json.loads(content[left : right + 1])
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def _hint_from_item(self, item: Any) -> str | None:
        if isinstance(item, str):
            text = item.strip()
            return text[:280] if text else None
        if not isinstance(item, dict):
            return None

        for key in ("recommendation", "repair_action", "next_step_fix", "recommend", "summary"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()[:280]

        if item.get("pattern") == "failure_followed_by_repair":
            val = item.get("repair_action")
            if isinstance(val, str) and val.strip():
                return val.strip()[:280]
        return None

    @staticmethod
    def _memory_id_from_item(item: Any) -> str | None:
        if not isinstance(item, dict):
            return None
        for key in ("summary_id", "card_id", "experience_id", "belief_id", "pattern_id", "rule_id", "action_id"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return None

    def _resolve_confidence(self, payload: dict[str, Any], items: list[Any]) -> float:
        candidates: list[float] = []
        top = self._try_float(payload.get("confidence"))
        if top is not None:
            candidates.append(top)
        for item in items:
            if isinstance(item, dict):
                v = self._try_float(item.get("confidence"))
                if v is not None:
                    candidates.append(v)
        return max(candidates) if candidates else 0.0

    def _collect_hint_items(self, payload: dict[str, Any]) -> list[Any]:
        event_name = payload.get("event_handled")
        items: list[Any] = []
        if event_name == "plan_generated":
            hints = payload.get("planning_tips", [])
            if isinstance(hints, list):
                items.extend(hints)
        elif event_name == "action_error":
            hints = payload.get("repair_suggestions", [])
            if isinstance(hints, list):
                items.extend(hints)
            next_fix = payload.get("next_step_fix")
            if isinstance(next_fix, str) and next_fix.strip():
                items.append(next_fix.strip())
        recs = payload.get("recommendations")
        if isinstance(recs, list):
            items.extend(recs)
        return items

    @staticmethod
    def _hint_family_from_item(item: Any) -> str | None:
        if not isinstance(item, dict):
            return None
        for key in ("family_id",):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        rtype = str(item.get("type", "")).strip().lower()
        if rtype:
            if rtype == "failure_card_v2":
                err = item.get("error_signature") if isinstance(item.get("error_signature"), dict) else {}
                err_type = str(err.get("error_type", "")).strip().lower()
                rec = str(item.get("recommendation", "")).strip().lower()[:80]
                family = f"failure_card:{err_type}:{rec}"
                return family if family.strip(":") else None
            if rtype == "belief_tip":
                belief_type = str(item.get("belief_type", "")).strip().lower()
                rec = str(item.get("recommendation", "")).strip().lower()[:80]
                family = f"belief_tip:{belief_type}:{rec}"
                return family if family.strip(":") else None
            rec = str(
                item.get("recommendation")
                or item.get("repair_action")
                or item.get("summary")
                or ""
            ).strip().lower()[:80]
            family = f"{rtype}:{rec}"
            return family if family.strip(":") else None
        return None

    @staticmethod
    def _hint_selection_score(item: Any, *, batch_confidence: float) -> float:
        if not isinstance(item, dict):
            return batch_confidence
        selection = ExternalToolBridgeHook._try_float(item.get("selection_score")) or 0.0
        helpfulness = ExternalToolBridgeHook._try_float(item.get("helpfulness_score")) or 0.0
        novelty = ExternalToolBridgeHook._try_float(item.get("novelty_score")) or 0.0
        item_conf = ExternalToolBridgeHook._try_float(item.get("confidence")) or 0.0
        return max(selection, 0.45 * helpfulness + 0.25 * novelty + 0.20 * item_conf + 0.10 * batch_confidence)

    def _buffer_memory_hints(self, payload: dict[str, Any]) -> int:
        items = self._collect_hint_items(payload)
        if not items:
            return 0

        batch_confidence = self._resolve_confidence(payload, items)
        if batch_confidence < self.min_confidence:
            return 0

        added = 0
        for item in items:
            item_conf = None
            if isinstance(item, dict):
                item_conf = self._try_float(item.get("confidence"))
            if item_conf is not None and item_conf < self.min_item_confidence:
                continue
            hint = self._hint_from_item(item)
            if not hint:
                continue
            family_id = self._hint_family_from_item(item)
            if family_id and family_id.startswith("workflow_step:") and self._step_index > 3:
                continue
            card_type = item.get("card_type") if isinstance(item, dict) else None
            is_semantic_card = card_type in {"BugInvariantCard", "SuccessPathCard", "BugAntiPatternCard"}
            if not is_semantic_card and family_id and self._hint_family_injected_count.get(family_id, 0) >= 3:
                continue
            key = hint.lower()
            if key in self._recent_hint_fingerprints:
                continue
            self._pending_memory_hints.append(
                {
                    "hint": hint,
                    "memory_id": self._memory_id_from_item(item),
                    "summary_id": item.get("summary_id") if isinstance(item, dict) else None,
                    "family_id": family_id,
                    "source_event": payload.get("event_handled"),
                    "source_event_id": payload.get("trace_id"),
                    "trace_id": payload.get("trace_id"),
                    "item_confidence": item_conf,
                    "batch_confidence": self._try_float(payload.get("confidence")),
                    "selection_score": self._hint_selection_score(item, batch_confidence=batch_confidence),
                    "type": item.get("type") if isinstance(item, dict) else None,
                    "card_type": item.get("card_type") if isinstance(item, dict) else None,
                    "normalized_pattern_type": item.get("normalized_pattern_type") if isinstance(item, dict) else None,
                    "subproblem_type": item.get("subproblem_type") if isinstance(item, dict) else None,
                    "strategy_label": item.get("strategy_label") if isinstance(item, dict) else None,
                    "prefer_actions": list(item.get("prefer_actions", []))
                    if isinstance(item, dict) and isinstance(item.get("prefer_actions"), list)
                    else [],
                    "avoid_actions": list(item.get("avoid_actions", []))
                    if isinstance(item, dict) and isinstance(item.get("avoid_actions"), list)
                    else [],
                    "budget_hints": dict(item.get("budget_hints", {}))
                    if isinstance(item, dict) and isinstance(item.get("budget_hints"), dict)
                    else {},
                    "governance_hardness": item.get("governance_hardness") if isinstance(item, dict) else None,
                    "minimal_patch_signature": item.get("minimal_patch_signature")
                    if isinstance(item, dict)
                    else None,
                    "failed_patch_signature": item.get("failed_patch_signature")
                    if isinstance(item, dict)
                    else None,
                    "signature_hash": item.get("signature_hash")
                    if isinstance(item, dict)
                    else None,
                    "card_id": item.get("card_id")
                    if isinstance(item, dict)
                    else None,
                    "support_count": item.get("support_count")
                    if isinstance(item, dict)
                    else None,
                }
            )
            if isinstance(item, dict):
                if card_type == "BugInvariantCard":
                    self._v2_register_invariant(item)
                elif card_type == "BugAntiPatternCard":
                    self._v2_register_anti_pattern(item)
            self._recent_hint_fingerprints.add(key)
            added += 1

        soft_cap = self.max_hints if self.max_hints > 0 else 25
        max_pending = max(20, soft_cap * 8)
        while len(self._pending_memory_hints) > max_pending:
            self._pending_memory_hints.popleft()

        if len(self._recent_hint_fingerprints) > 2000:
            # Keep dedup state bounded in long runs.
            self._recent_hint_fingerprints = {
                str(h.get("hint", "")).lower()
                for h in list(self._pending_memory_hints)
                if str(h.get("hint", "")).strip()
            }
        if added > 0:
            self._metric_inc("memory_hint_buffered", added)
        return added

    def _append_tool_response_summary(self, *, tool_name: str, payload: dict[str, Any]) -> None:
        event_name = str(payload.get("event_handled", "unknown"))
        retrieval_debug = payload.get("retrieval_debug")
        summary: dict[str, Any] = {
            "version": "v1",
            "event": "external_tool_response",
            "agent": self._agent_name,
            "instance_id": self._instance_id,
            "attempt_id": self._attempt_id,
            "source_tool": tool_name,
            "source_event": event_name,
            "trace_id": payload.get("trace_id"),
            "confidence": payload.get("confidence"),
            "step_index": self._step_index,
        }
        if isinstance(retrieval_debug, dict):
            summary["retrieval_debug"] = {
                "query_type": retrieval_debug.get("query_type"),
                "embedding_view": retrieval_debug.get("embedding_view"),
                "candidate_count_before_filter": retrieval_debug.get("candidate_count_before_filter", 0),
                "candidate_count_after_filter": retrieval_debug.get("candidate_count_after_filter", 0),
                "candidate_task_count": retrieval_debug.get("candidate_task_count", 0),
                "selected_subgraph_count": retrieval_debug.get("selected_subgraph_count", 0),
                "recommendation_count": retrieval_debug.get("recommendation_count", 0),
                "injection_candidate_count": retrieval_debug.get("injection_candidate_count", 0),
                "injection_selected_count": retrieval_debug.get("injection_selected_count", 0),
            }

        if event_name == "plan_generated":
            tips = payload.get("planning_tips", [])
            summary["suggestion_count"] = len(tips) if isinstance(tips, list) else 0
        elif event_name == "action_error":
            recs = payload.get("repair_suggestions", [])
            summary["suggestion_count"] = len(recs) if isinstance(recs, list) else 0
            summary["error_type"] = payload.get("error_type")
        elif event_name == "run_done":
            summary["task_id"] = payload.get("task_id")
            summary["cleared"] = bool(payload.get("cleared", False))
        self._metric_inc("external_tool_response", 1)
        append_json_log("HOOK", summary)

    def _record_trigger(self, trigger: str, detail: str = "") -> None:
        marker = trigger if not detail else f"{trigger}:{detail}"
        self._recent_triggers.append(marker)
        if trigger == "over_exploration":
            self._closure_active = True
        append_json_log(
            "HOOK",
            {
                "version": "v1",
                "event": "proactive_trigger",
                "agent": self._agent_name,
                "instance_id": self._instance_id,
                "attempt_id": self._attempt_id,
                "trigger": trigger,
                "detail": detail,
                "step_index": self._step_index,
            },
        )

    def _has_recent_trigger(self, prefix: str) -> bool:
        return any(str(marker).startswith(prefix) for marker in self._recent_triggers)

    def _runtime_guard_payload(self) -> dict[str, Any]:
        blocked_families: list[str] = []
        if self._closure_active:
            blocked_families.extend(["workflow_step", "planning_loop"])
        return {
            "closure_active": self._closure_active,
            "ad_hoc_script_count": len(self._ad_hoc_script_names),
            "recent_triggers": list(self._recent_triggers)[-8:],
            "blocked_families": blocked_families,
            "blocked_action_patterns": sorted(self._blocked_action_patterns),
            "active_subproblem_type": self._active_subproblem_type,
            "active_strategy_label": self._active_strategy_label,
        }

    def _apply_runtime_guard_from_payload(self, payload: dict[str, Any]) -> None:
        for item in self._collect_hint_items(payload):
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).strip().lower()
            card_type = str(item.get("card_type", "")).strip()
            hardness = str(item.get("governance_hardness") or "").strip().lower()
            should_bind_context = item_type == "attempt_summary_v1" or (
                item_type in {"compiler_card", "abstract_pattern"}
                and (
                    bool(item.get("prefer_actions") or item.get("avoid_actions"))
                    or bool(item.get("subproblem_type") or item.get("strategy_label"))
                )
            )
            if not should_bind_context:
                continue
            subproblem_type = str(item.get("subproblem_type") or "").strip().lower()
            strategy_label = str(item.get("strategy_label") or "").strip().lower()
            if subproblem_type:
                self._active_subproblem_type = subproblem_type
            if strategy_label:
                self._active_strategy_label = strategy_label
            should_block_actions = item_type == "attempt_summary_v1" or (
                item_type in {"compiler_card", "abstract_pattern"}
                and (
                    card_type in {"TimeoutGovernanceCard", "ClosureGuardCard"}
                    or hardness in {"guardrail", "hard_stop"}
                )
            )
            if not should_block_actions:
                continue
            for raw in item.get("avoid_actions") or []:
                pattern = str(raw).strip().lower()
                if pattern:
                    self._blocked_action_patterns.add(pattern)

    def _enqueue_system_hint(
        self,
        *,
        hint: str,
        family_id: str,
        normalized_pattern_type: str,
        selection_score: float,
    ) -> None:
        key = hint.strip().lower()
        if not key or key in self._recent_hint_fingerprints:
            return
        self._pending_memory_hints.append(
            {
                "hint": hint.strip(),
                "memory_id": None,
                "family_id": family_id,
                "source_event": "hook_generated",
                "source_event_id": None,
                "trace_id": None,
                "item_confidence": 0.9,
                "batch_confidence": 0.9,
                "selection_score": selection_score,
                "type": "abstract_pattern",
                "normalized_pattern_type": normalized_pattern_type,
            }
        )
        self._recent_hint_fingerprints.add(key)

    def _build_trace_id(self, event_name: str) -> str:
        self._event_seq += 1
        run_id = str(self._run_id or "norun")
        instance_id = self._instance_id or self._agent_name or "unknown"
        return f"{run_id}:{instance_id}:{self._step_index}:{event_name}:{self._event_seq}"

    def _build_event_payload(self, event_name: str, **extra: Any) -> dict[str, Any]:
        payload = {
            "version": "v1",
            "event": event_name,
            "event_type": event_name,
            "source": "bridge_hook",
            "agent": self._agent_name,
            "instance_id": self._instance_id,
            "run_id": self._run_id,
            "attempt_id": self._attempt_id,
            "step_index": self._step_index,
            "timestamp": time.time(),
        }
        payload.update(extra)
        return payload

    def _metric_inc(self, key: str, amount: int = 1) -> None:
        self._metrics[key] = self._metrics.get(key, 0) + max(0, amount)

    @staticmethod
    def _is_error_like_observation(text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        patterns = (
            "error",
            "exception",
            "traceback",
            "assert",
            "failed",
            "no such file",
            "timed out",
            "non-zero",
        )
        return any(p in lowered for p in patterns)

    def _is_circuit_open(self, tool_name: str) -> bool:
        now = time.monotonic()
        until = float(self._tool_circuit_open_until.get(tool_name, 0.0))
        if until <= 0:
            return False
        if now >= until:
            self._tool_circuit_open_until.pop(tool_name, None)
            self._tool_failure_streak[tool_name] = 0
            return False
        return True

    def _record_tool_failure(self, tool_name: str) -> None:
        streak = self._tool_failure_streak.get(tool_name, 0) + 1
        self._tool_failure_streak[tool_name] = streak
        if streak >= self.tool_circuit_threshold:
            self._tool_circuit_open_until[tool_name] = time.monotonic() + max(1.0, self.tool_circuit_cooldown_sec)

    def _record_tool_success(self, tool_name: str) -> None:
        self._tool_failure_streak[tool_name] = 0
        self._tool_circuit_open_until.pop(tool_name, None)

    def _emit_tool_result(self, *, tool_name: str, payload: dict[str, Any], status: str, detail: str = "") -> None:
        append_json_log(
            "HOOK",
            {
                "version": "v1",
                "event": "external_tool_result",
                "agent": self._agent_name,
                "instance_id": self._instance_id,
                "attempt_id": self._attempt_id,
                "source_tool": tool_name,
                "source_event": payload.get("event"),
                "trace_id": payload.get("trace_id"),
                "step_index": self._step_index,
                "status": status,
                "detail": detail,
                "failure_streak": self._tool_failure_streak.get(tool_name, 0),
            },
        )

    def _default_timeout_for_tool(self, tool_name: str) -> float:
        if tool_name == "tool_a":
            return self.tool_a_timeout_sec
        if tool_name == "tool_b":
            return self.tool_b_timeout_sec
        return self.timeout_sec

    def _effective_timeout_for_tool(self, tool_name: str) -> float:
        timeout = self._adaptive_timeout_by_tool.get(tool_name)
        if timeout is not None:
            return timeout
        return self._default_timeout_for_tool(tool_name)

    def _handle_parsed_tool_payload(self, *, tool_name: str, parsed: dict[str, Any]) -> None:
        event_name = str(parsed.get("event_handled", "")).strip().lower()
        if event_name:
            self._last_success_payload_by_event[event_name] = parsed

        self._append_tool_response_summary(tool_name=tool_name, payload=parsed)
        if not self.inject_enabled:
            return
        self._apply_runtime_guard_from_payload(parsed)
        buffered = self._buffer_memory_hints(parsed)
        if buffered <= 0:
            return
        self._logger.info(
            "Buffered %d memory hints from %s (pending=%d, min_conf=%.2f)",
            buffered,
            tool_name,
            len(self._pending_memory_hints),
            self.min_confidence,
        )
        append_json_log(
            "HOOK",
            {
                "version": "v1",
                "event": "memory_hint_buffered",
                "agent": self._agent_name,
                "instance_id": self._instance_id,
                "attempt_id": self._attempt_id,
                "source_event": parsed.get("event_handled"),
                "source_tool": tool_name,
                "buffered_count": buffered,
                "pending_count": len(self._pending_memory_hints),
                "confidence": parsed.get("confidence"),
                "buffered_memory_ids": [
                    str(item.get("memory_id"))
                    for item in list(self._pending_memory_hints)[-buffered:]
                    if str(item.get("memory_id", "")).strip()
                ],
                "buffered_family_ids": [
                    str(item.get("family_id"))
                    for item in list(self._pending_memory_hints)[-buffered:]
                    if str(item.get("family_id", "")).strip()
                ],
            },
        )

    def _emit_timeout_event(self, *, tool_name: str, payload: dict[str, Any], timeout: float, attempt: int) -> None:
        self._metric_inc("tool_timeout_events", 1)
        append_json_log(
            "HOOK",
            {
                "version": "v1",
                "event": "external_tool_timeout",
                "agent": self._agent_name,
                "instance_id": self._instance_id,
                "attempt_id": self._attempt_id,
                "source_tool": tool_name,
                "source_event": payload.get("event"),
                "trace_id": payload.get("trace_id"),
                "step_index": self._step_index,
                "attempt": attempt,
                "timeout_sec": timeout,
            },
        )

    def _try_stale_fallback(self, *, tool_name: str, payload: dict[str, Any]) -> bool:
        if not self.enable_stale_fallback:
            return False
        source_event = str(payload.get("event", "")).strip().lower()
        if not source_event:
            return False
        stale = self._last_success_payload_by_event.get(source_event)
        if not stale:
            return False
        stale_payload = dict(stale)
        stale_payload["stale_fallback"] = True
        self._logger.warning(
            "External %s fallback to stale response for event=%s trace=%s",
            tool_name,
            source_event,
            payload.get("trace_id"),
        )
        self._handle_parsed_tool_payload(tool_name=f"{tool_name}_stale", parsed=stale_payload)
        append_json_log(
            "HOOK",
            {
                "version": "v1",
                "event": "external_tool_stale_fallback",
                "agent": self._agent_name,
                "instance_id": self._instance_id,
                "attempt_id": self._attempt_id,
                "source_tool": tool_name,
                "source_event": source_event,
                "trace_id": payload.get("trace_id"),
                "step_index": self._step_index,
            },
        )
        self._emit_tool_result(tool_name=tool_name, payload=payload, status="stale_fallback")
        return True

    def _scan_proactive_triggers(self, action_text: str) -> None:
        action_l = action_text.lower()
        if re.search(r"\b(pytest|unittest|tox|nox|nose|python\s+-m\s+pytest|make\s+test)\b", action_l):
            self._record_trigger("before_test_run")
        if re.search(r"\b(submit|submission|create[_\s-]?pr|open[_\s-]?pr|final answer|final patch)\b", action_l):
            self._record_trigger("before_submit")

        for m in re.finditer(
            r"\b[\w.\-]+\.(?:py|js|ts|java|go|rs|cpp|h|md|txt|json|yaml|yml|toml|ini|cfg|sh)\b",
            action_text,
            flags=re.IGNORECASE,
        ):
            fn = m.group(0).lower()
            if fn not in self._seen_files:
                self._seen_files.add(fn)
                self._record_trigger("first_file_access", detail=fn)
            if re.match(r"(?:test_|reproduce_|debug_).+\.py$", fn):
                self._ad_hoc_script_names.add(fn)
                if len(self._ad_hoc_script_names) >= 2:
                    self._record_trigger("over_exploration", detail=str(len(self._ad_hoc_script_names)))
                    self._enqueue_system_hint(
                        hint="Stop expanding ad-hoc scripts. Validate the minimal fix against the target failing path, then either submit or terminate.",
                        family_id="closure_signal:over_exploration_after_key_signal",
                        normalized_pattern_type="closure_signal",
                        selection_score=0.98,
                    )

    @staticmethod
    def _extract_ad_hoc_script_names(action_text: str) -> list[str]:
        names: list[str] = []
        for m in re.finditer(r"\b(?:test_|reproduce_|debug_)[\w.\-]+\.py\b", action_text, flags=re.IGNORECASE):
            names.append(m.group(0).lower())
        return names

    def _is_new_ad_hoc_script_creation(self, action_text: str) -> bool:
        if not self._closure_active:
            return False
        lowered = action_text.lower()
        if not any(tok in lowered for tok in ("cat ", "tee ", "touch ", "printf ", "echo ", "apply_patch")):
            return False
        for name in self._extract_ad_hoc_script_names(action_text):
            if name not in self._ad_hoc_script_names:
                return True
        return False

    def _blocked_pattern_for_action(self, action_text: str) -> str | None:
        lowered = action_text.lower()
        patterns = self._blocked_action_patterns
        if "create_new_repro_script_after_repro_confirmed" in patterns:
            if any(tok in lowered for tok in ("cat ", "tee ", "touch ", "printf ", "echo ", "apply_patch")):
                if any(name in lowered for name in ("reproduce_", "test_", "debug_")):
                    return "create_new_repro_script_after_repro_confirmed"
        if "expand_search_to_unrelated_module_after_localization" in patterns:
            if any(
                token in lowered
                for token in (
                    "utils.py",
                    "wcsaxes",
                    "visualization/wcsaxes",
                    "grep -r",
                    "rg -n",
                    "find /testbed",
                    "ls /testbed/astropy/visualization",
                )
            ):
                return "expand_search_to_unrelated_module_after_localization"
        if "expand_patch_scope_before_minimal_fix_is_tested" in patterns:
            if any(
                token in lowered
                for token in (
                    "str_replace_editor insert",
                    "str_replace_editor replace",
                    "apply_patch",
                    "cat >",
                    "tee ",
                )
            ) and any(
                token in lowered
                for token in (
                    "utils.py",
                    "wcsaxes",
                    "visualization/wcsaxes",
                    "conftest.py",
                )
            ):
                return "expand_patch_scope_before_minimal_fix_is_tested"
        if "run_broad_regression_before_patch_candidate_exists" in patterns:
            if any(token in lowered for token in ("pytest -q", "python -m pytest", "tox", "nox")) and any(
                token in lowered for token in ("all", "full", "regression")
            ):
                return "run_broad_regression_before_patch_candidate_exists"
        if "submit_without_target_validation" in patterns:
            if any(token in lowered for token in ("submit", "exit", "done", "finalize")):
                return "submit_without_target_validation"
        return None

    def _soft_block_ad_hoc_script_creation(self, step: StepOutput) -> None:
        original_action = step.action
        step.action = (
            "printf '%s\\n' "
            "'Runtime guard blocked new ad-hoc script creation after closure. "
            "Validate the current minimal fix or stop.'"
        )
        append_json_log(
            "HOOK",
            {
                "version": "v1",
                "event": "runtime_guard_block",
                "agent": self._agent_name,
                "instance_id": self._instance_id,
                "attempt_id": self._attempt_id,
                "step_index": self._step_index + 1,
                "block_type": "new_ad_hoc_script_after_closure",
                "original_action": original_action,
                "replacement_action": step.action,
                "runtime_guard": self._runtime_guard_payload(),
            },
        )

    def _soft_block_runtime_guard_pattern(self, step: StepOutput, pattern: str) -> None:
        original_action = step.action
        step.action = (
            "printf '%s\\n' "
            "'Runtime guard blocked an action pattern discouraged by previous failed attempts. "
            "Validate the current minimal fix or follow the recorded next-best action.'"
        )
        append_json_log(
            "HOOK",
            {
                "version": "v1",
                "event": "runtime_guard_block",
                "agent": self._agent_name,
                "instance_id": self._instance_id,
                "attempt_id": self._attempt_id,
                "step_index": self._step_index + 1,
                "block_type": pattern,
                "original_action": original_action,
                "replacement_action": step.action,
                "runtime_guard": self._runtime_guard_payload(),
            },
        )

    def _run_cmd(self, *, cmd: str | None, tool_name: str, payload: dict[str, Any]) -> None:
        if not cmd:
            return
        if self._is_circuit_open(tool_name):
            self._emit_tool_result(
                tool_name=tool_name,
                payload=payload,
                status="circuit_open",
                detail=f"cooldown_sec={self.tool_circuit_cooldown_sec}",
            )
            self._try_stale_fallback(tool_name=tool_name, payload=payload)
            return
        env = os.environ.copy()
        env["SWE_AGENT_EXT_EVENT_JSON"] = json.dumps(payload, ensure_ascii=False)
        attempts = max(1, self.max_retries + 1)
        success = False
        last_failure = "unknown"
        timeout_base = self._effective_timeout_for_tool(tool_name)

        _overhead_file = os.environ.get("SWEAGENT_BRIDGE_OVERHEAD_FILE", "").strip()

        for attempt in range(1, attempts + 1):
            timeout_for_attempt = timeout_base if attempt == 1 else max(timeout_base, self.retry_timeout_sec)
            _t0_overhead = time.monotonic()
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    check=False,
                    text=True,
                    capture_output=True,
                    timeout=timeout_for_attempt,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                last_failure = f"timeout@{timeout_for_attempt:.2f}s"
                self._logger.warning(
                    "External %s timed out after %.2fs (attempt %d/%d)",
                    tool_name,
                    timeout_for_attempt,
                    attempt,
                    attempts,
                )
                self._emit_timeout_event(
                    tool_name=tool_name,
                    payload=payload,
                    timeout=timeout_for_attempt,
                    attempt=attempt,
                )
                self._record_tool_failure(tool_name)
                if _overhead_file:
                    try:
                        with open(_overhead_file, "a", encoding="utf-8") as _fh:
                            _fh.write(f"{time.monotonic() - _t0_overhead:.6f}\n")
                    except Exception:
                        pass
                if self.enable_adaptive_timeout and timeout_for_attempt < self.retry_timeout_sec:
                    self._adaptive_timeout_by_tool[tool_name] = self.retry_timeout_sec
                continue
            except Exception as e:
                last_failure = f"exception:{e}"
                self._logger.warning("External %s failed: %s (attempt %d/%d)", tool_name, e, attempt, attempts)
                self._record_tool_failure(tool_name)
                if _overhead_file:
                    try:
                        with open(_overhead_file, "a", encoding="utf-8") as _fh:
                            _fh.write(f"{time.monotonic() - _t0_overhead:.6f}\n")
                    except Exception:
                        pass
                continue

            if _overhead_file:
                try:
                    with open(_overhead_file, "a", encoding="utf-8") as _fh:
                        _fh.write(f"{time.monotonic() - _t0_overhead:.6f}\n")
                except Exception:
                    pass

            parsed_ok = False
            if result.stdout.strip():
                self._logger.info("[%s stdout] %s", tool_name, result.stdout.strip())
                parsed = self._extract_json_payload(result.stdout)
                if parsed:
                    self._handle_parsed_tool_payload(tool_name=tool_name, parsed=parsed)
                    parsed_ok = True
            if result.stderr.strip():
                self._logger.warning("[%s stderr] %s", tool_name, result.stderr.strip())

            if result.returncode == 0:
                success = True
                self._record_tool_success(tool_name)
                self._emit_tool_result(tool_name=tool_name, payload=payload, status="ok")
                break

            last_failure = f"exit_code:{result.returncode}"
            self._logger.warning(
                "External %s exit code=%d (attempt %d/%d)",
                tool_name,
                result.returncode,
                attempt,
                attempts,
            )
            self._record_tool_failure(tool_name)
            if parsed_ok:
                # Accept parsed response even when helper exits non-zero.
                success = True
                self._record_tool_success(tool_name)
                self._emit_tool_result(
                    tool_name=tool_name,
                    payload=payload,
                    status="ok_with_nonzero_exit",
                    detail=f"exit_code={result.returncode}",
                )
                break

        if success:
            return
        if self._try_stale_fallback(tool_name=tool_name, payload=payload):
            return
        self._emit_tool_result(tool_name=tool_name, payload=payload, status="failed", detail=last_failure)
        self._logger.warning("External %s failed after %d attempts (%s)", tool_name, attempts, last_failure)

    @staticmethod
    def _is_code_edit_action(action_text: str) -> bool:
        lowered = action_text.lower()
        return any(
            tok in lowered
            for tok in ("str_replace_editor", "edit_file", "apply_patch", "str_replace ", "create_file", "write_file")
        )

    @staticmethod
    def _is_patch_submission(action_text: str) -> bool:
        lowered = action_text.lower()
        return any(tok in lowered for tok in ("submit", "<submit>", "final answer", "final patch"))

    def _update_step_budget(self, action_text: str) -> None:
        self._step_budget_state["total_steps"] += 1
        self._step_budget_state["steps_since_last_edit"] += 1
        self._step_budget_state["steps_since_last_patch"] += 1
        if self._step_budget_state["extra_repro_budget"] > 0:
            self._step_budget_state["extra_repro_budget"] -= 1
        if self._is_code_edit_action(action_text):
            self._step_budget_state["steps_since_last_edit"] = 0
            self._patch_candidate_exists = True
        if self._is_patch_submission(action_text):
            self._step_budget_state["steps_since_last_patch"] = 0
        if re.search(r"repro\w*_confirm|reproduction.*pass|test.*pass.*repro", action_text.lower()):
            self._step_budget_state["extra_repro_budget"] = max(
                self._step_budget_state["extra_repro_budget"], 5
            )

    def _check_step_budget_triggers(self) -> None:
        steps_no_edit = self._step_budget_state["steps_since_last_edit"]
        steps_no_patch = self._step_budget_state["steps_since_last_patch"]
        total = self._step_budget_state["total_steps"]
        extra_budget = self._step_budget_state["extra_repro_budget"]
        effective_guardrail = self._step_budget_guardrail_steps + extra_budget
        if steps_no_edit >= effective_guardrail and steps_no_edit % 5 == 0:
            self._enqueue_system_hint(
                hint=(
                    f"[StepBudget] {steps_no_edit} steps without code edit. "
                    "You must now generate a patch or submit best candidate. "
                    "Stop exploring and focus on editing the fix location."
                ),
                family_id=f"step_budget:no_edit:{steps_no_edit // 5}",
                normalized_pattern_type="closure_signal",
                selection_score=0.99,
            )
        if steps_no_patch >= self._step_budget_strong_steps and not self._patch_candidate_exists:
            self._blocked_action_patterns.add("run_broad_regression_before_patch_candidate_exists")
            if steps_no_patch % 10 == 0:
                self._enqueue_system_hint(
                    hint=(
                        f"[StepBudget] {steps_no_patch} steps without patch. "
                        "Broad regression blocked. Form a patch candidate before running further tests."
                    ),
                    family_id=f"step_budget:no_patch:{steps_no_patch // 10}",
                    normalized_pattern_type="closure_signal",
                    selection_score=1.0,
                )
        if total >= self._step_budget_hard_stop and not self._patch_candidate_exists:
            self._enqueue_system_hint(
                hint=(
                    f"[StepBudget:HardStop] {total} steps with no patch submitted. "
                    "Submit your best current candidate immediately or exit. "
                    "Further exploration will not help."
                ),
                family_id=f"step_budget:hard_stop:{total // 10}",
                normalized_pattern_type="closure_signal",
                selection_score=1.0,
            )

    def _select_hints_for_injection(self) -> list[dict[str, Any]]:
        if not self._pending_memory_hints:
            return []

        char_budget = max(self.max_hint_chars, self.hint_char_budget)
        selected: list[dict[str, Any]] = []
        kept: list[dict[str, Any]] = []
        used = 0
        family_selected: set[str] = set()
        bucket_selected: dict[str, int] = {}

        ordered_items = sorted(
            list(self._pending_memory_hints),
            key=lambda row: (
                self._hint_bucket_priority(row),
                float(row.get("selection_score", 0.0) or 0.0),
                float(row.get("item_confidence") or row.get("batch_confidence") or 0.0),
            ),
            reverse=True,
        )

        for item in ordered_items:
            hint = str(item.get("hint", "")).strip()
            if not hint:
                continue

            if self.require_trigger_for_injection and not self._recent_triggers:
                confidence = self._try_float(item.get("item_confidence"))
                if confidence is None:
                    confidence = self._try_float(item.get("batch_confidence"))
                if confidence is None or confidence < max(self.min_confidence, 0.7):
                    kept.append(item)
                    continue

            family_id = str(item.get("family_id", "")).strip()
            bucket = self._hint_bucket(item)
            if bucket == "workflow" and (self._step_index > 3 or self._has_recent_trigger("over_exploration")):
                continue
            if family_id:
                if family_id in family_selected:
                    kept.append(item)
                    continue
                if self.max_hints_per_family_per_attempt > 0:
                    injected_count = self._hint_family_injected_count.get(family_id, 0)
                    if injected_count >= self.max_hints_per_family_per_attempt:
                        continue
                if self.family_cooldown_steps > 0:
                    last_step = self._hint_family_last_injected_step.get(family_id)
                    if last_step is not None and (self._step_index - last_step) < self.family_cooldown_steps:
                        kept.append(item)
                        continue

            bucket_limit = 4 if bucket == "semantic" else (1 if bucket in {"closure", "repair", "validation", "workflow"} else 2)
            if bucket_selected.get(bucket, 0) >= bucket_limit:
                kept.append(item)
                continue

            hint_cost = len(hint) + 8
            is_semantic = bucket == "semantic"
            soft_cap_reached = not is_semantic and self.max_hints > 0 and len(selected) >= self.max_hints
            budget_reached = not is_semantic and selected and (used + hint_cost > char_budget)
            if soft_cap_reached or budget_reached:
                kept.append(item)
                continue

            selected.append(item)
            if family_id:
                family_selected.add(family_id)
            bucket_selected[bucket] = bucket_selected.get(bucket, 0) + 1
            used += hint_cost
            if used >= char_budget:
                continue

        selected_ids = {id(item) for item in selected}
        self._pending_memory_hints = deque([item for item in kept if id(item) not in selected_ids])
        return selected

    @staticmethod
    def _hint_bucket(item: dict[str, Any]) -> str:
        card_type = str(item.get("card_type", "")).strip()
        if card_type in {"BugInvariantCard", "SuccessPathCard", "BugAntiPatternCard"}:
            return "semantic"
        rtype = str(item.get("type", "")).strip().lower()
        family_id = str(item.get("family_id", "")).strip().lower()
        pattern = str(item.get("normalized_pattern_type", "")).strip().lower()
        if pattern == "closure_signal" or family_id.startswith("closure_signal:") or family_id.startswith("step_budget:"):
            return "closure"
        if rtype in {"repair_pattern_v2", "failure_card_v2"} or pattern in {"negative_strategy", "patch_risk"}:
            return "repair"
        if pattern in {"validation_gap", "validation_guard"}:
            return "validation"
        if rtype == "belief_tip" or family_id.startswith("belief_tip:") or family_id.startswith("workflow_step:") or pattern == "planning_loop":
            return "workflow"
        return "general"

    @classmethod
    def _hint_bucket_priority(cls, item: dict[str, Any]) -> int:
        bucket = cls._hint_bucket(item)
        order = {
            "semantic": 6,
            "closure": 5,
            "repair": 4,
            "validation": 3,
            "general": 2,
            "workflow": 1,
        }
        return order.get(bucket, 0)

    def _register_interventions(self, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, item in enumerate(selected, start=1):
            trace_id = str(item.get("trace_id", "") or "")
            intervention_id = f"intv:{trace_id or 'missing'}:{self._step_index}:{idx}"
            state = {
                "intervention_id": intervention_id,
                "trace_id": trace_id or None,
                "memory_id": item.get("memory_id"),
                "created_step": self._step_index,
                "remaining_steps": self.intervention_window_steps,
                "observed_events": [],
                "success_events": 0,
                "error_events": 0,
                "closed": False,
            }
            self._active_interventions.append(state)
            rows.append(
                {
                    "intervention_id": intervention_id,
                    "trace_id": trace_id or None,
                    "memory_id": item.get("memory_id"),
                }
            )
        return rows

    def _advance_interventions(self, *, event: str, detail: str = "") -> None:
        if not self._active_interventions:
            return
        to_close: list[dict[str, Any]] = []
        for row in self._active_interventions:
            if row.get("closed"):
                continue
            row["observed_events"].append(event if not detail else f"{event}:{detail}")
            if event == "action_error":
                row["error_events"] = int(row.get("error_events", 0)) + 1
                self._v2_accum_card_stat(row.get("memory_id"), "error_events", 1)
            elif event == "action_success":
                row["success_events"] = int(row.get("success_events", 0)) + 1
                self._v2_accum_card_stat(row.get("memory_id"), "success_events", 1)
            row["remaining_steps"] = int(row.get("remaining_steps", 0)) - 1
            if int(row["remaining_steps"]) <= 0:
                to_close.append(row)

        if not to_close:
            return

        survivors: list[dict[str, Any]] = []
        for row in self._active_interventions:
            if row in to_close:
                row["closed"] = True
                success_count = int(row.get("success_events", 0))
                error_count = int(row.get("error_events", 0))
                # Effective when there is at least one success and more successes than errors
                local_effective = success_count > 0 and success_count >= error_count
                append_json_log(
                    "HOOK",
                    {
                        "version": "v1",
                        "event": "memory_intervention_feedback",
                        "agent": self._agent_name,
                        "instance_id": self._instance_id,
                        "attempt_id": self._attempt_id,
                        "intervention_id": row.get("intervention_id"),
                        "trace_id": row.get("trace_id"),
                        "memory_id": row.get("memory_id"),
                        "created_step": row.get("created_step"),
                        "closed_step": self._step_index,
                        "success_events": success_count,
                        "error_events": error_count,
                        "observed_events": row.get("observed_events", [])[-20:],
                        "local_effective": local_effective,
                        "evaluation_effective": None,
                        "feedback_stage": "local",
                        "effective": local_effective,
                    },
                )
            else:
                survivors.append(row)
        self._active_interventions = survivors

    def _emit_gate_evaluation(self) -> None:
        plan_generated = max(1, self._metrics.get("plan_generated", 0))
        external_resp = self._metrics.get("external_tool_response", 0)
        hint_buffered = self._metrics.get("memory_hint_buffered", 0)
        error_like = self._metrics.get("error_like_observation", 0)
        action_error = self._metrics.get("action_error_events", 0)

        external_ratio = external_resp / plan_generated
        buffer_ratio = (hint_buffered / external_resp) if external_resp > 0 else 0.0
        action_error_coverage = (action_error / error_like) if error_like > 0 else 1.0
        gates = {
            "external_ratio": external_ratio >= self.gate_min_external_ratio,
            "buffer_ratio": buffer_ratio >= self.gate_min_buffer_ratio,
            "action_error_coverage": action_error_coverage >= self.gate_min_action_error_coverage,
        }
        passed = all(gates.values())
        payload = {
            "version": "v1",
            "event": "memory_gate_evaluation",
            "agent": self._agent_name,
            "instance_id": self._instance_id,
            "attempt_id": self._attempt_id,
            "metrics": dict(self._metrics),
            "ratios": {
                "external_tool_response_over_plan_generated": round(external_ratio, 6),
                "memory_hint_buffered_over_external_tool_response": round(buffer_ratio, 6),
                "action_error_over_error_like_observation": round(action_error_coverage, 6),
            },
            "thresholds": {
                "external_ratio": self.gate_min_external_ratio,
                "buffer_ratio": self.gate_min_buffer_ratio,
                "action_error_coverage": self.gate_min_action_error_coverage,
            },
            "gates": gates,
            "passed": passed,
            "hard_fail_enabled": self.gate_hard_fail,
        }
        append_json_log("HOOK", payload)
        if self.gate_hard_fail and not passed:
            append_json_log(
                "HOOK",
                {
                    "version": "v1",
                    "event": "memory_gate_hard_fail",
                    "agent": self._agent_name,
                    "instance_id": self._instance_id,
                    "attempt_id": self._attempt_id,
                    "reason": "threshold_not_met",
                    "gates": gates,
                },
            )

    def on_model_query(self, *, messages: list[dict[str, str]], agent: str):
        self._v2_decide_strategy()
        self._v2_inject_strategy_header(messages)

        if not self.inject_enabled or not self._pending_memory_hints:
            return

        # T1-A: Context-aware hint reformulation (runs before selection)
        if self._t1a_enabled and self._reformulation_agent is not None and self._pending_memory_hints:
            self._t1a_reformat_pending_hints()

        selected = self._select_hints_for_injection()
        if not selected:
            return

        lines = [
            "[AgentMem Hints]",
            "Treat matching hints as high-priority guidance for this attempt.",
            "Avoid over-exploration: reuse validated repro/fix paths first, and once a minimal plausible patch exists, validate it before broadening the search.",
        ]
        for idx, item in enumerate(selected, start=1):
            hint = str(item.get("hint", "")).strip()
            if not hint:
                continue
            lines.append(f"{idx}. {hint}")
        content = "\n".join(lines)
        if len(content) > self.max_hint_chars:
            content = content[: self.max_hint_chars - 3].rstrip() + "..."

        messages.append({"role": "user", "content": content})

        self._logger.info("Injected memory hints: count=%d agent=%s", len(selected), agent)
        trigger_markers = list(self._recent_triggers)
        self._recent_triggers.clear()
        selected_memory_ids = [str(item.get("memory_id")) for item in selected if item.get("memory_id")]
        selected_trace_ids = [str(item.get("trace_id")) for item in selected if item.get("trace_id")]
        selected_family_ids = [str(item.get("family_id")) for item in selected if str(item.get("family_id", "")).strip()]
        trace_id_missing = any(not str(item.get("trace_id", "")).strip() for item in selected)
        selected_interventions = self._register_interventions(selected)
        hints_preview = [str(item.get("hint", ""))[:120] for item in selected if str(item.get("hint", "")).strip()]
        for family_id in selected_family_ids:
            self._hint_family_last_injected_step[family_id] = self._step_index
            self._hint_family_injected_count[family_id] = self._hint_family_injected_count.get(family_id, 0) + 1
        self._metric_inc("memory_injected", len(selected))
        append_json_log(
            "HOOK",
            {
                "version": "v1",
                "event": "memory_injected",
                "agent": self._agent_name,
                "instance_id": self._instance_id,
                "attempt_id": self._attempt_id,
                "target_agent": agent,
                "hint_count": len(selected),
                "selected_memory_ids": selected_memory_ids,
                "selected_trace_ids": selected_trace_ids,
                "selected_family_ids": selected_family_ids,
                "selected_interventions": selected_interventions,
                "trace_id_missing": trace_id_missing,
                "hints_preview": hints_preview,
                "step_index": self._step_index,
                "max_hints": self.max_hints,
                "hint_char_budget": self.hint_char_budget,
                "max_hint_chars": self.max_hint_chars,
                "family_cooldown_steps": self.family_cooldown_steps,
                "max_hints_per_family_per_attempt": self.max_hints_per_family_per_attempt,
                "min_confidence": self.min_confidence,
                "triggers": trigger_markers,
            },
        )

    # ============================================================
    # ============================================================

    def _v2_register_invariant(self, item: dict[str, Any]) -> None:
        """Register a retrieved invariant card for the current attempt."""
        card_id = str(item.get("card_id") or item.get("memory_id") or "")
        if any(str(x.get("card_id") or "") == card_id and card_id for x in self._v2_active_invariants):
            return
        self._v2_active_invariants.append(item)

        if not self._v2_strategy_decided:
            return
        if self._v2_strategy == "explore_fresh" and self._v2_p_reuse == 0.0:
            self._v2_strategy_decided = False

    def _v2_register_anti_pattern(self, item: dict[str, Any]) -> None:
        """Register a retrieved anti-pattern card for the current attempt."""
        card_id = str(item.get("card_id") or item.get("memory_id") or "")
        if any(str(x.get("card_id") or "") == card_id and card_id for x in self._v2_active_anti_patterns):
            return
        self._v2_active_anti_patterns.append(item)
        if not self._v2_strategy_decided:
            return
        if self._v2_strategy == "explore_fresh" and self._v2_p_reuse == 0.0:
            self._v2_strategy_decided = False

    def _v2_decide_strategy(self) -> None:
        """Choose reuse or exploration once enough retrieval context exists."""
        if self._v2_strategy_decided:
            return
        if self._v2_reuse_mode_env in {"off", ""}:
            self._v2_strategy_decided = True
            self._v2_strategy = "explore_fresh"
            self._v2_p_reuse = 0.0
            return
        if not self._v2_active_invariants and not self._v2_active_anti_patterns:
            if self._step_index < 3:
                return
            self._v2_strategy_decided = True
            self._v2_strategy = "explore_fresh"
            self._v2_p_reuse = 0.0
            append_json_log(
                "HOOK",
                {
                    "version": "v1",
                    "event": "v2_reuse_explore_decision",
                    "agent": self._agent_name,
                    "instance_id": self._instance_id,
                    "attempt_id": self._attempt_id,
                    "n_resolved": 0,
                    "n_failed": 0,
                    "p_reuse": 0.0,
                    "strategy": "explore_fresh",
                    "force_strategy_env": self._v2_force_strategy,
                    "deferred_no_cards": True,
                },
            )
            return
        if self._v2_force_strategy in {"reuse"}:
            self._v2_strategy_decided = True
            self._v2_strategy = "VERBATIM_REUSE"
            self._v2_p_reuse = 1.0
        elif self._v2_force_strategy in {"explore"}:
            self._v2_strategy_decided = True
            self._v2_strategy = "EXPLORE_AROUND_INVARIANT"
            self._v2_p_reuse = 0.0
        else:
            n_resolved = len(self._v2_active_invariants)
            n_failed = len(self._v2_active_anti_patterns)
            if n_resolved == 0:
                p_reuse = 0.0
                strategy_pool = "explore_fresh"
            elif n_failed == 0 and n_resolved >= 1:
                p_reuse = min(0.5 + 0.2 * n_resolved, 0.9)
                strategy_pool = "reuse_dominant"
            else:
                p_reuse = max(0.4, min(0.85, 0.6 + 0.1 * (n_resolved - n_failed)))
                strategy_pool = "mixed"
            seed_input = f"{self._instance_id}::{self._attempt_id}::{self._run_id}"
            seed = int(hashlib.sha1(seed_input.encode("utf-8")).hexdigest()[:12], 16)
            rng = random.Random(seed)
            draw = rng.random()
            if strategy_pool == "explore_fresh":
                strategy = "explore_fresh"
            else:
                strategy = "VERBATIM_REUSE" if draw < p_reuse else "EXPLORE_AROUND_INVARIANT"
            self._v2_strategy_decided = True
            self._v2_strategy = strategy
            self._v2_p_reuse = p_reuse
        if self._v2_strategy == "VERBATIM_REUSE":
            self._step_budget_guardrail_steps = min(self._step_budget_guardrail_steps, 8)
        append_json_log(
            "HOOK",
            {
                "version": "v1",
                "event": "v2_reuse_explore_decision",
                "agent": self._agent_name,
                "instance_id": self._instance_id,
                "attempt_id": self._attempt_id,
                "n_resolved": len(self._v2_active_invariants),
                "n_failed": len(self._v2_active_anti_patterns),
                "p_reuse": self._v2_p_reuse,
                "strategy": self._v2_strategy,
                "force_strategy_env": self._v2_force_strategy,
            },
        )

    def _v2_signature_hash_of_text(self, patch_text: str) -> str:
        """Compute a SHA-1 signature from stable added lines in a patch."""
        if not patch_text:
            return ""
        added = []
        for line in patch_text.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                stripped = line[1:].strip()
                if stripped and not stripped.startswith("#") and len(stripped) > 5:
                    added.append("ln::" + stripped)
                if len(added) >= 6:
                    break
        return "sha1:" + hashlib.sha1("\n".join(added).encode("utf-8", errors="replace")).hexdigest()

    @staticmethod
    def _v2_extract_param_signatures_from_text(patch_text: str) -> list[list[str]]:
        """Extract function parameter signatures from patch text."""
        results: list[list[str]] = []
        if not patch_text:
            return results
        for line in patch_text.splitlines():
            if "def " not in line:
                continue
            target = line[1:] if line.startswith("+") else line
            m = re.search(r"def\s+\w+\s*\(([^)]*)\)", target)
            if m:
                params = [
                    p.strip().split("=")[0].split(":")[0].strip()
                    for p in m.group(1).split(",")
                    if p.strip()
                ]
                if params:
                    results.append(params)
        return results

    def _v2_extract_current_patch_text(self, action_text: str) -> str:
        """Extract an inline diff when the action already contains patch text."""
        text = str(action_text or "")
        m = re.search(r"(diff --git[\s\S]+?)(?:\n\Z|\nEOF|\nPATCH_EOF)", text)
        if m:
            return m.group(1)
        return ""

    def _v2_rewrite_submit_to_pre_check(self, step: StepOutput) -> None:
        """Rewrite submit into a pre-submit diff inspection step."""
        original = step.action
        best_inv = self._v2_pick_best_invariant() if self._v2_active_invariants else {}
        inv_sig = (best_inv.get("minimal_patch_signature") or {}) if isinstance(best_inv, dict) else {}
        anchors = inv_sig.get("anchors") or []
        primary_params: list[str] = []
        primary_func = ""
        for a in anchors:
            if isinstance(a, dict) and a.get("symbol_kind") == "function" and a.get("param_signature"):
                primary_params = list(a.get("param_signature") or [])
                primary_func = str(a.get("symbol_name") or "")
                break
        anti_signatures: list[list[str]] = []
        for ap in self._v2_active_anti_patterns:
            sig = (ap.get("failed_patch_signature") or {}) if isinstance(ap, dict) else {}
            ps = sig.get("param_signature") or []
            if ps:
                anti_signatures.append(list(ps))
        invariant_hint = ""
        if primary_func and primary_params:
            invariant_hint = f"  [INVARIANT] Function `{primary_func}` MUST have parameters: {primary_params}\\n"
        if anti_signatures:
            invariant_hint += f"  [ANTI-PATTERN] Avoid parameter signatures: {anti_signatures}\\n"
        new_cmd = (
            "cd /testbed && git add -A && \\\n"
            "echo '===== [V2-Gate] PRE-SUBMIT DIFF =====' && \\\n"
            "git diff HEAD && \\\n"
            "echo '===== [V2-Gate] END DIFF =====' && \\\n"
            f"printf '%b' '[V2-Gate] Inspection above is your pending submission.\\n"
            f"{invariant_hint}"
            "   Run `submit` next to finalize (this inspection does NOT submit for you).\\n'"
        )
        step.action = new_cmd
        self._metric_inc("v2_gate_rewrite", 1)
        append_json_log(
            "HOOK",
            {
                "version": "v1",
                "event": "v2_action_rewritten",
                "agent": self._agent_name,
                "instance_id": self._instance_id,
                "attempt_id": self._attempt_id,
                "step_index": self._step_index,
                "rewrite_kind": "pre_submit_gate_inspection",
                "original_action": str(original)[:200],
                "primary_function": primary_func,
                "primary_params": primary_params,
                "anti_signatures": anti_signatures,
            },
        )

    def _v2_check_consistency_gate(self, step: StepOutput) -> str:
        """Apply the configured consistency gate to a submission action."""
        if self._v2_gate_mode in {"off", ""}:
            return "allow"
        if not self._is_patch_submission(step.action):
            return "allow"
        if self._v2_gate_pending_pass:
            self._v2_gate_pending_pass = False
            self._v2_emit_gate_event(step, decision="allow", reason="pending_pass_after_inspection", card=None)
            return "allow"
        patch_text = self._v2_extract_current_patch_text(step.action)
        if not patch_text:
            if (self._v2_active_invariants or self._v2_active_anti_patterns) and self._v2_gate_mode == "enforce":
                self._v2_gate_pending_pass = True
                self._v2_emit_gate_event(step, decision="pre_check", reason="submit_without_inline_patch", card=None)
                return "pre_check"
            return "allow"
        cur_hash = self._v2_signature_hash_of_text(patch_text)
        cur_param_sigs = self._t1c_added_param_signatures_from_text(patch_text)


        for ap in self._v2_active_anti_patterns:
            sig = (ap.get("failed_patch_signature") or {}) if isinstance(ap, dict) else {}
            ap_hash = str(sig.get("key_added_lines_hash") or ap.get("signature_hash") or "")
            if ap_hash and ap_hash == cur_hash:
                self._v2_emit_gate_event(step, decision="rewrite", reason="matches_anti_pattern_signature_hash", card=ap)
                return "rewrite" if self._v2_gate_mode == "enforce" else "allow_with_advise"
            ap_params = sig.get("param_signature") or []
            if ap_params and any(list(ap_params) == p for p in cur_param_sigs):
                self._v2_emit_gate_event(step, decision="rewrite", reason="matches_anti_pattern_param_signature", card=ap)
                return "rewrite" if self._v2_gate_mode == "enforce" else "allow_with_advise"

        if self._v2_active_invariants:
            best = self._v2_pick_best_invariant()
            inv_sig = (best.get("minimal_patch_signature") or {}) if isinstance(best, dict) else {}
            inv_hash = str(inv_sig.get("key_added_lines_hash") or best.get("signature_hash") or "")
            if inv_hash and inv_hash == cur_hash:
                self._v2_emit_gate_event(step, decision="allow", reason="matches_invariant_hash", card=best)
                return "allow"
            inv_params = []
            for a in (inv_sig.get("anchors") or []):
                if isinstance(a, dict) and a.get("symbol_kind") == "function":
                    inv_params = list(a.get("param_signature") or [])
                    if inv_params:
                        break
            if inv_params and not any(list(inv_params) == p for p in cur_param_sigs):
                if self._v2_strategy == "VERBATIM_REUSE":
                    self._v2_emit_gate_event(step, decision="force_reuse", reason="param_mismatch_in_reuse_mode", card=best)
                    return "force_reuse" if self._v2_gate_mode == "enforce" else "allow_with_advise"
                else:
                    self._v2_emit_gate_event(step, decision="rewrite", reason="param_mismatch_with_invariant", card=best)
                    return "rewrite" if self._v2_gate_mode == "enforce" else "allow_with_advise"
        self._v2_emit_gate_event(step, decision="allow", reason="no_violation", card=None)
        return "allow"

    def _v2_pick_best_invariant(self) -> dict[str, Any]:
        if not self._v2_active_invariants:
            return {}
        return max(
            self._v2_active_invariants,
            key=lambda c: float(c.get("item_confidence") or c.get("batch_confidence") or 0.0),
        )

    def _v2_emit_gate_event(self, step: Any, *, decision: str, reason: str, card: Any) -> None:
        metric_key = f"v2_gate_{decision}" if decision in {"allow", "rewrite", "force_reuse"} else "v2_gate_allow"
        self._metric_inc(metric_key, 1)
        append_json_log(
            "HOOK",
            {
                "version": "v1",
                "event": "v2_patch_consistency_gate",
                "agent": self._agent_name,
                "instance_id": self._instance_id,
                "attempt_id": self._attempt_id,
                "step_index": self._step_index,
                "decision": decision,
                "reason": reason,
                "gate_mode": self._v2_gate_mode,
                "strategy": self._v2_strategy,
                "card_id": (card or {}).get("card_id") if isinstance(card, dict) else None,
            },
        )

    def _v2_apply_gate_decision(self, step: StepOutput, decision: str) -> None:
        """Apply a consistency-gate decision to the generated action."""
        if decision == "allow":
            return
        if decision == "allow_with_advise":
            self._enqueue_system_hint(
                hint=(
                    "[V2-Gate  ADVISE] Your current submission appears to violate either the BUG_INVARIANT signature "
                    "or matches a known BUG_ANTI_PATTERN. Reconsider parameter names / key lines before submitting."
                ),
                family_id=f"v2_gate:advise:{self._step_index}",
                normalized_pattern_type="closure_signal",
                selection_score=1.0,
            )
            return
        if decision == "pre_check":
            self._v2_rewrite_submit_to_pre_check(step)
            return
        if decision == "rewrite":
            self._enqueue_system_hint(
                hint=(
                    "[V2-Gate  REWRITE] Submission blocked: patch violates known constraints. "
                    "Either align with the BUG_INVARIANT signature or avoid the BUG_ANTI_PATTERN before submitting again."
                ),
                family_id=f"v2_gate:rewrite:{self._step_index}",
                normalized_pattern_type="closure_signal",
                selection_score=1.0,
            )
            self._soft_block_runtime_guard_pattern(step, "v2_gate_rewrite_block")
            return
        if decision == "force_reuse":
            best = self._v2_pick_best_invariant()
            verbatim = ""
            if isinstance(best, dict):
                verbatim = str(((best.get("minimal_patch_signature") or {}).get("verbatim_diff") or "")).strip()
            if verbatim:
                self._v2_rewrite_action_to_apply_invariant(step, verbatim)
            else:
                self._v2_apply_gate_decision(step, "rewrite")

    def _v2_rewrite_action_to_apply_invariant(self, step: StepOutput, verbatim_diff: str) -> None:
        """Rewrite the action to apply a verified diff and submit it."""
        original_action = step.action
        marker = "PHASE9_V2_PATCH_EOF"
        new_cmd = (
            f"cd /testbed && git apply --reject --whitespace=nowarn <<'{marker}'\n"
            f"{verbatim_diff}\n"
            f"{marker}\n"
            "echo '[V2] Verbatim invariant patch applied; running validation and submitting.'\n"
            "git add -A && git diff --cached > /root/model.patch\n"
            "submit\n"
        )
        step.action = new_cmd
        self._patch_candidate_exists = True
        self._step_budget_state["steps_since_last_edit"] = 0
        self._metric_inc("v2_l3_force_reuse", 1)
        append_json_log(
            "HOOK",
            {
                "version": "v1",
                "event": "v2_action_rewritten",
                "agent": self._agent_name,
                "instance_id": self._instance_id,
                "attempt_id": self._attempt_id,
                "step_index": self._step_index,
                "rewrite_kind": "apply_invariant_verbatim",
                "original_action": original_action,
                "replacement_action_preview": new_cmd[:400],
            },
        )

    def _v2_rewrite_action_to_force_submit(self, step: StepOutput) -> None:
        """Rewrite the action to submit the best current patch at the hard stop."""
        original_action = step.action
        new_cmd = (
            "cd /testbed && git add -A && git diff --cached > /root/model.patch\n"
            "echo '[V2] Step budget hard stop reached; submitting best current candidate.'\n"
            "submit\n"
        )
        step.action = new_cmd
        self._metric_inc("v2_l3_force_submit", 1)
        append_json_log(
            "HOOK",
            {
                "version": "v1",
                "event": "v2_action_rewritten",
                "agent": self._agent_name,
                "instance_id": self._instance_id,
                "attempt_id": self._attempt_id,
                "step_index": self._step_index,
                "rewrite_kind": "force_submit_no_invariant",
                "original_action": original_action,
                "replacement_action_preview": new_cmd[:400],
            },
        )

    def _v2_check_l3_hard_stop(self, step: StepOutput) -> None:
        """Apply the configured hard-stop policy after the step budget expires."""
        if self._v2_l3_mode in {"off", ""}:
            return
        total = self._step_budget_state.get("total_steps", 0)
        if total < self._step_budget_hard_stop:
            return
        if self._patch_candidate_exists:
            return
        # dry_run
        if self._v2_l3_mode == "dry_run":
            append_json_log(
                "HOOK",
                {
                    "version": "v1",
                    "event": "v2_l3_dry_run",
                    "agent": self._agent_name,
                    "instance_id": self._instance_id,
                    "attempt_id": self._attempt_id,
                    "step_index": self._step_index,
                    "total_steps": total,
                    "had_invariant": bool(self._v2_active_invariants),
                },
            )
            return
        # In enforce mode, reuse a verified invariant when one is available.
        if self._v2_active_invariants:
            best = self._v2_pick_best_invariant()
            verbatim = str(((best.get("minimal_patch_signature") or {}).get("verbatim_diff") or "")).strip()
            if verbatim:
                self._v2_rewrite_action_to_apply_invariant(step, verbatim)
                return
        self._v2_rewrite_action_to_force_submit(step)

    def _v2_accum_card_stat(self, memory_id: Any, key: str, delta: int) -> None:
        """Accumulate success or error evidence for one injected card."""
        cid = str(memory_id or "").strip()
        if not cid:
            return
        bucket = self._v2_injected_card_stats.setdefault(
            cid, {"success_events": 0, "error_events": 0, "injected_steps": 0}
        )
        bucket[key] = int(bucket.get(key, 0)) + int(delta)

    def _v2_apply_local_effective_feedback(self) -> None:
        """Adjust injected-card confidence using local attempt evidence."""
        if not self._v2_local_effective_feedback:
            return
        gs = None
        try:
            storage_dir = os.environ.get("AGENT_MEM_STORAGE_DIR", "").strip()
            if storage_dir:
                import sys as _sys
                if "/home/pt/SWE-bench/PDDL_work_mem" not in _sys.path:
                    _sys.path.insert(0, "/home/pt/SWE-bench/PDDL_work_mem")
                from sweagent_external_tools_v2.agent_mem.storage.graph_store import GraphStore  # type: ignore
                gs = GraphStore(storage_dir=storage_dir)
        except Exception as e:  # pragma: no cover
            self._logger.warning("v2 feedback: failed to load GraphStore (%s); falling back to event-only", e)
            gs = None
        applied = 0
        for card_id, stats in self._v2_injected_card_stats.items():
            success = int(stats.get("success_events", 0))
            error = int(stats.get("error_events", 0))
            if success == 0 and error == 0:
                continue
            local_effective = success > 0 and success >= error
            delta = 0.05 if local_effective else -0.10
            if local_effective:
                self._metric_inc("v2_card_reinforced", 1)
            else:
                self._metric_inc("v2_card_demoted", 1)
            append_json_log(
                "HOOK",
                {
                    "version": "v1",
                    "event": "v2_card_feedback_request",
                    "agent": self._agent_name,
                    "instance_id": self._instance_id,
                    "attempt_id": self._attempt_id,
                    "card_id": card_id,
                    "success_events": success,
                    "error_events": error,
                    "local_effective": local_effective,
                    "delta_confidence": delta,
                    "demote_if_below": 0.4,
                    "reason": "local_effective_feedback",
                },
            )
            if gs is not None and hasattr(gs, "adjust_card"):
                try:
                    ok = gs.adjust_card(
                        card_id,
                        delta_confidence=delta,
                        demote_if_below=0.4,
                        reason="local_effective_feedback",
                    )
                    if ok:
                        applied += 1
                except Exception as e:
                    self._logger.warning("v2 adjust_card failed for %s: %s", card_id, e)
        if gs is not None and applied > 0:
            try:
                gs.save()
                append_json_log(
                    "HOOK",
                    {
                        "version": "v1",
                        "event": "v2_card_feedback_applied",
                        "agent": self._agent_name,
                        "instance_id": self._instance_id,
                        "attempt_id": self._attempt_id,
                        "applied_count": applied,
                    },
                )
            except Exception as e:
                self._logger.warning("v2 graph_store.save failed: %s", e)

    def _v2_inject_strategy_header(self, messages: list[dict[str, str]]) -> None:
        """Inject the selected reuse or exploration strategy into the prompt."""
        if self._v2_reuse_mode_env in {"off", ""}:
            return
        if not self._v2_strategy_decided:
            return
        if self._v2_strategy == "VERBATIM_REUSE":
            best = self._v2_pick_best_invariant()
            verbatim = ""
            if isinstance(best, dict):
                verbatim = str(((best.get("minimal_patch_signature") or {}).get("verbatim_diff") or "")).strip()
            content = (
                "[V2-Strategy  REUSE] This attempt is in REUSE mode. "
                "A previously resolved patch exists for this instance. "
                "Apply the BUG_INVARIANT verbatim diff as-is, run the listed validation, then submit. "
                "Do not introduce new exploration or rename parameters.\n"
            )
            if verbatim:
                content += "--- VERBATIM PATCH (reuse this exactly) ---\n" + verbatim[:6000] + "\n--- END ---\n"
            messages.append({"role": "user", "content": content})
        elif self._v2_strategy == "EXPLORE_AROUND_INVARIANT":
            content = (
                "[V2-Strategy  EXPLORE] This attempt is in EXPLORE mode. "
                "A previously resolved patch is available for reference but you may modify within the listed anchor files. "
                "AVOID all listed BUG_ANTI_PATTERN signatures. "
                "Validate the focused tests before submitting."
            )
            messages.append({"role": "user", "content": content})

    def on_actions_generated(self, *, step: StepOutput):
        self._v2_decide_strategy()
        self._v2_latest_step = step
        blocked_pattern = self._blocked_pattern_for_action(step.action)
        if blocked_pattern:
            self._soft_block_runtime_guard_pattern(step, blocked_pattern)
        elif self._is_new_ad_hoc_script_creation(step.action):
            self._soft_block_ad_hoc_script_creation(step)
        if self._is_patch_submission(step.action):
            decision = self._v2_check_consistency_gate(step)
            # T1-C: Semantic critic review (runs after hash-based gate)
            if self._t1c_enabled and self._critic_agent is not None:
                if decision == "allow" and self._v2_active_invariants:
                    decision = self._t1c_run_critic_and_decide(step)
            if decision != "allow":
                self._v2_apply_gate_decision(step, decision)
        self._step_index += 1
        self._metric_inc("plan_generated", 1)
        self._update_step_budget(step.action)
        self._check_step_budget_triggers()
        self._v2_check_l3_hard_stop(step)
        self._scan_proactive_triggers(step.action)
        trace_id = self._build_trace_id("plan_generated")
        self._active_step_trace_id = trace_id
        payload = self._build_event_payload(
            "plan_generated",
            trace_id=trace_id,
            thought=step.thought,
            action=step.action,
            runtime_guard=self._runtime_guard_payload(),
        )
        self._run_cmd(
            cmd=self.tool_a_cmd,
            tool_name="tool_a",
            payload=payload,
        )

    def on_action_error(self, *, step: StepOutput, error_type: str, error_message: str):
        self._consecutive_failures += 1
        self._metric_inc("action_error_events", 1)
        self._advance_interventions(event="action_error", detail=error_type or "unknown")
        if self._consecutive_failures == 1:
            self._record_trigger("first_failure", detail=error_type or "unknown")
        if self._consecutive_failures >= 2:
            self._record_trigger("consecutive_failures", detail=str(self._consecutive_failures))
        trace_id = self._build_trace_id("action_error")
        payload = self._build_event_payload(
            "action_error",
            trace_id=trace_id,
            error_type=error_type,
            error_message=error_message,
            thought=step.thought,
            action=step.action,
        )
        self._run_cmd(
            cmd=self.tool_b_cmd,
            tool_name="tool_b",
            payload=payload,
        )

    def on_action_executed(self, *, step: StepOutput):
        observation = str(step.observation)
        if self._is_error_like_observation(observation):
            self._metric_inc("error_like_observation", 1)
        self._append_success_fact_hotpath(step=step, observation=observation)
        # Reset failure streak on normal command observations.
        if "simulated command error" not in observation.lower():
            if self._consecutive_failures > 0:
                self._record_trigger("after_success", detail=str(self._consecutive_failures))
            self._consecutive_failures = 0
        self._advance_interventions(event="action_success")

        # T1-A: Maintain sliding trajectory window
        self._trajectory_window.append({
            "step": self._step_index,
            "action": str(step.action)[:300],
            "observation": observation[:400],
        })

        # T1-C: Maintain recent test output window
        obs_lower = observation.lower()
        if any(kw in obs_lower for kw in ("passed", "failed", "error", "assert", "traceback")):
            self._t1c_recent_test_outputs.append(observation[:600])
        if self._t1c_use_precheck_diff:
            self._t1c_capture_precheck_diff(observation)

        # T1-B: Check interim localization triggers (non-blocking async write)
        if self._t1b_enabled and self._interim_cache is not None:
            self._t1b_check_triggers(step, observation)

    def on_run_done(self, *, trajectory: Trajectory, info: AgentInfo):
        trace_id = self._build_trace_id("run_done")
        payload = self._build_event_payload(
            "run_done",
            trace_id=trace_id,
            trajectory_steps=len(trajectory),
            exit_status=info.get("exit_status"),
            has_submission=bool(info.get("submission")),
        )
        self._run_cmd(
            cmd=self.tool_b_cmd or self.tool_a_cmd,
            tool_name="tool_b",
            payload=payload,
        )
        # Close all still-open interventions at run end.
        for row in self._active_interventions:
            row["remaining_steps"] = 0
        self._advance_interventions(event="run_done")
        self._v2_apply_local_effective_feedback()
        self._emit_gate_evaluation()
        self._active_step_trace_id = ""
        append_json_log(
            "HOOK",
            {
                "version": "v1",
                "event": "run_done",
                "agent": self._agent_name,
                "instance_id": self._instance_id,
                "attempt_id": self._attempt_id,
                "trajectory_steps": len(trajectory),
                "exit_status": info.get("exit_status"),
                "has_submission": bool(info.get("submission")),
            },
        )

    def _append_success_fact_hotpath(self, *, step: StepOutput, observation: str) -> None:
        if not self._v21_enable_success_fact_hotpath or not self._v21_enable_sidecar:
            return
        if self._episode_ledger_store is None:
            self._metric_inc("success_fact_hotpath_skipped", 1)
            return

        trace_id = str(self._active_step_trace_id or "").strip()
        record_id = build_success_fact_idempotency_key(trace_id, self._step_index)
        if not record_id:
            self._metric_inc("success_fact_hotpath_skipped", 1)
            append_json_log(
                "HOOK",
                {
                    "version": "v2.1",
                    "event": "success_fact_hotpath_skipped",
                    "instance_id": self._instance_id,
                    "run_id": self._run_id,
                    "attempt_id": self._attempt_id,
                    "step_index": self._step_index,
                    "reason": "missing_trace_or_step",
                    "timestamp": time.time(),
                },
            )
            return

        event = {
            "version": "v2.1",
            "event": "success_fact",
            "fact_source": "hotpath",
            "record_id": record_id,
            "idempotency_key": record_id,
            "instance_id": self._instance_id,
            "run_id": self._run_id,
            "attempt_id": self._attempt_id,
            "trace_id": trace_id,
            "step_index": self._step_index,
            "action_text": str(step.action or ""),
            "success_like": classify_success_like(observation=observation),
            "timestamp": time.time(),
        }
        result = self._episode_ledger_store.append(event, stream="episode_ledger")
        if result.get("written"):
            self._metric_inc("success_fact_hotpath_written", 1)
            return
        self._metric_inc("success_fact_hotpath_skipped", 1)
        if result.get("skipped_reason") != "duplicate_record_id":
            append_json_log(
                "HOOK",
                {
                    "version": "v2.1",
                    "event": "success_fact_hotpath_failed",
                    "instance_id": self._instance_id,
                    "run_id": self._run_id,
                    "attempt_id": self._attempt_id,
                    "step_index": self._step_index,
                    "trace_id": trace_id,
                    "reason": result.get("skipped_reason"),
                    "timestamp": time.time(),
                },
            )

    # ================================================================
    # Multi-Agent T1 Helper Methods
    # ================================================================


    def _t1a_reformat_pending_hints(self) -> None:
        """Rewrite pending hints to match the current execution phase."""
        if not self._pending_memory_hints:
            return
        if (
            self._t1a_max_reformats_per_attempt > 0
            and self._t1a_reformat_count >= self._t1a_max_reformats_per_attempt
        ):
            append_json_log(
                "HOOK",
                {
                    "version": "v1",
                    "event": "t1a_reformulation_skipped",
                    "agent": self._agent_name,
                    "instance_id": self._instance_id,
                    "attempt_id": self._attempt_id,
                    "step_index": self._step_index,
                    "reason": "max_reformats_per_attempt",
                    "max_reformats_per_attempt": self._t1a_max_reformats_per_attempt,
                },
            )
            return
        summary = self._build_trajectory_summary()
        phase = self._infer_step_phase()
        original_hints = list(self._pending_memory_hints)
        try:
            rewritten = self._reformulation_agent.reformat(original_hints, summary, phase)
            self._pending_memory_hints = deque(rewritten)
            self._metric_inc("t1a_reformat_called")
            self._t1a_reformat_count += 1
            append_json_log(
                "HOOK",
                {
                    "version": "v1",
                    "event": "t1a_reformulation_done",
                    "agent": self._agent_name,
                    "instance_id": self._instance_id,
                    "attempt_id": self._attempt_id,
                    "step_index": self._step_index,
                    "phase": phase,
                    "hint_count": len(rewritten),
                },
            )
        except Exception:
            self._metric_inc("t1a_reformat_failed")

    def _build_trajectory_summary(self) -> str:
        """Build a compact string from the recent trajectory window."""
        if not self._trajectory_window:
            return "(no trajectory yet)"
        lines: list[str] = []
        for entry in self._trajectory_window:
            s = entry.get("step", "?")
            act = str(entry.get("action", ""))[:120].replace("\n", " ")
            obs = str(entry.get("observation", ""))[:160].replace("\n", " ")
            lines.append(f"Step {s}: action={act!r}  obs={obs!r}")
        return "\n".join(lines)

    def _infer_step_phase(self) -> str:
        """Heuristically infer the agent's current execution phase."""
        if self._step_index < 8 and not self._patch_candidate_exists:
            return "explore"
        budget = self._step_budget_state
        if budget.get("steps_since_last_edit", 0) == 0 and budget.get("steps_since_last_patch", 0) == 0:
            return "pre_submit"
        if self._patch_candidate_exists or budget.get("steps_since_last_patch", 99) <= 5:
            return "fixing"
        if self._seen_files:
            return "localized"
        return "explore"


    def _t1b_check_triggers(self, step: StepOutput, observation: str) -> None:
        """Check if any T1-B trigger condition is met and fire async write."""
        action_text = str(step.action or "")
        obs_lower = observation.lower()

        # Trigger 1: consecutive localization pattern hits
        loc_matches = re.findall(r"([a-zA-Z_][a-zA-Z0-9_/]*\.py):(\d+)", observation)
        if loc_matches:
            self._t1b_localization_hits += 1
            self._t1b_last_localization = loc_matches
        else:
            self._t1b_localization_hits = 0

        trigger1 = self._t1b_localization_hits >= self._t1b_localize_threshold

        # Trigger 2: first successful test run
        trigger2 = (
            not self._t1b_first_pass_written
            and any(kw in obs_lower for kw in ("passed", "ok", "1 passed", "2 passed"))
            and "failed" not in obs_lower
        )

        # Trigger 3: first code edit action
        trigger3 = (
            not self._t1b_first_edit_written
            and self._is_code_edit_action(action_text)
        )

        if not (trigger1 or trigger2 or trigger3):
            return

        localization = self._t1b_extract_localization(observation)
        if not localization.get("file"):
            return

        card_type = (
            "InterimProgressCard" if trigger2 or trigger3
            else "InterimLocalizationCard"
        )

        # The asynchronous write must not stall the Watchdog budget.
        import threading
        t = threading.Thread(
            target=self._t1b_write_async,
            args=(card_type, localization),
            daemon=True,
        )
        t.start()

        if trigger2:
            self._t1b_first_pass_written = True
        if trigger3:
            self._t1b_first_edit_written = True

    def _t1b_extract_localization(self, observation: str) -> dict[str, Any]:
        """Extract file/function/line_range from an observation string."""
        matches = re.findall(r"([a-zA-Z_][a-zA-Z0-9_/]*\.py):(\d+)", observation)
        if not matches:
            matches = self._t1b_last_localization
        if not matches:
            return {}
        file_path, line_no = matches[0]
        # Try to extract a function name near the match
        func_match = re.search(r"def ([a-zA-Z_]\w+)\(", observation)
        function = func_match.group(1) if func_match else ""
        return {
            "file": file_path,
            "function": function,
            "line_range": f"{line_no}-{line_no}",
            "confidence": 0.6,
        }

    def _t1b_write_async(self, card_type: str, localization: dict[str, Any]) -> None:
        """Thread target: write interim card to cache (errors are swallowed)."""
        try:
            ok = self._interim_cache.write_interim_card(
                instance_id=self._instance_id,
                attempt_id=self._attempt_id,
                card_type=card_type,
                localization=localization,
                source_step=self._step_index,
            )
            if ok:
                self._metric_inc("t1b_interim_written")
                append_json_log(
                    "HOOK",
                    {
                        "version": "v1",
                        "event": "t1b_interim_written",
                        "instance_id": self._instance_id,
                        "attempt_id": self._attempt_id,
                        "card_type": card_type,
                        "step_index": self._step_index,
                        "localization": localization,
                    },
                )
        except Exception:
            pass


    def _t1c_capture_precheck_diff(self, observation: str) -> None:
        """Capture the diff printed by the v2 pre-submit inspection command."""
        if "===== [V2-Gate] PRE-SUBMIT DIFF =====" not in observation:
            return
        start_marker = "===== [V2-Gate] PRE-SUBMIT DIFF ====="
        end_marker = "===== [V2-Gate] END DIFF ====="
        start = observation.find(start_marker)
        end = observation.find(end_marker, start + len(start_marker))
        if start < 0 or end <= start:
            return
        diff_text = observation[start + len(start_marker):end].strip()
        if "diff --git" not in diff_text:
            return
        self._t1c_last_precheck_diff = diff_text[:12000]
        append_json_log(
            "HOOK",
            {
                "version": "v1",
                "event": "t1c_precheck_diff_captured",
                "agent": self._agent_name,
                "instance_id": self._instance_id,
                "attempt_id": self._attempt_id,
                "step_index": self._step_index,
                "diff_chars": len(self._t1c_last_precheck_diff),
            },
        )

    def _t1c_run_critic_and_decide(self, step: StepOutput) -> str:
        """Run semantic critic review; return gate decision string."""
        # Anti-infinite-loop: if a revision was just injected, auto-approve
        if self._t1c_revision_pending:
            self._t1c_revision_pending = False
            self._t1c_emit_event(verdict="approve", reason="auto_approve_after_revision", hint="", source="auto")
            return "allow"

        patch_text = self._v2_extract_current_patch_text(step.action)
        if not patch_text and self._t1c_use_precheck_diff:
            patch_text = self._t1c_last_precheck_diff
        if not patch_text:
            self._t1c_emit_skip_event(reason="missing_patch_text")
            return "allow"

        best_invariant = self._v2_pick_best_invariant()
        if not best_invariant:
            self._t1c_emit_skip_event(reason="missing_invariant_card")
            return "allow"

        patch_key = self._t1c_patch_review_key(patch_text)
        previous_review_count = self._t1c_seen_patch_hashes.get(patch_key, 0)
        self._t1c_seen_patch_hashes[patch_key] = previous_review_count + 1

        deterministic_verdict = self._t1c_deterministic_guard_verdict(
            patch_text=patch_text,
            best_invariant=best_invariant,
            previous_review_count=previous_review_count,
        )
        if deterministic_verdict is not None:
            self._t1c_emit_event(
                verdict=deterministic_verdict.verdict,
                reason=deterministic_verdict.reasoning,
                hint=deterministic_verdict.revision_hint,
                source="deterministic",
            )
            return self._t1c_apply_verdict(deterministic_verdict, source="deterministic")

        try:
            verdict = self._critic_agent.critique(
                patch_diff=patch_text,
                instance_id=self._instance_id,
                invariant_card=best_invariant,
                anti_pattern_cards=list(self._v2_active_anti_patterns),
                recent_test_outputs=list(self._t1c_recent_test_outputs),
            )
        except Exception:
            self._t1c_emit_skip_event(reason="critic_exception")
            return "allow"

        if self._t1c_is_fallback_approve(verdict):
            self._metric_inc("t1c_unavailable")
            self._t1c_unavailable_count += 1
            self._t1c_emit_unavailable_event(reason=verdict.reasoning or "fallback_approve")
            unavailable_verdict = self._t1c_unavailable_policy_verdict(
                reason=verdict.reasoning or "fallback_approve",
            )
            if unavailable_verdict is not None:
                self._t1c_emit_event(
                    verdict=unavailable_verdict.verdict,
                    reason=unavailable_verdict.reasoning,
                    hint=unavailable_verdict.revision_hint,
                    source="deterministic",
                )
                return self._t1c_apply_verdict(unavailable_verdict, source="deterministic")
            return "allow"

        self._t1c_emit_event(
            verdict=verdict.verdict,
            reason=verdict.reasoning,
            hint=verdict.revision_hint,
            source="llm",
        )

        return self._t1c_apply_verdict(verdict, source="llm")

    def _t1c_apply_verdict(self, verdict: Any, *, source: str) -> str:
        if verdict.verdict == "revise":
            self._metric_inc("t1c_revise")
            if source == "deterministic":
                self._metric_inc("t1c_deterministic_revise")
            # Inject revision hint for the next model query
            if verdict.revision_hint:
                self._enqueue_system_hint(
                    hint=f"[Critic Review] {verdict.revision_hint}",
                    family_id="t1c_critic_revision",
                    normalized_pattern_type="patch_risk",
                    selection_score=0.95,
                )
            self._t1c_revision_pending = True
            # Return "allow" so the current action proceeds, but agent will
            # see the revision hint on next query and likely re-submit
            return "allow"
        elif verdict.verdict == "reject":
            self._metric_inc("t1c_reject")
            if source == "deterministic":
                self._metric_inc("t1c_deterministic_reject")
            return "force_reuse"
        else:
            self._metric_inc("t1c_approve")
            return "allow"

    def _t1c_patch_review_key(self, patch_text: str) -> str:
        signature = self._v2_signature_hash_of_text(patch_text)
        if signature and signature != "sha1:da39a3ee5e6b4b0d3255bfef95601890afd80709":
            return signature
        return "sha1:" + hashlib.sha1(patch_text.encode("utf-8", errors="replace")).hexdigest()

    def _t1c_deterministic_guard_verdict(
        self,
        *,
        patch_text: str,
        best_invariant: dict[str, Any],
        previous_review_count: int,
    ) -> Any | None:
        if not self._t1c_deterministic_guard or CriticVerdict is None:
            return None

        cur_hash = self._v2_signature_hash_of_text(patch_text)
        cur_param_sigs = self._t1c_added_param_signatures_from_text(patch_text)

        for ap in self._v2_active_anti_patterns:
            if not isinstance(ap, dict):
                continue
            sig = ap.get("failed_patch_signature") or {}
            ap_hash = str(sig.get("key_added_lines_hash") or ap.get("signature_hash") or "")
            if ap_hash and ap_hash == cur_hash:
                return CriticVerdict(
                    semantic_match_with_invariant=0.1,
                    semantic_match_with_anti_pattern=1.0,
                    verdict="reject",
                    revision_hint="The patch matches a known failed patch signature; force reuse of the best invariant patch instead.",
                    reasoning="deterministic_matches_anti_pattern_hash",
                )
            ap_params = sig.get("param_signature") or []
            if ap_params and any(list(ap_params) == p for p in cur_param_sigs):
                return CriticVerdict(
                    semantic_match_with_invariant=0.2,
                    semantic_match_with_anti_pattern=0.95,
                    verdict="reject",
                    revision_hint="The patch repeats a known failed parameter signature; switch to the invariant signature before submitting.",
                    reasoning="deterministic_matches_anti_pattern_param_signature",
                )

        inv_params = self._t1c_invariant_param_signature(best_invariant)
        if inv_params and cur_param_sigs and not any(list(inv_params) == p for p in cur_param_sigs):
            return CriticVerdict(
                semantic_match_with_invariant=0.35,
                semantic_match_with_anti_pattern=0.2,
                verdict="revise",
                revision_hint=f"Revise the patch to match the invariant parameter signature: {inv_params}.",
                reasoning="deterministic_param_mismatch_with_invariant",
            )

        if self._t1c_revise_duplicate_precheck and previous_review_count > 0:
            return CriticVerdict(
                semantic_match_with_invariant=0.45,
                semantic_match_with_anti_pattern=0.4,
                verdict="revise",
                revision_hint="This pre-submit diff is identical to a previous review in this attempt; change strategy or reuse the invariant patch before submitting again.",
                reasoning="deterministic_duplicate_precheck_diff",
            )

        return None

    def _t1c_unavailable_policy_verdict(self, *, reason: str) -> Any | None:
        """Escalate Critic backend unavailability into an explicit guardrail when configured."""
        if not self._t1c_deterministic_guard or CriticVerdict is None:
            return None
        policy = self._t1c_unavailable_policy
        if policy in {"", "allow", "off", "none"}:
            return None
        if policy == "revise_once" and self._t1c_unavailable_count > 1:
            return None
        if policy not in {"revise", "revise_once", "reject"}:
            return None

        verdict = "reject" if policy == "reject" else "revise"
        hint = (
            "Critic backend was unavailable; do not blindly submit. Re-check the captured diff "
            "against the invariant key lines and target tests, then submit only if it matches; "
            "otherwise reuse the invariant patch."
        )
        if verdict == "reject":
            hint = (
                "Critic backend was unavailable under reject policy; force reuse of the invariant "
                "patch instead of submitting an unreviewed diff."
            )
        return CriticVerdict(
            semantic_match_with_invariant=0.5,
            semantic_match_with_anti_pattern=0.5,
            verdict=verdict,
            revision_hint=hint,
            reasoning=f"deterministic_critic_unavailable_policy:{reason}",
        )

    @staticmethod
    def _t1c_invariant_param_signature(invariant_card: dict[str, Any]) -> list[str]:
        inv_sig = (invariant_card.get("minimal_patch_signature") or {}) if isinstance(invariant_card, dict) else {}
        for anchor in inv_sig.get("anchors") or []:
            if isinstance(anchor, dict) and anchor.get("symbol_kind") == "function":
                params = list(anchor.get("param_signature") or [])
                if params:
                    return params
        return []

    @staticmethod
    def _t1c_added_param_signatures_from_text(patch_text: str) -> list[list[str]]:
        results: list[list[str]] = []
        for line in str(patch_text or "").splitlines():
            if not line.startswith("+") or line.startswith("+++"):
                continue
            target = line[1:]
            if "def " not in target:
                continue
            m = re.search(r"def\s+\w+\s*\(([^)]*)\)", target)
            if not m:
                continue
            params = [
                p.strip().split("=")[0].split(":")[0].strip()
                for p in m.group(1).split(",")
                if p.strip()
            ]
            if params:
                results.append(params)
        return results

    def _t1c_is_fallback_approve(self, verdict: Any) -> bool:
        if not self._t1c_split_fallback_approve:
            return False
        if str(getattr(verdict, "verdict", "") or "").lower() != "approve":
            return False
        reason = str(getattr(verdict, "reasoning", "") or "").strip().lower()
        return reason in {
            "fallback",
            "llm_error_or_timeout",
            "parse_error",
            "empty_patch",
            "no_invariant_card",
        } or reason.startswith("fallback_")

    def _t1c_emit_skip_event(self, *, reason: str) -> None:
        append_json_log(
            "HOOK",
            {
                "version": "v1",
                "event": "t1c_critic_skipped",
                "agent": self._agent_name,
                "instance_id": self._instance_id,
                "attempt_id": self._attempt_id,
                "step_index": self._step_index,
                "reason": reason,
            },
        )

    def _t1c_emit_unavailable_event(self, *, reason: str) -> None:
        append_json_log(
            "HOOK",
            {
                "version": "v1",
                "event": "t1c_critic_unavailable",
                "agent": self._agent_name,
                "instance_id": self._instance_id,
                "attempt_id": self._attempt_id,
                "step_index": self._step_index,
                "reason": reason,
            },
        )

    def _t1c_emit_event(self, *, verdict: str, reason: str, hint: str, source: str = "llm") -> None:
        append_json_log(
            "HOOK",
            {
                "version": "v1",
                "event": "t1c_critic_verdict",
                "agent": self._agent_name,
                "instance_id": self._instance_id,
                "attempt_id": self._attempt_id,
                "step_index": self._step_index,
                "verdict": verdict,
                "source": source,
                "reason": reason,
                "revision_hint": hint,
            },
        )
