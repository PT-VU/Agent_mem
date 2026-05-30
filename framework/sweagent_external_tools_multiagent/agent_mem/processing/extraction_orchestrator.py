"""
Extraction orchestrator for run_done post-processing.

It maps intermediate analysis into existing templates only:
- abstract_experience
- failure_card_v2
- repair_pattern_v2
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..core.problem_file import ActionType, Outcome, ProblemFile
from .llm_extractor import CriticalSignal, LLMExperienceExtractor
from .taxonomy import normalize_error_type
from .abstract_experience import (
    _normalize_advice_family,
    _normalize_pattern_family,
    _normalize_trigger_family,
    build_experience_family_id,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", (text or "").lower()).strip("_") or "unknown"


class ExtractionOrchestrator:
    """Coordinates step analysis, critical-signal detection, and schema mapping."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        min_item_confidence: float = 0.35,
        extractor: Optional[LLMExperienceExtractor] = None,
        max_signals_per_attempt: int = 0,
        max_abstracts_per_attempt: int = 0,
        max_repair_patterns_per_attempt: int = 0,
    ):
        self.enabled = enabled
        self.min_item_confidence = max(0.0, min(1.0, float(min_item_confidence)))
        self.extractor = extractor or LLMExperienceExtractor.from_env()
        # 0 means no fixed count cap.
        self.max_signals_per_attempt = max(0, int(max_signals_per_attempt))
        self.max_abstracts_per_attempt = max(0, int(max_abstracts_per_attempt))
        self.max_repair_patterns_per_attempt = max(0, int(max_repair_patterns_per_attempt))

    @classmethod
    def from_env(cls) -> "ExtractionOrchestrator":
        enabled_raw = os.getenv("AGENT_MEM_ENABLE_LLM_EXTRACTION", "1").strip().lower()
        enabled = enabled_raw in {"1", "true", "yes", "on"}
        min_conf = float(os.getenv("AGENT_MEM_EXTRACT_MIN_CONFIDENCE", "0.35"))
        max_signals = int(os.getenv("AGENT_MEM_EXTRACT_MAX_SIGNALS", "0"))
        max_abstracts = int(os.getenv("AGENT_MEM_EXTRACT_MAX_ABSTRACTS", "0"))
        max_patterns = int(os.getenv("AGENT_MEM_EXTRACT_MAX_REPAIR_PATTERNS", "0"))
        return cls(
            enabled=enabled,
            min_item_confidence=min_conf,
            max_signals_per_attempt=max_signals,
            max_abstracts_per_attempt=max_abstracts,
            max_repair_patterns_per_attempt=max_patterns,
        )

    def process_attempt(
        self,
        *,
        task_id: str,
        actions: List[ProblemFile],
        success: bool,
        task_summary: str,
        exit_status: str,
        source_instance_id: str = "",
        source_run_id: str = "",
        source_attempt_id: str = "",
        trace_id: Optional[str] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not actions:
            return {"enabled": True, "triggered": False, "reason": "no_actions"}

        trial_overview = self._build_trial_overview(
            actions=actions,
            task_summary=task_summary,
            success=success,
            exit_status=exit_status,
            extra_context=extra_context,
        )
        subblock_analysis = self._build_subblock_analysis(
            actions=actions,
            trial_overview=trial_overview,
            extra_context=extra_context,
        )
        attempt_summary = self._build_attempt_summary(
            task_id=task_id,
            source_instance_id=source_instance_id,
            source_run_id=source_run_id,
            source_attempt_id=source_attempt_id,
            trace_id=trace_id,
            trial_overview=trial_overview,
            subblock_analysis=subblock_analysis,
            actions=actions,
        )
        if not self.enabled:
            return {
                "enabled": False,
                "triggered": False,
                "reason": "disabled_by_config",
                "trial_overview": trial_overview,
                "subblock_analysis": subblock_analysis,
                "attempt_summary_v1": attempt_summary,
            }

        trigger_reason = self._trigger_reason(actions=actions, success=success)
        if not trigger_reason:
            return {
                "enabled": True,
                "triggered": False,
                "reason": "trigger_not_matched",
                "trial_overview": trial_overview,
                "subblock_analysis": subblock_analysis,
                "attempt_summary_v1": attempt_summary,
            }

        attempt_id = self._attempt_id(task_id, source_run_id, source_attempt_id)
        analyzed = self.extractor.analyze_steps(
            attempt_id=attempt_id,
            actions=actions,
            context=extra_context,
        )
        strategy_observations = list(analyzed.get("strategy_observations", []) or [])
        strategy_observations.extend(
            self._heuristic_strategy_observations(actions=actions, context=extra_context)
        )
        assessments = analyzed.get("assessments", [])
        critical_signals = self.extractor.detect_critical_signals(
            attempt_id=attempt_id,
            actions=actions,
            assessments=assessments,
            success=success,
            exit_status=exit_status,
            context=extra_context,
        )
        if self.max_signals_per_attempt > 0:
            critical_signals = critical_signals[: self.max_signals_per_attempt]
        if not critical_signals:
            return {
                "enabled": True,
                "triggered": True,
                "reason": trigger_reason,
                "attempt_id": attempt_id,
                "assessments": assessments,
                "critical_signal": {},
                "critical_signals": [],
                "abstract_experiences": [],
                "failure_card_patch": {},
                "repair_patterns": [],
                "quality_gate": {"accepted": False, "reason": "no_critical_signal"},
                "llm_used": analyzed.get("llm_used", False),
                "trial_overview": trial_overview,
                "subblock_analysis": subblock_analysis,
                "attempt_summary_v1": attempt_summary,
            }

        payloads: Dict[str, Any] = {
            "abstract_experiences": [],
            "failure_card_patch": {},
            "repair_patterns": [],
        }
        for idx, critical in enumerate(critical_signals):
            row = self._map_to_existing_templates(
                critical=critical,
                actions=actions,
                task_id=task_id,
                task_summary=task_summary,
                success=success,
                source_instance_id=source_instance_id,
                source_run_id=source_run_id,
                source_attempt_id=source_attempt_id,
                trace_id=trace_id,
                signal_index=idx,
                extra_context=extra_context,
            )
            payloads["abstract_experiences"].extend(row.get("abstract_experiences", []))
            payloads["repair_patterns"].extend(row.get("repair_patterns", []))
            if idx == 0:
                payloads["failure_card_patch"] = dict(row.get("failure_card_patch") or {})
            else:
                patch = payloads.get("failure_card_patch", {})
                extra_patch = row.get("failure_card_patch", {}) or {}
                if isinstance(patch, dict):
                    patch["candidate_fix_actions"] = self._dedup(
                        list(patch.get("candidate_fix_actions", []) or [])
                        + list(extra_patch.get("candidate_fix_actions", []) or []),
                        limit=None,
                    )
                    patch["verification_commands"] = self._dedup(
                        list(patch.get("verification_commands", []) or [])
                        + list(extra_patch.get("verification_commands", []) or []),
                        limit=12,
                    )
                    patch["evidence_refs"] = self._dedup(
                        list(patch.get("evidence_refs", []) or [])
                        + list(extra_patch.get("evidence_refs", []) or []),
                        limit=60,
                    )
                    patch["root_cause_nodes"] = self._dedup(
                        list(patch.get("root_cause_nodes", []) or [])
                        + list(extra_patch.get("root_cause_nodes", []) or []),
                        limit=40,
                    )
                    patch["propagation_chain"] = self._dedup(
                        list(patch.get("propagation_chain", []) or [])
                        + list(extra_patch.get("propagation_chain", []) or []),
                        limit=40,
                    )
                    payloads["failure_card_patch"] = patch

        payloads["abstract_experiences"].extend(
            self._map_strategy_observations(
                strategy_observations=strategy_observations,
                actions=actions,
                task_id=task_id,
                task_summary=task_summary,
                source_instance_id=source_instance_id,
                source_run_id=source_run_id,
                source_attempt_id=source_attempt_id,
                trace_id=trace_id,
                extra_context=extra_context,
            )
        )

        if self.max_abstracts_per_attempt > 0:
            payloads["abstract_experiences"] = payloads["abstract_experiences"][: self.max_abstracts_per_attempt]
        if self.max_repair_patterns_per_attempt > 0:
            payloads["repair_patterns"] = payloads["repair_patterns"][: self.max_repair_patterns_per_attempt]
        payloads["abstract_experiences"] = self._rebalance_abstract_experiences(payloads["abstract_experiences"])
        payloads["repair_patterns"] = self._dedup_repair_patterns(payloads["repair_patterns"])
        gated = self._apply_quality_gate(payloads)

        return {
            "enabled": True,
            "triggered": True,
            "reason": trigger_reason,
            "attempt_id": attempt_id,
            "assessments": assessments,
            "critical_signal": critical_signals[0].to_dict(),
            "critical_signals": [sig.to_dict() for sig in critical_signals],
            "abstract_experiences": gated["abstract_experiences"],
            "failure_card_patch": gated["failure_card_patch"],
            "repair_patterns": gated["repair_patterns"],
            "quality_gate": gated["quality_gate"],
            "llm_used": analyzed.get("llm_used", False),
            "strategy_observations": strategy_observations,
            "taxonomy_version": analyzed.get("taxonomy_version", "unknown"),
            "trial_overview": trial_overview,
            "subblock_analysis": subblock_analysis,
            "attempt_summary_v1": attempt_summary,
        }

    def _build_trial_overview(
        self,
        *,
        actions: List[ProblemFile],
        task_summary: str,
        success: bool,
        exit_status: str,
        extra_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        ctx = extra_context if isinstance(extra_context, dict) else {}
        initial_plan_outline: List[str] = []
        for action in actions[:4]:
            text = self._summarize_action(action)
            if not text:
                continue
            initial_plan_outline.append(text[:140])
            if len(initial_plan_outline) >= 3:
                break

        actual_execution_outline: List[str] = []
        seen_blocks = set()
        for action in actions:
            block = self._infer_subproblem_type(action=action)
            strategy = self._infer_strategy_label(
                actions=[action],
                context=ctx,
                subproblem_type=block,
            )
            label = f"{block}:{strategy}" if strategy and strategy != "unknown_strategy" else block
            if label in seen_blocks:
                continue
            seen_blocks.add(label)
            actual_execution_outline.append(label)
            if len(actual_execution_outline) >= 6:
                break

        key_divergences: List[str] = []
        if int(ctx.get("ad_hoc_script_count", 0) or 0) >= 2:
            key_divergences.append("reproduce_issue drifted into multiple ad-hoc scripts")
        if len(actions) >= 40:
            key_divergences.append("attempt became long-running before converging on a minimal patch")
        texts = " ".join(f"{action.intent_text} {action.action_text}" for action in actions).lower()
        validation_runs = sum(1 for action in actions if action.action_type == ActionType.RUN_TEST)
        if validation_runs >= 3 and not any(action.action_type == ActionType.CODE_EDIT for action in actions):
            key_divergences.append("repeated validation ran before a stable patch candidate was formed")
        if any(token in texts for token in ("world_to_pixel_values", "all_world2pix", "fitswcs")) and any(
            token in texts for token in ("wcsaxes", "utils.py", "visualization")
        ):
            key_divergences.append("search expanded beyond the target WCS path after likely localization")
        if str(exit_status).strip().lower() in {"submitted"}:
            final_outcome = "submitted"
        else:
            final_outcome = str(ctx.get("official_eval_status") or exit_status or ("resolved" if success else "unresolved")).strip().lower()

        return {
            "problem_goal": str(ctx.get("task_problem_excerpt") or task_summary or "").strip()[:280],
            "initial_plan_outline": initial_plan_outline,
            "actual_execution_outline": actual_execution_outline,
            "key_divergences": key_divergences[:6],
            "final_outcome": final_outcome or ("resolved" if success else "unresolved"),
        }

    def _build_subblock_analysis(
        self,
        *,
        actions: List[ProblemFile],
        trial_overview: Dict[str, Any],
        extra_context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ctx = extra_context if isinstance(extra_context, dict) else {}
        grouped: Dict[str, List[ProblemFile]] = defaultdict(list)
        order: List[str] = []
        for action in actions:
            block = self._infer_subproblem_type(action=action)
            if block not in grouped:
                order.append(block)
            grouped[block].append(action)

        out: List[Dict[str, Any]] = []
        for block in order:
            block_actions = grouped.get(block, [])
            if not block_actions:
                continue
            strategy_label = self._infer_strategy_label(actions=block_actions, context=ctx, subproblem_type=block)
            prefer_actions, avoid_actions = self._derive_prefer_avoid_actions(
                subproblem_type=block,
                strategy_label=strategy_label,
                context=ctx,
            )
            failure_source = "execution"
            if block == "unknown":
                failure_source = "mixed"
            positives = self._positive_contribution(block_actions=block_actions, subproblem_type=block)
            negative_contribution: List[str] = []
            if strategy_label in {"ad_hoc_repro_script_loop", "cross_module_expansion_after_key_signal", "broad_test_without_patch"}:
                negative_contribution.append(f"{strategy_label} consumed effort without converging to a minimal fix")
            if block == "broad_regression_check":
                negative_contribution.append("broad validation happened before a focused patch path was secured")
            if block == "localize_fix" and strategy_label == "cross_module_expansion_after_key_signal":
                negative_contribution.append("localization drifted into adjacent modules before testing the minimal target fix")
            if block == "target_validation" and not positives:
                negative_contribution.append("validation effort did not confirm a tighter patch candidate")
            plan_success = any(action.outcome == Outcome.SUCCESS for action in block_actions)
            if not plan_success and block in {"reproduce_issue", "localize_fix", "form_minimal_patch", "target_validation"}:
                failure_source = "plan"
            elif negative_contribution and positives:
                failure_source = "mixed"
            out.append(
                {
                    "subproblem_type": block,
                    "goal": self._goal_for_subproblem(block),
                    "is_goal_necessary": block in {"reproduce_issue", "localize_fix", "form_minimal_patch", "target_validation"},
                    "plan_role": "required" if block in {"reproduce_issue", "localize_fix", "form_minimal_patch", "target_validation"} else "optional",
                    "plan_success": bool(plan_success),
                    "execution_paths": [strategy_label],
                    "key_actions": [str(action.action_id) for action in block_actions[:6]],
                    "positive_contribution": positives,
                    "negative_contribution": negative_contribution[:4],
                    "failure_source": failure_source,
                    "prefer_actions": prefer_actions,
                    "avoid_actions": avoid_actions,
                }
            )
        return out

    def _build_attempt_summary(
        self,
        *,
        task_id: str,
        source_instance_id: str,
        source_run_id: str,
        source_attempt_id: str,
        trace_id: Optional[str],
        trial_overview: Dict[str, Any],
        subblock_analysis: List[Dict[str, Any]],
        actions: List[ProblemFile],
    ) -> Dict[str, Any]:
        confirmed_signals: List[str] = []
        failed_strategies: List[Dict[str, Any]] = []
        best_partial_progress: List[str] = []
        next_best_actions: List[str] = []
        source_action_ids: List[str] = []

        for block in subblock_analysis:
            source_action_ids.extend([str(x) for x in (block.get("key_actions") or []) if str(x).strip()])
            positives = [str(x).strip() for x in (block.get("positive_contribution") or []) if str(x).strip()]
            negatives = [str(x).strip() for x in (block.get("negative_contribution") or []) if str(x).strip()]
            prefers = [str(x).strip() for x in (block.get("prefer_actions") or []) if str(x).strip()]
            avoids = [str(x).strip() for x in (block.get("avoid_actions") or []) if str(x).strip()]
            if positives:
                best_partial_progress.extend(positives[:2])
                confirmed_signals.extend(positives[:2])
            if negatives:
                failed_strategies.append(
                    {
                        "subproblem_type": str(block.get("subproblem_type") or "unknown"),
                        "strategy_label": str((block.get("execution_paths") or ["unknown_strategy"])[0]),
                        "reason": negatives[0],
                        "avoid_actions": avoids[:4],
                    }
                )
            next_best_actions.extend(prefers[:2])
            if not positives and not negatives and str(block.get("subproblem_type") or "") in {"reproduce_issue", "localize_fix", "target_validation"}:
                failed_strategies.append(
                    {
                        "subproblem_type": str(block.get("subproblem_type") or "unknown"),
                        "strategy_label": str((block.get("execution_paths") or ["unknown_strategy"])[0]),
                        "reason": "the subproblem consumed effort without producing a verifiable local improvement",
                        "avoid_actions": avoids[:4],
                    }
                )

        if not confirmed_signals:
            confirmed_signals.extend([str(x).strip() for x in (trial_overview.get("key_divergences") or []) if str(x).strip()][:2])
        if not best_partial_progress:
            for block in subblock_analysis:
                if block.get("plan_success"):
                    best_partial_progress.append(
                        f"{block.get('subproblem_type')}: reached its local goal before the attempt stalled"
                    )
                    if len(best_partial_progress) >= 2:
                        break
        if not next_best_actions:
            if any(str(block.get("subproblem_type") or "") == "localize_fix" for block in subblock_analysis):
                next_best_actions.append("test_the_smallest_fix_on_the_localized_target_before_expanding_scope")
            if any(str(block.get("subproblem_type") or "") == "reproduce_issue" for block in subblock_analysis):
                next_best_actions.append("reuse_the_confirmed_reproduction_path_instead_of_creating_new_scripts")

        return {
            "schema_version": "1.0",
            "summary_id": "",
            "instance_id": source_instance_id,
            "run_id": source_run_id,
            "attempt_id": source_attempt_id,
            "trace_id": trace_id or "",
            "task_id": task_id,
            "problem_goal": str(trial_overview.get("problem_goal") or "")[:280],
            "initial_plan_outline": list(trial_overview.get("initial_plan_outline") or [])[:4],
            "actual_execution_outline": list(trial_overview.get("actual_execution_outline") or [])[:6],
            "plan_success": bool(any(bool(block.get("plan_success")) for block in subblock_analysis)),
            "final_outcome": str(trial_overview.get("final_outcome") or "unknown"),
            "confirmed_signals": self._dedup(confirmed_signals, limit=8),
            "failed_strategies": failed_strategies[:6],
            "best_partial_progress": self._dedup(best_partial_progress, limit=6),
            "unverified_hypotheses": [],
            "next_best_actions": self._dedup(next_best_actions, limit=6),
            "source_action_ids": self._dedup(source_action_ids, limit=20),
            "subblock_analysis": subblock_analysis[:6],
            "created_at": _now_iso(),
        }

    def _infer_subproblem_type(self, *, action: ProblemFile) -> str:
        text = f"{action.intent_text} {action.action_text}".lower()
        if action.action_type == ActionType.RUN_TEST or "pytest" in text or "test_" in text or "reproduce_" in text:
            if "regression" in text or "all tests" in text or "full suite" in text:
                return "broad_regression_check"
            return "reproduce_issue" if any(marker in text for marker in ("reproduce", "test_", "quiet", "plot", "failing")) else "target_validation"
        if action.action_type == ActionType.CODE_EDIT:
            return "form_minimal_patch"
        if any(marker in text for marker in ("grep", "search", "inspect", "read", "open", "locate")):
            return "localize_fix"
        if any(marker in text for marker in ("docker", "proxy", "retry", "install", "timeout")):
            return "tool_recovery"
        return "unknown"

    def _infer_strategy_label(
        self,
        *,
        actions: List[ProblemFile],
        context: Optional[Dict[str, Any]],
        subproblem_type: str,
    ) -> str:
        ctx = context if isinstance(context, dict) else {}
        ad_hoc_count = int(ctx.get("ad_hoc_script_count", 0) or 0)
        step_count = int(ctx.get("step_count", 0) or len(actions))
        texts = " ".join(f"{action.intent_text} {action.action_text}" for action in actions).lower()
        if subproblem_type == "reproduce_issue" and (ad_hoc_count >= 2 or texts.count("test_") + texts.count("reproduce_") >= 2):
            return "ad_hoc_repro_script_loop"
        if subproblem_type == "localize_fix" and any(marker in texts for marker in ("wcsaxes", "utils.py", "other module")) and step_count >= 20:
            return "cross_module_expansion_after_key_signal"
        if subproblem_type == "broad_regression_check":
            return "broad_test_without_patch"
        if subproblem_type in {"form_minimal_patch", "target_validation"} and any(action.action_type == ActionType.CODE_EDIT for action in actions):
            return "minimal_patch_then_target_validation"
        if "world_to_pixel_values" in texts or "all_world2pix" in texts:
            return "api_alternative_probe_without_fix"
        if subproblem_type == "tool_recovery":
            return "tool_recovery_retry_loop"
        return "unknown_strategy"

    def _derive_prefer_avoid_actions(
        self,
        *,
        subproblem_type: str,
        strategy_label: str,
        context: Optional[Dict[str, Any]],
    ) -> tuple[List[str], List[str]]:
        if strategy_label == "ad_hoc_repro_script_loop":
            return (
                ["reuse_confirmed_repro_path", "switch_to_target_validation_after_repro_confirmed"],
                ["create_new_repro_script_after_repro_confirmed"],
            )
        if strategy_label == "cross_module_expansion_after_key_signal":
            return (
                ["edit_target_function_before_new_module_search", "validate_minimal_fix_on_target_path"],
                ["expand_search_to_unrelated_module_after_localization"],
            )
        if strategy_label == "broad_test_without_patch":
            return (
                ["run_target_validation_before_broad_tests"],
                ["run_broad_regression_before_patch_candidate_exists"],
            )
        if subproblem_type == "form_minimal_patch":
            return (["edit_target_function_before_new_repro_script"], [])
        if subproblem_type == "target_validation":
            return (["run_target_validation_before_submission"], ["submit_without_target_validation"])
        return ([], [])

    @staticmethod
    def _action_template(action: ProblemFile) -> str:
        text = f"{action.intent_text} {action.action_text}".lower()
        if action.action_type == ActionType.RUN_TEST:
            if any(token in text for token in ("reproduce_", "test_")):
                return "run_repro_script"
            if "pytest" in text:
                return "run_targeted_validation"
            return "run_validation"
        if action.action_type == ActionType.CODE_EDIT:
            return "edit_target_code"
        if any(token in text for token in ("grep", "search", "find")):
            return "search_codebase"
        if any(token in text for token in ("read", "inspect", "open")):
            return "inspect_source"
        if any(token in text for token in ("docker", "install", "retry", "timeout")):
            return "recover_tooling"
        return _slug(action.action_type.value if action.action_type else "unknown_action")

    @staticmethod
    def _goal_for_subproblem(subproblem_type: str) -> str:
        goals = {
            "reproduce_issue": "stabilize the failing path with a minimal reproduction",
            "localize_fix": "identify the smallest code location that controls the failing behavior",
            "form_minimal_patch": "produce the smallest viable patch candidate",
            "target_validation": "verify the candidate fix on the target failing path",
            "broad_regression_check": "check broader regressions only after a targeted fix exists",
            "tool_recovery": "recover the runtime or toolchain before continuing",
        }
        return goals.get(subproblem_type, "progress the task without unnecessary scope expansion")

    @staticmethod
    def _positive_contribution(*, block_actions: List[ProblemFile], subproblem_type: str) -> List[str]:
        positives: List[str] = []
        texts = " ".join(f"{action.intent_text} {action.action_text}" for action in block_actions).lower()
        if subproblem_type == "reproduce_issue":
            if any(token in texts for token in ("quiet=true", "quiet true", "all_world2pix")):
                positives.append("confirmed a reproducible WCS path and narrowed the behavior difference")
            if "traceback" in texts or "assert" in texts:
                positives.append("captured a concrete failing signal for the target path")
        if subproblem_type == "localize_fix" and any(token in texts for token in ("world_to_pixel_values", "all_world2pix", "fitswcs")):
            positives.append("narrowed the likely fix surface to the target WCS path")
        if subproblem_type == "form_minimal_patch" and any(action.action_type == ActionType.CODE_EDIT for action in block_actions):
            positives.append("produced at least one concrete patch candidate")
        if subproblem_type == "target_validation" and any(action.action_type == ActionType.RUN_TEST for action in block_actions):
            positives.append("ran focused validation on the currently suspected failing path")
        if subproblem_type == "tool_recovery" and any(token in texts for token in ("docker", "retry", "timeout", "reconnect")):
            positives.append("recovered enough tooling state to continue the attempt")
        return positives[:2]

    def _rebalance_abstract_experiences(self, abstracts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: Dict[str, Dict[str, Any]] = {}
        for row in abstracts:
            family_id = str(row.get("family_id", "")).strip() or _slug(
                f"{row.get('pattern_type','')}::{row.get('abstracted_intent','')}"
            )
            current = deduped.get(family_id)
            score = float(row.get("confidence", 0.0))
            if current is None or score > float(current.get("confidence", 0.0)):
                deduped[family_id] = row

        limits = {
            "closure_signal": 1,
            "negative_strategy": 2,
            "patch_risk": 1,
            "planning_rule": 1,
            "execution_tip": 1,
            "failure_recovery": 2,
            "validation_guard": 1,
            "validation_gap": 1,
        }
        selected: List[Dict[str, Any]] = []
        used = defaultdict(int)
        priorities = {
            "closure_signal": 0,
            "negative_strategy": 0,
            "patch_risk": 1,
            "failure_recovery": 1,
            "validation_guard": 2,
            "validation_gap": 2,
            "execution_tip": 3,
            "planning_rule": 4,
        }
        ranked = sorted(
            deduped.values(),
            key=lambda row: (
                priorities.get(self._classify_abstract_bucket(row), 99),
                -float(row.get("confidence", 0.0)),
                -len(row.get("evidence_refs", []) or []),
            ),
        )
        high_value_present = any(
            self._classify_abstract_bucket(row)
            in {"closure_signal", "negative_strategy", "patch_risk", "failure_recovery", "validation_guard", "validation_gap"}
            for row in ranked
        )
        for row in ranked:
            bucket = self._classify_abstract_bucket(row)
            if bucket == "planning_rule" and (self._planning_payload_is_too_generic(row) or high_value_present):
                continue
            if used[bucket] >= limits.get(bucket, 1):
                continue
            selected.append(row)
            used[bucket] += 1
        return selected

    def _classify_abstract_bucket(self, payload: Dict[str, Any]) -> str:
        norm_pattern = str(payload.get("normalized_pattern_type", "")).strip().lower()
        norm_advice = str(payload.get("normalized_advice_family", "")).strip().lower()
        pattern_type = str(payload.get("pattern_type", "")).strip().lower()
        if norm_pattern == "closure_signal" or pattern_type.startswith("closure_signal"):
            return "closure_signal"
        if norm_pattern == "negative_strategy" or pattern_type.startswith("negative_strategy"):
            return "negative_strategy"
        if norm_pattern == "patch_risk" or pattern_type.startswith("patch_risk"):
            return "patch_risk"
        if norm_pattern.startswith("repair_") or norm_pattern == "failure_recovery" or pattern_type.startswith("repair_"):
            return "failure_recovery"
        if norm_pattern in {"validation_guard", "validation_gap"} or norm_advice == "add_local_validation":
            return "validation_gap" if norm_pattern == "validation_gap" else "validation_guard"
        if norm_pattern == "planning_loop":
            return "planning_rule"
        return "execution_tip"

    def _planning_payload_is_too_generic(self, payload: Dict[str, Any]) -> bool:
        text = str(payload.get("abstracted_intent", "")).strip().lower()
        evidence = payload.get("evidence_refs", []) or []
        if len(evidence) < 2:
            return True
        generic_markers = (
            "generate plan for task",
            "resolve task with incremental actions",
            "run_done exit_status",
            "unknown",
        )
        if not text or any(marker in text for marker in generic_markers):
            return True
        return False

    def _dedup_repair_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        selected: List[Dict[str, Any]] = []
        for row in sorted(patterns, key=lambda r: float(r.get("confidence", 0.0)), reverse=True):
            trigger = row.get("trigger_signature") if isinstance(row.get("trigger_signature"), dict) else {}
            key = (
                str(trigger.get("error_type", "")).strip().lower(),
                str(trigger.get("error_stage", "")).strip().lower(),
                str(trigger.get("error_module", "")).strip().lower(),
                _slug(str(row.get("fix_action_template", "")).strip())[:80],
            )
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
        return selected

    def _attempt_id(self, task_id: str, run_id: str, attempt_id: str) -> str:
        parts = [run_id.strip(), attempt_id.strip(), task_id.strip()]
        compact = ":".join([p for p in parts if p])
        return compact or task_id

    def _trigger_reason(self, *, actions: List[ProblemFile], success: bool) -> str:
        fail_actions = [a for a in actions if a.outcome == Outcome.FAIL]
        if not success:
            return "run_done_unresolved"
        if fail_actions:
            return "run_done_success_with_recovery"
        # High-cost success trigger.
        step_count = len(actions)
        if step_count >= 40:
            return "run_done_high_cost"
        return ""

    def _map_to_existing_templates(
        self,
        *,
        critical: CriticalSignal,
        actions: List[ProblemFile],
        task_id: str,
        task_summary: str,
        success: bool,
        source_instance_id: str,
        source_run_id: str,
        source_attempt_id: str,
        trace_id: Optional[str],
        signal_index: int = 0,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        evidence_refs = self._collect_evidence_refs(actions, critical_step=critical.critical_step)
        fix_actions = self._candidate_fix_actions(critical)
        verification = self._verification_commands(actions, error_type=critical.error_type)
        signal_density = round(
            len([a for a in actions if a.failure_signature or a.touched_files or a.test_stats]) / max(1, len(actions)),
            4,
        )
        critical_dict = critical.to_dict()
        metadata_common = {
            "extraction_mode": self.extractor.mode,
            "critical_signal": critical_dict,
            "task_summary": task_summary[:300],
            "source_instance_id": source_instance_id,
            "source_run_ids": [source_run_id] if source_run_id else [],
            "source_attempt_ids": [source_attempt_id] if source_attempt_id else [],
            "source_event_ids": [trace_id] if trace_id else [],
            **self._stage_metadata(success=success, extra_context=extra_context),
        }
        if isinstance(extra_context, dict):
            metadata_common["submission_status"] = str(extra_context.get("submission_status", ""))[:64]
            metadata_common["official_eval_status"] = str(extra_context.get("official_eval_status", ""))[:64]
            if isinstance(extra_context.get("patch_summary"), dict):
                metadata_common["patch_summary"] = dict(extra_context["patch_summary"])
            if isinstance(extra_context.get("validation_summary"), dict):
                metadata_common["validation_summary"] = dict(extra_context["validation_summary"])
        critical_actions = [action for action in actions if int(action.step_index or 0) >= max(0, critical.critical_step - 1)]
        subproblem_type = self._infer_subproblem_type(action=critical_actions[-1] if critical_actions else actions[-1])
        strategy_label = self._infer_strategy_label(
            actions=critical_actions or actions[-3:],
            context=extra_context,
            subproblem_type=subproblem_type,
        )
        prefer_actions, avoid_actions = self._derive_prefer_avoid_actions(
            subproblem_type=subproblem_type,
            strategy_label=strategy_label,
            context=extra_context,
        )
        applicability_scope = {
            "subproblem_type": subproblem_type,
            "strategy_label": strategy_label,
            "official_eval_status": str((extra_context or {}).get("official_eval_status") or "unknown"),
        }

        pattern_type = f"critical_{_slug(critical.critical_module)}_{_slug(critical.error_type)}"
        abstract_payload = {
            "schema_version": "2.1",
            "experience_id": "",
            "pattern_type": pattern_type,
            "abstracted_intent": critical.correction_guidance[:280],
            "variant_texts": [critical.correction_guidance[:280]],
            "success_conditions": self._success_conditions(critical),
            "failure_avoidance": [f"avoid_repeat_{_slug(critical.error_type)}_without_strategy_shift"],
            "evidence_refs": evidence_refs[:20],
            "source_task_ids": [task_id],
            "source_event_ids": [trace_id] if trace_id else [],
            "source_instance_id": source_instance_id,
            "source_run_ids": [source_run_id] if source_run_id else [],
            "source_attempt_ids": [source_attempt_id] if source_attempt_id else [],
            "source_action_ids": [str(action.action_id) for action in (critical_actions or actions[-3:])[:6]],
            "source_action_chain": [self._action_template(action) for action in (critical_actions or actions[-3:])[:3]],
            "subproblem_type": subproblem_type,
            "strategy_label": strategy_label,
            "prefer_actions": prefer_actions,
            "avoid_actions": avoid_actions,
            "applicability_scope": applicability_scope,
            "support_count": 1,
            "confidence": round(max(0.2, min(0.95, critical.confidence)), 4),
            "lifecycle_status": "new",
            "links": {"related_experience_ids": []},
            "quality": {
                "item_confidence": round(max(0.2, min(0.95, critical.confidence)), 4),
                "support_count": 1,
                "signal_density": signal_density,
            },
            "fingerprint": "",
            "created_at": _now_iso(),
            "last_updated": _now_iso(),
            "metadata": {
                **metadata_common,
                "task_success": success,
                "error_signature": _slug(critical.error_type),
                "tool_sequence": self._tool_sequence(actions),
                "changed_file_pattern": self._changed_file_pattern(actions),
                "test_signal": "has_test" if any(a.action_type.value == "run_test" for a in actions) else "no_test",
                "critical_signal_rank": signal_index + 1,
                "subproblem_type": subproblem_type,
                "strategy_label": strategy_label,
                "prefer_actions": prefer_actions,
                "avoid_actions": avoid_actions,
                "applicability_scope": applicability_scope,
            },
        }
        abstract_payload["normalized_pattern_type"] = _normalize_pattern_family(
            abstract_payload["pattern_type"],
            abstract_payload["abstracted_intent"],
            abstract_payload["metadata"],
        )
        abstract_payload["normalized_trigger_family"] = _normalize_trigger_family(
            pattern_type=abstract_payload["pattern_type"],
            success_conditions=abstract_payload["success_conditions"],
            failure_avoidance=abstract_payload["failure_avoidance"],
            metadata=abstract_payload["metadata"],
        )
        abstract_payload["normalized_advice_family"] = _normalize_advice_family(
            abstract_payload["abstracted_intent"],
            abstract_payload["success_conditions"],
            abstract_payload["failure_avoidance"],
        )
        abstract_payload["family_id"] = build_experience_family_id(
            normalized_pattern_type=abstract_payload["normalized_pattern_type"],
            normalized_trigger_family=abstract_payload["normalized_trigger_family"],
            normalized_advice_family=abstract_payload["normalized_advice_family"],
        )

        failure_patch = {
            "error_signature": {
                "error_type": normalize_error_type(critical.error_type),
                "error_stage": self._error_stage(critical),
                "error_module": critical.critical_module,
            },
            "candidate_fix_actions": fix_actions,
            "verification_commands": verification,
            "evidence_refs": evidence_refs[:25],
            "root_cause_nodes": [ref for ref in evidence_refs if ref.startswith("action://")][:5],
            "propagation_chain": [f"step_{critical.critical_step}"] + [
                f"step_{row.get('step')}" for row in critical.cascading_effects[:6]
            ],
            "error_module": critical.critical_module,
            "confidence": round(max(0.2, min(0.95, critical.confidence)), 4),
            "metadata": {
                **metadata_common,
                "critical_alignment": {
                    "error_type": _slug(critical.error_type),
                    "error_module": _slug(critical.critical_module),
                    "critical_step": critical.critical_step,
                },
            },
        }

        repair_patterns = []
        for idx, action_text in enumerate(fix_actions, start=1):
            repair_patterns.append(
                {
                    "trigger_signature": {
                        "error_type": normalize_error_type(critical.error_type),
                        "error_stage": self._error_stage(critical),
                        "error_module": critical.critical_module,
                    },
                    "fix_action_template": action_text,
                    "expected_verification": verification[:3],
                    "evidence_refs": evidence_refs[:12],
                    "support": 1,
                    "confidence": round(max(0.2, min(0.95, critical.confidence - 0.04 * (idx - 1))), 4),
                    "trace_id": trace_id,
                    "metadata": {
                        **metadata_common,
                        "source": "llm_assisted_extractor",
                        "critical_alignment": {
                            "error_type": _slug(critical.error_type),
                            "error_module": _slug(critical.critical_module),
                            "critical_step": critical.critical_step,
                        },
                    },
                }
            )

        return {
            "abstract_experiences": [abstract_payload],
            "failure_card_patch": failure_patch,
            "repair_patterns": repair_patterns,
        }

    def _apply_quality_gate(self, payloads: Dict[str, Any]) -> Dict[str, Any]:
        failure_card_patch = dict(payloads.get("failure_card_patch") or {})
        abstracts = []
        for row in list(payloads.get("abstract_experiences") or []):
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            if (
                str(metadata.get("official_eval_status", "")).strip().lower() == "unresolved"
                and str(metadata.get("experience_polarity", "neutral")).strip().lower() != "negative"
            ):
                continue
            if self._classify_abstract_bucket(row) == "planning_rule" and self._planning_payload_is_too_generic(row):
                continue
            abstracts.append(row)
        patterns = list(payloads.get("repair_patterns") or [])

        evidence_refs = list(failure_card_patch.get("evidence_refs") or [])
        confidence = float(failure_card_patch.get("confidence", 0.0))
        if not evidence_refs:
            for row in abstracts:
                refs = row.get("evidence_refs") or []
                if refs:
                    evidence_refs = list(refs)
                    confidence = max(confidence, float(row.get("confidence", 0.0)))
                    break
        if not evidence_refs:
            for row in patterns:
                refs = row.get("evidence_refs") or []
                if refs:
                    evidence_refs = list(refs)
                    confidence = max(confidence, float(row.get("confidence", 0.0)))
                    break
        if not evidence_refs:
            return {
                "abstract_experiences": [],
                "failure_card_patch": {},
                "repair_patterns": [],
                "quality_gate": {"accepted": False, "reason": "missing_evidence_refs"},
            }
        if confidence < self.min_item_confidence:
            return {
                "abstract_experiences": [],
                "failure_card_patch": {},
                "repair_patterns": [],
                "quality_gate": {
                    "accepted": False,
                    "reason": "confidence_below_threshold",
                    "confidence": round(confidence, 4),
                    "threshold": round(self.min_item_confidence, 4),
                },
            }
        return {
            "abstract_experiences": abstracts,
            "failure_card_patch": failure_card_patch,
            "repair_patterns": patterns,
            "quality_gate": {
                "accepted": True,
                "reason": "passed",
                "confidence": round(confidence, 4),
            },
        }

    def _stage_metadata(self, *, success: bool, extra_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        ctx = extra_context if isinstance(extra_context, dict) else {}
        submission_success = bool(ctx.get("submission_success", False))
        eval_status = str(ctx.get("official_eval_status", "")).strip().lower() or "unknown"
        if eval_status == "resolved":
            return {
                "experience_polarity": "positive",
                "promotion_state": "promoted",
                "evidence_stage": "official_eval",
            }
        if eval_status in {"unresolved", "incomplete", "infra_failure"}:
            return {
                "experience_polarity": "negative" if eval_status == "unresolved" else "neutral",
                "promotion_state": "candidate",
                "evidence_stage": "official_eval",
            }
        if submission_success:
            return {
                "experience_polarity": "neutral",
                "promotion_state": "candidate",
                "evidence_stage": "submission",
            }
        if success:
            return {
                "experience_polarity": "positive",
                "promotion_state": "candidate",
                "evidence_stage": "trial_local",
            }
        return {
            "experience_polarity": "neutral",
            "promotion_state": "candidate",
            "evidence_stage": "trial_local",
        }

    def _map_strategy_observations(
        self,
        *,
        strategy_observations: List[Dict[str, Any]],
        actions: List[ProblemFile],
        task_id: str,
        task_summary: str,
        source_instance_id: str,
        source_run_id: str,
        source_attempt_id: str,
        trace_id: Optional[str],
        extra_context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not strategy_observations:
            return []
        stage_meta = self._stage_metadata(success=False, extra_context=extra_context)
        evidence_refs = self._collect_evidence_refs(actions, critical_step=max(0, len(actions) - 3))
        changed_file_pattern = self._changed_file_pattern(actions)
        subproblem_type = self._infer_subproblem_type(action=actions[-1]) if actions else "unknown"
        default_strategy_label = self._infer_strategy_label(
            actions=actions,
            context=extra_context,
            subproblem_type=subproblem_type,
        )
        out: List[Dict[str, Any]] = []
        for idx, row in enumerate(strategy_observations, start=1):
            strategy_type = _slug(str(row.get("strategy_type", "")))
            if not strategy_type:
                continue
            bucket = "negative_strategy"
            if "closure" in strategy_type or "over_exploration" in strategy_type:
                bucket = "closure_signal"
            elif "validation" in strategy_type:
                bucket = "validation_gap"
            elif "risk" in strategy_type or "patch" in strategy_type:
                bucket = "patch_risk"
            abstracted_intent = (
                str(row.get("recommended_avoidance", "")).strip()
                or str(row.get("why_failed_or_risky", "")).strip()
            )
            if not abstracted_intent:
                continue
            confidence = round(max(0.25, min(0.9, float(row.get("confidence", 0.0) or 0.0))), 4)
            strategy_label = default_strategy_label
            if "ad_hoc" in strategy_type:
                strategy_label = "ad_hoc_repro_script_loop"
            elif "closure" in strategy_type and default_strategy_label == "unknown_strategy":
                strategy_label = "cross_module_expansion_after_key_signal"
            prefer_actions, avoid_actions = self._derive_prefer_avoid_actions(
                subproblem_type=subproblem_type,
                strategy_label=strategy_label,
                context=extra_context,
            )
            payload = {
                "schema_version": "2.1",
                "experience_id": "",
                "pattern_type": f"{bucket}_{strategy_type}",
                "abstracted_intent": abstracted_intent[:280],
                "variant_texts": [abstracted_intent[:280]],
                "success_conditions": (
                    ["run_targeted_validation_before_submission"]
                    if bucket == "validation_gap"
                    else ["change_strategy_after_failed_attempt"]
                ),
                "failure_avoidance": [abstracted_intent[:180]],
                "evidence_refs": evidence_refs[:20],
                "source_task_ids": [task_id],
                "source_event_ids": [trace_id] if trace_id else [],
                "source_instance_id": source_instance_id,
                "source_run_ids": [source_run_id] if source_run_id else [],
                "source_attempt_ids": [source_attempt_id] if source_attempt_id else [],
                "source_action_ids": [str(action.action_id) for action in actions[-6:]],
                "source_action_chain": [strategy_label],
                "subproblem_type": subproblem_type,
                "strategy_label": strategy_label,
                "prefer_actions": prefer_actions,
                "avoid_actions": avoid_actions,
                "applicability_scope": {
                    "subproblem_type": subproblem_type,
                    "strategy_label": strategy_label,
                },
                "support_count": 1,
                "confidence": confidence,
                "lifecycle_status": "new",
                "links": {"related_experience_ids": []},
                "quality": {"item_confidence": confidence},
                "fingerprint": "",
                "created_at": _now_iso(),
                "last_updated": _now_iso(),
                "metadata": {
                    **stage_meta,
                    "strategy_observation": {
                        "strategy_type": strategy_type,
                        "why_failed_or_risky": str(row.get("why_failed_or_risky", ""))[:300],
                        "evidence": str(row.get("evidence", ""))[:240],
                    },
                    "task_summary": task_summary[:300],
                    "source_instance_id": source_instance_id,
                    "source_run_ids": [source_run_id] if source_run_id else [],
                    "source_attempt_ids": [source_attempt_id] if source_attempt_id else [],
                    "source_event_ids": [trace_id] if trace_id else [],
                    "changed_file_pattern": changed_file_pattern,
                    "critical_signal_rank": idx,
                    "subproblem_type": subproblem_type,
                    "strategy_label": strategy_label,
                    "prefer_actions": prefer_actions,
                    "avoid_actions": avoid_actions,
                    "applicability_scope": {
                        "subproblem_type": subproblem_type,
                        "strategy_label": strategy_label,
                    },
                },
            }
            payload["normalized_pattern_type"] = bucket
            if bucket == "validation_gap":
                payload["normalized_trigger_family"] = "missing_validation"
                payload["normalized_advice_family"] = "add_local_validation"
            elif bucket == "closure_signal":
                payload["normalized_trigger_family"] = "over_exploration_after_key_signal"
                payload["normalized_advice_family"] = "stop_expand_and_validate_minimal_fix"
            else:
                payload["normalized_trigger_family"] = "official_eval_unresolved"
                payload["normalized_advice_family"] = "avoid_repeat_failed_strategy"
            payload["family_id"] = build_experience_family_id(
                normalized_pattern_type=payload["normalized_pattern_type"],
                normalized_trigger_family=payload["normalized_trigger_family"],
                normalized_advice_family=payload["normalized_advice_family"],
            )
            out.append(payload)
        return out

    def _heuristic_strategy_observations(
        self,
        *,
        actions: List[ProblemFile],
        context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ctx = context if isinstance(context, dict) else {}
        step_count = int(ctx.get("step_count", 0) or len(actions))
        ad_hoc_names = [str(x).strip() for x in (ctx.get("ad_hoc_script_names") or []) if str(x).strip()]
        ad_hoc_count = int(ctx.get("ad_hoc_script_count", 0) or len(ad_hoc_names))
        validation_count = 0
        for action in actions:
            if action.action_type == ActionType.RUN_TEST:
                validation_count += 1
                continue
            if isinstance(action.action_text, str) and "pytest" in action.action_text.lower():
                validation_count += 1

        observations: List[Dict[str, Any]] = []
        if step_count >= 40 and validation_count >= 3:
            observations.append(
                {
                    "strategy_type": "closure_signal_over_exploration",
                    "why_failed_or_risky": "The attempt kept expanding validation and exploration after the core failing path had likely been identified.",
                    "evidence": f"step_count={step_count}, validation_count={validation_count}",
                    "recommended_avoidance": "Stop expanding investigation after the key failing path is confirmed; move to the minimal fix and targeted validation.",
                    "confidence": 0.74,
                }
            )
        if ad_hoc_count >= 2:
            observations.append(
                {
                    "strategy_type": "negative_strategy_ad_hoc_script_sprawl",
                    "why_failed_or_risky": "The attempt created multiple ad-hoc reproduction or test scripts, which tends to add local noise instead of converging on the minimal fix.",
                    "evidence": ", ".join(ad_hoc_names[:6]),
                    "recommended_avoidance": "Limit one-off reproduction scripts and prefer a single targeted validation path once the failing route is known.",
                    "confidence": 0.71,
                }
            )
        texts = " ".join(f"{action.intent_text} {action.action_text}" for action in actions).lower()
        if any(token in texts for token in ("world_to_pixel_values", "all_world2pix", "fitswcs")) and any(
            token in texts for token in ("wcsaxes", "utils.py", "visualization")
        ):
            observations.append(
                {
                    "strategy_type": "negative_strategy",
                    "why_failed_or_risky": "Search expanded into adjacent modules after the target WCS path was already localized.",
                    "recommended_avoidance": "avoid cross-module expansion after localizing the target fix surface",
                    "confidence": 0.74,
                }
            )
        if validation_count >= 3 and not any(action.action_type == ActionType.CODE_EDIT for action in actions):
            observations.append(
                {
                    "strategy_type": "negative_strategy",
                    "why_failed_or_risky": "Repeated validation happened before forming a concrete patch candidate.",
                    "recommended_avoidance": "avoid repeated target validation before preparing a minimal patch candidate",
                    "confidence": 0.71,
                }
            )
        return observations

    def _collect_evidence_refs(self, actions: List[ProblemFile], *, critical_step: int) -> List[str]:
        refs: List[str] = []
        for idx, action in enumerate(actions):
            step = action.step_index if isinstance(action.step_index, int) else idx
            if step < max(0, critical_step - 1):
                continue
            refs.append(f"action://{action.action_id}")
            if action.stderr_ref and action.stderr_ref.location:
                refs.append(action.stderr_ref.location)
            if action.stdout_ref and action.stdout_ref.location:
                refs.append(action.stdout_ref.location)
            for ptr in action.evidence_index[:2]:
                if ptr.location:
                    refs.append(ptr.location)
            if len(refs) >= 30:
                break
        return self._dedup(refs, limit=30)

    def _candidate_fix_actions(self, critical: CriticalSignal) -> List[str]:
        actions = [
            critical.correction_guidance.strip(),
            f"add_precheck_for_{_slug(critical.error_type)}_before_retry",
            "run_targeted_validation_after_fix_and_before_submission",
            f"verify_root_cause_{_slug(critical.root_cause)[:80]}_before_patch",
        ]
        for row in critical.cascading_effects[:20]:
            impact = _slug(str(row.get("impact", "cascade")))
            actions.append(f"break_cascade_at_{impact}_and_revalidate")
        return self._dedup([a for a in actions if a], limit=None)

    def _verification_commands(self, actions: List[ProblemFile], *, error_type: str) -> List[str]:
        out: List[str] = []
        if "test" in error_type:
            out.append("pytest -q")
        touched = [path for action in actions for path in action.touched_files]
        if any(path.endswith(".py") for path in touched):
            out.append("pytest -q")
        if any(path.endswith((".js", ".ts")) for path in touched):
            out.append("npm test -- --runInBand")
        if not out:
            out.append("pytest -q")
        return self._dedup(out, limit=3)

    def _success_conditions(self, critical: CriticalSignal) -> List[str]:
        out = [
            "validate_fix_with_targeted_check_before_submission",
            f"apply_strategy_shift_for_{_slug(critical.error_type)}",
        ]
        if critical.critical_module == "planning":
            out.append("include_explicit_test_plan_before_code_edit")
        return self._dedup(out, limit=5)

    def _tool_sequence(self, actions: List[ProblemFile]) -> List[str]:
        return [(a.action_family or a.action_type.value).lower() for a in actions[-6:]]

    def _changed_file_pattern(self, actions: List[ProblemFile]) -> List[str]:
        exts: List[str] = []
        for action in actions:
            for path in action.touched_files:
                m = re.search(r"\.([a-zA-Z0-9]+)$", path)
                if m:
                    exts.append(m.group(1).lower())
        counts: Dict[str, int] = {}
        for ext in exts:
            counts[ext] = counts.get(ext, 0) + 1
        ranked = sorted(counts.items(), key=lambda row: row[1], reverse=True)
        return [f".{ext}" for ext, _n in ranked[:4]]

    def _error_stage(self, critical: CriticalSignal) -> str:
        if critical.critical_module == "planning":
            return "planning"
        if "test" in critical.error_type:
            return "test"
        return "execution"

    @staticmethod
    def _dedup(items: List[str], *, limit: Optional[int]) -> List[str]:
        seen = set()
        out: List[str] = []
        for item in items:
            text = str(item).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
            if limit is not None and len(out) >= limit:
                break
        return out

    def _summarize_action(self, action: ProblemFile) -> str:
        subproblem = self._infer_subproblem_type(action=action)
        strategy = self._infer_strategy_label(actions=[action], context=None, subproblem_type=subproblem)
        template = self._action_template(action)
        detail = str(action.intent_text or action.action_text or "").strip()
        detail = re.sub(r"\s+", " ", detail)[:96]
        base = f"{subproblem}:{template}"
        if strategy and strategy != "unknown_strategy":
            base = f"{base}:{strategy}"
        if detail:
            return f"{base}:{detail}"
        return base
