"""
Failure card builder for unresolved task outcomes.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..core.experience_models import FailureCardV2
from ..core.problem_file import ActionType, Outcome, ProblemFile


class FailureCardBuilder:
    """Build FailureCardV2 from task actions and optional RCA report."""

    INFRA_ERROR_TYPES = {"environment_error", "tool_timeout", "permission_error", "timeout"}
    INFRA_TEXT_MARKERS = (
        "insufficient balance",
        "dockerpullerror",
        "docker pull",
        "docker build",
        "docker daemon",
        "daemon unavailable",
        "no space left on device",
        "sigbus",
    )

    def build_from_unresolved(
        self,
        *,
        task_id: str,
        actions: List[ProblemFile],
        task_summary: str = "",
        rca_report: Optional[Dict[str, Any]] = None,
    ) -> FailureCardV2:
        if not actions:
            return FailureCardV2(
                task_id=task_id,
                action_trace_snippet=[],
                error_signature={"error_type": "unknown", "reason": "no_actions"},
                candidate_fix_actions=["collect_additional_context_before_retry"],
                verification_commands=["pytest -q"],
                evidence_refs=[],
                confidence=0.1,
            )

        failed = [a for a in actions if a.outcome == Outcome.FAIL]
        pivot = failed[-1] if failed else actions[-1]

        error_signature = self._build_error_signature(pivot)
        action_trace = self._build_trace_snippet(actions)
        candidate_fix_actions = self._build_candidate_fix_actions(error_signature, actions)
        verification_commands = self._build_verification_commands(actions)
        evidence_refs = self._collect_evidence_refs(actions)

        root_cause_nodes: List[str] = []
        propagation_chain: List[str] = []
        error_module = "unknown"
        confidence = 0.35
        if isinstance(rca_report, dict):
            root_cause_nodes = [str(x) for x in (rca_report.get("root_cause_nodes") or []) if x]
            propagation_chain = [str(x) for x in (rca_report.get("propagation_chain") or []) if x]
            if not propagation_chain:
                all_chains = rca_report.get("propagation_chains") or []
                if all_chains and isinstance(all_chains[0], list):
                    propagation_chain = [str(x) for x in all_chains[0] if x]
            error_module = str(rca_report.get("error_module") or "unknown")
            confidence = max(confidence, float(rca_report.get("confidence", 0.0) or 0.0))

        return FailureCardV2(
            task_id=task_id,
            action_id=pivot.action_id,
            instance_id=pivot.instance_id,
            run_id=pivot.run_id,
            source_event=pivot.source_event,
            step_index=pivot.step_index,
            trace_id=pivot.trace_id,
            error_signature=error_signature,
            action_trace_snippet=action_trace,
            candidate_fix_actions=candidate_fix_actions,
            verification_commands=verification_commands,
            evidence_refs=evidence_refs,
            root_cause_nodes=root_cause_nodes,
            propagation_chain=propagation_chain,
            error_module=error_module,
            failure_class=self._classify_failure_class(error_signature, actions),
            confidence=min(0.95, max(confidence, 0.2)),
            status="unresolved",
            metadata={
                "task_summary": task_summary[:500],
                "action_count": len(actions),
                "failed_action_count": len(failed),
                "has_rca": bool(rca_report),
                "source_event_ids": [a.trace_id for a in actions if a.trace_id][:30],
                "source_instance_id": next((a.instance_id for a in actions if a.instance_id), ""),
                "source_run_ids": list(dict.fromkeys([a.run_id for a in actions if a.run_id]))[:6],
                "links": {"repair_pattern_ids": []},
            },
        )

    def _build_error_signature(self, action: ProblemFile) -> Dict[str, Any]:
        failure = action.failure_signature
        error_type = "unknown"
        error_tokens: List[str] = []
        if failure:
            error_type = failure.error_type or "unknown"
            error_tokens = list(failure.error_tokens or [])[:20]

        exec_stats = action.execution_stats or {}
        is_timeout = bool(exec_stats.get("is_timeout") or exec_stats.get("timeout"))
        exit_code = exec_stats.get("exit_code")
        nonzero_exit = bool(exec_stats.get("nonzero_exit")) or (
            isinstance(exit_code, int) and exit_code != 0
        )

        error_stage = "execution"
        if action.action_type == ActionType.RUN_TEST or (action.test_stats or {}).get("tests_failed", 0) > 0:
            error_stage = "test"

        return {
            "error_type": error_type,
            "error_stage": error_stage,
            "exit_code": exit_code,
            "is_timeout": is_timeout,
            "nonzero_exit": nonzero_exit,
            "error_tokens": error_tokens,
        }

    def _build_trace_snippet(self, actions: List[ProblemFile], max_steps: int = 6) -> List[str]:
        snippet: List[str] = []
        start = max(0, len(actions) - max_steps)
        for idx, action in enumerate(actions[start:], start=start):
            intent = (action.intent_text or "").strip().replace("\n", " ")
            if len(intent) > 120:
                intent = intent[:117] + "..."
            snippet.append(
                f"step={idx} type={action.action_type.value} outcome={action.outcome.value} intent={intent}"
            )
        return snippet

    def _build_candidate_fix_actions(
        self,
        error_signature: Dict[str, Any],
        actions: List[ProblemFile],
    ) -> List[str]:
        error_type = str(error_signature.get("error_type", "")).lower()
        candidates: List[str] = []
        if "import" in error_type:
            candidates.append("verify_dependency_and_import_path_before_retry")
        if "file" in error_type or "path" in error_type:
            candidates.append("validate_target_path_and_working_directory")
        if "assert" in error_type or "test" in error_type:
            candidates.append("run_related_tests_first_then_apply_targeted_fix")
        if error_signature.get("is_timeout"):
            candidates.append("reduce_scope_of_command_and_split_long_running_steps")

        repeated_actions = self._detect_repeated_action_families(actions)
        if repeated_actions:
            candidates.append("avoid_identical_retry_without_new_evidence")

        if not candidates:
            candidates.extend(
                [
                    "collect_failure_evidence_and_narrow_context",
                    "apply_small_fix_then_revalidate_with_targeted_test",
                ]
            )
        return candidates[:6]

    def _build_verification_commands(self, actions: List[ProblemFile]) -> List[str]:
        touched_files: List[str] = []
        for action in actions[-5:]:
            touched_files.extend(action.touched_files[:3])
        unique_files = list(dict.fromkeys(touched_files))

        commands: List[str] = []
        has_python = any(path.endswith(".py") for path in unique_files)
        has_js = any(path.endswith((".js", ".ts")) for path in unique_files)

        if has_python:
            commands.append("pytest -q")
        if has_js:
            commands.append("npm test -- --runInBand")

        if not commands:
            commands.append("pytest -q")
        return commands[:3]

    def _collect_evidence_refs(self, actions: List[ProblemFile], max_refs: int = 25) -> List[str]:
        refs: List[str] = []
        for action in reversed(actions):
            refs.append(action.action_id)
            for ptr in action.evidence_index[:3]:
                if ptr.location:
                    refs.append(ptr.location)
            if len(refs) >= max_refs:
                break
        # preserve order and dedup
        deduped: List[str] = []
        seen = set()
        for ref in refs:
            key = ref.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
            if len(deduped) >= max_refs:
                break
        return deduped

    def _detect_repeated_action_families(self, actions: List[ProblemFile]) -> bool:
        counts: Dict[str, int] = {}
        for action in actions:
            family = (action.action_family or action.action_type.value).strip().lower()
            if not family:
                continue
            family = re.sub(r"\s+", "_", family)
            counts[family] = counts.get(family, 0) + 1
            if counts[family] >= 3:
                return True
        return False

    def _classify_failure_class(
        self,
        error_signature: Dict[str, Any],
        actions: List[ProblemFile],
    ) -> str:
        error_type = str(error_signature.get("error_type", "")).strip().lower()
        if error_type in self.INFRA_ERROR_TYPES:
            return "infra_failure_card"

        snippets: List[str] = []
        for action in actions[-3:]:
            snippets.append(str(action.intent_text or ""))
            snippets.append(str(action.action_text or ""))
            stderr_excerpt = getattr(action.failure_signature, "stderr_excerpt", None)
            if stderr_excerpt:
                snippets.append(str(stderr_excerpt))
            meta = action.metadata if isinstance(action.metadata, dict) else {}
            for key in ("error_message", "stderr_excerpt", "raw_error"):
                val = meta.get(key)
                if isinstance(val, str):
                    snippets.append(val)

        body = " ".join(snippets).lower()
        if any(marker in body for marker in self.INFRA_TEXT_MARKERS):
            return "infra_failure_card"
        return "agent_failure_card"
