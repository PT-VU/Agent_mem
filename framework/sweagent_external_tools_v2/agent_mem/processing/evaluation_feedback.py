"""
Official evaluation feedback loop for AgentMem.

This module keeps the post-eval path explicit and lightweight:
- promote candidate memories on resolved
- suppress candidate memories on unresolved
- write small negative/validation memories from real external feedback
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from ..types import CompilerCardType
from ..storage.graph_store import GraphStore
from ..storage.episode_ledger_store import EpisodeLedgerStore
from .v21_shared import stable_patch_digest


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", (text or "").lower()).strip("_") or "unknown"


def _env_flag(name: str, default: str = "0") -> bool:
    """Resolve a v2 flag from the environment or the persisted run config.

    Reading the persisted config keeps subprocess behavior aligned with the
    launcher when environment variables are not forwarded explicitly.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        cfg_path = os.environ.get("AGENT_MEM_V2_CONFIG_FILE", "").strip()
        if not cfg_path:
            run_root = os.environ.get("RUN_ROOT", "").strip()
            if run_root:
                candidate = os.path.join(run_root, "agent_mem_logs", "v2_config.json")
                if os.path.isfile(candidate):
                    cfg_path = candidate
        if cfg_path and os.path.isfile(cfg_path):
            try:
                import json as _json
                with open(cfg_path, "r", encoding="utf-8") as fp:
                    data = _json.load(fp) or {}
                if name in data:
                    raw = str(data[name])
            except Exception:
                pass
    raw = (raw or default or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


_VERBATIM_DIFF_MAX_BYTES = 8 * 1024

# Non-code files/extensions that should never be treated as the representative
# file for a bug fix (e.g. AUTHORS, CHANGELOG, *.md).
_NON_CODE_BASENAMES: frozenset = frozenset({
    "AUTHORS", "CHANGELOG", "CHANGES", "NEWS", "NOTICE",
    "LICENSE", "COPYING", "CREDITS", "CONTRIBUTORS",
})
_NON_CODE_EXTENSIONS: frozenset = frozenset({
    ".md", ".rst", ".txt", ".cfg", ".ini",
    ".yaml", ".yml", ".json", ".xml", ".html", ".css",
})
_PREFERRED_CODE_EXTENSION = ".py"


class EvaluationFeedbackProcessor:
    """Maps official eval outcomes back into existing memory objects."""

    def __init__(
        self,
        graph_store: GraphStore,
        *,
        governance_policy: Any = None,
        episode_ledger_store: Optional[EpisodeLedgerStore] = None,
    ):
        self.graph_store = graph_store
        self.governance_policy = governance_policy
        self.episode_ledger_store = episode_ledger_store

    def apply_feedback(
        self,
        *,
        instance_id: str,
        outcome: str,
        eval_ref: str = "",
        patch_text: str = "",
        patch_summary: Optional[Dict[str, Any]] = None,
        changed_files: Optional[List[str]] = None,
        validation_summary: Optional[Dict[str, Any]] = None,
        task_summary: str = "",
        run_id: str = "",
        attempt_id: str = "",
        trace_id: str = "",
    ) -> Dict[str, Any]:
        outcome_norm = self._normalize_outcome(outcome)
        related = self.graph_store.list_candidate_abstract_experiences(
            instance_id=instance_id,
            run_id=run_id,
            attempt_id=attempt_id,
            trace_id=trace_id,
            max_results=200,
        )
        related_ids = [str(row.get("experience_id")) for row in related if str(row.get("experience_id", "")).strip()]

        report: Dict[str, Any] = {
            "instance_id": instance_id,
            "outcome": outcome_norm,
            "eval_ref": eval_ref,
            "related_candidate_count": len(related_ids),
            "related_candidate_ids": list(related_ids),
            "promoted_ids": [],
            "suppressed_ids": [],
            "written_ids": [],
            "written_card_ids": [],
            "failure_card_id": None,
            "patch_summary": {},
            "promoted_card_ids": [],
            "suppressed_card_ids": [],
            "subtask_eval_updates": {"subtasks": 0, "subtask_edges": 0},
        }

        if outcome_norm == "resolved":
            summary = self._build_patch_summary(
                patch_text=patch_text,
                changed_files=changed_files or [],
                validation_summary=validation_summary or {},
                supplied_summary=patch_summary or {},
            )
            report["patch_summary"] = summary
            for exp_id in related_ids:
                if self.graph_store.mark_experience_promoted(exp_id, eval_ref=eval_ref):
                    report["promoted_ids"].append(exp_id)
            success_card_id = self._write_success_path_card(
                instance_id=instance_id,
                eval_ref=eval_ref,
                patch_text=patch_text,
                summary=summary,
                validation_summary=validation_summary or {},
                task_summary=task_summary,
                run_id=run_id,
                attempt_id=attempt_id,
                trace_id=trace_id,
            )
            if success_card_id:
                report["written_card_ids"].append(success_card_id)
            bug_invariant_card_id = self._write_bug_invariant_card(
                instance_id=instance_id,
                eval_ref=eval_ref,
                patch_text=patch_text,
                summary=summary,
                validation_summary=validation_summary or {},
                task_summary=task_summary,
                run_id=run_id,
                attempt_id=attempt_id,
                trace_id=trace_id,
            )
            if bug_invariant_card_id:
                report["written_card_ids"].append(bug_invariant_card_id)
            self._apply_v21_feedback(
                report=report,
                instance_id=instance_id,
                run_id=run_id,
                attempt_id=attempt_id,
                outcome=outcome_norm,
                eval_ref=eval_ref,
            )
            return report

        if outcome_norm == "infra_failure":
            card_id = self.graph_store.upsert_failure_card_v2(
                {
                    "task_id": f"official_eval::{instance_id}",
                    "instance_id": instance_id,
                    "run_id": run_id or None,
                    "trace_id": trace_id or None,
                    "source_event": "official_eval_feedback",
                    "error_signature": {"error_type": "environment_error"},
                    "candidate_fix_actions": ["stabilize_runtime_environment_before_retrying"],
                    "verification_commands": ["docker info"],
                    "failure_class": "infra_failure_card",
                    "status": "incomplete",
                    "confidence": 0.9,
                    "metadata": {
                        "experience_polarity": "neutral",
                        "evaluation_outcome": outcome_norm,
                        "official_eval_refs": [eval_ref] if eval_ref else [],
                        "task_summary": task_summary[:280],
                    },
                }
            )
            report["failure_card_id"] = card_id
            self._apply_v21_feedback(
                report=report,
                instance_id=instance_id,
                run_id=run_id,
                attempt_id=attempt_id,
                outcome=outcome_norm,
                eval_ref=eval_ref,
            )
            return report

        for exp_id in related_ids:
            if self.graph_store.mark_experience_suppressed(
                exp_id,
                eval_ref=eval_ref,
                reason=f"official_eval_{outcome_norm}",
            ):
                report["suppressed_ids"].append(exp_id)

        summary = self._build_patch_summary(
            patch_text=patch_text,
            changed_files=changed_files or [],
            validation_summary=validation_summary or {},
            supplied_summary=patch_summary or {},
        )
        report["patch_summary"] = summary

        for payload in self._build_negative_experiences(
            instance_id=instance_id,
            outcome=outcome_norm,
            eval_ref=eval_ref,
            summary=summary,
            task_summary=task_summary,
            run_id=run_id,
            attempt_id=attempt_id,
            trace_id=trace_id,
        ):
            exp_id = self.graph_store.upsert_abstract_experience(payload)
            self.graph_store.mark_experience_promoted(exp_id, eval_ref=eval_ref)
            report["written_ids"].append(exp_id)

        if outcome_norm == "incomplete":
            timeout_card_id = self._write_timeout_governance_card(
                instance_id=instance_id,
                eval_ref=eval_ref,
                summary=summary,
                task_summary=task_summary,
                run_id=run_id,
                attempt_id=attempt_id,
                trace_id=trace_id,
            )
            if timeout_card_id:
                report["written_card_ids"].append(timeout_card_id)

        if outcome_norm == "unresolved":
            anti_anchors: List[Dict[str, Any]] = []
            anti_key_lines: List[str] = []
            anti_signature_hash = ""
            if _env_flag("AGENT_MEM_BUG_ANTI_PATTERN", "0") and patch_text.strip():
                anti_anchors = self._extract_anchors(patch_text)
                anti_key_lines = self._extract_key_added_lines(patch_text)
                anti_signature_hash = self._compute_signature_hash(
                    anchors=anti_anchors, key_added_lines=anti_key_lines
                )

            error_signature: Dict[str, Any] = {"error_type": "evaluation_unresolved"}
            if anti_signature_hash:
                error_signature["patch_signature_hash"] = anti_signature_hash

                primary_params: List[str] = []
                for a in anti_anchors:
                    if a.get("symbol_kind") == "function" and a.get("param_signature"):
                        primary_params = list(a.get("param_signature") or [])
                        break
                if primary_params:
                    error_signature["param_signature_hash"] = (
                        "sha1:" + hashlib.sha1(",".join(primary_params).encode("utf-8")).hexdigest()
                    )

            card_id = self.graph_store.upsert_failure_card_v2(
                {
                    "task_id": f"official_eval::{instance_id}",
                    "instance_id": instance_id,
                    "run_id": run_id or None,
                    "trace_id": trace_id or None,
                    "source_event": "official_eval_feedback",
                    "error_signature": error_signature,
                    "candidate_fix_actions": self._candidate_fix_actions(summary),
                    "verification_commands": self._verification_commands(summary),
                    "confidence": 0.76,
                    "status": "unresolved",
                    "metadata": {
                        "evaluation_outcome": outcome_norm,
                        "official_eval_refs": [eval_ref] if eval_ref else [],
                        "task_summary": task_summary[:280],
                        "patch_summary": summary,
                    },
                }
            )
            report["failure_card_id"] = card_id

            anti_card_id = self._write_bug_anti_pattern_card(
                instance_id=instance_id,
                eval_ref=eval_ref,
                patch_text=patch_text,
                summary=summary,
                validation_summary=validation_summary or {},
                task_summary=task_summary,
                run_id=run_id,
                attempt_id=attempt_id,
                trace_id=trace_id,
                official_eval_status=outcome_norm,
            )
            if anti_card_id:
                report["written_card_ids"].append(anti_card_id)

        self._apply_v21_feedback(
            report=report,
            instance_id=instance_id,
            run_id=run_id,
            attempt_id=attempt_id,
            outcome=outcome_norm,
            eval_ref=eval_ref,
        )

        return report

    def _apply_v21_feedback(
        self,
        *,
        report: Dict[str, Any],
        instance_id: str,
        run_id: str,
        attempt_id: str,
        outcome: str,
        eval_ref: str,
    ) -> None:
        if self.governance_policy is not None:
            cards = self.graph_store.list_candidate_compiler_cards(
                instance_id=instance_id,
                run_id=run_id,
                attempt_id=attempt_id,
                max_results=200,
            )
            if cards:
                card_update = self.governance_policy.apply_official_feedback(
                    compiler_cards=cards,
                    outcome=outcome,
                    eval_ref=eval_ref,
                )
                for row in card_update.get("cards", []):
                    card_id = str(row.get("card_id") or "").strip()
                    if not card_id:
                        continue
                    self.graph_store.update_compiler_card_state(
                        card_id,
                        promotion_state=str(row.get("promotion_state") or ""),
                        eval_ref=eval_ref,
                        reason=f"official_eval_{outcome}",
                    )
                report["promoted_card_ids"] = list(card_update.get("promoted_ids", []) or [])
                report["suppressed_card_ids"] = list(card_update.get("suppressed_ids", []) or [])

        if self.episode_ledger_store is None or self.governance_policy is None:
            return
        filters: Dict[str, Any] = {"instance_id": instance_id}
        if attempt_id:
            filters["attempt_id"] = attempt_id
        if run_id:
            filters["run_id"] = run_id
        subtasks = self.episode_ledger_store.load_latest_records(
            stream="subtask_instances",
            key_field="subtask_instance_id",
            filters=filters,
        )
        edges = self.episode_ledger_store.load_latest_records(
            stream="subtask_edges",
            key_field="edge_id",
            filters=filters,
        )
        if not subtasks and not edges:
            return
        attachment = self.governance_policy.attach_eval_context(
            subtasks=subtasks,
            subtask_edges=edges,
            outcome=outcome,
            eval_ref=eval_ref,
        )
        subtask_report = self.episode_ledger_store.append_batch(
            [
                {
                    **row,
                    "record_id": stable_patch_digest(
                        {
                            "stream": "subtask_instances",
                            "logical_id": str(row.get("subtask_instance_id") or ""),
                            "eval_ref": eval_ref,
                            "outcome": outcome,
                            "payload": row,
                        }
                    ),
                }
                for row in attachment.get("subtasks", [])
            ],
            stream="subtask_instances",
        )
        edge_report = self.episode_ledger_store.append_batch(
            [
                {
                    **row,
                    "record_id": stable_patch_digest(
                        {
                            "stream": "subtask_edges",
                            "logical_id": str(row.get("edge_id") or ""),
                            "eval_ref": eval_ref,
                            "outcome": outcome,
                            "payload": row,
                        }
                    ),
                }
                for row in attachment.get("subtask_edges", [])
            ],
            stream="subtask_edges",
        )
        report["subtask_eval_updates"] = {
            "subtasks": int(subtask_report.get("written", 0) or 0),
            "subtask_edges": int(edge_report.get("written", 0) or 0),
        }

    def _write_success_path_card(
        self,
        *,
        instance_id: str,
        eval_ref: str,
        patch_text: str,
        summary: Dict[str, Any],
        validation_summary: Dict[str, Any],
        task_summary: str,
        run_id: str,
        attempt_id: str,
        trace_id: str,
    ) -> Optional[str]:
        changed_files = [str(x).strip() for x in (summary.get("changed_files") or []) if str(x).strip()]
        validation_commands = [str(x).strip() for x in (summary.get("validation_commands") or []) if str(x).strip()]
        if not patch_text.strip() and not changed_files:
            return None
        patch_digest = stable_patch_digest(
            {
                "instance_id": instance_id,
                "patch_text": patch_text,
                "changed_files": changed_files,
            }
        )
        patch_family = f"{instance_id}::{patch_digest[:12]}"
        repro_path, target_validation = self._split_validation_paths(validation_commands)
        submit_preconditions = ["target_validation_passed"]
        if changed_files:
            submit_preconditions.append("same_patch_family_revalidated")
        hint = self._build_success_path_hint(changed_files=changed_files, target_validation=target_validation)
        family_id = stable_patch_digest(
            {
                "card_type": CompilerCardType.SUCCESS_PATH.value,
                "instance_id": instance_id,
                "patch_family": patch_family,
            }
        )
        summary_id = self._find_attempt_summary_id(instance_id=instance_id, run_id=run_id, attempt_id=attempt_id)
        return self._upsert_compiler_card_with_support(
            {
                "card_id": stable_patch_digest(
                    {
                        "compiler_card": CompilerCardType.SUCCESS_PATH.value,
                        "family_id": family_id,
                    }
                ),
                "card_type": CompilerCardType.SUCCESS_PATH.value,
                "family_id": family_id,
                "instance_id": instance_id,
                "run_id": run_id or None,
                "attempt_id": attempt_id or None,
                "trace_id": trace_id or None,
                "hint": hint,
                "recommendation": hint,
                "confidence": 0.9,
                "support_count": 1,
                "instance_scope": "same-case",
                "patch_family": patch_family,
                "repro_path": repro_path[:2],
                "target_validation": target_validation[:2],
                "submit_preconditions": submit_preconditions[:3],
                "prefer_actions": [
                    "reuse_confirmed_repro_path",
                    "run_target_validation_before_submission",
                    "promote_minimal_fix_candidate_earlier",
                ],
                "avoid_actions": [],
                "subproblem_type": "form_minimal_patch",
                "strategy_label": "minimal_patch_then_target_validation",
                "governance_hardness": "strong",
                "changed_file_pattern": [
                    f".{path.rsplit('.', 1)[-1].lower()}"
                    for path in changed_files
                    if "." in path
                ][:4],
                "evidence_level": "official",
                "promotion_state": "promoted",
                "source_object_ids": [summary_id] if summary_id else [],
                "evidence_refs": [eval_ref] if eval_ref else [],
                "metadata": {
                    "evidence_stage": "official_eval",
                    "official_eval_refs": [eval_ref] if eval_ref else [],
                    "validation_summary": dict(validation_summary or {}),
                    "patch_summary": dict(summary),
                    "task_summary": task_summary[:280],
                    "governance_hardness": "strong",
                    "subproblem_type": "form_minimal_patch",
                    "strategy_label": "minimal_patch_then_target_validation",
                },
            },
            support_key=self._support_key(run_id=run_id, attempt_id=attempt_id, eval_ref=eval_ref),
        )

    def _write_timeout_governance_card(
        self,
        *,
        instance_id: str,
        eval_ref: str,
        summary: Dict[str, Any],
        task_summary: str,
        run_id: str,
        attempt_id: str,
        trace_id: str,
    ) -> Optional[str]:
        changed_files = [str(x).strip() for x in (summary.get("changed_files") or []) if str(x).strip()]
        validation_commands = [str(x).strip() for x in (summary.get("validation_commands") or []) if str(x).strip()]
        if changed_files and validation_commands:
            return None
        budget_hints = {
            "max_new_repro_scripts_after_repro_confirmed": 0,
            "max_broad_tests_before_patch": 0,
            "max_target_validation_retries_without_patch_upgrade": 1,
        }
        family_id = stable_patch_digest(
            {
                "card_type": CompilerCardType.TIMEOUT_GOVERNANCE.value,
                "instance_id": instance_id,
                "failure_family": "ad_hoc_repro_script_loop",
            }
        )
        summary_id = self._find_attempt_summary_id(instance_id=instance_id, run_id=run_id, attempt_id=attempt_id)
        hint = (
            "Timeout governance: once reproduction is confirmed, do not create new repro scripts or run broad "
            "regression before a real patch candidate exists. Reuse the confirmed repro path and either submit the "
            "minimal fix candidate after target validation or stop early."
        )
        return self._upsert_compiler_card_with_support(
            {
                "card_id": stable_patch_digest(
                    {
                        "compiler_card": CompilerCardType.TIMEOUT_GOVERNANCE.value,
                        "family_id": family_id,
                    }
                ),
                "card_type": CompilerCardType.TIMEOUT_GOVERNANCE.value,
                "family_id": family_id,
                "instance_id": instance_id,
                "run_id": run_id or None,
                "attempt_id": attempt_id or None,
                "trace_id": trace_id or None,
                "hint": hint,
                "recommendation": hint,
                "confidence": 0.86,
                "support_count": 1,
                "failure_family": "ad_hoc_repro_script_loop",
                "trigger_scope": "same-case",
                "prefer_actions": [
                    "reuse_confirmed_repro_path",
                    "promote_minimal_fix_candidate_earlier",
                    "run_target_validation_before_submission",
                ],
                "avoid_actions": [
                    "create_new_repro_script_after_repro_confirmed",
                    "run_broad_regression_before_patch_candidate_exists",
                ],
                "budget_hints": budget_hints,
                "normalized_pattern_type": "negative_strategy",
                "subproblem_type": "reproduce_issue",
                "strategy_label": "ad_hoc_repro_script_loop",
                "governance_hardness": "guardrail",
                "evidence_level": "official",
                "promotion_state": "promoted",
                "source_object_ids": [summary_id] if summary_id else [],
                "evidence_refs": [eval_ref] if eval_ref else [],
                "metadata": {
                    "evidence_stage": "official_eval",
                    "official_eval_refs": [eval_ref] if eval_ref else [],
                    "task_summary": task_summary[:280],
                    "budget_hints": budget_hints,
                    "governance_hardness": "guardrail",
                    "subproblem_type": "reproduce_issue",
                    "strategy_label": "ad_hoc_repro_script_loop",
                    "runtime_guard": {
                        "blocked_action_patterns": [
                            "create_new_repro_script_after_repro_confirmed",
                            "run_broad_regression_before_patch_candidate_exists",
                        ]
                    },
                },
            },
            support_key=self._support_key(run_id=run_id, attempt_id=attempt_id, eval_ref=eval_ref),
        )

    def _write_bug_invariant_card(
        self,
        *,
        instance_id: str,
        eval_ref: str,
        patch_text: str,
        summary: Dict[str, Any],
        validation_summary: Dict[str, Any],
        task_summary: str,
        run_id: str,
        attempt_id: str,
        trace_id: str,
    ) -> Optional[str]:
        changed_files = [str(x).strip() for x in (summary.get("changed_files") or []) if str(x).strip()]
        if not patch_text.strip() and not changed_files:
            return None
        change_type = self._infer_change_type(patch_text)
        key_lines = self._extract_key_added_lines_preferring_code(patch_text)
        target_function = self._extract_target_function(patch_text)
        validation_invariant = self._extract_validation_invariant(validation_summary)
        bug_semantic = self._build_bug_semantic(
            target_function=target_function,
            change_type=change_type,
            changed_files=changed_files,
        )
        key_line_hint = key_lines[0] if key_lines else "see diff"
        hint = (
            f"[BugInvariant] Fix: {bug_semantic}. "
            f"Key change: {key_line_hint[:80]}. "
            f"Validate: {validation_invariant}."
        )[:300]
        family_id = stable_patch_digest(
            {
                "card_type": CompilerCardType.BUG_INVARIANT.value,
                "instance_id": instance_id,
                "changed_files": changed_files[:2],
            }
        )

        v2_enabled = _env_flag("AGENT_MEM_BUG_INVARIANT_VERBATIM", "0")
        anchors: List[Dict[str, Any]] = []
        verbatim_diff = ""
        signature_hash = ""
        if v2_enabled and patch_text.strip():
            anchors = self._extract_anchors(patch_text)
            verbatim_diff = self._normalize_diff_for_storage(patch_text)
            signature_hash = self._compute_signature_hash(anchors=anchors, key_added_lines=key_lines)

        minimal_patch_signature: Dict[str, Any] = {
            "files_touched": changed_files[:3],
            "change_type": change_type,
            "key_lines": key_lines[:3],
        }
        if v2_enabled:
            minimal_patch_signature.update(
                {
                    "anchors": anchors,
                    "key_added_lines": key_lines[:8],
                    "key_added_lines_hash": signature_hash,
                    "verbatim_diff": verbatim_diff,
                }
            )

        validation_invariant_v2: Any = validation_invariant
        if v2_enabled:
            commands = (
                validation_summary.get("commands")
                or validation_summary.get("verification_commands")
                or []
            )
            validation_invariant_v2 = {
                "primary": validation_invariant,
                "commands": [str(c).strip() for c in commands if str(c).strip()][:6],
            }

        card_payload: Dict[str, Any] = {
            "card_id": stable_patch_digest(
                {
                    "compiler_card": CompilerCardType.BUG_INVARIANT.value,
                    "family_id": family_id,
                }
            ),
            "card_type": CompilerCardType.BUG_INVARIANT.value,
            "family_id": family_id,
            "instance_id": instance_id,
            "run_id": run_id or None,
            "attempt_id": attempt_id or None,
            "trace_id": trace_id or None,
            "hint": hint,
            "recommendation": hint,
            "confidence": 0.95,
            "support_count": 1,
            "instance_scope": "same-case",
            "bug_semantic": bug_semantic,
            "minimal_patch_signature": minimal_patch_signature,
            "validation_invariant": validation_invariant_v2,
            "prefer_actions": [
                "apply_same_patch_family",
                "run_target_validation_before_submission",
            ],
            "avoid_actions": [],
            "governance_hardness": "strong",
            "evidence_level": "official",
            "promotion_state": "promoted",
            "source_object_ids": [],
            "evidence_refs": [eval_ref] if eval_ref else [],
            "metadata": {
                "evidence_stage": "official_eval",
                "official_eval_refs": [eval_ref] if eval_ref else [],
                "bug_semantic": bug_semantic,
                "minimal_patch_signature": minimal_patch_signature,
                "validation_invariant": validation_invariant_v2,
                "task_summary": task_summary[:280],
                "governance_hardness": "strong",
                "schema_version": "1.1" if v2_enabled else "1.0",
                "v2_features_enabled": v2_enabled,
            },
        }
        if v2_enabled:
            card_payload["reuse_template"] = {
                "mode": "verbatim_first_then_anchor_align",
                "fallback_anchor": anchors[0] if anchors else None,
            }
            card_payload["signature_hash"] = signature_hash

        return self._upsert_compiler_card_with_support(
            card_payload,
            support_key=self._support_key(run_id=run_id, attempt_id=attempt_id, eval_ref=eval_ref),
        )

    # ---------------- v2 helpers for Modules A and B ----------------

    @staticmethod
    def _normalize_diff_for_storage(patch_text: str) -> str:
        """Normalize a unified diff and cap storage at 8 KiB."""
        text = patch_text or ""
        cleaned_lines = []
        for line in text.splitlines():
            if line.startswith("--- ") or line.startswith("+++ "):
                cleaned_lines.append(line.split("\t", 1)[0])
            else:
                cleaned_lines.append(line)
        cleaned = "\n".join(cleaned_lines)
        encoded = cleaned.encode("utf-8", errors="replace")
        if len(encoded) > _VERBATIM_DIFF_MAX_BYTES:
            cleaned = encoded[:_VERBATIM_DIFF_MAX_BYTES].decode("utf-8", errors="replace") + "\n... [truncated]\n"
        return cleaned

    @staticmethod
    def _extract_anchors(patch_text: str) -> List[Dict[str, Any]]:
        """Extract file and symbol anchors from a unified diff.

        Python anchors use AST parsing when possible and hunk context as a
        fallback.
        """
        anchors: List[Dict[str, Any]] = []
        if not patch_text:
            return anchors
        blocks: List[Tuple[str, List[str]]] = []
        current_file = ""
        current_lines: List[str] = []
        for line in patch_text.splitlines():
            if line.startswith("diff --git "):
                if current_file:
                    blocks.append((current_file, current_lines))
                current_file = ""
                current_lines = []
                m = re.search(r" b/(\S+)", line)
                if m:
                    current_file = m.group(1)
            elif line.startswith("+++ "):
                # +++ b/path
                m = re.match(r"\+\+\+\s+b/(\S+)", line)
                if m and not current_file:
                    current_file = m.group(1)
            else:
                current_lines.append(line)
        if current_file:
            blocks.append((current_file, current_lines))

        for file_path, lines in blocks:
            added = "\n".join(
                ln[1:] for ln in lines if ln.startswith("+") and not ln.startswith("+++")
            )
            file_anchors = EvaluationFeedbackProcessor._extract_anchors_for_file(file_path, added, lines)
            anchors.extend(file_anchors)
        return anchors[:8]

    @staticmethod
    def _extract_anchors_for_file(file_path: str, added_block: str, raw_lines: List[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if file_path.endswith(".py") and added_block.strip():
            try:
                tree = ast.parse(added_block)
                parent_class: Optional[str] = None
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        parent_class = node.name
                        results.append(
                            {
                                "file": file_path,
                                "symbol_kind": "class",
                                "symbol_name": node.name,
                                "parent": None,
                                "param_signature": [],
                            }
                        )
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        params = [a.arg for a in node.args.args]
                        results.append(
                            {
                                "file": file_path,
                                "symbol_kind": "function",
                                "symbol_name": node.name,
                                "parent": parent_class,
                                "param_signature": params,
                            }
                        )
            except SyntaxError:
                pass
        if not results:
            for line in raw_lines:
                m = re.search(r"@@ [^@]+ @@\s*(.*)", line)
                if m:
                    ctx = m.group(1).strip()
                    fn = re.search(r"def\s+(\w+)\s*\(([^)]*)\)", ctx)
                    if fn:
                        params = [p.strip().split("=")[0].split(":")[0].strip() for p in fn.group(2).split(",") if p.strip()]
                        results.append(
                            {
                                "file": file_path,
                                "symbol_kind": "function",
                                "symbol_name": fn.group(1),
                                "parent": None,
                                "param_signature": params,
                            }
                        )
                if line.startswith("+") and "def " in line:
                    fn = re.search(r"def\s+(\w+)\s*\(([^)]*)\)", line)
                    if fn:
                        params = [p.strip().split("=")[0].split(":")[0].strip() for p in fn.group(2).split(",") if p.strip()]
                        results.append(
                            {
                                "file": file_path,
                                "symbol_kind": "function",
                                "symbol_name": fn.group(1),
                                "parent": None,
                                "param_signature": params,
                            }
                        )
        if not results:
            results.append(
                {
                    "file": file_path,
                    "symbol_kind": "file",
                    "symbol_name": file_path.rsplit("/", 1)[-1],
                    "parent": None,
                    "param_signature": [],
                }
            )
        return results

    @staticmethod
    def _compute_signature_hash(*, anchors: List[Dict[str, Any]], key_added_lines: List[str]) -> str:
        """Compute a stable SHA-1 signature from anchors and added lines."""
        parts: List[str] = []
        for a in anchors[:6]:
            parts.append(
                "::".join(
                    [
                        str(a.get("file") or ""),
                        str(a.get("symbol_kind") or ""),
                        str(a.get("symbol_name") or ""),
                        str(a.get("parent") or ""),
                        ",".join(str(x) for x in (a.get("param_signature") or [])),
                    ]
                )
            )
        for ln in key_added_lines[:6]:
            parts.append("ln::" + str(ln).strip())
        joined = "\n".join(parts)
        return "sha1:" + hashlib.sha1(joined.encode("utf-8", errors="replace")).hexdigest()

    # ---------------- v2 Module B: BugAntiPatternCard ----------------

    def _write_bug_anti_pattern_card(
        self,
        *,
        instance_id: str,
        eval_ref: str,
        patch_text: str,
        summary: Dict[str, Any],
        validation_summary: Dict[str, Any],
        task_summary: str,
        run_id: str,
        attempt_id: str,
        trace_id: str,
        official_eval_status: str,
    ) -> Optional[str]:
        """unresolved
        gated by AGENT_MEM_BUG_ANTI_PATTERN.
        """
        if not _env_flag("AGENT_MEM_BUG_ANTI_PATTERN", "0"):
            return None
        changed_files = [str(x).strip() for x in (summary.get("changed_files") or []) if str(x).strip()]
        if not patch_text.strip() and not changed_files:
            return None
        anchors = self._extract_anchors(patch_text) if patch_text.strip() else []
        key_lines = self._extract_key_added_lines(patch_text)
        verbatim_diff = self._normalize_diff_for_storage(patch_text)
        signature_hash = self._compute_signature_hash(anchors=anchors, key_added_lines=key_lines)

        # Use (instance, signature_hash) as the stable family identity.
        family_id = stable_patch_digest(
            {
                "card_type": CompilerCardType.BUG_ANTI_PATTERN.value,
                "instance_id": instance_id,
                "signature_hash": signature_hash,
            }
        )
        card_id = stable_patch_digest(
            {
                "compiler_card": CompilerCardType.BUG_ANTI_PATTERN.value,
                "family_id": family_id,
            }
        )
        existing = self.graph_store.compiler_cards_v21.get(card_id) or {}
        support_n = int(existing.get("support_count", 0) or 0) + 1
        confidence = min(0.6 + 0.15 * (support_n - 1), 0.95)
        hardness = "strong" if support_n >= 2 else "guardrail"

        target_function = self._extract_target_function(patch_text)
        primary_param_sig: List[str] = []
        if anchors:
            for a in anchors:
                if a.get("symbol_kind") == "function" and a.get("param_signature"):
                    primary_param_sig = list(a.get("param_signature") or [])
                    break

        first_failed_line = key_lines[0][:100] if key_lines else ""
        hint = (
            f"[BugAntiPattern  failed {support_n}x] Avoid this submission shape on instance {instance_id}: "
            f"function `{target_function}` with params {primary_param_sig}. "
            f"Failed key change: {first_failed_line}."
        )[:300]

        failed_signature = {
            "files_touched": changed_files[:3],
            "anchors": anchors,
            "key_added_lines": key_lines[:8],
            "key_added_lines_hash": signature_hash,
            "param_signature": primary_param_sig,
            "verbatim_diff": verbatim_diff,
        }

        failure_reason = {
            "official_eval_status": official_eval_status or "unresolved",
            "failing_tests": (validation_summary or {}).get("failing_tests", []),
            "stderr_excerpt": (validation_summary or {}).get("stderr", "")[:2048],
        }

        payload = {
            "card_id": card_id,
            "card_type": CompilerCardType.BUG_ANTI_PATTERN.value,
            "family_id": family_id,
            "instance_id": instance_id,
            "run_id": run_id or None,
            "attempt_id": attempt_id or None,
            "trace_id": trace_id or None,
            "hint": hint,
            "recommendation": hint,
            "confidence": confidence,
            "support_count": support_n,
            "instance_scope": "same-case",
            "failed_patch_signature": failed_signature,
            "failure_reason": failure_reason,
            "prefer_actions": [],
            "avoid_actions": [
                "do_not_replicate_failed_patch_signature",
                "do_not_submit_with_same_param_signature",
            ],
            "governance_hardness": hardness,
            "evidence_level": "official",
            "promotion_state": "promoted",
            "source_object_ids": [],
            "evidence_refs": [eval_ref] if eval_ref else [],
            "signature_hash": signature_hash,
            "metadata": {
                "evidence_stage": "official_eval",
                "official_eval_refs": [eval_ref] if eval_ref else [],
                "failed_patch_signature": failed_signature,
                "failure_reason": failure_reason,
                "task_summary": task_summary[:280],
                "governance_hardness": hardness,
                "schema_version": "1.0",
                "v2_features_enabled": True,
                "applies_to_instance": instance_id,
            },
        }
        return self._upsert_compiler_card_with_support(
            payload,
            support_key=self._support_key(run_id=run_id, attempt_id=attempt_id, eval_ref=eval_ref),
        )

    @staticmethod
    def _infer_change_type(patch_text: str) -> str:
        if not patch_text:
            return "unknown"
        added = "\n".join(
            line[1:] for line in patch_text.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        if re.search(r"\bif\b[^:]*:", added):
            return "condition_fix"
        if re.search(r"\braise\b|\bassert\b", added):
            return "missing_check"
        if re.search(r"\breturn\b", added):
            return "wrong_return"
        if re.search(r"\bdef\b|\bclass\b", added):
            return "new_function_or_class"
        return "code_change"

    @staticmethod
    def _extract_key_added_lines(patch_text: str) -> List[str]:
        lines = [
            line[1:].strip()
            for line in patch_text.splitlines()
            if line.startswith("+") and not line.startswith("+++")
            and line[1:].strip()
            and not line[1:].strip().startswith("#")
        ]
        return [line for line in lines if len(line) > 5][:3]

    @staticmethod
    def _pick_representative_file(changed_files: List[str]) -> str:
        """Return the most representative file, preferring .py over non-code files.

        Skips well-known non-code files (AUTHORS, CHANGELOG, *.md, *.txt, etc.)
        when a code file is available.  Falls back to changed_files[0] if every
        file is non-code.
        """
        if not changed_files:
            return "target file"

        def _score(path: str) -> int:
            basename = os.path.basename(path).upper().split(".")[0]
            ext = os.path.splitext(path)[1].lower()
            if os.path.basename(path).upper() in _NON_CODE_BASENAMES:
                return 0
            if ext in _NON_CODE_EXTENSIONS:
                return 0
            if ext == _PREFERRED_CODE_EXTENSION:
                return 3
            if ext:
                return 2
            return 1

        return max(changed_files, key=_score)

    @staticmethod
    def _extract_key_added_lines_preferring_code(patch_text: str) -> List[str]:
        """Like _extract_key_added_lines but skips diff sections for non-code files.

        Parses the unified diff into per-file sections (split on 'diff --git')
        and discards sections whose file path belongs to _NON_CODE_BASENAMES or
        has a _NON_CODE_EXTENSIONS extension before collecting added lines.
        Falls back to the plain extractor if no code-file section exists.
        """
        sections: List[str] = re.split(r"(?=^diff --git )", patch_text, flags=re.MULTILINE)
        code_sections: List[str] = []
        for section in sections:
            m = re.match(r"diff --git a/(\S+)", section)
            if m:
                path = m.group(1)
                if os.path.basename(path).upper() in _NON_CODE_BASENAMES:
                    continue
                if os.path.splitext(path)[1].lower() in _NON_CODE_EXTENSIONS:
                    continue
            code_sections.append(section)

        has_code = any(s.strip() for s in code_sections)
        source = "".join(code_sections) if has_code else patch_text
        lines = [
            line[1:].strip()
            for line in source.splitlines()
            if line.startswith("+") and not line.startswith("+++")
            and line[1:].strip()
            and not line[1:].strip().startswith("#")
        ]
        return [line for line in lines if len(line) > 5][:3]

    @staticmethod
    def _extract_target_function(patch_text: str) -> str:
        # Try @@ header context first (function or class name)
        for m in re.finditer(r"@@ [^@]+ @@ (.*)", patch_text):
            context = m.group(1).strip()
            if not context:
                continue
            fn = re.search(r"def\s+(\w+)", context)
            if fn:
                return fn.group(1)
            cls = re.search(r"class\s+(\w+)", context)
            if cls:
                return cls.group(1)
        # Fall back to added lines
        for line in patch_text.splitlines():
            if line.startswith("+") and "def " in line:
                fn = re.search(r"def\s+(\w+)", line)
                if fn:
                    return fn.group(1)
        return "target_function"

    @staticmethod
    def _extract_validation_invariant(validation_summary: Dict[str, Any]) -> str:
        commands = (
            validation_summary.get("commands")
            or validation_summary.get("verification_commands")
            or []
        )
        if commands:
            return str(commands[0])[:80]
        return "run targeted test"

    @staticmethod
    def _build_bug_semantic(*, target_function: str, change_type: str, changed_files: List[str]) -> str:
        rep = EvaluationFeedbackProcessor._pick_representative_file(changed_files)
        file_hint = rep.split("/")[-1] if rep != "target file" else rep
        return f"{change_type} in {target_function}() at {file_hint}"

    def _upsert_compiler_card_with_support(
        self,
        payload: Dict[str, Any],
        *,
        support_key: str,
    ) -> str:
        row = dict(payload)
        card_id = str(row.get("card_id") or "").strip()
        existing = dict(self.graph_store.compiler_cards_v21.get(card_id, {})) if card_id else {}
        metadata = dict(existing.get("metadata") or {})
        metadata.update(dict(row.get("metadata") or {}))

        support_keys = [str(x).strip() for x in (metadata.get("support_keys") or []) if str(x).strip()]
        support_count = int(existing.get("support_count", 0) or 0)
        if not support_key:
            support_key = stable_patch_digest({"card_id": card_id, "payload": row})
        if support_key not in support_keys:
            support_keys.append(support_key)
            support_count += max(1, int(row.get("support_count", 1) or 1))

        row["support_count"] = max(1, support_count)
        row["confidence"] = max(float(existing.get("confidence", 0.0) or 0.0), float(row.get("confidence", 0.0) or 0.0))
        metadata["support_keys"] = support_keys[-20:]
        row["metadata"] = metadata
        return self.graph_store.upsert_compiler_card_v21(row)

    @staticmethod
    def _split_validation_paths(validation_commands: List[str]) -> tuple[List[str], List[str]]:
        repro_path: List[str] = []
        target_validation: List[str] = []
        for command in validation_commands:
            lowered = str(command).lower()
            if any(token in lowered for token in ("reproduce", "test_", "quiet", "plot")):
                repro_path.append(command)
            if "pytest" in lowered or "python -m pytest" in lowered:
                target_validation.append(command)
        if not repro_path and validation_commands:
            repro_path = validation_commands[:1]
        if not target_validation and validation_commands:
            target_validation = validation_commands[:1]
        return repro_path[:2], target_validation[:2]

    @staticmethod
    def _build_success_path_hint(*, changed_files: List[str], target_validation: List[str]) -> str:
        file_hint = EvaluationFeedbackProcessor._pick_representative_file(changed_files)
        validation_hint = target_validation[0] if target_validation else "the focused target validation"
        return (
            f"Reuse the last resolved path: keep the patch family centered on {file_hint}, rerun "
            f"{validation_hint}, and submit once the focused target validation passes."
        )[:280]

    def _find_attempt_summary_id(self, *, instance_id: str, run_id: str, attempt_id: str) -> str:
        best_id = ""
        best_score = -1
        for summary_id, row in self.graph_store.attempt_summaries_v1.items():
            if str(row.get("instance_id") or "").strip() != str(instance_id or "").strip():
                continue
            if attempt_id and str(row.get("attempt_id") or "").strip() != str(attempt_id or "").strip():
                continue
            score = 0
            if attempt_id and str(row.get("attempt_id") or "").strip() == str(attempt_id or "").strip():
                score += 5
            if run_id and str(row.get("run_id") or "").strip() == str(run_id or "").strip():
                score += 2
            if score > best_score:
                best_id = str(summary_id)
                best_score = score
        return best_id

    @staticmethod
    def _support_key(*, run_id: str, attempt_id: str, eval_ref: str) -> str:
        parts = [str(x).strip() for x in (run_id, attempt_id, eval_ref) if str(x).strip()]
        return "::".join(parts)

    @staticmethod
    def _normalize_outcome(outcome: str) -> str:
        raw = str(outcome or "").strip().lower()
        if raw in {"resolved", "success", "pass", "passed"}:
            return "resolved"
        if raw in {"infra_failure", "environment_error"}:
            return "infra_failure"
        if raw in {"incomplete", "timeout", "error"}:
            return "incomplete"
        return "unresolved"

    def _build_patch_summary(
        self,
        *,
        patch_text: str,
        changed_files: List[str],
        validation_summary: Dict[str, Any],
        supplied_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        summary = dict(supplied_summary)
        patch = patch_text or ""
        risk_flags: List[str] = [str(x).strip() for x in (summary.get("risk_flags") or []) if str(x).strip()]
        added_lines = [line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")]
        heuristics = {
            "broad_exception": r"(?m)^\+.*except\s+(?:Exception|BaseException)\b",
            "bare_exception": r"(?m)^\+.*except:\s*$",
            "quiet_flag": r"(?m)^\+.*quiet\s*=\s*True",
            "warning_suppression": r"(?m)^\+.*warnings\.filterwarnings",
            "silent_pass": r"(?m)^\+.*\bpass\b",
            "suppress_context": r"(?m)^\+.*contextlib\.suppress",
            "interface_expansion": r"(?m)^\+.*def\s+\w+\s*\(.*=.*\)",
            "behavior_change_default": r"(?m)^\+.*def\s+\w+\s*\(.*(?:quiet|strict|force|fallback)\s*=\s*(?:True|False|None)\b.*\)",
        }
        for name, pattern in heuristics.items():
            if re.search(pattern, patch):
                risk_flags.append(name)
        if len(added_lines) >= 40:
            risk_flags.append("large_patch")
        if len(set(changed_files)) >= 4:
            risk_flags.append("broad_surface_patch")
        validation_commands = [
            str(x).strip()
            for x in (validation_summary.get("commands") or validation_summary.get("verification_commands") or [])
            if str(x).strip()
        ]
        if not validation_commands:
            risk_flags.append("missing_target_validation")
        summary["risk_flags"] = list(dict.fromkeys(risk_flags))
        summary["changed_files"] = list(dict.fromkeys([str(x).strip() for x in changed_files if str(x).strip()]))
        summary["validation_commands"] = validation_commands[:6]
        summary["changed_file_count"] = len(summary["changed_files"])
        return summary

    def _build_negative_experiences(
        self,
        *,
        instance_id: str,
        outcome: str,
        eval_ref: str,
        summary: Dict[str, Any],
        task_summary: str,
        run_id: str,
        attempt_id: str,
        trace_id: str,
    ) -> List[Dict[str, Any]]:
        risk_flags = [str(x).strip() for x in (summary.get("risk_flags") or []) if str(x).strip()]
        changed_files = [str(x).strip() for x in (summary.get("changed_files") or []) if str(x).strip()]
        validation_commands = [str(x).strip() for x in (summary.get("validation_commands") or []) if str(x).strip()]

        payloads: List[Dict[str, Any]] = []
        if any(flag in risk_flags for flag in {"broad_exception", "bare_exception", "quiet_flag", "warning_suppression", "silent_pass", "suppress_context"}):
            payloads.append(
                self._negative_experience_payload(
                    instance_id=instance_id,
                    pattern_type="negative_strategy_suppressive_fix",
                    abstracted_intent="Avoid suppressive fixes that silence failures before target validation confirms the real bug is fixed.",
                    success_conditions=["require_target_validation_before_submission"],
                    failure_avoidance=["do_not_repeat_suppressive_fix_as_primary_strategy"],
                    eval_ref=eval_ref,
                    task_summary=task_summary,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    trace_id=trace_id,
                    changed_files=changed_files,
                )
            )
        if any(flag in risk_flags for flag in {"large_patch", "interface_expansion", "behavior_change_default"}):
            payloads.append(
                self._negative_experience_payload(
                    instance_id=instance_id,
                    pattern_type="patch_risk_overdesigned_fix",
                    abstracted_intent="Avoid broad or interface-expanding patches before proving that a minimal fix cannot solve the failing path.",
                    success_conditions=["prefer_minimal_patch_then_targeted_validation"],
                    failure_avoidance=["do_not_expand_public_behavior_before_minimal_fix_is_tested"],
                    eval_ref=eval_ref,
                    task_summary=task_summary,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    trace_id=trace_id,
                    changed_files=changed_files,
                )
            )
        if "missing_target_validation" in risk_flags:
            payloads.append(
                self._negative_experience_payload(
                    instance_id=instance_id,
                    pattern_type="validation_gap_missing_target_validation",
                    abstracted_intent="Submitted patches without target validation should be treated as risky until the failing path is verified directly.",
                    success_conditions=["run_targeted_validation_before_submission"],
                    failure_avoidance=["avoid_submission_without_target_validation"],
                    eval_ref=eval_ref,
                    task_summary=task_summary,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    trace_id=trace_id,
                    changed_files=changed_files,
                )
            )
        if outcome == "incomplete" and (validation_commands or not changed_files):
            payloads.append(
                self._negative_experience_payload(
                    instance_id=instance_id,
                    pattern_type="negative_strategy_timeout_after_overexploration",
                    abstracted_intent="If repeated reproduction or ad-hoc validation does not produce a patch before timeout, stop expanding scope and either persist the minimal fix candidate or terminate early.",
                    success_conditions=[
                        "limit_ad_hoc_validation_scripts",
                        "promote_minimal_fix_candidate_earlier",
                    ],
                    failure_avoidance=[
                        "avoid_repeated_reproduction_after_key_signal",
                        "avoid_timeout_without_patch",
                    ],
                    eval_ref=eval_ref,
                    task_summary=task_summary,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    trace_id=trace_id,
                    changed_files=changed_files,
                )
            )
        if outcome == "unresolved" and (risk_flags or not payloads):
            payloads.append(
                self._negative_experience_payload(
                    instance_id=instance_id,
                    pattern_type="patch_risk_submission_unresolved",
                    abstracted_intent="A submitted patch that remains unresolved in official evaluation should be treated as a patch-risk warning, not a success pattern.",
                    success_conditions=validation_commands[:3] or ["run_targeted_validation_before_submission"],
                    failure_avoidance=["do_not_promote_submitted_patch_without_official_resolution"],
                    eval_ref=eval_ref,
                    task_summary=task_summary,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    trace_id=trace_id,
                    changed_files=changed_files,
                )
            )
        return payloads

    def _negative_experience_payload(
        self,
        *,
        instance_id: str,
        pattern_type: str,
        abstracted_intent: str,
        success_conditions: List[str],
        failure_avoidance: List[str],
        eval_ref: str,
        task_summary: str,
        run_id: str,
        attempt_id: str,
        trace_id: str,
        changed_files: List[str],
    ) -> Dict[str, Any]:
        norm_pattern = "negative_strategy" if pattern_type.startswith("negative_strategy") else (
            "patch_risk" if pattern_type.startswith("patch_risk") else "validation_gap"
        )
        trigger = "official_eval_unresolved"
        if norm_pattern == "validation_gap":
            trigger = "missing_validation"
        subproblem_type = "target_validation"
        strategy_label = "unknown_strategy"
        prefer_actions: List[str] = []
        avoid_actions: List[str] = []
        if pattern_type == "negative_strategy_timeout_after_overexploration":
            subproblem_type = "reproduce_issue"
            strategy_label = "ad_hoc_repro_script_loop"
            prefer_actions = ["reuse_confirmed_repro_path", "promote_minimal_fix_candidate_earlier"]
            avoid_actions = ["create_new_repro_script_after_repro_confirmed"]
        elif pattern_type.startswith("patch_risk"):
            subproblem_type = "form_minimal_patch"
            strategy_label = "api_alternative_probe_without_fix"
            prefer_actions = ["edit_target_function_before_new_module_search", "run_target_validation_before_submission"]
            avoid_actions = ["expand_patch_scope_before_minimal_fix_is_tested"]
        elif pattern_type.startswith("validation_gap"):
            subproblem_type = "target_validation"
            strategy_label = "broad_test_without_patch"
            prefer_actions = ["run_target_validation_before_submission"]
            avoid_actions = ["submit_without_target_validation"]
        return {
            "schema_version": "2.1",
            "pattern_type": pattern_type,
            "abstracted_intent": abstracted_intent,
            "variant_texts": [abstracted_intent],
            "success_conditions": success_conditions[:5],
            "failure_avoidance": failure_avoidance[:5],
            "subproblem_type": subproblem_type,
            "strategy_label": strategy_label,
            "prefer_actions": prefer_actions,
            "avoid_actions": avoid_actions,
            "applicability_scope": {
                "subproblem_type": subproblem_type,
                "strategy_label": strategy_label,
                "official_eval_status": "unresolved",
            },
            "source_instance_id": instance_id,
            "source_run_ids": [run_id] if run_id else [],
            "source_attempt_ids": [attempt_id] if attempt_id else [],
            "source_event_ids": [trace_id] if trace_id else [],
            "support_count": 1,
            "confidence": 0.82,
            "normalized_pattern_type": norm_pattern,
            "normalized_trigger_family": trigger,
            "normalized_advice_family": "avoid_repeat_failed_strategy",
            "family_id": f"{norm_pattern}__{trigger}__avoid_repeat_failed_strategy",
            "metadata": {
                "experience_polarity": "negative",
                "promotion_state": "promoted",
                "evidence_stage": "official_eval",
                "official_eval_refs": [eval_ref] if eval_ref else [],
                "official_eval_status": "unresolved",
                "task_summary": task_summary[:280],
                "changed_file_pattern": [
                    f".{path.rsplit('.', 1)[-1].lower()}"
                    for path in changed_files
                    if "." in path
                ][:4],
                "subproblem_type": subproblem_type,
                "strategy_label": strategy_label,
                "prefer_actions": prefer_actions,
                "avoid_actions": avoid_actions,
                "applicability_scope": {
                    "subproblem_type": subproblem_type,
                    "strategy_label": strategy_label,
                    "official_eval_status": "unresolved",
                },
            },
        }

    @staticmethod
    def _candidate_fix_actions(summary: Dict[str, Any]) -> List[str]:
        actions = ["reproduce_the_failing_path_with_target_validation_before_next_patch"]
        risk_flags = [str(x).strip() for x in (summary.get("risk_flags") or []) if str(x).strip()]
        if any(flag in risk_flags for flag in {"broad_exception", "bare_exception", "silent_pass"}):
            actions.append("remove_suppressive_fix_pattern_and_patch_target_logic_instead")
        if "missing_target_validation" in risk_flags:
            actions.append("run_targeted_validation_before_submission")
        return list(dict.fromkeys(actions))

    @staticmethod
    def _verification_commands(summary: Dict[str, Any]) -> List[str]:
        commands = [str(x).strip() for x in (summary.get("validation_commands") or []) if str(x).strip()]
        if not commands:
            commands = ["pytest -q"]
        return commands[:4]
