"""
Memory Agent: A service that provides query rewriting, subgraph selection,
experience distillation, evidence referencing, and belief updates.

Responsibilities:
1) Rewrite natural-language queries into structured retrieval requests.
2) Select the most relevant knowledge-graph subgraph.
3) Distill prior experience into concise guidance.
4) Attach evidence references.
5) Promote or deprecate beliefs using feedback.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import uuid

from ..core.problem_file import ActionType, Outcome, ProblemFile
from ..core.belief_graph import (
    AtomicBelief,
    AttemptBelief,
    BeliefType,
    ConditionSignature,
    BeliefRule,
)
from ..core.observation_kg import EdgeType
from ..storage.graph_store import GraphStore


class QueryRewriter:
    """Rewrites natural language queries into structured retrieval queries."""

    def rewrite_for_planning(self,
                            user_instruction: str,
                            context_summary: str,
                            env_signature: Dict[str, Any]) -> Dict[str, Any]:
        """Rewrite query for planning-time retrieval."""
        # TODO: Use LLM or rule-based rewriting
        # For MVP, return structured query based on simple extraction
        return {
            "query_type": "planning",
            "task_semantic": user_instruction,
            "context": context_summary,
            "env_cluster": env_signature.get("env_cluster"),
            "file_scope": self._extract_file_scope(user_instruction),
            "intent_patterns": self._extract_intent_patterns(user_instruction),
        }

    def rewrite_for_repair(self,
                          error_type: str,
                          error_message: str,
                          current_action: str,
                          env_signature: Dict[str, Any]) -> Dict[str, Any]:
        """Rewrite query for execution-time repair retrieval."""
        return {
            "query_type": "error_recovery",
            "error_type": error_type,
            "error_signature": self._extract_error_signature(error_message),
            "current_action": current_action,
            "env_cluster": env_signature.get("env_cluster"),
            "action_type": self._infer_action_type(current_action),
        }

    def rewrite_for_query_type(
        self,
        *,
        query_type: str,
        user_instruction: str = "",
        context_summary: str = "",
        env_signature: Optional[Dict[str, Any]] = None,
        error_type: str = "",
        error_message: str = "",
        current_action: str = "",
    ) -> Dict[str, Any]:
        normalized = (query_type or "planning").strip().lower()
        env = env_signature or {}
        if normalized in {"planning", "regression_guard"}:
            query = self.rewrite_for_planning(
                user_instruction=user_instruction,
                context_summary=context_summary,
                env_signature=env,
            )
            query["query_type"] = normalized
            return query
        if normalized in {"error_recovery", "test_failure_fix"}:
            query = self.rewrite_for_repair(
                error_type=error_type,
                error_message=error_message,
                current_action=current_action,
                env_signature=env,
            )
            query["query_type"] = normalized
            return query
        query = self.rewrite_for_planning(
            user_instruction=user_instruction,
            context_summary=context_summary,
            env_signature=env,
        )
        query["query_type"] = "planning"
        return query

    def _extract_file_scope(self, text: str) -> List[str]:
        """Extract file scope patterns from text."""
        # Simplified extraction - look for file extensions
        import re
        patterns = [
            r'(\S+\.py)', r'(\S+\.js)', r'(\S+\.ts)', r'(\S+\.java)',
            r'(\S+\.cpp)', r'(\S+\.h)', r'(\S+\.md)', r'(\S+\.txt)'
        ]
        files = []
        for pattern in patterns:
            files.extend(re.findall(pattern, text))
        return files[:5]  # Limit to 5 files

    def _extract_intent_patterns(self, text: str) -> List[str]:
        """Extract intent patterns from text."""
        # Common intent keywords
        intents = []
        keywords = [
            "fix", "repair", "debug", "implement", "add", "remove",
            "update", "modify", "test", "run", "check", "verify"
        ]
        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                intents.append(keyword)
        return intents

    def _extract_error_signature(self, error_message: str) -> Dict[str, Any]:
        """Extract error signature from error message."""
        # Simplified error signature extraction
        lines = error_message.strip().split('\n')
        first_line = lines[0] if lines else ""

        return {
            "first_line": first_line[:100],  # First 100 chars
            "contains_error": "error" in first_line.lower(),
            "contains_exception": "exception" in first_line.lower(),
            "contains_fail": "fail" in first_line.lower(),
            "line_count": len(lines),
        }

    def _infer_action_type(self, action: str) -> Optional[str]:
        """Infer action type from action string."""
        action_lower = action.lower()
        if "test" in action_lower:
            return "run_test"
        elif "edit" in action_lower or "modify" in action_lower:
            return "code_edit"
        elif "run" in action_lower or "execute" in action_lower:
            return "command_exec"
        elif "search" in action_lower or "find" in action_lower:
            return "tool_call"
        else:
            return None


class SubgraphSelector:
    """Selects useful experience subgraphs from KG (not just top-k nodes)."""

    def __init__(self, graph_store: GraphStore):
        self.graph_store = graph_store

    def select_for_planning(self,
                           query: Dict[str, Any],
                           max_subgraphs: int = 3) -> List[Dict[str, Any]]:
        """Select subgraphs for planning-time experience injection."""
        selected = []

        # For each task subgraph, compute relevance score
        for task_id, subgraph in self.graph_store.observation_kg.task_subgraphs.items():
            relevance = self._compute_planning_relevance(subgraph, query)
            if relevance > 0:
                selected.append({
                    "task_id": task_id,
                    "subgraph": subgraph,
                    "relevance": relevance,
                    "success_chain": subgraph.get_success_chain(),
                    "total_actions": len(subgraph.action_nodes),
                })

        # Sort by relevance and limit
        selected.sort(key=lambda x: x["relevance"], reverse=True)
        return selected[:max_subgraphs]

    def select_for_repair(self,
                         query: Dict[str, Any],
                         max_subgraphs: int = 5) -> List[Dict[str, Any]]:
        """Select subgraphs for execution-time repair."""
        selected = []

        for task_id, subgraph in self.graph_store.observation_kg.task_subgraphs.items():
            relevance = self._compute_repair_relevance(subgraph, query)
            if relevance > 0:
                # Extract failure-retry chains
                failure_chains = subgraph.get_failure_retry_chains()

                selected.append({
                    "task_id": task_id,
                    "subgraph": subgraph,
                    "relevance": relevance,
                    "failure_chains": failure_chains,
                    "total_actions": len(subgraph.action_nodes),
                })

        selected.sort(key=lambda x: x["relevance"], reverse=True)
        return selected[:max_subgraphs]

    def _compute_planning_relevance(self,
                                   subgraph: Any,  # TaskSubgraph
                                   query: Dict[str, Any]) -> float:
        """Compute relevance score for planning."""
        score = 0.0

        # Check if subgraph has successful completion
        success_chain = subgraph.get_success_chain()
        if not success_chain:
            return 0.0

        # Basic score for having a success chain
        score += 0.3

        # Check file scope match
        query_files = query.get("file_scope", [])
        if query_files:
            # Check if any action in subgraph touches similar files
            file_match = False
            for action in subgraph.action_nodes.values():
                for touched_file in action.touched_files:
                    for query_file in query_files:
                        if query_file in touched_file or touched_file in query_file:
                            file_match = True
                            break
                if file_match:
                    break
            if file_match:
                score += 0.3

        # Check intent pattern match
        query_intents = query.get("intent_patterns", [])
        if query_intents:
            intent_match = False
            for action in subgraph.action_nodes.values():
                intent_lower = action.intent_text.lower()
                for intent in query_intents:
                    if intent in intent_lower:
                        intent_match = True
                        break
                if intent_match:
                    break
            if intent_match:
                score += 0.2

        # Normalize to 0-1 range
        return min(score, 1.0)

    def _compute_repair_relevance(self,
                                 subgraph: Any,  # TaskSubgraph
                                 query: Dict[str, Any]) -> float:
        """Compute relevance score for repair."""
        score = 0.0

        # Check if subgraph has failure-retry chains
        failure_chains = subgraph.get_failure_retry_chains()
        if not failure_chains:
            return 0.0

        # Basic score for having failure chains
        score += 0.2

        # Check error type match
        query_error_type = query.get("error_type", "")
        if query_error_type:
            # Look for similar error types in failure actions
            error_match = False
            for action in subgraph.action_nodes.values():
                if action.failure_signature:
                    if (query_error_type.lower() in action.failure_signature.error_type.lower() or
                        action.failure_signature.error_type.lower() in query_error_type.lower()):
                        error_match = True
                        break
            if error_match:
                score += 0.3

        # Check action type match
        query_action_type = query.get("action_type")
        if query_action_type:
            action_match = False
            for action in subgraph.action_nodes.values():
                if action.action_type.value == query_action_type:
                    action_match = True
                    break
            if action_match:
                score += 0.3

        # Normalize to 0-1 range
        return min(score, 1.0)


class ExperienceDistiller:
    """Distills experience into executable suggestions/flow tips/constraint candidates."""

    def distill_planning_experience(self,
                                   selected_subgraphs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Distill planning experience from selected subgraphs."""
        recommendations = []
        evidence_refs = []

        for subgraph_info in selected_subgraphs:
            subgraph = subgraph_info["subgraph"]
            task_id = subgraph_info["task_id"]

            # Extract success chain as workflow template
            success_chain = subgraph.get_success_chain()
            if success_chain:
                workflow_tips = self._extract_workflow_tips(subgraph, success_chain)
                recommendations.extend(workflow_tips)

                # Add evidence references
                for action_id in success_chain[:3]:  # First 3 actions
                    evidence_refs.append({
                        "task_id": task_id,
                        "action_id": action_id,
                        "type": "success_chain",
                    })

            # Extract preventive checks from beliefs
            # TODO: Integrate with belief graph

        # Deduplicate and rank recommendations
        unique_recommendations = self._deduplicate_recommendations(recommendations)

        return {
            "recommendations": unique_recommendations[:5],  # Top 5
            "evidence_refs": evidence_refs[:10],  # Limit evidence
            "confidence": self._compute_confidence(selected_subgraphs),
            "total_subgraphs": len(selected_subgraphs),
        }

    def distill_repair_experience(self,
                                 selected_subgraphs: List[Dict[str, Any]],
                                 current_problem: ProblemFile) -> Dict[str, Any]:
        """Distill repair experience from selected subgraphs."""
        repair_suggestions = []
        evidence_refs = []

        for subgraph_info in selected_subgraphs:
            subgraph = subgraph_info["subgraph"]
            task_id = subgraph_info["task_id"]
            failure_chains = subgraph_info.get("failure_chains", {})

            for fail_action_id, retry_chain in failure_chains.items():
                if len(retry_chain) >= 2:  # At least failure and repair
                    # Extract repair pattern
                    repair_pattern = self._extract_repair_pattern(
                        subgraph, fail_action_id, retry_chain
                    )
                    if repair_pattern:
                        repair_suggestions.append(repair_pattern)

                        # Add evidence references
                        evidence_refs.append({
                            "task_id": task_id,
                            "fail_action_id": fail_action_id,
                            "repair_action_id": retry_chain[1] if len(retry_chain) > 1 else None,
                            "type": "failure_repair",
                        })

        # Find most relevant repair suggestion
        next_step_fix = ""
        if repair_suggestions:
            # For MVP, use the first suggestion
            first_suggestion = repair_suggestions[0]
            next_step_fix = first_suggestion.get("repair_action", "")

        return {
            "recommendations": repair_suggestions[:3],  # Top 3
            "evidence_refs": evidence_refs[:5],  # Limit evidence
            "confidence": self._compute_confidence(selected_subgraphs),
            "next_step_fix": next_step_fix,
            "expected_outcome": "Error should be resolved or reduced",
            "total_subgraphs": len(selected_subgraphs),
        }

    def _extract_workflow_tips(self, subgraph: Any, success_chain: List[str]) -> List[Dict[str, Any]]:
        """Extract workflow tips from success chain."""
        tips = []

        for i, action_id in enumerate(success_chain):
            action = subgraph.action_nodes.get(action_id)
            if not action:
                continue

            # Create tip for key actions
            if i == 0 or i == len(success_chain) - 1 or i % 3 == 0:  # Sample points
                tip = {
                    "type": "workflow_step",
                    "task_id": getattr(subgraph, "task_id", None),
                    "action_id": action_id,
                    "step_number": i + 1,
                    "intent": action.intent_text[:100],  # First 100 chars
                    "action_type": action.action_type.value,
                    "recommendation": f"Consider: {action.intent_text[:50]}...",
                }
                tips.append(tip)

        return tips

    def _extract_repair_pattern(self,
                               subgraph: Any,
                               fail_action_id: str,
                               retry_chain: List[str]) -> Optional[Dict[str, Any]]:
        """Extract repair pattern from failure-retry chain."""
        if len(retry_chain) < 2:
            return None

        fail_action = subgraph.action_nodes.get(fail_action_id)
        repair_action = subgraph.action_nodes.get(retry_chain[1])  # First repair attempt

        if not fail_action or not repair_action:
            return None

        return {
            "fail_action_type": fail_action.action_type.value,
            "fail_intent": fail_action.intent_text[:100],
            "repair_action_type": repair_action.action_type.value,
            "repair_intent": repair_action.intent_text[:100],
            "repair_action": self._suggest_repair_action(fail_action, repair_action),
            "pattern": "failure_followed_by_repair",
        }

    def _suggest_repair_action(self, fail_action: ProblemFile, repair_action: ProblemFile) -> str:
        """Generate repair action suggestion."""
        # Simplified suggestion generation
        if fail_action.action_type.value == repair_action.action_type.value:
            return f"Retry similar action with adjustments: {repair_action.intent_text[:50]}..."
        else:
            return f"Try different approach: {repair_action.intent_text[:50]}..."

    def _deduplicate_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate similar recommendations."""
        seen = set()
        unique = []

        for rec in recommendations:
            # Create signature for deduplication
            if "intent" in rec:
                sig = f"{rec.get('type', '')}:{rec.get('intent', '')[:50]}"
            elif "repair_action" in rec:
                sig = f"{rec.get('pattern', '')}:{rec.get('repair_action', '')[:50]}"
            else:
                sig = str(rec)

            if sig not in seen:
                seen.add(sig)
                unique.append(rec)

        return unique

    def _compute_confidence(self, selected_subgraphs: List[Dict[str, Any]]) -> float:
        """Compute confidence score based on selected subgraphs."""
        if not selected_subgraphs:
            return 0.0

        # Average relevance score
        total_relevance = sum(sg.get("relevance", 0.0) for sg in selected_subgraphs)
        avg_relevance = total_relevance / len(selected_subgraphs)

        # Boost for multiple supporting subgraphs
        support_boost = min(len(selected_subgraphs) / 5.0, 0.3)  # Max 0.3 boost for 5+ subgraphs

        return min(avg_relevance + support_boost, 1.0)


class MemoryAgent:
    """
    Memory Agent: Main service coordinating query rewriting, subgraph selection,
    experience distillation, and evidence referencing.
    """

    def __init__(self, graph_store: GraphStore):
        self.graph_store = graph_store
        self.query_rewriter = QueryRewriter()
        self.subgraph_selector = SubgraphSelector(graph_store)
        self.experience_distiller = ExperienceDistiller()
        self._fusion_weights = {
            "structure": 0.55,
            "embedding": 0.35,
            "support": 0.10,
        }

    @staticmethod
    def _normalize_query_type(query_type: str) -> str:
        raw = (query_type or "planning").strip().lower()
        aliases = {
            "repair": "error_recovery",
            "execute_repair": "error_recovery",
            "test_repair": "test_failure_fix",
            "regression": "regression_guard",
        }
        return aliases.get(raw, raw if raw else "planning")

    def _retrieve_abstract_patterns(
        self,
        *,
        query_text: str,
        error_type: Optional[str] = None,
        subproblem_type: str = "",
        strategy_label: str = "",
        max_results: int = 2,
    ) -> List[Dict[str, Any]]:
        rows = self.graph_store.query_abstract_experiences(
            query_text=query_text,
            error_type=error_type,
            subproblem_type=subproblem_type,
            strategy_label=strategy_label,
            max_results=max_results,
        )
        patterns: List[Dict[str, Any]] = []
        for row in rows:
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            patterns.append(
                {
                    "type": "abstract_pattern",
                    "experience_id": row.get("experience_id"),
                    "family_id": row.get("family_id"),
                    "normalized_pattern_type": row.get("normalized_pattern_type"),
                    "normalized_trigger_family": row.get("normalized_trigger_family"),
                    "normalized_advice_family": row.get("normalized_advice_family"),
                    "pattern_type": row.get("pattern_type"),
                    "recommendation": row.get("abstracted_intent"),
                    "success_conditions": row.get("success_conditions", []),
                    "failure_avoidance": row.get("failure_avoidance", []),
                    "confidence": float(row.get("confidence", 0.0)),
                    "score": float(row.get("score", 0.0)),
                    "critical_alignment_score": float(row.get("critical_alignment_score", 0.0)),
                    "support_count": int(row.get("support_count", 0) or 0),
                    "experience_polarity": metadata.get("experience_polarity", "neutral"),
                    "promotion_state": metadata.get("promotion_state", "candidate"),
                    "evidence_stage": metadata.get("evidence_stage", "trial_local"),
                    "changed_file_pattern": metadata.get("changed_file_pattern", []),
                    "subproblem_type": row.get("subproblem_type") or metadata.get("subproblem_type", ""),
                    "strategy_label": row.get("strategy_label") or metadata.get("strategy_label", ""),
                    "prefer_actions": row.get("prefer_actions") or metadata.get("prefer_actions", []),
                    "avoid_actions": row.get("avoid_actions") or metadata.get("avoid_actions", []),
                    "applicability_scope": row.get("applicability_scope") or metadata.get("applicability_scope", {}),
                }
            )
        return patterns

    @staticmethod
    def _infer_subproblem_type_for_planning(
        *,
        current_action: str,
        current_problem_file: Optional[ProblemFile],
    ) -> str:
        text = str(current_action or "").lower()
        if current_problem_file is not None:
            text = f"{current_problem_file.intent_text} {current_problem_file.action_text} {text}".lower()
        if "pytest" in text or "reproduce_" in text or "test_" in text:
            if any(token in text for token in ("all tests", "full suite", "regression")):
                return "broad_regression_check"
            return "reproduce_issue" if any(token in text for token in ("reproduce", "test_", "quiet", "plot")) else "target_validation"
        if any(token in text for token in ("edit", "patch", "apply")):
            return "form_minimal_patch"
        if any(token in text for token in ("grep", "search", "inspect", "read", "open", "find")):
            return "localize_fix"
        if any(token in text for token in ("docker", "retry", "timeout", "install")):
            return "tool_recovery"
        return "unknown"

    @staticmethod
    def _infer_strategy_label_for_planning(
        *,
        current_action: str,
        subproblem_type: str,
    ) -> str:
        text = str(current_action or "").lower()
        if subproblem_type == "reproduce_issue" and sum(text.count(tok) for tok in ("test_", "reproduce_")) >= 2:
            return "ad_hoc_repro_script_loop"
        if subproblem_type == "localize_fix" and any(token in text for token in ("utils.py", "wcsaxes")):
            return "cross_module_expansion_after_key_signal"
        if subproblem_type == "broad_regression_check":
            return "broad_test_without_patch"
        if subproblem_type in {"form_minimal_patch", "target_validation"} and any(token in text for token in ("patch", "edit", "pytest")):
            return "minimal_patch_then_target_validation"
        if any(token in text for token in ("world_to_pixel_values", "all_world2pix")):
            return "api_alternative_probe_without_fix"
        return "unknown_strategy"

    def _retrieve_attempt_summary(
        self,
        *,
        instance_id: str,
        current_attempt_id: str,
        current_problem_file: Optional[ProblemFile],
    ) -> Optional[Dict[str, Any]]:
        if not str(instance_id or "").strip():
            return None
        if current_problem_file is not None and int(current_problem_file.step_index or 0) > 2:
            return None
        summary = self.graph_store.get_latest_attempt_summary(
            instance_id=instance_id,
            exclude_attempt_id=current_attempt_id,
        )
        if not summary:
            return None
        failed = summary.get("failed_strategies") if isinstance(summary.get("failed_strategies"), list) else []
        next_best = summary.get("next_best_actions") if isinstance(summary.get("next_best_actions"), list) else []
        lines: List[str] = []
        goal = str(summary.get("problem_goal") or "").strip()
        if goal:
            lines.append(f"Previous attempt goal: {goal}")
        if failed:
            first = failed[0] if isinstance(failed[0], dict) else {}
            reason = str(first.get("reason") or "").strip()
            avoid = ", ".join([str(x).strip() for x in (first.get("avoid_actions") or []) if str(x).strip()][:3])
            if reason:
                lines.append(f"Previous failed path: {reason}")
            if avoid:
                lines.append(f"Avoid: {avoid}")
        if next_best:
            lines.append(f"Next best actions: {', '.join([str(x).strip() for x in next_best[:3] if str(x).strip()])}")
        recommendation = " ".join(lines).strip()
        if not recommendation:
            return None
        failed_strategy = failed[0] if failed and isinstance(failed[0], dict) else {}
        return {
            "type": "attempt_summary_v1",
            "summary_id": summary.get("summary_id"),
            "family_id": f"attempt_summary:{str(summary.get('instance_id') or 'unknown').strip()}",
            "recommendation": recommendation[:280],
            "confidence": 0.9,
            "attempt_summary": summary,
            "subproblem_type": str(failed_strategy.get("subproblem_type") or "").strip(),
            "strategy_label": str(failed_strategy.get("strategy_label") or "").strip(),
            "prefer_actions": [
                str(x).strip()
                for x in (failed_strategy.get("prefer_actions") or summary.get("next_best_actions") or [])
                if str(x).strip()
            ][:5],
            "avoid_actions": [
                str(x).strip()
                for x in (failed_strategy.get("avoid_actions") or [])
                if str(x).strip()
            ][:5],
        }

    def _collect_embedding_candidates(
        self,
        *,
        current_problem_file: Optional[ProblemFile],
        embedding_view: str,
        max_results: int,
        min_similarity: float,
    ) -> Dict[str, Any]:
        if current_problem_file is None:
            return {
                "candidate_count_before_filter": 0,
                "candidate_count_after_filter": 0,
                "candidate_task_count": 0,
                "task_scores": {},
            }

        raw_candidates = self.graph_store.find_similar_actions(
            current_problem_file,
            max_results=max(10, max_results),
            embedding_view=embedding_view,
        )
        filtered = [row for row in raw_candidates if float(row[2]) >= min_similarity]
        if not filtered and raw_candidates:
            filtered = raw_candidates[: min(8, len(raw_candidates))]

        task_scores: Dict[str, float] = {}
        for action_id, _action, score in filtered:
            mapped = self.graph_store.observation_kg.get_action(action_id)
            if not mapped:
                continue
            task_id = mapped[1]
            task_scores[task_id] = max(task_scores.get(task_id, 0.0), float(score))

        return {
            "candidate_count_before_filter": len(raw_candidates),
            "candidate_count_after_filter": len(filtered),
            "candidate_task_count": len(task_scores),
            "task_scores": task_scores,
        }

    def _merge_hybrid_subgraphs(
        self,
        *,
        selected_subgraphs: List[Dict[str, Any]],
        embedding_task_scores: Dict[str, float],
        query_type: str,
        max_subgraphs: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for row in selected_subgraphs:
            merged[row["task_id"]] = dict(row)

        for task_id in embedding_task_scores:
            if task_id in merged:
                continue
            subgraph = self.graph_store.observation_kg.get_task_subgraph(task_id)
            if subgraph is None:
                continue
            row = {
                "task_id": task_id,
                "subgraph": subgraph,
                "relevance": 0.0,
                "total_actions": len(subgraph.action_nodes),
            }
            if query_type == "planning":
                row["success_chain"] = subgraph.get_success_chain()
            else:
                row["failure_chains"] = subgraph.get_failure_retry_chains()
            merged[task_id] = row

        scored_rows: List[Dict[str, Any]] = []
        for task_id, row in merged.items():
            subgraph = row["subgraph"]
            structural_score = float(row.get("relevance", 0.0))
            embedding_score = float(embedding_task_scores.get(task_id, 0.0))
            support_score = min(1.0, len(subgraph.action_nodes) / 15.0)
            fused = (
                self._fusion_weights["structure"] * structural_score
                + self._fusion_weights["embedding"] * embedding_score
                + self._fusion_weights["support"] * support_score
            )
            merged_row = dict(row)
            merged_row["structural_score"] = round(structural_score, 6)
            merged_row["embedding_score"] = round(embedding_score, 6)
            merged_row["support_score"] = round(support_score, 6)
            merged_row["fused_score"] = round(fused, 6)
            scored_rows.append(merged_row)

        scored_rows.sort(key=lambda item: float(item.get("fused_score", 0.0)), reverse=True)
        limited = scored_rows[:max(1, max_subgraphs)]
        debug = {
            "fusion_weights": dict(self._fusion_weights),
            "ranked_tasks": [
                {
                    "task_id": row["task_id"],
                    "structural_score": row.get("structural_score", 0.0),
                    "embedding_score": row.get("embedding_score", 0.0),
                    "fused_score": row.get("fused_score", 0.0),
                    "total_actions": row.get("total_actions", 0),
                }
                for row in limited
            ],
        }
        return limited, debug

    def _retrieve_failure_cards(
        self,
        *,
        query_text: str,
        error_type: Optional[str],
        max_results: int,
    ) -> List[Dict[str, Any]]:
        rows = self.graph_store.query_failure_cards_v2(
            query_text=query_text,
            error_type=error_type,
            max_results=max_results,
            include_infra=False,
        )
        cards: List[Dict[str, Any]] = []
        for row in rows:
            cards.append(
                {
                    "type": "failure_card_v2",
                    "card_id": row.get("card_id"),
                    "failure_class": row.get("failure_class", "agent_failure_card"),
                    "recommendation": "; ".join((row.get("candidate_fix_actions") or [])[:2]),
                    "verification_commands": row.get("verification_commands", [])[:3],
                    "error_signature": row.get("error_signature", {}),
                    "confidence": float(row.get("confidence", 0.0)),
                    "score": float(row.get("score", 0.0)),
                    "critical_alignment_score": float(row.get("critical_alignment_score", 0.0)),
                    "evidence_refs": row.get("evidence_refs", [])[:4],
                    "has_rca": bool(row.get("root_cause_nodes")),
                }
            )
        return cards

    def _retrieve_repair_patterns(
        self,
        *,
        query_text: str,
        error_type: Optional[str],
        max_results: int,
    ) -> List[Dict[str, Any]]:
        rows = self.graph_store.query_repair_patterns_v2(
            query_text=query_text,
            error_type=error_type,
            max_results=max_results,
        )
        out: List[Dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "type": "repair_pattern_v2",
                    "pattern_id": row.get("pattern_id"),
                    "recommendation": row.get("fix_action_template"),
                    "verification_commands": row.get("expected_verification", [])[:3],
                    "confidence": float(row.get("confidence", 0.0)),
                    "score": float(row.get("score", 0.0)),
                    "critical_alignment_score": float(row.get("critical_alignment_score", 0.0)),
                    "evidence_refs": row.get("evidence_refs", [])[:4],
                }
            )
        return out

    def _dynamic_budget(
        self,
        *,
        query_type: str,
        current_action: str = "",
        error_message: str = "",
    ) -> Dict[str, int]:
        complexity = len((current_action or "").split()) + len((error_message or "").split())
        if query_type in {"error_recovery", "test_failure_fix"}:
            base = 10
        elif query_type == "regression_guard":
            base = 8
        else:
            base = 7
        bonus = min(6, complexity // 20)
        candidate_budget = base + bonus
        recommendation_budget = max(6, min(18, candidate_budget))
        failure_card_budget = max(2, recommendation_budget // 4)
        return {
            "subgraph_budget": candidate_budget,
            "recommendation_budget": recommendation_budget,
            "failure_card_budget": failure_card_budget,
            "repair_pattern_budget": max(2, failure_card_budget),
        }

    @staticmethod
    def _recommendation_identity(item: Dict[str, Any]) -> str:
        family_id = str(item.get("family_id", "")).strip()
        if family_id:
            return f"family:{family_id}"
        keys = ("pattern_id", "card_id", "experience_id", "belief_id", "rule_id", "action_id")
        for k in keys:
            val = item.get(k)
            if isinstance(val, str) and val.strip():
                return f"{k}:{val.strip()}"
        text = str(item.get("recommendation") or item.get("repair_action") or item.get("summary") or "")
        return text[:140].strip().lower()

    @staticmethod
    def _recommendation_family(item: Dict[str, Any]) -> str:
        family_id = str(item.get("family_id", "")).strip()
        if family_id:
            return family_id

        rtype = str(item.get("type", "")).strip().lower()
        if rtype == "failure_card_v2":
            error_sig = item.get("error_signature") if isinstance(item.get("error_signature"), dict) else {}
            error_type = str(error_sig.get("error_type", "")).strip().lower()
            rec = str(item.get("recommendation", "")).strip().lower()[:80]
            return f"failure_card:{error_type}:{rec}"
        if rtype == "repair_pattern_v2":
            pattern_id = str(item.get("pattern_id", "")).strip()
            if pattern_id:
                return f"repair_pattern:{pattern_id}"
        if rtype == "belief_tip":
            belief_type = str(item.get("belief_type", "")).strip().lower()
            rec = str(item.get("recommendation", "")).strip().lower()[:80]
            return f"belief_tip:{belief_type}:{rec}"
        if rtype:
            rec = str(item.get("recommendation") or item.get("repair_action") or "").strip().lower()[:80]
            return f"{rtype}:{rec}"
        return ""

    @staticmethod
    def _recommendation_type_bias(rtype: str) -> float:
        if rtype == "compiler_card":
            return 0.15
        if rtype == "attempt_summary_v1":
            return 0.18
        if rtype == "repair_pattern_v2":
            return 0.16
        if rtype == "failure_card_v2":
            return 0.14
        if rtype == "abstract_pattern":
            return 0.06
        if rtype == "belief_tip":
            return -0.12
        if rtype == "legacy_experience":
            return -0.08
        return 0.0

    @staticmethod
    def _recommendation_helpfulness(item: Dict[str, Any]) -> float:
        conf = float(item.get("confidence", 0.0))
        score = float(item.get("score", conf))
        support = min(1.0, float(item.get("support_count", 0) or 0) / 5.0)
        critical = float(item.get("critical_alignment_score", 0.0))
        return round(0.45 * score + 0.25 * conf + 0.20 * support + 0.10 * critical, 6)

    def _retrieve_compiler_cards(
        self,
        *,
        query_text: str,
        query_type: str,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        rows = self.graph_store.query_compiler_cards_v21(
            query_text=query_text,
            query_type=query_type,
            max_results=max_results,
            promotion_states=["candidate", "promoted"],
        )
        out: List[Dict[str, Any]] = []
        for row in rows:
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            out.append(
                {
                    "type": "compiler_card",
                    "card_id": row.get("card_id"),
                    "card_type": row.get("card_type"),
                    "family_id": row.get("family_id"),
                    "recommendation": row.get("recommendation") or row.get("hint"),
                    "hint": row.get("hint") or row.get("recommendation"),
                    "confidence": float(row.get("confidence", 0.0) or 0.0),
                    "score": float(row.get("confidence", 0.0) or 0.0),
                    "evidence_level": row.get("evidence_level"),
                    "evidence_stage": metadata.get("evidence_stage", ""),
                    "promotion_state": row.get("promotion_state"),
                    "support_count": int(row.get("support_count", 0) or 0),
                    "source_object_ids": list(row.get("source_object_ids") or [])[:6],
                    "evidence_refs": list(row.get("evidence_refs") or [])[:4],
                    "changed_file_pattern": list(row.get("changed_file_pattern") or metadata.get("changed_file_pattern") or [])[:4],
                    "normalized_pattern_type": row.get("normalized_pattern_type") or metadata.get("normalized_pattern_type", ""),
                    "subproblem_type": row.get("subproblem_type") or metadata.get("subproblem_type", ""),
                    "strategy_label": row.get("strategy_label") or metadata.get("strategy_label", ""),
                    "prefer_actions": list(row.get("prefer_actions") or metadata.get("prefer_actions") or [])[:4],
                    "avoid_actions": list(row.get("avoid_actions") or metadata.get("avoid_actions") or [])[:4],
                    "budget_hints": row.get("budget_hints") if isinstance(row.get("budget_hints"), dict) else metadata.get("budget_hints", {}),
                    "governance_hardness": row.get("governance_hardness") or metadata.get("governance_hardness", ""),
                }
            )
        return out

    def _dedup_and_rerank_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
        *,
        limit: int,
        current_file_hints: Optional[List[str]] = None,
        runtime_guard: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        runtime_guard = runtime_guard or {}
        closure_active = bool(runtime_guard.get("closure_active"))
        blocked_families = {
            str(x).strip().lower()
            for x in runtime_guard.get("blocked_families", [])
            if str(x).strip()
        }
        blocked_action_patterns = {
            str(x).strip().lower()
            for x in runtime_guard.get("blocked_action_patterns", [])
            if str(x).strip()
        }
        active_subproblem_type = str(runtime_guard.get("active_subproblem_type") or "").strip().lower()
        active_strategy_label = str(runtime_guard.get("active_strategy_label") or "").strip().lower()
        filtered_recommendations: List[Dict[str, Any]] = []
        for row in recommendations:
            family = self._recommendation_family(row).strip().lower()
            pattern = str(row.get("normalized_pattern_type", "")).strip().lower()
            row_subproblem_type = str(row.get("subproblem_type") or "").strip().lower()
            row_strategy_label = str(row.get("strategy_label") or "").strip().lower()
            row_avoid_actions = {
                str(x).strip().lower()
                for x in (row.get("avoid_actions") or [])
                if str(x).strip()
            }
            if closure_active:
                if "workflow_step" in blocked_families and family.startswith("workflow_step:"):
                    continue
                if "planning_loop" in blocked_families and pattern == "planning_loop":
                    continue
            if blocked_action_patterns and row_avoid_actions & blocked_action_patterns:
                continue
            if active_subproblem_type and row_subproblem_type and row_subproblem_type != active_subproblem_type:
                if family.startswith("workflow_step:") or pattern == "planning_loop":
                    continue
            if active_strategy_label and row_strategy_label and row_strategy_label == active_strategy_label:
                if row_avoid_actions:
                    continue
            filtered_recommendations.append(row)

        family_counts: Dict[str, int] = {}
        for row in filtered_recommendations:
            family = self._recommendation_family(row)
            if family:
                family_counts[family] = family_counts.get(family, 0) + 1

        scored_rows: List[Tuple[float, Dict[str, Any]]] = []
        for raw_row in filtered_recommendations:
            row = dict(raw_row)
            conf = float(row.get("confidence", 0.0))
            score = float(row.get("score", conf))
            critical_bonus = 0.2 * float(row.get("critical_alignment_score", 0.0))
            rtype = str(row.get("type", ""))
            type_bias = self._recommendation_type_bias(rtype)
            family = self._recommendation_family(row)
            family_dup_count = family_counts.get(family, 1) if family else 1
            novelty_score = 1.0 / float(family_dup_count)
            helpfulness_score = self._recommendation_helpfulness(row)
            total_score = (
                score
                + type_bias
                + 0.1 * conf
                + critical_bonus
                + 0.12 * helpfulness_score
                + 0.08 * novelty_score
                + self._recommendation_family_bonus(row)
                + self._changed_file_overlap_bonus(row, current_file_hints=current_file_hints)
            )
            if family:
                row["family_id"] = family
            row["helpfulness_score"] = helpfulness_score
            row["novelty_score"] = round(novelty_score, 6)
            row["selection_score"] = round(total_score, 6)
            scored_rows.append((total_score, row))
        scored_rows.sort(key=lambda x: x[0], reverse=True)

        deduped: List[Dict[str, Any]] = []
        seen = set()
        seen_families = set()
        for _score, row in scored_rows:
            ident = self._recommendation_identity(row)
            if not ident or ident in seen:
                continue
            family = self._recommendation_family(row)
            if family and family in seen_families:
                continue
            seen.add(ident)
            if family:
                seen_families.add(family)
            deduped.append(row)
            if len(deduped) >= max(1, limit):
                break
        return deduped

    @staticmethod
    def _recommendation_family_bonus(item: Dict[str, Any]) -> float:
        rtype = str(item.get("type", "")).strip().lower()
        family_id = str(item.get("family_id", "")).strip().lower()
        normalized_pattern_type = str(item.get("normalized_pattern_type", "")).strip().lower()
        promotion_state = str(item.get("promotion_state", "candidate")).strip().lower()
        evidence_stage = str(item.get("evidence_stage", "")).strip().lower()

        bonus = 0.0
        if normalized_pattern_type in {"closure_signal", "negative_strategy", "patch_risk", "validation_gap"}:
            bonus += 0.14
        if rtype == "attempt_summary_v1":
            bonus += 0.1
        if family_id.startswith("belief_tip:") or family_id.startswith("workflow_step:"):
            bonus -= 0.24
        if normalized_pattern_type == "planning_loop":
            bonus -= 0.14
        if rtype == "belief_tip":
            bonus -= 0.1
        if promotion_state == "promoted":
            bonus += 0.08
        elif promotion_state == "candidate":
            bonus -= 0.04
        if evidence_stage == "submission":
            bonus -= 0.06
        return bonus

    @staticmethod
    def _changed_file_overlap_bonus(
        item: Dict[str, Any],
        *,
        current_file_hints: Optional[List[str]],
    ) -> float:
        if not current_file_hints:
            return 0.0
        row_patterns = item.get("changed_file_pattern")
        if not isinstance(row_patterns, list):
            return 0.0
        row_set = {str(x).strip().lower() for x in row_patterns if str(x).strip()}
        current_set = {str(x).strip().lower() for x in current_file_hints if str(x).strip()}
        if not row_set or not current_set:
            return 0.0
        overlap = len(row_set & current_set)
        return min(0.08, 0.04 * overlap)

    # Backward-compatible wrappers used by legacy test scripts.
    def query_rewriting(self, query: str) -> Dict[str, Any]:
        return self.query_rewriter.rewrite_for_planning(
            user_instruction=query,
            context_summary="",
            env_signature={},
        )

    def experience_distillation(self, experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
        recommendations = []
        for exp in experiences:
            recommendations.append(
                {
                    "type": "legacy_experience",
                    "recommendation": exp.get("solution") or exp.get("description", ""),
                    "source_task_id": exp.get("task_id", "unknown"),
                }
            )
        return {
            "recommendations": recommendations,
            "evidence_refs": [],
            "confidence": 0.5 if recommendations else 0.0,
        }

    def retrieve_for_planning(self,
                            task_context: Dict[str, Any],
                            current_action: str,
                            agent_name: str,
                            current_problem_file: Optional[ProblemFile] = None,
                            embedding_view: str = "emb_task_sem",
                            query_type: str = "planning",
                            runtime_guard: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Retrieve experience for planning-time injection.

          A: Planning-time
        """
        query_type = self._normalize_query_type(query_type)
        budget = self._dynamic_budget(query_type=query_type, current_action=current_action)
        # Rewrite query
        query = self.query_rewriter.rewrite_for_query_type(
            query_type=query_type,
            user_instruction=task_context.get("instruction", ""),
            context_summary=task_context.get("summary", ""),
            env_signature=task_context.get("env_signature", {}),
            current_action=current_action,
        )
        env_signature = task_context.get("env_signature") if isinstance(task_context.get("env_signature"), dict) else {}
        instance_id = str(env_signature.get("instance_id") or "").strip()
        attempt_id = str(env_signature.get("attempt_id") or "").strip()
        current_subproblem_type = self._infer_subproblem_type_for_planning(
            current_action=current_action,
            current_problem_file=current_problem_file,
        )
        current_strategy_label = self._infer_strategy_label_for_planning(
            current_action=current_action,
            subproblem_type=current_subproblem_type,
        )

        # Select relevant subgraphs
        max_subgraphs = max(6, budget["subgraph_budget"])
        structure_selected = self.subgraph_selector.select_for_planning(query, max_subgraphs=max_subgraphs)
        embedding_debug = self._collect_embedding_candidates(
            current_problem_file=current_problem_file,
            embedding_view=embedding_view,
            max_results=max(20, max_subgraphs * 4),
            min_similarity=0.28,
        )
        selected_subgraphs, fusion_debug = self._merge_hybrid_subgraphs(
            selected_subgraphs=structure_selected,
            embedding_task_scores=embedding_debug["task_scores"],
            query_type="planning",
            max_subgraphs=max(3, budget["subgraph_budget"] // 2),
        )

        # Distill experience
        result = self.experience_distiller.distill_planning_experience(selected_subgraphs)

        abstract_patterns = self._retrieve_abstract_patterns(
            query_text=f"{task_context.get('instruction', '')} {task_context.get('summary', '')} {current_action}",
            subproblem_type=current_subproblem_type,
            strategy_label=current_strategy_label,
            max_results=max(2, min(6, budget["recommendation_budget"] // 3)),
        )
        if abstract_patterns:
            abstract_refs = []
            for row in self.graph_store.query_abstract_experiences(
                query_text=f"{task_context.get('instruction', '')} {current_action}",
                subproblem_type=current_subproblem_type,
                strategy_label=current_strategy_label,
                max_results=2,
            ):
                for ref in (row.get("evidence_refs") or [])[:2]:
                    abstract_refs.append(ref)
            result["recommendations"] = abstract_patterns + result.get("recommendations", [])
            result["evidence_refs"] = abstract_refs + result.get("evidence_refs", [])
            result["abstract_patterns"] = abstract_patterns

        failure_cards = self._retrieve_failure_cards(
            query_text=f"{task_context.get('instruction', '')} {current_action}",
            error_type=None,
            max_results=max(1, budget["failure_card_budget"]),
        )
        if failure_cards:
            result["recommendations"] = failure_cards + result.get("recommendations", [])
            refs = []
            for card in failure_cards:
                refs.extend(card.get("evidence_refs", [])[:2])
            result["evidence_refs"] = refs + result.get("evidence_refs", [])
            result["failure_cards"] = failure_cards

        repair_patterns = self._retrieve_repair_patterns(
            query_text=f"{task_context.get('instruction', '')} {current_action}",
            error_type=None,
            max_results=max(1, budget["repair_pattern_budget"] // 2),
        )
        if repair_patterns:
            result["recommendations"] = repair_patterns + result.get("recommendations", [])
            refs = []
            for pattern in repair_patterns:
                refs.extend(pattern.get("evidence_refs", [])[:2])
            result["evidence_refs"] = refs + result.get("evidence_refs", [])
            result["repair_patterns"] = repair_patterns

        compiler_cards = self._retrieve_compiler_cards(
            query_text=f"{task_context.get('instruction', '')} {task_context.get('summary', '')} {current_action}",
            query_type=query_type,
            max_results=max(1, min(4, budget["recommendation_budget"] // 2)),
        )
        if compiler_cards:
            refs = []
            for card in compiler_cards:
                refs.extend(card.get("source_object_ids", [])[:2])
                refs.extend(card.get("evidence_refs", [])[:2])
            result["recommendations"] = compiler_cards + result.get("recommendations", [])
            result["evidence_refs"] = refs + result.get("evidence_refs", [])
            result["compiler_cards"] = compiler_cards

        current_file_hints: List[str] = []
        if current_problem_file is not None:
            for path in current_problem_file.touched_files:
                suffix = path.rsplit(".", 1)[-1].lower() if "." in path else ""
                if suffix:
                    current_file_hints.append(f".{suffix}")

        attempt_summary = self._retrieve_attempt_summary(
            instance_id=instance_id,
            current_attempt_id=attempt_id,
            current_problem_file=current_problem_file,
        )
        if attempt_summary:
            result["recommendations"] = [attempt_summary] + result.get("recommendations", [])
            result["attempt_summary"] = attempt_summary.get("attempt_summary", {})

        before_rerank = list(result.get("recommendations", []))
        reranked = self._dedup_and_rerank_recommendations(
            before_rerank,
            limit=budget["recommendation_budget"],
            current_file_hints=current_file_hints,
            runtime_guard=runtime_guard,
        )
        has_high_value = any(
            row.get("type") in {"repair_pattern_v2", "failure_card_v2"}
            or str(row.get("normalized_pattern_type", "")).strip().lower() in {"negative_strategy", "patch_risk", "validation_gap"}
            for row in reranked
        )

        if not has_high_value:
            beliefs = self.graph_store.get_relevant_beliefs(max_results=1)
            if beliefs:
                belief = beliefs[0]
                reranked = self._dedup_and_rerank_recommendations(
                    [
                        {
                            "type": "belief_tip",
                            "belief_id": belief.belief_id,
                            "belief_type": belief.belief_type.value,
                            "recommendation": belief.rule.recommend or belief.rule.trigger,
                            "confidence": belief.confidence,
                        }
                    ]
                    + reranked,
                    limit=budget["recommendation_budget"],
                    current_file_hints=current_file_hints,
                    runtime_guard=runtime_guard,
                )
        result["recommendations"] = reranked

        # Add query context
        result["query"] = query
        result["selected_subgraph_count"] = len(selected_subgraphs)
        result["retrieval_debug"] = {
            "query_type": query_type,
            "embedding_view": embedding_view,
            "candidate_count_before_filter": embedding_debug["candidate_count_before_filter"],
            "candidate_count_after_filter": embedding_debug["candidate_count_after_filter"],
            "candidate_task_count": embedding_debug["candidate_task_count"],
            "selected_subgraph_count": len(selected_subgraphs),
            "recommendation_count": len(before_rerank),
            "injection_candidate_count": len(before_rerank),
            "injection_selected_count": len(result.get("recommendations", [])),
            "ranked_tasks": fusion_debug["ranked_tasks"],
            "fusion_weights": fusion_debug["fusion_weights"],
            "budget": budget,
            "runtime_guard": runtime_guard or {},
            "subproblem_type": current_subproblem_type,
            "strategy_label": current_strategy_label,
            "compiler_card_count": len(result.get("compiler_cards", []) or []),
        }
        result["agent"] = agent_name
        result["timestamp"] = datetime.now(timezone.utc).isoformat()

        return result

    def retrieve_for_repair(self,
                           error_type: str,
                           error_message: str,
                           current_action: str,
                           problem_file: ProblemFile,
                           embedding_view: str = "emb_error_sig",
                           query_type: str = "error_recovery") -> Dict[str, Any]:
        """
        Retrieve experience for execution-time repair.

          B: Execution-time
        """
        query_type = self._normalize_query_type(query_type)
        budget = self._dynamic_budget(
            query_type=query_type,
            current_action=current_action,
            error_message=error_message,
        )
        # Create env signature from problem file
        env_signature = {}
        if problem_file.env_signature:
            env_signature = {
                "toolchain": problem_file.env_signature.toolchain_version,
                "path_hash": problem_file.env_signature.path_hash,
                "working_dir": problem_file.env_signature.working_dir,
            }

        # Rewrite query
        query = self.query_rewriter.rewrite_for_query_type(
            query_type=query_type,
            error_type=error_type,
            error_message=error_message,
            current_action=current_action,
            env_signature=env_signature,
        )

        # Select relevant subgraphs
        max_subgraphs = max(8, budget["subgraph_budget"])
        structure_selected = self.subgraph_selector.select_for_repair(query, max_subgraphs=max_subgraphs)
        embedding_debug = self._collect_embedding_candidates(
            current_problem_file=problem_file,
            embedding_view=embedding_view,
            max_results=max(30, max_subgraphs * 4),
            min_similarity=0.24,
        )
        selected_subgraphs, fusion_debug = self._merge_hybrid_subgraphs(
            selected_subgraphs=structure_selected,
            embedding_task_scores=embedding_debug["task_scores"],
            query_type="repair",
            max_subgraphs=max(5, budget["subgraph_budget"] // 2),
        )

        # Distill experience
        result = self.experience_distiller.distill_repair_experience(
            selected_subgraphs, problem_file
        )

        # Inject pitfall beliefs that match current error signature/action type.
        beliefs = self.graph_store.get_relevant_beliefs(
            action_type=query.get("action_type"),
            max_results=5,
        )
        if beliefs:
            filtered_beliefs = []
            for belief in beliefs:
                sig = (belief.condition_signature.error_signature or "").lower()
                if not sig or sig in error_type.lower() or sig in error_message.lower():
                    filtered_beliefs.append(belief)
            if filtered_beliefs:
                belief_suggestions = []
                belief_refs = []
                for belief in filtered_beliefs[:3]:
                    belief_suggestions.append(
                        {
                            "pattern": "belief_repair_hint",
                            "belief_id": belief.belief_id,
                            "repair_action": belief.rule.recommend or belief.rule.trigger,
                            "avoid_action": belief.rule.avoid,
                            "confidence": belief.confidence,
                        }
                    )
                    belief_refs.extend(belief.evidence_refs[:2])
                result["recommendations"] = belief_suggestions + result.get("recommendations", [])
                result["evidence_refs"] = belief_refs + result.get("evidence_refs", [])

        abstract_patterns = self._retrieve_abstract_patterns(
            query_text=f"{error_type} {error_message} {current_action}",
            error_type=error_type,
            max_results=max(2, min(8, budget["recommendation_budget"] // 3)),
        )
        if abstract_patterns:
            abstract_refs = []
            for row in self.graph_store.query_abstract_experiences(
                query_text=f"{error_type} {current_action}",
                error_type=error_type,
                max_results=2,
            ):
                for ref in (row.get("evidence_refs") or [])[:2]:
                    abstract_refs.append(ref)
            result["recommendations"] = abstract_patterns + result.get("recommendations", [])
            result["evidence_refs"] = abstract_refs + result.get("evidence_refs", [])
            result["abstract_patterns"] = abstract_patterns

        failure_cards = self._retrieve_failure_cards(
            query_text=f"{error_type} {error_message} {current_action}",
            error_type=error_type,
            max_results=max(2, budget["failure_card_budget"]),
        )
        if failure_cards:
            result["recommendations"] = failure_cards + result.get("recommendations", [])
            refs = []
            for card in failure_cards:
                refs.extend(card.get("evidence_refs", [])[:2])
            result["evidence_refs"] = refs + result.get("evidence_refs", [])
            result["failure_cards"] = failure_cards

        repair_patterns = self._retrieve_repair_patterns(
            query_text=f"{error_type} {error_message} {current_action}",
            error_type=error_type,
            max_results=max(2, budget["repair_pattern_budget"]),
        )
        if repair_patterns:
            result["recommendations"] = repair_patterns + result.get("recommendations", [])
            refs = []
            for pattern in repair_patterns:
                refs.extend(pattern.get("evidence_refs", [])[:2])
            result["evidence_refs"] = refs + result.get("evidence_refs", [])
            result["repair_patterns"] = repair_patterns

        compiler_cards = self._retrieve_compiler_cards(
            query_text=f"{error_type} {error_message} {current_action}",
            query_type=query_type,
            max_results=max(1, min(4, budget["recommendation_budget"] // 2)),
        )
        if compiler_cards:
            refs = []
            for card in compiler_cards:
                refs.extend(card.get("source_object_ids", [])[:2])
                refs.extend(card.get("evidence_refs", [])[:2])
            result["recommendations"] = compiler_cards + result.get("recommendations", [])
            result["evidence_refs"] = refs + result.get("evidence_refs", [])
            result["compiler_cards"] = compiler_cards

        before_rerank = list(result.get("recommendations", []))
        result["recommendations"] = self._dedup_and_rerank_recommendations(
            before_rerank,
            limit=budget["recommendation_budget"],
        )

        # Add query context
        result["query"] = query
        result["selected_subgraph_count"] = len(selected_subgraphs)
        result["retrieval_debug"] = {
            "query_type": query_type,
            "embedding_view": embedding_view,
            "candidate_count_before_filter": embedding_debug["candidate_count_before_filter"],
            "candidate_count_after_filter": embedding_debug["candidate_count_after_filter"],
            "candidate_task_count": embedding_debug["candidate_task_count"],
            "selected_subgraph_count": len(selected_subgraphs),
            "recommendation_count": len(before_rerank),
            "injection_candidate_count": len(before_rerank),
            "injection_selected_count": len(result.get("recommendations", [])),
            "ranked_tasks": fusion_debug["ranked_tasks"],
            "fusion_weights": fusion_debug["fusion_weights"],
            "budget": budget,
            "compiler_card_count": len(result.get("compiler_cards", []) or []),
        }
        result["error_type"] = error_type
        result["timestamp"] = datetime.now(timezone.utc).isoformat()

        return result

    def retrieve_for_query_type(
        self,
        *,
        query_type: str,
        task_context: Optional[Dict[str, Any]] = None,
        current_action: str = "",
        agent_name: str = "unknown",
        current_problem_file: Optional[ProblemFile] = None,
        error_type: str = "",
        error_message: str = "",
        embedding_view: Optional[str] = None,
        runtime_guard: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized = self._normalize_query_type(query_type)
        if normalized in {"planning", "regression_guard"}:
            return self.retrieve_for_planning(
                task_context=task_context or {},
                current_action=current_action,
                agent_name=agent_name,
                current_problem_file=current_problem_file,
                embedding_view=embedding_view or "emb_task_sem",
                query_type=normalized,
                runtime_guard=runtime_guard,
            )
        if current_problem_file is None:
            current_problem_file = ProblemFile(
                task_id="__repair_query__",
                action_type=ActionType.TOOL_CALL,
                intent_text=current_action,
                action_text=current_action,
                outcome=Outcome.UNKNOWN,
                metadata={"source": "retrieve_for_query_type"},
            )
        return self.retrieve_for_repair(
            error_type=error_type or "unknown",
            error_message=error_message,
            current_action=current_action,
            problem_file=current_problem_file,
            embedding_view=embedding_view or "emb_error_sig",
            query_type=normalized,
        )

    def update_beliefs(self,
                      task_id: str,
                      outcome: str,
                      evidence_refs: List[str]) -> Dict[str, Any]:
        """
        Update beliefs based on task outcome.

        Args:
            task_id: Task ID
            outcome: "success" or "fail"
            evidence_refs: Evidence references from the task

        Returns:
            Update report
        """
        subgraph = self.graph_store.observation_kg.get_task_subgraph(task_id)
        if not subgraph:
            return {
                "task_id": task_id,
                "outcome": outcome,
                "beliefs_updated": 0,
                "new_beliefs_created": 0,
                "error": "task_not_found",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        actions = self.graph_store.get_task_actions(task_id)
        if not actions:
            return {
                "task_id": task_id,
                "outcome": outcome,
                "beliefs_updated": 0,
                "new_beliefs_created": 0,
                "error": "task_has_no_actions",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        success = str(outcome).lower() == "success"
        success_count = sum(1 for a in actions if a.outcome.value == "success")
        fail_count = sum(1 for a in actions if a.outcome.value == "fail")
        attempt_evidence = list(dict.fromkeys(evidence_refs + [a.action_id for a in actions[:3]]))
        attempt_belief = AttemptBelief(
            task_id=task_id,
            summary=f"Task {task_id}: total={len(actions)}, success={success_count}, fail={fail_count}",
            failure_causal_chain=[a.action_id for a in actions if a.outcome.value == "fail"][:5],
            key_turning_points=[actions[0].action_id, actions[-1].action_id] if len(actions) > 1 else [actions[0].action_id],
            reusable_workflow_points=[a.intent_text[:120] for a in actions if a.outcome.value == "success"][:3],
            evidence_refs=attempt_evidence[:20],
            metadata={"source": "memory_agent.update_beliefs"},
        )
        self.graph_store.add_attempt_belief(attempt_belief)

        last_action = actions[-1]
        failed_actions = [a for a in actions if a.outcome.value == "fail"]
        latest_fail = failed_actions[-1] if failed_actions else None

        env_cluster = None
        repo_toolchain = None
        if last_action.env_signature:
            env_cluster = last_action.env_signature.path_hash
            repo_toolchain = last_action.env_signature.toolchain_version

        belief_type = BeliefType.WORKFLOW if success else BeliefType.PITFALL
        error_signature = None
        if latest_fail and latest_fail.failure_signature:
            error_signature = latest_fail.failure_signature.error_type

        recommend = None
        avoid = None
        if success:
            success_chain = subgraph.get_success_chain()
            if success_chain:
                chain_intents = [
                    subgraph.action_nodes[action_id].intent_text
                    for action_id in success_chain
                    if action_id in subgraph.action_nodes
                ]
                recommend = " -> ".join([i[:60] for i in chain_intents[:3]]) or last_action.intent_text[:120]
            else:
                recommend = last_action.intent_text[:120]
        else:
            avoid = latest_fail.intent_text[:120] if latest_fail else last_action.intent_text[:120]
            retry_chains = subgraph.get_failure_retry_chains()
            if latest_fail and latest_fail.action_id in retry_chains:
                chain = retry_chains[latest_fail.action_id]
                if len(chain) > 1 and chain[1] in subgraph.action_nodes:
                    recommend = subgraph.action_nodes[chain[1]].intent_text[:120]
            if not recommend:
                recommend = f"Repair after {error_signature or 'failure'} by narrowing context and re-validating"

        condition = ConditionSignature(
            env_cluster=env_cluster,
            repo_toolchain=repo_toolchain,
            action_type_pattern=last_action.action_type.value,
            intent_pattern=last_action.intent_text[:80],
            error_signature=error_signature,
        )
        rule = BeliefRule(
            trigger=f"When action_type={last_action.action_type.value}",
            recommend=recommend,
            avoid=avoid,
        )

        def _same_signature(belief: AtomicBelief) -> bool:
            cs = belief.condition_signature
            return (
                belief.belief_type == belief_type
                and cs.action_type_pattern == condition.action_type_pattern
                and cs.error_signature == condition.error_signature
                and cs.env_cluster == condition.env_cluster
            )

        existing = None
        for belief in self.graph_store.belief_graph.atomic_beliefs.values():
            if _same_signature(belief):
                existing = belief
                break

        new_created = 0
        updated = 0
        belief_ids = []
        belief_evidence = list(dict.fromkeys(evidence_refs + [a.action_id for a in actions[-5:]]))

        if existing:
            # Keep most recent actionable wording.
            existing.rule = rule
            existing.evidence_refs = list(dict.fromkeys(existing.evidence_refs + belief_evidence))[:50]
            self.graph_store.update_belief_stats(
                existing.belief_id,
                success_with=success,
                success_without=not success,
            )
            belief_ids.append(existing.belief_id)
            updated += 1
        else:
            new_belief = AtomicBelief(
                belief_type=belief_type,
                condition_signature=condition,
                rule=rule,
                confidence=0.0,
                evidence_refs=belief_evidence[:50],
                metadata={"source": "memory_agent.update_beliefs", "task_id": task_id},
            )
            belief_id = self.graph_store.add_atomic_belief(new_belief)
            self.graph_store.update_belief_stats(
                belief_id,
                success_with=success,
                success_without=not success,
            )
            belief_ids.append(belief_id)
            new_created += 1

        return {
            "task_id": task_id,
            "outcome": outcome,
            "attempt_belief_id": attempt_belief.attempt_id,
            "belief_ids": belief_ids,
            "beliefs_updated": updated,
            "new_beliefs_created": new_created,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        kg_stats = self.graph_store.observation_kg.get_statistics()
        belief_stats = self.graph_store.belief_graph.get_statistics()

        return {
            "observation_kg": kg_stats,
            "belief_graph": belief_stats,
            "total_experience_units": kg_stats.get("total_actions", 0),
            "total_beliefs": belief_stats.get("total_atomic_beliefs", 0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
