"""Root-cause analysis over failure and retry chains."""

from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime
from enum import Enum

from ..core.problem_file import ProblemFile, Outcome, ActionType
from ..core.observation_kg import ObservationKG, EdgeType, TaskSubgraph


class ErrorModule(Enum):
    """Subsystem categories used in root-cause analysis."""
    MEMORY = "memory"
    REFLECTION = "reflection"
    PLANNING = "planning"
    ACTION = "action"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class ErrorType(Enum):
    """ """
    UNKNOWN = "unknown"
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    FILE_NOT_FOUND = "file_not_found"
    PERMISSION_ERROR = "permission_error"
    TIMEOUT = "timeout"
    ASSERTION_FAILED = "assertion_failed"
    TEST_FAILURE = "test_failure"
    COMPILATION_ERROR = "compilation_error"
    RUNTIME_ERROR = "runtime_error"
    LOGIC_ERROR = "logic_error"
    CONFIGURATION_ERROR = "configuration_error"


class RCAAgent:
    """
    RCA Agent


    """

    def __init__(self, observation_kg: ObservationKG):
        """
         RCA Agent

        Args:
            observation_kg: Observation KG
        """
        self.observation_kg = observation_kg

    def analyze_task_failure(self, task_id: str) -> Dict[str, Any]:
        """


        Args:
            task_id:  ID

        Returns:
            RCA
        """

        raw_subgraph = self.observation_kg.get_task_subgraph(task_id)
        task_subgraph = self._normalize_task_subgraph(raw_subgraph)
        if not task_subgraph.get("nodes"):
            return {"error": "Task not found", "task_id": task_id}


        failure_nodes = self._identify_failure_nodes(task_subgraph)


        if not failure_nodes:
            return {
                "task_id": task_id,
                "has_failures": False,
                "root_cause_nodes": [],
                "propagation_chain": [],
                "propagation_chains": [],
                "corrective_actions": [],
                "error_module": "unknown",
                "confidence": 0.0
            }


        analysis_results = []
        for failure_node in failure_nodes:
            analysis = self._analyze_single_failure(failure_node, task_subgraph)
            analysis_results.append(analysis)


        combined_result = self._combine_analysis_results(analysis_results, task_id)

        return combined_result

    def _normalize_task_subgraph(self, subgraph: Optional[Any]) -> Dict[str, Any]:
        """Normalize TaskSubgraph object to legacy dict shape used by RCA logic."""
        if subgraph is None:
            return {"nodes": [], "edges": []}
        if isinstance(subgraph, dict):
            return subgraph
        if isinstance(subgraph, TaskSubgraph):
            nodes: List[Dict[str, Any]] = []
            for action_id, pf in subgraph.action_nodes.items():
                nodes.append({"id": action_id, "data": pf.to_dict()})
            edges: List[Dict[str, Any]] = []
            for edge in subgraph.edges:
                edges.append(
                    {
                        "source": edge.source_id,
                        "target": edge.target_id,
                        "edge_type": edge.edge_type.value,
                        "metadata": edge.metadata,
                    }
                )
            return {"nodes": nodes, "edges": edges}
        return {"nodes": [], "edges": []}

    def _identify_failure_nodes(self, task_subgraph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """ """
        failure_nodes = []
        for node in task_subgraph.get("nodes", []):
            node_data = node.get("data", {})
            outcome = node_data.get("outcome", "")

            if outcome == "fail":
                failure_nodes.append(node)

        return failure_nodes

    def _analyze_single_failure(self,
                               failure_node: Dict[str, Any],
                               task_subgraph: Dict[str, Any]) -> Dict[str, Any]:
        """ """
        node_id = failure_node.get("id")
        node_data = failure_node.get("data", {})


        failure_signature = node_data.get("failure_signature", {})
        error_type_str = failure_signature.get("error_type", "unknown")
        action_type = node_data.get("action_type", "")
        intent_text = node_data.get("intent_text", "")


        error_type = self._classify_error_type(error_type_str, intent_text)


        root_cause_candidates = self._find_root_cause_candidates(
            node_id, task_subgraph, error_type
        )


        ranked_candidates = self._rank_root_cause_candidates(
            root_cause_candidates=root_cause_candidates,
            failure_node=failure_node,
            task_subgraph=task_subgraph,
            error_type=error_type,
        )
        root_cause_nodes = [row["node_id"] for row in ranked_candidates[:2]]


        propagation_chain = self._build_propagation_chain(
            root_cause_nodes, node_id, task_subgraph
        )


        corrective_actions = self._generate_corrective_actions(
            root_cause_nodes, error_type, action_type
        )


        preventive_checks = self._generate_preventive_checks(
            root_cause_nodes, error_type, action_type
        )


        error_module = self._determine_error_module(root_cause_nodes, error_type, action_type)

        return {
            "failure_node_id": node_id,
            "error_type": error_type.value,
            "error_module": error_module.value,
            "root_cause_nodes": root_cause_nodes,
            "propagation_chain": propagation_chain,
            "corrective_actions": corrective_actions,
            "preventive_checks": preventive_checks,
            "decisive_steps": ranked_candidates[:5],
            "confidence": self._calculate_confidence(root_cause_nodes, error_type),
            "evidence_refs": list(dict.fromkeys(root_cause_nodes + [node_id]))
        }

    def _classify_error_type(self, error_type_str: str, intent_text: str) -> ErrorType:
        """ """
        error_type_str = error_type_str.lower()


        error_mapping = {
            "syntax": ErrorType.SYNTAX_ERROR,
            "import": ErrorType.IMPORT_ERROR,
            "file": ErrorType.FILE_NOT_FOUND,
            "permission": ErrorType.PERMISSION_ERROR,
            "timeout": ErrorType.TIMEOUT,
            "assert": ErrorType.ASSERTION_FAILED,
            "test": ErrorType.TEST_FAILURE,
            "compile": ErrorType.COMPILATION_ERROR,
            "runtime": ErrorType.RUNTIME_ERROR,
            "logic": ErrorType.LOGIC_ERROR,
            "config": ErrorType.CONFIGURATION_ERROR,
        }

        for key, error_enum in error_mapping.items():
            if key in error_type_str:
                return error_enum


        intent_lower = intent_text.lower()
        if "test" in intent_lower:
            return ErrorType.TEST_FAILURE
        elif "import" in intent_lower or "require" in intent_lower:
            return ErrorType.IMPORT_ERROR
        elif "file" in intent_lower:
            return ErrorType.FILE_NOT_FOUND

        return ErrorType.UNKNOWN

    def _find_root_cause_candidates(self,
                                   failure_node_id: str,
                                   task_subgraph: Dict[str, Any],
                                   error_type: ErrorType) -> List[Dict[str, Any]]:
        """ """
        candidates = []


        all_nodes = task_subgraph.get("nodes", [])

        for node in all_nodes:
            node_id = node.get("id")
            if node_id == failure_node_id:
                continue

            node_data = node.get("data", {})
            node_outcome = node_data.get("outcome", "")


            if node_outcome == "fail":
                node_error_type = self._classify_error_type(
                    node_data.get("failure_signature", {}).get("error_type", ""),
                    node_data.get("intent_text", "")
                )
                if node_error_type == error_type:
                    candidates.append(node)

        failure_node = next(
            (n for n in all_nodes if n.get("id") == failure_node_id),
            None
        )
        if failure_node:
            failure_data = failure_node.get("data", {})
            failure_files = set(failure_data.get("touched_files", []))

            for node in all_nodes:
                node_id = node.get("id")
                if node_id == failure_node_id:
                    continue

                node_data = node.get("data", {})
                node_files = set(node_data.get("touched_files", []))


                if failure_files.intersection(node_files):
                    candidates.append(node)

        deduped: Dict[str, Dict[str, Any]] = {}
        for node in candidates:
            node_id = str(node.get("id"))
            if not node_id:
                continue
            deduped[node_id] = node
        return list(deduped.values())

    def _rank_root_cause_candidates(
        self,
        *,
        root_cause_candidates: List[Dict[str, Any]],
        failure_node: Dict[str, Any],
        task_subgraph: Dict[str, Any],
        error_type: ErrorType,
    ) -> List[Dict[str, Any]]:
        """Rank candidates by decisive_score; tie-break by earliest step."""
        ranked: List[Dict[str, Any]] = []
        failure_id = str(failure_node.get("id", ""))
        for node in root_cause_candidates:
            node_id = str(node.get("id", ""))
            if not node_id:
                continue
            score, reasons = self._score_candidate(
                candidate=node,
                failure_node=failure_node,
                task_subgraph=task_subgraph,
                error_type=error_type,
            )
            step_index = self._extract_step_index(node.get("data", {}))
            ranked.append(
                {
                    "node_id": node_id,
                    "decisive_score": round(score, 6),
                    "reasons": reasons,
                    "step_index": step_index,
                    "is_failure_node": node_id == failure_id,
                }
            )

        ranked.sort(
            key=lambda row: (
                float(row["decisive_score"]),
                -int(row["step_index"]),
            ),
            reverse=True,
        )

        # Apply "earliest decisive step first" for equal scores.
        i = 0
        while i < len(ranked):
            j = i + 1
            while j < len(ranked) and ranked[j]["decisive_score"] == ranked[i]["decisive_score"]:
                j += 1
            if j - i > 1:
                ranked[i:j] = sorted(
                    ranked[i:j],
                    key=lambda row: int(row["step_index"]) if row["step_index"] >= 0 else 10**9,
                )
            i = j
        return ranked

    def _score_candidate(
        self,
        *,
        candidate: Dict[str, Any],
        failure_node: Dict[str, Any],
        task_subgraph: Dict[str, Any],
        error_type: ErrorType,
    ) -> Tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []
        cand_data = candidate.get("data", {})
        fail_data = failure_node.get("data", {})

        cand_error = self._classify_error_type(
            cand_data.get("failure_signature", {}).get("error_type", ""),
            cand_data.get("intent_text", ""),
        )
        if cand_error == error_type:
            score += 0.45
            reasons.append("same_error_type")

        cand_outcome = str(cand_data.get("outcome", "")).lower()
        if cand_outcome == "fail":
            score += 0.2
            reasons.append("candidate_failed")

        cand_files = set(cand_data.get("touched_files", []) or [])
        fail_files = set(fail_data.get("touched_files", []) or [])
        if cand_files and fail_files and cand_files.intersection(fail_files):
            score += 0.2
            reasons.append("shared_touched_files")

        cand_step = self._extract_step_index(cand_data)
        fail_step = self._extract_step_index(fail_data)
        if cand_step >= 0 and fail_step >= 0 and cand_step <= fail_step:
            score += 0.1
            reasons.append("happened_before_failure")

        if str(candidate.get("id", "")) in self._find_ancestors(str(failure_node.get("id", "")), task_subgraph):
            score += 0.15
            reasons.append("graph_ancestor")

        return min(score, 1.0), reasons

    def _find_ancestors(self, node_id: str, task_subgraph: Dict[str, Any]) -> Set[str]:
        ancestors: Set[str] = set()
        edges = task_subgraph.get("edges", [])
        reverse: Dict[str, List[str]] = {}
        for edge in edges:
            src = str(edge.get("source", ""))
            dst = str(edge.get("target", ""))
            if src and dst:
                reverse.setdefault(dst, []).append(src)
        queue = [node_id]
        while queue:
            current = queue.pop()
            for prev in reverse.get(current, []):
                if prev in ancestors:
                    continue
                ancestors.add(prev)
                queue.append(prev)
        return ancestors

    @staticmethod
    def _extract_step_index(node_data: Dict[str, Any]) -> int:
        value = node_data.get("step_index")
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return -1

    def _select_root_causes(self,
                           candidates: List[Dict[str, Any]],
                           task_subgraph: Dict[str, Any]) -> List[str]:
        """ """
        if not candidates:
            return []


        candidates_with_times = []
        for candidate in candidates:
            node_data = candidate.get("data", {})
            timestamp_str = node_data.get("timestamp", "")
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                candidates_with_times.append((candidate, timestamp))
            except:
                continue


        candidates_with_times.sort(key=lambda x: x[1])

        root_causes = [candidate[0].get("id") for candidate in candidates_with_times[:2]]

        return root_causes

    def _build_propagation_chain(self,
                                root_cause_ids: List[str],
                                failure_node_id: str,
                                task_subgraph: Dict[str, Any]) -> List[str]:
        """Build a minimal propagation chain ending at the failure node."""
        if not root_cause_ids:
            return []

        chain = root_cause_ids.copy()
        if failure_node_id not in chain:
            chain.append(failure_node_id)

        return chain

    def _generate_corrective_actions(self,
                                    root_cause_ids: List[str],
                                    error_type: ErrorType,
                                    action_type: str) -> List[Dict[str, Any]]:
        """Generate corrective actions for the classified error."""
        actions = []


        corrective_templates = {
            ErrorType.SYNTAX_ERROR: [
                {
                    "action": "review_syntax",
                    "description": "Review syntax near the reported line.",
                    "parameters": {"error_location": "near the reported line"}
                }
            ],
            ErrorType.IMPORT_ERROR: [
                {
                    "action": "install_missing_package",
                    "description": "Install the missing package.",
                    "parameters": {"package_name": "extract from error message"}
                },
                {
                    "action": "add_to_path",
                    "description": "Add the missing directory to PYTHONPATH.",
                    "parameters": {"path": "extract from error message"}
                }
            ],
            ErrorType.FILE_NOT_FOUND: [
                {
                    "action": "check_file_path",
                    "description": "Check the referenced file path.",
                    "parameters": {"expected_path": "from action context"}
                },
                {
                    "action": "create_missing_file",
                    "description": "Create the missing file when required.",
                    "parameters": {"file_path": "from action context"}
                }
            ],
            ErrorType.PERMISSION_ERROR: [
                {
                    "action": "change_permissions",
                    "description": "Adjust file or directory permissions.",
                    "parameters": {"path": "from action context", "mode": "755"}
                }
            ],
            ErrorType.TIMEOUT: [
                {
                    "action": "increase_timeout",
                    "description": "Increase the operation timeout.",
                    "parameters": {"timeout_seconds": "current * 2"}
                },
                {
                    "action": "optimize_operation",
                    "description": "Optimize the operation before retrying.",
                    "parameters": {}
                }
            ]
        }


        if error_type in corrective_templates:
            actions.extend(corrective_templates[error_type])


        for root_cause_id in root_cause_ids:
            actions.append({
                "action": "review_previous_action",
                "description": f"Review previous action {root_cause_id}.",
                "parameters": {"action_id": root_cause_id}
            })

        return actions

    def _generate_preventive_checks(self,
                                   root_cause_ids: List[str],
                                   error_type: ErrorType,
                                   action_type: str) -> List[Dict[str, Any]]:
        """ """
        checks = []


        preventive_templates = {
            ErrorType.SYNTAX_ERROR: [
                {
                    "check": "syntax_check_before_execution",
                    "description": " ",
                    "trigger": "before code execution"
                }
            ],
            ErrorType.IMPORT_ERROR: [
                {
                    "check": "verify_imports",
                    "description": " ",
                    "trigger": "before execution"
                }
            ],
            ErrorType.FILE_NOT_FOUND: [
                {
                    "check": "verify_file_existence",
                    "description": " ",
                    "trigger": "before file operations"
                }
            ],
            ErrorType.PERMISSION_ERROR: [
                {
                    "check": "check_permissions",
                    "description": " ",
                    "trigger": "before file operations"
                }
            ]
        }

        if error_type in preventive_templates:
            checks.extend(preventive_templates[error_type])

        return checks

    def _determine_error_module(self,
                               root_cause_ids: List[str],
                               error_type: ErrorType,
                               action_type: str) -> ErrorModule:
        """ """
        if error_type in [ErrorType.SYNTAX_ERROR, ErrorType.LOGIC_ERROR]:
            return ErrorModule.PLANNING
        elif error_type in [ErrorType.IMPORT_ERROR, ErrorType.FILE_NOT_FOUND, ErrorType.PERMISSION_ERROR]:
            return ErrorModule.ACTION
        elif error_type in [ErrorType.TEST_FAILURE, ErrorType.ASSERTION_FAILED]:
            return ErrorModule.SYSTEM
        else:
            return ErrorModule.UNKNOWN

    def _calculate_confidence(self,
                            root_cause_ids: List[str],
                            error_type: ErrorType) -> float:
        """ """
        if not root_cause_ids:
            return 0.0


        confidence = 0.5


        if len(root_cause_ids) == 1:
            confidence += 0.2
        elif len(root_cause_ids) > 2:
            confidence -= 0.1


        if error_type != ErrorType.UNKNOWN:
            confidence += 0.2

        return min(max(confidence, 0.0), 1.0)

    def _combine_analysis_results(self,
                                 analysis_results: List[Dict[str, Any]],
                                 task_id: str) -> Dict[str, Any]:
        """ """
        if not analysis_results:
            return {
                "task_id": task_id,
                "has_failures": False,
                "root_cause_nodes": [],
                "propagation_chain": [],
                "propagation_chains": [],
                "corrective_actions": [],
                "error_module": "unknown",
                "confidence": 0.0
            }


        all_root_causes = []
        for result in analysis_results:
            all_root_causes.extend(result.get("root_cause_nodes", []))


        unique_root_causes = list(set(all_root_causes))


        all_corrective_actions = []
        for result in analysis_results:
            all_corrective_actions.extend(result.get("corrective_actions", []))


        all_propagation_chains = [result.get("propagation_chain", []) for result in analysis_results]
        primary_chain = max(all_propagation_chains, key=lambda chain: len(chain), default=[])


        confidences = [result.get("confidence", 0.0) for result in analysis_results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0


        error_types = [result.get("error_type", "unknown") for result in analysis_results]
        error_modules = [result.get("error_module", "unknown") for result in analysis_results]
        from collections import Counter
        most_common_error = Counter(error_types).most_common(1)
        most_common_module = Counter(error_modules).most_common(1)
        primary_error_type = most_common_error[0][0] if most_common_error else "unknown"
        primary_error_module = most_common_module[0][0] if most_common_module else "unknown"

        return {
            "task_id": task_id,
            "has_failures": True,
            "primary_error_type": primary_error_type,
            "error_module": primary_error_module,
            "root_cause_nodes": unique_root_causes,
            "propagation_chain": primary_chain,
            "propagation_chains": all_propagation_chains,
            "corrective_actions": all_corrective_actions[:5],
            "confidence": avg_confidence,
            "num_failures_analyzed": len(analysis_results),
            "analysis_timestamp": datetime.now().isoformat()
        }
