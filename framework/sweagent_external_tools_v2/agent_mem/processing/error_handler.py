"""Execution-time error analysis and repair suggestion generation."""

from typing import Dict, List, Optional, Any, Tuple
import json
from datetime import datetime

from ..core.problem_file import ProblemFile, ActionType, Outcome
from ..core.observation_kg import ObservationKG, EdgeType
from ..retrieval.retriever import HierarchicalRetriever
from ..storage.graph_store import GraphStore


class ErrorHandler:
    """Build repair queries, retrieve prior failures, and rank suggestions."""

    def __init__(self,
                 graph_store: GraphStore,
                 observation_kg: ObservationKG,
                 retriever: HierarchicalRetriever = None):
        """Initialize the error handler.

        Args:
            graph_store: Persistent graph store
            observation_kg: Observation KG
            retriever: Optional hierarchical retriever
        """
        self.graph_store = graph_store
        self.observation_kg = observation_kg
        self.retriever = retriever

    def handle_error(self,
                     error_context: Dict[str, Any],
                     current_problem_file: Optional[ProblemFile] = None) -> Dict[str, Any]:
        """Analyze one execution error and return ranked repair suggestions.

        Args:
            error_context: Structured error details
            current_problem_file: Optional action record associated with the error

        Returns:
            Error report with retrieval results and suggestions
        """

        error_type = error_context.get("error_type", "unknown")
        error_message = error_context.get("error_message", "")
        current_action = error_context.get("action", "")
        thought = error_context.get("thought", "")
        env_signature = error_context.get("env_signature", {})


        query = self._build_repair_query(
            error_type, error_message, current_action, env_signature
        )


        if self.retriever:
            retrieval_results = self.retriever.retrieve_for_execution(query, limit=3)
        else:
            retrieval_results = self._simple_retrieval(query)


        suggestions = self._generate_repair_suggestions(
            retrieval_results, error_context, current_problem_file
        )


        if current_problem_file:
            self._record_error(current_problem_file, error_context, suggestions)

        return {
            "error_type": error_type,
            "error_message": error_message[:500],
            "retrieval_query": query,
            "retrieval_results": retrieval_results,
            "repair_suggestions": suggestions,
            "generated_at": datetime.now().isoformat()
        }

    def analyze_error(self, error_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Compatibility wrapper for legacy callers expecting a suggestion list."""
        result = self.handle_error(error_context, current_problem_file=None)
        return result.get("repair_suggestions", [])

    def _build_repair_query(self,
                           error_type: str,
                           error_message: str,
                           current_action: str,
                           env_signature: Dict[str, Any]) -> Dict[str, Any]:
        """Build a structured repair-retrieval query."""
        return {
            "error_type": error_type,
            "error_message": error_message,
            "current_action": current_action,
            "env_signature": env_signature,
            "key_tokens": self._extract_key_tokens(error_message),
            "timestamp": datetime.now().isoformat()
        }

    def _simple_retrieval(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve similar failures directly from the observation graph.

        Args:
            query: Structured repair query

        Returns:
            Similar failures and linked repair actions
        """
        error_type = query.get("error_type", "")
        key_tokens = query.get("key_tokens", [])

        similar_failures = []


        if hasattr(self.observation_kg, 'search_by_error_signature'):
            error_signature = {
                "error_type": error_type,
                "key_tokens": key_tokens
            }
            similar_failures = self.observation_kg.search_by_error_signature(error_signature)


        repair_suggestions = []
        for failure in similar_failures[:3]:
            repair_info = self._find_repair_for_failure(failure)
            if repair_info:
                repair_suggestions.append(repair_info)

        return {
            "similar_failures": similar_failures[:5],
            "repair_suggestions": repair_suggestions,
            "evidence_refs": []
        }

    def _generate_repair_suggestions(self,
                                    retrieval_results: Dict[str, Any],
                                    error_context: Dict[str, Any],
                                    current_problem_file: Optional[ProblemFile]) -> List[Dict[str, Any]]:
        """Generate and rank repair suggestions."""
        suggestions = []

        # 1.
        repair_suggestions = retrieval_results.get("repair_suggestions", [])
        for repair in repair_suggestions[:3]:
            suggestion = {
                "type": "retrieved_repair",
                "confidence": repair.get("similarity", 0.5),
                "description": repair.get("evidence", "Retrieved repair suggestion"),
                "action": repair.get("repair_action", ""),
                "source": repair.get("failure_id", "unknown"),
                "evidence_refs": [repair.get("failure_id"), repair.get("repair_id")]
            }
            suggestions.append(suggestion)

        # 2.
        error_type = error_context.get("error_type", "")
        generic_suggestions = self._get_generic_suggestions(error_type)
        suggestions.extend(generic_suggestions)

        # 3.
        if current_problem_file:
            context_suggestion = self._generate_context_suggestion(current_problem_file, error_context)
            if context_suggestion:
                suggestions.append(context_suggestion)


        suggestions.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)

        return suggestions

    def _find_repair_for_failure(self, failure: ProblemFile) -> Optional[Dict[str, Any]]:
        """Find the linked retry action for a known failure."""
        task_id = failure.task_id
        if not task_id:
            return None

        fail_retry_pairs = self.observation_kg.get_fail_retry_pairs(task_id)

        for fail_pf, repair_pf in fail_retry_pairs:
            if fail_pf.action_id == failure.action_id:
                return {
                    "failure_id": fail_pf.action_id,
                    "repair_id": repair_pf.action_id,
                    "repair_action": repair_pf.intent_text,
                    "similarity": 1.0,
                    "evidence": f"Exact repair match for failure {fail_pf.action_id}"
                }

        return None

    def _extract_key_tokens(self, error_message: str) -> List[str]:
        """Extract stable tokens from an error message."""
        import re


        uppercase_tokens = re.findall(r'\b[A-Z][A-Z_]+\b', error_message)


        underscore_tokens = re.findall(r'\b[a-zA-Z]+_[a-zA-Z_]+\b', error_message)


        quoted_tokens = re.findall(r'[\'"]([^\'"]+)[\'"]', error_message)


        number_tokens = re.findall(r'\b\d{3,}\b', error_message)

        all_tokens = uppercase_tokens + underscore_tokens + quoted_tokens + number_tokens


        unique_tokens = list(set(all_tokens))
        return unique_tokens[:15]

    def _get_generic_suggestions(self, error_type: str) -> List[Dict[str, Any]]:
        """Return generic suggestions for common error categories."""
        generic_suggestions = {
            "syntax_error": [
                {
                    "type": "generic",
                    "confidence": 0.6,
                    "description": "The submitted code has a syntax error.",
                    "action": "Review syntax around the error line"
                },
                {
                    "type": "generic",
                    "confidence": 0.5,
                    "description": "A delimiter or quote may be unmatched.",
                    "action": "Check for matching brackets/quotes"
                }
            ],
            "import_error": [
                {
                    "type": "generic",
                    "confidence": 0.7,
                    "description": "The requested module could not be imported.",
                    "action": "Check if module is installed or path is correct"
                },
                {
                    "type": "generic",
                    "confidence": 0.6,
                    "description": "The runtime import path may be incomplete.",
                    "action": "Check PYTHONPATH environment variable"
                }
            ],
            "file_not_found": [
                {
                    "type": "generic",
                    "confidence": 0.8,
                    "description": "The referenced file does not exist.",
                    "action": "Check if file path exists, note case sensitivity"
                },
                {
                    "type": "generic",
                    "confidence": 0.6,
                    "description": "The working directory may be incorrect.",
                    "action": "Check current working directory"
                }
            ],
            "permission_error": [
                {
                    "type": "generic",
                    "confidence": 0.7,
                    "description": "The process lacks permission for this path.",
                    "action": "Check file/directory permissions"
                }
            ],
            "timeout": [
                {
                    "type": "generic",
                    "confidence": 0.6,
                    "description": "The operation exceeded its timeout.",
                    "action": "Increase timeout or optimize operation"
                }
            ]
        }

        return generic_suggestions.get(error_type, [])

    def _generate_context_suggestion(self,
                                    problem_file: ProblemFile,
                                    error_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a suggestion based on the failed action type."""
        action_type = problem_file.action_type

        suggestions_by_type = {
            ActionType.TOOL_CALL.value: {
                "type": "context",
                "confidence": 0.5,
                "description": "A tool invocation failed.",
                "action": "Check tool parameters and formatting"
            },
            ActionType.CODE_EDIT.value: {
                "type": "context",
                "confidence": 0.5,
                "description": "A code edit may have introduced the failure.",
                "action": "Check if code changes introduced the error"
            },
            ActionType.RUN_TEST.value: {
                "type": "context",
                "confidence": 0.5,
                "description": "The test command or environment may be incomplete.",
                "action": "Check test environment and dependencies"
            }
        }

        return suggestions_by_type.get(action_type.value)

    def _record_error(self,
                     problem_file: ProblemFile,
                     error_context: Dict[str, Any],
                     suggestions: List[Dict[str, Any]]) -> None:
        """Attach the failure signature to the current action record."""
        problem_file.outcome = Outcome.FAIL
        problem_file.failure_signature = {
            "error_type": error_context.get("error_type"),
            "key_tokens": self._extract_key_tokens(error_context.get("error_message", "")),
            "context": error_context.get("error_message", "")[:500],
            "suggestions_tried": [s.get("type") for s in suggestions]
        }


        if self.graph_store:

            pass

    def get_handler_stats(self) -> Dict[str, Any]:
        """Return component availability statistics."""
        return {
            "initialized": True,
            "retriever_available": self.retriever is not None,
            "graph_store_available": self.graph_store is not None,
            "kg_available": self.observation_kg is not None
        }


class FixSuggester:
    """Convert error-handler output into executable repair suggestions."""

    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler

    def suggest_fixes(self,
                     error_context: Dict[str, Any],
                     max_suggestions: int = 3) -> List[Dict[str, Any]]:
        """Return a bounded list of executable suggestions.

        Args:
            error_context: Structured error details
            max_suggestions: Maximum number of suggestions

        Returns:
            Suggestions ordered by descending confidence
        """
        result = self.error_handler.handle_error(error_context)


        suggestions = result.get("repair_suggestions", [])


        executable_suggestions = []
        for i, suggestion in enumerate(suggestions[:max_suggestions]):
            executable_suggestion = {
                "id": f"suggestion_{i+1}",
                "priority": i + 1,
                "description": suggestion.get("description", ""),
                "action": suggestion.get("action", ""),
                "confidence": suggestion.get("confidence", 0.0),
                "type": suggestion.get("type", "unknown"),
                "expected_outcome": f"Error should be resolved or reduced",
                "validation_steps": [
                    "Execute the suggested action",
                    "Check if error disappears or changes"
                ]
            }
            executable_suggestions.append(executable_suggestion)

        return executable_suggestions
