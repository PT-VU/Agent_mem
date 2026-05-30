"""Generate action records and evidence pointers during execution."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import uuid

from ..core.problem_file import (
    ProblemFile, ActionType, Outcome, EvidencePointer,
    FailureSignature, EnvSignature, MultiViewEmbeddings, EpistemicTag
)
from ..storage.graph_store import GraphStore


class EvidenceCollector:
    """Collects and manages evidence for actions."""

    def __init__(self, evidence_dir: Optional[str] = None):
        """
        Initialize evidence collector.

        Args:
            evidence_dir: Directory to store evidence files. If None, uses temp directory.
        """
        if evidence_dir:
            self.evidence_dir = Path(evidence_dir)
            self.evidence_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.evidence_dir = Path(tempfile.mkdtemp(prefix="agent_mem_evidence_"))

    def store_stdout(self, content: str, action_id: str) -> EvidencePointer:
        """Store stdout content and return evidence pointer."""
        return self._store_evidence("stdout", content, action_id)

    def store_stderr(self, content: str, action_id: str) -> EvidencePointer:
        """Store stderr content and return evidence pointer."""
        return self._store_evidence("stderr", content, action_id)

    def store_exception(self, exception: Exception, action_id: str) -> EvidencePointer:
        """Store exception and return evidence pointer."""
        content = f"{type(exception).__name__}: {str(exception)}\n"
        import traceback
        content += traceback.format_exc()
        return self._store_evidence("exception", content, action_id)

    def store_diff(self, diff_content: str, action_id: str) -> EvidencePointer:
        """Store diff content and return evidence pointer."""
        return self._store_evidence("diff", diff_content, action_id)

    def store_test_output(self, test_output: str, action_id: str) -> EvidencePointer:
        """Store test output and return evidence pointer."""
        return self._store_evidence("test_output", test_output, action_id)

    def _store_evidence(self,
                       evidence_type: str,
                       content: str,
                       action_id: str) -> EvidencePointer:
        """Store evidence content and return pointer."""
        # Generate filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{action_id}_{evidence_type}_{timestamp}.txt"
        filepath = self.evidence_dir / filename

        # Write content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        # Calculate hash
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

        return EvidencePointer(
            type=evidence_type,
            location=str(filepath),
            hash=content_hash,
            length=len(content)
        )

    def get_evidence_content(self, pointer: EvidencePointer) -> Optional[str]:
        """Retrieve evidence content from pointer."""
        try:
            with open(pointer.location, 'r', encoding='utf-8') as f:
                return f.read()
        except (FileNotFoundError, IOError):
            return None


class ActionLogger:
    """Record actions, evidence pointers, outcomes, and embeddings."""

    def __init__(self,
                 graph_store: GraphStore,
                 evidence_dir: Optional[str] = None):
        """
        Initialize ActionLogger.

        Args:
            graph_store: GraphStore instance for storing actions
            evidence_dir: Directory for evidence storage
        """
        self.graph_store = graph_store
        self.evidence_collector = EvidenceCollector(evidence_dir)
        self.current_task_id: Optional[str] = None
        self.last_action_id: Optional[str] = None

    def start_task(
        self,
        task_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        resume_from_existing: bool = True,
    ) -> None:
        """Start logging a new task."""
        self.current_task_id = task_id
        self.last_action_id = None
        if resume_from_existing:
            self.last_action_id = self.graph_store.get_last_action_id(task_id)

        # Create initial task metadata
        task_metadata = metadata or {}
        task_metadata["started_at"] = datetime.now(timezone.utc).isoformat()

        # TODO: Store task metadata in graph store

    def log_action(self,
                   action_type: ActionType,
                   intent_text: str,
                   inputs: Dict[str, Any],
                   tool_calls: List[Dict[str, Any]],
                   action_text: Optional[str] = None,
                   action_family: Optional[str] = None,
                   instance_id: Optional[str] = None,
                   run_id: Optional[str] = None,
                   agent_name: Optional[str] = None,
                   source_event: Optional[str] = None,
                   step_index: Optional[int] = None,
                   trace_id: Optional[str] = None,
                   stdout: Optional[str] = None,
                   stderr: Optional[str] = None,
                   exception: Optional[Exception] = None,
                   diff_content: Optional[str] = None,
                   test_output: Optional[str] = None,
                   touched_files: Optional[List[str]] = None,
                   outcome: Outcome = Outcome.UNKNOWN,
                   env_info: Optional[Dict[str, Any]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> ProblemFile:
        """
        Log an action and create problem file.

        Args:
            action_type: Type of action
            intent_text: Executor's intent/purpose
            inputs: Action inputs/parameters
            tool_calls: Tool calls made
            stdout: Standard output (if any)
            stderr: Standard error (if any)
            exception: Exception (if any)
            diff_content: Diff content (for code edits)
            test_output: Test output (if any)
            touched_files: Files touched by the action
            outcome: Outcome of the action
            env_info: Environment information
            metadata: Additional metadata

        Returns:
            Created ProblemFile
        """
        if not self.current_task_id:
            raise ValueError("No active task. Call start_task() first.")

        # Create ProblemFile
        resolved_action_text = action_text or str(inputs.get("action", "") or "")
        resolved_action_family = action_family or self._infer_action_family(
            action_type=action_type,
            action_text=resolved_action_text,
        )
        patch_stats = self._build_patch_stats(diff_content)
        test_stats = self._build_test_stats(test_output)
        execution_stats = self._build_execution_stats(metadata)

        problem_file = ProblemFile(
            task_id=self.current_task_id,
            action_type=action_type,
            intent_text=intent_text,
            action_text=resolved_action_text,
            action_family=resolved_action_family,
            instance_id=instance_id,
            run_id=run_id,
            agent_name=agent_name,
            source_event=source_event,
            step_index=step_index,
            trace_id=trace_id,
            inputs=inputs,
            tool_calls=tool_calls,
            outcome=outcome,
            metadata=metadata or {},
            patch_stats=patch_stats,
            test_stats=test_stats,
            execution_stats=execution_stats,
        )

        # Store evidence
        evidence_pointers = []

        if stdout:
            ptr = self.evidence_collector.store_stdout(stdout, problem_file.action_id)
            problem_file.stdout_ref = ptr
            evidence_pointers.append(ptr)

        if stderr:
            ptr = self.evidence_collector.store_stderr(stderr, problem_file.action_id)
            problem_file.stderr_ref = ptr
            evidence_pointers.append(ptr)

        if exception:
            ptr = self.evidence_collector.store_exception(exception, problem_file.action_id)
            problem_file.exception_stack_ref = ptr
            evidence_pointers.append(ptr)

        if diff_content:
            ptr = self.evidence_collector.store_diff(diff_content, problem_file.action_id)
            problem_file.diff_summary_ref = ptr
            evidence_pointers.append(ptr)

        if test_output:
            ptr = self.evidence_collector.store_test_output(test_output, problem_file.action_id)
            problem_file.tests_ref = ptr
            evidence_pointers.append(ptr)

        # Add touched files
        if touched_files:
            problem_file.touched_files = touched_files

        # Create failure signature if failed
        if outcome == Outcome.FAIL and (stderr or exception):
            problem_file.failure_signature = self._create_failure_signature(stderr, exception)

        # Create environment signature
        if env_info:
            problem_file.env_signature = self._create_env_signature(env_info)

        # Add all evidence pointers to index
        for ptr in evidence_pointers:
            problem_file.add_evidence_pointer(ptr)

        # Set epistemic tags
        self._set_epistemic_tags(problem_file)

        # Store in graph
        self.graph_store.add_action(problem_file)

        # Create edge from previous action if exists
        if self.last_action_id:
            from ..core.observation_kg import EdgeType
            edge_type = EdgeType.SUCCESS_NEXT if outcome == Outcome.SUCCESS else EdgeType.FAIL_RETRY
            self.graph_store.add_edge(
                self.last_action_id,
                problem_file.action_id,
                edge_type
            )

        # Update last action
        self.last_action_id = problem_file.action_id

        return problem_file

    def _create_failure_signature(self,
                                 stderr: Optional[str],
                                 exception: Optional[Exception]) -> FailureSignature:
        """Create failure signature from stderr/exception."""
        error_type = "unknown"
        error_tokens = []

        if exception:
            error_type = type(exception).__name__
            error_tokens = [error_type.lower(), str(exception).lower()[:50]]
        elif stderr:
            # Extract error type from stderr (simplified)
            lines = stderr.strip().split('\n')
            if lines:
                first_line = lines[0].lower()
                error_tokens = first_line.split()[:5]

                # Common error patterns
                if "error" in first_line:
                    error_type = "error"
                elif "exception" in first_line:
                    error_type = "exception"
                elif "fail" in first_line:
                    error_type = "failure"

        return FailureSignature(
            error_type=error_type,
            error_tokens=error_tokens
        )

    def _create_env_signature(self, env_info: Dict[str, Any]) -> EnvSignature:
        """Create environment signature from env info."""
        # Extract key environment variables
        key_env_vars = {}
        for var in ["PATH", "PYTHONPATH", "HOME", "LANG", "USER"]:
            if var in env_info:
                key_env_vars[var] = env_info[var]

        # Create path hash
        path_hash = None
        if "PATH" in env_info:
            path_str = env_info["PATH"]
            path_hash = hashlib.sha256(path_str.encode('utf-8')).hexdigest()[:16]

        return EnvSignature(
            key_env_vars=key_env_vars,
            path_hash=path_hash,
            working_dir=env_info.get("PWD")
        )

    def _set_epistemic_tags(self, problem_file: ProblemFile) -> None:
        """Set epistemic tags for evidence vs inference fields."""
        # Evidence fields (direct observations)
        evidence_fields = [
            "stdout_ref", "stderr_ref", "exception_stack_ref",
            "diff_summary_ref", "tests_ref", "ui_state_ref",
            "touched_files"
        ]

        for field in evidence_fields:
            if getattr(problem_file, field, None):
                problem_file.set_epistemic_tag(field, EpistemicTag.EVIDENCE)

        # Inference fields (derived/interpreted)
        inference_fields = ["failure_signature", "intent_text"]
        for field in inference_fields:
            if getattr(problem_file, field, None):
                problem_file.set_epistemic_tag(field, EpistemicTag.INFERENCE)

    def _infer_action_family(self, *, action_type: ActionType, action_text: str) -> str:
        """Infer normalized action family for retrieval/attribution."""
        text = (action_text or "").strip().lower()
        if "str_replace_editor" in text:
            return "str_replace_editor"
        if any(tok in text for tok in ("pytest", "unittest", "tox", "nose", "make test")):
            return "test_command"
        if any(tok in text for tok in ("cat ", "ls ", "grep ", "find ")):
            return "read_command"
        if any(tok in text for tok in ("python ", "bash ", "sh ")):
            return "exec_command"
        return action_type.value

    def _build_patch_stats(self, diff_content: Optional[str]) -> Dict[str, int]:
        """Build patch summary stats from diff content."""
        if not diff_content:
            return {"files_changed": 0, "lines_added": 0, "lines_deleted": 0}

        files_changed = len(re.findall(r"^diff --git ", diff_content, flags=re.MULTILINE))
        lines_added = 0
        lines_deleted = 0
        for line in diff_content.splitlines():
            if line.startswith("+++ ") or line.startswith("--- "):
                continue
            if line.startswith("+"):
                lines_added += 1
            elif line.startswith("-"):
                lines_deleted += 1
        return {
            "files_changed": files_changed,
            "lines_added": lines_added,
            "lines_deleted": lines_deleted,
        }

    def _build_test_stats(self, test_output: Optional[str]) -> Dict[str, int]:
        """Build coarse test stats from textual test output."""
        if not test_output:
            return {
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "fail_to_pass_failed": 0,
                "pass_to_pass_failed": 0,
            }

        low = test_output.lower()
        tests_failed = 0
        tests_passed = 0

        for pattern in (r"(\d+)\s+failed", r"failures:\s*(\d+)"):
            m = re.search(pattern, low)
            if m:
                tests_failed = max(tests_failed, int(m.group(1)))
        for pattern in (r"(\d+)\s+passed", r"passed:\s*(\d+)"):
            m = re.search(pattern, low)
            if m:
                tests_passed = max(tests_passed, int(m.group(1)))

        tests_run = tests_failed + tests_passed
        return {
            "tests_run": tests_run,
            "tests_passed": tests_passed,
            "tests_failed": tests_failed,
            "fail_to_pass_failed": 0,
            "pass_to_pass_failed": 0,
        }

    def _build_execution_stats(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, int]:
        """Build execution stats using optional metadata hints."""
        metadata = metadata or {}

        def _to_int(name: str) -> int:
            try:
                return int(metadata.get(name, 0) or 0)
            except Exception:
                return 0

        return {
            "duration_ms": _to_int("duration_ms"),
            "tool_latency_ms": _to_int("tool_latency_ms"),
            "retry_count": _to_int("retry_count"),
        }

    def end_task(self, success: bool, summary: Optional[str] = None) -> Dict[str, Any]:
        """End current task and return task summary."""
        if not self.current_task_id:
            raise ValueError("No active task")

        task_summary = {
            "task_id": self.current_task_id,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "success": success,
            "total_actions": 0,  # TODO: Count actions
            "summary": summary or ""
        }

        # Reset state
        self.current_task_id = None
        self.last_action_id = None

        return task_summary

    def get_evidence_content(self, problem_file: ProblemFile) -> Dict[str, Optional[str]]:
        """Retrieve all evidence content for a problem file."""
        evidence = {}

        # Check all evidence pointers
        pointers = [
            ("stdout", problem_file.stdout_ref),
            ("stderr", problem_file.stderr_ref),
            ("exception", problem_file.exception_stack_ref),
            ("diff", problem_file.diff_summary_ref),
            ("tests", problem_file.tests_ref),
            ("ui_state", problem_file.ui_state_ref),
        ]

        for name, pointer in pointers:
            if pointer:
                evidence[name] = self.evidence_collector.get_evidence_content(pointer)
            else:
                evidence[name] = None

        # Check evidence index
        for i, pointer in enumerate(problem_file.evidence_index):
            evidence[f"index_{i}"] = self.evidence_collector.get_evidence_content(pointer)

        return evidence
