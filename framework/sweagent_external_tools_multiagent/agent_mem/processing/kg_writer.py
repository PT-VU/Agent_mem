"""Write action records, graph edges, and embeddings to the knowledge graph."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from ..core.problem_file import ProblemFile, MultiViewEmbeddings
from ..core.observation_kg import EdgeType
from ..storage.graph_store import GraphStore


class EmbeddingGenerator:
    """Generates multi-view embeddings for problem files."""

    def __init__(
        self,
        embedding_dim: int = 384,
        *,
        model: str = "sentence-transformers",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize embedding generator.

        Args:
            embedding_dim: Dimension of embedding vectors
            model: Embedding backend (must be "sentence-transformers")
            model_name: Sentence-transformers model name
        """
        self.embedding_dim = embedding_dim
        self.model = model
        self.model_name = model_name
        self._encoder = None
        self._provider = "sentence-transformers"

        if model != "sentence-transformers":
            raise ValueError(
                f"Unsupported embedding model '{model}'. "
                "Only 'sentence-transformers' is allowed."
            )

        # Lazy-init the encoder at first encode call. This avoids paying model
        # load cost for code paths that do not require embeddings.

    def generate_task_semantic_embedding(self,
                                        intent_text: str,
                                        task_context: Optional[str] = None) -> List[float]:
        """Generate task semantic embedding."""
        text = intent_text
        if task_context:
            text = f"{intent_text}\n{task_context}"
        return self._embed_text(text)

    def generate_file_scope_embedding(self,
                                     touched_files: List[str],
                                     file_content_patterns: Optional[List[str]] = None) -> List[float]:
        """Generate file scope embedding."""
        file_text = " ".join(touched_files or [])
        pattern_text = " ".join(file_content_patterns or [])
        return self._embed_text(f"{file_text}\n{pattern_text}".strip())

    def generate_error_signature_embedding(self,
                                          error_type: str,
                                          error_tokens: List[str]) -> List[float]:
        """Generate error signature embedding."""
        return self._embed_text(f"{error_type} {' '.join(error_tokens or [])}".strip())

    def generate_tool_output_embedding(self,
                                      tool_output: str,
                                      tool_type: str) -> List[float]:
        """Generate tool output embedding."""
        return self._embed_text(f"{tool_type}\n{tool_output}".strip())

    def generate_diff_summary_embedding(self,
                                       diff_content: str) -> List[float]:
        """Generate diff summary embedding."""
        return self._embed_text(diff_content)

    def generate_intent_embedding(self, intent_text: str) -> List[float]:
        """Generate intent embedding."""
        return self._embed_text(intent_text)

    @property
    def provider(self) -> str:
        return self._provider

    def _embed_text(self, text: str) -> List[float]:
        if self._encoder is None:
            self._ensure_encoder()
        try:
            vec = self._encoder.encode(text or "", convert_to_numpy=True)
            return vec.astype(float).tolist()
        except Exception as e:
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    def _ensure_encoder(self) -> None:
        if self._encoder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize sentence-transformers model '{self.model_name}': {e}"
            ) from e


class KGWriter:
    """Write action nodes and their embeddings to the persistent graph."""

    def __init__(self,
                 graph_store: GraphStore,
                 embedding_generator: Optional[EmbeddingGenerator] = None):
        """
        Initialize KGWriter.

        Args:
            graph_store: GraphStore instance
            embedding_generator: EmbeddingGenerator instance (optional)
        """
        self.graph_store = graph_store
        self.embedding_generator = embedding_generator or EmbeddingGenerator()

    def write_action_with_embeddings(self, problem_file: ProblemFile) -> ProblemFile:
        """
        Write action to KG with generated embeddings.

        Args:
            problem_file: ProblemFile to write

        Returns:
            Updated ProblemFile with embeddings
        """
        # Generate embeddings
        embeddings = self._generate_embeddings(problem_file)
        problem_file.embeddings = embeddings

        # The action is already added to graph store by ActionLogger
        # This method ensures embeddings are generated and attached

        return problem_file

    def _generate_embeddings(self, problem_file: ProblemFile) -> MultiViewEmbeddings:
        """Generate multi-view embeddings for problem file."""
        embeddings = MultiViewEmbeddings()

        # Task semantic embedding
        if problem_file.intent_text:
            embeddings.emb_task_sem = self.embedding_generator.generate_task_semantic_embedding(
                problem_file.intent_text
            )

        # File scope embedding
        if problem_file.touched_files:
            embeddings.emb_file_scope = self.embedding_generator.generate_file_scope_embedding(
                problem_file.touched_files
            )

        # Error signature embedding
        if problem_file.failure_signature:
            embeddings.emb_error_sig = self.embedding_generator.generate_error_signature_embedding(
                problem_file.failure_signature.error_type,
                problem_file.failure_signature.error_tokens
            )

        # Tool output embedding (from stdout/stderr/tests evidence)
        tool_output_summary = self._build_tool_output_summary(problem_file)
        if tool_output_summary:
            embeddings.emb_tool_output = self.embedding_generator.generate_tool_output_embedding(
                tool_output_summary, problem_file.action_family or problem_file.action_type.value
            )

        # Diff summary embedding
        if problem_file.diff_summary_ref:
            diff_text = self._read_evidence_text(problem_file.diff_summary_ref.location)
            if diff_text:
                embeddings.emb_diff_summary = self.embedding_generator.generate_diff_summary_embedding(
                    self._truncate_text(diff_text, max_chars=8000)
                )
            else:
                problem_file.metadata["emb_diff_summary_error"] = "diff_ref_unreadable"

        # Intent embedding
        if problem_file.intent_text:
            embeddings.emb_intent = self.embedding_generator.generate_intent_embedding(
                problem_file.intent_text
            )

        return embeddings

    def _read_evidence_text(self, location: str) -> str:
        """Read evidence file content by location with safe fallback."""
        if not location:
            return ""
        try:
            path = Path(location)
            if not path.exists() or not path.is_file():
                return ""
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

    def _truncate_text(self, text: str, *, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 64] + "\n...[truncated]..."

    def _build_tool_output_summary(self, problem_file: ProblemFile) -> str:
        """Build tool output summary from evidence refs."""
        parts: List[str] = []
        stdout_text = self._read_evidence_text(problem_file.stdout_ref.location) if problem_file.stdout_ref else ""
        stderr_text = self._read_evidence_text(problem_file.stderr_ref.location) if problem_file.stderr_ref else ""
        tests_text = self._read_evidence_text(problem_file.tests_ref.location) if problem_file.tests_ref else ""

        if stdout_text:
            parts.append("stdout:\n" + self._truncate_text(stdout_text, max_chars=3000))
        if stderr_text:
            parts.append("stderr:\n" + self._truncate_text(stderr_text, max_chars=3000))
        if tests_text:
            parts.append("tests:\n" + self._truncate_text(tests_text, max_chars=2000))

        if not parts:
            if problem_file.stdout_ref or problem_file.stderr_ref or problem_file.tests_ref:
                problem_file.metadata["emb_tool_output_error"] = "tool_output_ref_unreadable"
            return ""

        return "\n\n".join(parts)

    def update_edge_metadata(self,
                            source_id: str,
                            target_id: str,
                            edge_type: EdgeType,
                            metadata_updates: Dict[str, Any]) -> bool:
        """
        Update metadata for an existing edge.

        Args:
            source_id: Source action ID
            target_id: Target action ID
            edge_type: Edge type
            metadata_updates: Metadata updates to apply

        Returns:
            True if successful, False if edge not found
        """
        # TODO: Implement edge metadata update
        # This would require modifying the GraphStore to support edge updates
        # For MVP, we'll just log the update
        print(f"Edge metadata update requested: {source_id} -> {target_id} ({edge_type})")
        print(f"Updates: {metadata_updates}")
        return True

    def batch_write_actions(self, problem_files: List[ProblemFile]) -> List[ProblemFile]:
        """
        Batch write multiple actions with embeddings.

        Args:
            problem_files: List of ProblemFiles to write

        Returns:
            List of updated ProblemFiles with embeddings
        """
        updated_files = []
        for pf in problem_files:
            updated = self.write_action_with_embeddings(pf)
            updated_files.append(updated)

        return updated_files

    def validate_action_integrity(self, problem_file: ProblemFile) -> List[str]:
        """
        Validate integrity of action data in KG.

        Args:
            problem_file: ProblemFile to validate

        Returns:
            List of validation errors/warnings
        """
        issues = []

        # Check if action exists in KG
        action_info = self.graph_store.observation_kg.get_action(problem_file.action_id)
        if not action_info:
            issues.append(f"Action {problem_file.action_id} not found in KG")
            return issues

        # Check embeddings
        if not problem_file.embeddings:
            issues.append("No embeddings generated for action")

        # Check evidence pointers
        for ptr in problem_file.evidence_index:
            if not ptr.location:
                issues.append(f"Evidence pointer missing location: {ptr.type}")

        # Validate problem file
        pf_issues = problem_file.validate()
        issues.extend(pf_issues)

        return issues

    def get_action_embeddings_summary(self, action_id: str) -> Optional[Dict[str, Any]]:
        """
        Get embeddings summary for an action.

        Args:
            action_id: Action ID

        Returns:
            Embeddings summary or None if not found
        """
        action_info = self.graph_store.observation_kg.get_action(action_id)
        if not action_info:
            return None

        problem_file, _ = action_info
        embeddings = problem_file.embeddings

        summary = {
            "action_id": action_id,
            "has_task_sem": embeddings.emb_task_sem is not None,
            "has_file_scope": embeddings.emb_file_scope is not None,
            "has_error_sig": embeddings.emb_error_sig is not None,
            "has_tool_output": embeddings.emb_tool_output is not None,
            "has_diff_summary": embeddings.emb_diff_summary is not None,
            "has_intent": embeddings.emb_intent is not None,
            "has_ui_state": embeddings.emb_ui_state is not None,
        }

        # Add vector dimensions if available
        for key, vector in [
            ("task_sem", embeddings.emb_task_sem),
            ("file_scope", embeddings.emb_file_scope),
            ("error_sig", embeddings.emb_error_sig),
            ("tool_output", embeddings.emb_tool_output),
            ("diff_summary", embeddings.emb_diff_summary),
            ("intent", embeddings.emb_intent),
            ("ui_state", embeddings.emb_ui_state),
        ]:
            if vector is not None:
                summary[f"{key}_dim"] = len(vector)

        return summary

    def export_embeddings_for_analysis(self,
                                      output_file: str,
                                      embedding_view: str = "emb_task_sem") -> Dict[str, Any]:
        """
        Export embeddings for external analysis.

        Args:
            output_file: Output file path
            embedding_view: Which embedding view to export

        Returns:
            Export statistics
        """
        embeddings_data = []
        missing_count = 0

        for task_id, subgraph in self.graph_store.observation_kg.task_subgraphs.items():
            for action_id, problem_file in subgraph.action_nodes.items():
                # Get the requested embedding view
                embedding_vector = getattr(problem_file.embeddings, embedding_view, None)

                if embedding_vector is not None:
                    embeddings_data.append({
                        "action_id": action_id,
                        "task_id": task_id,
                        "action_type": problem_file.action_type.value,
                        "outcome": problem_file.outcome.value,
                        "intent_text": problem_file.intent_text,
                        "embedding": embedding_vector,
                    })
                else:
                    missing_count += 1

        # Save to file
        export_data = {
            "embedding_view": embedding_view,
            "total_actions": len(embeddings_data) + missing_count,
            "exported_actions": len(embeddings_data),
            "missing_embeddings": missing_count,
            "embeddings": embeddings_data,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)

        return {
            "exported_file": output_file,
            "total_actions": export_data["total_actions"],
            "exported_actions": export_data["exported_actions"],
            "missing_embeddings": export_data["missing_embeddings"],
        }
