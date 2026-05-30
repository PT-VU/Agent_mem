"""Generate and cache multi-view sentence-transformer embeddings."""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Using dummy embeddings.")


class EmbeddingType(Enum):
    """Supported embedding views."""
    TASK_SEM = "task_sem"
    FILE_SCOPE = "file_scope"
    ERROR_SIG = "error_sig"
    TOOL_OUTPUT = "tool_output"
    DIFF_SUMMARY = "diff_summary"  # diff
    INTENT = "intent"
class MultiViewEmbedder:
    """Generate embedding views for action-level problem files."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_size: int = 1000):
        """Initialize the embedder.

        Args:
            model_name: Sentence-transformers model name
            cache_size: Maximum number of cached embeddings
        """
        self.model_name = model_name
        self.cache_size = cache_size
        self.embedding_cache: Dict[str, List[float]] = {}
        self.model = None

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                print(f"Loaded sentence-transformers model: {model_name}")
            except Exception as e:
                print(f"Failed to load model {model_name}: {e}")
                self.model = None
        else:
            self.model = None

    def embed_text(self, text: str) -> List[float]:
        """Embed text with caching and a deterministic fallback.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if not text:
            return self._dummy_embedding()


        cache_key = self._get_cache_key(text)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        if self.model:
            try:
                embedding = self.model.encode(text, convert_to_numpy=True).tolist()
            except Exception as e:
                print(f"Embedding generation failed: {e}")
                embedding = self._dummy_embedding()
        else:
            embedding = self._dummy_embedding()


        self._update_cache(cache_key, embedding)

        return embedding

    def generate_problem_file_embeddings(self, problem_file_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """Generate all available embedding views for one action.

        Args:
            problem_file_data: Serialized problem-file payload

        Returns:
            Mapping from embedding view name to vector
        """
        embeddings = {}

        # 1. Task-semantic embedding from intent and tool calls.
        task_sem_text = self._extract_task_semantic_text(problem_file_data)
        embeddings[EmbeddingType.TASK_SEM.value] = self.embed_text(task_sem_text)

        # 2. File-scope embedding from touched files.
        file_scope_text = self._extract_file_scope_text(problem_file_data)
        embeddings[EmbeddingType.FILE_SCOPE.value] = self.embed_text(file_scope_text)

        # 3. Error-signature embedding when an error is available.
        error_sig_text = self._extract_error_signature_text(problem_file_data)
        if error_sig_text:
            embeddings[EmbeddingType.ERROR_SIG.value] = self.embed_text(error_sig_text)

        # 4. Tool-output embedding from stdout and stderr.
        tool_output_text = self._extract_tool_output_text(problem_file_data)
        if tool_output_text:
            embeddings[EmbeddingType.TOOL_OUTPUT.value] = self.embed_text(tool_output_text)

        # 5. diff embedding ( diff)
        diff_summary_text = self._extract_diff_summary_text(problem_file_data)
        if diff_summary_text:
            embeddings[EmbeddingType.DIFF_SUMMARY.value] = self.embed_text(diff_summary_text)

        # 6. Intent embedding from the raw intent text.
        intent_text = problem_file_data.get("intent_text", "")
        if intent_text:
            embeddings[EmbeddingType.INTENT.value] = self.embed_text(intent_text)

        return embeddings

    def _extract_task_semantic_text(self, problem_file_data: Dict[str, Any]) -> str:
        """ """
        intent_text = problem_file_data.get("intent_text", "")
        tool_calls = problem_file_data.get("tool_calls", [])


        tool_info = []
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                tool_name = tool_call.get("tool", "")
                args = tool_call.get("args", {})
                tool_info.append(f"{tool_name}: {args}")

        return f"Intent: {intent_text}. Tools: {'; '.join(tool_info)}"

    def _extract_file_scope_text(self, problem_file_data: Dict[str, Any]) -> str:
        """ """
        touched_files = problem_file_data.get("touched_files", [])
        if not touched_files:
            return ""


        extensions = {}
        for file_path in touched_files:
            if '.' in file_path:
                ext = file_path.split('.')[-1]
                extensions[ext] = extensions.get(ext, 0) + 1

        ext_summary = ", ".join([f"{ext}:{count}" for ext, count in extensions.items()])
        return f"Files: {', '.join(touched_files[:5])}. Extensions: {ext_summary}"

    def _extract_error_signature_text(self, problem_file_data: Dict[str, Any]) -> str:
        """ """
        failure_signature = problem_file_data.get("failure_signature")
        if not failure_signature:
            return ""

        error_type = failure_signature.get("error_type", "")
        key_tokens = failure_signature.get("key_tokens", [])
        error_context = failure_signature.get("context", "")

        tokens_text = ", ".join(key_tokens[:10])
        return f"Error type: {error_type}. Tokens: {tokens_text}. Context: {error_context[:200]}"

    def _extract_tool_output_text(self, problem_file_data: Dict[str, Any]) -> str:
        """ """
        # MVP
        outcome = problem_file_data.get("outcome", "")
        return f"Outcome: {outcome}"

    def _extract_diff_summary_text(self, problem_file_data: Dict[str, Any]) -> str:
        """ diff """
        # MVP
        return ""

    def _get_cache_key(self, text: str) -> str:
        """ """
        return hashlib.md5(text.encode()).hexdigest()

    def _update_cache(self, key: str, embedding: List[float]) -> None:
        """ """
        if len(self.embedding_cache) >= self.cache_size:
            first_key = next(iter(self.embedding_cache))
            del self.embedding_cache[first_key]

        self.embedding_cache[key] = embedding

    def _dummy_embedding(self, dimension: int = 384) -> List[float]:
        """ embedding """
        return [0.0] * dimension

    def get_stats(self) -> Dict[str, Any]:
        """ """
        return {
            "model_name": self.model_name,
            "cache_size": len(self.embedding_cache),
            "max_cache_size": self.cache_size,
            "model_available": self.model is not None
        }


class EmbeddingManager:
    """
    Embedding

     embedding
    """

    def __init__(self, storage_backend=None, embedder: MultiViewEmbedder = None):
        """
         Embedding

        Args:
            storage_backend:
            embedder: Embedder
        """
        self.storage_backend = storage_backend
        self.embedder = embedder or MultiViewEmbedder()

    def generate_and_store_embeddings(self, problem_file_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """
         embedding

        Args:
            problem_file_data: ProblemFile

        Returns:
             embedding
        """
        embeddings = self.embedder.generate_problem_file_embeddings(problem_file_data)

        if self.storage_backend:
            problem_file_id = problem_file_data.get("action_id")
            if problem_file_id:
                for emb_type, emb_vector in embeddings.items():
                    self.storage_backend.save_embedding(
                        problem_file_id,
                        emb_type,
                        emb_vector
                    )

        return embeddings

    def search_similar(self,
                      query_text: str,
                      embedding_type: str,
                      limit: int = 10) -> List[Dict[str, Any]]:
        """
         embedding

        Args:
            query_text:
            embedding_type: embedding
            limit:

        Returns:

        """
        query_embedding = self.embedder.embed_text(query_text)


        if self.storage_backend:
            return self.storage_backend.search_similar(
                query_embedding,
                embedding_type,
                limit
            )
        else:

            return []

    def get_embeddings_for_problem_file(self, problem_file_id: str) -> Dict[str, List[float]]:
        """
         ProblemFile embedding

        Args:
            problem_file_id: ProblemFile ID

        Returns:
            embedding
        """
        if not self.storage_backend:
            return {}

        embeddings = {}
        for emb_type in EmbeddingType:
            emb_vector = self.storage_backend.load_embedding(
                problem_file_id,
                emb_type.value
            )
            if emb_vector:
                embeddings[emb_type.value] = emb_vector

        return embeddings
