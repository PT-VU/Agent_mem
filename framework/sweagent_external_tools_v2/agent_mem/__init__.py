"""
Agent-mem: Work-Experience Memory module for SWE-agent.

Keep package import best-effort so lightweight helpers can be reused without
forcing heavyweight optional dependencies at import time.
"""

from .config.config_manager import ConfigManager
from .core.problem_file import ProblemFile

__version__ = "0.1.0"

try:
    from .core.observation_kg import ObservationKG, EdgeType
except Exception:  # pragma: no cover - optional dependency path
    ObservationKG = None
    EdgeType = None

try:
    from .core.belief_graph import BeliefGraph
except Exception:  # pragma: no cover - optional dependency path
    BeliefGraph = None

try:
    from .storage.graph_store import GraphStore
except Exception:  # pragma: no cover - optional dependency path
    GraphStore = None

try:
    from .processing.action_logger import ActionLogger
    from .processing.kg_writer import KGWriter
    from .retrieval.memory_agent import MemoryAgent
    from .integration.sweagent_adapter import SWEAgentAdapter
except Exception:  # pragma: no cover - optional dependency path
    ActionLogger = None
    KGWriter = None
    MemoryAgent = None
    SWEAgentAdapter = None

__all__ = [
    "ProblemFile",
    "ObservationKG",
    "EdgeType",
    "BeliefGraph",
    "GraphStore",
    "ActionLogger",
    "KGWriter",
    "MemoryAgent",
    "SWEAgentAdapter",
    "ConfigManager",
]
