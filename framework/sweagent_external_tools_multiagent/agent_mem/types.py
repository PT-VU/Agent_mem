"""Shared Agent-mem enum types."""

from enum import Enum


class ActionType(str, Enum):
    """Recorded action categories."""

    TOOL_CALL = "tool_call"
    CODE_EDIT = "code_edit"
    RUN_TEST = "run_test"
    GUI_CLICK = "gui_click"
    GUI_TYPE = "gui_type"
    WEB_NAV = "web_nav"


class Outcome(str, Enum):
    """Recorded action outcomes."""

    SUCCESS = "success"
    FAIL = "fail"
    UNKNOWN = "unknown"


class EdgeType(str, Enum):
    """Observation-graph edge types."""

    SUCCESS_NEXT = "success_next"
    FAIL_RETRY = "fail_retry"
class GovernanceState(str, Enum):
    """Governance  """

    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    SUPPRESSED = "suppressed"
    DEPRECATED = "deprecated"


class EvidenceLevel(str, Enum):
    """ """

    LOCAL = "local"
    ATTEMPT = "attempt"
    OFFICIAL = "official"


class SubtaskRelationType(str, Enum):
    """Subtask candidate  """

    PRECEDES = "PRECEDES"
    RETRY_OF = "RETRY_OF"
    ALTERNATIVE_TO = "ALTERNATIVE_TO"


class SubtaskState(str, Enum):
    """SubtaskInstance  """

    PROJECTED_CANDIDATE = "projected_candidate"
    LOCALLY_SUPPORTED = "locally_supported"
    LOCALLY_FAILED = "locally_failed"
    EVAL_CONTEXT_ATTACHED = "eval_context_attached"
    DEPRECATED = "deprecated"


class SubtaskEdgeState(str, Enum):
    """SubtaskEdge  """

    CANDIDATE = "candidate"
    SUPPORTED_CANDIDATE = "supported_candidate"
    SUPPRESSED = "suppressed"
    DEPRECATED = "deprecated"


class CompilerCardType(str, Enum):
    """Compiler  """

    PLAN_HINT = "PlanHintCard"
    SUCCESS_PATH = "SuccessPathCard"
    BUG_INVARIANT = "BugInvariantCard"
    BUG_ANTI_PATTERN = "BugAntiPatternCard"
    SUBTASK_RISK = "SubtaskRiskCard"
    RETRY_HINT = "RetryHintCard"
    TIMEOUT_GOVERNANCE = "TimeoutGovernanceCard"
    CLOSURE_GUARD = "ClosureGuardCard"
