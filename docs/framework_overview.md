# Framework Overview

Agent-mem adds cross-attempt memory to SWE-agent through a non-invasive hook.
The hook records execution events, retrieves promoted cards before later model
calls, injects bounded hints, and feeds official evaluation outcomes back into
the card lifecycle.

## Core Data Flow

1. `bridge_hook.py` intercepts model-query, action, error, and run-completion
   events.
2. Tool A retrieves relevant cards from `GraphStore`; Tool B records execution
   evidence.
3. Retrieved cards are inserted into the next model query under
   `[AgentMem Hints]`.
4. Official SWE-bench evaluation produces `resolved` or `unresolved` feedback.
5. `evaluation_feedback.py` writes or updates structured memory cards for later
   attempts.

## Agent-mem v2 Modules

| Module | Purpose | Primary implementation |
| --- | --- | --- |
| A | Store verbatim diffs, anchors, and signature hashes for successful repairs | `agent_mem/processing/evaluation_feedback.py` |
| B | Write failure-side `BugAntiPatternCard` records | `agent_mem/processing/evaluation_feedback.py` |
| C | Check patches against known failure signatures before submission | `bridge_hook.py` |
| D | Select verbatim reuse, exploration around an invariant, or fresh exploration | `bridge_hook.py` |
| E | Adjust card confidence from local intervention-window evidence | `bridge_hook.py`, `agent_mem/storage/graph_store.py` |

## Multi-Agent Extension

The multi-agent copy extends v2 without replacing the SWE-agent executor:

| Module | Purpose | Primary implementation |
| --- | --- | --- |
| T1-A | Reformulate retrieved hints for the current execution phase | `agent_mem/processing/reformulation_agent.py` |
| T1-B | Persist interim localization progress for later attempts | `agent_mem/storage/interim_cache.py` |
| T1-C | Review a candidate diff before submission | `agent_mem/processing/critic_agent.py` |

The multi-agent study is auxiliary. T1-A was active in the main aligned run.
T1-C initially produced no verdicts because ordinary submit actions lacked an
inline diff. Later stages improved observability but did not establish reliable
semantic correction.

