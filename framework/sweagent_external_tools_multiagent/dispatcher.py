from __future__ import annotations

import argparse
import os
import sys

# Try to import Agent-mem, fall back to old tools
try:
    from .agent_mem_main import main as agent_mem_main
    AGENT_MEM_AVAILABLE = True
except ImportError:
    try:
        from agent_mem_main import main as agent_mem_main
        AGENT_MEM_AVAILABLE = True
    except ImportError:
        AGENT_MEM_AVAILABLE = False


def _load_event_from_env():
    try:
        from .adapters.sweagent_payload import load_event_from_env
    except ImportError:
        from adapters.sweagent_payload import load_event_from_env
    return load_event_from_env(os.getenv("SWE_AGENT_EXT_EVENT_JSON"))


def _append_tool_log(tool_name: str, payload: dict) -> None:
    try:
        from .tools.io_utils import append_json_log
    except ImportError:
        from tools.io_utils import append_json_log
    append_json_log(tool_name, payload)


def _infer_tool_from_event(event: str) -> str:
    if event == "plan_generated":
        return "A"
    if event in {"action_error", "run_done"}:
        return "B"
    return "A"


def main() -> int:
    parser = argparse.ArgumentParser(description="Dispatch SWE-agent external tool events.")

    if AGENT_MEM_AVAILABLE:
        # Use Agent-mem enhanced parser
        parser.add_argument("--tool", choices=["A", "B"],
                          help="Tool type (A for planning, B for error handling). "
                               "If not specified, inferred from event.")
        parser.add_argument("--config", help="Path to Agent-mem configuration file")
        parser.add_argument("--stats", action="store_true",
                          help="Show statistics instead of processing event")
        parser.add_argument("--export", metavar="DIR",
                          help="Export data to directory")
        parser.add_argument("--task-id", help="Task ID for statistics or export")

        args = parser.parse_args()

        # Pass arguments to Agent-mem main
        sys.argv = [sys.argv[0]]  # Reset argv for agent_mem_main

        if args.config:
            sys.argv.extend(["--config", args.config])
        if args.tool:
            sys.argv.extend(["--tool", args.tool])
        if args.stats:
            sys.argv.append("--stats")
        if args.export:
            sys.argv.extend(["--export", args.export])
        if args.task_id:
            sys.argv.extend(["--task-id", args.task_id])

        # Keep event-level logs stable across with-mem / no-mem comparison runs.
        if not args.stats and not args.export:
            payload = _load_event_from_env()
            if payload.get("valid", False):
                tool_name = args.tool or _infer_tool_from_event(payload.get("event", ""))
                _append_tool_log(tool_name, payload)

        return agent_mem_main()
    else:
        # Fall back to old tools
        try:
            from .tools.tool_a import handle_event as handle_tool_a
            from .tools.tool_b import handle_event as handle_tool_b
        except ImportError:
            from tools.tool_a import handle_event as handle_tool_a
            from tools.tool_b import handle_event as handle_tool_b

        parser.add_argument("--tool", choices=["A", "B"], required=True)
        args = parser.parse_args()

        payload = _load_event_from_env()
        if args.tool == "A":
            handle_tool_a(payload)
        else:
            handle_tool_b(payload)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
