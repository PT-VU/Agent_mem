"""
Main entry point for Agent-mem integration with SWE-agent external tools.

This replaces the simple Tool A/Tool B with full Agent-mem capabilities.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from .agent_mem.integration.sweagent_adapter import SWEAgentAdapter
    from .agent_mem.config.config_manager import ConfigManager
except ImportError:
    # Fallback for direct execution from this directory.
    from agent_mem.integration.sweagent_adapter import SWEAgentAdapter
    from agent_mem.config.config_manager import ConfigManager


def load_event_from_env() -> Dict[str, Any]:
    """Load event data from environment variable."""
    raw = os.getenv("SWE_AGENT_EXT_EVENT_JSON")
    if not raw:
        return {"event": "unknown", "raw": "", "valid": False}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"event": "unknown", "raw": raw, "valid": False}

    if not isinstance(parsed, dict):
        return {"event": "unknown", "raw": raw, "valid": False}

    parsed.setdefault("valid", True)
    parsed.setdefault("version", "v1")
    return parsed


def setup_agent_mem(config_path: Optional[str] = None) -> SWEAgentAdapter:
    """Set up Agent-mem system with configuration."""
    # Load configuration
    config_manager = ConfigManager(config_path)
    config_manager.update_from_env()

    # Validate configuration
    validation = config_manager.validate()
    if not validation["valid"]:
        print(f"Configuration validation issues: {validation['issues']}")
        # Continue anyway for MVP

    # Get storage directories from config
    storage_dir = config_manager.get("storage.graph_store_dir")
    evidence_dir = config_manager.get("storage.evidence_dir")
    embedding_model = config_manager.get("embeddings.model", "sentence-transformers")
    embedding_model_name = config_manager.get("embeddings.model_name", "all-MiniLM-L6-v2")
    embedding_dimension = int(config_manager.get("embeddings.dimension", 384))

    # Create adapter
    adapter = SWEAgentAdapter(
        storage_dir=storage_dir,
        evidence_dir=evidence_dir,
        embedding_model=embedding_model,
        embedding_model_name=embedding_model_name,
        embedding_dimension=embedding_dimension,
        v21_config=config_manager.get("agent_mem.v21", {}),
    )

    return adapter, config_manager


def handle_event(adapter: SWEAgentAdapter, event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle event using Agent-mem adapter."""
    event_type = event_data.get("event")

    if event_type == "plan_generated":
        return adapter.handle_plan_generated(event_data)
    elif event_type == "action_error":
        return adapter.handle_action_error(event_data)
    elif event_type == "run_done":
        return adapter.handle_run_done(event_data)
    elif event_type == "official_eval_feedback":
        return adapter.apply_evaluation_feedback(event_data)
    else:
        return {
            "error": f"Unsupported event type: {event_type}",
            "supported_events": ["plan_generated", "action_error", "run_done", "official_eval_feedback"],
            "event_data": event_data,
        }


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Agent-mem: Work-Experience Memory for SWE-agent"
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file (JSON or YAML)"
    )
    parser.add_argument(
        "--tool",
        choices=["A", "B"],
        help="Tool type (A for planning, B for error handling). "
             "If not specified, inferred from event."
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics instead of processing event"
    )
    parser.add_argument(
        "--export",
        metavar="DIR",
        help="Export data to directory"
    )
    parser.add_argument(
        "--task-id",
        help="Task ID for statistics or export"
    )

    args = parser.parse_args()

    # Set up Agent-mem
    try:
        adapter, config_manager = setup_agent_mem(args.config)
    except Exception as e:
        print(f"Failed to set up Agent-mem: {e}", file=sys.stderr)
        return 1

    # Handle different modes
    if args.stats:
        # Show statistics
        stats = adapter.get_task_statistics(args.task_id)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return 0

    if args.export:
        # Export data
        export_report = adapter.export_data(args.export)
        print(json.dumps(export_report, indent=2, ensure_ascii=False))
        return 0

    # Normal event processing
    event_data = load_event_from_env()

    if not event_data.get("valid", False):
        print(f"Invalid event data: {event_data}", file=sys.stderr)
        return 1

    # Handle event
    try:
        response = handle_event(adapter, event_data)
        print(json.dumps(response, indent=2, ensure_ascii=False))

        # Auto-save if configured
        if config_manager.get("storage.auto_save", True):
            adapter.graph_store.save()

        return 0
    except Exception as e:
        error_response = {
            "error": str(e),
            "event_data": event_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        print(json.dumps(error_response, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
