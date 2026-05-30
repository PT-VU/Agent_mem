from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sweagent.agent.hooks.abstract import AbstractAgentHook
from sweagent.types import AgentInfo, StepOutput, Trajectory
from sweagent.utils.log import get_logger

from .tools.io_utils import append_json_log


class BaselineLoggingHook(AbstractAgentHook):
    """Hook for no-mem baseline runs.

    It records comparable action/error/run summary signals without invoking Agent-mem.
    """

    def __init__(self, *, mode: str = "no-mem"):
        self.mode = mode
        self._agent_name = "unknown"
        self._logger = get_logger("swea-baseline-hook", emoji="")

    def on_init(self, *, agent):
        self._agent_name = agent.name

    def on_model_query(self, *, messages: list[dict[str, str]], agent: str):
        # Baseline mode intentionally does not inject any memory hints.
        return

    def _record(self, *, event: str, payload: dict[str, Any]) -> None:
        append_json_log(
            "BASELINE",
            {
                "version": "v1",
                "mode": self.mode,
                "event": event,
                "agent": self._agent_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **payload,
            },
        )

    def on_actions_generated(self, *, step: StepOutput):
        self._record(
            event="plan_generated",
            payload={
                "thought": step.thought,
                "action": step.action,
            },
        )

    def on_action_error(self, *, step: StepOutput, error_type: str, error_message: str):
        self._record(
            event="action_error",
            payload={
                "error_type": error_type,
                "error_message": error_message,
                "thought": step.thought,
                "action": step.action,
            },
        )

    def on_run_done(self, *, trajectory: Trajectory, info: AgentInfo):
        self._record(
            event="run_done",
            payload={
                "trajectory_steps": len(trajectory),
                "exit_status": info.get("exit_status"),
                "has_submission": bool(info.get("submission")),
            },
        )
        self._logger.info("Baseline log written for %d steps", len(trajectory))
