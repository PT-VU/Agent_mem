from __future__ import annotations

from typing import Any

from .io_utils import append_json_log


def handle_event(payload: dict[str, Any]) -> None:
    error_type = payload.get("error_type", "unknown")
    print(f"[Tool B] event={payload.get('event')} error_type={error_type}")
    append_json_log("B", payload)

