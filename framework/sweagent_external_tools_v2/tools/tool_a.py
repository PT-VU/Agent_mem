from __future__ import annotations

from typing import Any

from .io_utils import append_json_log


def handle_event(payload: dict[str, Any]) -> None:
    print(f"[Tool A] event={payload.get('event')} action={payload.get('action', '')}")
    append_json_log("A", payload)

