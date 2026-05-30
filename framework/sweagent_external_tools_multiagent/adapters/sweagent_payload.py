from __future__ import annotations

import json
from typing import Any


def load_event_from_env(raw: str | None) -> dict[str, Any]:
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

