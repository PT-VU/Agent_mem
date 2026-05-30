from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _default_log_path() -> Path:
    return Path(os.getenv("SWE_AGENT_EXT_TOOLS_LOG_FILE", "/tmp/sweagent_ext_tools.log"))


def append_json_log(tool_name: str, payload: dict[str, Any]) -> None:
    path = _default_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "payload": payload,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

