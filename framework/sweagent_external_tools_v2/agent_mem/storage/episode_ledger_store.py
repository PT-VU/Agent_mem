"""
Append-only sidecar store for v2.1 episode ledger records.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4


class EpisodeLedgerStore:
    """Stores append-only JSONL records for v2.1 sidecar streams."""

    STREAM_FILENAMES = {
        "episode_ledger": "episode_ledger.jsonl",
        "subtask_instances": "subtask_instances.jsonl",
        "subtask_edges": "subtask_edges.jsonl",
        "compiler_cards": "compiler_cards.jsonl",
    }

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir).expanduser().resolve()
        self._seen_record_ids: Dict[str, set[str]] = {}

    def append(self, event: Dict[str, Any], *, stream: str = "episode_ledger") -> Dict[str, Any]:
        """Append a single record to the requested sidecar stream."""
        if not isinstance(event, dict):
            return {
                "written": False,
                "record_id": None,
                "stream": stream,
                "skipped_reason": "invalid_event",
            }

        record_id = str(event.get("record_id") or uuid4().hex)
        payload = dict(event)
        payload.setdefault("record_id", record_id)
        payload.setdefault("written_at", datetime.now(timezone.utc).isoformat())

        try:
            path = self._stream_path(stream)
            seen_ids = self._ensure_seen_ids(stream)
            if record_id in seen_ids:
                return {
                    "written": False,
                    "record_id": record_id,
                    "stream": stream,
                    "skipped_reason": "duplicate_record_id",
                    "path": str(path),
                }
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            seen_ids.add(record_id)
        except ValueError as exc:
            return {
                "written": False,
                "record_id": None,
                "stream": stream,
                "skipped_reason": str(exc),
            }
        except TypeError:
            return {
                "written": False,
                "record_id": None,
                "stream": stream,
                "skipped_reason": "serialization_error",
            }
        except OSError:
            return {
                "written": False,
                "record_id": None,
                "stream": stream,
                "skipped_reason": "write_error",
            }

        return {
            "written": True,
            "record_id": record_id,
            "stream": stream,
            "skipped_reason": None,
            "path": str(path),
        }

    def append_batch(
        self,
        events: Iterable[Dict[str, Any]],
        *,
        stream: str = "episode_ledger",
    ) -> Dict[str, Any]:
        """Append a batch of records, preserving per-record failures."""
        results = [self.append(event, stream=stream) for event in events]
        return {
            "written": sum(1 for row in results if row.get("written")),
            "failed": sum(1 for row in results if not row.get("written")),
            "results": results,
            "stream": stream,
        }

    def stream_path(self, stream: str = "episode_ledger") -> Path:
        """Expose the resolved path for a known stream."""
        return self._stream_path(stream)

    def load_records(
        self,
        *,
        stream: str = "episode_ledger",
        limit: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Load records from a stream, optionally applying equality filters."""
        path = self._stream_path(stream)
        if not path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for raw in f:
                    text = raw.strip()
                    if not text:
                        continue
                    try:
                        row = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    if filters and not self._matches_filters(row, filters):
                        continue
                    rows.append(row)
        except OSError:
            return []
        if limit is not None and limit >= 0:
            return rows[-limit:] if limit > 0 else []
        return rows

    def load_latest_records(
        self,
        *,
        stream: str,
        key_field: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Load the latest record for each logical object key."""
        latest: Dict[str, Dict[str, Any]] = {}
        for row in self.load_records(stream=stream, filters=filters):
            key = str(row.get(key_field) or "").strip()
            if not key:
                continue
            latest[key] = row
        return list(latest.values())

    def _stream_path(self, stream: str) -> Path:
        filename = self.STREAM_FILENAMES.get(stream)
        if not filename:
            raise ValueError(f"unknown_stream:{stream}")
        return self.base_dir / filename

    def _ensure_seen_ids(self, stream: str) -> set[str]:
        cached = self._seen_record_ids.get(stream)
        if cached is not None:
            return cached
        seen: set[str] = set()
        path = self._stream_path(stream)
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    for raw in f:
                        text = raw.strip()
                        if not text:
                            continue
                        try:
                            row = json.loads(text)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(row, dict):
                            record_id = str(row.get("record_id") or "").strip()
                            if record_id:
                                seen.add(record_id)
            except OSError:
                seen = set()
        self._seen_record_ids[stream] = seen
        return seen

    @staticmethod
    def _matches_filters(row: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for key, expected in filters.items():
            actual = row.get(key)
            if isinstance(expected, (list, tuple, set)):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True
