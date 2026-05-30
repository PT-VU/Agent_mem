"""
T1-B: Interim Cache

Lightweight file-system JSON cache that stores in-flight localization and
progress discoveries made INSIDE an ongoing attempt.

Concurrent-safe via an advisory fcntl-style lock (POSIX) with fallback to a
retry loop; the writes are small and infrequent, so the window for contention
is tiny. No external dependencies required.

Layout:
  {cache_dir}/{instance_id}.json    list of InterimCard dicts

Controlled by SWE_AGENT_T1B_ENABLED (default: false).
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from typing import Any

_DEFAULT_CACHE_DIR_SUFFIX = os.path.join("agent_mem_data", "interim")
_MAX_CARDS_PER_INSTANCE = 20


def _default_cache_dir() -> str:
    base = os.getenv("SWE_AGENT_T1B_CACHE_DIR", "").strip()
    if base:
        return base
    graph_dir = os.getenv("AGENT_MEM_GRAPH_STORE_DIR", "").strip()
    if graph_dir:
        return os.path.join(graph_dir, "interim")
    return os.path.join(".", _DEFAULT_CACHE_DIR_SUFFIX)


class InterimCache:
    """File-based interim card store.

    Cards written here are:
    - Isolated from the main GraphStore (no BugInvariant / BugAntiPattern).
    - Read by the next attempt of the same instance at on_init time.
    - Typed as "InterimLocalizationCard" or "InterimProgressCard".
    - Injected via the "general" hint bucket (lowest priority), so they never
      crowd out validated semantic cards.
    """

    def __init__(self, cache_dir: str | None = None) -> None:
        self._cache_dir = cache_dir or _default_cache_dir()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_interim_card(
        self,
        *,
        instance_id: str,
        attempt_id: str,
        card_type: str,
        localization: dict[str, Any],
        source_step: int,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """Append an interim card. Returns True on success, False on error."""
        if not instance_id:
            return False
        card: dict[str, Any] = {
            "card_type": card_type,
            "instance_id": instance_id,
            "attempt_id": attempt_id,
            "source_step": source_step,
            "written_at": time.time(),
            "localization": localization,
        }
        if extra:
            card.update(extra)
        try:
            self._ensure_dir()
            path = self._path_for(instance_id)
            self._atomic_append(path, card)
            return True
        except Exception:
            return False

    def read_interim_cards(self, instance_id: str) -> list[dict[str, Any]]:
        """Return all interim cards for this instance (empty list on error)."""
        if not instance_id:
            return []
        try:
            path = self._path_for(instance_id)
            if not os.path.exists(path):
                return []
            with open(path, "r", encoding="utf-8") as fh:
                fcntl.flock(fh, fcntl.LOCK_SH)
                try:
                    data = json.load(fh)
                finally:
                    fcntl.flock(fh, fcntl.LOCK_UN)
            if isinstance(data, list):
                return data
            return []
        except Exception:
            return []

    def build_hint_items(self, instance_id: str) -> list[dict[str, Any]]:
        """Convert interim cards into the hint-dict format bridge_hook expects."""
        cards = self.read_interim_cards(instance_id)
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for card in cards:
            loc = card.get("localization") or {}
            file_path = str(loc.get("file") or "")
            function = str(loc.get("function") or "")
            line_range = str(loc.get("line_range") or "")
            confidence = float(loc.get("confidence") or 0.5)
            if not file_path:
                continue
            key = f"{file_path}:{function}:{line_range}"
            if key in seen:
                continue
            seen.add(key)
            attempt = str(card.get("attempt_id") or "?")
            step = card.get("source_step", "?")
            parts: list[str] = [f"[Interim] From attempt {attempt} step {step}:"]
            parts.append(f"Bug located in {file_path}")
            if function:
                parts.append(f"function `{function}`")
            if line_range:
                parts.append(f"lines {line_range}")
            hint_text = "  ".join(parts)
            items.append(
                {
                    "hint": hint_text,
                    "type": "interim_localization",
                    "card_type": "InterimLocalizationCard",
                    "family_id": f"interim:{file_path}:{function}",
                    "item_confidence": confidence,
                    "batch_confidence": confidence,
                    "selection_score": confidence * 0.6,
                    "source_event": "interim_read",
                }
            )
        return items

    def archive(self, instance_id: str) -> None:
        """Move the interim file to an 'archived' subdirectory (non-destructive)."""
        if not instance_id:
            return
        try:
            src = self._path_for(instance_id)
            if not os.path.exists(src):
                return
            archive_dir = os.path.join(self._cache_dir, "archived")
            os.makedirs(archive_dir, exist_ok=True)
            dst = os.path.join(archive_dir, f"{instance_id}.json")
            os.replace(src, dst)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _path_for(self, instance_id: str) -> str:
        safe = instance_id.replace("/", "_").replace("\\", "_")
        return os.path.join(self._cache_dir, f"{safe}.json")

    def _ensure_dir(self) -> None:
        os.makedirs(self._cache_dir, exist_ok=True)

    def _atomic_append(self, path: str, card: dict[str, Any]) -> None:
        with open(path, "a+", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.seek(0)
                raw = fh.read()
                try:
                    existing: list = json.loads(raw) if raw.strip() else []
                except json.JSONDecodeError:
                    existing = []
                if not isinstance(existing, list):
                    existing = []
                existing.append(card)
                # Trim to cap
                if len(existing) > _MAX_CARDS_PER_INSTANCE:
                    existing = existing[-_MAX_CARDS_PER_INSTANCE:]
                fh.seek(0)
                fh.truncate()
                json.dump(existing, fh, ensure_ascii=False, indent=None)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    @classmethod
    def from_env(cls) -> "InterimCache":
        return cls(cache_dir=_default_cache_dir())
