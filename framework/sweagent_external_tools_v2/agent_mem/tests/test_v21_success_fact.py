from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ..config.config_manager import ConfigManager
from ..processing.v21_shared import build_success_fact_idempotency_key, classify_success_like
from ..storage.episode_ledger_store import EpisodeLedgerStore


def test_config_manager_exposes_v21_defaults(monkeypatch):
    for name in (
        "AGENT_MEM_V21_ENABLE_SUCCESS_FACT_HOTPATH",
        "AGENT_MEM_V21_ENABLE_SIDECAR",
        "AGENT_MEM_V21_ENABLE_SUBTASK_PROJECTION",
        "AGENT_MEM_V21_ENABLE_CARD_COMPILER",
        "AGENT_MEM_V21_ENABLE_GOVERNANCE",
        "AGENT_MEM_V21_SIDECAR_DIR",
        "AGENT_MEM_V21_HOTPATH_TIMEOUT_MS",
        "AGENT_MEM_V21_COLDPATH_TIMEOUT_MS",
        "AGENT_MEM_V21_MAX_CARDS_PER_QUERY",
    ):
        monkeypatch.delenv(name, raising=False)

    config = ConfigManager()
    assert config.get("agent_mem.v21.enable_success_fact_hotpath") is False
    assert config.get("agent_mem.v21.enable_sidecar") is False
    assert config.get("agent_mem.v21.enable_subtask_projection") is False
    assert config.get("agent_mem.v21.enable_card_compiler") is False
    assert config.get("agent_mem.v21.enable_governance") is False
    assert config.get("agent_mem.v21.hotpath_timeout_ms") == 50
    assert config.get("agent_mem.v21.coldpath_timeout_ms") == 5000
    assert config.get("agent_mem.v21.max_cards_per_query") == 4


def test_config_manager_updates_v21_from_env(monkeypatch):
    monkeypatch.setenv("AGENT_MEM_V21_ENABLE_SIDECAR", "1")
    monkeypatch.setenv("AGENT_MEM_V21_ENABLE_CARD_COMPILER", "true")
    monkeypatch.setenv("AGENT_MEM_V21_SIDECAR_DIR", "/tmp/agent_mem_sidecar_test")
    monkeypatch.setenv("AGENT_MEM_V21_HOTPATH_TIMEOUT_MS", "75")
    monkeypatch.setenv("AGENT_MEM_V21_COLDPATH_TIMEOUT_MS", "9000")
    monkeypatch.setenv("AGENT_MEM_V21_MAX_CARDS_PER_QUERY", "6")

    config = ConfigManager()
    config.update_from_env()

    assert config.get("agent_mem.v21.enable_sidecar") is True
    assert config.get("agent_mem.v21.enable_card_compiler") is True
    assert config.get("agent_mem.v21.sidecar_dir") == "/tmp/agent_mem_sidecar_test"
    assert config.get("agent_mem.v21.hotpath_timeout_ms") == 75
    assert config.get("agent_mem.v21.coldpath_timeout_ms") == 9000
    assert config.get("agent_mem.v21.max_cards_per_query") == 6


def test_episode_ledger_store_append_and_batch():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EpisodeLedgerStore(tmpdir)

        result = store.append({"event_name": "action_success", "trace_id": "trace-1"})
        assert result["written"] is True
        assert result["record_id"]

        path = store.stream_path()
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1
        assert rows[0]["event_name"] == "action_success"
        assert rows[0]["record_id"] == result["record_id"]

        batch = store.append_batch(
            [
                {"subtask_instance_id": "sub-1"},
                {"subtask_instance_id": "sub-2"},
            ],
            stream="subtask_instances",
        )
        assert batch["written"] == 2
        subtask_path = store.stream_path("subtask_instances")
        assert subtask_path.exists()


def test_episode_ledger_store_is_idempotent_by_record_id():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EpisodeLedgerStore(tmpdir)
        payload = {"record_id": "trace-1::1", "event": "success_fact", "trace_id": "trace-1"}
        first = store.append(payload)
        second = store.append(payload)
        assert first["written"] is True
        assert second["written"] is False
        assert second["skipped_reason"] == "duplicate_record_id"
        rows = store.load_records(stream="episode_ledger")
        assert len(rows) == 1


def test_success_fact_helper_contract():
    assert build_success_fact_idempotency_key("trace-1", 3) == "trace-1::3"
    assert build_success_fact_idempotency_key("", 3) == ""
    assert classify_success_like(observation="command completed") is True
    assert classify_success_like(observation="Traceback: boom") is False
    assert classify_success_like(exit_status="submitted", has_submission=True) is True


def test_adapter_initializes_sidecar_store_with_v21_config(monkeypatch):
    pytest.importorskip("networkx")
    from ..integration.sweagent_adapter import SWEAgentAdapter

    monkeypatch.setenv("AGENT_MEM_ENABLE_ONLINE_EMBEDDINGS", "0")
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = SWEAgentAdapter(
            storage_dir=tmpdir,
            evidence_dir=tmpdir,
            v21_config={"enable_sidecar": True},
        )
        assert adapter.v21_config["enable_sidecar"] is True
        assert adapter.episode_ledger_store is not None
        assert adapter.v21_config["sidecar_dir"] == str(Path(tmpdir) / "sidecar")
