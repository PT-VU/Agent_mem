#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _load_instance_order(cache_path: Path | None, outcome_map: dict[str, str]) -> list[str]:
    if not cache_path or not cache_path.exists():
        return sorted(outcome_map)
    try:
        values = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return sorted(outcome_map)
    if not isinstance(values, list):
        return sorted(outcome_map)
    ordered = [str(x).strip() for x in values if str(x).strip() in outcome_map]
    trailing = [iid for iid in sorted(outcome_map) if iid not in ordered]
    return ordered + trailing


def main() -> int:
    ap = argparse.ArgumentParser(description="Apply official SWE-bench evaluation results back into AgentMem.")
    ap.add_argument("--workspace-root", required=True)
    ap.add_argument("--report-json", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--cache-file", default="")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--config", default="")
    args = ap.parse_args()

    ws_root = Path(args.workspace_root).resolve()
    report_path = Path(args.report_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    out_json = Path(args.output_json).resolve()
    cache_file = Path(args.cache_file).resolve() if args.cache_file else None

    sys.path.insert(0, str(ws_root))
    # Use the package selected by V2_FW_DIR. Core v2 is the artifact default;
    # multi-agent launchers override this path with their package directory.
    default_fw_dir = Path(__file__).resolve().parents[3] / "framework" / "sweagent_external_tools_v2"
    _v2_fw_dir = os.environ.get("V2_FW_DIR", str(default_fw_dir)).strip()
    _v2_parent = str(Path(_v2_fw_dir).parent)
    if _v2_parent not in sys.path:
        sys.path.insert(0, _v2_parent)
    framework_module = Path(_v2_fw_dir).name
    bridge = importlib.import_module(
        f"{framework_module}.agent_mem.processing.official_eval_bridge"
    )
    entrypoint = importlib.import_module(f"{framework_module}.agent_mem_main")
    build_feedback_event = bridge.build_feedback_event
    build_outcome_map = bridge.build_outcome_map
    handle_event = entrypoint.handle_event
    setup_agent_mem = entrypoint.setup_agent_mem

    report = _load_json(report_path)
    outcome_map = build_outcome_map(report)
    ordered_ids = _load_instance_order(cache_file, outcome_map)

    adapter, config_manager = setup_agent_mem(args.config or None)

    records: list[dict[str, Any]] = []
    outcome_counter: Counter[str] = Counter()
    for instance_id in ordered_ids:
        instance_dir = output_dir / instance_id
        event = build_feedback_event(
            instance_id=instance_id,
            base_outcome=outcome_map[instance_id],
            report_path=str(report_path),
            instance_dir=instance_dir,
            run_id=args.run_id,
        )
        response = handle_event(adapter, event)
        feedback_report = response.get("feedback_report", {}) if isinstance(response, dict) else {}
        effective_outcome = str(feedback_report.get("outcome") or event.get("official_eval_status") or "unknown")
        outcome_counter[effective_outcome] += 1
        records.append(
            {
                "instance_id": instance_id,
                "event": event,
                "response": response,
            }
        )

    if config_manager.get("storage.auto_save", True):
        adapter.graph_store.save()

    summary = {
        "schema_version": "v1",
        "report_json": str(report_path),
        "output_dir": str(output_dir),
        "processed_instances": len(records),
        "outcome_counts": dict(outcome_counter),
        "records": records,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
