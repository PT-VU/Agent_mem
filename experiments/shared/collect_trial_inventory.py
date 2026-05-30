#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def iso_mtime(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def infer_status(summary: dict[str, Any], attempt_dir: Path, report_path: Path) -> str:
    if summary:
        if int(summary.get("resolved_instances", 0) or 0) > 0:
            return "resolved"
        if int(summary.get("incomplete_instances", 0) or 0) > 0:
            return "incomplete"
        if int(summary.get("submitted_instances", 0) or 0) > 0:
            return "submitted"
        return "unresolved"
    if report_path.exists():
        report = load_json(report_path)
        if report.get("official_eval_status"):
            return str(report["official_eval_status"])
        if report.get("local_reason"):
            return f"report_only:{report['local_reason']}"
        return "report_only"
    if attempt_dir.exists():
        return "in_progress_or_unfinished"
    return "pending"


def build_trial_record(run_root: Path, group: str, instance_id: str, attempt: int) -> dict[str, Any]:
    tag = f"{attempt:02d}"
    if group == "nomem":
        base_root = run_root / "same_problem" / "nomem"
        attempt_dir = base_root / "outputs" / instance_id / f"attempt_{tag}"
        log_path = base_root / "logs" / instance_id / f"attempt_{tag}.log"
        event_log = base_root / "logs" / instance_id / f"attempt_{tag}.jsonl"
        metrics_root = base_root / "metrics"
        context_path = metrics_root / "attempt_context" / f"{instance_id}.attempt_{tag}.context.json"
    else:
        safe_iid = instance_id.replace("/", "_")
        candidate_root = run_root / "same_problem" / "with_mem" / safe_iid / "candidates" / instance_id
        attempt_dir = candidate_root / "outputs" / instance_id / f"attempt_{tag}"
        log_path = candidate_root / "logs" / f"{instance_id}.attempt_{tag}.log"
        event_log = candidate_root / "logs" / f"{instance_id}.attempt_{tag}.jsonl"
        metrics_root = candidate_root / "metrics"
        context_path = metrics_root / "attempt_context" / f"{instance_id}.attempt_{tag}.context.json"

    summary_path = metrics_root / "attempt_summaries" / f"{instance_id}.attempt_{tag}.summary.json"
    report_path = metrics_root / "attempt_reports" / f"{instance_id}.attempt_{tag}.official_eval.json"
    feedback_path = metrics_root / "attempt_feedback" / f"{instance_id}.attempt_{tag}.feedback.json"
    pred_path = metrics_root / "attempt_reports" / f"{instance_id}.attempt_{tag}.predictions.json"
    run_context_path = metrics_root / "run_context.json"
    summary = load_json(summary_path)
    report = load_json(report_path)

    return {
        "group": group,
        "instance_id": instance_id,
        "attempt": attempt,
        "status": infer_status(summary, attempt_dir, report_path),
        "summary": {
            "resolved_instances": int(summary.get("resolved_instances", 0) or 0),
            "incomplete_instances": int(summary.get("incomplete_instances", 0) or 0),
            "submitted_instances": int(summary.get("submitted_instances", 0) or 0),
            "solved_rate_on_planned": summary.get("solved_rate_on_planned"),
        },
        "report": {
            "official_eval_status": report.get("official_eval_status"),
            "local_reason": report.get("local_reason"),
            "schema_version": report.get("schema_version"),
        },
        "paths": {
            "attempt_dir": str(attempt_dir),
            "log_path": str(log_path),
            "event_log": str(event_log),
            "predictions_json": str(pred_path),
            "report_json": str(report_path),
            "summary_json": str(summary_path),
            "feedback_json": str(feedback_path),
            "context_json": str(context_path),
            "run_context_json": str(run_context_path),
        },
        "last_update": {
            "attempt_dir": iso_mtime(attempt_dir),
            "log_path": iso_mtime(log_path),
            "event_log": iso_mtime(event_log),
            "report_json": iso_mtime(report_path),
            "summary_json": iso_mtime(summary_path),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a same-only trial inventory for analysis.")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()

    run_root = Path(args.run_root).resolve()
    same_json = run_root / "same_instances.json"
    if not same_json.exists():
        raise SystemExit(f"same_instances.json not found under {run_root}")

    payload = load_json(same_json)
    instance_ids = payload.get("instance_ids", [])
    repeats = int(payload.get("repeats", 10) or 10)
    if not isinstance(instance_ids, list) or not instance_ids:
        raise SystemExit(f"instance_ids missing in {same_json}")

    records: list[dict[str, Any]] = []
    for group in ("nomem", "with_mem"):
        for instance_id in instance_ids:
            for attempt in range(1, repeats + 1):
                records.append(build_trial_record(run_root, group, str(instance_id), attempt))

    out_path = Path(args.out).resolve() if args.out else run_root / "orchestrator_state" / "trial_inventory.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "instance_count": len(instance_ids),
        "repeats": repeats,
        "trial_count": len(records),
        "instance_ids": instance_ids,
        "records": records,
    }
    out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
