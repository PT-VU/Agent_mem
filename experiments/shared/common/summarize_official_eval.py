#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object")
    return data


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _safe_div(a: int, b: int) -> float:
    if b <= 0:
        return 0.0
    return float(a / b)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize official SWE-bench evaluation report.")
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()

    report_path = Path(args.report_json).resolve()
    report = _load_json(report_path)

    planned = _safe_int(report.get("total_instances"))
    submitted = _safe_int(report.get("submitted_instances"))
    completed = _safe_int(report.get("completed_instances"))
    resolved = _safe_int(report.get("resolved_instances"))
    unresolved = _safe_int(report.get("unresolved_instances"))
    errors = _safe_int(report.get("error_instances"))
    incomplete = len(report.get("incomplete_ids", [])) if isinstance(report.get("incomplete_ids"), list) else 0

    summary = {
        "run_id": args.run_id,
        "dataset": args.dataset,
        "scope": args.scope,
        "planned_instances": planned,
        "submitted_instances": submitted,
        "completed_instances": completed,
        "resolved_instances": resolved,
        "unresolved_instances": unresolved,
        "error_instances": errors,
        "incomplete_instances": incomplete,
        "incomplete_ids": report.get("incomplete_ids", []),
        "solved_rate_on_planned": round(_safe_div(resolved, planned), 6),
        "solved_rate_on_submitted": round(_safe_div(resolved, submitted), 6),
        "report_json": str(report_path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    out = Path(args.summary_out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
