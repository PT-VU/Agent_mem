#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_report(instance_id: str, outcome: str, reason: str) -> dict:
    outcome = str(outcome or "incomplete").strip().lower()
    payload = {
        "schema_version": "local_eval_stub_v1",
        "total_instances": 1,
        "submitted_instances": 0,
        "completed_instances": 0,
        "resolved_instances": 0,
        "unresolved_instances": 0,
        "error_instances": 0,
        "submitted_ids": [],
        "completed_ids": [],
        "resolved_ids": [],
        "unresolved_ids": [],
        "error_ids": [],
        "empty_patch_ids": [],
        "incomplete_ids": [],
        "local_reason": reason,
    }
    if outcome == "resolved":
        payload["submitted_instances"] = 1
        payload["completed_instances"] = 1
        payload["resolved_instances"] = 1
        payload["submitted_ids"] = [instance_id]
        payload["completed_ids"] = [instance_id]
        payload["resolved_ids"] = [instance_id]
    elif outcome == "unresolved":
        payload["submitted_instances"] = 1
        payload["completed_instances"] = 1
        payload["unresolved_instances"] = 1
        payload["submitted_ids"] = [instance_id]
        payload["completed_ids"] = [instance_id]
        payload["unresolved_ids"] = [instance_id]
    elif outcome == "error":
        payload["error_instances"] = 1
        payload["error_ids"] = [instance_id]
        payload["incomplete_ids"] = [instance_id]
    else:
        payload["incomplete_ids"] = [instance_id]
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a local official-eval-like stub report.")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--outcome", default="incomplete")
    parser.add_argument("--reason", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(build_report(args.instance_id, args.outcome, args.reason), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
