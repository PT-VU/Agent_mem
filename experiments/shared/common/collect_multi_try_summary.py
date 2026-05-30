#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


FAIL_SIGNAL_RE = re.compile(r"(error|exception|traceback|failed|timeout|not found)", re.IGNORECASE)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_div(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num / den)


def _extract_injection_steps(traj_payload: dict[str, Any]) -> list[int]:
    steps: list[int] = []
    trajectory = traj_payload.get("trajectory", [])
    if not isinstance(trajectory, list):
        return steps

    for idx, step in enumerate(trajectory, start=1):
        if not isinstance(step, dict):
            continue
        query = step.get("query", [])
        if not isinstance(query, list):
            continue
        for msg in query:
            if not isinstance(msg, dict):
                continue
            content = str(msg.get("content", ""))
            if "[AgentMem Hints]" in content:
                steps.append(idx)
                break
    return steps


def _extract_fail_signal_count(traj_payload: dict[str, Any]) -> int:
    cnt = 0
    trajectory = traj_payload.get("trajectory", [])
    if not isinstance(trajectory, list):
        return cnt
    for step in trajectory:
        if not isinstance(step, dict):
            continue
        obs = str(step.get("observation", ""))
        if FAIL_SIGNAL_RE.search(obs):
            cnt += 1
    return cnt


def _parse_event_log(event_log: Path) -> dict[str, Any]:
    if not event_log.exists():
        return {
            "exists": False,
            "event_counts": {},
            "important_points": [],
        }

    counters = Counter()
    points: list[dict[str, Any]] = []
    for raw in event_log.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        payload = row.get("payload")
        if not isinstance(payload, dict):
            continue
        event = str(payload.get("event", ""))
        if not event:
            continue
        counters[event] += 1

        if event in {
            "external_tool_response",
            "memory_hint_buffered",
            "memory_injected",
            "action_error",
            "proactive_trigger",
            "external_tool_timeout",
            "external_tool_stale_fallback",
        }:
            points.append(
                {
                    "ts": row.get("ts"),
                    "event": event,
                    "trace_id": payload.get("trace_id"),
                    "step_index": payload.get("step_index"),
                    "source_event": payload.get("source_event"),
                    "source_tool": payload.get("source_tool"),
                    "trigger": payload.get("trigger"),
                    "buffered_count": payload.get("buffered_count"),
                    "hint_count": payload.get("hint_count"),
                    "confidence": payload.get("confidence"),
                    "timeout_sec": payload.get("timeout_sec"),
                }
            )

    return {
        "exists": True,
        "event_counts": dict(counters),
        "important_points": points,
    }


def _load_instance_ids(instance_list_file: Path) -> list[str]:
    data = _load_json(instance_list_file)
    if not data:
        return []
    ids = data.get("instance_ids")
    if isinstance(ids, list):
        return [str(x) for x in ids]
    return []


def _collect_attempt(
    *,
    output_root: Path,
    event_log_root: Path | None,
    instance_id: str,
    attempt: int,
) -> dict[str, Any]:
    attempt_tag = f"{attempt:02d}"
    attempt_dir = output_root / instance_id / f"attempt_{attempt_tag}"
    traj_path = attempt_dir / instance_id / f"{instance_id}.traj"
    info_path = attempt_dir / instance_id / f"{instance_id}.info"
    pred_path = attempt_dir / instance_id / f"{instance_id}.pred"
    patch_path = attempt_dir / instance_id / f"{instance_id}.patch"

    event_log = None
    event_data = {"exists": False, "event_counts": {}, "important_points": []}
    if event_log_root is not None:
        event_log = event_log_root / instance_id / f"attempt_{attempt_tag}.jsonl"
        event_data = _parse_event_log(event_log)

    traj_payload = _load_json(traj_path) if traj_path.exists() else None

    if not traj_payload:
        return {
            "attempt": attempt,
            "attempt_tag": attempt_tag,
            "status": "missing",
            "success": False,
            "exit_status": "missing",
            "step_count": 0,
            "api_calls": 0,
            "instance_cost": 0.0,
            "fail_signal_count": 0,
            "memory_injection_steps": [],
            "paths": {
                "attempt_dir": str(attempt_dir),
                "traj": str(traj_path),
                "info": str(info_path),
                "pred": str(pred_path),
                "patch": str(patch_path),
                "event_log": str(event_log) if event_log is not None else "",
            },
            "event_log": event_data,
        }

    info = traj_payload.get("info", {})
    if not isinstance(info, dict):
        info = {}
    exit_status = str(info.get("exit_status", "unknown"))
    success = bool(info.get("submission")) or exit_status == "submitted"
    model_stats = info.get("model_stats") if isinstance(info.get("model_stats"), dict) else {}

    return {
        "attempt": attempt,
        "attempt_tag": attempt_tag,
        "status": "done",
        "success": success,
        "exit_status": exit_status,
        "step_count": len(traj_payload.get("trajectory", []) or []),
        "api_calls": int(model_stats.get("api_calls", 0) or 0),
        "instance_cost": _safe_float(model_stats.get("instance_cost", 0.0), 0.0),
        "fail_signal_count": _extract_fail_signal_count(traj_payload),
        "memory_injection_steps": _extract_injection_steps(traj_payload),
        "paths": {
            "attempt_dir": str(attempt_dir),
            "traj": str(traj_path),
            "info": str(info_path),
            "pred": str(pred_path),
            "patch": str(patch_path),
            "event_log": str(event_log) if event_log is not None else "",
        },
        "event_log": event_data,
    }


def _first_success_attempt(attempts: list[dict[str, Any]]) -> int | None:
    for item in attempts:
        if bool(item.get("success")):
            return int(item.get("attempt"))
    return None


def _aggregate(per_instance: list[dict[str, Any]], repeats: int, mode: str) -> dict[str, Any]:
    instance_total = len(per_instance)
    solved_any = 0
    solved_last = 0
    first_success_attempts: list[int] = []
    success_by_attempt = [0 for _ in range(repeats)]
    step_values: list[list[float]] = [[] for _ in range(repeats)]
    cost_values: list[list[float]] = [[] for _ in range(repeats)]
    fail_values: list[list[float]] = [[] for _ in range(repeats)]
    injection_instances = 0
    memory_event_counts_by_attempt: list[Counter] = [Counter() for _ in range(repeats)]

    for row in per_instance:
        attempts = row.get("attempts", [])
        if not isinstance(attempts, list):
            continue
        success_flags = [bool(a.get("success")) for a in attempts]
        if any(success_flags):
            solved_any += 1
        if success_flags and success_flags[-1]:
            solved_last += 1

        fsa = row.get("first_success_attempt")
        if isinstance(fsa, int):
            first_success_attempts.append(fsa)

        injected_any = False
        for i in range(repeats):
            if i >= len(attempts):
                continue
            at = attempts[i]
            if bool(at.get("success")):
                success_by_attempt[i] += 1
            if at.get("status") == "done":
                step_values[i].append(float(at.get("step_count", 0)))
                cost_values[i].append(float(at.get("instance_cost", 0.0)))
                fail_values[i].append(float(at.get("fail_signal_count", 0)))

            steps = at.get("memory_injection_steps")
            if isinstance(steps, list) and steps:
                injected_any = True

            event_counts = at.get("event_log", {}).get("event_counts", {})
            if isinstance(event_counts, dict):
                for k, v in event_counts.items():
                    memory_event_counts_by_attempt[i][str(k)] += int(v)

        if injected_any:
            injection_instances += 1

    return {
        "instances_total": instance_total,
        "solved_any_instances": solved_any,
        "solved_last_attempt_instances": solved_last,
        "solved_any_rate": round(_safe_div(solved_any, instance_total), 6),
        "solved_last_attempt_rate": round(_safe_div(solved_last, instance_total), 6),
        "avg_first_success_attempt": round(float(mean(first_success_attempts)), 4) if first_success_attempts else 0.0,
        "success_count_by_attempt": {str(i + 1): success_by_attempt[i] for i in range(repeats)},
        "success_rate_by_attempt": {
            str(i + 1): round(_safe_div(success_by_attempt[i], instance_total), 6) for i in range(repeats)
        },
        "avg_steps_by_attempt": {
            str(i + 1): round(float(mean(step_values[i])), 4) if step_values[i] else 0.0 for i in range(repeats)
        },
        "avg_cost_by_attempt": {
            str(i + 1): round(float(mean(cost_values[i])), 6) if cost_values[i] else 0.0 for i in range(repeats)
        },
        "avg_fail_signals_by_attempt": {
            str(i + 1): round(float(mean(fail_values[i])), 4) if fail_values[i] else 0.0 for i in range(repeats)
        },
        "instances_with_memory_injection_detected": injection_instances if mode == "with-mem" else 0,
        "event_counts_by_attempt": {
            str(i + 1): dict(memory_event_counts_by_attempt[i]) for i in range(repeats)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect repeated-try SWE-bench experiment summary.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--mode", required=True, choices=["with-mem", "no-mem"])
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--instance-list-file", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--event-log-root", default="")
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).resolve()
    output_root = Path(args.output_root).resolve()
    instance_list_file = Path(args.instance_list_file).resolve()
    summary_out = Path(args.summary_out).resolve()
    event_log_root = Path(args.event_log_root).resolve() if args.event_log_root else None

    instance_ids = _load_instance_ids(instance_list_file)
    per_instance: list[dict[str, Any]] = []
    for instance_id in instance_ids:
        attempts: list[dict[str, Any]] = []
        for attempt in range(1, args.repeats + 1):
            attempts.append(
                _collect_attempt(
                    output_root=output_root,
                    event_log_root=event_log_root,
                    instance_id=instance_id,
                    attempt=attempt,
                )
            )

        first_success = _first_success_attempt(attempts)
        per_instance.append(
            {
                "instance_id": instance_id,
                "attempts": attempts,
                "first_success_attempt": first_success,
                "solved_any": any(bool(x.get("success")) for x in attempts),
                "solved_last_attempt": bool(attempts[-1].get("success")) if attempts else False,
            }
        )

    summary = {
        "schema_version": "v1",
        "generated_at": _now_iso(),
        "experiment_id": args.experiment_id,
        "mode": args.mode,
        "workspace_root": str(workspace_root),
        "output_root": str(output_root),
        "instance_list_file": str(instance_list_file),
        "event_log_root": str(event_log_root) if event_log_root is not None else "",
        "repeats": args.repeats,
        "aggregate": _aggregate(per_instance, args.repeats, args.mode),
        "per_instance": per_instance,
    }

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(summary_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
