#!/usr/bin/env python3
"""Stage3 decision helper for closure/exploit micro experiments."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


T1_EVENTS = {
    "t1a_reformulation_done",
    "t1a_reformulation_skipped",
    "t1c_precheck_diff_captured",
    "t1c_critic_verdict",
    "t1c_critic_unavailable",
    "v2_patch_consistency_gate",
}

CLOSURE_EVENTS = {
    "runtime_guard_block",
    "v2_l3_dry_run",
    "v2_action_rewritten",
}


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def parse_latest_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(errors="replace").splitlines():
        if "=" not in line:
            continue
        key, raw = line.split("=", 1)
        values[key.strip()] = raw.strip().strip("'\"")
    return values


def iter_events(path: Path):
    if not path.exists():
        return
    try:
        lines = path.read_text(errors="replace").splitlines()
    except Exception:
        return
    for line in lines:
        try:
            row = json.loads(line)
        except Exception:
            continue
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else row
        if isinstance(payload, dict):
            yield payload


def tmux_alive(session: str) -> bool | None:
    if not session:
        return None
    result = subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if "Operation not permitted" in (result.stderr or ""):
        return None
    return result.returncode == 0


def record_resolved(record: dict[str, Any]) -> bool:
    report = record.get("report") if isinstance(record.get("report"), dict) else {}
    return str(report.get("official_eval_status") or "") == "resolved" or str(record.get("status") or "") == "resolved"


def record_done(record: dict[str, Any]) -> bool:
    status = str(record.get("status") or "")
    report = record.get("report") if isinstance(record.get("report"), dict) else {}
    eval_status = str(report.get("official_eval_status") or "")
    return status in {"resolved", "unresolved", "error", "incomplete", "submitted"} or eval_status in {
        "resolved",
        "unresolved",
        "error",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="")
    parser.add_argument("--session", default="")
    parser.add_argument("--from-latest-env", default="")
    parser.add_argument("--stale-sec", type=int, default=900)
    parser.add_argument("--write-json", default="")
    args = parser.parse_args()

    env_values: dict[str, str] = {}
    if args.from_latest_env:
        env_values = parse_latest_env(Path(args.from_latest_env))
    run_root = Path(args.run_root or env_values.get("RUN_ROOT", ""))
    session = args.session or env_values.get("SESSION_NAME", "")
    if not str(run_root):
        raise SystemExit("--run-root or --from-latest-env with RUN_ROOT is required")

    now = time.time()
    inventory = load_json(run_root / "orchestrator_state" / "trial_inventory.json")
    records = [r for r in inventory.get("records", []) if r.get("group") == "with_mem"]

    event_totals: Counter[str] = Counter()
    verdicts: Counter[str] = Counter()
    closure_totals: Counter[str] = Counter()
    per_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    active_event_logs: list[Path] = []

    for record in records:
        iid = str(record.get("instance_id") or "")
        per_instance[iid].append(record)
        status = str(record.get("status") or "")
        event_log = Path((record.get("paths") or {}).get("event_log") or "")
        if status in {"in_progress_or_unfinished", "running"} and event_log:
            active_event_logs.append(event_log)
        for payload in iter_events(event_log):
            event = str(payload.get("event") or "")
            if event in T1_EVENTS:
                event_totals[event] += 1
            if event in CLOSURE_EVENTS or str(payload.get("normalized_pattern_type") or "") == "closure_signal":
                closure_totals[event or "closure_signal"] += 1
            family = str(payload.get("family_id") or "")
            if family.startswith("closure_signal:") or family.startswith("step_budget:"):
                closure_totals["closure_hint"] += 1
            if event == "t1c_critic_verdict":
                key = f"{payload.get('source') or 'unknown'}:{payload.get('verdict') or 'unknown'}"
                verdicts[key] += 1

    total = len(records)
    done = sum(1 for r in records if record_done(r))
    pending = sum(1 for r in records if str(r.get("status") or "") == "pending")
    active = total - done - pending
    resolved = sum(1 for r in records if record_resolved(r))
    incomplete = sum(1 for r in records if str(r.get("status") or "") == "incomplete")

    first_resolved_by_instance: dict[str, int | None] = {}
    for iid, rows in per_instance.items():
        attempts = sorted(int(r.get("attempt") or 0) for r in rows if record_resolved(r))
        first_resolved_by_instance[iid] = attempts[0] if attempts else None

    newest_event_age = None
    existing_logs = [p for p in active_event_logs if p.exists()]
    if existing_logs:
        newest_event_age = round(now - max(p.stat().st_mtime for p in existing_logs), 1)
    resume_log_age = None
    resume_log = run_root / "orchestrator_logs" / "same_only_resume.tmux.log"
    if resume_log.exists():
        resume_log_age = round(now - resume_log.stat().st_mtime, 1)

    tmux_ok = tmux_alive(session)
    recent_active_log = active > 0 and newest_event_age is not None and newest_event_age <= args.stale_sec
    healthy = bool(recent_active_log or tmux_ok is True or (done == total and total > 0))
    if active > 0 and newest_event_age is not None and newest_event_age > args.stale_sec:
        healthy = False

    instances = set(per_instance)
    is_sympy = instances == {"sympy__sympy-13031"} or "sympy__sympy-13031" in instances
    is_django = instances == {"django__django-11278"} or "django__django-11278" in instances
    closure_events = sum(closure_totals.values())
    first_sympy = first_resolved_by_instance.get("sympy__sympy-13031")
    first_django = first_resolved_by_instance.get("django__django-11278")

    if not healthy:
        decision = "inspect_unhealthy_run"
        next_action = "inspect_tmux_processes_and_resume_log_before_code_changes"
    elif done < total:
        decision = "continue_monitoring"
        next_action = "wait_until_all_attempts_finish"
    elif is_sympy and resolved >= 2 and incomplete <= 1 and (first_sympy is not None and first_sympy <= 4) and closure_events >= 1:
        decision = "run_django_sanity"
        next_action = "run_stage3_django_sanity_before_writing_final_claim"
    elif is_sympy:
        decision = "stop_and_write_boundary_analysis"
        next_action = "do_not_expand_experiments; summarize_closure_limits_for_paper"
    elif is_django and incomplete == 0 and resolved >= 1:
        decision = "stage3_ready_for_summary"
        next_action = "write_stage3_summary_and_stop_experimentation"
    elif is_django:
        decision = "stop_due_to_django_regression"
        next_action = "do_not_expand; preserve_stage2_main_result_and_note_regression"
    else:
        decision = "inspect_trace_before_next_action"
        next_action = "unknown_instance_mix"

    result = {
        "run_root": str(run_root),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
        "health": {
            "healthy": healthy,
            "tmux_session": session,
            "tmux_alive": tmux_ok,
            "newest_active_event_age_sec": newest_event_age,
            "resume_log_age_sec": resume_log_age,
        },
        "progress": {
            "done": done,
            "total": total,
            "active": active,
            "pending": pending,
            "resolved": resolved,
            "incomplete": incomplete,
            "first_resolved_by_instance": first_resolved_by_instance,
        },
        "mechanism": {
            "events": dict(event_totals),
            "verdicts": dict(verdicts),
            "closure_events": dict(closure_totals),
            "closure_event_total": closure_events,
        },
        "decision": decision,
        "next_action": next_action,
    }

    if args.write_json:
        out = Path(args.write_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

