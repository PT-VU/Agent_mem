#!/usr/bin/env python3
"""Produce a machine-readable next-step decision for Stage2-02 validation."""

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
    "t1c_critic_skipped",
    "v2_patch_consistency_gate",
}


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


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


def pid_alive(pid_path: Path) -> bool | None:
    try:
        pid = int(pid_path.read_text().strip())
    except Exception:
        return None
    try:
        os.kill(pid, 0)
    except PermissionError:
        return None
    except ProcessLookupError:
        return False
    except Exception:
        return None
    return True


def record_resolved(record: dict[str, Any]) -> bool:
    report = record.get("report") if isinstance(record.get("report"), dict) else {}
    return str(report.get("official_eval_status") or "") == "resolved" or str(record.get("status") or "") == "resolved"


def record_done(record: dict[str, Any]) -> bool:
    status = str(record.get("status") or "")
    report = record.get("report") if isinstance(record.get("report"), dict) else {}
    eval_status = str(report.get("official_eval_status") or "")
    if status in {"resolved", "unresolved", "error", "incomplete", "submitted"}:
        return True
    if eval_status in {"resolved", "unresolved", "error"}:
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--session", default="l3_stage2_critic_guard_sympy5")
    parser.add_argument("--stale-sec", type=int, default=900)
    parser.add_argument("--write-json", default="")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    now = time.time()
    inventory = load_json(run_root / "orchestrator_state" / "trial_inventory.json")
    records = [r for r in inventory.get("records", []) if r.get("group") == "with_mem"]

    event_totals: Counter[str] = Counter()
    verdicts: Counter[str] = Counter()
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
            if event == "t1c_critic_verdict":
                key = f"{payload.get('source') or 'unknown'}:{payload.get('verdict') or 'unknown'}"
                verdicts[key] += 1

    total = len(records)
    done = sum(1 for r in records if record_done(r))
    incomplete = sum(1 for r in records if str(r.get("status") or "") == "incomplete")
    resolved = sum(1 for r in records if record_resolved(r))
    pending = sum(1 for r in records if str(r.get("status") or "") == "pending")
    active = total - done - pending

    first_resolved_by_instance: dict[str, int | None] = {}
    for iid, rows in per_instance.items():
        attempts = sorted(
            int(r.get("attempt") or 0)
            for r in rows
            if record_resolved(r)
        )
        first_resolved_by_instance[iid] = attempts[0] if attempts else None

    t1c_revise_reject = sum(
        count
        for key, count in verdicts.items()
        if key.endswith(":revise") or key.endswith(":reject")
    )
    t1c_real = sum(count for key, count in verdicts.items() if not key.startswith("auto:"))
    unavailable = event_totals["t1c_critic_unavailable"]

    tmux_ok = tmux_alive(args.session)
    slot_paths = [
        run_root / "resource_slots" / "same_global" / "slot_1" / "pid",
        run_root / "resource_slots" / "same_with_mem" / "slot_1" / "pid",
    ]
    slots_alive = {str(path): pid_alive(path) for path in slot_paths}

    newest_event_age = None
    existing_logs = [p for p in active_event_logs if p.exists()]
    if existing_logs:
        newest_mtime = max(p.stat().st_mtime for p in existing_logs)
        newest_event_age = round(now - newest_mtime, 1)
    resume_log_age = None
    resume_log_text = ""
    resume_log = run_root / "orchestrator_logs" / "same_only_resume.tmux.log"
    if resume_log.exists():
        resume_log_age = round(now - resume_log.stat().st_mtime, 1)
        try:
            resume_log_text = resume_log.read_text(errors="replace")[-12000:]
        except Exception:
            resume_log_text = ""
    env_prepare_failed = (
        "environment prepare failed rc=86" in resume_log_text
        or "environment_prepare:preflight_failed" in resume_log_text
        or "docker info failed" in resume_log_text
    )

    recent_active_log = active > 0 and newest_event_age is not None and newest_event_age <= args.stale_sec
    recent_orchestrator_log = pending > 0 and resume_log_age is not None and resume_log_age <= args.stale_sec
    runner_visible = tmux_ok is True or any(v is True for v in slots_alive.values())
    healthy = bool(recent_active_log or recent_orchestrator_log or runner_visible or (done == total and total > 0))
    if active > 0 and newest_event_age is not None and newest_event_age > args.stale_sec:
        healthy = False

    if env_prepare_failed:
        healthy = False
        decision = "environment_prepare_failed"
        next_action = "restore_docker_or_runtime_then_rerun_validation"
    elif not healthy:
        decision = "unhealthy_or_stale"
        next_action = "inspect_process_and_logs_before_code_changes"
    elif done < total:
        decision = "continue_monitoring"
        next_action = "wait_for_patch_or_attempt_completion"
    elif t1c_revise_reject == 0:
        decision = "needs_critic_guard_upgrade"
        next_action = "add_or_tighten_deterministic_critic_rules"
    elif incomplete > 1:
        decision = "needs_closure_exploit_upgrade"
        next_action = "add_hard_stop_for_repeated_exploration_or_unproductive_closure"
    elif resolved >= 2:
        decision = "promote_to_two_case_validation"
        next_action = "run_short_validation_on_sympy_and_django"
    else:
        decision = "inspect_trace_before_next_upgrade"
        next_action = "read_failed_attempt_trajectories_and_patch_family"

    result = {
        "run_root": str(run_root),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
        "health": {
            "healthy": healthy,
            "tmux_session": args.session,
            "tmux_alive": tmux_ok,
            "slot_pids_alive": slots_alive,
            "newest_active_event_age_sec": newest_event_age,
            "resume_log_age_sec": resume_log_age,
            "environment_prepare_failed": env_prepare_failed,
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
            "t1c_revise_reject": t1c_revise_reject,
            "t1c_real_or_deterministic": t1c_real,
            "t1c_unavailable": unavailable,
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
