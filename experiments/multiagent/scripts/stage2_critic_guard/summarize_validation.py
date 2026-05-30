#!/usr/bin/env python3
"""Summarize a Stage2-02 validation run for quick iteration decisions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


T1_EVENTS = {
    "t1a_reformulation_done",
    "t1a_reformulation_skipped",
    "t1c_precheck_diff_captured",
    "t1c_critic_verdict",
    "t1c_critic_unavailable",
    "t1c_critic_skipped",
    "v2_patch_consistency_gate",
}


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def iter_events(path: Path):
    if not path.exists():
        return
    for line in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else row
        yield payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    inventory = load_json(run_root / "orchestrator_state" / "trial_inventory.json")
    records = [r for r in inventory.get("records", []) if r.get("group") == "with_mem"]

    by_instance: dict[str, list[dict]] = defaultdict(list)
    event_totals: Counter = Counter()
    verdicts: Counter = Counter()
    per_attempt: list[tuple[str, int, str, bool, Counter, Counter]] = []

    for record in records:
        iid = str(record.get("instance_id") or "")
        attempt = int(record.get("attempt") or 0)
        status = str(record.get("status") or "")
        eval_status = str((record.get("report") or {}).get("official_eval_status") or "")
        resolved = eval_status == "resolved" or status == "resolved"
        events = Counter()
        attempt_verdicts = Counter()
        event_log = Path((record.get("paths") or {}).get("event_log") or "")
        for payload in iter_events(event_log):
            event = str(payload.get("event") or "")
            if event in T1_EVENTS:
                events[event] += 1
                event_totals[event] += 1
            if event == "t1c_critic_verdict":
                key = f"{payload.get('source') or 'unknown'}:{payload.get('verdict') or 'unknown'}"
                verdicts[key] += 1
                attempt_verdicts[key] += 1
        by_instance[iid].append(record)
        per_attempt.append((iid, attempt, status, resolved, events, attempt_verdicts))

    print("\n== Stage2-02 Validation Summary ==")
    print(f"run_root: {run_root}")
    print(f"inventory_generated_at: {inventory.get('generated_at')}")
    print(f"with_mem_records: {len(records)}")

    print("\n-- Outcomes --")
    for iid, rows in sorted(by_instance.items()):
        rows = sorted(rows, key=lambda r: int(r.get("attempt") or 0))
        resolved_attempts = [
            int(r.get("attempt") or 0)
            for r in rows
            if str((r.get("report") or {}).get("official_eval_status") or "") == "resolved"
            or str(r.get("status") or "") == "resolved"
        ]
        incomplete = sum(1 for r in rows if str(r.get("status") or "") == "incomplete")
        print(
            f"{iid}: resolved={len(resolved_attempts)}/{len(rows)} "
            f"first={resolved_attempts[0] if resolved_attempts else '-'} incomplete={incomplete}"
        )

    print("\n-- T1 Events --")
    for key in sorted(event_totals):
        print(f"{key}: {event_totals[key]}")
    print("verdicts:", dict(verdicts))

    print("\n-- Per Attempt --")
    for iid, attempt, status, resolved, events, attempt_verdicts in sorted(per_attempt):
        if status == "pending":
            continue
        print(
            f"{iid} attempt_{attempt:02d} status={status} resolved={resolved} "
            f"events={dict(events)} verdicts={dict(attempt_verdicts)}"
        )

    print("\n-- Decision Hint --")
    t1c_real = sum(v for k, v in verdicts.items() if not k.startswith("auto:"))
    t1c_revise_reject = sum(v for k, v in verdicts.items() if k.endswith(":revise") or k.endswith(":reject"))
    unavailable = event_totals["t1c_critic_unavailable"]
    incomplete_total = sum(1 for _, _, status, _, _, _ in per_attempt if status == "incomplete")
    resolved_total = sum(1 for _, _, _, resolved, _, _ in per_attempt if resolved)
    if t1c_revise_reject == 0:
        print("NO-GO: no revise/reject yet; continue Critic guard work.")
    elif incomplete_total > 1:
        print("HOLD: Critic changed behavior, but incomplete attempts remain high; add closure/exploit control.")
    elif resolved_total >= 2:
        print("GO: mechanism and score are promising; run the two-case short validation next.")
    else:
        print("HOLD: mechanism fired, but score is still weak; inspect per-attempt traces.")
    if unavailable:
        print(f"NOTE: Critic unavailable events observed: {unavailable}; do not count them as approve.")
    if t1c_real:
        print(f"real_or_deterministic_t1c_verdicts: {t1c_real}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
