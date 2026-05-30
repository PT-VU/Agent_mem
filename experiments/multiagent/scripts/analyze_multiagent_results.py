#!/usr/bin/env python3
"""
Analyze Level-3 multi-agent experiment results.

The control uses historical phase9-v2 Kimi results. Treatment groups are
t1a_only, t1c_only, t1a_t1c, t1b_only, and all_t1.

The aligned analysis uses four instances:
astropy__astropy-12057, sympy__sympy-13551, sympy__sympy-13031, and
django__django-11278.

Examples:
  python3 analyze_multiagent_results.py [--l3-root PATH] [--group GROUP]
  python3 analyze_multiagent_results.py --json-out results.json
"""

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Optional

L3_ROOT_DEFAULT = (
    "/home/pt/SWE-bench/PDDL_work_mem/06_artificial_intelligence/"
    "experiments/final_validation/multiagent_l3"
)
PHASE9_ROOT = (
    "/home/pt/SWE-bench/PDDL_work_mem/06_artificial_intelligence/"
    "experiments/final_validation/phase9_v2"
)

# Explicit Kimi control runs. Do not mix in DeepSeek or unrelated runs.
PHASE9_KIMI_RUN_ALLOWLIST = {
    "astropy__astropy-12057":  "phase9_v2_moonshot_kimi_full_12057_11693_20260426_134648",
    "sympy__sympy-13551":      "phase9_f3_9inst_both_watchdog_20260506_045411",
    "sympy__sympy-13031":      "phase9_f3_9inst_both_watchdog_20260506_045411",
    "django__django-11278":    "phase9_v2c_expand4_20260429_075835",
}

INSTANCES = [
    "astropy__astropy-12057",
    "sympy__sympy-13551",
    "sympy__sympy-13031",
    "django__django-11278",
]
TREATMENT_GROUPS = ["t1a_only", "t1c_only", "t1a_t1c", "t1b_only", "all_t1"]
ALL_GROUPS = ["control"] + TREATMENT_GROUPS


# Helpers

def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def find_latest_run(l3_root: Path, group: str) -> Optional[Path]:
    group_dir = l3_root / group
    if not group_dir.exists():
        return None
    runs = sorted(group_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for r in runs:
        if r.is_dir() and (r / "orchestrator_state").exists():
            return r
    return None


def count_traj_steps(traj_path: Path) -> int:
    try:
        data = json.loads(traj_path.read_text())
        traj = data.get("trajectory", data.get("history", []))
        return len([s for s in traj if isinstance(s, dict) and s.get("role") == "assistant"])
    except Exception:
        return -1


def load_hook_events(log_path: Path) -> list:
    events = []
    if not log_path.exists():
        return events
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except Exception:
                pass
    return events


def classify_t1_events(events: list) -> dict:
    counts = {
        "t1a": 0,
        "t1b_cache_write": 0,
        "t1b_cache_hit": 0,
        "t1c_fire": 0,
        "t1c_reject": 0,
        "t1c_skip": 0,
        "t1c_unavailable": 0,
    }
    for e in events:
        if not isinstance(e, dict):
            continue
        payload = e.get("payload") if isinstance(e.get("payload"), dict) else e
        event_name = str(payload.get("event") or "").lower()
        s = json.dumps(payload).lower()
        if event_name == "t1a_reformulation_done" or (not event_name and ("reformulat" in s or "t1a" in s)):
            counts["t1a"] += 1
        if event_name == "t1b_interim_written" or ("interim_cache" in s and "write" in s):
            counts["t1b_cache_write"] += 1
        if "interim_cache" in s and ("hit" in s or "load" in s or "read" in s):
            counts["t1b_cache_hit"] += 1
        if event_name == "t1c_critic_verdict" or ("critic" in s and "t1c" in s and "fire" in s):
            counts["t1c_fire"] += 1
        if event_name == "t1c_critic_skipped":
            counts["t1c_skip"] += 1
        if event_name == "t1c_critic_unavailable":
            counts["t1c_unavailable"] += 1
        if event_name == "t1c_critic_verdict" and str(payload.get("verdict") or "").lower() == "reject":
            counts["t1c_reject"] += 1
        elif "critic" in s and "reject" in s:
            counts["t1c_reject"] += 1
    return counts


def load_attempt_events_from_records(records: list, *, branch_key: str = "with_mem") -> list:
    """Load per-attempt JSONL hook records referenced by trial_inventory.json.

    The shared agent_mem_logs/hook_events.jsonl can be empty for tmux L3 runs because
    each SWE-agent attempt writes its own event_log. Stage2 relies on these attempt
    logs for trustworthy T1 observability.
    """
    events = []
    seen: set[str] = set()
    for r in records:
        if r.get("group") != branch_key:
            continue
        event_log = ((r.get("paths") or {}).get("event_log") or "").strip()
        if not event_log or event_log in seen:
            continue
        seen.add(event_log)
        events.extend(load_hook_events(Path(event_log)))
    return events


# Resolution metrics from a list of (attempt, is_resolved) pairs.

def resolution_metrics(attempt_list: list[tuple[int, bool]], branch: str, total_planned: int) -> dict:
    """
    attempt_list: sorted [(attempt_no, is_resolved), ...]
    total_planned: total number of attempts planned (for done/total display)
    """
    by_inst: dict[str, list] = collections.defaultdict(list)
    # attempt_list may be per-instance already or flat; handle both via caller.
    # Here we treat the whole list as one instance for flexibility.
    # Callers that have multiple instances pass per-instance data.
    raise NotImplementedError("use per_instance_resolution_metrics instead")


def per_instance_resolution_metrics(by_inst: dict[str, list[tuple[int, bool]]],
                                     done: int, in_prog: int, total: int) -> dict:
    """
    by_inst: {instance_id: [(attempt, is_resolved), ...]} sorted by attempt
    """
    num_instances = len(by_inst)
    resolved_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
    first_success_attempts = []
    post_success_trials = 0
    post_success_resolved = 0

    for iid, attempts in by_inst.items():
        attempts_sorted = sorted(attempts, key=lambda x: x[0])
        resolved_attempts = [a for a, r in attempts_sorted if r]
        first_resolved_at = resolved_attempts[0] if resolved_attempts else None

        for k in [1, 3, 5, 10]:
            if any(a <= k for a in resolved_attempts):
                resolved_at_k[k] += 1

        if first_resolved_at is not None:
            first_success_attempts.append(first_resolved_at)
            after = [(a, r) for a, r in attempts_sorted if a > first_resolved_at]
            for _, r in after:
                post_success_trials += 1
                if r:
                    post_success_resolved += 1

    def safe_div(a, b):
        return a / b if b else None

    return {
        "num_instances": num_instances,
        "total_trials": total,
        "done": done,
        "in_progress": in_prog,
        "resolved_at_1":  safe_div(resolved_at_k[1],  num_instances),
        "resolved_at_3":  safe_div(resolved_at_k[3],  num_instances),
        "resolved_at_5":  safe_div(resolved_at_k[5],  num_instances),
        "resolved_at_10": safe_div(resolved_at_k[10], num_instances),
        "num_resolved_at_10": resolved_at_k[10],
        "avg_attempts_to_first_resolve": safe_div(sum(first_success_attempts), len(first_success_attempts)),
        "post_success_consistency": safe_div(post_success_resolved, post_success_trials),
    }


# Load historical control data from phase9-v2.

def load_control_from_phase9(phase9_root: Path) -> dict:
    """
    Load withmem results for the 4 instances from phase9-v2 Kimi allowlisted runs.
    Resolution is determined by status == 'resolved' (official_eval_status not available
    in historical data; the framework wrote resolved directly into status).
    """
    base = Path(phase9_root)
    by_inst: dict[str, list[tuple[int, bool]]] = {iid: [] for iid in INSTANCES}

    for iid, run_suffix in PHASE9_KIMI_RUN_ALLOWLIST.items():
        # find the inventory that belongs to this run
        inv_path = None
        for p in base.rglob("trial_inventory.json"):
            if run_suffix in str(p):
                inv_path = p
                break
        if inv_path is None:
            continue
        d = load_json(inv_path)
        for r in d.get("records", []):
            if r.get("group") not in ("with_mem", "withmem"):
                continue
            if r["instance_id"] != iid:
                continue
            att = r["attempt"]
            is_res = (r.get("status") == "resolved")
            by_inst[iid].append((att, is_res))

    total = sum(len(v) for v in by_inst.values())
    done = sum(
        sum(1 for _, r in v)  # all historical attempts are "done"
        for v in by_inst.values()
    )
    metrics = per_instance_resolution_metrics(by_inst, done=done, in_prog=0, total=total)

    return {
        "group": "control",
        "source": "phase9_v2_historical",
        "run_dir": str(base),
        "withmem": metrics,
        "nomem": None,
        "t1_events_total": {"t1a": 0, "t1b_cache_write": 0, "t1b_cache_hit": 0, "t1c_fire": 0, "t1c_reject": 0, "t1c_skip": 0, "t1c_unavailable": 0},
        "t1_events_per_attempt": {"t1a": 0, "t1b_cache_write": 0, "t1b_cache_hit": 0, "t1c_fire": 0, "t1c_reject": 0, "t1c_skip": 0, "t1c_unavailable": 0},
        "hook_events_total": 0,
    }


# Load treatment groups from Level-3 runs.

def analyze_treatment_group(l3_root: Path, group: str) -> dict:
    run_dir = find_latest_run(l3_root, group)
    if run_dir is None:
        return {"group": group, "error": "no run dir found"}

    inv_path = run_dir / "orchestrator_state" / "trial_inventory.json"
    if not inv_path.exists():
        return {"group": group, "error": "no trial_inventory.json"}

    d = load_json(inv_path)
    records = d.get("records", [])

    def branch_metrics(branch_key: str) -> dict:
        recs = [r for r in records if r.get("group") == branch_key]
        by_inst: dict[str, list[tuple[int, bool]]] = collections.defaultdict(list)
        for r in recs:
            iid = r["instance_id"]
            att = r["attempt"]
            # prefer official_eval_status, fall back to status field
            eval_s = r["report"].get("official_eval_status")
            is_res = (eval_s == "resolved") if eval_s else (r.get("status") == "resolved")
            by_inst[iid].append((att, is_res))
        done    = sum(1 for r in recs if r["status"] not in ("in_progress_or_unfinished", "pending"))
        in_prog = sum(1 for r in recs if r["status"] == "in_progress_or_unfinished")
        return per_instance_resolution_metrics(by_inst, done=done, in_prog=in_prog, total=len(recs))

    withmem_metrics = branch_metrics("with_mem")
    nomem_metrics   = branch_metrics("nomem")

    hook_log = run_dir / "agent_mem_logs" / "hook_events.jsonl"
    shared_hook_events = load_hook_events(hook_log)
    attempt_events = load_attempt_events_from_records(records, branch_key="with_mem")
    hook_events = shared_hook_events + attempt_events
    t1_counts = classify_t1_events(hook_events)
    denom = max(withmem_metrics["done"], 1)
    t1_per_attempt = {k: v / denom for k, v in t1_counts.items()}

    return {
        "group": group,
        "source": "l3_run",
        "run_dir": str(run_dir),
        "withmem": withmem_metrics,
        "nomem":   nomem_metrics,
        "t1_events_total": t1_counts,
        "t1_events_per_attempt": t1_per_attempt,
        "hook_events_total": len(hook_events),
        "shared_hook_events_total": len(shared_hook_events),
        "attempt_event_records_total": len(attempt_events),
    }


# Formatting

def fmt(v, spec=".1%"):
    if v is None:
        return "    "
    return format(v, spec)


def print_comparison_table(results: list):
    print("\n" + "=" * 92)
    print("  Level-3 multi-agent comparison")
    print("  control = historical phase9-v2 Kimi run | treatments = Level-3 runs")
    print("=" * 92)

    # Completion status
    print("\n  Completion status")
    print(f"  {'group':<12}  {'withmem done/total':>20}  {'source'}")
    print("  " + "-" * 58)
    for r in results:
        if "error" in r:
            print(f"  {r['group']:<12}  ERROR: {r['error']}")
            continue
        wm = r["withmem"]
        pct = 100 * wm["done"] / wm["total_trials"] if wm["total_trials"] else 0
        src = "historical phase9-v2" if r.get("source") == "phase9_v2_historical" else \
              ("complete" if pct == 100 else f"partial: {pct:.0f}%")
        print(f"  {r['group']:<12}  {wm['done']:>4}/{wm['total_trials']:<5} ({pct:5.1f}%)   {src}")

    # Resolution metrics for the with-memory branch.
    print("\n  withmem resolution metrics")
    hdr = f"  {'group':<12}  {'r@1':>6}  {'r@3':>6}  {'r@5':>6}  {'r@10':>6}  {'avg_att':>8}  {'post_ok':>8}"
    print(hdr)
    print("  " + "-" * 60)
    control_r10 = None
    for r in results:
        if "error" in r:
            continue
        wm = r["withmem"]
        if r["group"] == "control":
            control_r10 = wm["resolved_at_10"]
        delta = ""
        if r["group"] != "control" and control_r10 is not None and wm["resolved_at_10"] is not None:
            d = wm["resolved_at_10"] - control_r10
            delta = f"  ({d:+.0%})"
        print(f"  {r['group']:<12}  "
              f"{fmt(wm['resolved_at_1']):>6}  "
              f"{fmt(wm['resolved_at_3']):>6}  "
              f"{fmt(wm['resolved_at_5']):>6}  "
              f"{fmt(wm['resolved_at_10']):>6}  "
              f"{fmt(wm['avg_attempts_to_first_resolve'], '.1f'):>8}  "
              f"{fmt(wm['post_success_consistency']):>8}"
              f"{delta}")

    # Tier-1 events normalized by completed with-memory attempts.
    print("\n  Tier-1 events per withmem attempt")
    print(f"  {'group':<12}  {'T1-A/att':>9}  {'T1B_w/att':>10}  {'T1B_hit/att':>12}  {'T1C_fire/att':>13}  {'T1C_rej/att':>12}  {'T1C_skip/att':>13}  {'T1C_unavail/att':>15}")
    print("  " + "-" * 106)
    for r in results:
        if "error" in r or r.get("source") == "phase9_v2_historical":
            continue
        pp = r["t1_events_per_attempt"]
        print(f"  {r['group']:<12}  "
              f"{pp.get('t1a',0):>9.2f}  "
              f"{pp.get('t1b_cache_write',0):>10.2f}  "
              f"{pp.get('t1b_cache_hit',0):>12.2f}  "
              f"{pp.get('t1c_fire',0):>13.2f}  "
              f"{pp.get('t1c_reject',0):>12.2f}  "
              f"{pp.get('t1c_skip',0):>13.2f}  "
              f"{pp.get('t1c_unavailable',0):>15.2f}")

    # Preliminary Go / No-Go signal.
    print("\n  Preliminary Go / No-Go: t1a_t1c vs control")
    t1a_t1c_r10 = None
    for r in results:
        if "error" in r:
            continue
        if r["group"] == "t1a_t1c":
            t1a_t1c_r10 = r["withmem"]["resolved_at_10"]

    if control_r10 is not None and t1a_t1c_r10 is not None:
        delta = t1a_t1c_r10 - control_r10
        if delta >= 0.10:
            verdict = "GO: T1-A+C improves resolved@10 by at least 10 pp"
        elif delta >= 0:
            verdict = f"HOLD: improvement below 10 pp ({delta:+.1%})"
        else:
            verdict = f"NO-GO: t1a_t1c underperforms control ({delta:+.1%})"
        print(f"  control  resolved@10 = {control_r10:.1%}  (historical phase9-v2, n=4)")
        print(f"  t1a_t1c resolved@10 = {t1a_t1c_r10:.1%}")
        print(f"  delta = {delta:+.1%}  {verdict}")
    else:
        remaining = sum(
            r["withmem"]["total_trials"] - r["withmem"]["done"]
            for r in results
            if "error" not in r and r["group"] != "control"
        )
        print(f"  pending trials: {remaining}")

    print("\n" + "=" * 92 + "\n")


# Main entry point

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l3-root",    default=L3_ROOT_DEFAULT)
    parser.add_argument("--phase9-root", default=PHASE9_ROOT)
    parser.add_argument("--group",      default=None, help="analyze one treatment group")
    parser.add_argument("--verbose",    action="store_true")
    parser.add_argument("--json-out",   default=None)
    args = parser.parse_args()

    l3_root    = Path(args.l3_root)
    phase9_root = Path(args.phase9_root)
    groups = [args.group] if args.group else ALL_GROUPS

    results = []
    for group in groups:
        print(f"  Analyzing {group}...", end="", flush=True)
        if group == "control":
            r = load_control_from_phase9(phase9_root)
        else:
            r = analyze_treatment_group(l3_root, group)
        results.append(r)
        if "error" in r:
            print(f" ERROR: {r['error']}")
        else:
            wm = r["withmem"]
            print(f" {wm['done']}/{wm['total_trials']} done")

    print_comparison_table(results)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2, default=str))
        print(f"Results written to: {args.json_out}")


if __name__ == "__main__":
    main()
