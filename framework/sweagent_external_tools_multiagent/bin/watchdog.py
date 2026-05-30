#!/usr/bin/env python3
"""
watchdog.py: Wall-clock timeout that excludes bridge-hook overhead.

Instead of the bare shell `timeout N cmd`, use:
    python3 watchdog.py --budget N --overhead-file /path/to/overhead.log -- cmd args...

The watchdog allows the child process to run as long as:
    (wall_elapsed - cumulative_bridge_overhead) < budget

This ensures nomem and with_mem agents both get the same amount of
"pure agent time" (LLM calls + bash execution), while bridge-hook
memory queries do not consume the budget.

Exit codes mirror `timeout(1)`:
  124  killed by budget exhaustion
  130  killed by SIGINT (Ctrl-C propagated)
  else child's own exit code
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time


def _read_cumulative_overhead(path: str) -> float:
    """Sum all overhead values (one float per line) in *path*.
    Returns 0.0 on any read/parse error."""
    try:
        with open(path, encoding="utf-8") as fh:
            return sum(float(line) for line in fh if line.strip())
    except Exception:
        return 0.0


def _clear_file(path: str) -> None:
    try:
        open(path, "w").close()
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run a command with a budget-aware wall-clock timeout.",
        add_help=True,
    )
    ap.add_argument(
        "--budget",
        type=float,
        required=True,
        metavar="SEC",
        help="Agent time budget in seconds (bridge overhead excluded)",
    )
    ap.add_argument(
        "--overhead-file",
        default="",
        metavar="PATH",
        help="Path to bridge-hook overhead log (one float per line). "
             "If empty, behaves like plain `timeout --budget SEC`.",
    )
    ap.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="Command to run (everything after --)",
    )
    args = ap.parse_args()

    cmd: list[str] = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("watchdog: no command specified", file=sys.stderr)
        return 1

    overhead_file = args.overhead_file.strip()
    if overhead_file:
        _clear_file(overhead_file)

    # start_new_session=True puts the child in its own process group so that
    # killpg() below reaches sweagent (a grandchild) as well as the shell wrapper.
    proc = subprocess.Popen(cmd, start_new_session=True)
    t_start = time.monotonic()

    def _kill_group() -> None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            pass

    try:
        while True:
            ret = proc.poll()
            if ret is not None:
                return ret

            elapsed = time.monotonic() - t_start
            overhead = _read_cumulative_overhead(overhead_file) if overhead_file else 0.0
            agent_time = elapsed - overhead

            if agent_time >= args.budget:
                _kill_group()
                print(
                    f"[watchdog] budget exhausted: "
                    f"elapsed={elapsed:.1f}s overhead={overhead:.1f}s "
                    f"agent_time={agent_time:.1f}s >= budget={args.budget:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
                return 124

            time.sleep(1.0)

    except KeyboardInterrupt:
        _kill_group()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
