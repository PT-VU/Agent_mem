# Reproducibility Notes

The root [`README.md`](../README.md) is the primary reviewer guide. This file
records the verification boundaries and the relationship between quick checks
and thesis-scale reruns.

## Reviewer Quick Check

After installing SWE-agent and the Agent-mem runtime dependencies:

```bash
bash scripts/reviewer_quick_check.sh \
  --swe-agent-root ../SWE-agent
```

This command does not require Docker, network access, or model API credentials.
It runs publication checks and three SWE-agent dummy-runtime smoke tests:

- core Agent-mem with memory;
- core baseline without memory;
- multi-agent extension with memory.

Each smoke test covers successful actions, action errors, recovery logging, and
run completion. The with-memory tests additionally require Tool A retrieval and
at least one memory-hint injection.

## Static Publication Check

Run:

```bash
bash scripts/verify_artifact.sh
```

The script checks ASCII-only public content, absence of common secret patterns,
excluded runtime files, Shell syntax, Python compilation, JSON validity, and
the retained core evidence counts.

## Optional Real Single-Case Run

Preview a guarded single-case integration run:

```bash
bash scripts/run_reviewer_case.sh \
  --swe-agent-root ../SWE-agent \
  --mode with-mem \
  --instance django__django-16139 \
  --repeats 1
```

The script starts real work only when `--execute` is appended. Execution
requires Docker, the SWE-bench harness, and `MOONSHOT_API_KEY`. Use at least two
attempts to observe cross-attempt state reuse:

```bash
export MOONSHOT_API_KEY="your-key"
bash scripts/run_reviewer_case.sh \
  --swe-agent-root ../SWE-agent \
  --mode with-mem \
  --instance django__django-16139 \
  --repeats 2 \
  --execute
```

The reviewer wrapper disables online embeddings by default to reduce setup
time. The paper configuration enables them.

## Full Core Experiment Launcher

The thesis F3 launch chain is represented by:

```bash
WS_ROOT="$(cd ../ && pwd)" \
RUNS_ROOT="$PWD/reviewer_runs/f3" \
MODEL_CONFIG="$PWD/config/kimi_k2_5_moonshot.yaml" \
bash experiments/core_f3/scripts/launch_core_f3_watchdog.sh
```

Before launching, place `MOONSHOT_API_KEY` in `${WS_ROOT}/SWE-agent/.env`.
The script schedules 9 cases x 10 attempts x 2 conditions and is intentionally
separate from the low-cost reviewer path.

## Framework Unit Tests

Install `pytest` into the SWE-agent virtual environment, then run:

```bash
../SWE-agent/.venv/bin/python -m pytest -q \
  framework/sweagent_external_tools_v2

../SWE-agent/.venv/bin/python -m pytest -q \
  framework/sweagent_external_tools_multiagent
```

## Evidence Policy

This snapshot retains selected JSON evidence rather than full raw
trajectories. The retained files are sufficient to recount resolved,
unresolved, submitted, and incomplete outcomes at attempt granularity. Raw
event streams can be archived separately for a deeper audit.
