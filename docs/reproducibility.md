# Reproducibility Notes

## Lightweight Verification

Run:

```bash
bash scripts/verify_artifact.sh
```

This checks ASCII-only content, absence of common secret patterns, shell
syntax, Python compilation, retained evidence counts, and JSON validity.

## No-API Hook Smoke Tests

The following commands verify that each public framework package starts its
hook dispatcher, invokes Tool A, and injects at least one memory hint. LLM
extraction and online embeddings are disabled so the smoke tests do not require
API credentials or network access.

```bash
PERSIST_ROOT=/tmp/agent_mem_core_smoke \
SWE_AGENT_EXT_TOOLS_LOG_FILE=/tmp/agent_mem_core_smoke.log \
AGENT_MEM_ENABLE_LLM_EXTRACTION=0 \
AGENT_MEM_ENABLE_ONLINE_EMBEDDINGS=0 \
bash framework/sweagent_external_tools_v2/bin/start_with_external_tools.sh \
  --mode with-mem --smoke-test

PERSIST_ROOT=/tmp/agent_mem_multi_smoke \
SWE_AGENT_EXT_TOOLS_LOG_FILE=/tmp/agent_mem_multi_smoke.log \
AGENT_MEM_ENABLE_LLM_EXTRACTION=0 \
AGENT_MEM_ENABLE_ONLINE_EMBEDDINGS=0 \
bash framework/sweagent_external_tools_multiagent/bin/start_with_external_tools.sh \
  --mode with-mem --smoke-test
```

The expected output contains `SMOKE TEST PASS`, `tool_a_count=3`, and
`memory_injection_count=1`.

## Framework Unit Tests

The retained framework tests use `pytest`:

```bash
cd framework/sweagent_external_tools_v2
python3 -m pytest -q

cd ../sweagent_external_tools_multiagent
python3 -m pytest -q
```

## Core Experiment Launcher

The original F3 launch chain is represented by:

```bash
bash experiments/core_f3/scripts/launch_core_f3_watchdog.sh
```

It expects:

- an external SWE-agent checkout at `${WS_ROOT}/SWE-agent`;
- a compatible Kimi model config;
- Docker and SWE-bench images;
- `MOONSHOT_API_KEY` in the environment;
- sufficient disk space for trajectories and evaluation outputs.

## Multi-Agent Launchers

The stage launchers are under `experiments/multiagent/scripts/`. The shared
group launcher is:

```bash
bash experiments/multiagent/scripts/stage1/launch_multiagent_l3_group.sh
```

Stage-specific wrappers set the corresponding T1 and closure-control flags.

## Evidence Policy

This public snapshot keeps selected JSON evidence rather than full raw
trajectories. The retained files are sufficient to recount resolved,
unresolved, submitted, and incomplete outcomes at attempt granularity. Raw
event streams can be archived separately when a committee requires a deeper
audit.
