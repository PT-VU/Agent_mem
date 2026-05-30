# Agent-mem Thesis Artifact

This repository is the public artifact snapshot for the Agent-mem thesis. It
contains the Agent-mem v2 implementation, the lightweight multi-agent extension,
the experiment launch chain, concise English reports, and selected result
evidence. Raw trajectories, runtime environments, caches, external papers, and
secrets are intentionally excluded.

## Repository Map

| Path | Purpose |
| --- | --- |
| `framework/sweagent_external_tools_v2/` | Agent-mem v2 used by the core F3 experiment |
| `framework/sweagent_external_tools_multiagent/` | Multi-agent extension with T1-A, T1-B, and T1-C |
| `experiments/shared/` | Shared orchestration, evaluation, watchdog, and Docker helpers |
| `experiments/core_f3/` | Core 9-instance experiment launcher, config, and selected evidence |
| `experiments/multiagent/` | Multi-agent launchers, stage scripts, config, and selected evidence |
| `docs/` | Short architecture, experiment, result, and scope reports |
| `paper/` | Current LNCS thesis source and architecture figure specification |

The retained evidence inventory is documented in
`docs/evidence_manifest.md`.

## Core Experiment Scale

The main controlled study uses 9 diagnostic SWE-bench Full instances, 10
attempts per condition, and 2 conditions: `9 x 10 x 2 = 180` attempts. The
selected cases represent 9 of the 2,294 SWE-bench Full test instances, or about
0.39%. The study is diagnostic rather than a random population estimate.

The core result is small in aggregate: `nomem` resolves 62/90 attempts (68.9%)
and `with_mem` resolves 63/90 attempts (70.0%). The artifact preserves the
heterogeneous per-instance results and treats the zone pattern as exploratory.

## Verification

Run lightweight checks without API calls:

```bash
bash scripts/verify_artifact.sh
```

Run the no-API hook smoke tests to verify the public launcher wiring:

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

Framework unit tests require `pytest` and can be run from either implementation
directory:

```bash
cd framework/sweagent_external_tools_v2
python3 -m pytest -q

cd ../sweagent_external_tools_multiagent
python3 -m pytest -q
```

Full SWE-bench reruns require an external SWE-agent checkout, Docker, benchmark
images, a compatible model config, and `MOONSHOT_API_KEY` supplied through the
environment. See `docs/reproducibility.md`.

## Public Snapshot Policy

This is an English-only public copy assembled from the research workspace.
Implementation logic and selected evidence are retained. Historical Chinese
notes were replaced by short English reports. Large raw run directories were
not copied because they contain redundant trajectories and runtime state.
