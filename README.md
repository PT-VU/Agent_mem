# Agent-mem Thesis Artifact

This repository is the public artifact for the Agent-mem thesis. It contains
the Agent-mem v2 framework, the lightweight multi-agent extension, experiment
launchers, concise reports, and selected attempt-level evidence.

Repository URL: <https://github.com/PT-VU/Agent_mem>

The fastest verification path is intentionally offline: after installing
SWE-agent, one command checks the public snapshot and exercises both Agent-mem
implementations with SWE-agent's dummy runtime. Docker and API credentials are
only required for optional real SWE-bench execution.

## Validation Levels

| Level | Purpose | Requires Docker | Requires API Key | Typical Cost |
| --- | --- | --- | --- | --- |
| Static | Check publication hygiene, syntax, JSON, and retained evidence counts | No | No | Under one minute |
| Offline hook smoke | Exercise baseline logging, Tool A, Tool B, memory retrieval, and hint injection | No | No | About one minute |
| Real single case | Run Agent-mem against one SWE-bench Full instance and evaluate the submitted patch | Yes | Yes | Model quota and Docker image download |
| Full F3 reproduction | Re-run the thesis experiment: 9 cases x 10 attempts x 2 conditions | Yes | Yes | Expensive |

Start with the offline hook smoke test. It validates the integration surface
without spending model quota.

## Prerequisites

### Offline Verification

- Linux or WSL2.
- Git.
- Python 3.11 or 3.12.
- A SWE-agent source checkout with a local virtual environment.

### Real SWE-bench Execution

- Docker Engine with access to the Docker daemon.
- The SWE-bench evaluation harness installed in the SWE-agent virtual
  environment.
- A Moonshot API key for the bundled Kimi K2.5 reviewer configuration.
- Sufficient disk space for SWE-bench Docker images. The official SWE-bench
  documentation recommends reserving approximately 120 GB for evaluation
  images and caches.

Official references:

- SWE-agent source installation:
  <https://swe-agent.com/latest/installation/source/>
- SWE-agent repository:
  <https://github.com/SWE-agent/SWE-agent>
- SWE-bench Docker setup:
  <https://www.swebench.com/SWE-bench/guides/docker_setup/>
- SWE-bench repository:
  <https://github.com/SWE-bench/SWE-bench>

## Install SWE-agent

The public framework is tested against the SWE-agent v1.1.0 interface. Keep the
artifact and SWE-agent checkout in the same parent directory:

```text
review-workspace/
  Agent-mem-paper-artifact/
  SWE-agent/
  SWE-bench/                  # Optional for offline checks
```

From `review-workspace/`:

```bash
git clone --branch v1.1.0 --depth 1 \
  https://github.com/SWE-agent/SWE-agent.git

python3.12 -m venv SWE-agent/.venv
SWE-agent/.venv/bin/python -m pip install --upgrade pip
SWE-agent/.venv/bin/python -m pip install -e SWE-agent

SWE-agent/.venv/bin/python -m pip install \
  -r Agent-mem-paper-artifact/framework/sweagent_external_tools_v2/requirements.txt
```

For the optional framework unit tests:

```bash
SWE-agent/.venv/bin/python -m pip install pytest
```

For real SWE-bench execution, install the official harness into the same
virtual environment:

```bash
git clone --depth 1 https://github.com/SWE-bench/SWE-bench.git
SWE-agent/.venv/bin/python -m pip install -e SWE-bench
```

## Quick Offline Verification

From the artifact root:

```bash
bash scripts/reviewer_quick_check.sh \
  --swe-agent-root ../SWE-agent
```

This runs:

1. Public artifact checks: English-only text, excluded runtime files, obvious
   secret patterns, Shell syntax, Python compilation, JSON syntax, and retained
   core evidence counts.
2. Core Agent-mem smoke test in `with-mem` mode.
3. Core baseline smoke test in `no-mem` mode.
4. Multi-agent extension smoke test in `with-mem` mode.

The expected final line is:

```text
[ok] reviewer quick-check passed
```

To validate the snapshot without installing SWE-agent:

```bash
bash scripts/reviewer_quick_check.sh --static-only
```

## Configure a Real Reviewer Case

Real execution uses
[`config/kimi_k2_5_moonshot.yaml`](config/kimi_k2_5_moonshot.yaml). The file
contains no secret. Export the key in your shell:

```bash
export MOONSHOT_API_KEY="your-key"
```

Alternatively, copy the environment template to the external SWE-agent
checkout and edit it locally:

```bash
cp config/reviewer.env.example ../SWE-agent/.env
```

Never commit the populated `.env` file.

Check Docker before running a case:

```bash
docker info
docker run --rm hello-world
```

Preview the resolved configuration without using Docker or model quota:

```bash
bash scripts/run_reviewer_case.sh \
  --swe-agent-root ../SWE-agent \
  --mode with-mem \
  --instance django__django-16139 \
  --repeats 1
```

Add `--execute` only when the environment is ready:

```bash
bash scripts/run_reviewer_case.sh \
  --swe-agent-root ../SWE-agent \
  --mode with-mem \
  --instance django__django-16139 \
  --repeats 1 \
  --execute
```

Use `--repeats 2` or more to exercise cross-attempt card persistence and reuse.
The reviewer wrapper disables online embeddings by default to reduce setup
time. Set `AGENT_MEM_ENABLE_ONLINE_EMBEDDINGS=1` before execution for a
closer-to-paper run.

Run the same case without memory for a direct comparison:

```bash
bash scripts/run_reviewer_case.sh \
  --swe-agent-root ../SWE-agent \
  --mode no-mem \
  --instance django__django-16139 \
  --repeats 1 \
  --execute
```

## Full F3 Reproduction

The thesis-scale launcher remains available:

```bash
WS_ROOT="$(cd ../ && pwd)" \
RUNS_ROOT="$PWD/reviewer_runs/f3" \
MODEL_CONFIG="$PWD/config/kimi_k2_5_moonshot.yaml" \
bash experiments/core_f3/scripts/launch_core_f3_watchdog.sh
```

This launches the complete controlled study with one `nomem` lane and one
`with_mem` lane. It is not a quick test: the launcher schedules 180 attempts,
uses Docker, and consumes model API quota.

## Core Experiment Scale

The main controlled study uses 9 diagnostic SWE-bench Full instances, 10
attempts per condition, and 2 conditions: `9 x 10 x 2 = 180` attempts. The
selected cases represent 9 of the 2,294 SWE-bench Full test instances, or about
0.39%. This is a controlled diagnostic study, not a random population estimate.

Aggregate resolution changes only slightly: `nomem` resolves 62/90 attempts
(68.9%) and `with_mem` resolves 63/90 attempts (70.0%). The retained evidence
supports attempt-level recounting and the exploratory heterogeneous analysis
reported in the thesis.

## Repository Map

| Path | Purpose |
| --- | --- |
| `config/` | Secret-free reviewer configuration templates |
| `framework/sweagent_external_tools_v2/` | Agent-mem v2 used by the core F3 experiment |
| `framework/sweagent_external_tools_multiagent/` | Reproducibility snapshot derived from v2, with T1-A, T1-B, and T1-C |
| `experiments/shared/` | Shared orchestration, evaluation, watchdog, and Docker helpers |
| `experiments/core_f3/` | Core 9-instance experiment launcher, config, and selected evidence |
| `experiments/multiagent/` | Multi-agent launchers, stage scripts, config, and selected evidence |
| `scripts/reviewer_quick_check.sh` | Unified offline reviewer validation |
| `scripts/run_reviewer_case.sh` | Guarded optional real single-case execution |
| `docs/` | Short architecture, experiment, result, and scope reports |
| `paper/` | Current LNCS thesis source and architecture figure specification |

The retained evidence inventory is documented in
[`docs/evidence_manifest.md`](docs/evidence_manifest.md). More detailed
reproduction notes are available in
[`docs/reproducibility.md`](docs/reproducibility.md).

The two framework directories intentionally preserve experiment-specific
snapshots. `sweagent_external_tools_multiagent` reuses the v2 core and adds the
T1 modules, integration points, and tests. Keeping the snapshots separate
avoids rewriting the historical launcher paths used by the reported
experiments.

## Public Snapshot Policy

This is an English-only public copy assembled from the research workspace.
Raw trajectories, populated environment files, runtime caches, external
papers, and secrets are intentionally excluded.
