# Experiment Design

## Core F3 Study

The core controlled study compares standard SWE-agent (`nomem`) with Agent-mem
v2 (`with_mem`) on 9 manually selected SWE-bench Full instances. Each instance
receives 10 attempts in each condition, for 180 attempts total.

Both conditions use the same model family, prompt profile, Docker evaluation
environment, and 1,200-second net agent budget. The with-memory lane uses a
watchdog that subtracts bridge-call overhead before enforcing the budget.

The instance set is diagnostic, not random. It was selected to expose different
failure regimes under a limited API budget. Any grouping by nomem solve rate is
post hoc and should be read as hypothesis generation.

## Evidence Layout

`experiments/core_f3/evidence/` contains:

- the fixed instance list;
- the recorded run manifest;
- the active v2 feature configuration;
- 180 per-attempt summary JSON files;
- 180 official evaluation JSON files.

The raw trajectories are omitted. They occupy substantially more space and are
not necessary to verify the attempt-level outcome counts.

## Multi-Agent Auxiliary Study

The auxiliary study uses targeted aligned subsets to inspect T1-A and T1-C.
Selected per-attempt summaries and official evaluation records are retained
under `experiments/multiagent/evidence/`. The reports distinguish performance
signals from engineering observability fixes.

The Stage-1 evidence directory contains 40 `with_mem` treatment records and 6
retained `nomem` reference records. The reported T1-A + T1-C treatment metrics
use the 40 treatment attempts only.
