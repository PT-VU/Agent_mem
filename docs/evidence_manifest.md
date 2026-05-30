# Evidence Manifest

This snapshot keeps compact, attempt-level evidence for the thesis claims. It
does not include raw trajectories, event logs, caches, or runtime state.

## Core F3

| Evidence | Count |
| --- | ---: |
| Diagnostic SWE-bench Full instances | 9 |
| Attempt summaries | 180 |
| Official evaluation JSON records | 180 |
| Conditions | `nomem`, `with_mem` |

The core design is `9 instances x 10 attempts x 2 conditions = 180 attempts`.

## Multi-Agent Auxiliary Study

| Stage | Attempt summaries | Official evaluation JSON records | Purpose |
| --- | ---: | ---: | --- |
| Stage 1: T1-A + T1-C | 46 | 46 | 40 treatment attempts plus 6 retained `nomem` references |
| Stage 2: observability | 20 | 20 | Critic observability repair |
| Stage 2: critic guard | 10 | 10 | Short guarded-critic validation |
| Stage 3: closure | 5 | 5 | Exploration-closure boundary check |

The multi-agent stages are targeted diagnostics. They are not presented as a
benchmark-scale performance estimate.

## Omitted Material

Raw trajectories and large event streams remain in the private research
workspace. They can be archived separately for a deeper committee audit if
required.
