# Core F3 Results

## Aggregate Result

| Condition | Resolved | Total | Rate |
| --- | ---: | ---: | ---: |
| `nomem` | 62 | 90 | 68.9% |
| `with_mem` | 63 | 90 | 70.0% |
| Difference |  |  | +1.1 pp |

The aggregate improvement is small. The useful observation is heterogeneity
across diagnostic instances, not a benchmark-wide performance claim.

## Per-Instance Result

| Instance | `nomem` | `with_mem` | Input-token change |
| --- | ---: | ---: | ---: |
| `django__django-12284` | 10/10 | 9/10 | -21.9% |
| `django__django-16139` | 6/10 | 9/10 | -65.3% |
| `django__django-12497` | 6/10 | 7/10 | -5.4% |
| `sympy__sympy-24066` | 10/10 | 8/10 | -44.8% |
| `sympy__sympy-13031` | 0/10 | 6/10 | -29.6% |
| `sympy__sympy-13551` | 10/10 | 5/10 | +138.3% |
| `astropy__astropy-12907` | 10/10 | 9/10 | +42.8% |
| `astropy__astropy-14995` | 10/10 | 10/10 | -23.1% |
| `astropy__astropy-13033` | 0/10 | 0/10 | -1.6% |

## Exploratory Grouping

| Post-hoc group | Instances | `nomem` | `with_mem` | Difference |
| --- | ---: | ---: | ---: | ---: |
| Zero-resolution | 2 | 0% | 30% | +30 pp |
| Intermediate | 2 | 60% | 80% | +20 pp |
| Ceiling | 5 | 100% | 82% | -18 pp |

These groups are defined from the same data being analyzed. They are useful for
diagnosis and future validation design, but they are not an independently
validated theory.

