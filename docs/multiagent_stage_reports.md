# Multi-Agent Stage Reports

## Stage 1: Aligned T1-A + T1-C Study

The aligned four-instance treatment run resolves 17/40 attempts. Relative to
historical phase9-v2 control data, instance-level `resolved@10` changes from
75% to 100%. T1-A fires 251 times. T1-C produces zero verdicts because normal
submit actions do not carry inline patch text. The run supports a coverage
signal and T1-A activity, but not a T1-C correction claim.

## Stage 2-01: Critic Observability Repair

The two-case follow-up runs 20 attempts. Both instances are resolved at least
once. T1-C becomes observable with 20 verdicts and 28 captured precheck diffs,
but every verdict is `approve`. This repairs instrumentation, not semantic
correction.

## Stage 2-02: Critic Guard Validation

A short two-case validation runs 10 attempts and resolves 1/10, with 4
incomplete attempts. Separating `unavailable` from `approve` improves metric
meaning, but the critic remains unstable.

## Stage 3: Closure Boundary

The SymPy closure micro-experiment runs 5 attempts, resolves 0/5, and leaves 4
attempts incomplete. The dominant failure occurs before a candidate patch
exists, so a pre-submit critic is structurally late. The next useful direction
is exploration-phase closure control with deterministic validation.

