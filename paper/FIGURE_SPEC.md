# Figure Specifications for Agent-mem

## Figure 1: Agent-mem System Architecture (required)

Target file used by `main.tex`: `../fig1.png` (that is, `fig1.png` beside `main.tex`).

Purpose: replace the fallback schematic used by `main.tex`. The final figure must describe
the implemented Agent-mem base framework only. Do not add speculative components.

### Required visual structure

Use a clean academic systems-diagram style. Arrange the figure as two connected flows:

1. **Per-attempt execution flow** on the left or upper half.
2. **Cross-attempt memory lifecycle** on the right or lower half.

Add a narrow side annotation for the Watchdog timer and a compact legend for Modules A--E.

### Exact nodes and arrows

Per-attempt execution:

1. `SWE-agent Executor`
2. Arrow to `BridgeHook`
   - Small annotation: `captures model-query, action, error, and run-done events`
3. From `BridgeHook`, draw two branches:
   - `Tool A: Memory Query and Retrieval`
   - `Tool B: Event Logging`
4. `Tool A` reads promoted or active cards from `GraphStore`
5. Arrow from `Tool A` back to the executor:
   - `[AgentMem Hints] injected before the next model query`
6. Before submission, show:
   - `Module C: Patch Consistency Gate`
   - Inputs: `candidate patch signature` and `known failure signatures`
7. Output:
   - `Submitted Patch`

Cross-attempt lifecycle:

1. Inputs:
   - `Trajectory`
   - `Patch`
   - `Test Output`
   - `Official SWE-bench Evaluation Result`
2. Arrow to:
   - `LLM Experience Extractor`
3. Arrow to:
   - `EvaluationFeedbackProcessor`
4. Arrow to:
   - `GraphStore`
5. Inside or next to `GraphStore`, list:
   - `BugInvariantCard`
   - `SuccessPathCard`
   - `BugAntiPatternCard`
   - `PlanHintCard / RetryHintCard`
   - `TimeoutGovernanceCard / ClosureGuardCard`
6. Add a compact lifecycle annotation:
   - `CANDIDATE -> PROMOTED or SUPPRESSED after evaluation feedback`
7. Draw an arrow from `GraphStore` back to `Tool A`, labeled:
   - `retrieved by later attempts`

Watchdog side annotation:

- `Watchdog Timer`
- Formula: `net agent time = wall time - bridge overhead`
- Arrow to `SWE-agent Executor`
- Annotation: `enforces equal effective time budget`

Module legend:

- `A: Verbatim-diff augmentation`
- `B: Failure-side anti-pattern cards`
- `C: Patch consistency gate`
- `D: Reuse/explore scheduler`
- `E: Local card-confidence feedback`

### Style constraints

- White background.
- Black or dark-gray text.
- Restrained use of blue for execution arrows, green for verified feedback, and muted red
  for failure signatures.
- Use rectangular boxes with small corner radius.
- Avoid decorative illustrations, gradients, icons without meaning, or invented metrics.
- Use short labels only; preserve capitalization and component names exactly.
- Export as vector PDF when possible. Ensure all labels remain readable at LNCS column width.

### AI-generation prompt

Create a clean academic systems architecture diagram on a white background for a computer
science paper. Use two connected flows: a per-attempt SWE-agent execution loop and a
cross-attempt memory lifecycle. Include only the exact components and labels listed in this
specification. Use restrained blue execution arrows, green evaluation-feedback arrows, and
muted-red failure-signature annotations. Use rectangular boxes, no decorative imagery, no
gradients, and no invented components. Optimize for readability when embedded in a Springer
LNCS paper at full text width.

After generation, manually verify every label against this specification. Image models often
corrupt technical text; a diagram editor is preferred for the final vector export.

## Figure 2: Per-instance resolution comparison (optional, generate with plotting code)

This is not currently referenced by `main.tex`. It is recommended only if space remains after
LNCS compilation.

Do not use an image-generation model for this chart. Generate it with plotting code so that
all values remain exact.

Use a grouped bar chart with `nomem` and `with_mem` bars for these instances:

| Instance | nomem | with_mem | Diagnostic group |
| --- | ---: | ---: | --- |
| django-12284 | 100% | 90% | ceiling |
| django-16139 | 60% | 90% | intermediate |
| django-12497 | 60% | 70% | intermediate |
| sympy-24066 | 100% | 80% | ceiling |
| sympy-13031 | 0% | 60% | zero |
| sympy-13551 | 100% | 50% | ceiling |
| astropy-12907 | 100% | 90% | ceiling |
| astropy-14995 | 100% | 100% | ceiling |
| astropy-13033 | 0% | 0% | zero |

Required caption message: memory effects are heterogeneous across instances; the groups are
post-hoc diagnostic categories, not independently validated strata.

## Figure 3: Multi-agent auxiliary extension (optional)

This is not currently referenced by `main.tex`. Add it only if the auxiliary study remains in
the final 20-page version and space is available.

Required nodes:

1. `Retrieved Agent-mem Cards`
2. `T1-A: Memory Reformulation Agent`
   - output: `phase-appropriate hints`
3. `SWE-agent Executor`
4. `Candidate Patch Diff`
5. `T1-C: Pre-Submit Critic Agent`
   - inputs: `patch diff`, `issue context`, `memory cards`, `test output`
   - outputs: `approve`, `revise`, `reject`
6. Dashed optional node:
   - `T1-B: Async Interim Memory Mining`
   - annotation: `implemented but excluded from the main auxiliary experiment`

The figure must visually distinguish evaluated modules (`T1-A`, `T1-C`) from the
implemented-but-not-evaluated module (`T1-B`).
