# W2D-E3 preregistration amendment A1 — evaluator-label binding

2026-07-29.  This amendment was written after the E3 preregistration was
committed and before an E3 runner, pilot, result, or E3-specific measurement
existed.

## Correction

The preregistration requires every Panel B cell to report false-promotion and
false-refusal counts over the 512 frozen D1 test items.  Its frozen-parent table
binds the D1 scores and decisions, but accidentally omits the already-frozen
evaluator labels needed to classify those decisions.  Neither the safe detector
input nor the score artifact contains evaluator truth, and the 150 unique items
used by the E1/E2 protocol grid cannot reconstruct the required 512-item
denominator.

Add this read-only transitive parent input:

| Input | SHA256 |
| --- | --- |
| `results/w2d/W2D-labels.json` | `05d1363510a1790d690b05640c3a2830be4e62df56b8592182904ccb29019a9e` |

That digest is the `artifact_hashes.labels` value already bound inside the
preregistered `W2D-E1-INMEMORY.json` parent.  E3 must verify both the file bytes
and that transitive binding before using the file.

For Panel B, the runner may read only the `poison` evaluator field for the 512
item keys in `W2D-test-scores.json`.  It must not expose evaluator labels to the
detector-decision provider, use them to change the frozen decision or service
time, select or order events, drop a cell, or alter any threshold.  Labels are
used only after a lifecycle event to classify the already-frozen decision as
TP/FP/TN/FN and to derive false-promotion/false-refusal accounting.

All E3 hypotheses, population sizes, seeds, ANN settings, workload cells,
orders, durations, validity gates, stopping rules, and interpretation
boundaries remain unchanged.
