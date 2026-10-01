# W2D Preregistration — Amendment A15

2026-07-29, after the A14-authorized E1 whole-grid retry again failed the
frozen retrieval-schedule gate and before any post-A14 E1 or E2 result
artifact existed.

## Repeated E1 schedule failure

The A14 E1 retry completed 86 of 140 cells.  The next cell,
`seed-4__attack-recipe__backlog-normal__baseline-B2__arm-oracle`, again failed
the A11 requirement that every cell retain exactly 191 retrieval events in the
frozen role and ordinal sequence.  The runner raised
`runtime retrieval schedule is not the frozen 191-event sequence`.

The attempt wrote no canonical E1 result.  No partial cell result, aggregate,
comparison, or acceptance outcome is retained or used.  The corresponding E2
replay had not started.  A second whole-grid failure at a different cell means
the event-count loss is not treated as an isolated machine disturbance or
cleared by another unchanged retry.

## Root cause in the legacy wall-clock loop

The legacy loop increments the query counter and then calculates its next
sleep against `(i + 1) / qps`.  Its ideal starts are therefore

\[
0,\;2/qps,\;3/qps,\ldots,191/qps ,
\]

not \(0,1/qps,\ldots,190/qps\).  At 24 queries per second in an eight-second
window, this skips the \(1/qps\) release and leaves only about 40 ms between
the last ideal start and the horizon.  The loop also terminates from the live
wall clock before issuing the next query.  A sufficiently late scheduler wake
therefore silently produces a 190-event suffix even though the independent
gate correctly still requires 191.

The query observer is invoked only after a query completes, while control-plane
shutdown occurs only after the query and ingestion coroutines have both
returned.  Shutdown does not cancel an otherwise scheduled query here.  The
defect is the wall-clock-controlled event count and the skipped release, not
the observer, verifier cancellation, or the 191-event gate.

## W2D-only fixed event schedule

The legacy default workload path remains unchanged.  W2D alone opts into an
explicit schedule of exactly 191 target releases:

\[
0/qps,\;1/qps,\;2/qps,\ldots,190/qps .
\]

For each position, the runtime waits until that position's absolute target
relative to the cell clock, then records the actual query start and completion
times already required by A11.  A late wake starts immediately rather than
shifting later targets.  Before executing each query, the runtime must still
require its actual start to be strictly inside the frozen eight-second
horizon.  Missing that condition fails the cell loudly; it does not synthesize
an event, relabel a nominal release as an actual start, or run the query as
though it began inside the window.

The opt-in schedule preserves all frozen W2D quantities:

- exactly 191 events per cell;
- 24 target releases per second and an eight-second observation horizon;
- 64 attack-associated, 64 held-out-same-topic, and 63
  negative-other-topic events;
- role by position and ordinal `floor(position / 3) mod 6`; and
- the existing rule that every actual query start is inside the horizon,
  while a query that starts inside may complete after it.

This repair does not extend the window, lower the query rate, reduce or
condition the denominator, duplicate the final event, fill a missing suffix
after execution, or loosen an acceptance gate.  It removes the skipped
\(1/qps\) target and makes the already frozen event count an input to W2D
rather than an incidental outcome of scheduler timing.

## Implementation and negative gates

The fixed schedule must be an explicit W2D opt-in supplied by the W2D runner to
the shared cell runtime.  Calls that omit it retain the legacy loop and its
existing command behavior.  Regression tests must establish that:

1. the W2D opt-in invokes the query path and observer exactly 191 times in the
   frozen role and ordinal order;
2. its target releases are positions \(0\) through \(190\) divided by `qps`,
   with no skipped first interval and no 192nd event;
3. an actual start at or after the horizon raises a schedule failure instead
   of returning a shortened cell; and
4. the non-W2D default path remains unchanged.

These gates test construction and failure behavior only.  They do not supply
or alter any retrieval outcome.

## New fingerprints and mandatory full replay

A15 and its implementation enter the runtime fingerprint.  The failed A14 E1
attempt cannot be resumed at cell 87 or combined with any earlier execution.
E1 and E2 require new tier-specific manifests under a new shared authority and
complete 140-cell replays from clean tier state.  Only a complete result whose
191-event schedule and all other independent gates pass may be analyzed or
reported.

A15 changes no detector score or threshold, attack, landing assignment,
runtime item assignment, query material, baseline, arm, backlog, seed,
decision, lifecycle rule, latency estimand, backend contract, scientific
denominator, or acceptance threshold.  It repairs only how W2D realizes the
already frozen 191-event schedule.
