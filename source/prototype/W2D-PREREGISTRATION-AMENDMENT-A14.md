# W2D Preregistration — Amendment A14

2026-07-28, after the first post-A13 E1 replay failed a frozen runtime gate and
before any post-A13 E1 or E2 result artifact existed.

## Failed post-A13 E1 attempt

The E1 replay under runtime fingerprint
`1943928f5e84de38b285f4247a73ee467a64c4a413990c0fc5f8a62fec85cabd`
completed 135 of 140 cells.  The next cell,
`seed-5__attack-natural_cover_suffix_v1__backlog-normal__baseline-B2__arm-oracle`,
failed the A11 requirement that every cell retain exactly 191 retrieval
events in the frozen role and ordinal sequence.  The runner raised
`runtime retrieval schedule is not the frozen 191-event sequence`.

The attempt wrote no canonical E1 result.  No cell summary, aggregate,
comparison, or acceptance outcome from the in-memory partial execution was
retained or used.  The post-A13 E2 replay had not started.

## Gate and scientific inputs remain frozen

This failure does not authorize a smaller event denominator, a longer
observation window, a changed query rate, a patched cell, or a favorable
retry.  The A11 gate remains exactly 191 events per cell: 64
attack-associated, 64 held-out-same-topic, and 63 negative-other-topic
events, all starting inside the eight-second window in the frozen role and
ordinal order.

A14 changes no detector score or threshold, attack, landing assignment,
runtime item assignment, baseline, arm, backlog, seed, query material,
query schedule, observation horizon, decision, lifecycle rule, latency
estimand, denominator, backend contract, or acceptance gate.

## Whole-grid replay under a new fingerprint

The original stopping rule permits a rerun only to clear a failed gate and
requires the whole grid to restart under a new fingerprint rather than
patching cells.  A14 therefore enters the runtime fingerprint as an audit
record.  Both E1 and E2 receive new tier manifests and a new shared manifest
authority.  E1 restarts at cell 1, and E2 may start only from its own clean
backend state.  Only complete results that independently pass every existing
gate may be analyzed or reported.
