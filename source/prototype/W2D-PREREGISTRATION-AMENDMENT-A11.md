# W2D Preregistration — Amendment A11

2026-07-28, after Amendments A9/A10 and their frozen landing artifacts, but
before any complete promotion-path result artifact existed.

A10 closes internal result contradictions.  A follow-up audit showed that two
core quantities still could not be independently recomputed because the
canonical result discarded the retrieval top-5 evidence and lifecycle
transition ledger from which the producer calculated them.  The same audit
also reproduced hash-then-reopen mixed snapshots in formal readers.  A11
freezes evidence-preserving and byte-snapshot repairs before the full grid is
rerun.

## Result schema v3: retrieval evidence

Every retrieval event retains opaque-key versions of:

1. the exact returned top-5 list;
2. the exact eligible-ground-truth top-5 list used for Recall@5; and
3. the exact poison-free top-5 list used for displacement.

For the negative-query role, the latter two lists and their two metrics remain
null exactly as in the frozen producer.  Each event also retains query start
and completion times so the start can be proven inside the eight-second query
window without pretending that an in-flight query must complete before the
boundary.

The runner maps every backend ID to a frozen opaque item key and fails on an
unknown, duplicated, under-filled, or over-filled top-5 list.  The independent
verifier then recomputes:

- poison and landed-poison lists by filtering the returned top-5;
- both hit flags;
- eligible Recall@5 from returned versus eligible top-5;
- poison-free displacement from returned versus poison-free top-5; and
- every role summary and selected-attack landing count.

The frozen schedule contains exactly 191 events per cell: 64
attack-associated, 64 held-out-same-topic, and 63 negative-other-topic events,
with role determined by event position and ordinal
`floor(position / 3) mod 6`.  Missing suffixes or added events fail.

## Result schema v3: lifecycle evidence

Every lifecycle row retains:

1. its item arrival time relative to the cell clock; and
2. the ordered state-transition ledger as relative time, old state, and new
   state.

The independent verifier validates the legal state-machine transitions and
recomputes, through the fixed horizon:

- final state and whether visibility ever started;
- restricted visible and unavailable time;
- exposure status and observed time;
- the single provisional/unvetted episode status and observed time;
- quarantine status; and
- readmission count.

The recomputed values must exactly match the canonical lifecycle fields within
the fixed numerical tolerance used for wall-clock arithmetic.  Provider
decision/state and queue/service timing closure from A10 remain separate,
mandatory gates.  The result schema is bumped from
`W2D-protocol-result-v2` to `W2D-protocol-result-v3`; no v2 result is accepted.

These extra fields are audit evidence only.  They do not change runtime
ordering, state transitions, queries, admission decisions, or any reported
estimand.

## One-byte-instance rule

Every formal reader follows one byte instance:

1. Snapshot the manifest, result, every JSON artifact, the NPZ, and relevant
   code bytes before parsing.
2. Compute the cited SHA256 from those snapshot bytes.
3. Parse JSON only from those bytes and load NPZ only from a `BytesIO` snapshot.
4. Pass the same parsed/snapshotted objects to runtime input construction,
   verification, and analysis; never reopen a cited path as the source of a
   scientific value.
5. Re-hash all live paths before success and, for a writer, after output
   creation.  Drift fails and rolls back only the invocation-owned output.

The runner, independent verifier, and analyzer each receive regression tests
that swap a file between hash and parse and require a loud failure.  Stable
committed inputs must continue to produce the same A9 landing SHA and the same
pre-runtime plan assignments.

A11 enters the runtime fingerprint and supersedes the first A10 manifest before
the 140-cell replay restarts.  It changes no D1 score, threshold, metric,
landing outcome, attack, denominator, or selection.
