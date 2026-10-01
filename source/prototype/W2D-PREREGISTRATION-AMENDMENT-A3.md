# W2D Preregistration — Amendment A3

2026-07-28, after Amendment A2 was independently audited and still before any
detector implementation, detector score, threshold, or W2D result exists. A3
wins wherever A2, A1, or the original preregistration differs.

## A3.1 — Duplicate equivalence classes and exact topic allocation

A2 incorrectly put `source_id` inside the source-group digest and then claimed
that exact duplicate posts share a group. Those statements cannot both hold.

The group identity is instead:

```text
group_id = SHA256(Unicode-NFC, whitespace-collapsed raw post content)
```

Topic and source id are member metadata and are not part of the identity.
Consequently, exact normalized duplicates always share one group, including a
duplicate that appears under different topics or source files.

Each duplicate group has one canonical member, the lexicographically smallest
`(topic, source_id)` pair. Only the canonical member is eligible for allocation;
the others are recorded as `duplicate_noncanonical` exclusions. Allocation uses
the canonical member's topic. This preserves the exact per-topic counts in
A2.1 without splitting a duplicate group. Every derived query, clean passage,
recipe poison, and natural-cover attack inherits that group id.

## A3.2 — Unified base eligibility

Before a group can enter any reference, clean, hard-negative, poison, cover, or
source-control stratum, its canonical normalized post must satisfy all of:

- 300 through 2,000 Unicode code points after whitespace collapse, inclusive;
- at least 30 whitespace-delimited words;
- both word halves defined in A2.3 are non-empty;
- the full, first-half, second-half, and query embeddings have the expected 384
  dimensions; and
- every embedding coordinate is finite and every normalized embedding has
  nonzero norm before normalization.

The same base eligibility applies to every stratum. Hard-negative structural
rules are applied only after this filter. Every excluded group and its reason
are recorded in `W2D-DATA-FREEZE.json`. If the eligible pool cannot satisfy the
exact A2.1 allocation without replacement, data construction fails; it may not
silently reduce a denominator or draw a replacement from a different topic.

## A3.3 — Finite threshold representation and strict JSON

Every detector score, z statistic, threshold candidate, latency, and reported
rate must be finite. JSON is written with non-finite values forbidden.

The two extreme threshold candidates from A2.3 are represented by the finite
values:

```text
nextafter(min_calibration_score, -infinity)
nextafter(max_calibration_score, +infinity)
```

The builder asserts that both results are finite. Interior candidates remain
the finite midpoint between adjacent distinct finite scores. Classification is
still `suspicious iff C >= threshold`, and the largest-threshold F1 tie-break is
unchanged. No `NaN`, `Infinity`, or `-Infinity` token may enter any artifact.

## A3.4 — Item keys are harness metadata, not detector input

The scorer harness may read an opaque `item_key` solely to establish the
label-blind order, locate the safe input row, time the call, and join the output
after the detector returns. It must not pass that key to `detector.score()`.

The detector call signature receives only:

```text
normalized text, full embedding, source-evidence tuple,
reference embeddings and reference group exclusions,
pinned model revision, frozen z statistics, frozen threshold
```

The score result contains no key. The harness attaches the held key to the
result after the call. A gate invokes the detector with two otherwise identical
safe rows carrying different harness keys and requires byte-identical score and
decision outputs.

Item keys themselves are opaque SHA256 digests with no stratum, split, template,
topic, attack, clean, or poison token. This is defense in depth for logs and
joins, not permission to pass them into the detector.

## A3.5 — What the integration experiment is called

A2 deliberately chose score-once measurement followed by item-level
decision/service-time replay. That design gives a paired policy comparison, but
it is not repeated online detector execution.

The experiment is therefore named:

> **promotion-path replay driven by real D1 outputs and measured service times**

It provides two distinct latency results:

1. actual detector-only latency from the isolated, score-once D1 process; and
2. queue-entry-to-commit latency when those measured service times and real D1
   decisions drive B2/B3/B4 state transitions.

No result, artifact, log, or summary may call the second measurement “online
detector latency” or claim that D1 was repeatedly executed inside the live
promotion path. A future genuine online integration would be a separate
experiment and may not overwrite this one.

## Additional gates

27. Duplicate equivalence uses normalized-content hash only; every
    noncanonical duplicate is excluded and recorded; no group crosses a split.
28. Every selected item satisfies A3.2, and exclusions and reasons balance the
    source inventory.
29. Every numeric artifact value is finite and every JSON reader and writer
    rejects non-standard constants.
30. Changing only a harness item key cannot change detector output, and the
    detector signature contains no key argument.
31. Result metadata names the protocol measurement as real-output/service-time
    replay and keeps actual detector-only latency separate from integrated
    replay latency.

