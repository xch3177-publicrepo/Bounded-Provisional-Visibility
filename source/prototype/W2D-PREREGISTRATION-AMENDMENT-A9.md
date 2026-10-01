# W2D Preregistration — Amendment A9

2026-07-28, after the D1 score artifacts and the first complete B1 landing
artifact were observed, but before a runtime manifest or any promotion-path
cell existed.

This amendment records a protocol-timing deviation and freezes two
landing-consumer integrity repairs.  It is deliberately post-outcome.  It
cannot justify a new attack selection, query, candidate population, ranking
rule, denominator, or favorable replacement.

## A4.1 timing deviation

Amendment A4.1 and the protocol plan's `phase_constraint` said that the plan
would be created before the first test-item detector evaluation.  That timing
condition was not met.  The frozen test-score commit
`b2d4d2110e58d057edcbefe43a85501470ba05a6` has commit time
`2026-07-28T22:48:19-07:00`; the first protocol-plan file has filesystem birth
time `2026-07-28T22:48:53-07:00`, approximately 34 seconds later.

A9 supersedes A4.1 only as to this timing claim.  The deviation is not relabeled
as preregistration success.  The plan builder has no detector-score, threshold,
detector-metric, landing-result, or protocol-result input.  Its choices are
deterministic functions of the earlier frozen data, labels, constants, opaque
keys, and domain-separated hashes.  The first plan bytes are retained at
`results/w2d/W2D-PROTOCOL-PLAN-preA7-landing-contract-failure.json`, SHA256
`cfcf57938019339ddc6573f66a0d345b42328654bd560ba8fe1197c9a2b7c5c3`.
The later formal plan at the same SHA256 was byte-identical, which establishes
that the A7 consumer repair changed no selection or query material.

The plan's `phase_constraint` is now changed to a truthful provenance statement:
construction occurred after D1 scoring, while the builder consumed no D1
outcome, threshold, metric, landing, or protocol-result input.  Regeneration
may change only that statement and the plan-code provenance hash.  A structural
comparison against the archived `cfcf...` plan must prove every selection,
assignment, query material, runtime cell, execution order, landing record,
population, constant, and reporting rule identical before the replacement plan
is accepted.

## Observed landing result and frozen replay target

The first complete landing artifact is SHA256
`88aa0cf6503209c3aef93535db46e7456fde0cab9661d870bd2ddf0121e7301d`.
It contains 192 items: recipe-family landing is 128/128 and
natural-cover-suffix landing is 58/64.  Those outcomes are known.  Its exact
bytes are archived before the repairs below.

An audit then deterministically reproduced a mixed-snapshot failure: the
evaluator could parse one version of the plan, compute all ranks, and bind the
hash of a later version of that path.  A separate audit found that the consumer
did not reject an over-complete landing schema until after computing records.
Neither audit found evidence that the frozen `88aa...` artifact actually mixed
versions; its cited input hashes match the committed inputs.  They nevertheless
expose fail-open integrity defects.

## Frozen repairs

1. The evaluator snapshots the plan JSON, safe-input JSON, and NPZ bytes and
   their SHA256 values before parsing.  All parsing and numerical work uses
   only those snapshots.
2. It re-hashes all three live input paths immediately before output and again
   after output creation.  Any mismatch aborts and removes only the output
   created by that invocation.
3. Before any cosine computation, it requires exactly 192 landing records,
   exactly the two frozen families with counts 128 and 64, the exact population
   key lists, a query-material key universe exactly equal to the record key
   universe, and exact agreement of order positions, query binding,
   background, candidate-size, variant, topic, and source-group fields.
4. The formal plan's corrected provenance text may change its SHA256, so the
   replay artifact's plan-hash field may change.  Every per-item key, attack
   family, rank, landed flag, and top-5 key list, both population numerators and
   denominators, and every other scientific field must match the archived
   `88aa...` artifact exactly.  No tolerance and no replacement are allowed.
5. Regression tests must reproduce and reject concurrent drift of each input,
   reject missing and extra records/materials and malformed population/order
   fields, and verify rollback of an exclusively created output.
6. D1 code, scores, threshold, quality metrics, source controls, the 768-item
   background, attack texts, embeddings, and all legacy artifacts remain
   immutable and are not rerun.
7. A9 enters the formal runtime fingerprint.  Where A4.1, A7, or the old plan
   timing sentence conflicts with the factual timing above, A9 controls.

These repairs add provenance and fail-closed validation only.  They do not turn
the post-score plan into a preregistered plan and do not change an estimand.
