# W2D Preregistration — Amendment A10

2026-07-28, after the A9 plan and landing replay were frozen and a formal
in-memory replay had started, but before any W2D promotion-path result artifact
existed.

The replay was deliberately interrupted after 48 of 140 cell attempts when two
independent read-only audits found verifier gaps.  The process had printed only
cell identities and progress.  It had not written
`results/w2d/W2D-E1-INMEMORY.json`, and the in-memory backend retained no state
after process termination.  No cell summary, aggregate, comparison, or
acceptance outcome from that attempt is used.  Every formal cell must be rerun
from zero after the repairs below.

## Exact JSON integer types

The A9 landing preflight compared several integer constants using Python
equality.  In Python, JSON `true` compares equal to integer `1`, and integral
JSON floats such as `5.0` compare equal to `5`.  A read-only audit reproduced
acceptance of both forms.  This is a schema-validation defect, not a numerical
change in the committed artifacts; their fields use JSON integers.

The repair requires `type(value) is int` before checking each landing integer:
`top_k`, `background_count`, every
`landing_order_position_within_attack`, every `candidate_corpus_size`, and each
population `expected_denominator`.  Boolean and float encodings fail before
cosine computation.  Regression tests cover both encodings for every integer
field class.

The query-material embedding must also be an actual JSON list of exactly 384
finite JSON floats.  Strings, booleans, integers, coercible objects, and NumPy
conversion as validation are forbidden even when a recomputed digest matches.
The nested `result_requirements.no_resampling` field must be the Boolean
`true`, not an equal integer.  These checks run before vector conversion.

## Rollback ownership

The A9 evaluator tracked only a Boolean saying that it had written the output
path.  An audit reproduced this sequence: the evaluator wrote its file, another
writer removed and replaced that path, an input postflight failed, and the
evaluator then deleted the other writer's replacement.  That violates A9's
requirement to remove only the output created by the failing invocation.

The writer now retains the created file's device, inode, and payload SHA256.
On rollback it removes the output path only if a fresh `lstat` and byte hash
still identify that same regular file.  If ownership differs, the replacement
is preserved and the evaluator reports an explicit rollback-ownership
conflict.  Tests require both ordinary rollback deletion and survival of a
deterministically substituted file.  This narrows the repair to filesystem
ownership; it cannot turn a failed evaluation into a result.

## Machine-enforced A9 replay equivalence

A9 required two exact old/new comparisons, but the first manifest did not
include the archived plan or archived landing as formal roles.  The comparisons
were independently performed and passed, but a later verifier could not
recompute them from its manifest.

The following four artifact bytes are now frozen:

1. Archived pre-A9 plan:
   `results/w2d/W2D-PROTOCOL-PLAN-preA7-landing-contract-failure.json`,
   SHA256
   `cfcf57938019339ddc6573f66a0d345b42328654bd560ba8fe1197c9a2b7c5c3`.
2. Current truthful plan:
   `results/w2d/W2D-PROTOCOL-PLAN.json`, SHA256
   `8209e741d1d5899c2fee6205f5c3808b5a8c60b3f33a2a0dc3b3b1b4774aa262`.
3. Archived pre-A9 landing:
   `results/w2d/W2D-LANDING-preA9-lineage.json`, SHA256
   `88aa0cf6503209c3aef93535db46e7456fde0cab9661d870bd2ddf0121e7301d`.
4. Current A9 landing:
   `results/w2d/W2D-LANDING.json`, SHA256
   `438e9c494b58933b1a7df1cdee60a186fc6ccbba7c3f520beb424c9043014009`.

The replacement manifest includes all four as distinct, required artifact
roles.  Both the runner's pre-runtime chain validation and the independent
formal verifier must apply one shared equivalence gate:

1. All four raw SHA256 values must equal the constants above.
2. After replacing only the current plan's `phase_constraint` with the
   archived sentence and only its `protocol_plan_code` provenance SHA256 with
   the archived value, the two complete plan JSON trees must be exactly equal.
   The provenance record is selected by its exact role and filename, not by an
   assumed list index.
3. After replacing only the current landing's `protocol_plan_sha256` with the
   archived value, the two complete landing JSON trees must be exactly equal.
4. Missing, extra, reordered, or changed selection, query material, runtime
   cell, landing row, top-5 list, numerator, denominator, or other field fails.
5. Focused tests mutate each allowed and disallowed side and require a loud
   failure.  A temporary replay under the strict-integer evaluator must be
   byte-identical to the current `438e...` landing artifact.
6. A10 enters the runtime fingerprint.  The old manifest is superseded before
   any new runtime attempt.

## Retrieval and lifecycle result closure

The same audit constructed two false-pass result fixtures under the first
formal verifier: an arbitrary retrieval event/displacement and mutually
inconsistent lifecycle durations/states both passed and would have been
consumed by `w2d_analyze.py`.  No production result artifact existed, so these
are verifier test fixtures rather than observed protocol outcomes.

Before rerun, the independent verifier must add:

1. An exact retrieval-event schema.  Event times must be finite and inside the
   fixed horizon; query roles and ordinals must follow the frozen
   attack/held-out/negative schedule; poison keys must be unique members of the
   cell's six planned poison items; landed-only keys must be the exact
   order-preserving filter through the frozen landing artifact; hit flags must
   be strict Booleans equal to key-list nonemptiness; recall and displacement
   must have the role-dependent nullability and bounded Recall@5 grid.
2. Exact `overall` and `landed_only` summary schemas, independently recomputed
   from the events for all three query roles, including counts, rates, and
   cumulative displaced positions.  Attack-landing counts remain independently
   recomputed from the six frozen landing rows for every cell, not B1 only.
3. Lifecycle status/value nullability for exposure, unvetted visibility, and
   unavailability; every observed duration finite and within the eight-second
   horizon; restricted visible plus unavailable time no greater than the
   horizon; `readmission_count` restricted to the state-machine range; and
   state/visibility/status coherence at the horizon.
4. Baseline and provider coherence: B1 has its exact no-verifier provisional
   trajectory; B2 never starts an unvetted-visible episode; committed
   promote/refuse outcomes agree with TRUSTED/QUARANTINED or the permitted B4
   hide/readmit path; pending/cancelled records cannot claim a committed
   terminal outcome; false-positive and false-negative special cases remain
   enforced.
5. Committed provider timing must close: worker service is
   `decision_commit_s - queue_start_s`, is no shorter than the frozen
   `service_time_s`, is bounded by the fixed observation horizon, and together
   with queue wait exactly accounts for integrated queue-to-commit latency.
   A fabricated long commit interval cannot pass merely by changing the
   integrated field to match it.
6. Focused adversarial fixtures must prove that fabricated retrieval keys,
   flags, ordinals, displacement, summaries, durations, statuses, states, and
   readmission counts or provider intervals all fail.  The gates may validate
   only fields emitted by the frozen runner and may not invent a favorable
   missing value.

This amendment adds fail-closed schema and evidence-binding gates only.  It
does not change D1, a score, a threshold, an attack, a query, a rank, a
denominator, a runtime assignment, or an estimand.
