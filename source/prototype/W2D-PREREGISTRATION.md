# W2D — Real Detector in the Promotion Path: Preregistration

2026-07-28, written before the detector is coded and before any detector code
touches the runtime modules. No W2D number exists yet. Nothing in this document
was chosen after seeing a result, because there is no result to see.

## What is frozen before this document

v7 is the submission version and does not change on account of anything here.

| Frozen | Value |
| --- | --- |
| Manuscript commit | `ab5b390` |
| `paper/main.pdf` sha256 | `81a5dd84117d51657cb65049c1bf2a890d5360d9dabedb9dd71f36f639057d25` |
| `paper/main.tex` sha256 | `010500097da684514365563eb0e0d133e7f2bc8add73d0556352d9031edcbb10` |
| Authoritative runtime fingerprint | `8b868792d2a387bb…` |
| Authoritative result files | the 7 listed in `results/AUTHORITATIVE.json` |

None of these is overwritten, re-run, or reinterpreted by W2D. W2D writes new
files under new names and carries its own fingerprint.

## Why

The paper's standing criticism is that its verifier is a label oracle, so no
result speaks to what happens when a real detector is wrong. W2F already
answered half of that by forcing $k$ promotion errors, but a forced error is
uncorrelated with content, arrives with no latency of its own, and cannot be
provoked. A real detector's errors have all three properties.

W2D puts an actual detector in the promotion path and measures what the
protocol does around it. The oracle arm is retained and re-run under the same
code, so every detector number has a mechanism upper bound beside it.

**What W2D does not become.** It is one detector against one attack family on
one corpus. It does not establish that the protocol is safe under detectors in
general, and no sentence may say so. If the detector turns out to be good, that
is a fact about this attack's detectability, not a validation of the protocol.

## Detector specification

**`detector.py`, D1 v1.0.** The exact score function is fixed in code and its
sha256 recorded in every result file before any test-split item is scored. The
threshold is a constant in the result file, not a runtime search.

The paper's contract (Sec. IV-C) requires an affirmative decision from two
semantically distinct families, one of source/integrity and one of
content/behavior, neither able to compensate for the other. D1 implements both
as a conjunction. There is no aggregate score and no weighting: an item is
promoted only if both families affirm.

**Family S — source/integrity.** Credential validity, provenance-chain
consistency, embedding-model version match. Under this threat model the
adversary controls an already-authorized source, so S is expected to affirm for
every item, clean and poisoned alike. It is implemented and recorded anyway,
because its inertness is a finding about when the contract binds, not a detail
to omit. See prediction P2.

**Family C — content/behavior.** Scores the item from its own text and
embedding plus the frozen corpus. Two components, both computable without any
label and without any query:

- `rep`: cosine similarity between the embeddings of the passage's first and
  second half. The published recipe restates the target query inside the
  passage, so a poison is self-similar in a way an ordinary post is not.
- `knn`: one minus the mean cosine to the item's 10 nearest corpus posts, i.e.
  how far off the corpus manifold the item sits.

`C = z(rep) + z(knn)`, z-scored against the calibration corpus statistics,
which are frozen with the threshold.

**What the detector may not see.** It never receives the ground-truth label, and
it never receives $Q_{\text{craft}}$ or $Q_{\text{tgt}}$. A detector allowed to
compare an item against the evaluation queries would trivially win, for the same
reason the paper already flags the query-coupled hiding pattern as a stress
construction rather than a realistic one. A gate asserts the query sets are not
reachable from the detector's inputs.

## Labels, splits, and threshold selection

Labels are exact by construction: the harness plants the poison and knows which
ids are poisoned. There is no annotation noise and no annotator. Labels are used
**only** to score the detector after the fact, never as a detector input.

| Split | Corpus posts | Poison templates | Used for |
| --- | --- | --- | --- |
| Calibration | disjoint subset A | T0, T1, T2 | z-statistics and the threshold |
| Test | disjoint subset B | T3, T4 | every reported number |

The threshold is chosen on the calibration split alone, at the operating point
that maximizes F1 there, and then frozen. Test-split items are scored exactly
once, with that constant. **The test templates are attack variants the threshold
has never seen.** No number in this experiment may be produced by a threshold
that was adjusted after any test item was scored; a gate asserts the recorded
threshold equals the one recomputed from calibration data alone.

If the calibration-optimal threshold produces a degenerate test result (all
items on one side), that is reported as the result. It is not re-tuned.

## What is measured

**Detector quality**, on the test split only: TP, FP, TN, FN as counts, then
precision, recall and F1. Counts are reported alongside every rate, because with
six poison items per cell a rate alone implies a resolution the design does not
have -- the same rule W2F committed to.

**Detection latency**, per item, from the verifier receiving the item to it
committing a decision: P50, P95, P99 and the count, reported separately for the
no-backlog and heavy-backlog conditions. The distribution is recorded, not just
the quantiles.

**Exposure, re-measured for B1--B4 in both arms**, oracle and detector:

- $E_u$ — must still satisfy $E_u \le T_p + \delta_{hide}$. This is the
  protocol's claim and it is verifier-agnostic by specification, so a real
  detector must not move it. See P1.
- $E_p$ — the additional exposure produced by real false promotions, using the
  existing three-state accounting.
- **Clean-content ledger** — first visibility, durable visibility, and the
  outage between them, plus the new quarantine episodes that false positives
  create.

**Right-censoring is mandatory.** A poison item that is falsely promoted and
never subsequently detected has no containment instant. Its exposure is
right-censored at the observation horizon and reported as `RIGHT_CENSORED`. It
may not be recorded as contained, may not contribute a finite duration to any
median, and may not be dropped from a denominator. The existing three-state
model already enforces this and its identities are already gated in
`verify_w2.py`; W2D adds no exception to them.

**False positives on clean content are new code.** The current harness has no
path by which a clean item fails verification -- `passes = not content_bad or
id in false_promote` makes a clean item pass unconditionally. W2D adds the
symmetric path and the quarantine accounting that goes with it. This is the one
change that touches a runtime module, and it is why the whole grid re-runs
rather than being appended to v7.

## Runs

5 seeds × {normal, heavy} backlog × {B1, B2, B3, B4} × {oracle arm, detector
arm}. Seeds are 1--5, the same as v7. E1 (exact in-memory) is the primary tier;
E2 (Milvus Lite) runs only if E1 completes and passes every gate with time left.

All runs in a single chain from one frozen working tree. One runtime
fingerprint covers the whole grid; a mid-chain edit voids every cell produced
before it, as it did once already.

## Gates, written before the run

1. **Oracle arm reproduces v7 exactly.** The oracle arm under the new code must
   equal the v7 numbers cell for cell. If it does not, the detector plumbing
   changed behaviour it has no business touching and W2D is void. This is W2F's
   gate 1 applied to a larger edit.
2. **$E_u$ holds in every detector cell.** If a real detector moves the unvetted
   bound, that is a specification error in the paper, not a result to write
   around.
3. **The threshold came from calibration only.** Recomputed from calibration
   data and compared against the constant in the result file.
4. **Splits are disjoint.** Calibration and test post ids share no element, and
   no test template appears in calibration.
5. **The detector never saw a label or a query.** Asserted on the detector's
   input signature, not on inspection.
6. **Censoring identities hold**, unchanged from `verify_w2.py`: started =
   completed + censored, one age per open episode, null is not zero, and no
   undetected false promotion is counted as contained.
7. **Clean quarantine accounting balances**: every false positive opens exactly
   one quarantine episode, which is either closed by a later affirmative
   decision or censored at the horizon.
8. **Counts accompany every rate.**

A gate that only ever passes is not evidence. Gates 1 and 2 have a failing
counterpart in the failure-path suite before the grid runs.

## Predictions, recorded before the run

1. **$E_u$ is unchanged between arms.** The deadline scheduler does not wait on
   the verifier, so detector latency -- including its P99 tail under backlog --
   raises promotion latency without raising unvetted exposure. This is the
   single prediction the paper's central claim depends on.
2. **Family S affirms for every item in every cell.** The adversary holds a
   valid credential, so all discrimination comes from family C. The two-family
   contract will not be observed to do any work here, and reporting it as though
   it were validated would be false.
3. **The detector has nonzero FN on the unseen templates**, producing real false
   promotions, and some of those are never subsequently detected and are
   therefore right-censored.
4. **The detector has nonzero FP on clean content**, so the detector arm's clean
   availability is strictly worse than the oracle arm's. The protocol's clean
   cost under a real detector is higher than v7 reports.
5. **Detection latency P99 under heavy backlog is far above P50**, and $E_u$ is
   unmoved by it. If P99 rises and $E_u$ rises with it, prediction 1 is wrong
   and that is the headline.
6. **The attack may be easy.** The published recipe restates the query twice
   inside the passage, which is close to the `rep` signal by construction. If
   precision and recall on unseen templates land near 1.0, the honest reading is
   that this attack family is detectable, not that the protocol is validated,
   and the conclusion is that a second, harder attack is required. This is
   predicted in advance so that a clean result cannot later be presented as a
   stronger claim than it is.

## Interpretation rules

1. The oracle arm stays in every table as the mechanism upper bound. Detector
   numbers are reported beside it, never in place of it.
2. "The detector runs" is not a result. If FP, FN, latency quantiles and the
   re-measured ledgers are not all present and gated, there is no W2D section.
3. The threshold is not touched after the first test item is scored. If it turns
   out to be badly chosen, that is reported, not fixed.
4. Unflattering outcomes are reported at the same prominence as the rest.
   Predictions 3, 4 and 6 are all unflattering and are all recorded here.
5. Undetected false promotions are right-censored, and no sentence may imply
   containment for them.
6. W2D is one detector against one attack family. A second detector or a second
   attack strengthens external validity and is out of scope for 7/31.

## Stopping rule

The grid is 5 seeds and stops at 5 seeds. Seeds are not added because a result
is not significant, and the grid is not re-run because a result is unwelcome. A
re-run is permitted only to clear a failed gate, and it restarts the whole grid
under a new fingerprint rather than patching cells.

**Cutoff.** If every gate has not passed by 2026-07-30 evening, nothing from
W2D enters the 7/31 manuscript. It is held for camera-ready. This is not a
target to be met by weakening a gate; the gates above are fixed by this
document.
