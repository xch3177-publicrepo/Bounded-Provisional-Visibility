# W2D Preregistration — Amendment A1

2026-07-28, still before any detector code exists and before any W2D datum is
produced. `W2D-PREREGISTRATION.md` is not modified; this document adds to it and
wins wherever the two differ.

A1 exists because the original design would have produced a defensible-looking
precision and recall that did not answer the criticism it was written for. Four
of the items below are gaps in the original, not refinements of it.

## A1.0 — Standalone scale: what the evidence actually supports

Not a W2D item; recorded here because it was found while checking A1 and it
changes the manuscript.

The v7 text said the standalone deployment runs "over a $200$-item,
$384$-dimensional collection". Checked against the code rather than against the
config keys:

- `n_items` is the collection size, and each item is inserted, deadlined and
  hidden exactly once, so 200 items and 200 hide events per cell are the same
  200 things. This part was right.
- `DIM = 384` is a constant in `standalone_experiments.py` annotated *matches
  the paper's all-MiniLM-L6-v2 embedding size*, and the vectors come from
  `vecs_np`, which is `rng.standard_normal(...)` normalized. **They are
  synthetic random unit vectors, not encoded text.**

Every individual number was true and the sentence as a whole was not: three
pages earlier the manuscript says the real-text corpus is "encoded by
all-MiniLM-L6-v2 at $384$ dimensions", so a reader joining the two would
conclude the standalone tier ran on the encoded newsgroup corpus. It did not.

The text now reads "over a $200$-item collection of synthetic unit vectors",
which states the scale and forecloses the misreading. Dimension is dropped: it
carries no information here except the one wrong inference.

## A1.1 — Label isolation is enforced in code, not asserted in prose

The feature path must not be able to read the answer. Concretely, no module
reachable from `detector.score()` may import or receive `content_bad`, the
poison id set, `Q_craft`, `Q_tgt`, or the template table. The detector receives
one item's text and embedding plus the frozen calibration corpus statistics,
and nothing else.

A gate inspects the detector's call signature and module imports and fails on
any of those names. The original preregistration asserted this; A1 makes it
checkable.

## A1.2 — Grouped split, by source document and topic

The original split calibration and test by post subset and by poison template.
That is not sufficient. A poison passage is built from a query, and a query is
built from the opening words of a held-out post; if that post's siblings sit in
calibration while the poison derived from it sits in test, the threshold has
seen the neighbourhood it is scored on.

Splitting is therefore by **source document group**: every passage, query and
derived poison tracing to the same original post travels together, and groups
are assigned to calibration or test whole. Topic membership is balanced across
the two sides so neither is a different corpus. A gate asserts the group
partition, not merely the id partition.

## A1.3 — Gold labels are frozen, and are evaluator-only

An unsupervised feature is not a label-free evaluation. FP, FN, precision and
recall all require ground truth, and that ground truth must be fixed before
scoring rather than derived alongside it.

Labels are written once to `results/W2D-labels.json` with a sha256, before any
item is scored, and are read only by the evaluator. The gate in A1.1 is what
keeps them out of the feature path.

## A1.4 — Clean hard negatives

This is the item most likely to have inflated the original design's precision.
`rep`, the first-half/second-half self-similarity, fires on any legitimately
repetitive passage. A 20 Newsgroups corpus loaded with `remove=("headers",
"footers", "quotes")` has had most of its naturally repetitive material
stripped, so the clean side of the original design was unrepresentatively easy
and precision would have been flattered by the loader, not earned by the
detector.

The clean set gains an explicit hard-negative stratum, drawn from the same
corpus and labelled clean:

1. quoted-reply posts, loaded **without** the `quotes` removal, so the passage
   restates earlier text the way a poison restates a query;
2. FAQ and boilerplate posts, which repeat their own headings;
3. cross-posted near-duplicates appearing under two groups;
4. posts carrying signature blocks and template footers, loaded without the
   `footers` removal;
5. topic-boundary posts, nearest-neighbour to a cluster other than their own,
   which is what `knn` keys on.

Hard negatives are at least 25\% of the clean test stratum. FP is reported
separately for the ordinary and the hard-negative strata; a single pooled FP
rate that hides a high hard-negative FP rate is not an acceptable report.

## A1.5 — What D1 may be called

Until a second attack family with no surface feature in common with the
published recipe has been run and gated, D1 is named a **recipe-specific
feasibility detector** in every sentence that mentions it, in the paper, in the
result files and in this document's successors. It is not a poisoning detector,
not a general detector, and not evidence about detectors as a class.

If the second attack lands, the naming is revisited in an amendment, not by
editing this line.

## A1.6 — Negative controls for the source/integrity family

The original preregistration predicted that family S affirms for every item and
recorded that as a finding. That is not enough: a family that is never
exercised negatively has not been tested, and reporting a conjunctive
two-signal rule on the strength of it would be a claim the run does not support.

A negative-control stratum is added in which S must refuse: invalid signature,
provenance conflict against the lineage record, and unknown or uncredentialed
source. These items are clean in content, so only S can stop them. If S does
not refuse all of them, the S implementation is broken and W2D is void.

With this stratum, the reportable claim is that the conjunction was exercised on
both sides. Without it, the only defensible sentence is that family C was
tested, and that sentence is what gets written if the stratum does not run.

## A1.7 — One primary threshold, and intervals on every rate

One primary threshold, chosen on calibration by F1, frozen, and applied once to
the test split. No secondary threshold, no per-cell threshold, no re-tuning
after any test item is scored.

Every reported rate carries a Wilson 95\% interval alongside its numerator and
denominator. With six poison items per cell the interval on recall is wide and
saying so is the point. **A bare "0\% false positives" is not reportable**: it
is written as $0/n$ with its interval, because zero of a small denominator is
not evidence of zero.

FPR and FNR are reported alongside precision and recall, since precision alone
moves with prevalence and prevalence here is a design choice.

## A1.8 — Paired comparison across baselines

B2, B3 and B4 within a cell consume **the same detector decisions and the same
latency draws**. The baselines differ in admission and deadline policy, and
that is the only thing that may differ between them; re-drawing the detector's
behaviour per baseline would put detector variance into a comparison meant to
isolate protocol variance.

A gate asserts the decision and latency vectors are identical across B2, B3 and
B4 for a given seed and cell.

## A1.9 — Lifecycle semantics for both error types

**False positive on clean content.** The item enters QUARANTINED and stays
there until an affirmative decision arrives or the horizon ends the episode. It
counts into the misquarantine total, into first-visibility delay, and into the
clean-availability ledger. A false positive that is silently re-admitted
without an episode being recorded is a measurement bug, not a lucky recovery.

**False negative on poison.** A falsely promoted item stays TRUSTED. Unless a
second detection pass is preregistered -- and none is -- there is no later
event that contains it, and its exposure is right-censored at the horizon. **No
containment event may be synthesized for it**, it contributes no finite
duration to any median, and it is not dropped from a denominator.

## A1.10 — Latency is reported twice

- **Detector-only**: time inside `detector.score()`, the compute cost of the
  decision.
- **Integrated**: time from the item entering the verification queue to its
  decision committing, which includes queueing.

Both as P50, P95, P99 with counts, for both backlog conditions, and the queue
length is recorded at each decision. The gap between the two is the backlog,
and reporting only one of them would let a queueing effect read as detector
cost or the reverse.

## Additional gates

Numbered from the original's eight.

9. No label, poison id, query set or template name is reachable from the
   detector's inputs (A1.1).
10. The calibration/test partition is a partition of source-document groups,
    and topic balance across sides is within tolerance (A1.2).
11. `W2D-labels.json` exists with its sha256 recorded before the first score
    is computed (A1.3).
12. The hard-negative stratum is present and is at least 25\% of the clean test
    stratum, and FP is reported per stratum (A1.4).
13. Family S refuses every item in the negative-control stratum (A1.6).
14. Exactly one threshold constant appears in the result files, and it equals
    the value recomputed from calibration alone (A1.7).
15. Detector decision and latency vectors are identical across B2, B3, B4
    within a seed and cell (A1.8).
16. Every rate in the result files carries a numerator, a denominator and an
    interval (A1.7).

## What this changes in the recorded predictions

Original prediction P6 said the attack may be easy and a near-perfect result
would be a fact about the attack rather than a validation. A1.4 makes that
prediction testable rather than rhetorical: if precision stays high on the
hard-negative stratum, the detector is discriminating something real; if it
collapses there while staying high on ordinary clean posts, then `rep` was
measuring repetitiveness and the original design would have reported that as
detection.

Original prediction P2 said family S would be inert. A1.6 keeps the prediction
and adds the negative stratum that lets its inertness be distinguished from its
being unimplemented.

The cutoff is unchanged. If every gate, original and amended, has not passed by
2026-07-30 evening, nothing from W2D enters the 7/31 manuscript.
