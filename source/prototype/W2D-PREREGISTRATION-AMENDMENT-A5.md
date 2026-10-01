# W2D Preregistration — Amendment A5

2026-07-28, after the W2D implementation tests and legacy W2/W2R replay, but
before the production W2D data freeze, detector calibration, threshold, test
score, attack-landing result, or protocol result exists.  No production D1
outcome has been observed.  A5 changes only the legacy-oracle regression gate.
It does not change D1, either attack, any allocation, threshold rule, protocol
cell, estimand, or acceptance rule for the new W2D data. The v2 protocol-result
schema only makes the raw E_u and queue fields already required by A2.7--A2.8
mandatory instead of dropping them during canonicalization.

## Why the literal A2.6 cross-execution gate is invalid

A2.6 required every deterministic legacy cell field and per-query hit vector
to equal the archived W2/W2R execution.  The full replay showed that this
classification was wrong.  `T_p=0.3 s` is exactly the configured legacy
verifier service time.  A query tick and the verification commit race at that
boundary, so a returned-id vector partitioned "before" versus "after" the
commit is a wall-clock observation, not a deterministic function of the
inputs.

This is not inferred only from old-code versus new-code disagreement.  Two
complete executions of the *same* clean commit `5fa7058`, with no intervening
file change, produced:

```text
W2 run A  sha256 2122e80d66891276ddcb31369c2e71ae4de5fc6d9bf2af89306f6ef666f7fca8
W2 run B  sha256 754b613a7abce666c50b3f79d7ff18cdd416cb99c00319c61d76bf58729d46fc
```

Both independently passed the full W2 admissibility verifier.  Nevertheless,
for `B4/normal/T_p=0.3/seed=4`, the crafted-query poisoned retrieval count was
one in run A and three in run B.  Per-query vectors and the corresponding
phase-split displacement moved with it.  The paper-level aggregates were
unchanged.  A gate that rejects identical code according to scheduler timing
cannot distinguish a code regression from a second execution and is not a
valid regression test.

The full source ledger is frozen here, so the word "authoritative" cannot be
reassigned by passing a different file to the verifier:

```text
results/AUTHORITATIVE.json
  fdca94b312900f01024a94491b4d100328eced2d2681566ccd956356111422db
authoritative W2 in-memory
  1f5efd3390039bf930ceffac7464d06e0a178ca9e46bb0a3b7c961abfff7a24d
authoritative W2R in-memory
  58b9a8dc1fe3124621733e97c2e3ad0e253f9bdce513c70d8139f432b0d81e8d
candidate W2 run A
  2122e80d66891276ddcb31369c2e71ae4de5fc6d9bf2af89306f6ef666f7fca8
candidate W2R
  be980da694e9d09a59d8e25d162bc85828465906a112fe70fefe5c6b54de8c75
candidate W2 same-code run B
  754b613a7abce666c50b3f79d7ff18cdd416cb99c00319c61d76bf58729d46fc
canonical exhaustive run-A/run-B difference projection
  88b38fbac32b68c1ae749f33560d8a02b5eaf6f409bf76948684c5273f324d22
```

A5 records this failed assumption rather than relabelling either execution as
an exact reproduction.  The original literal cross-execution equality gate is
therefore **inconclusive**, not passed.

## Replacement legacy-oracle acceptance gate

The following checks are frozen now and are implemented by
`verify_w2d_legacy.py`. The expanded authority-bound artifact uses schema
`W2D-legacy-regression-v2`; the formal manifest and protocol result use their
respective v2 schemas because the artifact roles and raw E_u/queue fields were
expanded before any production W2D artifact existed.

1. Each new W2 and W2R run is a full, clean-tree, five-seed run and independently
   passes its existing baseline, state-machine, containment, censoring,
   isolation, and admissibility invariants.
2. W2R retains the frozen corpus, model, dataset digest, geometry, and protocol
   configuration. The authority ledger and all five input runs must have the
   exact SHA256 values above; role aliases or byte substitutions fail.
3. The off-path B1 control has exactly the same cell set, retrieval counts, and
   per-query hit vectors as the corresponding authoritative artifact.  B1 has
   no verifier or deadline and therefore does not use the disputed timing
   boundary.
4. Every cell has no foreign candidate, under-fill, residual row, live control
   task/timer, or background error.  B2 never exposes unvetted poison; B1 never
   claims containment; and every B4 cell has six defined finite exposure
   observations and respects its deadline plus the already frozen slack. The
   exact 70-cell grid, query-event denominator, configuration digest, zero
   false-promotion setting, and state-machine projection are fail-closed.
5. The default-oracle path is checked on fixed inputs, rather than by comparing
   two wall-clock executions: for B2/B3/B4 it rejects labelled poison, accepts
   clean content, preserves the legacy false-promotion override, and uses the
   same configured service time.  The existing `test_w2d_runtime.py`,
   `test_w2.py`, and `test_invariants.py` are part of this gate.
6. Scheduling-dependent phase partitions, first-visibility times, integrated
   queue times, and finite-duration summaries are reported exhaustively but are
   not asserted equal across executions.  No numeric tolerance is introduced
   or fitted. All other identity, configuration, data, geometry, schema, and
   structural fields remain pinned.
7. The exhaustive same-code difference projection has the exact canonical
   digest above. The formal verifier does not trust recorded PASS fields: it
   reruns the exact eight legacy gates with the current interpreter and fails
   if any source, runtime dependency, or gate file changes between the
   pre-execution and post-execution hash snapshots.

The repeated legacy regression in this amendment is E1/in-memory only. It does
not claim a new Milvus/E2 replay; the already archived E2 evidence remains a
separate artifact and cannot be described as part of this A5 gate.

The same-code counterexample remains in the evidence bundle.  It is not an
independent replication and is not used as a W2D observation.

## Interpretation

Passing the replacement gate supports the narrow statement that adding the
custom-verifier seam did not change the legacy default-oracle mechanism.  It
does not support byte-for-byte reproducibility of asynchronous wall-clock
trajectories, and no W2D report may claim that it does.
