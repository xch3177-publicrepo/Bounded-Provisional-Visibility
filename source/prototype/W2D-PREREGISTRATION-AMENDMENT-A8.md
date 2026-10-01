# W2D Preregistration — Amendment A8

2026-07-28, after detector and landing artifacts were frozen, but before a
formal runtime manifest or any promotion-path cell existed.

The first manifest build stopped before writing a manifest because the archived
legacy-regression record binds the pre-A6 SHA256 of `w2d_metrics.py`,
`27b775e40904354f46d3237420820f926667ffcaa91da1572dbd87cde839025d`,
while the production tree contains the A6 SHA256
`d379f132da17eb966f32cc04c6fe5cf31b697a097d56515a264031950f91fc3b`.
The A6 edit occurred before production data construction and before every D1
score.  It added a strict JSON parser over already snapshotted bytes so the
scorer cannot validate one file instance and then score another.  It did not
change a metric, legacy runtime file, legacy source artifact, or legacy
acceptance rule.

The archived regression artifact is not rewritten.  Its runtime-code SHA256
still equals the current legacy runtime SHA256.  The other ten files in its
legacy gate-code ledger still match byte for byte.

## Frozen compatibility gate

1. The archived and current legacy gate ledgers must have the identical
   filename set.
2. Every entry except `w2d_metrics.py` must match exactly.
3. The only accepted `w2d_metrics.py` transition is the exact old and A6
   SHA256 pair written above.  A third hash, a missing entry, or any other gate
   change fails.
4. The legacy runtime hash, six source-artifact hashes, independent
   recomputation, same-code projection, and exact eight rerun commands remain
   mandatory and unchanged.
5. Both the runner and the independent bundle verifier apply the same
   compatibility gate.  A regression test must reject drift in any unaffected
   file and either side of the one allowed pair.
6. This amendment enters the formal runtime fingerprint.  No detector score,
   threshold, metric, landing result, plan, or legacy artifact is regenerated
   by this repair.

Where an earlier check demanded equality of the whole legacy gate-code mapping,
A8 replaces it only with the exact compatibility rule above.
