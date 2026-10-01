# W2D Preregistration — Amendment A12

2026-07-28, after Amendments A9–A11 and before any complete
`W2D-protocol-result-v3` artifact existed.

This amendment records the final execution-integrity closures found while
implementing A11.  It changes no detector score, threshold, label, attack,
landing outcome, runtime assignment, query schedule, baseline, arm, seed,
backlog, observation horizon, denominator, or estimand.  D1 remains frozen and
is not rerun.

## Exact legacy-runtime compatibility

The archived A5 replacement gate remains evaluated against its archived
runtime digest
`f403c4e58c2526a70ada0dbaaa1d22c5cead42fb97b0e888ff70da9ad3a429e7`.
Current execution is accepted only at runtime digest
`3a599541311c27c524b0add29461534272b4e84b62a7cfa1becc8cfca1509fea`.

The only permitted per-file runtime transition is:

- `poison_exposure.py`:
  `7b08cd02814f6bc3e9f475f0234de29f7371365df5b524378c1b52bf7e4a4ff7`
  to
  `94a778d266b94a2aa1496792794fc131e0f2dc8137b95032f359e52c33a2df37`.

Every other file in the archived legacy-runtime digest must retain its exact
archived hash.  The change above retains raw retrieval and lifecycle evidence;
it does not change the frozen runtime schedule or legacy scientific
projection.  The eight legacy gates are rerun under the current code after
the archived projection is independently recomputed.

## Exact legacy-test compatibility

The A11 raw-evidence fields changed the expected query-observer record shape.
Consequently, the sole additional permitted legacy-gate transition is:

- `test_w2d_runtime.py`:
  `d53c2861b7774f4e4d6dafa7aa467764dd215d3519bffe1b30a42ab8c4a72fc6`
  to
  `05fd48acc80adc6411bf4861ba1a4d58c8a7ae5d52d1ece34e355f80c941d9b4`.

The new test requires `query_started_s`, `eligible_topk_ids`, and
`poisonfree_topk_ids` in addition to the prior fields.  It removes no
assertion and changes no pass threshold.

The frozen-production regression test must also compare the archived
candidate files against the archived runtime digest rather than recomputing
the current A11 digest:

- `test_verify_w2d_legacy.py`:
  `0a07113434630db2fafee4ef12c4f1a5e5792b882eab589173d1f6c555561ed2`
  to
  `5733c1d3b13cf53d5654fdcfa3abf4a9d751dfa54de6af2729d7a064de8ae888`.

This change replaces one dynamic current-runtime argument with the exact
archived digest and removes no test.  The separate A8
`w2d_metrics.py` transition remains exact; all other gate files must be
byte-identical to their archived hashes.

## Backend provenance

Every v3 result must contain one exact `runtime_backend` descriptor.

- E1 is `InMemoryBackend`, process-local exact brute force, 384 dimensions,
  cosine metric, and strong in-process consistency.
- E2 is `MilvusBackend` through Milvus Lite, effective `FLAT`, 384 dimensions,
  cosine metric, `Strong` consistency, `pymilvus==2.4.15`, and
  `milvus-lite==2.4.12`.

Mixed descriptors, coerced numeric types, a non-FLAT E2 index, an unknown
deployment, or an omitted package version fail.  A database URI is not
recorded as scientific provenance.

Each v3 result also contains one `runtime_backend_evidence.pre` receipt and one
`runtime_backend_evidence.post` receipt.  These are live descriptive receipts
captured outside the measured run window.  They are not remote attestation or
cryptographic proof that a remote service has a claimed identity; result and
manifest hashes bind the recorded bytes, not the truth of an unobservable
deployment claim.  The pre receipt does not inspect logical rows.  The E2 post
receipt performs one `Strong`-consistency query over the final logical rows and
must match the independently derived expected terminal-state summary.

For E2, the live server-version call must either return one canonical,
nonempty version string or fail with the typed gRPC status
`StatusCode.UNIMPLEMENTED`.  Only that typed capability absence is represented
as `{"status": "unimplemented", "version": null}`; every other exception or
malformed version fails closed.  Neither the backend descriptor nor either
receipt persists the database URI.  The URI is used only at capture time to
confirm the local Milvus Lite `.db` deployment.

## Formal byte and worktree closure

The manifest fingerprints `w2d_snapshot.py`, `w2d_equivalence.py`,
`w2d_backend_evidence.py`, A10, A11, and this amendment.  Thus the helper that
collects and validates the live backend receipts is part of the runtime
fingerprint.  Runner, verifier, and analyzer consume scientific values from one
captured byte instance.  The analyzer consumes the verified context returned
by the verifier rather than reopening the bundle.

Manifest creation requires an uncached clean-code worktree check.  Runtime
postflight also performs an uncached worktree query; it does not reuse the
legacy run-start cache.  Failure handling never deletes, unlinks, or replaces a
public output path.  An invocation-owned failed output remains in place for
explicit later quarantine; an external replacement is likewise preserved and
reported.  This no-delete rule supersedes A10's check-then-unlink rollback
because POSIX provides no atomic compare-and-unlink operation.

Python imports occur before the registry captures the corresponding source
files; in particular, `MilvusBackend` is imported before the runner captures
`milvus_backend.py`.  Therefore the formal operational boundary additionally
requires one committed, unchanged HEAD and no concurrent worktree or branch
mutation from process start through postflight.  The registry and uncached
worktree checks detect later persistent drift; they do not claim that
already-imported module objects were reconstructed from captured bytes.  The
eight historical path-only compatibility subprocesses retain this same
no-concurrent-mutation operational assumption; none supplies a scientific
measurement used by W2D.

## Backend-bound manifests and analysis semantics

E1 and E2 use separate canonical preregistered manifests and result filenames,
but they are bound by one canonical two-entry
`W2D-manifest-authority-v1` file.  That authority contains exactly one entry
for each execution tier and binds each canonical sibling manifest basename to
its exact SHA-256 digest.  A tier manifest records one exact execution tier,
its expected result basename, and its complete expected runtime-backend
descriptor.

Each result records both `manifest_sha256`, for the selected tier's captured
manifest, and `authority_sha256`, for the same two-entry authority.  Runner CLI
selection, authority entry, manifest identity, result basename, measured
result descriptor, and both recorded hashes must all agree; relabeling only a
manifest, result, or descriptor fails.

Production execution and production verification accept only the canonical
authority and canonical tier-manifest paths under `results/w2d`; the authority
must equal its committed `HEAD` bytes.  Portable authorities remain available
only for isolated unit fixtures whose artifact and fingerprint roots are also
outside the production tree.  Verification captures and checks both manifest
byte instances named by the authority, including the unselected tier, so a
missing or post-authority sibling manifest also fails.

Final analysis reports poison lifecycle, exposure, censoring, horizon state,
and \(E_u\) under both `overall` and frozen-landing `landed_only` populations.
It reports D1 classification errors separately from realized provider outcomes.
Only detector-arm `COMMITTED` false promotions or misquarantines contribute to
paired \(E_p\) or false-positive unavailability cost.  Paired latency reports
the counts for both committed, detector-only committed, oracle-only committed,
and neither committed; latency deltas use the both-committed subset.  Detector
F1 is a point estimate only and receives no binomial Wilson interval.

## A9 archived/current closure

The formal manifest contains distinct roles for both archived/current plan
files and both archived/current landing files.  Runner and independent
verifier apply the shared A9 equivalence gate before accepting a result.
Only the exact registered provenance differences are permitted.

## Latency reporting boundary

Detector-only service latency, queue wait, and queue-entry-to-commit latency
remain formal outputs because their timestamps and arithmetic closure are
verified.  Raw queue-depth counters remain in the result as diagnostics but
are not summarized as scientific measurements: the retained ledger does not
independently identify every same-timestamp scheduler ordering needed to
reconstruct those counters.

A12 enters the runtime fingerprint and supersedes the A11 manifest before the
first complete v3 E1 or E2 replay.
