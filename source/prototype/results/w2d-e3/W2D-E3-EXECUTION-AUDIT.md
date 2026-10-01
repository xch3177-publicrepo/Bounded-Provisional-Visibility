# W2D E3 execution audit

## Bottom line

The preregistered five-run experiment is **cross-run accepted**. Run numbers
1--5 each have one locally accepted execution from a distinct Milvus server
process. After a user-requested interruption was retained as a technical
failure, run 2 was restarted on the quiet host without changing the frozen
code, labels, thresholds, or validity gates and passed all twelve gates.

The independent validator accepted exactly those five executions and wrote
`W2D-E3-SUMMARY.json`. Its experimental unit is one accepted Milvus Standalone
process run; it reports only run-level median/min/max, does not pool requests,
and performs no significance test.

## Frozen provenance

- Preregistration commit: `e09b056`
- Label-binding amendment commit: `a4f31c6`
- Runtime implementation commit:
  `899c06ac3b20e80c22690391f10b19ca878e6ba3`
- Runtime fingerprint:
  `8032d0b15b445c1f95dd7bcf0901e971b8e0a6508a189978641d55c29c1093fc`
- Frozen detector-label SHA-256:
  `05d1363510a1790d690b05640c3a2830be4e62df56b8592182904ccb29019a9e`
- Server: Milvus Standalone v2.4.15, native arm64 under Colima
- Index: HNSW/COSINE, `M=16`, `efConstruction=200`, Strong consistency
- Static population: 100,000 vectors, comprising 1,752 frozen embeddings and
  98,248 deterministic normalized Gaussian pressure vectors
- Primary search setting: `ef=64`
- Query concurrency points: 1, 4, 8, and 16
- Admission load: 20 items/s
- Verifier concurrency: 4

The synthetic extension is a systems-pressure population. It is not a
100,000-document real-text corpus.

## Formal attempts

All ten retained artifacts use the same runtime commit and fingerprint. Every
locally rejected artifact failed only gate 11,
`sentinel_p95_movement`; its other eleven validity gates and final collection
cleanup passed. Attempt 04 of run 2 was interrupted at the user's request and
is retained as a technical failure rather than reclassified or deleted.

| Artifact | Run | Process start metric | Status | Maximum sentinel ratio | SHA-256 |
|---|---:|---:|---|---:|---|
| `W2D-E3-RUN-01.json` | 1 | 1785386469.59 | rejected | 1.97702 | `6d3407cdb6d4a9a0071b75449a93a9da79844c9f96b1c4356e58d60b6dbe1a3c` |
| `W2D-E3-RUN-01-ATTEMPT-02.json` | 1 | 1785389785.64 | accepted | 1.03129 | `41018781eea197a1c0f945822ac0143c6ad04e05356ef7811f705a0a1f7c3dbb` |
| `W2D-E3-RUN-02.json` | 2 | 1785391972.37 | rejected | 1.83448 | `79684fd84f1de1f2dc7a067c8f8ceb929217dfb4c1470448044c271aba5a542e` |
| `W2D-E3-RUN-02-ATTEMPT-02.json` | 2 | 1785394044.23 | rejected | 2.01778 | `adfd95889620d27dfc9a3a71a2f90176a5e49388b86404bc9cb678f4862a6fe9` |
| `W2D-E3-RUN-02-ATTEMPT-03.json` | 2 | 1785402999.58 | rejected | 1.81539 | `4a8d5fabec34cc6f3c34e0a362d1edf0bb2a1c3e1dd1e19ab192b5cfdebd6538` |
| `W2D-E3-RUN-02-ATTEMPT-04.json` | 2 | 1785428300.96 | technical failure (`KeyboardInterrupt`) | -- | `38b43edc2bf2b7571f56221cc743e4dae538863513167bddff4813653b5ec0d0` |
| `W2D-E3-RUN-02-ATTEMPT-05.json` | 2 | 1785429162.40 | accepted | 1.03282 | `dff5e5e84c6600c46492a7cf665896994c3626691ffc72f9f2b89cf18200f04e` |
| `W2D-E3-RUN-03.json` | 3 | 1785396283.03 | accepted | 1.02446 | `3054e4513308404789f8d4e370c3a2deac8d4721849f0509518993e52f2a14cd` |
| `W2D-E3-RUN-04.json` | 4 | 1785398314.65 | accepted | 1.02470 | `39155fc1e3c9392916b9e470e3748cbca7153ea477321435cfc4ca4d0658cbc5` |
| `W2D-E3-RUN-05.json` | 5 | 1785400948.32 | accepted | 1.05856 | `236ad8aae4d0374cd2d4a9ad44f18d1664f14228bc468f7aa7978ff9a0689433` |

The five accepted process-start metrics are distinct, so no two accepted runs
claim the same Milvus server process.

## Fail-closed cross-run validation

The earlier validator invocation correctly rejected run-2 attempt 03 and
created no summary. After the quiet-host execution, the same validator was
invoked with the accepted run-1 artifact, run-2 attempt 05, and accepted runs
3--5:

```text
./.venv312/bin/python verify_w2d_e3.py --runs \
  results/w2d-e3/W2D-E3-RUN-01-ATTEMPT-02.json \
  results/w2d-e3/W2D-E3-RUN-02-ATTEMPT-05.json \
  results/w2d-e3/W2D-E3-RUN-03.json \
  results/w2d-e3/W2D-E3-RUN-04.json \
  results/w2d-e3/W2D-E3-RUN-05.json \
  --summary-out results/w2d-e3/W2D-E3-SUMMARY.json

verified five independent formal W2D-E3 runs and wrote
results/w2d-e3/W2D-E3-SUMMARY.json
```

The summary status is `cross_run_accepted`; its SHA-256 is
`35cf224d8c16c9e7b7a9ab8d4d72877fb026d6ca03221636d519a1874fae8a6a`.

## Safe five-run descriptive facts

The following are run-level descriptive aggregates from the validator.

- Every accepted execution built and loaded a 100,000-row HNSW/COSINE index
  with zero pending index rows before measurement. After the dynamic panels,
  the index again reached `Finished`/`Loaded`, the live row count returned to
  exactly 100,000, and final collection deletion was confirmed.
- At the primary `ef=64`, Recall@5 had median `0.9979167` and range
  `0.9979167--0.9989583`; exact-top-5 match had median `0.9895833` and range
  `0.9895833--0.9947917`; landing concordance was `1.0` in every run.
- At `ef=256`, Recall@5 and exact-top-5 match were `1.0` in every run. At
  `ef=20`, Recall@5 had median `0.9729167` and range
  `0.9666667--0.975`.
- Each accepted execution completed all 16 baseline-by-concurrency cells.
  Across baselines, run-level median completed-query rates were
  `3.27--3.33` queries/s at concurrency 1 and `48.0--51.27` queries/s at
  concurrency 16. Every cell admitted exactly 20 items/s. These are controlled
  observations, not production-capacity estimates.
- Across the five accepted executions there were zero measured query errors,
  zero admission failures, zero cells with asynchronous errors, and zero
  cleanup-failure cells.
- The frozen 512-item detector shadow population contained 255 correct
  promotions, 129 correct refusals, 63 false promotions, and 65 false
  refusals. Thus the detector correctly refused 129 of 192 poisoned items
  (`67.19%`) and correctly promoted 255 of 320 clean items (`79.69%`).
  These counts are the same frozen detector population in every cell; they are
  not independent detector-quality replicates.
- Every 300-admission measured cell realized 156 correct promotions, 77 correct
  refusals, 38 false promotions, and 29 false refusals. Item-level lifecycle
  records show the mechanism-level consequence: all 38 false promotions were
  visible at the window end under every baseline; the 29 false refusals were
  never visible under verify-before-visible, and were briefly visible but absent
  at the window end under the provisional baselines.

## What this experiment does and does not answer

It directly weakens the claim that the evidence uses only FLAT retrieval,
contains no concurrency/throughput exercise, or models the verifier as a
perfect oracle. It supplies real Milvus HNSW retrieval, controlled concurrent
query load, measured detector FP/FN outcomes, measured verifier service-time
replay, and live exposure accounting.

It does **not** establish production capacity, distributed scaling, high
availability, or fault tolerance: Milvus remains single-node Standalone. The
detector's frozen decisions and measured service times are replayed in the
lifecycle harness; the detector model is not computed online and therefore its
CPU/GPU contention is not part of the throughput measurement. The synthetic
background vectors do not establish real-corpus or unknown-attack
generalization.

## Next defensible action

Treat `W2D-E3-SUMMARY.json` as the sole cross-run authority. Cite only its
run-level median/min/max and retain every rejected or interrupted artifact as
provenance. Any online-detector, distributed, production-capacity, or
unknown-attack claim requires a separately preregistered experiment.
