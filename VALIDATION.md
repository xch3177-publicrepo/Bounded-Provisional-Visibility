# Release validation, 2026-10-01

The public artifact was checked with Python 3.12.14 and NumPy 2.3.5 for observation reanalysis. No Milvus server, external corpus download or model download was needed for these checks.

| Check performed | Outcome |
|---|---|
| Source-to-public export comparison | 997 exported source files; 31 historical runtime snapshots. Source SHA/public SHA pairs are recorded. |
| Strict public JSON and numerical preservation | 83 JSON documents parsed; 2,395,213 numeric/Boolean/null leaves preserved outside explicitly enumerated third-party input redactions. |
| D1 metrics recomputed from held-out item scores and labels | PASS, including TP/FP/TN/FN = 129/65/255/63, 512 unique source groups, attack strata, clean false refusals and latency quantiles. |
| Source-family controls | PASS, recomputed from released control scores and labels. |
| E1 and E2 analyses recomputed from full recorded cells | PASS, detector quality, landing, exposure, availability, retrieval, latency and paired detector-minus-oracle contrasts exactly matched the corresponding frozen analysis objects. |
| E3 accepted attempts independently verified and aggregated | PASS, exact accepted attempt files, runtime fingerprint, five run identities, panel A, panel B, transfer condition and aggregation semantics. |
| W2 in-memory and Milvus acceptance verifier | PASS on both frozen files. |
| W2R in-memory and Milvus acceptance verifier | PASS on both frozen files. |
| W2F false-promotion acceptance verifier | PASS against the frozen W2 baseline. |
| Standalone S2 single-run observations | PASS, observable P50/P95 outward-rounded ranges 288–332/380–431 ms; no Strong/Session stale read in the retained cells. All five inconclusive replication attempts remain in the package and are not pooled. |
| `test_w2.py` offline regression checks | 57/57 PASS. |
| `test_invariants.py` offline state-machine checks | 20/20 PASS. |
| External-input reconstruction helper | CLI and source inspected; `--help` exercised. Network/model reconstruction not run. |

The privacy audit checked for private home paths, account/workstation identifiers, email strings, private remotes and credential patterns. Original passage/query text and embedding arrays were removed according to the per-leaf redaction record. Public localhost configuration and upstream example development defaults are documented in `README.md`.

These are file-integrity, source-linkage, numerical reanalysis and offline fixture checks. They do not prove new detector generality, new production capacity or a new runtime replication. They do not certify the private historical Git lineage as reconstructed in the public repository. The paper's interpretation boundaries remain those of the frozen experiments.

Not executed for this release: complete W2/W2R/W2F timing grids, a fresh detector score pass, E1/E2 replay grids, standalone S2 replication, the five HNSW runs, model/corpus reacquisition, or a complete fresh dependency installation. The public release is an independently checkable results-and-source artifact with explicit external-input requirements.
