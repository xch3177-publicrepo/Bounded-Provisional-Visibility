# Bounded Provisional Visibility: public reproducibility artifact

This artifact accompanies the accepted CBDCom 2026 paper. It contains experiment source, frozen observations, preregistrations and amendments, unsuccessful attempts, and historical runtime source snapshots. Public repository: [Bounded-Provisional-Visibility](https://github.com/xch3177-publicrepo/Bounded-Provisional-Visibility).

The export was prepared on 2026-10-01 from source revision `f7d2e90bf27da111bf52b9d86a2c8dd7b182cbcb`. Source revisions are identifiers, not bundled private Git history. The manuscript is maintained separately in the camera-ready package.

## What can be reproduced immediately

The offline checks below verify the public file inventory and independently recompute reported summaries from the released observations. They reproduce the D1 confusion matrix, attack-specific recall and clean false-refusal rates, latency quantiles, both oracle/detector protocol analyses, the five-run HNSW analysis, and the original W2/W2R/W2F acceptance checks. They do not execute a new long experiment.

```sh
python3 tools/verify_public.py
python3 -m venv /tmp/cbdcom-public-check
/tmp/cbdcom-public-check/bin/python -m pip install numpy
PYTHONDONTWRITEBYTECODE=1 /tmp/cbdcom-public-check/bin/python tools/verify_results.py
```

`verify_public.py` uses only the Python standard library. Python 3.12 is the preferred full-experiment interpreter. `verify_results.py` additionally needs NumPy. The pinned requirements in `source/prototype/requirements-w2d.txt` record the experiment environment; installing the latest NumPy above is sufficient for observation reanalysis, but is not a replacement for that environment when rerunning timed experiments.

The public checks fail on missing files, altered numerical leaves, mismatched authority links, malformed JSON, incorrect accepted-attempt selection, or disagreement between observations and the frozen summaries. The E3 check uses the historical verifier and runtime snapshot recorded by the accepted runs, including its runtime-byte fingerprint check.

## Layout and authority

| Location | Role |
|---|---|
| `PUBLIC_AUTHORITY.json` | Exact files supporting each result family; distinguishes the single-run standalone observation from inconclusive replication attempts. |
| `PUBLIC_PROVENANCE.json` | Source/public SHA256 pairs, numeric-projection hashes, input redactions, metadata transformations, excluded files, and historical source inventories. |
| `source/prototype/` | Frozen experimental code, configuration, preregistrations, amendments, and results. |
| `source/prototype/results/` | W2, W2R, W2F, eligibility, transfer and standalone observations, including superseded and unsuccessful records. |
| `source/prototype/results/w2d/` | Detector scores, labels without raw query material, threshold, oracle/detector traces, and analyses. |
| `source/prototype/results/w2d-e3/` | All tracked HNSW attempts, accepted runs, and the five-run summary. |
| `historical-runtime/` | Source-only snapshots for 31 recorded revisions; no original commit objects, author metadata, remotes or private history. |
| `data-provenance/` | Corpus-selection/freezing metadata; original passages and embedding arrays removed. |
| `source/paper/` | Original numeric/layout checkers and plot-generation source, preserved as historical tools. |
| `tools/` | Public verification and external-input reconstruction helpers. |

The accepted E3 attempts are RUN-01-ATTEMPT-02, RUN-02-ATTEMPT-05, RUN-03, RUN-04 and RUN-05. Earlier attempts remain visible and must not be silently promoted into the reported aggregate. The standalone S2 replication remained inconclusive; retaining all its attempts does not create a stronger performance claim.

## Sanitization and provenance

See `SANITIZATION.md`. Local home paths, personal workstation identifiers, private remotes and email strings were removed or made portable. No credentials from the original workspace, raw third-party corpus passages, query text, embedding arrays, model weights, model cache or private Git history are included.

Experimental observations, event timings, counts, labels, denominators, detector scores, thresholds, censoring states and aggregate values were preserved. Third-party query text and associated input embedding vectors in label/plan documents are explicitly marked as redacted. The original SHA256 is retained as a source identifier; it is not claimed to equal the sanitized file's SHA256.

The original `AUTHORITATIVE.json` and nested manifests are historical records of original bytes. Use `PUBLIC_AUTHORITY.json` and `PUBLIC_PROVENANCE.json` to locate and authenticate this transformed distribution. The original full `check_numbers.py` expects a manuscript, private historical Git objects and original input bytes; it is not an offline public-package entry point.

## Reconstructing external inputs

The raw 20 Newsgroups corpus and MiniLM weights must be obtained from their original sources under their terms. The W2D builder selects eight topics, preserves source-group separation and uses model revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` of `sentence-transformers/all-MiniLM-L6-v2`.

Install the archived dependencies in a separate Python 3.12 environment, then explicitly request input reconstruction in a new directory outside this artifact:

```sh
python3.12 -m venv /tmp/cbdcom-full-environment
/tmp/cbdcom-full-environment/bin/python -m pip install -r source/prototype/requirements.txt -r source/prototype/requirements-w2d.txt
/tmp/cbdcom-full-environment/bin/python tools/regenerate_inputs.py --work-dir /tmp/cbdcom-input-reconstruction --download-and-build
```

This helper copies the source into a fresh local Git provenance root, calls the original data builder, and compares reconstructed scorer inputs, embeddings, labels and label checksum against their historical hashes. It stops with failure if they differ. It does not alter the frozen public observations or rewrite historical hashes. This network/model reconstruction was **not run during public packaging**.

The older W2R `realtext_workload.py --build-cache` builder did not pin a model revision. Its source, corpus selection and recorded cache SHA256 are retained, but exact byte reproduction of that old cache cannot be promised from the public distribution. Rebuilding produces a new run and must record its own provenance.

## Full experiment reruns

The executable experiment code and original run sheets are included. A complete fresh rerun additionally requires reconstruction of the excluded inputs, Milvus Lite or Milvus Standalone as applicable, and a separately initialized run workspace. Wall-clock timings and concurrency measurements will vary by hardware and server state. Never overwrite the published observations.

The legacy runners are:

```sh
cd source/prototype
python poison_exposure.py --help
python standalone_runner.py --help
python w2d_runner.py --help
python w2d_e3_runner.py --help
```

Use `W2-PREREGISTRATION.md`, `W2R-PREREGISTRATION.md`, `W2F-PREREGISTRATION.md`, the W2D preregistration and amendments A1–A15, and `W2D-E3-RUN-SHEET.md` to preserve the actual experiment contract. The sequence is: reconstruct inputs; freeze calibration-only threshold; score the held-out set once; build protocol plan and landing evidence; execute E1/E2 with their separate manifests; analyze; execute five valid standalone HNSW process runs while retaining failures; aggregate exactly the accepted attempts.

The original formal W2D workflow intentionally binds original file bytes and historical Git revisions. Those gates cannot be truthfully reported as passing after sanitization and a new public Git root. A full public rerun therefore requires an explicit new provenance manifest/registration for the new run, or access to the original private provenance. This package provides tested observation reanalysis and source for fresh experimental replication; it does **not** claim a tested one-command regeneration of every historical run. No complete fresh runtime experiment or large model download was performed for this release.

The historical Docker Compose file uses upstream development defaults and binds service ports. For a local reproduction, restrict published ports to loopback and use throwaway credentials/volumes appropriate to the isolated experiment environment. The `minioadmin` strings in the historical compose file are public example defaults, not recovered private credentials. Localhost/loopback bind addresses in source remain because they are functional configuration, not personal machine identities.

## License and third-party material

See `NOTICE.md`. The source snapshot did not contain an explicit project license. This public availability does not invent an MIT/Apache or other reuse grant. The authors must specify any additional intended license. Dependencies, datasets, container images and model weights remain governed by their upstream terms and are not bundled.

## What was checked for this release

`VALIDATION.md` records the completed checks and the deliberate limits of those checks. `SHA256SUMS` covers every distributed file except itself. Use `shasum -a 256 -c SHA256SUMS` on macOS or `sha256sum -c SHA256SUMS` on Linux.
