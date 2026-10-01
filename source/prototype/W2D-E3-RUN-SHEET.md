# W2D-E3 formal run sheet

This is the operator procedure for the five independent Milvus Standalone runs
defined by `W2D-E3-PREREGISTRATION.md` (preregistered at commit `e09b056`) and
its evaluator-label correction
`W2D-E3-PREREGISTRATION-AMENDMENT-A1.md` (commit `a4f31c6`). These two files,
not this checklist, control every scientific definition. Do not use this sheet
to change a grid point, threshold, workload, timeout, or acceptance rule after
seeing an outcome.

## 0. Fixed operating boundary

- Work from `prototype/` on one committed runtime revision containing
  `w2d_e3_runner.py`, `verify_w2d_e3.py`, their tests, and this sheet.
- Treat both `W2D-E3-PREREGISTRATION.md` and
  `W2D-E3-PREREGISTRATION-AMENDMENT-A1.md` as read-only control files. The
  runtime revision must descend from `a4f31c6`.
- All five accepted files must come from that same revision and runtime
  fingerprint. Do not edit tracked files while the sequence is in progress.
- The existing Colima profile is native `aarch64`/`vz`, 4 vCPU, 8 GiB RAM, with
  a 100 GiB data disk. The host is an M1 Pro with 16 GiB RAM. Use that profile
  as-is; do not allocate the host's full 16 GiB to Colima.
- The matched client environment is `.venv312` with Python 3.12,
  `pymilvus==2.4.15`, and `setuptools==80.10.2`. The Compose deployment pins
  `milvusdb/milvus:v2.4.15`.
- Do not run `docker compose down -v`, delete `volumes/`, prune Docker, or
  recreate the Colima profile during the five-run sequence.
- Do not inspect or summarize favorable/unfavorable performance values between
  runs. Operational gate failures may be diagnosed, but the frozen matrix is
  not changed in response.

A1 binds `results/w2d/W2D-labels.json` at SHA256
`05d1363510a1790d690b05640c3a2830be4e62df56b8592182904ccb29019a9e`.
That same digest must appear as `artifact_hashes.labels` in the frozen
`W2D-E1-INMEMORY.json` parent. For the 512 keys in
`W2D-test-scores.json`, the runner may read only the labels document's
`poison` evaluator field, and only to classify an already-completed lifecycle
event as TP/FP/TN/FN. Labels must never reach the detector-decision provider or
change a frozen decision, service time, event selection/order, cell, or
threshold.

The 2026-07-29 pre-run inspection saw load averages above 8 and at least four
parallel `node_repl` processes. That is not a suitable measurement
environment. Before the first run, stop or let finish parallel Codex agents,
tests, builds, model inference, LaTeX compilation, sync jobs, and other
CPU/disk-heavy work. Keep the Mac connected to power, prevent sleep, and leave
it otherwise idle. This is an operational precaution; the preregistered
sentinel remains the formal environmental gate.

Budget 60--90 minutes per independent run plus 2--4 minutes for each restart
and health check. Reserve one uninterrupted 6--8 hour window for all five runs.
These are scheduling allowances, not reportable performance measurements.

## 1. One-time preflight

Run these read-only checks from `prototype/`:

```bash
git rev-parse HEAD
git merge-base --is-ancestor a4f31c6 HEAD
git diff --quiet
git diff --cached --quiet
shasum -a 256 results/w2d/W2D-labels.json
jq -e \
  '.artifact_hashes.labels == "05d1363510a1790d690b05640c3a2830be4e62df56b8592182904ccb29019a9e"' \
  results/w2d/W2D-E1-INMEMORY.json
colima list
docker context show
docker compose config --quiet
./.venv312/bin/python -c \
  'import pymilvus,sys; print(sys.version); print(pymilvus.__version__)'
./.venv312/bin/python -m unittest test_w2d_e3
./.venv312/bin/python w2d_e3_runner.py --help
./.venv312/bin/python verify_w2d_e3.py --help
uptime
```

The ancestry check, the two `git diff` commands, and the `jq` binding check must
exit zero. The `shasum` value must be exactly
`05d1363510a1790d690b05640c3a2830be4e62df56b8592182904ccb29019a9e`.
The runner/verifier tests must also confirm that the 512 unique test-score keys
all have Boolean `poison` labels and that no other evaluator field enters the
decision or schedule path. Untracked archival result files are not tracked-tree
mutations, but they must not collide with any output name below. Record the
full commit printed by `git rev-parse`; do not continue if it changes before
the final verification.

If Colima is stopped, start the existing profile without changing its
resources:

```bash
colima start
docker context use colima
```

Bring up the pinned stack and wait for the initial health check:

```bash
docker compose up -d etcd minio standalone
docker compose ps
docker inspect --format '{{.State.Health.Status}}' milvus-standalone
curl -fsS http://127.0.0.1:9091/healthz
```

Proceed only when `milvus-standalone` is `healthy`, `/healthz` succeeds, and
ports 19530 and 9091 belong to this Compose stack. On a cold start, health can
take 60--90 seconds. A timeout or unhealthy state is an operational failure:
save the terminal log and diagnose it; do not begin a formal run.

Create the output directory, then confirm that none of the five canonical
output files or the summary already exists:

```bash
mkdir -p results/w2d-e3
test ! -e results/w2d-e3/W2D-E3-RUN-01.json
test ! -e results/w2d-e3/W2D-E3-RUN-02.json
test ! -e results/w2d-e3/W2D-E3-RUN-03.json
test ! -e results/w2d-e3/W2D-E3-RUN-04.json
test ! -e results/w2d-e3/W2D-E3-RUN-05.json
test ! -e results/w2d-e3/W2D-E3-SUMMARY.json
```

The runner and verifier must use exclusive creation. Never add a force,
overwrite, resume, skip-cell, or append option.

## 2. Procedure for each independent run

Perform this complete sequence for run numbers 1 through 5. Run 1 also receives
an explicit Standalone restart; an already-running process is not accepted as
run 1's fresh process.

### 2.1 Restart only Milvus Standalone

```bash
docker compose restart standalone
docker compose ps
```

Do not restart etcd or MinIO between formal runs. Wait until both checks below
pass:

```bash
docker inspect --format '{{.State.Health.Status}}' milvus-standalone
curl -fsS http://127.0.0.1:9091/healthz
```

If the container does not become healthy within three minutes, retain the
terminal log, mark that attempt as an operational failure, and stop. Do not run
the formal command against a starting or unhealthy process.

### 2.2 Capture and distinguish the process start

After health succeeds, obtain exactly one Prometheus process-start value:

```bash
curl -fsS http://127.0.0.1:9091/metrics |
  awk '$1 == "process_start_time_seconds" {print $2}'
```

The value must be present, finite, and different from the value printed after
every preceding formal restart. If it is absent or repeats, the restart did not
establish the preregistered independent process: stop before running the
matrix. Do not manufacture a replacement value from container creation time.

The manual check is only an early guard. The formal runner must fetch the same
port-9091 metric into its exclusively-created result, and the final verifier
must reject missing or repeated values across the five files.

### 2.3 Run the frozen matrix

Use the corresponding command exactly once. `caffeinate` prevents host sleep;
it does not change the experimental grid.

Run 1:

```bash
caffeinate -dimsu ./.venv312/bin/python w2d_e3_runner.py \
  --run-number 1 \
  --uri http://127.0.0.1:19530 \
  --metrics-url http://127.0.0.1:9091/metrics \
  --out results/w2d-e3/W2D-E3-RUN-01.json
```

Run 2:

```bash
caffeinate -dimsu ./.venv312/bin/python w2d_e3_runner.py \
  --run-number 2 \
  --uri http://127.0.0.1:19530 \
  --metrics-url http://127.0.0.1:9091/metrics \
  --out results/w2d-e3/W2D-E3-RUN-02.json
```

Run 3:

```bash
caffeinate -dimsu ./.venv312/bin/python w2d_e3_runner.py \
  --run-number 3 \
  --uri http://127.0.0.1:19530 \
  --metrics-url http://127.0.0.1:9091/metrics \
  --out results/w2d-e3/W2D-E3-RUN-03.json
```

Run 4:

```bash
caffeinate -dimsu ./.venv312/bin/python w2d_e3_runner.py \
  --run-number 4 \
  --uri http://127.0.0.1:19530 \
  --metrics-url http://127.0.0.1:9091/metrics \
  --out results/w2d-e3/W2D-E3-RUN-04.json
```

Run 5:

```bash
caffeinate -dimsu ./.venv312/bin/python w2d_e3_runner.py \
  --run-number 5 \
  --uri http://127.0.0.1:19530 \
  --metrics-url http://127.0.0.1:9091/metrics \
  --out results/w2d-e3/W2D-E3-RUN-05.json
```

After each zero exit, confirm only that the expected file exists and is
non-empty. Do not hand-edit it:

```bash
test -s results/w2d-e3/W2D-E3-RUN-0N.json
git diff --quiet
git diff --cached --quiet
```

Replace `N` with the current run number. Also confirm the container remains
healthy. Then return to step 2.1 and restart Standalone before the next run.

## 3. Failure and technical-rerun rule

Every failure is evidence about execution and must remain recoverable.

- Never delete, truncate, overwrite, or hand-repair a failed JSON or terminal
  log.
- If the runner exits nonzero without creating its requested output, preserve
  the complete terminal transcript and use a new attempt-specific output name
  for any preregistration-permitted technical rerun.
- If it creates a rejected output, leave that byte instance at its original
  path. A retry uses a new filename such as
  `W2D-E3-RUN-03-ATTEMPT-02.json`; it never reuses the failed pathname.
- Before a retry, fix only the explicit operational or gate defect, commit any
  code repair, and restart the whole five-run sequence if that repair changes
  the runtime commit or fingerprint. A cell-level resume is forbidden.
- Low recall, low throughput, high error rate, a null-crossing range, or B4
  being slower than B1 are outcomes, not technical failures and not reasons to
  rerun.

If an accepted technical retry has an attempt-specific filename, pass that
exact accepted file to the final verifier in place of the corresponding
canonical argument. Do not copy or rename it merely to make the happy-path
command look uniform.

## 4. Verification first, summary second

Do not create or interpret an aggregate summary until five independent result
files exist and the fifth run has finished. Recheck the tracked tree and the
frozen runtime commit, then invoke the independent verifier once over the five
accepted byte instances:

```bash
git diff --quiet
git diff --cached --quiet
./.venv312/bin/python verify_w2d_e3.py \
  --runs \
    results/w2d-e3/W2D-E3-RUN-01.json \
    results/w2d-e3/W2D-E3-RUN-02.json \
    results/w2d-e3/W2D-E3-RUN-03.json \
    results/w2d-e3/W2D-E3-RUN-04.json \
    results/w2d-e3/W2D-E3-RUN-05.json \
  --summary-out results/w2d-e3/W2D-E3-SUMMARY.json
```

`verify_w2d_e3.py` must verify all per-run and cross-run gates, including the
five distinct `process_start_time_seconds` values, before exclusively creating
`W2D-E3-SUMMARY.json`. There is no separate analyzer. A verifier failure must
not leave a summary that can be mistaken for accepted output; retain any
invocation-owned failure artifact under an explicit rejected-attempt name.

Only after a zero verifier exit:

```bash
test -s results/w2d-e3/W2D-E3-SUMMARY.json
git diff --quiet
git diff --cached --quiet
```

may the summary be inspected and used for reporting. Archive all five accepted
run files, every rejected attempt, the final summary, terminal logs, the full
runtime commit, and the Compose image/version information together.
