# Vertical slice — fail-closed provisional-visibility protocol

Prototype for the topic-C paper (`../output/topic-c-paper-draft.md`,
`../output/topic-c-cbdcom-alignment.md`). Two execution models share the same
protocol logic and event schema:

| File | What | Backend | Clock | Status |
|---|---|---|---|---|
| `slice.py` + `test_invariants.py` | I1–I7 correctness harness | in-memory | logical (sim) | ✅ 20/20 |
| `pareto.py` | protocol-level freshness–exposure characterization (Fig A) | in-memory | logical (sim) | ✅ runs |
| `backend.py` | `VectorBackend` interface + `InMemoryBackend` | — | — | ✅ |
| `functional_slice.py` | real-time lifecycle (Gate 1) | in-memory **or Milvus** | wall-clock | ✅ 8/8 (in-mem + **Milvus Lite**, both modes) |
| `milvus_backend.py` | `MilvusBackend` (pymilvus) | Milvus | — | ✅ **validated on Milvus Lite**; full standalone = Gate 2/3 |

## ⚠️ Version matching (important)
The pymilvus client must match the Milvus server major.minor.
**Validated combo:** Python ≤3.12 + `pymilvus 2.4.x` + `setuptools<81` → matches
Milvus **v2.4.15** (the `docker-compose.yml` image) and Milvus Lite.
On **Python 3.14** pymilvus 2.4 can't build (no grpcio wheel) so pip gives
pymilvus 2.5.x, whose Lite (3.1.0) has a `function_score` search bug — use
Python 3.12 for the Lite path, or bump the compose image to `v2.5.x` for Docker.

## Validated (no-Docker) parts
```bash
# sim harness — any Python 3
python3 test_invariants.py                      # I1–I7  (20/20)
python3 pareto.py                               # Fig A characterization
python3 functional_slice.py --backend inmemory --mode postfilter   # 8/8
python3 functional_slice.py --backend inmemory --mode infilter     # 8/8

# Gate 1 on Milvus Lite (real Milvus API, NO Docker) — Python 3.12
python3.12 -m venv .venv312
./.venv312/bin/pip install "pymilvus>=2.4,<2.5" "setuptools<81"
./.venv312/bin/python functional_slice.py --backend milvus --uri /tmp/pv.db --mode postfilter
./.venv312/bin/python functional_slice.py --backend milvus --uri /tmp/pv.db --mode infilter
# -> 8/8 both modes; emits real delta_control / delta_query (~7–29ms on Lite)
```

## Gate 2/3 — full Milvus standalone (needs Docker)
Lite is single-process (no real concurrency/consistency), so the eligibility
mode comparison and overload experiments need the standalone server:
```bash
docker compose up -d                # etcd + minio + milvus v2.4.15 standalone
docker compose ps                   # wait for milvus-standalone (healthy), ~60–90s
# use the SAME matched client (Python 3.12 venv, pymilvus 2.4.x):
./.venv312/bin/python functional_slice.py --backend milvus --uri http://localhost:19530 --mode postfilter
docker compose down                 # (add -v && rm -rf volumes to wipe)
```
- **Gate 2** eligibility comparison (RQ-elig, §6.6): postfilter vs infilter —
  query P50/P95/P99, `delta_query`, top-k under-fill, refill cost, residual
  retrieval; sweep Milvus consistency level for the in-index mode.
- **Gate 3** overload validation (§6.7): verifier backlog vs the independent
  scheduler on real Milvus — `delta_query` stability, coupled-baseline blow-up.

Only after Gate 3 does §7 carry real-system throughput/latency numbers. The
in-memory backend is a correctness harness, **not** a Milvus performance baseline.
