"""
S1/S2/S5 standalone experiments -- the frozen §7.5 contract, evidence tier E3.

These are the measurements that Milvus Lite structurally CANNOT produce: real
gRPC concurrency (S1), consistency-level propagation delay (S2), and ANN
approximation error (S5). Everything emits through analysis.make_record with
evidence_level="milvus_standalone", so §7 tables are regenerated from JSON and
no number is hand-copied.

Requires a reachable Milvus standalone (docker compose up -d) and a MATCHED
client (Python 3.12 + pymilvus 2.4.x for the v2.4.15 image).

  ./.venv312/bin/python standalone_experiments.py --exp all
  ./.venv312/bin/python standalone_experiments.py --exp S2 --quick
"""

import argparse
import asyncio
import json
import os
import random
import time

import analysis
import gate2

HERE = os.path.dirname(os.path.abspath(__file__))
DIM = 32
K = 10

# Fixed ANN configuration for S1/S2 (S5 is the sweep). Reported, not tuned.
HNSW = {"index_type": "HNSW",
        "index_params": {"M": 16, "efConstruction": 200},
        "search_params": {"ef": 64}}
FLAT = {"index_type": "FLAT", "index_params": {}, "search_params": None}

# Execution context. --smoke runs the SAME code against Milvus Lite to prove the
# paths execute; Lite has no HNSW, no real consistency, no concurrency, so its
# output is tagged E2 and prefixed "smoke/" and must NEVER be read as a §7.5
# result. The real run leaves this untouched.
CTX = {"index": HNSW, "evidence": "milvus_standalone", "prefix": "standalone"}


def pctl(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))]


def stats(xs):
    return {"p50": pctl(xs, 0.50), "p95": pctl(xs, 0.95), "p99": pctl(xs, 0.99),
            "max": max(xs) if xs else None, "n": len(xs)}


def mk(uri, collection, consistency="Bounded", **idx):
    from milvus_backend import MilvusBackend
    return MilvusBackend(dim=DIM, uri=uri, collection=collection,
                         consistency_level=consistency, **idx)


def preflight(uri):
    try:
        from pymilvus import MilvusClient
    except ImportError:
        raise SystemExit("pymilvus not installed. Use the matched venv:\n"
                         "  python3.12 -m venv .venv312 && "
                         "./.venv312/bin/pip install 'pymilvus>=2.4,<2.5' 'setuptools<81'")
    try:
        MilvusClient(uri=uri).list_collections()
    except Exception as e:
        raise SystemExit(
            f"Cannot reach Milvus at {uri}: {e}\n"
            "Start it first:  docker compose up -d && docker compose ps\n"
            "(wait for milvus-standalone to report (healthy), ~60-90s)")


# --------------------------------------------------------------------------
# S1: B1-B4 x concurrency -> ingestion throughput, query P95/P99   [Fig B(a)]
# --------------------------------------------------------------------------
BASELINES = {
    "B1": {"desc": "immediate admission, no verification", "verify": False, "sync": False, "deadline": False},
    "B2": {"desc": "synchronous verify-before-visible", "verify": True, "sync": True, "deadline": False},
    "B3": {"desc": "async visible, no deadline", "verify": True, "sync": False, "deadline": False},
    "B4": {"desc": "fail-closed provisional visibility", "verify": True, "sync": False, "deadline": True},
}


async def s1_one(uri, name, conc, N, verify_cost, Tp, query_gap,
                 n_queriers=3, min_samples=150, topup_budget=5.0):
    """One (baseline, concurrency) cell: ingest N items under a concurrent query
    stream. Ingestion is per-item (insert(), not insert_many) because the
    protocol admits, deadlines, and verifies per item."""
    bl = BASELINES[name]
    b = mk(uri, f"s1_{name}_{conc}", "Bounded", **CTX["index"])
    rng = random.Random(0)
    vecs = [[rng.gauss(0, 1) for _ in range(DIM)] for _ in range(N)]
    state, samples, tasks = {}, [], []          # samples: (completion_time, latency)
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def hide(i):                                     # deadline scheduler (B4 only)
        if state.get(i) == "PROVISIONAL":
            state[i] = "HIDDEN"

    async def verify(i):
        await asyncio.sleep(verify_cost)
        if state.get(i) == "PROVISIONAL":
            state[i] = "TRUSTED"

    async def querier():                             # concurrent retrieval (W1)
        while not stop.is_set():
            q = vecs[rng.randrange(N)]
            t0 = time.monotonic()
            try:
                raw = await asyncio.to_thread(b.search, q, 2 * K, False)
            except asyncio.CancelledError:
                return
            except Exception:
                return
            # post-filter eligibility against the control store (§5)
            [i for i in raw if state.get(i) in ("PROVISIONAL", "TRUSTED")][:K]
            t1 = time.monotonic()
            samples.append((t1, t1 - t0))
            if query_gap:
                await asyncio.sleep(query_gap)

    queue = asyncio.Queue()
    for i in range(N):
        queue.put_nowait(i)

    async def worker():
        while True:
            try:
                i = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if bl["sync"]:                            # B2: verify BEFORE visible
                await asyncio.sleep(verify_cost)
                state[i] = "TRUSTED"
                await asyncio.to_thread(b.insert, i, vecs[i], True)
            else:
                state[i] = "PROVISIONAL"
                await asyncio.to_thread(b.insert, i, vecs[i], True)
                if bl["verify"]:
                    tasks.append(asyncio.create_task(verify(i)))
                if bl["deadline"]:
                    loop.call_later(Tp, hide, i)

    qts = [asyncio.create_task(querier()) for _ in range(n_queriers)]
    t0 = time.monotonic()
    await asyncio.gather(*[worker() for _ in range(conc)])
    t_end = time.monotonic()
    elapsed = t_end - t0

    # A tail percentile computed from a handful of samples is noise. If ingestion
    # finished before the query stream had min_samples, keep querying briefly and
    # say so, rather than reporting a P99 over ~8 points.
    topped_up = False
    if sum(1 for t, _ in samples if t <= t_end) < min_samples:
        topped_up = True
        deadline = time.monotonic() + topup_budget
        while (len(samples) < min_samples and time.monotonic() < deadline
               and not stop.is_set()):
            await asyncio.sleep(0.02)
    stop.set()
    for t in qts + tasks:
        t.cancel()
    await asyncio.gather(*qts, *tasks, return_exceptions=True)

    during = [l for t, l in samples if t <= t_end]
    used = samples_used = during if not topped_up else [l for _, l in samples]
    ql = stats(used)
    return {"throughput": N / elapsed, "elapsed_s": elapsed,
            "query_p50": ql["p50"], "query_p95": ql["p95"], "query_p99": ql["p99"],
            "n_queries": ql["n"], "n_queries_during_ingest": len(during),
            "query_window": "post_ingest_topup" if topped_up else "during_ingest"}


def _smoke():
    return CTX["prefix"] == "smoke"


async def run_s1(uri, quick):
    N = 120 if _smoke() else (400 if quick else 1500)
    concs = [4] if (quick or _smoke()) else [4, 16]
    out = []
    for conc in concs:
        for name in BASELINES:
            print(f"  S1 {name} conc={conc} N={N} ...", flush=True)
            m = await s1_one(uri, name, conc, N, verify_cost=0.05, Tp=1.0, query_gap=0.0)
            out.append(analysis.make_record(
                f"{CTX['prefix']}/S1/{name}/conc{conc}", "milvus",
                CTX["index"]["index_type"], CTX["evidence"],
                {"baseline": name, "desc": BASELINES[name]["desc"], "concurrency": conc,
                 "N": N, "k": K, "dim": DIM, "consistency": "Bounded",
                 "verify_cost_s": 0.05, "Tp_s": 1.0, "over_fetch": 2,
                 "query_load": "closed-loop, 3 queriers, no think time (saturating)",
                 **CTX["index"]}, m))
            print(f"     -> {m['throughput']:.0f} items/s, "
                  f"query P99 {1000*m['query_p99']:.1f} ms "
                  f"({m['n_queries']} queries, {m['query_window']})")
    return out


# --------------------------------------------------------------------------
# S2: in-index x consistency x backlog -> delta_hide P95/P99      [Fig B(b)]
# --------------------------------------------------------------------------
async def s2_one(uri, consistency, backlog, n_items, Tp, verify_cost, probe_int, probe_budget):
    """In-index (Milvus scalar) enforcement. Each item gets its own deadline; a
    sentinel probe (the item's own vector, so it is its top-1 when eligible)
    observes t_hide_effective. delta_hide decomposes into scheduler firing,
    control-store commit, and query-path propagation."""
    b = mk(uri, f"s2_{consistency}_{'bk' if backlog else 'nb'}", consistency, **CTX["index"])
    rng = random.Random(7)
    sem = asyncio.Semaphore(2) if backlog else None
    d_hide, d_sched, d_state, d_prop, censored = [], [], [], [], 0
    vtasks = []

    async def verifier():                    # backlogged, must not affect the timer (I6)
        if sem is not None:
            async with sem:
                await asyncio.sleep(verify_cost)
        else:
            await asyncio.sleep(verify_cost)

    async def one(i):
        nonlocal censored
        vec = [rng.gauss(0, 1) for _ in range(DIM)]
        await asyncio.to_thread(b.insert, i, vec, True)
        t_visible = time.monotonic()
        deadline = t_visible + Tp
        vtasks.append(asyncio.create_task(verifier()))
        await asyncio.sleep(max(0.0, deadline - time.monotonic()))   # independent timer
        t_sched = time.monotonic()
        await asyncio.to_thread(b.set_visible, i, False)             # control commit
        t_commit = time.monotonic()
        t_eff, end = None, time.monotonic() + probe_budget
        while time.monotonic() < end:
            try:
                hits = await asyncio.to_thread(b.search, vec, 1, True)
            except Exception:
                break
            if i not in hits:
                t_eff = time.monotonic()
                break
            await asyncio.sleep(probe_int)
        if t_eff is None:
            censored += 1
            return
        d_sched.append(t_sched - deadline)
        d_state.append(t_commit - t_sched)
        d_prop.append(t_eff - t_commit)
        d_hide.append(t_eff - deadline)

    items = []
    for i in range(n_items):
        items.append(asyncio.create_task(one(i)))
        await asyncio.sleep(0.10)            # stagger deadlines
    await asyncio.gather(*items, return_exceptions=True)
    for t in vtasks:
        t.cancel()
    await asyncio.gather(*vtasks, return_exceptions=True)

    h, s, st, p = stats(d_hide), stats(d_sched), stats(d_state), stats(d_prop)
    return {"delta_hide_p95": h["p95"], "delta_hide_p99": h["p99"], "delta_hide_max": h["max"],
            "delta_sched": s["p95"], "delta_state": st["p95"], "delta_prop": p["p95"],
            "delta_prop_p99": p["p99"], "n_observed": h["n"], "n_censored": censored,
            "probe_resolution_s": probe_int}


async def run_s2(uri, quick):
    # Lite has no real consistency propagation -> a single level is all a smoke
    # run can exercise; the sweep is the whole point of S2 on standalone.
    levels = (["Strong"] if _smoke()
              else (["Strong", "Bounded"] if quick
                    else ["Strong", "Bounded", "Session", "Eventual"]))
    n_items = 8 if _smoke() else (30 if quick else 60)
    out = []
    for lvl in levels:
        for backlog in (False, True):
            print(f"  S2 consistency={lvl} backlog={'heavy' if backlog else 'none'} "
                  f"n={n_items} ...", flush=True)
            m = await s2_one(uri, lvl, backlog, n_items, Tp=1.0, verify_cost=3.0,
                             probe_int=0.02, probe_budget=5.0)
            out.append(analysis.make_record(
                f"{CTX['prefix']}/S2/{lvl}/{'backlog' if backlog else 'nobacklog'}",
                "milvus", CTX["index"]["index_type"], CTX["evidence"],
                {"mode": "infilter", "consistency": lvl,
                 "verifier_backlog": "heavy" if backlog else "none",
                 "n_items": n_items, "Tp_s": 1.0, "verify_cost_s": 3.0,
                 "probe_interval_s": 0.02, "dim": DIM, **CTX["index"]}, m))
            dh = m["delta_hide_p95"]
            print(f"     -> delta_hide P95 {1000*dh:.0f} ms" if dh is not None
                  else "     -> no observation", flush=True)
            if m["n_censored"]:
                print(f"        ({m['n_censored']} censored: still visible after "
                      f"the 5 s probe budget)")
    return out


# --------------------------------------------------------------------------
# S5: HNSW parameters -> conditional Recall@k, query P99          [secondary]
# --------------------------------------------------------------------------
S5_GRID = [
    ("FLAT", {"index_type": "FLAT", "index_params": {}, "search_params": None}),
    ("HNSW_M8_ef32", {"index_type": "HNSW", "index_params": {"M": 8, "efConstruction": 64},
                      "search_params": {"ef": 32}}),
    ("HNSW_M16_ef64", {"index_type": "HNSW", "index_params": {"M": 16, "efConstruction": 200},
                       "search_params": {"ef": 64}}),
    ("HNSW_M32_ef128", {"index_type": "HNSW", "index_params": {"M": 32, "efConstruction": 400},
                        "search_params": {"ef": 128}}),
]


def s5_one(uri, label, idx, N, queries, over_fetch, hidden_mode):
    """Under-fill / recall against exact brute-force ground truth over the SAME
    vectors. On FLAT this reproduces the exact result; on HNSW, conditional
    Recall@k below 1.0 is pure ANN approximation error (unlockable only here)."""
    random.seed(0)
    vecs = gate2.gen_vecs(N, DIM)
    random.seed(1)
    qvecs = [[random.gauss(0, 1) for _ in range(DIM)] for _ in range(queries)]
    hidden = gate2.hidden_mask(hidden_mode, vecs, N, 0.30, qvecs, seed=2)
    elig = set(range(N)) - hidden

    b = mk(uri, f"s5_{label}", "Strong", **idx)
    batch = []
    for i in range(N):
        batch.append((i, vecs[i], i not in hidden))
        if len(batch) == 500:
            b.insert_many(batch)
            batch = []
    if batch:
        b.insert_many(batch)

    m = gate2.evaluate(b, vecs, qvecs, elig, hidden, K, over_fetch)
    lat = []
    for q in qvecs:                                   # latency of the same query set
        t0 = time.monotonic()
        b.search(q, over_fetch * K, use_index_filter=False)
        lat.append(time.monotonic() - t0)
    ql = stats(lat)
    return {**m, "query_p50": ql["p50"], "query_p95": ql["p95"], "query_p99": ql["p99"]}


def run_s5(uri, quick):
    N = 300 if _smoke() else (800 if quick else 2000)
    queries = 20 if _smoke() else (60 if quick else 150)
    grid = S5_GRID[:1] if _smoke() else (S5_GRID[:3] if quick else S5_GRID)  # Lite: FLAT only
    out = []
    for label, idx in grid:
        print(f"  S5 {label} N={N} queries={queries} ...", flush=True)
        m = s5_one(uri, label, idx, N, queries, over_fetch=2, hidden_mode="rankbiased")
        out.append(analysis.make_record(
            f"{CTX['prefix']}/S5/{label}", "milvus", idx["index_type"], CTX["evidence"],
            {"label": label, "N": N, "queries": queries, "k": K, "dim": DIM,
             "over_fetch": 2, "hidden_mode": "rankbiased", "hidden_ratio": 0.30,
             "consistency": "Strong", **idx}, m))
        cr = m["cond_recall"]
        print(f"     -> cond_recall {'n/a' if cr is None else f'{cr:.3f}'}, "
              f"query P99 {1000*m['query_p99']:.1f} ms")
    return out


# --------------------------------------------------------------------------
def tables(recs):
    s1 = [r for r in recs if "/S1/" in r["experiment_id"]]
    s2 = [r for r in recs if "/S2/" in r["experiment_id"]]
    s5 = [r for r in recs if "/S5/" in r["experiment_id"]]
    if s1:
        print("\n[S1] baseline x concurrency -> ingestion throughput, query tail latency")
        print(f"  {'baseline':>9} {'conc':>5} {'items/s':>9} {'P50 ms':>8} {'P95 ms':>8} "
              f"{'P99 ms':>8} {'n_q':>5}  window")
        for r in sorted(s1, key=lambda r: r["experiment_id"]):
            m, c = r["metrics"], r["config"]
            print(f"  {c['baseline']:>9} {c['concurrency']:>5} {m['throughput']:9.0f} "
                  f"{1000*m['query_p50']:8.1f} {1000*m['query_p95']:8.1f} "
                  f"{1000*m['query_p99']:8.1f} {m['n_queries']:>5}  {m['query_window']}")
        if any(r["metrics"]["query_window"] == "post_ingest_topup" for r in s1):
            print("  NOTE: post_ingest_topup rows measured latency partly AFTER ingestion")
            print("        finished (ingest was too short to fill the sample) -- they are")
            print("        NOT 'query latency under concurrent ingestion'. Raise N.")
    if s2:
        print("\n[S2] in-index enforcement -> delta_hide and its components (ms)")
        print(f"  {'consistency':>12} {'backlog':>8} {'P95':>7} {'P99':>7} {'max':>7} "
              f"{'sched':>7} {'state':>7} {'prop':>7} {'obs':>4} {'cens':>5}")
        for r in sorted(s2, key=lambda r: r["experiment_id"]):
            m, c = r["metrics"], r["config"]
            def ms(x):
                return "  n/a" if x is None else f"{1000*x:7.1f}"
            print(f"  {c['consistency']:>12} {c['verifier_backlog']:>8} {ms(m['delta_hide_p95'])} "
                  f"{ms(m['delta_hide_p99'])} {ms(m['delta_hide_max'])} {ms(m['delta_sched'])} "
                  f"{ms(m['delta_state'])} {ms(m['delta_prop'])} {m['n_observed']:>4} "
                  f"{m['n_censored']:>5}")
    if s5:
        print("\n[S5] index -> completeness and latency (cond_recall < 1.0 = ANN error)")
        print(f"  {'index':>16} {'underfill':>10} {'pf_rec':>7} {'cond_rec':>9} {'if_rec':>7} {'P99 ms':>8}")
        for r in sorted(s5, key=lambda r: r["experiment_id"]):
            m, c = r["metrics"], r["config"]
            cr = "n/a" if m["cond_recall"] is None else f"{m['cond_recall']:.3f}"
            print(f"  {c['label']:>16} {m['underfill']*100:9.1f}% {m['pf_recall']:7.3f} "
                  f"{cr:>9} {m['if_recall']:7.3f} {1000*m['query_p99']:8.1f}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", choices=["S1", "S2", "S5", "all"])
    ap.add_argument("--uri", default="http://localhost:19530")
    ap.add_argument("--quick", action="store_true", help="smaller grid, ~3x faster")
    ap.add_argument("--smoke", action="store_true",
                    help="execute the same code against Milvus Lite (no Docker) to "
                         "validate the code paths. Tagged E2/smoke, NOT §7.5 evidence.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    if a.smoke:
        CTX.update(index=FLAT, evidence="milvus_lite_flat", prefix="smoke")
        if a.uri.startswith("http"):
            a.uri = os.path.join(HERE, "results", "smoke.db")
    preflight(a.uri)
    out = a.out or os.path.join(HERE, "results",
                                "smoke.json" if a.smoke else "standalone.json")
    recs, failed = [], []

    plan = ["S1", "S2", "S5"] if a.exp == "all" else [a.exp]
    tier = ("E2 SMOKE on Milvus Lite -- code-path validation only, NOT a §7.5 result"
            if a.smoke else f"evidence tier E3{', quick grid' if a.quick else ''}")
    print(f"Standalone experiments {plan} on {a.uri}  ({tier})\n")
    for exp in plan:
        print(f"[{exp}]", flush=True)
        try:
            if exp == "S1":
                recs += await run_s1(a.uri, a.quick)
            elif exp == "S2":
                recs += await run_s2(a.uri, a.quick)
            else:
                recs += run_s5(a.uri, a.quick)
        except Exception as e:                     # one experiment failing must not
            failed.append((exp, repr(e)))          # discard the others' results
            print(f"  !! {exp} FAILED: {e!r}")
        json.dump(recs, open(out, "w"), indent=2)  # incremental, survives a crash

    print(f"\nwrote {len(recs)} records -> {out}")
    tables(recs)
    if failed:
        print("\nFAILED:", ", ".join(f"{e} ({m})" for e, m in failed))
    if a.smoke:
        print("\nSMOKE ONLY (evidence_level=milvus_lite_flat). Proves the code runs; "
              "it is NOT §7.5 evidence -- Lite has no HNSW, no real consistency, "
              "no concurrency. Now run it for real against Milvus standalone.")
    else:
        print("\nEvidence tier: milvus_standalone (E3). Send results/standalone.json "
              "back to fill §7.5 and the abstract's X/Y/Z.")


if __name__ == "__main__":
    asyncio.run(main())
