"""
S1/S2/S5 standalone experiments -- the frozen §7.5 contract, evidence tier E3.

These are the measurements Milvus Lite structurally CANNOT produce: real gRPC
concurrency (S1), write-to-query visibility enforcement (S2), and ANN behaviour
(S5). Everything emits through analysis.make_record with
evidence_level="milvus_standalone" and carries the deployment/index metadata
that proves what actually served the queries.

Design notes (each fixes a defect found in the 2026-07-25 pilot run):
- S1 uses a FIXED measurement window and a FIXED query arrival rate, in
  randomized paired blocks. The pilot compared baselines whose query samples
  differed 15x in size (621 vs 9031) over windows of different length, with B1
  running first and absorbing all cold-start cost -- it measured the no-defence
  baseline as SLOWER than the full protocol.
- S2 probes with BOTH a lightweight scalar point lookup and a full ANN search.
  The pilot's "delta_prop" (~470 ms) was the same order as one Strong search
  (~420 ms), so propagation was not separable from the probe's own cost.
  Note set_visible() is an UPSERT in Milvus (delete+insert), so hiding traverses
  the write path; this is a Milvus implementation property, not a property of
  in-index filtering in general.
- S5 uses the paper's 384-d embedding size and verifies the effective index
  type/state/load state. Success is a readable recall-latency trade-off, NOT
  "cond_recall < 1" -- reporting no observed recall loss is a valid outcome.

  ./.venv312/bin/python standalone_experiments.py --exp all
  ./.venv312/bin/python standalone_experiments.py --exp all --full
  ./.venv312/bin/python standalone_experiments.py --exp all --smoke   # Lite, E2
"""

import argparse
import asyncio
import itertools
import json
import os
import random
import time

import analysis

HERE = os.path.dirname(os.path.abspath(__file__))
DIM = 384          # matches the paper's all-MiniLM-L6-v2 embedding size (§5)
K = 10

HNSW = {"index_type": "HNSW",
        "index_params": {"M": 16, "efConstruction": 200},
        "search_params": {"ef": 64}}
FLAT = {"index_type": "FLAT", "index_params": {}, "search_params": None}

# --smoke runs the SAME code against Milvus Lite to prove the paths execute.
# Lite has no HNSW, no real consistency, no concurrency -> tagged E2 and
# prefixed "smoke/", and must NEVER be read as a §7.5 result.
CTX = {"index": HNSW, "evidence": "milvus_standalone", "prefix": "standalone",
       "env": {}}


def pctl(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))]


def stats(xs):
    return {"p50": pctl(xs, 0.50), "p95": pctl(xs, 0.95), "p99": pctl(xs, 0.99),
            "max": max(xs) if xs else None, "n": len(xs)}


def _smoke():
    return CTX["prefix"] == "smoke"


def mk(uri, collection, consistency="Bounded", **idx):
    from milvus_backend import MilvusBackend
    return MilvusBackend(dim=DIM, uri=uri, collection=collection,
                         consistency_level=consistency, **idx)


def rec(experiment_id, config, metrics, index_type=None):
    """Every record carries the deployment evidence, so no future reader has to
    reconstruct from latency whether this was Lite or a real server."""
    return analysis.make_record(
        f"{CTX['prefix']}/{experiment_id}", "milvus",
        index_type or CTX["index"]["index_type"], CTX["evidence"],
        {**config, **CTX["env"]}, metrics)


def preflight(uri):
    try:
        from pymilvus import MilvusClient
    except ImportError:
        raise SystemExit("pymilvus not installed. Use the matched venv:\n"
                         "  python3.12 -m venv .venv312 && "
                         "./.venv312/bin/pip install 'pymilvus>=2.4,<2.5' 'setuptools<81'")
    try:
        c = MilvusClient(uri=uri)
        c.list_collections()
    except Exception as e:
        raise SystemExit(
            f"Cannot reach Milvus at {uri}: {e}\n"
            "Start it first:  docker compose up -d && docker compose ps\n"
            "(wait for milvus-standalone to report (healthy), ~60-90s)")
    ver = None
    try:
        ver = c.get_server_version()
    except Exception:
        pass
    mode = "lite" if not str(uri).startswith("http") else "standalone"
    CTX["env"] = {"server_version": ver, "server_uri": uri, "deployment_mode": mode}
    print(f"Deployment: {mode}  server_version={ver}  uri={uri}")
    if mode == "lite" and not _smoke():
        raise SystemExit(
            "Refusing to emit E3 records from Milvus Lite: it has no HNSW "
            "(index_type is ignored), no real consistency levels, and no "
            "concurrency. Use --smoke for a code-path check, or point --uri at "
            "a standalone server.")


def vecs_np(n, dim, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, dim), dtype="float32")
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V


# --------------------------------------------------------------------------
# S1: B1-B4 x concurrency -> ingestion throughput, query tail latency [Fig B(a)]
# --------------------------------------------------------------------------
BASELINES = {
    "B1": {"desc": "immediate admission, no verification", "verify": False, "sync": False, "deadline": False},
    "B2": {"desc": "synchronous verify-before-visible", "verify": True, "sync": True, "deadline": False},
    "B3": {"desc": "async visible, no deadline", "verify": True, "sync": False, "deadline": False},
    "B4": {"desc": "fail-closed provisional visibility", "verify": True, "sync": False, "deadline": True},
}


async def s1_cell(uri, name, conc, window_s, qps, verify_cost, Tp, warmup, block):
    """Ingest as fast as `conc` workers allow for exactly `window_s` seconds,
    under an OPEN-loop query stream at a fixed arrival rate. Fixed window + fixed
    query rate is what makes tail latency comparable across baselines: a
    closed-loop stream would give the slow baseline (B2) an idle server and
    therefore a flatteringly low P99."""
    bl = BASELINES[name]
    b = mk(uri, f"s1_{name}_{conc}_b{block}", "Bounded", **CTX["index"])
    V = vecs_np(2048, DIM, seed=0)
    pool = [V[i].tolist() for i in range(len(V))]
    state, lat, inflight, dropped = {}, [], [0], [0]
    loop = asyncio.get_running_loop()
    ids = itertools.count()

    def hide(i):
        if state.get(i) == "PROVISIONAL":
            state[i] = "HIDDEN"

    async def verify(i):
        await asyncio.sleep(verify_cost)
        if state.get(i) == "PROVISIONAL":
            state[i] = "TRUSTED"

    async def admit(i, sync):
        if sync:                                   # B2: verify BEFORE visible
            await asyncio.sleep(verify_cost)
            state[i] = "TRUSTED"
            await asyncio.to_thread(b.insert, i, pool[i % len(pool)], True)
        else:
            state[i] = "PROVISIONAL"
            await asyncio.to_thread(b.insert, i, pool[i % len(pool)], True)
            if bl["verify"]:
                asyncio.create_task(verify(i))
            if bl["deadline"]:
                loop.call_later(Tp, hide, i)

    # ---- warm-up: connection, collection load, channel. Not measured. -----
    for w in range(warmup):
        await admit(next(ids), bl["sync"] and w < 3)   # keep warm-up cheap for B2
    for _ in range(10):
        await asyncio.to_thread(b.search, pool[0], 2 * K, False)

    qrng = random.Random(block * 97 + 13)

    async def one_query():
        inflight[0] += 1
        t0 = time.monotonic()
        try:
            raw = await asyncio.to_thread(b.search, pool[qrng.randrange(len(pool))],
                                          2 * K, False)
            [i for i in raw if state.get(i) in ("PROVISIONAL", "TRUSTED")][:K]
            lat.append((time.monotonic(), time.monotonic() - t0))
        except Exception:
            pass
        finally:
            inflight[0] -= 1

    stop_at = [0.0]
    admitted = [0]

    async def ingest_worker():
        while time.monotonic() < stop_at[0]:
            await admit(next(ids), bl["sync"])
            admitted[0] += 1

    lag = []

    async def query_driver():                      # open loop, fixed arrival rate
        gap = 1.0 / qps
        nxt = time.monotonic()
        while time.monotonic() < stop_at[0]:
            lag.append(time.monotonic() - nxt)     # offered-vs-actual dispatch lag
            if inflight[0] < 256:
                asyncio.create_task(one_query())
            else:
                dropped[0] += 1                    # server saturated; recorded
            nxt += gap
            await asyncio.sleep(max(0.0, nxt - time.monotonic()))

    t0 = time.monotonic()
    stop_at[0] = t0 + window_s
    qd = asyncio.create_task(query_driver())
    await asyncio.gather(*[ingest_worker() for _ in range(conc)])
    t_end = time.monotonic()
    qd.cancel()
    await asyncio.gather(qd, return_exceptions=True)
    while inflight[0] > 0 and time.monotonic() < t_end + 5:
        await asyncio.sleep(0.05)

    inwin = [l for t, l in lat if t <= t_end]
    ql, lg = stats(inwin), stats(lag)
    win = t_end - t0
    return {"throughput": admitted[0] / win, "window_s": win,
            "admitted": admitted[0], "query_p50": ql["p50"], "query_p95": ql["p95"],
            "query_p99": ql["p99"], "n_queries": ql["n"], "queries_dropped": dropped[0],
            "qps_offered": qps, "qps_achieved": ql["n"] / win,
            "query_dispatch_lag_p95": lg["p95"]}


async def run_s1(uri, quick, full):
    # Default grid is SMOKE-GRADE for S1: 15 s x 50 qps is ~750 samples, whose
    # P99 rests on the slowest 7-8 points. --full is the grid whose P99 may be
    # quoted: 45 s x 100 qps = ~4500 samples per cell, 5 paired blocks. Sample
    # count and block count are separate knobs; neither has to ride on window
    # length alone.
    window = 8 if _smoke() else (15 if not full else 45)
    reps = 1 if _smoke() else (3 if not full else 5)
    # One concurrency level even at --full: 45 s x 4 baselines x 5 blocks is
    # already 15 min per level. A second level is a separate run, not a default.
    concs = [4] if (quick or _smoke()) else [8]
    qps = 10 if _smoke() else (50 if not full else 100)
    if not (_smoke() or full):
        print("  NOTE: default S1 grid is smoke-grade (~750 query samples per cell).")
        print("        Use --full for the grid whose P99 is quotable.")
    out, by_block = [], {}

    # Global warm-up on a throwaway collection: the per-cell warm-up cannot
    # absorb database-level first-touch cost, which otherwise lands entirely on
    # whichever baseline happens to run first.
    print("  S1 global warm-up ...", flush=True)
    wb = mk(uri, "s1_warmup", "Bounded", **CTX["index"])
    wv = vecs_np(64, DIM, seed=99)
    for i in range(64):
        wb.insert(i, wv[i].tolist(), True)
    for _ in range(20):
        wb.search(wv[0].tolist(), 2 * K, False)
    for conc in concs:
        for block in range(reps):
            # Randomized order within each block: the pilot ran B1,B1,B1,B4,B4,B4
            # so B1 alone paid cold-start and came out 20% SLOWER than B4.
            order = list(BASELINES)
            random.Random(1000 + block).shuffle(order)
            print(f"  S1 block {block+1}/{reps} conc={conc} order={'>'.join(order)} "
                  f"window={window}s", flush=True)
            for name in order:
                m = await s1_cell(uri, name, conc, window, qps, verify_cost=0.05,
                                  Tp=1.0, warmup=30, block=block)
                by_block.setdefault((conc, name), []).append(m)
                print(f"     {name}: {m['throughput']:.0f} items/s, "
                      f"P99 {1000*m['query_p99']:.1f} ms, n_q={m['n_queries']}",
                      flush=True)
    for (conc, name), runs in by_block.items():
        tp = [r["throughput"] for r in runs]
        med = sorted(runs, key=lambda r: r["throughput"])[len(runs) // 2]
        # Per-block P99 aggregated across blocks -- never a single pooled P99 over
        # concatenated samples, which would hide between-block variation.
        p99 = sorted(r["query_p99"] for r in runs if r["query_p99"] is not None)
        agg = {"query_p99_blocks": p99,
               "query_p99_median": p99[len(p99) // 2] if p99 else None,
               "query_p99_min": min(p99) if p99 else None,
               "query_p99_max": max(p99) if p99 else None}
        out.append(rec(f"S1/{name}/conc{conc}",
                       {"baseline": name, "desc": BASELINES[name]["desc"],
                        "concurrency": conc, "window_s": window, "blocks": len(runs),
                        "k": K, "dim": DIM, "consistency": "Bounded",
                        "verify_cost_s": 0.05, "Tp_s": 1.0, "over_fetch": 2,
                        "query_load": f"open-loop, fixed {qps} qps",
                        "block_order": "randomized per block", **CTX["index"]},
                       {**med, **agg, "throughput_runs": tp, "throughput_min": min(tp),
                        "throughput_max": max(tp), "blocks": len(runs)}))
    # Paired ratio within each block -- only valid because blocks are paired.
    for conc in {c for c, _ in by_block}:
        b1 = by_block.get((conc, "B1"), [])
        b4 = by_block.get((conc, "B4"), [])
        if b1 and b4 and len(b1) == len(b4):
            R = [x["throughput"] / y["throughput"] for x, y in zip(b4, b1)]
            out.append(rec(f"S1/paired/B4_over_B1/conc{conc}",
                           {"concurrency": conc, "blocks": len(R),
                            "definition": "per-block B4 throughput / B1 throughput"},
                           {"ratio_median": sorted(R)[len(R) // 2],
                            "ratio_min": min(R), "ratio_max": max(R),
                            "ratios": R,
                            "same_direction": all(r > 1 for r in R) or all(r < 1 for r in R)}))
    return out


# --------------------------------------------------------------------------
# S2: in-index enforcement -> hide confirmation delay, two-layer probe [Fig B(b)]
# --------------------------------------------------------------------------
async def s2_one(uri, consistency, backlog, n_items, Tp, verify_cost, probe_int,
                 probe_budget, probe_mode="both"):
    """set_visible() is an upsert in Milvus, so a hide traverses the write path.
    Two probes run per item:
      - lightweight scalar point lookup  -> when the state itself reads back false
      - full ANN search (supported path) -> when the query path stops returning it
    The gap between them is the query path's own cost. Neither is a pure
    propagation delay; both are reported as observed upper bounds."""
    # FLAT, not HNSW: S2 isolates state visibility. Under an approximate index an
    # item can drop out of the result because ANN missed it, which is
    # indistinguishable from the hide taking effect.
    b = mk(uri, f"s2_{consistency}_{'bk' if backlog else 'nb'}", consistency, **FLAT)
    V = vecs_np(max(n_items, 8), DIM, seed=7)
    sem = asyncio.Semaphore(2) if backlog else None
    d_hide, d_ack, d_state_vis, d_confirm = [], [], [], []
    lat_state, lat_search, lag = [], [], []
    cens_state = cens_search = 0
    vtasks = []

    async def verifier():                # backlogged; must not affect the timer (I6)
        if sem is not None:
            async with sem:
                await asyncio.sleep(verify_cost)
        else:
            await asyncio.sleep(verify_cost)

    async def one(i):
        nonlocal cens_state, cens_search
        vec = V[i].tolist()
        await asyncio.to_thread(b.insert, i, vec, True)
        t_visible = time.monotonic()
        deadline = t_visible + Tp
        vtasks.append(asyncio.create_task(verifier()))
        await asyncio.sleep(max(0.0, deadline - time.monotonic()))    # independent timer
        t_sched = time.monotonic()
        await asyncio.to_thread(b.set_visible, i, False)              # upsert
        t_ack = time.monotonic()

        # The two probes MUST run concurrently from t_ack. Interleaving them in
        # one loop pushes each ANN observation behind a point lookup (inflating
        # query-confirm) and changes the sampling cadence mid-measurement once
        # the scalar probe stops firing.
        end = time.monotonic() + probe_budget

        async def probe_state():
            nxt = time.monotonic()
            while time.monotonic() < end:
                lag.append(time.monotonic() - nxt)
                ts = time.monotonic()
                try:
                    vis = await asyncio.to_thread(b.point_get, i)
                except Exception:
                    vis = None
                lat_state.append(time.monotonic() - ts)
                if vis is False:
                    return time.monotonic()
                nxt += probe_int
                await asyncio.sleep(max(0.0, nxt - time.monotonic()))
            return None

        async def probe_query():
            nxt = time.monotonic()
            while time.monotonic() < end:
                ts = time.monotonic()
                try:
                    hits = await asyncio.to_thread(b.search, vec, 1, True)
                except Exception:
                    return None
                lat_search.append(time.monotonic() - ts)
                if i not in hits:
                    return time.monotonic()
                nxt += probe_int
                await asyncio.sleep(max(0.0, nxt - time.monotonic()))
            return None

        if probe_mode == "query_only":       # control: is the scalar probe itself
            t_state = None                   # inflating the ANN search latency?
            t_conf = await probe_query()
        else:
            t_state, t_conf = await asyncio.gather(probe_state(), probe_query())

        if t_state is None:
            cens_state += 1
        else:
            d_state_vis.append(t_state - t_ack)
        if t_conf is None:
            cens_search += 1
            return
        d_ack.append(t_ack - t_sched)
        d_confirm.append(t_conf - t_ack)
        d_hide.append(t_conf - deadline)

    items = []
    for i in range(n_items):
        items.append(asyncio.create_task(one(i)))
        await asyncio.sleep(0.10)
    await asyncio.gather(*items, return_exceptions=True)
    for t in vtasks:
        t.cancel()
    await asyncio.gather(*vtasks, return_exceptions=True)

    h, a, sv, c = stats(d_hide), stats(d_ack), stats(d_state_vis), stats(d_confirm)
    ls, lq, lg = stats(lat_state), stats(lat_search), stats(lag)
    return {"delta_hide_p95": h["p95"], "delta_hide_p99": h["p99"], "delta_hide_max": h["max"],
            "delta_ack_p95": a["p95"], "delta_state_visible_p95": sv["p95"],
            "delta_confirm_p95": c["p95"], "delta_confirm_p99": c["p99"],
            "probe_state_p50": ls["p50"], "probe_search_p50": lq["p50"],
            "probe_scheduling_lag_p95": lg["p95"],
            "n_observed": h["n"], "n_censored": cens_search,
            "n_censored_state": cens_state, "probe_resolution_s": probe_int}


async def run_s2(uri, quick, full):
    levels = (["Strong"] if _smoke()
              else (["Strong", "Bounded"] if not full
                    else ["Strong", "Bounded", "Session", "Eventual"]))
    n_items = 6 if _smoke() else (25 if not full else 60)
    out = []
    # Instrumentation-overhead control: same cell with the scalar probe switched
    # off. If probe_search_p50 drops materially, the two-probe design is paying
    # for itself in the very number it reports.
    if full:
        print("  S2 control: query-only probe (instrumentation overhead) ...", flush=True)
        mc = await s2_one(uri, "Strong", False, n_items, Tp=1.0, verify_cost=3.0,
                          probe_int=0.005, probe_budget=5.0, probe_mode="query_only")
        out.append(rec("S2/Strong/nobacklog/query_only_control",
                       {"mode": "infilter", "consistency": "Strong",
                        "verifier_backlog": "none", "n_items": n_items,
                        "probes": "ANN search only (scalar probe disabled)",
                        "role": "instrumentation-overhead control", **FLAT},
                       mc, index_type="FLAT"))
        _p = mc["probe_search_p50"]
        print(f"     -> probe_search_p50 {'n/a' if _p is None else f'{1000*_p:.1f} ms'}")
    for lvl in levels:
        for backlog in (False, True):
            print(f"  S2 consistency={lvl} backlog={'heavy' if backlog else 'none'} "
                  f"n={n_items} ...", flush=True)
            m = await s2_one(uri, lvl, backlog, n_items, Tp=1.0, verify_cost=3.0,
                             probe_int=0.005, probe_budget=5.0)
            out.append(rec(f"S2/{lvl}/{'backlog' if backlog else 'nobacklog'}",
                           {"mode": "infilter", "consistency": lvl,
                            "verifier_backlog": "heavy" if backlog else "none",
                            "n_items": n_items, "Tp_s": 1.0, "verify_cost_s": 3.0,
                            "probe_interval_s": 0.005, "dim": DIM,
                            "probes": "scalar point lookup and ANN search, run "
                                      "concurrently from upsert ack",
                            "hide_mechanism": "override upsert on Milvus v2.4.15 "
                                              "(no in-place scalar update); cost is "
                                              "version- and update-mode-dependent",
                            **FLAT}, m, index_type="FLAT"))
            def _ms(x):
                return "n/a" if x is None else f"{1000*x:.0f} ms"
            print(f"     -> delta_hide P95 {_ms(m['delta_hide_p95'])}, "
                  f"state {_ms(m['delta_state_visible_p95'])}, "
                  f"confirm {_ms(m['delta_confirm_p95'])}, "
                  f"probe_search {_ms(m['probe_search_p50'])}", flush=True)
    return out


# --------------------------------------------------------------------------
# S5: index sweep -> conditional Recall@k vs latency               [secondary]
# --------------------------------------------------------------------------
def s5_grid(full):
    g = [("FLAT", FLAT),
         ("HNSW_ef16", {"index_type": "HNSW", "index_params": {"M": 16, "efConstruction": 200},
                        "search_params": {"ef": 16}}),
         ("HNSW_ef64", {"index_type": "HNSW", "index_params": {"M": 16, "efConstruction": 200},
                        "search_params": {"ef": 64}})]
    if full:
        g.append(("HNSW_ef256", {"index_type": "HNSW",
                                 "index_params": {"M": 16, "efConstruction": 200},
                                 "search_params": {"ef": 256}}))
    return g


def s5_one(uri, label, idx, N, queries, over_fetch, hidden_ratio=0.30):
    """Rank-biased hiding (adversarial worst case) at the paper's 384-d embedding
    size. Ground truth is exact brute force over the SAME vectors, computed with
    numpy -- at 384-d the pure-Python cosine loop of gate2 is not viable."""
    import numpy as np
    V = vecs_np(N, DIM, seed=0)
    Q = vecs_np(queries, DIM, seed=1)
    S = Q @ V.T                                        # cosine (both normalized)

    n_hidden = int(hidden_ratio * N)
    score = S.max(axis=0)                              # rank-biased: top-ranked hidden
    hidden_arr = np.argsort(-score)[:n_hidden]
    hidden = set(int(x) for x in hidden_arr)

    Sg = S.copy()
    Sg[:, hidden_arr] = -np.inf                        # ground truth over ELIGIBLE only
    part = np.argpartition(-Sg, K, axis=1)[:, :K]
    rows = np.arange(queries)[:, None]
    gt = part[rows, np.argsort(-Sg[rows, part], axis=1)]

    b = mk(uri, f"s5_{label}_{N}", "Bounded", **idx)
    batch = []
    for i in range(N):
        batch.append((i, V[i].tolist(), i not in hidden))
        if len(batch) == 500:
            b.insert_many(batch)
            batch = []
    if batch:
        b.insert_many(batch)
    # Without seal + build + reload, queries hit an unindexed growing segment and
    # every index_type degrades to the same brute-force scan.
    b.wait_index()
    info = b.index_info()

    underfill = 0
    pf_rec, cond, if_rec, hM = [], [], [], []
    lat_pf, lat_if = [], []
    for qi in range(queries):
        q = Q[qi].tolist()
        g = set(int(x) for x in gt[qi])
        t0 = time.monotonic()
        raw = b.search(q, over_fetch * K, use_index_filter=False)
        lat_pf.append(time.monotonic() - t0)
        hM.append(sum(1 for i in raw if i in hidden) / max(1, len(raw)))
        pf = [i for i in raw if i not in hidden][:K]
        r = len(set(pf) & g) / K
        pf_rec.append(r)
        if len(pf) == K:
            cond.append(r)
        else:
            underfill += 1
        t0 = time.monotonic()
        inf = b.search(q, K, use_index_filter=True)
        lat_if.append(time.monotonic() - t0)
        if_rec.append(len(set(inf) & g) / K)

    lp, li = stats(lat_pf), stats(lat_if)
    return {"underfill": underfill / queries,
            "hM50": pctl(hM, 0.50), "hM95": pctl(hM, 0.95),
            "pf_recall": sum(pf_rec) / len(pf_rec),
            "cond_recall": (sum(cond) / len(cond)) if cond else float("nan"),
            "if_recall": sum(if_rec) / len(if_rec),
            "query_p50": lp["p50"], "query_p95": lp["p95"], "query_p99": lp["p99"],
            "infilter_p99": li["p99"], **info}


def run_s5(uri, quick, full):
    Ns = [300] if _smoke() else ([20000] if not full else [20000, 100000])
    queries = 20 if _smoke() else (100 if not full else 150)
    grid = s5_grid(full)[:1] if _smoke() else s5_grid(full)
    out = []
    for N in Ns:
        for label, idx in grid:
            print(f"  S5 {label} N={N} d={DIM} queries={queries} ...", flush=True)
            m = s5_one(uri, label, idx, N, queries, over_fetch=2)
            out.append(rec(f"S5/{label}/N{N}",
                           {"label": label, "N": N, "queries": queries, "k": K,
                            "dim": DIM, "over_fetch": 2, "hidden_mode": "rankbiased",
                            "hidden_ratio": 0.30, "consistency": "Bounded", **idx},
                           m, index_type=idx["index_type"]))
            cr = m["cond_recall"]
            print(f"     -> cond_recall {'n/a' if cr != cr else f'{cr:.4f}'}, "
                  f"P99 {1000*m['query_p99']:.1f} ms, "
                  f"index={m['index_type_effective']}/{m['index_state']}")
    return out


# --------------------------------------------------------------------------
def tables(recs):
    s1 = [r for r in recs if "/S1/" in r["experiment_id"] and "paired" not in r["experiment_id"]]
    pr = [r for r in recs if "paired" in r["experiment_id"]]
    s2 = [r for r in recs if "/S2/" in r["experiment_id"]]
    s5 = [r for r in recs if "/S5/" in r["experiment_id"]]
    env = recs[0]["config"] if recs else {}
    print(f"\nDeployment: {env.get('deployment_mode')} {env.get('server_version')} "
          f"| commit={recs[0]['git_commit'] if recs else '?'}")

    if s1:
        print("\n[S1] fixed window, fixed query arrival rate, randomized paired blocks")
        print(f"  {'baseline':>9} {'items/s':>9} {'(min-max)':>15} "
              f"{'P99 med':>8} {'P99 (min-max)':>16} {'n_q':>6} {'qps a/o':>10} {'drop':>5}")
        for r in sorted(s1, key=lambda r: r["experiment_id"]):
            m, c = r["metrics"], r["config"]
            rng = f"({m['throughput_min']:.0f}-{m['throughput_max']:.0f})"
            qq = f"{m['qps_achieved']:.0f}/{m['qps_offered']}"
            pm = m.get("query_p99_median")
            p99rng = (f"({1000*m['query_p99_min']:.0f}-{1000*m['query_p99_max']:.0f})"
                      if m.get("query_p99_min") is not None else "n/a")
            print(f"  {c['baseline']:>9} {m['throughput']:9.0f} {rng:>15} "
                  f"{1000*pm if pm else float('nan'):8.1f} {p99rng:>16} {m['n_queries']:>6} "
                  f"{qq:>10} {m['queries_dropped']:>5}")
        print("  P99 is the median of per-block P99s (blocks never pooled), with the")
        print("  across-block range beside it.")
        if any(r["metrics"]["n_queries"] < 2000 for r in s1):
            print("  NOTE: cells with <2000 query samples have a P99 resting on <20 points.")
            print("        Treat as smoke-grade; re-run with --full before quoting a P99.")
    for r in pr:
        m = r["metrics"]
        print(f"\n  paired B4/B1 (conc {r['config']['concurrency']}): median "
              f"{m['ratio_median']:.3f}, range {m['ratio_min']:.3f}-{m['ratio_max']:.3f}, "
              f"consistent direction: {m['same_direction']}")
        print("  A ratio band spanning 1.0 means the protocol's throughput cost was not")
        print("  resolved above run-to-run variability -- report that, not a point figure.")

    if s2:
        print("\n[S2] in-index hide: upsert ack -> state visible -> query confirmation (ms)")
        print(f"  {'consistency':>12} {'backlog':>8} {'hideP95':>8} {'ack':>7} {'state':>7} "
              f"{'confirm':>8} {'probe_st':>9} {'probe_q':>8} {'obs':>4} {'cens':>5}")
        for r in sorted(s2, key=lambda r: r["experiment_id"]):
            m, c = r["metrics"], r["config"]
            def ms(x, w=7):
                return " " * (w - 3) + "n/a" if x is None else f"{1000*x:{w}.1f}"
            print(f"  {c['consistency']:>12} {c['verifier_backlog']:>8} "
                  f"{ms(m['delta_hide_p95'], 8)} {ms(m['delta_ack_p95'])} "
                  f"{ms(m['delta_state_visible_p95'])} {ms(m['delta_confirm_p95'], 8)} "
                  f"{ms(m['probe_state_p50'], 9)} {ms(m['probe_search_p50'], 8)} "
                  f"{m['n_observed']:>4} {m['n_censored']:>5}")
        print("  state = lightweight scalar point lookup; confirm = full ANN search path.")
        print("  Neither is a pure propagation delay. If confirm ~ probe_q, the ANN probe's")
        print("  own cost dominates and propagation is not separately resolvable.")

    if s5:
        print("\n[S5] index -> completeness and latency at the paper's 384-d size")
        print(f"  {'index':>14} {'N':>7} {'underfill':>10} {'pf_rec':>7} {'cond_rec':>9} "
              f"{'if_rec':>7} {'P99 ms':>8}  effective_index/state/load")
        for r in sorted(s5, key=lambda r: r["experiment_id"]):
            m, c = r["metrics"], r["config"]
            cr = "n/a" if m["cond_recall"] is None else f"{m['cond_recall']:.4f}"
            print(f"  {c['label']:>14} {c['N']:>7} {m['underfill']*100:9.1f}% "
                  f"{m['pf_recall']:7.3f} {cr:>9} {m['if_recall']:7.3f} "
                  f"{1000*m['query_p99']:8.1f}  {m['index_type_effective']}/"
                  f"{m['index_state']}/{m['load_state']}")
        bad = [r for r in s5 if r["config"]["index_type"] == "HNSW"
               and str(r["metrics"]["index_type_effective"]).upper() != "HNSW"]
        if bad:
            print("\n  !! The effective index is not HNSW on some rows -- those rows measure")
            print("     something other than the requested ANN configuration.")
        key = lambda m: (round(m["underfill"], 6), round(m["pf_recall"], 6),
                         round(m["if_recall"], 6))
        flat = [r for r in s5 if r["config"]["index_type"] == "FLAT"]
        ann = [r for r in s5 if r["config"]["index_type"] != "FLAT"]
        if flat and ann and all(key(r["metrics"]) == key(flat[0]["metrics"]) for r in ann):
            print("\n  NOTE: every HNSW row matches FLAT exactly. Two readings: (a) the ANN")
            print("  index is not serving these queries, or (b) HNSW is exact at this scale.")
            print("  Disambiguate with the effective_index/state column above. Either way")
            print("  this does not evidence ANN-induced recall loss.")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", choices=["S1", "S2", "S5", "all"])
    ap.add_argument("--uri", default="http://localhost:19530")
    ap.add_argument("--quick", action="store_true", help="single concurrency level")
    ap.add_argument("--full", action="store_true",
                    help="5 blocks, 60 s windows, 4 consistency levels, N up to 100k")
    ap.add_argument("--smoke", action="store_true",
                    help="run the same code against Milvus Lite (no Docker) to validate "
                         "the code paths. Tagged E2/smoke, NOT §7.5 evidence.")
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
            if a.smoke else f"evidence tier E3{', FULL grid' if a.full else ''}")
    print(f"Experiments {plan}  ({tier})\n")
    for exp in plan:
        print(f"[{exp}]", flush=True)
        try:
            if exp == "S1":
                recs += await run_s1(a.uri, a.quick, a.full)
            elif exp == "S2":
                recs += await run_s2(a.uri, a.quick, a.full)
            else:
                recs += run_s5(a.uri, a.quick, a.full)
        except Exception as e:
            failed.append((exp, repr(e)))
            print(f"  !! {exp} FAILED: {e!r}")
        json.dump(recs, open(out, "w"), indent=2)

    print(f"\nwrote {len(recs)} records -> {out}")
    if recs:
        tables(recs)
    if failed:
        print("\nFAILED:", ", ".join(f"{e} ({m})" for e, m in failed))
    if a.smoke:
        print("\nSMOKE ONLY (evidence_level=milvus_lite_flat). Proves the code runs; "
              "NOT §7.5 evidence.")
    else:
        print("\nEvidence tier: milvus_standalone (E3). Send results/standalone.json back.")


if __name__ == "__main__":
    asyncio.run(main())
