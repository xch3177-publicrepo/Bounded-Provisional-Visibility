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
       "env": {}, "profile": "default"}


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
        {**config, **CTX["env"], "run_profile": CTX["profile"]}, metrics)


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


# Williams square for four treatments: every baseline occupies every position
# exactly once, and all 12 ordered pairs of distinct baselines occur exactly
# once, so first-order carryover is balanced too.
WILLIAMS4 = [["B1", "B2", "B4", "B3"],
             ["B2", "B3", "B1", "B4"],
             ["B3", "B4", "B2", "B1"],
             ["B4", "B1", "B3", "B2"]]
S1_BLOCKS = [4]          # set by --s1-blocks; must stay a multiple of 4


async def s1_sentinel(b, qv, n=200):
    """Fixed workload run before each block against an unchanging collection.

    Cleaning up per-cell collections removes the known cause of cross-block
    decay; the sentinel is how we see whether any decay SURVIVES that fix. It
    does identical work every block, so a drift in its latency is a property of
    the environment and not of any baseline."""
    lat = []
    for i in range(n):
        t0 = time.monotonic()
        await asyncio.to_thread(b.search, qv[i % len(qv)].tolist(), 2 * K, False)
        lat.append(time.monotonic() - t0)
    st = stats(lat)
    return {"sentinel_p50": st["p50"], "sentinel_p95": st["p95"], "sentinel_n": st["n"]}


async def s1_cell(uri, name, conc, window_s, qps, verify_cost, Tp, warmup, block,
                  position=None):
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
    res = {"throughput": admitted[0] / win, "window_s": win,
           "admitted": admitted[0], "query_p50": ql["p50"], "query_p95": ql["p95"],
           "query_p99": ql["p99"], "n_queries": ql["n"], "queries_dropped": dropped[0],
           "qps_offered": qps, "qps_achieved": ql["n"] / win,
           "query_dispatch_lag_p95": lg["p95"],
           "block": block, "position_in_block": position}
    # Release and drop before the next cell starts. Leaving 20 loaded
    # collections behind is what made throughput decay with block index.
    b.drop()
    return res


async def run_s1(uri, quick, full, out=None):
    out = [] if out is None else out
    # Default grid is SMOKE-GRADE for S1: 15 s x 50 qps is ~750 samples, whose
    # P99 rests on the slowest 7-8 points. --full is the grid whose P99 may be
    # quoted: 45 s x 100 qps = ~4500 samples per cell, 5 paired blocks. Sample
    # count and block count are separate knobs; neither has to ride on window
    # length alone.
    window = 8 if _smoke() else (15 if not full else 45)
    # Block count MUST be a multiple of 4: the Williams square below balances
    # four treatments over four positions, so 5 blocks would hand one baseline
    # an extra turn in some position and reintroduce exactly the imbalance the
    # design exists to remove. 4 blocks give exact balance; 8 is that balance
    # replicated for precision, not a stronger balance.
    reps = 1 if _smoke() else (4 if not full else S1_BLOCKS[0])
    # One concurrency level even at --full: 45 s x 4 baselines x 5 blocks is
    # already 15 min per level. A second level is a separate run, not a default.
    concs = [4] if (quick or _smoke()) else [8]
    qps = 10 if _smoke() else (50 if not full else 100)
    if not (_smoke() or full):
        print("  NOTE: default S1 grid is smoke-grade (~750 query samples per cell).")
        print("        Use --full for the grid whose P99 is quotable.")
    # NOTE: do NOT rebind `out` here. main() passes the shared record list and
    # discards the return value, so a fresh list silently drops every S1 record.
    by_block = {}

    # Global warm-up on a throwaway collection: the per-cell warm-up cannot
    # absorb database-level first-touch cost, which otherwise lands entirely on
    # whichever baseline happens to run first.
    print("  S1 global warm-up ...", flush=True)
    # A crashed earlier run leaves loaded collections behind, and the next run
    # would then start from a server state it never chose. Sweep first.
    try:
        from pymilvus import MilvusClient
        _c = MilvusClient(uri=uri)
        stale = [n for n in _c.list_collections() if n.startswith("s1_")]
        for n in stale:
            try:
                _c.release_collection(n)
            except Exception:
                pass
            _c.drop_collection(n)
        if stale:
            print(f"     swept {len(stale)} leftover s1_* collections", flush=True)
    except Exception as e:
        print(f"     (could not sweep stale collections: {e!r})", flush=True)

    wb = mk(uri, "s1_warmup", "Bounded", **CTX["index"])
    wv = vecs_np(64, DIM, seed=99)
    for i in range(64):
        wb.insert(i, wv[i].tolist(), True)
    for _ in range(20):
        wb.search(wv[0].tolist(), 2 * K, False)

    # Sentinel: a constant collection and a constant query burst, run once per
    # block. Identical work every time, so any trend in it is the environment.
    sb = mk(uri, "s1_sentinel", "Bounded", **CTX["index"])
    sv = vecs_np(5000, DIM, seed=123)
    sb.insert_many([(i, sv[i].tolist(), True) for i in range(5000)])
    sb.wait_index()

    for conc in concs:
        for block in range(reps):
            sentinel = await s1_sentinel(sb, sv)
            print(f"     sentinel: P50 {1000*sentinel['sentinel_p50']:.2f} ms  "
                  f"P95 {1000*sentinel['sentinel_p95']:.2f} ms", flush=True)
            # Per-block shuffling is not a design. With a fixed seed the 5-block
            # run put B4 last in 4 of 5 blocks while server throughput decayed
            # with block index, so the treatment effect and the position effect
            # were the same number. This Williams square balances both.
            order = WILLIAMS4[block % 4]
            print(f"  S1 block {block+1}/{reps} conc={conc} order={'>'.join(order)} "
                  f"window={window}s", flush=True)
            for name in order:
                m = await s1_cell(uri, name, conc, window, qps, verify_cost=0.05,
                                  Tp=1.0, warmup=30, block=block,
                                  position=order.index(name))
                m.update(sentinel)
                by_block.setdefault((conc, name), []).append(m)
                print(f"     {name}: {m['throughput']:.0f} items/s, "
                      f"P99 {1000*m['query_p99']:.1f} ms, n_q={m['n_queries']}",
                      flush=True)
    for (conc, name), runs in by_block.items():
        tp = [r["throughput"] for r in runs]
        med = sorted(runs, key=lambda r: r["throughput"])[len(runs) // 2]
        # Per-block P99 aggregated across blocks -- never a single pooled P99 over
        # concatenated samples, which would hide between-block variation.
        # Keep BLOCK ORDER: sorting here silently destroys the pairing that
        # the paired B4-B1 delta depends on.
        p99 = [r["query_p99"] for r in runs]
        ok = [x for x in p99 if x is not None]
        agg = {"query_p99_blocks": p99,
               "query_p50_blocks": [r["query_p50"] for r in runs],
               "query_p95_blocks": [r["query_p95"] for r in runs],
               "n_queries_blocks": [r["n_queries"] for r in runs],
               "query_p99_median": sorted(ok)[len(ok) // 2] if ok else None,
               "query_p99_min": min(ok) if ok else None,
               "query_p99_max": max(ok) if ok else None}
        out.append(rec(f"S1/{name}/conc{conc}",
                       {"baseline": name, "desc": BASELINES[name]["desc"],
                        "concurrency": conc, "window_s": window, "blocks": len(runs),
                        "k": K, "dim": DIM, "consistency": "Bounded",
                        "verify_cost_s": 0.05, "Tp_s": 1.0, "over_fetch": 2,
                        "query_load": f"open-loop, fixed {qps} qps",
                        "expected_queries_per_block": int(qps * window),
                        "block_order": "Williams square (position- and "
                                       "carryover-balanced over 4 treatments)",
                        "positions": sorted(r.get("position_in_block") for r in runs),
                        **CTX["index"]},
                       {**med, **agg, "throughput_runs": tp, "throughput_min": min(tp),
                        "throughput_max": max(tp), "blocks": len(runs)}))
    for _b in (wb, sb):                     # the run's own fixtures must not leak
        try:
            _b.drop()
        except Exception:
            pass

    # Paired ratio within each block -- only valid because blocks are paired.
    for conc in {c for c, _ in by_block}:
        b1 = by_block.get((conc, "B1"), [])
        b4 = by_block.get((conc, "B4"), [])
        if b1 and b4 and len(b1) == len(b4):
            R = [x["throughput"] / y["throughput"] for x, y in zip(b4, b1)]
            # P99 is paired WITHIN a block too -- comparing each baseline's
            # across-block min-max would confound protocol cost with block noise.
            D = [x["query_p99"] - y["query_p99"] for x, y in zip(b4, b1)
                 if x["query_p99"] is not None and y["query_p99"] is not None]
            Ds = sorted(D)
            out.append(rec(f"S1/paired/B4_over_B1/conc{conc}",
                           {"concurrency": conc, "blocks": len(R),
                            "definition": "per-block B4/B1; throughput as ratio, "
                                          "P99 as difference (B4 - B1), seconds"},
                           {"ratio_median": sorted(R)[len(R) // 2],
                            "ratio_min": min(R), "ratio_max": max(R), "ratios": R,
                            "same_direction": all(r > 1 for r in R) or all(r < 1 for r in R),
                            "p99_delta_median": Ds[len(Ds) // 2] if Ds else None,
                            "p99_delta_min": min(Ds) if Ds else None,
                            "p99_delta_max": max(Ds) if Ds else None,
                            "p99_deltas": D,
                            "p99_same_direction": bool(Ds) and (all(d > 0 for d in Ds)
                                                               or all(d < 0 for d in Ds))}))
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
                    vis = await asyncio.to_thread(b.point_get, i, consistency)
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
                    hits = await asyncio.to_thread(b.search, vec, 1, True, consistency)
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

    # Recorded, not asserted: writer and both probes go through this one backend
    # instance, so a Session cell carries read-your-own-write semantics.
    session_shared = b.session_id()
    h, a, sv, c = stats(d_hide), stats(d_ack), stats(d_state_vis), stats(d_confirm)
    ls, lq, lg = stats(lat_state), stats(lat_search), stats(lag)
    # Raw samples are stored, not just percentiles. Two earlier defects cost a
    # full re-run because a statistic that was never emitted could not be
    # recomputed offline. At 200 events per cell this is a few KB and it makes
    # every future percentile, residual, or CI recoverable from the file alone.
    raw = {"delta_hide_samples": [round(x, 6) for x in d_hide],
           "delta_confirm_samples": [round(x, 6) for x in d_confirm],
           "delta_state_visible_samples": [round(x, 6) for x in d_state_vis],
           "probe_search_samples": [round(x, 6) for x in lat_search],
           "probe_state_samples": [round(x, 6) for x in lat_state]}
    # The residual the propagation reading actually rests on. Reporting it
    # directly beats inferring resolvability from a binary threshold note.
    resid = (c["p95"] - lq["p95"]) if (c["p95"] is not None and lq["p95"] is not None) else None
    return {"delta_hide_p50": h["p50"], "delta_hide_p95": h["p95"],
            "delta_hide_p99": h["p99"], "delta_hide_max": h["max"],
            "delta_ack_p50": a["p50"], "delta_ack_p95": a["p95"],
            "delta_state_visible_p50": sv["p50"], "delta_state_visible_p95": sv["p95"],
            "delta_confirm_p50": c["p50"], "delta_confirm_p95": c["p95"],
            "delta_confirm_p99": c["p99"],
            "confirm_minus_probe_p95": resid, **raw,
            "probe_state_p50": ls["p50"], "probe_state_p95": ls["p95"],
            "probe_search_p50": lq["p50"], "probe_search_p95": lq["p95"],
            "probe_search_p99": lq["p99"], "probe_scheduling_lag_p95": lg["p95"],
            "n_probe_state": ls["n"], "n_probe_search": lq["n"],
            "writer_client_id": session_shared, "state_probe_client_id": session_shared,
            "search_probe_client_id": session_shared,
            # The level actually put ON each probe request, not just declared on
            # the collection. If this is null the sweep did not exercise its axis.
            "consistency_sent_on_request": consistency,
            "n_observed": h["n"], "n_censored": cens_search,
            "n_censored_state": cens_state, "probe_resolution_s": probe_int}


async def run_s2(uri, quick, full, n_items_override=None, out=None):
    out = [] if out is None else out
    levels = (["Strong"] if _smoke()
              else (["Strong", "Bounded"] if not full
                    else ["Strong", "Bounded", "Session", "Eventually"]))
    # 25 events makes P95 the 24th value and P99 the maximum itself. Raise this
    # (--s2-items) before quoting anything above a median. Each event costs one
    # stagger interval, so 200 events is ~25 s per cell.
    n_items = n_items_override or (6 if _smoke() else (25 if not full else 60))
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
    # Milvus 2.4 HNSW requires ef >= top_k, and the post-filter search asks for
    # over_fetch*K = 20 candidates. ef=16 was therefore an ILLEGAL operating
    # point: out of range, with server-side behaviour (reject, clamp, or default)
    # that is not part of the contract. 20 is the legal minimum and is kept as
    # the most-degraded point; see the assertion in s5_one, which now refuses an
    # out-of-range grid instead of silently measuring something undefined.
    g = [("FLAT", FLAT),
         ("HNSW_ef20", {"index_type": "HNSW", "index_params": {"M": 16, "efConstruction": 200},
                        "search_params": {"ef": 20}}),
         ("HNSW_ef64", {"index_type": "HNSW", "index_params": {"M": 16, "efConstruction": 200},
                        "search_params": {"ef": 64}})]
    if full:
        g.append(("HNSW_ef256", {"index_type": "HNSW",
                                 "index_params": {"M": 16, "efConstruction": 200},
                                 "search_params": {"ef": 256}}))
    return g


def s5_one(uri, label, idx, N, queries, over_fetch, hidden_mode="random",
           hidden_ratio=0.30):
    """Hidden-set workloads at the paper's 384-d embedding size. Ground truth is
    exact brute force over the SAME vectors, computed with numpy -- at 384-d the
    pure-Python cosine loop of gate2 is not viable."""
    import numpy as np
    # Milvus 2.4 HNSW defines ef over [top_k, int_max]. Below top_k the server's
    # behaviour is out of contract, so the cell would measure something the
    # documentation does not define -- and it looks exactly like a legitimate
    # low-recall result. Refuse rather than record it.
    _ef = (idx.get("search_params") or {}).get("ef")
    _limit = over_fetch * K
    if _ef is not None and _ef < _limit:
        raise SystemExit(
            f"illegal operating point {label}: ef={_ef} < top_k={_limit} "
            f"(= over_fetch {over_fetch} x K {K}). Milvus 2.4 HNSW requires "
            "ef >= top_k; raise ef or lower over_fetch.")
    V = vecs_np(N, DIM, seed=0)
    Q = vecs_np(queries, DIM, seed=1)                  # evaluation queries (held out)
    A = vecs_np(queries, DIM, seed=11)                 # attack queries, disjoint from Q
    S = Q @ V.T                                        # cosine (both normalized)

    n_hidden = int(hidden_ratio * N)
    if hidden_mode == "random":
        hidden_arr = np.random.default_rng(2).choice(N, n_hidden, replace=False)
    elif hidden_mode == "clustered":                   # lineage burst: semantic cluster
        seeds = np.random.default_rng(3).choice(N, 5, replace=False)
        hidden_arr = np.argsort(-(V @ V[seeds].T).max(axis=1))[:n_hidden]
    elif hidden_mode == "heldout_attack_ranked":
        # Chosen from an INDEPENDENT attack-query set: selecting the hidden set with
        # the evaluation queries' own similarity matrix leaks the test set into the
        # workload. That leak is what made this mode degenerate at 384-d -- it hid
        # every evaluation query's entire neighbourhood (0 eligible in any top-20),
        # so under-fill was 100% by construction. Still an adversarial upper bound,
        # but now a legitimate one.
        hidden_arr = np.argsort(-(A @ V.T).max(axis=0))[:n_hidden]
    else:
        raise SystemExit(f"unknown hidden mode {hidden_mode}")
    hidden = set(int(x) for x in hidden_arr)

    Sg = S.copy()
    Sg[:, hidden_arr] = -np.inf                        # ground truth over ELIGIBLE only
    part = np.argpartition(-Sg, K, axis=1)[:, :K]
    rows = np.arange(queries)[:, None]
    gt = part[rows, np.argsort(-Sg[rows, part], axis=1)]

    # Feasibility of the workload itself, independent of any backend: how many
    # eligible items sit in the exact candidate pool a post-filter would fetch.
    # If this is ~0, under-fill is guaranteed by construction and the cell
    # measures the workload, not the system.
    pool_exact = np.argpartition(-S, over_fetch * K, axis=1)[:, :over_fetch * K]
    elig_in_pool = np.array([sum(1 for i in row if int(i) not in hidden)
                             for row in pool_exact])
    feas = {"eligible_in_pool_mean": float(elig_in_pool.mean()),
            "queries_with_full_k_available": int((elig_in_pool >= K).sum()),
            "eligible_total": N - len(hidden)}

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
    # cond_recall_n is the denominator behind "no recall loss observed": with zero
    # imperfect queries, the 95% upper bound on the loss rate is only ~3/n.
    return {**feas,
            "cond_recall_n": len(cond),
            "cond_recall_imperfect": sum(1 for r in cond if r < 1.0),
            "underfill": underfill / queries,
            "hM50": pctl(hM, 0.50), "hM95": pctl(hM, 0.95),
            "pf_recall": sum(pf_rec) / len(pf_rec),
            "cond_recall": (sum(cond) / len(cond)) if cond else float("nan"),
            "if_recall": sum(if_rec) / len(if_rec),
            "query_p50": lp["p50"], "query_p95": lp["p95"], "query_p99": lp["p99"],
            "infilter_p99": li["p99"], **info}


def run_s5(uri, quick, full, out=None):
    out = [] if out is None else out
    Ns = [300] if _smoke() else ([20000] if not full else [20000, 100000])
    # random is the workload that measures the system; clustered is the realistic
    # lineage-burst adversary and the one the paper's tail argument rests on.
    # heldout_attack_ranked selects the hidden set with an INDEPENDENT attack-query
    # set (seed 11) and is scored on held-out evaluation queries (seed 1). The
    # earlier query-coupled version of this mode was degenerate at 384-d (it
    # emptied every candidate pool), but that was the leakage, not the workload:
    # once the query sets are disjoint it is expected to resemble random, and that
    # negative result is the finding, not a failure. It stays --full-only because
    # it is a generalization check, not a system measurement. The oracle upper
    # bound lives in gate2 under a deliberately different name
    # (oracle_rank_coupled) so the two can never be compared by accident.
    # Smoke runs random+clustered (not just random): the clustered branch builds
    # the hidden set differently, and the validator's tail-pressure check needs
    # both arms present. At N=300 the second mode is nearly free, and a defect in
    # clustered hiding should not first appear in a 30-minute standalone run.
    modes = (["random", "clustered"] if _smoke() else
             (["random", "clustered"] if not full
              else ["random", "clustered", "heldout_attack_ranked"]))
    # Queries are cheap relative to index build, and they are what bounds the
    # recall claim: 100 -> <=3% loss rate at 95%, 200 -> <=1.5% (rule of three).
    queries = 20 if _smoke() else (100 if not full else 200)
    grid = s5_grid(full)[:1] if _smoke() else s5_grid(full)
    for N in Ns:
        for mode in modes:
            for label, idx in grid:
                print(f"  S5 {label} N={N} mode={mode} queries={queries} ...", flush=True)
                m = s5_one(uri, f"{label}_{mode}", idx, N, queries, over_fetch=2,
                           hidden_mode=mode)
                if m["queries_with_full_k_available"] == 0:
                    print(f"     !! DEGENERATE: 0 of {queries} queries have k eligible "
                          f"items in the candidate pool; this cell measures the "
                          f"workload, not the system")
                out.append(rec(f"S5/{label}/{mode}/N{N}",
                           {"label": label, "hidden_mode": mode, "N": N,
                            "queries": queries, "k": K,
                            "query_sampling": "iid standard normal, L2-normalized, "
                                              "eval seed=1, attack seed=11 (disjoint); "
                                              "hidden set never uses eval queries",
                            "dim": DIM, "over_fetch": 2,
                            "hidden_ratio": 0.30, "consistency": "Bounded", **idx},
                           m, index_type=idx["index_type"]))
                cr = m["cond_recall"]
                print(f"     -> cond_recall {'n/a' if cr != cr else f'{cr:.4f}'}, "
                      f"if_recall {m['if_recall']:.4f}, "
                      f"elig_in_pool {m['eligible_in_pool_mean']:.1f}, "
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
        nb = r["config"].get("blocks", len(m.get("ratios", [])))   # blocks live in config
        print(f"\n  paired B4/B1 (conc {r['config']['concurrency']}, {nb} blocks)")
        print(f"    throughput ratio: median {m['ratio_median']:.3f}, "
              f"range {m['ratio_min']:.3f}-{m['ratio_max']:.3f}, "
              f"same direction: {m['same_direction']}")
        if m.get("p99_delta_median") is not None:
            print(f"    P99 delta (B4-B1): median {1000*m['p99_delta_median']:.1f} ms, "
                  f"range {1000*m['p99_delta_min']:.1f}..{1000*m['p99_delta_max']:.1f} ms, "
                  f"same direction: {m['p99_same_direction']}")
        print("  A ratio band spanning 1.0 (or a P99 delta band spanning 0) means the")
        print("  protocol's cost was not resolved above run-to-run variability -- report")
        print("  that, not a point figure.")

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
        print(f"  {'index':>14} {'mode':>10} {'N':>7} {'underfill':>10} {'pf_rec':>7} "
              f"{'cond_rec':>9} {'if_rec':>7} {'elig/pool':>10} {'P99 ms':>8}  idx/state/load")
        for r in sorted(s5, key=lambda r: r["experiment_id"]):
            m, c = r["metrics"], r["config"]
            cr = "n/a" if m["cond_recall"] is None else f"{m['cond_recall']:.4f}"
            eip = m.get('eligible_in_pool_mean')
            print(f"  {c['label']:>14} {c.get('hidden_mode','?'):>10} {c['N']:>7} "
                  f"{m['underfill']*100:9.1f}% {m['pf_recall']:7.3f} {cr:>9} "
                  f"{m['if_recall']:7.3f} {'n/a' if eip is None else f'{eip:10.1f}'} "
                  f"{1000*m['query_p99']:8.1f}  {m['index_type_effective']}/"
                  f"{m['index_state']}/{m['load_state']}")
            if eip is not None and m.get('queries_with_full_k_available') == 0:
                print(f"                 !! DEGENERATE workload: no query has k eligible "
                      f"items in its candidate pool")
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
    ap.add_argument("--s1-blocks", type=int, default=4,
                    help="S1 blocks; MUST be a multiple of 4 so the Williams "
                         "square stays position- and carryover-balanced")
    ap.add_argument("--s2-items", type=int, default=None,
                    help="hide events per S2 cell; raise above the default 25 "
                         "before quoting any S2 percentile (200 for P95, 1000 for P99)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting an existing results file (default is "
                         "to refuse, so a prior run's evidence is never truncated)")
    a = ap.parse_args()

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    if a.smoke:
        CTX.update(index=FLAT, evidence="milvus_lite_flat", prefix="smoke")
        if a.uri.startswith("http"):
            a.uri = os.path.join(HERE, "results", "smoke.db")
    if a.s1_blocks % 4:
        raise SystemExit(f"--s1-blocks must be a multiple of 4 (got {a.s1_blocks}); "
                         "otherwise a baseline gets an extra turn in one position "
                         "and the position effect contaminates the treatment effect")
    S1_BLOCKS[0] = a.s1_blocks
    CTX["profile"] = ("smoke" if a.smoke else
                      "full" if a.full else "quick" if a.quick else "default")
    preflight(a.uri)
    # Per-experiment default path. A single shared standalone.json meant that
    # running S2 on its own would truncate the file and destroy the S1 records
    # from an earlier invocation -- and S1's raw latency samples are not
    # recoverable offline, so that loss costs a full re-run.
    out = a.out or os.path.join(
        HERE, "results",
        "smoke.json" if a.smoke else f"standalone-{a.exp}.json")
    if not a.smoke and not a.force and os.path.exists(out):
        raise SystemExit(
            f"refusing to overwrite {out}\n"
            "  That file is prior evidence. Move or rename it first (results from a\n"
            "  superseded run are kept, not deleted), or pass --out/--force.")
    recs, failed = [], []

    # S2 before S1: S2 unlocks the deadline/visibility result the paper's thesis
    # rests on, S1 the throughput/P99 result, and S5 is a secondary ANN study.
    # If the environment breaks partway, this order sacrifices the least.
    plan = ["S2", "S1", "S5"] if a.exp == "all" else [a.exp]
    tier = ("E2 SMOKE on Milvus Lite -- code-path validation only, NOT a §7.5 result"
            if a.smoke else f"evidence tier E3{', FULL grid' if a.full else ''}")
    print(f"Experiments {plan}  ({tier})\n")
    for exp in plan:
        print(f"[{exp}]", flush=True)
        before = len(recs)
        try:
            if exp == "S1":
                await run_s1(a.uri, a.quick, a.full, out=recs)
            elif exp == "S2":
                await run_s2(a.uri, a.quick, a.full, a.s2_items, out=recs)
            else:
                run_s5(a.uri, a.quick, a.full, out=recs)
        except Exception as e:
            # Cells finished before the failure stay in recs -- losing 3 of 4
            # consistency levels because the 4th raised is not acceptable.
            failed.append((exp, repr(e)))
            print(f"  !! {exp} FAILED after {len(recs)} records: {e!r}")
        # A runner that completes without contributing a record is a wiring bug
        # (it happened: run_s1 rebound `out` to a fresh list), not an empty grid.
        if len(recs) == before and not any(e == exp for e, _ in failed):
            failed.append((exp, "produced 0 records -- runner did not append to `recs`"))
            print(f"  !! {exp} produced 0 records (wiring bug, not an empty grid)")
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
