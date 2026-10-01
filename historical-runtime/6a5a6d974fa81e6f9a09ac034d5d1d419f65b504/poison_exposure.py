#!/usr/bin/env python3
"""
W2 -- end-to-end poison exposure and containment.

Closes the hole the paper opens itself: §III says poisoning exposure E_p "is
therefore measured empirically (Sec. VII)", §VI-B declares workload W2 (poison
burst), and §VII reports neither.

How these numbers may be read is frozen in W2-PREREGISTRATION.md, written
before this ran (and amended, on the record, before the first full run). The
short version:

  * That B4 hides at the deadline is CONSTRUCTED (I3/I4, 20/20 tests). A curve
    showing it is not a finding. What is measured here and nowhere else is how
    many retrievals were actually poisoned before containment, how that responds
    to verifier backlog, and what clean freshness and clean recall it costs --
    all four baselines in one run, in one unit system.
  * THREE query sets, because measuring only on the queries the poison was built
    against can prove nothing except an oracle worst case:
      Q_craft    the queries the poison was optimised against -> upper bound,
                 worst case BY CONSTRUCTION (same status as the oracle
                 rank-coupled hidden set in §VI-C)
      Q_target   different queries from the SAME topic, never used to build
                 poison -> the headline end-to-end result; a transfer
                 measurement, not a construction
      Q_negative unrelated topics -> contamination-spread check, expected ~0
  * E_p keeps its §III definition, t_contain - t_first-poison-visible. The
    observed first/last poisoned retrieval is a separate, interval-censored
    quantity and is never called E_p. A baseline that never contains (B1) is
    RIGHT-CENSORED by the window, not "exposed for 7 seconds".

Run:
  python3 poison_exposure.py --backend inmemory                  # E1 exact
  python3 poison_exposure.py --backend milvus --uri /tmp/w2.db   # E2 Milvus Lite
  python3 poison_exposure.py --smoke                             # code-path check
"""

import argparse
import asyncio
import json
import math
import os
import random
import statistics
import time

from backend import InMemoryBackend, cosine
from functional_slice import State, System, VISIBLE

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_VERSION = "1.0"
EXPERIMENT_ID = "W2"

# Identical to standalone_experiments.BASELINES. Kept local so an E1 run does
# not pull the standalone runner's pymilvus import path in; test_w2.py asserts
# the two definitions still agree.
BASELINES = {
    "B1": {"desc": "immediate admission, no verification", "verify": False, "sync": False, "deadline": False},
    "B2": {"desc": "synchronous verify-before-visible",    "verify": True,  "sync": True,  "deadline": False},
    "B3": {"desc": "async visible, no deadline",           "verify": True,  "sync": False, "deadline": False},
    "B4": {"desc": "fail-closed provisional visibility",   "verify": True,  "sync": False, "deadline": True},
}

CFG = {
    "dim": 32, "corpus": 600, "k": 5, "over_fetch": 3,
    "n_topics": 8, "topic_sigma": 0.15, "poison_jitter": 0.05,
    "qps": 24, "dur": 8.0, "inject_at": 1.0,
    "n_poison": 6, "n_clean": 6,
    "n_craft_q": 8, "n_target_q": 8, "n_neg_q": 8,
    # Every craft and target query is scored, not every fourth. Sampling was a
    # concession to a 600-cosine scan that no longer happens: with the corpus
    # half precomputed per seed the online cost is a twelve-cosine merge, so
    # cumulative displacement can be summed exactly instead of extrapolated
    # from a sample.
    "recall_every": 1,
    # Primary keys never repeat anywhere in a run, corpus included. Each
    # seed owns [seed*pk_stride, seed*pk_stride + pk_stride).
    "pk_stride": 1_000_000,
    "verify_cost": 0.3, "tp": 1.0, "bin": 0.25,
    # Backlog is real queueing, not a longer sleep: a semaphore of 1 plus items
    # already in the queue makes the poison wait behind them, which is the
    # condition I6 exists for. `normal` has enough concurrency that nothing
    # queues at all. heavy: 12 filler items x 0.3 s = ~3.6 s of queue.
    "backlog": {"normal": {"conc": 8, "items": 0},
                "heavy":  {"conc": 1, "items": 12}},
    # T_p points are anchored to the verifier, not chosen for looks. With
    # verify_cost 0.3 s and a heavy queue of ~3.6 s:
    #   0.15  deadline FASTER than normal verification -> even clean content is
    #         hidden before it is vetted, the aggressive-freshness extreme
    #   0.3   deadline == normal verification latency, the crossover
    #   1.0   between normal and heavy: slack when the verifier keeps up,
    #         binding when it does not  (the main grid's T_p)
    #   2.0   comfortably above normal latency, still well below the heavy queue
    "tp_sweep": [0.15, 0.3, 1.0, 2.0],
    "seeds": [1, 2, 3, 4, 5],
}


# ---------------------------------------------------------------- vectors ---
def unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def rvec(rng, dim):
    return unit([rng.gauss(0, 1) for _ in range(dim)])


def perturb(rng, base, sigma):
    return unit([x + rng.gauss(0, sigma) for x in base])


def make_world(seed, cfg):
    """Corpus, the three query sets and the poison for one seed. Identical
    across all four baselines at that seed -- the baselines differ only in
    admission policy, never in what they are shown."""
    rng = random.Random(seed)
    dim, T, sig = cfg["dim"], cfg["n_topics"], cfg["topic_sigma"]
    centers = [rvec(rng, dim) for _ in range(T)]

    corpus = [perturb(rng, centers[i % T], sig) for i in range(cfg["corpus"])]
    # Q_craft and Q_target are drawn from the SAME topic and are disjoint. The
    # poison is optimised against Q_craft only, so any hit on Q_target is
    # transfer within the topic rather than the construction coming back.
    q_craft = [perturb(rng, centers[0], sig) for _ in range(cfg["n_craft_q"])]
    q_target = [perturb(rng, centers[0], sig) for _ in range(cfg["n_target_q"])]
    q_neg = [perturb(rng, centers[1 + i % (T - 1)], sig)
             for i in range(cfg["n_neg_q"])]
    poison = [perturb(rng, q_craft[i % len(q_craft)], cfg["poison_jitter"])
              for i in range(cfg["n_poison"])]
    clean = [perturb(rng, centers[0], sig) for _ in range(cfg["n_clean"])]
    filler = [rvec(rng, dim)
              for _ in range(max(b["items"] for b in cfg["backlog"].values()))]

    def mean_cos(A, B):
        return round(statistics.mean(cosine(a, b) for a in A for b in B), 4)

    def kth_cos(qs, k):
        """The bar the poison has to clear: the k-th best corpus similarity for
        a query. The MEAN corpus similarity is not the competition -- most of
        the corpus is in other topics -- so reporting it would understate how
        hard entering the top-k is."""
        out = []
        for q in qs:
            s = sorted((cosine(q, c) for c in corpus), reverse=True)
            out.append(s[k - 1])
        return round(statistics.median(out), 4)

    # Reported, not tuned: a reader can see exactly how hard the transfer was
    # instead of taking "same topic" on faith. poison_vs_own_craft pairs each
    # poison with the query it was actually built against; averaging it over all
    # craft queries would dilute a 0.99 with three unrelated 0.57s.
    geometry = {
        "craft_vs_target": mean_cos(q_craft, q_target),
        "poison_vs_own_craft": round(statistics.mean(
            cosine(poison[i], q_craft[i % len(q_craft)])
            for i in range(len(poison))), 4),
        "poison_vs_target": mean_cos(poison, q_target),
        "poison_vs_negative": mean_cos(poison, q_neg),
        "corpus_kth_cos_target": kth_cos(q_target, cfg["k"]),
        "corpus_kth_cos_negative": kth_cos(q_neg, cfg["k"]),
    }
    return {"corpus": corpus, "q_craft": q_craft, "q_target": q_target,
            "q_neg": q_neg, "poison": poison, "clean": clean, "filler": filler,
            "geometry": geometry}


# ------------------------------------------------------------------- cell ---
def id_base(seed, cfg):
    """Primary keys are unique across the WHOLE run, corpus included.

    Cell items were already unique (a monotonic counter that is not reset
    between cells), but the corpus was re-inserted at ids 0..N-1 for every seed,
    with different vectors each time. That reuse is the one leak the residual
    check structurally could not see: a stale row under id 42 from an earlier
    seed is a key the CURRENT control store lists as TRUSTED, so post-filtering
    would hand it back as though it were this seed's item 42. Giving each seed
    its own key space removes the aliasing class outright, and it does not touch
    the query path."""
    return seed * cfg["pk_stride"]


def seed_control_store(sys, corpus, base):
    """The corpus is pre-existing trusted content. Without this the post-filter
    would drop every corpus id as unknown and the top-k would be empty."""
    for i, v in enumerate(corpus):
        sys.items[base + i] = {"id": base + i, "vec": v, "state": State.TRUSTED,
                               "prov": 0, "pending": False,
                               "content_bad": False, "ts": {}}


def corpus_topk(corpus, q, k, base):
    """The corpus half of the ground truth, computed ONCE per seed, before any
    cell is timed. The corpus never changes state during a cell, so this is
    constant for the whole seed. Emits BACKEND ids, not list positions, so it
    can be compared directly with what the query path returns."""
    cand = sorted(((cosine(q, v), base + i) for i, v in enumerate(corpus)),
                  reverse=True)
    return cand[:k]


def merged_topk(pre, sys, q, k, inserted, exclude=()):
    """Exact top-k over (corpus union currently-eligible ingested items).

    Identical to a full scan, and provably so: a corpus item in the true top-k
    must be among the corpus's own top-k, so carrying those k is enough. That
    matters because the full scan is 600 cosines -- about 5 ms -- and running it
    inside the measurement window blocks the event loop, which is where the
    deadline scheduler lives. The measured overshoot of E_p over T_p (1.8 ms at
    T_p = 0.15 s, 7.5 ms at 2 s) is the size of exactly that interference. Here
    at most twelve items are ever ingested, so the online half is a twelve-cosine
    scan and does not need a thread, a lock, or a snapshot.

    A reference frozen entirely before the run would be wrong rather than
    merely cheap: clean items are ingested DURING the window, and B2's whole
    cost is that its clean items do not appear until the verifier drains.
    """
    cand = list(pre)
    for i in inserted:
        if i in exclude:
            continue
        it = sys.items.get(i)
        if it is not None and it["state"] in VISIBLE:
            cand.append((cosine(q, it["vec"]), i))
    cand.sort(reverse=True)
    return [i for _, i in cand[:k]]


async def run_cell(backend, base, backlog_name, Tp, seed, cfg, world, base_id, pre):
    bl = BASELINES[base]
    bk = cfg["backlog"][backlog_name]
    sys = System(backend, mode="postfilter", Tp=Tp, verify_cost=cfg["verify_cost"],
                 over_fetch=cfg["over_fetch"], decouple=True,
                 verifier_concurrency=bk["conc"],
                 verify=bl["verify"], sync=bl["sync"], deadline=bl["deadline"])
    idbase = id_base(seed, cfg)
    assert idbase <= base_id < idbase + cfg["pk_stride"], (
        "primary keys have run past this seed's stride and would collide "
        "with the next seed's space")
    seed_control_store(sys, world["corpus"], idbase)

    poison_ids, clean_ids, filler_ids = [], [], []
    qlog = []                       # (t, kind, qidx, hit, recall, displacement)
    # Checked at BOTH ends. A row the store could not see when the previous cell
    # swept for it can surface afterwards, so a clean sweep is not by itself
    # proof that this cell started clean.
    residual_at_start = backend_residual(backend, cfg, idbase)
    qstats = {"foreign_candidates": 0, "queries": 0, "underfill": 0}
    t0 = time.monotonic()

    async def queries():
        interval = 1.0 / cfg["qps"]
        sets = [("craft", world["q_craft"]), ("target", world["q_target"]),
                ("negative", world["q_neg"])]
        i = 0
        while True:
            t = time.monotonic() - t0
            if t >= cfg["dur"]:
                return
            kind, pool = sets[i % 3]            # equal power on all three sets
            qidx = (i // 3) % len(pool)
            q = pool[qidx]
            ids = await asyncio.to_thread(sys.query, q, cfg["k"], qstats)
            rec = disp = None
            # Sampled on craft AND target. Sampling only target reported 0.0
            # displacement everywhere and read as "the poison is harmless" --
            # but the poison mostly does not reach a target query's top-k at
            # all, so the set where displacement actually happens was the one
            # not being measured.
            if kind in ("craft", "target") and qidx % cfg["recall_every"] == 0:
                pset = set(poison_ids)
                ins = poison_ids + clean_ids + filler_ids
                # Inline, not in a thread: the expensive half was precomputed
                # per seed, so what is left is a twelve-cosine merge.
                truth = merged_topk(pre[(kind, qidx)], sys, q, cfg["k"], ins)
                # Eligible Recall@k (§VI-C): does the query path return the
                # exact eligible top-k. Poison is eligible while exposed, so
                # this stays 1.0 during exposure -- it is a correctness check,
                # not a measure of harm.
                rec = (len(set(ids) & set(truth)) / len(truth)) if truth else None
                # Displacement against a POISON-FREE reference: how much of the
                # top-k the query would have had, had the poison never been
                # admitted, the poison pushed out. The reference tracks the
                # currently-eligible set and therefore MOVES as clean items are
                # ingested -- deliberately. Freezing it to the static corpus
                # would score a clean item's legitimate arrival as displacement,
                # and would erase B2's entire cost, since B2's clean items do
                # not appear until the verifier drains.
                clean = merged_topk(pre[(kind, qidx)], sys, q, cfg["k"], ins, pset)
                disp = (len(set(clean) - set(ids)) / len(clean)) if clean else None
            qlog.append((time.monotonic() - t0, kind, qidx,
                         any(j in poison_ids for j in ids), rec, disp))
            i += 1
            slack = (i + 1) * interval - (time.monotonic() - t0)
            if slack > 0:
                await asyncio.sleep(slack)

    async def ingest():
        nonlocal base_id
        await asyncio.sleep(cfg["inject_at"])
        # Filler first: it is what the poison's verification has to queue
        # behind. With conc=8 and items=0 this loop does nothing. The filler is
        # drawn once per seed, so `heavy` shows every baseline the same queue.
        for v in world["filler"][:bk["items"]]:
            sys.admit(base_id, v, content_bad=False)
            filler_ids.append(base_id)
            base_id += 1
        for v in world["poison"]:
            sys.admit(base_id, v, content_bad=True)
            poison_ids.append(base_id)
            base_id += 1
        for v in world["clean"]:
            sys.admit(base_id, v, content_bad=False)
            clean_ids.append(base_id)
            base_id += 1

    await asyncio.gather(queries(), ingest())
    # Stop the control plane BEFORE summarising and sweeping. Without this a
    # B2 admission still queued behind the heavy verifier resumes during the
    # next cell and inserts its item there, after this cell's sweep has already
    # run -- twelve leaked rows per heavy cell, caught by gate 5.
    bg_errors = await sys.shutdown()
    m = summarise(sys, qlog, poison_ids, clean_ids, cfg, Tp, t0)
    delete_errors, sweep_attempts = await sweep(
        backend, poison_ids + clean_ids + filler_ids, cfg, idbase)
    m.update(residual_at_start=residual_at_start, sweep_attempts=sweep_attempts,
             foreign_candidates=qstats["foreign_candidates"],
             candidate_queries=qstats["queries"],
             underfill_queries=qstats["underfill"],
             baseline=base, backlog=backlog_name, Tp=Tp, seed=seed,
             next_id=base_id, delete_errors=delete_errors,
             residual_rows=backend_residual(backend, cfg, idbase),
             # Nothing of this cell's control plane may still be armed: a
             # surviving task or timer is what leaked rows into the next cell
             # before shutdown() existed, and row counting alone would not see
             # it until after the damage.
             live_tasks=len([t for t in sys.tasks if not t.done()]),
             live_timers=len(sys.timers),
             bg_errors=[repr(e) for e in bg_errors])
    return m


async def sweep(backend, ids, cfg, base, attempts=8, wait=0.05):
    """Delete a cell's rows and VERIFY they are gone, rather than call delete
    once and assume it worked.

    Milvus applies a delete against what it can currently see, and the rows
    inserted most recently are not always visible yet. Measured on Milvus Lite
    with twelve ids inserted and immediately deleted, every form left the newest
    ids behind: per-id filter left 3, a primary-key list left 4, one range
    filter left 1, and inserting a flush first did not help (5 and 2). The
    survivors were always the highest ids -- the signature of write-visibility
    lag, not of a wrong delete API. So the sweep loops until the collection
    says it is clean, and reports how many passes that took. A cell's poison
    sits at cosine ~0.96 to the craft queries; one leaked row eats an
    over-fetch slot in every cell that follows.
    """
    errs = 0
    for attempt in range(attempts):
        for i in ids:
            try:
                backend.delete(i)
            except Exception:
                errs += 1
        if hasattr(backend, "flush"):
            try:
                backend.flush()
            except Exception:
                pass
        if backend_residual(backend, cfg, base) == 0:
            return errs, attempt + 1
        await asyncio.sleep(wait)
    return errs, attempts


def backend_residual(backend, cfg, base):
    """DISTINCT ids at or above the corpus that a cell failed to clean up.

    Not `count(*)` minus the corpus size, which is what this asked first and
    which reported 2400 phantom rows on Milvus Lite: `count(*)` includes the
    superseded version of every overwritten primary key until compaction runs.
    Direct probing showed the collection was in fact clean -- every id resolved
    to exactly one entity and no id >= corpus existed at all -- so the gate, not
    the data, was wrong. Counting the ids that should not be there measures the
    invariant the gate exists for, and a same-size-but-wrong-contents leak (12
    rows gone, 12 different rows left) fails it where a row count would not.
    """
    # Anything that is not this seed's corpus. Evaluated only between cells,
    # where the corpus is the entire legitimate contents, so this is exactly
    # "not in the active id set" -- a previous CELL's items live at
    # base+corpus+n and fall outside the window too.
    lo, hi = base, base + cfg["corpus"]
    if hasattr(backend, "v"):
        return sum(1 for i in backend.v if not (lo <= i < hi))
    try:
        rows = backend.client.query(backend.coll,
                                    filter=f"id < {lo} or id >= {hi}",
                                    output_fields=["id"], limit=8192)
        return len({r["id"] for r in rows})
    except Exception:
        return None


def reset_corpus(backend, corpus, cfg, base):
    """Put the fixed collection back to exactly the corpus before each seed.

    All four baselines have to start from one snapshot, and re-inserting over
    the same primary keys does not give that: it relies on how the store
    shadows an overwritten key, and it grows the physical row count once per
    seed. Delete first, then insert once. The collection itself is never
    dropped -- that is the S1 lesson this experiment is built around."""
    if hasattr(backend, "v"):
        backend.v.clear()
        backend.vis.clear()
    else:
        backend.client.delete(backend.coll, filter="id >= 0")
    for i, v in enumerate(corpus):
        backend.insert(base + i, v, visible=True)
    if hasattr(backend, "wait_index"):
        backend.wait_index()


def _set_metrics(rows, cfg, w):
    """N_p, PRR, AUC_p and the observed window for one query set."""
    hits = [t for t, _, hit, _, _ in rows if hit]
    bins, per_q = {}, {}
    for t, qidx, hit, _, _ in rows:
        b = int(t / w)
        n, h = bins.get(b, (0, 0))
        bins[b] = (n + 1, h + int(hit))
        n, h = per_q.get(qidx, (0, 0))
        per_q[qidx] = (n + 1, h + int(hit))    # for the conditional analysis
    return {
        # Which individual queries the poison reached, kept so the defence
        # effect can later be computed on the subset B1 actually compromised
        # rather than diluted by queries the attack never reached.
        "per_query": {str(q): {"n": n, "hits": h} for q, (n, h) in sorted(per_q.items())},
        "queries": len(rows),
        "Np": len(hits),                               # poisoned retrievals
        "prr": round(len(hits) / max(1, len(rows)), 4),
        "auc_query_seconds": round(sum(h / n * w for n, h in bins.values()), 4),
        "series": [{"t": round(b * w, 3), "prr": round(h / n, 4), "n": n}
                   for b, (n, h) in sorted(bins.items())],
        "obs_first": round(min(hits), 4) if hits else None,
        "obs_last": round(max(hits), 4) if hits else None,
    }


def summarise(sys, qlog, poison_ids, clean_ids, cfg, Tp, t0):
    by = {k: [(t, qi, hit, rec, dsp) for t, kk, qi, hit, rec, dsp in qlog if kk == k]
          for k in ("craft", "target", "negative")}
    sets = {k: _set_metrics(v, cfg, cfg["bin"]) for k, v in by.items()}

    # E_p (§III): t_contain - t_first-poison-visible, per poisoned item.
    # Never visible -> contributes nothing (B2). Never contained -> RIGHT-
    # CENSORED by the window (B1) and counted, never averaged in as if the
    # exposure had ended when the measurement did.
    ep, ep_censored, never_visible, contain_rel = [], 0, 0, []
    for i in poison_ids:
        ts = sys.items[i]["ts"]
        if "visible" not in ts:
            never_visible += 1
            continue
        if "contain" in ts:
            ep.append(ts["contain"] - ts["visible"])
            contain_rel.append(ts["contain"] - t0)
        else:
            ep_censored += 1

    # Clean freshness. An item still in the verifier queue when the window
    # closed is right-censored too: reporting only the ones that made it would
    # make B2 under backlog look fast by dropping exactly what its queue delayed.
    df = [sys.items[i]["ts"]["visible"] - sys.items[i]["ts"]["arrival"]
          for i in clean_ids if "visible" in sys.items[i]["ts"]]
    df_censored = sum(1 for i in clean_ids if "visible" not in sys.items[i]["ts"])

    # Eligible Recall@k against exact ground truth (§VI-C definition), split at
    # containment: during exposure the poison legitimately occupies eligible
    # slots, so the question is whether recall RETURNS once containment lands.
    c_med = statistics.median(contain_rel) if contain_rel else None

    # Which samples belong to "during exposure" has three cases, and getting
    # them wrong silently swaps B1's and B2's numbers:
    #   B2  no poison was ever visible  -> there is no exposure phase at all
    #   B1  poison visible, never contained -> everything after injection is
    #       exposure, and nothing is "after containment"
    #   B3/B4  split at the median containment instant
    exposed = never_visible < len(poison_ids)

    def phase_split(samples):
        after_inject = [(t, v) for t, v in samples if t >= cfg["inject_at"]]
        if not exposed:
            return [], [v for _, v in after_inject]
        if c_med is None:
            return [v for _, v in after_inject], []
        return ([v for t, v in after_inject if t <= c_med],
                [v for t, v in after_inject if t > c_med])

    # Displacement is only meaningful once the poison exists; before injection
    # there is nothing to displace and the zeros would dilute the figure.
    phased = {}
    for st in ("craft", "target"):
        phased[st] = {
            "rec": phase_split([(t, r) for t, _, _, r, _ in by[st] if r is not None]),
            "dsp": phase_split([(t, d) for t, _, _, _, d in by[st] if d is not None]),
        }
    rec_during, rec_after = phased["target"]["rec"]
    dsp_during, dsp_after = phased["target"]["dsp"]
    dspc_during, dspc_after = phased["craft"]["dsp"]

    # Cumulative displacement over the fixed window: how many top-k positions
    # the poison took in total, summed over every query in the window rather
    # than averaged per query. This is the metric that separates B4 from B1 in
    # HARM: while a poison item is visible both lose the same fraction of their
    # top-k, so only duration distinguishes them, and only a cumulative figure
    # shows it. Exact, not extrapolated -- every craft and target query is
    # scored.
    def cum_disp(st):
        s = [d for t, _, _, _, d in by[st]
             if d is not None and t >= cfg["inject_at"]]
        return round(sum(s) * cfg["k"], 3), len(s)

    dh_craft, n_dh_craft = cum_disp("craft")
    dh_target, n_dh_target = cum_disp("target")

    # Gap between authoritative containment and the last poisoned retrieval.
    # In post-filter mode the control store is authoritative, so this is bounded
    # by the query interval BY CONSTRUCTION -- it measures the censoring, not a
    # system delay. The in-index path's real propagation delay is §VII-D/§VII-E.
    last_t = sets["craft"]["obs_last"]
    gap = round(c_med - last_t, 4) if (c_med is not None and last_t is not None) else None

    prov_ok = all(sys.items[i]["prov"] <= 1 for i in poison_ids + clean_ids)
    watched = set(poison_ids + clean_ids)
    expiry_promoted = any(
        o == State.PROVISIONAL and n == State.TRUSTED and
        sys.items[i]["ts"].get("verify_commit", 0) > sys.items[i]["deadline"]
        for _, i, o, n in sys.transitions if i in watched)

    return {
        "sets": sets,
        "poisoned_retrievals_targeted": sets["target"]["Np"],   # headline
        "poisoned_retrievals_craft": sets["craft"]["Np"],
        "poisoned_retrievals_negative": sets["negative"]["Np"],
        "obs_censoring_bracket_s": round(3.0 / cfg["qps"], 4),  # per set: 1 in 3
        "Ep_p50": round(statistics.median(ep), 4) if ep else None,
        "Ep_max": round(max(ep), 4) if ep else None,
        "Ep_n": len(ep),
        "Ep_right_censored_n": ep_censored,
        "poison_never_visible_n": never_visible,
        "contain_rel_p50": round(c_med, 4) if c_med is not None else None,
        "containment_to_last_hit_s": gap,
        "Df_clean_p50": round(statistics.median(df), 4) if df else None,
        "Df_clean_max": round(max(df), 4) if df else None,
        "Df_right_censored_n": df_censored,
        # Eligible Recall@k (§VI-C). Reference: the currently-eligible set,
        # which INCLUDES the poison while it is exposed. A correctness check on
        # the query path, not a measure of harm -- it is expected to be 1.0.
        "elig_recall_during_exposure_p50": round(statistics.median(rec_during), 4) if rec_during else None,
        "elig_recall_after_containment_p50": round(statistics.median(rec_after), 4) if rec_after else None,
        # Displacement against a poison-free reference over the same
        # currently-eligible set. Per-query intensity, on each set.
        "displaced_vs_poisonfree_target_during_p50": round(statistics.median(dsp_during), 4) if dsp_during else None,
        "displaced_vs_poisonfree_target_after_p50": round(statistics.median(dsp_after), 4) if dsp_after else None,
        "displaced_vs_poisonfree_craft_during_p50": round(statistics.median(dspc_during), 4) if dspc_during else None,
        "displaced_vs_poisonfree_craft_after_p50": round(statistics.median(dspc_after), 4) if dspc_after else None,
        "displaced_vs_poisonfree_craft_during_max": round(max(dspc_during), 4) if dspc_during else None,
        # Cumulative: top-k positions lost over the whole window.
        "D_H_craft_positions": dh_craft, "D_H_craft_queries_scored": n_dh_craft,
        "D_H_target_positions": dh_target, "D_H_target_queries_scored": n_dh_target,
        # B1 never contains, so its observed exposure window is bounded by the
        # measurement, not by the system. Flagged rather than left to be
        # inferred from Ep_right_censored_n.
        "W_obs_right_censored": exposed and c_med is None,
        "inv_I7_prov_once": prov_ok,
        "inv_I3_no_expiry_promotion": not expiry_promoted,
    }


# ----------------------------------------------- conditional defence effect ---
def attach_conditional(records, cfg):
    """Split what the unconditional rate conflates: how many queries the attack
    reached at all, and how much exposure the protocol removed where it did.

    On Q_target the attack reaches a minority of queries, so an unconditional
    PRR is small for every baseline -- including the undefended one -- and reads
    as "the attack was weak" rather than "the defence worked". The qualified set
    is therefore fixed from **B1**, the baseline with no defence, at the same
    seed and backlog, and then applied unchanged to all four baselines and to
    every T_p. It is never re-derived per baseline: that would be selecting on
    the outcome being measured.
    """
    keys = sorted({(r["seed"], r["backlog"]) for r in records})
    for seed, backlog in keys:
        group = [r for r in records
                 if r["seed"] == seed and r["backlog"] == backlog]
        b1 = [r for r in group
              if r["baseline"] == "B1" and r["Tp"] == cfg["tp"]]
        if not b1:
            continue
        for st in ("craft", "target"):
            pq1 = b1[0]["sets"][st]["per_query"]
            qual = {q for q, v in pq1.items() if v["hits"] > 0}
            cov = len(qual) / len(pq1) if pq1 else None
            for r in group:
                pq = r["sets"][st]["per_query"]
                n = sum(pq[q]["n"] for q in qual if q in pq)
                h = sum(pq[q]["hits"] for q in qual if q in pq)
                r[f"attack_coverage_{st}"] = round(cov, 4) if cov is not None else None
                r[f"cond_{st}_qualified_q"] = len(qual)
                r[f"cond_{st}_queries"] = n
                r[f"cond_{st}_Np"] = h
                r[f"cond_{st}_prr"] = round(h / n, 4) if n else None
    return records


# ------------------------------------------------------------------ gates ---
def admissibility(records, cfg):
    """Preregistration §5. A failing gate means the RUN is inadmissible, never
    that the protocol was shown to work."""
    msgs, ok = [], True

    def cells(**kw):
        return [r for r in records if all(r.get(k) == v for k, v in kw.items())]

    b1 = cells(baseline="B1", Tp=cfg["tp"])
    if not b1 or not all(r["poisoned_retrievals_craft"] > 0 for r in b1):
        ok = False
        msgs.append("GATE 1 FAIL: the attack never entered the top-k under B1; "
                    "the run measures nothing and is not evidence for the protocol")
    else:
        msgs.append(f"gate 1 ok: B1 poisoned retrievals on Q_craft "
                    f"{[r['poisoned_retrievals_craft'] for r in b1]}, "
                    f"on Q_target {[r['poisoned_retrievals_targeted'] for r in b1]}")

    for r in cells(baseline="B2"):
        if r["poison_never_visible_n"] != cfg["n_poison"]:
            ok = False
            msgs.append("GATE 2 FAIL: B2 made unvetted content visible")
            break
    for r in cells(baseline="B1"):
        if r["Ep_right_censored_n"] != cfg["n_poison"]:
            ok = False
            msgs.append("GATE 2 FAIL: B1 contained something, but B1 has no verifier")
            break
    for r in cells(baseline="B4"):
        if r["Ep_max"] is not None and r["Ep_max"] > r["Tp"] + 0.25:
            ok = False
            msgs.append(f"GATE 2 FAIL: B4 Ep_max {r['Ep_max']}s exceeds Tp+slack "
                        f"at Tp={r['Tp']} -- invariant violation, fix and re-run")
            break
    if ok:
        msgs.append("gate 2 ok: every baseline behaved as its own definition says")

    if not all(r["inv_I7_prov_once"] and r["inv_I3_no_expiry_promotion"]
               for r in records):
        ok = False
        msgs.append("GATE 3 FAIL: I7 or I3 violated in-run")
    else:
        msgs.append("gate 3 ok: I7 and I3 held in every cell")

    for bkl in cfg["backlog"]:
        per_seed = {}
        for s in cfg["seeds"]:
            row = {b: cells(baseline=b, backlog=bkl, Tp=cfg["tp"], seed=s)
                   for b in BASELINES}
            if all(row.values()):
                per_seed[s] = {b: row[b][0]["poisoned_retrievals_targeted"]
                               for b in row}
        bad = [s for s, v in per_seed.items()
               if not (v["B1"] >= v["B3"] >= v["B4"] >= v["B2"])]
        if bad:
            msgs.append(f"gate 4 NOTE ({bkl}): seeds {bad} do not order "
                        f"B1>=B3>=B4>=B2 on Q_target; report the disagreement, "
                        f"not the mean")
        else:
            msgs.append(f"gate 4 ok ({bkl}): all {len(per_seed)} seeds order "
                        f"B1>=B3>=B4>=B2 on Q_target")

    neg = [r["poisoned_retrievals_negative"] for r in records]
    msgs.append(f"Q_negative poisoned retrievals: max {max(neg) if neg else 0} "
                f"across {len(neg)} cells (spread check)")

    # Gate 5: the fixed collection has to come back to the corpus between cells.
    resid = [r.get("residual_rows") for r in records]
    start = [r.get("residual_at_start") for r in records if r.get("residual_at_start") is not None]
    derr = sum(r.get("delete_errors", 0) for r in records)
    if start and max(start) > 0:
        ok = False
        msgs.append(f"GATE 5 FAIL: a cell STARTED with {max(start)} foreign ids "
                    f"present; the previous sweep reported clean but rows "
                    f"surfaced afterwards")
    passes = [r.get("sweep_attempts", 1) for r in records]
    if passes and max(passes) > 1:
        msgs.append(f"gate 5 note: the sweep needed up to {max(passes)} passes "
                    f"to verify a cell clean (write-visibility lag)")
    # The residual harm, measured rather than argued. A stale row cannot be
    # returned -- post-filtering decides eligibility in the control store -- but
    # it can occupy an over-fetch slot. This counts every time one did.
    foreign = sum(r.get("foreign_candidates", 0) for r in records)
    nq = sum(r.get("candidate_queries", 0) for r in records)
    if foreign:
        ok = False
        msgs.append(f"GATE 5 FAIL: {foreign} foreign ids entered the over-fetch "
                    f"candidate set across {nq} queries; an earlier cell's rows "
                    f"were competing for candidate slots")
    else:
        uf = sum(r.get("underfill_queries", 0) for r in records)
        msgs.append(f"gate 5 ok: no foreign id entered the candidate set in any "
                    f"of {nq} queries; {uf} under-filled")
    live = max([r.get("live_tasks", 0) + r.get("live_timers", 0) for r in records])
    if any(x is None for x in resid):
        msgs.append("GATE 5 NOTE: the backend could not be counted, so residual "
                    "rows between cells are unverified")
    elif max(resid) > 0 or derr:
        ok = False
        msgs.append(f"GATE 5 FAIL: {max(resid)} ids left in the collection "
                    f"between cells ({derr} delete errors); a later cell's "
                    f"over-fetch was competing with an earlier cell's poison")
    elif live:
        ok = False
        msgs.append(f"GATE 5 FAIL: {live} verifier tasks or deadline timers were "
                    f"still armed when a cell was summarised; they fire during "
                    f"the next cell")
    else:
        msgs.append("gate 5 ok: every cell returned the collection to the corpus "
                    "with no task or timer still armed")
    bg = [e for r in records for e in r.get("bg_errors", [])]
    if bg:
        ok = False
        msgs.append(f"GATE 5 FAIL: {len(bg)} background task(s) raised, e.g. "
                    f"{bg[0]}; a cell whose verifier died is not a cell whose "
                    f"baseline behaved as its definition says")

    # Transfer breadth. Each seed draws a fresh set of topic centres, so the
    # attacked topic differs from seed to seed -- five independently drawn
    # topics, not one topic five times. Reported explicitly so that "an
    # eight-topic corpus" is never read as "validated on eight topics".
    per_topic = {}
    for r in records:
        if r["baseline"] == "B1" and r["Tp"] == cfg["tp"]:
            per_topic[r["seed"]] = r.get("attack_coverage_target")
    if per_topic:
        vals = [v for v in per_topic.values() if v is not None]
        hit = sum(1 for v in vals if v > 0)
        msgs.append(f"transfer breadth: {hit}/{len(vals)} independently drawn "
                    f"attacked topics showed >=1 transferred query; per-topic "
                    f"coverage {[round(v, 3) for v in vals]}")

    cov_t = [r.get("attack_coverage_target") for r in records
             if r.get("attack_coverage_target") is not None]
    cov_c = [r.get("attack_coverage_craft") for r in records
             if r.get("attack_coverage_craft") is not None]
    if cov_t:
        msgs.append(f"attack coverage (B1-defined): craft "
                    f"{min(cov_c):.2f}-{max(cov_c):.2f}, target "
                    f"{min(cov_t):.2f}-{max(cov_t):.2f}")
    return ok, msgs


# ------------------------------------------------------------------- main ---
def agg(records, cfg, **sel):
    rs = [r for r in records if all(r.get(k) == v for k, v in sel.items())]
    if not rs:
        return None

    def med(key):
        vals = [r[key] for r in rs if r.get(key) is not None]
        return round(statistics.median(vals), 4) if vals else None

    def rng_(key):
        vals = [r[key] for r in rs if r.get(key) is not None]
        return [min(vals), max(vals)] if vals else None

    out = {"n_seeds": len(rs)}
    for s in ("craft", "target", "negative"):
        vals = [r["sets"][s]["Np"] for r in rs]
        out[f"Np_{s}_p50"] = statistics.median(vals)
        out[f"Np_{s}_range"] = [min(vals), max(vals)]
        aucs = [r["sets"][s]["auc_query_seconds"] for r in rs]
        out[f"auc_{s}_p50"] = round(statistics.median(aucs), 4)
    out.update({
        "prr_target_p50": round(statistics.median(
            [r["sets"]["target"]["prr"] for r in rs]), 4),
        "Ep_p50": med("Ep_p50"), "Ep_p50_range": rng_("Ep_p50"),
        "Ep_right_censored_n": sum(r["Ep_right_censored_n"] for r in rs),
        "poison_never_visible_n": sum(r["poison_never_visible_n"] for r in rs),
        "Df_clean_p50": med("Df_clean_p50"), "Df_clean_range": rng_("Df_clean_p50"),
        "Df_right_censored_n": sum(r["Df_right_censored_n"] for r in rs),
        "elig_recall_during_exposure_p50": med("elig_recall_during_exposure_p50"),
        "elig_recall_after_containment_p50": med("elig_recall_after_containment_p50"),
        "displaced_craft_during_p50": med("displaced_vs_poisonfree_craft_during_p50"),
        "displaced_craft_during_max": med("displaced_vs_poisonfree_craft_during_max"),
        "displaced_craft_after_p50": med("displaced_vs_poisonfree_craft_after_p50"),
        "displaced_target_during_p50": med("displaced_vs_poisonfree_target_during_p50"),
        "D_H_craft_p50": med("D_H_craft_positions"),
        "D_H_craft_range": rng_("D_H_craft_positions"),
        "D_H_target_p50": med("D_H_target_positions"),
        "D_H_target_range": rng_("D_H_target_positions"),
        "attack_coverage_craft": med("attack_coverage_craft"),
        "attack_coverage_target": med("attack_coverage_target"),
        "cond_craft_prr_p50": med("cond_craft_prr"), "cond_craft_prr_range": rng_("cond_craft_prr"),
        "cond_target_prr_p50": med("cond_target_prr"), "cond_target_prr_range": rng_("cond_target_prr"),
        "cond_craft_Np_p50": med("cond_craft_Np"),
        "cond_target_Np_p50": med("cond_target_Np"),
        "W_obs_right_censored_n": sum(1 for r in rs if r.get("W_obs_right_censored")),
        "containment_to_last_hit_s": med("containment_to_last_hit_s"),
    })
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="inmemory", choices=["inmemory", "milvus"])
    ap.add_argument("--uri", default="/tmp/w2_lite.db",
                    help="Milvus Lite .db path (no Docker) or http://host:19530")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = json.loads(json.dumps(CFG))
    if args.smoke:
        cfg.update(dur=3.0, seeds=[1], tp_sweep=[1.0], corpus=200, qps=21)

    if args.backend == "inmemory":
        backend = InMemoryBackend()
        evidence, index_type = "exact_in_memory", "exact_bruteforce"
    else:
        from milvus_backend import MilvusBackend
        backend = MilvusBackend(dim=cfg["dim"], uri=args.uri, collection="w2_poison")
        evidence = ("milvus_lite_flat" if not str(args.uri).startswith("http")
                    else "milvus_standalone")
        index_type = "FLAT"

    print(f"W2 poison exposure  backend={args.backend}  evidence={evidence}")
    print("  ONE collection, created once, never dropped during measurement.")

    records, next_id, geom = [], None, None
    for seed in cfg["seeds"]:
        world = make_world(seed, cfg)
        geom = geom or world["geometry"]
        idbase = id_base(seed, cfg)
        # Monotonic across every cell of the seed; never reset per cell.
        next_id = idbase + cfg["corpus"]
        reset_corpus(backend, world["corpus"], cfg, idbase)
        # Corpus half of the ground truth: computed once per seed, outside every
        # measurement window, because the corpus is identical and static across
        # this seed's cells.
        pre = {(kind, i): corpus_topk(world["corpus"], q, cfg["k"], idbase)
               for kind, key in (("craft", "q_craft"), ("target", "q_target"))
               for i, q in enumerate(world[key])}

        grid = [(b, bkl, cfg["tp"]) for bkl in cfg["backlog"] for b in BASELINES]
        grid += [("B4", bkl, tp) for bkl in cfg["backlog"]
                 for tp in cfg["tp_sweep"] if tp != cfg["tp"]]
        # Preregistered: cell order randomised per seed. Rows are removed
        # between cells but the collection persists, so a fixed order would let
        # any position effect load onto the same baseline in every seed.
        random.Random(9000 + seed).shuffle(grid)

        for base, bkl, tp in grid:
            m = await run_cell(backend, base, bkl, tp, seed, cfg, world,
                               next_id, pre)
            next_id = m.pop("next_id")
            records.append(m)
            print(f"  s{seed} {base} {bkl:<6} Tp={tp:<4} "
                  f"Np craft/target/neg "
                  f"{m['poisoned_retrievals_craft']:>3}/"
                  f"{m['poisoned_retrievals_targeted']:>3}/"
                  f"{m['poisoned_retrievals_negative']:<3} "
                  f"Ep50 {str(m['Ep_p50']):>6} Df50 {str(m['Df_clean_p50']):>6} "
                  f"DH_c {str(m["D_H_craft_positions"]):>6}",
                  flush=True)

    attach_conditional(records, cfg)
    ok, msgs = admissibility(records, cfg)
    print("\nAdmissibility (W2-PREREGISTRATION.md §5):")
    for m in msgs:
        print("  " + m)

    summary = {}
    for bkl in cfg["backlog"]:
        for b in BASELINES:
            summary[f"{b}/{bkl}"] = agg(records, cfg, baseline=b, backlog=bkl,
                                        Tp=cfg["tp"])
        for tp in cfg["tp_sweep"]:
            summary[f"B4/{bkl}/Tp={tp}"] = agg(records, cfg, baseline="B4",
                                               backlog=bkl, Tp=tp)

    import analysis
    out = args.out or os.path.join(HERE, "results", f"W2-{args.backend}.json")
    doc = {
        "schema_version": SCHEMA_VERSION,
        "run_id": f"W2-{args.backend}-{'smoke' if args.smoke else 'full'}",
        "experiment_id": EXPERIMENT_ID,
        "backend": args.backend,
        "index_type": index_type,
        "evidence_level": evidence,
        "git_commit": analysis.git_commit(),
        "git_dirty": analysis.git_dirty(),
        "smoke": args.smoke,
        "config": cfg,
        "workload_geometry": geom,
        "admissible": ok,
        "admissibility_messages": msgs,
        "metrics": {"cells": records, "summary": summary},
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and not args.smoke:
        raise SystemExit(f"{out} exists; rename it rather than overwrite "
                         f"(RUN-SHEET.md §5)")
    with open(out, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"\nwrote {out}   admissible={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
