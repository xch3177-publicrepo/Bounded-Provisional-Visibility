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
    "recall_every": 4,          # score exact recall on every 4th Q_target query
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
def seed_control_store(sys, corpus):
    """The corpus is pre-existing trusted content. Without this the post-filter
    would drop every corpus id as unknown and the top-k would be empty."""
    for i, v in enumerate(corpus):
        sys.items[i] = {"id": i, "vec": v, "state": State.TRUSTED, "prov": 0,
                        "pending": False, "content_bad": False, "ts": {}}


def exact_topk(sys, q, k, exclude=()):
    """Ground truth: exact top-k over everything the control store currently
    calls eligible. Computed here rather than in the backend so E1 and E2 are
    scored against the identical definition. With `exclude` set to the poison
    ids it gives the top-k the query WOULD have returned had the poison never
    been admitted."""
    # list() first: this runs in a worker thread while the event loop is still
    # admitting items into the same dict (see InMemoryBackend.search).
    cand = [(cosine(q, it["vec"]), i) for i, it in list(sys.items.items())
            if it["state"] in VISIBLE and i not in exclude]
    cand.sort(reverse=True)
    return [i for _, i in cand[:k]]


async def run_cell(backend, base, backlog_name, Tp, seed, cfg, world, base_id):
    bl = BASELINES[base]
    bk = cfg["backlog"][backlog_name]
    sys = System(backend, mode="postfilter", Tp=Tp, verify_cost=cfg["verify_cost"],
                 over_fetch=cfg["over_fetch"], decouple=True,
                 verifier_concurrency=bk["conc"],
                 verify=bl["verify"], sync=bl["sync"], deadline=bl["deadline"])
    seed_control_store(sys, world["corpus"])

    poison_ids, clean_ids, filler_ids = [], [], []
    qlog = []                       # (t, kind, hit, recall|None)
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
            ids = await asyncio.to_thread(sys.query, q, cfg["k"])
            rec = disp = None
            # Sampled on craft AND target. Sampling only target reported 0.0
            # displacement everywhere and read as "the poison is harmless" --
            # but the poison mostly does not reach a target query's top-k at
            # all, so the set where displacement actually happens was the one
            # not being measured.
            if kind in ("craft", "target") and qidx % cfg["recall_every"] == 0:
                pset = set(poison_ids)
                truth = await asyncio.to_thread(exact_topk, sys, q, cfg["k"])
                # Eligible Recall@k (§VI-C): does the query path return the
                # exact eligible top-k. Poison is eligible while exposed, so
                # this stays 1.0 during exposure -- it is a correctness check,
                # not a measure of harm.
                rec = (len(set(ids) & set(truth)) / len(truth)) if truth else None
                # Clean displacement: how much of the top-k the query WOULD
                # have had, absent the poison, the poison pushed out. This is
                # the harm, and it is what returns to 0 at containment.
                clean = await asyncio.to_thread(exact_topk, sys, q, cfg["k"], pset)
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
    await sys.shutdown()
    m = summarise(sys, qlog, poison_ids, clean_ids, cfg, Tp, t0)
    # Rows go, the collection stays. A swallowed delete would leave a previous
    # cell's poison in the store at cosine ~0.96 to the craft queries, where it
    # eats over-fetch slots in every later cell -- so the failure is recorded
    # and gated on, not caught and ignored.
    delete_errors = 0
    for i in poison_ids + clean_ids + filler_ids:
        try:
            backend.delete(i)
        except Exception:
            delete_errors += 1
    m.update(baseline=base, backlog=backlog_name, Tp=Tp, seed=seed,
             next_id=base_id, delete_errors=delete_errors,
             residual_rows=backend_residual(backend, cfg))
    return m


def backend_residual(backend, cfg):
    """Rows above the corpus that a cell failed to clean up. None if the
    backend cannot be counted -- which is itself worth seeing in the record."""
    if hasattr(backend, "v"):
        return len(backend.v) - cfg["corpus"]
    try:
        rows = backend.client.query(backend.coll, filter="id >= 0",
                                    output_fields=["count(*)"])
        return int(rows[0]["count(*)"]) - cfg["corpus"]
    except Exception:
        return None


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
        "recall_during_exposure_p50": round(statistics.median(rec_during), 4) if rec_during else None,
        "recall_after_containment_p50": round(statistics.median(rec_after), 4) if rec_after else None,
        "clean_displaced_during_p50": round(statistics.median(dsp_during), 4) if dsp_during else None,
        "clean_displaced_after_p50": round(statistics.median(dsp_after), 4) if dsp_after else None,
        "clean_displaced_during_max": round(max(dsp_during), 4) if dsp_during else None,
        # Same quantity on the crafted set, where the poison actually reaches
        # the top-k and therefore actually displaces something.
        "craft_displaced_during_p50": round(statistics.median(dspc_during), 4) if dspc_during else None,
        "craft_displaced_after_p50": round(statistics.median(dspc_after), 4) if dspc_after else None,
        "craft_displaced_during_max": round(max(dspc_during), 4) if dspc_during else None,
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
    derr = sum(r.get("delete_errors", 0) for r in records)
    if any(x is None for x in resid):
        msgs.append("GATE 5 NOTE: the backend could not be counted, so residual "
                    "rows between cells are unverified")
    elif max(resid) > 0 or derr:
        ok = False
        msgs.append(f"GATE 5 FAIL: {max(resid)} rows left in the collection "
                    f"between cells ({derr} delete errors); a later cell's "
                    f"over-fetch was competing with an earlier cell's poison")
    else:
        msgs.append("gate 5 ok: every cell returned the collection to the corpus")

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
        "recall_during_exposure_p50": med("recall_during_exposure_p50"),
        "recall_after_containment_p50": med("recall_after_containment_p50"),
        "clean_displaced_during_p50": med("clean_displaced_during_p50"),
        "clean_displaced_after_p50": med("clean_displaced_after_p50"),
        "craft_displaced_during_p50": med("craft_displaced_during_p50"),
        "craft_displaced_during_max": med("craft_displaced_during_max"),
        "craft_displaced_after_p50": med("craft_displaced_after_p50"),
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

    records, next_id, geom = [], cfg["corpus"], None
    for seed in cfg["seeds"]:
        world = make_world(seed, cfg)
        geom = geom or world["geometry"]
        for i, v in enumerate(world["corpus"]):
            backend.insert(i, v, visible=True)
        if hasattr(backend, "wait_index"):
            backend.wait_index()

        grid = [(b, bkl, cfg["tp"]) for bkl in cfg["backlog"] for b in BASELINES]
        grid += [("B4", bkl, tp) for bkl in cfg["backlog"]
                 for tp in cfg["tp_sweep"] if tp != cfg["tp"]]
        # Preregistered: cell order randomised per seed. Rows are removed
        # between cells but the collection persists, so a fixed order would let
        # any position effect load onto the same baseline in every seed.
        random.Random(9000 + seed).shuffle(grid)

        for base, bkl, tp in grid:
            m = await run_cell(backend, base, bkl, tp, seed, cfg, world, next_id)
            next_id = m.pop("next_id")
            records.append(m)
            print(f"  s{seed} {base} {bkl:<6} Tp={tp:<4} "
                  f"Np craft/target/neg "
                  f"{m['poisoned_retrievals_craft']:>3}/"
                  f"{m['poisoned_retrievals_targeted']:>3}/"
                  f"{m['poisoned_retrievals_negative']:<3} "
                  f"Ep50 {str(m['Ep_p50']):>6} Df50 {str(m['Df_clean_p50']):>6} "
                  f"R_after {str(m['recall_after_containment_p50']):>5}",
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
