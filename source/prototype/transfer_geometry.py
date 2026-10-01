#!/usr/bin/env python3
"""Per-query transfer geometry: the statistic that actually governs retrieval.

Both workloads reported transfer against a MEAN. `poison_vs_target` averages
cosine over every (poison, target-query) pair, and it was compared against the
median of the per-query top-k cutoffs. Neither side of that comparison is what
decides whether a query retrieves poison. What decides it is, for each query q
on its own:

    s(q) = max over the poison items of cos(q, x)      vs      c_k(q)

the best poison the query can see, against the k-th best clean corpus
similarity for that same query. A mean over pairs mixes in five poison items
written for other queries and understates s(q); a median over queries answers
a question no single query asks.

So the sentence "on average it fails to clear the bar, yet it transfers" was
explaining a retrieval outcome with a quantity that does not produce it. This
recomputes the right one from the frozen worlds -- no new measurement, the same
inputs the runs used.

RULE, fixed before running (the outputs below did not exist when this was
written): whatever comes out is what gets reported, including the case where
the geometry predicts more or fewer transferring queries than the runs
observed. A gap between predicted and observed is a finding about eligibility,
over-fetch and timing, not an error to tune away. Specifically:

  * predicted > observed is expected and benign: clearing the cutoff makes a
    query eligible to retrieve poison, but the poison must also be visible at
    the moment that query runs, and B1's window starts after injection.
  * predicted < observed would be a real inconsistency and must be chased, not
    reported around.

Usage:  python3 transfer_geometry.py            (writes results/transfer-geometry.json)
"""

import json
import os
import statistics

from backend import cosine
from poison_exposure import CFG, CFG_REALTEXT, make_world

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "transfer-geometry.json")


def per_query(world, cfg):
    """For each held-out target query: its own cutoff, the best poison it can
    see, and whether that poison is inside its top-k.

    The cutoff is taken over every non-poison item a query can see during the
    cell -- the corpus AND the clean and filler content admitted while it runs.
    Computing it over the corpus alone was wrong and the error was not
    hypothetical: the clean items are drawn from the attacked topic, so they
    compete for the same slots, and leaving them out predicted a transfer for
    one query that never happened. With them in, the prediction agrees with
    what B1 actually retrieved for all 80 queries of both workloads
    (verify_transfer_consistency.py)."""
    poison = world["poison"]
    alternatives = world["corpus"] + world["clean"] + world["filler"]
    rows = []
    for i, q in enumerate(world["q_target"]):
        ck = sorted((cosine(q, c) for c in alternatives), reverse=True)[cfg["k"] - 1]
        best = max(cosine(q, p) for p in poison)
        rows.append({"q": i, "cutoff": round(ck, 4), "best_poison": round(best, 4),
                     "margin": round(best - ck, 4), "clears": best >= ck})
    return rows


def analyse(name, cfg, build):
    seeds = []
    for seed in cfg["seeds"]:
        rows = per_query(build(seed, cfg), cfg)
        n_clear = sum(r["clears"] for r in rows)
        seeds.append({
            "seed": seed, "n_queries": len(rows), "n_clearing": n_clear,
            "frac_clearing": round(n_clear / len(rows), 4),
            "margin_p50": round(statistics.median(r["margin"] for r in rows), 4),
            "margin_min": min(r["margin"] for r in rows),
            "margin_max": max(r["margin"] for r in rows),
            "best_poison_p50": round(statistics.median(r["best_poison"] for r in rows), 4),
            "cutoff_p50": round(statistics.median(r["cutoff"] for r in rows), 4),
            "rows": rows,
        })
    fracs = [s["frac_clearing"] for s in seeds]
    return {
        "workload": name,
        "n_workloads": len(seeds),
        "queries_per_workload": seeds[0]["n_queries"],
        "distinct_target_queries": sum(s["n_queries"] for s in seeds),
        "clearing_per_workload": [s["n_clearing"] for s in seeds],
        "frac_clearing_range": [min(fracs), max(fracs)],
        "margin_p50_range": [min(s["margin_p50"] for s in seeds),
                             max(s["margin_p50"] for s in seeds)],
        "best_poison_p50_range": [min(s["best_poison_p50"] for s in seeds),
                                  max(s["best_poison_p50"] for s in seeds)],
        "cutoff_p50_range": [min(s["cutoff_p50"] for s in seeds),
                             max(s["cutoff_p50"] for s in seeds)],
        "per_seed": seeds,
    }


def negatives(cfg, build):
    """Distinct unrelated queries, counted once each. The manuscript said 'no
    poisoned retrieval in 70 cells', but the 70 cells re-query the SAME eight
    unrelated queries per workload under different baselines and deadlines --
    repeated measurement presented as independent evidence. The honest
    denominator is the number of distinct unrelated queries."""
    n = 0
    clears = 0
    for seed in cfg["seeds"]:
        w = build(seed, cfg)
        alt = w["corpus"] + w["clean"] + w["filler"]
        for q in w["q_neg"]:
            ck = sorted((cosine(q, c) for c in alt), reverse=True)[cfg["k"] - 1]
            best = max(cosine(q, p) for p in w["poison"])
            n += 1
            clears += best >= ck
    return {"distinct_unrelated_queries": n, "clearing": clears}


if __name__ == "__main__":
    from realtext_workload import make_world_realtext
    doc = {"synthetic": analyse("synthetic", CFG, make_world),
           "realtext": analyse("realtext", CFG_REALTEXT, make_world_realtext)}
    doc["synthetic"]["negatives"] = negatives(CFG, make_world)
    doc["realtext"]["negatives"] = negatives(CFG_REALTEXT, make_world_realtext)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1)
    for k, v in doc.items():
        print(f"\n=== {k}")
        print(f"  target queries per workload {v['queries_per_workload']}, "
              f"{v['distinct_target_queries']} distinct across "
              f"{v['n_workloads']} workloads")
        print(f"  clearing their own top-k cutoff, per workload: "
              f"{v['clearing_per_workload']}  "
              f"= {v['frac_clearing_range'][0]*100:.0f}-"
              f"{v['frac_clearing_range'][1]*100:.0f}%")
        print(f"  best poison (median over queries) {v['best_poison_p50_range']}  "
              f"vs cutoff {v['cutoff_p50_range']}")
        print(f"  margin best-poison minus cutoff, median {v['margin_p50_range']}")
        print(f"  unrelated: {v['negatives']['clearing']} of "
              f"{v['negatives']['distinct_unrelated_queries']} distinct queries clear")
    print(f"\nwrote {OUT}")
