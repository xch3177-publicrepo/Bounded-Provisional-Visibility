#!/usr/bin/env python3
"""Per-query check: does the geometry prediction agree with what B1 retrieved?

The manuscript explains a gap -- geometry says up to 4 of 8 held-out queries can
retrieve poison, the runs retrieved for 3 -- by calling the prediction an upper
bound that timing brackets from above. That explanation is only allowed if the
missing query had no opportunity: B1 never contains, so a query that executed
after the poison became visible and still returned no poison contradicts the
prediction rather than being bracketed by it.

Each target query executes n times per cell (recorded), so opportunity is
checkable rather than assumable. This joins the two and prints every
disagreement with its margin, so a mismatch is located rather than absorbed.

eps is fixed here at 1e-9 and is not to be widened after seeing an outcome; a
disagreement inside float noise is reported as a tie, not as agreement.
"""
import json
import os
import sys

from backend import cosine
from poison_exposure import CFG, CFG_REALTEXT, make_world

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-9


def predict(world, cfg, extra_visible):
    """max poison a query can see, against that query's own k-th best clean
    score. `extra_visible` is the clean content admitted during the cell: it is
    same-topic and competes for the same top-k slots, so leaving it out of the
    cutoff over-predicts."""
    out = []
    clean_pool = world["corpus"] + (world["clean"] if extra_visible else [])
    for q in world["q_target"]:
        ck = sorted((cosine(q, c) for c in clean_pool), reverse=True)[cfg["k"] - 1]
        best = max(cosine(q, p) for p in world["poison"])
        out.append({"cutoff": ck, "best_poison": best, "hit": best > ck + EPS})
    return out


def run(name, path, cfg, build, expect_legacy_miss):
    bad = 0
    doc = json.load(open(os.path.join(HERE, path)))
    print(f"\n=== {name}  ({doc['run_id']})")
    bad = 0
    # The corpus-only pass is a NEGATIVE CONTROL, kept deliberately. It is the
    # cutoff the manuscript used to compute, and it mispredicts one real-text
    # query; reproducing that miss is how this file proves the bug was real and
    # would notice it coming back. Its 39/40 is not a failing result and is
    # never quoted -- it is an expected mismatch, and the run is wrong if it
    # DISAPPEARS.
    for variant, extra in (("legacy corpus-only cutoff (expected mismatch)", False),
                           ("query-time candidate set (the fix)", True)):
        agree = mismatch = no_opportunity = 0
        rows = []
        for seed in cfg["seeds"]:
            world = build(seed, cfg)
            pred = predict(world, cfg, extra)
            cell = next(c for c in doc["metrics"]["cells"]
                        if c["baseline"] == "B1" and c["backlog"] == "heavy"
                        and c["seed"] == seed and c["Tp"] == cfg["tp"])
            pq = cell["sets"]["target"]["per_query"]
            for i, p in enumerate(pred):
                rec = pq.get(str(i), {"n": 0, "hits": 0})
                actual = rec["hits"] > 0
                if rec["n"] == 0:
                    no_opportunity += 1
                    continue
                if actual == p["hit"]:
                    agree += 1
                else:
                    mismatch += 1
                    rows.append((seed, i, p["best_poison"], p["cutoff"],
                                 p["hit"], actual, rec["n"], rec["hits"]))
        # A check with no subject is not a passing check. Zero comparisons here
        # would mean the cells, the per-query records or the world builder
        # stopped producing what this reads, and printing PASS for it is how a
        # gate goes quiet without going away.
        expect_n = len(cfg["seeds"]) * cfg["n_target_q"]
        if agree + mismatch != expect_n:
            print(f"  {variant}\n    FAIL: compared {agree + mismatch} queries, "
                  f"expected {expect_n}")
            bad += 1
            continue
        if extra:
            tag = "PASS" if not mismatch else "FAIL"
        else:
            tag = ("as declared" if mismatch == expect_legacy_miss
                   else f"NOT as declared, expected {expect_legacy_miss}")
        print(f"  {variant}\n    {tag}: {agree}/{agree + mismatch}"
              f"   never executed {no_opportunity}")
        for r in rows[:6]:
            s, i, b, c, ph, ah, n, h = r
            print(f"     seed {s} q{i}: best_poison {b:.4f} cutoff {c:.4f} "
                  f"margin {b - c:+.4f}  predicted {ph} actual {ah} "
                  f"({h}/{n} executions hit)")
        if extra:
            bad += mismatch
        elif mismatch != expect_legacy_miss:
            # Declared per workload, because the old cutoff only ever
            # mispredicted on real text: the synthetic clean items sit far
            # enough from the target queries that adding them moves no top-5
            # boundary. Requiring a counterexample where none exists would be
            # inventing one; requiring none where one exists would let the
            # regression back in.
            bad += 1
    return bad


if __name__ == "__main__":
    from realtext_workload import make_world_realtext
    bad = 0
    # Expected legacy mismatches, frozen: none in synthetic, exactly one in
    # real text (seed 1, q5, poison 0.5339 over a corpus-only cutoff of 0.5127
    # that a concurrently ingested clean item raises above it).
    bad += run("synthetic", "results/W2-inmemory.json", CFG, make_world, 0)
    bad += run("real text", "results/W2R-inmemory.json", CFG_REALTEXT,
               make_world_realtext, 1)
    print(f"\n==== verify_transfer_consistency: "
          f"{'AGREES per query' if not bad else f'{bad} DISAGREEMENTS'} ====")
    raise SystemExit(0 if not bad else 1)
