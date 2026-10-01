#!/usr/bin/env python3
"""Independent acceptance check for the real-text W2R result files.

    ./.venv312/bin/python verify_w2r.py results/W2R-inmemory.json results/W2R-milvus.json

W2R records have the same schema as W2, so every W2 check applies unchanged and
is reused rather than reimplemented -- a second copy would drift. What this adds
are the checks specific to the claim W2R exists to support:

  * No timing parameter was retuned. This is the load-bearing one. W2R's whole
    argument is that only the DATA changed; if a knob moved, a W2-vs-W2R
    difference could be the knob. Compared field by field against the synthetic
    CFG, not eyeballed.
  * The poison was not placed. Its mean similarity to the held-out target
    queries must sit BELOW the corpus top-k bar those queries set -- meaning
    whatever transfer occurred was earned by the text against MiniLM, not
    arranged by construction. A poison that sat above the bar on average would
    make the transfer result vacuous.
  * The frozen embedding cache is the one the manifest describes, in both runs.
  * The attack actually engaged real queries, with the denominators printed.
"""
import json
import sys

import verify_w2
from verify_w2 import check
from poison_exposure import CFG, CFG_REALTEXT

# Everything except what the workload legitimately changes. dim is the vector
# width; the two sigma/jitter knobs generate synthetic geometry and have no
# meaning for real passages.
WORKLOAD_ONLY = {"dim", "topic_sigma", "poison_jitter"}


def verify_realtext(path):
    doc, h = verify_w2.verify(path)
    cfg = doc["config"]
    print(f"    workload={doc.get('workload')} "
          f"corpus={(doc.get('workload_provenance') or {}).get('source')}")

    check("run is tagged as the real-text workload",
          doc.get("workload") == "realtext" and doc.get("experiment_id") == "W2R")

    prov = doc.get("workload_provenance") or {}
    check("embedding model recorded", prov.get("model") == "all-MiniLM-L6-v2",
          str(prov.get("model")))
    check("corpus provenance recorded", "newsgroup" in str(prov.get("source")).lower(),
          str(prov.get("source")))
    check("embedding cache sha256 recorded", len(prov.get("sha256", "")) == 64,
          prov.get("sha256", "")[:12])
    check("vectors are 384-d, as the paper's system description says",
          cfg["dim"] == 384)

    # --- the anti-tuning check ---------------------------------------------
    drift = {k: (CFG[k], cfg[k]) for k in CFG
             if k not in WORKLOAD_ONLY and cfg.get(k) != CFG[k]}
    check("no timing or protocol parameter differs from the synthetic W2 config",
          not drift, str(drift))
    check("the declared real-text config is what actually ran",
          all(cfg.get(k) == v for k, v in CFG_REALTEXT.items()))

    # --- the poison was measured into the top-k, not placed there -----------
    g = doc["workload_geometry"]
    check("poison sits BELOW the corpus top-k bar for held-out target queries "
          "(transfer is earned, not constructed)",
          g["poison_vs_target"] < g["corpus_kth_cos_target"],
          f'poison~target {g["poison_vs_target"]} < bar {g["corpus_kth_cos_target"]}')
    check("poison does reach the queries it was written against",
          g["poison_clears_bar_on_craft"] > 0,
          f'{g["poison_clears_bar_on_craft"]}/{g["n_poison"]}')
    check("poison is near-orthogonal to unrelated topics",
          g["poison_vs_negative"] < g["poison_vs_target"],
          f'negative {g["poison_vs_negative"]}')

    # --- transfer, with its denominator -------------------------------------
    s = doc["metrics"]["summary"]
    b1h = s["B1/heavy"]
    print(f"    B1 heavy: craft {b1h['Np_craft_p50']} "
          f"target {b1h['Np_target_p50']} negative {b1h['Np_negative_p50']} "
          f"(per-workload medians over {len(cfg['seeds'])} seeds; "
          f"{cfg['n_target_q']} target queries per seed)")
    check("unrelated topics stayed clean",
          s["B1/heavy"]["Np_negative_p50"] == 0,
          str(s["B1/heavy"]["Np_negative_p50"]))
    return doc, h


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    docs = [verify_realtext(p) for p in sys.argv[1:]]

    if len(docs) == 2:
        print("\n=== cross-tier agreement (mechanism, not point equality)")
        (a, _), (b, _) = docs
        for bkl in a["config"]["backlog"]:
            order = []
            for doc in (a, b):
                s = doc["metrics"]["summary"]
                order.append([s[f"{x}/{bkl}"]["Np_craft_p50"]
                              for x in ("B1", "B2", "B3", "B4")])
            check(f"{bkl}: baseline ordering identical across tiers",
                  sorted(range(4), key=lambda i: -order[0][i]) ==
                  sorted(range(4), key=lambda i: -order[1][i]),
                  f"{order[0]} vs {order[1]}")
        for doc, tag in ((a, "E1"), (b, "E2")):
            s = doc["metrics"]["summary"]
            check(f"{tag}: B3 exposure grows from normal to heavy backlog",
                  s["B3/heavy"]["Np_craft_p50"] > s["B3/normal"]["Np_craft_p50"],
                  f'{s["B3/normal"]["Np_craft_p50"]} -> {s["B3/heavy"]["Np_craft_p50"]}')
            check(f"{tag}: B4 grows less than B3 under the same backlog",
                  (s["B4/heavy"]["Np_craft_p50"] - s["B4/normal"]["Np_craft_p50"]) <
                  (s["B3/heavy"]["Np_craft_p50"] - s["B3/normal"]["Np_craft_p50"]))
            check(f"{tag}: B2 pays clean freshness for its zero exposure",
                  s["B2/heavy"]["Df_clean_p50"] is None or
                  s["B2/heavy"]["Df_clean_p50"] > s["B4/heavy"]["Df_clean_p50"])
            tps = sorted(doc["config"]["tp_sweep"])
            np_by_tp = [s[f"B4/heavy/Tp={t}"]["Np_craft_p50"] for t in tps]
            check(f"{tag}: B4 exposure is monotone in T_p",
                  all(x <= y for x, y in zip(np_by_tp, np_by_tp[1:])), str(np_by_tp))

    print(f"\n==== verify_w2r: "
          f"{'ALL CHECKS PASSED' if verify_w2.ok_all else 'FAILURES ABOVE'} ====")
    raise SystemExit(0 if verify_w2.ok_all else 1)
