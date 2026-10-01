#!/usr/bin/env python3
"""Independent acceptance check for a W2 result file.

    ./.venv312/bin/python verify_w2.py results/W2-inmemory.json [results/W2-milvus.json]

The runner grades its own run. This does not trust that grading: it recomputes
the gates from the per-cell records, checks the manifest against the frozen
config, and checks that the row-level observations and the gate-level verdict
agree. That last one is not hypothetical -- a variable-shadowing bug once made
every record carry `baseline=1000000`, and the run printed rows showing the
attack landing thirteen times while its own gate 1 concluded the attack had
never worked. Evidence and verdict disagreeing is the signature worth testing
for, whatever caused it.

Prints a sha256 of the metrics block so a figure or a paper sentence can be tied
to the exact numbers it came from.
"""
import hashlib
import json
import sys

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def key(c):
    return (c["baseline"], c["backlog"], c["Tp"], c["seed"],
            c.get("false_promoted_k", 0))


def verify(path):
    global ok_all
    with open(path) as f:
        doc = json.load(f)
    cfg, cells = doc["config"], doc["metrics"]["cells"]
    print(f"\n=== {path}")
    print(f"    evidence={doc['evidence_level']} commit={doc['git_commit']} "
          f"dirty={doc['git_dirty']} smoke={doc.get('smoke')}")

    # --- manifest: the run is the shape the frozen config asks for -----------
    baselines = sorted({c["baseline"] for c in cells})
    check("baseline field holds names, not integers",
          all(isinstance(c["baseline"], str) for c in cells), str(baselines))
    n_expect = len(cfg["seeds"]) * (
        len(cfg["backlog"]) * 4 + len(cfg["backlog"]) * (len(cfg["tp_sweep"]) - 1))
    check("cell count matches the config", len(cells) == n_expect,
          f"{len(cells)} vs {n_expect}")
    check("every seed present", sorted({c["seed"] for c in cells}) == sorted(cfg["seeds"]))
    check("all four baselines present at the main T_p",
          sorted({c["baseline"] for c in cells if c["Tp"] == cfg["tp"]}) ==
          ["B1", "B2", "B3", "B4"])
    check("not a smoke run", doc.get("smoke") is False)

    # --- row level vs gate level -------------------------------------------
    # Recomputed from per_query, which the gates do not read.
    b1 = [c for c in cells if c["baseline"] == "B1" and c["Tp"] == cfg["tp"]]
    raw_hits = [sum(v["hits"] for v in c["sets"]["craft"]["per_query"].values())
                for c in b1]
    check("B1 craft hits recomputed from per-query equal the reported count",
          all(h == c["poisoned_retrievals_craft"] for h, c in zip(raw_hits, b1)))
    check("row level agrees with gate 1 (the attack did land)",
          all(h > 0 for h in raw_hits) and
          any("gate 1 ok" in m for m in doc["admissibility_messages"]),
          f"hits {raw_hits}")

    # The qualified set must come from B1 and be identical for every baseline
    # at the same seed and backlog; a per-baseline set would select on outcome.
    bad = []
    for seed in cfg["seeds"]:
        for bkl in cfg["backlog"]:
            grp = [c for c in cells if c["seed"] == seed and c["backlog"] == bkl]
            qn = {c.get("cond_craft_qualified_q") for c in grp}
            if len(qn) > 1:
                bad.append((seed, bkl, qn))
    check("qualified set is shared across baselines, not re-derived", not bad, str(bad[:2]))

    # --- isolation ----------------------------------------------------------
    check("no foreign candidate in any query",
          sum(c.get("foreign_candidates", 0) for c in cells) == 0)
    check("no under-fill",
          sum(c.get("underfill_queries", 0) for c in cells) == 0)
    check("under-fill denominator equals the query total",
          all(c.get("candidate_queries", 0) > 0 for c in cells))
    check("no cell started with foreign ids",
          max(c.get("residual_at_start", 0) or 0 for c in cells) == 0)
    check("no cell ended with residual rows",
          max(c.get("residual_rows", 0) or 0 for c in cells) == 0)
    check("no task or timer survived a cell",
          max(c.get("live_tasks", 0) + c.get("live_timers", 0) for c in cells) == 0)
    check("no background task raised",
          not [e for c in cells for e in c.get("bg_errors", [])])

    # --- baseline definitions -----------------------------------------------
    check("B2 never made unvetted content visible",
          all(c["poison_never_visible_n"] == cfg["n_poison"]
              for c in cells if c["baseline"] == "B2"))
    check("B1 never contained anything",
          all(c["Ep_right_censored_n"] == cfg["n_poison"]
              for c in cells if c["baseline"] == "B1"))
    over = [(c["seed"], c["Tp"], c["Ep_max"]) for c in cells
            if c["baseline"] == "B4" and c["Ep_max"] is not None
            and c["Ep_max"] > c["Tp"] + 0.25]
    check("B4 exposure never exceeded its deadline plus slack", not over, str(over[:2]))
    check("B1 windows are flagged right-censored",
          all(c.get("W_obs_right_censored") for c in cells if c["baseline"] == "B1"))

    # --- episode bookkeeping identities -------------------------------------
    # Mechanical, and in the verifier rather than only in a unit test, because
    # the failures they catch are the ones that look like results: a None
    # rendered as 0, an empty list whose median becomes 0, a censored episode
    # dropped from a denominator, a pending episode counted as finished. Each
    # has to hold in every cell.
    def ident(name, ok, detail=""):
        check(f"episode bookkeeping: {name}", ok, detail)

    bad = [key(c) for c in cells
           if c["Eu_started_n"] != c["Eu_completed_n"] + c["Eu_right_censored_n"]]
    ident("started = completed + censored", not bad, str(bad[:2]))
    bad = [key(c) for c in cells
           if len(c["Eu_open_episode_ages"]) != c["Eu_right_censored_n"]]
    ident("one recorded age per open episode", not bad, str(bad[:2]))
    bad = [key(c) for c in cells if c["Eu_completed_n"] == 0
           and not (c["Eu_completed_p50"] is None and c["Eu_completed_max"] is None)]
    ident("no completed episode gives null, not zero", not bad, str(bad[:2]))
    bad = [key(c) for c in cells
           if c["Eu_completed_n"] > 0 and c["Eu_completed_p50"] is None]
    ident("a completed episode gives a duration", not bad, str(bad[:2]))
    bad = [key(c) for c in cells if c["Eu_started_n"] != len(
        [x for x in c["Eu_status"] if x != "NOT_STARTED"])]
    ident("the per-item statuses agree with the counts", not bad, str(bad[:2]))
    bad = [key(c) for c in cells
           if c["Eu_right_censored_n"] and c["Eu_observed_time_at_risk"] <= 0]
    ident("a censored episode contributes its observed age", not bad, str(bad[:2]))
    bad = [key(c) for c in cells if set(c["Eu_status"]) -
           {"NOT_STARTED", "COMPLETED", "RIGHT_CENSORED"}]
    ident("no status outside the three", not bad, str(bad[:2]))
    # Clean freshness: three quantities, and the arithmetic that keeps them from
    # collapsing back into one. A deadline hide opens a gap, a readmission
    # closes it, and the horizon censors what is still open -- so every expiry
    # is accounted for exactly once, and "no gap" is not a gap of length zero.
    n_clean = cfg["n_clean"]
    bad = [key(c) for c in cells
           if c["clean_expired_n"] != c["clean_gap_completed_n"]
           + c["clean_gap_right_censored_n"]]
    ident("every expiry is a completed or a censored gap", not bad, str(bad[:2]))
    bad = [key(c) for c in cells
           if c["clean_readmitted_n"] != c["clean_gap_completed_n"]]
    ident("a readmission is exactly what closes a gap", not bad, str(bad[:2]))
    bad = [key(c) for c in cells if c["clean_gap_completed_n"] == 0
           and not (c["clean_gap_p50"] is None and c["clean_gap_max"] is None)]
    ident("no completed gap gives null, not a gap of zero", not bad, str(bad[:2]))
    bad = [key(c) for c in cells
           if c["Df_trusted_completed_n"] + c["Df_trusted_right_censored_n"]
           != n_clean]
    ident("every clean item is durably visible or censored", not bad, str(bad[:2]))
    bad = [key(c) for c in cells if c["Df_trusted_completed_n"] == 0
           and c["Df_trusted_p50"] is not None]
    ident("no durably visible item gives null", not bad, str(bad[:2]))

    # Not JSON-representable, and the one that would survive every other check.
    import math as _m
    bad = [(key(c), f) for c in cells
           for f in ("Eu_completed_p50", "Eu_completed_max",
                     "Eu_observed_time_at_risk", "Ep_p50", "Ep_max")
           if isinstance(c.get(f), float) and (_m.isnan(c[f]) or _m.isinf(c[f]))]
    ident("no NaN or infinity reached the file", not bad, str(bad[:2]))

    # --- the run's own verdict ----------------------------------------------
    check("runner declared the run admissible", doc["admissible"] is True)

    h = hashlib.sha256(json.dumps(doc["metrics"], sort_keys=True).encode()).hexdigest()
    print(f"    metrics sha256 {h[:32]}")
    return doc, h


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    docs = [verify(p) for p in sys.argv[1:]]

    if len(docs) == 2:
        # E1 and E2 are the same experiment on two backends. They are not
        # independent replication -- one process, one implementation, one
        # machine -- so the check is that the MECHANISM agrees, not that the
        # numbers coincide.
        print("\n=== cross-tier agreement (mechanism, not point equality)")
        (a, _), (b, _) = docs
        for bkl in a["config"]["backlog"]:
            order = []
            for doc in (a, b):
                s = doc["metrics"]["summary"]
                order.append([s[f"{x}/{bkl}"]["Np_craft_p50"] for x in
                              ("B1", "B2", "B3", "B4")])
            check(f"{bkl}: baseline ordering identical across tiers",
                  [sorted(range(4), key=lambda i: -order[0][i])] ==
                  [sorted(range(4), key=lambda i: -order[1][i])],
                  f"{order[0]} vs {order[1]}")
        for doc, tag in ((a, "E1"), (b, "E2")):
            s = doc["metrics"]["summary"]
            check(f"{tag}: B3 exposure grows from normal to heavy backlog",
                  s["B3/heavy"]["Np_craft_p50"] > s["B3/normal"]["Np_craft_p50"],
                  f'{s["B3/normal"]["Np_craft_p50"]} -> {s["B3/heavy"]["Np_craft_p50"]}')
            check(f"{tag}: B4 grows less than B3 under the same backlog",
                  (s["B4/heavy"]["Np_craft_p50"] - s["B4/normal"]["Np_craft_p50"]) <
                  (s["B3/heavy"]["Np_craft_p50"] - s["B3/normal"]["Np_craft_p50"]))
            check(f"{tag}: verify-before-visible buys clean freshness only "
                  f"when the verifier keeps up",
                  s["B2/normal"]["Df_clean_p50"] > s["B4/normal"]["Df_clean_p50"],
                  f'normal B2 {s["B2/normal"]["Df_clean_p50"]} vs B4 '
                  f'{s["B4/normal"]["Df_clean_p50"]}; heavy B2 '
                  f'{s["B2/heavy"]["Df_clean_p50"]} vs B4 '
                  f'{s["B4/heavy"]["Df_clean_p50"]}')
            tps = sorted(doc["config"]["tp_sweep"])
            np_by_tp = [s[f"B4/heavy/Tp={t}"]["Np_craft_p50"] for t in tps]
            check(f"{tag}: B4 exposure is monotone in T_p",
                  all(x <= y for x, y in zip(np_by_tp, np_by_tp[1:])), str(np_by_tp))

    print(f"\n==== verify_w2: {'ALL CHECKS PASSED' if ok_all else 'FAILURES ABOVE'} ====")
    raise SystemExit(0 if ok_all else 1)
