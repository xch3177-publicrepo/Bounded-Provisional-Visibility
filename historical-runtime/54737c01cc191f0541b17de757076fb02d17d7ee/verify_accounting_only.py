#!/usr/bin/env python3
"""Core retrieval outcomes are reproducible across two independent runs.

NOT a proof that the accounting change was confined, and the distinction
matters. This compares two EXECUTIONS. Two executions of identical code already
differ in asyncio scheduling, so no cross-run comparison can separate "the
analysis changed" from "these are two runs" for any wall-clock quantity. What
it does establish is the stronger-than-it-sounds fact that everything
deterministic -- every retrieval count, every per-query hit vector -- came back
identical across the change.

The isolation claim is made where it can be made: test_w2.py feeds one fixed
input to the old rule and the new one and shows they diverge only on
trajectories where a hide was not durable.


    ./.venv312/bin/python verify_accounting_only.py \
        results/W2-inmemory-preEpFix.json results/W2-inmemory.json

Written while the re-run was still in flight, so the thresholds cannot be
fitted to its outcome.

Under a correct oracle no poison is ever readmitted after its hide, so every
retrieval-side number must come back identical. If 49/34/7 moves, the edit
reached runtime state, the query path, or the measurement window, and the new
numbers are not to be accepted -- the first diverging cell is to be found
instead.
"""
import json
import sys

# Retrieval-side. None of these may move.
FROZEN = [
    "poisoned_retrievals_craft", "poisoned_retrievals_targeted",
    "poisoned_retrievals_negative", "poison_never_visible_n",
    "attack_coverage_craft", "attack_coverage_target",
    "cond_craft_prr", "cond_craft_qualified_q",
    "D_H_craft_positions", "D_H_target_positions",
    # `displaced_craft_during` and `displaced_craft_after` used to be listed
    # here. No cell has ever carried them -- they are summary-level names -- so
    # they compared None to None in every cell and were counted among the
    # fields this gate reported as checked.
    "foreign_candidates", "candidate_queries", "underfill_queries",
    "residual_at_start", "residual_rows",
]
# Analysis-side, changed on purpose by the fix.
EXPECTED_TO_MOVE = {"Ep_p50", "Ep_max", "Ep_n", "Ep_right_censored_n",
                    "contain_rel_p50", "containment_to_last_hit_s",
                    "poison_readmitted_after_hide_n", "Eu_p50", "Eu_max", "Eu_n",
                    "poison_visible_at_window_end_n", "first_readmission_rel_s",
                    "post_readmission_poison_hits",
                    "terminal_nonvisible_at_window_end_n",
                    "pending_visibility_transition_at_window_end_n",
                    "false_promoted_k", "false_promoted_ids"}

# Wall-clock, and therefore outside what this gate can decide.
#
# The comparison is between two EXECUTIONS, not between two readings of one set
# of events, so anything derived from when an asyncio sleep actually returned
# differs whether or not the analysis changed. That includes the phase-split
# medians: which queries count as "during exposure" depends on the containment
# instant, so with a short deadline a few milliseconds move one query across the
# boundary and a median over seven samples steps by a whole position. Their
# cumulative counterparts (D_H_*) carry no phase boundary, are in FROZEN, and are
# the ones the paper prints.
#
# Listing these as untestable here rather than tolerating them numerically is
# deliberate: a tolerance picked after seeing the deltas would be fitted to them.
# What separates "the edit changed behaviour" from "two runs differ" is one
# input through two accountants (test_w2.py), not a looser threshold here and
# not a same-code control run, which would carry the identical noise.
WALL_CLOCK = {"Df_clean_p50", "Df_clean_max", "Df_censored_n",
              "displaced_vs_poisonfree_craft_during_p50",
              "displaced_vs_poisonfree_craft_after_p50",
              "displaced_vs_poisonfree_craft_during_max",
              "displaced_vs_poisonfree_target_during_p50",
              "displaced_vs_poisonfree_target_after_p50",
              "elig_recall_during_exposure_p50",
              "elig_recall_after_containment_p50",
              "obs_first", "obs_last", "sweep_attempts",
              "run_started_at", "run_finished_at"}


def key(c):
    return (c["baseline"], c["backlog"], c["Tp"], c["seed"],
            c.get("false_promoted_k", 0))


def main(before_path, after_path):
    A = json.load(open(before_path))
    B = json.load(open(after_path))
    a = {key(c): c for c in A["metrics"]["cells"]}
    b = {key(c): c for c in B["metrics"]["cells"]}
    print(f"=== {A['run_id']} -> {B['run_id']}")
    print(f"    {len(a)} cells before, {len(b)} after")

    ok = True
    if set(a) != set(b):
        print(f"  [FAIL] cell set changed  "
              f"only-before {sorted(set(a) - set(b))[:3]} "
              f"only-after {sorted(set(b) - set(a))[:3]}")
        ok = False

    # A field nobody emits compares None to None forever and is counted in the
    # "17 fields checked" line while checking nothing. Two of the original
    # seventeen were like that; the message was the only evidence they existed.
    sample = next(iter(a.values()))
    missing = [f for f in FROZEN if f not in sample]
    print(f"  [{'PASS' if not missing else 'FAIL'}] every frozen field is one the "
          f"runner emits" + (f"  absent: {missing}" if missing else ""))
    if missing:
        ok = False

    moved = []
    for k in sorted(set(a) & set(b)):
        for f in FROZEN:
            if a[k].get(f) != b[k].get(f):
                moved.append((k, f, a[k].get(f), b[k].get(f)))
    print(f"  [{'PASS' if not moved else 'FAIL'}] no retrieval-side field moved"
          + (f"  first: {moved[0]}" if moved else
             f"  ({len(FROZEN)} fields x {len(a)} cells checked)"))
    if moved:
        ok = False
        for m in moved[:8]:
            print(f"     {m[0]} {m[1]}: {m[2]} -> {m[3]}")

    # The per-query hit vectors are the closest thing to a returned-id log that
    # these files carry, and they are what a query-path change would disturb
    # first. Compared element by element rather than by their totals.
    pq_moved = []
    for k in sorted(set(a) & set(b)):
        for st in ("craft", "target", "negative"):
            pa = a[k]["sets"][st]["per_query"]
            pb = b[k]["sets"][st]["per_query"]
            if pa != pb:
                pq_moved.append((k, st))
    print(f"  [{'PASS' if not pq_moved else 'FAIL'}] per-query hit vectors "
          f"identical" + (f"  first: {pq_moved[0]}" if pq_moved else ""))
    if pq_moved:
        ok = False

    # The aggregates the manuscript prints. A single cell at the shortest
    # deadline can move on a boundary between two executions -- two or three
    # retrievals fit inside a 0.3 s exposure, so which side of a query the
    # deadline lands on decides the count. The medians and ranges over five
    # workloads are what the paper quotes, and those must not move.
    agg_moved = []
    for cell in ("B1/heavy", "B2/heavy", "B3/heavy", "B4/heavy",
                 "B1/normal", "B2/normal", "B3/normal", "B4/normal"):
        for f in ("Np_craft_p50", "Np_craft_range", "Np_target_p50",
                  "D_H_craft_p50", "D_H_craft_range"):
            x = A["metrics"]["summary"].get(cell, {}).get(f)
            y = B["metrics"]["summary"].get(cell, {}).get(f)
            if x != y:
                agg_moved.append((cell, f, x, y))
    for t in sorted(A["config"]["tp_sweep"]):
        for bkl in ("heavy", "normal"):
            k = f"B4/{bkl}/Tp={t}"
            x = A["metrics"]["summary"].get(k, {}).get("Np_craft_p50")
            y = B["metrics"]["summary"].get(k, {}).get("Np_craft_p50")
            if x != y:
                agg_moved.append((k, "Np_craft_p50", x, y))
    print(f"  [{'PASS' if not agg_moved else 'FAIL'}] every aggregate the paper "
          f"quotes is unchanged" + (f"  {agg_moved[:3]}" if agg_moved else ""))
    if agg_moved:
        ok = False

    changed_fields = set()
    for k in sorted(set(a) & set(b)):
        for f in set(a[k]) | set(b[k]):
            if f not in ("sets",) and a[k].get(f) != b[k].get(f):
                changed_fields.add(f)
    added = {f for k in set(b) for f in b[k]} - {f for k in set(a) for f in a[k]}
    unexpected = changed_fields - EXPECTED_TO_MOVE - WALL_CLOCK - added
    print(f"  [{'PASS' if not unexpected else 'FAIL'}] every field that moved is "
          f"either the fix or wall-clock")
    print(f"     the fix:     {sorted(changed_fields & EXPECTED_TO_MOVE)}")
    print(f"     wall-clock:  {sorted(changed_fields & WALL_CLOCK)}")
    if added:
        print(f"     new fields:  {sorted(added)}")
    if unexpected:
        print(f"     UNEXPECTED:  {sorted(unexpected)}")
        ok = False

    print(f"\n==== verify_accounting_only: "
          f"{'RETRIEVAL OUTCOMES REPRODUCIBLE' if ok else 'THE RUN CHANGED -- do not accept'} ====")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
