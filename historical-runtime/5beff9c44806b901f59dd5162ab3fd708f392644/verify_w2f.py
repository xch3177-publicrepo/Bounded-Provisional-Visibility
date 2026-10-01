#!/usr/bin/env python3
"""Acceptance gates for the false-promotion sweep (W2F).

    ./.venv312/bin/python verify_w2f.py results/W2F-inmemory.json results/W2-inmemory.json

Written before the sweep was re-run under nested promotion sets, so the
thresholds are not fitted to an outcome. The gates encode the one distinction
this experiment exists to make and is easiest to garble:

    a false promotion does NOT break the operational bound on unvetted
    visibility. It breaks the CONDITIONAL bound on poisoning exposure.

Gate 1  k=0 reproduces the W2 run cell for cell. The harness is exact and
        deterministic, so this is equality, not a tolerance. A drift here means
        the promotion switch reached something it has no business touching.
Gate 2  B1 is identical across k. B1 runs no verifier, so any movement means
        the schedule, seed, or corpus moved with k and the sweep is void.
Gate 3  E_u is unaffected at every k -- zero for B2, within the deadline for
        B4 -- while the poisoning counts are free to rise. This is the gate
        that catches the semantic error.
Gate 4  Under nested promotion sets, poisoning exposure is non-decreasing in k
        for each baseline and seed.
Gate 5  Promotion sets are nested, and are chosen without reference to which
        poison transfers best.
Gate 6  Exposure after a false promotion is right-censored, not contained. The
        window ending is not containment, and an item readmitted after its
        deadline hide must be counted as still exposed.
"""
import json
import sys

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def cells(doc, **kw):
    return [c for c in doc["metrics"]["cells"]
            if all(c.get(k) == v for k, v in kw.items())]


def main(fp_path, w2_path):
    F = json.load(open(fp_path))
    W = json.load(open(w2_path))
    ks = F["config"]["fp_sweep"]
    seeds = F["config"]["seeds"]
    Tp = F["config"]["tp"]
    print(f"=== {F['run_id']}  vs  {W['run_id']}")
    print(f"    k sweep {ks}, {len(seeds)} workloads, heavy backlog, Tp={Tp}")

    # ---- Gate 1: k=0 is the W2 run -----------------------------------------
    diffs = []
    for b in ("B1", "B2", "B3", "B4"):
        for s in seeds:
            a = cells(F, baseline=b, seed=s, false_promoted_k=0, backlog="heavy",
                      Tp=Tp)
            c = cells(W, baseline=b, seed=s, backlog="heavy", Tp=Tp)
            if not a or not c:
                diffs.append((b, s, "missing"))
                continue
            for key in ("poisoned_retrievals_craft", "poisoned_retrievals_targeted",
                        "poison_never_visible_n", "Ep_right_censored_n"):
                if a[0][key] != c[0][key]:
                    diffs.append((b, s, key, a[0][key], c[0][key]))
    check("gate 1: k=0 reproduces the W2 cells exactly", not diffs, str(diffs[:3]))

    # ---- Gate 2: B1 does not move with k -----------------------------------
    moved = []
    for s in seeds:
        vals = {k: cells(F, baseline="B1", seed=s, false_promoted_k=k,
                         backlog="heavy", Tp=Tp)[0]["poisoned_retrievals_craft"]
                for k in ks}
        if len(set(vals.values())) > 1:
            moved.append((s, vals))
    check("gate 2: B1 is identical at every k (it runs no verifier)",
          not moved, str(moved[:2]))

    # ---- Gate 3: E_u is untouched; only E_p moves ---------------------------
    # The first version of this gate asserted that B2 kept every poison item
    # invisible, and failed the moment a falsely promoted one appeared. That was
    # the confusion the gate exists to catch, committed by the gate: B2 makes an
    # item visible only AFTER verification returns, so a falsely promoted item
    # is visible-and-vetted, never visible-and-unvetted. Its E_u is zero and its
    # E_p is not, which is the entire distinction. What is asserted is E_u.
    # Asserted on whether an unvetted-visible episode ever STARTED, not on how
    # many resolved. Counting resolutions made Eu_n = 0 mean two opposite
    # things -- B2, which shows nothing unvetted, and B1, which shows poison
    # unvetted for the whole window and never resolves it -- so a gate written
    # against that count would have passed the undefended baseline.
    # A table with a positive control in it. B1 is required to show exactly the
    # failure the others must not, because a gate that only ever passes is not
    # evidence that anything was enforced.
    #
    #   baseline  started  completed  censored
    #   B1        all      0          all        unvetted throughout, unresolved
    #   B2        0        0          0          never unvetted on the query path
    #   B4        all      all        0          every episode ended at a deadline
    #
    # B3 is not given a fixed row: it ends its episodes when verification
    # commits, so whether any are still open depends on whether the queue
    # drained inside the window. It is checked against the rule instead.
    N = F["config"]["n_poison"]
    SLACK = 0.25
    bad_eu = []
    for k in ks:
        for s in seeds:
            def cell(b):
                return cells(F, baseline=b, seed=s, false_promoted_k=k,
                             backlog="heavy", Tp=Tp)[0]
            b1, b2, b4 = cell("B1"), cell("B2"), cell("B4")
            if (b1["Eu_started_n"], b1["Eu_completed_n"],
                    b1["Eu_right_censored_n"]) != (N, 0, N):
                bad_eu.append(("B1", k, s, "the control did not show unbounded "
                                           "unvetted exposure"))
            if (b2["Eu_started_n"], b2["Eu_right_censored_n"]) != (0, 0):
                bad_eu.append(("B2", k, s, "an unvetted item reached the query path"))
            if (b4["Eu_started_n"], b4["Eu_completed_n"]) != (N, N):
                bad_eu.append(("B4", k, s, "not every provisional episode closed"))
            if b4["Eu_completed_max"] is None or b4["Eu_completed_max"] > Tp + SLACK:
                bad_eu.append(("B4", k, s, f"Eu max {b4['Eu_completed_max']} "
                                           f"exceeds Tp+{SLACK}"))
            # The bound is violated by an OPEN episode only once the horizon has
            # passed the deadline it should have been closed by. Derived from
            # the recorded age rather than from this experiment's constants, so
            # the rule survives a different window.
            for b in ("B3", "B4"):
                for age in cell(b)["Eu_open_episode_ages"]:
                    if age > Tp + SLACK:
                        bad_eu.append((b, k, s, f"an episode has been open {age}s, "
                                                f"past Tp+{SLACK}"))
    check("gate 3: E_u holds at every k -- B2 admits nothing unvetted, B4 stays "
          "inside its deadline", not bad_eu, str(bad_eu[:3]))

    # ---- Gate 4: monotone in k under nesting -------------------------------
    nonmono = []
    for b in ("B2", "B3", "B4"):
        for s in seeds:
            seq = [cells(F, baseline=b, seed=s, false_promoted_k=k,
                         backlog="heavy", Tp=Tp)[0]["poisoned_retrievals_craft"]
                   for k in sorted(ks)]
            if any(x > y for x, y in zip(seq, seq[1:])):
                nonmono.append((b, s, seq))
    check("gate 4: poisoning exposure is non-decreasing in k",
          not nonmono, str(nonmono[:3]))

    # ---- Gate 5: the promotion sets are nested ------------------------------
    # Offsets, not primary keys. Each cell allocates ids from its own base, so
    # two k conditions have disjoint id ranges by construction and comparing
    # them reported a properly nested sweep as unnested -- the gate's error,
    # found only because the sweep it accused was checkable another way.
    notnested = []
    for b in ("B2", "B3", "B4"):
        for s in seeds:
            sets = [set(cells(F, baseline=b, seed=s, false_promoted_k=k,
                              backlog="heavy", Tp=Tp)[0]["false_promoted_offsets"])
                    for k in sorted(ks)]
            for a, c in zip(sets, sets[1:]):
                if not a < c or len(c) != len(a) + 1:
                    notnested.append((b, s, [sorted(x) for x in sets]))
                    break
    # The same k must select the same six-item positions for every baseline, or
    # the baselines are not being shown the same failure.
    across = []
    for s in seeds:
        for k in ks:
            got = {b: tuple(cells(F, baseline=b, seed=s, false_promoted_k=k,
                                  backlog="heavy", Tp=Tp)[0]["false_promoted_offsets"])
                   for b in ("B2", "B3", "B4")}
            if len(set(got.values())) != 1:
                across.append((s, k, got))
    check("gate 5: promotion sets are nested, one position added per step",
          not notnested, str(notnested[:1]))
    check("gate 5b: every baseline is given the same positions at the same k",
          not across, str(across[:1]))

    # Intended set versus the set the run actually promoted. B4 hides at the
    # deadline and a falsely promoted item comes back through HIDDEN->TRUSTED,
    # so the count of readmissions is observable and must equal k exactly --
    # not at least k, and not whatever the sweep produced.
    wrong_k = [(s, k, c["poison_readmitted_after_hide_n"])
               for k in ks for s in seeds
               for c in [cells(F, baseline="B4", seed=s, false_promoted_k=k,
                               backlog="heavy", Tp=Tp)[0]]
               if c["poison_readmitted_after_hide_n"] != k]
    check("gate 5c: exactly the k intended items were promoted, and each came "
          "back through the hide", not wrong_k, str(wrong_k[:3]))

    # ---- Gate 6: readmitted poison is censored, not contained ---------------
    bad_cens = []
    for k in ks:
        for s in seeds:
            for b in ("B3", "B4"):
                c = cells(F, baseline=b, seed=s, false_promoted_k=k,
                          backlog="heavy", Tp=Tp)[0]
                if c["poison_readmitted_after_hide_n"] > c["Ep_right_censored_n"]:
                    bad_cens.append((b, k, s, c["poison_readmitted_after_hide_n"],
                                     c["Ep_right_censored_n"]))
    check("gate 6: poison readmitted after its hide is counted as still exposed, "
          "not as contained", not bad_cens, str(bad_cens[:3]))

    # ---- what the paper may say --------------------------------------------
    print("\n=== the sweep, for the record")
    sm = F["metrics"]["summary"]
    print(f"    {'k':<4}{'B1':>8}{'B2':>8}{'B3':>8}{'B4':>8}   "
          f"{'B4 Eu_max':>10}{'B4 readmit':>12}")
    for k in sorted(ks):
        row = [sm[f"{b}/heavy/fp={k}"]["Np_craft_p50"] for b in
               ("B1", "B2", "B3", "B4")]
        b4 = cells(F, baseline="B4", false_promoted_k=k, backlog="heavy", Tp=Tp)
        eu = max((c["Eu_completed_max"] or 0) for c in b4)
        rd = sorted({c["poison_readmitted_after_hide_n"] for c in b4})
        print(f"    {k}/{F['config']['n_poison']:<2}" +
              "".join(f"{int(x):>8}" for x in row) + f"{eu:>10.3f}{str(rd):>12}")

    print(f"\n==== verify_w2f: {'ALL GATES PASSED' if ok_all else 'FAILURES ABOVE'} ====")
    return 0 if ok_all else 1


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
