"""
Acceptance gate for a standalone run. Run this BEFORE any number reaches the
paper; it answers "is this run admissible as E3 evidence?", not "are the numbers
good?".

  ./.venv312/bin/python validate_results.py results/standalone.json

Exit code 0 = admissible (warnings may still apply), 1 = do not use.
"""

import json
import os
import subprocess
import sys

import analysis

HERE = os.path.dirname(os.path.abspath(__file__))

REQUIRED = ("schema_version", "run_id", "experiment_id", "backend", "index_type",
            "evidence_level", "git_commit", "metrics", "config")

# Frozen reporting thresholds. A percentile needs ~10 observations in its own
# tail to be anything but the extreme order statistic: n >= 10/(1-p). At n=25,
# P95 is the 24th value and P99 IS the maximum -- quoting either is self-
# deception, which is why this is a machine check and not a judgement call.
MIN_N = {"median": 30, "p95": 200, "p99": 1000}


def sign_p(k, n):
    """Two-sided sign test. With 5 blocks even a 5/5 split only reaches p=0.06,
    so this is here to stop a consistent direction being written up as either
    significance or absence of effect."""
    from math import comb
    if not n:
        return None
    tail = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(tail + 1)) / 2 ** n)


def strength(n):
    """Strongest statistic this sample size may carry."""
    if n >= MIN_N["p99"]:
        return "p99"
    if n >= MIN_N["p95"]:
        return "p95"
    if n >= MIN_N["median"]:
        return "median"
    return "none"


def main(path):
    recs = json.load(open(path))
    errs, warns, notes = [], [], []

    if not recs:
        print("EMPTY result file")
        return 1

    # ---- schema / provenance ------------------------------------------------
    for r in recs:
        for f in REQUIRED:
            if f not in r or r[f] is None:
                errs.append(f"{r.get('experiment_id','?')}: missing field {f}")
        if r.get("evidence_level") not in analysis.EVIDENCE_LEVELS:
            errs.append(f"{r['experiment_id']}: bad evidence_level {r.get('evidence_level')}")
        if r.get("status") == "pilot_invalidated":
            errs.append(f"{r['experiment_id']}: invalidated pilot record mixed into this file")

    tiers = {r["evidence_level"] for r in recs}
    if tiers != {"milvus_standalone"}:
        errs.append(f"expected only milvus_standalone records, found {sorted(tiers)}")
    modes = {r["config"].get("deployment_mode") for r in recs}
    if modes != {"standalone"}:
        errs.append(f"deployment_mode is {sorted(modes)} -- Lite output is not E3 evidence")
    vers = {r["config"].get("server_version") for r in recs}
    commits = {r["git_commit"] for r in recs}
    runs = {r["run_id"] for r in recs}
    profs = {r["config"].get("run_profile") for r in recs}
    notes.append(f"server_version={sorted(vers)} git_commit={sorted(commits)} "
                 f"run_id={sorted(runs)} run_profile={sorted(str(p) for p in profs)}")
    # The profile must be declared, not inferred from sample counts -- that is
    # exactly the confusion this field exists to prevent.
    if None in profs:
        warns.append("records without run_profile (produced before the field existed); "
                     "grid identity cannot be verified from metadata alone")
    if "smoke" in profs:
        errs.append("smoke-profile records present -- these are Lite code-path checks")

    s1 = [r for r in recs if "/S1/" in r["experiment_id"] and "paired" not in r["experiment_id"]]
    pr = [r for r in recs if "paired" in r["experiment_id"]]
    s2 = [r for r in recs if "/S2/" in r["experiment_id"]]
    s5 = [r for r in recs if "/S5/" in r["experiment_id"]]
    # Make a silently-absent family visible: a runner that drops all its records
    # otherwise reads here as "that experiment simply wasn't in this run".
    notes.append(f"families present: S1={len(s1)} paired={len(pr)} "
                 f"S2={len(s2)} S5={len(s5)}")

    # ---- provenance: is this run reproducible? ------------------------------
    # A result does not expire because the repository moved on. A run at a real
    # commit with a clean tree can be reproduced by checking that commit out, so
    # it stays valid evidence and is merely labelled historical. What is fatal is
    # a run whose code cannot be recovered: an unknown commit, or a dirty tree
    # whose uncommitted edits were never saved anywhere.
    def _git(*a):
        try:
            r = subprocess.run(["git", *a], cwd=HERE, capture_output=True,
                               text=True, timeout=5)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None

    head = _git("rev-parse", "--short", "HEAD")
    commits = {r.get("git_commit") for r in recs} - {None, ""}
    dirty = {r.get("git_dirty") for r in recs}
    if not head:
        warns.append("could not read git HEAD; provenance is unverified")
    elif not commits or commits == {"unknown"}:
        errs.append("records carry no usable git_commit -- the code that produced "
                    "them cannot be identified, so they are not reproducible")
    else:
        unknown = [c for c in commits if _git("cat-file", "-e", f"{c}^{{commit}}") is None]
        if unknown:
            errs.append(f"git_commit {sorted(unknown)} does not exist in this "
                        "repository -- the producing code cannot be recovered")
        elif True in dirty:
            errs.append(f"run at commit(s) {sorted(commits)} with a DIRTY working "
                        "tree -- the exact code that produced these numbers was "
                        "never committed and cannot be checked out. Re-run clean.")
        elif commits == {head}:
            notes.append(f"provenance: current-valid (run at HEAD {head}, clean tree)")
        else:
            if None in dirty:
                warns.append(f"provenance: historical-valid but the run predates the "
                             f"git_dirty field, so a clean tree at {sorted(commits)} "
                             "is assumed rather than recorded")
            else:
                notes.append(f"provenance: historical-valid (clean run at "
                             f"{sorted(commits)}; HEAD is now {head}). Still quotable "
                             "-- reproduce with: git checkout " + sorted(commits)[0])
            if len(commits) > 1:
                errs.append(f"records in ONE file span {len(commits)} commits "
                            f"{sorted(commits)} -- a single result file must come "
                            "from a single build of the measurement code")

    # ---- S1: was the treatment order balanced, and did the box hold still? --
    # A 5-block run with per-block shuffling put B4 last in 4 of 5 blocks while
    # throughput decayed with block index; the treatment effect and the position
    # effect were then the same number. Balance is a property of the design and
    # can be checked, so it is checked rather than assumed.
    if s1:
        pos = {}
        for r in s1:
            for i in (r["metrics"].get("positions") or r["config"].get("positions") or []):
                pos.setdefault(r["config"].get("baseline"), []).append(i)
        if pos and all(v for v in pos.values()):
            counts = {b: sorted(v) for b, v in pos.items()}
            nblocks = max(len(v) for v in counts.values())
            balanced = (nblocks % 4 == 0 and
                        all(sorted(v) == sorted(list(range(4)) * (nblocks // 4))
                            for v in counts.values()))
            if not balanced:
                errs.append(f"S1 treatment order is NOT position-balanced: {counts}. "
                            "Each baseline must occupy each position equally often, "
                            "or a position effect is indistinguishable from the "
                            "protocol effect. Re-run with the Williams square.")
            else:
                notes.append(f"S1 position balance: exact over {nblocks} blocks")
        else:
            warns.append("S1 records carry no position_in_block, so treatment-order "
                         "balance cannot be verified -- this run predates the check")
        sent = [(r["config"].get("baseline"), r["metrics"].get("sentinel_p95"))
                for r in s1 if r["metrics"].get("sentinel_p95") is not None]
        if sent:
            vals = [v for _, v in sent]
            if min(vals) > 0 and max(vals) / min(vals) > 1.5:
                errs.append(f"S1 sentinel P95 moved {1000*min(vals):.1f} -> "
                            f"{1000*max(vals):.1f} ms across blocks (x"
                            f"{max(vals)/min(vals):.2f}) -- the environment drifted "
                            "during the run, so between-block differences are not "
                            "attributable to the baselines")
            else:
                notes.append(f"S1 sentinel P95 stable across blocks "
                             f"({1000*min(vals):.1f}-{1000*max(vals):.1f} ms)")
        else:
            warns.append("S1 has no sentinel measurements, so cross-block "
                         "environment drift is unmonitored")
    if s1:
        wins = {round(r["metrics"]["window_s"]) for r in s1}
        if len(wins) > 1:
            errs.append(f"S1 windows differ across baselines: {sorted(wins)} "
                        "-- tail latencies are not comparable")
        qo = {r["metrics"]["qps_offered"] for r in s1}
        if len(qo) > 1:
            errs.append(f"S1 offered QPS differs across baselines: {sorted(qo)}")
        for r in s1:
            m = r["metrics"]
            # Sample count must be explained by the grid, not just be "small".
            # A cell far below offered_qps * window means the open loop did not
            # hold (serialized queries, lost results), which more samples cannot fix.
            exp_n = m["qps_offered"] * m["window_s"]
            got = m["n_queries"] / exp_n if exp_n else 0
            if got < 0.7:
                errs.append(f"{r['experiment_id']}: {m['n_queries']} queries vs "
                            f"{exp_n:.0f} expected ({m['qps_offered']} qps x "
                            f"{m['window_s']:.0f}s) = {got:.0%} -- the open loop did not "
                            "hold; a longer window will not fix this")
            else:
                notes.append(f"{r['experiment_id']}: {m['n_queries']}/{exp_n:.0f} "
                             f"expected samples ({got:.0%})")
            if m["n_queries"] < MIN_N["p99"]:
                warns.append(f"{r['experiment_id']}: only {m['n_queries']} query samples "
                             "-- P99 is smoke-grade, do not quote")
            if m["queries_dropped"]:
                warns.append(f"{r['experiment_id']}: {m['queries_dropped']} queries dropped "
                             "(client-side saturation)")
            if m["qps_achieved"] < 0.8 * m["qps_offered"]:
                warns.append(f"{r['experiment_id']}: achieved {m['qps_achieved']:.0f} of "
                             f"{m['qps_offered']} offered QPS -- open loop did not hold")
            if m.get("blocks", 0) < 3:
                warns.append(f"{r['experiment_id']}: only {m.get('blocks')} blocks")
        if not pr:
            errs.append("S1 present but no paired B4/B1 record -- pairing did not run")
    for r in pr:
        m = r["metrics"]
        span_tp = m["ratio_min"] <= 1.0 <= m["ratio_max"]
        # A band spanning 1.0 does NOT mean "no difference". Report the direction
        # too: k of n blocks below 1.0, with the sign-test p-value, so a
        # consistent-but-imprecise effect is not written up as absence of effect.
        R = m.get("ratios") or []
        below = sum(1 for x in R if x < 1.0)
        p = sign_p(below, len(R)) if R else None
        notes.append(
            f"paired throughput ratio {m['ratio_median']:.3f} "
            f"[{m['ratio_min']:.3f},{m['ratio_max']:.3f}], {below}/{len(R)} blocks < 1.0"
            + (f", sign test p={p:.3f}" if p is not None else "") + " -> "
            + (f"band spans 1.0: report the direction ({below}/{len(R)} favour lower B4) "
               "and that the magnitude is unresolved -- NOT 'no difference'"
               if span_tp else "band excludes 1.0: a retention figure may be quoted"))
        if span_tp and R:
            # The sentence to paste into 7.5, so the hedging survives the trip
            # from this gate into the manuscript intact.
            notes.append(
                f"  -> write exactly: \"{below}/{len(R)} paired blocks showed lower B4 "
                "throughput, but the magnitude remains unresolved because the small "
                "number of blocks and run-to-run variability prevent a precise "
                "estimate.\"  (No retention percentage may accompany this.)")
        if m.get("p99_delta_min") is not None:
            span99 = m["p99_delta_min"] <= 0 <= m["p99_delta_max"]
            notes.append(
                f"paired P99 delta {1000*m['p99_delta_median']:.1f} ms "
                f"[{1000*m['p99_delta_min']:.1f},{1000*m['p99_delta_max']:.1f}] -> "
                + ("band spans 0: P99 impact NOT resolved"
                   if span99 else "band excludes 0: a P99 delta may be quoted"))

    # ---- S2: probes independent, delays separable, censoring visible --------
    ctrl = [r for r in s2 if "query_only" in r["experiment_id"]]
    main2 = [r for r in s2 if "query_only" not in r["experiment_id"]]
    for r in main2:
        m = r["metrics"]
        if r["config"].get("index_type") != "FLAT":
            errs.append(f"{r['experiment_id']}: S2 must use FLAT; an approximate index "
                        "makes 'item absent' ambiguous between hide and ANN miss")
        if m["n_observed"] < 20:
            warns.append(f"{r['experiment_id']}: only {m['n_observed']} observations")
        if m["n_censored"]:
            notes.append(f"{r['experiment_id']}: {m['n_censored']} censored "
                         "(never confirmed within probe budget) -- keep in the reported tail")
        ids = {m.get("writer_client_id"), m.get("state_probe_client_id"),
               m.get("search_probe_client_id")} - {None}
        if r["config"].get("consistency") == "Session" and len(ids) > 1:
            errs.append(f"{r['experiment_id']}: writer and probes use different clients "
                        f"({len(ids)}) -- Session reads at a cached last-write ts "
                        "(pymilvus 2.4: a process-wide GTsDict keyed by collection), so "
                        "split them across processes and this stops being "
                        "read-your-own-write")
        # A level declared on the collection but absent from the request means
        # pymilvus stamped a cached mutation ts instead: the axis is not exercised.
        sent = m.get("consistency_sent_on_request")
        if sent is None:
            warns.append(f"{r['experiment_id']}: no consistency level on the probe "
                         "requests (collection default only) -- this cell cannot be "
                         "read as a point on the consistency axis")
        elif sent != r["config"].get("consistency"):
            errs.append(f"{r['experiment_id']}: sent consistency {sent!r} != configured "
                        f"{r['config'].get('consistency')!r}")
        # Report the residual itself, and compare it against the measurement
        # uncertainty it has to clear. Resolvability is a quantitative question;
        # a binary threshold note is not a substitute for the number the paper
        # would actually quote.
        c, p = m.get("delta_confirm_p95"), m.get("probe_search_p95")
        if c is not None and p is not None:
            resid = c - p
            unc = (m.get("probe_resolution_s") or 0) + (m.get("probe_scheduling_lag_p95") or 0)
            if c <= 2 * p:
                notes.append(f"{r['experiment_id']}: confirm P95 {1000*c:.1f} ms vs probe "
                             f"service P95 {1000*p:.1f} ms (residual {1000*resid:.1f} ms) "
                             "-- probe cost dominates, propagation NOT separately "
                             "resolvable; report confirmation latency, not propagation")
            elif resid <= unc:
                notes.append(f"{r['experiment_id']}: residual {1000*resid:.1f} ms does not "
                             f"exceed probe cadence + scheduling lag ({1000*unc:.1f} ms) "
                             "-- inside measurement uncertainty, do not quote a "
                             "propagation magnitude")
            else:
                notes.append(f"{r['experiment_id']}: residual {1000*resid:.1f} ms "
                             f"(confirm P95 {1000*c:.1f} - probe P95 {1000*p:.1f}) exceeds "
                             f"measurement uncertainty {1000*unc:.1f} ms -- propagation is "
                             "separately resolvable at P95 in this cell")
    if ctrl and main2:
        on = [r["metrics"]["probe_search_p95"] for r in main2
              if r["config"].get("consistency") == "Strong"
              and r["config"].get("verifier_backlog") == "none"
              and r["metrics"].get("probe_search_p95") is not None]
        off = ctrl[0]["metrics"].get("probe_search_p95")
        if on and off:
            ratio = on[0] / off
            notes.append(f"probe-overhead control: ANN search P95 with scalar probe "
                         f"{1000*on[0]:.1f} ms vs without {1000*off:.1f} ms (x{ratio:.2f})")
            if ratio > 1.2:
                warns.append("scalar probe inflates ANN search P95 by >20% -- report both "
                             "arms, do not present the two-probe numbers alone")
    elif s2:
        warns.append("no probe-off control cell (only produced under --full)")
    # Four levels that return bit-identical numbers are far more likely to be an
    # unwired axis than a real null result; the two must not be reported alike.
    lv = {r["config"].get("consistency"): r["metrics"].get("delta_confirm_p95")
          for r in main2 if r["config"].get("verifier_backlog") == "none"}
    vals = [v for v in lv.values() if v is not None]
    if len(lv) > 2 and len(set(vals)) == 1:
        errs.append(f"S2: {len(lv)} consistency levels returned an identical "
                    f"delta_confirm_p95 ({vals[0]}) -- verify the axis is wired before "
                    "reporting 'consistency-independent'")

    # ---- S5: the requested index actually served the queries ----------------
    for r in s5:
        m, c = r["metrics"], r["config"]
        eff = str(m.get("index_type_effective") or "").upper()
        if c["index_type"] != "FLAT" and eff != "HNSW":
            errs.append(f"{r['experiment_id']}: requested {c['index_type']} but effective "
                        f"index is {eff or 'unknown'} -- this row measures something else")
        if str(m.get("index_state", "")).lower() not in ("finished", "indexstatefinished", "3"):
            errs.append(f"{r['experiment_id']}: index_state={m.get('index_state')} "
                        "(not Finished)")
        if str(m.get("load_state", "")).lower().find("load") < 0:
            warns.append(f"{r['experiment_id']}: load_state={m.get('load_state')}")
        avail = m.get("queries_with_full_k_available")
        if avail is not None and avail < 0.5 * c["queries"]:
            errs.append(f"{r['experiment_id']}: only {avail}/{c['queries']} queries have "
                        "k eligible items available -- the workload, not the system, "
                        "determines this cell's under-fill")
        if c.get("dim") != 384:
            warns.append(f"{r['experiment_id']}: dim={c.get('dim')}, paper specifies 384")
        # Overall and conditional recall are not independent: conditional recall
        # is measured on the full-k subset, and the under-filled remainder can
        # contribute at most 1.0 each. If the overall figure exceeds that bound
        # the two columns are not measuring what their names say -- different
        # ground truth, mismatched query sets, or a mislabelled column.
        pf, cond, uf = m.get("pf_recall"), m.get("cond_recall"), m.get("underfill")
        if None not in (pf, cond, uf):
            bound = (1 - uf) * cond + uf
            if pf > bound + 1e-6:
                errs.append(f"{r['experiment_id']}: pf_recall {pf:.4f} exceeds the "
                            f"bound {bound:.4f} implied by cond_recall {cond:.4f} at "
                            f"{uf:.1%} under-fill -- the two recall columns cannot "
                            "both be measured as documented")
        # An out-of-contract ef would look like a real low-recall result.
        ef = (c.get("search_params") or {}).get("ef")
        if ef is not None and ef < c.get("over_fetch", 2) * c.get("k", 10):
            errs.append(f"{r['experiment_id']}: ef={ef} is below top_k="
                        f"{c.get('over_fetch', 2) * c.get('k', 10)}; Milvus 2.4 HNSW "
                        "defines ef only on [top_k, int_max], so this cell is "
                        "undefined, not a low-recall measurement")
    # ---- S5: did the adversarial workload actually apply pressure? ----------
    # clustered is the workload the paper leans on for "the upper tail, not the
    # mean, drives under-fill". That reading only holds if clustered really does
    # shift the h_M upper tail relative to random at the same global hidden
    # ratio. If it does not, the cell ran but exerted no adversarial pressure,
    # and its under-fill number must not be presented as an adversarial result.
    # heldout_attack_ranked is exempt on purpose: with a disjoint attack set it is
    # expected
    # to look like random, and that negative finding must not block the run.
    by_cell = {}
    for r in s5:
        c = r["config"]
        by_cell.setdefault((c.get("label"), c.get("N"), c.get("over_fetch")), {})[
            c.get("hidden_mode")] = r
    for key, modes in sorted(by_cell.items(), key=lambda kv: str(kv[0])):
        rand, clus = modes.get("random"), modes.get("clustered")
        if not (rand and clus):
            continue
        r95, c95 = rand["metrics"].get("hM95"), clus["metrics"].get("hM95")
        r50, c50 = rand["metrics"].get("hM50"), clus["metrics"].get("hM50")
        if r95 is None or c95 is None:
            continue
        if c95 <= r95:
            warns.append(f"S5 {key}: clustered h_M P95 {c95:.2f} does not exceed random "
                         f"{r95:.2f} -- this cell produced no tail pressure. Report it as "
                         "a null workload result; do not call it an adversarial workload.")
        elif r50 is not None and c50 is not None and abs(c50 - r50) > 0.05:
            notes.append(f"S5 {key}: clustered shifted the MEDIAN h_M too "
                         f"({r50:.2f} -> {c50:.2f}), so this pair does not isolate a "
                         "tail effect at equal mean. Say 'heavier tail and higher mean'.")
        else:
            notes.append(f"S5 {key}: clustered raises h_M P95 {r95:.2f} -> {c95:.2f} at "
                         f"comparable median ({r50} -> {c50}) -- tail pressure at equal "
                         "mean, the comparison the paper claims.")
    # Recall evidence is per operating point. Pooling n across ef settings would
    # let a high-recall configuration vouch for a weaker one, so each row gets
    # its own denominator and its own bound.
    ann = [r for r in s5 if r["config"]["index_type"] != "FLAT"]
    if ann:
        print("\n  --- S5 recall evidence per operating point ---")
        print(f"  {'config':>16} {'N':>7} {'n_fullk':>8} {'losses':>7} {'loss rate':>10} "
              f"{'95% upper':>10}")
        for r in sorted(ann, key=lambda r: r["experiment_id"]):
            m, c = r["metrics"], r["config"]
            n = m.get("cond_recall_n") or c["queries"]
            bad = m.get("cond_recall_imperfect")
            if bad is None:                       # older runs lack the counter
                bad = 0 if m.get("cond_recall") == 1.0 else None
            if bad == 0:
                ub, rate = 3.0 / n if n else 1.0, 0.0
                ubs = f"<={ub:.1%}"
            elif bad is None:
                ub, rate, ubs = None, None, "unknown"
            else:
                rate = bad / n if n else None
                ub, ubs = None, "use binomial CI"
            print(f"  {c['label']+'/'+str(c.get('hidden_mode','?')):>16} {c['N']:>7} {n:>8} "
                  f"{'?' if bad is None else bad:>7} "
                  f"{'?' if rate is None else f'{rate:.1%}':>10} {ubs:>10}")
            if bad == 0 and ub and ub > 0.05:
                warns.append(f"{r['experiment_id']}: zero losses over only {n} queries "
                             f"bounds the loss rate at {ub:.1%} -- too weak to state "
                             "recall is preserved. Raise queries to 100 (3%) or 200 "
                             "(1.5%); this does not require a larger N.")
        if all(r["metrics"].get("cond_recall") in (1.0, None) for r in ann):
            notes.append("S5: P1 stays PENDING. Write 'no loss observed over n queries "
                         "(<=X% at 95%)' per operating point -- never 'recall preserved', "
                         "and never a pooled n. Do not tune ef down to manufacture error.")

    # ---- per-experiment reporting strength (the upgrade decision) -----------
    verdicts = {}
    if s1:
        ns = [r["metrics"]["n_queries"] for r in s1]
        st = strength(min(ns))
        verdicts["S1"] = (st, f"{min(ns)}-{max(ns)} query samples/cell",
                          "UPGRADE (--full)" if st != "p99" else "sufficient")
    if main2:
        ns = [r["metrics"]["n_observed"] for r in main2]
        st = strength(min(ns))
        nxt = {"none": MIN_N["median"], "median": MIN_N["p95"], "p95": MIN_N["p99"]}
        # Percentiles from one run share a server state, a compaction schedule and
        # a backlog phase, so they are empirical tail estimates, not replicated
        # ones. Only a second independent run makes them replicated.
        label = st if st in ("none",) else f"empirical_{st}"
        verdicts["S2"] = (label, f"{min(ns)}-{max(ns)} hide events/cell, "
                                 f"replicated=false (single run)",
                          "sufficient" if st == "p99" else
                          f"UPGRADE only if the paper quotes above {st} "
                          f"(--exp S2 --s2-items {nxt[st]})")
    if s5:
        Ns = {r["config"]["N"] for r in s5}
        nq = {r["config"]["queries"] for r in s5}
        ok = all(str(r["metrics"].get("index_type_effective") or "").upper() == "HNSW"
                 for r in s5 if r["config"]["index_type"] != "FLAT")
        verdicts["S5"] = ("valid" if ok else "INVALID",
                          f"N={sorted(Ns)}, {sorted(nq)} queries",
                          "usable, scope every claim to the evaluated scale" if ok
                          else "UPGRADE: effective index is not HNSW")

    print("\n  --- reporting strength (frozen thresholds: median>=30, "
          "P95>=200, P99>=1000) ---")
    for k in ("S1", "S2", "S5"):
        if k in verdicts:
            st, detail, action = verdicts[k]
            print(f"  {k}: max quotable = {st:>6}  ({detail})  -> {action}")

    # ---- verdict ------------------------------------------------------------
    for n in notes:
        print(f"  note   {n}")
    for w in warns:
        print(f"  WARN   {w}")
    for e in errs:
        print(f"  ERROR  {e}")
    print(f"\n{len(recs)} records | {len(errs)} errors, {len(warns)} warnings")
    if errs:
        print("VERDICT: NOT admissible as E3 evidence. Fix and re-run.")
        return 1
    print("VERDICT: admissible. Warnings constrain what may be claimed, not whether "
          "the run is usable.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "results/standalone.json"))
