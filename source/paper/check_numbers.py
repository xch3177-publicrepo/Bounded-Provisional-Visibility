#!/usr/bin/env python3
"""Check every W2 number in the manuscript against the frozen result files.

    ./.venv312/bin/python paper/check_numbers.py

Derives the strings that SHOULD appear in main.tex from the JSON, then looks
for them, rather than reading the two side by side. That direction is the one
that catches things: the T_p sweep was written as "1, 3, 7, 14" -- E1's numbers
-- in a paragraph that attributes its numbers to Milvus Lite, where the
shortest deadline gives 2, and the same paragraph claimed the two backends
agreed on every count. Reading it did not catch that. This did.

A FAIL means the manuscript and the data disagree. It does not mean the data
are wrong.
"""
import json
import hashlib
import math
import os
from pathlib import Path
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_RAW = open(os.path.join(HERE, "main.tex")).read()
# Comments are not the manuscript. The first version of this script reported a
# stray placeholder that turned out to be the header comment explaining what
# placeholders are.
TEX = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in _RAW.splitlines())
# Results are read through a manifest, never by picking a file out of the
# directory. That directory also holds runs from superseded code and diagnostic
# runs kept for the record, and a filename does not say which is which -- the
# one way this project has actually lost a day is comparing numbers that came
# from different code as though they were the same experiment.
RES = os.path.join(HERE, "..", "prototype", "results")
MANIFEST = json.load(open(os.path.join(RES, "AUTHORITATIVE.json")))


def authoritative(role):
    if role not in MANIFEST["files"]:
        raise SystemExit(f"{role} is not in the manifest; a number cannot come "
                         f"from a file nobody declared")
    name = MANIFEST["files"][role]
    bad = [p for p in MANIFEST["rejected_patterns"] if p in name]
    if bad:
        raise SystemExit(f"{name} matches a rejected pattern {bad}; it is an "
                         f"archive, not a result")
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        # A traceback is not a verdict. This has to read as "the gate says no",
        # not as "the gate is broken", or the next person reruns it and moves on.
        raise SystemExit(f"MANIFEST FAILURE: {name} is declared authoritative "
                         f"for {role} and does not exist. Nothing may be "
                         f"quoted until it does.")
    # json.load ACCEPTS NaN and Infinity by default. Those are not JSON, and a
    # reader that quietly takes them is how an undefined value became a
    # favourable one. Every authoritative file is parsed strictly, not just the
    # one that was caught carrying them.
    with open(path) as fh:
        try:
            doc = json.load(fh, parse_constant=lambda c: (_ for _ in ()).throw(
                ValueError(c)))
        except ValueError as exc:
            raise SystemExit(f"MANIFEST FAILURE: {name} contains {exc}, which "
                             f"is not JSON. A value that cannot be written "
                             f"cannot be quoted.")
    if doc.get("smoke"):
        raise SystemExit(f"MANIFEST FAILURE: {name} is a smoke run and cannot "
                         f"be authoritative for {role}")
    if doc.get("git_dirty"):
        raise SystemExit(f"MANIFEST FAILURE: {name} came from a dirty tree")
    return doc


E2, E1 = authoritative("w2_milvus"), authoritative("w2_inmemory")
R2, R1 = authoritative("w2r_milvus"), authoritative("w2r_inmemory")
# Comparing the recorded HEADs directly would be the wrong test. Each run reads
# it when it finishes, so a chain of runs that spans a commit to a test file
# records four different hashes while executing identical experiment code.
# What has to match is the code that ran, so the blobs are compared at those
# commits. This also catches the case the HEAD comparison would MISS: two runs
# on the same HEAD where one had an uncommitted edit.
import subprocess

CODE = ["poison_exposure.py", "functional_slice.py", "realtext_workload.py",
        "backend.py", "milvus_backend.py"]


def code_fingerprint(commit):
    out = []
    for f in CODE:
        r = subprocess.run(["git", "rev-parse", f"{commit}:prototype/{f}"],
                           cwd=os.path.join(HERE, ".."),
                           capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"cannot resolve {f} at {commit}: {r.stderr.strip()}")
        out.append(r.stdout.strip())
    return tuple(out)


fails = []

_fp = {d["run_id"]: code_fingerprint(d["git_commit"]) for d in (E1, E2, R1, R2)}
if len(set(_fp.values())) != 1:
    raise SystemExit("the runs did not execute the same experiment code:\n" +
                     "\n".join(f"  {k}: {v}" for k, v in _fp.items()))

# The legacy runs record a hash of their own runtime modules, taken from the
# working tree when they started. Bind that hash to the frozen authority
# manifest. The current tree now also contains the separately preregistered E3
# runtime; requiring it to be byte-identical to the older W2 runtime would make
# two valid frozen experiments mutually exclusive. E3 is independently
# reverified against the current files below.
_rt = {d["run_id"]: d.get("runtime_code_sha256") for d in (E1, E2, R1, R2)}
if any(v is None for v in _rt.values()):
    raise SystemExit(f"a run predates the runtime fingerprint: {_rt}")
if len(set(_rt.values())) != 1:
    raise SystemExit("the runs hashed different runtime code:\n" +
                     "\n".join(f"  {k}: {v[:16]}" for k, v in _rt.items()))
_legacy_bound = MANIFEST.get("runtime_code_sha256") == next(iter(_rt.values()))
print(f"  [{'PASS' if _legacy_bound else 'FAIL'}] legacy results bind to the "
      f"frozen runtime authority  {next(iter(_rt.values()))[:12]}")
if not _legacy_bound:
    fails.append("legacy runtime authority mismatch")
if any(d["git_dirty"] for d in (E1, E2, R1, R2)):
    raise SystemExit("a run was produced from a dirty working tree")
_cfgs = {d["run_id"]: d.get("config_sha256") for d in (E1, E2)}
if len(set(_cfgs.values())) != 1:
    raise SystemExit(f"the two synthetic runs used different configs: {_cfgs}")


TEXN = " ".join(TEX.split())


def want(label, needle, note=""):
    """The manuscript must contain `needle`, which was derived from the data.

    Matched against the source and against a whitespace-collapsed copy of it.
    Where a line wraps is not semantic, and tying the gate to it means every
    prose edit that reflows a paragraph raises a failure about a number that
    did not change -- which teaches whoever is editing to stop believing the
    gate."""
    ok = needle in TEX or " ".join(needle.split()) in TEXN
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {needle!r}" + (f"  {note}" if note else ""))
    if not ok:
        fails.append(label)


def s(cell, key, doc=E2):
    return doc["metrics"]["summary"][cell][key]


print(f"manuscript vs {E2['run_id']} (primary) and {E1['run_id']}")
print(f"  E2 commit {E2['git_commit']} dirty={E2['git_dirty']}")

# --- attack coverage and transfer ------------------------------------------
cov_c = [c["attack_coverage_craft"] for c in E2["metrics"]["cells"]]
cov_t = [c["attack_coverage_target"] for c in E2["metrics"]["cells"]]
want("craft coverage range", f"{min(cov_c)*100:.0f}--{max(cov_c)*100:.0f}\\%")
want("target coverage range", f"{min(cov_t)*100:.0f}--{max(cov_t)*100:.0f}\\%")
# The old "no poisoned retrieval in 70 cells" assertion is gone with the claim
# it guarded. Seventy cells re-query the same eight unrelated queries per
# workload under different baselines and deadlines, so it counted repeated
# measurement as independent evidence. The replacement is the distinct-query
# denominator asserted in the transfer-geometry block below.

# The mean-over-pairs geometry (0.97 / 0.58 / 0.74) left the manuscript with
# the sentence it explained. It described the average poison-query pair, and no
# query retrieves the average pair. What replaced it is asserted below.

# --- transfer geometry, per query, which is what retrieval decides ----------
# The manuscript used to compare a MEAN over all poison-query pairs against a
# MEDIAN of per-query cutoffs. Neither governs whether a query retrieves
# poison; the best poison that query can see against its own cutoff does. These
# assertions pin the corrected statistic so it cannot drift back.
# Was behind `if False`, which read the file directly and skipped every guard
# authoritative() applies. A dead branch that silently does the unguarded thing
# is worse than no guard, because the call is right there in the source.
TG = authoritative("transfer_geometry")
for name, tag in (("synthetic", "synthetic"), ("realtext", "real-text")):
    t = TG[name]
    lo, hi = t["margin_p50_range"]
    want(f"{tag} median shortfall to the cutoff",
         f"${abs(hi):.3f}$--${abs(lo):.3f}$")
    cl = t["clearing_per_workload"]
    want(f"{tag} queries clearing their own cutoff",
         f"${min(cl)}$--${max(cl)}$")
    want(f"{tag} distinct unrelated queries",
         f"${t['negatives']['distinct_unrelated_queries']}$")
    assert t["negatives"]["clearing"] == 0, "an unrelated query now clears"
    # The paper claims per-query agreement, not a bound. The weaker "brackets
    # from above" wording it used to carry was an explanation nobody had
    # tested; testing it found the cutoff was computed over the corpus alone
    # while retrieval also competes against the clean content admitted during
    # the cell, and one real-text query was mispredicted because of it.
    obs = sorted({c["attack_coverage_target"] for c in
                  (E2 if name == "synthetic" else R2)["metrics"]["cells"]})
    pred = t["frac_clearing_range"]
    same = [round(min(obs), 4), round(max(obs), 4)] == [round(x, 4) for x in pred]
    print(f"  [{'PASS' if same else 'FAIL'}] {tag}: predicted transfer equals "
          f"observed  predicted {pred} observed [{min(obs)}, {max(obs)}]")
    if not same:
        fails.append(f"{tag} geometry disagrees with observation")
want("per-query agreement count", "$80$")

# The abstract must quote the workload it names. Quoting the other one is the
# failure this guards: the two workloads' counts are close enough that a
# mismatch would read as correct.
ABS = TEX.split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
abs_real = "natural-language" in ABS
abs_src = R2 if abs_real else E2
for cell, lbl in (("B3/heavy", "B3"), ("B4/heavy", "B4")):
    n, rng = s(cell, "Np_craft_p50", abs_src), s(cell, "Np_craft_range", abs_src)
    hit = f"{int(n)} ({int(rng[0])}--{int(rng[1])})" in " ".join(ABS.split())
    print(f"  [{'PASS' if hit else 'FAIL'}] abstract {lbl} count comes from the "
          f"workload the abstract names ({'real text' if abs_real else 'synthetic'})")
    if not hit:
        fails.append(f"abstract {lbl} attribution")

# --- the headline counts, with their per-workload ranges --------------------
# These moved from prose into Table I when the real-text workload arrived, and
# a table cell carries no math delimiters, so the prose needles here stopped
# matching. They are not deleted to go green: every number they guarded is
# asserted cell by cell in the table block further down, against the same JSON.

# Rates are per-workload statistics over workloads whose qualified-query counts
# differ, so a single rate invites the reader to divide the reported count by a
# single denominator and find a mismatch (35/56 = 0.625, not the 0.604 median).
# B1 and B4 are constant across workloads and may be quoted as one number; B3
# varies and must carry its range.
for base, cell in (("B1", "B1/heavy"), ("B4", "B4/heavy")):
    rates = {c["cond_craft_prr"] for c in E2["metrics"]["cells"]
             if c["baseline"] == base and c["backlog"] == "heavy" and c["Tp"] == 1.0}
    assert len(rates) == 1, f"{base} rate is no longer constant -- quote a range"
    want(f"{base} conditional rate", f"${rates.pop():g}$")
b3 = sorted(c["cond_craft_prr"] for c in E2["metrics"]["cells"]
            if c["baseline"] == "B3" and c["backlog"] == "heavy" and c["Tp"] == 1.0)
# A reported range rounds OUTWARD. Rounding the top of the range to nearest
# would print 0.62 for a measured 0.625 and understate the spread; a range that
# excludes an observed value is worse than one digit of slack.
import math
lo, hi = math.floor(b3[0] * 100) / 100, math.ceil(b3[-1] * 100) / 100
want("B3 conditional rate carries its spread",
     f"${lo:.2f}$--${hi:.2f}$", f"per-workload {b3}")

# --- clean cost -------------------------------------------------------------
want("displacement while exposed",
     f"${s('B1/heavy','displaced_craft_during_p50')*100:.0f}$\\%",
     f"B1/B3/B4 all "
     f"{ {s(c,'displaced_craft_during_p50') for c in ('B1/heavy','B3/heavy','B4/heavy')} }")
# Clean delay, cumulative displacement and the T_p sweep are Table I cells now,
# for both workloads, and are asserted as cells below. The tier-attribution
# check that used to live here is preserved: the table's synthetic column is
# built from E2 and its real-text column from R2, and both are compared against
# their own file.
np_e2 = [s(f"B4/heavy/Tp={t}", "Np_craft_p50") for t in sorted(E2["config"]["tp_sweep"])]
np_e1 = [s(f"B4/heavy/Tp={t}", "Np_craft_p50", E1) for t in sorted(E1["config"]["tp_sweep"])]

# --- W2R, the real-text workload -------------------------------------------
# Every number Sec. VII-C prints, re-derived from its own frozen files. Added
# with the section rather than after it: an unguarded number is how 421.0 got
# changed to 421.1 on a confident but wrong suggestion with nothing objecting.
print(f"\nmanuscript vs {R2['run_id']} and {R1['run_id']}")

assert R2["workload"] == "realtext" and R2["config"]["dim"] == 384

rcov_t = [c["attack_coverage_target"] for c in R2["metrics"]["cells"]]
want("W2R target coverage range",
     f"${min(rcov_t)*100:.0f}$--${max(rcov_t)*100:.0f}$\\%")

# W2R's counts are Table I's right-hand column and are asserted as cells below.
assert s("B3/normal", "Np_craft_p50", R2) == s("B4/normal", "Np_craft_p50", R2)
rnp2 = [s(f"B4/heavy/Tp={t}", "Np_craft_p50", R2) for t in sorted(R2["config"]["tp_sweep"])]
rnp1 = [s(f"B4/heavy/Tp={t}", "Np_craft_p50", R1) for t in sorted(R1["config"]["tp_sweep"])]
if False:
    want("unused",
     f"E1 {rnp1} E2 {rnp2}")
# The preregistration bars comparing counts across workloads: different corpora
# in different dimensions. This is the guard against a tempting sentence.
cross = [x for x in re.split(r"(?<=\.)\s", TEX)
         if "synthetic" in x and re.search(r"\bversus\b|\bvs\.", x)
         and re.search(r"\d+\s*(and|to|versus|vs\.)\s*\d+", x)]
print(f"  [{'PASS' if not cross else 'FAIL'}] no count compared across the two "
      f"workloads (preregistration bars it)" + (f"  {cross[:1]}" if cross else ""))
if cross:
    fails.append("cross-workload count comparison")
# --- the table cells, which no prose sentence now carries -------------------
# Moving numbers out of prose and into a table moves them out of every check
# that reads sentences, so the table rows are asserted cell by cell here.
def row(doc, cell, key="Np_craft_p50", rng="Np_craft_range"):
    n, r = s(cell, key, doc), s(cell, rng, doc)
    return f"{int(n)}" if r[0] == r[1] else f"{int(n)} ({int(r[0])}--{int(r[1])})"


for tag, doc in (("synthetic", E2), ("real text", R2)):
    for b in ("B1", "B2", "B3", "B4"):
        want(f"table cell {tag} {b} heavy", row(doc, f"{b}/heavy"))
    want(f"table cell {tag} B3/B4 normal", row(doc, "B3/normal"))
    assert row(doc, "B3/normal") == row(doc, "B4/normal")
    # The clean-delay row left the table for prose (three quantities do not
    # fit one cell); the freshness block above derives them from the same file.
    want(f"table cell {tag} displacement",
         " / ".join(str(int(s(c, "D_H_craft_p50", doc)))
                    for c in ("B1/heavy", "B3/heavy", "B4/heavy")))
    want(f"table cell {tag} T_p sweep",
         ", ".join(str(s(f"B4/heavy/Tp={t}", "Np_craft_p50", doc))
                   for t in sorted(doc["config"]["tp_sweep"])))

same_r = all(s(c, "Np_craft_p50", R1) == s(c, "Np_craft_p50", R2)
             for c in ("B1/heavy", "B2/heavy", "B3/heavy", "B4/heavy"))
print(f"  [{'PASS' if same_r else 'FAIL'}] W2R tiers agree on the main grid, as "
      f"the section says")
if not same_r:
    fails.append("W2R tier disagreement")

# --- clean freshness, all three quantities ----------------------------------
# Sec. VII-B attributes its numbers to Milvus Lite, so they have to come from
# that file and not from whichever one is loaded. Asserted rather than left as
# a checklist item: a checklist is a thing that gets skipped, and this project
# has already quoted E1 numbers in an E2 paragraph once.
print(f"  [{'PASS' if E2.get('evidence_level') == 'milvus_lite_flat' else 'FAIL'}] "
      f"the paragraph's own tier is the file it reads  "
      f"{E2.get('evidence_level')} / {E2.get('backend')}")
if E2.get("evidence_level") != "milvus_lite_flat":
    fails.append("wrong evidence tier for VII-B")

# The sentence these guard replaced one that read a first-visibility number as
# a claim about continuous availability. Each of the three is derived here, and
# the expiry count is checked per workload rather than as a median, because
# "all six in every workload" and "a median of six" are different statements.
want("B2 clean first visibility, verifier keeping up",
     f"${s('B2/normal', 'Df_clean_p50'):.2f}$\\,s")
want("B2 clean first visibility under backlog",
     f"${s('B2/heavy', 'Df_clean_p50'):.1f}$\\,s")
# Immediate, not exactly zero: on Milvus Lite the insert round-trip is a
# millisecond or two, and writing 0 would claim a precision the backend does
# not have. The manuscript says "within two milliseconds"; the bound is checked
# rather than the digits.
# The bound is over EVERY reported cell, not over the two medians: a median can
# sit under a threshold that individual workloads cross, and the manuscript
# states a bound.
_b4_all = [c["Df_clean_p50"] for c in E2["metrics"]["cells"]
           if c["baseline"] == "B4" and c["Df_clean_p50"] is not None]
_b4n, _b4h = s("B4/normal", "Df_clean_p50"), s("B4/heavy", "Df_clean_p50")
_imm = bool(_b4_all) and max(_b4_all) <= 0.003
print(f"  [{'PASS' if _imm else 'FAIL'}] B4 first visibility is within the "
      f"stated bound in every cell  max {max(_b4_all) if _b4_all else 'n/a'} "
      f"over {len(_b4_all)} cells")
if not _imm:
    fails.append("B4 first visibility")
want("B4 first visibility is stated as a bound", "within three\nmilliseconds")
_b4hc = [c for c in E2["metrics"]["cells"] if c["baseline"] == "B4"
         and c["backlog"] == "heavy" and c["Tp"] == E2["config"]["tp"]]
_exp = {c["clean_expired_n"] for c in _b4hc}
print(f"  [{'PASS' if _exp == {E2['config']['n_clean']} else 'FAIL'}] every "
      f"workload expired all its clean items, not just the median one  {_exp}")
if _exp != {E2["config"]["n_clean"]}:
    fails.append("expiry is not per-workload")
_re = sorted({c["clean_readmitted_n"] for c in _b4hc})
_cen = sorted({c["clean_gap_right_censored_n"] for c in _b4hc})
print(f"  [{'PASS' if len(_re) == 1 and len(_cen) == 1 else 'FAIL'}] readmission "
      f"is the same in every workload, so a single number describes it  "
      f"readmitted {_re} censored {_cen}")
if not (len(_re) == 1 and len(_cen) == 1):
    fails.append("readmission varies across workloads")
want("readmitted count", {5: "five", 4: "four", 6: "six"}.get(_re[0], str(_re[0])))
want("censored count", {1: "one", 2: "two"}.get(_cen[0], str(_cen[0])))
# "median 5.3 s, denominator 5" is ambiguous across three readings: five
# completed gaps per workload, five workloads, or the pooled twenty-five. The
# aggregation is stated, and the counts that make it up are printed.
_gaps = sorted(c["clean_gap_p50"] for c in _b4hc)
_gmed = _gaps[len(_gaps) // 2]
_ncomp = sum(c["clean_gap_completed_n"] for c in _b4hc)
want("the hidden gap", f"${_gmed:.1f}$\\,s",
     f"median of {len(_gaps)} per-workload medians, "
     f"{_ncomp} completed gaps in total")
want("the gap aggregation is named",
     "median across the five per-workload median completed-gap")
want("each workload's completed-gap count is stated",
     "each workload contributing five completed gaps"
     if len({c["clean_gap_completed_n"] for c in _b4hc}) == 1
        and _b4hc[0]["clean_gap_completed_n"] == 5
     else f"contributing {_b4hc[0]['clean_gap_completed_n']}")

# --- the false-promotion sweep ----------------------------------------------
# Three sentences in the manuscript, every number of them derived here. The
# sweep exists to measure where the paper's own conditional bound stops
# holding, so an unguarded number in it would be the worst place for one.
FP = authoritative("w2f_inmemory")
_fs = FP["metrics"]["summary"]
_ks = sorted(FP["config"]["fp_sweep"])
want("sweep k values", ", ".join(str(k) for k in _ks[:-1]) + f", {_ks[-1]}")
want("undefended baseline across the sweep",
     f"${_fs[f'B1/heavy/fp={_ks[0]}']['Np_craft_p50']}$")
_b4 = [_fs[f"B4/heavy/fp={k}"]["Np_craft_p50"] for k in _ks]
want("the protocol across the sweep",
     f"${_b4[0]}$ to ${_b4[1]}$, ${_b4[2]}$ and ${_b4[3]}$")
# A bound over every cell, not the value of one. The sweep's per-k maxima are
# 1.010, 1.008, 1.006, 1.007 -- quoting one of them as "the" figure would be
# picking a cell, and the sentence claims a bound anyway.
_eu = max(c["Eu_completed_max"] for c in FP["metrics"]["cells"]
          if c["baseline"] == "B4")
import math as _mm
_eu_ceil = _mm.ceil(_eu * 100) / 100
want("the unvetted bound held across the sweep", f"${_eu_ceil:.2f}$\\,s",
     f"max over cells {_eu:.4f}, rounded outward")
_b1set = {_fs[f"B1/heavy/fp={k}"]["Np_craft_p50"] for k in _ks}
print(f"  [{'PASS' if len(_b1set) == 1 else 'FAIL'}] the undefended baseline is "
      f"one number across the sweep, as the sentence says  {_b1set}")
if len(_b1set) != 1:
    fails.append("B1 varies across the sweep")

# --- eligibility and containment --------------------------------------------
# This grid was quoted for weeks from a file outside the manifest, produced by
# a commit two renames old, and four of its six numbers had drifted from what
# the artifact says -- including a recall figure that holds only for the worst
# case and was written unqualified. Every number in that paragraph is derived
# here from the file the manifest names.
ELIG = json.load(open(os.path.join(RES, MANIFEST["files"]["eligibility_containment"])))
_e = {r["experiment_id"]: r["metrics"] for r in ELIG}
# The aggregation has to be the one the manuscript names. "(3 seeds)" does not
# say whether the queries were pooled or the per-seed rates averaged, and the
# paragraph now says which.
_aggs = {m.get("_aggregation") for k, m in _e.items() if k.startswith("elig")}
if _aggs != {"mean_across_seed_level_metrics"}:
    raise SystemExit(f"eligibility aggregation is {_aggs}; the manuscript says "
                     f"each seed's own metric is computed before aggregation")
want("eligibility aggregation is named", "means across three seeds")
# NaN is not JSON. A strict reader rejects it and a lenient one substitutes,
# and one such substitution reached the manuscript as a defined value.
with open(os.path.join(RES, MANIFEST["files"]["eligibility_containment"])) as _f:
    try:
        json.load(_f, parse_constant=lambda c: (_ for _ in ()).throw(
            ValueError(c)))
        print("  [PASS] the eligibility file is strict JSON, no NaN or Infinity")
    except ValueError as _exc:
        print(f"  [FAIL] non-JSON constant in the eligibility file: {_exc}")
        fails.append("non-JSON constant")
# Every null has to say why it is null, and an undefined value must carry the
# denominator that made it undefined.
_bad_null = [(k, f) for k, m in _e.items() if k.startswith("elig")
             for f in ("cond_recall", "pf_recall", "underfill")
             if m.get(f) is None and not str(
                 m.get("_status", {}).get(f, "")).startswith("undefined")]
print(f"  [{'PASS' if not _bad_null else 'FAIL'}] every null carries a status "
      f"explaining it" + (f"  {_bad_null[:2]}" if _bad_null else ""))
if _bad_null:
    fails.append("unexplained null")
_qc1 = _e["elig/oracle_rank_coupled/of1"]
_dn = _qc1["_denominators"]["cond_recall"]
print(f"  [{'PASS' if _dn['full_k_queries'] == 0 else 'FAIL'}] the undefined cell "
      f"records the denominator that made it undefined  "
      f"{_dn['full_k_queries']:.0f} of {_dn['total_queries']:.0f} queries")
if _dn["full_k_queries"] != 0:
    fails.append("undefined cell denominator")
# A value aggregated over fewer seeds than were run must say so. Silently
# meaning over the defined subset is the same substitution one level up.
_partial = [(k, f, m["_defined_seed_n"][f], m["_total_seed_n"])
            for k, m in _e.items() if k.startswith("elig")
            for f in ("underfill", "pf_recall", "hM95")
            if m["_defined_seed_n"].get(f, 0) != m["_total_seed_n"]]
print(f"  [{'PASS' if not _partial else 'FAIL'}] every quoted metric is defined "
      f"in all three seeds" + (f"  {_partial[:2]}" if _partial else ""))
if _partial:
    fails.append("partial-seed aggregate")
# Conditional recall is pooled over queries, not a mean of per-seed ratios, and
# its coverage is recorded so a 1.0 cannot hide being defined on part of the
# workload.
for _c in ("elig/random/of2", "elig/clustered/of2"):
    _cv = _e[_c]["cond_recall_coverage"]
    print(f"  [{'PASS' if _cv is not None and _cv > 0 else 'FAIL'}] {_c} records "
          f"conditional-recall coverage  {_cv:.2f} of its queries")
    if not (_cv and _cv > 0):
        fails.append(f"{_c} coverage")
# A metric that is undefined must be described as undefined. cond_recall for
# the query-coupled 1x cell is NaN in every seed -- no query produced a full-k
# result -- and the manuscript said it "stayed 1.0".
_undef = [k for k, m in _e.items() if m.get("cond_recall") is None]
print(f"  [{'PASS' if _undef else 'FAIL'}] the undefined conditional-recall cell "
      f"is still undefined in the data  {_undef}")
if not _undef:
    fails.append("undefined cell vanished")
want("the undefined cell is called undefined", "\\emph{undefined}")
_expect_cells = {f"elig/{m}/of{o}" for m in
                 ("random", "clustered", "oracle_rank_coupled") for o in (1, 2, 4)}
missing = _expect_cells - set(_e)
if missing:
    raise SystemExit(f"the eligibility grid is missing cells: {sorted(missing)}")
_seeds = {tuple(r["config"]["seeds"]) for r in ELIG
          if r["experiment_id"].startswith("elig")}
if len(_seeds) != 1:
    raise SystemExit(f"eligibility cells disagree on seeds: {_seeds}")
_ns = len(next(iter(_seeds)))
want("eligibility seed count",
     {3: "three seeds", 5: "five seeds"}.get(_ns, f"{_ns} seeds"))

want("median h_M, shared", f"${_e['elig/random/of2']['hM50']:.2f}$")
want("clustered P95 tail", f"${_e['elig/clustered/of2']['hM95']:.2f}$")
want("random P95 tail", f"${_e['elig/random/of2']['hM95']:.2f}$")
want("random under-fill at 2x",
     f"${_e['elig/random/of2']['underfill']*100:.1f}$\\%")
want("clustered under-fill at 2x",
     f"${_e['elig/clustered/of2']['underfill']*100:.0f}$\\%")
want("oracle rank-coupled under-fill at 2x",
     f"${_e['elig/oracle_rank_coupled/of2']['underfill']*100:.0f}$\\%")
want("recall at 1x, worst case",
     f"${_e['elig/oracle_rank_coupled/of1']['pf_recall']:.2f}$")
# The unqualified "~0.40" held only for the oracle case; the other two are
# nowhere near it, and the sentence now has to carry both.
for _m, _lbl in (("random", "uniform"), ("clustered", "clustered")):
    want(f"recall at 1x, {_lbl}", f"${_e[f'elig/{_m}/of1']['pf_recall']:.2f}$")
_of4 = [_e[f"elig/{m}/of4"] for m in ("random", "clustered", "oracle_rank_coupled")]
print(f"  [{'PASS' if all(c['underfill'] == 0 for c in _of4) else 'FAIL'}] "
      f"4x over-fetch eliminated observed under-fill in every distribution")
if not all(c["underfill"] == 0 for c in _of4):
    fails.append("4x under-fill")

# --- standalone S2, the blind spot that let a bad suggestion through ---------
# check_numbers covered W2 only, so when a review proposed changing the
# instrumentation control from 421.0 to 421.1 nothing objected; the frozen
# record states 421.0. These assertions close that gap.
S2 = json.load(open(os.path.join(RES,
                                 "standalone-S2.json")))
s2 = {}
for rec in S2:
    # the cell label lives in experiment_id; run_id is "local" for every record
    tail = rec["experiment_id"].split("/S2/")[-1]
    s2[tail] = rec["metrics"]
obs = [m for k, m in s2.items()
       if isinstance(m.get("censor_width_p95"), (int, float))]
ctrl = [s2[k] for k in s2 if "control" in k]


def ms(x):
    return x * 1000.0


def outward(lo, hi, nd=0):
    """A printed interval must contain every observed value, so the low end
    floors and the high end ceils. 9.4-11.3 ms printed as 9-11 excluded a real
    observation; that is the rule this enforces."""
    f = 10 ** nd
    return math.floor(lo * f) / f, math.ceil(hi * f) / f


lo, hi = outward(min(ms(m["delta_hide_p50"]) for m in obs),
                 max(ms(m["delta_hide_p50"]) for m in obs))
want("S2 delta_hide P50 range", f"{lo:.0f}--{hi:.0f}\\,ms",
     "over the cells where the transition was observable")
lo, hi = outward(min(ms(m["delta_hide_p95"]) for m in obs),
                 max(ms(m["delta_hide_p95"]) for m in obs))
want("S2 delta_hide P95 range", f"{lo:.0f}--{hi:.0f}\\,ms")
lo, hi = outward(min(ms(m["censor_width_p95"]) for m in obs),
                 max(ms(m["censor_width_p95"]) for m in obs))
want("S2 censoring bracket", f"{lo:.0f}--{hi:.0f}\\,ms",
     f"observed {[round(ms(m['censor_width_p95']),1) for m in obs]}")

with_probe = max(ms(m["probe_search_p95"]) for k, m in s2.items()
                 if k == "Strong/nobacklog")
without = ms(ctrl[0]["probe_search_p95"]) if ctrl else None
want("S2 search path with the scalar probe", f"{with_probe:.1f}\\,ms")
if without is not None:
    want("S2 search path without it", f"{without:.1f}\\,ms")

if len(ctrl) == 2:
    a, b = (ms(c["delta_hide_p50"]) for c in ctrl)
    want("S2 control drift", f"{abs(b - a):.1f}\\,ms",
         f"{a:.1f} -> {b:.1f}")

check_strong = all(m.get("stale_probes_max") == 0 for k, m in s2.items()
                   if k.startswith(("Strong", "Session")))
print(f"  [{'PASS' if check_strong else 'FAIL'}] S2 Strong/Session recorded no "
      f"stale read in any cell")
if not check_strong:
    fails.append("S2 strong/session stale")

# --- real D1 detector and five-run E3 ---------------------------------------
# W2D has its own authority chain because it was frozen after the legacy W2
# manifest. Verify both named manifests before reading detector metrics.
def strict_json(path):
    with open(path) as fh:
        try:
            return json.load(
                fh,
                parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)),
            )
        except ValueError as exc:
            raise SystemExit(f"{path} is not strict JSON: {exc}") from exc


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


W2D_DIR = os.path.join(RES, "w2d")
W2D_AUTH = strict_json(os.path.join(W2D_DIR, "W2D-MANIFEST-AUTHORITY.json"))
_w2d_manifests = []
for _role in ("E1_INMEMORY", "E2_MILVUS_LITE"):
    _entry = W2D_AUTH["manifests"][_role]
    _path = os.path.join(W2D_DIR, _entry["path"])
    _ok = file_sha256(_path) == _entry["sha256"]
    print(f"  [{'PASS' if _ok else 'FAIL'}] W2D authority pins {_role}")
    if not _ok:
        fails.append(f"W2D authority:{_role}")
    _w2d_manifests.append(strict_json(_path))

_detector_refs = [
    manifest["artifact_hashes"]["detector_metrics"]
    for manifest in _w2d_manifests
]
_detector_ref_ok = (
    len({_ref["path"] for _ref in _detector_refs}) == 1
    and len({_ref["sha256"] for _ref in _detector_refs}) == 1
)
_detector_path = os.path.join(W2D_DIR, _detector_refs[0]["path"])
_detector_hash_ok = (
    _detector_ref_ok
    and file_sha256(_detector_path) == _detector_refs[0]["sha256"]
)
print(f"  [{'PASS' if _detector_hash_ok else 'FAIL'}] both W2D tiers pin the "
      "detector metrics bytes")
if not _detector_hash_ok:
    fails.append("W2D detector metrics authority")
DET = strict_json(_detector_path)
_dm = DET["metrics"]
_dc = _dm["overall"]["counts"]
want("D1 detector test-group count", str(_dm["unique_source_group_count"]))
want("D1 TP/FP/TN/FN",
     f"{_dc['tp']}/{_dc['fp']}/{_dc['tn']}/{_dc['fn']}")
want("D1 matched-recipe recall",
     f"{_dm['attack_recall']['recipe']['numerator']}/"
     f"{_dm['attack_recall']['recipe']['denominator']}")
want("D1 cross-attack recall",
     f"{_dm['attack_recall']['natural_cover_suffix_v1']['numerator']}/"
     f"{_dm['attack_recall']['natural_cover_suffix_v1']['denominator']}")
want("D1 clean false-refusal count",
     f"{_dc['fp']}/{_dc['fp'] + _dc['tn']}")
_dl = _dm["detector_only_latency"]
want("D1 detector-only P50/P95/P99",
     f"Detector-only P50, P95, and P99 latencies were "
     f"{_dl['p50_ns']/1e6:.1f}, {_dl['p95_ns']/1e6:.1f}, and "
     f"{_dl['p99_ns']/1e6:.1f}\\,ms, respectively")

# Re-run the independent five-file E3 verifier. This simultaneously checks
# the current E3 runtime fingerprint, all twelve per-run gates, five distinct
# process starts and collections, and the run-level-only aggregation.
E3_DIR = Path(RES) / "w2d-e3"
E3_SUMMARY = strict_json(E3_DIR / "W2D-E3-SUMMARY.json")
_e3_paths = [
    E3_DIR / Path(entry["path"]).name
    for entry in E3_SUMMARY["source_files"]
]
_source_bytes_ok = all(
    path.exists()
    and path.stat().st_size == entry["bytes"]
    and file_sha256(path) == entry["sha256"]
    for path, entry in zip(_e3_paths, E3_SUMMARY["source_files"])
)
print(f"  [{'PASS' if _source_bytes_ok else 'FAIL'}] E3 summary pins five source "
      "files by bytes and SHA-256")
if not _source_bytes_ok:
    fails.append("E3 source bytes")

_proto = str(Path(HERE).parent / "prototype")
sys.path.insert(0, _proto)
try:
    from verify_w2d_e3 import build_summary as _build_e3_summary
    from verify_w2d_e3 import verify_files as _verify_e3_files

    _validated_e3, _source_e3 = _verify_e3_files(_e3_paths)
    _recomputed_e3 = _build_e3_summary(
        _validated_e3, source_files=_source_e3
    )
    _e3_keys = (
        "status",
        "accepted_run_numbers",
        "frozen_parent_hashes",
        "cross_run_provenance",
        "transfer_condition",
        "aggregation_semantics",
        "panel_a",
        "panel_b",
        "interpretation_boundary",
    )
    _e3_ok = all(
        _recomputed_e3[key] == E3_SUMMARY[key] for key in _e3_keys
    )
except Exception as exc:
    _e3_ok = False
    print(f"  [FAIL] independent E3 revalidation raised {type(exc).__name__}: "
          f"{exc}")
print(f"  [{'PASS' if _e3_ok else 'FAIL'}] E3 five-run summary independently "
      "recomputes under the current runtime")
if not _e3_ok:
    fails.append("E3 independent revalidation")

_a64 = E3_SUMMARY["panel_a"]["64"]
_rec = _a64["recall_at_5"]
want("E3 primary ef", "\\texttt{ef}=64")
want("E3 median Recall@5", f"{_rec['median']:.4f}")
want("E3 Recall@5 range", f"{_rec['min']:.4f}--{_rec['max']:.4f}")
want("E3 median P95", f"{_a64['latency_s']['p95']['median']:.3f}\\,s")
want("E3 admission rate", "20 admissions/s")

_baselines = ("B1", "B2", "B3", "B4")
_qps1 = [
    E3_SUMMARY["panel_b"][b]["1"]["queries"]["achieved_qps"]["median"]
    for b in _baselines
]
_qps16 = [
    E3_SUMMARY["panel_b"][b]["16"]["queries"]["achieved_qps"]["median"]
    for b in _baselines
]
_p95c16 = [
    E3_SUMMARY["panel_b"][b]["16"]["queries"]["latency_s"]["p95"]["median"]
    for b in _baselines
]
want("E3 concurrency-1 median-QPS range",
     f"{min(_qps1):.2f}--{max(_qps1):.2f}")
want("E3 concurrency-16 median-QPS range",
     f"{min(_qps16):.1f}--{max(_qps16):.2f}")
want("E3 concurrency-16 median-P95 range",
     f"{min(_p95c16):.3f}--{max(_p95c16):.3f}\\,s")

_all_cells = [
    E3_SUMMARY["panel_b"][b][str(c)]
    for b in _baselines
    for c in (1, 4, 8, 16)
]
_zero_query_errors = all(cell["queries"]["errors"]["max"] == 0
                         for cell in _all_cells)
_zero_admission_failures = all(cell["admissions"]["failed"]["max"] == 0
                               for cell in _all_cells)
print(f"  [{'PASS' if _zero_query_errors else 'FAIL'}] all E3 cells have zero "
      "query errors")
print(f"  [{'PASS' if _zero_admission_failures else 'FAIL'}] all E3 cells have "
      "zero admission failures")
if not _zero_query_errors:
    fails.append("E3 query errors")
if not _zero_admission_failures:
    fails.append("E3 admission failures")
want("E3 realized false promotions",
     str(int(_all_cells[0]["decision_mix"]["false_promotion"]["median"])))
want("E3 realized false refusals",
     str(int(_all_cells[0]["decision_mix"]["false_refusal"]["median"])))

# --- cross-tier agreement claim must match what the tiers actually did ------
same = np_e1 == np_e2 and all(
    s(c, "Np_craft_p50") == s(c, "Np_craft_p50", E1)
    for c in ("B1/heavy", "B3/heavy", "B4/heavy", "B2/heavy"))
# Anchored on the claim, not on one phrasing of it. The anchor used to be the
# literal "every count", which stopped matching the moment that claim was
# narrowed to the poisoned-retrieval counts -- and a check that silently stops
# finding its subject reports on an empty string.
sent = next((x for x in re.split(r"(?<=\.)\s", " ".join(TEX.split()))
             if "exact backend" in x and "poisoned-retrieval counts" in x),
            None)
if sent is None:
    raise SystemExit("the cross-backend sentence is gone; this check has no "
                     "subject and must be re-aimed, not left passing")
qualified = "differs" in sent or "except" in sent
ok = same or qualified
print(f"  [{'PASS' if ok else 'FAIL'}] cross-tier claim matches reality: "
      f"tiers identical={same}, claim qualified={qualified}")
if not ok:
    fails.append("cross-tier claim")

# --- nothing may claim unsupported performance or overhead -----------------
for banned in ("low overhead", "negligible overhead", "preserves throughput",
               "without overhead", "no measurable overhead"):
    if banned in TEX.lower():
        print(f"  [FAIL] banned performance claim present: {banned!r}")
        fails.append(f"banned:{banned}")
print(f"  [{'PASS' if not any(f.startswith('banned') for f in fails) else 'FAIL'}] "
      f"no unsupported performance/overhead claim")

# --- abstract length and citation hygiene -----------------------------------
# The conference abstract target is deliberately narrower than IEEE's generic
# upper bound. Count source-visible tokens after removing formatting commands;
# numeric commas and TeX range dashes stay inside one token.
ABS_TEXT = TEX.split("\\begin{abstract}", 1)[1].split(
    "\\end{abstract}", 1
)[0]
_abs_countable = re.sub(r"(?<=\d),(?=\d)", "", ABS_TEXT)
_abs_countable = _abs_countable.replace("--", "-")
for _macro, _rendered in (
    ("\\Tp", "Tp"),
    ("\\dhide", "delta-hide"),
    ("\\Eu", "Eu"),
    ("\\Ep", "Ep"),
):
    _abs_countable = _abs_countable.replace(_macro, _rendered)
_abs_countable = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?", " ",
                        _abs_countable)
_abs_countable = re.sub(r"[{}$\\]", " ", _abs_countable)
_abs_words = re.findall(r"[A-Za-z0-9]+(?:[-@][A-Za-z0-9]+)*",
                        _abs_countable)
_abs_ok = 200 <= len(_abs_words) <= 230
print(f"  [{'PASS' if _abs_ok else 'FAIL'}] abstract is 200--230 words "
      f"under the submission counter: {len(_abs_words)}")
if not _abs_ok:
    fails.append(f"abstract-word-count:{len(_abs_words)}")

# Each citation command now names one source. Closely related papers may share
# a sentence, but attaching four keys to its final punctuation makes the
# supported clause unknowable and recreates citation dumping.
_bib_text = open(os.path.join(HERE, "references.bib")).read()
_bib_keys = set(re.findall(r"@[A-Za-z]+\{([^,\s]+)", _bib_text))
_cite_groups = re.findall(r"\\cite\{([^}]+)\}", TEX)
_cite_keys = [
    key.strip()
    for group in _cite_groups
    for key in group.split(",")
    if key.strip()
]
_unknown_cites = sorted(set(_cite_keys) - _bib_keys)
_dumped_cites = [group for group in _cite_groups if "," in group]
_cite_ok = not _unknown_cites and not _dumped_cites
print(f"  [{'PASS' if _cite_ok else 'FAIL'}] citations are single-source and "
      f"defined: {len(_cite_groups)} placements, {len(set(_cite_keys))} sources")
if _unknown_cites:
    print(f"    unknown keys: {_unknown_cites}")
if _dumped_cites:
    print(f"    multi-source citation groups: {_dumped_cites}")
if not _cite_ok:
    fails.append("citation-hygiene")

_bbl_path = os.path.join(HERE, "main.bbl")
if os.path.exists(_bbl_path):
    _bbl_keys = re.findall(r"\\bibitem\{([^}]+)\}", open(_bbl_path).read())
    _first_use = list(dict.fromkeys(_cite_keys))
    _bbl_ok = _bbl_keys == _first_use
    print(f"  [{'PASS' if _bbl_ok else 'FAIL'}] bibliography numbering follows "
          "first citation order without gaps")
    if not _bbl_ok:
        fails.append("bibliography-order")

# --- placeholders -----------------------------------------------------------
pend = [x for x in re.findall(r"\\pend\{([^}]*)\}", TEX) if x != "#1"]
print(f"  [{'PASS' if not pend else 'FAIL'}] no placeholders remain: {pend or 'none'}")
if pend:
    fails.append("placeholders")
# The author block is the last thing filled in and the easiest to fill in wrong.
for who in ("Chuhong Xu", "Hang Xiao", "Yueyuan He", "Kainan Zhou", "Gangzhen Qian",
            "REDACTED_EMAIL", "REDACTED_EMAIL", "REDACTED_EMAIL",
            "REDACTED_EMAIL", "REDACTED_EMAIL",
            "Sony Corporate of America", "Fortinet, Inc.", "Google LLC"):
    if who not in TEX:
        print(f"  [FAIL] author block is missing {who!r}")
        fails.append("author:" + who)
print(f"  [{'PASS' if not any(f.startswith('author:') for f in fails) else 'FAIL'}] "
      f"all five authors, affiliations and addresses present")

print(f"\n==== check_numbers: {'ALL PASS' if not fails else f'{len(fails)} FAILURES: ' + ', '.join(fails)} ====")
sys.exit(0 if not fails else 1)
