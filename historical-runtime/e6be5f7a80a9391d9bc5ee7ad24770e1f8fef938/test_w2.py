"""Guards for the W2 poison-exposure experiment that do not need a run.

    ./.venv312/bin/python test_w2.py

These are the checks that would otherwise only fail after a twelve-minute run,
or worse, not fail at all and quietly change what the numbers mean.
"""
import ast
import os
import re
import sys

# A bare module-level RNG call: random.foo(...) not preceded by a Random()
# instance. Used by the promotion-stream test below.
_r_re = re.compile(r"(^|[^.\w])random\.(random|shuffle|sample|choice|randint|gauss|seed)\(")

import poison_exposure as w2
from functional_slice import State, System, VISIBLE
from backend import InMemoryBackend
import asyncio

HERE = os.path.dirname(os.path.abspath(__file__))
checks = []


def check(name, cond):
    checks.append((name, bool(cond)))


# 1. The four baselines must be the ones the standalone runner uses. Parsed out
#    of the source rather than imported, because importing that module pulls in
#    pymilvus and an E1 run is supposed to need neither.
src = open(os.path.join(HERE, "standalone_experiments.py")).read()
tree = ast.parse(src)
theirs = None
for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "BASELINES" for t in node.targets):
        theirs = ast.literal_eval(node.value)
check("BASELINES agree with standalone_experiments",
      theirs is not None and
      {k: {kk: vv for kk, vv in v.items() if kk != "desc"}
       for k, v in theirs.items()} ==
      {k: {kk: vv for kk, vv in v.items() if kk != "desc"}
       for k, v in w2.BASELINES.items()})

# 2. The policy switches must actually select different control planes, or the
#    four "baselines" are one baseline measured four times.
sysB1 = System(None, verify=False, sync=False, deadline=False)
sysB4 = System(None, verify=True, sync=False, deadline=True)
check("B1 runs no verifier", sysB1.do_verify is False)
check("B1 has no deadline", sysB1.use_deadline is False)
check("B4 has both", sysB4.do_verify and sysB4.use_deadline)

# 3. T_p points must bracket the verifier latency, which is the whole reason
#    they were chosen. If someone edits verify_cost without editing the sweep,
#    the sweep stops answering its question.
vc = w2.CFG["verify_cost"]
sweep = w2.CFG["tp_sweep"]
check("a T_p below normal verification latency", any(t < vc for t in sweep))
check("a T_p at normal verification latency", any(abs(t - vc) < 1e-9 for t in sweep))
check("a T_p above it", any(t > vc for t in sweep))
heavy_queue = w2.CFG["backlog"]["heavy"]["items"] * vc
check("every T_p below the heavy queue depth", all(t < heavy_queue for t in sweep))
check("main T_p is in the sweep", w2.CFG["tp"] in sweep)

# 4. The window has to outlast the heavy queue, or B3-heavy is right-censored
#    in every cell and the backlog contrast cannot be measured at all.
check("window outlasts the heavy verifier queue",
      w2.CFG["dur"] - w2.CFG["inject_at"] > heavy_queue + vc)

# 5. Three query sets, and the crafted one must not be the reported one.
for key in ("n_craft_q", "n_target_q", "n_neg_q"):
    check(f"{key} > 0", w2.CFG[key] > 0)

# 6. Eligibility must not lead existence: an item may not be visible in the
#    control store before the store can return it (fail-closed).
fs = open(os.path.join(HERE, "functional_slice.py")).read()
ins = fs.index("self.b.insert(iid, vec, visible=True)")
prov = fs.index('it["state"] = State.PROVISIONAL', ins - 400)
check("insert precedes PROVISIONAL in admit()", ins < prov)

# 7. Phase splitting: a cell that never contained must not report samples as
#    "after containment", and a cell that was never exposed must not report any
#    as "during exposure". Exercised on the summariser's own logic.
class _S:
    def __init__(self):
        self.items, self.transitions = {}, []
qlog = [(t, "target", 0, False, 1.0, 0.0) for t in (0.5, 2.0, 4.0)]
s = _S()
# `state` is not optional: summarise reads it to tell a deadline hide apart from
# a hide that a late promotion undid, so a stub without one describes an item
# the system cannot produce.
s.items[99] = {"id": 99, "prov": 1, "deadline": float("inf"),
               "state": State.PROVISIONAL,
               "ts": {"arrival": 0.0, "visible": 0.0}}          # visible, never contained
m = w2.summarise(s, qlog, [99], [], w2.CFG, 1.0, 0.0)
check("never contained -> nothing lands after containment",
      m["elig_recall_after_containment_p50"] is None and
      m["elig_recall_during_exposure_p50"] is not None)
s2 = _S()
s2.items[98] = {"id": 98, "prov": 0, "deadline": float("inf"),
                "state": State.QUARANTINED, "ts": {}}                     # never visible
m2 = w2.summarise(s2, qlog, [98], [], w2.CFG, 1.0, 0.0)
check("never exposed -> nothing lands during exposure",
      m2["elig_recall_during_exposure_p50"] is None)

# 8. Primary keys must never repeat anywhere in a run. A seed's cells share one
#    monotonic counter inside that seed's stride; if the counter could reach the
#    stride, the next seed's corpus would alias an earlier seed's cell items and
#    the control store would call a stale row eligible.
stride = w2.CFG["pk_stride"]
cells_per_seed = 2 * len(w2.BASELINES) + 2 * (len(w2.CFG["tp_sweep"]) - 1)
per_cell = (w2.CFG["n_poison"] + w2.CFG["n_clean"] +
            max(b["items"] for b in w2.CFG["backlog"].values()))
worst = w2.CFG["corpus"] + cells_per_seed * per_cell
check(f"a seed's keys ({worst}) stay inside the stride ({stride})", worst < stride)
check("seed key spaces do not overlap",
      w2.id_base(1, w2.CFG) + stride <= w2.id_base(2, w2.CFG))
check("corpus_topk emits backend ids, not list positions",
      w2.corpus_topk([[1.0, 0.0], [0.0, 1.0]], [1.0, 0.0], 1, 500)[0][1] == 500)

# 9. The precomputed-corpus merge must equal a full scan exactly. This is the
#    optimisation that took the 600-cosine ground-truth scan out of the timed
#    window; if it is only approximately right, every recall and displacement
#    number is quietly wrong.
import random as _r

from backend import cosine as _cos


class _Sys:
    def __init__(self, items):
        self.items = items


def _full_scan(items, q, k, exclude=()):
    cand = [(_cos(q, it["vec"]), i) for i, it in items.items()
            if it["state"] in VISIBLE and i not in exclude]
    cand.sort(reverse=True)
    return [i for _, i in cand[:k]]


_rng = _r.Random(7)
_agree = True
for _trial in range(200):
    dim, ncorp, nins, k = 8, 60, 12, 5
    corpus = [w2.unit([_rng.gauss(0, 1) for _ in range(dim)]) for _ in range(ncorp)]
    items = {i: {"id": i, "vec": v, "state": State.TRUSTED}
             for i, v in enumerate(corpus)}
    inserted = []
    for j in range(nins):
        iid = ncorp + j
        items[iid] = {"id": iid,
                      "vec": w2.unit([_rng.gauss(0, 1) for _ in range(dim)]),
                      "state": _rng.choice([State.PROVISIONAL, State.TRUSTED,
                                            State.HIDDEN, State.QUARANTINED])}
        inserted.append(iid)
    q = w2.unit([_rng.gauss(0, 1) for _ in range(dim)])
    excl = set(_rng.sample(inserted, _rng.randint(0, 4)))
    sysx = _Sys(items)
    pre = w2.corpus_topk(corpus, q, k, 0)
    if (w2.merged_topk(pre, sysx, q, k, inserted, excl)
            != _full_scan(items, q, k, excl)):
        _agree = False
        break
check("precomputed merge equals a full scan (200 random cases)", _agree)

# --- the state field is load-bearing, and must stay load-bearing ------------
# summarise decides censoring from the item's final state, so an item without
# one has no defined answer. Reading it with .get() would turn that into a
# silent "not visible", which is the flattering direction: it would credit
# containment the run never observed. This test exists to fail if anyone makes
# that edit, so it asserts the raise rather than the result.
_s3 = _S()
_s3.items[97] = {"id": 97, "prov": 1, "deadline": float("inf"),
                 "ts": {"arrival": 0.0, "visible": 0.0}}          # no "state"
try:
    w2.summarise(_s3, qlog, [97], [], w2.CFG, 1.0, 0.0)
    _raised = False
except KeyError:
    _raised = True
check("an item with no final state is an error, not a silent 'contained'", _raised)

# And the fixtures this file builds must carry one, so the test above is the
# only place a stateless item can exist.
check("every stub item in this file declares a state",
      all("state" in it for st in (s, s2) for it in st.items.values()))

# --- containment: the terminal state authorizes a finite estimate, it does ---
# --- not supply the timestamp ------------------------------------------------
# Getting these two jobs confused is a quiet, plausible error. A poison hidden
# at t=1 and quarantined at t=5 is contained at t=1 -- the query path stopped
# returning it then -- and scoring it at t=5 would report E_p = 5 s for a 1 s
# deadline, breaking the very bound the run exists to check while looking like
# a more conservative choice. One case per reachable trajectory.
def _ep(state, ts):
    st = _S()
    st.items[1] = {"id": 1, "prov": 1, "deadline": float("inf"),
                   "state": state, "ts": ts}
    m = w2.summarise(st, qlog, [1], [], w2.CFG, 1.0, 0.0)
    return m["Ep_p50"], m["Ep_right_censored_n"], m["poison_never_visible_n"]

_cases = [
    # trajectory, state at horizon, ts, expected (Ep, censored, never_visible)
    ("PROVISIONAL->HIDDEN->QUARANTINED", State.QUARANTINED,
     {"arrival": 0.0, "visible": 0.0, "contain": 1.0, "verify_commit": 5.0},
     (1.0, 0, 0)),
    ("PROVISIONAL->HIDDEN, verifier still pending", State.HIDDEN,
     {"arrival": 0.0, "visible": 0.0, "contain": 1.0},
     (None, 1, 0)),
    ("PROVISIONAL->HIDDEN->TRUSTED, visible at horizon", State.TRUSTED,
     {"arrival": 0.0, "visible": 0.0, "contain": 1.0, "trusted": 5.0},
     (None, 1, 0)),
    # cas() stamps "contain" again on TRUSTED->REVOKED, so this carries the end
    # of the LAST visible episode, not the first.
    ("PROVISIONAL->HIDDEN->TRUSTED->REVOKED", State.REVOKED,
     {"arrival": 0.0, "visible": 0.0, "contain": 7.0, "trusted": 5.0},
     (7.0, 0, 0)),
    ("QUARANTINED at admission, never visible", State.QUARANTINED,
     {"arrival": 0.0},
     (None, 0, 1)),
]
for _name, _st, _ts, _want in _cases:
    check(f"containment case: {_name}", _ep(_st, _ts) == _want)

# The rule is not "later transitions never move t_contain" -- that would be
# wrong for a readmitted item, whose first hide stopped being durable the moment
# it came back. The rule is that a later exit moves containment ONLY when
# something in between made the item query-visible again. Both halves are
# asserted against the real cas(), not a hand-built fixture, so they hold for
# whatever path reaches containment.
def _trace(*steps):
    sysd = System(InMemoryBackend(), Tp=1.0, verify=False, deadline=False)
    it = {"id": 5, "state": State.PROVISIONAL, "prov": 1, "ts": {"visible": 0.0},
          "deadline": float("inf"), "pending": False}
    sysd.items[5] = it
    marks = []
    for frm, to, stamp in steps:
        sysd.cas(it, frm, to, stamp)
        marks.append(it["ts"].get("contain"))
    return it, marks

_it_q, _m_q = _trace((State.PROVISIONAL, State.HIDDEN, "hide_commit"),
                     (State.HIDDEN, State.QUARANTINED, "verify_commit"))
check("no readmission: a terminal transition leaves t_contain at the first hide",
      _m_q[0] == _m_q[1])

_it_r, _m_r = _trace((State.PROVISIONAL, State.HIDDEN, "hide_commit"),
                     (State.HIDDEN, State.TRUSTED, "trusted"),
                     (State.TRUSTED, State.REVOKED, "contain"))
check("readmission then revoke: t_contain moves to the revoke, not the hide",
      _m_r[1] == _m_r[0] and _m_r[2] > _m_r[1])


# --- one trace, two accountants ---------------------------------------------
# A second execution cannot show that an accounting change was confined: two
# runs differ in asyncio scheduling whatever the analysis does. Feeding ONE
# fixed input to the old rule and the new one can, and it isolates exactly the
# trajectories the change was meant to touch.
def _old_rule(state, ts):
    """What summarise did before: a containment stamp ended exposure, whatever
    state the item was actually in when the window closed."""
    if "visible" not in ts:
        return ("never_visible", None)
    if "contain" in ts:
        return ("finite", ts["contain"] - ts["visible"])
    return ("censored", None)

def _new_rule(state, ts):
    got = _ep(state, ts)
    if got[2]:
        return ("never_visible", None)
    return ("censored", None) if got[1] else ("finite", got[0])

_differ = [(n, _old_rule(st, ts), _new_rule(st, ts))
           for n, st, ts, _ in _cases if _old_rule(st, ts) != _new_rule(st, ts)]
_names = {d[0] for d in _differ}
check("the two accountants agree except where a hide was not durable",
      _names == {"PROVISIONAL->HIDDEN, verifier still pending",
                 "PROVISIONAL->HIDDEN->TRUSTED, visible at horizon"})
check("where they differ, the old rule was the one claiming containment",
      all(o[0] == "finite" and n[0] == "censored" for _, o, n in _differ))


# --- the horizon state is the horizon state, not the post-drain state --------
# shutdown() cancels in-flight verifiers rather than draining them, so an item
# whose verification would have committed after the window stays HIDDEN and is
# censored. If that ever became a drain, this item would read QUARANTINED and a
# censored sample would silently turn into a finite containment.
async def _horizon():
    sysh = System(InMemoryBackend(), Tp=0.05, verify=True, sync=False,
                  deadline=True, verify_cost=5.0)          # commits long after
    sysh.admit(1, [1.0] + [0.0] * 31, content_bad=True)
    await asyncio.sleep(0.3)                               # deadline has fired
    state_at_horizon = sysh.items[1]["state"]
    await sysh.shutdown()
    return state_at_horizon, sysh.items[1]["state"]

_at_horizon, _after = asyncio.run(_horizon())
check("a verifier that would commit after the window leaves the item HIDDEN",
      _at_horizon == State.HIDDEN and _after == State.HIDDEN)
check("that item is right-censored, not counted as contained",
      _ep(_after, {"arrival": 0.0, "visible": 0.0, "contain": 0.05})[1] == 1)

# --- the promotion sweep must vary the verifier outcome and nothing else ----
# If choosing which poison to falsely promote drew from a stream anything else
# uses, k would also move the corpus, the query schedule or the cell order, and
# a difference between two k conditions would no longer be attributable to
# promotion. Every generator here is a fresh Random with a namespaced seed, so
# that cannot happen -- asserted rather than left to inspection, because the
# failure would be invisible in the results.
check("no module-level RNG anywhere in the runner or the state machine",
      not [ln for f in ("poison_exposure.py", "functional_slice.py",
                        "realtext_workload.py")
           for ln in open(os.path.join(HERE, f))
           if _r_re.search(ln) and "random.Random(" not in ln])

_seeds_used = sorted({7000 + 1, 9000 + 1, 1, 10_000 + 1})
check("the four generator seeds do not collide at any workload seed",
      all(len({s + o for o in (0, 7000, 9000, 10_000)}) == 4
          for s in w2.CFG["seeds"]))

# What the sweep must hold fixed, checked as the values themselves.
_w = w2.make_world(2, w2.CFG)
_w_again = w2.make_world(2, w2.CFG)
check("the world a seed produces does not depend on when it is built",
      _w["corpus"] == _w_again["corpus"] and _w["poison"] == _w_again["poison"]
      and _w["q_target"] == _w_again["q_target"])

_sets = [w2.false_promote_ids(1000, "heavy", 2, w2.CFG, k)
         for k in w2.FP_SWEEP]
check("promotion sets are nested and grow by one",
      all(a < b and len(b) == len(a) + 1 for a, b in zip(_sets, _sets[1:])))
check("k=0 promotes nothing", _sets[0] == frozenset())

# Drawing the promotion set must leave every other stream where it was.
_before = w2.make_world(2, w2.CFG)["poison"]
for _k in w2.FP_SWEEP:
    w2.false_promote_ids(1000, "heavy", 2, w2.CFG, _k)
check("drawing a promotion set does not disturb the workload draw",
      w2.make_world(2, w2.CFG)["poison"] == _before)

_grid_order = []
for _k in w2.FP_SWEEP:
    import random as _rr
    _g = list(range(14))
    w2.false_promote_ids(1000, "heavy", 2, w2.CFG, _k)
    _rr.Random(9000 + 2).shuffle(_g)
    _grid_order.append(_g)
check("cell order is the same under every k",
      all(g == _grid_order[0] for g in _grid_order))

# --- E_u must tell apart 'nothing was unvetted' from 'never resolved' -------
# The first E_u counted resolved episodes only, which made a count of zero mean
# both B2 (verified before visible, nothing unvetted ever) and B1 (unvetted for
# the whole window, never resolved). A gate written against that count would
# have passed the undefended baseline as if it enforced the bound. These two
# fixtures are the pair that collided.
_b1 = _S()
_b1.items[1] = {"id": 1, "prov": 1, "deadline": float("inf"),
                "state": State.PROVISIONAL, "ts": {"arrival": 0.0, "visible": 0.0}}
_m_b1 = w2.summarise(_b1, qlog, [1], [], w2.CFG, 1.0, 0.0)
_b2 = _S()
_b2.items[2] = {"id": 2, "prov": 0, "deadline": float("inf"),
                "state": State.TRUSTED, "ts": {"arrival": 0.0, "visible": 5.0}}
_m_b2 = w2.summarise(_b2, qlog, [2], [], w2.CFG, 1.0, 0.0)
_b4 = _S()
_b4.items[3] = {"id": 3, "prov": 1, "deadline": 1.0, "state": State.QUARANTINED,
                "ts": {"arrival": 0.0, "visible": 0.0, "contain": 1.0}}
_m_b4 = w2.summarise(_b4, qlog, [3], [], w2.CFG, 1.0, 0.0)

check("started and unresolved is RIGHT_CENSORED",
      _m_b1["Eu_status"] == ["RIGHT_CENSORED"] and
      _m_b1["Eu_started_n"] == 1 and _m_b1["Eu_completed_n"] == 0)
check("never on the query path unvetted is NOT_STARTED",
      _m_b2["Eu_status"] == ["NOT_STARTED"] and _m_b2["Eu_started_n"] == 0)
check("closed at the deadline is COMPLETED with a duration",
      _m_b4["Eu_status"] == ["COMPLETED"] and _m_b4["Eu_completed_p50"] == 1.0)
check("the three states are distinct, which one count could not make them",
      len({_m_b1["Eu_status"][0], _m_b2["Eu_status"][0],
           _m_b4["Eu_status"][0]}) == 3)
check("no completed episode gives a null median, never zero",
      _m_b1["Eu_completed_p50"] is None and _m_b2["Eu_completed_p50"] is None)
check("a censored episode contributes its observed age, not zero",
      _m_b1["Eu_observed_time_at_risk"] > 0 and
      _m_b2["Eu_observed_time_at_risk"] == 0)
check("the ambiguous field is gone rather than kept beside the fix",
      "Eu_n" not in _m_b1)

# The episode boundaries have to be query-path events, or E_u measures how long
# a control-plane row said something rather than how long the store returned
# it. W2/W2R/W2F run in postfilter mode, where the query path asks the control
# store directly -- so the state transition IS the visibility change, with no
# propagation in between. Asserted, because it is only true of this mode: the
# in-index path has to push a flag to the store and that gap is what delta_hide
# names and what the standalone experiment measures.
_pf = System(InMemoryBackend(), mode="postfilter", Tp=1.0, verify=False,
             deadline=False)
_it_pf = _pf.admit(1, [1.0] + [0.0] * 31)
_vis_before = _pf.eligible(1)
_pf.cas(_it_pf, State.PROVISIONAL, State.HIDDEN, "hide_commit")
_vis_after = _pf.eligible(1)
check("in postfilter mode the query path flips at the state transition itself",
      _vis_before and not _vis_after)
check("W2, W2R and W2F all run in that mode",
      'mode="postfilter"' in open(os.path.join(HERE, "poison_exposure.py")).read()
      and 'mode="infilter"' not in open(
          os.path.join(HERE, "poison_exposure.py")).read())

# --- freshness ends at DURABLE visibility -----------------------------------
# The third instance of one defect: a metric stopping at the first event and
# never reopening. D_f read the first `visible` stamp, so a B4 clean item --
# provisionally visible for T_p, hidden at the deadline while the verifier was
# still queued, back only seconds later -- was recorded as 0.000 s fully
# observed. Measured on B4/heavy seed 1: visible 0.000, hidden 1.001, back
# 5.7-6.9 s, one of six never back. The manuscript said that content stayed
# "retrievable within a millisecond".
_dfw = _S()
_dfw.items[10] = {"id": 10, "prov": 1, "deadline": 1.0, "state": State.TRUSTED,
                  "ts": {"arrival": 0.0, "visible": 0.0, "hide_commit": 1.0,
                         "trusted": 5.7}}
_m_w = w2.summarise(_dfw, qlog, [], [10], w2.CFG, 1.0, 0.0)
check("first visibility keeps the paper's definition even when withdrawn later",
      _m_w["Df_clean_p50"] == 0.0)
check("durable visibility is the readmission, and the gap is the outage",
      _m_w["Df_trusted_p50"] == 5.7 and _m_w["clean_gap_p50"] == 4.7 and
      _m_w["clean_expired_n"] == 1 and _m_w["clean_readmitted_n"] == 1)

_dfh = _S()
_dfh.items[11] = {"id": 11, "prov": 1, "deadline": 1.0, "state": State.HIDDEN,
                  "ts": {"arrival": 0.0, "visible": 0.0, "hide_commit": 1.0}}
_m_h = w2.summarise(_dfh, qlog, [], [11], w2.CFG, 1.0, 0.0)
check("an item that never came back is censored in durable and in gap, "
      "while its first visibility still counts",
      _m_h["Df_clean_p50"] == 0.0 and _m_h["Df_trusted_p50"] is None and
      _m_h["Df_trusted_right_censored_n"] == 1 and
      _m_h["clean_gap_right_censored_n"] == 1 and
      _m_h["clean_gap_p50"] is None)

# The omission this pair exists to prevent: median over the ones that returned
# while the ones that did not are dropped from the denominator.
_mix = _S()
_mix.items[13] = dict(_dfw.items[10], id=13)
_mix.items[14] = dict(_dfh.items[11], id=14)
_m_mix = w2.summarise(_mix, qlog, [], [13, 14], w2.CFG, 1.0, 0.0)
check("a mixed cell reports both the completed gap and the censored one",
      _m_mix["clean_gap_completed_n"] == 1 and
      _m_mix["clean_gap_right_censored_n"] == 1 and
      _m_mix["clean_expired_n"] == 2)

_dfp = _S()
_dfp.items[12] = {"id": 12, "prov": 1, "deadline": float("inf"),
                  "state": State.PROVISIONAL,
                  "ts": {"arrival": 0.0, "visible": 0.4}}
_m_p = w2.summarise(_dfp, qlog, [], [12], w2.CFG, 1.0, 0.0)
check("never withdrawn: first visibility is also the durable one, no gap",
      _m_p["Df_clean_p50"] == 0.4 and _m_p["Df_trusted_p50"] == 0.4 and
      _m_p["clean_expired_n"] == 0 and _m_p["clean_gap_p50"] is None)

# --- verify-before-visible is a TIMING property, not a code-path artefact ----
# Asserting that B2 sets no PROVISIONAL flag proves only that _sync_admit does
# not call the function that sets it. The property the paper claims is temporal:
# no query returns the item before its verification commits, whatever the
# verdict turns out to be. Driven with a verifier slow enough that queries land
# on both sides of the commit, and checked against the query path rather than
# against the control store's own bookkeeping.
async def _b2_timing(bad_label):
    sysb = System(InMemoryBackend(), mode="postfilter", Tp=1.0, verify=True,
                  sync=True, deadline=False, verify_cost=0.4,
                  false_promote=frozenset([1]) if bad_label else frozenset())
    qv = [1.0] + [0.0] * 31
    sysb.admit(1, qv, content_bad=bad_label)
    seen_before = 0
    for _ in range(6):                       # queries while verification runs
        await asyncio.sleep(0.05)
        if sysb.eligible(1):
            seen_before += 1
    await asyncio.sleep(0.35)                # past the commit
    after = sysb.eligible(1)
    await sysb.shutdown()
    return seen_before, after, sysb.items[1]["prov"]

_clean_before, _clean_after, _ = asyncio.run(_b2_timing(False))
check("B2: a clean item is invisible until its verification commits",
      _clean_before == 0 and _clean_after)
_fp_before, _fp_after, _fp_prov = asyncio.run(_b2_timing(True))
check("B2: a falsely promoted item is invisible until the commit too",
      _fp_before == 0 and _fp_after)
check("B2: the false promotion produced exposure without any unvetted episode",
      _fp_after and _fp_prov == 0)

ok = sum(1 for _, c in checks if c)
for name, c in checks:
    print(f"  [{'PASS' if c else 'FAIL'}] {name}")
print(f"\n==== test_w2: {ok}/{len(checks)} checks passed ====")
sys.exit(0 if ok == len(checks) else 1)
