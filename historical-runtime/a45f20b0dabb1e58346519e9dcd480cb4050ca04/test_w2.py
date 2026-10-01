"""Guards for the W2 poison-exposure experiment that do not need a run.

    ./.venv312/bin/python test_w2.py

These are the checks that would otherwise only fail after a twelve-minute run,
or worse, not fail at all and quietly change what the numbers mean.
"""
import ast
import os
import sys

import poison_exposure as w2
from functional_slice import State, System, VISIBLE

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
s.items[99] = {"id": 99, "prov": 1, "deadline": float("inf"),
               "ts": {"arrival": 0.0, "visible": 0.0}}          # visible, never contained
m = w2.summarise(s, qlog, [99], [], w2.CFG, 1.0, 0.0)
check("never contained -> nothing lands after containment",
      m["elig_recall_after_containment_p50"] is None and
      m["elig_recall_during_exposure_p50"] is not None)
s2 = _S()
s2.items[98] = {"id": 98, "prov": 0, "deadline": float("inf"), "ts": {}}   # never visible
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

ok = sum(1 for _, c in checks if c)
for name, c in checks:
    print(f"  [{'PASS' if c else 'FAIL'}] {name}")
print(f"\n==== test_w2: {ok}/{len(checks)} checks passed ====")
sys.exit(0 if ok == len(checks) else 1)
