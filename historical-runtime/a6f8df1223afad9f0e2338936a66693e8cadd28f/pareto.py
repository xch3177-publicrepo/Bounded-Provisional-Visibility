"""
Protocol-Level Freshness-Exposure Characterization (Fig A).

SIMULATION / correctness harness -- NOT a real-system Pareto frontier; carries no
Milvus/gRPC/consistency cost. All items are clean, so:
  - E_u = exposure BUDGET (how long an unvetted item is query-visible; the window
    poison *would* exploit). It is NOT measured poisoning exposure E_p, which is a
    system-level (Gate 3) result.
  - A_f = clean-content availability = fraction of item-targeting probes over the
    fresh-content interval [t_visible, t_visible + W] that actually retrieve it.

Sweeps T_p across three verifier regimes (fast / moderate / overloaded) plus a
burst workload. Run:  python3 pareto.py
"""

from collections import defaultdict
from slice import Sim, System, Item, State

VISIBLE = (State.PROVISIONAL, State.TRUSTED)
W = 8.0            # fresh-content interval length (spans verify latency)
PROBE_DT = 0.1     # probe cadence (also the A_f / delta_query resolution)


def first_exit_prov(sys, iid):
    ts = [t for (t, i, o, n) in sys.cs.transitions if i == iid and o == State.PROVISIONAL]
    return min(ts) if ts else None


def pctl(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))]


def run(Tp, verify_cost, arrival_interval, N=100, burst=False):
    sim = Sim()
    sys = System(sim, Tp=Tp, verify_cost=verify_cost)
    items = [Item(k) for k in range(N)]
    for k, it in enumerate(items):
        arr = 0.0 if burst else k * arrival_interval
        sim.at(arr, (lambda it=it: sys.admit(it)))
        sim.at(arr, (lambda it=it, a=arr: sys.schedule_probe(it, PROBE_DT, a + W)))
    sim.run()

    probes = defaultdict(list)
    for (t, i, s) in sys.query_log:
        probes[i].append(s)

    eus, afs = [], []
    expired = 0
    for it in items:
        tep = first_exit_prov(sys, it.id)
        if tep is not None and "visible" in it.ts:
            eus.append(tep - it.ts["visible"])
        ps = probes.get(it.id, [])
        if ps:
            afs.append(sum(1 for s in ps if s in VISIBLE) / len(ps))
        if "hide_commit" in it.ts:
            expired += 1
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {"meanE_u": mean(eus), "p95E_u": pctl(eus, 0.95),
            "A_f": mean(afs), "expRate": expired / N}


def sweep(label, verify_cost, arrival_interval, burst=False):
    print(f"\n{label}")
    print(f"  {'T_p':>5} {'E_u(budget)':>12} {'p95E_u':>8} {'A_f(avail)':>11} {'expRate':>8}")
    for Tp in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
        r = run(Tp, verify_cost, arrival_interval, burst=burst)
        print(f"  {Tp:5.2f} {r['meanE_u']:12.3f} {r['p95E_u']:8.3f} "
              f"{r['A_f']:11.3f} {r['expRate']:8.2f}")


if __name__ == "__main__":
    print("Protocol-Level Freshness-Exposure Characterization (Fig A)")
    print("simulation / correctness harness -- NOT a system Pareto; all-clean items")
    print(f"fresh interval W={W}s, probe cadence={PROBE_DT}s")
    sweep("FAST verifier (verify_cost=0.2s, stable)",     verify_cost=0.2, arrival_interval=1.0)
    sweep("MODERATE verifier (verify_cost=1.0s, stable)", verify_cost=1.0, arrival_interval=1.5)
    sweep("OVERLOADED verifier (vc=1.0s, backlog)",       verify_cost=1.0, arrival_interval=0.5)
    sweep("BURST workload (vc=1.0s, all arrive t=0)",     verify_cost=1.0, arrival_interval=0.0, burst=True)
    print("\nPareto reading (per regime): each T_p is one point; raising T_p raises")
    print("the exposure budget E_u and clean availability A_f together. The frontier")
    print("shifts unfavorably fast->moderate->overloaded as verify latency grows.")
