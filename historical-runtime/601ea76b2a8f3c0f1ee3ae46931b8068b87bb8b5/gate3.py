"""
Gate 3 -- overload validation of I6, on a REAL-TIME backend.

Floods N items at once against a bounded verifier pool (backlog), and compares:
  - decoupled: the deadline scheduler is independent (loop.call_later) -> hiding
    fires on time regardless of verifier backlog (invariant I6)
  - coupled:   hiding queues behind the same verifier pool -> backlog delays
    enforcement, so unvetted content stays visible for the whole backlog

Metric: per-item E_u = (first exit from PROVISIONAL) - t_visible, read from the
audit log. Expect decoupled E_u ~ T_p (bounded), coupled E_u >> T_p.

This is the real-time analogue of slice.py's T7 (which showed the same on the
logical-clock sim). Runs on in-memory or Milvus Lite; the full standalone run
adds real gRPC/consistency cost. Run:
  python3 gate3.py --backend inmemory
  ./.venv312/bin/python gate3.py --backend milvus --uri /tmp/g3.db
"""

import argparse
import asyncio

from functional_slice import System, State, make_backend


def eu_list(sys, items):
    out = []
    for it in items:
        exits = [t for (t, i, o, n) in sys.transitions
                 if i == it["id"] and o == State.PROVISIONAL]
        if exits and "visible" in it["ts"]:
            out.append(min(exits) - it["ts"]["visible"])
    return out


def pctl(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))]


async def run(backend, decouple, N, Tp, vc, conc):
    sys = System(backend, mode="postfilter", Tp=Tp, verify_cost=vc,
                 decouple=decouple, verifier_concurrency=conc)
    vec = [1.0] + [0.0] * 7
    items = [sys.admit(k, vec, source_ok=True, content_bad=False) for k in range(N)]
    await asyncio.sleep(N * vc / conc + Tp + 4)      # let the backlog fully drain
    return eu_list(sys, items)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="inmemory", choices=["inmemory", "milvus"])
    ap.add_argument("--uri", default="http://localhost:19530")
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--Tp", type=float, default=1.0)
    ap.add_argument("--vc", type=float, default=0.4)
    ap.add_argument("--conc", type=int, default=2)
    a = ap.parse_args()

    print(f"Gate 3 -- overload (I6): backend={a.backend}, N={a.N} flooded, "
          f"T_p={a.Tp}s, verify={a.vc}s x {a.conc} workers")
    print(f"  verifier backlog ~ N*vc/conc = {a.N*a.vc/a.conc:.1f}s\n")
    for decouple in (True, False):
        backend = make_backend(a.backend, a.uri)      # fresh backend/collection
        eus = await run(backend, decouple, a.N, a.Tp, a.vc, a.conc)
        label = "decoupled (independent scheduler, I6)" if decouple \
            else "coupled (hide behind verifier pool) "
        print(f"  {label}:  E_u  mean={sum(eus)/len(eus):5.2f}s  "
              f"p95={pctl(eus,0.95):5.2f}s  max={max(eus):5.2f}s   (T_p={a.Tp}s)")
    print("\n  I6: decoupled keeps E_u ~ T_p; coupled lets it grow with the backlog.")


if __name__ == "__main__":
    asyncio.run(main())
