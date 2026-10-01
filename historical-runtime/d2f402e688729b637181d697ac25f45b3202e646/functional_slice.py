"""
Real-time (asyncio) functional slice of the fail-closed provisional-visibility
protocol over a VectorBackend. Unlike slice.py (logical-clock simulation), this
runs in wall-clock time against a real backend, so it produces real delta_query.

Gate 1 (this file): the full lifecycle runs and emits the §6.1 timestamps:
  INGESTED -> PROVISIONAL (visible) -> deadline -> HIDDEN (invisible)
  -> late positive verification -> TRUSTED (visible) -> lineage alert -> REVOKED

Run:  python3 functional_slice.py --backend inmemory --mode postfilter
      python3 functional_slice.py --backend milvus   --mode infilter   (needs Docker)

The deadline scheduler uses loop.call_later -> it fires independently of the
(awaiting) verifier coroutines, which is invariant I6.
"""

import argparse
import asyncio
import time
from enum import Enum

from backend import InMemoryBackend


class State(str, Enum):
    PROVISIONAL = "PROVISIONAL"
    TRUSTED = "TRUSTED"
    QUARANTINED = "QUARANTINED"
    HIDDEN = "HIDDEN"
    REVOKED = "REVOKED"


VISIBLE = {State.PROVISIONAL, State.TRUSTED}


def now():
    return time.monotonic()


class System:
    def __init__(self, backend, mode="postfilter", Tp=1.0, verify_cost=0.3, over_fetch=3,
                 decouple=True, verifier_concurrency=0):
        self.b = backend
        self.mode = mode
        self.Tp = Tp
        self.vc = verify_cost
        self.of = over_fetch
        # decouple=True (default, Gate 1): the deadline scheduler is independent (I6).
        # decouple=False (Gate 3 coupled baseline): hiding queues behind the verifier
        # worker pool, so backlog delays enforcement.
        self.decouple = decouple
        self.sem = asyncio.Semaphore(verifier_concurrency) if verifier_concurrency else None
        self.items = {}                 # control store (authoritative state)
        self.transitions = []

    def cas(self, it, expected, new, stamp=None):
        if it["state"] != expected:
            return False
        old = it["state"]
        it["state"] = new
        if new == State.PROVISIONAL:
            it["prov"] += 1
        if stamp:
            it["ts"][stamp] = now()
        self.transitions.append((now(), it["id"], old, new))
        return True

    def eligible(self, iid):
        return self.items[iid]["state"] in VISIBLE

    # ---- admission -------------------------------------------------------
    def admit(self, iid, vec, source_ok=True, content_bad=False):
        it = {"id": iid, "vec": list(vec), "state": None, "deadline": None,
              "prov": 0, "pending": False, "content_bad": content_bad,
              "ts": {"arrival": now(), "admit": now()}}
        self.items[iid] = it
        if not source_ok:                       # cheap hard gate at admission
            it["state"] = State.QUARANTINED
            self.transitions.append((now(), iid, None, State.QUARANTINED))
            return it
        it["state"] = State.PROVISIONAL
        it["prov"] = 1
        it["deadline"] = now() + self.Tp
        self.b.insert(iid, vec, visible=True)
        it["ts"].update(insert_ack=now(), visible=now(), deadline=it["deadline"])
        it["was_visible"] = True
        self.transitions.append((now(), iid, None, State.PROVISIONAL))
        asyncio.get_running_loop().call_later(self.Tp, self._on_deadline, it)
        asyncio.create_task(self._verify(it))
        return it

    # ---- async verifier (commit-time rule) -------------------------------
    async def _verify(self, it):
        it["ts"]["verify_enqueue"] = now()
        if self.sem is not None:
            async with self.sem:                 # models a bounded verifier pool
                it["ts"]["verify_start"] = now()
                await asyncio.sleep(self.vc)
        else:
            it["ts"]["verify_start"] = now()
            await asyncio.sleep(self.vc)
        it["ts"]["verify_end"] = now()
        passes = not it["content_bad"]
        it["ts"]["verify_commit"] = now()
        if passes:
            if now() <= it["deadline"] and self.cas(it, State.PROVISIONAL, State.TRUSTED, "trusted"):
                return
            if self.cas(it, State.HIDDEN, State.TRUSTED, "trusted"):
                if self.mode == "infilter":
                    self.b.set_visible(it["id"], True)
                return
            if it["state"] == State.PROVISIONAL:     # scheduler not fired yet
                it["pending"] = True
        else:
            if not self.cas(it, State.PROVISIONAL, State.QUARANTINED, "verify_commit"):
                self.cas(it, State.HIDDEN, State.QUARANTINED, "verify_commit")
            if self.mode == "infilter":
                self.b.set_visible(it["id"], False)

    # ---- deadline scheduler (I6) -----------------------------------------
    def _on_deadline(self, it):
        if self.decouple or self.sem is None:
            self._do_hide(it)                    # independent, immediate (I6)
        else:
            asyncio.create_task(self._coupled_hide(it))   # queues behind verifier pool

    async def _coupled_hide(self, it):
        async with self.sem:                     # backlog delays enforcement -> E_u > T_p
            self._do_hide(it)

    def _do_hide(self, it):
        if self.cas(it, State.PROVISIONAL, State.HIDDEN, "hide_commit"):
            if self.mode == "infilter":
                self.b.set_visible(it["id"], False)
            if it["pending"]:
                self.cas(it, State.HIDDEN, State.TRUSTED, "trusted")
                if self.mode == "infilter":
                    self.b.set_visible(it["id"], True)

    # ---- lineage revocation ----------------------------------------------
    def revoke(self, it):
        it["ts"]["alert"] = now()
        if self.cas(it, State.TRUSTED, State.REVOKED, "contain"):
            if self.mode == "infilter":
                self.b.set_visible(it["id"], False)

    # ---- query service (two eligibility modes) ---------------------------
    def query(self, qvec, k=5):
        if self.mode == "infilter":
            return self.b.search(qvec, k, use_index_filter=True)
        raw = self.b.search(qvec, self.of * k, use_index_filter=False)  # over-fetch
        return [i for i in raw if self.eligible(i)][:k]

    def is_visible(self, it, qvec):
        return it["id"] in self.query(qvec, k=5)


async def gate1(backend, mode):
    print(f"Gate 1 functional slice  (backend={backend.__class__.__name__}, mode={mode})\n")
    sys = System(backend, mode=mode, Tp=1.0, verify_cost=5.0)  # verify AFTER deadline
    qv = [1.0] + [0.0] * 7
    it = sys.admit(0, qv, source_ok=True, content_bad=False)

    # dense probe to estimate delta_query (first-hidden after deadline)
    first_hidden = [None]

    async def probe():
        while True:
            vis = sys.is_visible(it, qv)
            if (not vis and first_hidden[0] is None and it["state"] == State.HIDDEN):
                first_hidden[0] = now()
                it["ts"]["hide_effective"] = now()
            await asyncio.sleep(0.02)

    ptask = asyncio.create_task(probe())

    await asyncio.sleep(0.3);  v_prov = sys.is_visible(it, qv)     # PROVISIONAL visible
    await asyncio.sleep(1.0);  v_hidden = sys.is_visible(it, qv)   # past deadline -> HIDDEN
    await asyncio.sleep(4.3);  v_trusted = sys.is_visible(it, qv)  # late verify -> TRUSTED
    sys.revoke(it);            v_revoked = sys.is_visible(it, qv)  # REVOKED
    ptask.cancel()

    ts = it["ts"]
    def rel(k):
        return f"{ts[k]-ts['arrival']:.3f}s" if k in ts else "--"
    print("  timeline (relative to arrival):")
    for k in ["visible", "deadline", "hide_commit", "hide_effective",
              "verify_commit", "trusted", "alert", "contain"]:
        print(f"    {k:16} {rel(k)}")
    dq = (ts["hide_effective"] - ts["deadline"]) if "hide_effective" in ts else None
    dc = (ts["hide_commit"] - ts["deadline"]) if "hide_commit" in ts else None
    print(f"\n  delta_control = {dc:.4f}s   delta_query(upper) = "
          f"{dq:.4f}s" if dq is not None else "  delta_query n/a")

    checks = [
        ("PROVISIONAL visible", v_prov is True),
        ("HIDDEN after deadline invisible", v_hidden is False),
        ("TRUSTED after late verify visible", v_trusted is True),
        ("REVOKED invisible", v_revoked is False),
        ("entered PROVISIONAL exactly once (I7)", it["prov"] == 1),
        ("went PROVISIONAL->HIDDEN->TRUSTED (commit-time)",
         any(o == State.PROVISIONAL and n == State.HIDDEN for _, i, o, n in sys.transitions)
         and any(o == State.HIDDEN and n == State.TRUSTED for _, i, o, n in sys.transitions)),
        ("never PROVISIONAL->TRUSTED directly",
         not any(o == State.PROVISIONAL and n == State.TRUSTED for _, i, o, n in sys.transitions)),
        ("final state REVOKED", it["state"] == State.REVOKED),
    ]
    print()
    ok = 0
    for name, c in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
        ok += bool(c)
    print(f"\n==== Gate 1: {ok}/{len(checks)} checks passed ====")
    return ok == len(checks)


def make_backend(name, uri):
    if name == "inmemory":
        return InMemoryBackend()
    if name == "milvus":
        from milvus_backend import MilvusBackend
        return MilvusBackend(dim=8, uri=uri)
    raise SystemExit(f"unknown backend {name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="inmemory", choices=["inmemory", "milvus"])
    ap.add_argument("--mode", default="postfilter", choices=["postfilter", "infilter"])
    ap.add_argument("--uri", default="http://localhost:19530",
                    help="Milvus server URI, or a local .db path for Milvus Lite (no Docker)")
    args = ap.parse_args()
    ok = asyncio.run(gate1(make_backend(args.backend, args.uri), args.mode))
    raise SystemExit(0 if ok else 1)
