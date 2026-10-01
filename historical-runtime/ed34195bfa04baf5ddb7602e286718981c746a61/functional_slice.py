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
    # `verify`/`sync`/`deadline` select the admission policy, so B1-B4 are the
    # SAME control plane under different switches rather than four
    # implementations. Defaults are B4, which is what gate1 has always tested;
    # nothing above this line changes behaviour for existing callers.
    #   B1 verify=False sync=False deadline=False   immediate, no verification
    #   B2 verify=True  sync=True  deadline=False   verify before visible
    #   B3 verify=True  sync=False deadline=False   async, no deadline
    #   B4 verify=True  sync=False deadline=True    fail-closed provisional
    def __init__(self, backend, mode="postfilter", Tp=1.0, verify_cost=0.3, over_fetch=3,
                 decouple=True, verifier_concurrency=0,
                 verify=True, sync=False, deadline=True,
                 false_promote=frozenset()):
        self.b = backend
        self.mode = mode
        self.Tp = Tp
        self.vc = verify_cost
        self.of = over_fetch
        self.do_verify = verify
        self.sync = sync
        self.use_deadline = deadline
        # Ids the oracle verifier passes despite a bad label (W2F). Fixed before
        # the cell runs, never derived from what the cell observes. Empty in
        # every other experiment, so the verifier stays exactly as correct as it
        # has always been unless something asked for it not to be.
        self.false_promote = frozenset(false_promote)
        # decouple=True (default, Gate 1): the deadline scheduler is independent (I6).
        # decouple=False (Gate 3 coupled baseline): hiding queues behind the verifier
        # worker pool, so backlog delays enforcement.
        self.decouple = decouple
        self.sem = asyncio.Semaphore(verifier_concurrency) if verifier_concurrency else None
        self.items = {}                 # control store (authoritative state)
        self.transitions = []
        # Verifier coroutines and deadline timers outlive the workload that
        # started them. Abandoned rather than cancelled, a pending B2 admission
        # resumes during whatever runs next and inserts its item there -- so a
        # measurement that thought it had cleaned up is competing with the
        # previous one's rows. Tracked so a caller can shut the plane down.
        self.tasks = set()
        self.timers = []

    def _spawn(self, coro):
        t = asyncio.create_task(coro)
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)
        return t

    async def shutdown(self):
        """Cancel everything still in flight and wait for it to land. Must be
        awaited before a caller reuses or tears down the backend.

        Returns the exceptions the background tasks raised, cancellations
        excluded. They are returned rather than swallowed: `return_exceptions`
        would hide a verifier that died mid-run behind a clean-looking
        shutdown, and a cell whose verifier crashed is not a cell whose
        baseline behaved as its definition says."""
        for h in self.timers:
            h.cancel()
        self.timers.clear()
        done = [t for t in self.tasks if t.done()]
        pending = [t for t in self.tasks if not t.done()]
        for t in pending:
            t.cancel()
        results = []
        if pending:
            results = await asyncio.gather(*pending, return_exceptions=True)
        errs = [r for r in results
                if isinstance(r, BaseException)
                and not isinstance(r, asyncio.CancelledError)]
        for t in done:
            if t.exception() is not None:
                errs.append(t.exception())
        self.tasks.clear()
        return errs

    def cas(self, it, expected, new, stamp=None):
        if it["state"] != expected:
            return False
        old = it["state"]
        it["state"] = new
        if new == State.PROVISIONAL:
            it["prov"] += 1
        if stamp:
            it["ts"][stamp] = now()
        # t_contain (§III): the first transition that takes the item OUT of the
        # query path, whichever mechanism did it -- deadline (HIDDEN), verifier
        # (QUARANTINED) or lineage (REVOKED). Stamped here rather than at each
        # call site so no path can reach containment without recording when.
        if old in VISIBLE and new not in VISIBLE:
            it["ts"].setdefault("contain", now())
        self.transitions.append((now(), it["id"], old, new))
        return True

    def eligible(self, iid):
        it = self.items.get(iid)        # ids the control store never saw are
        return it is not None and it["state"] in VISIBLE   # not eligible

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
        if self.sync:                           # B2: nothing is visible unvetted
            self._spawn(self._sync_admit(it))
            return it
        # Without a deadline (B1/B3) there is no instant a promotion can miss,
        # so the commit-time rule in _verify compares against +inf rather than
        # being special-cased there.
        it["deadline"] = now() + self.Tp if self.use_deadline else float("inf")
        # Insert BEFORE the control store calls it eligible. The other order
        # leaves a window in which the authoritative state says "visible" and
        # the store cannot return it yet -- harmless for a lifecycle check, but
        # it shows up as spurious recall loss the moment anything scores the
        # query path against exact ground truth. Fail-closed means eligibility
        # never leads existence.
        self.b.insert(iid, vec, visible=True)
        it["state"] = State.PROVISIONAL
        it["prov"] = 1
        it["ts"].update(insert_ack=now(), visible=now(), deadline=it["deadline"])
        it["was_visible"] = True
        self.transitions.append((now(), iid, None, State.PROVISIONAL))
        if self.use_deadline:
            self.timers.append(
                asyncio.get_running_loop().call_later(self.Tp, self._on_deadline, it))
        if self.do_verify:
            self._spawn(self._verify(it))
        return it

    # ---- B2: verify before visible ---------------------------------------
    async def _sync_admit(self, it):
        """The verification runs on the admission path, so freshness delay
        carries the whole verifier wait -- including the queueing wait, which is
        why B2's D_f grows under backlog while its exposure stays zero."""
        it["ts"]["verify_enqueue"] = now()
        if self.sem is not None:
            async with self.sem:
                it["ts"]["verify_start"] = now()
                await asyncio.sleep(self.vc)
        else:
            it["ts"]["verify_start"] = now()
            await asyncio.sleep(self.vc)
        it["ts"]["verify_end"] = it["ts"]["verify_commit"] = now()
        # A falsely promoted item passes here too. B2's zero exposure is a
        # property of a verifier that is right, not of the admission discipline
        # on its own, and hard-coding the label here would have hidden that.
        if it["content_bad"] and it["id"] not in self.false_promote:
            it["state"] = State.QUARANTINED     # never inserted, never visible
            self.transitions.append((now(), it["id"], None, State.QUARANTINED))
            return
        self.b.insert(it["id"], it["vec"], visible=True)   # exists before eligible
        it["state"] = State.TRUSTED
        it["ts"].update(insert_ack=now(), visible=now())
        it["was_visible"] = True
        self.transitions.append((now(), it["id"], None, State.TRUSTED))

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
        passes = not it["content_bad"] or it["id"] in self.false_promote
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
            self._spawn(self._coupled_hide(it))          # queues behind verifier pool

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
    def query(self, qvec, k=5, stats=None):
        if self.mode == "infilter":
            return self.b.search(qvec, k, use_index_filter=True)
        raw = self.b.search(qvec, self.of * k, use_index_filter=False)  # over-fetch
        if stats is not None:
            # Ids the control store has never heard of. Post-filtering means
            # such a row can never be RETURNED -- eligibility is decided here,
            # not by the store -- but it can still occupy one of the over-fetch
            # slots and push a legitimate candidate out. That is the entire
            # residual harm a stale row from an earlier workload can do, so it
            # is counted rather than argued about. Forcing a namespace filter
            # into the search would prevent it, but it would also turn every
            # query into an in-index-filtered query, and post-filter versus
            # in-index is one of the axes this paper measures.
            stats["foreign_candidates"] += sum(1 for i in raw if i not in self.items)
            stats["queries"] += 1
        out = [i for i in raw if self.eligible(i)][:k]
        if stats is not None and len(out) < k:
            # Post-filter under-fill (Sec. VI-C). Expected to be zero here --
            # the eligible population is far larger than k -- so a non-zero
            # count means the over-fetch pool was being consumed by something,
            # which is the same failure a foreign candidate causes.
            stats["underfill"] += 1
        return out

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
