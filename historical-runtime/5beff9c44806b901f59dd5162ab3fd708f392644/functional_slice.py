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
import inspect
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
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


@dataclass(frozen=True)
class VerifierDecision:
    """Result returned by an injected verifier.

    `service_time_s` is the verifier service time to replay while holding a
    worker slot. A real-detector-output/service-time replay harness can therefore
    return a frozen decision and the detector time measured on its one scoring
    pass; every baseline then sees the same decision and service vector while
    queueing remains endogenous to that baseline. This is replay, not an online
    detector integration. `metadata` is deliberately opaque to the state
    machine.
    """

    passes: bool
    service_time_s: float | None = None
    metadata: Mapping | None = None


@dataclass(frozen=True)
class VerifierRequest:
    """Only data an injected decision provider receives.

    `item_key` is an opaque lookup key for a frozen decision bank.
    `verifier_input` is the detector-safe payload. A provider doing the one real
    score must call detector.score(verifier_input), never detector.score(request)
    and never pass item_key through to the detector. Neither field contains the
    runtime item dictionary or evaluator truth.
    """

    item_key: object
    verifier_input: object


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
                 false_promote=frozenset(), verifier=None):
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
        # Optional decision provider for real-detector-output/service-time
        # replay. It receives ONLY a VerifierRequest containing the opaque key
        # and detector-safe input supplied at admission -- never this item
        # dictionary, which also contains evaluator truth for the legacy oracle.
        # None keeps the exact oracle/false-promote behaviour used by W2/W2R/W2F.
        self.verifier = verifier
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
        # Explicit queue instrumentation. asyncio.Semaphore intentionally does
        # not expose a public queue-depth API, and reading its private waiters
        # would make a measurement depend on an implementation detail.
        self._verifier_waiting = 0
        self._verifier_active = 0
        self.verifier_records = []

    def _spawn(self, coro):
        t = asyncio.create_task(coro)
        self.tasks.add(t)
        # Keep completed tasks until shutdown() has retrieved their result.
        # Discarding them in a done callback loses a verifier exception that
        # lands before the observation window closes: shutdown() then sees an
        # empty set, reports no background error, and asyncio emits only
        # "Task exception was never retrieved".  A System is scoped to one
        # measurement cell, so retaining this bounded set also gives shutdown
        # one authoritative error sink without accumulating across cells.
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
            try:
                exc = t.exception()
            except asyncio.CancelledError:
                continue
            if exc is not None:
                errs.append(exc)
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
    def admit(self, iid, vec, source_ok=True, content_bad=False,
              verifier_input=None, verifier_key=None):
        it = {"id": iid, "vec": list(vec), "state": None, "deadline": None,
              "prov": 0, "pending": False, "content_bad": content_bad,
              # `verifier_key` is an opaque, stable key used only to pair a
              # score-once decision across baselines. It defaults to the runtime
              # id for legacy callers and is never interpreted here.
              "verifier_input": verifier_input,
              "verifier_key": iid if verifier_key is None else verifier_key,
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
        passes = await self._obtain_verdict(it)
        # A falsely promoted item passes here too. B2's zero exposure is a
        # property of a verifier that is right, not of the admission discipline
        # on its own, and hard-coding the label here would have hidden that.
        if not passes:
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
        passes = await self._obtain_verdict(it)
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

    async def _call_custom_verifier(self, request):
        """Call a custom provider without ever handing it the runtime item.

        Synchronous detector code runs in a worker thread so it cannot stall the
        event loop that owns the independent deadline scheduler. Async callables
        are awaited directly. The caller times this operation separately from
        queueing and from any replayed service delay.
        """
        call = self.verifier
        is_async = inspect.iscoroutinefunction(call) or inspect.iscoroutinefunction(
            getattr(call, "__call__", None))
        if is_async:
            return await call(request)
        return await asyncio.to_thread(call, request)

    @staticmethod
    def _normalise_decision(raw, provider_elapsed):
        """Normalise the deliberately small custom-verifier result contract."""
        if isinstance(raw, VerifierDecision):
            passes = raw.passes
            service = raw.service_time_s
            metadata = dict(raw.metadata or {})
        elif isinstance(raw, bool):
            passes, service, metadata = raw, None, {}
        elif isinstance(raw, Mapping):
            if "passes" not in raw:
                raise ValueError("custom verifier result requires a 'passes' field")
            passes = raw["passes"]
            service = raw.get("service_time_s")
            nested = raw.get("metadata", {})
            if nested is not None and not isinstance(nested, Mapping):
                raise TypeError("custom verifier metadata must be a mapping or None")
            metadata = dict(nested or {})
            metadata.update({k: v for k, v in raw.items()
                             if k not in {"passes", "service_time_s", "metadata"}})
        else:
            raise TypeError("custom verifier must return bool, VerifierDecision, "
                            "or a mapping")
        if not isinstance(passes, bool):
            raise TypeError("custom verifier 'passes' must be bool")
        if service is None:
            service = provider_elapsed
        if (not isinstance(service, (int, float)) or isinstance(service, bool)
                or not math.isfinite(service) or service < 0):
            raise ValueError("custom verifier service_time_s must be finite and >= 0")
        return VerifierDecision(passes, float(service), metadata)

    async def _serve_verifier(self, it, record):
        """Return one verdict while occupying a verifier worker slot."""
        provider_started = now()
        if self.verifier is None:
            # Legacy oracle. The decision itself is free; `vc` is its configured
            # service time, exactly as the old duplicated sleeps implemented it.
            raw = VerifierDecision(
                passes=(not it["content_bad"] or it["id"] in self.false_promote),
                service_time_s=self.vc,
                metadata={"provider": "oracle"})
        else:
            # This is the only call site for a custom verifier. The opaque input
            # request is the sole argument; evaluator truth stays in this module.
            # A live score provider strips the lookup key and passes only
            # request.verifier_input into detector.score().
            request = VerifierRequest(it["verifier_key"], it["verifier_input"])
            raw = await self._call_custom_verifier(request)
        provider_elapsed = now() - provider_started
        decision = self._normalise_decision(raw, provider_elapsed)
        # Record a frozen outcome as soon as the provider returns. If the
        # observation window later cancels a replay while its service time is
        # still elapsing, the result is known but explicitly not committed.
        record.update(
            decision_available_s=now(),
            provider_elapsed_s=provider_elapsed,
            service_time_s=decision.service_time_s,
            passes=decision.passes,
            metadata=dict(decision.metadata or {}),
        )
        # A replay provider normally returns immediately with the frozen detector
        # service time. A live async provider may already have consumed all of it;
        # only wait for the remainder, never double-charge it.
        remaining = decision.service_time_s - provider_elapsed
        if remaining > 0:
            await asyncio.sleep(remaining)
        return decision, provider_elapsed

    async def _obtain_verdict(self, it):
        """Shared queue/service/commit path for B2, B3 and B4."""
        enqueued = now()
        it["ts"]["verify_enqueue"] = enqueued
        record = {
            "item_id": it["id"],
            "item_key": it["verifier_key"],
            "status": "QUEUED",
            "queue_enter_s": enqueued,
            "queue_depth_at_enqueue": self._verifier_waiting,
            "active_at_enqueue": self._verifier_active,
            "queue_start_s": None,
            "decision_available_s": None,
            "decision_commit_s": None,
            "queue_depth_at_start": None,
            "queue_depth_at_commit": None,
            "active_at_start": None,
            "queue_wait_s": None,
            "provider_elapsed_s": None,
            "service_time_s": None,
            "service_observed_s": None,
            "integrated_latency_s": None,
            "passes": None,
            "metadata": {},
        }
        self.verifier_records.append(record)
        it["verifier_record"] = record

        self._verifier_waiting += 1
        acquired = False
        active = False
        try:
            if self.sem is not None:
                await self.sem.acquire()
                acquired = True
            self._verifier_waiting -= 1
            self._verifier_active += 1
            active = True
            started = now()
            it["ts"]["verify_start"] = started
            record.update(
                status="RUNNING",
                queue_start_s=started,
                queue_depth_at_start=self._verifier_waiting,
                active_at_start=self._verifier_active,
                queue_wait_s=started - enqueued,
            )
            decision, provider_elapsed = await self._serve_verifier(it, record)
            ended = now()
            it["ts"]["verify_end"] = ended
            # Commit remains a distinct timestamp even though no deliberate
            # delay exists between service completion and the state transition.
            committed = now()
            it["ts"]["verify_commit"] = committed
            record.update(
                status="COMMITTED",
                decision_commit_s=committed,
                queue_depth_at_commit=self._verifier_waiting,
                provider_elapsed_s=provider_elapsed,
                service_time_s=decision.service_time_s,
                service_observed_s=ended - started,
                integrated_latency_s=committed - enqueued,
                passes=decision.passes,
                metadata=dict(decision.metadata or {}),
            )
            return decision.passes
        except asyncio.CancelledError:
            record["status"] = "CANCELLED"
            record["queue_depth_at_commit"] = self._verifier_waiting
            if record["queue_start_s"] is not None:
                record["service_observed_s"] = now() - record["queue_start_s"]
            raise
        except BaseException as exc:
            record["status"] = "FAILED"
            record["error_type"] = type(exc).__name__
            record["queue_depth_at_commit"] = self._verifier_waiting
            if record["queue_start_s"] is not None:
                record["service_observed_s"] = now() - record["queue_start_s"]
            raise
        finally:
            if not active:
                # Cancelled or failed while waiting for a worker.
                self._verifier_waiting -= 1
            else:
                self._verifier_active -= 1
            if acquired:
                self.sem.release()

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
