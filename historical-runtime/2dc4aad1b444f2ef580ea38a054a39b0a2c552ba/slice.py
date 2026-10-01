"""
Vertical slice for the fail-closed provisional-visibility protocol
(topic-c-cbdcom-alignment.md §0.11 invariants I1-I7; topic-c-paper-draft.md §4-6).

Discrete-event simulation with a logical clock -> deterministic, reproducible
delta_hide numbers (no real-time flakiness). In-memory vector store behind a
minimal interface; the authoritative state machine lives in a separate control
store (as in §5). Run:  python3 slice.py
"""

import heapq
import itertools
from enum import Enum


class State(str, Enum):
    PROVISIONAL = "PROVISIONAL"
    TRUSTED = "TRUSTED"
    QUARANTINED = "QUARANTINED"
    HIDDEN = "HIDDEN"
    REVOKED = "REVOKED"


VISIBLE = {State.PROVISIONAL, State.TRUSTED}  # supported query path eligibility


class Sim:
    """Minimal discrete-event engine with a logical clock."""
    def __init__(self):
        self.t = 0.0
        self._q = []
        self._seq = itertools.count()

    def at(self, t, cb):
        heapq.heappush(self._q, (t, next(self._seq), cb))

    def run(self):
        while self._q:
            self.t, _, cb = heapq.heappop(self._q)
            cb()


class Item:
    __slots__ = ("id", "poisoned", "source_ok", "content_bad",
                 "state", "deadline", "prov_entries", "pending_pass", "ts",
                 "was_visible")

    def __init__(self, id, poisoned=False, source_ok=True, content_bad=None):
        self.id = id
        self.poisoned = poisoned
        self.source_ok = source_ok               # cheap hard-gate signal
        self.content_bad = poisoned if content_bad is None else content_bad
        self.state = None
        self.deadline = None
        self.prov_entries = 0                     # I7 counter
        self.pending_pass = False                 # late pass awaiting hide
        self.ts = {}                              # §6.1 timestamps
        self.was_visible = False


class ControlStore:
    """Separate transactional state store (not the vector DB). CAS is atomic
    because each callback runs to completion with no interleaving (sim = single
    logical thread), modelling an atomic compare-and-set on the state column."""
    def __init__(self, sim):
        self.sim = sim
        self.items = {}
        self.transitions = []                     # audit: (t, id, old, new)

    def add(self, it):
        self.items[it.id] = it

    def cas(self, it, expected, new, stamp=None):
        if it.state != expected:
            return False
        old = it.state
        it.state = new
        if new == State.PROVISIONAL:
            it.prov_entries += 1
        if stamp:
            it.ts[stamp] = self.sim.t
        self.transitions.append((self.sim.t, it.id, old, new))
        return True

    def eligible(self, it):
        return it.state in VISIBLE


class System:
    """Admission control plane + independent deadline scheduler + async verifier
    + query service, over an in-memory vector store."""
    def __init__(self, sim, Tp=1.0, verify_cost=0.1, decouple_hide=True,
                 detector_fn_rate=0.0):
        self.sim = sim
        self.cs = ControlStore(sim)
        self.Tp = Tp
        self.verify_cost = verify_cost
        self.decouple_hide = decouple_hide        # I6: independent scheduler
        self.detector_fn_rate = detector_fn_rate  # false-negative (for E_p study)
        self.vbusy_until = 0.0                     # verifier single-server FIFO tail
        self.query_log = []                        # (t, item_id, state_at_return)

    # ---- admission -------------------------------------------------------
    def admit(self, it):
        it.ts["arrival"] = self.sim.t
        it.ts["admit"] = self.sim.t
        if not it.source_ok:                       # cheap hard gate at admission
            it.state = State.QUARANTINED
            self.cs.add(it)
            self.cs.transitions.append((self.sim.t, it.id, None, State.QUARANTINED))
            return
        it.state = State.PROVISIONAL               # initial entry into PROVISIONAL
        it.prov_entries = 1
        it.deadline = self.sim.t + self.Tp
        it.ts["insert_ack"] = self.sim.t
        it.ts["visible"] = self.sim.t
        it.ts["deadline"] = it.deadline
        it.was_visible = True
        self.cs.add(it)
        self.cs.transitions.append((self.sim.t, it.id, None, State.PROVISIONAL))
        # independent deadline scheduler event (fires regardless of verifier)
        self.sim.at(it.deadline, lambda: self._deadline(it))
        # async verifier (single-server FIFO -> backlog under overload)
        it.ts["verify_start"] = max(self.sim.t, self.vbusy_until)
        ready = max(self.sim.t, self.vbusy_until) + self.verify_cost
        self.vbusy_until = ready
        self.sim.at(ready, lambda: self._verify_done(it))

    # ---- deadline scheduler (I3, I6) -------------------------------------
    def _deadline(self, it):
        if self.decouple_hide:
            hide_at = self.sim.t                   # O(1), independent of verifier
        else:                                      # broken variant: hide queued
            hide_at = max(self.sim.t, self.vbusy_until)  # behind verifier backlog
        if hide_at == self.sim.t:
            self._do_hide(it)
        else:
            self.sim.at(hide_at, lambda: self._do_hide(it))

    def _do_hide(self, it):
        if self.cs.cas(it, State.PROVISIONAL, State.HIDDEN, "hide_commit"):
            if it.pending_pass:                    # late pass arrived first
                self.cs.cas(it, State.HIDDEN, State.TRUSTED, "verify_commit")
                it.ts["trusted"] = self.sim.t

    # ---- async verifier (I1, I2, commit-time rule) -----------------------
    def _verify_done(self, it):
        it.ts["verify_end"] = self.sim.t
        # detector: clean iff not content_bad (with optional false-negative)
        passes = (not it.content_bad) or (it.content_bad and self._fn())
        # commit time == now (§4.4: compare commit time, not compute time)
        it.ts["verify_commit"] = self.sim.t
        if passes:
            if self.sim.t <= it.deadline:
                if self.cs.cas(it, State.PROVISIONAL, State.TRUSTED, "verify_commit"):
                    it.ts["trusted"] = self.sim.t
                    return
            # past deadline (or lost race): must route through HIDDEN
            if self.cs.cas(it, State.HIDDEN, State.TRUSTED, "verify_commit"):
                it.ts["trusted"] = self.sim.t
            elif it.state == State.PROVISIONAL:    # scheduler not yet fired
                it.pending_pass = True             # hide handler will promote
        else:
            if not self.cs.cas(it, State.PROVISIONAL, State.QUARANTINED, "verify_commit"):
                self.cs.cas(it, State.HIDDEN, State.QUARANTINED, "verify_commit")

    def _fn(self):
        # deterministic pseudo false-negative gate (no Math.random dependence)
        return False if self.detector_fn_rate <= 0 else (hash(("fn",)) % 100) < 0

    # ---- post-hoc revocation (I5) ----------------------------------------
    def revoke(self, it):
        it.ts["alert"] = self.sim.t
        if self.cs.cas(it, State.TRUSTED, State.REVOKED, "contain"):
            it.ts["contain"] = self.sim.t

    # ---- query service / probe (records state AS OBSERVED) ---------------
    def probe(self, it):
        vis = self.cs.eligible(it)
        self.query_log.append((self.sim.t, it.id, it.state))
        if it.was_visible and not vis and "hide_effective" not in it.ts \
                and it.state in (State.HIDDEN, State.QUARANTINED, State.REVOKED):
            it.ts["hide_effective"] = self.sim.t

    def schedule_probe(self, it, interval, horizon):
        def tick():
            if self.sim.t > horizon:
                return
            self.probe(it)
            self.sim.at(self.sim.t + interval, tick)
        self.sim.at(self.sim.t, tick)


# ---- derived metrics (§6.2) ---------------------------------------------
def E_u(it):
    exit_prov = it.ts.get("trusted") if it.ts.get("verify_commit", 1e18) <= it.deadline \
        else it.ts.get("hide_effective")
    if exit_prov is None:
        exit_prov = it.ts.get("hide_effective") or it.ts.get("trusted")
    return None if exit_prov is None else exit_prov - it.ts["visible"]


def delta_hide(it):
    return None if "hide_effective" not in it.ts else it.ts["hide_effective"] - it.ts["deadline"]
