"""
Invariant validation for the vertical slice (§6.7).
Run:  python3 test_invariants.py     (self-contained; no pytest needed)

Validates I1-I7 + commit-time rule + the I6 overload guarantee: E_u stays bounded
(~T_p) under verifier backlog ONLY when the deadline scheduler is decoupled;
a coupled (broken) scheduler lets unvetted content stay visible for the whole
backlog. E_u is read from the audit log as (first exit from PROVISIONAL) - t_visible.
"""

from slice import Sim, System, Item, State, delta_hide

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"  ({detail})" if detail else ""))


def promoted_directly(sys, iid):
    return any(i == iid and o == State.PROVISIONAL and n == State.TRUSTED
               for (_, i, o, n) in sys.cs.transitions)


def hidden_then_trusted(sys, iid):
    seq = [(o, n) for (_, i, o, n) in sys.cs.transitions if i == iid]
    return (State.PROVISIONAL, State.HIDDEN) in seq and (State.HIDDEN, State.TRUSTED) in seq


def t_exit_prov(sys, iid):
    ts = [t for (t, i, o, n) in sys.cs.transitions if i == iid and o == State.PROVISIONAL]
    return min(ts) if ts else None


def eu_obs(sys, it):
    """E_u = (first exit from PROVISIONAL) - t_visible  (§3, §6.2)."""
    tep = t_exit_prov(sys, it.id)
    return None if tep is None else tep - it.ts["visible"]


# --- T1: minimal path, verification COMMITS AFTER deadline -----------------
def t1_minimal_path_late_verify():
    sim = Sim()
    sys = System(sim, Tp=1.0, verify_cost=2.0)           # verify(2.0) > Tp(1.0)
    it = Item(0)
    sys.admit(it)
    sys.schedule_probe(it, interval=0.05, horizon=5.0)
    sim.run()
    check("T1 final state TRUSTED", it.state == State.TRUSTED, it.state)
    check("T1 PROVISIONAL->HIDDEN->TRUSTED (commit-time rule)", hidden_then_trusted(sys, 0))
    check("T1 NEVER PROVISIONAL->TRUSTED directly (I3/commit-time)", not promoted_directly(sys, 0))
    check("T1 entered PROVISIONAL exactly once (I7)", it.prov_entries == 1, f"n={it.prov_entries}")
    eu, du = eu_obs(sys, it), delta_hide(it)
    check("T1 E_u == T_p (hidden promptly at deadline)", abs(eu - sys.Tp) < 1e-9, f"E_u={eu:.4f}")
    check("T1 delta_hide measured & small", du is not None and du <= 0.06, f"delta_hide={du:.4f}")


# --- T2: verification COMMITS BEFORE deadline -> direct promotion -----------
def t2_before_deadline():
    sim = Sim()
    sys = System(sim, Tp=2.0, verify_cost=0.5)
    it = Item(0)
    sys.admit(it)
    sim.run()
    check("T2 final state TRUSTED", it.state == State.TRUSTED, it.state)
    check("T2 PROVISIONAL->TRUSTED directly (committed <= deadline)", promoted_directly(sys, 0))
    check("T2 never hidden", not any(n == State.HIDDEN for (_, i, o, n) in sys.cs.transitions))


# --- T3: poisoned content rejected -----------------------------------------
def t3_reject_poison():
    sim = Sim()
    sys = System(sim, Tp=2.0, verify_cost=0.5)
    it = Item(0, poisoned=True)
    sys.admit(it)
    sim.run()
    check("T3 poisoned -> QUARANTINED", it.state == State.QUARANTINED, it.state)
    check("T3 poisoned never TRUSTED", not any(n == State.TRUSTED for (_, i, o, n) in sys.cs.transitions))


# --- T4: cheap hard gate at admission --------------------------------------
def t4_hard_gate():
    sim = Sim()
    sys = System(sim, Tp=2.0, verify_cost=0.5)
    it = Item(0, source_ok=False)
    sys.admit(it)
    sys.schedule_probe(it, 0.1, 3.0)
    sim.run()
    check("T4 bad source -> QUARANTINED at admission", it.state == State.QUARANTINED)
    check("T4 never entered PROVISIONAL / never visible", it.prov_entries == 0)
    check("T4 no probe ever saw it visible",
          all(s not in (State.PROVISIONAL, State.TRUSTED) for (_, _, s) in sys.query_log))


# --- T5: query path never returns HIDDEN -----------------------------------
def t5_query_excludes_hidden():
    sim = Sim()
    sys = System(sim, Tp=1.0, verify_cost=5.0)           # never verifies in time
    it = Item(0)
    sys.admit(it)
    sys.schedule_probe(it, 0.05, 3.0)
    sim.run()
    hide_t = it.ts.get("hide_commit")
    bad = [(t, s) for (t, i, s) in sys.query_log
           if t > hide_t + 1e-9 and s in (State.PROVISIONAL, State.TRUSTED)]
    check("T5 no visible observation after hide", len(bad) == 0, f"violations={len(bad)}")


# --- T6: post-promotion revocation -----------------------------------------
def t6_revocation():
    sim = Sim()
    sys = System(sim, Tp=1.0, verify_cost=0.2)
    it = Item(0)
    sys.admit(it)
    sim.at(3.0, lambda: sys.revoke(it))
    sim.run()
    check("T6 TRUSTED then REVOKED", it.state == State.REVOKED, it.state)
    check("T6 has logical-containment timestamp", "contain" in it.ts)


# --- T7: I6 overload -- E_u bounded ONLY when scheduler decoupled -----------
def t7_i6_overload():
    def run(decouple):
        sim = Sim()
        sys = System(sim, Tp=1.0, verify_cost=1.0, decouple_hide=decouple)
        items = [Item(k) for k in range(50)]         # 50 arrive at t=0 -> ~50s backlog
        for it in items:
            sim.at(0.0, (lambda it=it: sys.admit(it)))
        sim.run()
        target = items[40]                            # deep in the backlog
        eus = [eu_obs(sys, it) for it in items]
        return eu_obs(sys, target), max(eus)

    eu_dec, max_dec = run(True)
    eu_cpl, max_cpl = run(False)
    print(f"      decoupled: target E_u={eu_dec:.3f}s   max E_u={max_dec:.3f}s")
    print(f"      coupled:   target E_u={eu_cpl:.3f}s   max E_u={max_cpl:.3f}s   (T_p=1.0)")
    check("T7 decoupled scheduler keeps E_u ~ T_p under backlog (I6)",
          eu_dec <= sys_tp() + 1e-9, f"E_u={eu_dec:.3f}s <= T_p")
    check("T7 coupled (broken) scheduler lets E_u blow up under backlog",
          eu_cpl > 10.0, f"E_u={eu_cpl:.3f}s >> T_p")
    check("T7 -> I6 (decoupled hide) is what bounds unvetted exposure under overload",
          max_cpl > 10 * max_dec, f"{max_cpl:.1f}s vs {max_dec:.3f}s")


def sys_tp():
    return 1.0


if __name__ == "__main__":
    print("Vertical-slice invariant validation (§6.7)\n")
    for fn in [t1_minimal_path_late_verify, t2_before_deadline, t3_reject_poison,
               t4_hard_gate, t5_query_excludes_hidden, t6_revocation, t7_i6_overload]:
        print(f"{fn.__name__}:")
        fn()
        print()
    n_pass = sum(1 for _, c in results if c)
    print(f"==== {n_pass}/{len(results)} checks passed ====")
    raise SystemExit(0 if n_pass == len(results) else 1)
