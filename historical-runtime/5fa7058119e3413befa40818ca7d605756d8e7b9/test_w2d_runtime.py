#!/usr/bin/env python3
"""Focused guards for the W2D runtime seam.

These tests deliberately use tiny, deterministic fixtures. They validate the
plumbing needed by a later real-detector-output/service-time replay runner
without importing a detector, a workload builder, or evaluator labels.

    ./.venv312/bin/python test_w2d_runtime.py
"""

import asyncio
import json
import time

from backend import InMemoryBackend
from functional_slice import State, System, VerifierDecision, VerifierRequest
import poison_exposure as w2


checks = []


def check(name, cond, detail=""):
    checks.append((name, bool(cond)))
    if detail and not cond:
        print(f"      {detail}")


async def finish(sys):
    tasks = list(sys.tasks)
    if tasks:
        await asyncio.gather(*tasks)
    errs = await sys.shutdown()
    check("completed verifier tasks raised no background error", not errs, repr(errs))


async def custom_rejection_paths():
    """B2/B3/B4 share one provider contract and all honor a clean rejection."""
    safe_inputs = {}
    seen = []

    def reject(request):
        # If System accidentally passed its item dictionary, this assertion sees
        # `content_bad` and the test fails inside the background task.
        assert isinstance(request, VerifierRequest)
        assert request.verifier_input in safe_inputs.values()
        assert not isinstance(request.verifier_input, dict)
        assert request.item_key.startswith("opaque-")
        seen.append(request.verifier_input)
        return VerifierDecision(
            passes=False,
            service_time_s=0.004,
            metadata={"family_s_pass": True, "family_c_pass": False,
                      "integration": "real-detector-output/service-time replay"},
        )

    out = {}
    for baseline, sync, deadline in (
            ("B2", True, False), ("B3", False, False), ("B4", False, True)):
        safe = object()
        safe_inputs[baseline] = safe
        sys = System(
            InMemoryBackend(), Tp=0.2, verify_cost=9.0,
            verifier_concurrency=1, verify=True, sync=sync,
            deadline=deadline, verifier=reject)
        t0 = time.monotonic()
        it = sys.admit(
            1, [1.0, 0.0], content_bad=False,
            verifier_input=safe, verifier_key=f"opaque-{baseline}")
        await finish(sys)
        m = w2.summarise(sys, [], [], [1], w2.CFG, 0.2, t0)
        out[baseline] = (sys, it, m)

    check("custom provider received exactly the three opaque inputs",
          len(seen) == 3 and set(seen) == set(safe_inputs.values()))
    for baseline, (sys, it, _) in out.items():
        check(f"{baseline}: clean custom rejection reaches QUARANTINED",
              it["state"] == State.QUARANTINED)
        rec = sys.verifier_records[0]
        check(f"{baseline}: shared path records queue/start/commit",
              rec["queue_enter_s"] is not None
              and rec["queue_start_s"] is not None
              and rec["decision_commit_s"] is not None)
        check(f"{baseline}: record exposes replay service and integrated latency",
              rec["service_time_s"] == 0.004
              and rec["service_observed_s"] >= 0
              and rec["integrated_latency_s"] >= rec["service_observed_s"])
        check(f"{baseline}: family outcomes survive as opaque metadata",
              rec["metadata"]["family_s_pass"] is True
              and rec["metadata"]["family_c_pass"] is False)
        check(f"{baseline}: record contains no evaluator truth or verifier input",
              "content_bad" not in rec and "verifier_input" not in rec)

    b2m = out["B2"][2]
    check("B2 false positive never starts a visibility-withdrawal episode",
          b2m["clean_withdrawal_started_n"] == 0)
    check("B2 false positive opens a right-censored quarantine episode",
          b2m["clean_quarantine_started_n"] == 1
          and b2m["clean_quarantine_right_censored_n"] == 1)
    check("B2 false positive is censored in first and durable visibility",
          b2m["Df_right_censored_n"] == 1
          and b2m["Df_trusted_right_censored_n"] == 1)

    for baseline in ("B3", "B4"):
        m = out[baseline][2]
        check(f"{baseline}: false-positive withdrawal is not called durable",
              m["Df_trusted_completed_n"] == 0
              and m["Df_trusted_right_censored_n"] == 1)
        check(f"{baseline}: false-positive withdrawal and quarantine are censored",
              m["clean_withdrawal_started_n"] == 1
              and m["clean_withdrawal_right_censored_n"] == 1
              and m["clean_quarantine_started_n"] == 1
              and m["clean_quarantine_right_censored_n"] == 1)
        check(f"{baseline}: verifier rejection does not masquerade as deadline expiry",
              m["clean_expired_n"] == 0
              and m["clean_gap_completed_n"] == 0
              and m["clean_gap_right_censored_n"] == 0)


async def legacy_oracle_paths():
    """The optional seam must not move the legacy oracle or W2F behaviour."""

    async def state(sync, bad, promoted):
        sys = System(
            InMemoryBackend(), verify_cost=0.001, verify=True, sync=sync,
            deadline=False, false_promote={1} if promoted else frozenset())
        it = sys.admit(1, [1.0, 0.0], content_bad=bad)
        await finish(sys)
        return it["state"], sys.verifier_records[0]

    for sync, tag in ((True, "B2"), (False, "B3")):
        clean, clean_rec = await state(sync, False, False)
        rejected, rejected_rec = await state(sync, True, False)
        promoted, promoted_rec = await state(sync, True, True)
        check(f"{tag}: default oracle still accepts clean", clean == State.TRUSTED)
        check(f"{tag}: default oracle still rejects labelled poison",
              rejected == State.QUARANTINED)
        check(f"{tag}: legacy false_promote still overrides the oracle",
              promoted == State.TRUSTED)
        check(f"{tag}: default path identifies itself only as oracle outcome",
              all(r["metadata"] == {"provider": "oracle"}
                  for r in (clean_rec, rejected_rec, promoted_rec)))


async def queue_and_service_records():
    """Service draws are paired; integrated latency remains queue-dependent."""

    def replay(_request):
        return {"passes": True, "service_time_s": 0.025,
                "family_s_pass": True, "family_c_pass": True}

    sys = System(
        InMemoryBackend(), verify=True, sync=False, deadline=False,
        verifier_concurrency=1, verifier=replay)
    sys.admit(1, [1.0, 0.0], verifier_input=("safe", 1), verifier_key="k1")
    sys.admit(2, [0.0, 1.0], verifier_input=("safe", 2), verifier_key="k2")
    await finish(sys)
    r1, r2 = sys.verifier_records
    check("paired replay records the same service draw",
          r1["service_time_s"] == r2["service_time_s"] == 0.025)
    check("the second item observed real queueing",
          r2["queue_wait_s"] > r1["queue_wait_s"]
          and r2["integrated_latency_s"] > r1["integrated_latency_s"],
          f"{r1['queue_wait_s']}, {r2['queue_wait_s']}")
    check("queue instrumentation records active work ahead",
          r2["active_at_enqueue"] >= 1 or r2["queue_depth_at_enqueue"] >= 1)


async def cancelled_replay_record():
    """A known decision is not the same thing as a committed decision."""

    def slow_replay(request):
        assert request.verifier_input == "safe-payload"
        return {"passes": True, "service_time_s": 0.2,
                "family_s_pass": True, "family_c_pass": True}

    sys = System(
        InMemoryBackend(), Tp=0.01, verify=True, sync=False, deadline=True,
        verifier_concurrency=1, verifier=slow_replay)
    it = sys.admit(
        1, [1.0, 0.0], content_bad=True,
        verifier_input="safe-payload", verifier_key="opaque-slow")
    await asyncio.sleep(0.04)  # provider returned; deadline fired; no commit yet
    errs = await sys.shutdown()
    rec = sys.verifier_records[0]
    check("cancelled replay raised no background error", not errs, repr(errs))
    check("cancelled replay preserves the frozen outcome without committing it",
          rec["status"] == "CANCELLED" and rec["passes"] is True
          and rec["decision_available_s"] is not None
          and rec["decision_commit_s"] is None)
    check("an uncommitted positive replay cannot readmit the item",
          it["state"] == State.HIDDEN and "trusted" not in it["ts"])


async def completed_failure_is_reported():
    """A verifier that fails before shutdown must not disappear from its sink."""

    def fail(_request):
        raise RuntimeError("deliberate verifier failure")

    sys = System(
        InMemoryBackend(), verify=True, sync=False, deadline=False,
        verifier_concurrency=1, verifier=fail)
    sys.admit(1, [1.0, 0.0], verifier_input="safe", verifier_key="opaque-fail")
    await asyncio.sleep(0.02)  # let the task fail before shutdown inspects it
    errs = await sys.shutdown()
    check("completed verifier failure is retained until shutdown",
          len(errs) == 1 and isinstance(errs[0], RuntimeError), repr(errs))
    check("failed verifier record remains explicit",
          sys.verifier_records[0]["status"] == "FAILED"
          and sys.verifier_records[0]["error_type"] == "RuntimeError")


async def run_cell_hooks():
    """The common cell runner exposes enough hooks for a later W2D runner."""
    cfg = json.loads(json.dumps(w2.CFG))
    cfg.update(
        dim=2, corpus=4, k=1, over_fetch=2, qps=30, dur=0.18,
        inject_at=0.02, n_poison=1, n_clean=1, n_craft_q=1,
        n_target_q=1, n_neg_q=1, verify_cost=0.001,
        backlog={"normal": {"conc": 1, "items": 0}},
    )
    world = {
        "corpus": [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
        "q_craft": [[1.0, 0.0]],
        "q_target": [[0.9, 0.1]],
        "q_neg": [[0.0, 1.0]],
        "poison": [[1.0, 0.0]],
        "clean": [[0.0, 1.0]],
        "filler": [],
    }
    backend = InMemoryBackend()
    idbase = w2.id_base(1, cfg)
    w2.reset_corpus(backend, world["corpus"], cfg, idbase)
    pre = {(kind, i): w2.corpus_topk(world["corpus"], q, cfg["k"], idbase)
           for kind, key in (("craft", "q_craft"), ("target", "q_target"))
           for i, q in enumerate(world[key])}

    detector_inputs = []
    observed_queries = []

    class RecordingDetector:
        def score(self, payload):
            # This is the security boundary the replay runner must preserve:
            # detector features contain neither the evaluator role/label nor
            # the opaque key used to pair a frozen verdict across baselines.
            assert set(payload) == {"embedding"}
            assert all(isinstance(x, float) for x in payload["embedding"])
            detector_inputs.append(payload)
            return payload["embedding"][0] < 0.5

    detector = RecordingDetector()

    def admission_hook(iid, vec, **_):
        return {"verifier_input": {"embedding": tuple(vec)},
                "verifier_key": f"seed1/item/{iid}"}

    def replay(request):
        # The provider owns the opaque pairing key, but strips it before the
        # one-time live detector call. A later replay provider can instead use
        # item_key to return the frozen result from that scoring pass.
        assert request.item_key.startswith("seed1/item/")
        passes = detector.score(request.verifier_input)
        return {"passes": passes, "service_time_s": 0.001,
                "family_s_pass": True, "family_c_pass": passes}

    def metrics_hook(sys, **_):
        return {"hook_observed_commits": sum(
            r["status"] == "COMMITTED" for r in sys.verifier_records)}

    m = await w2.run_cell(
        backend, "B3", "normal", 1.0, 1, cfg, world,
        idbase + cfg["corpus"], pre, verifier=replay,
        admission_hook=admission_hook, metrics_hook=metrics_hook,
        query_observer=lambda event: observed_queries.append(event))
    check("run_cell emits per-item verifier records for custom runs",
          len(m["verifier_records"]) == 2)
    check("run_cell preserves opaque stable keys",
          {r["item_key"] for r in m["verifier_records"]}
          == {f"seed1/item/{idbase + cfg['corpus']}",
              f"seed1/item/{idbase + cfg['corpus'] + 1}"})
    check("detector.score received only safe payloads, never pairing keys or truth",
          len(detector_inputs) == 2
          and all(set(payload) == {"embedding"}
                  for payload in detector_inputs))
    check("metrics hook can append without replacing common fields",
          m["hook_observed_commits"] == 2
          and "poisoned_retrievals_craft" in m)
    check("run_cell custom records do not serialise verifier inputs or truth",
          all("verifier_input" not in r and "content_bad" not in r
              for r in m["verifier_records"]))
    check("query observer receives exact returned and poison ids",
          bool(observed_queries)
          and all(set(event) == {
              "t", "kind", "qidx", "returned_ids",
              "poison_ids_retrieved", "recall", "displacement",
          } for event in observed_queries)
          and all(
              set(event["poison_ids_retrieved"])
              <= set(event["returned_ids"])
              for event in observed_queries
          ))


async def main():
    await custom_rejection_paths()
    await legacy_oracle_paths()
    await queue_and_service_records()
    await cancelled_replay_record()
    await completed_failure_is_reported()
    await run_cell_hooks()


if __name__ == "__main__":
    asyncio.run(main())
    passed = sum(ok for _, ok in checks)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n==== test_w2d_runtime: {passed}/{len(checks)} checks passed ====")
    raise SystemExit(0 if passed == len(checks) else 1)
