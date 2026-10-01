# Measurement specification (artifact)

This was Appendix A of the manuscript. It moved here for the CBDCom page budget
(regular papers get 8 pages, up to 10 at $100 per extra page), not because it
became optional: the paper cites this file for the full schema, and §6.1 keeps
the three distinctions that are load-bearing for the results.

The analysis derives every metric from the versioned event schema alone, so the
standalone deployment emits the same fields without changing any definition.
Emitter of record: `analysis.py`. Acceptance gate: `validate_results.py`.

## A.1 Per-item lifecycle timestamps

| Symbol | Event |
|---|---|
| `t_arrival` | enters ingestion service |
| `t_admit` | cheap `r(x)` + hard gates done |
| `t_insert-ack` | vector-store write acknowledged |
| `t_visible` | first returnable by an ordinary query |
| `t_deadline` | provisional-visibility deadline (= `t_visible` + `T_p`) |
| `t_verify-start` / `t_verify-end` | async verification starts / finishes computing |
| `t_verify-commit` | verification result atomically committed |
| `t_trusted` | enters TRUSTED |
| `t_hide-commit` | state set HIDDEN (control plane) |
| `t_hide-effective` | query path actually stops returning it |
| `t_alert` | post-hoc malicious / lineage alert |
| `t_contain` | logical containment effective |
| `t_purge` | physical deletion complete |

## A.2 Derived metrics

- `D_f = t_visible - t_arrival`
- `delta_control = t_hide-commit - t_deadline`
- `delta_hide = t_hide-effective - t_deadline = delta_sched + delta_state + delta_prop`
- `E_u = t_exit-provisional - t_visible`
- `L_v = t_verify-commit - t_verify-start`
- `E_p = t_contain - t_first-poison-visible`
- `L_contain = t_contain - t_alert`
- `W_recovery = t_purge - t_contain`
- latency decomposition `L_total = L_filter + L_control + L_DB-insert + L_visibility`

## A.3 Probe methodology

`t_hide-effective` is observed by a sentinel probe stream whose target enters
the top-1 when eligible. Probes run at a fixed cadence around each deadline,
identical across baselines and not throttled under overload. The probe yields
`delta_hide_hat = t_first-hidden-probe - t_deadline`, an **upper bound** on the
true delay with measurement error `<= probe_interval + jitter`. The probe
interval is never attributed to system delay.

S2 runs **two probes concurrently** — a lightweight scalar point lookup (state
visibility) and a full FLAT search (query-path confirmation). They must be
concurrent: run serially, the scalar probe changes the sampling cadence of the
search probe mid-measurement. S2 must use FLAT, because under an approximate
index "the item is gone" is ambiguous between a hide taking effect and an ANN
miss.

## A.4 Experimental validity

Durations use a monotonic clock; wall time is kept only for cross-component
correlation. Instrumentation writes asynchronously off the query critical path,
and its on/off overhead is reported (S2 carries an explicit probe-off control
cell).

A single versioned event schema backs both backends and the analysis is
backend-agnostic: `schema_version`, `run_id`, `config_hash`, and per-event
`item_id`, `state_before/after`, `commit_time`, `component`. Records also carry
`git_commit`, `server_version`, `server_uri`, `deployment_mode`, and
`run_profile`, so no future reader has to reconstruct from latency whether a
number came from Lite or a real server.

**Evidence tiers are not interchangeable.** The in-memory backend is a
correctness harness (I1–I7, races, CAS), *not* a vector-store performance
baseline. Milvus Lite (FLAT, single process) is a functional microbenchmark,
*not* production performance. Only `deployment_mode=standalone` records are E3.

### Frozen reporting thresholds

A percentile needs roughly ten observations in its own tail, so:
`median >= 30`, `P95 >= 200`, `P99 >= 1000` samples. At n=25 the "P99" is just
the maximum. A paired interval spanning 1.0 does **not** mean "no difference" —
report the direction, the count of blocks, and a sign test. Zero observed
failures over n trials bounds the rate at about `3/n` (rule of three), computed
**per operating point**, never pooled.

## A.5 Query log

Per query: `query_id`, `timestamp`, `returned_vector_ids`,
`returned_item_states` (**as observed at return time**, never reconstructed
from the item's later state), `rank`, `similarity`, `query_latency`.

From this:
- Poisoned-Query Exposure Count = #{queries returning an uncontained poisoned item}
- Residual Exposure Rate = #{post-containment queries still returning x} / #{post-containment queries}
