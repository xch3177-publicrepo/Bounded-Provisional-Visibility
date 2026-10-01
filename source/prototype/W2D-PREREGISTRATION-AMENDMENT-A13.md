# W2D Preregistration — Amendment A13

2026-07-28, after the first complete E2 execution reached 140/140 cells but
before that execution passed the independent verifier or any E2 result was
released.

## Triggering verifier rejection

The completed E2 execution is not an accepted result.  The independent
verifier rejected its provider-decision/state-transition closure.  Among 2,160
committed B2/B3/B4 Milvus item records, five had a gap greater than 10 ms
between the provider's recorded decision commit and the corresponding terminal
state transition; all five were B2 records (5/720 B2 records).  The largest
observed E2 gap was 14.057 ms.  In the completed E1 execution, the largest
corresponding gap was 0.176 ms.

These observations are disclosed because they motivated this amendment.  They
are diagnostic values from executions made under the superseded gate, not
accepted protocol estimates and not calibration data for a replacement
threshold.

## Defect in the former closure gate

The former verifier required the terminal-transition timestamp to be within
10 ms of `decision_commit_s`.  That symmetric near-equality gate conflated two
distinct events:

1. the provider committing a decision; and
2. the runtime applying that decision as a state transition.

The 10 ms constant was therefore a measurement-model defect.  It was neither a
protocol safety bound nor a preregistered service-level objective.  This
amendment does not increase it to 15 ms, tune it to the observed maximum, or
replace it with another favorable pass threshold.  The near-equality gate is
removed.

## Strict transition identity and temporal order

For every item whose provider record has status `COMMITTED`, the verifier must
instead establish all of the following from the retained provider and
lifecycle ledgers:

1. the committed decision implies exactly one terminal state,
   `TRUSTED` or `QUARANTINED`, under the frozen arm decision;
2. the lifecycle ledger contains exactly one transition into that implied
   terminal state;
3. that transition occurs at or after `decision_commit_s`, with no symmetric
   closeness tolerance and no alternative transition accepted as closure; and
4. the lifecycle final state equals the decision-implied terminal state.

The provider-to-state actuation gap is then defined for each committed item as

\[
G_{\mathrm{act}}
  = t_{\mathrm{terminal\ transition}} - t_{\mathrm{decision\ commit}} .
\]

Every accepted gap must be finite and nonnegative.  There is no upper
pass/fail threshold on \(G_{\mathrm{act}}\); a slow but correctly ordered
transition remains an observed latency rather than being relabelled as a
protocol violation.

## Formal latency reporting

\(G_{\mathrm{act}}\) becomes a formal latency distribution alongside the
existing detector-service, queue-wait, and queue-entry-to-commit
distributions.  The analyzer retains every committed-item observation and
reports count, P50, P95, P99, and maximum separately for E1 and E2 under the
existing baseline, backlog, and arm groupings.  It also reports the count and
rate above the superseded 10 ms diagnostic boundary, explicitly as
description rather than an acceptance gate.

The gap is not detector inference latency and is not substituted for any
existing latency estimand.  It measures the observed delay from provider
decision commit to the corresponding logical state actuation in the measured
runtime.

## Frozen scientific inputs and mandatory replay

This amendment changes no D1 score or decision, detector threshold, label,
attack construction, landing assignment, plan cell, baseline, arm, seed,
backlog, query schedule, observation horizon, promotion outcome, denominator,
or exposure estimand.  It adds a correct causal closure rule and a latency
report derived from already required provider and lifecycle timestamps.

The rejected E2 execution must not be relabelled as accepted, edited, or
reported as a protocol result; an explicitly rejected archival filename does
not change that status.  Because A13 and the implementing verifier/analyzer
enter the runtime fingerprint, both E1 and E2 require new fingerprints, new
tier-specific manifests under the shared manifest authority, and complete
replays before either post-A13 result is accepted.  No pre-A13 result may be
mixed with a post-A13 result.

## Backend-receipt boundary

A13 does not strengthen the A12 backend receipts into identity proof.  The
pre/post receipts remain live descriptive evidence collected outside the
measured run window.  They are not remote attestation or cryptographic proof
of backend identity; hashes bind the recorded receipt bytes, not an
unobservable deployment claim.

A13 supersedes only the former 10 ms provider-decision/state-transition
near-equality gate and enters the runtime fingerprint before either mandatory
replay.
