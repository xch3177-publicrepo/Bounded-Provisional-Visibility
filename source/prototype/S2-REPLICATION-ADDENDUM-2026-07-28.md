# S2 Replication — Addendum 1, 2026-07-28

Filed after rep1 completed and BEFORE rep2--rep5 finished, so it cannot be a
reading of how the sequence turned out. Appended rather than edited into
`S2-REPLICATION-PREREGISTRATION.md`, per that file's own rule.

## What happened

rep1 returned `VERDICT: NOT admissible as E3 evidence` on the preregistered
control bracket: control shift 31.9 ms (9.8%), classified DRIFTED. The original
S2 run this replicates drifted 22.3 ms and classified MILD.

## The confound, stated before the rest of the data exists

The replication is running on a different host from the original: a laptop that
had Docker installed minutes earlier for this purpose, and which was
concurrently compiling the manuscript, running package installs, and running
the W2R experiments while rep1 measured. Sub-millisecond latency percentiles on
a contended host are exactly what the control bracket exists to catch, and it
caught them.

This is a host-level confound identified by what the machine was doing, not by
which way the numbers came out. I am recording it now so that distinction is
verifiable later.

## What this does and does not license

DOES NOT license restarting rep1, or restarting the sequence, or relaxing the
gate. The project's own history is the reason: the S1 replacement run failed
its environment gate and was reported as a failure rather than retried into
success, and that decision is what makes the S1 section credible.

DOES license the following, decided now:

1. The sequence runs to completion. Every run is reported with its verdict.
2. Host load is reduced for the remainder (no manuscript builds, no installs
   during measurement) -- a change to the environment, recorded, not a change
   to the rules.
3. **If fewer than 5 runs are admissible, the paper's "single run" wording
   stands unchanged.** A replication that cannot pass its own environment gate
   is not evidence that the original result replicates. The manuscript gains
   nothing and claims nothing.
4. If the admissible runs among rep1--rep5 disagree with the original run's
   categorical Strong/Session finding, that disagreement is reported even
   though the runs are inadmissible on the timing bracket -- a stale read
   either happened or it did not, and the bracket governs the latency numbers,
   not the categorical observation.

## Expected outcome

Stated in advance so it cannot be reconstructed afterwards: I expect the
remaining runs to drift as well, because the host is a laptop with a VM and no
isolation. The likely honest conclusion is that this machine cannot produce E3
evidence to the standard the original host met, and that the "single run"
caveat survives to submission.
