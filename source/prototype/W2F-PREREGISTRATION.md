# W2F — False-Promotion Sensitivity: Preregistration

2026-07-28, before the experiment is coded past the point of running. No W2F
number exists yet.

## Why

The paper's poisoning bound is conditional: $E_p \le T_p + \delta_{hide}$ holds
for content the verifier does not falsely promote, and the paper says so in
three places. It has never measured what happens on the other side of that
condition. A reviewer is entitled to read an unmeasured caveat as a failure
mode the authors would rather not look at.

W2F looks at it, without needing a real detector. The verifier stays a label
oracle; we make it wrong on purpose, for a chosen number of poison items, and
measure the exposure that results.

This does NOT become evidence about detectors. A real detector's error is
correlated with the content, arrives with its own latency, and can be adaptively
provoked; forcing k of six labels is none of those things. W2F measures the
protocol's sensitivity to promotion error, not any detector's error rate, and
no sentence may say otherwise.

## Design

Same runner, same workloads, same timing configuration as W2, opt-in by flag.
One knob: $k$, the number of the six poison items the verifier wrongly passes,
swept over $k \in \{0, 1, 2, 3\}$. Reported as counts out of six, never as a
percentage: with six items the resolution is $1/6$, and printing "17\%" would
imply a precision the design does not have.

Which items are falsely promoted is fixed per cell before the cell runs, from
the cell's seed, and recorded. All four baselines run at every $k$, because B2's
zero exposure is a claim about the verifier as much as about admission control,
and it should be seen to break.

$k = 0$ must reproduce the existing W2 numbers exactly. If it does not, the
flag has changed behaviour it should not touch and the experiment is void.

## Predictions, recorded before the run

1. **B4 exposure rises with $k$**, because a falsely promoted item is TRUSTED
   and no deadline fires for it. At $k = 6$ B4 would equal B1; at $k = 3$ we
   expect roughly half of B1's exposure plus B4's usual bounded share from the
   items still handled correctly.
2. **B2 exposure becomes nonzero for $k > 0$.** Verify-before-visible admits
   whatever the verifier passes. Its zero in the main table is a property of a
   correct verifier, not of the admission discipline alone. This is the result
   most likely to be uncomfortable and it is being predicted in advance.
3. **B1 is unchanged**, since it never verifies.
4. **$E_u$ is unaffected in all cases.** A falsely promoted item stops being
   unvetted at promotion; the unvetted bound is untouched by promotion error,
   which is exactly the asymmetry between the two bounds the paper defines.

If prediction 4 fails, that is a specification error in the paper, not a
measurement to write around.

## Interpretation rules

1. W2F is a sensitivity analysis, reported as such. It is not a defence, not an
   evaluation, and not a robustness claim.
2. Counts of poison items, not percentages of a detector's error rate.
3. E1 (exact in-memory) only. The mechanism is control-plane; a second tier
   would add cost without adding a distinguishing observation, and saying so
   here prevents a later "we only ran one tier" that reads like an omission.
4. If the result is unflattering -- and prediction 2 is unflattering -- it goes
   in the paper anyway, at the same prominence as the rest.
5. If W2F is not finished and verified by 2026-07-30 evening, nothing from it
   enters the manuscript.
