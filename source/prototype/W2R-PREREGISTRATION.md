# W2R — Real-Text Poison Exposure: Preregistration

Written 2026-07-28, BEFORE any full W2R run. The only numbers seen at the time
of writing are from a feasibility spike on seed 0 (a seed W2R does not use),
which established that the attack is constructible at all: query-stuffed
passages cleared the top-5 corpus bar for 6 of 8 craft queries, reached 2 of 8
held-out same-topic queries, and 0 of 7 unrelated-topic queries. Those numbers
are a go/no-go check on the instrument, not results, and the spike script is
not the runner.

## What W2R is for

An external review named the sharpest objection to W2: the corpus is 600
synthetic Gaussian vectors, the poison is placed in vector space near its
target, and the verifier is an oracle. So W2 demonstrates that the lifecycle
behaves as specified, but not that any of it engages a realistic retrieval
setting.

W2R changes exactly one thing -- what the vectors are:

| | W2 | W2R |
|---|---|---|
| corpus | 600 Gaussian draws, 8 synthetic topic centres, dim 32 | 600 passages, 8 newsgroups, 75 each, MiniLM dim 384 |
| queries | perturbed topic centroid | first 30 words of held-out real passages |
| poison | vector nudged toward a craft query | English passage restating the query, then assertive filler; never moved in vector space |
| everything else | -- | identical: baselines, backlog, T_p grid, seeds, gates, metrics |

`CFG_REALTEXT` is `CFG` with `dim=384` and the two synthetic-only knobs
nulled. No timing parameter is retuned. This is load-bearing: if a knob moved,
a W2/W2R difference could be the knob.

## What W2R can and cannot establish

CAN: that a poisoning attack built in text space against a real embedding
model does enter the top-k of real queries, and that the deadline shortens how
long and how often it does so, on a corpus nobody in this project generated.

CANNOT: anything about detector quality. The verifier remains a label oracle
in W2R exactly as in W2. W2R makes the WORKLOAD real, not the DEFENCE. Any
sentence implying W2R evaluates detection is false and must not be written.

CANNOT: anything about end-to-end LLM answer manipulation. W2R measures
retrieval placement -- which documents reach the top-k -- and stops there. No
generation is run, so "the answer was manipulated" is not claimable.

## Frozen interpretation rules

1. **The existing W2 admissibility gates apply unchanged** (`admissibility()`,
   W2-PREREGISTRATION.md §5). They are properties of the protocol, not of the
   corpus. A gate failure means the W2R RUN is inadmissible; it never means the
   protocol failed, and it never licenses relaxing the gate.

2. **Gate 1 is the workload-validity gate here.** If the poison never enters
   B1's top-k on Q_craft, W2R measured nothing about poisoning and no W2R
   number goes in the paper -- including favourable-looking ones. The spike says
   this will pass; if it does not, W2R is reported as an unsuccessful attack
   construction and the paper keeps W2 alone. It is not retried with a
   different attack until one works: that would be tuning the attack to the
   result. ONE amendment is permitted, must be committed before re-running, and
   must state what changed and why.

3. **Craft/target/negative keep their W2 meanings.** Q_craft is an upper bound
   by construction. The headline is Q_target (transfer). Q_negative is a
   contamination check expected near zero.

4. **The transfer rate is reported as measured, with no success threshold.**
   Whatever fraction of Q_target the poison reaches -- higher than synthetic,
   lower, or zero -- is printed with its denominator. A low real-text transfer
   rate is a finding about text-space attacks, not a failed run, and it will be
   stated plainly rather than omitted.

5. **W2 vs W2R is a mechanism comparison, never a point comparison.** The two
   run on different corpora in different dimensions; retrieval counts are not
   commensurable. Permitted claims are ordinal and per-workload:
   B1 >= B3 >= B4 >= B2 ordering, heavy-vs-normal backlog direction, T_p
   monotonicity, and B2's clean-freshness cost. Any sentence of the form "35 in
   synthetic vs N in real text" is barred.

6. **Both evidence tiers or neither.** W2R runs on E1 (exact in-memory) and E2
   (Milvus Lite FLAT), as W2 does. A mechanism that appears in one tier only is
   reported as tier-specific.

7. **Five seeds, medians and full ranges.** No confidence intervals, no
   significance tests. Same rule as W2.

8. **The frozen embedding cache is the unit of reproducibility.** Its sha256 is
   recorded in every W2R result file. If the cache is rebuilt, every prior W2R
   result is superseded, not merged.

9. **Failure to finish is not failure.** If W2R does not produce admissible
   results in both tiers by 2026-07-30 evening, the paper ships with W2 alone
   and the real-text workload becomes future work. No partial W2R number enters
   the manuscript.

## Verification

`verify_w2r.py` recomputes the mechanism assertions from the frozen JSONs,
independently of the runner's own admissibility output, on the pattern of
`verify_w2.py`. Runner self-assessment is not evidence.
