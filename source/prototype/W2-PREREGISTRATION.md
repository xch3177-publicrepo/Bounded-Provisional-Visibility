# W2 — end-to-end poison exposure: how the numbers will be judged

Frozen 2026-07-27, **before the experiment was run**. Same discipline as
`RUN-SHEET.md` (how a run is judged) and `s1-writeup-decision.md` (how the
judgement is written). Both had to precede the data, and so does this.

> **Amendment 1, 2026-07-27, before any full run.** A smoke run (3 s cells, one
> seed, 200-vector corpus) had been executed as a code-path check when this was
> amended. The amendments come from external review of the *design*, not from
> those numbers, and none of them changes how an outcome is judged in a
> direction the smoke could have suggested. Recorded rather than silently
> applied, because an unlogged amendment to a preregistration is worth less than
> no preregistration at all. What changed:
> 1. **Three query sets instead of two.** Measuring PRR only on the queries the
>    poison was built against can establish nothing but an oracle worst case.
>    `Q_target` — different queries from the same topic, never used to craft
>    poison — is added and becomes the headline set; the crafted set becomes an
>    explicit upper bound. This required giving the corpus topic structure.
> 2. **Clean Eligible Recall@k is measured**, split at containment. It was
>    missing: the run measured what exposure cost in freshness but not in
>    retrieval quality.
> 3. **The $T_p$ sweep is anchored to verifier latency** (below / at / between /
>    above), replacing four round numbers chosen for looks.
> 4. **Five seeds, not three**, reporting seed-level median and range.
> 5. **The containment-to-last-hit gap is reported as a censoring bound**, not
>    as a system delay: in post-filter mode the control store is authoritative,
>    so that gap is bounded by the query interval by construction.

The experiment closes a hole the paper itself opens: §III states that poisoning
exposure $E_p$ "is therefore *measured empirically* (Sec. VII)", §VI-B declares
workload **W2 (poison burst)**, and §VII reports neither. `claim-evidence-matrix.md`
records the same gap from the other side — P8-X was withdrawn because "the only
experiment that would have produced it (W2, poison burst) was never executed."

---

> **Amendment 2, 2026-07-27, after a full E1 run and before E2.** Unlike
> amendment 1, this one was written **with the E1 numbers in hand**, and that has
> to be said plainly rather than buried. What was seen: the unconditional
> poisoned-retrieval counts on $Q_{\text{target}}$ were small for every baseline
> (B1 7 of 64, B4 1 of 64), because the attack reaches only a minority of
> non-crafted queries at all. What changed, and why it is not outcome-fitting:
> 1. **The defence effect is reported conditional on the queries B1 actually
>    compromised**, alongside the attack's coverage. This *separates* two
>    quantities the unconditional rate mixes; it does not change any gate, any
>    branch of §6, or which direction counts as success. The qualified set is
>    fixed from **B1 only** and applied unchanged to all four baselines and every
>    $T_p$ — deriving it per baseline would be selecting on the outcome.
> 2. **Recall and displacement are sampled on $Q_{\text{craft}}$ as well.** The
>    first run measured displacement only on $Q_{\text{target}}$ and reported
>    0.0 everywhere — not because the poison is harmless but because it rarely
>    reaches a target query's top-$k$ at all, so the set where displacement
>    happens was the one not being measured. This is a **defect fix**, not a
>    reinterpretation.
> 3. **Residual rows between cells are counted and gated** (gate 5). Deletion
>    failures were being swallowed; a leaked poison vector sits at cosine
>    $\approx 0.96$ to the craft queries and would eat over-fetch slots in every
>    later cell.
> 4. Right-censoring of the observed window is emitted as an explicit flag
>    rather than inferred.
>
> The pre-amendment E1 run is kept as
> `results/W2-inmemory-unconditional-2026-07-27.json`. **Both tiers are re-run
> from scratch under the amended analysis** so E1 and E2 are never compared
> across two different analyses.

## 1. What is genuinely new here, and what is not

**Not new, and must never be written as a finding.** That B4 hides at the
deadline is *constructed*: invariants I3 (expiry never promotes) and I4
(visibility $\le T_p + \delta_{\text{hide}}$) are design properties, asserted by
20/20 tests already reported in §VII-A. A curve showing B4 dropping near $T_p$
re-displays a guarantee; it does not discover one. Any sentence of the form
"the experiment shows the protocol works" is banned.

**New, and the only things this experiment may claim:**

1. **Retrieval count.** How many top-$k$ results actually contained poison
   before containment. This is not derivable from the bound: it depends on
   query rate, on whether the poison wins the top-$k$ competition, and on where
   in the exposure window queries land.
2. **Backlog response.** B3's exposure grows with verifier backlog; B4's does
   not. §VII-A already shows this as a *scheduling* property ($\Eu$ vs
   $50T_p$). Here it is measured end to end **on the query path**, in poisoned
   retrievals.
3. **The trade-off in one unit system.** B2's freshness cost and B1's exposure
   measured in the same run, so the freshness–exposure trade-off is *measured*
   rather than asserted.
4. **What containment cost clean retrieval.** Clean freshness delay $D_f$ and
   Eligible Recall@$k$ against exact ground truth, split at containment, so
   B2's zero exposure is visibly paid for and B4's truncation is visibly not
   free. Measured in the same run and the same units as the exposure.

The gap between $t_{\text{contain}}$ (control plane) and the last observed
poisoned retrieval (query path) is computed and reported, but it is **not** a
finding: in post-filter mode the control store is authoritative, so the gap is
bounded by the query interval by construction. It is reported as a censoring
bound. The in-index path's real propagation delay is §VII-D and §VII-E.

## 2. Definitions — the paper's, not new ones

$E_p$ keeps its §III definition, $E_p(x) = t_{\text{contain}}(x) -
t_{\text{first-poison-visible}}(x)$: a **system interval**. It is *not*
redefined as "last poisoned retrieval minus first poisoned retrieval" — that is
an **observed** interval, it is bounded by $E_p$, and it shrinks with query rate
for reasons that have nothing to do with the protocol. Both are reported, under
different names:

| Symbol | Definition | Source |
|---|---|---|
| $E_p$ | $t_{\text{contain}} - t_{\text{first-poison-visible}}$ | §III, unchanged |
| $D_f$ | $t_{\text{visible}} - t_{\text{arrive}}$ (clean items) | §III, unchanged |
| `poisoned_retrievals` | count of returned top-$k$ results containing $\ge 1$ poisoned item | new, observational |
| `PRR(t)` | poisoned_retrievals / queries issued, in a time bin | new, observational |
| `exposure_auc` | $\int$ PRR $dt$ over the run, in query-seconds | derived from PRR |
| `obs_window` | last $-$ first poisoned retrieval | new, observational, **censored** |

`obs_window` and the containment instant it implies are **interval-censored** by
the query interval, exactly as $\delta_{\text{hide}}$ is censored by the probe
interval in §VII-E. Report the conservative bound and the bracket width. Never
quote a containment time finer than the bracket.

## 3. Design

**One collection, created and loaded once, never dropped during measurement.**
This is the direct lesson of the S1 abort: `s1_cell()` built a fresh collection
per cell, twenty over the run, and the drift the sentinel caught between blocks
could not be separated from the server digesting that churn. Here the collection
lifecycle is constant across all four baselines by construction.

**Baselines** are the four already defined in `standalone_experiments.py`:
B1 immediate admission / B2 verify-before-visible / B3 async without deadline /
B4 the protocol. All four see an identical poison set, an identical query
stream, and an identical verifier.

**Verifier is deterministic and label-based.** A poisoned item is flagged iff it
carries the poison label, after a fixed delay. No detector error is injected:
the object of study is the lifecycle protocol, not detector quality, and this is
what the paper already claims ("detectors are deliberately pluggable and
non-SOTA", §IX). Detector false negatives are out of scope here and stay out of
the write-up.

**Conditions:** `{normal, heavy}` verifier backlog $\times$ B1–B4, plus a $T_p$
sweep for B4 alone over 3–4 points. Nothing larger. Replicates: 3 seeds, and the
baseline order is randomised per seed.

**Targeting — three query sets.** The corpus has topic structure, and:

| Set | Role | Status of its PRR |
|---|---|---|
| $Q_{\text{craft}}$ | the queries the poison was optimised against | **upper bound, worst case by construction** — same status as the oracle rank-coupled hidden set in §VI-C, under the same ban (matrix C5: never claim it bounds an adversary against *unseen* queries) |
| $Q_{\text{target}}$ | different queries, **same topic**, never used to craft poison | **the headline result** — a transfer measurement, not a construction |
| $Q_{\text{negative}}$ | unrelated topics | contamination-spread check, expected $\approx 0$ |

Measuring PRR *only* on unrelated queries would drive every baseline including
B1 to zero and measure nothing; measuring it *only* on the crafted set would
prove only that the construction worked. The transfer level on $Q_{\text{target}}$
is a property of the workload's embedding geometry, so the realised mean
cosines (craft-vs-target, poison-vs-target, poison-vs-negative,
corpus-vs-target) are **emitted with the results** rather than asserted, and the
topic spread is fixed in the config before the run. It is not tuned until the
transfer looks interesting.

**Between-baseline contrasts are valid on every set** because all baselines face
the identical poison, queries and verifier. "By construction" qualifies the
absolute level of $Q_{\text{craft}}$, never the B1-vs-B4 difference.

**$T_p$ points are anchored to the verifier, not chosen for looks:** below
normal verification latency (even clean content is hidden before it is vetted),
at it (the crossover), between normal and the heavy queue (slack when the
verifier keeps up, binding when it does not), and comfortably above normal
latency while still well below the heavy queue. Each point answers a stated
question about what happens when the deadline is faster than, equal to, or
slower than verification.

## 4. Evidence tier

Runs at **E1** (exact in-memory backend, exact ground truth) **and E2** (Milvus
Lite: real collection, real insert path, real top-$k$, both eligibility modes).
Reported side by side, as §VII-C/§VII-D already do for eligibility — E1 is not
"the simulation" and E2 is not "the result". Per matrix C9, Lite carries
functional validation and exact-trend reproduction, never production
performance, and no Lite latency is quoted as system performance.

E3 (standalone Docker) is **not** a submission prerequisite. If it happens it is
a narrow confirmatory slice — one normal backlog, one heavy, B1–B4, one
representative $T_p$ — not a new grid, and it must not gate the deadline.

## 5. Admissibility gates, fixed now

A run is admissible for the paper only if all of these hold:

1. **The attack works at all.** B1 poisoned-retrieval count > 0 on
   $Q_{\text{craft}}$. If the poison never enters the top-$k$ even on the
   queries it was built against, the run measures nothing and is discarded — it
   is not evidence that the protocol helped. A zero on $Q_{\text{target}}$ is
   *not* a gate failure: it is the finding that the attack did not transfer,
   and it is reported as such.
2. **The control plane did what the baseline says.** B2 emits zero provisional
   visibility; B1 emits no containment; B3 contains only after verification;
   B4 contains at $\le T_p + \delta_{\text{hide}}$. A baseline that violates its
   own definition invalidates the run, not the protocol.
3. **Invariants hold in-run.** I7 (each item PROVISIONAL at most once) and I3
   (no PROVISIONAL$\to$TRUSTED on expiry) are asserted continuously; a violation
   aborts.
4. **Seeds agree in direction.** All 5 seeds must order
   B1 $\ge$ B3 $\ge$ B4 $\ge$ B2 on poisoned retrievals in $Q_{\text{target}}$.
   If they disagree, report the disagreement and the range, not the mean.
5. **Censoring is stated.** Any containment-time figure carries its bracket.

## 6. Wording, fixed by outcome

**If the expected ordering holds (B1 $\gg$ B3 > B4 $\ge$ B2 on poisoned
retrievals, B2 worst on $D_f$):**
> Report counts and the ratio to B1, name B1 as the denominator, state the
> targeted-by-construction caveat in the same sentence, and attribute the B4
> result to the deadline rather than to detection. The backlog contrast is the
> headline: B3's exposure grew by $\cdot\times$ from normal to heavy backlog
> while B4's changed by $\cdot\times$.

**If B4 and B3 do not separate under normal backlog:** that is expected when
verification finishes well inside $T_p$ — say so plainly, and report the
separation where the protocol is supposed to produce it (heavy backlog). Do not
tune $T_p$ down until they separate; report the $T_p$ at which they cross.

**If B4 shows poisoned retrievals after $T_p + \delta_{\text{hide}}$:** that is
an invariant violation. Stop, fix the implementation, and re-run. It is not a
result and it is not reportable as a limitation.

**If the attack fails (gate 1):** no figure, no claim, and the paper ships with
$E_p$ still unmeasured. Do not weaken the poison until something appears.

## 7. What this does not license

- No claim about detector quality, ASR against a real detector, or robustness to
  adaptive attacks.
- No claim that measured PRR estimates production risk: the query set is
  targeted by construction and the corpus is synthetic.
- No latency, throughput, or tail figure from either tier. Lite is a
  single-process microbenchmark and the in-memory backend is a correctness
  harness — both already stated in §VI-A.
- No "reproducible" or "independent replication": three seeds inside one
  process on one machine.
