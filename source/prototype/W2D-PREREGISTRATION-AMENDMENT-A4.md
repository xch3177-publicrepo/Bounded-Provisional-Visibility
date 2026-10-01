# W2D Preregistration — Amendment A4

2026-07-28, after implementation-only tests on synthetic fixtures and before
the production W2D data freeze, protocol plan, calibration score, threshold,
test score, or protocol result exists.  No production detector outcome or
attack-landing outcome has been inspected.  A4 wins where the earlier W2D
documents leave the protocol replay or retrieval population ambiguous.

This amendment does not change D1, its threshold rule, either attack, the
detector-quality populations, or the fixed eight-second horizon.  It freezes
which already allocated items enter repeated protocol cells and how retrieval
queries are constructed, so those choices cannot be made after seeing a false
positive, false negative, or landing result.

## A4.1 — An immutable protocol plan precedes test scoring

After `W2D-DATA-FREEZE.json` and `W2D-labels.json` exist, but before the first
test item is scored, the harness writes exactly once:

```text
results/w2d/W2D-PROTOCOL-PLAN.json
```

The plan records and hashes its data freeze, safe JSON/NPZ, labels, plan code,
and this amendment.  It contains every runtime item key, role, query donor,
injection ordinal, cell id, and execution-order position.  It is strict JSON
and is opened with exclusive creation.  A second build is an error rather than
an overwrite.

No plan rule may read a D1 score, decision, service time, threshold, landing
result, or protocol result.  Opaque item keys are ordered only by SHA256 domain
separators fixed in the plan code.

## A4.2 — Runtime item assignment

Seeds remain 1 through 5.  Each seed receives, without replacement across
seeds:

- six recipe-test poison items and six natural-cover poison items;
- six injected clean test items, with the five per-seed
  `(ordinary, hard-negative)` counts `(4,2), (3,3), (4,2), (3,3), (4,2)`;
- twelve heavy-backlog filler items, disjoint from injected clean items, with
  the five per-seed counts `(ordinary, hard-negative)` equal to
  `(7,5), (7,5), (7,5), (7,5), (8,4)`.

The clean and filler assignments are shared by the two attacks.  The poison
assignment is attack-specific.  Topic selection is round-robin over the frozen
topic order, and candidates within a required topic/stratum/attack are ordered
by the plan's recorded SHA256 domain rule.  Recipe candidates alternate T3 and
T4 where both are available.  Failure to meet an exact count aborts planning;
there is no replacement after any outcome is known.

Normal backlog has no filler.  Heavy backlog uses the twelve frozen filler
items for that seed.  Within a cell the injection order is filler, poison,
clean, with the exact ordinal order recorded in the plan.  Every baseline and
arm reuses those same item keys and ordinals.

Repeated protocol use never enlarges the 512-group detector test denominator.

## A4.3 — Retrieval background and three query roles

A2 called the 768 clean reference items "detector reference only" but did not
freeze a separate retrieval background.  A4 explicitly assigns those same
immutable 768 clean items a second logical role: the fixed retrieval background
for W2D.  They remain excluded from detector-quality denominators and from
calibration/test candidates.  No corpus is selected or altered after a landing
result.

For each planned poison item the evaluator records three query roles:

1. `attack_associated`: the frozen 30-word source-prefix query.  For recipe
   poison this is the query used in construction; for natural-cover poison it
   is the source-prefix relevance query required by A2.5.
2. `heldout_same_topic`: the source-prefix query of a distinct, unselected
   test-attack item from the same topic.
3. `negative_other_topic`: the source-prefix query of a distinct, unselected
   test-attack item from the next topic in the frozen cyclic topic order.

Donors may be reused as queries but are never inserted into that cell.  A donor
may not be the poison item whose associated query it supplies.  All query
item keys and vectors are taken from the evaluator-only frozen label artifact
and are named in the protocol plan.  They are never visible to D1.

These names replace the overloaded craft/target/negative shorthand for new W2D
results.  Result artifacts may include compatibility aliases, but must retain
the three explicit roles above.

## A4.4 — Off-path landing population

Before interpreting detector decisions, the evaluator computes deterministic
B1 landing for every test poison item: all 128 recipe-test items and all 64
natural-cover items, with no resampling.  For each item, its modified passage
is added alone to the 768-item retrieval background and queried by its
`attack_associated` query.  Landing is whether that item is in exact cosine
top-5.  Ties use descending cosine followed by ascending opaque item key.

The landing artifact reports every item, plus numerator, denominator and
Wilson interval by attack.  Runtime summaries report both:

- `overall`, containing every preregistered protocol poison item; and
- `landed_only`, restricted by the separately frozen B1 landing record.

Failure to land is an outcome and never causes replacement.

## A4.5 — Arms, baselines, and service replay

B1 is one undefended off-path control per `(seed, attack, backlog)` and has no
oracle/detector arm distinction.  Shadow scoring cannot alter its state.

B2, B3 and B4 each have:

- a detector arm using D1's frozen promote/refuse decision; and
- an oracle arm using the construction label as the decision.

Both arms replay the same measured D1 service time for a given item.  Thus an
arm comparison changes decision correctness, while a baseline comparison
changes only admission/deadline policy.  This remains a
"promotion-path replay driven by real D1 outputs and measured service times",
not online detector execution.

The primary grid is:

```text
5 seeds × 2 attacks × 2 backlogs ×
  [1 B1 control + (3 verified baselines × 2 arms)] = 140 cells
```

The main deadline is `T_p = 1.0 s`; W2D does not add a data-dependent deadline
sweep.  Every cell uses the existing eight-second window, injection at one
second, exact in-memory cosine retrieval, top-5, and the existing query cadence.
The seed-specific execution order is ascending
`SHA256("W2D-cell-order|" + seed + "|" + cell_id)` and is frozen in the plan.

## A4.6 — Acceptance additions

32. The protocol plan exists and predates the test-score artifact; all of its
    input and code digests match, and rebuilding refuses to overwrite it.
33. Every B2/B3/B4 cell consumes the plan's exact item/role/ordinal mapping.
    Detector decisions and replayed service times match item-by-item across the
    three baselines; integrated latency may differ.
34. Oracle and detector arms replay identical item service times.  The oracle
    arm differs only by replacing the decision with construction truth.
35. The B1 landing artifact contains all 192 test poison items.  Natural-cover
    landing always has denominator 64, and no failed item is replaced.
36. The retrieval background is exactly the 768 frozen clean reference items.
    Query donors and all three query roles match the immutable plan.
37. Detector-quality summaries remain exactly 512 unique source groups.
    Protocol replay and landed-only summaries cannot change that denominator.
38. Formal scorer, threshold, metric, landing, runtime and manifest artifacts
    are exclusive-create, strict JSON, and carry sufficient SHA256 lineage to
    reject a mixed-code or mixed-data chain.
