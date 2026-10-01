# W2D Preregistration — Amendment A6

2026-07-28, after the first production data-construction attempt failed before
writing any artifact, and before any production D1 score, calibration
threshold, detector result, landing result, or promotion-path result exists.
A6 changes only the per-topic allocation of ordinary-clean source groups. It
does not change source eligibility, duplicate handling, reference selection,
hard-negative rules or counts, poison construction, either attack, the global
calibration/test denominator, D1, the threshold rule, or any runtime cell.
Where A6 differs from an earlier preregistration document, A6 controls. The
repair is outcome-blind but data-adaptive: it uses the observed eligible
ordinary capacities, which depend on the frozen corpus and embedding-derived
hard-negative classifier, but it uses no D1 score, threshold, landing result,
or promotion-path outcome. It preserves poison transformations and counts;
because clean groups are removed first, the exact later poison/control source
IDs may differ from the infeasible allocation. The repaired mixture is not
claimed to be topic-balanced.

## Observed feasibility failure

Amendment A3 correctly required construction to fail rather than silently
change a denominator. With the pinned 20 Newsgroups training set, pinned
MiniLM revision, A3 eligibility, the first 96 reference groups per topic, and
the frozen hard-negative classifier, the eligible ordinary-clean capacities after
reference selection are:

| topic | ordinary capacity |
|---|---:|
| `rec.sport.baseball` | 58 |
| `sci.space` | 47 |
| `comp.graphics` | 130 |
| `talk.politics.mideast` | 26 |
| `rec.autos` | 59 |
| `sci.med` | 61 |
| `soc.religion.christian` | 35 |
| `misc.forsale` | 247 |

The earlier rule demanded 24 calibration plus 24 test ordinary groups from
every topic. Three topics cannot supply 48 such groups. The first attempted
build stopped at `sci.space` with `need 24 groups, found 23`; it produced no
data, label, score, threshold, or result artifact. The complete capacity table
above was then computed using only the already frozen corpus, model,
eligibility, reference, and hard-negative rules.

## Frozen outcome-blind quota repair

The global ordinary-clean totals remain exactly 192 calibration and 192 test
groups, without replacement. For each topic:

1. If ordinary capacity is at least 48, initialize its calibration and test
   quotas to 24 and 24.
2. Otherwise allocate all available ordinary groups as evenly as possible:
   calibration receives `ceil(capacity/2)` and test receives
   `floor(capacity/2)`.
3. For calibration and then test separately, distribute the remaining global
   quota one group at a time in the existing frozen `TOPICS` order. Skip a
   topic once its combined calibration-plus-test quota reaches its ordinary
   capacity. Repeat the topic order until the global split total reaches 192.
4. If either global total cannot be reached, construction fails before writing
   anything. No group classified as a hard negative may be relabelled
   ordinary, and no source group may be reused.

This yields the following fixed quotas:

| topic | calibration ordinary | test ordinary |
|---|---:|---:|
| `rec.sport.baseball` | 28 | 28 |
| `sci.space` | 24 | 23 |
| `comp.graphics` | 28 | 28 |
| `talk.politics.mideast` | 13 | 13 |
| `rec.autos` | 27 | 28 |
| `sci.med` | 27 | 28 |
| `soc.religion.christian` | 18 | 17 |
| `misc.forsale` | 27 | 27 |

Both columns sum to 192. All other per-topic counts remain exactly as frozen
in A2. The test-quality denominator therefore remains 512 unique source
groups, and calibration remains 448.

## Additional gates

32. The data freeze records the eight ordinary capacities, the repaired
    calibration/test quotas, the frozen topic order, both global totals, and
    the complete eligible-group hard-rule vector used to recompute capacity.
33. Recomputing the quota function from the recorded capacities produces the
    recorded quotas exactly; any insufficient global pool fails before write.
34. Every selected ordinary item has no hard-negative rule, every selected
    hard negative has at least one rule, each split has 192 ordinary groups,
    and no source group crosses a split or appears twice.
35. Production safe-input and label artifacts identify construction version
    `W2D-A6-v1`; the formal runtime fingerprint includes this amendment, and
    score artifacts bind the A6 safe-input bytes.
