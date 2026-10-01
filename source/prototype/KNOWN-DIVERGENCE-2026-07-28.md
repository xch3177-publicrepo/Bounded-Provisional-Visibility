# Known divergence: one W2R cell, and why the gate stays strict

2026-07-28. `verify_accounting_only.py` failed comparing
`W2R-inmemory-preEpFix.json` (commit `a45f20b`) with `W2R-inmemory.json`
(commit `d2c22bd`). Recorded rather than tolerated.

## What moved

Five values across four cells, all B4 under heavy backlog. The first pass
through this only reported the first of them, which understated it; the full
list:

| cell | field | before -> after |
|---|---|---|
| Tp=0.3, seed 2 | poisoned retrievals, craft | 3 -> 2 |
| Tp=0.3, seed 2 | conditional rate | 0.0536 -> 0.0357 |
| Tp=0.3, seed 2 | cumulative displacement, craft | 4 -> 2 |
| Tp=0.3, seed 3 | cumulative displacement, target | 0 -> 1 |
| **Tp=1.0, seed 4** | cumulative displacement, craft | 9 -> 10 |

The last one is on the main grid, so this is not confined to the sweep.
Nothing moved in the synthetic workload, across all seventy cells.

Displacement can move while the retrieval count does not, because it counts
POSITIONS lost, not queries hit. All six poison items take their deadline at
about the same instant; a query landing inside that window sees one or several
of them and is one hit either way, but loses a different number of top-5 slots.
That is the Tp=1.0 case.

## Why it is not the edit

Structural, from the diff between the two producing commits, not from the size
of the change:

* `functional_slice.py` gained a `false_promote` set defaulting to empty. The
  two lines that read it are `it["id"] not in self.false_promote` and
  `it["id"] in self.false_promote`; against an empty frozenset those are
  constant True and constant False, so both call sites reduce to exactly the
  expressions they replaced.
* `false_promote_ids()` returns immediately when k is 0 and consumes no
  randomness. It would matter if it did: the cell-order shuffle draws from
  `random.Random(9000 + seed)`, and a stray draw from a shared generator would
  reorder cells. It uses its own instance, and only when k > 0.
* Everything else is inside `summarise()`, which runs after the cell has
  finished and can only read.

W2 and W2R run with k = 0 throughout, so the promotion switch is off in every
cell of both.

## What it is

The deadline boundary against the query schedule. At Tp = 0.3 s and 24 queries
per second split across three query sets, a poison item is visible for about
seven queries, of which two or three are crafted. Whether the deadline fires
just before or just after one of them decides a count of 2 versus 3. At
Tp = 1 s the exposure spans roughly three times as many queries and one
boundary crossing does not move the count; at Tp = 0.3 s it does.

This is a property of the measurement at its shortest deadline, not of the
protocol and not of the edit.

## Effect on the manuscript

Checked rather than assumed. Main grid, heavy backlog, both files: B1 49
(42--49), B2 0, B3 34 (29--34), B4 7 (6--7). T_p sweep medians [1, 3, 7, 14]
in both. Cumulative displacement medians 49 / 34 / 7 in both.

One aggregate did move: B4/heavy cumulative displacement RANGE, [7, 9] ->
[7, 10]. The manuscript quotes the median of that quantity and not its range,
so no printed number changes -- but the range moved, and writing "no aggregate
moved" would have been false. `check_numbers.py` re-derives every printed
number from these files and is the authority on that claim; it is re-run at the
end of this round rather than trusted from here.

## What was NOT done

`poisoned_retrievals_craft` was not reclassified as a wall-clock field. It is
the number the whole experiment exists to produce, and moving it into the
untestable set to make a gate green would remove the only automatic check that
a future edit did not change what queries retrieve. The gate keeps failing on
this pair, and this file is the reason it may be overridden -- once, for this
pair, for this cause.

What was added instead is a summary-level comparison: the aggregates the paper
actually prints must be identical across the pair. Those are stable where the
single-cell count is not, and they are what a reader depends on.

---

## Provenance caveat on this round's files

`git_commit` and `git_dirty` are read when the result document is written, at
the end of a run, not when it starts. Commits landed during this round's runs
(tests and verifiers), so a file's recorded commit can be later than the one
whose code produced it.

What is verifiable now, and what is not:

* The experiment code -- `poison_exposure.py`, `functional_slice.py`,
  `realtext_workload.py`, `backend.py`, `milvus_backend.py` -- was not touched
  after the chain started. `git log` over those paths shows nothing between the
  chain's start and its finish. Only `test_w2.py`, `verify_*.py` and markdown
  moved, and none of them is imported by a run.
* The recorded hash does not prove that by itself. It is the end-of-run HEAD.

Fix queued rather than applied: capturing the hash at run start is a change to
`poison_exposure.py`, and editing experiment code while runs are in flight is
the thing this note exists to avoid. It lands after this round freezes, along
with a `git_diff_sha256_at_start`.
