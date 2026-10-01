# W2D Preregistration — Amendment A7

2026-07-28, after D1 calibration, threshold selection, test scoring, detector
metrics, and source-family controls were frozen, but before any landing result,
runtime manifest, promotion-path cell, or protocol result existed.

The first production protocol plan was constructed from clean committed inputs
at commit `8717bbb4a0c55f9c1f65bd6bf81f980f5aa9f2ba`.  Its SHA256 is
`cfcf57938019339ddc6573f66a0d345b42328654bd560ba8fe1197c9a2b7c5c3`.
It contains the preregistered 140 cells, 60 runtime poison assignments, all 192
off-path landing records, and all 192 corresponding query embeddings.
It is archived as
`results/w2d/W2D-PROTOCOL-PLAN-preA7-landing-contract-failure.json`.

The first landing evaluation stopped before writing an output.  The evaluator
incorrectly required every one of the 192 off-path landing items to appear in
`planned_poison_queries`, even though that field intentionally contains only
the 60 poison items assigned to runtime cells.  Each landing record already
binds its own `query_item_key`, and `query_materials` already contains the
corresponding frozen embedding for all 192 items.  The failure therefore
exposes a consumer-contract bug, not a missing query, selection failure, or
detector outcome.

## Frozen repair

1. Landing evaluation obtains each query key from that item's
   `landing_plan.records[*].query_item_key`, requires
   `query_role == "attack_associated"`, and then reads the already frozen
   embedding from `query_materials`.
2. `planned_poison_queries` remains the 60-item runtime query-role assignment
   and is not expanded or reinterpreted.
3. The 192 landing items, order, attack labels, query embeddings, immutable
   768-item background, candidate rule, top-5 rule, and failure-retention rule
   do not change.
4. D1 code, calibration scores, threshold, test scores, detector metrics, and
   source-control results are immutable and are not rerun.
5. A regression fixture must include landing items absent from
   `planned_poison_queries`; all 192 must still be evaluated from their landing
   records.  Missing, duplicated, or inconsistent landing query keys fail
   before output.
6. This amendment is added to the formal runtime fingerprint.  The plan is
   regenerated only after the repair is committed and the worktree is clean.
   Its semantic fields and SHA256 are expected to remain identical; any change
   other than provenance forced by this amendment must be investigated before
   continuing.

Where this amendment differs from an earlier landing-consumer description, A7
controls.  It changes no estimand or outcome-dependent choice.
