# S2 Replication (5 runs) — Preregistration

Written 2026-07-28, BEFORE any replication run. Frozen on commit; amendments
must be committed as dated addenda, never edits.

## Purpose — and what this does NOT reopen

The paper's §7.5 standalone results rest on ONE full S2 run
(`results/standalone-S2.json`, frozen in `results/S2-FROZEN-2026-07-26.md`).
An external review flagged "single run" as a weakness. This replication adds
four more full runs so the paper can state its standalone numbers with a
5-run median and range instead of a single-run point.

Explicitly out of scope:
- The original S2 verdicts and numbers stay frozen. Nothing here reinterprets
  `S2-FROZEN-2026-07-26.md`. The original run becomes run 1 of 5 ONLY if its
  config hash matches the new runs (same code path, same `--s2-items 200`);
  otherwise the replication is reported as its own 5-run set and the original
  stays a separate, superseded-by-replication data point.
- No new metrics, no P99 (still needs 1000/cell), no new consistency levels.
- S1 stays closed ("no third attempt" rule stands). S5 is not touched here.

## Run protocol (operator: Docker machine)

```bash
cd ~/projects/paper-venue-scout && git pull        # must reach the commit adding this file
cd prototype
docker compose down && rm -rf volumes && docker compose up -d   # stale-etcd lesson from S1
docker compose ps                                   # wait (healthy), ~90 s on fresh init

for i in 1 2 3 4 5; do
  ./.venv312/bin/python standalone_experiments.py --exp S2 --full --s2-items 200 \
      --out results/standalone-S2-rep$i.json
  ./.venv312/bin/python validate_results.py results/standalone-S2-rep$i.json
  sleep 60                                          # fixed cooldown, all runs
done

git add results/standalone-S2-rep*.json && git commit -m "S2 replication: 5 full runs" && git push
```

Design choice, frozen: the five runs share ONE Milvus instance brought up
fresh at the start (no per-run wipe). Rationale: replication should measure
run-to-run variability in the environment the paper describes, and each run
carries its own control bracket, so environmental drift is detected per run
rather than silently averaged. If Milvus itself crashes mid-sequence: bring it
back up (`rm -rf volumes` again) and continue with the remaining runs; note
which runs preceded the crash in the commit message. Do NOT restart a
completed run because its verdict looked wrong.

A run whose validator prints errors is KEPT and the sequence continues; the
interpretation layer below handles it. Only an environment failure (Milvus
unreachable) pauses the sequence.

## Interpretation rules (frozen before data)

1. Every run is validated independently by `validate_results.py`. A run's
   admissibility verdict is recorded and never edited.
2. **Strong/Session categorical claim** ("no completed query still returned
   unvetted content once its hide committed"):
   - 5/5 runs FIRST_PROBE_HIDDEN → the categorical wording stands, upgraded
     from "(single run)" to "(5 runs)".
   - ≥1 run shows a stale read in Strong or Session → the categorical wording
     is DOWNGRADED to "in k of 5 runs" everywhere it appears, including the
     abstract. No exclusion of the offending run for any reason.
3. **Bounded/Eventually δ_hide P95**: report across-run median and full range.
   - All five P95 values in [200 ms, 600 ms) → the paper keeps "a few hundred
     milliseconds at P95" and upgrades the run count.
   - Any value outside → keep the conservative wording AND print the full
     range in §7.5. No dropping of outliers.
4. **Stale-window probe counts** (Bounded/Eventually): report median of
   per-run medians plus range. No pass/fail line; whatever it is, it is.
5. **Environment gate**: each run's control-drift classification (TIGHT/MILD/
   worse) is reported per run. Runs with MILD-or-worse drift are flagged, not
   dropped. Sensitivity check: if excluding flagged runs flips any wording
   decision above, report both readings in the paper's own words.
6. Five runs are five runs: median + range only. No confidence intervals, no
   significance tests, no "±" notation.
7. Paper edits allowed from this replication: run-count upgrades, range
   additions, and any downgrade forced by rule 2/3. Nothing else changes.

## Freeze artifacts

After the five runs land: a dated `S2-REPLICATION-FROZEN-<date>.md` recording
the five sha256s, the per-run verdicts, and the across-run numbers the paper
may cite. §7.5 is edited against that file only, not against memory.
