# Standalone run sheet — §7.5 (E3) evidence

Target: CBDCom regular, **2026-07-31**. Total machine time ~50 min.
Run on the machine that has Docker. Nothing here runs on a laptop without it —
the runner refuses to emit E3 records from Milvus Lite by design.

Order is **S2 → S1 → S5** and it is deliberate: S2 unlocks the
deadline/visibility result the paper's thesis rests on, S1 the throughput/P99
result, S5 is a secondary ANN study. If the environment breaks partway, this
order loses the least.

---

## 0. Preconditions (do not skip)

```bash
cd ~/projects/paper-venue-scout
git pull                        # MUST include the S1/sentinel record (2026-07-26)
git log --oneline -1            # verify

cd prototype
# Matched client. Python 3.12 + pymilvus 2.4.x <-> milvusdb/milvus:v2.4.15.
# Python 3.13/3.14 cannot build pymilvus 2.4 (no grpcio wheel) and pip will
# silently give you 2.5.x, whose Lite has a function_score search bug.
python3.12 -m venv .venv312
./.venv312/bin/pip install "pymilvus>=2.4,<2.5" "setuptools<81"

docker compose up -d
docker compose ps               # wait for milvus-standalone (healthy), ~60-90s
```

**If `create_collection` fails with `node not match[expectedNodeID=..]`** —
preflight now catches this in ~13 s rather than at warm-up. It is a stale proxy
session in etcd, and neither `restart` nor `down`/`up` clears it: this compose
file **bind-mounts** `./volumes/etcd`, so etcd's data survives everything except
deleting that directory. Each restart registers a new node ID while the dead
registration stays, and the meta-cache broadcast hits the corpse.

```bash
docker compose down
rm -rf volumes                  # the bind mounts; `down -v` does NOT cover these
docker compose up -d
docker compose ps               # (healthy) takes ~90 s on a fresh init
```

This wipes all collections. Nothing of evidentiary value lives there — the
experiments create what they need and every result is already a committed JSON —
so the only casualty is `pv_slice`, which `functional_slice.py` recreates in
seconds. Expect the banner to say `0 unrelated collection(s)` afterwards.

Surgical alternative, if the collections must survive:
`docker compose stop standalone && docker compose exec etcd etcdctl del --prefix by-dev/meta/session/ && docker compose start standalone`.

The validator now checks that the records' `git_commit` matches HEAD. If you
run with a dirty or stale checkout it will refuse the results, so pull first.

---

## 1. Preflight (~4 min) — catches environment faults before the long runs

```bash
./.venv312/bin/python standalone_experiments.py --exp S2 --quick --out results/pre2-S2.json
./.venv312/bin/python standalone_experiments.py --exp S5 --quick --out results/pre2-S5.json
```

(`pre-S2.json` / `pre-S5.json` are the 07-26 preflights and are committed; the
runner refuses to overwrite an existing output, hence the new names.)

Expected in the banner: `Deployment: standalone  server_version=v2.4.15`.
If it says `lite`, the URI is wrong and the runner will refuse to continue.

Preflight results are throwaway. Do not push them and do not quote them.

---

## 2. S2 — DONE, do not re-run

S2 is closed. `results/S2-FROZEN-2026-07-26.md` is the only authority on what
it supports; `results/standalone-S2.json` holds the data. Re-run it only if the
paper needs a claim that record marks unavailable — not to improve a number and
not because S1 turned out awkward.

---

## 3. S1 — replacement run under band enforcement (~40–50 min)

**Restart Milvus first.** Then:

```bash
docker compose restart          # and wait for (healthy) again
./.venv312/bin/python standalone_experiments.py --exp S1 --full --s1-blocks 8
./.venv312/bin/python validate_results.py results/standalone-S1.json
```

Two earlier S1 runs are archived and neither is merged into this one:
`standalone-S1-cycle1only.json` (4 blocks) and
`standalone-S1-envgate-failed-8blocks.json` (8 blocks, commit `900edf4`). The
second one's **design checks all passed** — Williams order rebuilt from the
cells with all 12 carryover pairs in each cycle, position balance exact over 8
blocks, 4500/4500 queries per cell, no drops, boundary exclusion 0.007% and
balanced across every pair — and it failed on **one** thing: the between-block
sentinel gate, P95 spanning 2.2 → 6.4 ms (×2.95). So the fix belongs in the
environment, not in the analysis.

### What changed: the environment is now enforced, not just observed

`QUIESCE` in `standalone_experiments.py` (every constant fixed before this run
produced any data):

1. five idle sentinel reads before block 0 fix a band at
   `median ± max(3·MAD, 25%·median)` — measured now, not taken from the quiet
   blocks of a previous run, which would be choosing a threshold after seeing
   which readings were convenient;
2. a fixed 90 s cooldown after every block;
3. sentinel probes every 20 s until **two consecutive** readings land in band;
4. caps of 300 s **and** 15 probes per block — whichever comes first;
5. hitting either cap **aborts the whole run**. It never skips a block, never
   drops one, and never widens the band.

Every cooldown, probe, wait and pass/fail is written to the `S1/sentinel`
record, and `test_quiesce.py` covers the four behaviours (band from MAD,
recovery, the consecutive rule resetting on a single stray reading, and the
abort).

**What enforcement buys and what it does not.** Once the band is enforced the
block-start readings are in band by construction, so the acceptance gate over
them becomes a precondition check rather than evidence — the validator says so
rather than letting it look passed. The environment is then judged on the
**wait each block needed** and the **within-block rise**, both of which
enforcement does not touch. The previous run rose ×2.31 and ×2.15 *within*
blocks 4 and 7; if that persists, the run can pass the gate and still return a
wide band.

### The judgement, fixed before the data exists

Three gates, each answering a different question, none of them able to rescue
another.

**1. Admission (during the run).** The band above. Two consecutive in-band
readings or the run aborts. It answers *"is the box badly broken right now"* —
nothing more.

**2. Environment tier (after the run), from the WITHIN-block movement.**
Enforcement holds the block *starts* in band by construction, so those carry no
information. What it does not touch is how far the box moves between a block's
start and its end, measured in both directions — a block whose sentinel fell
6.4 → 2.6 ms did not recover nicely, it ran its four cells 2.5$\times$ apart.

| worst within-block movement | tier | consequence |
|---|---|---|
| $\le 1.18\times$ | quiet | eligible to interpret a point estimate as a protocol effect |
| $1.18$–$1.5\times$ | moderate | no catastrophic excursion, but not quiet enough for a few-percent cost |
| $> 1.5\times$ | excursion | feasibility facts stand; the protocol cost does not |

A block over $1.5\times$ **downgrades** the run; it does not void it and it
never removes the block. One noisy block out of eight is one noisy observation,
not grounds to discard the 4500/4500 completion facts alongside it.

**3. Paired outcome (after the run),** per pairing and *per metric* — a P99 band
crossing zero says nothing about a throughput ratio, so they are judged apart.
Unresolved if the band crosses its null, or the two cycles disagree in
direction. Resolved only if neither.

A point estimate may be **interpreted as a protocol effect** only when tier 2 is
`quiet` and tier 3 is resolved. Otherwise the point estimate is still printed and
still reportable — as a **description**, with the cost written as unresolved.

**The realised SD and the effect it implies are context, never a gate.** Both
that spread and the effect come from the same eight numbers, so promoting it to
a criterion would add a third rule that can contradict the two frozen above —
construct 0.80…0.92 with one block at 1.02 and the band says unresolved while
the t-statistic says resolved — with no tiebreak fixed in advance. It stays in
the output to explain *why* eight blocks struggle with a small effect.

**If the admission gate fails again: stop.** No third run. Keep the partial file,
label it, and ship the version with no performance-preservation claim. Note that
the gate firing evidences only that the environment control executed as
designed; it says nothing whatsoever about the protocol.

The goal is **not** to recover the 0.907 ratio: that is the observed median of a
drifted run, and three independent four-block groups have now given 0.882, 1.023
and 1.031. The goal is a band narrow enough to say something. The realised
block-to-block SD was 13.2% (7 d.f. — a planning figure, not a power result); if
stabilisation brings it near the 4% the earlier run showed, an 8-block band of
roughly ±5% would support "the ingestion cost is within a few percent", which is
worth more than resolving any particular effect size.

Fixed before the run and not to be revisited while it is in progress:

- **Exactly 8 blocks**, i.e. two full Williams cycles. Both cycles run to
  completion; the run does not stop early because an interim result looks
  decided, and no block is appended afterwards to firm one up.
- **No block is dropped**, whatever it shows.
- Report **cycle 1, cycle 2, and the pooled 8** as three layers. The
  experimental unit is the block, not the 36 000 individual queries.
- If the paired band still spans 1.0 at 8 blocks, that is the result: write
  "unresolved" and stop. It is not grounds for a ninth block.

### Decision rule, fixed 2026-07-26 while the run was still in flight

Written before the data exists, so that no reading of it can be chosen after
the fact.

**Name it correctly.** The eight blocks are **two complete Williams cycles
within one run** — same server instance, same process, same warm-up, same
sentinel fixture, same seeds. They add block-level experimental units and
balance position and carryover twice over. They are **not** independent
replication and the paper must not call them that; a run on a separately
started deployment is the only thing that earns that word.

**The two cycles are asymmetric evidence.** If cycle 2 disagrees with cycle 1,
that is strong evidence of within-run non-stationarity and the combined median
summarises nothing. If cycle 2 agrees, that is *weak* evidence — the two cycles
share every source of drift, so agreement is equally consistent with the same
bias acting twice. Agreement licenses reporting a combined figure with its
range; it does not license the word "reproducible".

**What eight blocks can and cannot resolve** (two-sided sign test on the paired
direction, and a paired power calculation using the 4.07% block-to-block SD of
cycle 1's four ratios — which has 3 degrees of freedom and is therefore itself
uncertain by roughly a factor of two):

| Outcome | $p$ | What may be written |
|---|---|---|
| 8/8 blocks same direction | 0.008 | a resolved direction, with the block-level range |
| 7/8 | 0.070 | **not** significant — direction plus range, "unresolved" |
| 6/8 | 0.289 | unresolved |

Power, same caveat: 8 blocks resolves an effect of roughly **4%**; 3% would need
~14 blocks and 2% would need ~32. So a true protocol overhead below about 3%
will come back unresolved **by design**. That is a stated limit of the
experiment, not a null result, and it is not a reason to add blocks — the
alternative on offer would be dozens of blocks, not one more cycle.

**Name the denominator.** "Retained $Y$\% of baseline throughput" is ambiguous
while four baselines exist. The denominator is **B1** (immediate admission, no
verification) — the throughput ceiling, hence the cost of the protocol against
doing nothing. B4-vs-B3 (async visible, no deadline) isolates the cost of the
deadline specifically and is reported separately if it resolves.

**Zero dropped queries is not an abstract-grade result.** `queries_dropped`
counts dispatches abandoned because 256 requests were already in flight, so 0
does say the server absorbed the offered 100 q/s. But all four baselines showed
0, including unprotected B1, so it distinguishes nothing about the protocol. It
belongs in the results as evidence that the open loop held and the latency
percentiles are therefore not queueing artifacts.

**This must run as one whole run.** Throughput and P99 have to come from the
same paired blocks; quoting an earlier run's throughput beside a new run's P99
would source one performance conclusion from two different sets of conditions.
Raw latency samples are not recoverable offline, so a partial run cannot be
repaired — it can only be redone.

Cycle 1's direction (1/4 blocks favouring lower B4, ratio median 1.031, range
0.970–1.066) is **pilot evidence only** and is superseded by this run.

---

## 4. S5 — ANN approximation effects (~20-30 min, secondary)

```bash
./.venv312/bin/python standalone_experiments.py --exp S5 --full
./.venv312/bin/python validate_results.py results/standalone-S5.json
```

Observing **no** recall loss is a valid outcome and simply leaves P1 pending.
Do not tune `ef` down to manufacture an error.

---

## 5. Re-runs

Output files are per-experiment and the runner refuses to overwrite an
existing one, so a re-run cannot silently destroy prior evidence:

```bash
mv results/standalone-S2.json results/standalone-S2-attempt1.json
./.venv312/bin/python standalone_experiments.py --exp S2 --full --s2-items 1000
```

Never delete a superseded run. Rename it and note why, the way
`run2-2026-07-25-superseded.json` is kept.

---

## 6. Hand back

```bash
git add results/standalone-*.json
git commit -m "standalone S2/S1/S5 results"
git push
docker compose down             # add -v and rm -rf volumes to wipe
```

Then say which of the three validators printed `VERDICT: admissible`. Nothing
reaches §7.5 that has not passed its validator, and each claim is written only
at the strength the validator permits.
