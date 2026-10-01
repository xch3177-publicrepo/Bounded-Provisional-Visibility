"""
Golden-file regression + schema-validity tests for the analysis pipeline.
Run:  python3 test_golden.py   (self-contained)

Guards that: metrics reproduce deterministically; the versioned schema is well
formed; FLAT stays an exact tier (never mislabelled approximate/standalone); a
NaN metric is emitted as null (not silently 0); required fields and the evidence
enum are enforced.
"""

import json
import math
import os

import analysis

HERE = os.path.dirname(os.path.abspath(__file__))
PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"  ({detail})" if detail else ""))


def approx(a, b, tol=1e-6):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol


if __name__ == "__main__":
    print("Golden-file regression + schema validity\n")
    m = json.load(open(os.path.join(HERE, "config_manifest.json")))
    m = {**m, **m["golden"]}
    golden = json.load(open(os.path.join(HERE, "results", "golden.json")))
    fresh = analysis.collect(m, "inmemory")

    gid = {r["experiment_id"] for r in golden}
    fid = {r["experiment_id"] for r in fresh}
    check("same experiment set as golden", gid == fid, f"{len(gid)} experiments")

    gmap = {r["experiment_id"]: r["metrics"] for r in golden}
    ok, worst = True, ""
    for r in fresh:
        for k, v in r["metrics"].items():
            if not approx(v, gmap[r["experiment_id"]].get(k)):
                ok = False
                worst = f"{r['experiment_id']}.{k}: {v} vs {gmap[r['experiment_id']].get(k)}"
    check("metrics reproduce golden (deterministic)", ok, worst)

    check("evidence_level in enum",
          all(r["evidence_level"] in analysis.EVIDENCE_LEVELS for r in fresh))
    check("FLAT records stay exact/lite tiers (not standalone/approximate)",
          all(r["evidence_level"] in ("exact_in_memory", "milvus_lite_flat")
              for r in fresh if r["index_type"] == "FLAT"))

    rb1 = next(r for r in fresh if r["experiment_id"] == "elig/oracle_rank_coupled/of1")
    check("NaN metric emitted as null, not 0", rb1["metrics"]["cond_recall"] is None,
          f"cond_recall={rb1['metrics']['cond_recall']}")

    raised = False
    try:
        analysis.make_record("x", "inmemory", "FLAT", "exact_in_memory", {}, None)
    except ValueError:
        raised = True
    check("missing required field raises (not defaulted to 0)", raised)

    raised = False
    try:
        analysis.make_record("x", "inmemory", "FLAT", "bogus_tier", {}, {"a": 1})
    except ValueError:
        raised = True
    check("unknown evidence_level raises", raised)

    n = sum(1 for c in results if c)
    print(f"\n==== {n}/{len(results)} checks passed ====")
    raise SystemExit(0 if n == len(results) else 1)
