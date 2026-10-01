"""
Canonical result emitter + table regenerator (step-5 validatable pipeline).

Runs the frozen configs (config_manifest.json) through the existing gate2/gate4
measurement functions and emits ONE versioned JSON schema tagged with the
evidence tier, so the standalone deployment later emits the same fields without
changing any metric definition. Tables are regenerated from the JSON — no paper
number is hard-coded.

  python3 analysis.py                 # full config -> results/results.json + tables
  python3 analysis.py --golden        # small fixed config -> results/golden.json
  python3 analysis.py --backend milvus --uri /tmp/a.db   # Milvus Lite (FLAT)
"""

import argparse
import json
import math
import os
import random
import subprocess

import gate2
import gate4
from backend import InMemoryBackend

EVIDENCE_LEVELS = ("simulation", "exact_in_memory", "milvus_lite_flat", "milvus_standalone")
SCHEMA_VERSION = "1.0"
REQUIRED = ("schema_version", "run_id", "experiment_id", "backend", "index_type",
            "evidence_level", "git_commit", "metrics")
HERE = os.path.dirname(os.path.abspath(__file__))


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=HERE, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def git_dirty():
    """Whether the working tree had uncommitted changes when the run started.

    A result does NOT expire because the repository moved on: a run at a real
    commit with a clean tree stays reproducible by checking that commit out.
    What actually destroys reproducibility is a run against edits that were
    never committed, because the exact code that produced the numbers no longer
    exists anywhere. That is the distinction this field records, and it is the
    one the acceptance gate keys on -- not equality with the current HEAD."""
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"],
                                      cwd=HERE, stderr=subprocess.DEVNULL).decode()
        return bool(out.strip())
    except Exception:
        return None


def _clean(metrics):
    # JSON has no NaN; a NaN metric (e.g. conditional recall with no full-k query)
    # becomes null rather than being silently coerced to 0.
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in metrics.items()}


def make_record(experiment_id, backend, index_type, evidence_level, config, metrics, run_id="local"):
    if evidence_level not in EVIDENCE_LEVELS:
        raise ValueError(f"unknown evidence_level {evidence_level!r}")
    if not isinstance(metrics, dict):
        raise ValueError("missing required field: metrics (must be a dict, not defaulted)")
    rec = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "experiment_id": experiment_id,
           "backend": backend, "index_type": index_type, "evidence_level": evidence_level,
           "git_commit": git_commit(), "git_dirty": git_dirty(),
           "config": config, "metrics": _clean(metrics)}
    for f in REQUIRED:
        if rec.get(f) is None:
            raise ValueError(f"missing required field: {f}")
    return rec


def make_backend(backend, dim, uri):
    if backend == "inmemory":
        return InMemoryBackend()
    if backend == "milvus":
        from milvus_backend import MilvusBackend
        return MilvusBackend(dim=dim, uri=uri)
    raise SystemExit(f"unknown backend {backend}")


def collect(m, backend="inmemory", uri=None):
    evidence = "exact_in_memory" if backend == "inmemory" else "milvus_lite_flat"
    dim, k = m["dim"], m["k"]
    random.seed(0)
    vecs = gate2.gen_vecs(m["N_eligibility"], dim)
    random.seed(1)
    qvecs = [[random.gauss(0, 1) for _ in range(dim)] for _ in range(m["queries"])]
    recs = []

    for mode in m["hidden_modes"]:
        for of in m["over_fetch"]:
            accs = []
            for sd in m["seeds"]:
                hidden = gate2.hidden_mask(mode, vecs, m["N_eligibility"], m["hidden_ratio"], qvecs, seed=sd)
                elig = set(range(m["N_eligibility"])) - hidden
                b = make_backend(backend, dim, uri)
                for i in range(m["N_eligibility"]):
                    b.insert(i, vecs[i], visible=(i not in hidden))
                accs.append(gate2.evaluate(b, vecs, qvecs, elig, hidden, k, of))
            met = {kk: sum(a[kk] for a in accs) / len(accs) for kk in accs[0]}
            recs.append(make_record(f"elig/{mode}/of{of}", backend, m["index_type"], evidence,
                                    {"seeds": m["seeds"], "mode": mode, "over_fetch": of,
                                     "hidden_ratio": m["hidden_ratio"], "k": k}, met))

    items = gate4.gen(m["N_containment"], dim, m["n_sources"], m["n_batches"],
                      m["compromised_source"], m["n_poison_batches"], m["poison_frac"])
    for strat in m["revocation_granularities"]:
        rev = gate4.revoked_set(items, strat, m["compromised_source"])
        b = make_backend(backend, dim, uri)
        met = gate4.evaluate(b, items, rev)
        recs.append(make_record(f"contain/{strat}", backend, m["index_type"], evidence,
                                {"seed": 0, "granularity": strat, "n_revoked": len(rev)}, met))
    return recs


def tables(records):
    """Regenerate the paper tables FROM the records (no hard-coded numbers)."""
    elig = [r for r in records if r["experiment_id"].startswith("elig/")]
    con = [r for r in records if r["experiment_id"].startswith("contain/")]
    print(f"\nEvidence: {sorted({r['evidence_level'] for r in records})}  "
          f"commit={records[0]['git_commit']}")
    print("\n[Eligibility] mode/over-fetch -> under-fill, hM tail, recall")
    print(f"  {'experiment':>20} {'underfill':>9} {'hM95':>6} {'pf_rec':>7} {'cond_rec':>8} {'if_rec':>7}")
    for r in sorted(elig, key=lambda r: r["experiment_id"]):
        mt = r["metrics"]
        cr = "n/a" if mt["cond_recall"] is None else f"{mt['cond_recall']:.3f}"
        print(f"  {r['experiment_id']:>20} {mt['underfill']*100:8.1f}% {mt['hM95']*100:5.0f}% "
              f"{mt['pf_recall']:7.3f} {cr:>8} {mt['if_recall']:7.3f}")
    print("\n[Containment] granularity -> caught, collateral, residual(post-filter/in-index)")
    print(f"  {'experiment':>18} {'caught':>7} {'collat':>7} {'res_pf':>7} {'res_if':>7}")
    for r in sorted(con, key=lambda r: r["experiment_id"]):
        mt = r["metrics"]
        print(f"  {r['experiment_id']:>18} {mt['caught']*100:6.1f}% {mt['collateral']*100:6.1f}% "
              f"{mt['res_pf']*100:6.1f}% {mt['res_if']*100:6.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="inmemory", choices=["inmemory", "milvus"])
    ap.add_argument("--uri", default="http://localhost:19530")
    ap.add_argument("--golden", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    m = json.load(open(os.path.join(HERE, "config_manifest.json")))
    if a.golden:
        m = {**m, **m["golden"]}
    recs = collect(m, a.backend, a.uri)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = a.out or os.path.join(HERE, "results", "golden.json" if a.golden else "results.json")
    json.dump(recs, open(out, "w"), indent=2)
    print(f"wrote {len(recs)} records -> {out}")
    tables(recs)
