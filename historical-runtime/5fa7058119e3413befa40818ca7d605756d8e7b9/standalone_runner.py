"""
Standalone experiment runner for the frozen §7.5 contract (S1/S2/S5).

STATUS: **Interface-tested, deployment-UNVALIDATED.** The CLI, config parsing,
output-record schema, and --dry-run are tested here; Milvus standalone API
behaviour, gRPC concurrency, consistency propagation, HNSW effects, and real
P99/throughput are NOT validated until a Docker run. It reads the SAME
config_manifest.json and emits the SAME versioned schema as analysis.py, so a
Docker run is "execute + fill", not "redesign".

  python3 standalone_runner.py --exp S1 --dry-run
  python3 standalone_runner.py --exp S2 --dry-run --consistency Bounded
"""

import argparse
import json
import os

import analysis

HERE = os.path.dirname(os.path.abspath(__file__))

SPECS = {
    "S1": {"desc": "B1–B4 × concurrency → ingestion throughput, query P95/P99",
           "metrics": ["throughput", "query_p95", "query_p99"], "figure": "Fig B(a)"},
    "S2": {"desc": "in-index × consistency × verifier backlog → δ_hide P95/P99",
           "metrics": ["delta_hide_p95", "delta_hide_p99", "delta_sched", "delta_state", "delta_prop"],
           "figure": "Fig B(b)"},
    "S5": {"desc": "HNSW parameters → conditional Recall@k, query P99",
           "metrics": ["cond_recall", "query_p99"], "figure": "secondary"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=list(SPECS))
    ap.add_argument("--uri", default="http://localhost:19530")
    ap.add_argument("--consistency", default="Strong",
                    choices=["Strong", "Bounded", "Session", "Eventually"])
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--index-type", default="HNSW")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    m = json.load(open(os.path.join(HERE, "config_manifest.json")))
    spec = SPECS[a.exp]
    cfg = {"exp": a.exp, "uri": a.uri, "consistency": a.consistency,
           "concurrency": a.concurrency, "index_type": a.index_type,
           "k": m["k"], "seeds": m["seeds"]}
    print(f"[{a.exp}] {spec['desc']}  →  {spec['figure']}")

    if a.dry_run:
        placeholder = {k: None for k in spec["metrics"]}
        rec = analysis.make_record(f"standalone/{a.exp}", "milvus", a.index_type,
                                   "milvus_standalone", cfg, {**placeholder, "status": "planned"})
        print("  config:", json.dumps(cfg))
        print("  output-record skeleton:",
              json.dumps({k: rec[k] for k in ("experiment_id", "evidence_level",
                                              "index_type", "schema_version", "metrics")}))
        print("  DRY-RUN ok. STATUS: Interface-tested, deployment-UNVALIDATED "
              "(needs Docker + Milvus standalone with a matched client).")
        return

    raise SystemExit(
        "This runner only emits the §7.5 contract skeleton. The executable\n"
        "experiments live in standalone_experiments.py:\n\n"
        f"  ./.venv312/bin/python standalone_experiments.py --exp {a.exp} --uri {a.uri}\n"
        "  ./.venv312/bin/python standalone_experiments.py --exp all\n"
        "  ./.venv312/bin/python standalone_experiments.py --exp all --smoke   "
        "(Milvus Lite, no Docker; E2 code-path check only)")


if __name__ == "__main__":
    main()
