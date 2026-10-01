#!/usr/bin/env python3
"""Recompute published result summaries from the sanitized observations.

This verifies observations and arithmetic independently of the export hash.
The original private Git/file-byte provenance gates are not represented as
having passed on the transformed public files.
"""
import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path


def load(p):
    return json.loads(p.read_text(),parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def same(label,actual,expected):
    if actual!=expected:
        raise ValueError(label+" differs from frozen summary")
    print("PASS: "+label)


def main(root):
    proto=root/"source/prototype"
    sys.path.insert(0,str(proto))
    import w2d_metrics as metrics
    import w2d_analyze as analyze
    result_dir=proto/"results/w2d"
    labels=load(result_dir/"W2D-labels.json")
    scores=load(result_dir/"W2D-test-scores.json")
    controls=load(result_dir/"W2D-source-controls.json")
    landing=load(result_dir/"W2D-LANDING.json")
    same("D1 confusion matrix, 512-group denominators, attack/clean strata and latency quantiles",metrics.detector_quality_summary(scores,labels),load(result_dir/"W2D-detector-metrics.json")["metrics"])
    same("source-family controls",metrics.source_control_gate_summary(controls,labels),load(result_dir/"W2D-source-control-gate.json")["gate"])
    landing_by_key={str(r["item_key"]):r for r in landing["items"]}
    for stem in ("E1-INMEMORY","E2-MILVUS"):
        doc=load(result_dir/("W2D-"+stem+".json"))
        summary=load(result_dir/("W2D-"+stem[:2]+"-ANALYSIS.json"))
        cells=doc["cells"]
        same(stem+" detector section",analyze._detector_section(scores,labels),summary["detector_quality"])
        same(stem+" landing section",analyze._landing_section(landing,cells),summary["landing"])
        same(stem+" full exposure/availability/latency/retrieval runtime summaries",analyze._runtime_section(cells,landing_by_key=landing_by_key),summary["runtime"])
        same(stem+" detector-minus-oracle paired contrasts",analyze._paired_section(cells),summary["paired_detector_minus_oracle"])
    authority=load(root/"PUBLIC_AUTHORITY.json")
    recorded=load(root/authority["hnsw_summary"])
    commit=recorded["cross_run_provenance"]["git_commit"]
    verifier_file=root/"historical-runtime"/commit/"verify_w2d_e3.py"
    spec=importlib.util.spec_from_file_location("historical_e3_verifier",verifier_file)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    validated, sources=module.verify_files([root/p for p in authority["hnsw_accepted_runs"]])
    recomputed=module.build_summary(validated,source_files=sources)
    for field in ("accepted_run_numbers","aggregation_semantics","panel_a","panel_b","transfer_condition","interpretation_boundary"):
        same("E3 five-run "+field,recomputed[field],recorded[field])
    s2=load(root/authority['standalone_s2']['single_run_observation'])
    observed=[r['metrics'] for r in s2 if isinstance(r['metrics'].get('censor_width_p95'),(int,float))]
    def interval(key):
        values=[m[key]*1000 for m in observed]
        return [math.floor(min(values)),math.ceil(max(values))]
    same('S2 single-run outward-rounded observable P50 interval (ms)',interval('delta_hide_p50'),[288,332])
    same('S2 single-run outward-rounded observable P95 interval (ms)',interval('delta_hide_p95'),[380,431])
    if not all(r['metrics'].get('stale_probes_max')==0 for r in s2 if r['config']['consistency'] in ('Strong','Session')):
        raise ValueError('S2 Strong/Session stale-read observation differs')
    print('PASS: S2 Strong/Session no stale read in the retained single-run cells')
    if len(authority['standalone_s2']['replication_attempts'])!=5:
        raise ValueError('S2 incomplete replication attempt inventory')
    for p in authority['standalone_s2']['replication_attempts']:
        if len(load(root/p))!=10:
            raise ValueError('S2 replication has missing cells: '+p)
    print('PASS: all five inconclusive S2 replication attempts retained (not pooled)')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1")
    commands=[
        ["verify_w2.py","results/W2-inmemory.json","results/W2-milvus.json"],
        ["verify_w2r.py","results/W2R-inmemory.json","results/W2R-milvus.json"],
        ["verify_w2f.py","results/W2F-inmemory.json","results/W2-inmemory.json"],
    ]
    for cmd in commands:
        result=subprocess.run([sys.executable,*cmd],cwd=proto,env=env,text=True,capture_output=True)
        if result.returncode:
            print(result.stdout)
            print(result.stderr,file=sys.stderr)
            raise ValueError("Frozen verifier failed: "+" ".join(cmd))
        print("PASS: "+" ".join(cmd))
    print("PASS: numerical reanalysis complete; no long experiment was run")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    args=ap.parse_args()
    try:
        main(args.root)
    except Exception as exc:
        print("FAIL: "+str(exc),file=sys.stderr)
        raise SystemExit(1)
