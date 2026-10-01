#!/usr/bin/env python3
"""Check the public artifact's files, JSON, numeric projections and lineage."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical(v):
    return json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def reject_constant(s):
    raise ValueError("Nonstandard JSON constant: " + s)


def strict_load(p):
    def pairs(v):
        out={}
        for k,x in v:
            if k in out:
                raise ValueError("Duplicate JSON key: " + k)
            out[k]=x
        return out
    return json.loads(p.read_text(), parse_constant=reject_constant, object_pairs_hook=pairs)


def numerical(v, excluded=(), prefix=""):
    if prefix in excluded:
        return []
    if isinstance(v,dict):
        if set(v)=={"_public_nonfinite"}:
            return [[prefix,"nonfinite",v["_public_nonfinite"]]]
        return [r for k in sorted(v) for r in numerical(v[k],excluded,prefix+"/"+str(k).replace("~","~0").replace("/","~1"))]
    if isinstance(v,list):
        return [r for i,z in enumerate(v) for r in numerical(z,excluded,prefix+"/"+str(i))]
    if v is None or isinstance(v,(int,float,bool)):
        if isinstance(v,float) and not math.isfinite(v):
            raise ValueError("Non-finite numerical output")
        return [[prefix,type(v).__name__,v]]
    return []


def verify(root):
    manifest=strict_load(root/"PUBLIC_PROVENANCE.json")
    entries={e["path"]:e for e in manifest["files"]}
    seen_json=0
    numeric_count=0
    for name,entry in entries.items():
        p=root/name
        if not p.is_file() or p.is_symlink():
            raise ValueError("Missing/non-regular artifact: "+name)
        if sha(p.read_bytes())!=entry["public_sha256"]:
            raise ValueError("Public SHA mismatch: "+name)
        if "numeric_projection_sha256" in entry:
            doc=strict_load(p)
            projection=numerical(doc,set(entry["excluded_input_pointers"]))
            if sha(canonical(projection))!=entry["numeric_projection_sha256"]:
                raise ValueError("Numeric projection mismatch: "+name)
            if len(projection)!=entry["numeric_leaf_count"]:
                raise ValueError("Numeric leaf count mismatch: "+name)
            seen_json+=1
            numeric_count+=len(projection)
    authority=strict_load(root/"PUBLIC_AUTHORITY.json")
    referenced=list(authority["legacy"].values())+list(authority["detector_protocol"].values())+[authority["hnsw_summary"]]+authority["hnsw_accepted_runs"]+[authority["standalone_s2"]["single_run_observation"]]+authority["standalone_s2"]["replication_attempts"]
    for name in referenced:
        if name not in entries:
            raise ValueError("Authority points outside export: "+name)
    legacy=strict_load(root/"source/prototype/results/AUTHORITATIVE.json")
    for filename,source_sha in legacy["sha256"].items():
        name="source/prototype/results/"+filename
        if entries[name]["source_sha256"]!=source_sha:
            raise ValueError("Historical authority/source link mismatch: "+filename)
    e3=strict_load(root/authority["hnsw_summary"])
    expected=["source/prototype/results/w2d-e3/"+Path(r["path"]).name for r in e3["source_files"]]
    if expected!=authority["hnsw_accepted_runs"]:
        raise ValueError("E3 authority does not preserve exact accepted attempts")
    for r,name in zip(e3["source_files"],expected):
        if entries[name]["source_sha256"]!=r["sha256"]:
            raise ValueError("E3 historical SHA/source link mismatch: "+name)
    forbidden=[p for folder in ('source','historical-runtime','data-provenance') for p in (root/folder).rglob('*') if p.name=='.git' or p.suffix in {'.npz','.pyc'} or '__pycache__' in p.parts]
    if forbidden:
        raise ValueError("Forbidden bundled history/cache: "+str(forbidden[0].relative_to(root)))
    sums=root/"SHA256SUMS"
    if sums.exists():
        listed=[]
        for line in sums.read_text().splitlines():
            expected,name=line.split("  ",1)
            if sha((root/name).read_bytes())!=expected:
                raise ValueError("SHA256SUMS mismatch: "+name)
            listed.append(name)
        actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p!=sums and '.git' not in p.parts}
        if set(listed)!=actual:
            raise ValueError("SHA256SUMS file inventory mismatch")
    return {"status":"PASS","exported_source_files":len(entries),"strict_json_files":seen_json,"preserved_numeric_leaves":numeric_count,"authority_references":len(referenced),"scope":"public artifact integrity and recorded source linkage; not a new experiment or a reconstruction of private Git history"}


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    args=ap.parse_args()
    try:
        print(json.dumps(verify(args.root),indent=2))
    except Exception as exc:
        print("FAIL: "+str(exc),file=sys.stderr)
        raise SystemExit(1)
