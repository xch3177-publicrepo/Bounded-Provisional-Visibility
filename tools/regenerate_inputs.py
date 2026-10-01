#!/usr/bin/env python3
"""Reacquire third-party W2D inputs outside the immutable public artifact.

Explicitly downloads 20 Newsgroups and pinned model weights. This command is
not part of the offline quick check and was not run during public packaging.
It compares reconstructed input bytes with the historical recorded hashes;
any discrepancy is a failure, not a reason to change the frozen manifest.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir',type=Path,required=True,help='new separate directory for external data and a fresh local Git provenance root')
    ap.add_argument('--download-and-build',action='store_true',help='explicitly allow fetching upstream corpus and model weights')
    args=ap.parse_args()
    if not args.download_and_build:
        ap.error('reacquisition requires --download-and-build')
    root=Path(__file__).resolve().parents[1]
    work=args.work_dir.resolve()
    if work.exists() or work==root or root in work.parents:
        raise SystemExit('Use a new directory outside the public artifact')
    proto=work/'prototype'
    proto.mkdir(parents=True)
    for p in (root/'source/prototype').iterdir():
        if p.is_file() and p.suffix in {'.py','.md','.txt','.json','.yml'}:
            shutil.copyfile(p,proto/p.name)
    (work/'.gitignore').write_text('generated/\n__pycache__/\n.venv/\n')
    def run(*cmd):
        subprocess.run(cmd,cwd=work,check=True)
    run('git','init','--quiet')
    run('git','add','prototype','.gitignore')
    run('git','-c','user.name=Artifact Reproduction','-c','user.email=artifact@invalid','commit','--quiet','-m','Public input-reconstruction source snapshot')
    generated=work/'generated'
    run(sys.executable,str(proto/'w2d_workload.py'),'--build','--data-dir',str(generated/'data/w2d'),'--results-dir',str(generated/'results/w2d'))
    provenance=json.loads((root/'PUBLIC_PROVENANCE.json').read_text())
    expected={e['source_path']:e['source_sha256'] for e in provenance['files']+provenance['excluded_source_files']}
    names=('data/w2d/W2D-detector-inputs.json','data/w2d/W2D-detector-inputs.npz','results/w2d/W2D-labels.json','results/w2d/W2D-labels.sha256')
    checks=[]
    for name in names:
        actual=hashlib.sha256((generated/name).read_bytes()).hexdigest()
        wanted=expected['prototype/'+name]
        checks.append({'relative_path':name,'expected_historical_sha256':wanted,'reconstructed_sha256':actual,'matches':actual==wanted})
    report={'scope':'external-input byte reconstruction only; no protocol replay or HNSW run performed','checks':checks,'pass':all(c['matches'] for c in checks)}
    (generated/'INPUT_RECONSTRUCTION_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if report['pass'] else 1


if __name__=='__main__':
    raise SystemExit(main())
