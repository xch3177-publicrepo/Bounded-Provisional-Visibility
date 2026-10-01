#!/usr/bin/env python3
"""Exercise the verifiers' FAILURE paths.

    ./.venv312/bin/python test_verifier_failure_paths.py

Every gate in this project is run almost exclusively on data that passes it, so
the code that runs when a gate fires is the least exercised code here. That is
not hypothetical: the runtime-fingerprint check appended to `fails` several
lines before `fails` existed, so the one path that reports a moved codebase
would have raised NameError instead of failing, and nothing noticed until the
day it finally had something to report.

Each case below takes a real result file, breaks exactly one thing, and asserts
the verifier says so: a specific message, a non-zero exit, and no NameError,
AttributeError, KeyError or TypeError. A gate that crashes when it should fail
is worse than no gate, because a crash reads as "the tool is broken" and gets
rerun rather than believed.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv312", "bin", "python")
RES = os.path.join(HERE, "results")
checks = []


def check(name, cond, detail=""):
    checks.append((name, bool(cond)))
    if detail and not cond:
        print(f"      {detail}")


def run_verifier(script, args):
    r = subprocess.run([PY if os.path.exists(PY) else sys.executable,
                        os.path.join(HERE, script), *args],
                       capture_output=True, text=True, cwd=HERE)
    return r.returncode, r.stdout + r.stderr


CRASHES = ("NameError", "AttributeError", "KeyError", "TypeError",
           "IndexError", "Traceback")


def break_it(src, mutate, name):
    """Copy a result file, corrupt it, and hand back the path."""
    doc = json.loads(open(os.path.join(RES, src)).read())
    mutate(doc)
    fd, path = tempfile.mkstemp(prefix=f"broken-{name}-", suffix=".json",
                                dir=RES)
    with os.fdopen(fd, "w") as f:
        json.dump(doc, f)
    return path


def case(name, src, mutate, script, extra_args, expect_in_output):
    if not os.path.exists(os.path.join(RES, src)):
        checks.append((f"{name} (skipped: {src} absent)", True))
        return
    path = break_it(src, mutate, name)
    try:
        code, out = run_verifier(script, [path, *extra_args])
        crashed = [c for c in CRASHES if c in out]
        check(f"{name}: reports the problem", expect_in_output in out,
              f"looked for {expect_in_output!r} in:\n{out[-600:]}")
        check(f"{name}: exits non-zero", code != 0, f"exit {code}")
        check(f"{name}: fails rather than crashes", not crashed, str(crashed))
    finally:
        os.unlink(path)


# --- verify_w2: the episode identities --------------------------------------
def _break_identity(doc):
    doc["metrics"]["cells"][0]["Eu_started_n"] += 1     # started != completed + censored


case("started != completed + censored", "W2-inmemory.json", _break_identity,
     "verify_w2.py", [], "started = completed + censored")


def _break_null_rule(doc):
    c = doc["metrics"]["cells"][0]
    c["Eu_completed_n"] = 0
    c["Eu_completed_p50"] = 0.0                        # zero standing in for null


case("a zero standing in for null", "W2-inmemory.json", _break_null_rule,
     "verify_w2.py", [], "null, not zero")


def _break_gap_arithmetic(doc):
    c = doc["metrics"]["cells"][0]
    c["clean_expired_n"] = c["clean_gap_completed_n"] + \
        c["clean_gap_right_censored_n"] + 1


case("an expiry that is neither completed nor censored", "W2-inmemory.json",
     _break_gap_arithmetic, "verify_w2.py", [],
     "completed or a censored gap")


def _break_readmission(doc):
    doc["metrics"]["cells"][0]["clean_readmitted_n"] += 1


case("a readmission that closes no gap", "W2-inmemory.json", _break_readmission,
     "verify_w2.py", [], "closes a gap")


def _break_b4_bound(doc):
    for c in doc["metrics"]["cells"]:
        if c["baseline"] == "B4" and c["Ep_max"] is not None:
            c["Ep_max"] = c["Tp"] + 5.0
            break


case("B4 exposure past its deadline", "W2-inmemory.json", _break_b4_bound,
     "verify_w2.py", [], "exceeded its deadline")


def _break_nan(doc):
    doc["metrics"]["cells"][0]["Ep_p50"] = float("nan")


case("a NaN in a result file", "W2-inmemory.json", _break_nan,
     "verify_w2.py", [], "NaN")


# --- verify_w2r: the anti-tuning check --------------------------------------
def _break_knob(doc):
    doc["config"]["tp"] = 2.5                          # a retuned timing knob


case("a retuned timing parameter", "W2R-inmemory.json", _break_knob,
     "verify_w2r.py", [], "no timing or protocol parameter differs")


def _break_provenance(doc):
    doc["workload_provenance"]["model"] = "some-other-encoder"


case("a different embedding model", "W2R-inmemory.json", _break_provenance,
     "verify_w2r.py", [], "embedding model recorded")


if __name__ == "__main__":
    ok = sum(1 for _, c in checks if c)
    for n, c in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {n}")
    print(f"\n==== test_verifier_failure_paths: {ok}/{len(checks)} ====")
    sys.exit(0 if ok == len(checks) else 1)
