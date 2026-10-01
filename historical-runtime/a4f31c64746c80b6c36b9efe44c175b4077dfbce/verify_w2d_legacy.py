#!/usr/bin/env python3
"""Fail-loud legacy-oracle regression gate for W2D Amendment A5.

The gate compares only genuinely deterministic cross-execution evidence.  It
also records, but does not grade, scheduling-sensitive differences between two
full executions of the same code.  Every candidate run is independently
checked for the legacy state-machine and accounting invariants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from w2d_metrics import file_sha256, strict_json_dump, strict_json_load


HERE = Path(__file__).resolve().parent
SCHEMA_VERSION = "W2D-legacy-regression-v2"
HEX = frozenset("0123456789abcdef")
FROZEN_CANDIDATE_COMMIT = "5fa7058"
FROZEN_SAME_CODE_DIFFERENCE_SHA256 = (
    "88b38fbac32b68c1ae749f33560d8a02b5eaf6f409bf76948684c5273f324d22"
)

FROZEN_SOURCE_SHA256 = {
    "authority_manifest": (
        "fdca94b312900f01024a94491b4d100328eced2d2681566ccd956356111422db"
    ),
    "authoritative_w2": (
        "1f5efd3390039bf930ceffac7464d06e0a178ca9e46bb0a3b7c961abfff7a24d"
    ),
    "authoritative_w2r": (
        "58b9a8dc1fe3124621733e97c2e3ad0e253f9bdce513c70d8139f432b0d81e8d"
    ),
    "candidate_w2": (
        "2122e80d66891276ddcb31369c2e71ae4de5fc6d9bf2af89306f6ef666f7fca8"
    ),
    "candidate_w2r": (
        "be980da694e9d09a59d8e25d162bc85828465906a112fe70fefe5c6b54de8c75"
    ),
    "same_code_repeat": (
        "754b613a7abce666c50b3f79d7ff18cdd416cb99c00319c61d76bf58729d46fc"
    ),
}

LEGACY_RUNTIME_FILES = (
    "poison_exposure.py",
    "functional_slice.py",
    "backend.py",
    "realtext_workload.py",
    "analysis.py",
    "milvus_backend.py",
)

LEGACY_GATE_FILES = (
    "verify_w2d_legacy.py",
    "test_verify_w2d_legacy.py",
    "verify_w2.py",
    "verify_w2r.py",
    "test_w2d_runtime.py",
    "test_w2.py",
    "test_invariants.py",
    "slice.py",
    "gate2.py",
    "gate4.py",
    "w2d_metrics.py",
)

LEGACY_GATE_COMMANDS = (
    ("verify_w2.py", "candidate_w2"),
    ("verify_w2.py", "candidate_w2r"),
    ("verify_w2r.py", "candidate_w2r"),
    ("verify_w2.py", "same_code_repeat"),
    ("test_w2d_runtime.py", None),
    ("test_w2.py", None),
    ("test_invariants.py", None),
    ("test_verify_w2d_legacy.py", None),
)

B1_FROZEN_FIELDS = (
    "poisoned_retrievals_craft",
    "poisoned_retrievals_targeted",
    "poisoned_retrievals_negative",
    "poison_never_visible_n",
    "attack_coverage_craft",
    "attack_coverage_target",
    "cond_craft_prr",
    "cond_craft_qualified_q",
    "D_H_craft_positions",
    "D_H_target_positions",
    "foreign_candidates",
    "candidate_queries",
    "underfill_queries",
    "residual_at_start",
    "residual_rows",
)

SCHEDULING_DIAGNOSTIC_FIELDS = (
    "poisoned_retrievals_craft",
    "poisoned_retrievals_targeted",
    "cond_craft_prr",
    "D_H_craft_positions",
    "D_H_target_positions",
    "Ep_p50",
    "Ep_max",
    "Eu_completed_p50",
    "Eu_completed_max",
    "Df_clean_p50",
    "Df_trusted_p50",
    "clean_gap_p50",
)


class LegacyGateError(ValueError):
    pass


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LegacyGateError(f"{name} must be an object")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in HEX for character in value)
    )


def canonical_strict_json_sha256(value: Any) -> str:
    """Hash one strict-JSON value with a single canonical serialization."""

    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LegacyGateError(
            f"same-code difference projection is not strict JSON: {exc}"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def validate_frozen_same_code_differences(value: Any) -> str:
    """Require the exact, preregistered same-code counterexample projection."""

    digest = canonical_strict_json_sha256(value)
    if digest != FROZEN_SAME_CODE_DIFFERENCE_SHA256:
        raise LegacyGateError(
            "same-code difference projection differs from Amendment A5"
        )
    return digest


def legacy_runtime_code_sha256(root: Path = HERE) -> str:
    digest = hashlib.sha256()
    for filename in sorted(LEGACY_RUNTIME_FILES):
        path = root / filename
        if not path.is_file():
            raise LegacyGateError(f"legacy runtime file is missing: {filename}")
        digest.update(filename.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def legacy_gate_code_sha256(root: Path = HERE) -> dict[str, str]:
    return {
        filename: file_sha256(str(root / filename))
        for filename in LEGACY_GATE_FILES
    }


def _config_sha256(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _frozen_top_level_projection(document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: document.get(field)
        for field in (
            "schema_version",
            "experiment_id",
            "backend",
            "index_type",
            "evidence_level",
            "workload",
            "config",
            "config_sha256",
            "dataset_sha256",
            "workload_geometry",
            "workload_provenance",
        )
    }


def _expected_cell_keys(config: Mapping[str, Any]) -> set[tuple[Any, ...]]:
    seeds = config.get("seeds")
    backlog = config.get("backlog")
    tp_sweep = config.get("tp_sweep")
    main_tp = config.get("tp")
    if (
        seeds != [1, 2, 3, 4, 5]
        or not isinstance(backlog, Mapping)
        or set(backlog) != {"normal", "heavy"}
        or not isinstance(tp_sweep, list)
        or main_tp not in tp_sweep
    ):
        raise LegacyGateError("legacy config does not define the frozen grid")
    result = set()
    for seed in seeds:
        for backlog_name in sorted(backlog):
            for baseline in ("B1", "B2", "B3", "B4"):
                result.add((baseline, backlog_name, main_tp, seed, 0))
            for deadline in tp_sweep:
                if deadline != main_tp:
                    result.add(("B4", backlog_name, deadline, seed, 0))
    return result


def _cells(document: Mapping[str, Any]) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    metrics = _mapping(document.get("metrics"), "metrics")
    raw = metrics.get("cells")
    if not isinstance(raw, list):
        raise LegacyGateError("metrics.cells must be a list")
    result: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for index, value in enumerate(raw):
        cell = _mapping(value, f"cell {index}")
        key = (
            cell.get("baseline"),
            cell.get("backlog"),
            cell.get("Tp"),
            cell.get("seed"),
            cell.get("false_promoted_k", 0),
        )
        if key in result:
            raise LegacyGateError(f"duplicate cell key {key}")
        result[key] = cell
    return result


def _check(condition: bool, name: str, checks: list[str]) -> None:
    if not condition:
        raise LegacyGateError(name)
    checks.append(name)


def _b1_equality(
    authoritative: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    tag: str,
    checks: list[str],
) -> None:
    old = _cells(authoritative)
    new = _cells(candidate)
    old_b1 = {key: value for key, value in old.items() if key[0] == "B1"}
    new_b1 = {key: value for key, value in new.items() if key[0] == "B1"}
    _check(set(old_b1) == set(new_b1), f"{tag}: B1 cell set changed", checks)
    for key in sorted(old_b1):
        before, after = old_b1[key], new_b1[key]
        for field in B1_FROZEN_FIELDS:
            _check(
                field in before and field in after,
                f"{tag}: B1 field {field} is absent",
                checks,
            )
            _check(
                before[field] == after[field],
                f"{tag}: B1 {key} field {field} moved",
                checks,
            )
        for role in ("craft", "target", "negative"):
            before_vector = before["sets"][role]["per_query"]
            after_vector = after["sets"][role]["per_query"]
            _check(
                before_vector == after_vector,
                f"{tag}: B1 {key}/{role} per-query vector moved",
                checks,
            )


def _candidate_invariants(
    document: Mapping[str, Any], *, tag: str, checks: list[str]
) -> None:
    cells = _cells(document)
    config = _mapping(document.get("config"), f"{tag}.config")
    seeds = config.get("seeds")
    backlog = config.get("backlog")
    tp_sweep = config.get("tp_sweep")
    if not isinstance(seeds, list) or not isinstance(backlog, Mapping):
        raise LegacyGateError(f"{tag}: malformed run config")
    _check(
        document.get("config_sha256") == _config_sha256(config),
        f"{tag}: config digest is stale",
        checks,
    )
    expected_keys = _expected_cell_keys(config)
    _check(
        set(cells) == expected_keys,
        f"{tag}: exact 70-cell grid changed",
        checks,
    )
    _check(len(cells) == 70, f"{tag}: not a full 70-cell run", checks)
    _check(document.get("smoke") is False, f"{tag}: smoke run", checks)
    _check(document.get("git_dirty") is False, f"{tag}: dirty run", checks)
    _check(document.get("admissible") is True, f"{tag}: runner rejected run", checks)
    schemas = {tuple(sorted(cell)) for cell in cells.values()}
    set_schemas = {
        tuple(sorted(_mapping(value, f"{tag}.sets").keys()))
        for cell in cells.values()
        for value in [_mapping(cell.get("sets"), f"{tag}.sets")]
    }
    _check(len(schemas) == 1, f"{tag}: cell schema varies by cell", checks)
    _check(
        set_schemas == {("craft", "negative", "target")},
        f"{tag}: query-role schema changed",
        checks,
    )
    expected_query_count = int(config.get("qps", 0) * config.get("dur", 0)) - 1
    _check(
        expected_query_count > 0
        and all(
            cell.get("candidate_queries") == expected_query_count
            for cell in cells.values()
        ),
        f"{tag}: fixed query-event denominator changed",
        checks,
    )
    _check(
        all(
            cell.get("false_promoted_k") == 0
            and cell.get("false_promoted_ids") == []
            and cell.get("false_promoted_offsets") == []
            for cell in cells.values()
        ),
        f"{tag}: legacy oracle false-promotion setting changed",
        checks,
    )
    _check(
        all(cell.get("foreign_candidates") == 0 for cell in cells.values()),
        f"{tag}: foreign candidate entered a query",
        checks,
    )
    _check(
        all(cell.get("underfill_queries") == 0 for cell in cells.values()),
        f"{tag}: an under-filled query occurred",
        checks,
    )
    _check(
        all(
            cell.get("residual_at_start") == 0
            and cell.get("residual_rows") == 0
            for cell in cells.values()
        ),
        f"{tag}: collection isolation failed",
        checks,
    )
    _check(
        all(
            cell.get("live_tasks") == 0
            and cell.get("live_timers") == 0
            and not cell.get("bg_errors")
            for cell in cells.values()
        ),
        f"{tag}: control-plane state survived a cell",
        checks,
    )
    n_poison = config.get("n_poison")
    _check(
        isinstance(n_poison, int)
        and not isinstance(n_poison, bool)
        and n_poison == 6,
        f"{tag}: poison denominator changed",
        checks,
    )
    _check(
        all(
            cell.get("poison_never_visible_n") == n_poison
            and cell.get("Eu_started_n") == 0
            and cell.get("Ep_n") == 0
            for cell in cells.values()
            if cell.get("baseline") == "B2"
        ),
        f"{tag}: B2 exposed unvetted poison",
        checks,
    )
    _check(
        all(
            cell.get("Ep_right_censored_n") == n_poison
            and cell.get("Eu_started_n") == n_poison
            for cell in cells.values()
            if cell.get("baseline") == "B1"
        ),
        f"{tag}: B1 synthesized containment",
        checks,
    )
    b4_bad = [
        key
        for key, cell in cells.items()
        if cell.get("baseline") == "B4"
        and (
            cell.get("Ep_n") != n_poison
            or not isinstance(cell.get("Ep_max"), (int, float))
            or isinstance(cell.get("Ep_max"), bool)
            or not math.isfinite(float(cell["Ep_max"]))
            or float(cell["Ep_max"]) < 0
            or float(cell["Ep_max"]) > float(cell["Tp"]) + 0.25
        )
    ]
    _check(
        not b4_bad,
        f"{tag}: B4 exposure is missing or exceeded deadline slack",
        checks,
    )
    _check(
        all(
            cell.get("sweep_attempts") == 1
            and cell.get("delete_errors") == 0
            and cell.get("inv_I3_no_expiry_promotion") is True
            and cell.get("inv_I7_prov_once") is True
            and cell.get("poison_readmitted_after_hide_n") == 0
            for cell in cells.values()
        ),
        f"{tag}: deterministic state-machine projection changed",
        checks,
    )
    for cell in cells.values():
        _check(
            cell.get("Eu_started_n")
            == cell.get("Eu_completed_n") + cell.get("Eu_right_censored_n"),
            f"{tag}: exposure censoring identity failed",
            checks,
        )
        _check(
            cell.get("clean_expired_n")
            == cell.get("clean_gap_completed_n")
            + cell.get("clean_gap_right_censored_n"),
            f"{tag}: clean gap identity failed",
            checks,
        )


def _same_code_differences(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> list[dict[str, Any]]:
    left, right = _cells(first), _cells(second)
    if set(left) != set(right):
        raise LegacyGateError("same-code diagnostic cell sets differ")
    changes: list[dict[str, Any]] = []
    for key in sorted(left):
        for field in sorted(set(left[key]) | set(right[key])):
            first_value = left[key].get(field)
            second_value = right[key].get(field)
            if first_value == second_value:
                continue
            record: dict[str, Any] = {
                "cell": list(key),
                "field": field,
                "first_type": type(first_value).__name__,
                "second_type": type(second_value).__name__,
            }
            if isinstance(first_value, (type(None), bool, int, float, str)) and (
                isinstance(second_value, (type(None), bool, int, float, str))
            ):
                record.update(first=first_value, second=second_value)
            else:
                record.update(
                    first_sha256=hashlib.sha256(
                        json.dumps(
                            first_value, sort_keys=True, allow_nan=False
                        ).encode("utf-8")
                    ).hexdigest(),
                    second_sha256=hashlib.sha256(
                        json.dumps(
                            second_value, sort_keys=True, allow_nan=False
                        ).encode("utf-8")
                    ).hexdigest(),
                )
            changes.append(record)
    return changes


def evaluate(
    *,
    authority_manifest: Mapping[str, Any],
    authoritative_w2: Mapping[str, Any],
    authoritative_w2r: Mapping[str, Any],
    candidate_w2: Mapping[str, Any],
    candidate_w2r: Mapping[str, Any],
    same_code_repeat: Mapping[str, Any],
    source_sha256: Mapping[str, str],
    expected_runtime_code_sha256: str,
    enforce_frozen_same_code_differences: bool = True,
) -> dict[str, Any]:
    checks: list[str] = []
    _check(
        set(source_sha256) == set(FROZEN_SOURCE_SHA256)
        and all(_is_sha256(value) for value in source_sha256.values())
        and len(set(source_sha256.values())) == len(source_sha256),
        "legacy source SHA256 roles are missing, malformed, or aliased",
        checks,
    )
    authority_files = _mapping(
        authority_manifest.get("files"), "authority_manifest.files"
    )
    authority_hashes = _mapping(
        authority_manifest.get("sha256"), "authority_manifest.sha256"
    )
    _check(
        authority_files.get("w2_inmemory") == "W2-inmemory.json"
        and authority_files.get("w2r_inmemory") == "W2R-inmemory.json",
        "authority manifest selected different in-memory runs",
        checks,
    )
    _check(
        authority_hashes.get("W2-inmemory.json")
        == source_sha256["authoritative_w2"]
        and authority_hashes.get("W2R-inmemory.json")
        == source_sha256["authoritative_w2r"],
        "authority manifest hashes do not bind W2/W2R sources",
        checks,
    )
    for old, new, tag in (
        (authoritative_w2, candidate_w2, "W2"),
        (authoritative_w2r, candidate_w2r, "W2R"),
    ):
        _check(
            _frozen_top_level_projection(old)
            == _frozen_top_level_projection(new),
            f"{tag}: frozen identity/config/data/geometry changed",
            checks,
        )
        _b1_equality(old, new, tag=tag, checks=checks)
        _candidate_invariants(new, tag=tag, checks=checks)
    _candidate_invariants(
        same_code_repeat, tag="W2 same-code repeat", checks=checks
    )
    _check(
        _frozen_top_level_projection(candidate_w2)
        == _frozen_top_level_projection(same_code_repeat),
        "W2 same-code repeat changed identity/config/data/geometry",
        checks,
    )
    candidate_schemas = {
        tuple(sorted(next(iter(_cells(document).values()))))
        for document in (candidate_w2, candidate_w2r, same_code_repeat)
    }
    _check(
        len(candidate_schemas) == 1,
        "candidate runs do not share one frozen cell schema",
        checks,
    )
    git_commit = candidate_w2.get("git_commit")
    _check(
        isinstance(git_commit, str)
        and git_commit == FROZEN_CANDIDATE_COMMIT
        == candidate_w2r.get("git_commit")
        == same_code_repeat.get("git_commit"),
        "candidate runs do not share the frozen code commit",
        checks,
    )
    runtime_code_sha256 = candidate_w2.get("runtime_code_sha256")
    _check(
        _is_sha256(runtime_code_sha256)
        and runtime_code_sha256 == expected_runtime_code_sha256
        and runtime_code_sha256
        == candidate_w2r.get("runtime_code_sha256")
        == same_code_repeat.get("runtime_code_sha256"),
        "candidate runs do not share one runtime fingerprint",
        checks,
    )
    _check(
        candidate_w2r.get("workload") == "realtext"
        and candidate_w2r.get("workload_provenance", {}).get("model")
        == "all-MiniLM-L6-v2",
        "W2R real-text/model identity changed",
        checks,
    )
    _b1_equality(
        candidate_w2,
        same_code_repeat,
        tag="W2 same-code repeat",
        checks=checks,
    )
    scheduling_differences = _same_code_differences(
        candidate_w2, same_code_repeat
    )
    _check(
        bool(scheduling_differences),
        "same-code counterexample contains no recorded scheduling difference",
        checks,
    )
    if enforce_frozen_same_code_differences:
        validate_frozen_same_code_differences(scheduling_differences)
        checks.append(
            "same-code counterexample has the exact frozen difference projection"
        )
    return {
        "passed": True,
        "checks": checks,
        "same_code_scheduling_differences": scheduling_differences,
        "literal_A2_6_cross_execution_equality": "INCONCLUSIVE",
        "reason": (
            "same committed runtime changes query-boundary observations "
            "between complete executions; Amendment A5 replaces that invalid "
            "classifier without changing W2D data, detector, or outcomes"
        ),
    }


def _run_gate(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=HERE,
        text=True,
        capture_output=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "passed": completed.returncode == 0,
    }


def rerun_legacy_gates(
    source_paths: Mapping[str, str | Path],
) -> list[dict[str, Any]]:
    """Execute the exact eight legacy gates with the current interpreter."""

    required = {
        role for _script, role in LEGACY_GATE_COMMANDS if role is not None
    }
    missing = required - set(source_paths)
    if missing:
        raise LegacyGateError(
            f"legacy gate source paths are missing roles {sorted(missing)}"
        )
    executable = str(Path(sys.executable).resolve())
    executions: list[dict[str, Any]] = []
    for script, role in LEGACY_GATE_COMMANDS:
        command = [executable, script]
        if role is not None:
            command.append(str(Path(source_paths[role]).resolve()))
        executions.append(_run_gate(command))
    if not all(record["passed"] for record in executions):
        failed = [
            LEGACY_GATE_COMMANDS[index][0]
            for index, record in enumerate(executions)
            if not record["passed"]
        ]
        raise LegacyGateError(
            f"legacy/runtime gate rerun failed: {failed}"
        )
    return executions


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-manifest", required=True)
    parser.add_argument("--authoritative-w2", required=True)
    parser.add_argument("--authoritative-w2r", required=True)
    parser.add_argument("--candidate-w2", required=True)
    parser.add_argument("--candidate-w2r", required=True)
    parser.add_argument("--same-code-repeat", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"legacy gate is no-overwrite: {output}")
    paths = {
        "authority_manifest": str(Path(args.authority_manifest).resolve()),
        "authoritative_w2": str(Path(args.authoritative_w2).resolve()),
        "authoritative_w2r": str(Path(args.authoritative_w2r).resolve()),
        "candidate_w2": str(Path(args.candidate_w2).resolve()),
        "candidate_w2r": str(Path(args.candidate_w2r).resolve()),
        "same_code_repeat": str(Path(args.same_code_repeat).resolve()),
    }
    loaded = {
        name: _mapping(strict_json_load(path), name)
        for name, path in paths.items()
    }
    source_sha256_before = {
        name: file_sha256(path) for name, path in paths.items()
    }
    if source_sha256_before != FROZEN_SOURCE_SHA256:
        raise LegacyGateError(
            "legacy source SHA256 set differs from Amendment A5"
        )
    runtime_sha256_before = legacy_runtime_code_sha256()
    gate_code_before = legacy_gate_code_sha256()
    result = evaluate(
        **loaded,
        source_sha256=source_sha256_before,
        expected_runtime_code_sha256=runtime_sha256_before,
    )
    external = rerun_legacy_gates(paths)
    source_sha256_after = {
        name: file_sha256(path) for name, path in paths.items()
    }
    runtime_sha256_after = legacy_runtime_code_sha256()
    gate_code_after = legacy_gate_code_sha256()
    if (
        source_sha256_after != source_sha256_before
        or runtime_sha256_after != runtime_sha256_before
        or gate_code_after != gate_code_before
    ):
        raise LegacyGateError(
            "legacy source/runtime/gate code changed during gate execution"
        )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "amendment": "W2D-PREREGISTRATION-AMENDMENT-A5.md",
        "source_artifacts": {
            name: {
                "filename": Path(path).name,
                "sha256": source_sha256_before[name],
            }
            for name, path in paths.items()
        },
        "runtime_code_sha256": runtime_sha256_before,
        "git_commit": FROZEN_CANDIDATE_COMMIT,
        "gate_code_sha256": gate_code_before,
        "replacement_gate": result,
        "existing_gate_executions": external,
    }
    strict_json_dump(artifact, str(output))
    print(f"wrote {output}")
    print("verify_w2d_legacy: REPLACEMENT GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
