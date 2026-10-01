#!/usr/bin/env python3
"""Independent, fail-closed verifier and run-level summariser for W2D-E3.

This module intentionally does not import :mod:`w2d_e3_runner`.  The constants
and checks below are a second implementation of the contract frozen in
``W2D-E3-PREREGISTRATION.md`` and its amendment A1.  A summary is created only
after all five byte instances pass both the per-run and cross-run checks.

The summary keeps the server run as the experimental unit.  It records the five
run-level values and their median/minimum/maximum; request observations are
never pooled and no significance calculation is performed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
SCHEMA_VERSION = "W2D-E3-result-v1"
SUMMARY_SCHEMA_VERSION = "W2D-E3-summary-v1"
RUN_NUMBERS = (1, 2, 3, 4, 5)
STATIC_N = 100_000
FROZEN_N = 1_752
BACKGROUND_N = 98_248
DIM = 384
QUERY_N = 192
PANEL_A_EF = (20, 64, 256)
PANEL_A_REPEATS = 5
PANEL_A_LATENCY_N = 960
PRIMARY_EF = 64
TOP_K = 5
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
CONSISTENCY = "Strong"
PANEL_B_CONCURRENCY = (1, 4, 8, 16)
PANEL_B_WARMUP_S = 5.0
PANEL_B_MEASURED_S = 15.0
ADMISSION_RATE_PER_S = 20.0
TP_S = 1.0
VERIFIER_CONCURRENCY = 4
SENTINEL_N = 50
MAX_JSON_INT = (1 << 63) - 1
RUNTIME_COMPONENTS = (
    "w2d_e3_runner.py",
    "verify_w2d_e3.py",
    "milvus_backend.py",
    "functional_slice.py",
    "backend.py",
    "docker-compose.yml",
)
QUERY_KEYS_SHA256 = (
    "97d1ca7eac50989e79f902876a21202604804a7153752332d0889a4d699719ce"
)
QUERY_FAMILIES_SHA256 = (
    "ababae3822c5de19852b0b0b5e53b0ba4ed1facb41bd92f8ce71e72dac99731e"
)
QUERY_ASSOCIATED_IDS_SHA256 = (
    "e3e2e11e10b41134b7e4f983b99413b301479c9e7666f93baa2621e9b852014c"
)

FROZEN_PARENT_SHA256 = {
    "data/w2d/W2D-detector-inputs.json":
        "5784dbf469f004068f3f6b183f44ff51e36de0035ae8cc146ddb7291d22cf39b",
    "data/w2d/W2D-detector-inputs.npz":
        "1249c7cd23a03d3cde4e4c46ff3ec67ff9d59e6a3fd05978bb7c6da00bba1c45",
    "results/w2d/W2D-test-scores.json":
        "f928416f8adf7277ec6eeb536d2ebce668ae4655f97f425a3f57501137c91e01",
    "results/w2d/W2D-threshold.json":
        "52d87cdde2d868862764b18e134f6f28b0fc06c4092c1da9d0ec91e9cfc6eca3",
    "results/w2d/W2D-PROTOCOL-PLAN.json":
        "8209e741d1d5899c2fee6205f5c3808b5a8c60b3f33a2a0dc3b3b1b4774aa262",
    "results/w2d/W2D-E1-INMEMORY.json":
        "767248c0669fdde87f9c26072a0dde649b892b0a0153aa4729f8126104f1a18c",
    "results/w2d/W2D-E2-MILVUS.json":
        "90d4309c649ff090023ce5599d77802b77cbfe173a8099536d2f74a28d1ac2ef",
    "results/w2d/W2D-labels.json":
        "05d1363510a1790d690b05640c3a2830be4e62df56b8592182904ccb29019a9e",
}

WILLIAMS4 = (
    ("B1", "B2", "B4", "B3"),
    ("B2", "B3", "B1", "B4"),
    ("B3", "B4", "B2", "B1"),
    ("B4", "B1", "B3", "B2"),
)

GATE_NAMES = (
    "frozen_parent_hashes",
    "clean_tree_same_commit_runtime",
    "standalone_hnsw",
    "independent_process_start",
    "static_population_and_queries",
    "panel_a_complete",
    "panel_b_complete",
    "cell_accounting_balances",
    "baseline_semantics",
    "dynamic_cleanup",
    "sentinel_p95_movement",
    "strict_json",
)

ACCOUNTING_CHECKS = {
    "all_admissions_offered_equals_accepted_plus_failed",
    "admission_offered_equals_accepted_plus_failed",
    "query_attempts_equal_completed_plus_errors",
    "verifier_records_terminal_after_shutdown",
    "verifier_records_match_accepted_admissions",
}

POLICY_CHECKS = {
    "postfilter_static_passthrough_mode",
    "B1_no_verifier_calls",
    "B1_immediate_visibility",
    "B2_no_precommit_visibility",
    "B3_no_deadline",
    "B4_independent_deadline",
}

BASELINE_SWITCHES = {
    "B1": {"verify": False, "sync": False, "deadline": False, "decouple": True},
    "B2": {"verify": True, "sync": True, "deadline": False, "decouple": True},
    "B3": {"verify": True, "sync": False, "deadline": False, "decouple": True},
    "B4": {"verify": True, "sync": False, "deadline": True, "decouple": True},
}

HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX_COMMIT = re.compile(r"^[0-9a-f]{7,64}$")


class VerificationError(RuntimeError):
    """A result byte instance does not satisfy the frozen E3 contract."""


def _fail(where: str, message: str) -> None:
    raise VerificationError(f"{where}: {message}")


def strict_json_load_bytes(payload: bytes, source: str = "<bytes>") -> Any:
    """Parse strict UTF-8 JSON, rejecting duplicate, non-finite and huge numbers."""
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("strict_json_load_bytes requires bytes")

    def reject_constant(token: str) -> None:
        raise ValueError(f"non-standard numeric token {token}")

    def parse_integer(token: str) -> int:
        value = int(token, 10)
        if value < -MAX_JSON_INT - 1 or value > MAX_JSON_INT:
            raise ValueError(f"integer outside signed 64-bit range: {token}")
        return value

    def parse_floating(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise ValueError(f"non-finite or overflowing number: {token}")
        return value

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate object key {key!r}")
            result[key] = value
        return result

    try:
        text = bytes(payload).decode("utf-8", errors="strict")
        value = json.loads(
            text,
            parse_constant=reject_constant,
            parse_int=parse_integer,
            parse_float=parse_floating,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise VerificationError(f"{source}: strict JSON parse failed: {exc}") from exc
    assert_finite_tree(value, source)
    return value


def assert_finite_tree(value: Any, where: str = "$") -> None:
    """Reject values that cannot be represented by the strict result format."""
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        if value < -MAX_JSON_INT - 1 or value > MAX_JSON_INT:
            _fail(where, "integer outside signed 64-bit range")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(where, "NaN or Infinity")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite_tree(child, f"{where}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                _fail(where, "non-string object key")
            assert_finite_tree(child, f"{where}.{key}")
        return
    _fail(where, f"unsupported value type {type(value).__name__}")


def strict_json_bytes(value: Any) -> bytes:
    """Encode and immediately reparse a deterministic strict JSON document."""
    assert_finite_tree(value)
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"cannot encode strict JSON: {exc}") from exc
    payload = (text + "\n").encode("utf-8")
    strict_json_load_bytes(payload, "serialized summary")
    return payload


def exclusive_write_json(path: str | os.PathLike[str], value: Any) -> Path:
    """Exclusively create a summary; serialization happens before reservation."""
    target = Path(path)
    payload = strict_json_bytes(value)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise VerificationError(f"refusing to overwrite {target}") from exc
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # The verifier owns this just-created path.  A failed invocation must
        # not leave a partial file that can be mistaken for an accepted summary.
        try:
            target.unlink()
        except OSError:
            pass
        raise
    return target


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(where, "expected object")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(where, "expected array")
    return value


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        _fail(where, "expected boolean")
    return value


def _int(value: Any, where: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(where, "expected integer")
    if minimum is not None and value < minimum:
        _fail(where, f"expected integer >= {minimum}")
    return value


def _number(value: Any, where: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(where, "expected finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail(where, "expected finite number")
    if minimum is not None and result < minimum:
        _fail(where, f"expected number >= {minimum}")
    return result


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(where, "expected non-empty string")
    return value


def _eq(actual: Any, expected: Any, where: str) -> None:
    if actual != expected:
        _fail(where, f"expected {expected!r}, got {actual!r}")


def _close(actual: Any, expected: float, where: str, tol: float = 1e-9) -> None:
    number = _number(actual, where)
    if not math.isclose(number, float(expected), rel_tol=tol, abs_tol=tol):
        _fail(where, f"expected {expected!r}, got {actual!r}")


def _optional_close(actual: Any, expected: float | None, where: str) -> None:
    if expected is None:
        if actual is not None:
            _fail(where, f"expected null, got {actual!r}")
    else:
        _close(actual, expected, where)


def _sum_int_mapping(value: Any, where: str) -> int:
    mapping = _mapping(value, where)
    return sum(_int(item, f"{where}.{key}", minimum=0) for key, item in mapping.items())


def _timestamp(value: Any, where: str) -> datetime:
    text = _string(value, where)
    if not text.endswith("Z"):
        _fail(where, "timestamp must end in Z")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise VerificationError(f"{where}: invalid ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        _fail(where, "timestamp is not UTC")
    return parsed


def baseline_order(run_number: int) -> tuple[str, ...]:
    if run_number not in RUN_NUMBERS:
        raise ValueError("run number must be 1..5")
    return WILLIAMS4[(run_number - 1) % 4]


def concurrency_order(run_number: int) -> tuple[int, ...]:
    if run_number not in RUN_NUMBERS:
        raise ValueError("run number must be 1..5")
    shift = (run_number - 1) % 4
    return PANEL_B_CONCURRENCY[shift:] + PANEL_B_CONCURRENCY[:shift]


def expected_panel_b_pairs(run_number: int) -> list[tuple[str, int]]:
    return [
        (baseline, concurrency)
        for baseline in baseline_order(run_number)
        for concurrency in concurrency_order(run_number)
    ]


def _sequence_sha256(values: Sequence[Any]) -> str:
    payload = json.dumps(
        list(values),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def current_runtime_fingerprint(root: Path = HERE) -> dict[str, Any]:
    """Recompute the runner's recorded runtime binding without importing it."""
    components: dict[str, str] = {}
    combined = hashlib.sha256()
    for name in sorted(RUNTIME_COMPONENTS):
        path = root / name
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise VerificationError(f"cannot read runtime component {path}: {exc}") from exc
        components[name] = hashlib.sha256(payload).hexdigest()
        combined.update(name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(payload)
        combined.update(b"\0")
    return {"sha256": combined.hexdigest(), "components": components}


def _percentile_summary(
    value: Any,
    where: str,
    *,
    expected_count: int | None = None,
    allow_empty: bool = False,
) -> Mapping[str, Any]:
    summary = _mapping(value, where)
    count = _int(summary.get("count"), f"{where}.count", minimum=0)
    if expected_count is not None and count != expected_count:
        _fail(f"{where}.count", f"expected {expected_count}, got {count}")
    fields = ("p50", "p95", "p99", "max")
    if count == 0:
        if not allow_empty:
            _fail(where, "empty percentile population")
        for field in fields:
            if summary.get(field) is not None:
                _fail(f"{where}.{field}", "must be null for an empty population")
        return summary
    numbers = [_number(summary.get(field), f"{where}.{field}", minimum=0.0)
               for field in fields]
    if numbers != sorted(numbers):
        _fail(where, "percentiles/max are not monotone")
    return summary


def _ready_hnsw(
    value: Any,
    where: str,
    *,
    expected_indexed_rows: int | None = STATIC_N,
    minimum_indexed_rows: int | None = None,
) -> None:
    info = _mapping(value, where)
    if str(info.get("index_type_effective", "")).upper() != "HNSW":
        _fail(where, "effective index is not HNSW")
    if str(info.get("metric_type_effective", "")).upper() != "COSINE":
        _fail(where, "effective metric is not COSINE")
    _eq(info.get("M_effective"), HNSW_M, f"{where}.M_effective")
    _eq(
        info.get("efConstruction_effective"),
        HNSW_EF_CONSTRUCTION,
        f"{where}.efConstruction_effective",
    )
    indexed_rows = _int(
        info.get("indexed_rows"),
        f"{where}.indexed_rows",
        minimum=0,
    )
    if expected_indexed_rows is not None:
        _eq(
            indexed_rows,
            expected_indexed_rows,
            f"{where}.indexed_rows",
        )
    if minimum_indexed_rows is not None and indexed_rows < minimum_indexed_rows:
        _fail(
            f"{where}.indexed_rows",
            f"expected at least {minimum_indexed_rows}, got {indexed_rows}",
        )
    _eq(info.get("pending_index_rows"), 0, f"{where}.pending_index_rows")
    state = str(info.get("index_state", "")).lower()
    if "finish" not in state and state != "3":
        _fail(where, f"index is not finished: {state!r}")
    if "loaded" not in str(info.get("load_state", "")).lower():
        _fail(where, "collection is not loaded")


def _validate_accuracy_group(
    value: Any,
    where: str,
    records: Sequence[Mapping[str, Any]],
) -> None:
    group = _mapping(value, where)
    _eq(group.get("query_count"), len(records), f"{where}.query_count")
    successful = sum(record.get("returned_ids") is not None for record in records)
    _eq(
        group.get("successful_query_count"),
        successful,
        f"{where}.successful_query_count",
    )
    denominator = len(records)
    recall = (
        sum(float(record.get("recall_at_5") or 0.0) for record in records)
        / denominator
        if denominator
        else None
    )
    exact = (
        sum(record.get("exact_top5_match") is True for record in records)
        / denominator
        if denominator
        else None
    )
    landing = (
        sum(record.get("landing_concordant") is True for record in records)
        / denominator
        if denominator
        else None
    )
    _optional_close(group.get("recall_at_5"), recall, f"{where}.recall_at_5")
    _optional_close(
        group.get("exact_top5_match_rate"),
        exact,
        f"{where}.exact_top5_match_rate",
    )
    _optional_close(
        group.get("landing_concordance_rate"),
        landing,
        f"{where}.landing_concordance_rate",
    )
    _eq(
        group.get("failure_scoring"),
        "a query with no measured result contributes zero",
        f"{where}.failure_scoring",
    )


def _validate_panel_a(value: Any, where: str) -> dict[int, Mapping[str, Any]]:
    panel = _mapping(value, where)
    _eq(panel.get("formal_query_count"), QUERY_N, f"{where}.formal_query_count")
    _eq(panel.get("executed_query_count"), QUERY_N, f"{where}.executed_query_count")
    _eq(panel.get("repeats"), PANEL_A_REPEATS, f"{where}.repeats")
    _eq(
        panel.get("exact_reference"),
        "NumPy COSINE top-5 over the same static snapshot",
        f"{where}.exact_reference",
    )
    points = _list(panel.get("ef_points"), f"{where}.ef_points")
    _eq([point.get("ef") if isinstance(point, Mapping) else None for point in points],
        list(PANEL_A_EF), f"{where}.ef_points order")
    result: dict[int, Mapping[str, Any]] = {}
    reference_records: list[Mapping[str, Any]] | None = None
    for point_index, raw_point in enumerate(points):
        point_where = f"{where}.ef_points[{point_index}]"
        point = _mapping(raw_point, point_where)
        ef = _int(point.get("ef"), f"{point_where}.ef")
        _eq(point.get("primary"), ef == PRIMARY_EF, f"{point_where}.primary")
        _eq(point.get("warmup_query_count"), QUERY_N,
            f"{point_where}.warmup_query_count")
        _eq(point.get("measured_repeats"), PANEL_A_REPEATS,
            f"{point_where}.measured_repeats")
        _eq(point.get("measured_attempt_count"), PANEL_A_LATENCY_N,
            f"{point_where}.measured_attempt_count")
        _percentile_summary(
            point.get("latency_s"),
            f"{point_where}.latency_s",
            expected_count=PANEL_A_LATENCY_N,
        )
        errors = _int(point.get("query_errors"), f"{point_where}.query_errors",
                      minimum=0)
        if errors > PANEL_A_LATENCY_N:
            _fail(f"{point_where}.query_errors", "exceeds measured attempts")
        _eq(
            _sum_int_mapping(point.get("error_types"), f"{point_where}.error_types"),
            errors,
            f"{point_where}.error_types sum",
        )
        search_observed = _mapping(
            point.get("search_kwargs_observed"),
            f"{point_where}.search_kwargs_observed",
        )
        _eq(
            search_observed.get("consistency_level"),
            CONSISTENCY,
            f"{point_where}.search_kwargs_observed.consistency_level",
        )
        _eq(
            search_observed.get("search_params"),
            {"metric_type": "COSINE", "params": {"ef": ef}},
            f"{point_where}.search_kwargs_observed.search_params",
        )
        timeouts = _int(
            point.get("query_timeouts"),
            f"{point_where}.query_timeouts",
            minimum=0,
        )
        if timeouts > errors:
            _fail(f"{point_where}.query_timeouts", "exceeds all query errors")
        records_raw = _list(
            point.get("per_query_accuracy"),
            f"{point_where}.per_query_accuracy",
        )
        _eq(len(records_raw), QUERY_N, f"{point_where}.per_query_accuracy length")
        records: list[Mapping[str, Any]] = []
        query_keys: set[str] = set()
        for query_index, raw_record in enumerate(records_raw):
            record_where = f"{point_where}.per_query_accuracy[{query_index}]"
            record = _mapping(raw_record, record_where)
            key = _string(record.get("query_key"), f"{record_where}.query_key")
            if key in query_keys:
                _fail(f"{record_where}.query_key", "duplicate query key")
            query_keys.add(key)
            _string(record.get("attack_family"), f"{record_where}.attack_family")
            associated = _int(
                record.get("associated_static_id"),
                f"{record_where}.associated_static_id",
                minimum=0,
            )
            if associated >= STATIC_N:
                _fail(f"{record_where}.associated_static_id", "outside population")
            exact_ids = _list(record.get("exact_ids"), f"{record_where}.exact_ids")
            if len(exact_ids) != TOP_K or len(set(exact_ids)) != TOP_K:
                _fail(f"{record_where}.exact_ids", "must contain five unique ids")
            for ordinal, item_id in enumerate(exact_ids):
                parsed = _int(item_id, f"{record_where}.exact_ids[{ordinal}]",
                              minimum=0)
                if parsed >= STATIC_N:
                    _fail(f"{record_where}.exact_ids[{ordinal}]", "outside population")
            returned_raw = record.get("returned_ids")
            if returned_raw is None:
                _close(
                    record.get("recall_at_5"),
                    0.0,
                    f"{record_where}.recall_at_5",
                )
                for field in ("exact_top5_match", "ann_landed",
                              "landing_concordant"):
                    _eq(record.get(field), False, f"{record_where}.{field}")
                _eq(
                    record.get("accuracy_failure_zero"),
                    True,
                    f"{record_where}.accuracy_failure_zero",
                )
            else:
                returned = _list(returned_raw, f"{record_where}.returned_ids")
                if len(returned) != TOP_K or len(set(returned)) != TOP_K:
                    _fail(
                        f"{record_where}.returned_ids",
                        "must contain five unique ids",
                    )
                for ordinal, item_id in enumerate(returned):
                    parsed = _int(
                        item_id,
                        f"{record_where}.returned_ids[{ordinal}]",
                        minimum=0,
                    )
                    if parsed >= STATIC_N:
                        _fail(
                            f"{record_where}.returned_ids[{ordinal}]",
                            "outside population",
                        )
                expected_recall = len(set(exact_ids) & set(returned)) / TOP_K
                _close(
                    record.get("recall_at_5"),
                    expected_recall,
                    f"{record_where}.recall_at_5",
                )
                _eq(
                    record.get("exact_top5_match"),
                    returned == exact_ids,
                    f"{record_where}.exact_top5_match",
                )
                _eq(
                    record.get("ann_landed"),
                    associated in returned,
                    f"{record_where}.ann_landed",
                )
                _eq(
                    record.get("landing_concordant"),
                    (associated in exact_ids) == (associated in returned),
                    f"{record_where}.landing_concordant",
                )
                _eq(
                    record.get("accuracy_failure_zero"),
                    False,
                    f"{record_where}.accuracy_failure_zero",
                )
            _eq(
                record.get("exact_landed"),
                associated in exact_ids,
                f"{record_where}.exact_landed",
            )
            records.append(record)
        _eq(
            _sequence_sha256([record["query_key"] for record in records]),
            QUERY_KEYS_SHA256,
            f"{point_where}.query_key sequence SHA256",
        )
        _eq(
            _sequence_sha256([record["attack_family"] for record in records]),
            QUERY_FAMILIES_SHA256,
            f"{point_where}.attack_family sequence SHA256",
        )
        _eq(
            _sequence_sha256(
                [record["associated_static_id"] for record in records]
            ),
            QUERY_ASSOCIATED_IDS_SHA256,
            f"{point_where}.associated_static_id sequence SHA256",
        )
        _validate_accuracy_group(point.get("overall"), f"{point_where}.overall", records)
        families = sorted({str(record["attack_family"]) for record in records})
        family_doc = _mapping(
            point.get("by_attack_family"),
            f"{point_where}.by_attack_family",
        )
        _eq(set(family_doc), set(families), f"{point_where}.by_attack_family keys")
        for family in families:
            family_records = [
                record for record in records if record["attack_family"] == family
            ]
            _validate_accuracy_group(
                family_doc[family],
                f"{point_where}.by_attack_family.{family}",
                family_records,
            )
        if reference_records is None:
            reference_records = records
        else:
            reference = [
                (
                    record.get("query_key"),
                    record.get("attack_family"),
                    record.get("associated_static_id"),
                    record.get("exact_ids"),
                    record.get("exact_landed"),
                )
                for record in reference_records
            ]
            observed = [
                (
                    record.get("query_key"),
                    record.get("attack_family"),
                    record.get("associated_static_id"),
                    record.get("exact_ids"),
                    record.get("exact_landed"),
                )
                for record in records
            ]
            _eq(observed, reference, f"{point_where}.exact reference stability")
        result[ef] = point
    return result


SHADOW_CONFUSION_512 = {
    "denominator": 512,
    "promote": 318,
    "refuse": 194,
    "correct_promotion": 255,
    "false_promotion": 63,
    "correct_refusal": 129,
    "false_refusal": 65,
}

MEASURED_DECISION_MIX_300 = {
    "promote": 194,
    "refuse": 106,
    "correct_promotion": 156,
    "false_promotion": 38,
    "correct_refusal": 77,
    "false_refusal": 29,
}


def _validate_shadow_confusion(value: Any, where: str) -> None:
    """Validate the A1-bound 512-item detector confusion separately from lifecycle."""
    confusion = _mapping(value, where)
    for field, expected in SHADOW_CONFUSION_512.items():
        _eq(confusion.get(field), expected, f"{where}.{field}")
    _eq(
        confusion["promote"],
        confusion["correct_promotion"] + confusion["false_promotion"],
        f"{where}.promote balance",
    )
    _eq(
        confusion["refuse"],
        confusion["correct_refusal"] + confusion["false_refusal"],
        f"{where}.refuse balance",
    )
    _eq(
        confusion["promote"] + confusion["refuse"],
        confusion["denominator"],
        f"{where}.denominator balance",
    )


def _validate_decision_mix(
    value: Any,
    where: str,
    *,
    expected_denominator: int,
) -> None:
    mix = _mapping(value, where)
    expected_keys = {
        "promote",
        "refuse",
        "correct_promotion",
        "false_promotion",
        "correct_refusal",
        "false_refusal",
    }
    if set(mix) != expected_keys:
        _fail(where, f"expected keys {sorted(expected_keys)}, got {sorted(mix)}")
    counts = {
        key: _int(mix[key], f"{where}.{key}", minimum=0)
        for key in expected_keys
    }
    _eq(
        counts["promote"],
        counts["correct_promotion"] + counts["false_promotion"],
        f"{where}.promote balance",
    )
    _eq(
        counts["refuse"],
        counts["correct_refusal"] + counts["false_refusal"],
        f"{where}.refuse balance",
    )
    _eq(
        counts["promote"] + counts["refuse"],
        expected_denominator,
        f"{where}.denominator",
    )


def _validate_live_group(
    value: Any,
    where: str,
) -> dict[str, int]:
    group = _mapping(value, where)
    expected_keys = {"denominator", "ever_visible", "at_end", "exposure_seconds"}
    if set(group) != expected_keys:
        _fail(where, f"expected keys {sorted(expected_keys)}, got {sorted(group)}")
    denominator = _int(group.get("denominator"), f"{where}.denominator", minimum=0)
    ever_visible = _int(
        group.get("ever_visible"),
        f"{where}.ever_visible",
        minimum=0,
    )
    at_end = _int(group.get("at_end"), f"{where}.at_end", minimum=0)
    if ever_visible > denominator or at_end > denominator:
        _fail(where, "visible count exceeds denominator")
    if at_end > ever_visible:
        _fail(where, "at-end visibility exceeds ever-visible count")
    exposure = _percentile_summary(
        group.get("exposure_seconds"),
        f"{where}.exposure_seconds",
        expected_count=denominator,
        allow_empty=True,
    )
    if denominator:
        maximum = _number(
            exposure.get("max"),
            f"{where}.exposure_seconds.max",
            minimum=0.0,
        )
        if maximum > PANEL_B_MEASURED_S + 1e-9:
            _fail(
                f"{where}.exposure_seconds.max",
                "exceeds the measured 15-second horizon",
            )
    return {
        "denominator": denominator,
        "ever_visible": ever_visible,
        "at_end": at_end,
    }


def _validate_live_outcomes(
    value: Any,
    where: str,
    *,
    baseline: str,
    accepted_within_window: int,
    accepted_measured: int,
    decision_mix: Mapping[str, Any],
) -> None:
    outcomes = _mapping(value, where)
    expected_keys = {
        "denominator",
        "cohort",
        "poison",
        "clean",
        "d1_false_promotion_realized",
        "d1_false_refusal_realized",
    }
    if set(outcomes) != expected_keys:
        _fail(where, f"expected keys {sorted(expected_keys)}, got {sorted(outcomes)}")
    _eq(
        outcomes.get("cohort"),
        "measured admissions accepted by the fixed window horizon",
        f"{where}.cohort",
    )
    denominator = _int(
        outcomes.get("denominator"),
        f"{where}.denominator",
        minimum=0,
    )
    _eq(denominator, accepted_within_window, f"{where}.denominator")
    poison = _validate_live_group(outcomes.get("poison"), f"{where}.poison")
    clean = _validate_live_group(outcomes.get("clean"), f"{where}.clean")
    false_promotion = _validate_live_group(
        outcomes.get("d1_false_promotion_realized"),
        f"{where}.d1_false_promotion_realized",
    )
    false_refusal = _validate_live_group(
        outcomes.get("d1_false_refusal_realized"),
        f"{where}.d1_false_refusal_realized",
    )
    _eq(
        poison["denominator"] + clean["denominator"],
        denominator,
        f"{where}.poison_clean denominator balance",
    )
    if false_promotion["denominator"] > poison["denominator"]:
        _fail(where, "false-promotion denominator exceeds poison denominator")
    if false_refusal["denominator"] > clean["denominator"]:
        _fail(where, "false-refusal denominator exceeds clean denominator")
    if false_promotion["denominator"] > _int(
        decision_mix.get("false_promotion"),
        f"{where}.decision_mix.false_promotion",
        minimum=0,
    ):
        _fail(where, "live false promotions exceed accepted decision mix")
    if false_refusal["denominator"] > _int(
        decision_mix.get("false_refusal"),
        f"{where}.decision_mix.false_refusal",
        minimum=0,
    ):
        _fail(where, "live false refusals exceed accepted decision mix")
    if accepted_within_window == accepted_measured:
        _eq(
            poison["denominator"],
            int(decision_mix["false_promotion"])
            + int(decision_mix["correct_refusal"]),
            f"{where}.poison.denominator",
        )
        _eq(
            clean["denominator"],
            int(decision_mix["correct_promotion"])
            + int(decision_mix["false_refusal"]),
            f"{where}.clean.denominator",
        )
        _eq(
            false_promotion["denominator"],
            int(decision_mix["false_promotion"]),
            f"{where}.d1_false_promotion_realized.denominator",
        )
        _eq(
            false_refusal["denominator"],
            int(decision_mix["false_refusal"]),
            f"{where}.d1_false_refusal_realized.denominator",
        )
    if baseline == "B1":
        for name, group in (
            ("poison", poison),
            ("clean", clean),
            ("d1_false_promotion_realized", false_promotion),
            ("d1_false_refusal_realized", false_refusal),
        ):
            _eq(
                group["ever_visible"],
                group["denominator"],
                f"{where}.{name}.ever_visible",
            )
            _eq(
                group["at_end"],
                group["denominator"],
                f"{where}.{name}.at_end",
            )
    elif baseline == "B2":
        _eq(
            false_refusal["ever_visible"],
            0,
            f"{where}.d1_false_refusal_realized.ever_visible",
        )
        _eq(
            false_refusal["at_end"],
            0,
            f"{where}.d1_false_refusal_realized.at_end",
        )
    elif baseline in {"B3", "B4"}:
        for name, group in (
            ("d1_false_promotion_realized", false_promotion),
            ("d1_false_refusal_realized", false_refusal),
        ):
            _eq(
                group["ever_visible"],
                group["denominator"],
                f"{where}.{name}.ever_visible",
            )


def _validate_cell(
    value: Any,
    where: str,
    *,
    run_number: int,
    position: int,
    baseline: str,
    concurrency: int,
) -> Mapping[str, Any]:
    cell = _mapping(value, where)
    expected_id = (
        f"run-{run_number}__baseline-{baseline}__concurrency-{concurrency}"
    )
    _eq(cell.get("cell_id"), expected_id, f"{where}.cell_id")
    _eq(cell.get("position"), position, f"{where}.position")
    _eq(cell.get("baseline"), baseline, f"{where}.baseline")
    _eq(cell.get("query_concurrency"), concurrency, f"{where}.query_concurrency")
    _eq(
        cell.get("system_mode"),
        "postfilter_static_passthrough",
        f"{where}.system_mode",
    )
    _eq(cell.get("ef"), PRIMARY_EF, f"{where}.ef")
    _eq(cell.get("top_k"), TOP_K, f"{where}.top_k")
    _close(cell.get("tp_s"), TP_S, f"{where}.tp_s")
    _eq(
        cell.get("verifier_concurrency"),
        VERIFIER_CONCURRENCY,
        f"{where}.verifier_concurrency",
    )
    _eq(
        cell.get("policy_switches"),
        BASELINE_SWITCHES[baseline],
        f"{where}.policy_switches",
    )
    _close(cell.get("warmup_s"), PANEL_B_WARMUP_S, f"{where}.warmup_s")
    _close(cell.get("measured_s"), PANEL_B_MEASURED_S, f"{where}.measured_s")
    _number(
        cell.get("actual_cell_elapsed_s"),
        f"{where}.actual_cell_elapsed_s",
        minimum=PANEL_B_WARMUP_S + PANEL_B_MEASURED_S,
    )
    _close(
        cell.get("fixed_horizon_s"),
        PANEL_B_WARMUP_S + PANEL_B_MEASURED_S,
        f"{where}.fixed_horizon_s",
    )
    _eq(
        cell.get("snapshot_before_drain"),
        True,
        f"{where}.snapshot_before_drain",
    )
    _eq(cell.get("thread_pool_workers"), 32, f"{where}.thread_pool_workers")
    _eq(cell.get("async_error_types"), [], f"{where}.async_error_types")

    admissions = _mapping(cell.get("admissions"), f"{where}.admissions")
    _close(
        admissions.get("offered_rate_per_s"),
        ADMISSION_RATE_PER_S,
        f"{where}.admissions.offered_rate_per_s",
    )
    offered_all = _int(
        admissions.get("offered_including_warmup"),
        f"{where}.admissions.offered_including_warmup",
        minimum=0,
    )
    accepted_all = _int(
        admissions.get("accepted_including_warmup"),
        f"{where}.admissions.accepted_including_warmup",
        minimum=0,
    )
    failed_all = _int(
        admissions.get("failed_including_warmup"),
        f"{where}.admissions.failed_including_warmup",
        minimum=0,
    )
    offered = _int(admissions.get("offered"), f"{where}.admissions.offered",
                   minimum=0)
    accepted = _int(admissions.get("accepted"), f"{where}.admissions.accepted",
                    minimum=0)
    accepted_window = _int(
        admissions.get("accepted_within_window"),
        f"{where}.admissions.accepted_within_window",
        minimum=0,
    )
    accepted_after = _int(
        admissions.get("accepted_after_window"),
        f"{where}.admissions.accepted_after_window",
        minimum=0,
    )
    failed = _int(admissions.get("failed"), f"{where}.admissions.failed",
                  minimum=0)
    _eq(offered_all, 400, f"{where}.admissions.offered_including_warmup")
    _eq(offered, 300, f"{where}.admissions.offered")
    _eq(offered_all, accepted_all + failed_all, f"{where}.admissions all balance")
    _eq(offered, accepted + failed, f"{where}.admissions measured balance")
    _eq(
        accepted,
        accepted_window + accepted_after,
        f"{where}.admissions completion-window balance",
    )
    horizon_snapshot = _mapping(
        cell.get("horizon_admission_snapshot"),
        f"{where}.horizon_admission_snapshot",
    )
    horizon_offered = _int(
        horizon_snapshot.get("offered"),
        f"{where}.horizon_admission_snapshot.offered",
        minimum=0,
    )
    horizon_accepted = _int(
        horizon_snapshot.get("accepted"),
        f"{where}.horizon_admission_snapshot.accepted",
        minimum=0,
    )
    _eq(horizon_offered, offered_all,
        f"{where}.horizon_admission_snapshot.offered")
    if not accepted_window <= horizon_accepted <= accepted_all:
        _fail(
            f"{where}.horizon_admission_snapshot.accepted",
            "outside measured-horizon..all-accepted bounds",
        )
    _close(
        admissions.get("achieved_rate_per_s"),
        accepted_window / PANEL_B_MEASURED_S,
        f"{where}.admissions.achieved_rate_per_s",
    )
    _percentile_summary(
        admissions.get("dispatch_lag_s"),
        f"{where}.admissions.dispatch_lag_s",
        expected_count=offered,
    )
    _eq(
        _sum_int_mapping(
            admissions.get("error_types"),
            f"{where}.admissions.error_types",
        ),
        failed,
        f"{where}.admissions.error_types sum",
    )
    dynamic_range = _list(
        admissions.get("dynamic_id_range"),
        f"{where}.admissions.dynamic_id_range",
    )
    expected_lo = 1_000_000_000 + run_number * 10_000_000 + (position - 1) * 100_000
    _eq(dynamic_range, [expected_lo, expected_lo + 401],
        f"{where}.admissions.dynamic_id_range")

    queries = _mapping(cell.get("queries"), f"{where}.queries")
    attempts = _int(queries.get("attempts"), f"{where}.queries.attempts", minimum=1)
    completed = _int(
        queries.get("completed"), f"{where}.queries.completed", minimum=0
    )
    completed_window = _int(
        queries.get("completed_within_window"),
        f"{where}.queries.completed_within_window",
        minimum=0,
    )
    finished_after = _int(
        queries.get("finished_after_window"),
        f"{where}.queries.finished_after_window",
        minimum=0,
    )
    errors = _int(queries.get("errors"), f"{where}.queries.errors", minimum=0)
    _eq(attempts, completed + errors, f"{where}.queries attempt balance")
    if completed_window > completed:
        _fail(f"{where}.queries.completed_within_window", "exceeds completions")
    if finished_after > attempts:
        _fail(f"{where}.queries.finished_after_window", "exceeds attempts")
    _close(
        queries.get("achieved_qps"),
        completed_window / PANEL_B_MEASURED_S,
        f"{where}.queries.achieved_qps",
    )
    _percentile_summary(
        queries.get("latency_s"),
        f"{where}.queries.latency_s",
        expected_count=attempts,
    )
    _eq(
        _sum_int_mapping(queries.get("error_types"), f"{where}.queries.error_types"),
        errors,
        f"{where}.queries.error_types sum",
    )
    max_inflight = _int(
        queries.get("max_inflight"),
        f"{where}.queries.max_inflight",
        minimum=0,
    )
    if max_inflight > concurrency:
        _fail(f"{where}.queries.max_inflight", "exceeds configured concurrency")

    verifier = _mapping(cell.get("verifier"), f"{where}.verifier")
    verifier_total = _int(
        verifier.get("records_total_including_warmup"),
        f"{where}.verifier.records_total_including_warmup",
        minimum=0,
    )
    measured_records = _int(
        verifier.get("measured_records"),
        f"{where}.verifier.measured_records",
        minimum=0,
    )
    statuses = _mapping(verifier.get("status_counts"), f"{where}.verifier.status_counts")
    allowed_statuses = {"COMMITTED", "CANCELLED", "FAILED"}
    if not set(statuses).issubset(allowed_statuses):
        _fail(f"{where}.verifier.status_counts", "contains a non-terminal status")
    _eq(
        _sum_int_mapping(statuses, f"{where}.verifier.status_counts"),
        verifier_total,
        f"{where}.verifier.status_counts sum",
    )
    window_statuses = _mapping(
        verifier.get("window_snapshot_status_counts"),
        f"{where}.verifier.window_snapshot_status_counts",
    )
    allowed_window_statuses = {
        "QUEUED",
        "RUNNING",
        "COMMITTED",
        "CANCELLED",
        "FAILED",
    }
    if not set(window_statuses).issubset(allowed_window_statuses):
        _fail(
            f"{where}.verifier.window_snapshot_status_counts",
            "contains an unknown status",
        )
    window_record_count = _sum_int_mapping(
        window_statuses,
        f"{where}.verifier.window_snapshot_status_counts",
    )
    if window_record_count > horizon_accepted:
        _fail(
            f"{where}.verifier.window_snapshot_status_counts",
            "contains more records than horizon-accepted admissions",
        )
    _eq(
        verifier.get("cancelled"),
        int(statuses.get("CANCELLED", 0)),
        f"{where}.verifier.cancelled",
    )
    _eq(
        verifier.get("failed"),
        int(statuses.get("FAILED", 0)),
        f"{where}.verifier.failed",
    )
    shutdown_errors = _list(
        verifier.get("shutdown_error_types"),
        f"{where}.verifier.shutdown_error_types",
    )
    if shutdown_errors:
        _fail(f"{where}.verifier.shutdown_error_types", "async worker errors observed")
    if baseline == "B1":
        _eq(verifier.get("execution"), "none_undefended", f"{where}.verifier.execution")
        _eq(verifier_total, 0, f"{where}.verifier.records_total_including_warmup")
        _eq(measured_records, 0, f"{where}.verifier.measured_records")
        _eq(window_record_count, 0,
            f"{where}.verifier.window_snapshot_status_counts sum")
    else:
        _eq(
            verifier.get("execution"),
            "measured-service trace replay, not online detector computation",
            f"{where}.verifier.execution",
        )
        _eq(verifier_total, accepted_all,
            f"{where}.verifier.records_total_including_warmup")
        _eq(measured_records, accepted, f"{where}.verifier.measured_records")
    for metric in (
        "queue_wait_s",
        "service_time_s",
        "integrated_latency_s",
        "queue_depth_at_enqueue",
    ):
        summary = _percentile_summary(
            verifier.get(metric),
            f"{where}.verifier.{metric}",
            allow_empty=True,
        )
        if _int(summary.get("count"), f"{where}.verifier.{metric}.count") > measured_records:
            _fail(f"{where}.verifier.{metric}.count", "exceeds measured records")

    decision_mix = _mapping(cell.get("decision_mix"), f"{where}.decision_mix")
    _validate_decision_mix(
        decision_mix,
        f"{where}.decision_mix",
        expected_denominator=accepted,
    )
    # A1 requires the full, independently frozen 512-item confusion to remain
    # visible as a separate denominator.  The measured-cell mix above is not a
    # substitute for it.
    _validate_shadow_confusion(
        cell.get("shadow_confusion_512"),
        f"{where}.shadow_confusion_512",
    )
    for field in (
        "promote",
        "refuse",
        "correct_promotion",
        "false_promotion",
        "correct_refusal",
        "false_refusal",
    ):
        if _int(
            decision_mix.get(field),
            f"{where}.decision_mix.{field}",
            minimum=0,
        ) > SHADOW_CONFUSION_512[field]:
            _fail(
                f"{where}.decision_mix.{field}",
                "exceeds the frozen 512-item shadow count",
            )
    if accepted == 300:
        _eq(
            dict(decision_mix),
            MEASURED_DECISION_MIX_300,
            f"{where}.decision_mix full measured cohort",
        )
    _validate_live_outcomes(
        cell.get("live_outcomes"),
        f"{where}.live_outcomes",
        baseline=baseline,
        accepted_within_window=accepted_window,
        accepted_measured=accepted,
        decision_mix=decision_mix,
    )
    truth_checks = _mapping(
        cell.get("truth_separation_checks"),
        f"{where}.truth_separation_checks",
    )
    expected_truth_checks = {
        "no_truth_field_in_pre_snapshot_events",
        "system_received_no_evaluator_truth",
        "decision_provider_has_no_truth_mapping",
        "classification_ran_after_snapshot",
    }
    if set(truth_checks) != expected_truth_checks:
        _fail(
            f"{where}.truth_separation_checks",
            "unexpected truth-separation assertion set",
        )
    for key, check in truth_checks.items():
        _eq(check, True, f"{where}.truth_separation_checks.{key}")

    lifecycle = _mapping(cell.get("lifecycle"), f"{where}.lifecycle")
    for field in (
        "deadline_hidden_at_window_end",
        "right_censored_provisional_at_window_end",
        "pending_not_visible_at_window_end",
        "B2_precommit_visibility_violations",
        "B1_immediate_visibility_violations",
        "B3_hidden_transition_violations",
    ):
        count = _int(lifecycle.get(field), f"{where}.lifecycle.{field}", minimum=0)
        if count > accepted_window:
            _fail(f"{where}.lifecycle.{field}", "exceeds horizon cohort")
    if lifecycle.get("B2_precommit_visibility_violations") != 0:
        _fail(f"{where}.lifecycle", "B2 precommit visibility violation")
    if lifecycle.get("B1_immediate_visibility_violations") != 0:
        _fail(f"{where}.lifecycle", "B1 visibility violation")
    if lifecycle.get("B3_hidden_transition_violations") != 0:
        _fail(f"{where}.lifecycle", "B3 hidden transition violation")

    sentinel = _mapping(cell.get("sentinel"), f"{where}.sentinel")
    sentinel_p95: list[float] = []
    for side in ("before", "after"):
        burst = _mapping(sentinel.get(side), f"{where}.sentinel.{side}")
        _eq(burst.get("started"), SENTINEL_N, f"{where}.sentinel.{side}.started")
        _eq(burst.get("completed"), SENTINEL_N,
            f"{where}.sentinel.{side}.completed")
        _eq(burst.get("errors"), 0, f"{where}.sentinel.{side}.errors")
        _eq(
            _sum_int_mapping(
                burst.get("error_types"),
                f"{where}.sentinel.{side}.error_types",
            ),
            0,
            f"{where}.sentinel.{side}.error_types sum",
        )
        latency = _percentile_summary(
            burst.get("latency_s"),
            f"{where}.sentinel.{side}.latency_s",
            expected_count=SENTINEL_N,
        )
        sentinel_p95.append(_number(latency["p95"], f"{where}.sentinel.{side}.p95"))
    if min(sentinel_p95) <= 0:
        _fail(f"{where}.sentinel", "P95 must be positive")
    expected_movement = max(sentinel_p95) / min(sentinel_p95)
    _close(
        sentinel.get("p95_movement_ratio"),
        expected_movement,
        f"{where}.sentinel.p95_movement_ratio",
    )
    if expected_movement > 1.5:
        _fail(f"{where}.sentinel.p95_movement_ratio", "exceeds 1.5x gate")
    _eq(sentinel.get("within_1_5x"), True, f"{where}.sentinel.within_1_5x")

    cleanup = _mapping(cell.get("cleanup"), f"{where}.cleanup")
    _eq(cleanup.get("delete_range_lo"), dynamic_range[0],
        f"{where}.cleanup.delete_range_lo")
    _eq(cleanup.get("delete_range_hi"), dynamic_range[1],
        f"{where}.cleanup.delete_range_hi")
    _eq(cleanup.get("delete_confirmed"), True, f"{where}.cleanup.delete_confirmed")
    _eq(cleanup.get("delete_remaining_n"), 0,
        f"{where}.cleanup.delete_remaining_n")
    _eq(cleanup.get("residual_dynamic_row_count"), 0,
        f"{where}.cleanup.residual_dynamic_row_count")
    _eq(cleanup.get("residual_dynamic_ids"), [],
        f"{where}.cleanup.residual_dynamic_ids")
    _eq(cleanup.get("live_tasks_after_shutdown"), 0,
        f"{where}.cleanup.live_tasks_after_shutdown")
    _eq(cleanup.get("live_timers_after_shutdown"), 0,
        f"{where}.cleanup.live_timers_after_shutdown")
    physical_preinsert = _int(
        cleanup.get("physical_preinsert_count"),
        f"{where}.cleanup.physical_preinsert_count",
        minimum=0,
    )
    logical_noop = _int(
        cleanup.get("logical_noop_insert_count"),
        f"{where}.cleanup.logical_noop_insert_count",
        minimum=0,
    )
    if physical_preinsert < accepted_all or physical_preinsert > offered_all:
        _fail(
            f"{where}.cleanup.physical_preinsert_count",
            "outside accepted..offered bounds",
        )
    if logical_noop > accepted_all or logical_noop > physical_preinsert:
        _fail(
            f"{where}.cleanup.logical_noop_insert_count",
            "exceeds accepted/preinserted count",
        )
    if baseline in {"B1", "B3", "B4"}:
        _eq(
            logical_noop,
            accepted_all,
            f"{where}.cleanup.logical_noop_insert_count",
        )

    accounting = _mapping(cell.get("accounting_checks"), f"{where}.accounting_checks")
    if set(accounting) != ACCOUNTING_CHECKS:
        _fail(f"{where}.accounting_checks", "unexpected accounting assertion set")
    if any(_bool(value, f"{where}.accounting_checks.{key}") is not True
           for key, value in accounting.items()):
        _fail(f"{where}.accounting_checks", "one or more assertions failed")
    policy = _mapping(cell.get("policy_checks"), f"{where}.policy_checks")
    if set(policy) != POLICY_CHECKS:
        _fail(f"{where}.policy_checks", "unexpected policy assertion set")
    if any(_bool(value, f"{where}.policy_checks.{key}") is not True
           for key, value in policy.items()):
        _fail(f"{where}.policy_checks", "one or more assertions failed")
    return cell


def _validate_panel_b(
    value: Any,
    where: str,
    *,
    run_number: int,
) -> dict[tuple[str, int], Mapping[str, Any]]:
    panel = _mapping(value, where)
    _eq(panel.get("formal_matrix_cell_count"), 16,
        f"{where}.formal_matrix_cell_count")
    _eq(panel.get("executed_matrix_cell_count"), 16,
        f"{where}.executed_matrix_cell_count")
    _eq(panel.get("baseline_order"), list(baseline_order(run_number)),
        f"{where}.baseline_order")
    _eq(panel.get("concurrency_order"), list(concurrency_order(run_number)),
        f"{where}.concurrency_order")
    _eq(panel.get("query_order_seed"), 8300 + run_number,
        f"{where}.query_order_seed")
    _eq(
        panel.get("run5_balance_note"),
        (
            "run 5 repeats Williams row 1 and is reported separately"
            if run_number == 5
            else None
        ),
        f"{where}.run5_balance_note",
    )
    ordinals = _list(panel.get("query_order_ordinals"),
                     f"{where}.query_order_ordinals")
    if len(ordinals) != QUERY_N:
        _fail(f"{where}.query_order_ordinals", "expected 192 ordinals")
    parsed_ordinals = [
        _int(value, f"{where}.query_order_ordinals[{index}]", minimum=0)
        for index, value in enumerate(ordinals)
    ]
    _eq(sorted(parsed_ordinals), list(range(QUERY_N)),
        f"{where}.query_order_ordinals permutation")
    expected_ordinals = [
        int(value)
        for value in np.random.default_rng(8300 + run_number).permutation(QUERY_N)
    ]
    _eq(
        parsed_ordinals,
        expected_ordinals,
        f"{where}.query_order_ordinals frozen order",
    )
    cells = _list(panel.get("cells"), f"{where}.cells")
    _eq(len(cells), 16, f"{where}.cells length")
    expected = expected_panel_b_pairs(run_number)
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for index, ((baseline, concurrency), raw_cell) in enumerate(zip(expected, cells)):
        cell = _validate_cell(
            raw_cell,
            f"{where}.cells[{index}]",
            run_number=run_number,
            position=index + 1,
            baseline=baseline,
            concurrency=concurrency,
        )
        result[(baseline, concurrency)] = cell
    return result


def _validate_configuration(
    value: Any,
    where: str,
    *,
    run_number: int,
) -> None:
    config = _mapping(value, where)
    _eq(config.get("static_n"), STATIC_N, f"{where}.static_n")
    _eq(config.get("formal_static_n"), STATIC_N, f"{where}.formal_static_n")
    hnsw = _mapping(config.get("hnsw"), f"{where}.hnsw")
    _eq(hnsw.get("M"), HNSW_M, f"{where}.hnsw.M")
    _eq(
        hnsw.get("efConstruction"),
        HNSW_EF_CONSTRUCTION,
        f"{where}.hnsw.efConstruction",
    )
    _eq(hnsw.get("metric"), "COSINE", f"{where}.hnsw.metric")
    _eq(hnsw.get("consistency"), CONSISTENCY, f"{where}.hnsw.consistency")
    _eq(config.get("panel_a_ef"), list(PANEL_A_EF), f"{where}.panel_a_ef")
    _eq(
        config.get("panel_a_repeats"),
        PANEL_A_REPEATS,
        f"{where}.panel_a_repeats",
    )
    _eq(
        config.get("panel_b_baseline_order"),
        list(baseline_order(run_number)),
        f"{where}.panel_b_baseline_order",
    )
    _eq(
        config.get("panel_b_concurrency_order"),
        list(concurrency_order(run_number)),
        f"{where}.panel_b_concurrency_order",
    )
    _eq(
        config.get("thread_pool_workers"),
        32,
        f"{where}.thread_pool_workers",
    )


def _validate_provenance(value: Any, where: str) -> dict[str, Any]:
    provenance = _mapping(value, where)
    git = _mapping(provenance.get("git"), f"{where}.git")
    commit = _string(git.get("commit"), f"{where}.git.commit")
    if HEX_COMMIT.fullmatch(commit) is None:
        _fail(f"{where}.git.commit", "not a hexadecimal Git object id")
    _eq(git.get("tracked_tree_clean"), True, f"{where}.git.tracked_tree_clean")
    _eq(git.get("tracked_status"), [], f"{where}.git.tracked_status")

    fingerprint = _mapping(
        provenance.get("runtime_fingerprint"),
        f"{where}.runtime_fingerprint",
    )
    fingerprint_sha = _string(
        fingerprint.get("sha256"),
        f"{where}.runtime_fingerprint.sha256",
    )
    if HEX64.fullmatch(fingerprint_sha) is None:
        _fail(f"{where}.runtime_fingerprint.sha256", "not a SHA256")
    components = _mapping(
        fingerprint.get("components"),
        f"{where}.runtime_fingerprint.components",
    )
    expected_components = set(RUNTIME_COMPONENTS)
    _eq(set(components), expected_components,
        f"{where}.runtime_fingerprint.components keys")
    for name, digest in components.items():
        if not isinstance(digest, str) or HEX64.fullmatch(digest) is None:
            _fail(f"{where}.runtime_fingerprint.components.{name}", "not a SHA256")

    environment = _mapping(provenance.get("environment"), f"{where}.environment")
    _string(environment.get("python"), f"{where}.environment.python")
    _string(
        environment.get("python_executable"),
        f"{where}.environment.python_executable",
    )
    pymilvus_version = _string(
        environment.get("pymilvus_version"),
        f"{where}.environment.pymilvus_version",
    )
    if pymilvus_version != "2.4.15":
        _fail(f"{where}.environment.pymilvus_version", "expected 2.4.15")
    _string(environment.get("platform"), f"{where}.environment.platform")
    _string(environment.get("machine"), f"{where}.environment.machine")
    _int(environment.get("cpu_count"), f"{where}.environment.cpu_count", minimum=1)
    memory = _mapping(environment.get("memory"), f"{where}.environment.memory")
    _int(
        memory.get("physical_memory_bytes"),
        f"{where}.environment.memory.physical_memory_bytes",
        minimum=1,
    )

    process = _mapping(
        provenance.get("process_start_metric"),
        f"{where}.process_start_metric",
    )
    _eq(
        process.get("metric"),
        "process_start_time_seconds",
        f"{where}.process_start_metric.metric",
    )
    process_value = _number(
        process.get("value"),
        f"{where}.process_start_metric.value",
        minimum=1.0,
    )
    metric_url = _string(
        process.get("url"),
        f"{where}.process_start_metric.url",
    )
    if not metric_url.startswith("http://") or not metric_url.endswith(":9091/metrics"):
        _fail(f"{where}.process_start_metric.url", "unexpected metrics endpoint")
    _int(
        process.get("matching_series_count"),
        f"{where}.process_start_metric.matching_series_count",
        minimum=1,
    )

    prereg = _mapping(provenance.get("preregistration"), f"{where}.preregistration")
    _eq(prereg.get("parent_commit"), "2c414c1",
        f"{where}.preregistration.parent_commit")
    _eq(prereg.get("document_commit"), "e09b056",
        f"{where}.preregistration.document_commit")
    _eq(prereg.get("amendment_a1_commit"), "a4f31c6",
        f"{where}.preregistration.amendment_a1_commit")
    _eq(
        prereg.get("documents"),
        [
            "W2D-E3-PREREGISTRATION.md",
            "W2D-E3-PREREGISTRATION-AMENDMENT-A1.md",
        ],
        f"{where}.preregistration.documents",
    )
    return {
        "commit": commit,
        "fingerprint_sha256": fingerprint_sha,
        "fingerprint_components": dict(components),
        "environment": dict(environment),
        "process_start": process_value,
        "metrics_url": metric_url,
    }


def _validate_backend(value: Any, where: str) -> dict[str, Any]:
    backend = _mapping(value, where)
    _eq(backend.get("deployment_mode"), "standalone", f"{where}.deployment_mode")
    version = _string(backend.get("server_version"), f"{where}.server_version")
    if version.lstrip("v") != "2.4.15":
        _fail(f"{where}.server_version", f"expected v2.4.15, got {version!r}")
    uri = _string(backend.get("server_uri"), f"{where}.server_uri")
    if not uri.startswith("http://"):
        _fail(f"{where}.server_uri", "formal Standalone URI must be HTTP")
    _eq(backend.get("implementation"), "MilvusBackend", f"{where}.implementation")
    _eq(backend.get("metric"), "COSINE", f"{where}.metric")
    _eq(backend.get("vector_dim"), DIM, f"{where}.vector_dim")
    _eq(
        backend.get("consistency_level"),
        CONSISTENCY,
        f"{where}.consistency_level",
    )
    _eq(
        str(backend.get("index_type_requested", "")).upper(),
        "HNSW",
        f"{where}.index_type_requested",
    )
    requested = _mapping(
        backend.get("index_params_requested"),
        f"{where}.index_params_requested",
    )
    _eq(requested.get("M"), HNSW_M, f"{where}.index_params_requested.M")
    _eq(
        requested.get("efConstruction"),
        HNSW_EF_CONSTRUCTION,
        f"{where}.index_params_requested.efConstruction",
    )
    _eq(backend.get("search_primary"), {"ef": PRIMARY_EF},
        f"{where}.search_primary")
    _ready_hnsw(backend.get("index_info"), f"{where}.index_info")
    _ready_hnsw(
        backend.get("index_info_after_panels"),
        f"{where}.index_info_after_panels",
        expected_indexed_rows=None,
        minimum_indexed_rows=STATIC_N,
    )
    state_after = str(backend.get("index_state_observed_after_panels", "")).lower()
    if "finish" not in state_after and state_after != "3":
        _fail(
            f"{where}.index_state_observed_after_panels",
            f"post-panel wait did not observe a finished index: {state_after!r}",
        )
    _number(
        backend.get("index_wait_after_panels_s"),
        f"{where}.index_wait_after_panels_s",
        minimum=0.0,
    )
    _eq(
        backend.get("server_row_count_after_panels"),
        STATIC_N,
        f"{where}.server_row_count_after_panels",
    )
    _string(backend.get("collection"), f"{where}.collection")
    thread_pool = _mapping(backend.get("thread_pool"), f"{where}.thread_pool")
    _eq(
        thread_pool.get("implementation"),
        "concurrent.futures.ThreadPoolExecutor",
        f"{where}.thread_pool.implementation",
    )
    _eq(thread_pool.get("max_workers"), 32, f"{where}.thread_pool.max_workers")
    _eq(
        thread_pool.get("purpose"),
        "Milvus insert/search/cleanup RPC isolation",
        f"{where}.thread_pool.purpose",
    )
    return {
        "server_version": version,
        "server_uri": uri,
        "collection": backend["collection"],
    }


def _validate_population(value: Any, where: str) -> None:
    population = _mapping(value, where)
    _eq(population.get("static_n"), STATIC_N, f"{where}.static_n")
    _eq(population.get("formal_static_n"), STATIC_N, f"{where}.formal_static_n")
    _eq(population.get("frozen_n"), FROZEN_N, f"{where}.frozen_n")
    _eq(
        population.get("synthetic_extension_n"),
        BACKGROUND_N,
        f"{where}.synthetic_extension_n",
    )
    _eq(
        population.get("synthetic_extension_seed"),
        7301,
        f"{where}.synthetic_extension_seed",
    )
    _eq(population.get("graph_build_count"), 1, f"{where}.graph_build_count")
    _eq(population.get("background_seed"), 7301, f"{where}.background_seed")
    _eq(population.get("single_graph_build"), True, f"{where}.single_graph_build")
    _eq(population.get("dimension"), DIM, f"{where}.dimension")
    _eq(population.get("dtype"), "float32", f"{where}.dtype")
    _eq(population.get("primary_key_range"), [0, STATIC_N],
        f"{where}.primary_key_range")
    _eq(
        population.get("unique_primary_keys"),
        STATIC_N,
        f"{where}.unique_primary_keys",
    )
    _number(
        population.get("normalization_max_abs_error"),
        f"{where}.normalization_max_abs_error",
        minimum=0.0,
    )
    _eq(
        population.get("synthetic_extension_interpretation"),
        (
            "systems pressure load; not a real-text corpus and not a "
            "detector-quality/attack-generalization denominator"
        ),
        f"{where}.synthetic_extension_interpretation",
    )
    insert = _mapping(population.get("insert"), f"{where}.insert")
    _eq(insert.get("attempted"), STATIC_N, f"{where}.insert.attempted")
    _eq(insert.get("inserted"), STATIC_N, f"{where}.insert.inserted")
    _int(insert.get("batch_size"), f"{where}.insert.batch_size", minimum=1)
    _number(insert.get("insertion_s"), f"{where}.insert.insertion_s", minimum=0.0)
    _eq(
        population.get("server_row_count_after_index"),
        STATIC_N,
        f"{where}.server_row_count_after_index",
    )
    _number(population.get("index_build_s"), f"{where}.index_build_s", minimum=0.0)


def _validate_gates(value: Any, where: str) -> None:
    gates = _list(value, where)
    _eq(len(gates), 12, f"{where} length")
    for index, (gate, expected_name) in enumerate(zip(gates, GATE_NAMES), start=1):
        gate_where = f"{where}[{index - 1}]"
        gate_map = _mapping(gate, gate_where)
        _eq(gate_map.get("gate_id"), index, f"{gate_where}.gate_id")
        _eq(gate_map.get("name"), expected_name, f"{gate_where}.name")
        _eq(gate_map.get("passed"), True, f"{gate_where}.passed")
        if "detail" not in gate_map:
            _fail(gate_where, "missing gate detail")


def validate_run(document: Any, source: str = "<run>") -> dict[str, Any]:
    """Validate one formal run and return the independently extracted values."""
    root = _mapping(document, source)
    _eq(root.get("schema_version"), SCHEMA_VERSION, f"{source}.schema_version")
    _eq(root.get("experiment"), "W2D-E3", f"{source}.experiment")
    run_number = _int(root.get("run_number"), f"{source}.run_number")
    if run_number not in RUN_NUMBERS:
        _fail(f"{source}.run_number", "must be 1..5")
    _eq(root.get("formal"), True, f"{source}.formal")
    _eq(root.get("quick_nonformal"), False, f"{source}.quick_nonformal")
    _eq(root.get("status"), "locally_accepted", f"{source}.status")
    _eq(
        root.get("local_validity_accepted"),
        True,
        f"{source}.local_validity_accepted",
    )
    _eq(
        root.get("cross_run_acceptance"),
        "pending five-file verifier",
        f"{source}.cross_run_acceptance",
    )
    started = _timestamp(root.get("started_at_utc"), f"{source}.started_at_utc")
    finished = _timestamp(root.get("finished_at_utc"), f"{source}.finished_at_utc")
    if finished < started:
        _fail(source, "finished timestamp precedes start")
    _eq(
        root.get("frozen_parent_hashes"),
        FROZEN_PARENT_SHA256,
        f"{source}.frozen_parent_hashes",
    )
    provenance = _validate_provenance(root.get("provenance"), f"{source}.provenance")
    _validate_configuration(
        root.get("configuration"),
        f"{source}.configuration",
        run_number=run_number,
    )
    backend = _validate_backend(root.get("runtime_backend"), f"{source}.runtime_backend")
    _validate_population(root.get("population"), f"{source}.population")
    panel_a = _validate_panel_a(root.get("panel_a"), f"{source}.panel_a")
    panel_b = _validate_panel_b(
        root.get("panel_b"),
        f"{source}.panel_b",
        run_number=run_number,
    )
    _validate_gates(root.get("validity_gates"), f"{source}.validity_gates")
    _eq(
        root.get("interpretation_boundary"),
        [
            "single-node Milvus Standalone only",
            "measured-service trace replay, not online detector computation",
            "synthetic extension is not a 100k real-text corpus",
            "no distributed scaling, production capacity, high availability, "
            "fault tolerance, unknown-attack robustness, or end-to-end RAG "
            "answer-safety claim",
        ],
        f"{source}.interpretation_boundary",
    )

    cleanup = _mapping(
        root.get("final_collection_cleanup"),
        f"{source}.final_collection_cleanup",
    )
    _eq(cleanup.get("drop_confirmed"), True,
        f"{source}.final_collection_cleanup.drop_confirmed")
    _eq(cleanup.get("passed"), True, f"{source}.final_collection_cleanup.passed")

    primary_recall = _number(
        panel_a[PRIMARY_EF].get("overall", {}).get("recall_at_5"),
        f"{source}.panel_a.ef64.overall.recall_at_5",
        minimum=0.0,
    )
    if primary_recall > 1.0:
        _fail(f"{source}.panel_a.ef64.overall.recall_at_5", "exceeds 1")
    transfer = _mapping(root.get("transfer_condition"), f"{source}.transfer_condition")
    _eq(transfer.get("primary_ef"), PRIMARY_EF,
        f"{source}.transfer_condition.primary_ef")
    _close(
        transfer.get("per_run_recall_at_5"),
        primary_recall,
        f"{source}.transfer_condition.per_run_recall_at_5",
    )
    _close(
        transfer.get("required_minimum"),
        0.95,
        f"{source}.transfer_condition.required_minimum",
    )
    _eq(
        transfer.get("supported_this_run"),
        primary_recall >= 0.95,
        f"{source}.transfer_condition.supported_this_run",
    )
    return {
        "run_number": run_number,
        "document": root,
        "provenance": provenance,
        "backend": backend,
        "panel_a": panel_a,
        "panel_b": panel_b,
        "primary_recall": primary_recall,
        "started_at_utc": root["started_at_utc"],
        "finished_at_utc": root["finished_at_utc"],
    }


def _metric_summary(
    runs: Sequence[Mapping[str, Any]],
    extractor: Callable[[Mapping[str, Any]], Any],
    *,
    where: str,
) -> dict[str, Any]:
    run_values: list[dict[str, Any]] = []
    defined: list[float] = []
    for run in runs:
        raw = extractor(run)
        if raw is None:
            value = None
        else:
            value = _number(raw, f"{where}.run_{run['run_number']}")
            defined.append(value)
        run_values.append({"run_number": run["run_number"], "value": value})
    if defined:
        median = float(statistics.median(defined))
        minimum = float(min(defined))
        maximum = float(max(defined))
    else:
        median = minimum = maximum = None
    return {
        "experimental_unit": "one accepted Milvus Standalone process run",
        "run_values": run_values,
        "defined_run_count": len(defined),
        "median": median,
        "min": minimum,
        "max": maximum,
    }


def _nested_metric(
    runs: Sequence[Mapping[str, Any]],
    base_extractor: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    fields: Sequence[str],
    where: str,
) -> dict[str, Any]:
    return {
        field: _metric_summary(
            runs,
            lambda run, field=field: base_extractor(run).get(field),
            where=f"{where}.{field}",
        )
        for field in fields
    }


def build_summary(
    validated_runs: Sequence[Mapping[str, Any]],
    *,
    source_files: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Cross-check five runs and build a run-level-only aggregate document."""
    if len(validated_runs) != 5:
        _fail("runs", "exactly five validated runs are required")
    ordered = sorted(validated_runs, key=lambda run: int(run["run_number"]))
    _eq([run["run_number"] for run in ordered], list(RUN_NUMBERS), "run numbers")

    commits = {run["provenance"]["commit"] for run in ordered}
    fingerprints = {run["provenance"]["fingerprint_sha256"] for run in ordered}
    fingerprint_components = {
        json.dumps(run["provenance"]["fingerprint_components"], sort_keys=True)
        for run in ordered
    }
    process_starts = [run["provenance"]["process_start"] for run in ordered]
    server_versions = {run["backend"]["server_version"] for run in ordered}
    server_uris = {run["backend"]["server_uri"] for run in ordered}
    collections = [run["backend"]["collection"] for run in ordered]
    environments = {
        json.dumps(run["provenance"]["environment"], sort_keys=True)
        for run in ordered
    }
    if len(commits) != 1:
        _fail("cross_run.git_commit", "runs do not share one commit")
    if len(fingerprints) != 1 or len(fingerprint_components) != 1:
        _fail("cross_run.runtime_fingerprint", "runs do not share one runtime")
    current_fingerprint = current_runtime_fingerprint()
    _eq(
        next(iter(fingerprints)),
        current_fingerprint["sha256"],
        "cross_run.runtime_fingerprint vs current files",
    )
    _eq(
        ordered[0]["provenance"]["fingerprint_components"],
        current_fingerprint["components"],
        "cross_run.runtime components vs current files",
    )
    if len(set(process_starts)) != 5:
        _fail("cross_run.process_start", "five distinct process starts required")
    if len(server_versions) != 1 or len(server_uris) != 1:
        _fail("cross_run.server", "server version/URI changed across runs")
    if len(set(collections)) != 5:
        _fail("cross_run.collection", "five freshly named collections required")
    if len(environments) != 1:
        _fail("cross_run.environment", "client/host descriptor changed across runs")

    panel_a: dict[str, Any] = {}
    for ef in PANEL_A_EF:
        point = lambda run, ef=ef: run["panel_a"][ef]
        panel_a[str(ef)] = {
            "recall_at_5": _metric_summary(
                ordered,
                lambda run, point=point: point(run)["overall"]["recall_at_5"],
                where=f"panel_a.{ef}.recall_at_5",
            ),
            "exact_top5_match_rate": _metric_summary(
                ordered,
                lambda run, point=point:
                    point(run)["overall"]["exact_top5_match_rate"],
                where=f"panel_a.{ef}.exact_top5_match_rate",
            ),
            "landing_concordance_rate": _metric_summary(
                ordered,
                lambda run, point=point:
                    point(run)["overall"]["landing_concordance_rate"],
                where=f"panel_a.{ef}.landing_concordance_rate",
            ),
            "query_errors": _metric_summary(
                ordered,
                lambda run, point=point: point(run)["query_errors"],
                where=f"panel_a.{ef}.query_errors",
            ),
            "latency_s": _nested_metric(
                ordered,
                lambda run, point=point: point(run)["latency_s"],
                ("p50", "p95", "p99", "max"),
                f"panel_a.{ef}.latency_s",
            ),
        }

    panel_b: dict[str, Any] = {}
    for baseline in ("B1", "B2", "B3", "B4"):
        baseline_doc: dict[str, Any] = {}
        for concurrency in PANEL_B_CONCURRENCY:
            cell = lambda run, baseline=baseline, concurrency=concurrency: (
                run["panel_b"][(baseline, concurrency)]
            )
            cell_where = f"panel_b.{baseline}.concurrency_{concurrency}"
            baseline_doc[str(concurrency)] = {
                "admissions": {
                    "offered_rate_per_s": _metric_summary(
                        ordered,
                        lambda run, cell=cell:
                            cell(run)["admissions"]["offered_rate_per_s"],
                        where=f"{cell_where}.admissions.offered_rate_per_s",
                    ),
                    "achieved_rate_per_s": _metric_summary(
                        ordered,
                        lambda run, cell=cell:
                            cell(run)["admissions"]["achieved_rate_per_s"],
                        where=f"{cell_where}.admissions.achieved_rate_per_s",
                    ),
                    "failed": _metric_summary(
                        ordered,
                        lambda run, cell=cell: cell(run)["admissions"]["failed"],
                        where=f"{cell_where}.admissions.failed",
                    ),
                },
                "queries": {
                    "achieved_qps": _metric_summary(
                        ordered,
                        lambda run, cell=cell: cell(run)["queries"]["achieved_qps"],
                        where=f"{cell_where}.queries.achieved_qps",
                    ),
                    "errors": _metric_summary(
                        ordered,
                        lambda run, cell=cell: cell(run)["queries"]["errors"],
                        where=f"{cell_where}.queries.errors",
                    ),
                    "latency_s": _nested_metric(
                        ordered,
                        lambda run, cell=cell: cell(run)["queries"]["latency_s"],
                        ("p50", "p95", "p99", "max"),
                        f"{cell_where}.queries.latency_s",
                    ),
                },
                "verifier": {
                    metric: _nested_metric(
                        ordered,
                        lambda run, cell=cell, metric=metric:
                            cell(run)["verifier"][metric],
                        ("p50", "p95", "p99", "max"),
                        f"{cell_where}.verifier.{metric}",
                    )
                    for metric in (
                        "queue_wait_s",
                        "service_time_s",
                        "integrated_latency_s",
                    )
                },
                "decision_mix": _nested_metric(
                    ordered,
                    lambda run, cell=cell: cell(run)["decision_mix"],
                    (
                        "promote",
                        "refuse",
                        "correct_promotion",
                        "false_promotion",
                        "correct_refusal",
                        "false_refusal",
                    ),
                    f"{cell_where}.decision_mix",
                ),
                "shadow_confusion_512": _nested_metric(
                    ordered,
                    lambda run, cell=cell: cell(run)["shadow_confusion_512"],
                    (
                        "false_promotion",
                        "false_refusal",
                        "correct_promotion",
                        "correct_refusal",
                    ),
                    f"{cell_where}.shadow_confusion_512",
                ),
                "live_outcomes": {
                    "denominator": _metric_summary(
                        ordered,
                        lambda run, cell=cell:
                            cell(run)["live_outcomes"]["denominator"],
                        where=f"{cell_where}.live_outcomes.denominator",
                    ),
                    **{
                        group: {
                            **_nested_metric(
                                ordered,
                                lambda run, cell=cell, group=group:
                                    cell(run)["live_outcomes"][group],
                                ("denominator", "ever_visible", "at_end"),
                                f"{cell_where}.live_outcomes.{group}",
                            ),
                            "exposure_seconds": _nested_metric(
                                ordered,
                                lambda run, cell=cell, group=group:
                                    cell(run)["live_outcomes"][group][
                                        "exposure_seconds"
                                    ],
                                ("p50", "p95", "p99", "max"),
                                (
                                    f"{cell_where}.live_outcomes.{group}."
                                    "exposure_seconds"
                                ),
                            ),
                        }
                        for group in (
                            "poison",
                            "clean",
                            "d1_false_promotion_realized",
                            "d1_false_refusal_realized",
                        )
                    },
                },
                "lifecycle": _nested_metric(
                    ordered,
                    lambda run, cell=cell: cell(run)["lifecycle"],
                    (
                        "deadline_hidden_at_window_end",
                        "right_censored_provisional_at_window_end",
                        "pending_not_visible_at_window_end",
                    ),
                    f"{cell_where}.lifecycle",
                ),
                "sentinel_p95_movement_ratio": _metric_summary(
                    ordered,
                    lambda run, cell=cell:
                        cell(run)["sentinel"]["p95_movement_ratio"],
                    where=f"{cell_where}.sentinel_p95_movement_ratio",
                ),
            }
        panel_b[baseline] = baseline_doc

    transfer_values = [
        {"run_number": run["run_number"], "recall_at_5": run["primary_recall"]}
        for run in ordered
    ]
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "W2D-E3",
        "status": "cross_run_accepted",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "accepted_run_numbers": list(RUN_NUMBERS),
        "source_files": list(source_files),
        "frozen_parent_hashes": dict(FROZEN_PARENT_SHA256),
        "cross_run_provenance": {
            "git_commit": next(iter(commits)),
            "runtime_fingerprint_sha256": next(iter(fingerprints)),
            "runtime_fingerprint_components":
                ordered[0]["provenance"]["fingerprint_components"],
            "server_version": next(iter(server_versions)),
            "server_uri": next(iter(server_uris)),
            "collections": [
                {"run_number": run["run_number"], "name": run["backend"]["collection"]}
                for run in ordered
            ],
            "process_start_time_seconds": [
                {"run_number": run["run_number"], "value": run["provenance"]["process_start"]}
                for run in ordered
            ],
            "five_distinct_process_starts": True,
        },
        "transfer_condition": {
            "primary_ef": PRIMARY_EF,
            "required_recall_at_5_per_run": 0.95,
            "run_values": transfer_values,
            "supported_across_all_five_runs": all(
                run["primary_recall"] >= 0.95 for run in ordered
            ),
            "negative_result_does_not_invalidate_runs": True,
        },
        "aggregation_semantics": {
            "experimental_unit": "one accepted Milvus Standalone process run",
            "run_count": 5,
            "request_observations_pooled": False,
            "significance_test_performed": False,
            "reported_cross_run_statistics": ["median", "min", "max"],
            "note": (
                "Each value was computed inside one run; request counts are not "
                "treated as independent experimental units."
            ),
        },
        "panel_a": panel_a,
        "panel_b": panel_b,
        "interpretation_boundary": [
            "single-node Milvus Standalone only",
            "measured-service trace replay, not online detector computation",
            "synthetic extension is not a 100k real-text corpus",
            "no distributed scaling, production capacity, high availability, "
            "fault tolerance, unknown-attack robustness, or end-to-end RAG "
            "answer-safety claim",
        ],
    }
    assert_finite_tree(summary)
    return summary


def verify_files(
    run_paths: Sequence[str | os.PathLike[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Strictly read and validate exactly five distinct run byte instances."""
    if len(run_paths) != 5:
        _fail("--runs", "exactly five paths are required")
    resolved = [Path(path).resolve() for path in run_paths]
    if len(set(resolved)) != 5:
        _fail("--runs", "duplicate paths are forbidden")
    validated: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    seen_bytes: set[str] = set()
    for path in resolved:
        if not path.is_file():
            _fail(str(path), "not a regular file")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest in seen_bytes:
            _fail(str(path), "duplicate run byte instance")
        seen_bytes.add(digest)
        document = strict_json_load_bytes(payload, str(path))
        validated.append(validate_run(document, str(path)))
        source_files.append(
            {"path": str(path), "sha256": digest, "bytes": len(payload)}
        )
    run_numbers = [run["run_number"] for run in validated]
    if sorted(run_numbers) != list(RUN_NUMBERS):
        _fail("--runs", f"must contain run numbers 1..5 exactly once: {run_numbers}")
    # Keep provenance aligned with run order in the summary.
    paired = sorted(zip(validated, source_files), key=lambda pair: pair[0]["run_number"])
    return [pair[0] for pair in paired], [pair[1] for pair in paired]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed verification of exactly five formal W2D-E3 run files; "
            "writes one exclusive run-level summary only after every gate passes."
        )
    )
    parser.add_argument(
        "--runs",
        nargs=5,
        required=True,
        metavar=("RUN1", "RUN2", "RUN3", "RUN4", "RUN5"),
        help="exactly five formal E3 JSON byte instances",
    )
    parser.add_argument(
        "--summary-out",
        required=True,
        type=Path,
        help="new summary path; an existing path is never overwritten",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.summary_out.exists():
        print(
            f"verification failed: refusing to overwrite {args.summary_out}",
            file=sys.stderr,
        )
        return 2
    try:
        validated, source_files = verify_files(args.runs)
        summary = build_summary(validated, source_files=source_files)
        exclusive_write_json(args.summary_out, summary)
    except (OSError, VerificationError, TypeError, ValueError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 2
    print(
        "verified five independent formal W2D-E3 runs and wrote "
        f"{args.summary_out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
