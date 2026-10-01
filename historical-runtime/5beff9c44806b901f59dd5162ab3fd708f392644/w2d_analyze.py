#!/usr/bin/env python3
"""Recompute the final W2D statistical report from a verified formal bundle.

This module is intentionally downstream of :mod:`verify_w2d`.  The first
operation performed by :func:`analyze_bundle` is the complete fail-loud bundle
verification; no result or manifest field is read before that gate succeeds.

The report keeps three concepts separate:

* detector quality is computed once over 512 unique source groups;
* B1 is an undefended control and never receives FP/FN terminology; and
* B2/B3/B4 are repeated, paired policy episodes, not new detector samples.

No ordinary median of an episode duration is emitted.  Right-censored ages
remain in fixed-horizon restricted totals and are never called containment
times.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from w2d_metrics import (
    detector_quality_summary,
    empirical_quantile,
    file_sha256,
    latency_summary,
    rate_record,
    source_control_gate_summary,
    strict_json_load,
)


HERE = Path(__file__).resolve().parent
SCHEMA_VERSION = "W2D-analysis-v1"
QUERY_ROLES = (
    "attack_associated",
    "heldout_same_topic",
    "negative_other_topic",
)
VISIBLE_STATES = frozenset({"PROVISIONAL", "TRUSTED"})


class AnalysisError(RuntimeError):
    """The verified bundle cannot be reduced to the preregistered statistics."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{path} must be an object")
    return value


def _list(value: Any, path: str) -> list[Any]:
    _require(isinstance(value, list), f"{path} must be a list")
    return value


def _finite_tree(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        _require(math.isfinite(value), f"{path} contains NaN or infinity")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require(isinstance(key, str), f"{path} has a non-string key")
            _finite_tree(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_tree(child, f"{path}[{index}]")
        return
    raise AnalysisError(f"{path} has unsupported type {type(value).__name__}")


def _exclusive_json_dump(value: Any, path: str | os.PathLike[str]) -> None:
    _finite_tree(value)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    with target.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _artifact_paths(
    manifest_path: Path, manifest: Mapping[str, Any]
) -> dict[str, Path]:
    entries = _mapping(manifest.get("artifact_hashes"), "manifest.artifact_hashes")
    result: dict[str, Path] = {}
    for name, raw in entries.items():
        record = _mapping(raw, f"manifest.artifact_hashes.{name}")
        path_value = record.get("path")
        _require(
            isinstance(path_value, str) and bool(path_value),
            f"artifact {name} path is absent",
        )
        path = Path(path_value)
        result[str(name)] = (
            path if path.is_absolute() else manifest_path.parent / path
        )
    return result


def _snapshot_hashes(
    *,
    manifest_path: Path,
    results_path: Path,
    artifact_paths: Mapping[str, Path],
) -> dict[str, str]:
    paths = {
        "manifest": manifest_path,
        "results": results_path,
        **{f"artifact:{name}": path for name, path in artifact_paths.items()},
    }
    return {
        name: file_sha256(str(path))
        for name, path in sorted(paths.items())
    }


def _label_map(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = document.get("items")
    if isinstance(raw, Mapping):
        result = {
            str(key): _mapping(value, f"labels.items.{key}")
            for key, value in raw.items()
        }
    else:
        result = {}
        for index, value in enumerate(_list(raw, "labels.items")):
            record = _mapping(value, f"labels.items[{index}]")
            key = record.get("item_key")
            _require(
                isinstance(key, str) and key and key not in result,
                "labels contain a missing or duplicate item_key",
            )
            result[key] = record
    return result


def _score_map(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(_list(document.get("items"), "scores.items")):
        record = _mapping(value, f"scores.items[{index}]")
        key = record.get("item_key")
        _require(
            isinstance(key, str) and key and key not in result,
            "scores contain a missing or duplicate item_key",
        )
        result[key] = record
    return result


def _numeric_summary(values: Iterable[int | float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    _require(
        all(math.isfinite(value) for value in numbers),
        "numeric summary received a non-finite value",
    )
    if not numbers:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(numbers),
        "min": min(numbers),
        "p50": empirical_quantile(numbers, 0.50),
        "p95": empirical_quantile(numbers, 0.95),
        "p99": empirical_quantile(numbers, 0.99),
        "max": max(numbers),
    }


def _latency_seconds(values: Iterable[int | float]) -> dict[str, Any]:
    nanoseconds = [
        int(round(float(value) * 1_000_000_000.0)) for value in values
    ]
    return latency_summary(nanoseconds)


def _status_partition(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Any]:
    total = len(rows)
    counts = Counter(str(row[field]) for row in rows)
    statuses = ("NOT_STARTED", "COMPLETED", "RIGHT_CENSORED", "NOT_APPLICABLE")
    return {
        "denominator": total,
        "counts": {status: counts.get(status, 0) for status in statuses},
        "rates": {
            status: rate_record(counts.get(status, 0), total)
            for status in statuses
        },
    }


def _observed_or_zero(row: Mapping[str, Any], field: str) -> float:
    value = row.get(field)
    if value is None:
        return 0.0
    number = float(value)
    _require(math.isfinite(number) and number >= 0.0, f"{field} is invalid")
    return number


def _detector_section(
    test_scores: Mapping[str, Any],
    labels_document: Mapping[str, Any],
) -> dict[str, Any]:
    labels = _label_map(labels_document)
    scores = _score_map(test_scores)
    formal = detector_quality_summary(test_scores, labels_document)

    def records_where(predicate: Any) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
        return [
            (score, labels[key])
            for key, score in scores.items()
            if predicate(score, labels[key])
        ]

    attack: dict[str, Any] = {}
    for family, expected in (
        ("recipe", 128),
        ("natural_cover_suffix_v1", 64),
    ):
        rows = records_where(
            lambda _score, label, family=family: (
                label.get("poison") is True
                and label.get("attack_family") == family
            )
        )
        _require(len(rows) == expected, f"{family} detector denominator drifted")
        tp = sum(not bool(score["promote"]) for score, _label in rows)
        fn = len(rows) - tp
        attack[family] = {
            "truth_poison_n": len(rows),
            "tp_refused_n": tp,
            "fn_promoted_n": fn,
            "recall": rate_record(tp, len(rows)),
            "fnr": rate_record(fn, len(rows)),
            "detector_only_latency": latency_summary(
                [int(score["detector_service_ns"]) for score, _label in rows]
            ),
        }

    clean: dict[str, Any] = {}
    for name, stratum, expected in (
        ("ordinary_clean", "ordinary_clean", 192),
        ("hard_negative_clean", "hard_negative_clean", 128),
    ):
        rows = records_where(
            lambda _score, label, stratum=stratum: (
                label.get("poison") is False and label.get("stratum") == stratum
            )
        )
        _require(len(rows) == expected, f"{name} detector denominator drifted")
        fp = sum(not bool(score["promote"]) for score, _label in rows)
        tn = len(rows) - fp
        clean[name] = {
            "truth_clean_n": len(rows),
            "fp_refused_n": fp,
            "tn_promoted_n": tn,
            "fpr": rate_record(fp, len(rows)),
            "specificity": rate_record(tn, len(rows)),
            "detector_only_latency": latency_summary(
                [int(score["detector_service_ns"]) for score, _label in rows]
            ),
        }

    return {
        "unique_item_count": formal["unique_item_count"],
        "unique_source_group_count": formal["unique_source_group_count"],
        "overall": formal["overall"],
        "clean_by_stratum": clean,
        "poison_by_attack": attack,
        "detector_only_latency_all_512": formal["detector_only_latency"],
        "denominator_rule": (
            "one score-once observation per unique source group; protocol "
            "replay never enlarges this denominator"
        ),
    }


def _landing_section(
    landing: Mapping[str, Any],
    runtime_cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [
        _mapping(value, f"landing.items[{index}]")
        for index, value in enumerate(_list(landing.get("items"), "landing.items"))
    ]
    by_key = {str(row["item_key"]): row for row in rows}
    selected: dict[str, set[str]] = defaultdict(set)
    for cell in runtime_cells:
        family = str(cell["attack_family"])
        for item in _list(cell.get("protocol_items"), "cell.protocol_items"):
            record = _mapping(item, "cell protocol item")
            if record.get("role") == "poison":
                selected[family].add(str(record["item_key"]))

    result: dict[str, Any] = {}
    for family, expected in (
        ("recipe", 128),
        ("natural_cover_suffix_v1", 64),
    ):
        family_rows = [row for row in rows if row.get("attack_family") == family]
        _require(len(family_rows) == expected, f"{family} landing count drifted")
        landed = sum(bool(row["landed_top5"]) for row in family_rows)
        selected_keys = selected[family]
        _require(len(selected_keys) == 30, f"{family} selected runtime set is not 30")
        selected_landed = sum(
            bool(_mapping(by_key.get(key), f"landing item {key}")["landed_top5"])
            for key in selected_keys
        )
        result[family] = {
            "all_test_poison": rate_record(landed, expected),
            "runtime_selected_poison": rate_record(selected_landed, 30),
            "failed_landing_policy": "retain as outcome; never replace",
        }
    return result


def _cell_maps(
    cell: Mapping[str, Any],
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
]:
    items: dict[str, Mapping[str, Any]] = {}
    for raw in _list(cell.get("protocol_items"), "cell.protocol_items"):
        item = _mapping(raw, "cell protocol item")
        items[str(item["item_key"])] = item
    lifecycle: dict[str, Mapping[str, Any]] = {}
    for raw in _list(cell.get("lifecycle"), "cell.lifecycle"):
        row = _mapping(raw, "cell lifecycle row")
        lifecycle[str(row["item_key"])] = row
    providers: dict[str, Mapping[str, Any]] = {}
    for raw in _list(cell.get("verifier_records"), "cell.verifier_records"):
        row = _mapping(raw, "cell verifier record")
        providers[str(row["item_key"])] = row
    _require(set(items) == set(lifecycle), "cell item/lifecycle keys differ")
    return items, lifecycle, providers


def _rows_for_role(
    cells: Sequence[Mapping[str, Any]], role: str
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for cell in cells:
        items, lifecycle, providers = _cell_maps(cell)
        for key, item in items.items():
            if item.get("role") != role:
                continue
            result.append(
                {
                    "item": item,
                    "lifecycle": lifecycle[key],
                    "provider": providers.get(key),
                    "cell": cell,
                }
            )
    return result


def _poison_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline: str,
    arm: str,
    landing_by_key: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    lifecycle = [row["lifecycle"] for row in rows]
    total = len(rows)
    started = [
        row for row in lifecycle if row.get("exposure_status") != "NOT_STARTED"
    ]
    right_censored = sum(
        row.get("exposure_status") == "RIGHT_CENSORED" for row in started
    )
    exposure_total = sum(
        _observed_or_zero(row, "exposure_observed_s") for row in started
    )
    landed = sum(
        bool(
            _mapping(
                landing_by_key.get(str(row["item"]["item_key"])),
                f"landing {row['item']['item_key']}",
            ).get("landed_top5")
        )
        for row in rows
    )
    common = {
        "poison_item_episode_n": total,
        "visibility_started": rate_record(len(started), total),
        "exposure_episode_status": _status_partition(
            lifecycle, "exposure_status"
        ),
        "right_censored_among_started": rate_record(
            right_censored, len(started)
        ),
        "right_censored_among_all": rate_record(right_censored, total),
        "restricted_exposure_total_s": exposure_total,
        "restricted_exposure_per_started_s": (
            exposure_total / len(started) if started else None
        ),
        "state_at_horizon": {
            state: {
                "count": count,
                "rate": rate_record(count, total),
            }
            for state, count in sorted(
                Counter(str(row["state_at_horizon"]) for row in lifecycle).items()
            )
        },
        "selected_item_landing": rate_record(landed, total),
        "duration_policy": (
            "fixed-horizon restricted totals; no ordinary duration median"
        ),
    }
    if baseline == "B1":
        common["control_interpretation"] = (
            "undefended off-path control; shadow D1 values do not drive state"
        )
        return common

    d1_fn = sum(bool(row["item"]["detector_promote"]) for row in rows)
    committed_promotions = sum(
        row["provider"] is not None
        and row["provider"].get("status") == "COMMITTED"
        and row["provider"].get("passes") is True
        for row in rows
    )
    common["d1_false_negative_decisions"] = rate_record(d1_fn, total)
    common["actual_arm_poison_promotions"] = rate_record(
        committed_promotions, total
    )
    if arm == "detector":
        common["realized_false_promotions"] = rate_record(
            committed_promotions, total
        )
    else:
        common["oracle_mechanism_upper_bound"] = True
    return common


def _clean_summary(
    rows: Sequence[Mapping[str, Any]], *, baseline: str, arm: str
) -> dict[str, Any]:
    lifecycle = [row["lifecycle"] for row in rows]
    total = len(rows)
    never_visible = sum(
        row.get("visibility_status") == "NOT_STARTED" for row in lifecycle
    )
    first_visible = total - never_visible
    durable = sum(
        row.get("state_at_horizon") in VISIBLE_STATES for row in lifecycle
    )
    unavailable_total = sum(
        _observed_or_zero(row, "unavailable_observed_s") for row in lifecycle
    )
    common = {
        "clean_item_episode_n": total,
        "never_visible": rate_record(never_visible, total),
        "first_visible": rate_record(first_visible, total),
        "durable_visible_at_horizon": rate_record(durable, total),
        "unavailable_episode_status": _status_partition(
            lifecycle, "unavailable_status"
        ),
        "quarantine_episode_status": _status_partition(
            lifecycle, "quarantine_status"
        ),
        "restricted_unavailable_total_s": unavailable_total,
        "restricted_unavailable_per_item_s": (
            unavailable_total / total if total else None
        ),
        "readmission_total_n": sum(
            int(row.get("readmission_count", 0)) for row in lifecycle
        ),
        "duration_policy": (
            "fixed-horizon restricted totals; no ordinary duration median"
        ),
    }
    if baseline == "B1":
        common["control_interpretation"] = (
            "undefended off-path availability control; no verifier error semantics"
        )
        return common

    d1_fp = sum(not bool(row["item"]["detector_promote"]) for row in rows)
    misquarantine = sum(
        row["provider"] is not None
        and row["provider"].get("status") == "COMMITTED"
        and row["provider"].get("passes") is False
        and row["lifecycle"].get("quarantine_status") != "NOT_STARTED"
        for row in rows
    )
    common["d1_false_positive_decisions"] = rate_record(d1_fp, total)
    common["actual_arm_misquarantine"] = rate_record(misquarantine, total)
    if arm == "detector":
        common["realized_false_positive_misquarantine"] = rate_record(
            misquarantine, total
        )
    else:
        common["oracle_mechanism_upper_bound"] = True
    return common


def _eu_summary(
    rows: Sequence[Mapping[str, Any]], *, baseline: str
) -> dict[str, Any]:
    lifecycle = [row["lifecycle"] for row in rows]
    total = len(rows)
    started = [
        row
        for row in lifecycle
        if row.get("unvetted_visibility_status") != "NOT_STARTED"
    ]
    observed = [
        _observed_or_zero(row, "unvetted_visibility_observed_s")
        for row in started
    ]
    if baseline == "B2":
        violations = len(started)
        bound = "must not start"
        bound_denominator = total
    elif baseline == "B4":
        violations = sum(
            row.get("unvetted_visibility_status") != "COMPLETED"
            or _observed_or_zero(
                row, "unvetted_visibility_observed_s"
            ) > 1.02
            for row in started
        )
        bound = "each started episode <= 1.02 s and completed"
        bound_denominator = len(started)
    else:
        violations = 0
        bound = "not applicable"
        bound_denominator = 0
    return {
        "episode_status": _status_partition(
            lifecycle, "unvetted_visibility_status"
        ),
        "started": rate_record(len(started), total),
        "observed_time_at_risk_total_s": sum(observed),
        "observed_time_at_risk_per_started_s": (
            sum(observed) / len(started) if started else None
        ),
        "max_observed_s": max(observed) if observed else None,
        "bound": bound,
        "bound_violations": rate_record(violations, bound_denominator),
        "censoring_interpretation": (
            "RIGHT_CENSORED contributes observed age to restricted total, "
            "never a finite containment time"
        ),
    }


def _retrieval_summary(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    events: list[Mapping[str, Any]] = []
    landing_num = landing_den = 0
    for cell in cells:
        retrieval = _mapping(cell.get("retrieval"), "cell.retrieval")
        landing_num += int(retrieval.get("attack_landing_numerator", 0))
        landing_den += int(retrieval.get("attack_landing_denominator", 0))
        raw_events = retrieval.get("events", [])
        if raw_events is None:
            raw_events = []
        for index, raw in enumerate(_list(raw_events, "cell.retrieval.events")):
            events.append(_mapping(raw, f"retrieval event {index}"))
    by_role: dict[str, Any] = {}
    for role in QUERY_ROLES:
        rows = [row for row in events if row.get("query_role") == role]
        overall_hits = sum(bool(row.get("hit_overall")) for row in rows)
        landed_hits = sum(bool(row.get("hit_landed_only")) for row in rows)
        displacements = [
            float(row["poisonfree_displacement_at_5"])
            for row in rows
            if row.get("poisonfree_displacement_at_5") is not None
        ]
        by_role[role] = {
            "query_event_n": len(rows),
            "poisoned_query_events_overall": rate_record(
                overall_hits, len(rows)
            ),
            "poisoned_query_events_landed_only": rate_record(
                landed_hits, len(rows)
            ),
            "poison_item_returns_overall_n": sum(
                len(row.get("poison_item_keys", [])) for row in rows
            ),
            "poison_item_returns_landed_only_n": sum(
                len(row.get("landed_poison_item_keys", [])) for row in rows
            ),
            "cumulative_displaced_top5_positions": sum(
                value * 5.0 for value in displacements
            ),
            "displacement_observation_n": len(displacements),
        }
    return {
        "selected_attack_landing": rate_record(landing_num, landing_den),
        "query_roles": by_role,
        "query_event_interval_warning": (
            "repeated query ticks are correlated; Wilson intervals are "
            "descriptive and not independent-sample inference"
        ),
    }


def _queue_summary(cells: Sequence[Mapping[str, Any]], *, baseline: str) -> dict[str, Any]:
    records = [
        _mapping(raw, "cell verifier record")
        for cell in cells
        for raw in _list(cell.get("verifier_records"), "cell.verifier_records")
    ]
    if baseline == "B1":
        _require(not records, "B1 unexpectedly contains verifier records")
        return {
            "applicable": False,
            "requested_n": 0,
            "started": rate_record(0, 0),
            "committed": rate_record(0, 0),
            "cancelled": rate_record(0, 0),
            "detector_service_replay_latency": latency_summary([]),
            "queue_wait_latency": latency_summary([]),
            "integrated_queue_to_commit_latency": latency_summary([]),
            "queue_depth_at_enqueue": _numeric_summary([]),
            "queue_depth_at_start": _numeric_summary([]),
        }
    requested = len(records)
    started_rows = [row for row in records if row.get("queue_start_s") is not None]
    committed = [row for row in records if row.get("status") == "COMMITTED"]
    cancelled = [row for row in records if row.get("status") == "CANCELLED"]
    return {
        "applicable": True,
        "requested_n": requested,
        "started": rate_record(len(started_rows), requested),
        "committed": rate_record(len(committed), requested),
        "cancelled": rate_record(len(cancelled), requested),
        "detector_service_replay_latency": _latency_seconds(
            row["service_time_s"]
            for row in records
            if row.get("service_time_s") is not None
        ),
        "queue_wait_latency": _latency_seconds(
            row["queue_wait_s"]
            for row in started_rows
            if row.get("queue_wait_s") is not None
        ),
        "integrated_queue_to_commit_latency": _latency_seconds(
            row["integrated_latency_s"]
            for row in committed
            if row.get("integrated_latency_s") is not None
        ),
        "queue_depth_at_enqueue": _numeric_summary(
            int(row["queue_depth_at_enqueue"]) for row in records
        ),
        "queue_depth_at_start": _numeric_summary(
            int(row["queue_depth_at_start"])
            for row in started_rows
            if row.get("queue_depth_at_start") is not None
        ),
    }


def _runtime_group_summary(
    cells: Sequence[Mapping[str, Any]],
    *,
    landing_by_key: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _require(bool(cells), "runtime group is empty")
    first = cells[0]
    baseline = str(first["baseline"])
    arm = str(first["arm"])
    poison = _rows_for_role(cells, "poison")
    clean = _rows_for_role(cells, "clean")
    return {
        "cell_n": len(cells),
        "poison": _poison_summary(
            poison,
            baseline=baseline,
            arm=arm,
            landing_by_key=landing_by_key,
        ),
        "clean": _clean_summary(clean, baseline=baseline, arm=arm),
        "E_u_unvetted_visibility": _eu_summary(poison, baseline=baseline),
        "retrieval": _retrieval_summary(cells),
        "queue_and_integrated_latency": _queue_summary(
            cells, baseline=baseline
        ),
    }


def _runtime_section(
    cells: Sequence[Mapping[str, Any]],
    *,
    landing_by_key: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    by_seed: dict[tuple[str, str, str, str, int], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    across_seed: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for cell in cells:
        key5 = (
            str(cell["attack_family"]),
            str(cell["arm"]),
            str(cell["baseline"]),
            str(cell["backlog"]),
            int(cell["seed"]),
        )
        by_seed[key5].append(cell)
        across_seed[key5[:4]].append(cell)
    _require(len(by_seed) == 140, "runtime does not have 140 exact cell strata")
    _require(len(across_seed) == 28, "runtime does not have 28 across-seed strata")
    return {
        "by_attack_arm_baseline_backlog_seed": {
            "/".join(map(str, key)): _runtime_group_summary(
                rows, landing_by_key=landing_by_key
            )
            for key, rows in sorted(by_seed.items())
        },
        "across_five_seeds": {
            "/".join(map(str, key)): _runtime_group_summary(
                rows, landing_by_key=landing_by_key
            )
            for key, rows in sorted(across_seed.items())
        },
        "replay_denominator_warning": (
            "the same frozen items recur across baselines, backlogs, and arms; "
            "these episodes do not enlarge the 512-item detector denominator"
        ),
    }


def _cell_pair_key(cell: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(cell["attack_family"]),
        str(cell["baseline"]),
        str(cell["backlog"]),
        int(cell["seed"]),
    )


def _paired_retrieval_difference(
    detector: Mapping[str, Any], oracle: Mapping[str, Any]
) -> dict[str, Any]:
    d = _retrieval_summary([detector])["query_roles"]
    o = _retrieval_summary([oracle])["query_roles"]
    return {
        role: {
            "detector_query_denominator": d[role]["query_event_n"],
            "oracle_query_denominator": o[role]["query_event_n"],
            "overall_hit_n_detector_minus_oracle": (
                d[role]["poisoned_query_events_overall"]["numerator"]
                - o[role]["poisoned_query_events_overall"]["numerator"]
            ),
            "landed_only_hit_n_detector_minus_oracle": (
                d[role]["poisoned_query_events_landed_only"]["numerator"]
                - o[role]["poisoned_query_events_landed_only"]["numerator"]
            ),
            "displaced_positions_detector_minus_oracle": (
                d[role]["cumulative_displaced_top5_positions"]
                - o[role]["cumulative_displaced_top5_positions"]
            ),
        }
        for role in QUERY_ROLES
    }


def _paired_contrast(
    detector: Mapping[str, Any], oracle: Mapping[str, Any]
) -> dict[str, Any]:
    d_items, d_lifecycle, d_provider = _cell_maps(detector)
    o_items, o_lifecycle, o_provider = _cell_maps(oracle)
    _require(set(d_items) == set(o_items), "paired arms have different item keys")
    item_differences: list[dict[str, Any]] = []
    for key in sorted(d_items):
        _require(
            d_items[key]["role"] == o_items[key]["role"],
            f"paired role differs for {key}",
        )
        d_life = d_lifecycle[key]
        o_life = o_lifecycle[key]
        d_record = d_provider.get(key)
        o_record = o_provider.get(key)
        integrated_delta = None
        if (
            d_record is not None
            and o_record is not None
            and d_record.get("status") == "COMMITTED"
            and o_record.get("status") == "COMMITTED"
        ):
            integrated_delta = float(d_record["integrated_latency_s"]) - float(
                o_record["integrated_latency_s"]
            )
        item_differences.append(
            {
                "item_key": key,
                "role": str(d_items[key]["role"]),
                "restricted_exposure_s_detector_minus_oracle": (
                    _observed_or_zero(d_life, "exposure_observed_s")
                    - _observed_or_zero(o_life, "exposure_observed_s")
                ),
                "restricted_unavailable_s_detector_minus_oracle": (
                    _observed_or_zero(d_life, "unavailable_observed_s")
                    - _observed_or_zero(o_life, "unavailable_observed_s")
                ),
                "E_u_observed_s_detector_minus_oracle": (
                    _observed_or_zero(
                        d_life, "unvetted_visibility_observed_s"
                    )
                    - _observed_or_zero(
                        o_life, "unvetted_visibility_observed_s"
                    )
                ),
                "integrated_latency_s_detector_minus_oracle": integrated_delta,
                "detector_provider_status": (
                    d_record.get("status") if d_record is not None else None
                ),
                "oracle_provider_status": (
                    o_record.get("status") if o_record is not None else None
                ),
                "detector_arm_passes": (
                    d_record.get("passes") if d_record is not None else None
                ),
                "oracle_arm_passes": (
                    o_record.get("passes") if o_record is not None else None
                ),
            }
        )
    poison = [row for row in item_differences if row["role"] == "poison"]
    clean = [row for row in item_differences if row["role"] == "clean"]
    false_negative_poison = [
        row
        for row in poison
        if bool(d_items[row["item_key"]]["detector_promote"])
    ]
    false_positive_clean = [
        row
        for row in clean
        if not bool(d_items[row["item_key"]]["detector_promote"])
    ]
    latency_deltas = [
        row["integrated_latency_s_detector_minus_oracle"]
        for row in item_differences
        if row["integrated_latency_s_detector_minus_oracle"] is not None
    ]
    ep_total = sum(
        row["restricted_exposure_s_detector_minus_oracle"]
        for row in false_negative_poison
    )
    fp_unavailable_total = sum(
        row["restricted_unavailable_s_detector_minus_oracle"]
        for row in false_positive_clean
    )
    return {
        "attack_family": str(detector["attack_family"]),
        "baseline": str(detector["baseline"]),
        "backlog": str(detector["backlog"]),
        "seed": int(detector["seed"]),
        "item_differences": item_differences,
        "poison_item_n": len(poison),
        "poison_restricted_exposure_delta_total_s": sum(
            row["restricted_exposure_s_detector_minus_oracle"] for row in poison
        ),
        "E_p_false_negative_item_n": len(false_negative_poison),
        "E_p_additional_restricted_exposure_total_s": ep_total,
        "E_p_additional_restricted_exposure_per_false_negative_s": (
            ep_total / len(false_negative_poison)
            if false_negative_poison
            else None
        ),
        "clean_item_n": len(clean),
        "clean_restricted_unavailable_delta_total_s": sum(
            row["restricted_unavailable_s_detector_minus_oracle"] for row in clean
        ),
        "false_positive_clean_item_n": len(false_positive_clean),
        "false_positive_additional_unavailable_total_s": fp_unavailable_total,
        "false_positive_additional_unavailable_per_item_s": (
            fp_unavailable_total / len(false_positive_clean)
            if false_positive_clean
            else None
        ),
        "E_u_delta_total_s": sum(
            row["E_u_observed_s_detector_minus_oracle"] for row in poison
        ),
        "integrated_latency_delta_s": _numeric_summary(latency_deltas),
        "retrieval_difference": _paired_retrieval_difference(detector, oracle),
        "pairing_rule": "same seed, attack, baseline, backlog, item_key and role",
    }


def _paired_section(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    arms: dict[tuple[str, str, str, int], dict[str, Mapping[str, Any]]] = (
        defaultdict(dict)
    )
    for cell in cells:
        if cell.get("baseline") == "B1":
            continue
        arms[_cell_pair_key(cell)][str(cell["arm"])] = cell
    contrasts = []
    for key, pair in sorted(arms.items()):
        _require(
            set(pair) == {"detector", "oracle"},
            f"paired cell {key} lacks detector/oracle arm",
        )
        contrasts.append(_paired_contrast(pair["detector"], pair["oracle"]))
    _require(len(contrasts) == 60, "expected 60 detector-oracle cell contrasts")
    return {
        "contrast_n": len(contrasts),
        "contrasts": contrasts,
        "inference_warning": (
            "paired descriptive contrasts only; five seeds and repeated items "
            "do not justify treating replay episodes as independent samples"
        ),
    }


def analyze_bundle(
    *,
    manifest_path: str | os.PathLike[str],
    results_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Verify, recompute, and exclusive-create the W2D analysis artifact."""

    # This must remain the first operation: no partial or failed bundle is read.
    import verify_w2d

    verify_w2d.verify_or_raise(manifest_path, results_path)

    manifest_file = Path(manifest_path)
    results_file = Path(results_path)
    output_file = Path(output_path)
    manifest = _mapping(strict_json_load(str(manifest_file)), "manifest")
    result = _mapping(strict_json_load(str(results_file)), "results")
    paths = _artifact_paths(manifest_file, manifest)
    snapshot_before = _snapshot_hashes(
        manifest_path=manifest_file,
        results_path=results_file,
        artifact_paths=paths,
    )
    required = {"labels", "test_scores", "source_controls", "landing"}
    _require(
        required <= set(paths),
        f"manifest lacks analysis artifacts {sorted(required - set(paths))}",
    )
    labels_document = _mapping(strict_json_load(str(paths["labels"])), "labels")
    test_scores = _mapping(
        strict_json_load(str(paths["test_scores"])), "test_scores"
    )
    source_controls = _mapping(
        strict_json_load(str(paths["source_controls"])), "source_controls"
    )
    landing = _mapping(strict_json_load(str(paths["landing"])), "landing")
    cells = [
        _mapping(value, f"results.cells[{index}]")
        for index, value in enumerate(_list(result.get("cells"), "results.cells"))
    ]
    landing_by_key = {
        str(row["item_key"]): row
        for row in (
            _mapping(value, f"landing.items[{index}]")
            for index, value in enumerate(
                _list(landing.get("items"), "landing.items")
            )
        )
    }

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "measurement_name": result["measurement_name"],
        "verified_before_analysis": True,
        "verified_again_before_write": True,
        "provenance": {
            "manifest_sha256": snapshot_before["manifest"],
            "results_sha256": snapshot_before["results"],
            "analyzer_sha256": file_sha256(str(Path(__file__).resolve())),
            "artifact_sha256": {
                name: snapshot_before[f"artifact:{name}"]
                for name in sorted(paths)
            },
        },
        "detector_quality": _detector_section(test_scores, labels_document),
        "source_family_controls": source_control_gate_summary(
            source_controls, labels_document
        ),
        "landing": _landing_section(landing, cells),
        "runtime": _runtime_section(cells, landing_by_key=landing_by_key),
        "paired_detector_minus_oracle": _paired_section(cells),
        "statistical_contract": {
            "rate_interval": (
                "Wilson 95% with explicit numerator and denominator"
            ),
            "censoring": (
                "NOT_STARTED is not zero; RIGHT_CENSORED contributes observed "
                "age only to fixed-horizon restricted totals"
            ),
            "survival": (
                "no ordinary duration median and no implied containment for "
                "open episodes; no post-hoc survival model"
            ),
            "independence": (
                "detector and landing denominators are unique source groups; "
                "runtime cells and query ticks are paired/correlated replay"
            ),
            "latency": (
                "detector-only score-once latency is separate from replay "
                "queue wait and queue-entry-to-commit latency"
            ),
        },
    }
    _require(
        _snapshot_hashes(
            manifest_path=manifest_file,
            results_path=results_file,
            artifact_paths=paths,
        )
        == snapshot_before,
        "formal bundle changed while analysis was being computed",
    )
    verify_w2d.verify_or_raise(manifest_path, results_path)
    _require(
        _snapshot_hashes(
            manifest_path=manifest_file,
            results_path=results_file,
            artifact_paths=paths,
        )
        == snapshot_before,
        "formal bundle changed across the final verification",
    )
    _exclusive_json_dump(artifact, output_file)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    artifact = analyze_bundle(
        manifest_path=args.manifest,
        results_path=args.results,
        output_path=args.output,
    )
    print(
        f"Wrote {args.output}: "
        f"{artifact['detector_quality']['unique_item_count']} unique scores, "
        f"{len(artifact['runtime']['by_attack_arm_baseline_backlog_seed'])} cells"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
