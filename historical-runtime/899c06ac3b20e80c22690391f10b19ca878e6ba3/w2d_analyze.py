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
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from w2d_metrics import (
    detector_quality_summary,
    empirical_quantile,
    latency_summary,
    rate_record,
    source_control_gate_summary,
)
from w2d_snapshot import (
    OutputOwnership,
    OutputOwnershipError,
    SnapshotError,
    assert_owned_output,
    exclusive_create_bytes,
    preserve_failed_output,
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


def _exclusive_json_dump(
    value: Any,
    path: str | os.PathLike[str],
) -> OutputOwnership:
    _finite_tree(value)
    payload = (
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return exclusive_create_bytes(path, payload)


def _assert_verified_inputs_unchanged(snapshot: Any, message: str) -> None:
    try:
        snapshot.registry.assert_unchanged()
    except (AttributeError, SnapshotError) as exc:
        raise AnalysisError(message) from exc


def _reverify_same_snapshot(
    verifier: Any,
    snapshot: Any,
    message: str,
) -> None:
    try:
        returned = verifier.verify_snapshot_or_raise(snapshot)
        _require(
            returned is snapshot,
            "formal verifier replaced the captured byte context",
        )
        snapshot.registry.assert_unchanged()
    except AnalysisError:
        raise
    except Exception as exc:
        raise AnalysisError(message) from exc


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

    overall = dict(_mapping(formal["overall"], "detector overall"))
    f1_record = _mapping(overall["f1"], "detector overall.f1")
    overall["f1"] = {
        "defined": bool(f1_record["defined"]),
        "point_estimate": f1_record["rate"],
        "estimator": "2*TP/(2*TP+FP+FN)",
        "interval": None,
        "interval_policy": (
            "descriptive harmonic-mean point estimate only; no binomial "
            "Wilson interval is defined for F1"
        ),
    }

    return {
        "unique_item_count": formal["unique_item_count"],
        "unique_source_group_count": formal["unique_source_group_count"],
        "overall": overall,
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


def _landed_rows(
    rows: Sequence[Mapping[str, Any]],
    landing_by_key: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    landed: list[Mapping[str, Any]] = []
    for row in rows:
        key = str(row["item"]["item_key"])
        landing = _mapping(landing_by_key.get(key), f"landing {key}")
        if bool(landing.get("landed_top5")):
            landed.append(row)
    return landed


def _poison_summary_view(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline: str,
    arm: str,
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


def _poison_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline: str,
    arm: str,
    landing_by_key: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    landed = _landed_rows(rows, landing_by_key)
    return {
        "overall": _poison_summary_view(
            rows,
            baseline=baseline,
            arm=arm,
        ),
        "landed_only": _poison_summary_view(
            landed,
            baseline=baseline,
            arm=arm,
        ),
        "landing_filter": (
            "landed_only contains exactly frozen landing records with "
            "landed_top5=true; failed landings remain in overall"
        ),
        "selected_item_landing": rate_record(len(landed), len(rows)),
    }


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


def _eu_summary_view(
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


def _eu_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline: str,
    landing_by_key: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    landed = _landed_rows(rows, landing_by_key)
    return {
        "overall": _eu_summary_view(rows, baseline=baseline),
        "landed_only": _eu_summary_view(landed, baseline=baseline),
        "landing_filter": (
            "landed_only contains exactly frozen landing records with "
            "landed_top5=true; failed landings remain in overall"
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


def _decision_commit_to_state_transition_latency(
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    elapsed_s: list[float] = []
    for cell_index, cell in enumerate(cells):
        cell_name = str(cell.get("plan_cell_id", f"cell[{cell_index}]"))
        lifecycle_by_key: dict[str, Mapping[str, Any]] = {}
        for row_index, raw in enumerate(
            _list(cell.get("lifecycle"), f"{cell_name}.lifecycle")
        ):
            row = _mapping(raw, f"{cell_name}.lifecycle[{row_index}]")
            item_key = row.get("item_key")
            _require(
                isinstance(item_key, str)
                and bool(item_key)
                and item_key not in lifecycle_by_key,
                f"{cell_name} has a missing or duplicate lifecycle item_key",
            )
            lifecycle_by_key[item_key] = row

        committed_keys: set[str] = set()
        for record_index, raw in enumerate(
            _list(
                cell.get("verifier_records"),
                f"{cell_name}.verifier_records",
            )
        ):
            record = _mapping(
                raw,
                f"{cell_name}.verifier_records[{record_index}]",
            )
            if record.get("status") != "COMMITTED":
                continue
            item_key = record.get("item_key")
            _require(
                isinstance(item_key, str)
                and bool(item_key)
                and item_key not in committed_keys,
                f"{cell_name} has a missing or duplicate committed item_key",
            )
            committed_keys.add(item_key)
            lifecycle = lifecycle_by_key.get(item_key)
            _require(
                lifecycle is not None,
                f"{cell_name} committed item {item_key} has no lifecycle row",
            )

            decision_commit = record.get("decision_commit_s")
            _require(
                isinstance(decision_commit, (int, float))
                and not isinstance(decision_commit, bool)
                and math.isfinite(float(decision_commit)),
                f"{cell_name} committed item {item_key} has an invalid "
                "decision_commit_s",
            )
            passes = record.get("passes")
            _require(
                isinstance(passes, bool),
                f"{cell_name} committed item {item_key} has no Boolean verdict",
            )

            terminal_transitions: list[Mapping[str, Any]] = []
            for transition_index, raw_transition in enumerate(
                _list(
                    lifecycle.get("transitions"),
                    f"{cell_name}.lifecycle[{item_key}].transitions",
                )
            ):
                transition = _mapping(
                    raw_transition,
                    (
                        f"{cell_name}.lifecycle[{item_key}].transitions"
                        f"[{transition_index}]"
                    ),
                )
                if transition.get("new_state") in {
                    "TRUSTED",
                    "QUARANTINED",
                }:
                    terminal_transitions.append(transition)
            _require(
                len(terminal_transitions) == 1,
                f"{cell_name} committed item {item_key} must have exactly one "
                "TRUSTED/QUARANTINED lifecycle transition",
            )
            transition = terminal_transitions[0]
            expected_state = "TRUSTED" if passes else "QUARANTINED"
            _require(
                transition.get("new_state") == expected_state,
                f"{cell_name} committed item {item_key} lifecycle transition "
                "disagrees with the committed verdict",
            )
            transition_time = transition.get("t_s")
            _require(
                isinstance(transition_time, (int, float))
                and not isinstance(transition_time, bool)
                and math.isfinite(float(transition_time)),
                f"{cell_name} committed item {item_key} has an invalid "
                "lifecycle transition time",
            )
            elapsed = float(transition_time) - float(decision_commit)
            _require(
                elapsed >= 0.0,
                f"{cell_name} committed item {item_key} transitions before "
                "decision_commit_s",
            )
            elapsed_s.append(elapsed)
    elapsed_ns = [
        int(round(value * 1_000_000_000.0)) for value in elapsed_s
    ]
    summary = latency_summary(elapsed_ns)
    summary["max_ns"] = max(elapsed_ns) if elapsed_ns else None
    summary["above_superseded_10ms_diagnostic"] = rate_record(
        sum(value > 10_000_000 for value in elapsed_ns),
        len(elapsed_ns),
    )
    return summary


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
            "decision_commit_to_state_transition_latency": (
                _decision_commit_to_state_transition_latency([])
            ),
            "queue_depth_policy": (
                "raw producer counters retained only as diagnostics; no "
                "scientific summary without independent event-order replay"
            ),
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
        "decision_commit_to_state_transition_latency": (
            _decision_commit_to_state_transition_latency(cells)
        ),
        "queue_depth_policy": (
            "raw producer counters retained only as diagnostics; no "
            "scientific summary without independent event-order replay"
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
        "E_u_unvetted_visibility": _eu_summary(
            poison,
            baseline=baseline,
            landing_by_key=landing_by_key,
        ),
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
        role = str(d_items[key]["role"])
        d_life = d_lifecycle[key]
        o_life = o_lifecycle[key]
        d_record = d_provider.get(key)
        o_record = o_provider.get(key)
        d_committed = (
            d_record is not None and d_record.get("status") == "COMMITTED"
        )
        o_committed = (
            o_record is not None and o_record.get("status") == "COMMITTED"
        )
        if d_committed and o_committed:
            latency_pair_status = "both_committed"
        elif d_committed:
            latency_pair_status = "detector_only_committed"
        elif o_committed:
            latency_pair_status = "oracle_only_committed"
        else:
            latency_pair_status = "neither_committed"
        integrated_delta = None
        if d_committed and o_committed:
            integrated_delta = float(d_record["integrated_latency_s"]) - float(
                o_record["integrated_latency_s"]
            )
        detector_promote = bool(d_items[key]["detector_promote"])
        d1_classification_false_negative = (
            role == "poison" and detector_promote
        )
        d1_classification_false_positive = (
            role == "clean" and not detector_promote
        )
        realized_committed_false_promotion = (
            role == "poison"
            and d_committed
            and d_record.get("passes") is True
        )
        realized_committed_misquarantine = (
            role == "clean"
            and d_committed
            and d_record.get("passes") is False
            and d_life.get("quarantine_status") != "NOT_STARTED"
        )
        item_differences.append(
            {
                "item_key": key,
                "role": role,
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
                "integrated_latency_pair_status": latency_pair_status,
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
                "d1_classification_false_negative": (
                    d1_classification_false_negative
                ),
                "d1_classification_false_positive": (
                    d1_classification_false_positive
                ),
                "realized_committed_false_promotion": (
                    realized_committed_false_promotion
                ),
                "realized_committed_misquarantine": (
                    realized_committed_misquarantine
                ),
            }
        )
    poison = [row for row in item_differences if row["role"] == "poison"]
    clean = [row for row in item_differences if row["role"] == "clean"]
    d1_false_negative_poison = [
        row
        for row in poison
        if row["d1_classification_false_negative"]
    ]
    realized_false_promotion_poison = [
        row
        for row in poison
        if row["realized_committed_false_promotion"]
    ]
    d1_false_positive_clean = [
        row
        for row in clean
        if row["d1_classification_false_positive"]
    ]
    realized_misquarantine_clean = [
        row
        for row in clean
        if row["realized_committed_misquarantine"]
    ]
    latency_deltas = [
        row["integrated_latency_s_detector_minus_oracle"]
        for row in item_differences
        if row["integrated_latency_s_detector_minus_oracle"] is not None
    ]
    ep_total = sum(
        row["restricted_exposure_s_detector_minus_oracle"]
        for row in realized_false_promotion_poison
    )
    fp_unavailable_total = sum(
        row["restricted_unavailable_s_detector_minus_oracle"]
        for row in realized_misquarantine_clean
    )
    latency_pair_counts = Counter(
        str(row["integrated_latency_pair_status"])
        for row in item_differences
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
        "d1_classification_false_negative_item_n": len(
            d1_false_negative_poison
        ),
        "realized_committed_false_promotion_item_n": len(
            realized_false_promotion_poison
        ),
        "E_p_additional_restricted_exposure_total_s": ep_total,
        "E_p_additional_restricted_exposure_per_realized_false_promotion_s": (
            ep_total / len(realized_false_promotion_poison)
            if realized_false_promotion_poison
            else None
        ),
        "E_p_population": (
            "poison items with a detector-arm COMMITTED passes=true decision; "
            "D1 classification false negatives that were CANCELLED are excluded"
        ),
        "clean_item_n": len(clean),
        "clean_restricted_unavailable_delta_total_s": sum(
            row["restricted_unavailable_s_detector_minus_oracle"] for row in clean
        ),
        "d1_classification_false_positive_item_n": len(
            d1_false_positive_clean
        ),
        "realized_committed_misquarantine_item_n": len(
            realized_misquarantine_clean
        ),
        "false_positive_additional_unavailable_total_s": fp_unavailable_total,
        "false_positive_additional_unavailable_per_realized_misquarantine_s": (
            fp_unavailable_total / len(realized_misquarantine_clean)
            if realized_misquarantine_clean
            else None
        ),
        "false_positive_cost_population": (
            "clean items with a detector-arm COMMITTED passes=false decision "
            "and an observed quarantine episode; CANCELLED D1 false positives "
            "are excluded"
        ),
        "E_u_delta_total_s": sum(
            row["E_u_observed_s_detector_minus_oracle"] for row in poison
        ),
        "integrated_latency_pair_availability": {
            "both_committed_n": latency_pair_counts.get(
                "both_committed", 0
            ),
            "detector_only_committed_n": latency_pair_counts.get(
                "detector_only_committed", 0
            ),
            "oracle_only_committed_n": latency_pair_counts.get(
                "oracle_only_committed", 0
            ),
            "neither_committed_n": latency_pair_counts.get(
                "neither_committed", 0
            ),
            "total_item_n": len(item_differences),
        },
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

    verified = verify_w2d.verify_or_raise(manifest_path, results_path)
    output_file = Path(output_path)

    manifest = _mapping(verified.manifest, "manifest")
    result = _mapping(verified.result, "results")
    artifacts = _mapping(verified.artifacts, "verified artifacts")
    artifact_hashes = _mapping(
        verified.artifact_hashes,
        "verified artifact hashes",
    )
    required = {"labels", "test_scores", "source_controls", "landing"}
    _require(
        required <= set(artifacts),
        f"manifest lacks analysis artifacts "
        f"{sorted(required - set(artifacts))}",
    )
    labels_document = _mapping(artifacts["labels"], "labels")
    test_scores = _mapping(artifacts["test_scores"], "test_scores")
    source_controls = _mapping(artifacts["source_controls"], "source_controls")
    landing = _mapping(artifacts["landing"], "landing")
    try:
        analyzer_snapshot = verified.component_snapshots["analyzer"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise AnalysisError(
            "verified context omits the analyzer runtime snapshot"
        ) from exc
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
        "execution_mode": result["execution_mode"],
        "runtime_backend": dict(
            _mapping(result["runtime_backend"], "results.runtime_backend")
        ),
        "observation_horizon_s": result["observation_horizon_s"],
        "injection_at_s": result["injection_at_s"],
        "runtime_fingerprint_sha256": result[
            "runtime_fingerprint_sha256"
        ],
        "manifest_sha256": result["manifest_sha256"],
        "authority_sha256": result["authority_sha256"],
        "runtime_backend_evidence": {
            phase: dict(
                _mapping(
                    result["runtime_backend_evidence"][phase],
                    f"results.runtime_backend_evidence.{phase}",
                )
            )
            for phase in ("pre", "post")
        },
        "verified_before_analysis": True,
        "verified_again_before_write": True,
        "provenance": {
            "authority_sha256": verified.authority_snapshot.sha256,
            "manifest_sha256": verified.manifest_snapshot.sha256,
            "results_sha256": verified.result_snapshot.sha256,
            "analyzer_sha256": analyzer_snapshot.sha256,
            "artifact_sha256": dict(sorted(artifact_hashes.items())),
        },
        "detector_quality": _detector_section(test_scores, labels_document),
        "source_family_controls": source_control_gate_summary(
            source_controls, labels_document
        ),
        "landing": _landing_section(landing, cells),
        "runtime": _runtime_section(cells, landing_by_key=landing_by_key),
        "paired_detector_minus_oracle": _paired_section(cells),
        "statistical_contract": {
            "binomial_rate_interval": (
                "Wilson 95% only for directly count-based Bernoulli "
                "proportions with explicit numerator and denominator"
            ),
            "f1_interval": (
                "F1 is a descriptive harmonic-mean point estimate; no "
                "binomial Wilson interval is emitted"
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
    _assert_verified_inputs_unchanged(
        verified,
        "formal bundle changed while analysis was being computed",
    )
    _reverify_same_snapshot(
        verify_w2d,
        verified,
        "formal bundle changed across the final verification",
    )
    ownership: OutputOwnership | None = None
    try:
        ownership = _exclusive_json_dump(artifact, output_file)
        _assert_verified_inputs_unchanged(
            verified,
            "formal bundle changed after analysis output creation",
        )
        assert_owned_output(ownership)
    except BaseException as exc:
        if ownership is not None:
            try:
                preserve_failed_output(ownership)
            except OutputOwnershipError as conflict:
                raise conflict from exc
        raise
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
