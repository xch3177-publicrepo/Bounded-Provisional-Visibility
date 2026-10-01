#!/usr/bin/env python3
"""Strict artifact I/O and preregistered W2D detector summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence


Z_95 = 1.959963984540054
TEST_ITEM_COUNT = 512
SOURCE_CONTROL_ITEM_COUNT = 24
KEY_RE = re.compile(r"^[0-9a-f]{64}$")
FINAL_SCORE_KEYS = frozenset(
    {
        "item_key",
        "family_s_affirms",
        "c_score",
        "promote",
        "detector_service_ns",
    }
)


class MetricsError(ValueError):
    pass


def _reject_constant(token: str) -> None:
    raise MetricsError(f"non-standard JSON constant {token!r}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MetricsError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def strict_json_load_bytes(payload: bytes, *, source: str = "<bytes>") -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MetricsError(f"{source} is not UTF-8 JSON") from exc
    try:
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except json.JSONDecodeError as exc:
        raise MetricsError(f"{source} is not valid JSON: {exc}") from exc
    _validate_finite(value)
    return value


def strict_json_load(path: str) -> Any:
    with open(path, "rb") as handle:
        payload = handle.read()
    return strict_json_load_bytes(payload, source=path)


def _validate_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MetricsError(f"{path} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_finite(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_finite(child, f"{path}[{index}]")
        return
    raise MetricsError(f"{path} contains unsupported type {type(value).__name__}")


def strict_json_dump(value: Any, path: str) -> None:
    """Write one strict JSON artifact, refusing to replace an existing file."""
    _validate_finite(value)
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
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    created = False
    try:
        with open(path, "xb") as handle:
            created = True
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        if created:
            try:
                os.unlink(path)
            except OSError:
                pass
        raise


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float] | None:
    if not isinstance(successes, int) or not isinstance(total, int):
        raise MetricsError("Wilson counts must be integers")
    if total < 0 or successes < 0 or successes > total:
        raise MetricsError(f"invalid Wilson counts {successes}/{total}")
    if total == 0:
        return None
    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = (p + z2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total))
        / denominator
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def rate_record(successes: int, total: int) -> dict[str, Any]:
    interval = wilson_interval(successes, total)
    if total == 0:
        return {
            "defined": False,
            "numerator": successes,
            "denominator": total,
            "rate": None,
            "wilson95": None,
        }
    return {
        "defined": True,
        "numerator": successes,
        "denominator": total,
        "rate": successes / total,
        "wilson95": list(interval),
    }


def confusion_summary(tp: int, fp: int, tn: int, fn: int) -> dict[str, Any]:
    for name, value in {"tp": tp, "fp": fp, "tn": tn, "fn": fn}.items():
        if not isinstance(value, int) or value < 0:
            raise MetricsError(f"{name} must be a non-negative integer")
    return {
        "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": rate_record(tp, tp + fp),
        "recall": rate_record(tp, tp + fn),
        "f1": rate_record(2 * tp, 2 * tp + fp + fn),
        "fpr": rate_record(fp, fp + tn),
        "fnr": rate_record(fn, fn + tp),
    }


def empirical_quantile(values: Sequence[float], probability: float) -> float:
    if not 0.0 <= probability <= 1.0:
        raise MetricsError("quantile probability must be between zero and one")
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise MetricsError("cannot take a quantile of no observations")
    if not all(math.isfinite(value) for value in ordered):
        raise MetricsError("latency contains a non-finite observation")
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def latency_summary(values_ns: Sequence[int | float]) -> dict[str, Any]:
    values = [float(value) for value in values_ns]
    if not values:
        return {"count": 0, "p50_ns": None, "p95_ns": None, "p99_ns": None}
    if any(value < 0 or not math.isfinite(value) for value in values):
        raise MetricsError("latencies must be finite and non-negative")
    return {
        "count": len(values),
        "p50_ns": empirical_quantile(values, 0.50),
        "p95_ns": empirical_quantile(values, 0.95),
        "p99_ns": empirical_quantile(values, 0.99),
    }


def _label_map(value: Any) -> dict[str, Mapping[str, Any]]:
    if isinstance(value, Mapping) and isinstance(value.get("items"), (Mapping, list)):
        value = value["items"]
    if isinstance(value, Mapping) and isinstance(value.get("labels"), Mapping):
        value = value["labels"]
    if isinstance(value, list):
        result = {}
        for record in value:
            if not isinstance(record, Mapping):
                raise MetricsError("evaluator item is not a mapping")
            key = str(record.get("item_key", ""))
            if not key or key in result:
                raise MetricsError(f"missing or duplicate evaluator key {key!r}")
            result[key] = record
        return result
    if not isinstance(value, Mapping):
        raise MetricsError("label artifact must be a mapping")
    result = {}
    for key, record in value.items():
        if not isinstance(record, Mapping):
            raise MetricsError(f"label for {key} is not a mapping")
        result[str(key)] = record
    return result


def _score_records(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        value = value.get("items", value.get("scores"))
    if not isinstance(value, list):
        raise MetricsError("score artifact must contain an items list")
    if not all(isinstance(record, Mapping) for record in value):
        raise MetricsError("every score record must be a mapping")
    return value


def _require_score_artifact(
    value: Any, *, phase: str, require_provenance: bool
) -> list[Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise MetricsError("score artifact must be a mapping")
    if value.get("schema_version") != "W2D-score-v1":
        raise MetricsError("unexpected score artifact schema")
    if value.get("phase") != phase:
        raise MetricsError(f"expected a {phase} score artifact")
    if require_provenance:
        for field in (
            "safe_inputs_sha256",
            "embeddings_sha256",
            "detector_py_sha256",
            "scorer_py_sha256",
            "threshold_artifact_sha256",
        ):
            if not KEY_RE.fullmatch(str(value.get(field, ""))):
                raise MetricsError(f"score artifact has no valid {field}")
    records = _score_records(value)
    for record in records:
        actual = set(record)
        if actual != FINAL_SCORE_KEYS:
            raise MetricsError(
                "final score record schema mismatch; "
                f"missing={sorted(FINAL_SCORE_KEYS - actual)}, "
                f"extra={sorted(actual - FINAL_SCORE_KEYS)}"
            )
        if not isinstance(record.get("family_s_affirms"), bool):
            raise MetricsError("family_s_affirms must be bool")
        if not isinstance(record.get("promote"), bool):
            raise MetricsError("promote must be bool")
        score = record.get("c_score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise MetricsError("c_score must be finite and numeric")
        latency = record.get("detector_service_ns")
        if isinstance(latency, bool) or not isinstance(latency, int) or latency < 0:
            raise MetricsError("detector_service_ns must be a non-negative integer")
    return records


def _truth(label: Mapping[str, Any], key: str) -> bool:
    value = label.get("poison", label.get("is_poison"))
    if not isinstance(value, bool):
        raise MetricsError(f"label {key} poison must be bool")
    return value


def _detector_quality_summary(
    score_artifact: Any,
    label_artifact: Any,
    *,
    expected_item_count: int | None,
    require_provenance: bool,
) -> dict[str, Any]:
    """Join once-frozen scores to evaluator-only records and report all rates."""
    scores = _require_score_artifact(
        score_artifact, phase="test", require_provenance=require_provenance
    )
    labels = _label_map(label_artifact)
    expected_keys = {
        key for key, label in labels.items() if label.get("split") == "test"
    }
    if expected_item_count is not None and len(expected_keys) != expected_item_count:
        raise MetricsError(
            f"labels contain {len(expected_keys)} test items; "
            f"expected {expected_item_count}"
        )
    if len(scores) != len(expected_keys):
        raise MetricsError(
            f"score artifact has {len(scores)} items but labels freeze "
            f"{len(expected_keys)} test items"
        )
    seen_keys: set[str] = set()
    seen_groups: set[str] = set()
    joined: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for score in scores:
        key = str(score.get("item_key", ""))
        if not key or key in seen_keys:
            raise MetricsError(f"missing or duplicate score item_key {key!r}")
        seen_keys.add(key)
        if key not in labels:
            raise MetricsError(f"score {key} has no evaluator record")
        label = labels[key]
        group = str(label.get("source_group", ""))
        if not group:
            raise MetricsError(f"evaluator record {key} has no source_group")
        if group in seen_groups:
            raise MetricsError(
                f"source group {group} appears more than once in detector denominator"
            )
        seen_groups.add(group)
        joined.append((score, label))
    if seen_keys != expected_keys:
        raise MetricsError(
            "test score keys do not exactly equal the frozen test denominator"
        )

    def counts(rows: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> tuple[int, int, int, int]:
        tp = fp = tn = fn = 0
        for score, label in rows:
            actual = _truth(label, str(score["item_key"]))
            predicted = not score["promote"]
            if actual and predicted:
                tp += 1
            elif not actual and predicted:
                fp += 1
            elif not actual and not predicted:
                tn += 1
            else:
                fn += 1
        return tp, fp, tn, fn

    overall = confusion_summary(*counts(joined))

    clean_fpr: dict[str, Any] = {}
    for name, matcher in {
        "ordinary": lambda text: "ordinary" in text and "clean" in text,
        "hard_negative": lambda text: "hard" in text and "negative" in text,
    }.items():
        rows = [
            pair
            for pair in joined
            if not _truth(pair[1], str(pair[0]["item_key"]))
            and matcher(str(pair[1].get("stratum", "")).lower().replace("-", "_"))
        ]
        _tp, fp, tn, _fn = counts(rows)
        clean_fpr[name] = rate_record(fp, fp + tn)

    attack_recall: dict[str, Any] = {}
    families = sorted(
        {
            str(label.get("attack_family"))
            for _score, label in joined
            if _truth(label, str(_score["item_key"]))
            and label.get("attack_family") is not None
        }
    )
    for family in families:
        rows = [
            pair
            for pair in joined
            if _truth(pair[1], str(pair[0]["item_key"]))
            and str(pair[1].get("attack_family")) == family
        ]
        tp, _fp, _tn, fn = counts(rows)
        attack_recall[family] = rate_record(tp, tp + fn)

    result = {
        "unique_item_count": len(joined),
        "unique_source_group_count": len(seen_groups),
        "overall": overall,
        "clean_fpr_by_stratum": clean_fpr,
        "attack_recall": attack_recall,
        "detector_only_latency": latency_summary(
            [score["detector_service_ns"] for score, _label in joined]
        ),
    }
    _validate_finite(result)
    return result


def detector_quality_summary(score_artifact: Any, label_artifact: Any) -> dict[str, Any]:
    """Formal 512-item test summary."""
    return _detector_quality_summary(
        score_artifact,
        label_artifact,
        expected_item_count=TEST_ITEM_COUNT,
        require_provenance=True,
    )


def detector_quality_summary_test_fixture(
    score_artifact: Any, label_artifact: Any
) -> dict[str, Any]:
    """Named small-fixture entry point; exact key equality remains enforced."""
    return _detector_quality_summary(
        score_artifact,
        label_artifact,
        expected_item_count=None,
        require_provenance=False,
    )


def source_control_gate_summary(
    score_artifact: Any, label_artifact: Any
) -> dict[str, Any]:
    """Evaluate the 24 implementation-only Family-S negative controls."""
    scores = _require_score_artifact(
        score_artifact, phase="source_control", require_provenance=True
    )
    labels = _label_map(label_artifact)
    expected_keys = {
        key
        for key, label in labels.items()
        if label.get("split") == "implementation_control"
    }
    if len(expected_keys) != SOURCE_CONTROL_ITEM_COUNT:
        raise MetricsError(
            f"labels contain {len(expected_keys)} source controls; "
            f"expected {SOURCE_CONTROL_ITEM_COUNT}"
        )
    if len(scores) != SOURCE_CONTROL_ITEM_COUNT:
        raise MetricsError(
            f"source-control score artifact has {len(scores)} items; "
            f"expected {SOURCE_CONTROL_ITEM_COUNT}"
        )
    by_key = {str(record["item_key"]): record for record in scores}
    if len(by_key) != len(scores) or set(by_key) != expected_keys:
        raise MetricsError(
            "source-control score keys do not exactly equal the frozen controls"
        )
    kind_counts: dict[str, int] = {}
    affirmed = promoted = 0
    for key in sorted(expected_keys):
        label = labels[key]
        if _truth(label, key):
            raise MetricsError(f"source control {key} is not labelled clean")
        kind = label.get("source_control_kind")
        if kind not in {
            "invalid_signature",
            "provenance_conflict",
            "unknown_source",
        }:
            raise MetricsError(f"source control {key} has invalid kind {kind!r}")
        kind_counts[str(kind)] = kind_counts.get(str(kind), 0) + 1
        affirmed += int(by_key[key]["family_s_affirms"])
        promoted += int(by_key[key]["promote"])
    expected_kind_counts = {
        "invalid_signature": 8,
        "provenance_conflict": 8,
        "unknown_source": 8,
    }
    if kind_counts != expected_kind_counts:
        raise MetricsError(
            f"source-control kinds {kind_counts} != {expected_kind_counts}"
        )
    return {
        "passed": affirmed == 0 and promoted == 0,
        "total": SOURCE_CONTROL_ITEM_COUNT,
        "family_s_refused_n": SOURCE_CONTROL_ITEM_COUNT - affirmed,
        "family_s_affirmed_n": affirmed,
        "promoted_n": promoted,
        "kind_counts": kind_counts,
    }


def restricted_time_summary(
    observed_times: Sequence[float], censored: Sequence[bool], horizon: float
) -> dict[str, Any]:
    if len(observed_times) != len(censored):
        raise MetricsError("time and censoring vectors differ in length")
    horizon = float(horizon)
    if not math.isfinite(horizon) or horizon <= 0:
        raise MetricsError("horizon must be finite and positive")
    times = [float(value) for value in observed_times]
    if any(not math.isfinite(value) or value < 0 or value > horizon for value in times):
        raise MetricsError("restricted time lies outside the observation horizon")
    return {
        "started_n": len(times),
        "right_censored_n": sum(bool(value) for value in censored),
        "restricted_total": sum(times),
        "restricted_per_started": statistics.fmean(times) if times else None,
        "horizon": horizon,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument(
        "--phase", choices=("test", "source_control"), default="test"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    scores = strict_json_load(args.scores)
    labels = strict_json_load(args.labels)
    if args.phase == "test":
        artifact = {
            "schema_version": "W2D-detector-metrics-v1",
            "scores_sha256": file_sha256(args.scores),
            "labels_sha256": file_sha256(args.labels),
            "metrics": detector_quality_summary(scores, labels),
        }
    else:
        artifact = {
            "schema_version": "W2D-source-control-gate-v1",
            "scores_sha256": file_sha256(args.scores),
            "labels_sha256": file_sha256(args.labels),
            "gate": source_control_gate_summary(scores, labels),
        }
    strict_json_dump(artifact, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
