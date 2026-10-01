#!/usr/bin/env python3
"""Calibration-only threshold selection and opaque phase-key manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from detector import ZStatistics
from w2d_metrics import (
    confusion_summary,
    file_sha256,
    strict_json_dump,
    strict_json_load,
)


class CalibrationError(ValueError):
    pass


CALIBRATION_ITEM_COUNT = 448
TEST_ITEM_COUNT = 512
SOURCE_CONTROL_ITEM_COUNT = 24
KEY_RE = re.compile(r"^[0-9a-f]{64}$")


def score_order_key(item_key: str) -> str:
    return hashlib.sha256(
        ("W2D-score-order|" + item_key).encode("utf-8")
    ).hexdigest()


def _labels(value: Any) -> dict[str, Mapping[str, Any]]:
    if isinstance(value, Mapping) and isinstance(value.get("items"), (Mapping, list)):
        value = value["items"]
    if isinstance(value, Mapping) and isinstance(value.get("labels"), Mapping):
        value = value["labels"]
    if isinstance(value, list):
        result: dict[str, Mapping[str, Any]] = {}
        for record in value:
            if not isinstance(record, Mapping):
                raise CalibrationError("evaluator item is not a mapping")
            key = str(record.get("item_key", ""))
            if not key or key in result:
                raise CalibrationError(f"missing or duplicate evaluator key {key!r}")
            result[key] = record
        return result
    if not isinstance(value, Mapping):
        raise CalibrationError("evaluator artifact must be a key-to-record mapping")
    result: dict[str, Mapping[str, Any]] = {}
    for key, record in value.items():
        if not isinstance(record, Mapping):
            raise CalibrationError(f"record for {key} is not a mapping")
        result[str(key)] = record
    return result


def _scores(
    value: Any, *, require_provenance: bool
) -> tuple[list[Mapping[str, Any]], ZStatistics]:
    if not isinstance(value, Mapping):
        raise CalibrationError("calibration score artifact must be a mapping")
    if value.get("schema_version") != "W2D-score-v1":
        raise CalibrationError("unexpected calibration score schema")
    records = value.get("items")
    if not isinstance(records, list) or not all(
        isinstance(record, Mapping) for record in records
    ):
        raise CalibrationError("calibration score artifact has no items list")
    if value.get("phase") != "calibration":
        raise CalibrationError("threshold may only use a calibration score artifact")
    if require_provenance:
        for field in (
            "safe_inputs_sha256",
            "embeddings_sha256",
            "detector_py_sha256",
            "scorer_py_sha256",
        ):
            if not KEY_RE.fullmatch(str(value.get(field, ""))):
                raise CalibrationError(
                    f"calibration score artifact has no valid {field}"
                )
    z_value = value.get("z_statistics", value.get("z_stats"))
    return records, ZStatistics.from_value(z_value)


def _phase(record: Mapping[str, Any]) -> str | None:
    explicit = record.get("split", record.get("phase", record.get("role")))
    if explicit is not None:
        text = str(explicit).strip().lower().replace("-", "_")
        if text in {"calibration", "cal", "train"}:
            return "calibration"
        if text in {"test", "testing"}:
            return "test"
        if text in {"reference", "implementation", "control"}:
            return None
    stratum = str(record.get("stratum", "")).strip().lower().replace("-", "_")
    if "calibration" in stratum or stratum.startswith("cal_"):
        return "calibration"
    if "test" in stratum:
        return "test"
    # A2 fixes the second attack as test-only and source controls as
    # implementation-only.  These names are used only to construct opaque key
    # manifests; the scorer never receives this record.
    attack = str(record.get("attack_family", "")).lower()
    if attack == "natural_cover_suffix_v1":
        return "test"
    if "source" in stratum and "control" in stratum:
        return None
    return None


def build_phase_key_manifests(label_artifact: Any) -> dict[str, list[str]]:
    labels = _labels(label_artifact)
    result = {"calibration": [], "test": [], "source_control": []}
    for key, record in labels.items():
        phase = _phase(record)
        if phase is not None:
            result[phase].append(key)
        elif record.get("split") == "implementation_control":
            result["source_control"].append(key)
    for phase in result:
        result[phase].sort(key=score_order_key if phase == "test" else None)
        if len(result[phase]) != len(set(result[phase])):
            raise CalibrationError(f"{phase} key manifest contains a duplicate")
        if any(not KEY_RE.fullmatch(key) for key in result[phase]):
            raise CalibrationError(f"{phase} manifest contains a non-opaque key")
    if not all(result.values()):
        raise CalibrationError(
            "could not derive non-empty calibration, test, and source-control manifests"
        )
    expected_counts = {
        "calibration": CALIBRATION_ITEM_COUNT,
        "test": TEST_ITEM_COUNT,
        "source_control": SOURCE_CONTROL_ITEM_COUNT,
    }
    for phase, expected in expected_counts.items():
        if len(result[phase]) != expected:
            raise CalibrationError(
                f"{phase} manifest has {len(result[phase])} keys; expected {expected}"
            )
    phases = list(result)
    for index, left in enumerate(phases):
        for right in phases[index + 1 :]:
            overlap = set(result[left]) & set(result[right])
            if overlap:
                raise CalibrationError(
                    f"{len(overlap)} keys cross {left}/{right}"
                )
    return result


def _finite_score(record: Mapping[str, Any]) -> float:
    try:
        value = float(record.get("c_score"))
    except (TypeError, ValueError) as exc:
        raise CalibrationError("calibration record lacks c_score") from exc
    if not math.isfinite(value):
        raise CalibrationError("calibration c_score is non-finite")
    return value


def threshold_candidates(scores: Sequence[float]) -> list[float]:
    distinct = sorted(set(float(score) for score in scores))
    if not distinct or not all(math.isfinite(score) for score in distinct):
        raise CalibrationError("threshold selection needs finite calibration scores")
    low = math.nextafter(distinct[0], -math.inf)
    high = math.nextafter(distinct[-1], math.inf)
    if not math.isfinite(low) or not math.isfinite(high):
        raise CalibrationError("finite extreme threshold could not be represented")
    candidates = [low]
    for left, right in zip(distinct, distinct[1:]):
        midpoint = left + (right - left) * 0.5
        if not math.isfinite(midpoint):
            midpoint = left * 0.5 + right * 0.5
        if not math.isfinite(midpoint):
            raise CalibrationError("interior threshold midpoint is non-finite")
        candidates.append(midpoint)
    candidates.append(high)
    return candidates


def _counts(
    scores_and_truth: Sequence[tuple[float, bool]], threshold: float
) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for score, actual in scores_and_truth:
        predicted = score >= threshold
        if actual and predicted:
            tp += 1
        elif not actual and predicted:
            fp += 1
        elif not actual and not predicted:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else 0.0


def _select_threshold(
    score_artifact: Any,
    label_artifact: Any,
    *,
    expected_item_count: int | None,
    require_provenance: bool,
) -> dict[str, Any]:
    records, z_statistics = _scores(
        score_artifact, require_provenance=require_provenance
    )
    labels = _labels(label_artifact)
    expected_keys = {
        key for key, label in labels.items() if _phase(label) == "calibration"
    }
    if expected_item_count is not None and len(expected_keys) != expected_item_count:
        raise CalibrationError(
            f"labels contain {len(expected_keys)} calibration items; "
            f"expected {expected_item_count}"
        )
    if len(records) != len(expected_keys):
        raise CalibrationError(
            f"score artifact has {len(records)} items but labels freeze "
            f"{len(expected_keys)} calibration items"
        )
    joined: list[tuple[float, bool]] = []
    seen_keys: set[str] = set()
    seen_groups: set[str] = set()
    canonical_pairs: list[dict[str, Any]] = []
    for record in records:
        key = str(record.get("item_key", ""))
        if not key or key in seen_keys:
            raise CalibrationError(f"missing or duplicate calibration key {key!r}")
        seen_keys.add(key)
        if key not in labels:
            raise CalibrationError(f"calibration score {key} has no evaluator record")
        label = labels[key]
        phase = _phase(label)
        if phase != "calibration":
            raise CalibrationError(f"non-calibration item {key} entered calibration")
        group = str(label.get("source_group", ""))
        if not group or group in seen_groups:
            raise CalibrationError(
                f"missing or repeated source group in calibration: {group!r}"
            )
        seen_groups.add(group)
        score = _finite_score(record)
        truth = label.get("poison", label.get("is_poison"))
        if not isinstance(truth, bool):
            raise CalibrationError(f"calibration label {key} poison must be bool")
        joined.append((score, truth))
        canonical_pairs.append({"item_key": key, "c_score": score, "poison": truth})

    if seen_keys != expected_keys:
        missing = sorted(expected_keys - seen_keys)
        extra = sorted(seen_keys - expected_keys)
        raise CalibrationError(
            f"calibration score keys differ from the frozen labels; "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    if not joined or not any(truth for _score, truth in joined):
        raise CalibrationError("calibration has no suspicious-class examples")
    if not any(not truth for _score, truth in joined):
        raise CalibrationError("calibration has no clean examples")

    best: tuple[float, float, tuple[int, int, int, int]] | None = None
    candidates = threshold_candidates([score for score, _truth in joined])
    for threshold in candidates:
        counts = _counts(joined, threshold)
        tp, fp, _tn, fn = counts
        candidate = (_f1(tp, fp, fn), threshold, counts)
        # Tuple ordering implements the preregistered largest-threshold tie-break.
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    _selected_f1, selected_threshold, counts = best
    pairs_blob = json.dumps(
        sorted(canonical_pairs, key=lambda row: row["item_key"]),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema_version": "W2D-threshold-v1",
        "selection_rule": "maximize poison-class F1; ties choose largest threshold",
        "selected_threshold": selected_threshold,
        "candidate_count": len(candidates),
        "calibration_item_count": len(joined),
        "calibration_source_group_count": len(seen_groups),
        "calibration_summary": confusion_summary(*counts),
        "z_statistics": z_statistics.to_dict(),
        "calibration_score_label_digest": hashlib.sha256(pairs_blob).hexdigest(),
    }


def select_threshold(score_artifact: Any, label_artifact: Any) -> dict[str, Any]:
    """Select the formal threshold from exactly 448 frozen calibration items."""
    return _select_threshold(
        score_artifact,
        label_artifact,
        expected_item_count=CALIBRATION_ITEM_COUNT,
        require_provenance=True,
    )


def select_threshold_test_fixture(
    score_artifact: Any, label_artifact: Any
) -> dict[str, Any]:
    """Named small-fixture entry point; exact key equality is still enforced."""
    return _select_threshold(
        score_artifact,
        label_artifact,
        expected_item_count=None,
        require_provenance=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    keys = subparsers.add_parser("keys", help="write opaque phase key manifests")
    keys.add_argument("--labels", required=True)
    keys.add_argument("--output", required=True)

    threshold = subparsers.add_parser(
        "threshold", help="select the one calibration-only threshold"
    )
    threshold.add_argument("--scores", required=True)
    threshold.add_argument("--labels", required=True)
    threshold.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    labels = strict_json_load(args.labels)
    if args.command == "keys":
        artifact = {
            "schema_version": "W2D-phase-keys-v1",
            "labels_sha256": file_sha256(args.labels),
            **build_phase_key_manifests(labels),
        }
    else:
        scores = strict_json_load(args.scores)
        artifact = select_threshold(scores, labels)
        artifact["calibration_scores_sha256"] = file_sha256(args.scores)
        artifact["labels_sha256"] = file_sha256(args.labels)
    strict_json_dump(artifact, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
