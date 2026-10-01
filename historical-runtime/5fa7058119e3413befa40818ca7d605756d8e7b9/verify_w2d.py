#!/usr/bin/env python3
"""Independent, fail-loud gates for a formal W2D replay bundle.

This verifier intentionally consumes only frozen artifacts plus the final
protocol result.  It never builds data, scores an item, or runs a protocol cell.
The runner may import the public constants and :func:`verify_bundle` before
publishing a result.

Usage:
    ./.venv312/bin/python verify_w2d.py \
        --manifest results/w2d/W2D-MANIFEST.json \
        --results results/w2d/W2D-PROTOCOL-RESULTS.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from w2d_calibrate import CalibrationError, select_threshold
from w2d_metrics import (
    MetricsError,
    detector_quality_summary,
    file_sha256,
    rate_record,
    source_control_gate_summary,
    strict_json_load,
)


MEASUREMENT_NAME = (
    "promotion-path replay driven by real D1 outputs and measured service times"
)
EXECUTION_MODE = "score_once_frozen_decision_service_replay"
FROZEN_SERVICE_REPLAY = "frozen_D1_item_service_time"
FROZEN_PROVIDER = "frozen_score_lookup"
RESULT_SCHEMA = "W2D-protocol-result-v1"
MANIFEST_SCHEMA = "W2D-manifest-v1"
EXPECTED_TEST_ITEMS = 512
EXPECTED_SOURCE_CONTROLS = 24
EXPECTED_NATURAL_COVER = 64
EXPECTED_RECIPE_TEST = 128
EXPECTED_RUNTIME_POISON_PER_FAMILY = 30
EXPECTED_RUNTIME_UNITS = 20
EXPECTED_PROTOCOL_CELLS = 140
EXPECTED_SEEDS = frozenset({1, 2, 3, 4, 5})
EXPECTED_CLEAN_STRATA = {
    1: {"ordinary_clean": 4, "hard_negative_clean": 2},
    2: {"ordinary_clean": 3, "hard_negative_clean": 3},
    3: {"ordinary_clean": 4, "hard_negative_clean": 2},
    4: {"ordinary_clean": 3, "hard_negative_clean": 3},
    5: {"ordinary_clean": 4, "hard_negative_clean": 2},
}
EXPECTED_FILLER_STRATA = {
    1: {"ordinary_clean": 7, "hard_negative_clean": 5},
    2: {"ordinary_clean": 7, "hard_negative_clean": 5},
    3: {"ordinary_clean": 7, "hard_negative_clean": 5},
    4: {"ordinary_clean": 7, "hard_negative_clean": 5},
    5: {"ordinary_clean": 8, "hard_negative_clean": 4},
}
OPAQUE_KEY_RE = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_ARTIFACTS = frozenset(
    {
        "data_freeze",
        "safe_inputs_json",
        "safe_inputs_npz",
        "labels",
        "protocol_plan",
        "calibration_scores",
        "threshold",
        "test_scores",
        "source_controls",
        "source_control_gate",
        "detector_metrics",
        "landing",
    }
)
REQUIRED_FINGERPRINT_COMPONENTS = frozenset(
    {
        "runtime",
        "detector",
        "workload",
        "protocol",
        "scorer",
        "calibrator",
        "metrics",
        "runner",
        "verifier",
        "model_revision",
    }
)
RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "measurement_name",
        "execution_mode",
        "runtime_fingerprint_sha256",
        "artifact_hashes",
        "detector_execution",
        "observation_horizon_s",
        "injection_at_s",
        "detector_denominator",
        "cells",
    }
)
CELL_FIELDS = frozenset(
    {
        "plan_cell_id",
        "attack_family",
        "arm",
        "baseline",
        "backlog",
        "seed",
        "protocol_items",
        "verifier_records",
        "lifecycle",
        "retrieval",
    }
)
PROTOCOL_ITEM_FIELDS = frozenset(
    {
        "item_key",
        "role",
        "ordinal",
        "detector_promote",
        "detector_service_ns",
        "score_record_sha256",
    }
)
LIFECYCLE_FIELDS = frozenset(
    {
        "item_key",
        "truth_poison",
        "detector_promote",
        "retry_count",
        "state_at_horizon",
        "visibility_status",
        "exposure_status",
        "exposure_observed_s",
        "unavailable_status",
        "unavailable_observed_s",
        "quarantine_status",
        "readmission_count",
        "observed_at_horizon_s",
    }
)
VERIFIER_RECORD_FIELDS = frozenset(
    {
        "item_key",
        "status",
        "passes",
        "service_time_s",
        "integrated_latency_s",
        "decision_commit_s",
    }
)
FINAL_STATES = frozenset(
    {"PROVISIONAL", "TRUSTED", "QUARANTINED", "HIDDEN", "REVOKED"}
)
EPISODE_STATUSES = frozenset(
    {"NOT_STARTED", "COMPLETED", "RIGHT_CENSORED", "NOT_APPLICABLE"}
)
PROVIDER_STATUSES = frozenset({"COMMITTED", "CANCELLED"})
CALIBRATION_SCORE_FIELDS = frozenset(
    {
        "item_key",
        "rep",
        "knn",
        "c_score",
        "family_s_affirms",
        "detector_service_ns",
    }
)
LANDING_FIELDS = frozenset(
    {
        "schema_version",
        "measurement",
        "top_k",
        "tie_break",
        "candidate_population",
        "protocol_plan_sha256",
        "safe_inputs_sha256",
        "embeddings_sha256",
        "retrieval_background_sha256",
        "populations",
        "items",
    }
)
LANDING_ITEM_FIELDS = frozenset(
    {
        "item_key",
        "attack_family",
        "rank",
        "candidate_rank",
        "landed_top5",
        "landed",
        "top5_item_keys",
    }
)


class W2DVerificationError(ValueError):
    """A formal bundle violated its frozen schema or a W2D gate."""


@dataclass
class VerificationReport:
    checks: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def passed(self, name: str) -> None:
        self.checks.append(name)

    def failed(self, message: str) -> None:
        self.failures.append(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise W2DVerificationError(message)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise W2DVerificationError(f"{path} must be a mapping")
    return value


def _sequence(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise W2DVerificationError(f"{path} must be a list")
    return value


def _exact_fields(value: Mapping[str, Any], expected: frozenset[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise W2DVerificationError(
            f"{path} schema mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _required_fields(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], path: str
) -> None:
    missing = set(expected) - set(value)
    if missing:
        raise W2DVerificationError(f"{path} missing required fields {sorted(missing)}")


def _sha256(value: Any, path: str) -> str:
    text = value if isinstance(value, str) else ""
    if not OPAQUE_KEY_RE.fullmatch(text):
        raise W2DVerificationError(f"{path} must be a lowercase SHA256")
    return text


def _item_key(value: Any, path: str) -> str:
    return _sha256(value, path)


def _strict_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise W2DVerificationError(f"{path} must be bool")
    return value


def _nonnegative_number(value: Any, path: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise W2DVerificationError(f"{path} must be finite and non-negative")
    return float(value)


def canonical_record_sha256(record: Mapping[str, Any]) -> str:
    """Digest one strict-JSON score record for replay back-references."""
    try:
        payload = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise W2DVerificationError(f"score record is not strict JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def runtime_fingerprint_digest(components: Mapping[str, Mapping[str, Any]]) -> str:
    """Canonical aggregate over named component digests.

    The aggregate contains names as well as hashes, so swapping two component
    files cannot preserve the fingerprint.
    """
    normalized = {
        str(name): _sha256(_mapping(record, f"component {name}").get("sha256"),
                           f"component {name}.sha256")
        for name, record in components.items()
    }
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_runtime_fingerprint(
    components: Mapping[str, Mapping[str, Any]], base_dir: str | Path
) -> dict[str, Any]:
    """Build (but do not write) the independent fingerprint record.

    This helper exists so the runner and verifier use exactly one aggregate
    definition.  File entries use ``{"path": ...}``; the model revision uses
    ``{"value": ...}``.
    """
    root = Path(base_dir)
    frozen: dict[str, dict[str, str]] = {}
    for name, raw in components.items():
        record = _mapping(raw, f"component {name}")
        has_path = isinstance(record.get("path"), str)
        has_value = isinstance(record.get("value"), str)
        _require(has_path ^ has_value, f"component {name} needs exactly path or value")
        if has_path:
            path = Path(record["path"])
            actual = file_sha256(str(path if path.is_absolute() else root / path))
            frozen[str(name)] = {"path": record["path"], "sha256": actual}
        else:
            value = record["value"]
            _require(bool(value), f"component {name}.value must be non-empty")
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
            frozen[str(name)] = {"value": value, "sha256": digest}
    return {
        "algorithm": "sha256",
        "components": frozen,
        "sha256": runtime_fingerprint_digest(frozen),
    }


def _resolve(root: Path, raw: Any, path: str) -> Path:
    _require(isinstance(raw, str) and bool(raw), f"{path} must be a path string")
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else root / candidate


def _load_artifacts(
    manifest_path: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    entries = _mapping(manifest.get("artifact_hashes"), "manifest.artifact_hashes")
    missing = REQUIRED_ARTIFACTS - set(entries)
    _require(not missing, f"manifest missing artifact hashes {sorted(missing)}")
    loaded: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for name, raw in entries.items():
        record = _mapping(raw, f"manifest.artifact_hashes.{name}")
        _exact_fields(record, frozenset({"path", "sha256"}),
                      f"manifest.artifact_hashes.{name}")
        expected = _sha256(record["sha256"], f"artifact {name}.sha256")
        path = _resolve(manifest_path.parent, record["path"], f"artifact {name}.path")
        _require(path.is_file(), f"artifact {name} is absent: {path}")
        actual = file_sha256(str(path))
        _require(actual == expected, f"artifact {name} sha256 {actual} != {expected}")
        hashes[str(name)] = actual
        if path.suffix.lower() == ".json":
            try:
                loaded[str(name)] = strict_json_load(str(path))
            except (MetricsError, json.JSONDecodeError, OSError) as exc:
                raise W2DVerificationError(
                    f"artifact {name} is not strict JSON: {exc}"
                ) from exc
        else:
            loaded[str(name)] = path
    return loaded, hashes


def _verify_fingerprint(manifest_path: Path, manifest: Mapping[str, Any]) -> str:
    fp = _mapping(manifest.get("runtime_fingerprint"), "manifest.runtime_fingerprint")
    _exact_fields(
        fp, frozenset({"algorithm", "components", "sha256"}),
        "manifest.runtime_fingerprint",
    )
    _require(fp["algorithm"] == "sha256", "fingerprint algorithm must be sha256")
    components = _mapping(fp["components"], "runtime_fingerprint.components")
    missing = REQUIRED_FINGERPRINT_COMPONENTS - set(components)
    _require(not missing, f"fingerprint missing components {sorted(missing)}")
    for name, raw in components.items():
        record = _mapping(raw, f"fingerprint component {name}")
        expected = _sha256(record.get("sha256"), f"component {name}.sha256")
        if "path" in record:
            _exact_fields(record, frozenset({"path", "sha256"}), f"component {name}")
            path = _resolve(manifest_path.parent, record["path"], f"component {name}.path")
            _require(path.is_file(), f"fingerprint component {name} is absent")
            actual = file_sha256(str(path))
        elif "value" in record:
            _exact_fields(record, frozenset({"value", "sha256"}), f"component {name}")
            _require(
                isinstance(record["value"], str) and bool(record["value"]),
                f"component {name}.value must be a non-empty string",
            )
            actual = hashlib.sha256(record["value"].encode("utf-8")).hexdigest()
        else:
            raise W2DVerificationError(
                f"fingerprint component {name} has neither path nor value"
            )
        _require(actual == expected, f"fingerprint component {name} moved")
    aggregate = runtime_fingerprint_digest(components)
    expected_aggregate = _sha256(fp["sha256"], "runtime_fingerprint.sha256")
    _require(aggregate == expected_aggregate, "runtime fingerprint aggregate mismatch")
    return aggregate


def _label_map(value: Any) -> dict[str, Mapping[str, Any]]:
    root = _mapping(value, "labels")
    raw = root.get("items")
    if isinstance(raw, Mapping):
        result = {str(key): _mapping(record, f"labels.items.{key}")
                  for key, record in raw.items()}
    elif isinstance(raw, list):
        result = {}
        for index, record_raw in enumerate(raw):
            record = _mapping(record_raw, f"labels.items[{index}]")
            key = _item_key(record.get("item_key"), f"labels.items[{index}].item_key")
            _require(key not in result, f"duplicate label key {key}")
            result[key] = record
    else:
        raise W2DVerificationError("labels.items must be a mapping or list")
    for key in result:
        _item_key(key, f"labels key {key}")
    return result


def _score_map(value: Any, phase: str) -> dict[str, Mapping[str, Any]]:
    root = _mapping(value, f"{phase} scores")
    _require(root.get("schema_version") == "W2D-score-v1",
             f"{phase} score schema mismatch")
    _require(root.get("phase") == phase, f"expected {phase} score phase")
    _require(root.get("measurement_name") == MEASUREMENT_NAME,
             f"{phase} score measurement name mismatch")
    records = _sequence(root.get("items"), f"{phase} scores.items")
    result: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(records):
        record = _mapping(raw, f"{phase} scores.items[{index}]")
        key = _item_key(record.get("item_key"), f"{phase} score item_key")
        _require(key not in result, f"duplicate {phase} score key {key}")
        result[key] = record
    return result


def _verify_score_provenance(
    score: Mapping[str, Any],
    hashes: Mapping[str, str],
    fingerprint: Mapping[str, Any],
    *,
    phase: str,
) -> None:
    _require(score.get("safe_inputs_sha256") == hashes["safe_inputs_json"],
             f"{phase} safe-input digest mismatch")
    _require(score.get("embeddings_sha256") == hashes["safe_inputs_npz"],
             f"{phase} embedding digest mismatch")
    components = _mapping(fingerprint["components"], "fingerprint.components")
    _require(score.get("detector_py_sha256") == components["detector"]["sha256"],
             f"{phase} detector.py digest mismatch")
    _require(score.get("scorer_py_sha256") == components["scorer"]["sha256"],
             f"{phase} scorer.py digest mismatch")
    if phase in {"test", "source_control"}:
        _require(score.get("threshold_artifact_sha256") == hashes["threshold"],
                 f"{phase} threshold digest mismatch")


def _verify_detector_artifacts(
    artifacts: Mapping[str, Any],
    hashes: Mapping[str, str],
    fingerprint_record: Mapping[str, Any],
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    Mapping[str, Any],
]:
    labels = _label_map(artifacts["labels"])
    test_root = _mapping(artifacts["test_scores"], "test_scores")
    control_root = _mapping(artifacts["source_controls"], "source_controls")
    scores = _score_map(test_root, "test")
    controls = _score_map(control_root, "source_control")
    _verify_score_provenance(
        test_root, hashes, fingerprint_record, phase="test"
    )
    _verify_score_provenance(
        control_root, hashes, fingerprint_record, phase="source_control"
    )
    try:
        recomputed_metrics = detector_quality_summary(test_root, artifacts["labels"])
        control_gate = source_control_gate_summary(control_root, artifacts["labels"])
    except MetricsError as exc:
        raise W2DVerificationError(f"detector denominator/control gate failed: {exc}") from exc
    _require(control_gate.get("passed") is True, "source controls did not all refuse")

    metric_artifact = _mapping(artifacts["detector_metrics"], "detector_metrics")
    _required_fields(
        metric_artifact,
        {"schema_version", "scores_sha256", "labels_sha256", "metrics"},
        "detector_metrics",
    )
    _require(metric_artifact["scores_sha256"] == hashes["test_scores"],
             "detector metrics point to a different score artifact")
    _require(metric_artifact["labels_sha256"] == hashes["labels"],
             "detector metrics point to different labels")
    _require(metric_artifact["metrics"] == recomputed_metrics,
             "detector metrics are not the independent recomputation")

    gate_artifact = _mapping(
        artifacts["source_control_gate"], "source_control_gate"
    )
    _exact_fields(
        gate_artifact,
        frozenset({"schema_version", "scores_sha256", "labels_sha256", "gate"}),
        "source_control_gate",
    )
    _require(
        gate_artifact["schema_version"] == "W2D-source-control-gate-v1",
        "source-control gate schema mismatch",
    )
    _require(
        gate_artifact["scores_sha256"] == hashes["source_controls"],
        "source-control gate points to a different score artifact",
    )
    _require(
        gate_artifact["labels_sha256"] == hashes["labels"],
        "source-control gate points to different labels",
    )
    _require(
        gate_artifact["gate"] == control_gate,
        "source-control gate is not the independent recomputation",
    )
    _require(len(scores) == EXPECTED_TEST_ITEMS,
             f"test score denominator is {len(scores)}, expected {EXPECTED_TEST_ITEMS}")
    _require(len(controls) == EXPECTED_SOURCE_CONTROLS,
             f"source controls are {len(controls)}, expected {EXPECTED_SOURCE_CONTROLS}")
    return labels, scores, controls, recomputed_metrics


def _verify_calibration_lineage(
    artifacts: Mapping[str, Any],
    hashes: Mapping[str, str],
    fingerprint_record: Mapping[str, Any],
) -> None:
    calibration = _mapping(artifacts["calibration_scores"], "calibration_scores")
    calibration_map = _score_map(calibration, "calibration")
    _verify_score_provenance(
        calibration, hashes, fingerprint_record, phase="calibration"
    )
    _require(
        len(calibration_map) == 448,
        "calibration score denominator is not 448",
    )
    for key, record in calibration_map.items():
        _exact_fields(
            record, CALIBRATION_SCORE_FIELDS, f"calibration score {key}"
        )
        _strict_bool(
            record["family_s_affirms"],
            f"calibration score {key}.family_s_affirms",
        )
        for field_name in ("rep", "knn", "c_score"):
            value = record[field_name]
            _require(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value)),
                f"calibration score {key}.{field_name} must be finite numeric",
            )
        _require(
            isinstance(record["detector_service_ns"], int)
            and not isinstance(record["detector_service_ns"], bool)
            and record["detector_service_ns"] >= 0,
            f"calibration score {key}.detector_service_ns is invalid",
        )

    threshold = _mapping(artifacts["threshold"], "threshold")
    _required_fields(
        threshold,
        {
            "schema_version",
            "calibration_item_count",
            "calibration_source_group_count",
            "calibration_scores_sha256",
            "labels_sha256",
        },
        "threshold",
    )
    _require(
        threshold["schema_version"] == "W2D-threshold-v1",
        "threshold schema mismatch",
    )
    _require(
        threshold["calibration_item_count"] == 448
        and threshold["calibration_source_group_count"] == 448,
        "threshold was not selected from 448 unique calibration groups",
    )
    _require(
        threshold["calibration_scores_sha256"] == hashes["calibration_scores"],
        "threshold points to different calibration scores",
    )
    _require(
        threshold["labels_sha256"] == hashes["labels"],
        "threshold points to different labels",
    )
    try:
        recomputed = select_threshold(calibration, artifacts["labels"])
    except (CalibrationError, KeyError, TypeError, ValueError) as exc:
        raise W2DVerificationError(
            f"independent calibration recomputation failed: {exc}"
        ) from exc
    published_core = {
        key: value
        for key, value in threshold.items()
        if key not in {"calibration_scores_sha256", "labels_sha256"}
    }
    _require(
        published_core == recomputed,
        "threshold artifact is not the independent calibration recomputation",
    )


def _canonical_json_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise W2DVerificationError(f"value is not strict JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _verify_landing_artifact(
    artifacts: Mapping[str, Any],
    hashes: Mapping[str, str],
    labels: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    landing = _mapping(artifacts["landing"], "landing")
    _exact_fields(landing, LANDING_FIELDS, "landing")
    _require(
        landing["schema_version"] == "W2D-landing-v1",
        "landing schema mismatch",
    )
    _require(
        landing["top_k"] == 5,
        "landing artifact did not use top-5",
    )
    _require(
        landing["protocol_plan_sha256"] == hashes["protocol_plan"],
        "landing artifact points to a different protocol plan",
    )
    _require(
        landing["safe_inputs_sha256"] == hashes["safe_inputs_json"],
        "landing artifact points to different safe inputs",
    )
    _require(
        landing["embeddings_sha256"] == hashes["safe_inputs_npz"],
        "landing artifact points to different embeddings",
    )
    plan = _mapping(artifacts["protocol_plan"], "protocol_plan")
    background = _mapping(
        plan.get("retrieval_background"), "protocol_plan.retrieval_background"
    )
    background_keys = _sequence(
        background.get("item_keys"),
        "protocol_plan.retrieval_background.item_keys",
    )
    _require(
        len(background_keys) == 768
        and len(set(background_keys)) == len(background_keys),
        "retrieval background is not 768 unique items",
    )
    for index, key in enumerate(background_keys):
        _item_key(key, f"retrieval background item {index}")
    _require(
        landing["retrieval_background_sha256"]
        == _canonical_json_sha256(background_keys),
        "landing retrieval-background digest mismatch",
    )

    rows = _sequence(landing["items"], "landing.items")
    _require(len(rows) == 192, "landing artifact does not contain 192 test poisons")
    by_key: dict[str, Mapping[str, Any]] = {}
    counts: dict[str, tuple[int, int]] = {}
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"landing.items[{index}]")
        _exact_fields(row, LANDING_ITEM_FIELDS, f"landing.items[{index}]")
        key = _item_key(row["item_key"], f"landing.items[{index}].item_key")
        _require(key not in by_key, f"landing item {key} is duplicated")
        label = _mapping(labels.get(key), f"landing label {key}")
        _require(label.get("split") == "test", f"landing item {key} is not test")
        _require(
            _strict_bool(
                label.get("poison", label.get("is_poison")),
                f"landing label {key}.poison",
            ),
            f"landing item {key} is not poison",
        )
        family = row["attack_family"]
        _require(
            family in {"recipe", "natural_cover_suffix_v1"}
            and label.get("attack_family") == family,
            f"landing item {key} attack family differs from labels",
        )
        rank = row["rank"]
        _require(
            isinstance(rank, int)
            and not isinstance(rank, bool)
            and rank > 0
            and row["candidate_rank"] == rank,
            f"landing item {key} has an invalid rank",
        )
        landed = _strict_bool(row["landed_top5"], f"landing item {key}.landed_top5")
        _require(
            _strict_bool(row["landed"], f"landing item {key}.landed") == landed
            and landed == (rank <= 5),
            f"landing item {key} landed flag disagrees with rank",
        )
        top5 = _sequence(row["top5_item_keys"], f"landing item {key}.top5_item_keys")
        _require(
            len(top5) == 5 and len(set(top5)) == 5,
            f"landing item {key} does not report five unique top-5 keys",
        )
        for position, candidate in enumerate(top5):
            _item_key(candidate, f"landing item {key}.top5[{position}]")
        total, hits = counts.get(str(family), (0, 0))
        counts[str(family)] = (total + 1, hits + int(landed))
        by_key[key] = row

    expected_counts = {
        "recipe": EXPECTED_RECIPE_TEST,
        "natural_cover_suffix_v1": EXPECTED_NATURAL_COVER,
    }
    populations = _mapping(landing["populations"], "landing.populations")
    _require(
        set(populations) == set(expected_counts),
        "landing populations are not the exact two attack families",
    )
    for family, expected_n in expected_counts.items():
        total, hits = counts.get(family, (0, 0))
        _require(
            total == expected_n,
            f"landing {family} denominator is {total}, expected {expected_n}",
        )
        expected_summary = {
            **rate_record(hits, expected_n),
            "landed_n": hits,
            "n": expected_n,
        }
        _require(
            populations[family] == expected_summary,
            f"landing {family} summary is not recomputed from item rows",
        )

    landing_plan = _mapping(plan.get("landing_plan"), "protocol_plan.landing_plan")
    planned_rows = _sequence(
        landing_plan.get("records"), "protocol_plan.landing_plan.records"
    )
    planned_keys: set[str] = set()
    for index, raw in enumerate(planned_rows):
        if isinstance(raw, str):
            key = raw
        else:
            record = _mapping(raw, f"landing plan record {index}")
            key = record.get("item_key", record.get("poison_item_key"))
        planned_keys.add(_item_key(key, f"landing plan record {index}.item_key"))
    _require(
        len(planned_rows) == len(planned_keys) == 192
        and planned_keys == set(by_key),
        "landing artifact keys differ from the frozen landing plan",
    )
    return by_key


def _verify_plan(
    plan: Any,
    labels: Mapping[str, Mapping[str, Any]],
    landing_keys: set[str],
) -> tuple[
    dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]
]:
    root = _mapping(plan, "protocol_plan")
    units_raw = _sequence(root.get("runtime_units"), "protocol_plan.runtime_units")
    cells_raw = _sequence(root.get("cells"), "protocol_plan.cells")
    _require(len(units_raw) == EXPECTED_RUNTIME_UNITS,
             f"protocol plan has {len(units_raw)} runtime units, expected 20")
    _require(len(cells_raw) == EXPECTED_PROTOCOL_CELLS,
             f"protocol plan has {len(cells_raw)} cells, expected 140")
    units: dict[str, Mapping[str, Any]] = {}
    poison_by_family: dict[str, set[str]] = {
        "recipe": set(),
        "natural_cover_suffix_v1": set(),
    }
    poison_by_family_seed: dict[tuple[str, int], dict[str, set[str]]] = {}
    role_vectors: dict[
        tuple[str, int, str], dict[str, tuple[str, ...]]
    ] = {}
    for index, raw in enumerate(units_raw):
        unit = _mapping(raw, f"runtime_units[{index}]")
        _required_fields(
            unit,
            {
                "runtime_unit_id",
                "seed",
                "attack_family",
                "backlog",
                "poison_item_keys",
                "clean_item_keys",
                "filler_item_keys",
                "query_poison_item_keys",
                "injection_sequence",
            },
            f"runtime_units[{index}]",
        )
        uid = str(unit["runtime_unit_id"])
        _require(uid and uid not in units, f"duplicate/empty runtime_unit_id {uid!r}")
        sequence = _sequence(
            unit["injection_sequence"], f"runtime_unit {uid}.injection_sequence"
        )
        seen: set[str] = set()
        unit_poison: set[str] = set()
        ordered_by_role: dict[str, list[str]] = {
            "poison": [],
            "clean": [],
            "filler": [],
        }
        for position, event_raw in enumerate(sequence):
            event = _mapping(event_raw, f"runtime_unit {uid}.injection[{position}]")
            _required_fields(
                event, {"injection_ordinal", "item_key", "role"},
                f"runtime_unit {uid}.injection[{position}]",
            )
            key = _item_key(event["item_key"], f"runtime_unit {uid} item_key")
            _require(key not in seen, f"runtime unit {uid} repeats item {key}")
            seen.add(key)
            _require(
                event["injection_ordinal"] == position,
                f"runtime unit {uid} injection ordinal is not dense",
            )
            _require(event["role"] in {"poison", "clean", "filler"},
                     f"runtime unit {uid} has invalid role {event['role']!r}")
            ordered_by_role[str(event["role"])].append(key)
            label = _mapping(labels.get(key), f"runtime unit {uid} label {key}")
            truth = _strict_bool(
                label.get("poison", label.get("is_poison")),
                f"runtime unit {uid} label {key}.poison",
            )
            if event["role"] == "poison":
                _require(truth, f"runtime unit {uid} marks clean item {key} as poison")
                family = str(unit["attack_family"])
                _require(
                    family in poison_by_family
                    and label.get("attack_family") == family,
                    f"runtime unit {uid} poison {key} has the wrong attack family",
                )
                _require(
                    key in landing_keys,
                    f"runtime unit {uid} poison {key} has no landing record",
                )
                unit_poison.add(key)
                poison_by_family[family].add(key)
            else:
                _require(
                    not truth,
                    f"runtime unit {uid} marks poison item {key} as {event['role']}",
                )
                _require(
                    label.get("stratum")
                    in {"ordinary_clean", "hard_negative_clean"},
                    f"runtime unit {uid} clean/filler {key} has invalid stratum",
                )
        frozen_vectors: dict[str, list[str]] = {}
        for role, field_name in (
            ("poison", "poison_item_keys"),
            ("clean", "clean_item_keys"),
            ("filler", "filler_item_keys"),
        ):
            vector = _sequence(
                unit[field_name], f"runtime unit {uid}.{field_name}"
            )
            keys = [
                _item_key(value, f"runtime unit {uid}.{field_name}[{offset}]")
                for offset, value in enumerate(vector)
            ]
            _require(
                len(keys) == len(set(keys)) and keys == ordered_by_role[role],
                f"runtime unit {uid} {field_name} differs from injection sequence",
            )
            frozen_vectors[role] = keys
        _require(
            sequence
            == sorted(
                sequence,
                key=lambda event: (
                    {"filler": 0, "poison": 1, "clean": 2}[event["role"]],
                    event["injection_ordinal"],
                ),
            ),
            f"runtime unit {uid} injection order is not filler/poison/clean",
        )
        query_poison = _sequence(
            unit["query_poison_item_keys"],
            f"runtime unit {uid}.query_poison_item_keys",
        )
        _require(
            query_poison == frozen_vectors["poison"],
            f"runtime unit {uid} query poison order differs from injection",
        )
        _require(
            len(unit_poison) == len(frozen_vectors["poison"]) == 6,
            f"runtime unit {uid} does not contain exactly six poison items",
        )
        seed = unit["seed"]
        _require(
            isinstance(seed, int) and not isinstance(seed, bool),
            f"runtime unit {uid} seed must be an integer",
        )
        _require(seed in EXPECTED_SEEDS, f"runtime unit {uid} has invalid seed {seed}")
        backlog = str(unit["backlog"])
        _require(
            backlog in {"normal", "heavy"},
            f"runtime unit {uid} has invalid backlog {backlog!r}",
        )
        expected_filler_n = 12 if backlog == "heavy" else 0
        _require(
            len(frozen_vectors["clean"]) == 6
            and len(frozen_vectors["filler"]) == expected_filler_n,
            f"runtime unit {uid} does not have 6 clean/{expected_filler_n} filler",
        )
        clean_strata = Counter(
            str(_mapping(labels[key], f"label {key}")["stratum"])
            for key in frozen_vectors["clean"]
        )
        filler_strata = Counter(
            str(_mapping(labels[key], f"label {key}")["stratum"])
            for key in frozen_vectors["filler"]
        )
        _require(
            dict(clean_strata) == EXPECTED_CLEAN_STRATA[seed],
            f"runtime unit {uid} clean strata differ from A4",
        )
        expected_filler_strata = (
            EXPECTED_FILLER_STRATA[seed] if backlog == "heavy" else {}
        )
        _require(
            dict(filler_strata) == expected_filler_strata,
            f"runtime unit {uid} filler strata differ from A4",
        )
        family = str(unit["attack_family"])
        role_key = (family, seed, backlog)
        _require(role_key not in role_vectors, f"duplicate runtime tuple {role_key}")
        role_vectors[role_key] = {
            role: tuple(values) for role, values in frozen_vectors.items()
        }
        family_seed = (family, seed)
        by_backlog = poison_by_family_seed.setdefault(family_seed, {})
        _require(
            backlog not in by_backlog,
            f"runtime unit pair {family_seed} repeats backlog {backlog}",
        )
        by_backlog[backlog] = unit_poison
        units[uid] = unit

    for family, keys in poison_by_family.items():
        _require(
            len(keys) == EXPECTED_RUNTIME_POISON_PER_FAMILY,
            f"runtime plan has {len(keys)} unique {family} poisons, expected 30",
        )
    for family_seed, by_backlog in poison_by_family_seed.items():
        _require(
            set(by_backlog) == {"normal", "heavy"},
            f"runtime pair {family_seed} lacks normal/heavy backlog",
        )
        _require(
            by_backlog["normal"] == by_backlog["heavy"],
            f"runtime pair {family_seed} changes poison items across backlog",
        )
    for family in poison_by_family:
        family_seeds = {
            seed
            for (candidate_family, seed) in poison_by_family_seed
            if candidate_family == family
        }
        _require(
            family_seeds == EXPECTED_SEEDS,
            f"runtime plan seeds for {family} are {sorted(family_seeds)}",
        )

    expected_role_keys = {
        (family, seed, backlog)
        for family in poison_by_family
        for seed in EXPECTED_SEEDS
        for backlog in ("normal", "heavy")
    }
    _require(
        set(role_vectors) == expected_role_keys,
        "runtime plan does not contain the exact attack/seed/backlog product",
    )
    all_clean_by_seed: dict[int, set[str]] = {}
    all_filler_by_seed: dict[int, set[str]] = {}
    for seed in EXPECTED_SEEDS:
        recipe_normal = role_vectors[("recipe", seed, "normal")]
        recipe_heavy = role_vectors[("recipe", seed, "heavy")]
        natural_normal = role_vectors[
            ("natural_cover_suffix_v1", seed, "normal")
        ]
        natural_heavy = role_vectors[
            ("natural_cover_suffix_v1", seed, "heavy")
        ]
        clean_vectors = {
            recipe_normal["clean"],
            recipe_heavy["clean"],
            natural_normal["clean"],
            natural_heavy["clean"],
        }
        _require(
            len(clean_vectors) == 1,
            f"seed {seed} clean set/order changes across attack or backlog",
        )
        _require(
            recipe_normal["filler"] == natural_normal["filler"] == (),
            f"seed {seed} normal backlog unexpectedly has filler",
        )
        _require(
            recipe_heavy["filler"] == natural_heavy["filler"],
            f"seed {seed} heavy filler changes across attack",
        )
        all_clean_by_seed[seed] = set(recipe_normal["clean"])
        all_filler_by_seed[seed] = set(recipe_heavy["filler"])

    for left in EXPECTED_SEEDS:
        _require(
            not (all_clean_by_seed[left] & all_filler_by_seed[left]),
            f"seed {left} reuses an item as clean and filler",
        )
        for right in EXPECTED_SEEDS:
            if left >= right:
                continue
            _require(
                not (all_clean_by_seed[left] & all_clean_by_seed[right]),
                f"seeds {left}/{right} reuse injected clean items",
            )
            _require(
                not (all_filler_by_seed[left] & all_filler_by_seed[right]),
                f"seeds {left}/{right} reuse heavy filler items",
            )
            _require(
                not (all_clean_by_seed[left] & all_filler_by_seed[right])
                and not (all_filler_by_seed[left] & all_clean_by_seed[right]),
                f"seeds {left}/{right} cross-reuse clean/filler items",
            )

    cells: dict[str, Mapping[str, Any]] = {}
    per_unit: dict[str, list[tuple[str, str]]] = {}
    for index, raw in enumerate(cells_raw):
        cell = _mapping(raw, f"protocol_plan.cells[{index}]")
        _required_fields(
            cell,
            {
                "cell_id",
                "runtime_unit_id",
                "seed",
                "attack_family",
                "backlog",
                "baseline",
                "arm",
                "service_replay",
            },
            f"protocol_plan.cells[{index}]",
        )
        cid = str(cell["cell_id"])
        uid = str(cell["runtime_unit_id"])
        _require(cid and cid not in cells, f"duplicate/empty plan cell id {cid!r}")
        _require(uid in units, f"plan cell {cid} has unknown runtime unit {uid!r}")
        baseline, arm = cell["baseline"], cell["arm"]
        _require(baseline in {"B1", "B2", "B3", "B4"},
                 f"plan cell {cid} has invalid baseline")
        if baseline == "B1":
            _require(arm == "control", f"B1 plan cell {cid} must be control arm")
        else:
            _require(arm in {"detector", "oracle"},
                     f"plan cell {cid} has invalid arm {arm!r}")
            _require(cell["service_replay"] == FROZEN_SERVICE_REPLAY,
                     f"plan cell {cid} does not freeze item service times")
        unit = units[uid]
        for field in ("seed", "attack_family", "backlog"):
            _require(cell[field] == unit[field],
                     f"plan cell {cid} {field} differs from runtime unit")
        per_unit.setdefault(uid, []).append((baseline, arm))
        cells[cid] = cell
    expected = {
        ("B1", "control"),
        ("B2", "detector"), ("B3", "detector"), ("B4", "detector"),
        ("B2", "oracle"), ("B3", "oracle"), ("B4", "oracle"),
    }
    for uid, memberships in per_unit.items():
        _require(
            len(memberships) == len(expected) and set(memberships) == expected,
            f"runtime unit {uid} does not have the exact seven planned cells",
        )
    return units, cells


def _protocol_items(
    cell: Mapping[str, Any],
    unit: Mapping[str, Any],
    scores: Mapping[str, Mapping[str, Any]],
    path: str,
) -> list[Mapping[str, Any]]:
    items_raw = _sequence(cell["protocol_items"], f"{path}.protocol_items")
    sequence = _sequence(unit["injection_sequence"], f"{path}.plan_sequence")
    _require(len(items_raw) == len(sequence),
             f"{path} protocol item count differs from its runtime unit")
    items: list[Mapping[str, Any]] = []
    for index, (raw, event_raw) in enumerate(zip(items_raw, sequence)):
        item = _mapping(raw, f"{path}.protocol_items[{index}]")
        event = _mapping(event_raw, f"{path}.plan_sequence[{index}]")
        _exact_fields(item, PROTOCOL_ITEM_FIELDS, f"{path}.protocol_items[{index}]")
        key = _item_key(item["item_key"], f"{path}.protocol item key")
        _require(key in scores, f"{path} item {key} has no unique score-once record")
        _require(item["role"] == event["role"]
                 and item["ordinal"] == event["injection_ordinal"]
                 and key == event["item_key"],
                 f"{path} protocol item {index} differs from the frozen plan")
        score = scores[key]
        promote = _strict_bool(item["detector_promote"],
                               f"{path}.protocol_items[{index}].detector_promote")
        _require(promote == score["promote"],
                 f"{path} item {key} decision differs from score-once record")
        latency = item["detector_service_ns"]
        _require(
            isinstance(latency, int) and not isinstance(latency, bool) and latency >= 0,
            f"{path} item {key} detector_service_ns must be non-negative int",
        )
        _require(latency == score["detector_service_ns"],
                 f"{path} item {key} service time differs from score-once record")
        _require(
            item["score_record_sha256"] == canonical_record_sha256(score),
            f"{path} item {key} score-record digest mismatch",
        )
        items.append(item)
    return items


def _verify_provider_records(
    cell: Mapping[str, Any],
    item_map: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    path: str,
) -> dict[str, Mapping[str, Any]]:
    records = _sequence(cell["verifier_records"], f"{path}.verifier_records")
    if cell["baseline"] == "B1":
        _require(not records, f"{path} B1 control must not run a verifier")
        return {}
    by_key: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(records):
        record = _mapping(raw, f"{path}.verifier_records[{index}]")
        _exact_fields(
            record,
            VERIFIER_RECORD_FIELDS,
            f"{path}.verifier_records[{index}]",
        )
        key = _item_key(record["item_key"], f"{path}.verifier_records[{index}].item_key")
        _require(key not in by_key, f"{path} repeats verifier record {key}")
        _require(key in item_map, f"{path} verifier record {key} is not planned")
        status = record["status"]
        _require(status in PROVIDER_STATUSES,
                 f"{path} provider status {status!r} is not terminal/legal")
        if record["passes"] is not None:
            passes = _strict_bool(record["passes"], f"{path} record {key}.passes")
            if cell["arm"] == "detector":
                _require(passes == item_map[key]["detector_promote"],
                         f"{path} record {key} replayed a different decision")
            else:
                label = _mapping(labels.get(key), f"{path} label {key}")
                truth = _strict_bool(
                    label.get("poison", label.get("is_poison")),
                    f"{path} label {key}.poison",
                )
                _require(
                    passes is (not truth),
                    f"{path} oracle record {key} differs from evaluator truth",
                )
        if record["service_time_s"] is not None:
            service = _nonnegative_number(
                record["service_time_s"], f"{path} record {key}.service_time_s"
            )
            expected = item_map[key]["detector_service_ns"] / 1_000_000_000
            _require(
                math.isclose(service, expected, rel_tol=0.0, abs_tol=1e-12),
                f"{path} record {key} replayed a different service time",
            )
        if status == "COMMITTED":
            _strict_bool(record["passes"], f"{path} committed record {key}.passes")
            _nonnegative_number(
                record["service_time_s"], f"{path} committed record {key}.service_time_s"
            )
            _nonnegative_number(
                record["integrated_latency_s"],
                f"{path} committed record {key}.integrated_latency_s",
            )
            _nonnegative_number(
                record["decision_commit_s"],
                f"{path} committed record {key}.decision_commit_s",
            )
        else:
            _require(record["decision_commit_s"] is None,
                     f"{path} cancelled record {key} claims a commit")
            _nonnegative_number(
                record["integrated_latency_s"],
                f"{path} cancelled record {key}.integrated_latency_s",
                allow_none=True,
            )
        by_key[key] = record
    _require(set(by_key) == set(item_map),
             f"{path} verifier records do not exactly cover protocol items")
    return by_key


def _verify_lifecycle(
    cell: Mapping[str, Any],
    item_map: Mapping[str, Mapping[str, Any]],
    provider_records: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    horizon: float,
    path: str,
) -> None:
    rows = _sequence(cell["lifecycle"], f"{path}.lifecycle")
    by_key: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"{path}.lifecycle[{index}]")
        _exact_fields(row, LIFECYCLE_FIELDS, f"{path}.lifecycle[{index}]")
        key = _item_key(row["item_key"], f"{path}.lifecycle[{index}].item_key")
        _require(key not in by_key and key in item_map,
                 f"{path} lifecycle key {key} is duplicate or unplanned")
        label = _mapping(labels.get(key), f"label {key}")
        truth = label.get("poison")
        if truth is None:
            truth = label.get("is_poison")
        truth = _strict_bool(truth, f"label {key}.poison")
        promote = _strict_bool(
            row["detector_promote"], f"{path}.lifecycle[{index}].detector_promote"
        )
        _require(_strict_bool(row["truth_poison"], f"{path} lifecycle truth") == truth,
                 f"{path} lifecycle truth for {key} differs from evaluator label")
        _require(promote == item_map[key]["detector_promote"],
                 f"{path} lifecycle decision for {key} differs from score")
        _require(
            isinstance(row["retry_count"], int)
            and not isinstance(row["retry_count"], bool)
            and row["retry_count"] == 0,
            f"{path} item {key} violates the no-retry design",
        )
        _require(
            isinstance(row["readmission_count"], int)
            and not isinstance(row["readmission_count"], bool)
            and row["readmission_count"] >= 0,
            f"{path} item {key} has invalid readmission_count",
        )
        _require(row["state_at_horizon"] in FINAL_STATES,
                 f"{path} item {key} has invalid horizon state")
        _require(row["visibility_status"] in {"NOT_STARTED", "STARTED"},
                 f"{path} item {key} has invalid visibility status")
        for field_name in (
            "exposure_status", "unavailable_status", "quarantine_status"
        ):
            _require(row[field_name] in EPISODE_STATUSES,
                     f"{path} item {key} has invalid {field_name}")
        _nonnegative_number(
            row["exposure_observed_s"], f"{path} item {key}.exposure_observed_s",
            allow_none=True,
        )
        _nonnegative_number(
            row["unavailable_observed_s"], f"{path} item {key}.unavailable_observed_s",
            allow_none=True,
        )
        observed = _nonnegative_number(
            row["observed_at_horizon_s"], f"{path} item {key}.observed_at_horizon_s"
        )
        _require(math.isclose(observed, horizon, rel_tol=0.0, abs_tol=1e-9),
                 f"{path} item {key} was not accounted at the fixed horizon")

        decision_committed = (
            key in provider_records
            and provider_records[key]["status"] == "COMMITTED"
        )
        if (
            cell["arm"] == "detector"
            and decision_committed
            and not truth
            and not promote
        ):
            # Clean false positive: one terminal misquarantine, no retry or
            # readmission, with the unavailable/quarantine episode censored at
            # the fixed horizon.
            _require(row["state_at_horizon"] == "QUARANTINED",
                     f"{path} clean false positive {key} is not quarantined")
            _require(row["quarantine_status"] == "RIGHT_CENSORED"
                     and row["unavailable_status"] == "RIGHT_CENSORED",
                     f"{path} clean false positive {key} lost censoring")
            _require(row["readmission_count"] == 0,
                     f"{path} clean false positive {key} was retried/readmitted")
            expected_visibility = (
                "NOT_STARTED" if cell["baseline"] == "B2" else "STARTED"
            )
            _require(row["visibility_status"] == expected_visibility,
                     f"{path} clean false-positive visibility is inconsistent")

        if (
            cell["arm"] == "detector"
            and decision_committed
            and truth
            and promote
        ):
            # Poison false negative: the decision can commit before or after a
            # B4 hide, but after any readmission it remains exposed at horizon.
            _require(row["state_at_horizon"] == "TRUSTED"
                     and row["visibility_status"] == "STARTED",
                     f"{path} poison false negative {key} is not visible/trusted")
            _require(row["exposure_status"] == "RIGHT_CENSORED",
                     f"{path} poison false negative {key} was called contained")
            age = _nonnegative_number(
                row["exposure_observed_s"],
                f"{path} poison false negative {key}.exposure_observed_s",
            )
            _require(age <= horizon,
                     f"{path} poison false negative {key} exceeds the horizon")
            _require(row["quarantine_status"] == "NOT_STARTED",
                     f"{path} poison false negative {key} also claims quarantine")
        by_key[key] = row
    _require(set(by_key) == set(item_map),
             f"{path} lifecycle does not exactly cover protocol items")


def _verify_result(
    result: Any,
    artifacts: Mapping[str, Any],
    hashes: Mapping[str, str],
    fingerprint_sha256: str,
    labels: Mapping[str, Mapping[str, Any]],
    scores: Mapping[str, Mapping[str, Any]],
    landing_items: Mapping[str, Mapping[str, Any]],
    detector_metrics: Mapping[str, Any],
    units: Mapping[str, Mapping[str, Any]],
    plan_cells: Mapping[str, Mapping[str, Any]],
) -> None:
    root = _mapping(result, "results")
    _exact_fields(root, RESULT_FIELDS, "results")
    _require(root["schema_version"] == RESULT_SCHEMA, "result schema mismatch")
    _require(root["measurement_name"] == MEASUREMENT_NAME,
             "result measurement name is not the preregistered replay name")
    _require(root["execution_mode"] == EXECUTION_MODE,
             "result execution mode could be mistaken for online detector latency")
    _require(root["runtime_fingerprint_sha256"] == fingerprint_sha256,
             "result points to a different runtime fingerprint")
    result_hashes = _mapping(root["artifact_hashes"], "results.artifact_hashes")
    _require(set(result_hashes) == set(hashes),
             "result artifact hash set differs from the manifest")
    for name, expected in hashes.items():
        _require(result_hashes[name] == expected,
                 f"result artifact hash {name} differs from the manifest")
    execution = _mapping(root["detector_execution"], "results.detector_execution")
    _exact_fields(
        execution,
        frozenset({"protocol_provider", "detector_invocations_during_replay"}),
        "results.detector_execution",
    )
    _require(execution["protocol_provider"] == FROZEN_PROVIDER,
             "protocol provider is not the frozen score lookup")
    _require(
        isinstance(execution["detector_invocations_during_replay"], int)
        and not isinstance(execution["detector_invocations_during_replay"], bool)
        and execution["detector_invocations_during_replay"] == 0,
        "live detector execution occurred during replay",
    )
    horizon = _nonnegative_number(
        root["observation_horizon_s"], "results.observation_horizon_s"
    )
    injection_at = _nonnegative_number(root["injection_at_s"], "results.injection_at_s")
    _require(horizon == 8.0 and injection_at == 1.0 and injection_at < horizon,
             "result does not use the frozen 8 s horizon / 1 s injection")

    denominator = _mapping(root["detector_denominator"], "results.detector_denominator")
    _exact_fields(
        denominator,
        frozenset(
            {
                "score_once_item_count",
                "unique_source_group_count",
                "reported_detector_quality_denominator",
                "protocol_replay_observation_count",
            }
        ),
        "results.detector_denominator",
    )
    expected_items = int(detector_metrics["unique_item_count"])
    expected_groups = int(detector_metrics["unique_source_group_count"])
    _require(
        denominator["score_once_item_count"] == expected_items
        and denominator["unique_source_group_count"] == expected_groups
        and denominator["reported_detector_quality_denominator"] == expected_items,
        "detector denominator expanded or differs from the unique score-once set",
    )

    cells_raw = _sequence(root["cells"], "results.cells")
    _require(len(cells_raw) == len(plan_cells),
             "result cell count differs from the protocol plan")
    result_cells: dict[str, Mapping[str, Any]] = {}
    replay_vectors: dict[
        tuple[str, str, int, str], dict[str, tuple[tuple[Any, ...], ...]]
    ] = {}
    replay_observations = 0
    for index, raw in enumerate(cells_raw):
        cell = _mapping(raw, f"results.cells[{index}]")
        _exact_fields(cell, CELL_FIELDS, f"results.cells[{index}]")
        cid = str(cell["plan_cell_id"])
        _require(cid in plan_cells and cid not in result_cells,
                 f"result cell {cid!r} is absent from plan or duplicated")
        planned = plan_cells[cid]
        uid = str(planned["runtime_unit_id"])
        unit = units[uid]
        for field in ("seed", "attack_family", "backlog", "baseline", "arm"):
            _require(cell[field] == planned[field],
                     f"result cell {cid} {field} differs from plan")
        path = f"result cell {cid}"
        items = _protocol_items(cell, unit, scores, path)
        item_map = {str(item["item_key"]): item for item in items}
        _require(len(item_map) == len(items), f"{path} repeats a protocol item")
        provider_records = _verify_provider_records(
            cell, item_map, labels, path
        )
        _verify_lifecycle(
            cell, item_map, provider_records, labels, horizon, path
        )
        retrieval = _mapping(cell["retrieval"], f"{path}.retrieval")
        if cell["baseline"] == "B1":
            _required_fields(
                retrieval,
                {"attack_landing_numerator", "attack_landing_denominator"},
                f"{path}.retrieval",
            )
            numerator = retrieval["attack_landing_numerator"]
            denominator_value = retrieval["attack_landing_denominator"]
            _require(
                isinstance(numerator, int)
                and not isinstance(numerator, bool)
                and isinstance(denominator_value, int)
                and not isinstance(denominator_value, bool)
                and 0 <= numerator <= denominator_value,
                f"{path} has invalid B1 attack-landing counts",
            )
            poison_keys = [
                str(event["item_key"])
                for event in _sequence(
                    unit["injection_sequence"], f"{path}.plan_sequence"
                )
                if _mapping(event, f"{path}.plan event").get("role") == "poison"
            ]
            expected_landed = sum(
                int(
                    _strict_bool(
                        _mapping(
                            landing_items.get(key), f"{path} landing item {key}"
                        ).get("landed_top5"),
                        f"{path} landing item {key}.landed_top5",
                    )
                )
                for key in poison_keys
            )
            _require(
                denominator_value == len(poison_keys) == 6
                and numerator == expected_landed,
                f"{path} B1 landing counts differ from six frozen landing rows",
            )
        if cell["baseline"] != "B1":
            pair = (
                str(cell["attack_family"]),
                str(cell["backlog"]),
                int(cell["seed"]),
                str(cell["arm"]),
            )
            vector = tuple(
                (
                    item["item_key"],
                    item["role"],
                    item["ordinal"],
                    item["detector_promote"],
                    item["detector_service_ns"],
                    item["score_record_sha256"],
                )
                for item in items
            )
            replay_vectors.setdefault(pair, {})[str(cell["baseline"])] = vector
        if cell["arm"] == "detector":
            replay_observations += len(items)
        result_cells[cid] = cell
    _require(set(result_cells) == set(plan_cells),
             "result cells do not exactly equal the protocol-plan cells")
    for pair, by_baseline in replay_vectors.items():
        _require(set(by_baseline) == {"B2", "B3", "B4"},
                 f"replay pair {pair} lacks B2/B3/B4")
        _require(by_baseline["B2"] == by_baseline["B3"] == by_baseline["B4"],
                 f"replay pair {pair} changed decision/service vectors")
    _require(
        denominator["protocol_replay_observation_count"] == replay_observations,
        "reported replay observation count does not equal cell records",
    )
    expected_family_counts = {
        "natural_cover_suffix_v1": EXPECTED_NATURAL_COVER,
        "recipe": EXPECTED_RECIPE_TEST,
    }
    for family, expected in expected_family_counts.items():
        label_keys = {
            key
            for key, label in labels.items()
            if label.get("split") == "test"
            and label.get("attack_family") == family
            and label.get("poison", label.get("is_poison")) is True
        }
        _require(
            len(label_keys) == expected,
            f"labels retain {len(label_keys)} {family} test poisons, expected {expected}",
        )


def verify_bundle(
    manifest_path: str | Path, results_path: str | Path
) -> VerificationReport:
    """Verify the complete W2D bundle without mutating it."""
    report = VerificationReport()
    try:
        manifest_file = Path(manifest_path)
        try:
            manifest = strict_json_load(str(manifest_file))
            result = strict_json_load(str(results_path))
        except (MetricsError, json.JSONDecodeError, OSError) as exc:
            raise W2DVerificationError(f"strict JSON load failed: {exc}") from exc
        manifest = _mapping(manifest, "manifest")
        _required_fields(
            manifest,
            {"schema_version", "measurement_name", "artifact_hashes",
             "runtime_fingerprint"},
            "manifest",
        )
        _require(manifest["schema_version"] == MANIFEST_SCHEMA,
                 "manifest schema mismatch")
        _require(manifest["measurement_name"] == MEASUREMENT_NAME,
                 "manifest measurement name mismatch")
        artifacts, hashes = _load_artifacts(manifest_file, manifest)
        report.passed("strict JSON and every formal artifact hash")
        fingerprint_sha = _verify_fingerprint(manifest_file, manifest)
        report.passed("independent runtime fingerprint and component hashes")
        fingerprint_record = _mapping(
            manifest["runtime_fingerprint"], "runtime_fingerprint"
        )
        _verify_calibration_lineage(artifacts, hashes, fingerprint_record)
        report.passed("calibration threshold recomputation and artifact lineage")
        labels, scores, _controls, detector_metrics = _verify_detector_artifacts(
            artifacts, hashes, fingerprint_record
        )
        report.passed("unique score-once denominator and 24 source controls")
        landing_items = _verify_landing_artifact(artifacts, hashes, labels)
        report.passed("192-item landing artifact and retrieval-background lineage")
        units, plan_cells = _verify_plan(
            artifacts["protocol_plan"], labels, set(landing_items)
        )
        report.passed("20 runtime units and 140 frozen protocol cells")
        _verify_result(
            result,
            artifacts,
            hashes,
            fingerprint_sha,
            labels,
            scores,
            landing_items,
            detector_metrics,
            units,
            plan_cells,
        )
        report.passed(
            "paired replay, score back-references, provider states, and FP/FN lifecycle"
        )
    except (W2DVerificationError, MetricsError, KeyError, TypeError, ValueError) as exc:
        report.failed(str(exc))
    return report


def verify_or_raise(manifest_path: str | Path, results_path: str | Path) -> None:
    report = verify_bundle(manifest_path, results_path)
    if not report.ok:
        raise W2DVerificationError("; ".join(report.failures))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args(argv)
    report = verify_bundle(args.manifest, args.results)
    for name in report.checks:
        print(f"  [PASS] {name}")
    for failure in report.failures:
        print(f"  [FAIL] {failure}")
    print(
        "\n==== verify_w2d: "
        + ("ALL GATES PASSED" if report.ok else "FAILURES ABOVE")
        + " ===="
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
