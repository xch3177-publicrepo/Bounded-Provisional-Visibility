#!/usr/bin/env python3
"""Isolated score-once harness for D1.

The harness owns opaque join keys; the detector call never receives them.
Protocol experiments consume the resulting decisions and measured service
times as a replay artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import math
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np

import detector as detector_module
from detector import (
    D1Detector,
    DetectorInputError,
    EXPECTED_DIM,
    MODEL_NAME,
    MODEL_REVISION,
    ZStatistics,
    normalize_text,
)
from w2d_metrics import (
    file_sha256,
    strict_json_dump,
    strict_json_load_bytes,
)


KEY_RE = re.compile(r"^[0-9a-f]{64}$")
PHASE_FIELDS = {
    "calibration": "calibration_item_keys",
    "test": "test_score_order",
    "source_control": "source_control_item_keys",
}
PHASE_COUNTS = {"calibration": 448, "test": 512, "source_control": 24}
SAFE_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "construction_version",
        "model_revision",
        "model",
        "embedding_artifact",
        "items",
        "reference_index",
        "sets",
    }
)
SAFE_ITEM_KEYS = frozenset(
    {
        "item_key",
        "normalized_text",
        "embedding_row",
        "source_evidence",
        "source_group",
        "reference",
        "model_revision",
    }
)
MODEL_KEYS = frozenset({"name", "revision", "dimension"})
EMBEDDING_ARTIFACT_KEYS = frozenset(
    {"filename", "sha256", "array", "dtype", "shape"}
)
REFERENCE_INDEX_KEYS = frozenset({"item_keys", "embedding_rows", "groups"})
SET_KEYS = frozenset(
    {
        "calibration_item_keys",
        "test_score_order",
        "source_control_item_keys",
        "warmup_reference_item_key",
    }
)
THRESHOLD_KEYS = frozenset(
    {
        "schema_version",
        "selection_rule",
        "selected_threshold",
        "candidate_count",
        "calibration_item_count",
        "calibration_source_group_count",
        "calibration_summary",
        "z_statistics",
        "calibration_score_label_digest",
        "calibration_scores_sha256",
        "labels_sha256",
    }
)
MEASUREMENT_NAME = (
    "promotion-path replay driven by real D1 outputs and measured service times"
)


class ScorerError(ValueError):
    pass


def _require_exact_keys(value: Any, expected: frozenset[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScorerError(f"{name} must be a mapping")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ScorerError(
            f"{name} schema mismatch; missing={missing}, extra={extra}"
        )
    return value


def _require_sha256(value: Any, name: str) -> str:
    text = str(value)
    if not KEY_RE.fullmatch(text):
        raise ScorerError(f"{name} must be a lowercase hexadecimal SHA256")
    return text


def score_order_key(item_key: str) -> str:
    return hashlib.sha256(
        ("W2D-score-order|" + item_key).encode("utf-8")
    ).hexdigest()


def _items(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        value = value.get("items")
    if not isinstance(value, list):
        raise ScorerError("safe input artifact must contain an items list")
    if not all(isinstance(item, Mapping) for item in value):
        raise ScorerError("every safe input item must be a mapping")
    return value


def _item_key(item: Mapping[str, Any]) -> str:
    key = str(item.get("item_key", ""))
    if not KEY_RE.fullmatch(key):
        raise ScorerError("item_key must be a lowercase hexadecimal SHA256")
    return key


def _text(item: Mapping[str, Any]) -> str:
    text = item.get("text", item.get("normalized_text"))
    if not isinstance(text, str) or not text:
        raise ScorerError(f"{_item_key(item)} has no normalized text")
    if normalize_text(text) != text:
        raise ScorerError(f"{_item_key(item)} text is not canonically normalized")
    return text


def _embedding_row(item: Mapping[str, Any]) -> int:
    row = item.get("embedding_row")
    if isinstance(row, bool) or not isinstance(row, int) or row < 0:
        raise ScorerError(f"{_item_key(item)} has an invalid embedding_row")
    return row


def _embedding_array(npz: Any, safe_root: Any) -> np.ndarray:
    name = safe_root["embedding_artifact"]["array"]
    if npz.files != [name]:
        raise ScorerError(
            f"NPZ arrays {npz.files} do not equal the frozen array {[name]}"
        )
    arr = np.asarray(npz[name])
    if arr.dtype != np.dtype("float32"):
        raise ScorerError(f"embedding array dtype {arr.dtype} is not float32")
    expected_shape = tuple(safe_root["embedding_artifact"]["shape"])
    if arr.shape != expected_shape:
        raise ScorerError(
            f"embedding array shape {arr.shape} != frozen {expected_shape}"
        )
    if not np.isfinite(arr).all():
        raise ScorerError("embedding array contains NaN or infinity")
    return arr


def _key_manifest(value: Any, phase: str) -> list[str]:
    if isinstance(value, Mapping):
        if phase in value:
            value = value[phase]
        else:
            value = value.get("item_keys", value.get("keys"))
    if not isinstance(value, list) or not all(isinstance(key, str) for key in value):
        raise ScorerError("key manifest must contain a list of opaque item keys")
    if len(value) != len(set(value)):
        raise ScorerError("key manifest contains a duplicate")
    for key in value:
        if not KEY_RE.fullmatch(key):
            raise ScorerError("key manifest contains a non-opaque item key")
    return value


def _safe_phase_keys(safe_root: Any, phase: str) -> list[str]:
    if not isinstance(safe_root, Mapping) or not isinstance(
        safe_root.get("sets"), Mapping
    ):
        raise ScorerError("safe input artifact has no frozen phase sets")
    field = PHASE_FIELDS[phase]
    return _key_manifest(safe_root["sets"].get(field), phase)


def _model_revision(safe_root: Any) -> str:
    if not isinstance(safe_root, Mapping) or not isinstance(
        safe_root.get("model"), Mapping
    ):
        raise ScorerError("safe input artifact has no model descriptor")
    revision = safe_root["model"].get("revision")
    if not isinstance(revision, str) or not revision:
        raise ScorerError("safe input model revision is absent")
    return revision


def _reference_descriptor(
    safe_root: Any,
) -> tuple[list[str], list[int], list[str], str]:
    if not isinstance(safe_root, Mapping) or not isinstance(
        safe_root.get("reference_index"), Mapping
    ):
        raise ScorerError("safe input artifact has no reference index")
    descriptor = safe_root["reference_index"]
    keys = descriptor.get("item_keys")
    rows = descriptor.get("embedding_rows")
    groups = descriptor.get("groups")
    if not all(isinstance(value, list) for value in (keys, rows, groups)):
        raise ScorerError("reference index vectors are malformed")
    if not (len(keys) == len(rows) == len(groups)):
        raise ScorerError("reference index vectors differ in length")
    if len(keys) != len(set(keys)):
        raise ScorerError("reference index contains a duplicate item key")
    for key in keys:
        if not isinstance(key, str) or not KEY_RE.fullmatch(key):
            raise ScorerError("reference index contains a non-opaque item key")
    for row in rows:
        if isinstance(row, bool) or not isinstance(row, int) or row < 0:
            raise ScorerError("reference index contains an invalid embedding row")
    for group in groups:
        if not isinstance(group, str) or not group:
            raise ScorerError("reference index contains an invalid group")
    warm = safe_root.get("sets", {}).get("warmup_reference_item_key")
    if warm not in keys:
        raise ScorerError("frozen warm-up key is not in the reference index")
    return list(keys), list(rows), list(groups), str(warm)


def _validate_safe_root(safe_root: Any, *, frozen_production: bool) -> None:
    root = _require_exact_keys(safe_root, SAFE_ROOT_KEYS, "safe input root")
    if root.get("schema_version") != "1.0":
        raise ScorerError("unexpected safe-input schema version")
    if root.get("construction_version") != "W2D-A6-v1":
        raise ScorerError("unexpected safe-input construction version")

    model = _require_exact_keys(root.get("model"), MODEL_KEYS, "model descriptor")
    revision = model.get("revision")
    if root.get("model_revision") != revision:
        raise ScorerError("top-level and model revisions differ")
    dimension = model.get("dimension")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ScorerError("model dimension must be a positive integer")
    if frozen_production and (
        model.get("name") != MODEL_NAME
        or revision != MODEL_REVISION
        or dimension != EXPECTED_DIM
    ):
        raise ScorerError("safe input does not use the frozen production model")

    embedding = _require_exact_keys(
        root.get("embedding_artifact"),
        EMBEDDING_ARTIFACT_KEYS,
        "embedding artifact descriptor",
    )
    _require_sha256(embedding.get("sha256"), "embedding artifact sha256")
    if embedding.get("array") != "embeddings":
        raise ScorerError("embedding artifact array must be 'embeddings'")
    if embedding.get("dtype") != "float32":
        raise ScorerError("embedding artifact dtype must be float32")
    if frozen_production and embedding.get("filename") != "W2D-detector-inputs.npz":
        raise ScorerError("embedding artifact filename is not the frozen filename")
    shape = embedding.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in shape)
        or shape[1] != dimension
    ):
        raise ScorerError("embedding artifact shape is malformed")

    items = _items(root)
    keys: set[str] = set()
    rows: list[int] = []
    source_groups: set[str] = set()
    for item in items:
        _require_exact_keys(item, SAFE_ITEM_KEYS, "safe input item")
        key = _item_key(item)
        if key in keys:
            raise ScorerError(f"duplicate safe-input item_key {key}")
        keys.add(key)
        rows.append(_embedding_row(item))
        _text(item)
        group = _require_sha256(item.get("source_group"), f"{key} source_group")
        if group in source_groups:
            raise ScorerError(f"source group {group} appears more than once")
        source_groups.add(group)
        if not isinstance(item.get("reference"), bool):
            raise ScorerError(f"{key} reference flag must be bool")
        if item.get("model_revision") != revision:
            raise ScorerError(f"{key} model revision differs from the safe root")
        evidence = item.get("source_evidence")
        if (
            not isinstance(evidence, list)
            or len(evidence) != 3
            or not isinstance(evidence[0], bool)
            or not isinstance(evidence[1], bool)
            or not isinstance(evidence[2], str)
        ):
            raise ScorerError(f"{key} source_evidence has the wrong schema")
    if shape[0] != len(items):
        raise ScorerError("embedding shape row count differs from the item count")
    if sorted(rows) != list(range(len(items))):
        raise ScorerError("embedding rows are not a dense one-to-one index")

    _require_exact_keys(
        root.get("reference_index"), REFERENCE_INDEX_KEYS, "reference index"
    )
    _require_exact_keys(root.get("sets"), SET_KEYS, "frozen phase sets")
    ref_keys, _ref_rows, _groups, warm = _reference_descriptor(root)
    phase_keys = {
        phase: _safe_phase_keys(root, phase) for phase in PHASE_FIELDS
    }
    if frozen_production:
        if len(ref_keys) != 768:
            raise ScorerError(f"reference index has {len(ref_keys)} items; expected 768")
        for phase, expected in PHASE_COUNTS.items():
            if len(phase_keys[phase]) != expected:
                raise ScorerError(
                    f"{phase} set has {len(phase_keys[phase])} items; "
                    f"expected {expected}"
                )
    if phase_keys["calibration"] != sorted(phase_keys["calibration"]):
        raise ScorerError("calibration keys are not in frozen lexical order")
    if phase_keys["test"] != sorted(phase_keys["test"], key=score_order_key):
        raise ScorerError("test keys are not in frozen label-blind score order")
    if phase_keys["source_control"] != sorted(phase_keys["source_control"]):
        raise ScorerError("source-control keys are not in frozen lexical order")
    partitions = [set(ref_keys), *(set(values) for values in phase_keys.values())]
    if any(partitions[i] & partitions[j]
           for i in range(len(partitions)) for j in range(i + 1, len(partitions))):
        raise ScorerError("reference/calibration/test/source-control sets overlap")
    if set().union(*partitions) != keys:
        raise ScorerError("frozen sets do not partition every safe input item")
    if warm != min(ref_keys):
        raise ScorerError("warm-up key is not the frozen first reference key")
    ref_set = set(ref_keys)
    item_by_key = {_item_key(item): item for item in items}
    for key, row, group in zip(
        ref_keys,
        root["reference_index"]["embedding_rows"],
        root["reference_index"]["groups"],
    ):
        if item_by_key[key]["embedding_row"] != row:
            raise ScorerError(f"reference row for {key} disagrees with its item row")
        if item_by_key[key]["source_group"] != group:
            raise ScorerError(f"reference group for {key} disagrees with its item group")
    for item in items:
        if item["reference"] != (_item_key(item) in ref_set):
            raise ScorerError(f"{_item_key(item)} reference flag disagrees with index")


def _threshold_artifact(
    value: Any, *, require_provenance: bool
) -> tuple[float, ZStatistics]:
    if not isinstance(value, Mapping):
        raise ScorerError("threshold artifact must be a mapping")
    if require_provenance:
        _require_exact_keys(value, THRESHOLD_KEYS, "threshold artifact")
    if value.get("schema_version") != "W2D-threshold-v1":
        raise ScorerError("unexpected threshold artifact schema")
    if require_provenance:
        if (
            value.get("selection_rule")
            != "maximize poison-class F1; ties choose largest threshold"
        ):
            raise ScorerError("threshold selection rule is not frozen")
        if value.get("calibration_item_count") != PHASE_COUNTS["calibration"]:
            raise ScorerError("threshold calibration item count is not 448")
        if (
            value.get("calibration_source_group_count")
            != PHASE_COUNTS["calibration"]
        ):
            raise ScorerError("threshold calibration source-group count is not 448")
        _require_sha256(
            value.get("calibration_score_label_digest"),
            "threshold calibration_score_label_digest",
        )
        _require_sha256(
            value.get("calibration_scores_sha256"),
            "threshold calibration_scores_sha256",
        )
        _require_sha256(value.get("labels_sha256"), "threshold labels_sha256")
    threshold = value.get("selected_threshold", value.get("threshold"))
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ScorerError("threshold artifact has no numeric threshold") from exc
    if not math.isfinite(threshold):
        raise ScorerError("threshold must be finite")
    z_value = value.get("z_statistics", value.get("z_stats"))
    try:
        z = ZStatistics.from_value(z_value)
    except DetectorInputError as exc:
        raise ScorerError(str(exc)) from exc
    return threshold, z


def _row_embedding(
    item: Mapping[str, Any], embeddings: np.ndarray
) -> np.ndarray:
    row = _embedding_row(item)
    if row >= embeddings.shape[0]:
        raise ScorerError(
            f"{_item_key(item)} embedding row {row} exceeds "
            f"{embeddings.shape[0]} rows"
        )
    return embeddings[row]


def _raw_record(
    detector: D1Detector,
    item: Mapping[str, Any],
    embedding: np.ndarray,
    reference_embeddings: np.ndarray,
    reference_groups: Sequence[str],
    excluded_source_group: str,
    model_revision: str,
    z_statistics: ZStatistics,
) -> tuple[dict[str, Any], int]:
    text = _text(item)
    evidence = item.get("source_evidence")
    started = time.perf_counter_ns()
    features = detector.content_features(
        text,
        embedding,
        reference_embeddings,
        reference_groups,
        excluded_source_group,
    )
    c_score = detector.content_score(features, z_statistics)
    family_s = detector.source_affirms(
        evidence, model_revision
    )
    elapsed = time.perf_counter_ns() - started
    if elapsed < 0:
        raise ScorerError("perf_counter_ns moved backwards")
    return (
        {
            "rep": features.rep,
            "knn": features.knn,
            "c_score": c_score,
            "family_s_affirms": family_s,
        },
        elapsed,
    )


def _score_without_key(
    detector: D1Detector,
    item: Mapping[str, Any],
    embedding: np.ndarray,
    reference_embeddings: np.ndarray,
    reference_groups: Sequence[str],
    excluded_source_group: str,
    model_revision: str,
    z_statistics: ZStatistics,
    threshold: float,
) -> tuple[dict[str, Any], int]:
    # Keep this call visibly key-free.  Gate 30 also checks the detector
    # signature and changes only the harness key in an otherwise identical row.
    text = _text(item)
    evidence = item.get("source_evidence")
    started = time.perf_counter_ns()
    scored = detector.score(
        text,
        embedding,
        evidence,
        reference_embeddings,
        reference_groups,
        excluded_source_group,
        model_revision,
        z_statistics,
        threshold,
    )
    elapsed = time.perf_counter_ns() - started
    if elapsed < 0:
        raise ScorerError("perf_counter_ns moved backwards")
    return (
        {
            "c_score": scored.c_score,
            "family_s_affirms": scored.family_s_affirms,
            "promote": scored.promote,
        },
        elapsed,
    )


def _snapshot_bytes(path: str) -> tuple[bytes, str]:
    with open(path, "rb") as handle:
        payload = handle.read()
    return payload, hashlib.sha256(payload).hexdigest()


def _assert_frozen_paths_unchanged(
    frozen_paths: Mapping[str, str],
) -> None:
    for path, expected_sha256 in frozen_paths.items():
        try:
            actual_sha256 = file_sha256(path)
        except OSError as exc:
            raise ScorerError(
                f"frozen scorer input/code disappeared during execution: {path}"
            ) from exc
        if actual_sha256 != expected_sha256:
            raise ScorerError(
                f"frozen scorer input/code changed during execution: {path}"
            )


def _require_clean_production_tree() -> None:
    """Require every formal scoring input/code byte to start at a commit."""

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ScorerError(
            "formal scoring could not establish a clean Git worktree"
        ) from exc
    if status.strip():
        changed = [
            line[3:] if len(line) > 3 else line
            for line in status.splitlines()[:20]
        ]
        raise ScorerError(
            "formal scoring requires a clean committed worktree; "
            f"changed paths: {changed}"
        )


def _run_scorer(
    *,
    inputs_path: str,
    embeddings_path: str,
    keys_path: str | None,
    phase: str,
    output_path: str | None = None,
    threshold_artifact_path: str | None = None,
    detector: D1Detector | None = None,
    frozen_production: bool,
    enforce_phase_count: bool,
) -> dict[str, Any]:
    if phase not in PHASE_FIELDS:
        raise ScorerError(
            "phase must be calibration, test, or source_control"
        )
    if phase == "calibration" and threshold_artifact_path is not None:
        raise ScorerError("calibration scoring may not receive a threshold")
    if phase != "calibration" and not threshold_artifact_path:
        raise ScorerError(f"{phase} scoring requires a threshold artifact")

    inputs_bytes, inputs_sha256 = _snapshot_bytes(inputs_path)
    embeddings_bytes, actual_embeddings_sha256 = _snapshot_bytes(
        embeddings_path
    )
    keys_bytes: bytes | None = None
    keys_sha256: str | None = None
    if keys_path:
        keys_bytes, keys_sha256 = _snapshot_bytes(keys_path)
    threshold_bytes: bytes | None = None
    threshold_sha256: str | None = None
    if threshold_artifact_path:
        threshold_bytes, threshold_sha256 = _snapshot_bytes(
            threshold_artifact_path
        )
    detector_path = str(detector_module.__file__)
    scorer_path = str(__file__)
    _detector_bytes, detector_py_sha256 = _snapshot_bytes(detector_path)
    _scorer_bytes, scorer_py_sha256 = _snapshot_bytes(scorer_path)
    frozen_paths = {
        inputs_path: inputs_sha256,
        embeddings_path: actual_embeddings_sha256,
        detector_path: detector_py_sha256,
        scorer_path: scorer_py_sha256,
    }
    if keys_path and keys_sha256 is not None:
        frozen_paths[keys_path] = keys_sha256
    if threshold_artifact_path and threshold_sha256 is not None:
        frozen_paths[threshold_artifact_path] = threshold_sha256

    safe_root = strict_json_load_bytes(inputs_bytes, source=inputs_path)
    _validate_safe_root(safe_root, frozen_production=frozen_production)
    item_list = _items(safe_root)
    by_key = {_item_key(item): item for item in item_list}

    frozen_keys = _safe_phase_keys(safe_root, phase)
    if keys_path:
        assert keys_bytes is not None
        selected_keys = _key_manifest(
            strict_json_load_bytes(keys_bytes, source=keys_path), phase
        )
        if selected_keys != frozen_keys:
            raise ScorerError(
                f"external {phase} keys do not exactly match the frozen phase set"
            )
    else:
        selected_keys = frozen_keys
    if enforce_phase_count and len(selected_keys) != PHASE_COUNTS[phase]:
        raise ScorerError(
            f"{phase} has {len(selected_keys)} items; "
            f"expected {PHASE_COUNTS[phase]}"
        )
    selected_items = [by_key[key] for key in selected_keys]
    frozen_embeddings_sha256 = safe_root["embedding_artifact"]["sha256"]
    if actual_embeddings_sha256 != frozen_embeddings_sha256:
        raise ScorerError(
            "embedding NPZ sha256 does not match the safe input descriptor"
        )
    with np.load(io.BytesIO(embeddings_bytes), allow_pickle=False) as npz:
        embeddings = _embedding_array(npz, safe_root)
        ref_keys, ref_rows, reference_groups, warm_key = _reference_descriptor(
            safe_root
        )
        if set(selected_keys) & set(ref_keys):
            raise ScorerError("a reference item was selected for detector evaluation")
        missing_refs = [key for key in ref_keys if key not in by_key]
        if missing_refs:
            raise ScorerError("a reference key is absent from safe input items")
        reference_items = [by_key[key] for key in ref_keys]
        if any(row >= embeddings.shape[0] for row in ref_rows):
            raise ScorerError("a reference embedding row is out of range")
        reference_embeddings = np.asarray([embeddings[row] for row in ref_rows])
        reference_texts = [_text(item) for item in reference_items]
        model_revision = _model_revision(safe_root)

        if detector is None:
            dimension = int(safe_root["model"]["dimension"])
            detector = D1Detector(
                pinned_model_revision=model_revision, expected_dim=dimension
            )
        if (
            detector.pinned_model_revision != model_revision
            or detector.expected_dim != safe_root["model"]["dimension"]
            or detector.model_name != safe_root["model"]["name"]
        ):
            raise ScorerError("detector configuration differs from the safe input")

        threshold: float | None = None
        if phase == "calibration":
            z_statistics = detector.reference_z_statistics(
                reference_texts, reference_embeddings, reference_groups
            )
        else:
            assert threshold_bytes is not None
            threshold, z_statistics = _threshold_artifact(
                strict_json_load_bytes(
                    threshold_bytes, source=str(threshold_artifact_path)
                ),
                require_provenance=frozen_production,
            )

        # One fixed reference warm-up is deliberately excluded from the output.
        warm = by_key[warm_key]
        warm_embedding = _row_embedding(warm, embeddings)
        warm_group = reference_groups[ref_keys.index(warm_key)]
        if threshold is None:
            _raw_record(
                detector,
                warm,
                warm_embedding,
                reference_embeddings,
                reference_groups,
                warm_group,
                model_revision,
                z_statistics,
            )
        else:
            _score_without_key(
                detector,
                warm,
                warm_embedding,
                reference_embeddings,
                reference_groups,
                warm_group,
                model_revision,
                z_statistics,
                threshold,
            )

        records: list[dict[str, Any]] = []
        # Calibration/control manifests are frozen lexically, while test is
        # already in A2.4's label-blind score order. Scoring always uses that
        # one canonical order and never a caller-provided permutation.
        ordered = sorted(
            selected_items, key=lambda item: score_order_key(_item_key(item))
        )
        for item in ordered:
            embedding = _row_embedding(item, embeddings)
            if threshold is None:
                result, elapsed = _raw_record(
                    detector,
                    item,
                    embedding,
                    reference_embeddings,
                    reference_groups,
                    "",
                    model_revision,
                    z_statistics,
                )
            else:
                result, elapsed = _score_without_key(
                    detector,
                    item,
                    embedding,
                    reference_embeddings,
                    reference_groups,
                    "",
                    model_revision,
                    z_statistics,
                    threshold,
                )
            # The key is attached only after the detector call returns.
            records.append(
                {
                    "item_key": _item_key(item),
                    **result,
                    "detector_service_ns": elapsed,
                }
            )

    _assert_frozen_paths_unchanged(frozen_paths)

    artifact: dict[str, Any] = {
        "schema_version": "W2D-score-v1",
        "phase": phase,
        "measurement_name": MEASUREMENT_NAME,
        "model_name": detector.model_name,
        "model_revision": detector.pinned_model_revision,
        "warmup_item_key": warm_key,
        "score_order_rule": 'SHA256("W2D-score-order|" + item_key)',
        "safe_inputs_sha256": inputs_sha256,
        "embeddings_sha256": actual_embeddings_sha256,
        "detector_py_sha256": detector_py_sha256,
        "scorer_py_sha256": scorer_py_sha256,
        "z_statistics": z_statistics.to_dict(),
        "threshold": threshold,
        "items": records,
    }
    if keys_path:
        artifact["keys_sha256"] = keys_sha256
    else:
        artifact["keys_source"] = "safe input frozen sets"
    if threshold_artifact_path:
        artifact["threshold_artifact_sha256"] = threshold_sha256
    if output_path:
        wrote_output = False
        try:
            strict_json_dump(artifact, output_path)
            wrote_output = True
            _assert_frozen_paths_unchanged(frozen_paths)
        except BaseException:
            if wrote_output:
                Path(output_path).unlink(missing_ok=True)
            raise
    return artifact


def run_scorer(
    *,
    inputs_path: str,
    embeddings_path: str,
    keys_path: str | None,
    phase: str,
    output_path: str | None = None,
    threshold_artifact_path: str | None = None,
    detector: D1Detector | None = None,
) -> dict[str, Any]:
    """Run the frozen production contract with exact schemas and denominators."""
    _require_clean_production_tree()
    return _run_scorer(
        inputs_path=inputs_path,
        embeddings_path=embeddings_path,
        keys_path=keys_path,
        phase=phase,
        output_path=output_path,
        threshold_artifact_path=threshold_artifact_path,
        detector=detector,
        frozen_production=True,
        enforce_phase_count=True,
    )


def run_scorer_test_fixture(
    *,
    inputs_path: str,
    embeddings_path: str,
    keys_path: str | None,
    phase: str,
    output_path: str | None = None,
    threshold_artifact_path: str | None = None,
    detector: D1Detector | None = None,
) -> dict[str, Any]:
    """Named small-fixture entry point; schema/hash rules still remain active."""
    return _run_scorer(
        inputs_path=inputs_path,
        embeddings_path=embeddings_path,
        keys_path=keys_path,
        phase=phase,
        output_path=output_path,
        threshold_artifact_path=threshold_artifact_path,
        detector=detector,
        frozen_production=False,
        enforce_phase_count=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument(
        "--keys",
        help="optional opaque key manifest; defaults to the safe artifact's frozen sets",
    )
    parser.add_argument(
        "--phase",
        choices=("calibration", "test", "source_control"),
        required=True,
    )
    parser.add_argument("--threshold-artifact")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    run_scorer(
        inputs_path=args.inputs,
        embeddings_path=args.embeddings,
        keys_path=args.keys,
        phase=args.phase,
        output_path=args.output,
        threshold_artifact_path=args.threshold_artifact,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
