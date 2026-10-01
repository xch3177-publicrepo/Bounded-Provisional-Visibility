#!/usr/bin/env python3
"""Run the frozen W2D E1/E2 promotion-path replay grid.

D1 has already scored each unique test item exactly once.  This runner never
imports or calls D1.  It replays the frozen decision and measured service time
through B2/B3/B4, runs the paired construction-label oracle arm, and retains B1
as one off-path undefended control.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import io
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

import analysis
from backend import InMemoryBackend
from functional_slice import State, VISIBLE, VerifierDecision
from milvus_backend import MilvusBackend
from poison_exposure import (
    CFG_REALTEXT,
    corpus_topk,
    id_base,
    reset_corpus,
    run_cell,
)
from w2d_metrics import (
    FINAL_SCORE_KEYS,
    detector_quality_summary,
    latency_summary,
    rate_record,
    source_control_gate_summary,
    strict_json_load_bytes,
)
from w2d_calibrate import select_threshold
from w2d_backend_evidence import (
    BackendEvidenceError,
    capture_backend_evidence,
    summarize_expected_seed_state,
)
from w2d_equivalence import A9EquivalenceError, validate_a9_replay_equivalence
from verify_w2d import (
    EXECUTION_MODE,
    EXECUTION_TIER_CONTRACTS,
    FINGERPRINT_FILE_COMPONENTS,
    FROZEN_MODEL_REVISION,
    FROZEN_PROVIDER,
    LEGACY_RUNTIME_ARCHIVED_SHA256,
    MANIFEST_AUTHORITY_BASENAME,
    MANIFEST_AUTHORITY_SCHEMA,
    MANIFEST_SCHEMA,
    MEASUREMENT_NAME,
    RESULT_SCHEMA,
    RUNNER_BACKEND_TO_TIER,
    W2DVerificationError,
    canonical_record_sha256,
    capture_manifest_authority_binding,
    runtime_fingerprint_digest,
    validate_legacy_gate_code_compatibility,
    validate_legacy_runtime_code_compatibility,
    verify_a6_artifact_lineage,
    verify_or_raise,
)
from verify_w2d_legacy import evaluate as evaluate_legacy_regression
from verify_w2d_legacy import (
    FROZEN_CANDIDATE_COMMIT,
    FROZEN_SOURCE_SHA256,
    LEGACY_GATE_FILES,
    LEGACY_GATE_COMMANDS,
    LEGACY_RUNTIME_FILES,
    canonical_strict_json_sha256,
    FROZEN_SAME_CODE_DIFFERENCE_SHA256,
    rerun_legacy_gates,
    validate_frozen_same_code_differences,
)
from w2d_snapshot import (
    FileSnapshot,
    OutputOwnership,
    OutputOwnershipError,
    SnapshotError,
    SnapshotRegistry,
    assert_owned_output,
    exclusive_create_bytes,
    preserve_failed_output,
)


HERE = Path(__file__).resolve().parent
SCHEMA_VERSION = RESULT_SCHEMA
OPAQUE_KEY_LENGTH = 64
MODEL_REVISION = FROZEN_MODEL_REVISION

DEFAULT_PATHS = {
    "legacy_authority_manifest": HERE / "results" / "AUTHORITATIVE.json",
    "legacy_authoritative_w2": HERE / "results" / "W2-inmemory.json",
    "legacy_authoritative_w2r": HERE / "results" / "W2R-inmemory.json",
    "legacy_candidate_w2": (
        HERE / "results" / "w2d" / "W2-legacy-regression-run-a.json"
    ),
    "legacy_candidate_w2r": (
        HERE / "results" / "w2d" / "W2R-legacy-regression.json"
    ),
    "legacy_same_code_repeat": (
        HERE / "results" / "w2d"
        / "W2-legacy-regression-run-b-same-code.json"
    ),
    "legacy_regression": (
        HERE / "results" / "w2d" / "W2D-LEGACY-REGRESSION.json"
    ),
    "freeze": HERE / "data" / "w2d" / "W2D-DATA-FREEZE.json",
    "inputs": HERE / "data" / "w2d" / "W2D-detector-inputs.json",
    "embeddings": HERE / "data" / "w2d" / "W2D-detector-inputs.npz",
    "labels": HERE / "results" / "w2d" / "W2D-labels.json",
    "labels_checksum": HERE / "results" / "w2d" / "W2D-labels.sha256",
    "plan_pre_a9": (
        HERE
        / "results"
        / "w2d"
        / "W2D-PROTOCOL-PLAN-preA7-landing-contract-failure.json"
    ),
    "plan": HERE / "results" / "w2d" / "W2D-PROTOCOL-PLAN.json",
    "calibration_scores": (
        HERE / "results" / "w2d" / "W2D-calibration-scores.json"
    ),
    "threshold": HERE / "results" / "w2d" / "W2D-threshold.json",
    "test_scores": HERE / "results" / "w2d" / "W2D-test-scores.json",
    "source_controls": HERE / "results" / "w2d" / "W2D-source-controls.json",
    "source_control_gate": (
        HERE / "results" / "w2d" / "W2D-source-control-gate.json"
    ),
    "detector_metrics": (
        HERE / "results" / "w2d" / "W2D-detector-metrics.json"
    ),
    "landing_pre_a9": (
        HERE / "results" / "w2d" / "W2D-LANDING-preA9-lineage.json"
    ),
    "landing": HERE / "results" / "w2d" / "W2D-LANDING.json",
    "output": HERE / "results" / "w2d" / "W2D-E1-INMEMORY.json",
}
DEFAULT_MANIFEST = (
    HERE
    / "results"
    / "w2d"
    / EXECUTION_TIER_CONTRACTS["E1_INMEMORY"]["manifest_filename"]
)

ARTIFACT_NAMES = {
    "legacy_authority_manifest": "legacy_authority_manifest",
    "legacy_authoritative_w2": "legacy_authoritative_w2",
    "legacy_authoritative_w2r": "legacy_authoritative_w2r",
    "legacy_candidate_w2": "legacy_candidate_w2",
    "legacy_candidate_w2r": "legacy_candidate_w2r",
    "legacy_same_code_repeat": "legacy_same_code_repeat",
    "legacy_regression": "legacy_regression",
    "freeze": "data_freeze",
    "inputs": "safe_inputs_json",
    "embeddings": "safe_inputs_npz",
    "labels": "labels",
    "labels_checksum": "labels_checksum",
    "plan_pre_a9": "protocol_plan_pre_a9",
    "plan": "protocol_plan",
    "calibration_scores": "calibration_scores",
    "threshold": "threshold",
    "test_scores": "test_scores",
    "source_controls": "source_controls",
    "source_control_gate": "source_control_gate",
    "detector_metrics": "detector_metrics",
    "landing_pre_a9": "landing_pre_a9",
    "landing": "landing",
}


class RunnerError(RuntimeError):
    pass


def _verify_a6_from_snapshots(
    *,
    artifacts: Mapping[str, Any],
    hashes: Mapping[str, str],
    paths: Mapping[str, Any],
    artifact_snapshots: Mapping[str, FileSnapshot],
    component_snapshots: Mapping[str, FileSnapshot],
    registry: SnapshotRegistry,
) -> None:
    """Run the shared A6 gate entirely from the formal byte snapshots."""

    try:
        verify_a6_artifact_lineage(
            artifacts,
            hashes,
            paths,
            artifact_snapshots=artifact_snapshots,
            component_snapshots=component_snapshots,
        )
        registry.assert_unchanged()
    except SnapshotError as exc:
        raise RunnerError(
            "formal inputs moved during the shared A6 lineage gate"
        ) from exc


def _is_key(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == OPAQUE_KEY_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_tree(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RunnerError(f"{path} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_tree(child, f"{path}[{index}]")
        return
    raise RunnerError(f"{path} contains unsupported {type(value).__name__}")


def _strict_json_bytes(value: Any) -> bytes:
    _finite_tree(value)
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _exclusive_json_dump(
    value: Any, path: str | os.PathLike[str]
) -> OutputOwnership:
    return exclusive_create_bytes(path, _strict_json_bytes(value))


def _relative_to(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve())


def _capture_runtime_fingerprint(
    registry: SnapshotRegistry,
    base_dir: Path,
) -> tuple[dict[str, Any], dict[str, FileSnapshot]]:
    """Build the exact manifest fingerprint from one captured byte instance."""

    components: dict[str, dict[str, str]] = {}
    snapshots: dict[str, FileSnapshot] = {}
    resolved: dict[Path, str] = {}
    for component, filename in FINGERPRINT_FILE_COMPONENTS.items():
        snapshot = registry.capture(HERE / filename)
        if snapshot.path in resolved:
            raise RunnerError(
                f"runtime components {resolved[snapshot.path]} and {component} "
                "resolve to one path"
            )
        resolved[snapshot.path] = component
        snapshots[component] = snapshot
        components[component] = {
            "path": _relative_to(snapshot.path, base_dir),
            "sha256": snapshot.sha256,
        }
    model_digest = hashlib.sha256(MODEL_REVISION.encode("utf-8")).hexdigest()
    components["model_revision"] = {
        "value": MODEL_REVISION,
        "sha256": model_digest,
    }
    return (
        {
            "algorithm": "sha256",
            "components": components,
            "sha256": runtime_fingerprint_digest(components),
        },
        snapshots,
    )


def _runtime_hash_by_filename(
    snapshots: Mapping[str, FileSnapshot],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for snapshot in snapshots.values():
        filename = snapshot.path.name
        if filename in result:
            raise RunnerError(
                f"runtime fingerprint contains duplicate filename {filename}"
            )
        result[filename] = snapshot.sha256
    return result


def _legacy_runtime_sha256(
    runtime_snapshots: Mapping[str, FileSnapshot],
) -> str:
    by_filename = {value.path.name: value for value in runtime_snapshots.values()}
    digest = hashlib.sha256()
    for filename in sorted(LEGACY_RUNTIME_FILES):
        snapshot = by_filename.get(filename)
        if snapshot is None:
            raise RunnerError(
                f"legacy runtime file is absent from fingerprint: {filename}"
            )
        digest.update(filename.encode("utf-8"))
        digest.update(snapshot.payload)
    return digest.hexdigest()


def _legacy_runtime_hashes(
    runtime_snapshots: Mapping[str, FileSnapshot],
) -> dict[str, str]:
    by_filename = _runtime_hash_by_filename(runtime_snapshots)
    missing = set(LEGACY_RUNTIME_FILES) - set(by_filename)
    if missing:
        raise RunnerError(
            f"legacy runtime files are absent from fingerprint: {sorted(missing)}"
        )
    return {
        filename: by_filename[filename]
        for filename in LEGACY_RUNTIME_FILES
    }


def _legacy_gate_hashes(
    runtime_snapshots: Mapping[str, FileSnapshot],
) -> dict[str, str]:
    by_filename = {
        value.path.name: value.sha256 for value in runtime_snapshots.values()
    }
    missing = set(LEGACY_GATE_FILES) - set(by_filename)
    if missing:
        raise RunnerError(
            f"legacy gate files are absent from fingerprint: {sorted(missing)}"
        )
    return {filename: by_filename[filename] for filename in LEGACY_GATE_FILES}


def _safe_items(document: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(document, Mapping) or not isinstance(
        document.get("items"), list
    ):
        raise RunnerError("safe artifact has no items list")
    result: dict[str, Mapping[str, Any]] = {}
    for record in document["items"]:
        if not isinstance(record, Mapping):
            raise RunnerError("safe item is not an object")
        key = record.get("item_key")
        if not _is_key(key) or key in result:
            raise RunnerError("safe artifact has missing/duplicate opaque key")
        result[key] = record
    return result


def _label_items(document: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(document, Mapping) or not isinstance(
        document.get("items"), Mapping
    ):
        raise RunnerError("label artifact has no key-to-item mapping")
    result = dict(document["items"])
    if any(not _is_key(key) or not isinstance(value, Mapping)
           for key, value in result.items()):
        raise RunnerError("label artifact contains a malformed record")
    return result


def _score_records(
    document: Any,
    *,
    safe_inputs_sha256: str,
    embeddings_sha256: str,
    threshold_sha256: str,
    detector_py_sha256: str,
    scorer_py_sha256: str,
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(document, Mapping) or document.get("phase") != "test":
        raise RunnerError("test score artifact has the wrong phase")
    if document.get("schema_version") != "W2D-score-v1":
        raise RunnerError("test score artifact has the wrong schema")
    if document.get("measurement_name") != MEASUREMENT_NAME:
        raise RunnerError("test score artifact misnames the replay measurement")
    expected_lineage = {
        "safe_inputs_sha256": safe_inputs_sha256,
        "embeddings_sha256": embeddings_sha256,
        "threshold_artifact_sha256": threshold_sha256,
        "detector_py_sha256": detector_py_sha256,
        "scorer_py_sha256": scorer_py_sha256,
    }
    for field, expected in expected_lineage.items():
        if document.get(field) != expected:
            raise RunnerError(
                f"test score {field} does not match the frozen chain"
            )
    if document.get("keys_source") != "safe input frozen sets":
        raise RunnerError("formal test score did not use the frozen key set")
    raw = document.get("items")
    if not isinstance(raw, list) or len(raw) != 512:
        raise RunnerError("test score artifact must contain exactly 512 items")
    result: dict[str, Mapping[str, Any]] = {}
    for record in raw:
        if not isinstance(record, Mapping):
            raise RunnerError("test score record is not an object")
        key = record.get("item_key")
        if not _is_key(key) or key in result:
            raise RunnerError("test score artifact has a duplicate/missing key")
        if set(record) != FINAL_SCORE_KEYS:
            raise RunnerError("test score record does not have the exact schema")
        if not isinstance(record.get("family_s_affirms"), bool):
            raise RunnerError("test score family_s_affirms must be bool")
        if not isinstance(record.get("promote"), bool):
            raise RunnerError("test score promote must be bool")
        score = record.get("c_score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise RunnerError("test score c_score must be finite")
        _service_ns(record)
        result[key] = record
    return result


def _decision(record: Mapping[str, Any]) -> bool:
    value = record.get("promote")
    if not isinstance(value, bool):
        raise RunnerError("test score record promote must be a JSON boolean")
    return value


def _service_ns(record: Mapping[str, Any]) -> int:
    value = record.get("detector_service_ns")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RunnerError("test score record has invalid detector service time")
    return value


class ReplayBank:
    """Frozen O(1) result lookup; it never runs D1 or opens evaluator files."""

    def __init__(
        self,
        *,
        arm: str,
        scores: Mapping[str, Mapping[str, Any]],
        poison_truth: Mapping[str, bool],
        score_artifact_sha256: str,
    ) -> None:
        if arm not in {"oracle", "detector"}:
            raise RunnerError(f"invalid replay arm {arm!r}")
        self.arm = arm
        self.scores = scores
        self.poison_truth = poison_truth
        self.score_artifact_sha256 = score_artifact_sha256

    async def __call__(self, request: Any) -> VerifierDecision:
        key = request.item_key
        if not _is_key(key) or key not in self.scores:
            raise RunnerError("replay requested an unknown opaque item key")
        # A replay provider must not need detector features.  The payload is a
        # marker only and makes an accidental live-score provider conspicuous.
        if request.verifier_input != {"execution_mode": EXECUTION_MODE}:
            raise RunnerError("runtime supplied a non-replay verifier payload")
        score = self.scores[key]
        detector_passes = _decision(score)
        passes = (
            not self.poison_truth[key]
            if self.arm == "oracle"
            else detector_passes
        )
        metadata = {
            "execution_mode": EXECUTION_MODE,
            "measurement_name": MEASUREMENT_NAME,
            "arm": self.arm,
            "score_artifact_sha256": self.score_artifact_sha256,
            "score_item_key": key,
            "detector_decision": detector_passes,
            "family_s_affirms": bool(score.get("family_s_affirms")),
        }
        return VerifierDecision(
            passes=passes,
            service_time_s=_service_ns(score) / 1_000_000_000.0,
            metadata=metadata,
        )


def _unit_vector(value: Any, name: str) -> list[float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (384,) or not np.isfinite(array).all():
        raise RunnerError(f"{name} is not a finite 384-vector")
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm <= 0:
        raise RunnerError(f"{name} has invalid norm")
    return [float(number) for number in array / norm]


class FrozenInputs:
    def __init__(
        self,
        *,
        documents: Mapping[str, Mapping[str, Any]],
        embeddings_snapshot: FileSnapshot,
        artifact_hashes: Mapping[str, str],
        runtime_hashes_by_filename: Mapping[str, str],
    ) -> None:
        self.safe_document = documents["inputs"]
        self.labels_document = documents["labels"]
        self.plan = documents["plan"]
        self.score_document = documents["test_scores"]
        self.landing_document = documents["landing"]
        if not all(
            isinstance(value, Mapping)
            for value in (
                self.safe_document,
                self.labels_document,
                self.plan,
                self.score_document,
                self.landing_document,
            )
        ):
            raise RunnerError("one W2D artifact is not a JSON object")

        self.safe = _safe_items(self.safe_document)
        self.labels = _label_items(self.labels_document)
        self.scores = _score_records(
            self.score_document,
            safe_inputs_sha256=artifact_hashes["safe_inputs_json"],
            embeddings_sha256=artifact_hashes["safe_inputs_npz"],
            threshold_sha256=artifact_hashes["threshold"],
            detector_py_sha256=runtime_hashes_by_filename["detector.py"],
            scorer_py_sha256=runtime_hashes_by_filename["w2d_scorer.py"],
        )
        if set(self.scores) != set(
            self.safe_document.get("sets", {}).get("test_score_order", [])
        ):
            raise RunnerError("test score keys differ from frozen test set")
        if set(self.scores) - set(self.labels):
            raise RunnerError("a test score has no evaluator label")

        with np.load(
            io.BytesIO(embeddings_snapshot.payload), allow_pickle=False
        ) as archive:
            if archive.files != ["embeddings"]:
                raise RunnerError("safe NPZ must contain only embeddings")
            self.embeddings = np.asarray(
                archive["embeddings"], dtype=np.float64
            )
        if self.embeddings.ndim != 2 or self.embeddings.shape[1] != 384:
            raise RunnerError("safe embedding matrix has invalid dimensions")
        if not np.isfinite(self.embeddings).all():
            raise RunnerError("safe embedding matrix contains non-finite values")

        self.landed = {}
        raw_landing = self.landing_document.get("items")
        if not isinstance(raw_landing, list) or len(raw_landing) != 192:
            raise RunnerError("landing artifact must contain all 192 poisons")
        for record in raw_landing:
            key = record.get("item_key") if isinstance(record, Mapping) else None
            landed = (
                record.get("landed_top5", record.get("landed"))
                if isinstance(record, Mapping)
                else None
            )
            if not _is_key(key) or not isinstance(landed, bool):
                raise RunnerError("landing artifact contains malformed record")
            self.landed[key] = landed

        self.poison_truth = {}
        for key in self.scores:
            truth = self.labels[key].get("poison")
            if not isinstance(truth, bool):
                raise RunnerError(f"label {key} poison must be a JSON boolean")
            self.poison_truth[key] = truth
        self.query_materials = self.plan.get("query_materials")
        self.planned_queries = self.plan.get("planned_poison_queries")
        if not isinstance(self.query_materials, Mapping) or not isinstance(
            self.planned_queries, Mapping
        ):
            raise RunnerError("protocol plan has no frozen query materials")

    def vector(self, key: str) -> list[float]:
        if key not in self.safe:
            raise RunnerError(f"safe item {key} is missing")
        row = self.safe[key].get("embedding_row")
        if isinstance(row, bool) or not isinstance(row, int):
            raise RunnerError(f"safe item {key} has invalid embedding row")
        if not 0 <= row < self.embeddings.shape[0]:
            raise RunnerError(f"safe item {key} embedding row is out of range")
        return _unit_vector(self.embeddings[row], f"item {key}")

    def query(self, poison_key: str, role: str) -> list[float]:
        descriptor = self.planned_queries.get(poison_key)
        if not isinstance(descriptor, Mapping):
            raise RunnerError(f"{poison_key} has no planned queries")
        role_value = descriptor.get("roles", {}).get(role)
        if not isinstance(role_value, Mapping):
            raise RunnerError(f"{poison_key} has no {role} query")
        query_key = role_value.get("query_item_key")
        material = self.query_materials.get(query_key)
        if not _is_key(query_key) or not isinstance(material, Mapping):
            raise RunnerError(f"{poison_key}/{role} query material is absent")
        return _unit_vector(
            material.get("embedding"), f"query {query_key}"
        )


def _plan_units(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = plan.get("runtime_units")
    if not isinstance(raw, list) or len(raw) != 20:
        raise RunnerError("protocol plan must have exactly 20 runtime units")
    result = {}
    for unit in raw:
        if not isinstance(unit, Mapping):
            raise RunnerError("runtime unit is not an object")
        unit_id = unit.get("runtime_unit_id")
        if not isinstance(unit_id, str) or not unit_id or unit_id in result:
            raise RunnerError("runtime unit id is missing or duplicate")
        result[unit_id] = unit
    return result


def _plan_cells(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = plan.get("cells")
    if not isinstance(raw, list) or len(raw) != 140:
        raise RunnerError("protocol plan must have exactly 140 cells")
    seen = set()
    cells = []
    for cell in raw:
        if not isinstance(cell, Mapping):
            raise RunnerError("protocol cell is not an object")
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id or cell_id in seen:
            raise RunnerError("protocol cell id is missing or duplicate")
        seen.add(cell_id)
        cells.append(cell)
    for seed in range(1, 6):
        positions = sorted(
            int(cell["execution_order_position"])
            for cell in cells
            if int(cell.get("seed", -1)) == seed
        )
        if positions != list(range(1, 29)):
            raise RunnerError(f"seed {seed} execution positions are not 1..28")
    return sorted(
        cells,
        key=lambda cell: (
            int(cell["seed"]),
            int(cell["execution_order_position"]),
        ),
    )


def _world_for_unit(
    frozen: FrozenInputs, unit: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    poison = list(unit.get("poison_item_keys", []))
    clean = list(unit.get("clean_item_keys", []))
    filler = list(unit.get("filler_item_keys", []))
    query_poison = list(unit.get("query_poison_item_keys", poison))
    backlog = unit.get("backlog")
    expected_filler = 12 if backlog == "heavy" else 0
    if (
        len(poison) != 6
        or len(clean) != 6
        or len(filler) != expected_filler
    ):
        raise RunnerError(
            "runtime unit must contain 6 poison, 6 clean, and "
            f"{expected_filler} filler for {backlog}"
        )
    if query_poison != poison:
        raise RunnerError("query_poison_item_keys must preserve poison order")
    all_runtime = poison + clean + filler
    if len(all_runtime) != len(set(all_runtime)):
        raise RunnerError("runtime unit reuses an item across roles")
    if any(key not in frozen.scores for key in all_runtime):
        raise RunnerError("runtime unit references an unscored test item")
    if any(not frozen.poison_truth[key] for key in poison):
        raise RunnerError("runtime poison role contains a clean item")
    if any(frozen.poison_truth[key] for key in clean + filler):
        raise RunnerError("runtime clean/filler role contains poison")

    background = frozen.plan.get("retrieval_background", {}).get("item_keys")
    if not isinstance(background, list) or len(background) != 768:
        raise RunnerError("plan retrieval background is not 768 items")
    world = {
        "corpus": [frozen.vector(key) for key in background],
        "poison": [frozen.vector(key) for key in poison],
        "clean": [frozen.vector(key) for key in clean],
        "filler": [frozen.vector(key) for key in filler],
        "q_craft": [
            frozen.query(key, "attack_associated") for key in query_poison
        ],
        "q_target": [
            frozen.query(key, "heldout_same_topic") for key in query_poison
        ],
        "q_neg": [
            frozen.query(key, "negative_other_topic") for key in query_poison
        ],
        "geometry": {
            "retrieval_background_n": len(background),
            "query_role_aliases": {
                "craft": "attack_associated",
                "target": "heldout_same_topic",
                "negative": "negative_other_topic",
            },
        },
    }
    return world, {"poison": poison, "clean": clean, "filler": filler}


def _role_map(assignments: Mapping[str, Sequence[str]]) -> dict[str, str]:
    result = {}
    for role, keys in assignments.items():
        for key in keys:
            if key in result:
                raise RunnerError("item appears under multiple runtime roles")
            result[key] = role
    return result


def _protocol_item_records(
    unit: Mapping[str, Any], frozen: FrozenInputs
) -> list[dict[str, Any]]:
    sequence = unit.get("injection_sequence")
    if not isinstance(sequence, list):
        raise RunnerError("runtime unit has no injection sequence")
    result: list[dict[str, Any]] = []
    for expected_ordinal, event in enumerate(sequence):
        if not isinstance(event, Mapping):
            raise RunnerError("runtime injection event is not an object")
        key = event.get("item_key")
        role = event.get("role")
        ordinal = event.get("injection_ordinal")
        if (
            not _is_key(key)
            or key not in frozen.scores
            or role not in {"poison", "clean", "filler"}
            or ordinal != expected_ordinal
        ):
            raise RunnerError("runtime injection sequence is malformed")
        score = frozen.scores[key]
        result.append(
            {
                "item_key": key,
                "role": role,
                "ordinal": ordinal,
                "detector_promote": _decision(score),
                "detector_service_ns": _service_ns(score),
                "score_record_sha256": canonical_record_sha256(score),
            }
        )
    return result


def _visible_intervals(
    *,
    item: Mapping[str, Any],
    transitions: Sequence[tuple[float, Any, Any, Any]],
    horizon_abs: float,
) -> tuple[float, float]:
    """Return visible and unavailable time from arrival through the horizon."""
    arrival = float(item["ts"]["arrival"])
    cursor = arrival
    visible = False
    visible_total = 0.0
    unavailable_total = 0.0
    for event_time, _iid, _old, new in sorted(
        transitions, key=lambda value: value[0]
    ):
        event_time = min(max(float(event_time), arrival), horizon_abs)
        duration = max(0.0, event_time - cursor)
        if visible:
            visible_total += duration
        else:
            unavailable_total += duration
        visible = new in VISIBLE
        cursor = event_time
    duration = max(0.0, horizon_abs - cursor)
    if visible:
        visible_total += duration
    else:
        unavailable_total += duration
    horizon = max(0.0, horizon_abs - arrival)
    if abs((visible_total + unavailable_total) - horizon) > 1e-5:
        raise RunnerError("visibility ledger does not balance to the horizon")
    return visible_total, unavailable_total


def _unvetted_visibility_episode(
    *,
    transitions: Sequence[tuple[float, Any, Any, Any]],
    horizon_abs: float,
) -> tuple[str, float | None]:
    """Measure the first and only PROVISIONAL episode through the horizon."""

    entries = sorted(
        float(event_time)
        for event_time, _iid, old, new in transitions
        if new == State.PROVISIONAL
        and old != State.PROVISIONAL
        and float(event_time) <= horizon_abs
    )
    if not entries:
        return "NOT_STARTED", None
    if len(entries) != 1:
        raise RunnerError("an item opened more than one unvetted episode")
    started = entries[0]
    exits = sorted(
        float(event_time)
        for event_time, _iid, old, new in transitions
        if old == State.PROVISIONAL
        and new != State.PROVISIONAL
        and started <= float(event_time) <= horizon_abs
    )
    if exits:
        return "COMPLETED", max(0.0, exits[0] - started)
    return "RIGHT_CENSORED", max(0.0, horizon_abs - started)


def _lifecycle_records(
    *,
    sys: Any,
    assignments: Mapping[str, Sequence[str]],
    id_to_key: Mapping[int, str],
    frozen: FrozenInputs,
    arm: str,
    t0: float,
    horizon: float,
) -> dict[str, list[dict[str, Any]]]:
    key_to_id = {key: iid for iid, key in id_to_key.items()}
    result: dict[str, list[dict[str, Any]]] = {
        "poison": [],
        "clean": [],
        "filler": [],
    }
    horizon_abs = t0 + horizon
    terminal_nonvisible = {State.QUARANTINED, State.REVOKED}
    for role in result:
        for ordinal, key in enumerate(assignments[role]):
            iid = key_to_id.get(key)
            if iid is None or iid not in sys.items:
                raise RunnerError(f"{role} item {key} was not admitted")
            item = sys.items[iid]
            transitions = [
                transition
                for transition in sys.transitions
                if transition[1] == iid
                and float(transition[0]) <= horizon_abs
            ]
            if not transitions:
                raise RunnerError(f"{role} item {key} has no state transition")
            transitions.sort(key=lambda transition: float(transition[0]))
            visible_s, unavailable_s = _visible_intervals(
                item=item,
                transitions=transitions,
                horizon_abs=horizon_abs,
            )
            unvetted_status, unvetted_observed_s = (
                _unvetted_visibility_episode(
                    transitions=transitions,
                    horizon_abs=horizon_abs,
                )
            )
            score = frozen.scores[key]
            detector_passes = _decision(score)
            replay_passes = (
                not frozen.poison_truth[key]
                if arm == "oracle"
                else detector_passes
            )
            ts = item["ts"]
            ever_visible = any(
                transition[3] in VISIBLE for transition in transitions
            )
            visible_entries = [
                transition
                for transition in transitions
                if transition[3] in VISIBLE and transition[2] not in VISIBLE
            ]
            quarantine_entries = [
                transition
                for transition in transitions
                if transition[3] == State.QUARANTINED
            ]
            state_at_horizon = transitions[-1][3]
            visible_at_horizon = state_at_horizon in VISIBLE
            nonvisible_at_horizon = not visible_at_horizon
            record = {
                "item_key": key,
                "role": role,
                "ordinal": ordinal,
                "truth_poison": frozen.poison_truth[key],
                "detector_decision": detector_passes,
                "replay_decision": replay_passes,
                "detector_service_ns": _service_ns(score),
                "arrival_s": max(0.0, float(ts["arrival"]) - t0),
                "transitions": [
                    {
                        "t_s": max(0.0, float(event_time) - t0),
                        "old_state": (
                            None if old is None else old.value
                        ),
                        "new_state": new.value,
                    }
                    for event_time, _item_id, old, new in transitions
                ],
                "state_at_horizon": state_at_horizon.value,
                "ever_visible": ever_visible,
                "visible_at_horizon": visible_at_horizon,
                "first_visible_delay_s": (
                    max(0.0, float(ts["visible"]) - float(ts["arrival"]))
                    if ever_visible
                    else None
                ),
                "restricted_visible_time_s": visible_s,
                "restricted_unavailable_time_s": unavailable_s,
                "retry_count": 0,
                "readmission_count": max(0, len(visible_entries) - 1),
                "exposure_status": (
                    "NOT_STARTED"
                    if not ever_visible
                    else (
                        "RIGHT_CENSORED"
                        if visible_at_horizon
                        else "COMPLETED"
                    )
                ),
                "unvetted_visibility_status": unvetted_status,
                "unvetted_visibility_observed_s": unvetted_observed_s,
                "unavailable_status": (
                    "RIGHT_CENSORED"
                    if nonvisible_at_horizon
                    else (
                        "COMPLETED"
                        if unavailable_s > 0.0
                        else "NOT_STARTED"
                    )
                ),
                "quarantine_status": (
                    "RIGHT_CENSORED"
                    if state_at_horizon == State.QUARANTINED
                    else (
                        "COMPLETED"
                        if quarantine_entries
                        else "NOT_STARTED"
                    )
                ),
                "observed_at_horizon_s": horizon,
            }
            if role == "poison":
                record.update(
                    false_promotion=replay_passes,
                    right_censored=(
                        ever_visible and state_at_horizon not in terminal_nonvisible
                    ),
                    landed_top5=bool(frozen.landed[key]),
                )
            else:
                record.update(
                    false_positive=not replay_passes,
                    never_visible=not ever_visible,
                    durable_visible=state_at_horizon in VISIBLE,
                    quarantine_right_censored=(
                        state_at_horizon == State.QUARANTINED
                    ),
                )
            result[role].append(record)
    return result


def _canonical_lifecycle_records(
    lifecycle: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    """Flatten the evaluator ledger into the verifier's exact item schema."""

    result: list[dict[str, Any]] = []
    for role in ("filler", "poison", "clean"):
        for record in lifecycle.get(role, []):
            ever_visible = bool(record["ever_visible"])
            result.append(
                {
                    "item_key": record["item_key"],
                    "truth_poison": record["truth_poison"],
                    "detector_promote": record["detector_decision"],
                    "retry_count": record["retry_count"],
                    "arrival_s": record["arrival_s"],
                    "transitions": record["transitions"],
                    "state_at_horizon": record["state_at_horizon"],
                    "visibility_status": (
                        "STARTED" if ever_visible else "NOT_STARTED"
                    ),
                    "exposure_status": record["exposure_status"],
                    "exposure_observed_s": (
                        record["restricted_visible_time_s"]
                        if ever_visible
                        else None
                    ),
                    "unvetted_visibility_status": record[
                        "unvetted_visibility_status"
                    ],
                    "unvetted_visibility_observed_s": record[
                        "unvetted_visibility_observed_s"
                    ],
                    "unavailable_status": record["unavailable_status"],
                    "unavailable_observed_s": (
                        record["restricted_unavailable_time_s"]
                        if record["unavailable_status"] != "NOT_STARTED"
                        else None
                    ),
                    "quarantine_status": record["quarantine_status"],
                    "readmission_count": record["readmission_count"],
                    "observed_at_horizon_s": record[
                        "observed_at_horizon_s"
                    ],
                }
            )
    return result


def _retrieval_records(
    *,
    raw_events: Sequence[Mapping[str, Any]],
    id_to_key: Mapping[int, str],
    frozen: FrozenInputs,
) -> list[dict[str, Any]]:
    def map_top5(
        value: Any, *, field: str, allow_none: bool = False
    ) -> list[str] | None:
        if value is None and allow_none:
            return None
        if not isinstance(value, list) or len(value) != 5:
            raise RunnerError(f"retrieval {field} is not an exact top-5 list")
        if len(set(value)) != 5:
            raise RunnerError(f"retrieval {field} contains duplicate ids")
        keys: list[str] = []
        for item_id in value:
            if (
                not isinstance(item_id, int)
                or isinstance(item_id, bool)
                or item_id not in id_to_key
            ):
                raise RunnerError(
                    f"retrieval {field} contains an unmapped backend id"
                )
            key = id_to_key[item_id]
            if not _is_key(key):
                raise RunnerError(f"retrieval {field} mapped to an invalid key")
            keys.append(key)
        if len(set(keys)) != 5:
            raise RunnerError(f"retrieval {field} maps to duplicate keys")
        return keys

    records = []
    for event in raw_events:
        returned_keys = map_top5(
            event.get("returned_ids"), field="returned_ids"
        )
        assert returned_keys is not None
        eligible_keys = map_top5(
            event.get("eligible_topk_ids"),
            field="eligible_topk_ids",
            allow_none=True,
        )
        poisonfree_keys = map_top5(
            event.get("poisonfree_topk_ids"),
            field="poisonfree_topk_ids",
            allow_none=True,
        )
        raw_poison_ids = event.get("poison_ids_retrieved")
        if not isinstance(raw_poison_ids, list):
            raise RunnerError("retrieval poison_ids_retrieved is not a list")
        poison_keys = []
        for item_id in raw_poison_ids:
            if item_id not in id_to_key:
                raise RunnerError(
                    "retrieval poison_ids_retrieved contains an unmapped id"
                )
            poison_keys.append(id_to_key[item_id])
        expected_poison_keys = [
            key for key in returned_keys if frozen.poison_truth.get(key, False)
        ]
        if poison_keys != expected_poison_keys:
            raise RunnerError(
                "retrieval poison-id evidence differs from returned top-5"
            )
        landed_keys = [key for key in poison_keys if frozen.landed.get(key)]
        query_role = {
            "craft": "attack_associated",
            "target": "heldout_same_topic",
            "negative": "negative_other_topic",
        }[event["kind"]]
        if query_role == "negative_other_topic":
            if (
                eligible_keys is not None
                or poisonfree_keys is not None
                or event.get("recall") is not None
                or event.get("displacement") is not None
            ):
                raise RunnerError(
                    "negative retrieval unexpectedly has reference top-5 evidence"
                )
        elif eligible_keys is None or poisonfree_keys is None:
            raise RunnerError(
                "non-negative retrieval omits reference top-5 evidence"
            )
        records.append(
            {
                "query_started_s": float(event["query_started_s"]),
                "t_s": float(event["t"]),
                "query_role": query_role,
                "query_ordinal": int(event["qidx"]),
                "returned_top5_item_keys": returned_keys,
                "eligible_top5_item_keys": eligible_keys,
                "poisonfree_top5_item_keys": poisonfree_keys,
                "poison_item_keys": poison_keys,
                "landed_poison_item_keys": landed_keys,
                "hit_overall": bool(poison_keys),
                "hit_landed_only": bool(landed_keys),
                "eligible_recall_at_5": event["recall"],
                "poisonfree_displacement_at_5": event["displacement"],
            }
        )
    return records


def _retrieval_summary(
    records: Sequence[Mapping[str, Any]], hit_field: str
) -> dict[str, Any]:
    result = {}
    for role in (
        "attack_associated",
        "heldout_same_topic",
        "negative_other_topic",
    ):
        rows = [record for record in records if record["query_role"] == role]
        hits = sum(bool(record[hit_field]) for record in rows)
        result[role] = {
            "poisoned_retrieval_count": hits,
            "query_count": len(rows),
            "poisoned_retrieval_rate": rate_record(hits, len(rows)),
            "cumulative_displaced_positions": sum(
                float(record["poisonfree_displacement_at_5"]) * 5.0
                for record in rows
                if record["poisonfree_displacement_at_5"] is not None
            ),
        }
    return result


def _cell_latency(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    committed = [
        int(round(float(record["integrated_latency_s"]) * 1_000_000_000))
        for record in records
        if record.get("status") == "COMMITTED"
        and record.get("integrated_latency_s") is not None
    ]
    return latency_summary(committed)


def _canonical_verifier_records(
    *,
    records: Sequence[Mapping[str, Any]],
    id_to_key: Mapping[int, str],
    role_by_key: Mapping[str, str],
    t0: float,
) -> list[dict[str, Any]]:
    result = []
    for record in records:
        key = record.get("item_key")
        if not _is_key(key) or key not in role_by_key:
            raise RunnerError("verifier record has unknown item key")
        if id_to_key.get(record.get("item_id")) != key:
            raise RunnerError("runtime id/key pairing changed inside a cell")
        result.append(
            {
                "item_key": key,
                "status": record.get("status"),
                "passes": record.get("passes"),
                "service_time_s": record.get("service_time_s"),
                "queue_enter_s": (
                    None
                    if record.get("queue_enter_s") is None
                    else float(record["queue_enter_s"]) - t0
                ),
                "queue_start_s": (
                    None
                    if record.get("queue_start_s") is None
                    else float(record["queue_start_s"]) - t0
                ),
                "queue_wait_s": record.get("queue_wait_s"),
                "queue_depth_at_enqueue": record.get(
                    "queue_depth_at_enqueue"
                ),
                "queue_depth_at_start": record.get(
                    "queue_depth_at_start"
                ),
                "integrated_latency_s": record.get(
                    "integrated_latency_s"
                ),
                "decision_commit_s": (
                    None
                    if record.get("decision_commit_s") is None
                    else float(record["decision_commit_s"]) - t0
                ),
            }
        )
    return result


def _population_summary(
    cells: Sequence[Mapping[str, Any]], *, landed_only: bool
) -> dict[str, Any]:
    poison = [
        item
        for cell in cells
        for item in cell["lifecycle"]["poison"]
        if not landed_only or item["landed_top5"]
    ]
    false_promotions = sum(item["false_promotion"] for item in poison)
    started = [item for item in poison if item["ever_visible"]]
    right_censored = sum(item["right_censored"] for item in started)
    exposure_total = sum(
        float(item["restricted_visible_time_s"]) for item in started
    )
    state_counts = Counter(item["state_at_horizon"] for item in poison)
    retrieval = [
        event for cell in cells for event in cell["retrieval"]["events"]
    ]
    hit_field = "hit_landed_only" if landed_only else "hit_overall"
    return {
        "poison_item_episode_n": len(poison),
        "false_promotion_incidence": rate_record(
            false_promotions, len(poison)
        ),
        "started_n": len(started),
        "right_censored": rate_record(right_censored, len(started)),
        "restricted_exposure_total_s": exposure_total,
        "restricted_exposure_per_started_s": (
            exposure_total / len(started) if started else None
        ),
        "state_at_horizon": dict(sorted(state_counts.items())),
        "retrieval": _retrieval_summary(retrieval, hit_field),
    }


def _clean_summary(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    clean = [
        item for cell in cells for item in cell["lifecycle"]["clean"]
    ]
    false_positive = sum(item["false_positive"] for item in clean)
    unavailable = sum(
        float(item["restricted_unavailable_time_s"]) for item in clean
    )
    return {
        "clean_item_episode_n": len(clean),
        "false_positive_misquarantine": rate_record(
            false_positive, len(clean)
        ),
        "never_visible_n": sum(item["never_visible"] for item in clean),
        "first_visible_n": sum(not item["never_visible"] for item in clean),
        "durable_visible_n": sum(item["durable_visible"] for item in clean),
        "restricted_unavailable_total_s": unavailable,
        "restricted_unavailable_per_item_s": (
            unavailable / len(clean) if clean else None
        ),
        "right_censored_quarantine_n": sum(
            item["quarantine_right_censored"] for item in clean
        ),
    }


def _aggregate_cells(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for cell in cells:
        key = (
            cell["attack_family"],
            cell["arm"],
            cell["baseline"],
            cell["backlog"],
        )
        grouped[key].append(cell)
    result = {}
    for (attack, arm, baseline, backlog), rows in sorted(grouped.items()):
        verifier = [
            record for row in rows for record in row["verifier_records"]
        ]
        result[f"{attack}/{arm}/{baseline}/{backlog}"] = {
            "cell_n": len(rows),
            "overall": _population_summary(rows, landed_only=False),
            "landed_only": _population_summary(rows, landed_only=True),
            "clean": _clean_summary(rows),
            "integrated_queue_to_commit_latency": _cell_latency(verifier),
        }
    return result


def _capture_artifacts(
    paths: Mapping[str, str],
    registry: SnapshotRegistry,
) -> dict[str, FileSnapshot]:
    result: dict[str, FileSnapshot] = {}
    resolved: dict[Path, str] = {}
    for path_name, artifact_name in ARTIFACT_NAMES.items():
        try:
            snapshot = registry.capture(paths[path_name])
        except SnapshotError as exc:
            raise RunnerError(
                f"required artifact cannot be snapshotted: {paths[path_name]}"
            ) from exc
        if snapshot.path in resolved:
            raise RunnerError(
                f"artifact roles {resolved[snapshot.path]} and {artifact_name} "
                "resolve to one path"
            )
        resolved[snapshot.path] = artifact_name
        result[path_name] = snapshot
    return result


def _artifact_hashes(
    snapshots: Mapping[str, FileSnapshot],
) -> dict[str, str]:
    return {
        artifact_name: snapshots[path_name].sha256
        for path_name, artifact_name in ARTIFACT_NAMES.items()
    }


def _artifact_documents(
    snapshots: Mapping[str, FileSnapshot],
) -> dict[str, Mapping[str, Any]]:
    documents: dict[str, Mapping[str, Any]] = {}
    for path_name in ARTIFACT_NAMES:
        if path_name in {"embeddings", "labels_checksum"}:
            continue
        snapshot = snapshots[path_name]
        try:
            document = strict_json_load_bytes(
                snapshot.payload, source=str(snapshot.path)
            )
        except (TypeError, ValueError) as exc:
            raise RunnerError(
                f"formal artifact is not strict JSON: {snapshot.path}"
            ) from exc
        if not isinstance(document, Mapping):
            raise RunnerError(
                f"formal artifact is not a JSON object: {snapshot.path}"
            )
        documents[path_name] = document
    return documents


def _require_digest(
    document: Mapping[str, Any],
    field: str,
    expected: str,
    description: str,
) -> None:
    if document.get(field) != expected:
        raise RunnerError(f"{description} does not match the frozen chain")


def _validate_score_lineage(
    document: Mapping[str, Any],
    *,
    phase: str,
    hashes: Mapping[str, str],
    runtime_hashes_by_filename: Mapping[str, str],
    require_threshold: bool,
) -> None:
    if (
        document.get("schema_version") != "W2D-score-v1"
        or document.get("phase") != phase
        or document.get("measurement_name") != MEASUREMENT_NAME
    ):
        raise RunnerError(f"{phase} score artifact has the wrong identity")
    expected = {
        "safe_inputs_sha256": hashes["safe_inputs_json"],
        "embeddings_sha256": hashes["safe_inputs_npz"],
        "detector_py_sha256": runtime_hashes_by_filename["detector.py"],
        "scorer_py_sha256": runtime_hashes_by_filename["w2d_scorer.py"],
    }
    if require_threshold:
        expected["threshold_artifact_sha256"] = hashes["threshold"]
    elif "threshold_artifact_sha256" in document:
        raise RunnerError("calibration score artifact unexpectedly used a threshold")
    for field, value in expected.items():
        _require_digest(document, field, value, f"{phase} score {field}")
    if document.get("keys_source") != "safe input frozen sets":
        raise RunnerError(f"{phase} scores did not use the frozen key set")


def _plan_provenance(
    plan: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    raw = plan.get("provenance", {}).get("inputs")
    if not isinstance(raw, list):
        raise RunnerError("protocol plan has no provenance input list")
    result: dict[str, Mapping[str, Any]] = {}
    for record in raw:
        if not isinstance(record, Mapping):
            raise RunnerError("protocol provenance record is not an object")
        role = record.get("role")
        if not isinstance(role, str) or role in result:
            raise RunnerError("protocol provenance role is missing or duplicate")
        result[role] = record
    return result


def _validate_artifact_chain(
    paths: Mapping[str, str],
    hashes: Mapping[str, str],
    *,
    documents: Mapping[str, Mapping[str, Any]],
    artifact_snapshots: Mapping[str, FileSnapshot],
    runtime_snapshots: Mapping[str, FileSnapshot],
    registry: SnapshotRegistry,
) -> Mapping[str, Any]:
    """Recompute every derived artifact before the first protocol cell."""

    runtime_hashes_by_filename = _runtime_hash_by_filename(runtime_snapshots)
    try:
        validate_a9_replay_equivalence(
            archived_plan=documents["plan_pre_a9"],
            current_plan=documents["plan"],
            archived_landing=documents["landing_pre_a9"],
            current_landing=documents["landing"],
            artifact_hashes=hashes,
        )
    except (A9EquivalenceError, KeyError, TypeError) as exc:
        raise RunnerError(f"A9 replay equivalence failed: {exc}") from exc

    legacy = documents["legacy_regression"]
    expected_legacy_sources = {
        "authority_manifest": "legacy_authority_manifest",
        "authoritative_w2": "legacy_authoritative_w2",
        "authoritative_w2r": "legacy_authoritative_w2r",
        "candidate_w2": "legacy_candidate_w2",
        "candidate_w2r": "legacy_candidate_w2r",
        "same_code_repeat": "legacy_same_code_repeat",
    }
    legacy_paths_by_artifact = {
        artifact_name: paths[path_name]
        for path_name, artifact_name in ARTIFACT_NAMES.items()
        if artifact_name in expected_legacy_sources.values()
    }
    legacy_source_paths_by_role = {
        role: legacy_paths_by_artifact[artifact_name]
        for role, artifact_name in expected_legacy_sources.items()
    }
    source_hashes_before = {
        role: artifact_snapshots[path_name].sha256
        for role, artifact_name in expected_legacy_sources.items()
        for path_name, candidate_name in ARTIFACT_NAMES.items()
        if candidate_name == artifact_name
    }
    runtime_sha256_before = _legacy_runtime_sha256(runtime_snapshots)
    runtime_files_before = _legacy_runtime_hashes(runtime_snapshots)
    try:
        validate_legacy_runtime_code_compatibility(
            runtime_sha256_before,
            runtime_files_before,
        )
    except ValueError as exc:
        raise RunnerError(
            f"legacy runtime code compatibility failed: {exc}"
        ) from exc
    gate_code_before = _legacy_gate_hashes(runtime_snapshots)
    sources = legacy.get("source_artifacts")
    executions = legacy.get("existing_gate_executions")
    expected_executions = tuple(
        (
            script,
            expected_legacy_sources[source_role]
            if source_role is not None
            else None,
        )
        for script, source_role in LEGACY_GATE_COMMANDS
    )
    if (
        legacy.get("schema_version") != "W2D-legacy-regression-v2"
        or legacy.get("amendment")
        != "W2D-PREREGISTRATION-AMENDMENT-A5.md"
        or not isinstance(sources, Mapping)
        or set(sources) != set(expected_legacy_sources)
        or not isinstance(executions, list)
        or len(executions) != len(expected_executions)
        or any(
            not isinstance(record, Mapping)
            or record.get("passed") is not True
            or record.get("returncode") != 0
            for record in executions
        )
    ):
        raise RunnerError("legacy replacement-gate artifact is malformed")
    for record, (script, artifact_name) in zip(
        executions, expected_executions
    ):
        command = record.get("command")
        expected_length = 3 if artifact_name is not None else 2
        if (
            not isinstance(command, list)
            or len(command) != expected_length
            or command[0] != str(Path(sys.executable).resolve())
            or command[1] != script
            or (
                artifact_name is not None
                and Path(str(command[2])).resolve()
                != Path(
                    legacy_paths_by_artifact[artifact_name]
                ).resolve()
            )
        ):
            raise RunnerError("legacy existing-gate command set changed")
    for role, artifact_name in expected_legacy_sources.items():
        record = sources[role]
        if (
            not isinstance(record, Mapping)
            or record.get("sha256") != hashes[artifact_name]
            or record.get("filename")
            != Path(legacy_paths_by_artifact[artifact_name]).name
        ):
            raise RunnerError(
                f"legacy replacement gate does not bind {artifact_name}"
            )
    if {
        role: hashes[artifact_name]
        for role, artifact_name in expected_legacy_sources.items()
    } != FROZEN_SOURCE_SHA256:
        raise RunnerError("legacy source hashes differ from Amendment A5")
    if source_hashes_before != FROZEN_SOURCE_SHA256:
        raise RunnerError("legacy source bytes moved before runner verification")
    recomputed_legacy = evaluate_legacy_regression(
        authority_manifest=documents["legacy_authority_manifest"],
        authoritative_w2=documents["legacy_authoritative_w2"],
        authoritative_w2r=documents["legacy_authoritative_w2r"],
        candidate_w2=documents["legacy_candidate_w2"],
        candidate_w2r=documents["legacy_candidate_w2r"],
        same_code_repeat=documents["legacy_same_code_repeat"],
        source_sha256={
            role: hashes[artifact_name]
            for role, artifact_name in expected_legacy_sources.items()
        },
        expected_runtime_code_sha256=LEGACY_RUNTIME_ARCHIVED_SHA256,
    )
    if legacy.get("replacement_gate") != recomputed_legacy:
        raise RunnerError(
            "legacy replacement gate is not the independent recomputation"
        )
    try:
        validate_legacy_gate_code_compatibility(
            legacy.get("gate_code_sha256"),
            gate_code_before,
        )
    except ValueError as exc:
        raise RunnerError(
            f"legacy gate code compatibility failed: {exc}"
        ) from exc
    if legacy.get("runtime_code_sha256") != LEGACY_RUNTIME_ARCHIVED_SHA256:
        raise RunnerError(
            "legacy runtime fingerprint differs from its archived candidate "
            "code"
        )
    if legacy.get("git_commit") != FROZEN_CANDIDATE_COMMIT:
        raise RunnerError("legacy regression is not on the frozen commit")
    if (
        canonical_strict_json_sha256(
            recomputed_legacy["same_code_scheduling_differences"]
        )
        != FROZEN_SAME_CODE_DIFFERENCE_SHA256
    ):
        raise RunnerError("legacy same-code difference projection moved")
    validate_frozen_same_code_differences(
        recomputed_legacy["same_code_scheduling_differences"]
    )
    rerun_legacy_gates(legacy_source_paths_by_role)
    try:
        registry.assert_unchanged()
    except SnapshotError as exc:
        raise RunnerError(
            "legacy source/runtime/gate code changed during independent rerun"
        ) from exc

    freeze = documents["freeze"]
    try:
        _verify_a6_from_snapshots(
            registry=registry,
            artifacts={
                "data_freeze": freeze,
                "safe_inputs_json": documents["inputs"],
                "labels": documents["labels"],
            },
            hashes={
                "safe_inputs_json": hashes["safe_inputs_json"],
                "safe_inputs_npz": hashes["safe_inputs_npz"],
                "labels": hashes["labels"],
                "labels_checksum": hashes["labels_checksum"],
            },
            artifact_snapshots={
                "safe_inputs_npz": artifact_snapshots["embeddings"],
            },
            component_snapshots=runtime_snapshots,
            paths={
                "safe_inputs_json": Path(paths["inputs"]),
                "safe_inputs_npz": Path(paths["embeddings"]),
                "labels": Path(paths["labels"]),
                "labels_checksum": Path(paths["labels_checksum"]),
            },
        )
    except ValueError as exc:
        raise RunnerError(f"A6 data/label lineage gate failed: {exc}") from exc
    frozen_artifacts = freeze.get("artifacts")
    if not isinstance(frozen_artifacts, Mapping):
        raise RunnerError("data freeze has no artifact ledger")
    for path_name, artifact_name in (
        ("inputs", "safe_inputs_json"),
        ("embeddings", "safe_inputs_npz"),
        ("labels", "labels"),
        ("labels_checksum", "labels_checksum"),
    ):
        filename = Path(paths[path_name]).name
        record = frozen_artifacts.get(filename)
        if (
            not isinstance(record, Mapping)
            or record.get("sha256") != hashes[artifact_name]
        ):
            raise RunnerError(f"data freeze does not bind {filename}")
    try:
        checksum_line = artifact_snapshots[
            "labels_checksum"
        ].payload.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise RunnerError("label checksum sidecar is not ASCII") from exc
    expected_line = (
        f"{hashes['labels']}  {Path(paths['labels']).name}"
    )
    if checksum_line != expected_line:
        raise RunnerError("label checksum sidecar does not bind evaluator labels")

    safe = documents["inputs"]
    embedding_record = safe.get("embedding_artifact")
    if (
        not isinstance(embedding_record, Mapping)
        or embedding_record.get("filename")
        != Path(paths["embeddings"]).name
        or embedding_record.get("sha256") != hashes["safe_inputs_npz"]
    ):
        raise RunnerError("safe JSON does not bind the embedding NPZ")

    plan = documents["plan"]
    provenance = _plan_provenance(plan)
    expected_plan_inputs = {
        "data_freeze": (
            Path(paths["freeze"]).name,
            hashes["data_freeze"],
        ),
        "safe_item_schema": (
            Path(paths["inputs"]).name,
            hashes["safe_inputs_json"],
        ),
        "safe_item_embeddings": (
            Path(paths["embeddings"]).name,
            hashes["safe_inputs_npz"],
        ),
        "evaluator_only_labels_and_queries": (
            Path(paths["labels"]).name,
            hashes["labels"],
        ),
        "protocol_plan_code": (
            "w2d_protocol.py",
            runtime_hashes_by_filename["w2d_protocol.py"],
        ),
        "preregistration_amendment": (
            "W2D-PREREGISTRATION-AMENDMENT-A4.md",
            runtime_hashes_by_filename[
                "W2D-PREREGISTRATION-AMENDMENT-A4.md"
            ],
        ),
    }
    if set(provenance) != set(expected_plan_inputs):
        raise RunnerError("protocol plan provenance roles changed")
    for role, (filename, digest) in expected_plan_inputs.items():
        record = provenance[role]
        if record.get("filename") != filename or record.get("sha256") != digest:
            raise RunnerError(f"protocol plan provenance moved for {role}")

    calibration = documents["calibration_scores"]
    _validate_score_lineage(
        calibration,
        phase="calibration",
        hashes=hashes,
        runtime_hashes_by_filename=runtime_hashes_by_filename,
        require_threshold=False,
    )
    labels = documents["labels"]
    recomputed_threshold = select_threshold(calibration, labels)
    threshold = documents["threshold"]
    threshold_core = dict(threshold)
    _require_digest(
        threshold_core,
        "calibration_scores_sha256",
        hashes["calibration_scores"],
        "threshold calibration score digest",
    )
    _require_digest(
        threshold_core,
        "labels_sha256",
        hashes["labels"],
        "threshold label digest",
    )
    threshold_core.pop("calibration_scores_sha256", None)
    threshold_core.pop("labels_sha256", None)
    if threshold_core != recomputed_threshold:
        raise RunnerError("threshold is not the independent calibration result")

    test_scores = documents["test_scores"]
    source_controls = documents["source_controls"]
    _validate_score_lineage(
        test_scores,
        phase="test",
        hashes=hashes,
        runtime_hashes_by_filename=runtime_hashes_by_filename,
        require_threshold=True,
    )
    _validate_score_lineage(
        source_controls,
        phase="source_control",
        hashes=hashes,
        runtime_hashes_by_filename=runtime_hashes_by_filename,
        require_threshold=True,
    )
    recomputed_metrics = detector_quality_summary(test_scores, labels)
    metric_artifact = documents["detector_metrics"]
    if (
        metric_artifact.get("schema_version")
        != "W2D-detector-metrics-v1"
        or metric_artifact.get("scores_sha256") != hashes["test_scores"]
        or metric_artifact.get("labels_sha256") != hashes["labels"]
        or metric_artifact.get("metrics") != recomputed_metrics
    ):
        raise RunnerError("detector metric artifact is not an exact recomputation")

    recomputed_control_gate = source_control_gate_summary(
        source_controls, labels
    )
    control_artifact = documents["source_control_gate"]
    if (
        control_artifact.get("schema_version")
        != "W2D-source-control-gate-v1"
        or control_artifact.get("scores_sha256")
        != hashes["source_controls"]
        or control_artifact.get("labels_sha256") != hashes["labels"]
        or control_artifact.get("gate") != recomputed_control_gate
        or recomputed_control_gate.get("passed") is not True
    ):
        raise RunnerError("source-control gate is absent, stale, or failed")

    landing = documents["landing"]
    if (
        landing.get("schema_version") != "W2D-landing-v1"
        or landing.get("protocol_plan_sha256") != hashes["protocol_plan"]
        or landing.get("safe_inputs_sha256") != hashes["safe_inputs_json"]
        or landing.get("embeddings_sha256") != hashes["safe_inputs_npz"]
        or not isinstance(landing.get("items"), list)
        or len(landing["items"]) != 192
    ):
        raise RunnerError("landing artifact is absent, stale, or incomplete")
    return recomputed_metrics


MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "measurement_name",
        "execution_tier",
        "expected_result_filename",
        "expected_runtime_backend",
        "artifact_hashes",
        "runtime_fingerprint",
    }
)


def _validate_manifest_identity_document(
    manifest: Mapping[str, Any],
    *,
    manifest_file: Path,
    execution_tier: str,
) -> None:
    tier_contract = EXECUTION_TIER_CONTRACTS[execution_tier]
    if (
        set(manifest) != MANIFEST_FIELDS
        or manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("measurement_name") != MEASUREMENT_NAME
        or manifest.get("execution_tier") != execution_tier
        or manifest.get("expected_result_filename")
        != tier_contract["result_filename"]
        or manifest.get("expected_runtime_backend")
        != tier_contract["runtime_backend"]
        or manifest_file.name != tier_contract["manifest_filename"]
    ):
        raise RunnerError("formal manifest identity changed")


def build_manifest(
    *,
    paths: Mapping[str, str],
    manifest_path: str | os.PathLike[str],
    backend_name: str = "inmemory",
) -> dict[str, Any]:
    """Build an exclusive, pre-runtime manifest for later independent gates."""

    dirty_files = _uncached_git_dirty_files()
    if dirty_files:
        raise RunnerError(
            "formal W2D manifest requires a committed code tree; dirty files: "
            f"{dirty_files}"
        )
    manifest_file = Path(manifest_path).resolve()
    execution_tier = RUNNER_BACKEND_TO_TIER.get(backend_name)
    if execution_tier is None:
        raise RunnerError(f"unsupported formal backend {backend_name!r}")
    tier_contract = EXECUTION_TIER_CONTRACTS[execution_tier]
    if manifest_file.name != tier_contract["manifest_filename"]:
        raise RunnerError(
            "formal manifest filename differs from the selected execution tier"
        )
    if Path(paths["output"]).name != tier_contract["result_filename"]:
        raise RunnerError(
            "formal result filename differs from the selected execution tier"
        )
    registry = SnapshotRegistry()
    artifact_snapshots = _capture_artifacts(paths, registry)
    hashes = _artifact_hashes(artifact_snapshots)
    documents = _artifact_documents(artifact_snapshots)
    base = manifest_file.parent
    fingerprint, runtime_snapshots = _capture_runtime_fingerprint(
        registry, base
    )
    _validate_artifact_chain(
        paths,
        hashes,
        documents=documents,
        artifact_snapshots=artifact_snapshots,
        runtime_snapshots=runtime_snapshots,
        registry=registry,
    )
    entries = {
        artifact_name: {
            "path": _relative_to(Path(paths[path_name]), base),
            "sha256": hashes[artifact_name],
        }
        for path_name, artifact_name in ARTIFACT_NAMES.items()
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "measurement_name": MEASUREMENT_NAME,
        "execution_tier": execution_tier,
        "expected_result_filename": tier_contract["result_filename"],
        "expected_runtime_backend": dict(
            tier_contract["runtime_backend"]
        ),
        "artifact_hashes": entries,
        "runtime_fingerprint": fingerprint,
    }
    ownership: OutputOwnership | None = None
    try:
        ownership = _exclusive_json_dump(manifest, manifest_file)
        _assert_snapshot_registry_unchanged(registry)
        assert_owned_output(ownership)
    except BaseException:
        if ownership is not None:
            preserve_failed_output(ownership)
        raise
    return manifest


def build_manifest_authority(
    *,
    manifest_paths: Mapping[str, str | os.PathLike[str]],
    authority_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Bind both tier-specific pre-run manifests into one authority file.

    The resulting authority must be committed before a canonical repository run.
    Temporary directories use the same canonical basenames without a Git
    requirement, which keeps unit fixtures portable.
    """

    dirty_files = _uncached_git_dirty_files()
    if dirty_files:
        raise RunnerError(
            "formal W2D authority requires a committed code tree; dirty files: "
            f"{dirty_files}"
        )
    if set(manifest_paths) != set(EXECUTION_TIER_CONTRACTS):
        raise RunnerError(
            "formal authority requires exactly one manifest for every execution tier"
        )
    resolved_manifests = {
        tier: Path(manifest_paths[tier]).resolve()
        for tier in EXECUTION_TIER_CONTRACTS
    }
    parent_dirs = {path.parent for path in resolved_manifests.values()}
    if len(parent_dirs) != 1:
        raise RunnerError("formal tier manifests must be sibling files")
    manifest_parent = next(iter(parent_dirs))
    expected_authority = manifest_parent / MANIFEST_AUTHORITY_BASENAME
    authority_file = (
        expected_authority
        if authority_path is None
        else Path(authority_path).resolve()
    )
    if authority_file != expected_authority:
        raise RunnerError(
            "formal manifest authority must use the canonical sibling path"
        )
    registry = SnapshotRegistry()
    manifest_snapshots: dict[str, FileSnapshot] = {}
    for execution_tier, manifest_file in resolved_manifests.items():
        try:
            manifest_snapshot = registry.capture(manifest_file)
            manifest = strict_json_load_bytes(
                manifest_snapshot.payload,
                source=str(manifest_snapshot.path),
            )
        except (SnapshotError, TypeError, ValueError) as exc:
            raise RunnerError(
                f"formal manifest {execution_tier} cannot seed an authority"
            ) from exc
        if not isinstance(manifest, Mapping):
            raise RunnerError(
                f"formal manifest {execution_tier} is not an object"
            )
        _validate_manifest_identity_document(
            manifest,
            manifest_file=manifest_snapshot.path,
            execution_tier=execution_tier,
        )
        manifest_snapshots[execution_tier] = manifest_snapshot
    authority = {
        "schema_version": MANIFEST_AUTHORITY_SCHEMA,
        "measurement_name": MEASUREMENT_NAME,
        "manifests": {
            tier: {
                "path": manifest_snapshots[tier].path.name,
                "sha256": manifest_snapshots[tier].sha256,
            }
            for tier in EXECUTION_TIER_CONTRACTS
        },
    }
    ownership: OutputOwnership | None = None
    try:
        ownership = _exclusive_json_dump(authority, authority_file)
        _assert_snapshot_registry_unchanged(registry)
        assert_owned_output(ownership)
    except BaseException:
        if ownership is not None:
            preserve_failed_output(ownership)
        raise
    return authority


@dataclass(frozen=True)
class _FormalBinding:
    registry: SnapshotRegistry
    manifest: Mapping[str, Any]
    manifest_sha256: str
    authority_sha256: str
    artifact_snapshots: Mapping[str, FileSnapshot]
    documents: Mapping[str, Mapping[str, Any]]
    hashes: Mapping[str, str]
    runtime_snapshots: Mapping[str, FileSnapshot]
    runtime_fingerprint_sha256: str
    detector_metrics: Mapping[str, Any]


def _validate_manifest_binding(
    *,
    paths: Mapping[str, str],
    manifest_path: str | os.PathLike[str],
    backend_name: str,
    registry: SnapshotRegistry | None = None,
) -> _FormalBinding:
    active_registry = registry if registry is not None else SnapshotRegistry()
    try:
        authority_binding = capture_manifest_authority_binding(
            manifest_path,
            registry=active_registry,
        )
    except (SnapshotError, W2DVerificationError) as exc:
        raise RunnerError("formal manifest authority binding failed") from exc
    manifest_file = authority_binding.manifest_path
    manifest = authority_binding.manifest
    execution_tier = RUNNER_BACKEND_TO_TIER.get(backend_name)
    if execution_tier is None:
        raise RunnerError(f"unsupported formal backend {backend_name!r}")
    tier_contract = EXECUTION_TIER_CONTRACTS[execution_tier]
    _validate_manifest_identity_document(
        manifest,
        manifest_file=manifest_file,
        execution_tier=execution_tier,
    )
    if Path(paths["output"]).name != tier_contract["result_filename"]:
        raise RunnerError("formal manifest identity changed")
    artifact_snapshots = _capture_artifacts(paths, active_registry)
    hashes = _artifact_hashes(artifact_snapshots)
    documents = _artifact_documents(artifact_snapshots)
    entries = manifest.get("artifact_hashes")
    if not isinstance(entries, Mapping) or set(entries) != set(hashes):
        raise RunnerError("formal manifest artifact set changed")
    for path_name, artifact_name in ARTIFACT_NAMES.items():
        record = entries.get(artifact_name)
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "sha256",
        }:
            raise RunnerError(f"manifest entry {artifact_name} is malformed")
        candidate = Path(str(record["path"]))
        if not candidate.is_absolute():
            candidate = manifest_file.parent / candidate
        if (
            candidate.resolve() != Path(paths[path_name]).resolve()
            or record["sha256"] != hashes[artifact_name]
        ):
            raise RunnerError(f"manifest entry {artifact_name} moved")
    expected_fingerprint, runtime_snapshots = _capture_runtime_fingerprint(
        active_registry, manifest_file.parent
    )
    if manifest.get("runtime_fingerprint") != expected_fingerprint:
        raise RunnerError("runtime fingerprint differs from the manifest")
    metrics = _validate_artifact_chain(
        paths,
        hashes,
        documents=documents,
        artifact_snapshots=artifact_snapshots,
        runtime_snapshots=runtime_snapshots,
        registry=active_registry,
    )
    return _FormalBinding(
        registry=active_registry,
        manifest=manifest,
        manifest_sha256=authority_binding.manifest_snapshot.sha256,
        authority_sha256=authority_binding.authority_snapshot.sha256,
        artifact_snapshots=artifact_snapshots,
        documents=documents,
        hashes=hashes,
        runtime_snapshots=runtime_snapshots,
        runtime_fingerprint_sha256=expected_fingerprint["sha256"],
        detector_metrics=metrics,
    )


def _assert_formal_snapshots_unchanged(registry: SnapshotRegistry) -> None:
    """Fail if any captured input/code byte or tracked-worktree state moved."""

    dirty_files = _uncached_git_dirty_files()
    if dirty_files:
        raise RunnerError(
            "formal W2D inputs/code changed during replay; dirty files: "
            f"{dirty_files}"
        )
    _assert_snapshot_registry_unchanged(registry)


def _uncached_git_dirty_files() -> list[str]:
    """Read live code and fingerprint-component state without the legacy cache."""

    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=HERE,
            stderr=subprocess.DEVNULL,
        ).decode()
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError) as exc:
        raise RunnerError(
            "cannot independently recheck formal W2D worktree state"
        ) from exc
    fingerprint_paths = {
        (HERE / filename).resolve()
        for filename in FINGERPRINT_FILE_COMPONENTS.values()
    }
    dirty: list[str] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        filename = line.split(maxsplit=1)[-1]
        destination = filename.rsplit(" -> ", maxsplit=1)[-1]
        resolved = (HERE.parent / destination).resolve()
        if analysis._is_code(filename) or resolved in fingerprint_paths:
            dirty.append(filename)
    return sorted(dirty)[:10]


def _assert_snapshot_registry_unchanged(
    registry: SnapshotRegistry,
) -> None:
    """Recheck captured identities/bytes without grading expected new outputs."""

    try:
        registry.assert_unchanged()
    except SnapshotError as exc:
        raise RunnerError(
            "formal W2D artifact/runtime snapshot changed during replay"
        ) from exc


def _config_from_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    constants = plan.get("constants")
    if not isinstance(constants, Mapping):
        raise RunnerError("protocol plan has no constants")
    expected = {
        "seeds": [1, 2, 3, 4, 5],
        "deadline_T_p_s": 1.0,
        "window_horizon_s": 8.0,
        "injection_at_s": 1.0,
        "retrieval_top_k": 5,
    }
    for field, value in expected.items():
        if constants.get(field) != value:
            raise RunnerError(
                f"protocol constant {field}={constants.get(field)!r}, "
                f"expected {value!r}"
            )
    cfg = json.loads(json.dumps(CFG_REALTEXT))
    cfg.update(
        corpus=768,
        n_poison=6,
        n_clean=6,
        n_craft_q=6,
        n_target_q=6,
        n_neg_q=6,
        qps=24,
        dur=8.0,
        inject_at=1.0,
        tp=1.0,
        tp_sweep=[1.0],
        seeds=[1, 2, 3, 4, 5],
        backlog={
            "normal": {"conc": 8, "items": 0},
            "heavy": {"conc": 1, "items": 12},
        },
    )
    return cfg


def _runtime_backend_descriptor(
    backend_name: str,
    *,
    backend: Any | None = None,
    uri: str | None = None,
) -> dict[str, Any]:
    if backend_name == "inmemory":
        return {
            "backend": "inmemory",
            "implementation": "InMemoryBackend",
            "evidence_level": "exact_in_memory",
            "deployment_mode": "process_local",
            "vector_dim": 384,
            "metric": "COSINE",
            "index_type_requested": "exact_bruteforce",
            "index_type_effective": "exact_bruteforce",
            "consistency_level": "strong_in_process",
            "pymilvus_version": None,
            "milvus_lite_version": None,
        }
    if backend_name == "milvus":
        if backend is None or uri is None:
            raise RunnerError(
                "Milvus runtime provenance requires the constructed backend"
            )
        info = backend.index_info()
        effective = (
            info.get("index_type_effective")
            if isinstance(info, Mapping)
            else None
        )
        if effective != "FLAT":
            raise RunnerError(
                "Milvus formal replay requires effective FLAT index, "
                f"observed {effective!r}"
            )
        deployment = backend.server_info(uri).get("deployment_mode")
        if deployment != "lite":
            raise RunnerError(
                "formal backend 'milvus' is reserved for Milvus Lite"
            )
        try:
            pymilvus_version = importlib_metadata.version("pymilvus")
            milvus_lite_version = importlib_metadata.version("milvus-lite")
        except importlib_metadata.PackageNotFoundError as exc:
            raise RunnerError(
                "Milvus package versions cannot be resolved"
            ) from exc
        return {
            "backend": "milvus_lite",
            "implementation": "MilvusBackend",
            "evidence_level": "milvus_lite_flat",
            "deployment_mode": "lite",
            "vector_dim": 384,
            "metric": "COSINE",
            "index_type_requested": "FLAT",
            "index_type_effective": "FLAT",
            "consistency_level": "Strong",
            "pymilvus_version": pymilvus_version,
            "milvus_lite_version": milvus_lite_version,
        }
    raise RunnerError(f"unsupported backend {backend_name!r}")


async def run_grid(
    *,
    paths: Mapping[str, str],
    manifest_path: str | os.PathLike[str] = DEFAULT_MANIFEST,
    backend_name: str = "inmemory",
    uri: str = "/tmp/w2d_lite.db",
    snapshot_registry: SnapshotRegistry | None = None,
) -> dict[str, Any]:
    execution_tier = RUNNER_BACKEND_TO_TIER.get(backend_name)
    if execution_tier is None:
        raise RunnerError(f"unsupported backend {backend_name!r}")
    canonical_manifest = (
        HERE
        / "results"
        / "w2d"
        / EXECUTION_TIER_CONTRACTS[execution_tier]["manifest_filename"]
    ).resolve()
    if Path(manifest_path).resolve() != canonical_manifest:
        raise RunnerError(
            "formal W2D execution requires the canonical committed "
            "tier manifest"
        )
    if analysis.git_dirty():
        raise RunnerError(
            "formal W2D run requires a committed code tree; dirty files: "
            f"{analysis.git_dirty_files()}"
        )
    registry = (
        snapshot_registry
        if snapshot_registry is not None
        else SnapshotRegistry()
    )
    binding = _validate_manifest_binding(
        paths=paths,
        manifest_path=manifest_path,
        backend_name=backend_name,
        registry=registry,
    )
    hashes = binding.hashes
    runtime_fingerprint_sha256 = binding.runtime_fingerprint_sha256
    detector_metrics = binding.detector_metrics
    runtime_hashes_by_filename = _runtime_hash_by_filename(
        binding.runtime_snapshots
    )
    frozen = FrozenInputs(
        documents=binding.documents,
        embeddings_snapshot=binding.artifact_snapshots["embeddings"],
        artifact_hashes=hashes,
        runtime_hashes_by_filename=runtime_hashes_by_filename,
    )
    units = _plan_units(frozen.plan)
    cells = _plan_cells(frozen.plan)
    cfg = _config_from_plan(frozen.plan)
    if backend_name == "inmemory":
        backend = InMemoryBackend()
    elif backend_name == "milvus":
        backend = MilvusBackend(
            dim=384, uri=uri, collection="w2d_poison"
        )
    else:
        raise RunnerError(f"unsupported backend {backend_name!r}")
    runtime_backend = _runtime_backend_descriptor(
        backend_name, backend=backend, uri=uri
    )
    try:
        backend_evidence_pre = capture_backend_evidence(
            backend_name,
            backend,
            uri,
            "pre",
        )
    except BackendEvidenceError as exc:
        raise RunnerError(
            "live backend preflight evidence collection failed"
        ) from exc

    score_sha = hashes["test_scores"]
    emitted_cells: list[dict[str, Any]] = []
    next_id: int | None = None
    active_seed: int | None = None
    background_keys = frozen.plan["retrieval_background"]["item_keys"]
    background_vectors = [frozen.vector(key) for key in background_keys]

    for plan_cell in cells:
        seed = int(plan_cell["seed"])
        if active_seed != seed:
            active_seed = seed
            reset_corpus(backend, background_vectors, cfg, id_base(seed, cfg))
            next_id = id_base(seed, cfg) + cfg["corpus"]
        assert next_id is not None

        unit_id = plan_cell.get("runtime_unit_id")
        if unit_id not in units:
            raise RunnerError(f"cell references unknown runtime unit {unit_id}")
        unit = units[unit_id]
        for field in ("seed", "attack_family", "backlog"):
            if plan_cell.get(field) != unit.get(field):
                raise RunnerError(f"cell/runtime unit disagree on {field}")
        world, assignments = _world_for_unit(frozen, unit)
        baseline = str(plan_cell["baseline"])
        arm = str(plan_cell["arm"])
        backlog = str(plan_cell["backlog"])
        if baseline == "B1":
            if arm != "control":
                raise RunnerError("B1 must have the single control arm")
            provider = None
        else:
            provider = ReplayBank(
                arm=arm,
                scores=frozen.scores,
                poison_truth=frozen.poison_truth,
                score_artifact_sha256=score_sha,
            )

        pre = {
            (kind, index): corpus_topk(
                world["corpus"], query, cfg["k"], id_base(seed, cfg)
            )
            for kind, key in (("craft", "q_craft"), ("target", "q_target"))
            for index, query in enumerate(world[key])
        }
        id_to_key: dict[int, str] = {
            id_base(seed, cfg) + index: key
            for index, key in enumerate(background_keys)
        }
        raw_query_events: list[dict[str, Any]] = []

        def admission_hook(
            *, role: str, ordinal: int, iid: int, **_unused: Any
        ) -> dict[str, Any]:
            key = assignments[role][ordinal]
            id_to_key[iid] = key
            return {
                "verifier_key": key,
                "verifier_input": {"execution_mode": EXECUTION_MODE},
            }

        lifecycle_holder: dict[str, Any] = {}

        def metrics_hook(*, sys: Any, t0: float, **_unused: Any) -> dict[str, Any]:
            lifecycle_holder["t0"] = t0
            lifecycle_holder["lifecycle"] = _lifecycle_records(
                sys=sys,
                assignments=assignments,
                id_to_key=id_to_key,
                frozen=frozen,
                arm=arm,
                t0=t0,
                horizon=cfg["dur"],
            )
            return {}

        raw = await run_cell(
            backend,
            baseline,
            backlog,
            cfg["tp"],
            seed,
            cfg,
            world,
            next_id,
            pre,
            verifier=provider,
            admission_hook=admission_hook,
            metrics_hook=metrics_hook,
            query_observer=lambda event: raw_query_events.append(dict(event)),
            fixed_query_count=191,
        )
        next_id = raw.pop("next_id")
        if raw.get("bg_errors"):
            raise RunnerError(
                f"{plan_cell['cell_id']} background errors: {raw['bg_errors']}"
            )
        if raw.get("residual_rows") != 0 or raw.get("live_tasks") != 0:
            raise RunnerError(
                f"{plan_cell['cell_id']} left runtime state behind"
            )
        role_by_key = _role_map(assignments)
        verifier_records = _canonical_verifier_records(
            records=raw.pop("verifier_records", []),
            id_to_key=id_to_key,
            role_by_key=role_by_key,
            t0=float(lifecycle_holder["t0"]),
        )
        retrieval_events = _retrieval_records(
            raw_events=raw_query_events,
            id_to_key=id_to_key,
            frozen=frozen,
        )
        expected_query_roles = [
            (
                "attack_associated"
                if position % 3 == 0
                else (
                    "heldout_same_topic"
                    if position % 3 == 1
                    else "negative_other_topic"
                )
            )
            for position in range(191)
        ]
        if len(retrieval_events) != 191 or any(
            event["query_role"] != expected_query_roles[position]
            or event["query_ordinal"] != (position // 3) % 6
            for position, event in enumerate(retrieval_events)
        ):
            raise RunnerError(
                "runtime retrieval schedule is not the frozen 191-event sequence"
            )
        lifecycle = lifecycle_holder["lifecycle"]
        protocol_items = _protocol_item_records(unit, frozen)
        attack_landed_n = sum(
            bool(frozen.landed[key]) for key in assignments["poison"]
        )
        cell = {
            "plan_cell_id": plan_cell["cell_id"],
            "seed": seed,
            "attack_family": plan_cell["attack_family"],
            "backlog": backlog,
            "baseline": baseline,
            "arm": arm,
            "protocol_items": protocol_items,
            "verifier_records": verifier_records,
            "lifecycle": _canonical_lifecycle_records(lifecycle),
            "retrieval": {
                "events": retrieval_events,
                "overall": _retrieval_summary(
                    retrieval_events, "hit_overall"
                ),
                "landed_only": _retrieval_summary(
                    retrieval_events, "hit_landed_only"
                ),
                "attack_landing_numerator": attack_landed_n,
                "attack_landing_denominator": len(assignments["poison"]),
            },
        }
        _finite_tree(cell)
        emitted_cells.append(cell)
        print(
            f"{len(emitted_cells):3d}/140 "
            f"s{seed} {plan_cell['attack_family']:<23} "
            f"{arm:<8} {baseline}/{backlog}",
            flush=True,
        )

    replay_observation_count = sum(
        len(cell["protocol_items"])
        for cell in emitted_cells
        if cell["arm"] == "detector"
    )
    if active_seed != 5:
        raise RunnerError("formal replay did not finish on frozen seed 5")
    expected_terminal_rows = [
        {
            "id": id_base(active_seed, cfg) + index,
            "visible": True,
            "vector": vector,
        }
        for index, vector in enumerate(background_vectors)
    ]
    try:
        expected_terminal_state = summarize_expected_seed_state(
            expected_terminal_rows
        )
        backend_evidence_post = capture_backend_evidence(
            backend_name,
            backend,
            uri,
            "post",
            expected_seed_state=expected_terminal_state,
        )
    except BackendEvidenceError as exc:
        raise RunnerError(
            "live backend postflight evidence collection failed"
        ) from exc
    result = {
        "schema_version": SCHEMA_VERSION,
        "measurement_name": MEASUREMENT_NAME,
        "execution_mode": EXECUTION_MODE,
        "runtime_fingerprint_sha256": runtime_fingerprint_sha256,
        "manifest_sha256": binding.manifest_sha256,
        "authority_sha256": binding.authority_sha256,
        "artifact_hashes": hashes,
        "runtime_backend": runtime_backend,
        "runtime_backend_evidence": {
            "pre": backend_evidence_pre,
            "post": backend_evidence_post,
        },
        "detector_execution": {
            "protocol_provider": FROZEN_PROVIDER,
            "detector_invocations_during_replay": 0,
        },
        "observation_horizon_s": cfg["dur"],
        "injection_at_s": cfg["inject_at"],
        "detector_denominator": {
            "score_once_item_count": detector_metrics[
                "unique_item_count"
            ],
            "unique_source_group_count": detector_metrics[
                "unique_source_group_count"
            ],
            "reported_detector_quality_denominator": detector_metrics[
                "unique_item_count"
            ],
            "protocol_replay_observation_count": replay_observation_count,
        },
        "cells": emitted_cells,
    }
    _finite_tree(result)
    _assert_formal_snapshots_unchanged(registry)
    return result


def _paths_from_args(args: argparse.Namespace) -> dict[str, str]:
    return {
        name: str(getattr(args, name))
        for name in DEFAULT_PATHS
    }


def _write_verified_result(
    *,
    result: Mapping[str, Any],
    output_path: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    registry: SnapshotRegistry,
) -> None:
    """Write once, verify by path, and preserve any failed public output."""

    ownership: OutputOwnership | None = None
    try:
        ownership = _exclusive_json_dump(result, output_path)
        _assert_snapshot_registry_unchanged(registry)
        assert_owned_output(ownership)
        verify_or_raise(manifest_path, output_path)
        _assert_snapshot_registry_unchanged(registry)
        assert_owned_output(ownership)
    except BaseException:
        if ownership is not None:
            preserve_failed_output(ownership)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in DEFAULT_PATHS.items():
        parser.add_argument(
            "--" + name.replace("_", "-"),
            dest=name,
            default=str(default),
        )
    parser.add_argument(
        "--backend", choices=("inmemory", "milvus"), default="inmemory"
    )
    parser.add_argument("--uri", default="/tmp/w2d_lite.db")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--build-manifest-only", action="store_true")
    actions.add_argument("--build-authority-only", action="store_true")
    args = parser.parse_args(argv)
    paths = _paths_from_args(args)
    if args.build_manifest_only:
        build_manifest(
            paths=paths,
            manifest_path=args.manifest,
            backend_name=args.backend,
        )
        print(f"wrote {args.manifest}")
        return 0
    if args.build_authority_only:
        manifest_parent = Path(args.manifest).resolve().parent
        authority_path = manifest_parent / MANIFEST_AUTHORITY_BASENAME
        authority = build_manifest_authority(
            manifest_paths={
                tier: manifest_parent / contract["manifest_filename"]
                for tier, contract in EXECUTION_TIER_CONTRACTS.items()
            },
            authority_path=authority_path,
        )
        print(
            f"wrote {authority_path} with "
            f"{len(authority['manifests'])} tier manifests"
        )
        return 0
    registry = SnapshotRegistry()
    result = asyncio.run(
        run_grid(
            paths=paths,
            manifest_path=args.manifest,
            backend_name=args.backend,
            uri=args.uri,
            snapshot_registry=registry,
        )
    )
    _write_verified_result(
        result=result,
        output_path=paths["output"],
        manifest_path=args.manifest,
        registry=registry,
    )
    print(f"wrote {paths['output']}")
    print("verify_w2d: ALL GATES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
