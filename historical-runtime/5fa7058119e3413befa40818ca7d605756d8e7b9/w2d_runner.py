#!/usr/bin/env python3
"""Run the frozen W2D E1 promotion-path replay grid.

D1 has already scored each unique test item exactly once.  This runner never
imports or calls D1.  It replays the frozen decision and measured service time
through B2/B3/B4, runs the paired construction-label oracle arm, and retains B1
as one off-path undefended control.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import analysis
from backend import InMemoryBackend
from functional_slice import State, VISIBLE, VerifierDecision
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
    file_sha256,
    latency_summary,
    rate_record,
    source_control_gate_summary,
    strict_json_load,
)
from w2d_calibrate import select_threshold
from verify_w2d import (
    EXECUTION_MODE,
    FROZEN_PROVIDER,
    MANIFEST_SCHEMA,
    MEASUREMENT_NAME,
    RESULT_SCHEMA,
    build_runtime_fingerprint,
    canonical_record_sha256,
    verify_or_raise,
)


HERE = Path(__file__).resolve().parent
SCHEMA_VERSION = RESULT_SCHEMA
OPAQUE_KEY_LENGTH = 64
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

DEFAULT_PATHS = {
    "freeze": HERE / "data" / "w2d" / "W2D-DATA-FREEZE.json",
    "inputs": HERE / "data" / "w2d" / "W2D-detector-inputs.json",
    "embeddings": HERE / "data" / "w2d" / "W2D-detector-inputs.npz",
    "labels": HERE / "results" / "w2d" / "W2D-labels.json",
    "labels_checksum": HERE / "results" / "w2d" / "W2D-labels.sha256",
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
    "landing": HERE / "results" / "w2d" / "W2D-LANDING.json",
    "output": HERE / "results" / "w2d" / "W2D-E1-INMEMORY.json",
}
DEFAULT_MANIFEST = HERE / "results" / "w2d" / "W2D-MANIFEST.json"

ARTIFACT_NAMES = {
    "freeze": "data_freeze",
    "inputs": "safe_inputs_json",
    "embeddings": "safe_inputs_npz",
    "labels": "labels",
    "labels_checksum": "labels_checksum",
    "plan": "protocol_plan",
    "calibration_scores": "calibration_scores",
    "threshold": "threshold",
    "test_scores": "test_scores",
    "source_controls": "source_controls",
    "source_control_gate": "source_control_gate",
    "detector_metrics": "detector_metrics",
    "landing": "landing",
}


class RunnerError(RuntimeError):
    pass


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


def _relative_to(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve())


def _runtime_components(base_dir: Path) -> dict[str, dict[str, str]]:
    """Named, independently hashable runtime inputs for the formal manifest."""

    files = {
        "runtime": "functional_slice.py",
        "backend": "backend.py",
        "experiment_driver": "poison_exposure.py",
        "detector": "detector.py",
        "workload": "w2d_workload.py",
        "protocol": "w2d_protocol.py",
        "scorer": "w2d_scorer.py",
        "calibrator": "w2d_calibrate.py",
        "metrics": "w2d_metrics.py",
        "landing_evaluator": "w2d_landing.py",
        "runner": "w2d_runner.py",
        "verifier": "verify_w2d.py",
        "preregistration": "W2D-PREREGISTRATION.md",
        "amendment_a1": "W2D-PREREGISTRATION-AMENDMENT-A1.md",
        "amendment_a2": "W2D-PREREGISTRATION-AMENDMENT-A2.md",
        "amendment_a3": "W2D-PREREGISTRATION-AMENDMENT-A3.md",
        "amendment_a4": "W2D-PREREGISTRATION-AMENDMENT-A4.md",
    }
    result: dict[str, dict[str, str]] = {}
    for component, filename in files.items():
        path = HERE / filename
        if not path.is_file():
            raise RunnerError(
                f"runtime fingerprint file is missing: {filename}"
            )
        result[component] = {"path": _relative_to(path, base_dir)}
    result["model_revision"] = {"value": MODEL_REVISION}
    return result


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
        "detector_py_sha256": file_sha256(str(HERE / "detector.py")),
        "scorer_py_sha256": file_sha256(str(HERE / "w2d_scorer.py")),
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
        safe_path: str,
        embeddings_path: str,
        labels_path: str,
        plan_path: str,
        scores_path: str,
        threshold_path: str,
        landing_path: str,
    ) -> None:
        self.safe_document = strict_json_load(safe_path)
        self.labels_document = strict_json_load(labels_path)
        self.plan = strict_json_load(plan_path)
        self.score_document = strict_json_load(scores_path)
        self.landing_document = strict_json_load(landing_path)
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
            safe_inputs_sha256=file_sha256(safe_path),
            embeddings_sha256=file_sha256(embeddings_path),
            threshold_sha256=file_sha256(threshold_path),
        )
        if set(self.scores) != set(
            self.safe_document.get("sets", {}).get("test_score_order", [])
        ):
            raise RunnerError("test score keys differ from frozen test set")
        if set(self.scores) - set(self.labels):
            raise RunnerError("a test score has no evaluator label")

        with np.load(embeddings_path, allow_pickle=False) as archive:
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
            ]
            visible_s, unavailable_s = _visible_intervals(
                item=item,
                transitions=transitions,
                horizon_abs=horizon_abs,
            )
            score = frozen.scores[key]
            detector_passes = _decision(score)
            replay_passes = (
                not frozen.poison_truth[key]
                if arm == "oracle"
                else detector_passes
            )
            ts = item["ts"]
            ever_visible = "visible" in ts
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
            visible_at_horizon = item["state"] in VISIBLE
            nonvisible_at_horizon = not visible_at_horizon
            record = {
                "item_key": key,
                "role": role,
                "ordinal": ordinal,
                "truth_poison": frozen.poison_truth[key],
                "detector_decision": detector_passes,
                "replay_decision": replay_passes,
                "detector_service_ns": _service_ns(score),
                "state_at_horizon": item["state"].value,
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
                    if item["state"] == State.QUARANTINED
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
                        ever_visible and item["state"] not in terminal_nonvisible
                    ),
                    landed_top5=bool(frozen.landed[key]),
                )
            else:
                record.update(
                    false_positive=not replay_passes,
                    never_visible=not ever_visible,
                    durable_visible=item["state"] in VISIBLE,
                    quarantine_right_censored=item["state"] == State.QUARANTINED,
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
    records = []
    for event in raw_events:
        poison_keys = [
            id_to_key[item_id]
            for item_id in event["poison_ids_retrieved"]
            if item_id in id_to_key
        ]
        landed_keys = [key for key in poison_keys if frozen.landed.get(key)]
        records.append(
            {
                "t_s": float(event["t"]),
                "query_role": {
                    "craft": "attack_associated",
                    "target": "heldout_same_topic",
                    "negative": "negative_other_topic",
                }[event["kind"]],
                "query_ordinal": int(event["qidx"]),
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
                "integrated_latency_s": record.get(
                    "integrated_latency_s"
                ),
                "decision_commit_s": record.get("decision_commit_s"),
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


def _artifact_hashes(paths: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path_name, artifact_name in ARTIFACT_NAMES.items():
        path = paths[path_name]
        if not Path(path).is_file():
            raise RunnerError(f"required artifact is missing: {path}")
        result[artifact_name] = file_sha256(path)
    return result


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
        "detector_py_sha256": file_sha256(str(HERE / "detector.py")),
        "scorer_py_sha256": file_sha256(str(HERE / "w2d_scorer.py")),
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
    paths: Mapping[str, str], hashes: Mapping[str, str]
) -> Mapping[str, Any]:
    """Recompute every derived artifact before the first protocol cell."""

    documents = {
        name: strict_json_load(paths[name])
        for name in (
            "freeze",
            "inputs",
            "labels",
            "plan",
            "calibration_scores",
            "threshold",
            "test_scores",
            "source_controls",
            "source_control_gate",
            "detector_metrics",
            "landing",
        )
    }
    if not all(isinstance(value, Mapping) for value in documents.values()):
        raise RunnerError("one formal W2D artifact is not a JSON object")

    freeze = documents["freeze"]
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
    checksum_line = Path(paths["labels_checksum"]).read_text(
        encoding="ascii"
    ).strip()
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
            file_sha256(str(HERE / "w2d_protocol.py")),
        ),
        "preregistration_amendment": (
            "W2D-PREREGISTRATION-AMENDMENT-A4.md",
            file_sha256(
                str(HERE / "W2D-PREREGISTRATION-AMENDMENT-A4.md")
            ),
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
        require_threshold=True,
    )
    _validate_score_lineage(
        source_controls,
        phase="source_control",
        hashes=hashes,
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


def build_manifest(
    *, paths: Mapping[str, str], manifest_path: str | os.PathLike[str]
) -> dict[str, Any]:
    """Build an exclusive, pre-runtime manifest for later independent gates."""

    manifest_file = Path(manifest_path)
    hashes = _artifact_hashes(paths)
    _validate_artifact_chain(paths, hashes)
    base = manifest_file.parent
    entries = {
        artifact_name: {
            "path": _relative_to(Path(paths[path_name]), base),
            "sha256": hashes[artifact_name],
        }
        for path_name, artifact_name in ARTIFACT_NAMES.items()
    }
    fingerprint = build_runtime_fingerprint(
        _runtime_components(base), base
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "measurement_name": MEASUREMENT_NAME,
        "artifact_hashes": entries,
        "runtime_fingerprint": fingerprint,
    }
    _exclusive_json_dump(manifest, manifest_file)
    return manifest


def _validate_manifest_binding(
    *,
    paths: Mapping[str, str],
    manifest_path: str | os.PathLike[str],
) -> tuple[dict[str, str], str, Mapping[str, Any]]:
    manifest_file = Path(manifest_path)
    manifest = strict_json_load(str(manifest_file))
    if not isinstance(manifest, Mapping):
        raise RunnerError("formal manifest is not an object")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("measurement_name") != MEASUREMENT_NAME
    ):
        raise RunnerError("formal manifest identity changed")
    hashes = _artifact_hashes(paths)
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
    expected_fingerprint = build_runtime_fingerprint(
        _runtime_components(manifest_file.parent), manifest_file.parent
    )
    if manifest.get("runtime_fingerprint") != expected_fingerprint:
        raise RunnerError("runtime fingerprint differs from the manifest")
    metrics = _validate_artifact_chain(paths, hashes)
    return hashes, expected_fingerprint["sha256"], metrics


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


async def run_grid(
    *,
    paths: Mapping[str, str],
    manifest_path: str | os.PathLike[str] = DEFAULT_MANIFEST,
    backend_name: str = "inmemory",
    uri: str = "/tmp/w2d_lite.db",
) -> dict[str, Any]:
    if analysis.git_dirty():
        raise RunnerError(
            "formal W2D run requires a committed code tree; dirty files: "
            f"{analysis.git_dirty_files()}"
        )
    hashes, runtime_fingerprint_sha256, detector_metrics = (
        _validate_manifest_binding(
            paths=paths,
            manifest_path=manifest_path,
        )
    )
    frozen = FrozenInputs(
        safe_path=paths["inputs"],
        embeddings_path=paths["embeddings"],
        labels_path=paths["labels"],
        plan_path=paths["plan"],
        scores_path=paths["test_scores"],
        threshold_path=paths["threshold"],
        landing_path=paths["landing"],
    )
    units = _plan_units(frozen.plan)
    cells = _plan_cells(frozen.plan)
    cfg = _config_from_plan(frozen.plan)

    if backend_name == "inmemory":
        backend = InMemoryBackend()
        evidence_level = "exact_in_memory"
        index_type = "exact_bruteforce"
    elif backend_name == "milvus":
        from milvus_backend import MilvusBackend

        backend = MilvusBackend(
            dim=384, uri=uri, collection="w2d_poison"
        )
        evidence_level = "milvus_lite_flat"
        index_type = "FLAT"
    else:
        raise RunnerError(f"unsupported backend {backend_name!r}")

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
        id_to_key: dict[int, str] = {}
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
        )
        retrieval_events = _retrieval_records(
            raw_events=raw_query_events,
            id_to_key=id_to_key,
            frozen=frozen,
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
    result = {
        "schema_version": SCHEMA_VERSION,
        "measurement_name": MEASUREMENT_NAME,
        "execution_mode": EXECUTION_MODE,
        "runtime_fingerprint_sha256": runtime_fingerprint_sha256,
        "artifact_hashes": hashes,
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
    return result


def _paths_from_args(args: argparse.Namespace) -> dict[str, str]:
    return {
        name: str(getattr(args, name))
        for name in DEFAULT_PATHS
    }


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
    parser.add_argument("--build-manifest-only", action="store_true")
    args = parser.parse_args(argv)
    paths = _paths_from_args(args)
    if args.build_manifest_only:
        build_manifest(paths=paths, manifest_path=args.manifest)
        print(f"wrote {args.manifest}")
        return 0
    result = asyncio.run(
        run_grid(
            paths=paths,
            manifest_path=args.manifest,
            backend_name=args.backend,
            uri=args.uri,
        )
    )
    _exclusive_json_dump(result, paths["output"])
    verify_or_raise(args.manifest, paths["output"])
    print(f"wrote {paths['output']}")
    print("verify_w2d: ALL GATES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
