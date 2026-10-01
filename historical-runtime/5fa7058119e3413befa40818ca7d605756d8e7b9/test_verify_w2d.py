#!/usr/bin/env python3
"""Adversarial tests for the independent formal W2D bundle verifier."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from typing import Any, Callable

from w2d_calibrate import select_threshold
from w2d_metrics import (
    detector_quality_summary,
    file_sha256,
    rate_record,
    source_control_gate_summary,
)
from verify_w2d import (
    EXECUTION_MODE,
    FROZEN_PROVIDER,
    FROZEN_SERVICE_REPLAY,
    MANIFEST_SCHEMA,
    MEASUREMENT_NAME,
    RESULT_SCHEMA,
    W2DVerificationError,
    _verify_plan,
    build_runtime_fingerprint,
    canonical_record_sha256,
    verify_bundle,
)


def opaque(namespace: str, index: int) -> str:
    return hashlib.sha256(f"{namespace}|{index}".encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any, *, allow_nan: bool = False) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=allow_nan,
        )
        + "\n",
        encoding="utf-8",
    )


class FormalBundle:
    """Build a small-runtime but full-denominator formal artifact bundle."""

    def __init__(self, root: Path):
        self.root = root
        self.artifact_paths: dict[str, Path] = {}
        self.labels: dict[str, Any]
        self.test_scores: dict[str, Any]
        self.plan: dict[str, Any]
        self.result: dict[str, Any]
        self.manifest: dict[str, Any]
        self.recipe: list[str]
        self.natural: list[str]
        self.clean: list[str]
        self._build()

    def _artifact(self, name: str, filename: str, value: Any) -> Path:
        path = self.root / filename
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            write_json(path, value)
        self.artifact_paths[name] = path
        return path

    def _build_labels(self) -> None:
        calibration = [opaque("calibration", index) for index in range(448)]
        self.recipe = [opaque("test-recipe", index) for index in range(128)]
        self.natural = [opaque("test-natural", index) for index in range(64)]
        self.clean = [opaque("test-clean", index) for index in range(320)]
        controls = [opaque("source-control", index) for index in range(24)]

        items: list[dict[str, Any]] = []
        for index, key in enumerate(calibration):
            items.append(
                {
                    "item_key": key,
                    "split": "calibration",
                    "poison": index >= 224,
                    "source_group": f"calibration-group-{index}",
                    "attack_family": "recipe" if index >= 224 else None,
                    "stratum": (
                        "calibration_recipe"
                        if index >= 224
                        else "calibration_ordinary_clean"
                    ),
                }
            )
        for index, key in enumerate(self.recipe):
            items.append(
                {
                    "item_key": key,
                    "split": "test",
                    "poison": True,
                    "source_group": f"test-recipe-group-{index}",
                    "attack_family": "recipe",
                    "stratum": "test_recipe",
                }
            )
        for index, key in enumerate(self.natural):
            items.append(
                {
                    "item_key": key,
                    "split": "test",
                    "poison": True,
                    "source_group": f"test-natural-group-{index}",
                    "attack_family": "natural_cover_suffix_v1",
                    "stratum": "test_natural_cover",
                }
            )
        for index, key in enumerate(self.clean):
            items.append(
                {
                    "item_key": key,
                    "split": "test",
                    "poison": False,
                    "source_group": f"test-clean-group-{index}",
                    "attack_family": None,
                    "stratum": (
                        "ordinary_clean" if index < 256 else "hard_negative_clean"
                    ),
                }
            )
        kinds = (
            ["invalid_signature"] * 8
            + ["provenance_conflict"] * 8
            + ["unknown_source"] * 8
        )
        for index, (key, kind) in enumerate(zip(controls, kinds)):
            items.append(
                {
                    "item_key": key,
                    "split": "implementation_control",
                    "poison": False,
                    "source_group": f"source-control-group-{index}",
                    "attack_family": None,
                    "stratum": "source_control",
                    "source_control_kind": kind,
                }
            )
        self.calibration_keys = calibration
        self.control_keys = controls
        self.labels = {"schema_version": "W2D-labels-v1", "items": items}

    def _build_plan(self) -> None:
        background = [opaque("retrieval-background", index) for index in range(768)]
        units: list[dict[str, Any]] = []
        cells: list[dict[str, Any]] = []
        clean_counts = {
            1: (4, 2),
            2: (3, 3),
            3: (4, 2),
            4: (3, 3),
            5: (4, 2),
        }
        filler_counts = {
            1: (7, 5),
            2: (7, 5),
            3: (7, 5),
            4: (7, 5),
            5: (8, 4),
        }
        ordinary = iter(self.clean[:256])
        hard = iter(self.clean[256:])

        def selected_clean(counts: tuple[int, int]) -> list[str]:
            ordinary_n, hard_n = counts
            strata: list[str] = []
            while ordinary_n or hard_n:
                if ordinary_n:
                    strata.append("ordinary")
                    ordinary_n -= 1
                if hard_n:
                    strata.append("hard")
                    hard_n -= 1
            return [
                next(ordinary) if stratum == "ordinary" else next(hard)
                for stratum in strata
            ]

        clean_by_seed = {
            seed: selected_clean(clean_counts[seed]) for seed in range(1, 6)
        }
        filler_by_seed = {
            seed: selected_clean(filler_counts[seed]) for seed in range(1, 6)
        }

        for family, poison_keys in (
            ("recipe", self.recipe),
            ("natural_cover_suffix_v1", self.natural),
        ):
            for seed in range(1, 6):
                selected = poison_keys[(seed - 1) * 6 : seed * 6]
                clean_keys = clean_by_seed[seed]
                for backlog in ("normal", "heavy"):
                    uid = f"unit-{family}-{seed}-{backlog}"
                    filler_keys = (
                        filler_by_seed[seed] if backlog == "heavy" else []
                    )
                    ordered_roles = (
                        [("filler", key) for key in filler_keys]
                        + [("poison", key) for key in selected]
                        + [("clean", key) for key in clean_keys]
                    )
                    sequence = [
                        {
                            "injection_ordinal": ordinal,
                            "item_key": key,
                            "role": role,
                        }
                        for ordinal, (role, key) in enumerate(ordered_roles)
                    ]
                    units.append(
                        {
                            "runtime_unit_id": uid,
                            "seed": seed,
                            "attack_family": family,
                            "backlog": backlog,
                            "poison_item_keys": list(selected),
                            "clean_item_keys": list(clean_keys),
                            "filler_item_keys": list(filler_keys),
                            "query_poison_item_keys": list(selected),
                            "injection_sequence": sequence,
                        }
                    )
                    configurations = [
                        ("B1", "control"),
                        ("B2", "detector"),
                        ("B3", "detector"),
                        ("B4", "detector"),
                        ("B2", "oracle"),
                        ("B3", "oracle"),
                        ("B4", "oracle"),
                    ]
                    for position, (baseline, arm) in enumerate(configurations):
                        cells.append(
                            {
                                "cell_id": f"{uid}:{baseline}:{arm}",
                                "runtime_unit_id": uid,
                                "seed": seed,
                                "attack_family": family,
                                "backlog": backlog,
                                "baseline": baseline,
                                "arm": arm,
                                "execution_order_position": position,
                                "service_replay": (
                                    "none"
                                    if baseline == "B1"
                                    else FROZEN_SERVICE_REPLAY
                                ),
                            }
                        )
        landing_records = [
            {"item_key": key, "attack_family": "recipe"}
            for key in self.recipe
        ] + [
            {"item_key": key, "attack_family": "natural_cover_suffix_v1"}
            for key in self.natural
        ]
        self.background = background
        self.plan = {
            "schema_version": "W2D-protocol-plan-v1",
            "runtime_units": units,
            "cells": cells,
            "retrieval_background": {"item_keys": background},
            "landing_plan": {"records": landing_records},
        }

    def _build_components(self) -> dict[str, Any]:
        components: dict[str, Any] = {}
        for name in (
            "runtime",
            "detector",
            "workload",
            "protocol",
            "scorer",
            "calibrator",
            "metrics",
            "runner",
            "verifier",
        ):
            path = self.root / f"component-{name}.py"
            path.write_text(f"# frozen {name}\n", encoding="utf-8")
            components[name] = {"path": path.name}
        components["model_revision"] = {"value": "frozen-model-revision"}
        return build_runtime_fingerprint(components, self.root)

    def _score_provenance(self) -> dict[str, str]:
        components = self.fingerprint["components"]
        return {
            "safe_inputs_sha256": file_sha256(
                str(self.artifact_paths["safe_inputs_json"])
            ),
            "embeddings_sha256": file_sha256(
                str(self.artifact_paths["safe_inputs_npz"])
            ),
            "detector_py_sha256": components["detector"]["sha256"],
            "scorer_py_sha256": components["scorer"]["sha256"],
        }

    def _build_calibration(self) -> None:
        label_by_key = {
            item["item_key"]: item for item in self.labels["items"]
        }
        records = []
        for index, key in enumerate(self.calibration_keys):
            poison = label_by_key[key]["poison"]
            records.append(
                {
                    "item_key": key,
                    "rep": 0.1,
                    "knn": 0.2,
                    "c_score": 1.0 if poison else 0.0,
                    "family_s_affirms": False,
                    "detector_service_ns": 1000 + index,
                }
            )
        calibration = {
            "schema_version": "W2D-score-v1",
            "phase": "calibration",
            "measurement_name": MEASUREMENT_NAME,
            "z_statistics": {
                "rep": {"mean": 0.0, "std": 1.0},
                "knn": {"mean": 0.0, "std": 1.0},
            },
            **self._score_provenance(),
            "items": records,
        }
        self._artifact(
            "calibration_scores", "W2D-calibration-scores.json", calibration
        )
        threshold = select_threshold(calibration, self.labels)
        threshold["calibration_scores_sha256"] = file_sha256(
            str(self.artifact_paths["calibration_scores"])
        )
        threshold["labels_sha256"] = file_sha256(
            str(self.artifact_paths["labels"])
        )
        self._artifact("threshold", "W2D-threshold.json", threshold)

    def _build_final_scores(self) -> None:
        threshold_hash = file_sha256(str(self.artifact_paths["threshold"]))
        test_records: list[dict[str, Any]] = []
        ordered = self.recipe + self.natural + self.clean
        for index, key in enumerate(ordered):
            false_negative = key == self.recipe[0]
            false_positive = key == self.clean[0]
            poison = key in set(self.recipe) or key in set(self.natural)
            promote = false_negative or (not poison and not false_positive)
            test_records.append(
                {
                    "item_key": key,
                    "family_s_affirms": False,
                    "c_score": 0.0 if promote else 1.0,
                    "promote": promote,
                    "detector_service_ns": 1_000_000 + index,
                }
            )
        common = {
            "schema_version": "W2D-score-v1",
            "measurement_name": MEASUREMENT_NAME,
            **self._score_provenance(),
            "threshold_artifact_sha256": threshold_hash,
        }
        self.test_scores = {
            **common,
            "phase": "test",
            "items": test_records,
        }
        controls = {
            **common,
            "phase": "source_control",
            "items": [
                {
                    "item_key": key,
                    "family_s_affirms": False,
                    "c_score": 1.0,
                    "promote": False,
                    "detector_service_ns": 2_000_000 + index,
                }
                for index, key in enumerate(self.control_keys)
            ],
        }
        self._artifact("test_scores", "W2D-test-scores.json", self.test_scores)
        self._artifact(
            "source_controls", "W2D-source-controls.json", controls
        )
        metrics = {
            "schema_version": "W2D-detector-metrics-v1",
            "scores_sha256": file_sha256(
                str(self.artifact_paths["test_scores"])
            ),
            "labels_sha256": file_sha256(str(self.artifact_paths["labels"])),
            "metrics": detector_quality_summary(self.test_scores, self.labels),
        }
        gate = {
            "schema_version": "W2D-source-control-gate-v1",
            "scores_sha256": file_sha256(
                str(self.artifact_paths["source_controls"])
            ),
            "labels_sha256": file_sha256(str(self.artifact_paths["labels"])),
            "gate": source_control_gate_summary(controls, self.labels),
        }
        self._artifact("detector_metrics", "W2D-detector-metrics.json", metrics)
        self._artifact(
            "source_control_gate", "W2D-source-control-gate.json", gate
        )

    def _build_landing(self) -> None:
        records: list[dict[str, Any]] = []
        family_counts: dict[str, tuple[int, int]] = {}
        for index, (key, family) in enumerate(
            [(key, "recipe") for key in self.recipe]
            + [(key, "natural_cover_suffix_v1") for key in self.natural]
        ):
            rank = 1 if index % 2 == 0 else 6
            landed = rank <= 5
            top5 = (
                [key, *self.background[:4]]
                if landed
                else self.background[:5]
            )
            records.append(
                {
                    "item_key": key,
                    "attack_family": family,
                    "rank": rank,
                    "candidate_rank": rank,
                    "landed_top5": landed,
                    "landed": landed,
                    "top5_item_keys": top5,
                }
            )
            total, hits = family_counts.get(family, (0, 0))
            family_counts[family] = (total + 1, hits + int(landed))
        populations = {}
        for family, (total, hits) in family_counts.items():
            populations[family] = {
                **rate_record(hits, total),
                "landed_n": hits,
                "n": total,
            }
        landing = {
            "schema_version": "W2D-landing-v1",
            "measurement": "B1 off-path exact-cosine attack landing",
            "top_k": 5,
            "tie_break": "descending cosine, then ascending opaque item_key",
            "candidate_population": (
                "one poison item at a time plus the frozen 768-item "
                "retrieval background"
            ),
            "protocol_plan_sha256": file_sha256(
                str(self.artifact_paths["protocol_plan"])
            ),
            "safe_inputs_sha256": file_sha256(
                str(self.artifact_paths["safe_inputs_json"])
            ),
            "embeddings_sha256": file_sha256(
                str(self.artifact_paths["safe_inputs_npz"])
            ),
            "retrieval_background_sha256": canonical_sha256(self.background),
            "populations": populations,
            "items": records,
        }
        self.landing_by_key = {record["item_key"]: record for record in records}
        self._artifact("landing", "W2D-LANDING.json", landing)

    def _score_by_key(self) -> dict[str, dict[str, Any]]:
        return {record["item_key"]: record for record in self.test_scores["items"]}

    def _label_by_key(self) -> dict[str, dict[str, Any]]:
        return {record["item_key"]: record for record in self.labels["items"]}

    def _lifecycle_row(
        self,
        *,
        key: str,
        baseline: str,
        arm: str,
        detector_promote: bool,
    ) -> dict[str, Any]:
        truth = self._label_by_key()[key]["poison"]
        passes = detector_promote if arm == "detector" else not truth
        visibility = "NOT_STARTED" if baseline == "B2" and not passes else "STARTED"
        state = "TRUSTED" if passes else "QUARANTINED"
        exposure_status = (
            "RIGHT_CENSORED"
            if truth and passes
            else ("COMPLETED" if truth and visibility == "STARTED" else "NOT_STARTED")
        )
        exposure_age = (
            7.0
            if exposure_status == "RIGHT_CENSORED"
            else (0.5 if exposure_status == "COMPLETED" else None)
        )
        clean_false_positive = not truth and arm == "detector" and not detector_promote
        return {
            "item_key": key,
            "truth_poison": truth,
            "detector_promote": detector_promote,
            "retry_count": 0,
            "state_at_horizon": state,
            "visibility_status": visibility,
            "exposure_status": exposure_status,
            "exposure_observed_s": exposure_age,
            "unavailable_status": (
                "RIGHT_CENSORED" if clean_false_positive else "NOT_APPLICABLE"
            ),
            "unavailable_observed_s": 7.0 if clean_false_positive else None,
            "quarantine_status": (
                "RIGHT_CENSORED"
                if clean_false_positive
                else ("NOT_STARTED" if truth and passes else "NOT_APPLICABLE")
            ),
            "readmission_count": 0,
            "observed_at_horizon_s": 8.0,
        }

    def _build_result(self) -> None:
        scores = self._score_by_key()
        labels = self._label_by_key()
        unit_by_id = {
            unit["runtime_unit_id"]: unit for unit in self.plan["runtime_units"]
        }
        emitted: list[dict[str, Any]] = []
        replay_observations = 0
        for planned in self.plan["cells"]:
            unit = unit_by_id[planned["runtime_unit_id"]]
            protocol_items = []
            records = []
            lifecycle = []
            for event in unit["injection_sequence"]:
                key = event["item_key"]
                score = scores[key]
                protocol_items.append(
                    {
                        "item_key": key,
                        "role": event["role"],
                        "ordinal": event["injection_ordinal"],
                        "detector_promote": score["promote"],
                        "detector_service_ns": score["detector_service_ns"],
                        "score_record_sha256": canonical_record_sha256(score),
                    }
                )
                lifecycle.append(
                    self._lifecycle_row(
                        key=key,
                        baseline=planned["baseline"],
                        arm=planned["arm"],
                        detector_promote=score["promote"],
                    )
                )
                if planned["baseline"] != "B1":
                    passes = (
                        score["promote"]
                        if planned["arm"] == "detector"
                        else not labels[key]["poison"]
                    )
                    service = score["detector_service_ns"] / 1_000_000_000
                    baseline_delta = {"B2": 0.2, "B3": 0.4, "B4": 0.6}[
                        planned["baseline"]
                    ]
                    records.append(
                        {
                            "item_key": key,
                            "status": "COMMITTED",
                            "passes": passes,
                            "service_time_s": service,
                            "integrated_latency_s": service + baseline_delta,
                            "decision_commit_s": 1.0 + service + baseline_delta,
                        }
                    )
            if planned["arm"] == "detector":
                replay_observations += len(protocol_items)
            poison_keys = [
                item["item_key"]
                for item in protocol_items
                if item["role"] == "poison"
            ]
            attack_landed_n = sum(
                int(self.landing_by_key[key]["landed_top5"])
                for key in poison_keys
            )
            emitted.append(
                {
                    "plan_cell_id": planned["cell_id"],
                    "attack_family": planned["attack_family"],
                    "arm": planned["arm"],
                    "baseline": planned["baseline"],
                    "backlog": planned["backlog"],
                    "seed": planned["seed"],
                    "protocol_items": protocol_items,
                    "verifier_records": records,
                    "lifecycle": lifecycle,
                    "retrieval": {
                        "attack_landing_numerator": attack_landed_n,
                        "attack_landing_denominator": len(poison_keys),
                    },
                }
            )
        metrics = detector_quality_summary(self.test_scores, self.labels)
        artifact_hashes = {
            name: file_sha256(str(path))
            for name, path in self.artifact_paths.items()
        }
        self.result = {
            "schema_version": RESULT_SCHEMA,
            "measurement_name": MEASUREMENT_NAME,
            "execution_mode": EXECUTION_MODE,
            "runtime_fingerprint_sha256": self.fingerprint["sha256"],
            "artifact_hashes": artifact_hashes,
            "detector_execution": {
                "protocol_provider": FROZEN_PROVIDER,
                "detector_invocations_during_replay": 0,
            },
            "observation_horizon_s": 8.0,
            "injection_at_s": 1.0,
            "detector_denominator": {
                "score_once_item_count": metrics["unique_item_count"],
                "unique_source_group_count": metrics["unique_source_group_count"],
                "reported_detector_quality_denominator": metrics[
                    "unique_item_count"
                ],
                "protocol_replay_observation_count": replay_observations,
            },
            "cells": emitted,
        }

    def _build(self) -> None:
        self._build_labels()
        self._artifact("data_freeze", "W2D-DATA-FREEZE.json", {"frozen": True})
        self._artifact(
            "safe_inputs_json", "W2D-detector-inputs.json", {"frozen": True}
        )
        self._artifact(
            "safe_inputs_npz", "W2D-detector-inputs.npz", b"frozen-npz-placeholder"
        )
        self._artifact("labels", "W2D-evaluator-only.json", self.labels)
        self._build_plan()
        self._artifact("protocol_plan", "W2D-PROTOCOL-PLAN.json", self.plan)
        self.fingerprint = self._build_components()
        self._build_calibration()
        self._build_final_scores()
        self._build_landing()
        self._build_result()
        entries = {
            name: {"path": path.name, "sha256": file_sha256(str(path))}
            for name, path in self.artifact_paths.items()
        }
        self.manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "measurement_name": MEASUREMENT_NAME,
            "artifact_hashes": entries,
            "runtime_fingerprint": self.fingerprint,
        }
        self.manifest_path = self.root / "W2D-MANIFEST.json"
        self.results_path = self.root / "W2D-PROTOCOL-RESULTS.json"
        write_json(self.manifest_path, self.manifest)
        write_json(self.results_path, self.result)

    def rewrite_result(self, mutator: Callable[[dict[str, Any]], None]) -> None:
        self.result = copy.deepcopy(self.result)
        mutator(self.result)
        write_json(self.results_path, self.result)

    def rewrite_artifact(
        self, name: str, mutator: Callable[[dict[str, Any]], None]
    ) -> None:
        path = self.artifact_paths[name]
        document = json.loads(path.read_text(encoding="utf-8"))
        mutator(document)
        write_json(path, document)
        digest = file_sha256(str(path))
        self.manifest["artifact_hashes"][name]["sha256"] = digest
        self.result["artifact_hashes"][name] = digest
        write_json(self.manifest_path, self.manifest)
        write_json(self.results_path, self.result)


class VerifyW2DTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.bundle = FormalBundle(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_rejected(self, needle: str | None = None) -> None:
        report = verify_bundle(
            self.bundle.manifest_path, self.bundle.results_path
        )
        self.assertFalse(report.ok, report.checks)
        if needle is not None:
            self.assertIn(needle, " ".join(report.failures))

    def test_full_formal_bundle_passes_with_unpaired_integrated_latency(self) -> None:
        report = verify_bundle(
            self.bundle.manifest_path, self.bundle.results_path
        )
        self.assertTrue(report.ok, report.failures)
        detector_cells = [
            cell
            for cell in self.bundle.result["cells"]
            if cell["arm"] == "detector"
            and cell["attack_family"] == "recipe"
            and cell["backlog"] == "normal"
            and cell["seed"] == 1
        ]
        self.assertEqual(
            {
                cell["verifier_records"][0]["integrated_latency_s"]
                for cell in detector_cells
            },
            {
                detector_cells[0]["verifier_records"][0]["service_time_s"] + 0.2,
                detector_cells[0]["verifier_records"][0]["service_time_s"] + 0.4,
                detector_cells[0]["verifier_records"][0]["service_time_s"] + 0.6,
            },
        )

    def test_score_record_back_reference_is_exact(self) -> None:
        def mutate(result: dict[str, Any]) -> None:
            result["cells"][0]["protocol_items"][0][
                "score_record_sha256"
            ] = "0" * 64

        self.bundle.rewrite_result(mutate)
        self.assert_rejected("score-record digest")

    def test_detector_denominator_cannot_expand_with_replays(self) -> None:
        self.bundle.rewrite_result(
            lambda result: result["detector_denominator"].update(
                reported_detector_quality_denominator=513
            )
        )
        self.assert_rejected("detector denominator")

    def test_replay_decision_or_service_cannot_change_by_baseline(self) -> None:
        def mutate(result: dict[str, Any]) -> None:
            cell = next(
                row
                for row in result["cells"]
                if row["baseline"] == "B3" and row["arm"] == "detector"
            )
            cell["protocol_items"][0]["detector_service_ns"] += 1

        self.bundle.rewrite_result(mutate)
        self.assert_rejected("service time differs")

    def test_clean_false_positive_must_remain_quarantined_without_retry(self) -> None:
        clean_key = self.bundle.clean[0]

        def mutate(result: dict[str, Any]) -> None:
            for cell in result["cells"]:
                if cell["arm"] != "detector":
                    continue
                for row in cell["lifecycle"]:
                    if row["item_key"] == clean_key:
                        row["state_at_horizon"] = "TRUSTED"
                        return
            self.fail("false-positive fixture was absent")

        self.bundle.rewrite_result(mutate)
        self.assert_rejected("false positive")

    def test_poison_false_negative_must_remain_visible_and_exposed(self) -> None:
        poison_key = self.bundle.recipe[0]

        def mutate(result: dict[str, Any]) -> None:
            for cell in result["cells"]:
                if cell["arm"] != "detector":
                    continue
                for row in cell["lifecycle"]:
                    if row["item_key"] == poison_key:
                        row["state_at_horizon"] = "QUARANTINED"
                        return
            self.fail("false-negative fixture was absent")

        self.bundle.rewrite_result(mutate)
        self.assert_rejected("false negative")

    def test_failed_provider_state_blocks_publication(self) -> None:
        def mutate(result: dict[str, Any]) -> None:
            cell = next(row for row in result["cells"] if row["baseline"] == "B2")
            cell["verifier_records"][0]["status"] = "FAILED"

        self.bundle.rewrite_result(mutate)
        self.assert_rejected("not terminal/legal")

    def test_live_detector_invocation_during_replay_blocks_publication(self) -> None:
        self.bundle.rewrite_result(
            lambda result: result["detector_execution"].update(
                detector_invocations_during_replay=1
            )
        )
        self.assert_rejected("live detector execution")

    def test_strict_json_rejects_nan(self) -> None:
        self.bundle.result["observation_horizon_s"] = float("nan")
        write_json(self.bundle.results_path, self.bundle.result, allow_nan=True)
        self.assert_rejected("strict JSON")

    def test_artifact_hash_mismatch_blocks_publication(self) -> None:
        self.bundle.artifact_paths["data_freeze"].write_text(
            '{"frozen": false}\n', encoding="utf-8"
        )
        self.assert_rejected("sha256")

    def test_runtime_component_fingerprint_mismatch_blocks_publication(self) -> None:
        component = self.bundle.root / "component-runtime.py"
        component.write_text("# changed after run\n", encoding="utf-8")
        self.assert_rejected("component runtime moved")

    def test_source_control_gate_is_independently_recomputed(self) -> None:
        self.bundle.rewrite_artifact(
            "source_control_gate",
            lambda artifact: artifact["gate"].update(promoted_n=1),
        )
        self.assert_rejected("independent recomputation")

    def test_landing_rows_are_independently_checked(self) -> None:
        def mutate(artifact: dict[str, Any]) -> None:
            artifact["items"][0]["landed"] = not artifact["items"][0]["landed"]

        self.bundle.rewrite_artifact("landing", mutate)
        self.assert_rejected("landed flag")

    def test_b1_landing_counts_are_recomputed_from_six_frozen_rows(self) -> None:
        def mutate(result: dict[str, Any]) -> None:
            cell = next(row for row in result["cells"] if row["baseline"] == "B1")
            cell["retrieval"]["attack_landing_numerator"] -= 1

        self.bundle.rewrite_result(mutate)
        self.assert_rejected("six frozen landing rows")

    def test_plan_enforces_role_counts_strata_and_cross_attack_reuse(self) -> None:
        labels = self.bundle._label_by_key()
        landing_keys = set(self.bundle.landing_by_key)

        missing_clean = copy.deepcopy(self.bundle.plan)
        unit = missing_clean["runtime_units"][0]
        removed = unit["clean_item_keys"].pop()
        unit["injection_sequence"] = [
            event
            for event in unit["injection_sequence"]
            if event["item_key"] != removed
        ]
        for ordinal, event in enumerate(unit["injection_sequence"]):
            event["injection_ordinal"] = ordinal
        with self.assertRaisesRegex(
            W2DVerificationError, "does not have 6 clean"
        ):
            _verify_plan(missing_clean, labels, landing_keys)

        bad_strata = copy.deepcopy(self.bundle.plan)
        unit = next(
            row
            for row in bad_strata["runtime_units"]
            if row["seed"] == 1
            and row["attack_family"] == "recipe"
            and row["backlog"] == "normal"
        )
        hard_key = next(
            key
            for key in unit["clean_item_keys"]
            if labels[key]["stratum"] == "hard_negative_clean"
        )
        replacement = next(
            key
            for key in self.bundle.clean
            if labels[key]["stratum"] == "ordinary_clean"
            and all(
                key not in candidate["clean_item_keys"]
                and key not in candidate["filler_item_keys"]
                for candidate in bad_strata["runtime_units"]
            )
        )
        unit["clean_item_keys"][
            unit["clean_item_keys"].index(hard_key)
        ] = replacement
        for event in unit["injection_sequence"]:
            if event["item_key"] == hard_key:
                event["item_key"] = replacement
        with self.assertRaisesRegex(W2DVerificationError, "clean strata"):
            _verify_plan(bad_strata, labels, landing_keys)

        cross_attack_drift = copy.deepcopy(self.bundle.plan)
        unit = next(
            row
            for row in cross_attack_drift["runtime_units"]
            if row["seed"] == 1
            and row["attack_family"] == "natural_cover_suffix_v1"
            and row["backlog"] == "normal"
        )
        old_key = unit["clean_item_keys"][0]
        replacement = next(
            key
            for key in self.bundle.clean
            if labels[key]["stratum"] == labels[old_key]["stratum"]
            and all(
                key not in candidate["clean_item_keys"]
                and key not in candidate["filler_item_keys"]
                for candidate in cross_attack_drift["runtime_units"]
            )
        )
        unit["clean_item_keys"][0] = replacement
        for event in unit["injection_sequence"]:
            if event["item_key"] == old_key:
                event["item_key"] = replacement
        with self.assertRaisesRegex(
            W2DVerificationError, "clean set/order changes"
        ):
            _verify_plan(cross_attack_drift, labels, landing_keys)


if __name__ == "__main__":
    unittest.main()
