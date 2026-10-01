#!/usr/bin/env python3
"""Adversarial tests for the independent formal W2D bundle verifier."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from typing import Any, Callable

import numpy as np

from w2d_calibrate import select_threshold
from w2d_metrics import (
    detector_quality_summary,
    file_sha256,
    rate_record,
    source_control_gate_summary,
)
from test_verify_w2d_legacy import _document as legacy_document
from verify_w2d import (
    A6_HARD_NEGATIVE_RULES,
    A6_NOMINAL_PER_TOPIC,
    A6_ORDINARY_AVAILABLE,
    A6_ORDINARY_QUOTAS,
    A6_TOPICS,
    EXECUTION_MODE,
    FINGERPRINT_FILE_COMPONENTS,
    FROZEN_DATA_CONSTRUCTION,
    FROZEN_MODEL_REVISION,
    FROZEN_PROVIDER,
    FROZEN_SERVICE_REPLAY,
    MANIFEST_SCHEMA,
    MEASUREMENT_NAME,
    RESULT_SCHEMA,
    REQUIRED_FINGERPRINT_COMPONENTS,
    W2DVerificationError,
    _verify_plan,
    build_runtime_fingerprint,
    canonical_record_sha256,
    runtime_fingerprint_digest,
    verify_bundle,
)
from verify_w2d_legacy import evaluate as evaluate_legacy_regression
from verify_w2d_legacy import (
    legacy_gate_code_sha256,
    legacy_runtime_code_sha256,
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
        items: list[dict[str, Any]] = []
        allocation: list[dict[str, Any]] = []
        calibration: list[str] = []
        recipe: list[str] = []
        natural: list[str] = []
        ordinary_clean: list[str] = []
        hard_clean: list[str] = []
        controls: list[str] = []
        ordinal = 0

        def add(
            topic: str,
            split: str,
            stratum: str,
            count: int,
            *,
            family: str | None = None,
            variant: str | None = None,
            control: str | None = None,
            hard: bool = False,
            collect: list[str] | None = None,
        ) -> None:
            nonlocal ordinal
            for _ in range(count):
                source_group = opaque("formal-source-group", ordinal)
                key = hashlib.sha256(
                    f"W2D-item|{source_group}".encode("utf-8")
                ).hexdigest()
                normalized_text = f"formal scorer-safe text {key}"
                source_id = f"{topic}/{ordinal}"
                hard_rules = ["quoted_reply_lines"] if hard else []
                frozen = {
                    "item_key": key,
                    "source_group": source_group,
                    "source_id": source_id,
                    "source_aliases": [source_id],
                    "topic": topic,
                    "split": split,
                    "stratum": stratum,
                    "attack_family": family,
                    "attack_variant": variant,
                    "source_control_kind": control,
                    "hard_negative_rules": hard_rules,
                    "raw_sha256": opaque("formal-raw", ordinal),
                    "normalized_text_sha256": hashlib.sha256(
                        normalized_text.encode("utf-8")
                    ).hexdigest(),
                }
                if stratum == "natural_cover_suffix_poison":
                    frozen.update(
                        suffix_append_count=1,
                        longest_suffix_query_shared_ngram=0,
                    )
                allocation.append(frozen)
                items.append(
                    {
                        **frozen,
                        "poison": stratum
                        in {
                            "recipe_poison",
                            "natural_cover_suffix_poison",
                        },
                    }
                )
                if collect is not None:
                    collect.append(key)
                ordinal += 1

        for topic in A6_TOPICS:
            quota = A6_ORDINARY_QUOTAS[topic]
            add(topic, "reference", "clean_reference", 96)
            add(
                topic,
                "calibration",
                "ordinary_clean",
                quota["calibration"],
                collect=calibration,
            )
            add(
                topic,
                "calibration",
                "hard_negative_clean",
                8,
                hard=True,
                collect=calibration,
            )
            for variant in ("T0", "T1", "T2"):
                add(
                    topic,
                    "calibration",
                    "recipe_poison",
                    8,
                    family="recipe",
                    variant=variant,
                    hard=True,
                    collect=calibration,
                )
            add(
                topic,
                "test",
                "ordinary_clean",
                quota["test"],
                collect=ordinary_clean,
            )
            add(
                topic,
                "test",
                "hard_negative_clean",
                16,
                hard=True,
                collect=hard_clean,
            )
            for variant in ("T3", "T4"):
                add(
                    topic,
                    "test",
                    "recipe_poison",
                    8,
                    family="recipe",
                    variant=variant,
                    hard=True,
                    collect=recipe,
                )
            add(
                topic,
                "test",
                "natural_cover_suffix_poison",
                8,
                family="natural_cover_suffix_v1",
                variant="natural_cover_suffix_v1",
                hard=True,
                collect=natural,
            )
            for control in (
                "invalid_signature",
                "provenance_conflict",
                "unknown_source",
            ):
                add(
                    topic,
                    "implementation_control",
                    "source_family_negative_control",
                    1,
                    control=control,
                    hard=True,
                    collect=controls,
                )

        self.allocation = allocation
        self.calibration_keys = calibration
        self.recipe = recipe
        self.natural = natural
        self.ordinary_clean = ordinary_clean
        self.hard_clean = hard_clean
        self.clean = ordinary_clean + hard_clean
        self.control_keys = controls
        self.labels = {
            "schema_version": "1.0",
            "construction_version": FROZEN_DATA_CONSTRUCTION,
            "model_revision": FROZEN_MODEL_REVISION,
            "items": {
                item["item_key"]: {
                    key: value
                    for key, value in item.items()
                    if key != "item_key"
                }
                for item in items
            },
        }

    def _build_data_freeze(self) -> dict[str, Any]:
        expected_per_topic: dict[str, dict[str, int]] = {}
        for topic in A6_TOPICS:
            quota = A6_ORDINARY_QUOTAS[topic]
            expected = {
                **A6_NOMINAL_PER_TOPIC,
                "calibration_ordinary_clean": quota["calibration"],
                "test_ordinary_clean": quota["test"],
            }
            expected_per_topic[topic] = expected
        hard_keys = {
            split: [
                row["item_key"]
                for row in self.allocation
                if row["split"] == split
                and row["stratum"] == "hard_negative_clean"
            ]
            for split in ("calibration", "test")
        }
        eligible_unallocated: list[dict[str, Any]] = []
        classification_records = [
            {
                "group_id": row["source_group"],
                "topic": row["topic"],
                "rules": row["hard_negative_rules"],
            }
            for row in self.allocation
        ]
        for topic in A6_TOPICS:
            selected_ordinary = (
                A6_ORDINARY_QUOTAS[topic]["calibration"]
                + A6_ORDINARY_QUOTAS[topic]["test"]
            )
            for index in range(
                A6_ORDINARY_AVAILABLE[topic] - selected_ordinary
            ):
                group = opaque(f"formal-unallocated-{topic}", index)
                record = {"group_id": group, "topic": topic}
                eligible_unallocated.append(record)
                classification_records.append({**record, "rules": []})
        classification_records.sort(key=lambda row: row["group_id"])
        eligible_unallocated.sort(key=lambda row: row["group_id"])
        eligible_records = [
            {"group_id": row["group_id"], "topic": row["topic"]}
            for row in classification_records
        ]
        eligible_n = len(classification_records)

        return {
            "schema_version": "1.0",
            "construction_version": FROZEN_DATA_CONSTRUCTION,
            "build_provenance": {
                "git_commit": "a" * 40,
                "git_dirty": False,
                "dirty_paths": [],
                "w2d_workload_py_sha256": file_sha256(
                    str(self.root / "w2d_workload.py")
                ),
                "amendment_a6_sha256": file_sha256(
                    str(
                        self.root
                        / "W2D-PREREGISTRATION-AMENDMENT-A6.md"
                    )
                ),
            },
            "source": {
                "raw_group_count": eligible_n,
                "inventory": {
                    "raw_post_count": eligible_n,
                    "canonical_group_count": eligible_n,
                    "duplicate_noncanonical_count": 0,
                    "base_eligible_group_count": eligible_n,
                    "base_excluded_group_count": 0,
                    "allocated_group_count": 1752,
                    "eligible_unallocated_group_count": len(
                        eligible_unallocated
                    ),
                },
            },
            "model": {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "revision": FROZEN_MODEL_REVISION,
                "dimension": 384,
            },
            "nominal_per_topic": A6_NOMINAL_PER_TOPIC,
            "expected_per_topic": expected_per_topic,
            "ordinary_clean_quota_rule": {
                "amendment": "W2D-PREREGISTRATION-AMENDMENT-A6.md",
                "available_after_reference": A6_ORDINARY_AVAILABLE,
                "quota": A6_ORDINARY_QUOTAS,
                "global_calibration_total": 192,
                "global_test_total": 192,
                "topic_order": list(A6_TOPICS),
            },
            "counts": {
                "total_items": 1752,
                "unique_source_groups": 1752,
                "reference": 768,
                "calibration_quality": 448,
                "test_quality": 512,
                "source_controls": 24,
                "calibration_clean": 256,
                "calibration_recipe_poison": 192,
                "test_clean": 320,
                "test_recipe_poison": 128,
                "test_natural_cover_poison": 64,
                "per_topic": expected_per_topic,
            },
            "base_eligibility": {"records": eligible_records},
            "exclusions": [],
            "eligible_unallocated": eligible_unallocated,
            "hard_negative_classification": {
                "allowed_rules": list(A6_HARD_NEGATIVE_RULES),
                "records": classification_records,
            },
            "allocation": self.allocation,
            "hard_negative_selection": {
                split: {
                    "n": len(keys),
                    "item_keys": sorted(keys),
                    "rule_counts": {"quoted_reply_lines": len(keys)},
                    "items": {
                        key: ["quoted_reply_lines"] for key in sorted(keys)
                    },
                }
                for split, keys in hard_keys.items()
            },
            "artifacts": {
                self.artifact_paths[role].name: {
                    "sha256": file_sha256(str(self.artifact_paths[role])),
                    "role": expected_role,
                }
                for role, expected_role in {
                    "safe_inputs_json": "safe_detector_schema",
                    "safe_inputs_npz": "safe_full_item_embeddings",
                    "labels": "evaluator_only_labels_queries_and_attack_family",
                    "labels_checksum": "label_artifact_digest",
                }.items()
            },
        }

    def _build_safe_inputs(self) -> dict[str, Any]:
        ordered = sorted(self.allocation, key=lambda row: row["item_key"])
        items = [
            {
                "item_key": row["item_key"],
                "normalized_text": f"formal scorer-safe text {row['item_key']}",
                "embedding_row": index,
                "source_evidence": (
                    [False, True, FROZEN_MODEL_REVISION]
                    if row["source_control_kind"]
                    in {"invalid_signature", "unknown_source"}
                    else [True, False, FROZEN_MODEL_REVISION]
                    if row["source_control_kind"] == "provenance_conflict"
                    else [True, True, FROZEN_MODEL_REVISION]
                ),
                "source_group": row["source_group"],
                "reference": row["split"] == "reference",
                "model_revision": FROZEN_MODEL_REVISION,
            }
            for index, row in enumerate(ordered)
        ]
        reference_items = [
            item for item in items if item["reference"]
        ]
        calibration_keys = sorted(
            row["item_key"]
            for row in self.allocation
            if row["split"] == "calibration"
        )
        test_keys = [
            row["item_key"]
            for row in self.allocation
            if row["split"] == "test"
        ]
        test_order = sorted(
            test_keys,
            key=lambda key: hashlib.sha256(
                f"W2D-score-order|{key}".encode("utf-8")
            ).hexdigest(),
        )
        control_keys = sorted(
            row["item_key"]
            for row in self.allocation
            if row["split"] == "implementation_control"
        )
        npz_digest = hashlib.sha256(self.npz_payload).hexdigest()
        return {
            "schema_version": "1.0",
            "construction_version": FROZEN_DATA_CONSTRUCTION,
            "model_revision": FROZEN_MODEL_REVISION,
            "model": {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "revision": FROZEN_MODEL_REVISION,
                "dimension": 384,
            },
            "embedding_artifact": {
                "filename": "W2D-detector-inputs.npz",
                "sha256": npz_digest,
                "array": "embeddings",
                "dtype": "float32",
                "shape": [1752, 384],
            },
            "items": items,
            "reference_index": {
                "item_keys": [item["item_key"] for item in reference_items],
                "embedding_rows": [
                    item["embedding_row"] for item in reference_items
                ],
                "groups": [item["source_group"] for item in reference_items],
            },
            "sets": {
                "calibration_item_keys": calibration_keys,
                "test_score_order": test_order,
                "source_control_item_keys": control_keys,
                "warmup_reference_item_key": min(
                    item["item_key"] for item in reference_items
                ),
            },
        }

    def _build_legacy(self) -> None:
        authoritative_w2 = legacy_document()
        authoritative_w2r = legacy_document(realtext=True)
        candidate_w2 = copy.deepcopy(authoritative_w2)
        candidate_w2r = copy.deepcopy(authoritative_w2r)
        same_code_repeat = copy.deepcopy(candidate_w2)
        runtime_sha256 = legacy_runtime_code_sha256()
        for document in (candidate_w2, candidate_w2r, same_code_repeat):
            document["runtime_code_sha256"] = runtime_sha256
        next(
            cell
            for cell in same_code_repeat["metrics"]["cells"]
            if cell["baseline"] == "B4"
            and cell["backlog"] == "normal"
            and cell["Tp"] == 0.3
            and cell["seed"] == 4
        )["Ep_p50"] = 0.201
        source_values = {
            "authoritative_w2": (
                "legacy_authoritative_w2",
                authoritative_w2,
            ),
            "authoritative_w2r": (
                "legacy_authoritative_w2r",
                authoritative_w2r,
            ),
            "candidate_w2": ("legacy_candidate_w2", candidate_w2),
            "candidate_w2r": ("legacy_candidate_w2r", candidate_w2r),
            "same_code_repeat": (
                "legacy_same_code_repeat",
                same_code_repeat,
            ),
        }
        filenames = {
            "legacy_authoritative_w2": "W2-inmemory.json",
            "legacy_authoritative_w2r": "W2R-inmemory.json",
            "legacy_candidate_w2": "W2-legacy-regression-run-a.json",
            "legacy_candidate_w2r": "W2R-legacy-regression.json",
            "legacy_same_code_repeat": (
                "W2-legacy-regression-run-b-same-code.json"
            ),
        }
        for _role, (name, value) in source_values.items():
            self._artifact(name, filenames[name], value)
        authority = {
            "files": {
                "w2_inmemory": filenames["legacy_authoritative_w2"],
                "w2r_inmemory": filenames["legacy_authoritative_w2r"],
            },
            "sha256": {
                filenames["legacy_authoritative_w2"]: file_sha256(
                    str(self.artifact_paths["legacy_authoritative_w2"])
                ),
                filenames["legacy_authoritative_w2r"]: file_sha256(
                    str(self.artifact_paths["legacy_authoritative_w2r"])
                ),
            },
        }
        self._artifact(
            "legacy_authority_manifest", "AUTHORITATIVE.json", authority
        )
        source_roles = {
            "authority_manifest": "legacy_authority_manifest",
            **{role: name for role, (name, _value) in source_values.items()},
        }
        source_hashes = {
            role: file_sha256(str(self.artifact_paths[name]))
            for role, name in source_roles.items()
        }
        self.legacy_source_hashes = source_hashes
        replacement = evaluate_legacy_regression(
            authority_manifest=authority,
            authoritative_w2=authoritative_w2,
            authoritative_w2r=authoritative_w2r,
            candidate_w2=candidate_w2,
            candidate_w2r=candidate_w2r,
            same_code_repeat=same_code_repeat,
            source_sha256=source_hashes,
            expected_runtime_code_sha256=runtime_sha256,
            enforce_frozen_same_code_differences=False,
        )
        replacement["checks"].append(
            "same-code counterexample has the exact frozen difference projection"
        )
        gate = {
            "schema_version": "W2D-legacy-regression-v2",
            "amendment": "W2D-PREREGISTRATION-AMENDMENT-A5.md",
            "source_artifacts": {
                role: {
                    "filename": self.artifact_paths[name].name,
                    "sha256": file_sha256(str(self.artifact_paths[name])),
                }
                for role, name in source_roles.items()
            },
            "runtime_code_sha256": candidate_w2["runtime_code_sha256"],
            "git_commit": candidate_w2["git_commit"],
            "gate_code_sha256": legacy_gate_code_sha256(),
            "replacement_gate": replacement,
            "existing_gate_executions": [
                {
                    "command": (
                        [
                            str(Path(sys.executable).resolve()),
                            script,
                            str(self.artifact_paths[artifact]),
                        ]
                        if artifact is not None
                        else [str(Path(sys.executable).resolve()), script]
                    ),
                    "returncode": 0,
                    "output_sha256": "a" * 64,
                    "passed": True,
                }
                for script, artifact in (
                    ("verify_w2.py", "legacy_candidate_w2"),
                    ("verify_w2.py", "legacy_candidate_w2r"),
                    ("verify_w2r.py", "legacy_candidate_w2r"),
                    ("verify_w2.py", "legacy_same_code_repeat"),
                    ("test_w2d_runtime.py", None),
                    ("test_w2.py", None),
                    ("test_invariants.py", None),
                    ("test_verify_w2d_legacy.py", None),
                )
            ],
        }
        self._artifact(
            "legacy_regression", "W2D-LEGACY-REGRESSION.json", gate
        )

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
        ordinary = iter(self.ordinary_clean)
        hard = iter(self.hard_clean)

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
        for name, filename in sorted(FINGERPRINT_FILE_COMPONENTS.items()):
            path = self.root / filename
            path.write_text(f"# frozen {name}\n", encoding="utf-8")
            components[name] = {"path": path.name}
        components["model_revision"] = {"value": FROZEN_MODEL_REVISION}
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
        label_by_key = self.labels["items"]
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
        return dict(self.labels["items"])

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
        state = (
            "PROVISIONAL"
            if baseline == "B1"
            else ("TRUSTED" if passes else "QUARANTINED")
        )
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
        if baseline == "B2":
            unvetted_status, unvetted_age = "NOT_STARTED", None
        elif baseline == "B1":
            unvetted_status, unvetted_age = "RIGHT_CENSORED", 7.0
        else:
            unvetted_status, unvetted_age = "COMPLETED", 0.5
        return {
            "item_key": key,
            "truth_poison": truth,
            "detector_promote": detector_promote,
            "retry_count": 0,
            "state_at_horizon": state,
            "visibility_status": visibility,
            "exposure_status": exposure_status,
            "exposure_observed_s": exposure_age,
            "unvetted_visibility_status": unvetted_status,
            "unvetted_visibility_observed_s": unvetted_age,
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
                            "queue_enter_s": 1.0,
                            "queue_start_s": 1.0 + baseline_delta,
                            "queue_wait_s": baseline_delta,
                            "queue_depth_at_enqueue": 1,
                            "queue_depth_at_start": 0,
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
        self._build_legacy()
        self._build_labels()
        buffer = io.BytesIO()
        embeddings = np.zeros((1752, 384), dtype=np.float32)
        embeddings[:, 0] = 1.0
        np.savez_compressed(
            buffer, embeddings=embeddings
        )
        self.npz_payload = buffer.getvalue()
        self._artifact(
            "safe_inputs_json",
            "W2D-detector-inputs.json",
            self._build_safe_inputs(),
        )
        self._artifact(
            "safe_inputs_npz", "W2D-detector-inputs.npz", self.npz_payload
        )
        self._artifact("labels", "W2D-evaluator-only.json", self.labels)
        self._artifact(
            "labels_checksum",
            "W2D-labels.sha256",
            (
                f"{file_sha256(str(self.artifact_paths['labels']))}  "
                f"{self.artifact_paths['labels'].name}\n"
            ).encode("ascii"),
        )
        self.fingerprint = self._build_components()
        self._artifact(
            "data_freeze",
            "W2D-DATA-FREEZE.json",
            self._build_data_freeze(),
        )
        self._build_plan()
        self._artifact("protocol_plan", "W2D-PROTOCOL-PLAN.json", self.plan)
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

    def rewrite_bound_data_child(
        self, name: str, mutator: Callable[[dict[str, Any]], None]
    ) -> None:
        path = self.artifact_paths[name]
        document = json.loads(path.read_text(encoding="utf-8"))
        mutator(document)
        write_json(path, document)
        changed = {name}
        if name == "labels":
            checksum = self.artifact_paths["labels_checksum"]
            checksum.write_text(
                f"{file_sha256(str(path))}  {path.name}\n",
                encoding="ascii",
            )
            changed.add("labels_checksum")
        freeze_path = self.artifact_paths["data_freeze"]
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        for role in changed:
            child = self.artifact_paths[role]
            freeze["artifacts"][child.name]["sha256"] = file_sha256(
                str(child)
            )
        write_json(freeze_path, freeze)
        changed.add("data_freeze")
        for role in changed:
            digest = file_sha256(str(self.artifact_paths[role]))
            self.manifest["artifact_hashes"][role]["sha256"] = digest
            self.result["artifact_hashes"][role] = digest
        write_json(self.manifest_path, self.manifest)
        write_json(self.results_path, self.result)

    def rewrite_bound_binary_child(self, name: str, payload: bytes) -> None:
        path = self.artifact_paths[name]
        path.write_bytes(payload)
        changed = {name}
        if name == "safe_inputs_npz":
            safe_path = self.artifact_paths["safe_inputs_json"]
            safe = json.loads(safe_path.read_text(encoding="utf-8"))
            safe["embedding_artifact"]["sha256"] = file_sha256(str(path))
            write_json(safe_path, safe)
            changed.add("safe_inputs_json")
        freeze_path = self.artifact_paths["data_freeze"]
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        for role in changed:
            child = self.artifact_paths[role]
            freeze["artifacts"][child.name]["sha256"] = file_sha256(
                str(child)
            )
        write_json(freeze_path, freeze)
        changed.add("data_freeze")
        for role in changed:
            digest = file_sha256(str(self.artifact_paths[role]))
            self.manifest["artifact_hashes"][role]["sha256"] = digest
            self.result["artifact_hashes"][role] = digest
        write_json(self.manifest_path, self.manifest)
        write_json(self.results_path, self.result)


class VerifyW2DTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.bundle = FormalBundle(Path(self.temporary.name))
        gate = json.loads(
            self.bundle.artifact_paths["legacy_regression"].read_text(
                encoding="utf-8"
            )
        )
        difference_sha256 = canonical_sha256(
            gate["replacement_gate"]["same_code_scheduling_differences"]
        )
        artifact_paths = {
            name: path.name
            for name, path in self.bundle.artifact_paths.items()
        }
        self.patchers = [
            patch(
                "verify_w2d.FROZEN_SOURCE_SHA256",
                self.bundle.legacy_source_hashes,
            ),
            patch(
                "verify_w2d.FROZEN_SAME_CODE_DIFFERENCE_SHA256",
                difference_sha256,
            ),
            patch(
                "verify_w2d_legacy.FROZEN_SAME_CODE_DIFFERENCE_SHA256",
                difference_sha256,
            ),
            patch(
                "verify_w2d.EXPECTED_ARTIFACT_ROOT",
                self.bundle.root,
            ),
            patch(
                "verify_w2d.EXPECTED_ARTIFACT_RELATIVE_PATHS",
                artifact_paths,
            ),
            patch(
                "verify_w2d.EXPECTED_FINGERPRINT_ROOT",
                self.bundle.root,
            ),
            patch("verify_w2d.rerun_legacy_gates", return_value=[]),
        ]
        for patcher in self.patchers:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temporary.cleanup()

    def assert_rejected(self, needle: str | None = None) -> None:
        report = verify_bundle(
            self.bundle.manifest_path, self.bundle.results_path
        )
        self.assertFalse(report.ok, report.checks)
        if needle is not None:
            self.assertIn(needle, " ".join(report.failures))

    def test_full_formal_bundle_passes_with_unpaired_integrated_latency(self) -> None:
        with patch(
            "verify_w2d.rerun_legacy_gates", return_value=[]
        ) as rerun:
            report = verify_bundle(
                self.bundle.manifest_path, self.bundle.results_path
            )
        self.assertTrue(report.ok, report.failures)
        rerun.assert_called_once()
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

    def test_b2_cannot_start_an_unvetted_visible_episode(self) -> None:
        def mutate(result: dict[str, Any]) -> None:
            cell = next(
                row for row in result["cells"] if row["baseline"] == "B2"
            )
            cell["lifecycle"][0]["unvetted_visibility_status"] = "COMPLETED"
            cell["lifecycle"][0]["unvetted_visibility_observed_s"] = 0.001

        self.bundle.rewrite_result(mutate)
        self.assert_rejected("B2 item")

    def test_b4_unvetted_episode_bound_is_per_item(self) -> None:
        def mutate(result: dict[str, Any]) -> None:
            cell = next(
                row for row in result["cells"] if row["baseline"] == "B4"
            )
            cell["lifecycle"][0]["unvetted_visibility_observed_s"] = 1.021

        self.bundle.rewrite_result(mutate)
        self.assert_rejected("E_u <= 1.02")

    def test_queue_timestamps_must_balance_integrated_latency(self) -> None:
        def mutate(result: dict[str, Any]) -> None:
            cell = next(
                row for row in result["cells"] if row["baseline"] != "B1"
            )
            cell["verifier_records"][0]["queue_wait_s"] += 0.1

        self.bundle.rewrite_result(mutate)
        self.assert_rejected("queue timing does not balance")

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

    def test_a6_quota_ledger_is_independently_recomputed(self) -> None:
        def mutate(artifact: dict[str, Any]) -> None:
            artifact["ordinary_clean_quota_rule"]["quota"]["sci.space"][
                "test"
            ] += 1

        self.bundle.rewrite_artifact("data_freeze", mutate)
        self.assert_rejected("independent recomputation")

    def test_a6_ordinary_and_hard_negative_semantics_are_enforced(self) -> None:
        def mutate(artifact: dict[str, Any]) -> None:
            record = next(
                row
                for row in artifact["allocation"]
                if row["stratum"] == "ordinary_clean"
            )
            record["hard_negative_rules"] = ["quoted_reply_lines"]

        self.bundle.rewrite_artifact("data_freeze", mutate)
        self.assert_rejected("ordinary-clean record")

    def test_a6_rejects_unknown_split_or_stratum(self) -> None:
        def mutate(artifact: dict[str, Any]) -> None:
            record = next(
                row
                for row in artifact["allocation"]
                if row["stratum"] == "clean_reference"
            )
            record["split"] = "not_a_split"
            record["stratum"] = "not_a_stratum"

        self.bundle.rewrite_artifact("data_freeze", mutate)
        self.assert_rejected("unknown split/stratum")

    def test_a6_labels_must_match_allocation_itemwise(self) -> None:
        first_key = next(iter(self.bundle.labels["items"]))
        self.bundle.rewrite_bound_data_child(
            "labels",
            lambda labels: labels["items"][first_key].update(
                source_group="f" * 64
            ),
        )
        self.assert_rejected("differs from data-freeze allocation")

    def test_a6_labels_must_use_the_runtime_mapping_schema(self) -> None:
        def mutate(labels: dict[str, Any]) -> None:
            labels["items"] = [
                {"item_key": key, **record}
                for key, record in labels["items"].items()
            ]

        self.bundle.rewrite_bound_data_child("labels", mutate)
        self.assert_rejected("opaque-key mapping")

    def test_a6_safe_items_must_match_allocation_itemwise(self) -> None:
        self.bundle.rewrite_bound_data_child(
            "safe_inputs_json",
            lambda safe: safe["items"][0].update(source_group="e" * 64),
        )
        self.assert_rejected("differs from data-freeze allocation")

    def test_a6_natural_cover_diagnostics_are_exact(self) -> None:
        def mutate(freeze: dict[str, Any]) -> None:
            record = next(
                row
                for row in freeze["allocation"]
                if row["stratum"] == "natural_cover_suffix_poison"
            )
            record["suffix_append_count"] = 2

        self.bundle.rewrite_artifact("data_freeze", mutate)
        self.assert_rejected("natural-cover construction diagnostics")

    def test_a6_safe_text_is_normalized_and_bound_to_freeze(self) -> None:
        self.bundle.rewrite_bound_data_child(
            "safe_inputs_json",
            lambda safe: safe["items"][0].update(
                normalized_text=safe["items"][0]["normalized_text"] + " changed"
            ),
        )
        self.assert_rejected("text differs from the data-freeze allocation")

    def test_a6_source_evidence_matches_control_kind(self) -> None:
        def mutate(safe: dict[str, Any]) -> None:
            control_key = next(
                key
                for key, row in self.bundle.labels["items"].items()
                if row["source_control_kind"] == "invalid_signature"
            )
            next(
                row for row in safe["items"] if row["item_key"] == control_key
            )["source_evidence"] = [True, True, FROZEN_MODEL_REVISION]

        self.bundle.rewrite_bound_data_child("safe_inputs_json", mutate)
        self.assert_rejected("source evidence differs")

    def test_a6_embedding_rows_must_be_unit_norm(self) -> None:
        matrix = np.zeros((1752, 384), dtype=np.float32)
        matrix[:, 0] = 1.0
        matrix[0, 0] = 0.0
        buffer = io.BytesIO()
        np.savez_compressed(buffer, embeddings=matrix)
        self.bundle.rewrite_bound_binary_child(
            "safe_inputs_npz", buffer.getvalue()
        )
        self.assert_rejected("zero or non-normalized row")

    def test_manifest_rejects_extra_artifact_role(self) -> None:
        self.bundle.manifest["artifact_hashes"]["result_cycle"] = {
            "path": self.bundle.results_path.name,
            "sha256": file_sha256(str(self.bundle.results_path)),
        }
        write_json(self.bundle.manifest_path, self.bundle.manifest)
        self.assert_rejected("extra=['result_cycle']")

    def test_manifest_rejects_extra_top_level_field(self) -> None:
        self.bundle.manifest["unfrozen_note"] = "not part of the schema"
        write_json(self.bundle.manifest_path, self.bundle.manifest)
        self.assert_rejected("extra=['unfrozen_note']")

    def test_runtime_component_role_is_bound_to_declared_path(self) -> None:
        components = self.bundle.manifest["runtime_fingerprint"]["components"]
        components["runtime"] = copy.deepcopy(components["backend"])
        self.bundle.manifest["runtime_fingerprint"]["sha256"] = (
            runtime_fingerprint_digest(components)
        )
        write_json(self.bundle.manifest_path, self.bundle.manifest)
        self.assert_rejected("not its declared source file")

    def test_runtime_component_fingerprint_mismatch_blocks_publication(self) -> None:
        component = self.bundle.root / FINGERPRINT_FILE_COMPONENTS["runtime"]
        component.write_text("# changed after run\n", encoding="utf-8")
        self.assert_rejected("component runtime moved")

    def test_legacy_recorded_pass_cannot_replace_independent_execution(self) -> None:
        def mutate(artifact: dict[str, Any]) -> None:
            artifact["existing_gate_executions"][0]["command"][0] = "/bin/true"
            artifact["existing_gate_executions"][0]["output_sha256"] = "0" * 64

        self.bundle.rewrite_artifact("legacy_regression", mutate)
        self.assert_rejected("command changed")

    def test_legacy_regression_top_level_commit_is_frozen(self) -> None:
        self.bundle.rewrite_artifact(
            "legacy_regression",
            lambda artifact: artifact.update(git_commit="deadbeef"),
        )
        self.assert_rejected("frozen candidate commit")

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
