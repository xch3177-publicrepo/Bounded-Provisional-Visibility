"""Unit tests for the downstream W2D statistical analyzer.

The fixture is a complete-cardinality synthetic result chain.  The formal
verifier is replaced only at the call boundary: these tests exercise analysis
semantics without importing the numpy-backed detector stack.  Production code
still imports and calls ``verify_w2d.verify_or_raise`` before reading a byte.

    python test_w2d_analyze.py
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import w2d_analyze
from w2d_metrics import strict_json_load_bytes
from w2d_snapshot import OutputOwnershipError, SnapshotRegistry


MEASUREMENT = (
    "promotion-path replay driven by real D1 outputs and measured service times"
)
ATTACKS = ("recipe", "natural_cover_suffix_v1")
BACKLOGS = ("normal", "heavy")
BASELINE_ARMS = (
    ("B1", "control"),
    ("B2", "detector"),
    ("B2", "oracle"),
    ("B3", "detector"),
    ("B3", "oracle"),
    ("B4", "detector"),
    ("B4", "oracle"),
)


def _key(prefix: str, index: int) -> str:
    return hashlib.sha256(f"{prefix}|{index}".encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


@contextmanager
def _verifier(result=None, error: BaseException | None = None):
    calls = []
    module = types.ModuleType("verify_w2d")

    def verify_or_raise(manifest_path, results_path):
        calls.append((str(manifest_path), str(results_path)))
        if error is not None:
            raise error
        if result is not None:
            return result
        registry = SnapshotRegistry()
        authority_snapshot = registry.capture(
            Path(manifest_path).resolve().parent
            / "W2D-MANIFEST-AUTHORITY.json"
        )
        manifest_snapshot = registry.capture(manifest_path)
        result_snapshot = registry.capture(results_path)
        authority = strict_json_load_bytes(
            authority_snapshot.payload,
            source=str(authority_snapshot.path),
        )
        manifest = strict_json_load_bytes(
            manifest_snapshot.payload,
            source=str(manifest_snapshot.path),
        )
        result_document = strict_json_load_bytes(
            result_snapshot.payload,
            source=str(result_snapshot.path),
        )
        artifact_snapshots = {}
        artifacts = {}
        artifact_hashes = {}
        artifact_paths = {}
        for role, record in manifest["artifact_hashes"].items():
            path = Path(record["path"])
            if not path.is_absolute():
                path = manifest_snapshot.path.parent / path
            snapshot = registry.capture(path)
            artifact_snapshots[role] = snapshot
            artifacts[role] = strict_json_load_bytes(
                snapshot.payload,
                source=str(snapshot.path),
            )
            artifact_hashes[role] = snapshot.sha256
            artifact_paths[role] = snapshot.path
        analyzer_snapshot = registry.capture(Path(w2d_analyze.__file__))
        return types.SimpleNamespace(
            registry=registry,
            authority_path=authority_snapshot.path,
            authority_snapshot=authority_snapshot,
            manifest_snapshot=manifest_snapshot,
            result_snapshot=result_snapshot,
            authority=authority,
            manifest=manifest,
            result=result_document,
            artifacts=artifacts,
            artifact_hashes=artifact_hashes,
            artifact_paths=artifact_paths,
            artifact_snapshots=artifact_snapshots,
            component_snapshots={"analyzer": analyzer_snapshot},
        )

    def verify_snapshot_or_raise(snapshot):
        calls.append(("same_snapshot", id(snapshot)))
        snapshot.registry.assert_unchanged()
        return snapshot

    module.verify_or_raise = verify_or_raise
    module.verify_snapshot_or_raise = verify_snapshot_or_raise
    previous = sys.modules.get("verify_w2d")
    sys.modules["verify_w2d"] = module
    try:
        yield calls
    finally:
        if previous is None:
            sys.modules.pop("verify_w2d", None)
        else:
            sys.modules["verify_w2d"] = previous


def _rate_keys(value):
    if isinstance(value, dict):
        if {"numerator", "denominator", "rate", "wilson95"} <= set(value):
            yield value
        for child in value.values():
            yield from _rate_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _rate_keys(child)


def _recursive_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _recursive_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _recursive_keys(child)


class SyntheticBundle:
    def __init__(self, root: Path):
        self.root = root
        self.recipe = [_key("recipe", index) for index in range(128)]
        self.natural = [_key("natural", index) for index in range(64)]
        self.ordinary = [_key("ordinary", index) for index in range(192)]
        self.hard = [_key("hard", index) for index in range(128)]
        self.controls = [_key("control", index) for index in range(24)]
        self.labels = self._labels()
        self.test_scores = self._test_scores()
        self.source_controls = self._source_controls()
        self.landing = self._landing()
        self.result = self._result()
        self._write()

    def _labels(self):
        labels = {}

        def add(
            key: str,
            *,
            poison: bool,
            split: str,
            stratum: str,
            attack_family=None,
            source_control_kind=None,
        ):
            labels[key] = {
                "source_group": hashlib.sha256(
                    f"group|{key}".encode("utf-8")
                ).hexdigest(),
                "topic": f"topic-{int(key[0], 16) % 8}",
                "split": split,
                "stratum": stratum,
                "poison": poison,
                "attack_family": attack_family,
                "source_control_kind": source_control_kind,
            }

        for key in self.recipe:
            add(
                key,
                poison=True,
                split="test",
                stratum="recipe_poison",
                attack_family="recipe",
            )
        for key in self.natural:
            add(
                key,
                poison=True,
                split="test",
                stratum="natural_cover_suffix_poison",
                attack_family="natural_cover_suffix_v1",
            )
        for key in self.ordinary:
            add(
                key,
                poison=False,
                split="test",
                stratum="ordinary_clean",
            )
        for key in self.hard:
            add(
                key,
                poison=False,
                split="test",
                stratum="hard_negative_clean",
            )
        kinds = ("invalid_signature", "provenance_conflict", "unknown_source")
        for index, key in enumerate(self.controls):
            add(
                key,
                poison=False,
                split="implementation_control",
                stratum="source_family_negative_control",
                source_control_kind=kinds[index // 8],
            )
        return {
            "schema_version": "1.0",
            "model_revision": "synthetic",
            "items": labels,
        }

    @staticmethod
    def _score_document(phase: str, records):
        digest = "a" * 64
        return {
            "schema_version": "W2D-score-v1",
            "phase": phase,
            "measurement_name": MEASUREMENT,
            "safe_inputs_sha256": digest,
            "embeddings_sha256": digest,
            "detector_py_sha256": digest,
            "scorer_py_sha256": digest,
            "threshold_artifact_sha256": digest,
            "items": records,
        }

    def _test_scores(self):
        records = []
        all_keys = self.recipe + self.natural + self.ordinary + self.hard
        poison = set(self.recipe + self.natural)
        false_negative = {self.recipe[0], self.natural[0]}
        false_positive = {self.ordinary[0], self.hard[0]}
        for index, key in enumerate(all_keys):
            promote = key in false_negative or (
                key not in poison and key not in false_positive
            )
            records.append(
                {
                    "item_key": key,
                    "family_s_affirms": True,
                    "c_score": 0.0 if promote else 1.0,
                    "promote": promote,
                    "detector_service_ns": 1_000_000 + index,
                }
            )
        return self._score_document("test", records)

    def _source_controls(self):
        return self._score_document(
            "source_control",
            [
                {
                    "item_key": key,
                    "family_s_affirms": False,
                    "c_score": 1.0,
                    "promote": False,
                    "detector_service_ns": 2_000_000 + index,
                }
                for index, key in enumerate(self.controls)
            ],
        )

    def _landing(self):
        items = []
        for family, keys in (
            ("recipe", self.recipe),
            ("natural_cover_suffix_v1", self.natural),
        ):
            for index, key in enumerate(keys):
                landed = index % 2 == 0
                items.append(
                    {
                        "item_key": key,
                        "attack_family": family,
                        "rank": 1 if landed else 6,
                        "candidate_rank": 1 if landed else 6,
                        "landed_top5": landed,
                        "landed": landed,
                        "top5_item_keys": [key] if landed else [],
                    }
                )
        return {
            "schema_version": "W2D-landing-v1",
            "items": items,
        }

    def _score_by_key(self):
        return {
            record["item_key"]: record for record in self.test_scores["items"]
        }

    def _landing_by_key(self):
        return {record["item_key"]: record for record in self.landing["items"]}

    @staticmethod
    def _backend_evidence():
        identity = {"module": "backend", "class": "InMemoryBackend"}
        endpoint = {
            "kind": "process_local",
            "absolute_uri_omitted": True,
        }
        boundary = (
            "live_descriptive_receipt_not_attestation_"
            "outside_measurement_window"
        )
        pre = {
            "schema_version": "W2D-backend-evidence-v1",
            "phase": "pre",
            "backend_name": "inmemory",
            "backend_identity": dict(identity),
            "endpoint": dict(endpoint),
            "evidence_boundary": boundary,
            "server": None,
            "collection": None,
            "index": None,
            "load": None,
            "terminal_state": None,
        }
        post = {
            "schema_version": "W2D-backend-evidence-v1",
            "phase": "post",
            "backend_name": "inmemory",
            "backend_identity": dict(identity),
            "endpoint": dict(endpoint),
            "evidence_boundary": boundary,
            "server": None,
            "collection": None,
            "index": None,
            "load": None,
            "terminal_state": {
                "source": "in_memory_object",
                "consistency": "strong_in_process",
                "row_count": 768,
                "visible_true_count": 768,
                "vector_dim": 384,
                "logical_state_sha256": "e" * 64,
                "expected_seed_state_checked": True,
            },
        }
        return {"pre": pre, "post": post}

    def _lifecycle(self, key, role, baseline, arm):
        label = self.labels["items"][key]
        truth = label["poison"]
        score = self._score_by_key()[key]
        detector_promote = score["promote"]
        if baseline == "B1":
            passes = True
        else:
            passes = detector_promote if arm == "detector" else not truth

        if baseline == "B1":
            visibility = "STARTED"
            state = "PROVISIONAL"
            exposure_status = "RIGHT_CENSORED" if truth else "NOT_APPLICABLE"
            exposure = 7.0 if truth else None
            eu_status, eu_age = "RIGHT_CENSORED", 7.0
            unavailable_status, unavailable = "NOT_STARTED", None
            quarantine_status = "NOT_STARTED"
        else:
            if truth:
                visibility = (
                    "NOT_STARTED"
                    if baseline == "B2" and not passes
                    else "STARTED"
                )
                state = "TRUSTED" if passes else "QUARANTINED"
                exposure_status = (
                    "RIGHT_CENSORED"
                    if passes
                    else ("NOT_STARTED" if visibility == "NOT_STARTED" else "COMPLETED")
                )
                exposure = (
                    7.0
                    if exposure_status == "RIGHT_CENSORED"
                    else (0.5 if exposure_status == "COMPLETED" else None)
                )
                unavailable_status = (
                    "RIGHT_CENSORED"
                    if visibility == "NOT_STARTED"
                    else "NOT_APPLICABLE"
                )
                unavailable = 7.0 if visibility == "NOT_STARTED" else None
                quarantine_status = (
                    "RIGHT_CENSORED" if not passes else "NOT_STARTED"
                )
            else:
                visibility = (
                    "NOT_STARTED"
                    if baseline == "B2" and not passes
                    else "STARTED"
                )
                state = "TRUSTED" if passes else "QUARANTINED"
                exposure_status, exposure = "NOT_APPLICABLE", None
                unavailable_status = (
                    "RIGHT_CENSORED" if not passes else "COMPLETED"
                )
                unavailable = 7.0 if not passes else 0.2
                quarantine_status = (
                    "RIGHT_CENSORED" if not passes else "NOT_STARTED"
                )
            if baseline == "B2":
                eu_status, eu_age = "NOT_STARTED", None
            else:
                eu_status, eu_age = "COMPLETED", 0.5
        if baseline == "B1":
            transitions = [
                {
                    "t_s": 1.0,
                    "old_state": None,
                    "new_state": "PROVISIONAL",
                }
            ]
        else:
            wait = {"B2": 0.1, "B3": 0.2, "B4": 0.3}[baseline]
            decision_commit_s = (
                1.0
                + wait
                + score["detector_service_ns"] / 1_000_000_000
            )
            transitions = []
            if baseline in {"B3", "B4"}:
                transitions.append(
                    {
                        "t_s": 1.0,
                        "old_state": None,
                        "new_state": "PROVISIONAL",
                    }
                )
            transitions.append(
                {
                    "t_s": decision_commit_s,
                    "old_state": (
                        None if baseline == "B2" else "PROVISIONAL"
                    ),
                    "new_state": state,
                }
            )
        return {
            "item_key": key,
            "truth_poison": truth,
            "detector_promote": detector_promote,
            "retry_count": 0,
            "transitions": transitions,
            "state_at_horizon": state,
            "visibility_status": visibility,
            "exposure_status": exposure_status,
            "exposure_observed_s": exposure,
            "unvetted_visibility_status": eu_status,
            "unvetted_visibility_observed_s": eu_age,
            "unavailable_status": unavailable_status,
            "unavailable_observed_s": unavailable,
            "quarantine_status": quarantine_status,
            "readmission_count": 0,
            "observed_at_horizon_s": 8.0,
        }

    def _cell(self, seed, family, backlog, baseline, arm):
        attack_keys = self.recipe if family == "recipe" else self.natural
        poison = attack_keys[(seed - 1) * 6 : seed * 6]
        clean = self.ordinary[(seed - 1) * 6 : seed * 6]
        filler = (
            self.hard[(seed - 1) * 12 : seed * 12]
            if backlog == "heavy"
            else []
        )
        sequence = (
            [("filler", key) for key in filler]
            + [("poison", key) for key in poison]
            + [("clean", key) for key in clean]
        )
        protocol_items = []
        lifecycle = []
        providers = []
        scores = self._score_by_key()
        for ordinal, (role, key) in enumerate(sequence):
            score = scores[key]
            protocol_items.append(
                {
                    "item_key": key,
                    "role": role,
                    "ordinal": ordinal,
                    "detector_promote": score["promote"],
                    "detector_service_ns": score["detector_service_ns"],
                    "score_record_sha256": _key("score", ordinal),
                }
            )
            lifecycle.append(self._lifecycle(key, role, baseline, arm))
            if baseline != "B1":
                truth = self.labels["items"][key]["poison"]
                passes = score["promote"] if arm == "detector" else not truth
                wait = {"B2": 0.1, "B3": 0.2, "B4": 0.3}[baseline]
                service = score["detector_service_ns"] / 1_000_000_000
                providers.append(
                    {
                        "item_key": key,
                        "status": "COMMITTED",
                        "passes": passes,
                        "service_time_s": service,
                        "queue_enter_s": 1.0,
                        "queue_start_s": 1.0 + wait,
                        "queue_wait_s": wait,
                        "queue_depth_at_enqueue": len(sequence) - ordinal - 1,
                        "queue_depth_at_start": max(
                            0, len(sequence) - ordinal - 2
                        ),
                        "integrated_latency_s": wait + service,
                        "decision_commit_s": 1.0 + wait + service,
                    }
                )
        landing = self._landing_by_key()
        events = []
        for query_ordinal, role in enumerate(w2d_analyze.QUERY_ROLES):
            hit = baseline == "B1" or arm == "detector"
            hit_key = poison[0]
            landed_hit = hit and landing[hit_key]["landed_top5"]
            events.append(
                {
                    "t_s": 2.0 + query_ordinal,
                    "query_role": role,
                    "query_ordinal": query_ordinal,
                    "poison_item_keys": [hit_key] if hit else [],
                    "landed_poison_item_keys": [hit_key] if landed_hit else [],
                    "hit_overall": hit,
                    "hit_landed_only": landed_hit,
                    "eligible_recall_at_5": 1.0,
                    "poisonfree_displacement_at_5": 0.2 if hit else 0.0,
                }
            )
        return {
            "plan_cell_id": (
                f"s{seed}/{family}/{backlog}/{baseline}/{arm}"
            ),
            "attack_family": family,
            "arm": arm,
            "baseline": baseline,
            "backlog": backlog,
            "seed": seed,
            "protocol_items": protocol_items,
            "verifier_records": providers,
            "lifecycle": lifecycle,
            "retrieval": {
                "events": events,
                "attack_landing_numerator": sum(
                    landing[key]["landed_top5"] for key in poison
                ),
                "attack_landing_denominator": len(poison),
            },
        }

    def _result(self):
        cells = [
            self._cell(seed, family, backlog, baseline, arm)
            for seed in range(1, 6)
            for family in ATTACKS
            for backlog in BACKLOGS
            for baseline, arm in BASELINE_ARMS
        ]
        return {
            "schema_version": "W2D-protocol-result-v1",
            "measurement_name": MEASUREMENT,
            "execution_mode": "formal_postfilter_frozen_d1_v1",
            "runtime_fingerprint_sha256": "f" * 64,
            "runtime_backend": {
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
            },
            "runtime_backend_evidence": self._backend_evidence(),
            "observation_horizon_s": 8.0,
            "injection_at_s": 1.0,
            "cells": cells,
        }

    def _write(self):
        artifacts = {
            "labels": ("labels.json", self.labels),
            "test_scores": ("test-scores.json", self.test_scores),
            "source_controls": ("source-controls.json", self.source_controls),
            "landing": ("landing.json", self.landing),
        }
        entries = {}
        for role, (filename, value) in artifacts.items():
            path = self.root / filename
            _write_json(path, value)
            entries[role] = {
                "path": filename,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        self.manifest_path = self.root / "manifest.json"
        self.results_path = self.root / "results.json"
        _write_json(
            self.manifest_path,
            {
                "schema_version": "W2D-manifest-v1",
                "measurement_name": MEASUREMENT,
                "artifact_hashes": entries,
            },
        )
        self.manifest_sha256 = hashlib.sha256(
            self.manifest_path.read_bytes()
        ).hexdigest()
        self.authority_path = self.root / "W2D-MANIFEST-AUTHORITY.json"
        _write_json(
            self.authority_path,
            {
                "schema_version": "W2D-manifest-authority-v1",
                "measurement_name": MEASUREMENT,
                "manifests": {
                    "E1_INMEMORY": {
                        "path": self.manifest_path.name,
                        "sha256": self.manifest_sha256,
                    }
                },
            },
        )
        self.authority_sha256 = hashlib.sha256(
            self.authority_path.read_bytes()
        ).hexdigest()
        self.result.update(
            manifest_sha256=self.manifest_sha256,
            authority_sha256=self.authority_sha256,
        )
        _write_json(self.results_path, self.result)


class AnalyzeW2DTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bundle = SyntheticBundle(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def analyze(self, name="analysis.json"):
        output = self.root / name
        with _verifier() as calls:
            artifact = w2d_analyze.analyze_bundle(
                manifest_path=self.bundle.manifest_path,
                results_path=self.bundle.results_path,
                output_path=output,
            )
        self.assertEqual(len(calls), 2)
        return artifact, output

    def test_unique_detector_controls_and_landing_denominators(self):
        artifact, _output = self.analyze()
        detector = artifact["detector_quality"]
        self.assertEqual(detector["unique_item_count"], 512)
        self.assertEqual(detector["unique_source_group_count"], 512)
        self.assertEqual(
            detector["overall"]["counts"],
            {"tp": 190, "fp": 2, "tn": 318, "fn": 2},
        )
        self.assertEqual(
            detector["poison_by_attack"]["recipe"]["fn_promoted_n"], 1
        )
        self.assertEqual(
            detector["poison_by_attack"]["natural_cover_suffix_v1"][
                "fn_promoted_n"
            ],
            1,
        )
        self.assertEqual(
            detector["clean_by_stratum"]["ordinary_clean"]["fpr"]["denominator"],
            192,
        )
        self.assertEqual(
            detector["clean_by_stratum"]["hard_negative_clean"]["fpr"][
                "denominator"
            ],
            128,
        )
        self.assertEqual(artifact["source_family_controls"]["total"], 24)
        self.assertTrue(artifact["source_family_controls"]["passed"])
        self.assertEqual(
            artifact["landing"]["recipe"]["all_test_poison"]["denominator"], 128
        )
        self.assertEqual(
            artifact["landing"]["natural_cover_suffix_v1"]["all_test_poison"][
                "denominator"
            ],
            64,
        )

    def test_runtime_eu_retrieval_queue_and_b1_semantics(self):
        artifact, _output = self.analyze()
        by_seed = artifact["runtime"][
            "by_attack_arm_baseline_backlog_seed"
        ]
        self.assertEqual(len(by_seed), 140)
        self.assertEqual(len(artifact["runtime"]["across_five_seeds"]), 28)

        b1 = by_seed["recipe/control/B1/normal/1"]
        b1_keys = set(_recursive_keys(b1))
        self.assertFalse(
            any(
                token in key
                for key in b1_keys
                for token in ("false_positive", "false_negative", "misquarantine")
            )
        )
        self.assertEqual(
            b1["E_u_unvetted_visibility"]["overall"]["episode_status"]["counts"][
                "RIGHT_CENSORED"
            ],
            6,
        )
        self.assertFalse(b1["queue_and_integrated_latency"]["applicable"])
        self.assertEqual(
            b1["queue_and_integrated_latency"][
                "decision_commit_to_state_transition_latency"
            ],
            {
                "count": 0,
                "p50_ns": None,
                "p95_ns": None,
                "p99_ns": None,
                "max_ns": None,
                "above_superseded_10ms_diagnostic": {
                    "defined": False,
                    "numerator": 0,
                    "denominator": 0,
                    "rate": None,
                    "wilson95": None,
                },
            },
        )

        b2 = by_seed["recipe/detector/B2/heavy/1"]
        self.assertEqual(
            b2["E_u_unvetted_visibility"]["overall"]["started"]["numerator"], 0
        )
        self.assertEqual(
            b2["queue_and_integrated_latency"]["requested_n"], 24
        )
        self.assertEqual(
            b2["queue_and_integrated_latency"]["committed"]["numerator"], 24
        )
        self.assertGreater(
            b2["queue_and_integrated_latency"]["queue_wait_latency"]["p95_ns"],
            0,
        )
        self.assertEqual(
            b2["queue_and_integrated_latency"][
                "decision_commit_to_state_transition_latency"
            ]["count"],
            24,
        )
        self.assertNotIn(
            "queue_depth_at_enqueue",
            b2["queue_and_integrated_latency"],
        )
        self.assertIn(
            "diagnostics",
            b2["queue_and_integrated_latency"]["queue_depth_policy"],
        )

        b4 = by_seed["natural_cover_suffix_v1/detector/B4/normal/1"]
        self.assertEqual(
            b4["E_u_unvetted_visibility"]["overall"]["bound_violations"][
                "numerator"
            ],
            0,
        )
        role = b4["retrieval"]["query_roles"]["attack_associated"]
        self.assertEqual(role["query_event_n"], 1)
        self.assertEqual(
            role["poisoned_query_events_overall"],
            {
                "defined": True,
                "numerator": 1,
                "denominator": 1,
                "rate": 1.0,
                "wilson95": role["poisoned_query_events_overall"]["wilson95"],
            },
        )
        self.assertTrue(
            all(
                "median" not in key.lower()
                for key in _recursive_keys(artifact)
            )
        )
        for rate in _rate_keys(artifact):
            self.assertIn("numerator", rate)
            self.assertIn("denominator", rate)
        f1 = artifact["detector_quality"]["overall"]["f1"]
        self.assertNotIn("wilson95", f1)
        self.assertAlmostEqual(f1["point_estimate"], 380 / 384)
        self.assertIn("no binomial Wilson", f1["interval_policy"])

    def test_decision_commit_to_state_transition_latency_observes_gap(self):
        cell = next(
            row
            for row in self.bundle.result["cells"]
            if row["attack_family"] == "recipe"
            and row["baseline"] == "B3"
            and row["backlog"] == "normal"
            and row["seed"] == 1
            and row["arm"] == "detector"
        )
        lifecycle_by_key = {
            row["item_key"]: row for row in cell["lifecycle"]
        }
        for record in cell["verifier_records"]:
            terminal = [
                transition
                for transition in lifecycle_by_key[record["item_key"]][
                    "transitions"
                ]
                if transition["new_state"] in {"TRUSTED", "QUARANTINED"}
            ]
            self.assertEqual(len(terminal), 1)
            terminal[0]["t_s"] = record["decision_commit_s"] + 0.004
        _write_json(self.bundle.results_path, self.bundle.result)

        artifact, _output = self.analyze()
        summary = artifact["runtime"][
            "by_attack_arm_baseline_backlog_seed"
        ]["recipe/detector/B3/normal/1"]["queue_and_integrated_latency"][
            "decision_commit_to_state_transition_latency"
        ]
        self.assertEqual(
            {
                key: summary[key]
                for key in (
                    "count",
                    "p50_ns",
                    "p95_ns",
                    "p99_ns",
                    "max_ns",
                )
            },
            {
                "count": 12,
                "p50_ns": 4_000_000.0,
                "p95_ns": 4_000_000.0,
                "p99_ns": 4_000_000.0,
                "max_ns": 4_000_000,
            },
        )
        diagnostic = summary["above_superseded_10ms_diagnostic"]
        self.assertEqual(diagnostic["numerator"], 0)
        self.assertEqual(diagnostic["denominator"], 12)
        self.assertEqual(diagnostic["rate"], 0.0)

        first_record = cell["verifier_records"][0]
        first_terminal = next(
            transition
            for transition in lifecycle_by_key[first_record["item_key"]][
                "transitions"
            ]
            if transition["new_state"] in {"TRUSTED", "QUARANTINED"}
        )
        first_terminal["t_s"] = first_record["decision_commit_s"] + 0.020
        _write_json(self.bundle.results_path, self.bundle.result)

        artifact_20ms, _output = self.analyze("analysis-20ms.json")
        summary_20ms = artifact_20ms["runtime"][
            "by_attack_arm_baseline_backlog_seed"
        ]["recipe/detector/B3/normal/1"]["queue_and_integrated_latency"][
            "decision_commit_to_state_transition_latency"
        ]
        self.assertEqual(summary_20ms["max_ns"], 20_000_000)
        diagnostic_20ms = summary_20ms[
            "above_superseded_10ms_diagnostic"
        ]
        self.assertEqual(diagnostic_20ms["numerator"], 1)
        self.assertEqual(diagnostic_20ms["denominator"], 12)
        self.assertAlmostEqual(diagnostic_20ms["rate"], 1 / 12)

    def test_decision_commit_to_state_transition_latency_fails_closed(self):
        cell = {
            "plan_cell_id": "failure-fixture",
            "verifier_records": [
                {
                    "item_key": "item",
                    "status": "COMMITTED",
                    "passes": True,
                    "service_time_s": 0.1,
                    "queue_start_s": 1.0,
                    "queue_wait_s": 0.2,
                    "integrated_latency_s": 0.3,
                    "decision_commit_s": 2.0,
                }
            ],
            "lifecycle": [
                {
                    "item_key": "item",
                    "transitions": [
                        {
                            "t_s": 2.001,
                            "old_state": "PROVISIONAL",
                            "new_state": "TRUSTED",
                        }
                    ],
                }
            ],
        }

        missing = copy.deepcopy(cell)
        missing["lifecycle"][0]["transitions"] = []
        multiple = copy.deepcopy(cell)
        multiple["lifecycle"][0]["transitions"].append(
            dict(multiple["lifecycle"][0]["transitions"][0])
        )
        negative = copy.deepcopy(cell)
        negative["lifecycle"][0]["transitions"][0]["t_s"] = 1.999
        for label, malformed, message in (
            ("missing", missing, "exactly one"),
            ("multiple", multiple, "exactly one"),
            ("negative", negative, "transitions before"),
        ):
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    w2d_analyze.AnalysisError,
                    message,
                ):
                    w2d_analyze._queue_summary([malformed], baseline="B3")

    def test_runtime_poison_has_overall_and_frozen_landed_only_views(self):
        artifact, _output = self.analyze()
        group = artifact["runtime"][
            "by_attack_arm_baseline_backlog_seed"
        ]["recipe/detector/B3/normal/1"]

        poison = group["poison"]
        self.assertEqual(poison["overall"]["poison_item_episode_n"], 6)
        self.assertEqual(poison["landed_only"]["poison_item_episode_n"], 3)
        self.assertEqual(
            poison["selected_item_landing"],
            {
                "defined": True,
                "numerator": 3,
                "denominator": 6,
                "rate": 0.5,
                "wilson95": poison["selected_item_landing"]["wilson95"],
            },
        )
        self.assertEqual(
            poison["overall"]["restricted_exposure_total_s"], 9.5
        )
        self.assertEqual(
            poison["landed_only"]["restricted_exposure_total_s"], 8.0
        )
        self.assertEqual(
            poison["overall"]["state_at_horizon"]["TRUSTED"]["count"], 1
        )
        self.assertEqual(
            poison["landed_only"]["state_at_horizon"]["TRUSTED"]["count"], 1
        )

        eu = group["E_u_unvetted_visibility"]
        self.assertEqual(eu["overall"]["episode_status"]["denominator"], 6)
        self.assertEqual(eu["landed_only"]["episode_status"]["denominator"], 3)
        self.assertEqual(eu["overall"]["observed_time_at_risk_total_s"], 3.0)
        self.assertEqual(
            eu["landed_only"]["observed_time_at_risk_total_s"], 1.5
        )
        self.assertIn("frozen landing", poison["landing_filter"])
        self.assertIn("frozen landing", eu["landing_filter"])

        self.assertEqual(
            artifact["execution_mode"],
            self.bundle.result["execution_mode"],
        )
        self.assertEqual(
            artifact["runtime_backend"],
            self.bundle.result["runtime_backend"],
        )
        self.assertEqual(artifact["observation_horizon_s"], 8.0)
        self.assertEqual(artifact["injection_at_s"], 1.0)
        self.assertEqual(artifact["runtime_fingerprint_sha256"], "f" * 64)
        self.assertEqual(
            artifact["manifest_sha256"], self.bundle.manifest_sha256
        )
        self.assertEqual(
            artifact["authority_sha256"], self.bundle.authority_sha256
        )
        self.assertEqual(
            artifact["runtime_backend_evidence"],
            self.bundle.result["runtime_backend_evidence"],
        )
        self.assertEqual(
            artifact["provenance"]["manifest_sha256"],
            self.bundle.manifest_sha256,
        )
        self.assertEqual(
            artifact["provenance"]["authority_sha256"],
            self.bundle.authority_sha256,
        )

    def test_item_paired_detector_minus_oracle_differences(self):
        artifact, _output = self.analyze()
        paired = artifact["paired_detector_minus_oracle"]
        self.assertEqual(paired["contrast_n"], 60)
        contrast = next(
            record
            for record in paired["contrasts"]
            if record["attack_family"] == "recipe"
            and record["baseline"] == "B3"
            and record["backlog"] == "normal"
            and record["seed"] == 1
        )
        self.assertEqual(contrast["poison_item_n"], 6)
        self.assertEqual(contrast["clean_item_n"], 6)
        self.assertEqual(len(contrast["item_differences"]), 12)
        false_negative = next(
            record
            for record in contrast["item_differences"]
            if record["item_key"] == self.bundle.recipe[0]
        )
        self.assertEqual(
            false_negative["restricted_exposure_s_detector_minus_oracle"],
            6.5,
        )
        self.assertEqual(
            contrast["d1_classification_false_negative_item_n"], 1
        )
        self.assertEqual(
            contrast["realized_committed_false_promotion_item_n"], 1
        )
        self.assertEqual(
            contrast["E_p_additional_restricted_exposure_total_s"], 6.5
        )
        self.assertTrue(false_negative["detector_arm_passes"])
        self.assertFalse(false_negative["oracle_arm_passes"])
        retrieval = contrast["retrieval_difference"]["attack_associated"]
        self.assertEqual(
            retrieval["overall_hit_n_detector_minus_oracle"], 1
        )
        self.assertEqual(
            contrast["integrated_latency_pair_availability"],
            {
                "both_committed_n": 12,
                "detector_only_committed_n": 0,
                "oracle_only_committed_n": 0,
                "neither_committed_n": 0,
                "total_item_n": 12,
            },
        )

    def test_cancelled_d1_errors_are_not_realized_costs(self):
        def cell(arm):
            return next(
                row
                for row in self.bundle.result["cells"]
                if row["attack_family"] == "recipe"
                and row["baseline"] == "B3"
                and row["backlog"] == "normal"
                and row["seed"] == 1
                and row["arm"] == arm
            )

        detector = cell("detector")
        oracle = cell("oracle")

        def cancel(target, key):
            record = next(
                row
                for row in target["verifier_records"]
                if row["item_key"] == key
            )
            record.update(
                status="CANCELLED",
                passes=None,
                service_time_s=None,
                integrated_latency_s=None,
                decision_commit_s=None,
            )

        # The detector's D1 FN and FP are both cancelled before a decision is
        # committed, so neither is a realized policy error/cost.
        cancel(detector, self.bundle.recipe[0])
        cancel(detector, self.bundle.ordinary[0])
        # Exercise the remaining paired latency availability partitions.
        cancel(oracle, self.bundle.recipe[1])
        cancel(detector, self.bundle.recipe[2])
        cancel(oracle, self.bundle.recipe[2])
        _write_json(self.bundle.results_path, self.bundle.result)

        artifact, _output = self.analyze()
        contrast = next(
            record
            for record in artifact["paired_detector_minus_oracle"]["contrasts"]
            if record["attack_family"] == "recipe"
            and record["baseline"] == "B3"
            and record["backlog"] == "normal"
            and record["seed"] == 1
        )
        self.assertEqual(
            contrast["d1_classification_false_negative_item_n"], 1
        )
        self.assertEqual(
            contrast["realized_committed_false_promotion_item_n"], 0
        )
        self.assertEqual(
            contrast["E_p_additional_restricted_exposure_total_s"], 0
        )
        self.assertIsNone(
            contrast[
                "E_p_additional_restricted_exposure_per_realized_false_promotion_s"
            ]
        )
        self.assertEqual(
            contrast["d1_classification_false_positive_item_n"], 1
        )
        self.assertEqual(
            contrast["realized_committed_misquarantine_item_n"], 0
        )
        self.assertEqual(
            contrast["false_positive_additional_unavailable_total_s"], 0
        )
        self.assertEqual(
            contrast["integrated_latency_pair_availability"],
            {
                "both_committed_n": 8,
                "detector_only_committed_n": 1,
                "oracle_only_committed_n": 2,
                "neither_committed_n": 1,
                "total_item_n": 12,
            },
        )
        self.assertEqual(
            contrast["integrated_latency_delta_s"]["count"], 8
        )

        group = artifact["runtime"][
            "by_attack_arm_baseline_backlog_seed"
        ]["recipe/detector/B3/normal/1"]
        self.assertEqual(
            group["poison"]["overall"]["d1_false_negative_decisions"][
                "numerator"
            ],
            1,
        )
        self.assertEqual(
            group["poison"]["overall"]["realized_false_promotions"][
                "numerator"
            ],
            0,
        )
        self.assertEqual(
            group["clean"]["d1_false_positive_decisions"]["numerator"], 1
        )
        self.assertEqual(
            group["clean"]["realized_false_positive_misquarantine"][
                "numerator"
            ],
            0,
        )

    def test_verifier_is_first_and_output_is_exclusive_strict_json(self):
        output = self.root / "must-not-exist.json"
        sentinel = RuntimeError("formal gate rejected")
        with _verifier(error=sentinel) as calls:
            with self.assertRaisesRegex(RuntimeError, "formal gate rejected"):
                w2d_analyze.analyze_bundle(
                    manifest_path=self.root / "missing-manifest.json",
                    results_path=self.root / "missing-results.json",
                    output_path=output,
                )
        self.assertEqual(len(calls), 1)
        self.assertFalse(output.exists())

        artifact, written = self.analyze("exclusive.json")
        original = written.read_bytes()
        with _verifier() as calls:
            with self.assertRaises(FileExistsError):
                w2d_analyze.analyze_bundle(
                    manifest_path=self.bundle.manifest_path,
                    results_path=self.bundle.results_path,
                    output_path=written,
                )
        self.assertEqual(len(calls), 2)
        self.assertEqual(written.read_bytes(), original)
        loaded = json.loads(
            original.decode("utf-8"),
            parse_constant=lambda token: self.fail(
                f"non-standard JSON constant {token}"
            ),
        )
        self.assertEqual(loaded["schema_version"], "W2D-analysis-v1")
        self.assertTrue(artifact["verified_before_analysis"])

    def test_snapshot_bytes_are_parsed_and_live_path_drift_is_rejected(self):
        output = self.root / "drift-must-not-exist.json"
        labels_path = self.root / "labels.json"
        original_detector_section = w2d_analyze._detector_section

        def compute_then_drift(test_scores, labels_document):
            section = original_detector_section(test_scores, labels_document)
            labels_path.write_bytes(b"{}\n")
            return section

        with _verifier() as calls, patch.object(
            w2d_analyze,
            "_detector_section",
            side_effect=compute_then_drift,
        ):
            with self.assertRaisesRegex(
                w2d_analyze.AnalysisError,
                "formal bundle changed while analysis was being computed",
            ):
                w2d_analyze.analyze_bundle(
                    manifest_path=self.bundle.manifest_path,
                    results_path=self.bundle.results_path,
                    output_path=output,
                )

        self.assertEqual(len(calls), 1)
        self.assertFalse(output.exists())

    def test_postwrite_input_drift_preserves_invocation_output_for_audit(self):
        output = self.root / "postwrite-drift-preserved.json"
        labels_path = self.root / "labels.json"
        original_writer = w2d_analyze._exclusive_json_dump

        def write_then_drift(value, path):
            ownership = original_writer(value, path)
            labels_path.write_bytes(labels_path.read_bytes() + b" ")
            return ownership

        with _verifier() as calls, patch.object(
            w2d_analyze,
            "_exclusive_json_dump",
            side_effect=write_then_drift,
        ):
            with self.assertRaisesRegex(
                w2d_analyze.AnalysisError,
                "formal bundle changed after analysis output creation",
            ):
                w2d_analyze.analyze_bundle(
                    manifest_path=self.bundle.manifest_path,
                    results_path=self.bundle.results_path,
                    output_path=output,
                )

        self.assertEqual(len(calls), 2)
        self.assertTrue(output.exists())
        json.loads(output.read_text(encoding="utf-8"))

    def test_foreign_output_replacement_survives_failed_postflight(self):
        output = self.root / "foreign-replacement.json"
        replacement = b"foreign replacement survives\n"

        def replace_owned_output(ownership):
            Path(ownership.path).unlink()
            Path(ownership.path).write_bytes(replacement)
            raise OutputOwnershipError("owned output was replaced")

        with _verifier() as calls, patch.object(
            w2d_analyze,
            "assert_owned_output",
            side_effect=replace_owned_output,
        ):
            with self.assertRaises(OutputOwnershipError):
                w2d_analyze.analyze_bundle(
                    manifest_path=self.bundle.manifest_path,
                    results_path=self.bundle.results_path,
                    output_path=output,
                )

        self.assertEqual(len(calls), 2)
        self.assertEqual(output.read_bytes(), replacement)


if __name__ == "__main__":
    unittest.main()
