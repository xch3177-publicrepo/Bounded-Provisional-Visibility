"""Unit tests for the downstream W2D statistical analyzer.

The fixture is a complete-cardinality synthetic result chain.  The formal
verifier is replaced only at the call boundary: these tests exercise analysis
semantics without importing the numpy-backed detector stack.  Production code
still imports and calls ``verify_w2d.verify_or_raise`` before reading a byte.

    python test_w2d_analyze.py
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import w2d_analyze


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
        return result

    module.verify_or_raise = verify_or_raise
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

    def _lifecycle(self, key, role, baseline, arm):
        label = self.labels["items"][key]
        truth = label["poison"]
        detector_promote = self._score_by_key()[key]["promote"]
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
        return {
            "item_key": key,
            "truth_poison": truth,
            "detector_promote": detector_promote,
            "retry_count": 0,
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
            b1["E_u_unvetted_visibility"]["episode_status"]["counts"][
                "RIGHT_CENSORED"
            ],
            6,
        )
        self.assertFalse(b1["queue_and_integrated_latency"]["applicable"])

        b2 = by_seed["recipe/detector/B2/heavy/1"]
        self.assertEqual(
            b2["E_u_unvetted_visibility"]["started"]["numerator"], 0
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

        b4 = by_seed["natural_cover_suffix_v1/detector/B4/normal/1"]
        self.assertEqual(
            b4["E_u_unvetted_visibility"]["bound_violations"]["numerator"], 0
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
        self.assertEqual(contrast["E_p_false_negative_item_n"], 1)
        self.assertEqual(
            contrast["E_p_additional_restricted_exposure_total_s"], 6.5
        )
        self.assertTrue(false_negative["detector_arm_passes"])
        self.assertFalse(false_negative["oracle_arm_passes"])
        retrieval = contrast["retrieval_difference"]["attack_associated"]
        self.assertEqual(
            retrieval["overall_hit_n_detector_minus_oracle"], 1
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


if __name__ == "__main__":
    unittest.main()
