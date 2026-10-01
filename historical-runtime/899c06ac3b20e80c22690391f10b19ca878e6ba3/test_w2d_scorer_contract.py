"""Negative guards for the formal W2D scorer/calibrator/metrics contract."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
import unittest
from unittest import mock

from detector import D1Detector
from test_w2d_detector import DIM, REVISION, ScorerIntegrationTest, opaque
import w2d_calibrate
import w2d_metrics
import w2d_scorer


class ScorerContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (
            self.embedder,
            self.inputs_path,
            self.embeddings_path,
            self.calibration_keys,
            self.test_keys,
        ) = ScorerIntegrationTest()._bundle(self.root)
        self.detector = D1Detector(
            embedder=self.embedder,
            pinned_model_revision=REVISION,
            expected_dim=DIM,
            k_neighbors=3,
        )

    def tearDown(self):
        self.temp.cleanup()

    def _write_variant(self, name, mutate):
        value = w2d_metrics.strict_json_load(str(self.inputs_path))
        mutate(value)
        path = self.root / name
        w2d_metrics.strict_json_dump(value, str(path))
        return path

    def _calibration(self):
        return w2d_scorer.run_scorer_test_fixture(
            inputs_path=str(self.inputs_path),
            embeddings_path=str(self.embeddings_path),
            keys_path=None,
            phase="calibration",
            detector=self.detector,
        )

    def _labels(self):
        return {
            "items": [
                {
                    "item_key": self.calibration_keys[0],
                    "source_group": opaque("cal-group-0"),
                    "split": "calibration",
                    "stratum": "ordinary_clean",
                    "poison": False,
                    "attack_family": None,
                },
                {
                    "item_key": self.calibration_keys[1],
                    "source_group": opaque("cal-group-1"),
                    "split": "calibration",
                    "stratum": "recipe_poison",
                    "poison": True,
                    "attack_family": "recipe",
                },
                {
                    "item_key": self.test_keys[0],
                    "source_group": opaque("test-group-0"),
                    "split": "test",
                    "stratum": "ordinary_clean",
                    "poison": False,
                    "attack_family": None,
                },
                {
                    "item_key": self.test_keys[1],
                    "source_group": opaque("test-group-1"),
                    "split": "test",
                    "stratum": "recipe_poison",
                    "poison": True,
                    "attack_family": "recipe",
                },
            ]
        }

    def _test_scores(self):
        calibration = self._calibration()
        threshold = w2d_calibrate.select_threshold_test_fixture(
            calibration, self._labels()
        )
        threshold_path = self.root / "threshold.json"
        w2d_metrics.strict_json_dump(threshold, str(threshold_path))
        return w2d_scorer.run_scorer_test_fixture(
            inputs_path=str(self.inputs_path),
            embeddings_path=str(self.embeddings_path),
            keys_path=None,
            phase="test",
            threshold_artifact_path=str(threshold_path),
            detector=self.detector,
        )

    def test_forbidden_safe_field_is_rejected(self):
        bad = self._write_variant(
            "labels-leaked.json",
            lambda value: value["items"][0].__setitem__("poison", True),
        )
        with self.assertRaises(w2d_scorer.ScorerError):
            w2d_scorer.run_scorer_test_fixture(
                inputs_path=str(bad),
                embeddings_path=str(self.embeddings_path),
                keys_path=None,
                phase="calibration",
                detector=self.detector,
            )

    def test_external_manifest_must_exactly_equal_frozen_phase(self):
        manifest = self.root / "cross-phase.json"
        w2d_metrics.strict_json_dump(
            {"calibration": [self.test_keys[0]]}, str(manifest)
        )
        with self.assertRaises(w2d_scorer.ScorerError):
            w2d_scorer.run_scorer_test_fixture(
                inputs_path=str(self.inputs_path),
                embeddings_path=str(self.embeddings_path),
                keys_path=str(manifest),
                phase="calibration",
                detector=self.detector,
            )

    def test_npz_must_match_safe_descriptor_hash(self):
        import numpy as np

        changed = self.root / "changed.npz"
        with np.load(self.embeddings_path, allow_pickle=False) as archive:
            vectors = archive["embeddings"].copy()
        vectors[0, 0] += 0.01
        np.savez_compressed(changed, embeddings=vectors)
        with self.assertRaises(w2d_scorer.ScorerError):
            w2d_scorer.run_scorer_test_fixture(
                inputs_path=str(self.inputs_path),
                embeddings_path=str(changed),
                keys_path=None,
                phase="calibration",
                detector=self.detector,
            )

    def test_final_score_schema_and_code_provenance_are_exact(self):
        scores = self._test_scores()
        self.assertTrue(
            all(set(record) == w2d_metrics.FINAL_SCORE_KEYS
                for record in scores["items"])
        )
        for field in (
            "safe_inputs_sha256",
            "embeddings_sha256",
            "detector_py_sha256",
            "scorer_py_sha256",
            "threshold_artifact_sha256",
        ):
            self.assertRegex(scores[field], r"^[0-9a-f]{64}$")

    def test_source_control_is_a_distinct_scorer_phase(self):
        control_key = self.test_keys[0]

        def move_to_controls(value):
            value["sets"]["test_score_order"].remove(control_key)
            value["sets"]["source_control_item_keys"] = [control_key]
            by_key = {item["item_key"]: item for item in value["items"]}
            by_key[control_key]["source_evidence"][0] = False

        control_inputs = self._write_variant("controls.json", move_to_controls)
        calibration = self._calibration()
        threshold = w2d_calibrate.select_threshold_test_fixture(
            calibration, self._labels()
        )
        threshold_path = self.root / "control-threshold.json"
        w2d_metrics.strict_json_dump(threshold, str(threshold_path))
        scores = w2d_scorer.run_scorer_test_fixture(
            inputs_path=str(control_inputs),
            embeddings_path=str(self.embeddings_path),
            keys_path=None,
            phase="source_control",
            threshold_artifact_path=str(threshold_path),
            detector=self.detector,
        )
        self.assertEqual(scores["phase"], "source_control")
        self.assertEqual(len(scores["items"]), 1)
        self.assertFalse(scores["items"][0]["family_s_affirms"])
        self.assertFalse(scores["items"][0]["promote"])

    def test_service_timer_excludes_harness_text_lookup(self):
        original = w2d_scorer._text

        def slow_lookup(item):
            time.sleep(0.02)
            return original(item)

        with mock.patch("w2d_scorer._text", side_effect=slow_lookup):
            scores = self._test_scores()
        self.assertLess(
            max(record["detector_service_ns"] for record in scores["items"]),
            10_000_000,
        )

    def test_scorer_rejects_input_or_npz_drift_before_writing(self):
        original_raw_record = w2d_scorer._raw_record
        for name, path in (
            ("safe", self.inputs_path),
            ("embeddings", self.embeddings_path),
        ):
            with self.subTest(name=name):
                original_bytes = path.read_bytes()
                output = self.root / f"{name}-drift-output.json"
                changed = False

                def drift_once(*args, **kwargs):
                    nonlocal changed
                    if not changed:
                        path.write_bytes(original_bytes + b" ")
                        changed = True
                    return original_raw_record(*args, **kwargs)

                try:
                    with mock.patch(
                        "w2d_scorer._raw_record", side_effect=drift_once
                    ):
                        with self.assertRaisesRegex(
                            w2d_scorer.ScorerError,
                            "changed during execution",
                        ):
                            w2d_scorer.run_scorer_test_fixture(
                                inputs_path=str(self.inputs_path),
                                embeddings_path=str(self.embeddings_path),
                                keys_path=None,
                                phase="calibration",
                                output_path=str(output),
                                detector=self.detector,
                            )
                    self.assertFalse(output.exists())
                finally:
                    path.write_bytes(original_bytes)

    def test_formal_scorer_rejects_dirty_tree_before_reading_inputs(self):
        with (
            mock.patch(
                "w2d_scorer.subprocess.check_output",
                return_value=" M detector.py\n",
            ),
            mock.patch("w2d_scorer._run_scorer") as run,
        ):
            with self.assertRaisesRegex(
                w2d_scorer.ScorerError, "clean committed worktree"
            ):
                w2d_scorer.run_scorer(
                    inputs_path=str(self.inputs_path),
                    embeddings_path=str(self.embeddings_path),
                    keys_path=None,
                    phase="calibration",
                    detector=self.detector,
                )
            run.assert_not_called()

    def test_formal_denominators_reject_small_fixture(self):
        calibration = self._calibration()
        labels = self._labels()
        with self.assertRaises(w2d_calibrate.CalibrationError):
            w2d_calibrate.select_threshold(calibration, labels)
        test_scores = self._test_scores()
        with self.assertRaises(w2d_metrics.MetricsError):
            w2d_metrics.detector_quality_summary(test_scores, labels)
        self.assertEqual(
            w2d_metrics.detector_quality_summary_test_fixture(
                test_scores, labels
            )["unique_item_count"],
            2,
        )

    def test_strict_writer_is_exclusive(self):
        path = self.root / "one-shot.json"
        w2d_metrics.strict_json_dump({"value": 1}, str(path))
        with self.assertRaises(FileExistsError):
            w2d_metrics.strict_json_dump({"value": 2}, str(path))
        self.assertEqual(
            w2d_metrics.strict_json_load(str(path)), {"value": 1}
        )


class SourceControlGateTest(unittest.TestCase):
    def test_controls_have_a_separate_non_quality_gate(self):
        hashes = {
            name: ("a" if index % 2 == 0 else "b") * 64
            for index, name in enumerate(
                (
                    "safe_inputs_sha256",
                    "embeddings_sha256",
                    "detector_py_sha256",
                    "scorer_py_sha256",
                    "threshold_artifact_sha256",
                )
            )
        }
        kinds = (
            ["invalid_signature"] * 8
            + ["provenance_conflict"] * 8
            + ["unknown_source"] * 8
        )
        keys = [opaque(f"control-{index}") for index in range(24)]
        scores = {
            "schema_version": "W2D-score-v1",
            "phase": "source_control",
            **hashes,
            "items": [
                {
                    "item_key": key,
                    "family_s_affirms": False,
                    "c_score": 0.0,
                    "promote": False,
                    "detector_service_ns": 1,
                }
                for key in keys
            ],
        }
        labels = {
            "items": {
                key: {
                    "source_group": opaque(f"control-group-{index}"),
                    "split": "implementation_control",
                    "stratum": "source_family_negative_control",
                    "poison": False,
                    "source_control_kind": kinds[index],
                }
                for index, key in enumerate(keys)
            }
        }
        gate = w2d_metrics.source_control_gate_summary(scores, labels)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["family_s_refused_n"], 24)
        self.assertEqual(gate["promoted_n"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
