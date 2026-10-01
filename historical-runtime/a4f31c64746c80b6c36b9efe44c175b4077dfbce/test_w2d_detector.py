"""Offline tests for the W2D detector, scorer, calibration and metrics layers."""

from __future__ import annotations

import ast
import hashlib
import inspect
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

import detector
from detector import D1Detector, DetectorInputError, ZComponent, ZStatistics
import w2d_calibrate
import w2d_metrics
import w2d_scorer


REVISION = "test-model-revision"
DIM = 4


def opaque(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


class FakeEmbedder:
    def __call__(self, texts):
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            row = np.asarray(
                [1.0 + digest[index] / 255.0 for index in range(DIM)],
                dtype=np.float64,
            )
            rows.append(row / np.linalg.norm(row))
        return np.asarray(rows)


def source_evidence(*, credential=True, provenance=True, revision=REVISION):
    return [credential, provenance, revision]


class DetectorUnitTest(unittest.TestCase):
    def setUp(self):
        self.embedder = FakeEmbedder()
        self.detector = D1Detector(
            embedder=self.embedder,
            pinned_model_revision=REVISION,
            expected_dim=DIM,
            k_neighbors=3,
        )
        self.reference_texts = [
            f"reference {index} alpha beta gamma delta epsilon zeta"
            for index in range(12)
        ]
        self.reference_embeddings = self.embedder(self.reference_texts)
        self.reference_groups = [f"group-{index}" for index in range(12)]
        self.z = self.detector.reference_z_statistics(
            self.reference_texts,
            self.reference_embeddings,
            self.reference_groups,
        )

    def test_detector_signature_and_result_have_no_harness_key(self):
        parameters = inspect.signature(D1Detector.score).parameters
        self.assertNotIn("item_key", parameters)
        self.assertNotIn("key", parameters)
        text = "candidate alpha beta gamma delta epsilon zeta eta"
        result = self.detector.score(
            text,
            self.embedder([text])[0],
            source_evidence(),
            self.reference_embeddings,
            self.reference_groups,
            "",
            REVISION,
            self.z,
            100.0,
        )
        self.assertNotIn("item_key", result.to_dict())
        self.assertTrue(result.family_s_affirms)
        self.assertTrue(result.promote)

    def test_source_conjunction_and_content_boundary(self):
        text = "candidate one two three four five six seven eight"
        embedding = self.embedder([text])[0]
        features = self.detector.content_features(
            text,
            embedding,
            self.reference_embeddings,
            self.reference_groups,
            "",
        )
        c_score = self.detector.content_score(features, self.z)
        refused = self.detector.score(
            text,
            embedding,
            source_evidence(),
            self.reference_embeddings,
            self.reference_groups,
            "",
            REVISION,
            self.z,
            c_score,
        )
        self.assertFalse(refused.family_c_affirms)
        self.assertFalse(refused.promote)

        bad_source = self.detector.score(
            text,
            embedding,
            source_evidence(credential=False),
            self.reference_embeddings,
            self.reference_groups,
            "",
            REVISION,
            self.z,
            c_score + 1.0,
        )
        self.assertTrue(bad_source.family_c_affirms)
        self.assertFalse(bad_source.family_s_affirms)
        self.assertFalse(bad_source.promote)

    def test_source_evidence_rejects_truthy_non_booleans(self):
        for evidence in (
            ["false", True, REVISION],
            [True, 1, REVISION],
            {
                "credential_valid": "false",
                "provenance_consistent": True,
                "embedding_model_revision": REVISION,
            },
        ):
            with self.subTest(evidence=evidence):
                with self.assertRaises(DetectorInputError):
                    detector.SourceEvidence.from_value(evidence)

    def test_reference_group_is_excluded(self):
        with self.assertRaises(DetectorInputError):
            self.detector.content_features(
                self.reference_texts[0],
                self.reference_embeddings[0],
                self.reference_embeddings[:3],
                ["same", "same", "same"],
                "same",
            )

    def test_nonfinite_and_empty_half_fail_closed(self):
        with self.assertRaises(DetectorInputError):
            self.detector.content_features(
                "one",
                self.reference_embeddings[0],
                self.reference_embeddings,
                self.reference_groups,
                "",
            )
        bad = self.reference_embeddings[0].copy()
        bad[0] = np.nan
        with self.assertRaises(DetectorInputError):
            self.detector.content_features(
                "one two three four",
                bad,
                self.reference_embeddings,
                self.reference_groups,
                "",
            )

    def test_detector_imports_no_experiment_modules(self):
        tree = ast.parse(Path(detector.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        forbidden = {
            "w2d_workload",
            "realtext_workload",
            "poison_exposure",
            "w2d_calibrate",
            "w2d_metrics",
        }
        self.assertFalse(imported & forbidden)


class CalibrationAndMetricsTest(unittest.TestCase):
    def test_finite_extremes_and_largest_threshold_tie_break(self):
        z = ZStatistics(
            rep=ZComponent(0.0, 1.0), knn=ZComponent(0.0, 1.0)
        )
        keys = [opaque(f"cal-{index}") for index in range(4)]
        scores = {
            "phase": "calibration",
            "z_statistics": z.to_dict(),
            "items": [
                {"item_key": key, "c_score": score}
                for key, score in zip(keys, (0.0, 1.0, 2.0, 3.0))
            ],
        }
        # Sorted by score: suspicious, clean, clean, suspicious.  F1 is 2/3
        # both below the minimum and between 2 and 3; the frozen rule chooses
        # the larger threshold.
        labels = {
            "items": [
                {
                    "item_key": key,
                    "source_group": f"group-{index}",
                    "split": "calibration",
                    "stratum": "recipe_poison" if truth else "ordinary_clean",
                    "is_poison": truth,
                    "attack_family": "recipe" if truth else None,
                }
                for index, (key, truth) in enumerate(
                    zip(keys, (True, False, False, True))
                )
            ]
        }
        scores["schema_version"] = "W2D-score-v1"
        result = w2d_calibrate.select_threshold_test_fixture(scores, labels)
        self.assertEqual(result["selected_threshold"], 2.5)
        self.assertTrue(math.isfinite(result["selected_threshold"]))
        candidates = w2d_calibrate.threshold_candidates([0.0, 1.0, 2.0, 3.0])
        self.assertTrue(all(math.isfinite(value) for value in candidates))

    def test_wilson_and_unique_group_denominator(self):
        interval = w2d_metrics.wilson_interval(0, 10)
        self.assertEqual(interval[0], 0.0)
        self.assertGreater(interval[1], 0.0)
        keys = [opaque("a"), opaque("b")]
        scores = {
            "schema_version": "W2D-score-v1",
            "phase": "test",
            "items": [
                {
                    "item_key": keys[0],
                    "family_s_affirms": True,
                    "c_score": 1.0,
                    "promote": False,
                    "detector_service_ns": 10,
                },
                {
                    "item_key": keys[1],
                    "family_s_affirms": True,
                    "c_score": -1.0,
                    "promote": True,
                    "detector_service_ns": 20,
                },
            ]
        }
        labels = {
            "items": [
                {
                    "item_key": keys[0],
                    "source_group": "g0",
                    "split": "test",
                    "stratum": "recipe_poison",
                    "is_poison": True,
                    "attack_family": "recipe",
                },
                {
                    "item_key": keys[1],
                    "source_group": "g1",
                    "split": "test",
                    "stratum": "ordinary_clean",
                    "is_poison": False,
                    "attack_family": None,
                },
            ]
        }
        summary = w2d_metrics.detector_quality_summary_test_fixture(scores, labels)
        self.assertEqual(summary["overall"]["counts"], {"tp": 1, "fp": 0, "tn": 1, "fn": 0})
        labels["items"][1]["source_group"] = "g0"
        with self.assertRaises(w2d_metrics.MetricsError):
            w2d_metrics.detector_quality_summary_test_fixture(scores, labels)

    def test_strict_json_rejects_nonstandard_numbers_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "value.json"
            for payload in (
                '{"x": NaN}',
                '{"x": Infinity}',
                '{"x": 1e999}',
                '{"x": 1, "x": 2}',
            ):
                path.write_text(payload, encoding="utf-8")
                with self.assertRaises(w2d_metrics.MetricsError):
                    w2d_metrics.strict_json_load(str(path))
            with self.assertRaises(w2d_metrics.MetricsError):
                w2d_metrics.strict_json_dump({"x": float("nan")}, str(path))


class ScorerIntegrationTest(unittest.TestCase):
    def _bundle(self, root: Path):
        embedder = FakeEmbedder()
        reference_keys = [opaque(f"ref-{index}") for index in range(12)]
        calibration_keys = [opaque("cal-clean"), opaque("cal-suspicious")]
        test_keys = [opaque("test-clean"), opaque("test-suspicious")]
        all_keys = reference_keys + calibration_keys + test_keys
        texts = [
            f"document {index} alpha beta gamma delta epsilon zeta eta"
            for index in range(len(all_keys))
        ]
        vectors = embedder(texts).astype(np.float32)
        items = [
            {
                "item_key": key,
                "normalized_text": text,
                "embedding_row": index,
                "source_evidence": source_evidence(),
                "source_group": opaque(f"group-{index}"),
                "reference": key in set(reference_keys),
                "model_revision": REVISION,
            }
            for index, (key, text) in enumerate(zip(all_keys, texts))
        ]
        inputs = {
            "schema_version": "1.0",
            "construction_version": "W2D-A6-v1",
            "model_revision": REVISION,
            "model": {
                "name": detector.MODEL_NAME,
                "revision": REVISION,
                "dimension": DIM,
            },
            "embedding_artifact": {
                "filename": "inputs.npz",
                "sha256": "",
                "array": "embeddings",
                "dtype": "float32",
                "shape": [len(items), DIM],
            },
            "items": items,
            "reference_index": {
                "item_keys": reference_keys,
                "embedding_rows": list(range(len(reference_keys))),
                "groups": [
                    opaque(f"group-{index}") for index in range(12)
                ],
            },
            "sets": {
                "calibration_item_keys": sorted(calibration_keys),
                "test_score_order": sorted(
                    test_keys, key=w2d_scorer.score_order_key
                ),
                "source_control_item_keys": [],
                "warmup_reference_item_key": min(reference_keys),
            },
        }
        inputs_path = root / "inputs.json"
        embeddings_path = root / "inputs.npz"
        np.savez_compressed(embeddings_path, embeddings=vectors)
        inputs["embedding_artifact"]["sha256"] = w2d_metrics.file_sha256(
            str(embeddings_path)
        )
        w2d_metrics.strict_json_dump(inputs, str(inputs_path))
        return (
            embedder,
            inputs_path,
            embeddings_path,
            calibration_keys,
            test_keys,
        )

    def test_score_calibrate_test_pipeline_and_key_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embedder, inputs_path, embeddings_path, cal_keys, test_keys = self._bundle(root)
            detector_instance = D1Detector(
                embedder=embedder,
                pinned_model_revision=REVISION,
                expected_dim=DIM,
                k_neighbors=3,
            )
            calibration_scores = w2d_scorer.run_scorer_test_fixture(
                inputs_path=str(inputs_path),
                embeddings_path=str(embeddings_path),
                keys_path=None,
                phase="calibration",
                detector=detector_instance,
            )
            self.assertEqual(
                {record["item_key"] for record in calibration_scores["items"]},
                set(cal_keys),
            )
            labels = {
                "items": [
                    {
                        "item_key": cal_keys[0],
                        "source_group": "cal-group-0",
                        "split": "calibration",
                        "stratum": "ordinary_clean",
                        "is_poison": False,
                        "attack_family": None,
                    },
                    {
                        "item_key": cal_keys[1],
                        "source_group": "cal-group-1",
                        "split": "calibration",
                        "stratum": "recipe_poison",
                        "is_poison": True,
                        "attack_family": "recipe",
                    },
                ]
            }
            threshold = w2d_calibrate.select_threshold_test_fixture(
                calibration_scores, labels
            )
            threshold_path = root / "threshold.json"
            w2d_metrics.strict_json_dump(threshold, str(threshold_path))
            test_scores = w2d_scorer.run_scorer_test_fixture(
                inputs_path=str(inputs_path),
                embeddings_path=str(embeddings_path),
                keys_path=None,
                phase="test",
                threshold_artifact_path=str(threshold_path),
                detector=detector_instance,
            )
            self.assertEqual(
                [record["item_key"] for record in test_scores["items"]],
                sorted(test_keys, key=w2d_scorer.score_order_key),
            )
            self.assertTrue(
                all(record["promote"] is not None for record in test_scores["items"])
            )
            self.assertEqual(
                test_scores["measurement_name"],
                w2d_scorer.MEASUREMENT_NAME,
            )


if __name__ == "__main__":
    unittest.main()
