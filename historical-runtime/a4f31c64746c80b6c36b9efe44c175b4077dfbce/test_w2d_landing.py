#!/usr/bin/env python3
"""Offline tests for the exact W2D landing evaluator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import w2d_landing as landing


def key(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


class LandingTest(unittest.TestCase):
    def _fixture(self, root: Path):
        input_path = root / "W2D-detector-inputs.json"
        npz_path = root / "W2D-detector-inputs.npz"
        plan_path = root / "W2D-PROTOCOL-PLAN.json"

        ref_keys = [key(f"reference-{index}") for index in range(768)]
        recipe_keys = [key(f"recipe-{index}") for index in range(128)]
        cover_keys = [key(f"cover-{index}") for index in range(64)]
        poison = [(item_key, "recipe") for item_key in recipe_keys]
        poison.extend(
            (item_key, "natural_cover_suffix_v1") for item_key in cover_keys
        )
        all_keys = ref_keys + recipe_keys + cover_keys
        matrix = np.zeros((len(all_keys), 384), dtype=np.float32)
        matrix[: len(ref_keys), 1] = 1.0
        matrix[len(ref_keys) :, 0] = 1.0
        np.savez_compressed(npz_path, embeddings=matrix)
        safe = {
            "items": [
                {"item_key": item_key, "embedding_row": index}
                for index, item_key in enumerate(all_keys)
            ]
        }
        input_path.write_text(
            json.dumps(safe, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        query = [1.0] + [0.0] * 383
        query_digest = hashlib.sha256(
            (
                json.dumps(
                    query,
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest()
        landing_records = []
        query_materials = {}
        populations = {}
        offset = 0
        for family, expected_n in landing.EXPECTED_POPULATIONS.items():
            family_poison = poison[offset : offset + expected_n]
            offset += expected_n
            family_keys = []
            for position, (item_key, observed_family) in enumerate(
                family_poison, start=1
            ):
                self.assertEqual(observed_family, family)
                source_group = key(f"group-{item_key}")
                topic = "topic-a"
                variant = "T3" if family == "recipe" else family
                family_keys.append(item_key)
                landing_records.append(
                    {
                        "landing_order_position_within_attack": position,
                        "item_key": item_key,
                        "source_group": source_group,
                        "attack_family": family,
                        "attack_variant": variant,
                        "topic": topic,
                        "query_role": "attack_associated",
                        "query_item_key": item_key,
                        "retrieval_background_id": "frozen_reference_768",
                        "candidate_added_alone_item_key": item_key,
                        "candidate_corpus_size": 769,
                    }
                )
                query_materials[item_key] = {
                    "query_item_key": item_key,
                    "attack_family": family,
                    "attack_variant": variant,
                    "source_group": source_group,
                    "topic": topic,
                    "text": f"frozen query for {item_key}",
                    "embedding_encoding": (
                        "strict-JSON float array in pinned model dimension"
                    ),
                    "embedding": query,
                    "embedding_sha256": query_digest,
                }
            populations[family] = {
                "item_keys": family_keys,
                "expected_denominator": expected_n,
                "failed_landing_policy": "retain_as_failure_never_replace",
            }
        plan = {
            "provenance": {
                "inputs": [
                    {
                        "filename": input_path.name,
                        "sha256": landing.file_sha256(str(input_path)),
                    },
                    {
                        "filename": npz_path.name,
                        "sha256": landing.file_sha256(str(npz_path)),
                    },
                ]
            },
            "retrieval_background": {"item_keys": ref_keys},
            "query_materials": query_materials,
            "planned_poison_queries": {
                item_key: {
                    "attack_family": family,
                    "roles": {
                        "attack_associated": {"query_item_key": item_key}
                    },
                }
                for item_key, family in poison[:60]
            },
            "landing_plan": {
                "background_count": 768,
                "background_id": "frozen_reference_768",
                "baseline": "B1",
                "candidate_rule": (
                    "add exactly one modified test poison passage to the "
                    "immutable background; query with that item's "
                    "attack_associated query"
                ),
                "metric": "exact_cosine",
                "off_path": True,
                "populations": populations,
                "records": landing_records,
                "result_requirements": landing.EXPECTED_RESULT_REQUIREMENTS,
                "tie_break": (
                    "descending cosine then ascending opaque item_key"
                ),
                "top_k": 5,
            },
        }
        plan_path.write_text(
            json.dumps(plan, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        return plan_path, input_path, npz_path

    def test_all_items_and_denominators_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, inputs, embeddings = self._fixture(root)
            artifact = landing.evaluate_landing(
                plan_path=str(plan),
                inputs_path=str(inputs),
                embeddings_path=str(embeddings),
            )
            self.assertEqual(len(artifact["items"]), 192)
            self.assertEqual(artifact["populations"]["recipe"]["n"], 128)
            self.assertEqual(
                artifact["populations"]["natural_cover_suffix_v1"]["n"], 64
            )
            self.assertTrue(all(row["landed_top5"] for row in artifact["items"]))
            frozen_plan = json.loads(plan.read_text(encoding="utf-8"))
            self.assertEqual(len(frozen_plan["planned_poison_queries"]), 60)

    def test_landing_record_must_bind_its_own_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, inputs, embeddings = self._fixture(root)
            document = json.loads(plan.read_text(encoding="utf-8"))
            document["landing_plan"]["records"][60].pop("query_item_key")
            plan.write_text(
                json.dumps(document, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                landing.LandingError, "record fields"
            ):
                landing.evaluate_landing(
                    plan_path=str(plan),
                    inputs_path=str(inputs),
                    embeddings_path=str(embeddings),
                )

    def test_exact_landing_schema_rejects_over_and_under_fill(self):
        mutations = {
            "missing record": lambda document: document["landing_plan"][
                "records"
            ].pop(),
            "extra record": lambda document: document["landing_plan"][
                "records"
            ].append(dict(document["landing_plan"]["records"][-1])),
            "missing material": lambda document: document[
                "query_materials"
            ].pop(next(iter(document["query_materials"]))),
            "extra material": lambda document: document[
                "query_materials"
            ].update({key("extra-material"): dict(
                next(iter(document["query_materials"].values()))
            )}),
            "population order": lambda document: document["landing_plan"][
                "populations"
            ]["recipe"]["item_keys"].reverse(),
            "record order position": lambda document: document[
                "landing_plan"
            ]["records"][0].update(
                {"landing_order_position_within_attack": 2}
            ),
            "candidate corpus": lambda document: document["landing_plan"][
                "records"
            ][0].update({"candidate_corpus_size": 770}),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                plan, inputs, embeddings = self._fixture(root)
                document = json.loads(plan.read_text(encoding="utf-8"))
                mutate(document)
                plan.write_text(
                    json.dumps(document, sort_keys=True, allow_nan=False),
                    encoding="utf-8",
                )
                with self.assertRaises(landing.LandingError):
                    landing.evaluate_landing(
                        plan_path=str(plan),
                        inputs_path=str(inputs),
                        embeddings_path=str(embeddings),
                    )

    def test_json_integer_fields_require_exact_int_types(self):
        mutations = {
            "top_k float": lambda document: document["landing_plan"].update(
                {"top_k": 5.0}
            ),
            "top_k bool": lambda document: document["landing_plan"].update(
                {"top_k": True}
            ),
            "background_count float": lambda document: document[
                "landing_plan"
            ].update({"background_count": 768.0}),
            "background_count bool": lambda document: document[
                "landing_plan"
            ].update({"background_count": True}),
            "order bool": lambda document: document["landing_plan"][
                "records"
            ][0].update({"landing_order_position_within_attack": True}),
            "order float": lambda document: document["landing_plan"][
                "records"
            ][0].update({"landing_order_position_within_attack": 0.0}),
            "candidate size float": lambda document: document[
                "landing_plan"
            ]["records"][0].update({"candidate_corpus_size": 769.0}),
            "candidate size bool": lambda document: document[
                "landing_plan"
            ]["records"][0].update({"candidate_corpus_size": True}),
            "denominator float": lambda document: document["landing_plan"][
                "populations"
            ]["recipe"].update({"expected_denominator": 128.0}),
            "denominator bool": lambda document: document["landing_plan"][
                "populations"
            ]["recipe"].update({"expected_denominator": True}),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                plan, inputs, embeddings = self._fixture(root)
                document = json.loads(plan.read_text(encoding="utf-8"))
                mutate(document)
                plan.write_text(
                    json.dumps(document, sort_keys=True, allow_nan=False),
                    encoding="utf-8",
                )
                with self.assertRaises(landing.LandingError):
                    landing.evaluate_landing(
                        plan_path=str(plan),
                        inputs_path=str(inputs),
                        embeddings_path=str(embeddings),
                    )

    def test_no_resampling_requires_an_exact_json_boolean(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, inputs, embeddings = self._fixture(root)
            document = json.loads(plan.read_text(encoding="utf-8"))
            document["landing_plan"]["result_requirements"][
                "no_resampling"
            ] = 1
            plan.write_text(
                json.dumps(document, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                landing.LandingError, "result requirements"
            ):
                landing.evaluate_landing(
                    plan_path=str(plan),
                    inputs_path=str(inputs),
                    embeddings_path=str(embeddings),
                )

    def test_query_embedding_rejects_non_float_raw_values_before_numpy(self):
        invalid_embeddings = {
            "not a list": tuple([1.0] + [0.0] * 383),
            "wrong length": [1.0] + [0.0] * 382,
            "bool component": [True] + [0.0] * 383,
            "int component": [1] + [0.0] * 383,
            "string component": ["1.0"] + [0.0] * 383,
            "nonfinite component": [float("nan")] + [0.0] * 383,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, _inputs, _embeddings = self._fixture(root)
            document = json.loads(plan.read_text(encoding="utf-8"))
            record = document["landing_plan"]["records"][0]
            material = document["query_materials"][record["query_item_key"]]
            for name, raw_embedding in invalid_embeddings.items():
                with self.subTest(name=name):
                    changed = dict(material)
                    changed["embedding"] = raw_embedding
                    with (
                        mock.patch(
                            "w2d_landing.np.asarray",
                            side_effect=AssertionError(
                                "numpy conversion happened too early"
                            ),
                        ),
                        self.assertRaisesRegex(
                            landing.LandingError, "invalid embedding"
                        ),
                    ):
                        landing._query_embedding(changed, record)

    def test_each_input_drift_is_rejected_before_output(self):
        for input_name in ("plan", "inputs", "embeddings"):
            with (
                self.subTest(input_name=input_name),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                plan, inputs, embeddings = self._fixture(root)
                output = root / "landing.json"
                paths = {
                    "plan": plan,
                    "inputs": inputs,
                    "embeddings": embeddings,
                }
                original = landing._landing_records

                def drift_once(document):
                    result = original(document)
                    target = paths[input_name]
                    target.write_bytes(target.read_bytes() + b" ")
                    return result

                with mock.patch(
                    "w2d_landing._landing_records",
                    side_effect=drift_once,
                ):
                    with self.assertRaisesRegex(
                        landing.LandingError,
                        "changed during evaluation",
                    ):
                        landing.evaluate_landing(
                            plan_path=str(plan),
                            inputs_path=str(inputs),
                            embeddings_path=str(embeddings),
                            output_path=str(output),
                        )
                self.assertFalse(output.exists())

    def test_postwrite_drift_preserves_failed_output_for_quarantine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, inputs, embeddings = self._fixture(root)
            output = root / "landing.json"
            original = landing._exclusive_json_dump

            def write_then_drift(value, path):
                ownership = original(value, path)
                plan.write_bytes(plan.read_bytes() + b" ")
                return ownership

            with mock.patch(
                "w2d_landing._exclusive_json_dump",
                side_effect=write_then_drift,
            ):
                with self.assertRaisesRegex(
                    landing.LandingError,
                    "changed during evaluation",
                ):
                    landing.evaluate_landing(
                        plan_path=str(plan),
                        inputs_path=str(inputs),
                        embeddings_path=str(embeddings),
                        output_path=str(output),
                    )
            self.assertTrue(output.exists())

    def test_failure_handling_preserves_substituted_or_modified_output(self):
        replacement = b"user replacement must survive\n"
        for mode in (
            "different inode",
            "same payload different inode",
            "changed payload",
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                plan, inputs, embeddings = self._fixture(root)
                output = root / "landing.json"
                original = landing._exclusive_json_dump
                created_inode = {}

                def write_then_replace(value, path):
                    ownership = original(value, path)
                    created_inode["value"] = ownership.inode
                    target = Path(path)
                    expected = replacement
                    if mode == "same payload different inode":
                        expected = target.read_bytes()
                    if mode in {
                        "different inode",
                        "same payload different inode",
                    }:
                        target.unlink()
                    target.write_bytes(expected)
                    return ownership

                with mock.patch(
                    "w2d_landing._exclusive_json_dump",
                    side_effect=write_then_replace,
                ):
                    with self.assertRaisesRegex(
                        landing.LandingError,
                        "output ownership conflict",
                    ):
                        landing.evaluate_landing(
                            plan_path=str(plan),
                            inputs_path=str(inputs),
                            embeddings_path=str(embeddings),
                            output_path=str(output),
                        )
                self.assertTrue(output.exists())
                if mode == "same payload different inode":
                    self.assertNotEqual(
                        output.stat().st_ino, created_inode["value"]
                    )
                else:
                    self.assertEqual(output.read_bytes(), replacement)

    def test_output_is_exclusive_and_lineage_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, inputs, embeddings = self._fixture(root)
            output = root / "landing.json"
            landing.evaluate_landing(
                plan_path=str(plan),
                inputs_path=str(inputs),
                embeddings_path=str(embeddings),
                output_path=str(output),
            )
            with self.assertRaises(FileExistsError):
                landing.evaluate_landing(
                    plan_path=str(plan),
                    inputs_path=str(inputs),
                    embeddings_path=str(embeddings),
                    output_path=str(output),
                )
            inputs.write_text("{}", encoding="utf-8")
            with self.assertRaises(landing.LandingError):
                landing.evaluate_landing(
                    plan_path=str(plan),
                    inputs_path=str(inputs),
                    embeddings_path=str(embeddings),
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
