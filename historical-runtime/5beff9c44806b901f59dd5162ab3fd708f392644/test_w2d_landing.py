#!/usr/bin/env python3
"""Offline tests for the exact W2D landing evaluator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

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
            "query_materials": {
                item_key: {
                    "embedding": query,
                    "embedding_sha256": query_digest,
                }
                for item_key, _family in poison
            },
            "planned_poison_queries": {
                item_key: {
                    "attack_family": family,
                    "roles": {
                        "attack_associated": {"query_item_key": item_key}
                    },
                }
                for item_key, family in poison
            },
            "landing_plan": {
                "records": [
                    {"item_key": item_key, "attack_family": family}
                    for item_key, family in poison
                ]
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
