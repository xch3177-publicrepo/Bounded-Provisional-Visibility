#!/usr/bin/env python3
"""Contract and failure-path tests for the preregistered W2D-E3 runner.

These tests deliberately stop before any Milvus or Docker operation.  They
lock the parts of W2D-E3 that must be knowable before a formal run: constants,
cell construction/order, strict JSON, and exclusive result ownership.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

import w2d_e3_runner as e3


RUN_NUMBERS = (1, 2, 3, 4, 5)
CONTRACT_ERRORS = (TypeError, ValueError, e3.E3Error)
WILLIAMS4 = (
    ("B1", "B2", "B4", "B3"),
    ("B2", "B3", "B1", "B4"),
    ("B3", "B4", "B2", "B1"),
    ("B4", "B1", "B3", "B2"),
)
CONCURRENCIES = (1, 4, 8, 16)


def strict_stdlib_load(payload: bytes):
    """Independent strict reader for bytes emitted by the E3 writer."""

    def reject_constant(token: str):
        raise ValueError(f"non-standard numeric token: {token}")

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        payload.decode("utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


class FrozenContractTest(unittest.TestCase):
    def test_parent_byte_bindings_are_exact_and_complete(self) -> None:
        self.assertEqual(
            dict(e3.FROZEN_PARENT_SHA256),
            {
                "data/w2d/W2D-detector-inputs.json":
                    "5784dbf469f004068f3f6b183f44ff51e36de0035ae8cc146ddb7291d22cf39b",
                "data/w2d/W2D-detector-inputs.npz":
                    "1249c7cd23a03d3cde4e4c46ff3ec67ff9d59e6a3fd05978bb7c6da00bba1c45",
                "results/w2d/W2D-test-scores.json":
                    "f928416f8adf7277ec6eeb536d2ebce668ae4655f97f425a3f57501137c91e01",
                "results/w2d/W2D-threshold.json":
                    "52d87cdde2d868862764b18e134f6f28b0fc06c4092c1da9d0ec91e9cfc6eca3",
                "results/w2d/W2D-PROTOCOL-PLAN.json":
                    "8209e741d1d5899c2fee6205f5c3808b5a8c60b3f33a2a0dc3b3b1b4774aa262",
                "results/w2d/W2D-E1-INMEMORY.json":
                    "767248c0669fdde87f9c26072a0dde649b892b0a0153aa4729f8126104f1a18c",
                "results/w2d/W2D-E2-MILVUS.json":
                    "90d4309c649ff090023ce5599d77802b77cbfe173a8099536d2f74a28d1ac2ef",
            },
        )
        self.assertEqual(
            e3.TRANSITIVE_LABEL_PATH,
            "results/w2d/W2D-labels.json",
        )
        self.assertEqual(
            e3.TRANSITIVE_LABEL_SHA256,
            "05d1363510a1790d690b05640c3a2830be4e62df56b8592182904ccb29019a9e",
        )

    def test_all_preregistered_constants_are_exact(self) -> None:
        expected = {
            "RUN_NUMBERS": RUN_NUMBERS,
            "BACKGROUND_SEED": 7301,
            "QUERY_ORDER_SEED_BASE": 8300,
            "STATIC_N": 100_000,
            "FROZEN_N": 1_752,
            "BACKGROUND_N": 98_248,
            "DIM": 384,
            "QUERY_N": 192,
            "PANEL_A_EF": (20, 64, 256),
            "PRIMARY_EF": 64,
            "PANEL_A_REPEATS": 5,
            "PANEL_A_LATENCY_N": 960,
            "HNSW_M": 16,
            "HNSW_EF_CONSTRUCTION": 200,
            "TOP_K": 5,
            "CONSISTENCY_LEVEL": "Strong",
            "ADMISSION_RATE_PER_S": 20,
            "PANEL_B_WARMUP_S": 5.0,
            "PANEL_B_MEASURED_S": 15.0,
            "TP_S": 1.0,
            "VERIFIER_CONCURRENCY": 4,
        }
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertTrue(
                    hasattr(e3, name),
                    f"w2d_e3_runner must expose frozen constant {name}",
                )
                actual = getattr(e3, name)
                if isinstance(value, tuple):
                    actual = tuple(actual)
                self.assertEqual(actual, value)

        self.assertEqual(e3.FROZEN_N + e3.BACKGROUND_N, e3.STATIC_N)
        self.assertEqual(e3.QUERY_N * e3.PANEL_A_REPEATS, 960)

    def test_effective_hnsw_rejects_finished_but_pending_segments(self) -> None:
        ready = {
            "index_type_effective": "HNSW",
            "metric_type_effective": "COSINE",
            "M_effective": 16,
            "efConstruction_effective": 200,
            "indexed_rows": 100_000,
            "pending_index_rows": 0,
            "index_state": "Finished",
            "load_state": "Loaded",
        }
        self.assertTrue(e3._index_is_ready(ready))
        self.assertFalse(
            e3._index_is_ready(
                {**ready, "indexed_rows": 0, "pending_index_rows": 100_000}
            )
        )
        self.assertFalse(e3._index_is_ready({**ready, "indexed_rows": None}))

    def test_background_vectors_follow_the_frozen_numpy_recipe(self) -> None:
        expected_rng = np.random.default_rng(7301)
        expected = expected_rng.standard_normal((3, 4), dtype="float32")
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)

        actual = e3.deterministic_background(3, 4, seed=7301)
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.shape, (3, 4))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            actual,
            e3.deterministic_background(3, 4, seed=7301),
        )
        self.assertTrue(
            np.allclose(np.linalg.norm(actual, axis=1), 1.0, atol=1e-6)
        )
        self.assertFalse(
            np.array_equal(
                actual,
                e3.deterministic_background(3, 4, seed=7302),
            )
        )
        self.assertEqual(
            e3.deterministic_background(0, 4, seed=7301).shape,
            (0, 4),
        )

        for count, dim, seed in (
            (-1, 4, 7301),
            (3, 0, 7301),
            (True, 4, 7301),
            (3, 4.0, 7301),
            (3, 4, True),
        ):
            with self.subTest(count=count, dim=dim, seed=seed):
                with self.assertRaises((TypeError, ValueError)):
                    e3.deterministic_background(count, dim, seed=seed)

    def test_query_order_is_run_specific_frozen_and_label_blind(self) -> None:
        orders = []
        for run_number in RUN_NUMBERS:
            with self.subTest(run_number=run_number):
                expected_seed = 8300 + run_number
                self.assertEqual(e3.query_order_seed(run_number), expected_seed)
                expected = np.random.default_rng(expected_seed).permutation(192)
                actual = np.asarray(e3.query_order(run_number, n=192))
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(sorted(actual.tolist()), list(range(192)))
                np.testing.assert_array_equal(
                    actual,
                    np.asarray(e3.query_order(run_number, n=192)),
                )
                orders.append(tuple(actual))
        self.assertEqual(len(set(orders)), 5)

    def test_frozen_loader_closes_all_preregistered_populations(self) -> None:
        frozen = e3.load_frozen()
        self.assertEqual(frozen.embeddings.shape, (1_752, 384))
        self.assertEqual(frozen.embeddings.dtype, np.float32)
        self.assertTrue(np.isfinite(frozen.embeddings).all())
        self.assertEqual(len(frozen.key_to_row), 1_752)
        self.assertEqual(set(frozen.key_to_row.values()), set(range(1_752)))

        self.assertEqual(len(frozen.test_order), 512)
        self.assertEqual(len(set(frozen.test_order)), 512)
        self.assertEqual(set(frozen.scores), set(frozen.test_order))
        self.assertEqual(set(frozen.truth_poison), set(frozen.test_order))

        self.assertEqual(len(frozen.query_keys), 192)
        self.assertEqual(len(set(frozen.query_keys)), 192)
        self.assertEqual(frozen.query_vectors.shape, (192, 384))
        self.assertEqual(len(frozen.query_families), 192)
        self.assertEqual(len(frozen.query_associated_ids), 192)
        self.assertEqual(
            set(frozen.parent_hashes),
            {*e3.FROZEN_PARENT_SHA256, e3.TRANSITIVE_LABEL_PATH},
        )

    def test_baseline_and_concurrency_orders_are_frozen(self) -> None:
        expected_concurrency = {
            1: (1, 4, 8, 16),
            2: (4, 8, 16, 1),
            3: (8, 16, 1, 4),
            4: (16, 1, 4, 8),
            5: (1, 4, 8, 16),
        }
        for run_number in RUN_NUMBERS:
            with self.subTest(run_number=run_number):
                self.assertEqual(
                    tuple(e3.baseline_order(run_number)),
                    WILLIAMS4[(run_number - 1) % 4],
                )
                self.assertEqual(
                    tuple(e3.concurrency_order(run_number)),
                    expected_concurrency[run_number],
                )

        for invalid in (0, 6, -1, True, 1.0, "1", None):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    e3.baseline_order(invalid)
                with self.assertRaises((TypeError, ValueError)):
                    e3.concurrency_order(invalid)
                with self.assertRaises((TypeError, ValueError)):
                    e3.panel_b_matrix(invalid)

    def test_panel_b_matrix_is_complete_ordered_and_deterministic(self) -> None:
        all_cell_ids = set()
        for run_number in RUN_NUMBERS:
            with self.subTest(run_number=run_number):
                first = e3.panel_b_matrix(run_number)
                second = e3.panel_b_matrix(run_number)
                self.assertEqual(first, second)
                self.assertEqual(len(first), 16)

                expected_pairs = [
                    (baseline, concurrency)
                    for baseline in e3.baseline_order(run_number)
                    for concurrency in e3.concurrency_order(run_number)
                ]
                self.assertEqual(
                    [
                        (cell["baseline"], cell["query_concurrency"])
                        for cell in first
                    ],
                    expected_pairs,
                )
                self.assertEqual(
                    len(
                        {
                            (cell["baseline"], cell["query_concurrency"])
                            for cell in first
                        }
                    ),
                    16,
                )
                self.assertEqual(len({cell["cell_id"] for cell in first}), 16)
                all_cell_ids.update(cell["cell_id"] for cell in first)

                for cell in first:
                    self.assertTrue(
                        cell["cell_id"].startswith(f"run-{run_number}__")
                    )
                    self.assertEqual(cell["warmup_s"], 5)
                    self.assertEqual(cell["measured_s"], 15)
                    self.assertEqual(cell["admission_rate_per_s"], 20)
                    self.assertEqual(cell["tp_s"], 1.0)
                    self.assertEqual(cell["verifier_concurrency"], 4)
                    self.assertEqual(cell["top_k"], 5)
                    self.assertEqual(cell["ef"], 64)
        self.assertEqual(
            len(all_cell_ids),
            80,
            "cell ids must remain unique across all five independent runs",
        )


class StrictJsonAndOwnershipTest(unittest.TestCase):
    def test_percentiles_are_finite_and_defined_inside_one_run(self) -> None:
        self.assertEqual(
            e3.percentiles([1.0, 2.0, 3.0, 4.0]),
            {
                "count": 4,
                "p50": 2.5,
                "p95": 3.8499999999999996,
                "p99": 3.9699999999999998,
                "max": 4.0,
            },
        )
        for values in ([], [1.0, math.nan], [math.inf], ["1"]):
            with self.subTest(values=values):
                with self.assertRaises(CONTRACT_ERRORS):
                    e3.percentiles(values)

    def test_strict_json_bytes_are_deterministic_and_finite(self) -> None:
        left = e3.strict_json_bytes({"z": [1, 2.5], "a": {"x": "snowman \u2603"}})
        right = e3.strict_json_bytes({"a": {"x": "snowman \u2603"}, "z": [1, 2.5]})
        self.assertEqual(left, right)
        self.assertEqual(
            strict_stdlib_load(left),
            {"a": {"x": "snowman \u2603"}, "z": [1, 2.5]},
        )

        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(CONTRACT_ERRORS):
                    e3.strict_json_bytes({"nested": [0, {"bad": value}]})

    def test_strict_reader_rejects_duplicates_and_nonstandard_numbers(self) -> None:
        invalid = (
            b'{"x":1,"x":2}\n',
            b'{"x":NaN}\n',
            b'{"x":Infinity}\n',
            b'{"x":-Infinity}\n',
            b'{"x":1e400}\n',
        )
        for payload in invalid:
            with self.subTest(payload=payload):
                with self.assertRaises(CONTRACT_ERRORS):
                    e3.strict_json_load_bytes(payload, source="fixture")

        self.assertEqual(
            e3.strict_json_load_bytes(b'{"x":[1,2,3]}\n', source="fixture"),
            {"x": [1, 2, 3]},
        )

    def test_exclusive_writer_never_overwrites_or_leaves_invalid_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output = root / "RUN-01.json"
            document = {"run_number": 1, "finite": 1.25}

            e3.exclusive_write_json(output, document)
            original = output.read_bytes()
            self.assertEqual(strict_stdlib_load(original), document)

            with self.assertRaises((FileExistsError, e3.E3Error)):
                e3.exclusive_write_json(
                    output,
                    {"run_number": 1, "replacement": True},
                )
            self.assertEqual(output.read_bytes(), original)

            invalid_output = root / "RUN-02.json"
            with self.assertRaises(CONTRACT_ERRORS):
                e3.exclusive_write_json(invalid_output, {"bad": math.nan})
            self.assertFalse(
                invalid_output.exists(),
                "serialization failure must not reserve an empty formal output",
            )


if __name__ == "__main__":
    unittest.main()
