"""Offline acceptance tests for the immutable W2D A4 protocol plan.

The fixture has the production A4 cardinalities but eight-dimensional,
deterministic unit vectors.  It never invokes a detector or creates a formal
landing/runtime artifact.

    python test_w2d_protocol.py
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import w2d_protocol as protocol


TOPICS = tuple(f"topic-{index}" for index in range(8))
MODEL_REVISION = "synthetic-pinned-revision"
MODEL_DIMENSION = 8


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _unit_vector(value: str) -> np.ndarray:
    raw = hashlib.sha256(value.encode("utf-8")).digest()
    vector = np.asarray(
        [1.0 + raw[index] / 255.0 for index in range(MODEL_DIMENSION)],
        dtype=np.float32,
    )
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def _json_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _npz_bytes(matrix: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.savez(stream, embeddings=matrix)
    return stream.getvalue()


def build_frozen_fixture(root: Path) -> dict[str, Path | list[str]]:
    data = root / "data" / "w2d"
    results = root / "results" / "w2d"
    data.mkdir(parents=True)
    results.mkdir(parents=True)
    safe_npz = data / "W2D-detector-inputs.npz"
    safe_json = data / "W2D-detector-inputs.json"
    labels_json = results / "W2D-labels.json"
    freeze_json = data / "W2D-DATA-FREEZE.json"

    records: list[dict] = []

    def add(
        topic: str,
        kind: str,
        index: int,
        *,
        split: str,
        stratum: str,
        poison: bool,
        attack_family: str | None = None,
        attack_variant: str | None = None,
    ) -> None:
        item_key = _digest(f"item|{topic}|{kind}|{index}")
        source_group = _digest(f"group|{topic}|{kind}|{index}")
        query = (
            " ".join(
                [f"{topic}-query-{kind}-{index}"]
                + [f"word-{position}" for position in range(1, 30)]
            )
            if poison
            else None
        )
        records.append(
            {
                "item_key": item_key,
                "source_group": source_group,
                "topic": topic,
                "split": split,
                "stratum": stratum,
                "poison": poison,
                "attack_family": attack_family,
                "attack_variant": attack_variant,
                "target_query": query,
                "target_query_embedding": (
                    [float(value) for value in _unit_vector(f"query|{item_key}")]
                    if poison
                    else None
                ),
                "normalized_text": (
                    " ".join(f"text-{kind}-{index}-{word}" for word in range(40))
                ),
                "embedding": _unit_vector(f"document|{item_key}"),
            }
        )

    for topic in TOPICS:
        for index in range(96):
            add(
                topic,
                "reference",
                index,
                split="reference",
                stratum="clean_reference",
                poison=False,
            )
        for index in range(24):
            add(
                topic,
                "test-ordinary",
                index,
                split="test",
                stratum="ordinary_clean",
                poison=False,
            )
        for index in range(16):
            add(
                topic,
                "test-hard",
                index,
                split="test",
                stratum="hard_negative_clean",
                poison=False,
            )
        for variant in ("T3", "T4"):
            for index in range(8):
                add(
                    topic,
                    f"recipe-{variant}",
                    index,
                    split="test",
                    stratum="recipe_poison",
                    poison=True,
                    attack_family="recipe",
                    attack_variant=variant,
                )
        for index in range(8):
            add(
                topic,
                "natural",
                index,
                split="test",
                stratum="natural_cover_suffix_poison",
                poison=True,
                attack_family="natural_cover_suffix_v1",
                attack_variant="natural_cover_suffix_v1",
            )

    records.sort(key=lambda record: record["item_key"])
    matrix = np.asarray([record["embedding"] for record in records], dtype=np.float32)
    npz_payload = _npz_bytes(matrix)
    safe_npz.write_bytes(npz_payload)

    safe_items = []
    labels = {}
    row_by_key = {}
    for row, record in enumerate(records):
        key = record["item_key"]
        row_by_key[key] = row
        safe_items.append(
            {
                "item_key": key,
                "normalized_text": record["normalized_text"],
                "embedding_row": row,
                "source_evidence": [True, True, MODEL_REVISION],
                "source_group": record["source_group"],
                "reference": record["split"] == "reference",
                "model_revision": MODEL_REVISION,
            }
        )
        labels[key] = {
            "source_group": record["source_group"],
            "source_id": f"{record['topic']}/{key[:12]}",
            "topic": record["topic"],
            "split": record["split"],
            "stratum": record["stratum"],
            "poison": record["poison"],
            "attack_family": record["attack_family"],
            "attack_variant": record["attack_variant"],
            "source_control_kind": None,
            "hard_negative_rules": (
                ["synthetic_hard_rule"]
                if record["stratum"] == "hard_negative_clean"
                else []
            ),
            "target_query": record["target_query"],
            "target_query_embedding": record["target_query_embedding"],
        }

    reference_keys = [
        record["item_key"] for record in records if record["split"] == "reference"
    ]
    test_keys = [
        record["item_key"] for record in records if record["split"] == "test"
    ]
    safe_document = {
        "schema_version": "1.0",
        "construction_version": "synthetic-A4",
        "model_revision": MODEL_REVISION,
        "model": {
            "name": "synthetic-unit-model",
            "revision": MODEL_REVISION,
            "dimension": MODEL_DIMENSION,
        },
        "embedding_artifact": {
            "filename": safe_npz.name,
            "sha256": hashlib.sha256(npz_payload).hexdigest(),
            "array": "embeddings",
            "dtype": "float32",
            "shape": [len(records), MODEL_DIMENSION],
        },
        "items": safe_items,
        "reference_index": {
            "item_keys": reference_keys,
            "embedding_rows": [row_by_key[key] for key in reference_keys],
            "groups": [labels[key]["source_group"] for key in reference_keys],
        },
        "sets": {
            "calibration_item_keys": [],
            "test_score_order": test_keys,
            "source_control_item_keys": [],
            "warmup_reference_item_key": min(reference_keys),
        },
    }
    labels_document = {
        "schema_version": "1.0",
        "construction_version": "synthetic-A4",
        "model_revision": MODEL_REVISION,
        "items": labels,
    }
    safe_payload = _json_bytes(safe_document)
    labels_payload = _json_bytes(labels_document)
    safe_json.write_bytes(safe_payload)
    labels_json.write_bytes(labels_payload)
    freeze_document = {
        "schema_version": "1.0",
        "construction_version": "synthetic-A4",
        "source": {"topics": list(TOPICS)},
        "model": {
            "name": "synthetic-unit-model",
            "revision": MODEL_REVISION,
            "dimension": MODEL_DIMENSION,
        },
        "artifacts": {
            safe_json.name: {
                "sha256": hashlib.sha256(safe_payload).hexdigest(),
                "role": "safe_detector_schema",
            },
            safe_npz.name: {
                "sha256": hashlib.sha256(npz_payload).hexdigest(),
                "role": "safe_full_item_embeddings",
            },
            labels_json.name: {
                "sha256": hashlib.sha256(labels_payload).hexdigest(),
                "role": "evaluator_only_labels_queries_and_attack_family",
            },
        },
    }
    freeze_json.write_bytes(_json_bytes(freeze_document))
    return {
        "freeze": freeze_json,
        "safe_json": safe_json,
        "safe_npz": safe_npz,
        "labels": labels_json,
        "reference_keys": reference_keys,
        "test_keys": test_keys,
    }


class ProtocolPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.fixture = build_frozen_fixture(cls.root)
        cls.output = cls.root / "results" / "w2d" / "W2D-PROTOCOL-PLAN.json"
        cls.amendment = Path(protocol.__file__).with_name(
            "W2D-PREREGISTRATION-AMENDMENT-A4.md"
        )
        cls.build = protocol.build_protocol_plan(
            freeze_path=cls.fixture["freeze"],
            safe_json_path=cls.fixture["safe_json"],
            safe_npz_path=cls.fixture["safe_npz"],
            labels_path=cls.fixture["labels"],
            amendment_path=cls.amendment,
            output_path=cls.output,
        )
        cls.plan = protocol.strict_json_load(cls.output)
        cls.labels = protocol.strict_json_load(cls.fixture["labels"])["items"]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_exact_A4_assignments_and_denominators(self):
        assignments = self.plan["item_assignments"]["by_seed"]
        self.assertEqual([record["seed"] for record in assignments], [1, 2, 3, 4, 5])
        all_poison = []
        all_clean = []
        all_filler = []
        for seed_record in assignments:
            seed = seed_record["seed"]
            for attack in protocol.ATTACKS:
                poison = seed_record["poison_by_attack"][attack]
                self.assertEqual(len(poison), 6)
                self.assertEqual(
                    [record["topic"] for record in poison],
                    [
                        TOPICS[index % len(TOPICS)]
                        for index in range((seed - 1) * 6, seed * 6)
                    ],
                )
                all_poison.extend(record["item_key"] for record in poison)
            clean = seed_record["clean"]
            filler = seed_record["heavy_filler"]
            self.assertEqual(len(clean), 6)
            self.assertEqual(len(filler), 12)
            self.assertEqual(
                {
                    stratum: sum(
                        record["stratum"] == stratum for record in clean
                    )
                    for stratum in ("ordinary_clean", "hard_negative_clean")
                },
                {
                    stratum: protocol.CLEAN_COUNTS[seed][stratum]
                    for stratum in ("ordinary_clean", "hard_negative_clean")
                },
            )
            self.assertEqual(
                {
                    stratum: sum(
                        record["stratum"] == stratum for record in filler
                    )
                    for stratum in ("ordinary_clean", "hard_negative_clean")
                },
                {
                    stratum: protocol.FILLER_COUNTS[seed][stratum]
                    for stratum in ("ordinary_clean", "hard_negative_clean")
                },
            )
            all_clean.extend(record["item_key"] for record in clean)
            all_filler.extend(record["item_key"] for record in filler)

        recipe_records = [
            record
            for seed_record in assignments
            for record in seed_record["poison_by_attack"]["recipe"]
        ]
        self.assertEqual(
            [record["attack_variant"] for record in recipe_records],
            ["T3" if index % 2 == 0 else "T4" for index in range(30)],
        )
        self.assertEqual(len(all_poison), 60)
        self.assertEqual(len(set(all_poison)), 60)
        self.assertEqual(len(set(all_clean)), 30)
        self.assertEqual(len(set(all_filler)), 60)
        self.assertTrue(set(all_poison).isdisjoint(all_clean))
        self.assertTrue(set(all_poison).isdisjoint(all_filler))
        self.assertTrue(set(all_clean).isdisjoint(all_filler))
        self.assertEqual(
            self.plan["reporting_populations"]["detector_quality"]["denominator"],
            512,
        )

    def test_cross_arm_pairing_injection_order_and_140_cell_grid(self):
        units = self.plan["runtime_units"]
        cells = self.plan["cells"]
        self.assertEqual(len(units), 20)
        self.assertEqual(len(cells), 140)
        units_by_key = {
            (unit["seed"], unit["attack_family"], unit["backlog"]): unit
            for unit in units
        }
        for seed in protocol.SEEDS:
            for attack in protocol.ATTACKS:
                normal = units_by_key[(seed, attack, "normal")]
                heavy = units_by_key[(seed, attack, "heavy")]
                self.assertEqual(normal["poison_item_keys"], heavy["poison_item_keys"])
                self.assertEqual(normal["clean_item_keys"], heavy["clean_item_keys"])
                self.assertEqual(normal["filler_item_keys"], [])
                self.assertEqual(len(heavy["filler_item_keys"]), 12)
                for unit in (normal, heavy):
                    sequence = unit["injection_sequence"]
                    self.assertEqual(
                        [record["injection_ordinal"] for record in sequence],
                        list(range(len(sequence))),
                    )
                    roles = [record["role"] for record in sequence]
                    expected = (
                        ["filler"] * (12 if unit["backlog"] == "heavy" else 0)
                        + ["poison"] * 6
                        + ["clean"] * 6
                    )
                    self.assertEqual(roles, expected)
                    self.assertEqual(len(unit["reused_by_cells"]), 7)
                    matching = [
                        cell
                        for cell in cells
                        if cell["runtime_unit_id"] == unit["runtime_unit_id"]
                    ]
                    self.assertEqual(len(matching), 7)
                    b1 = [cell for cell in matching if cell["baseline"] == "B1"]
                    self.assertEqual(len(b1), 1)
                    self.assertEqual(b1[0]["arm"], "control")
                    for baseline in protocol.VERIFIED_BASELINES:
                        baseline_cells = [
                            cell
                            for cell in matching
                            if cell["baseline"] == baseline
                        ]
                        self.assertEqual(
                            {cell["arm"] for cell in baseline_cells},
                            {"detector", "oracle"},
                        )
                        self.assertEqual(
                            {
                                cell["service_replay"]
                                for cell in baseline_cells
                            },
                            {"frozen_D1_item_service_time"},
                        )

        for seed in protocol.SEEDS:
            order = self.plan["execution_order_by_seed"][str(seed)]
            self.assertEqual(len(order), 28)
            self.assertEqual(len(set(order)), 28)
            positions = {
                cell["cell_id"]: cell["execution_order_position"]
                for cell in cells
                if cell["seed"] == seed
            }
            self.assertEqual([positions[cell_id] for cell_id in order], list(range(1, 29)))

    def test_query_roles_retrieval_background_and_all_192_landing_items(self):
        background = self.plan["retrieval_background"]
        self.assertEqual(background["count"], 768)
        self.assertEqual(background["item_keys"], self.fixture["reference_keys"])
        self.assertEqual(len(set(background["source_groups"])), 768)
        self.assertTrue(
            set(background["item_keys"]).isdisjoint(self.fixture["test_keys"])
        )
        self.assertEqual(
            background["logical_roles"],
            ["detector_reference", "W2D_fixed_retrieval_background"],
        )

        selected_by_attack = {
            attack: {
                record["item_key"]
                for seed_record in self.plan["item_assignments"]["by_seed"]
                for record in seed_record["poison_by_attack"][attack]
            }
            for attack in protocol.ATTACKS
        }
        queries = self.plan["planned_poison_queries"]
        self.assertEqual(len(queries), 60)
        for owner, assignment in queries.items():
            roles = assignment["roles"]
            self.assertEqual(set(roles), set(protocol.QUERY_ROLES))
            self.assertEqual(
                roles["attack_associated"]["query_item_key"], owner
            )
            owner_label = self.labels[owner]
            heldout = roles["heldout_same_topic"]["donor_item_key"]
            negative = roles["negative_other_topic"]["donor_item_key"]
            self.assertNotIn(heldout, selected_by_attack[assignment["attack_family"]])
            self.assertNotIn(negative, selected_by_attack[assignment["attack_family"]])
            self.assertEqual(
                self.labels[heldout]["attack_family"],
                assignment["attack_family"],
            )
            self.assertEqual(self.labels[heldout]["topic"], owner_label["topic"])
            next_topic = TOPICS[
                (TOPICS.index(owner_label["topic"]) + 1) % len(TOPICS)
            ]
            self.assertEqual(self.labels[negative]["topic"], next_topic)
            for role in protocol.QUERY_ROLES:
                material_key = roles[role]["query_item_key"]
                material = self.plan["query_materials"][material_key]
                self.assertEqual(len(material["embedding"]), MODEL_DIMENSION)
                self.assertTrue(np.isfinite(material["embedding"]).all())

        landing = self.plan["landing_plan"]
        self.assertEqual(len(landing["records"]), 192)
        self.assertEqual(
            landing["populations"]["recipe"]["expected_denominator"], 128
        )
        self.assertEqual(
            landing["populations"]["natural_cover_suffix_v1"][
                "expected_denominator"
            ],
            64,
        )
        self.assertEqual(len(self.plan["query_materials"]), 192)
        self.assertTrue(
            all(
                record["candidate_corpus_size"] == 769
                and record["query_role"] == "attack_associated"
                for record in landing["records"]
            )
        )
        self.assertTrue(
            all("landed" not in record for record in landing["records"])
        )

    def test_deterministic_exclusive_create_and_no_outcome_input(self):
        self.assertEqual(
            self.plan["phase_constraint"],
            (
                "constructed after D1 test scoring; the builder consumes only "
                "frozen data, labels, constants, opaque keys, and "
                "domain-separated hashes, with no D1 score, threshold, "
                "metric, landing, or protocol result input"
            ),
        )
        second_output = self.root / "second" / "W2D-PROTOCOL-PLAN.json"
        protocol.build_protocol_plan(
            freeze_path=self.fixture["freeze"],
            safe_json_path=self.fixture["safe_json"],
            safe_npz_path=self.fixture["safe_npz"],
            labels_path=self.fixture["labels"],
            amendment_path=self.amendment,
            output_path=second_output,
        )
        self.assertEqual(self.output.read_bytes(), second_output.read_bytes())
        original = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            protocol.build_protocol_plan(
                freeze_path=self.fixture["freeze"],
                safe_json_path=self.fixture["safe_json"],
                safe_npz_path=self.fixture["safe_npz"],
                labels_path=self.fixture["labels"],
                amendment_path=self.amendment,
                output_path=self.output,
            )
        self.assertEqual(self.output.read_bytes(), original)

        parameters = set(inspect.signature(protocol.build_protocol_plan).parameters)
        self.assertEqual(
            parameters,
            {
                "freeze_path",
                "safe_json_path",
                "safe_npz_path",
                "labels_path",
                "amendment_path",
                "output_path",
            },
        )
        tree = ast.parse(Path(protocol.__file__).read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertFalse(
            imported
            & {
                "detector",
                "w2d_scorer",
                "w2d_calibrate",
                "w2d_metrics",
                "poison_exposure",
            }
        )

    def test_strict_json_and_frozen_hash_rejection(self):
        bad = self.root / "bad.json"
        for payload in (
            '{"x": NaN}',
            '{"x": Infinity}',
            '{"x": 1e999}',
            '{"x": 1, "x": 2}',
        ):
            bad.write_text(payload, encoding="utf-8")
            with self.assertRaises(protocol.ProtocolPlanError):
                protocol.strict_json_load(bad)

        tampered_labels = self.root / "tampered" / "W2D-labels.json"
        tampered_labels.parent.mkdir()
        tampered_labels.write_bytes(self.fixture["labels"].read_bytes() + b" ")
        with self.assertRaisesRegex(
            protocol.ProtocolPlanError, "sha256 .* != frozen"
        ):
            protocol.build_protocol_plan(
                freeze_path=self.fixture["freeze"],
                safe_json_path=self.fixture["safe_json"],
                safe_npz_path=self.fixture["safe_npz"],
                labels_path=tampered_labels,
                amendment_path=self.amendment,
                output_path=self.root / "tampered" / "plan.json",
            )


if __name__ == "__main__":
    unittest.main()
