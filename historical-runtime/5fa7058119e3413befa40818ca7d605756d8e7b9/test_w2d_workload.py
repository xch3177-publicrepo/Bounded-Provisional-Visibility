"""Offline guards for the A2/A3 W2D data freeze.

The fixtures are synthetic raw posts and a deterministic 384-dimensional
embedder.  No test downloads 20 Newsgroups or a Hugging Face model.

    ./.venv312/bin/python test_w2d_workload.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np

import w2d_workload as w2d


class FakeEmbedder:
    """Stable topic-shaped vectors with the SentenceTransformer call surface."""

    def __init__(self):
        self.calls = 0

    def encode(
        self,
        texts,
        *,
        normalize_embeddings=False,
        show_progress_bar=False,
        convert_to_numpy=True,
    ):
        self.calls += 1
        rows = []
        for text in texts:
            vector = np.zeros(w2d.MODEL_DIM, dtype=np.float32)
            match = re.search(r"TOPICMARKER([0-7])", text)
            topic_index = int(match.group(1)) if match else 0
            vector[topic_index] = 5.0
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            for index, value in enumerate(digest):
                vector[16 + index] = (value + 1) / 1024.0
            rows.append(vector)
        return np.asarray(rows, dtype=np.float32)


class NonfiniteEmbedder(FakeEmbedder):
    def encode(self, texts, **kwargs):
        rows = super().encode(texts, **kwargs)
        if len(rows):
            rows[0, 0] = np.nan
        return rows


def _body(topic_index: int, item_index: int) -> str:
    words = [
        f"topicword{topic_index}",
        f"TOPICMARKER{topic_index}",
        f"record{item_index:04d}",
    ]
    words.extend(
        f"unique{item_index:04d}token{position:02d}" for position in range(52)
    )
    return " ".join(words)


def synthetic_posts(per_topic: int = 320):
    posts = []
    for topic_index, topic in enumerate(w2d.TOPICS):
        for item_index in range(per_topic):
            header = (
                f"From: source{item_index}@example.test\n"
                f"Subject: Synthetic post {item_index}\n"
                f"Newsgroups: {topic}\n"
            )
            body = _body(topic_index, item_index)
            if item_index % 2 == 0:
                # Deterministically supplies multiple structural rules, without
                # using any detector score.
                raw = (
                    header.replace(
                        f"Subject: Synthetic post {item_index}",
                        f"Subject: Re: Synthetic post {item_index}",
                    )
                    + "> quoted material alpha\n"
                    + "> quoted material beta\n"
                    + body
                    + "\n-- \nFAQ\n"
                )
            else:
                raw = header + body + "\n"
            posts.append(
                w2d.RawPost(
                    topic=topic,
                    filename=f"/synthetic/{topic}/{item_index:05d}",
                    raw_bytes=raw.encode("latin1"),
                )
            )

    # One exact cross-topic duplicate.  A3 requires one content group, the
    # lexicographically first (topic, source_id) member as canonical, and an
    # explicit exclusion for the other source.
    duplicate = posts[1]
    posts.append(
        w2d.RawPost(
            topic=w2d.TOPICS[2],
            filename="/synthetic/duplicate/99999",
            raw_bytes=duplicate.raw_bytes,
        )
    )
    # One canonical group that fails the unified base eligibility.
    posts.append(
        w2d.RawPost(
            topic=w2d.TOPICS[0],
            filename="/synthetic/short/99998",
            raw_bytes=b"Subject: tiny\nonly a few words",
        )
    )
    return posts


def recursive_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from recursive_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_keys(child)


class UnitRulesTest(unittest.TestCase):
    def test_content_only_group_identity_and_canonical_member(self):
        raw = (
            "Subject: duplicate\n"
            + " ".join(f"word{i}" for i in range(60))
        ).encode("latin1")
        posts = [
            w2d.RawPost(w2d.TOPICS[4], "/z/2", raw),
            w2d.RawPost(w2d.TOPICS[1], "/a/1", raw),
        ]
        groups, excluded, raw_n = w2d.group_raw_posts(
            posts, return_inventory=True
        )
        normalized = w2d.normalize_text(w2d.decode_raw(raw))
        self.assertEqual(raw_n, 2)
        self.assertEqual(len(groups), 1)
        self.assertEqual(
            groups[0].group_id,
            hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(groups[0].topic, w2d.TOPICS[4])
        self.assertEqual(groups[0].source_id, f"{w2d.TOPICS[4]}/2")
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["reason"], "duplicate_noncanonical")

    def test_structural_hard_negative_rules(self):
        text = (
            "Subject: Re: question\n"
            "Newsgroups: one,two\n"
            "> quote one\n"
            "> quote two\n"
            "repeat\nrepeat\nrepeat\n"
            "-- \n"
            "Frequently Asked Questions contents\n"
            "A person writes: the answer follows\n"
        )
        rules = set(w2d.structural_hard_negative_rules(text))
        self.assertEqual(
            rules,
            {
                "quoted_reply_lines",
                "reply_attribution_or_re_subject",
                "signature_footer_separator",
                "faq_boilerplate_or_repeated_lines",
                "cross_post_header",
            },
        )

    def test_longest_shared_ngram(self):
        self.assertEqual(
            w2d.longest_shared_token_ngram(
                "one two THREE four five", "zero two three four nine"
            ),
            3,
        )
        self.assertEqual(w2d.longest_shared_token_ngram("", "anything"), 0)

    def test_recipe_templates_are_the_frozen_w2r_variants(self):
        from realtext_workload import TEMPLATES

        self.assertEqual(tuple(TEMPLATES), w2d.RECIPE_TEMPLATES)

    def test_nonfinite_base_embedding_is_excluded_with_reason(self):
        text = " ".join(["TOPICMARKER0"] + [f"longword{i}" for i in range(60)])
        text += " " + ("padding " * 30)
        raw = (
            f"Subject: ordinary\nNewsgroups: {w2d.TOPICS[0]}\n{text}"
        ).encode("latin1")
        groups = w2d.group_raw_posts(
            [w2d.RawPost(w2d.TOPICS[0], "/one/1", raw)]
        )
        eligible, _, excluded, _ = w2d._base_eligible_groups(
            groups, NonfiniteEmbedder()
        )
        self.assertEqual(eligible, [])
        self.assertIn("full_nonfinite", excluded[0]["reasons"])

    def test_strict_json_rejects_nonstandard_and_overflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            for payload in (
                '{"value": NaN}',
                '{"value": Infinity}',
                '{"value": 1e999}',
                '{"value": 1, "value": 2}',
            ):
                path.write_text(payload, encoding="utf-8")
                with self.assertRaises(w2d.W2DDataError):
                    w2d.strict_json_load(path)
            path.write_text('{"value": 1e-13}', encoding="utf-8")
            self.assertEqual(w2d.strict_json_load(path), {"value": 1e-13})


class FullFreezeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.data1 = root / "one" / "data"
        cls.results1 = root / "one" / "results"
        cls.data2 = root / "two" / "data"
        cls.results2 = root / "two" / "results"
        cls.posts = synthetic_posts()
        cls.embedder1 = FakeEmbedder()
        cls.first = w2d.build_w2d_data(
            raw_posts=cls.posts,
            embedder=cls.embedder1,
            data_dir=cls.data1,
            results_dir=cls.results1,
        )
        cls.second = w2d.build_w2d_data(
            raw_posts=cls.posts,
            embedder=FakeEmbedder(),
            data_dir=cls.data2,
            results_dir=cls.results2,
        )
        cls.freeze = w2d.load_freeze(cls.data1)
        cls.safe = w2d.load_detector_inputs(cls.data1)
        cls.labels = w2d.load_labels(
            data_dir=cls.data1, results_dir=cls.results1
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_exact_allocation_and_unique_groups(self):
        counts = self.freeze["counts"]
        self.assertEqual(counts["reference"], 768)
        self.assertEqual(counts["calibration_clean"], 256)
        self.assertEqual(counts["calibration_recipe_poison"], 192)
        self.assertEqual(counts["test_clean"], 320)
        self.assertEqual(counts["test_recipe_poison"], 128)
        self.assertEqual(counts["test_natural_cover_poison"], 64)
        self.assertEqual(counts["source_controls"], 24)
        self.assertEqual(counts["total_items"], 1752)
        self.assertEqual(counts["unique_source_groups"], 1752)
        for topic in w2d.TOPICS:
            self.assertEqual(
                counts["per_topic"][topic], w2d.PER_TOPIC_COUNTS
            )
        allocated = self.freeze["allocation"]
        groups = [record["source_group"] for record in allocated]
        self.assertEqual(len(groups), len(set(groups)))

    def test_a3_inventory_and_eligibility_are_fully_accounted(self):
        inventory = self.freeze["source"]["inventory"]
        self.assertEqual(
            inventory["raw_post_count"],
            inventory["canonical_group_count"]
            + inventory["duplicate_noncanonical_count"],
        )
        self.assertEqual(
            inventory["canonical_group_count"],
            inventory["base_eligible_group_count"]
            + inventory["base_excluded_group_count"],
        )
        self.assertEqual(
            inventory["base_eligible_group_count"],
            inventory["allocated_group_count"]
            + inventory["eligible_unallocated_group_count"],
        )
        reasons = [record["reason"] for record in self.freeze["exclusions"]]
        self.assertIn("duplicate_noncanonical", reasons)
        self.assertIn("base_eligibility", reasons)
        selected = {
            record["source_group"] for record in self.freeze["allocation"]
        }
        eligibility = {
            record["group_id"]: record
            for record in self.freeze["base_eligibility"]["records"]
        }
        self.assertTrue(selected <= set(eligibility))
        for group_id in selected:
            record = eligibility[group_id]
            self.assertGreaterEqual(record["normalized_codepoints"], 300)
            self.assertLessEqual(record["normalized_codepoints"], 2000)
            self.assertGreaterEqual(record["word_count"], 30)
            self.assertEqual(
                set(record["embedding_sha256"]),
                {"full", "first_half", "second_half", "query"},
            )

    def test_hard_negative_selection_and_ordinary_exclusion(self):
        labels = self.labels["items"]
        for split, expected in (("calibration", 64), ("test", 128)):
            summary = self.freeze["hard_negative_selection"][split]
            self.assertEqual(summary["n"], expected)
            for key in summary["item_keys"]:
                self.assertTrue(labels[key]["hard_negative_rules"])
        for item in self.labels["items"].values():
            if item["stratum"] == "ordinary_clean":
                self.assertEqual(item["hard_negative_rules"], [])

    def test_safe_schema_contains_no_labels_queries_or_attack_metadata(self):
        document = self.safe["document"]
        self.assertFalse(set(recursive_keys(document)) & w2d.FORBIDDEN_SAFE_KEYS)
        self.assertEqual(
            document["model"]["revision"], w2d.MODEL_REVISION
        )
        for record in document["items"]:
            self.assertEqual(set(record), w2d.SAFE_ITEM_KEYS)
            self.assertRegex(record["item_key"], r"^[0-9a-f]{64}$")
        test_keys = document["sets"]["test_score_order"]
        self.assertEqual(len(test_keys), 512)
        self.assertEqual(
            test_keys,
            sorted(test_keys, key=w2d.score_order_key),
        )
        key = test_keys[0]
        detector_args = w2d.materialize_detector_item(self.safe, key)
        self.assertEqual(
            set(detector_args),
            {
                "normalized_text",
                "full_embedding",
                "source_evidence",
                "excluded_source_group",
                "model_revision",
            },
        )
        self.assertNotIn("item_key", detector_args)
        self.assertEqual(w2d.reference_exclusion_rows(self.safe, key), ())
        reference_key = document["reference_index"]["item_keys"][0]
        self.assertEqual(
            len(w2d.reference_exclusion_rows(self.safe, reference_key)), 1
        )

    def test_labels_are_separate_and_carry_protocol_queries(self):
        items = [
            {"item_key": key, **record}
            for key, record in self.labels["items"].items()
        ]
        recipes = [
            item for item in items if item["attack_family"] == "recipe"
        ]
        covers = [
            item
            for item in items
            if item["attack_family"] == "natural_cover_suffix_v1"
        ]
        self.assertEqual(len(recipes), 320)
        self.assertEqual(len(covers), 64)
        variants = {
            name: sum(item["attack_variant"] == name for item in recipes)
            for name in ("T0", "T1", "T2", "T3", "T4")
        }
        self.assertEqual(
            variants, {"T0": 64, "T1": 64, "T2": 64, "T3": 64, "T4": 64}
        )
        safe_by_key = {
            item["item_key"]: item for item in self.safe["document"]["items"]
        }
        for item in recipes + covers:
            self.assertIsNotNone(item["target_query"])
            self.assertEqual(len(item["target_query_embedding"]), w2d.MODEL_DIM)
        for item in covers:
            text = safe_by_key[item["item_key"]]["normalized_text"]
            self.assertTrue(text.endswith(w2d.NATURAL_COVER_SUFFIX))
            self.assertEqual(text.count(w2d.NATURAL_COVER_SUFFIX), 1)
            self.assertLess(
                w2d.longest_shared_token_ngram(
                    w2d.NATURAL_COVER_SUFFIX, item["target_query"]
                ),
                6,
            )

    def test_label_schema_builds_exact_calibration_and_test_manifests(self):
        from w2d_calibrate import build_phase_key_manifests

        manifests = build_phase_key_manifests(self.labels)
        self.assertEqual(len(manifests["calibration"]), 448)
        self.assertEqual(len(manifests["test"]), 512)
        self.assertEqual(len(manifests["source_control"]), 24)
        self.assertFalse(
            set(manifests["calibration"]) & set(manifests["test"])
        )

    def test_source_controls_are_distinct_and_refuse_by_safe_evidence(self):
        labels = {
            key: item
            for key, item in self.labels["items"].items()
            if item["source_control_kind"]
        }
        safe = {
            item["item_key"]: item for item in self.safe["document"]["items"]
        }
        kinds = {}
        for key, label in labels.items():
            kinds[label["source_control_kind"]] = (
                kinds.get(label["source_control_kind"], 0) + 1
            )
            credential, provenance, revision = safe[key]["source_evidence"]
            self.assertEqual(revision, w2d.MODEL_REVISION)
            self.assertFalse(credential and provenance)
        self.assertEqual(
            kinds,
            {
                "invalid_signature": 8,
                "provenance_conflict": 8,
                "unknown_source": 8,
            },
        )

    def test_artifacts_are_byte_deterministic_and_hash_verified(self):
        for name in (
            "freeze",
            "inputs_json",
            "inputs_npz",
            "labels_json",
            "labels_sha256",
        ):
            self.assertEqual(
                self.first["digests"][name], self.second["digests"][name]
            )
        line = (
            self.results1 / w2d.LABEL_FILENAMES["labels_sha256"]
        ).read_text(encoding="ascii")
        labels_path = self.results1 / w2d.LABEL_FILENAMES["labels_json"]
        self.assertEqual(
            line,
            f"{w2d.sha256_file(labels_path)}  "
            f"{w2d.LABEL_FILENAMES['labels_json']}\n",
        )

    def test_no_overwrite_refuses_before_embedding(self):
        untouched = FakeEmbedder()
        with self.assertRaises(FileExistsError):
            w2d.build_w2d_data(
                raw_posts=self.posts,
                embedder=untouched,
                data_dir=self.data1,
                results_dir=self.results1,
            )
        self.assertEqual(untouched.calls, 0)

    def test_tamper_is_rejected(self):
        path = self.data2 / w2d.SAFE_FILENAMES["inputs_json"]
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(w2d.W2DDataError):
            w2d.load_detector_inputs(self.data2)

    def test_model_revision_and_finite_artifacts(self):
        self.assertEqual(
            self.freeze["model"],
            {
                "name": w2d.MODEL_NAME,
                "revision": w2d.MODEL_REVISION,
                "dimension": w2d.MODEL_DIM,
            },
        )
        self.assertTrue(np.isfinite(self.safe["embeddings"]).all())
        # Re-serialization with allow_nan=False is also a recursive finite check
        # over the JSON-decoded numeric values.
        json.dumps(self.freeze, allow_nan=False)
        json.dumps(self.labels, allow_nan=False)


class FailureBeforeFreezeTest(unittest.TestCase):
    def test_insufficient_pool_writes_nothing(self):
        posts = synthetic_posts(per_topic=4)
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            results = Path(tmp) / "results"
            with self.assertRaises(w2d.W2DDataError):
                w2d.build_w2d_data(
                    raw_posts=posts,
                    embedder=FakeEmbedder(),
                    data_dir=data,
                    results_dir=results,
                )
            self.assertFalse(any(data.glob("*")) if data.exists() else False)
            self.assertFalse(any(results.glob("*")) if results.exists() else False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
