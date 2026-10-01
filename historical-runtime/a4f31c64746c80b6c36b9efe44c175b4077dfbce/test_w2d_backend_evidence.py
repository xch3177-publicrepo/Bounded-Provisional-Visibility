"""Offline contract tests for W2D backend evidence receipts."""

from __future__ import annotations

import copy
import json
import math
import struct
import unittest

from grpc import StatusCode

from w2d_backend_evidence import (
    BackendEvidenceError,
    COLLECTION_FIELD_FIELDS,
    COLLECTION_FIELDS,
    ENDPOINT_FIELDS,
    EXPECTED_SUMMARY_FIELDS,
    INDEX_FIELDS,
    LOAD_FIELDS,
    SCHEMA_VERSION,
    SERVER_FIELDS,
    TERMINAL_FIELDS,
    TOP_LEVEL_FIELDS,
    capture_backend_evidence,
    summarize_expected_seed_state,
    validate_backend_evidence_pair,
)


DIM = 384
SECRET_URI = "/private/tmp/person-name-and-secret-token.db"


def vector(value: float) -> list[float]:
    return [value] * DIM


def rows() -> list[dict[str, object]]:
    return [
        {"id": 2, "visible": False, "vector": vector(0.25)},
        {"id": 1, "visible": True, "vector": vector(0.5)},
    ]


class FakeInMemoryBackend:
    def __init__(self):
        self.v = {
            2: vector(0.25),
            1: vector(0.5),
        }
        self.vis = {2: False, 1: True}


class FakeMilvusClient:
    def __init__(self, terminal_rows=None):
        self.terminal_rows = rows() if terminal_rows is None else terminal_rows
        self.calls: list[tuple[str, object]] = []

    def get_server_version(self):
        self.calls.append(("get_server_version", None))
        return "v2.4.15-lite"

    def describe_collection(self, collection):
        self.calls.append(("describe_collection", collection))
        return {
            "collection_name": collection,
            "auto_id": False,
            "enable_dynamic_field": False,
            "consistency_level": 0,
            "fields": [
                {
                    "field_id": 100,
                    "name": "id",
                    "type": "DataType.INT64",
                    "is_primary": True,
                    "auto_id": False,
                    "params": {},
                },
                {
                    "field_id": 101,
                    "name": "vector",
                    "type": "DataType.FLOAT_VECTOR",
                    "params": {"dim": DIM},
                },
                {
                    "field_id": 102,
                    "name": "visible",
                    "type": "DataType.BOOL",
                    "params": {},
                },
            ],
            "created_timestamp": 123456789,
        }

    def describe_index(self, collection, field_name):
        self.calls.append(("describe_index", (collection, field_name)))
        return {
            "field_name": field_name,
            # This is the actual default produced by IndexParams.add_index()
            # when the backend does not supply index_name.
            "index_name": "",
            "index_type": "FLAT",
            "metric_type": "COSINE",
            "state": "Finished",
            "total_rows": 2,
            "indexed_rows": 2,
            "pending_index_rows": 0,
            "params": {},
        }

    def get_load_state(self, collection):
        self.calls.append(("get_load_state", collection))
        return {"state": "LoadState.Loaded"}

    def query(self, collection, **kwargs):
        self.calls.append(("query", (collection, kwargs)))
        return self.terminal_rows


class FakeMilvusBackend:
    def __init__(self, client=None):
        self.client = FakeMilvusClient() if client is None else client
        self.coll = "w2d_poison"
        self.dim = DIM


class RaisingMilvusClient(FakeMilvusClient):
    def get_server_version(self):
        raise RuntimeError(f"endpoint failed: {SECRET_URI}")


class FakeGrpcError(RuntimeError):
    def __init__(self, code):
        super().__init__(f"typed RPC failure at {SECRET_URI}")
        self._code = code

    def code(self):
        return self._code


class UnimplementedVersionClient(FakeMilvusClient):
    def get_server_version(self):
        self.calls.append(("get_server_version", None))
        raise FakeGrpcError(StatusCode.UNIMPLEMENTED)


class UnavailableVersionClient(FakeMilvusClient):
    def get_server_version(self):
        self.calls.append(("get_server_version", None))
        raise FakeGrpcError(StatusCode.UNAVAILABLE)


class BackendEvidenceTest(unittest.TestCase):
    def assert_exact_shape(self, receipt):
        self.assertEqual(set(receipt), TOP_LEVEL_FIELDS)
        self.assertEqual(set(receipt["endpoint"]), ENDPOINT_FIELDS)
        if receipt["server"] is not None:
            self.assertEqual(set(receipt["server"]), SERVER_FIELDS)
        if receipt["collection"] is not None:
            self.assertEqual(set(receipt["collection"]), COLLECTION_FIELDS)
            for field in receipt["collection"]["fields"]:
                self.assertEqual(set(field), COLLECTION_FIELD_FIELDS)
        if receipt["index"] is not None:
            self.assertEqual(set(receipt["index"]), INDEX_FIELDS)
        if receipt["load"] is not None:
            self.assertEqual(set(receipt["load"]), LOAD_FIELDS)
        if receipt["terminal_state"] is not None:
            self.assertEqual(
                set(receipt["terminal_state"]), TERMINAL_FIELDS
            )
        json.dumps(receipt, allow_nan=False)

    def test_e1_pre_has_dynamic_identity_and_no_terminal_state(self):
        backend = FakeInMemoryBackend()
        receipt = capture_backend_evidence(
            "inmemory", backend, SECRET_URI, "pre"
        )
        self.assertEqual(receipt["schema_version"], SCHEMA_VERSION)
        self.assertEqual(
            receipt["backend_identity"],
            {
                "module": FakeInMemoryBackend.__module__,
                "class": FakeInMemoryBackend.__qualname__,
            },
        )
        self.assertIsNone(receipt["terminal_state"])
        self.assertIsNone(receipt["server"])
        self.assertIsNone(receipt["collection"])
        self.assertNotIn(SECRET_URI, json.dumps(receipt, sort_keys=True))
        self.assert_exact_shape(receipt)

    def test_e1_post_canonical_digest_and_expected_state(self):
        backend = FakeInMemoryBackend()
        expected = summarize_expected_seed_state(rows())
        self.assertEqual(set(expected), EXPECTED_SUMMARY_FIELDS)
        receipt = capture_backend_evidence(
            "inmemory",
            backend,
            SECRET_URI,
            "post",
            expected_seed_state=expected,
        )
        terminal = receipt["terminal_state"]
        self.assertEqual(terminal["source"], "in_memory_object")
        self.assertEqual(terminal["consistency"], "strong_in_process")
        self.assertEqual(terminal["row_count"], 2)
        self.assertEqual(terminal["visible_true_count"], 1)
        self.assertEqual(terminal["vector_dim"], DIM)
        self.assertEqual(
            terminal["logical_state_sha256"],
            expected["logical_state_sha256"],
        )
        self.assertTrue(terminal["expected_seed_state_checked"])
        self.assert_exact_shape(receipt)

        wrong = dict(expected)
        wrong["visible_true_count"] = 2
        with self.assertRaises(BackendEvidenceError):
            capture_backend_evidence(
                "inmemory",
                backend,
                None,
                "post",
                expected_seed_state=wrong,
            )

    def test_e2_pre_calls_metadata_only_and_omits_uri(self):
        client = FakeMilvusClient()
        backend = FakeMilvusBackend(client)
        receipt = capture_backend_evidence(
            "milvus", backend, SECRET_URI, "pre"
        )
        call_names = [name for name, _ in client.calls]
        self.assertEqual(
            call_names,
            [
                "get_server_version",
                "describe_collection",
                "describe_index",
                "get_load_state",
            ],
        )
        self.assertNotIn("query", call_names)
        self.assertEqual(
            receipt["backend_identity"]["class"],
            FakeMilvusBackend.__qualname__,
        )
        self.assertEqual(
            receipt["server"],
            {"status": "reported", "version": "v2.4.15-lite"},
        )
        self.assertEqual(receipt["collection"]["consistency_level"], "Strong")
        self.assertEqual(receipt["index"]["index_type"], "FLAT")
        self.assertEqual(receipt["index"]["index_name"], "")
        self.assertEqual(receipt["load"], {"state": "Loaded"})
        self.assertIsNone(receipt["terminal_state"])
        self.assertNotIn(SECRET_URI, json.dumps(receipt, sort_keys=True))
        self.assertNotIn("uri", receipt)
        self.assert_exact_shape(receipt)

    def test_e2_post_uses_one_strong_query_and_checks_expected_rows(self):
        client = FakeMilvusClient()
        backend = FakeMilvusBackend(client)
        receipt = capture_backend_evidence(
            "milvus",
            backend,
            SECRET_URI,
            "post",
            expected_seed_state={"rows": rows()},
        )
        query_calls = [
            detail for name, detail in client.calls if name == "query"
        ]
        self.assertEqual(len(query_calls), 1)
        collection, kwargs = query_calls[0]
        self.assertEqual(collection, "w2d_poison")
        self.assertEqual(
            kwargs,
            {
                "filter": "",
                "output_fields": ["id", "visible", "vector"],
                "limit": 8192,
                "consistency_level": "Strong",
            },
        )
        terminal = receipt["terminal_state"]
        self.assertEqual(terminal["source"], "milvus_strong_query")
        self.assertEqual(terminal["consistency"], "Strong")
        self.assertEqual(terminal["row_count"], 2)
        self.assertTrue(terminal["expected_seed_state_checked"])
        self.assertNotIn(SECRET_URI, json.dumps(receipt, sort_keys=True))
        self.assert_exact_shape(receipt)

    def test_e2_typed_unimplemented_server_version_is_controlled(self):
        client = UnimplementedVersionClient()
        backend = FakeMilvusBackend(client)
        expected = summarize_expected_seed_state(rows())
        pre = capture_backend_evidence(
            "milvus", backend, SECRET_URI, "pre"
        )
        post = capture_backend_evidence(
            "milvus",
            backend,
            SECRET_URI,
            "post",
            expected_seed_state=expected,
        )
        for receipt in (pre, post):
            self.assertEqual(
                receipt["server"],
                {"status": "unimplemented", "version": None},
            )
            self.assertNotIn(
                SECRET_URI, json.dumps(receipt, sort_keys=True)
            )
            receipt["backend_identity"] = {
                "module": "milvus_backend",
                "class": "MilvusBackend",
            }
        self.assertIsNone(
            validate_backend_evidence_pair(
                {"pre": pre, "post": post},
                backend_name="milvus",
                expected_post_state=expected,
            )
        )

    def test_digest_quantizes_runner_float64_and_milvus_float32_equally(self):
        binary64 = 1.0 / 3.0
        binary32 = struct.unpack("!f", struct.pack("!f", binary64))[0]
        self.assertNotEqual(binary64, binary32)
        runner_rows = [
            {"id": 1, "visible": True, "vector": vector(binary64)}
        ]
        milvus_rows = [
            {"id": 1, "visible": True, "vector": vector(binary32)}
        ]
        runner_summary = summarize_expected_seed_state(runner_rows)
        milvus_summary = summarize_expected_seed_state(milvus_rows)
        self.assertEqual(runner_summary, milvus_summary)

        receipt = capture_backend_evidence(
            "milvus",
            FakeMilvusBackend(FakeMilvusClient(milvus_rows)),
            SECRET_URI,
            "post",
            expected_seed_state=runner_summary,
        )
        self.assertEqual(
            receipt["terminal_state"]["logical_state_sha256"],
            runner_summary["logical_state_sha256"],
        )

    def test_pair_validator_accepts_only_complete_production_e1_pair(self):
        backend = FakeInMemoryBackend()
        expected = summarize_expected_seed_state(rows())
        pre = capture_backend_evidence(
            "inmemory", backend, SECRET_URI, "pre"
        )
        post = capture_backend_evidence(
            "inmemory",
            backend,
            SECRET_URI,
            "post",
            expected_seed_state=expected,
        )
        for receipt in (pre, post):
            receipt["backend_identity"] = {
                "module": "backend",
                "class": "InMemoryBackend",
            }
        pair = {"pre": pre, "post": post}
        self.assertIsNone(
            validate_backend_evidence_pair(
                pair,
                backend_name="inmemory",
                expected_post_state=expected,
            )
        )

        cases = {}
        cases["extra pair field"] = copy.deepcopy(pair)
        cases["extra pair field"]["unexpected"] = True
        cases["wrong phase"] = copy.deepcopy(pair)
        cases["wrong phase"]["pre"]["phase"] = "post"
        cases["fake class"] = copy.deepcopy(pair)
        cases["fake class"]["post"]["backend_identity"]["class"] = (
            "FakeInMemoryBackend"
        )
        cases["E2 metadata on E1"] = copy.deepcopy(pair)
        cases["E2 metadata on E1"]["pre"]["server"] = {"version": "v1"}
        cases["pre terminal"] = copy.deepcopy(pair)
        cases["pre terminal"]["pre"]["terminal_state"] = copy.deepcopy(
            pair["post"]["terminal_state"]
        )
        cases["unchecked post"] = copy.deepcopy(pair)
        cases["unchecked post"]["post"]["terminal_state"][
            "expected_seed_state_checked"
        ] = False
        cases["wrong expected digest"] = copy.deepcopy(pair)
        cases["wrong expected digest"]["post"]["terminal_state"][
            "logical_state_sha256"
        ] = "0" * 64

        for name, bad_pair in cases.items():
            with self.subTest(name):
                with self.assertRaises(BackendEvidenceError):
                    validate_backend_evidence_pair(
                        bad_pair,
                        backend_name="inmemory",
                        expected_post_state=expected,
                    )

    def test_pair_validator_enforces_production_e2_metadata(self):
        backend = FakeMilvusBackend()
        expected = summarize_expected_seed_state(rows())
        pre = capture_backend_evidence(
            "milvus", backend, SECRET_URI, "pre"
        )
        post = capture_backend_evidence(
            "milvus",
            backend,
            SECRET_URI,
            "post",
            expected_seed_state=expected,
        )
        for receipt in (pre, post):
            receipt["backend_identity"] = {
                "module": "milvus_backend",
                "class": "MilvusBackend",
            }
        pair = {"pre": pre, "post": post}
        self.assertIsNone(
            validate_backend_evidence_pair(
                pair,
                backend_name="milvus",
                expected_post_state=expected,
            )
        )

        cases = {}
        cases["wrong metric"] = copy.deepcopy(pair)
        cases["wrong metric"]["post"]["index"]["metric_type"] = "L2"
        cases["not loaded"] = copy.deepcopy(pair)
        cases["not loaded"]["pre"]["load"]["state"] = "Loading"
        cases["wrong schema consistency"] = copy.deepcopy(pair)
        cases["wrong schema consistency"]["post"]["collection"][
            "consistency_level"
        ] = "Session"
        cases["server drift"] = copy.deepcopy(pair)
        cases["server drift"]["post"]["server"]["version"] = "v2.4.16-lite"
        cases["index identity drift"] = copy.deepcopy(pair)
        cases["index identity drift"]["post"]["index"]["index_name"] = (
            "other_index"
        )

        for name, bad_pair in cases.items():
            with self.subTest(name):
                with self.assertRaises(BackendEvidenceError):
                    validate_backend_evidence_pair(
                        bad_pair,
                        backend_name="milvus",
                        expected_post_state=expected,
                    )

    def test_live_exception_fails_closed_without_leaking_uri(self):
        for client in (RaisingMilvusClient(), UnavailableVersionClient()):
            with self.subTest(client=type(client).__name__):
                backend = FakeMilvusBackend(client)
                with self.assertRaises(BackendEvidenceError) as caught:
                    capture_backend_evidence(
                        "milvus", backend, SECRET_URI, "pre"
                    )
                self.assertEqual(
                    str(caught.exception),
                    "live get_server_version call failed",
                )
                self.assertNotIn(SECRET_URI, str(caught.exception))

    def test_malformed_and_nonfinite_state_fail_closed(self):
        with self.subTest("mismatched E1 maps"):
            backend = FakeInMemoryBackend()
            del backend.vis[2]
            with self.assertRaises(BackendEvidenceError):
                capture_backend_evidence(
                    "inmemory", backend, None, "post"
                )

        with self.subTest("non-finite E2 vector"):
            bad_rows = rows()
            bad_rows[0]["vector"] = vector(math.nan)
            backend = FakeMilvusBackend(FakeMilvusClient(bad_rows))
            with self.assertRaises(BackendEvidenceError):
                capture_backend_evidence(
                    "milvus", backend, SECRET_URI, "post"
                )

        with self.subTest("remote endpoint"):
            backend = FakeMilvusBackend()
            with self.assertRaises(BackendEvidenceError):
                capture_backend_evidence(
                    "milvus",
                    backend,
                    "http://localhost:19530",
                    "pre",
                )
            self.assertEqual(backend.client.calls, [])

        with self.subTest("expected state forbidden at pre"):
            backend = FakeInMemoryBackend()
            with self.assertRaises(BackendEvidenceError):
                capture_backend_evidence(
                    "inmemory",
                    backend,
                    None,
                    "pre",
                    expected_seed_state=rows(),
                )


if __name__ == "__main__":
    unittest.main()
