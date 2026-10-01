#!/usr/bin/env python3
"""Unit tests for the E3-only Milvus bulk-load and cleanup helpers."""

from __future__ import annotations

import unittest

from milvus_backend import MilvusBackend


class _FakeClient:
    def __init__(self):
        self.inserted = []
        self.deleted_filters = []
        self.query_rows = []
        self.flushes = 0

    def insert(self, collection, data):
        self.inserted.append((collection, data))

    def delete(self, collection, *, filter):
        self.deleted_filters.append((collection, filter))

    def query(self, collection, *, filter, output_fields, limit):
        self.last_query = {
            "collection": collection,
            "filter": filter,
            "output_fields": output_fields,
            "limit": limit,
        }
        return list(self.query_rows)

    def flush(self, collection):
        self.flushes += 1

    def describe_index(self, collection, field):
        return {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "M": "16",
            "efConstruction": "200",
            "indexed_rows": "100000",
            "pending_index_rows": "0",
            "state": "Finished",
        }

    def get_load_state(self, collection):
        return {"state": "Loaded"}


def _backend(client: _FakeClient) -> MilvusBackend:
    backend = object.__new__(MilvusBackend)
    backend.client = client
    backend.coll = "e3"
    backend._vec = {}
    return backend


class MilvusBackendE3Tests(unittest.TestCase):
    def test_bulk_static_load_can_skip_python_vector_cache(self):
        client = _FakeClient()
        backend = _backend(client)

        backend.insert_many(
            [(1, [1.0, 0.0], True), (2, [0.0, 1.0], True)],
            cache_vectors=False,
        )

        self.assertEqual(backend._vec, {})
        self.assertEqual(
            client.inserted[0][1],
            [
                {"id": 1, "vector": [1.0, 0.0], "visible": True},
                {"id": 2, "vector": [0.0, 1.0], "visible": True},
            ],
        )

    def test_bulk_protocol_load_keeps_default_cache_for_visibility_upsert(self):
        client = _FakeClient()
        backend = _backend(client)

        backend.insert_many([(7, [0.25, 0.75], False)])

        self.assertEqual(backend._vec, {7: [0.25, 0.75]})

    def test_ids_in_range_uses_half_open_primary_key_filter(self):
        client = _FakeClient()
        client.query_rows = [{"id": 12}, {"id": 10}, {"id": 12}]
        backend = _backend(client)

        self.assertEqual(backend.ids_in_range(10, 20), [10, 12])
        self.assertEqual(client.last_query["filter"], "id >= 10 and id < 20")

    def test_delete_range_confirms_absence_and_clears_only_dynamic_cache(self):
        client = _FakeClient()
        client.query_rows = []
        backend = _backend(client)
        backend._vec = {2: [1.0], 101: [2.0], 102: [3.0], 999: [4.0]}

        result = backend.delete_range(100, 200, timeout=0.1, poll=0.0)

        self.assertTrue(result["delete_confirmed"])
        self.assertEqual(result["delete_remaining_n"], 0)
        self.assertEqual(client.deleted_filters, [("e3", "id >= 100 and id < 200")])
        self.assertEqual(backend._vec, {2: [1.0], 999: [4.0]})
        self.assertEqual(client.flushes, 1)

    def test_delete_range_rejects_empty_or_reversed_interval(self):
        backend = _backend(_FakeClient())
        with self.assertRaisesRegex(ValueError, "hi > lo"):
            backend.delete_range(3, 3)
        with self.assertRaisesRegex(ValueError, "hi > lo"):
            backend.delete_range(4, 3)

    def test_index_info_reports_effective_server_side_hnsw_parameters(self):
        backend = _backend(_FakeClient())

        self.assertEqual(
            backend.index_info(),
            {
                "index_type_effective": "HNSW",
                "metric_type_effective": "COSINE",
                "M_effective": 16,
                "efConstruction_effective": 200,
                "indexed_rows": 100_000,
                "pending_index_rows": 0,
                "index_state": "Finished",
                "load_state": "Loaded",
            },
        )


if __name__ == "__main__":
    unittest.main()
