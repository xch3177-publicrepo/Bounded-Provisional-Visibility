"""
Milvus adapter for VectorBackend.

STATUS: UNVALIDATED until the first Docker run (Gate 1). Written against the
pymilvus MilvusClient API (>= 2.4). The first successful
`python3 functional_slice.py --backend milvus` IS its validation; treat any
failure here as an adapter bug to fix, not a protocol result.

Design notes:
- The `visible` BOOL scalar field powers the in-index eligibility mode
  (search filter `visible == true`). The post-retrieval mode ignores it and
  filters raw candidates against the control store instead.
- Milvus upsert replaces the whole entity, so we cache vectors to re-upsert on
  set_visible without re-embedding.
- consistency_level is the delta_query knob for the in-index mode: "Strong"
  minimizes staleness (Gate 1 correctness), "Bounded"/"Eventual" trade freshness
  for latency (Gate 2/3 sweep). Fix and report it (§6.6).
"""

from backend import VectorBackend


class MilvusBackend(VectorBackend):
    def __init__(self, dim, uri="http://localhost:19530", collection="pv_slice",
                 consistency_level="Strong", index_type="FLAT"):
        from pymilvus import MilvusClient, DataType
        self.client = MilvusClient(uri=uri)
        self.coll = collection
        self.dim = dim
        self.consistency = consistency_level
        self._vec = {}  # id -> vector, so set_visible can upsert the full entity

        if self.client.has_collection(collection):
            self.client.drop_collection(collection)
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("visible", DataType.BOOL)
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="vector", metric_type="COSINE", index_type=index_type)
        self.client.create_collection(collection, schema=schema,
                                      index_params=index_params,
                                      consistency_level=consistency_level)
        self.client.load_collection(collection)

    def insert(self, id, vector, visible=True):
        v = [float(x) for x in vector]
        self._vec[id] = v
        self.client.insert(self.coll, {"id": int(id), "vector": v, "visible": bool(visible)})

    def set_visible(self, id, visible):
        v = self._vec[id]
        self.client.upsert(self.coll, {"id": int(id), "vector": v, "visible": bool(visible)})

    def delete(self, id):
        self.client.delete(self.coll, filter=f"id == {int(id)}")
        self._vec.pop(id, None)

    def search(self, qvec, k, use_index_filter=False):
        # consistency is set at collection creation (a per-search kwarg is not
        # accepted by MilvusClient.search in 2.4.x).
        flt = "visible == true" if use_index_filter else ""
        res = self.client.search(self.coll, data=[[float(x) for x in qvec]],
                                 limit=k, filter=flt, output_fields=["id"])
        return [hit["id"] for hit in res[0]]
