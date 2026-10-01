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
                 consistency_level="Strong", index_type="FLAT",
                 index_params=None, search_params=None):
        """index_params/search_params carry ANN knobs for the standalone S5 sweep
        (e.g. index_params={"M": 16, "efConstruction": 200}, search_params={"ef": 64}).
        Both default to None, which reproduces the FLAT/exact behaviour used by
        Gates 1-4 and Milvus Lite, so existing callers are unaffected."""
        from pymilvus import MilvusClient, DataType
        self.client = MilvusClient(uri=uri)
        self.coll = collection
        self.dim = dim
        self.consistency = consistency_level
        self.index_type = index_type
        self.search_params = dict(search_params) if search_params else None
        self._vec = {}  # id -> vector, so set_visible can upsert the full entity

        if self.client.has_collection(collection):
            self.client.drop_collection(collection)
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("visible", DataType.BOOL)
        ip = self.client.prepare_index_params()
        ip.add_index(field_name="vector", metric_type="COSINE", index_type=index_type,
                     **(index_params or {}))
        self.client.create_collection(collection, schema=schema, index_params=ip,
                                      consistency_level=consistency_level)
        self.client.load_collection(collection)

    def insert(self, id, vector, visible=True):
        v = [float(x) for x in vector]
        self._vec[id] = v
        self.client.insert(self.coll, {"id": int(id), "vector": v, "visible": bool(visible)})

    def flush(self):
        self.client.flush(self.coll)

    def session_id(self):
        """Identity of the underlying client object. Session consistency uses a
        cached last-write timestamp as the guarantee ts, so writer and probes
        must share that cache for a Session cell to mean anything.

        Measured on pymilvus 2.4.15: the cache (`ts_utils.GTsDict`) is a
        PROCESS-WIDE singleton keyed by collection name, not per client -- two
        MilvusClient objects in one process already share it. Sharing one client
        is the stronger condition and is what we do; this id is recorded so a
        future refactor that splits the writer and probes across processes is
        visible in the data rather than silently changing what Session means."""
        return id(self.client)

    def point_get(self, id, consistency=None):
        """Scalar point lookup of the visibility flag -- a lightweight read that
        does NOT run ANN search. Pairing it with search() separates state
        visibility from query-path confirmation, which a search-only probe
        cannot do (the search's own latency is the same order as the delay)."""
        kw = {"consistency_level": consistency} if consistency else {}
        rows = self.client.query(self.coll, filter=f"id == {int(id)}",
                                 output_fields=["visible"], limit=1, **kw)
        return rows[0]["visible"] if rows else None

    def index_info(self):
        """What is ACTUALLY serving queries -- not what we asked for."""
        info = {"index_type_effective": None, "index_state": None, "load_state": None}
        try:
            d = self.client.describe_index(self.coll, "vector")
            info["index_type_effective"] = d.get("index_type")
            info["index_state"] = str(d.get("state", d.get("index_state", "")))
        except Exception as e:
            info["index_state"] = f"unreadable: {e!r}"
        try:
            info["load_state"] = str(self.client.get_load_state(self.coll).get("state"))
        except Exception:
            pass
        return info

    def server_info(self, uri):
        try:
            ver = self.client.get_server_version()
        except Exception:
            ver = None
        return {"server_version": ver, "server_uri": uri,
                "deployment_mode": "lite" if not str(uri).startswith("http") else "standalone"}

    def wait_index(self, timeout=180, poll=2.0):
        """Seal + build + reload, then report whether an ANN index is actually
        serving queries.

        Without this, freshly inserted vectors stay in a growing segment that
        Milvus brute-force scans, so an HNSW collection returns exactly the FLAT
        result and an ANN sweep silently measures nothing. Returns the index
        state string it observed (None if it could not be read)."""
        import time as _t
        self.flush()
        state, end = None, _t.time() + timeout
        while _t.time() < end:
            try:
                d = self.client.describe_index(self.coll, "vector")
                state = str(d.get("state", d.get("index_state", ""))) or None
                if state and state.lower() in ("finished", "indexstatefinished", "3"):
                    break
            except Exception:
                pass
            _t.sleep(poll)
        try:                                  # reload so the built index is served
            self.client.release_collection(self.coll)
            self.client.load_collection(self.coll)
        except Exception:
            pass
        return state

    def insert_many(self, rows):
        """Batch load (corpus setup only). Per-item ingestion throughput is measured
        with insert(), because the protocol is per-item."""
        data = []
        for i, vec, vis in rows:
            v = [float(x) for x in vec]
            self._vec[i] = v
            data.append({"id": int(i), "vector": v, "visible": bool(vis)})
        self.client.insert(self.coll, data)

    def set_visible(self, id, visible):
        v = self._vec[id]
        self.client.upsert(self.coll, {"id": int(id), "vector": v, "visible": bool(visible)})

    def delete(self, id):
        self.client.delete(self.coll, filter=f"id == {int(id)}")
        self._vec.pop(id, None)

    def search(self, qvec, k, use_index_filter=False, consistency=None):
        # `consistency` sends the level ON THE REQUEST. Without it pymilvus takes
        # the use_default branch of construct_guarantee_ts() and stamps the
        # request with a cached mutation ts, so the request itself carries no
        # level and a consistency sweep can silently degenerate into four
        # identical cells. Verified on 2.4.15: MilvusClient.search forwards
        # **kwargs and Prepare.search_requests_with_expr reads consistency_level.
        flt = "visible == true" if use_index_filter else ""
        kw = {}
        if consistency:
            kw["consistency_level"] = consistency
        if self.search_params:
            kw["search_params"] = {"metric_type": "COSINE", "params": self.search_params}
        res = self.client.search(self.coll, data=[[float(x) for x in qvec]],
                                 limit=k, filter=flt, output_fields=["id"], **kw)
        return [hit["id"] for hit in res[0]]
