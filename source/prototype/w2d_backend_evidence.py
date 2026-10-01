"""Fail-closed backend receipts for the W2D detector experiment.

The receipt is deliberately descriptive rather than an attestation.  Callers
must invoke it before or after a measured window; this module never belongs in
the timed path.  In particular, the Milvus ``post`` receipt performs one
Strong-consistency query over the final logical rows, while the ``pre`` receipt
does not query rows at all.

No endpoint value is serialized.  For Milvus Lite, ``uri`` is used only to
verify that the requested backend is a local ``.db`` deployment.

``expected_seed_state`` is optional and may be either:

* the exact four-field summary returned by
  :func:`summarize_expected_seed_state`; or
* a sequence of row mappings (or ``{"rows": sequence}``) with exactly the
  fields ``id``, ``visible``, and ``vector``.

If supplied, a mismatch is an error rather than a receipt with a soft warning.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import numbers
import os
import re
import struct
from typing import Any


SCHEMA_VERSION = "W2D-backend-evidence-v1"
EXPECTED_VECTOR_DIM = 384
MAX_LOGICAL_ROWS = 8192
PAIR_FIELDS = frozenset({"pre", "post"})

TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "phase",
        "backend_name",
        "backend_identity",
        "endpoint",
        "evidence_boundary",
        "server",
        "collection",
        "index",
        "load",
        "terminal_state",
    }
)
IDENTITY_FIELDS = frozenset({"module", "class"})
ENDPOINT_FIELDS = frozenset({"kind", "absolute_uri_omitted"})
COLLECTION_FIELDS = frozenset(
    {
        "name",
        "auto_id",
        "enable_dynamic_field",
        "consistency_level",
        "fields",
    }
)
COLLECTION_FIELD_FIELDS = frozenset(
    {"name", "data_type", "is_primary", "auto_id", "dimension"}
)
INDEX_FIELDS = frozenset(
    {
        "field_name",
        "index_name",
        "index_type",
        "metric_type",
        "state",
        "total_rows",
        "indexed_rows",
        "pending_index_rows",
    }
)
LOAD_FIELDS = frozenset({"state"})
SERVER_FIELDS = frozenset({"status", "version"})
TERMINAL_FIELDS = frozenset(
    {
        "source",
        "consistency",
        "row_count",
        "visible_true_count",
        "vector_dim",
        "logical_state_sha256",
        "expected_seed_state_checked",
    }
)
EXPECTED_SUMMARY_FIELDS = frozenset(
    {
        "row_count",
        "visible_true_count",
        "vector_dim",
        "logical_state_sha256",
    }
)
ROW_FIELDS = frozenset({"id", "visible", "vector"})

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_DATA_TYPE_BY_NUMBER = {
    1: "BOOL",
    5: "INT64",
    101: "FLOAT_VECTOR",
}
_CONSISTENCY_BY_NUMBER = {
    0: "Strong",
    1: "Session",
    2: "Bounded",
    3: "Eventually",
    4: "Customized",
}
_EXPECTED_FIELD_ORDER = ("id", "vector", "visible")
_EXPECTED_FIELD_TYPES = {
    "id": "INT64",
    "vector": "FLOAT_VECTOR",
    "visible": "BOOL",
}
_EVIDENCE_BOUNDARY = (
    "live_descriptive_receipt_not_attestation_outside_measurement_window"
)
_DIGEST_DOMAIN = b"W2D-BACKEND-LOGICAL-STATE-v1\0"
_PRODUCTION_IDENTITIES = {
    "inmemory": {"module": "backend", "class": "InMemoryBackend"},
    "milvus": {"module": "milvus_backend", "class": "MilvusBackend"},
}
_EXPECTED_ENDPOINTS = {
    "inmemory": {
        "kind": "process_local",
        "absolute_uri_omitted": True,
    },
    "milvus": {
        "kind": "milvus_lite_local_file",
        "absolute_uri_omitted": True,
    },
}


class BackendEvidenceError(RuntimeError):
    """Raised when live evidence cannot be collected or validated."""


def _fail(message: str) -> None:
    raise BackendEvidenceError(message)


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{path} must be a mapping")
    return value


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    path: str,
) -> None:
    if set(value) != expected:
        _fail(f"{path} has the wrong fields")
    if not all(type(key) is str for key in value):
        _fail(f"{path} keys must be strings")


def _require_bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        _fail(f"{path} must be a boolean")
    return value


def _require_int(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        _fail(f"{path} must be an integer")
    result = int(value)
    if result < minimum:
        _fail(f"{path} is outside its allowed range")
    return result


def _optional_int(value: Any, path: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, path)


def _require_nonempty_string(value: Any, path: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{path} must be a non-empty canonical string")
    return value


def _require_string(value: Any, path: str) -> str:
    if type(value) is not str or value != value.strip():
        _fail(f"{path} must be a canonical string")
    return value


def _enum_token(value: Any, path: str) -> str:
    name = getattr(value, "name", None)
    if type(name) is str and name:
        return name.rsplit(".", 1)[-1]
    if type(value) is str:
        token = _require_nonempty_string(value, path)
        return token.rsplit(".", 1)[-1]
    if isinstance(value, bool):
        _fail(f"{path} has an invalid enum value")
    if isinstance(value, numbers.Integral):
        return str(int(value))
    _fail(f"{path} has an invalid enum value")


def _normalize_data_type(value: Any, path: str) -> str:
    token = _enum_token(value, path)
    if token.isdecimal():
        result = _DATA_TYPE_BY_NUMBER.get(int(token))
    else:
        result = token.upper()
    if result not in {"BOOL", "INT64", "FLOAT_VECTOR"}:
        _fail(f"{path} has an unsupported data type")
    return result


def _normalize_consistency(value: Any, path: str) -> str:
    token = _enum_token(value, path)
    if token.isdecimal():
        result = _CONSISTENCY_BY_NUMBER.get(int(token))
    else:
        result = token.capitalize()
    if result not in set(_CONSISTENCY_BY_NUMBER.values()):
        _fail(f"{path} has an unsupported consistency level")
    return result


def _normalize_state(value: Any, path: str) -> str:
    return _enum_token(value, path)


def _normalize_index_ready_state(value: Any, path: str) -> str:
    token = _normalize_state(value, path).lower()
    if token not in {"finished", "indexstatefinished", "3"}:
        _fail(f"{path} is not ready")
    return "Finished"


def _normalize_load_ready_state(value: Any, path: str) -> str:
    token = _normalize_state(value, path).lower()
    if token not in {"loaded", "loadstateloaded", "3"}:
        _fail(f"{path} is not loaded")
    return "Loaded"


def _normalize_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        _fail(f"{path} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{path} must be a finite real number")
    # Milvus FLOAT_VECTOR returns binary32 values, while the frozen runner
    # vectors are Python/binary64 floats.  Hashing the latter directly makes
    # identical logical rows disagree solely because they crossed the backend
    # boundary.  Quantize both sides to IEEE-754 binary32 before canonical JSON.
    try:
        result = struct.unpack("!f", struct.pack("!f", result))[0]
    except (OverflowError, struct.error) as exc:
        raise BackendEvidenceError(
            f"{path} is outside the finite binary32 range"
        ) from exc
    if not math.isfinite(result):
        _fail(f"{path} is outside the finite binary32 range")
    return result


def _normalize_row(value: Any, path: str) -> dict[str, Any]:
    row = _require_mapping(value, path)
    _require_exact_fields(row, ROW_FIELDS, path)
    row_id = _require_int(row["id"], f"{path}.id")
    visible = _require_bool(row["visible"], f"{path}.visible")
    vector = row["vector"]
    if (
        not isinstance(vector, Sequence)
        or isinstance(vector, (str, bytes, bytearray))
    ):
        _fail(f"{path}.vector must be a sequence")
    if len(vector) != EXPECTED_VECTOR_DIM:
        _fail(f"{path}.vector has the wrong dimension")
    normalized_vector = [
        _normalize_float(component, f"{path}.vector[{index}]")
        for index, component in enumerate(vector)
    ]
    return {
        "id": row_id,
        "visible": visible,
        "vector": normalized_vector,
    }


def _normalize_rows(rows: Any, path: str) -> list[dict[str, Any]]:
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes, bytearray))
    ):
        _fail(f"{path} must be a sequence")
    if not rows:
        _fail(f"{path} must contain at least one row")
    if len(rows) >= MAX_LOGICAL_ROWS:
        _fail(f"{path} is at or above the safe query limit")
    normalized = [
        _normalize_row(row, f"{path}[{index}]")
        for index, row in enumerate(rows)
    ]
    normalized.sort(key=lambda row: row["id"])
    ids = [row["id"] for row in normalized]
    if len(ids) != len(set(ids)):
        _fail(f"{path} contains duplicate ids")
    return normalized


def _state_summary(rows: Any, path: str) -> dict[str, Any]:
    normalized = _normalize_rows(rows, path)
    canonical = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(_DIGEST_DOMAIN + canonical).hexdigest()
    return {
        "row_count": len(normalized),
        "visible_true_count": sum(
            1 for row in normalized if row["visible"]
        ),
        "vector_dim": EXPECTED_VECTOR_DIM,
        "logical_state_sha256": digest,
    }


def summarize_expected_seed_state(rows: Any) -> dict[str, Any]:
    """Return the exact summary accepted by ``expected_seed_state``."""

    result = _state_summary(rows, "expected_seed_state")
    _validate_json_tree(result, "expected_seed_state_summary")
    return result


def _validate_expected_summary(value: Any) -> dict[str, Any]:
    summary = _require_mapping(value, "expected_seed_state")
    _require_exact_fields(
        summary, EXPECTED_SUMMARY_FIELDS, "expected_seed_state"
    )
    row_count = _require_int(
        summary["row_count"], "expected_seed_state.row_count", minimum=1
    )
    visible_true_count = _require_int(
        summary["visible_true_count"],
        "expected_seed_state.visible_true_count",
    )
    if visible_true_count > row_count:
        _fail("expected_seed_state has an impossible visible count")
    vector_dim = _require_int(
        summary["vector_dim"],
        "expected_seed_state.vector_dim",
        minimum=1,
    )
    if vector_dim != EXPECTED_VECTOR_DIM:
        _fail("expected_seed_state has the wrong vector dimension")
    digest = _require_nonempty_string(
        summary["logical_state_sha256"],
        "expected_seed_state.logical_state_sha256",
    )
    if _SHA256_RE.fullmatch(digest) is None:
        _fail("expected_seed_state has an invalid logical-state digest")
    return {
        "row_count": row_count,
        "visible_true_count": visible_true_count,
        "vector_dim": vector_dim,
        "logical_state_sha256": digest,
    }


def _expected_summary(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        if set(value) == EXPECTED_SUMMARY_FIELDS:
            return _validate_expected_summary(value)
        if set(value) == {"rows"} and all(type(key) is str for key in value):
            return _state_summary(value["rows"], "expected_seed_state.rows")
        _fail("expected_seed_state has the wrong fields")
    return _state_summary(value, "expected_seed_state")


def _terminal_state(
    *,
    rows: Any,
    source: str,
    consistency: str,
    expected_seed_state: Any,
) -> dict[str, Any]:
    summary = _state_summary(rows, "terminal_rows")
    expected = _expected_summary(expected_seed_state)
    if expected is not None and summary != expected:
        _fail("terminal state does not match expected_seed_state")
    return {
        "source": source,
        "consistency": consistency,
        **summary,
        "expected_seed_state_checked": expected is not None,
    }


def _backend_identity(backend: Any) -> dict[str, str]:
    try:
        cls = type(backend)
        module = cls.__module__
        class_name = cls.__qualname__
    except Exception as exc:
        raise BackendEvidenceError(
            "backend identity could not be inspected"
        ) from exc
    return {
        "module": _require_nonempty_string(
            module, "backend_identity.module"
        ),
        "class": _require_nonempty_string(
            class_name, "backend_identity.class"
        ),
    }


def _live_call(label: str, operation: Any) -> Any:
    try:
        return operation()
    except Exception as exc:
        raise BackendEvidenceError(f"live {label} call failed") from exc


def _server_receipt(client: Any) -> dict[str, Any]:
    try:
        version_value = client.get_server_version()
    except Exception as exc:
        # Milvus Lite 2.4.12 exposes the method through MilvusClient but its
        # embedded server does not implement the corresponding gRPC endpoint.
        # This one typed capability absence is evidence, not a generic success.
        try:
            from grpc import StatusCode as GrpcStatusCode

            code = exc.code()
        except Exception:
            GrpcStatusCode = None
            code = None
        if (
            GrpcStatusCode is not None
            and code is GrpcStatusCode.UNIMPLEMENTED
        ):
            return {"status": "unimplemented", "version": None}
        raise BackendEvidenceError(
            "live get_server_version call failed"
        ) from exc
    version = _require_nonempty_string(
        version_value, "get_server_version"
    )
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", version) is None:
        _fail("get_server_version returned a non-canonical version")
    return {"status": "reported", "version": version}


def _capture_inmemory(
    backend: Any,
    *,
    phase: str,
    expected_seed_state: Any,
) -> tuple[
    None,
    None,
    None,
    None,
    dict[str, Any] | None,
]:
    if phase == "pre":
        return None, None, None, None, None
    try:
        vectors = getattr(backend, "v")
        visibility = getattr(backend, "vis")
    except Exception as exc:
        raise BackendEvidenceError(
            "in-memory terminal state could not be inspected"
        ) from exc
    vectors = _require_mapping(vectors, "backend.v")
    visibility = _require_mapping(visibility, "backend.vis")
    if set(vectors) != set(visibility):
        _fail("backend.v and backend.vis contain different ids")
    rows = [
        {
            "id": row_id,
            "visible": visibility[row_id],
            "vector": vector,
        }
        for row_id, vector in vectors.items()
    ]
    terminal = _terminal_state(
        rows=rows,
        source="in_memory_object",
        consistency="strong_in_process",
        expected_seed_state=expected_seed_state,
    )
    return None, None, None, None, terminal


def _validate_local_lite_uri(uri: Any) -> None:
    try:
        value = os.fspath(uri)
    except (TypeError, ValueError, OSError) as exc:
        raise BackendEvidenceError(
            "Milvus Lite endpoint is not a valid local path"
        ) from exc
    if (
        type(value) is not str
        or not value.strip()
        or value != value.strip()
        or "\x00" in value
        or "://" in value
        or not value.lower().endswith(".db")
    ):
        _fail("Milvus Lite endpoint is not a valid local .db path")


def _collection_receipt(
    raw_value: Any,
    *,
    expected_name: str,
    expected_dim: int,
) -> dict[str, Any]:
    raw = _require_mapping(raw_value, "describe_collection")
    name = _require_nonempty_string(
        raw.get("collection_name"), "describe_collection.collection_name"
    )
    if name != expected_name:
        _fail("describe_collection returned a different collection")
    auto_id = _require_bool(
        raw.get("auto_id"), "describe_collection.auto_id"
    )
    dynamic = _require_bool(
        raw.get("enable_dynamic_field"),
        "describe_collection.enable_dynamic_field",
    )
    if auto_id or dynamic:
        _fail("describe_collection violates the fixed W2D schema")
    consistency = _normalize_consistency(
        raw.get("consistency_level"),
        "describe_collection.consistency_level",
    )
    if consistency != "Strong":
        _fail("describe_collection is not Strong-consistency")
    raw_fields = raw.get("fields")
    if (
        not isinstance(raw_fields, Sequence)
        or isinstance(raw_fields, (str, bytes, bytearray))
    ):
        _fail("describe_collection.fields must be a sequence")
    by_name: dict[str, dict[str, Any]] = {}
    for index, raw_field_value in enumerate(raw_fields):
        path = f"describe_collection.fields[{index}]"
        raw_field = _require_mapping(raw_field_value, path)
        field_name = _require_nonempty_string(
            raw_field.get("name"), f"{path}.name"
        )
        if field_name in by_name:
            _fail("describe_collection.fields contains duplicate names")
        data_type = _normalize_data_type(
            raw_field.get("type"), f"{path}.type"
        )
        is_primary = raw_field.get("is_primary", False)
        auto_field = raw_field.get("auto_id", False)
        is_primary = _require_bool(is_primary, f"{path}.is_primary")
        auto_field = _require_bool(auto_field, f"{path}.auto_id")
        params = raw_field.get("params", {})
        params = _require_mapping(params, f"{path}.params")
        dimension: int | None
        if data_type == "FLOAT_VECTOR":
            dimension = _require_int(
                params.get("dim"), f"{path}.params.dim", minimum=1
            )
        else:
            if "dim" in params:
                _fail(f"{path}.params has a scalar-field dimension")
            dimension = None
        by_name[field_name] = {
            "name": field_name,
            "data_type": data_type,
            "is_primary": is_primary,
            "auto_id": auto_field,
            "dimension": dimension,
        }
    if set(by_name) != set(_EXPECTED_FIELD_ORDER):
        _fail("describe_collection.fields is not the fixed W2D schema")
    for name_in_order in _EXPECTED_FIELD_ORDER:
        field = by_name[name_in_order]
        if field["data_type"] != _EXPECTED_FIELD_TYPES[name_in_order]:
            _fail("describe_collection.fields has the wrong data type")
    if not by_name["id"]["is_primary"]:
        _fail("describe_collection.id is not primary")
    if any(
        by_name[name]["is_primary"] for name in ("vector", "visible")
    ):
        _fail("describe_collection has an unexpected primary field")
    if any(by_name[name]["auto_id"] for name in _EXPECTED_FIELD_ORDER):
        _fail("describe_collection has an auto-id field")
    if by_name["vector"]["dimension"] != expected_dim:
        _fail("describe_collection vector dimension disagrees with backend")
    return {
        "name": name,
        "auto_id": auto_id,
        "enable_dynamic_field": dynamic,
        "consistency_level": consistency,
        "fields": [by_name[name] for name in _EXPECTED_FIELD_ORDER],
    }


def _index_receipt(raw_value: Any) -> dict[str, Any]:
    raw = _require_mapping(raw_value, "describe_index")
    field_name = _require_nonempty_string(
        raw.get("field_name"), "describe_index.field_name"
    )
    # MilvusClient's default IndexParams uses the empty string as the actual
    # index name when add_index() is called without index_name.
    index_name = _require_string(
        raw.get("index_name"), "describe_index.index_name"
    )
    index_type = _require_nonempty_string(
        raw.get("index_type"), "describe_index.index_type"
    ).upper()
    metric_type = _require_nonempty_string(
        raw.get("metric_type"), "describe_index.metric_type"
    ).upper()
    if field_name != "vector" or index_type != "FLAT" or metric_type != "COSINE":
        _fail("describe_index is not the fixed W2D FLAT/COSINE index")
    state_value = raw.get("state", raw.get("index_state"))
    state = _normalize_index_ready_state(
        state_value, "describe_index.state"
    )
    return {
        "field_name": field_name,
        "index_name": index_name,
        "index_type": index_type,
        "metric_type": metric_type,
        "state": state,
        "total_rows": _optional_int(
            raw.get("total_rows"), "describe_index.total_rows"
        ),
        "indexed_rows": _optional_int(
            raw.get("indexed_rows"), "describe_index.indexed_rows"
        ),
        "pending_index_rows": _optional_int(
            raw.get("pending_index_rows"),
            "describe_index.pending_index_rows",
        ),
    }


def _load_receipt(raw_value: Any) -> dict[str, str]:
    raw = _require_mapping(raw_value, "get_load_state")
    return {
        "state": _normalize_load_ready_state(
            raw.get("state"), "get_load_state.state"
        )
    }


def _capture_milvus(
    backend: Any,
    *,
    uri: Any,
    phase: str,
    expected_seed_state: Any,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    dict[str, Any] | None,
]:
    _validate_local_lite_uri(uri)
    try:
        client = getattr(backend, "client")
        collection_name = getattr(backend, "coll")
        backend_dim = getattr(backend, "dim")
    except Exception as exc:
        raise BackendEvidenceError(
            "Milvus backend attributes could not be inspected"
        ) from exc
    collection_name = _require_nonempty_string(
        collection_name, "backend.coll"
    )
    if (
        re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", collection_name) is None
    ):
        _fail("backend.coll is not a canonical collection name")
    backend_dim = _require_int(backend_dim, "backend.dim", minimum=1)
    if backend_dim != EXPECTED_VECTOR_DIM:
        _fail("backend.dim is not the preregistered W2D dimension")

    server = _server_receipt(client)
    collection = _collection_receipt(
        _live_call(
            "describe_collection",
            lambda: client.describe_collection(collection_name),
        ),
        expected_name=collection_name,
        expected_dim=backend_dim,
    )
    index = _index_receipt(
        _live_call(
            "describe_index",
            lambda: client.describe_index(collection_name, "vector"),
        )
    )
    load = _load_receipt(
        _live_call(
            "get_load_state",
            lambda: client.get_load_state(collection_name),
        )
    )

    terminal = None
    if phase == "post":
        raw_rows = _live_call(
            "Strong terminal query",
            lambda: client.query(
                collection_name,
                filter="",
                output_fields=["id", "visible", "vector"],
                limit=MAX_LOGICAL_ROWS,
                consistency_level="Strong",
            ),
        )
        terminal = _terminal_state(
            rows=raw_rows,
            source="milvus_strong_query",
            consistency="Strong",
            expected_seed_state=expected_seed_state,
        )
    return server, collection, index, load, terminal


def _validate_json_tree(value: Any, path: str) -> None:
    if value is None or type(value) in {bool, str, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            _fail(f"{path} contains a non-finite number")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_tree(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                _fail(f"{path} contains a non-string key")
            _validate_json_tree(item, f"{path}.{key}")
        return
    _fail(f"{path} contains a non-JSON value")


def _validate_receipt_shape(receipt: dict[str, Any]) -> None:
    _require_exact_fields(receipt, TOP_LEVEL_FIELDS, "receipt")
    identity = _require_mapping(receipt["backend_identity"], "backend_identity")
    _require_exact_fields(identity, IDENTITY_FIELDS, "backend_identity")
    endpoint = _require_mapping(receipt["endpoint"], "endpoint")
    _require_exact_fields(endpoint, ENDPOINT_FIELDS, "endpoint")
    if receipt["server"] is not None:
        server = _require_mapping(receipt["server"], "server")
        _require_exact_fields(server, SERVER_FIELDS, "server")
    if receipt["collection"] is not None:
        collection = _require_mapping(receipt["collection"], "collection")
        _require_exact_fields(collection, COLLECTION_FIELDS, "collection")
        fields = collection["fields"]
        if type(fields) is not list:
            _fail("collection.fields must be a JSON array")
        for index, field in enumerate(fields):
            field = _require_mapping(field, f"collection.fields[{index}]")
            _require_exact_fields(
                field,
                COLLECTION_FIELD_FIELDS,
                f"collection.fields[{index}]",
            )
    if receipt["index"] is not None:
        index = _require_mapping(receipt["index"], "index")
        _require_exact_fields(index, INDEX_FIELDS, "index")
    if receipt["load"] is not None:
        load = _require_mapping(receipt["load"], "load")
        _require_exact_fields(load, LOAD_FIELDS, "load")
    if receipt["terminal_state"] is not None:
        terminal = _require_mapping(
            receipt["terminal_state"], "terminal_state"
        )
        _require_exact_fields(terminal, TERMINAL_FIELDS, "terminal_state")
    _validate_json_tree(receipt, "receipt")
    try:
        json.dumps(receipt, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BackendEvidenceError("receipt is not strict JSON") from exc


def _validate_e2_metadata(receipt: Mapping[str, Any], path: str) -> None:
    server = _require_mapping(receipt["server"], f"{path}.server")
    status = server["status"]
    version = server["version"]
    if status == "reported":
        version = _require_nonempty_string(
            version, f"{path}.server.version"
        )
        if re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._+-]*", version
        ) is None:
            _fail(f"{path}.server.version is not a canonical version")
    elif status == "unimplemented":
        if version is not None:
            _fail(
                f"{path}.server.version must be null when unimplemented"
            )
    else:
        _fail(f"{path}.server.status is not controlled")

    expected_collection = {
        "name": "w2d_poison",
        "auto_id": False,
        "enable_dynamic_field": False,
        "consistency_level": "Strong",
        "fields": [
            {
                "name": "id",
                "data_type": "INT64",
                "is_primary": True,
                "auto_id": False,
                "dimension": None,
            },
            {
                "name": "vector",
                "data_type": "FLOAT_VECTOR",
                "is_primary": False,
                "auto_id": False,
                "dimension": EXPECTED_VECTOR_DIM,
            },
            {
                "name": "visible",
                "data_type": "BOOL",
                "is_primary": False,
                "auto_id": False,
                "dimension": None,
            },
        ],
    }
    if receipt["collection"] != expected_collection:
        _fail(f"{path}.collection is not the production W2D schema")

    index = _require_mapping(receipt["index"], f"{path}.index")
    if (
        index["field_name"] != "vector"
        or index["index_type"] != "FLAT"
        or index["metric_type"] != "COSINE"
        or index["state"] != "Finished"
    ):
        _fail(f"{path}.index is not a ready FLAT/COSINE vector index")
    index_name = _require_string(
        index["index_name"], f"{path}.index.index_name"
    )
    if index_name and re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", index_name
    ) is None:
        _fail(f"{path}.index.index_name is not canonical")
    counts = [
        _optional_int(index[field], f"{path}.index.{field}")
        for field in (
            "total_rows",
            "indexed_rows",
            "pending_index_rows",
        )
    ]
    if all(count is not None for count in counts):
        total_rows, indexed_rows, pending_rows = counts
        assert total_rows is not None
        assert indexed_rows is not None
        assert pending_rows is not None
        if indexed_rows > total_rows or pending_rows > total_rows:
            _fail(f"{path}.index row accounting is impossible")

    if receipt["load"] != {"state": "Loaded"}:
        _fail(f"{path}.load is not Loaded")


def validate_backend_evidence_pair(
    value: Any,
    *,
    backend_name: str,
    expected_post_state: Any,
) -> None:
    """Validate a complete production ``{"pre", "post"}`` evidence pair.

    This function is pure: it performs no backend calls and does not mutate its
    arguments.  It accepts only receipts produced for the production backend
    classes, then binds the post receipt to the caller-supplied logical-state
    summary.  Any disagreement raises :class:`BackendEvidenceError`.
    """

    if type(backend_name) is not str or backend_name not in {
        "inmemory",
        "milvus",
    }:
        _fail("backend_name must be 'inmemory' or 'milvus'")
    pair = _require_mapping(value, "backend_evidence")
    _require_exact_fields(pair, PAIR_FIELDS, "backend_evidence")
    _validate_json_tree(value, "backend_evidence")

    receipts: dict[str, Mapping[str, Any]] = {}
    for phase in ("pre", "post"):
        receipt = _require_mapping(
            pair[phase], f"backend_evidence.{phase}"
        )
        _validate_receipt_shape(receipt)
        if receipt["schema_version"] != SCHEMA_VERSION:
            _fail(f"backend_evidence.{phase} has the wrong schema version")
        if receipt["phase"] != phase:
            _fail(f"backend_evidence.{phase} has the wrong phase")
        if receipt["backend_name"] != backend_name:
            _fail(f"backend_evidence.{phase} has the wrong backend")
        if receipt["backend_identity"] != _PRODUCTION_IDENTITIES[backend_name]:
            _fail(
                f"backend_evidence.{phase} is not from the production "
                "backend class"
            )
        if receipt["endpoint"] != _EXPECTED_ENDPOINTS[backend_name]:
            _fail(f"backend_evidence.{phase} has the wrong endpoint boundary")
        if receipt["evidence_boundary"] != _EVIDENCE_BOUNDARY:
            _fail(
                f"backend_evidence.{phase} has the wrong measurement boundary"
            )
        receipts[phase] = receipt

    pre = receipts["pre"]
    post = receipts["post"]
    if pre["terminal_state"] is not None:
        _fail("backend_evidence.pre must not contain terminal state")

    if backend_name == "inmemory":
        for phase, receipt in receipts.items():
            if any(
                receipt[field] is not None
                for field in ("server", "collection", "index", "load")
            ):
                _fail(
                    f"backend_evidence.{phase} contains E2-only metadata"
                )
    else:
        for phase, receipt in receipts.items():
            _validate_e2_metadata(
                receipt, f"backend_evidence.{phase}"
            )
        if pre["server"] != post["server"]:
            _fail("backend evidence server changed between pre and post")
        if pre["collection"] != post["collection"]:
            _fail("backend evidence collection changed between pre and post")
        stable_index_fields = (
            "field_name",
            "index_name",
            "index_type",
            "metric_type",
            "state",
        )
        if any(
            pre["index"][field] != post["index"][field]
            for field in stable_index_fields
        ):
            _fail("backend evidence index identity changed pre to post")
        if pre["load"] != post["load"]:
            _fail("backend evidence load state changed between pre and post")

    expected = _validate_expected_summary(expected_post_state)
    terminal = _require_mapping(
        post["terminal_state"], "backend_evidence.post.terminal_state"
    )
    observed_summary = {
        field: terminal[field] for field in EXPECTED_SUMMARY_FIELDS
    }
    if observed_summary != expected:
        _fail("backend evidence post state differs from expected summary")
    if terminal["expected_seed_state_checked"] is not True:
        _fail("backend evidence post state was not checked at capture time")

    expected_source = (
        "in_memory_object"
        if backend_name == "inmemory"
        else "milvus_strong_query"
    )
    expected_consistency = (
        "strong_in_process" if backend_name == "inmemory" else "Strong"
    )
    if (
        terminal["source"] != expected_source
        or terminal["consistency"] != expected_consistency
    ):
        _fail("backend evidence post terminal-state semantics are wrong")


def capture_backend_evidence(
    backend_name: str,
    backend: Any,
    uri: Any,
    phase: str,
    expected_seed_state: Any = None,
) -> dict[str, Any]:
    """Capture one validated pre/post receipt outside the measured window.

    ``pre`` never reads logical rows.  ``post`` reads E1 state directly or
    issues one Strong Milvus query.  Any failed call, malformed response,
    non-finite value, schema drift, or expected-state mismatch raises
    :class:`BackendEvidenceError`.
    """

    if type(backend_name) is not str or backend_name not in {
        "inmemory",
        "milvus",
    }:
        _fail("backend_name must be 'inmemory' or 'milvus'")
    if type(phase) is not str or phase not in {"pre", "post"}:
        _fail("phase must be 'pre' or 'post'")
    if phase == "pre" and expected_seed_state is not None:
        _fail("expected_seed_state is only valid for phase='post'")

    identity = _backend_identity(backend)
    if backend_name == "inmemory":
        server, collection, index, load, terminal = _capture_inmemory(
            backend,
            phase=phase,
            expected_seed_state=expected_seed_state,
        )
        endpoint = {
            "kind": "process_local",
            "absolute_uri_omitted": True,
        }
    else:
        server, collection, index, load, terminal = _capture_milvus(
            backend,
            uri=uri,
            phase=phase,
            expected_seed_state=expected_seed_state,
        )
        endpoint = {
            "kind": "milvus_lite_local_file",
            "absolute_uri_omitted": True,
        }

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "backend_name": backend_name,
        "backend_identity": identity,
        "endpoint": endpoint,
        "evidence_boundary": _EVIDENCE_BOUNDARY,
        "server": server,
        "collection": collection,
        "index": index,
        "load": load,
        "terminal_state": terminal,
    }
    _validate_receipt_shape(receipt)
    return receipt


__all__ = [
    "BackendEvidenceError",
    "EXPECTED_SUMMARY_FIELDS",
    "EXPECTED_VECTOR_DIM",
    "PAIR_FIELDS",
    "SCHEMA_VERSION",
    "TOP_LEVEL_FIELDS",
    "capture_backend_evidence",
    "summarize_expected_seed_state",
    "validate_backend_evidence_pair",
]
