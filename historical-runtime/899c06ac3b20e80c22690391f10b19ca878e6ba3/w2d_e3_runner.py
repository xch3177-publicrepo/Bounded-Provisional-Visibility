#!/usr/bin/env python3
"""Independent W2D-E3 Milvus Standalone ANN/concurrency runner.

This module deliberately does not import or extend the formal E1/E2 runner.
It consumes the frozen W2D parent artifacts, builds one 100k-vector HNSW
collection, measures the preregistered ANN panel, and then runs the B1--B4
measured-service trace replay against the same graph.

Formal design:
  W2D-E3-PREREGISTRATION.md
  W2D-E3-PREREGISTRATION-AMENDMENT-A1.md

``--quick`` is a code-path exercise only.  Its population, queries, repetitions,
concurrencies, and timings are intentionally smaller and its output is marked
non-formal.  It cannot satisfy or substitute for any formal E3 run.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np

from functional_slice import State, System, VISIBLE, VerifierDecision
from milvus_backend import MilvusBackend


HERE = Path(__file__).resolve().parent
RUN_NUMBERS = (1, 2, 3, 4, 5)
BACKGROUND_SEED = 7301
QUERY_ORDER_SEED_BASE = 8300
STATIC_N = 100_000
FROZEN_N = 1_752
BACKGROUND_N = 98_248
DIM = 384
QUERY_N = 192
PANEL_A_EF = (20, 64, 256)
PRIMARY_EF = 64
PANEL_A_REPEATS = 5
PANEL_A_LATENCY_N = 960
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
TOP_K = 5
CONSISTENCY_LEVEL = "Strong"
PANEL_B_CONCURRENCY = (1, 4, 8, 16)
PANEL_B_WARMUP_S = 5.0
PANEL_B_MEASURED_S = 15.0
ADMISSION_RATE_PER_S = 20.0
TP_S = 1.0
VERIFIER_CONCURRENCY = 4
SENTINEL_N = 50
SCHEMA_VERSION = "W2D-E3-result-v1"
COLLECTION_PREFIX = "w2d_e3_"

WILLIAMS4 = (
    ("B1", "B2", "B4", "B3"),
    ("B2", "B3", "B1", "B4"),
    ("B3", "B4", "B2", "B1"),
    ("B4", "B1", "B3", "B2"),
)

BASELINE_SWITCHES = {
    "B1": {"verify": False, "sync": False, "deadline": False},
    "B2": {"verify": True, "sync": True, "deadline": False},
    "B3": {"verify": True, "sync": False, "deadline": False},
    "B4": {"verify": True, "sync": False, "deadline": True},
}

FROZEN_PARENT_SHA256 = {
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
}
TRANSITIVE_LABEL_PATH = "results/w2d/W2D-labels.json"
TRANSITIVE_LABEL_SHA256 = (
    "05d1363510a1790d690b05640c3a2830be4e62df56b8592182904ccb29019a9e"
)

RUNTIME_COMPONENTS = (
    "w2d_e3_runner.py",
    "verify_w2d_e3.py",
    "milvus_backend.py",
    "functional_slice.py",
    "backend.py",
    "docker-compose.yml",
)
THREAD_POOL_WORKERS = 32


class E3Error(RuntimeError):
    """A construction, provenance, or execution contract failed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_run_number(run_number: Any) -> int:
    if (
        isinstance(run_number, bool)
        or not isinstance(run_number, int)
        or run_number not in RUN_NUMBERS
    ):
        raise ValueError(f"run number must be one of {RUN_NUMBERS}")
    return run_number


def strict_json_load_bytes(payload: bytes, source: str = "<bytes>") -> Any:
    """Load UTF-8 JSON while rejecting duplicate keys and non-finite constants."""
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("strict_json_load_bytes requires bytes")

    def reject_constant(value: str) -> None:
        raise ValueError(f"{source}: non-standard JSON constant {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{source}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"{source}: non-finite JSON number {value}")
        return parsed

    try:
        text = bytes(payload).decode("utf-8", errors="strict")
        return json.loads(
            text,
            parse_constant=reject_constant,
            parse_float=finite_float,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise E3Error(f"strict JSON read failed for {source}: {exc}") from exc


def strict_json_bytes(value: Any) -> bytes:
    """Canonical-enough deterministic strict JSON for an E3 result."""
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
    except (TypeError, ValueError) as exc:
        raise E3Error(f"result is not strict JSON: {exc}") from exc
    payload = (text + "\n").encode("utf-8")
    strict_json_load_bytes(payload, "serialized E3 result")
    return payload


def exclusive_write_json(path: str | os.PathLike[str], value: Any) -> Path:
    """Strictly encode first, then exclusively create ``path`` without overwrite."""
    target = Path(path)
    payload = strict_json_bytes(value)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise E3Error(f"refusing to overwrite existing result: {target}") from exc
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # A failed write is visibly retained rather than silently reused.
        raise
    return target


def baseline_order(run_number: int) -> tuple[str, ...]:
    run_number = _validate_run_number(run_number)
    return WILLIAMS4[(run_number - 1) % len(WILLIAMS4)]


def concurrency_order(run_number: int) -> tuple[int, ...]:
    run_number = _validate_run_number(run_number)
    shift = (run_number - 1) % len(PANEL_B_CONCURRENCY)
    return PANEL_B_CONCURRENCY[shift:] + PANEL_B_CONCURRENCY[:shift]


def query_order_seed(run_number: int) -> int:
    return QUERY_ORDER_SEED_BASE + _validate_run_number(run_number)


def query_order(run_number: int, n: int = QUERY_N) -> tuple[int, ...]:
    _validate_run_number(run_number)
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("query order length must be a positive integer")
    return tuple(
        int(value)
        for value in np.random.default_rng(query_order_seed(run_number)).permutation(n)
    )


def panel_b_matrix(run_number: int) -> list[dict[str, Any]]:
    """Return the complete formal 16-cell matrix in its execution order."""
    run_number = _validate_run_number(run_number)
    return [
        {
            "cell_id": f"run-{run_number}__baseline-{baseline}__concurrency-{conc}",
            "baseline": baseline,
            "query_concurrency": conc,
            "warmup_s": PANEL_B_WARMUP_S,
            "measured_s": PANEL_B_MEASURED_S,
            "admission_rate_per_s": ADMISSION_RATE_PER_S,
            "tp_s": TP_S,
            "verifier_concurrency": VERIFIER_CONCURRENCY,
            "top_k": TOP_K,
            "ef": PRIMARY_EF,
        }
        for baseline in baseline_order(run_number)
        for conc in concurrency_order(run_number)
    ]


def percentiles(values: Sequence[float]) -> dict[str, float | int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("percentiles requires a non-empty numeric sequence")
    if not values:
        raise ValueError("percentiles requires at least one value")
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iuf":
        raise TypeError("percentiles requires a one-dimensional numeric sequence")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("percentiles rejects NaN and Infinity")
    return {
        "count": int(array.size),
        "p50": float(np.percentile(array, 50, method="linear")),
        "p95": float(np.percentile(array, 95, method="linear")),
        "p99": float(np.percentile(array, 99, method="linear")),
        "max": float(np.max(array)),
    }


def _maybe_percentiles(values: Sequence[float]) -> dict[str, Any]:
    return (
        percentiles(values)
        if values
        else {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    )


def deterministic_background(
    count: int,
    dim: int = DIM,
    seed: int = BACKGROUND_SEED,
) -> np.ndarray:
    """Frozen float32 Gaussian extension used by the 100k systems population."""
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        or isinstance(dim, bool)
        or not isinstance(dim, int)
        or dim <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ValueError(
            "background count/dimension/seed must be non-negative/positive/int"
        )
    values = np.random.default_rng(seed).standard_normal(
        (count, dim), dtype=np.float32
    )
    if count:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms <= 0):
            raise E3Error("Gaussian background produced an invalid norm")
        values /= norms
    return np.ascontiguousarray(values, dtype=np.float32)


def build_population(
    frozen_embeddings: np.ndarray,
    static_n: int = STATIC_N,
) -> np.ndarray:
    source = np.asarray(frozen_embeddings, dtype=np.float32)
    if source.shape != (FROZEN_N, DIM) or not np.isfinite(source).all():
        raise E3Error(f"frozen embeddings must have shape {(FROZEN_N, DIM)}")
    if isinstance(static_n, bool) or not isinstance(static_n, int) or static_n < FROZEN_N:
        raise ValueError(f"static_n must be an integer >= {FROZEN_N}")
    norms = np.linalg.norm(source, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise E3Error("frozen embeddings contain a zero/non-finite vector")
    normalized = source / norms
    result = np.empty((static_n, DIM), dtype=np.float32)
    result[:FROZEN_N] = normalized
    result[FROZEN_N:] = deterministic_background(static_n - FROZEN_N)
    return np.ascontiguousarray(result)


@dataclass(frozen=True)
class FrozenData:
    embeddings: np.ndarray
    key_to_row: Mapping[str, int]
    test_order: tuple[str, ...]
    scores: Mapping[str, Mapping[str, Any]]
    truth_poison: Mapping[str, bool]
    query_keys: tuple[str, ...]
    query_vectors: np.ndarray
    query_families: tuple[str, ...]
    query_associated_ids: tuple[int, ...]
    parent_hashes: Mapping[str, str]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _strict_file(path: Path) -> Any:
    return strict_json_load_bytes(path.read_bytes(), str(path))


def load_frozen() -> FrozenData:
    """Load and cross-check every frozen parent byte binding used by E3."""
    observed: dict[str, str] = {}
    for relative, expected in FROZEN_PARENT_SHA256.items():
        path = HERE / relative
        actual = _hash_file(path)
        observed[relative] = actual
        if actual != expected:
            raise E3Error(f"frozen parent hash mismatch for {relative}: {actual}")

    label_path = HERE / TRANSITIVE_LABEL_PATH
    label_sha = _hash_file(label_path)
    if label_sha != TRANSITIVE_LABEL_SHA256:
        raise E3Error(f"transitive label hash mismatch: {label_sha}")
    observed[TRANSITIVE_LABEL_PATH] = label_sha

    # A1 requires both the label bytes and their transitive E1 binding.
    e1 = _strict_file(HERE / "results/w2d/W2D-E1-INMEMORY.json")
    if e1.get("artifact_hashes", {}).get("labels") != label_sha:
        raise E3Error("E1 does not bind the transitive evaluator-label digest")
    del e1

    inputs = _strict_file(HERE / "data/w2d/W2D-detector-inputs.json")
    plan = _strict_file(HERE / "results/w2d/W2D-PROTOCOL-PLAN.json")
    score_doc = _strict_file(HERE / "results/w2d/W2D-test-scores.json")
    labels = _strict_file(label_path)
    _strict_file(HERE / "results/w2d/W2D-threshold.json")

    items = inputs.get("items")
    if not isinstance(items, list) or len(items) != FROZEN_N:
        raise E3Error(f"safe input must contain exactly {FROZEN_N} items")
    key_to_row: dict[str, int] = {}
    for expected_row, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise E3Error("safe input item is not an object")
        key, row = item.get("item_key"), item.get("embedding_row")
        if (
            not isinstance(key, str)
            or len(key) != 64
            or key in key_to_row
            or row != expected_row
        ):
            raise E3Error("safe input key/embedding row order is malformed")
        key_to_row[key] = row

    npz_path = HERE / "data/w2d/W2D-detector-inputs.npz"
    with np.load(npz_path, allow_pickle=False) as archive:
        if archive.files != ["embeddings"]:
            raise E3Error("embedding NPZ must contain only 'embeddings'")
        embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
    if embeddings.shape != (FROZEN_N, DIM) or not np.isfinite(embeddings).all():
        raise E3Error("frozen embedding matrix has the wrong shape/content")

    test_order_raw = inputs.get("sets", {}).get("test_score_order")
    score_items = score_doc.get("items")
    if (
        not isinstance(test_order_raw, list)
        or len(test_order_raw) != 512
        or not isinstance(score_items, list)
        or len(score_items) != 512
    ):
        raise E3Error("D1 test order/scores must each contain 512 items")
    test_order = tuple(test_order_raw)
    scores: dict[str, Mapping[str, Any]] = {}
    for position, record in enumerate(score_items):
        if not isinstance(record, Mapping):
            raise E3Error("D1 score record is not an object")
        key = record.get("item_key")
        if key != test_order[position] or key in scores:
            raise E3Error("D1 score order differs from the frozen test order")
        promote, service_ns = record.get("promote"), record.get("detector_service_ns")
        if (
            not isinstance(promote, bool)
            or isinstance(service_ns, bool)
            or not isinstance(service_ns, int)
            or service_ns < 0
        ):
            raise E3Error("D1 decision/service record is malformed")
        scores[key] = {
            "promote": promote,
            "detector_service_ns": service_ns,
        }

    label_items = labels.get("items")
    if not isinstance(label_items, Mapping):
        raise E3Error("label artifact has no item map")
    truth: dict[str, bool] = {}
    for key in test_order:
        label = label_items.get(key)
        poison = label.get("poison") if isinstance(label, Mapping) else None
        if not isinstance(poison, bool):
            raise E3Error("a frozen test item has no Boolean poison truth")
        truth[key] = poison

    query_materials = plan.get("query_materials")
    if not isinstance(query_materials, Mapping) or len(query_materials) != QUERY_N:
        raise E3Error(f"protocol plan must contain exactly {QUERY_N} query materials")
    query_keys: list[str] = []
    query_vectors: list[np.ndarray] = []
    query_families: list[str] = []
    associated_ids: list[int] = []
    for query_key, material in query_materials.items():
        if not isinstance(query_key, str) or not isinstance(material, Mapping):
            raise E3Error("query material is malformed")
        vector = np.asarray(material.get("embedding"), dtype=np.float32)
        norm = float(np.linalg.norm(vector)) if vector.shape == (DIM,) else 0.0
        associated_key = material.get("query_item_key")
        if (
            not math.isfinite(norm)
            or norm <= 0
            or associated_key not in key_to_row
        ):
            raise E3Error("query vector/associated item is malformed")
        query_keys.append(query_key)
        query_vectors.append(vector / norm)
        query_families.append(str(material.get("attack_family") or "unknown"))
        associated_ids.append(key_to_row[associated_key])

    return FrozenData(
        embeddings=embeddings,
        key_to_row=key_to_row,
        test_order=test_order,
        scores=scores,
        truth_poison=truth,
        query_keys=tuple(query_keys),
        query_vectors=np.ascontiguousarray(query_vectors, dtype=np.float32),
        query_families=tuple(query_families),
        query_associated_ids=tuple(associated_ids),
        parent_hashes=observed,
    )


def runtime_fingerprint() -> dict[str, Any]:
    components: dict[str, str] = {}
    combined = hashlib.sha256()
    for name in sorted(RUNTIME_COMPONENTS):
        path = HERE / name
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        components[name] = digest
        combined.update(name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(payload)
        combined.update(b"\0")
    return {"sha256": combined.hexdigest(), "components": components}


def _git_provenance() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=HERE,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    commit = run("rev-parse", "HEAD")
    tracked_status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit or None,
        "tracked_tree_clean": not tracked_status,
        "tracked_status": tracked_status.splitlines(),
    }


def _memory_descriptor() -> dict[str, Any]:
    page_size = page_count = None
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        pass
    return {
        "physical_memory_bytes": (
            page_size * page_count
            if page_size is not None and page_count is not None
            else None
        ),
        "source": "os.sysconf(SC_PAGE_SIZE*SC_PHYS_PAGES)",
    }


def _environment_descriptor() -> dict[str, Any]:
    try:
        pymilvus_version = importlib_metadata.version("pymilvus")
    except importlib_metadata.PackageNotFoundError:
        pymilvus_version = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "pymilvus_version": pymilvus_version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "memory": _memory_descriptor(),
    }


def _process_start_metric(
    metrics_url: str,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    """Read the one preregistered server-process identity from Prometheus."""
    parsed = urlparse(metrics_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise E3Error("formal E3 requires an http(s) Prometheus metrics URL")
    if not parsed.path.endswith("/metrics"):
        raise E3Error("Milvus metrics URL must end in /metrics")
    try:
        with urlopen(metrics_url, timeout=timeout_s) as response:
            payload = response.read().decode("utf-8", errors="strict")
    except Exception as exc:
        raise E3Error(
            f"cannot read Milvus process-start metric from {metrics_url}: {exc}"
        ) from exc
    found: list[tuple[str, float]] = []
    for line in payload.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        metric_name = stripped.split("{", 1)[0].split(None, 1)[0]
        if metric_name != "process_start_time_seconds":
            continue
        try:
            value = float(stripped.rsplit(None, 1)[1])
        except (IndexError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            found.append((stripped, value))
    if not found:
        raise E3Error(
            f"{metrics_url} exposed no finite process_start_time_seconds"
        )
    distinct = sorted({value for _line, value in found})
    if len(distinct) != 1:
        raise E3Error("Prometheus exposed ambiguous process-start values")
    return {
        "metric": "process_start_time_seconds",
        "value": distinct[0],
        "url": metrics_url,
        "matching_series_count": len(found),
    }


def _previous_formal_results() -> list[dict[str, Any]]:
    """Read every prior E3 JSON fail-closed.

    A malformed file in the authoritative directory is not skipped: doing so
    could make a previously accepted run or process start disappear from the
    stopping-rule audit.
    """
    directory = HERE / "results" / "w2d-e3"
    records: list[dict[str, Any]] = []
    if not directory.exists():
        return records
    # Technical attempts may use an explicit non-canonical filename.  Every JSON
    # declaring itself a formal E3 attempt participates in the restart audit.
    for path in sorted(directory.glob("*.json")):
        try:
            document = strict_json_load_bytes(path.read_bytes(), str(path))
        except E3Error as exc:
            raise E3Error(
                f"malformed prior JSON in authoritative E3 directory: {path}"
            ) from exc
        if not isinstance(document, Mapping):
            raise E3Error(f"prior E3 JSON root is not an object: {path}")
        if document.get("experiment") != "W2D-E3":
            # The final cross-run summary legitimately shares this directory.
            if document.get("schema_version") == "W2D-E3-summary-v1":
                continue
            raise E3Error(f"unrecognised JSON in authoritative E3 directory: {path}")
        formal = document.get("formal")
        if not isinstance(formal, bool):
            raise E3Error(f"prior E3 artifact has malformed formal flag: {path}")
        if not formal:
            continue
        run_number = document.get("run_number")
        if (
            isinstance(run_number, bool)
            or not isinstance(run_number, int)
            or run_number not in RUN_NUMBERS
        ):
            raise E3Error(f"prior formal E3 artifact has malformed run number: {path}")
        accepted = document.get("local_validity_accepted")
        if accepted is not None and not isinstance(accepted, bool):
            raise E3Error(f"prior E3 artifact has malformed acceptance flag: {path}")
        records.append({"path": str(path), "document": document})
    return records


def _enforce_formal_stopping_rule(
    run_number: int,
    previous: Sequence[Mapping[str, Any]],
) -> None:
    accepted = [
        record for record in previous
        if record.get("document", {}).get("local_validity_accepted") is True
    ]
    if len(accepted) >= len(RUN_NUMBERS):
        raise E3Error("formal E3 already has five accepted runs; stopping rule reached")
    if any(
        record.get("document", {}).get("run_number") == run_number
        for record in accepted
    ):
        raise E3Error(
            f"formal E3 run {run_number} already has an accepted byte instance"
        )


def _collection_row_count(
    backend: MilvusBackend,
    expected: int | None = None,
    timeout_s: float = 60.0,
) -> int | None:
    end = time.monotonic() + timeout_s
    last: int | None = None
    while time.monotonic() < end:
        try:
            stats = backend.client.get_collection_stats(backend.coll)
            raw = stats.get("row_count")
            if raw is not None:
                last = int(raw)
                if last >= 0 and (expected is None or last == expected):
                    return last
        except Exception:
            pass
        try:
            rows = backend.client.query(
                backend.coll,
                filter="id >= 0",
                output_fields=["count(*)"],
            )
            if rows and "count(*)" in rows[0]:
                last = int(rows[0]["count(*)"])
                if last >= 0 and (expected is None or last == expected):
                    return last
        except Exception:
            pass
        time.sleep(0.5)
    return last


def _index_is_ready(info: Mapping[str, Any]) -> bool:
    index_type = str(info.get("index_type_effective", "")).upper()
    metric_type = str(info.get("metric_type_effective", "")).upper()
    index_state = str(info.get("index_state", "")).lower()
    load_state = str(info.get("load_state", "")).lower()
    return (
        index_type == "HNSW"
        and metric_type == "COSINE"
        and info.get("M_effective") == HNSW_M
        and info.get("efConstruction_effective") == HNSW_EF_CONSTRUCTION
        and isinstance(info.get("indexed_rows"), int)
        and not isinstance(info.get("indexed_rows"), bool)
        and info.get("pending_index_rows") == 0
        and ("finish" in index_state or index_state in {"3"})
        and "loaded" in load_state
    )


def _wait_settled_hnsw(
    backend: MilvusBackend,
    *,
    expected_indexed_rows: int | None = None,
    minimum_indexed_rows: int | None = None,
    timeout_s: float = 180.0,
    poll_s: float = 0.5,
) -> dict[str, Any]:
    """Wait for a real, fully indexed HNSW snapshot and then reload it.

    Milvus can briefly report ``state=Finished`` for the collection index while
    newly flushed segments still appear as ``pending_index_rows``.  State alone
    would therefore let Panel A silently exercise growing-segment brute force.
    """
    if expected_indexed_rows is None and minimum_indexed_rows is None:
        raise ValueError("an exact or minimum indexed-row target is required")
    started = time.monotonic()
    deadline = started + float(timeout_s)
    observed_state = backend.wait_index(
        timeout=max(0.1, deadline - time.monotonic()),
        poll=poll_s,
    )
    last_info: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last_info = dict(backend.index_info())
        indexed_rows = last_info.get("indexed_rows")
        row_target_met = (
            isinstance(indexed_rows, int)
            and not isinstance(indexed_rows, bool)
            and (
                expected_indexed_rows is None
                or indexed_rows == expected_indexed_rows
            )
            and (
                minimum_indexed_rows is None
                or indexed_rows >= minimum_indexed_rows
            )
        )
        if _index_is_ready(last_info) and row_target_met:
            # The first wait may have reloaded during Milvus's transient
            # Finished/pending state. Reload once more only after all rows are
            # known to be indexed, as required by the preregistration.
            observed_state = backend.wait_index(
                timeout=max(0.1, min(5.0, deadline - time.monotonic())),
                poll=min(poll_s, 0.2),
            )
            final_info = dict(backend.index_info())
            final_indexed_rows = final_info.get("indexed_rows")
            final_target_met = (
                isinstance(final_indexed_rows, int)
                and not isinstance(final_indexed_rows, bool)
                and (
                    expected_indexed_rows is None
                    or final_indexed_rows == expected_indexed_rows
                )
                and (
                    minimum_indexed_rows is None
                    or final_indexed_rows >= minimum_indexed_rows
                )
            )
            if _index_is_ready(final_info) and final_target_met:
                return {
                    "state_observed": observed_state,
                    "wait_s": time.monotonic() - started,
                    "index_info": final_info,
                }
            last_info = final_info
        time.sleep(poll_s)
    raise E3Error(
        "HNSW did not settle before timeout "
        f"(exact={expected_indexed_rows}, minimum={minimum_indexed_rows}): "
        f"{last_info}"
    )


def _insert_population(
    backend: MilvusBackend,
    population: np.ndarray,
    batch_size: int = 500,
) -> dict[str, Any]:
    started = time.monotonic()
    inserted = 0
    for lo in range(0, len(population), batch_size):
        hi = min(lo + batch_size, len(population))
        backend.insert_many(
            [(index, population[index], True) for index in range(lo, hi)],
            cache_vectors=False,
        )
        inserted += hi - lo
    return {
        "attempted": int(len(population)),
        "inserted": inserted,
        "batch_size": batch_size,
        "insertion_s": time.monotonic() - started,
    }


def _exact_topk(population: np.ndarray, query: np.ndarray, k: int = TOP_K) -> list[int]:
    scores = population @ np.asarray(query, dtype=np.float32)
    candidates = np.argpartition(scores, -k)[-k:]
    ordered = candidates[np.lexsort((candidates, -scores[candidates]))]
    return [int(value) for value in ordered]


def _accuracy_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record.get("returned_ids") is not None]
    denominator = len(records)
    if denominator == 0:
        raise E3Error("accuracy group cannot have an empty denominator")
    return {
        "query_count": denominator,
        "successful_query_count": len(valid),
        "recall_at_5": float(
            np.mean([float(record.get("recall_at_5") or 0.0) for record in records])
        ),
        "exact_top5_match_rate": float(
            np.mean([bool(record.get("exact_top5_match")) for record in records])
        ),
        "landing_concordance_rate": float(
            np.mean([bool(record.get("landing_concordant")) for record in records])
        ),
        "failure_scoring": "a query with no measured result contributes zero",
    }


async def run_panel_a(
    backend: MilvusBackend,
    population: np.ndarray,
    frozen: FrozenData,
    *,
    quick: bool,
) -> dict[str, Any]:
    query_count = 12 if quick else QUERY_N
    repeats = 1 if quick else PANEL_A_REPEATS
    queries = frozen.query_vectors[:query_count]
    query_keys = frozen.query_keys[:query_count]
    families = frozen.query_families[:query_count]
    associated_ids = frozen.query_associated_ids[:query_count]
    exact = [_exact_topk(population, query) for query in queries]
    points: list[dict[str, Any]] = []

    for ef in PANEL_A_EF:
        # One complete, unreported warm-up pass immediately precedes this point.
        warmup_errors: Counter[str] = Counter()
        for query in queries:
            try:
                await asyncio.to_thread(
                    backend.search,
                    query.tolist(),
                    TOP_K,
                    False,
                    CONSISTENCY_LEVEL,
                    {"ef": ef},
                )
            except Exception as exc:
                warmup_errors[type(exc).__name__] += 1

        attempt_latency: list[float] = []
        error_types: Counter[str] = Counter()
        first_result: dict[int, list[int]] = {}
        result_variation: set[int] = set()
        for _repeat in range(repeats):
            for query_index, query in enumerate(queries):
                started = time.monotonic()
                try:
                    returned = await asyncio.to_thread(
                        backend.search,
                        query.tolist(),
                        TOP_K,
                        False,
                        CONSISTENCY_LEVEL,
                        {"ef": ef},
                    )
                    returned = [int(value) for value in returned]
                    if (
                        len(returned) != TOP_K
                        or len(set(returned)) != TOP_K
                        or any(value < 0 or value >= len(population) for value in returned)
                    ):
                        raise E3Error("ANN query did not return five unique static ids")
                    if query_index not in first_result:
                        first_result[query_index] = returned
                    elif first_result[query_index] != returned:
                        result_variation.add(query_index)
                except Exception as exc:
                    error_types[type(exc).__name__] += 1
                finally:
                    attempt_latency.append(time.monotonic() - started)

        per_query: list[dict[str, Any]] = []
        for index in range(query_count):
            returned = first_result.get(index)
            expected = exact[index]
            expected_set = set(expected)
            returned_set = set(returned or [])
            associated = associated_ids[index]
            per_query.append(
                {
                    "query_key": query_keys[index],
                    "attack_family": families[index],
                    "associated_static_id": associated,
                    "exact_ids": expected,
                    "returned_ids": returned,
                    "recall_at_5": (
                        len(expected_set & returned_set) / TOP_K
                        if returned is not None
                        else 0.0
                    ),
                    "exact_top5_match": (
                        returned == expected if returned is not None else False
                    ),
                    "exact_landed": associated in expected_set,
                    "ann_landed": (
                        associated in returned_set if returned is not None else False
                    ),
                    "landing_concordant": (
                        (associated in expected_set) == (associated in returned_set)
                        if returned is not None else False
                    ),
                    "accuracy_failure_zero": returned is None,
                }
            )
        by_family = {
            family: _accuracy_summary(
                [record for record in per_query if record["attack_family"] == family]
            )
            for family in sorted(set(families))
        }
        point = {
            "ef": ef,
            "primary": ef == PRIMARY_EF,
            "warmup_query_count": query_count,
            "warmup_errors": dict(sorted(warmup_errors.items())),
            "measured_repeats": repeats,
            "measured_attempt_count": len(attempt_latency),
            "latency_s": percentiles(attempt_latency),
            "latency_population": (
                "all measured attempts, including elapsed time to any error"
            ),
            "query_errors": int(sum(error_types.values())),
            "query_timeouts": int(error_types.get("TimeoutError", 0)),
            "error_types": dict(sorted(error_types.items())),
            "result_variation_query_count": len(result_variation),
            "overall": _accuracy_summary(per_query),
            "by_attack_family": by_family,
            "per_query_accuracy": per_query,
            "search_kwargs_observed": dict(
                getattr(backend, "last_search_kwargs", {})
            ),
        }
        points.append(point)

    return {
        "formal_query_count": QUERY_N,
        "executed_query_count": query_count,
        "repeats": repeats,
        "ef_points": points,
        "exact_reference": "NumPy COSINE top-5 over the same static snapshot",
    }


class FrozenReplay:
    """Decision/service provider that has no evaluator-label reference."""

    def __init__(self, scores: Mapping[str, Mapping[str, Any]]) -> None:
        self._scores = scores

    async def __call__(self, request: Any) -> VerifierDecision:
        if request.verifier_input != {
            "execution_mode": "score_once_frozen_decision_service_replay"
        }:
            raise E3Error("verifier received a non-replay payload")
        score = self._scores.get(request.item_key)
        if score is None:
            raise E3Error("verifier received an unknown frozen item key")
        return VerifierDecision(
            passes=bool(score["promote"]),
            service_time_s=int(score["detector_service_ns"]) / 1_000_000_000.0,
            metadata={
                "provider": "frozen_D1_decision_service_trace",
                "item_key": request.item_key,
            },
        )


class PreinsertedNoopBackendView:
    """Backend view used by the shared state machine in Panel B.

    The real Milvus insert is awaited before :meth:`System.admit` is called.
    The state machine's later insert is consequently a checked no-op.  Search
    still reaches the real collection, while post-filter eligibility remains
    entirely in the control store.
    """

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.preinserted: set[int] = set()
        self.noop_insert_ids: set[int] = set()

    def mark_preinserted(self, item_id: int) -> None:
        self.preinserted.add(int(item_id))

    def insert(self, item_id: int, _vector: Sequence[float], visible: bool = True) -> None:
        item_id = int(item_id)
        if item_id not in self.preinserted:
            raise E3Error(f"System attempted an insert before Milvus ack: {item_id}")
        if visible is not True:
            raise E3Error("Panel B preinsert view only accepts visible physical rows")
        if item_id in self.noop_insert_ids:
            raise E3Error(f"System attempted a duplicate logical insert: {item_id}")
        self.noop_insert_ids.add(item_id)

    def search(
        self,
        qvec: Sequence[float],
        k: int,
        use_index_filter: bool = False,
        consistency: str | None = None,
    ) -> list[int]:
        if use_index_filter:
            raise E3Error("Panel B must use control-plane post-filter eligibility")
        return self.backend.search(
            qvec,
            k,
            False,
            consistency or CONSISTENCY_LEVEL,
            {"ef": PRIMARY_EF},
        )

    def set_visible(self, _item_id: int, _visible: bool) -> None:
        raise E3Error("post-filter Panel B must not mutate Milvus visibility metadata")

    def delete(self, item_id: int) -> None:
        self.backend.delete(item_id)


class StaticPassthroughSystem(System):
    """The shared System with immutable static ids always query-eligible."""

    def __init__(self, *args: Any, static_n: int, **kwargs: Any) -> None:
        self.static_n = int(static_n)
        if self.static_n <= 0:
            raise ValueError("static_n must be positive")
        super().__init__(*args, **kwargs)

    def eligible(self, item_id: int) -> bool:
        parsed = int(item_id)
        if 0 <= parsed < self.static_n:
            return True
        return super().eligible(parsed)


async def _sentinel(
    backend: MilvusBackend,
    queries: np.ndarray,
    order: Sequence[int],
    n: int,
) -> dict[str, Any]:
    latencies: list[float] = []
    errors: Counter[str] = Counter()
    started_count = 0
    completed_count = 0
    for ordinal in range(n):
        started_count += 1
        started = time.monotonic()
        query = queries[order[ordinal % len(order)]]
        try:
            await asyncio.to_thread(
                backend.search,
                query.tolist(),
                TOP_K,
                False,
                CONSISTENCY_LEVEL,
                {"ef": PRIMARY_EF},
            )
            completed_count += 1
        except Exception as exc:
            errors[type(exc).__name__] += 1
        finally:
            latencies.append(time.monotonic() - started)
    return {
        "started": started_count,
        "completed": completed_count,
        "errors": int(sum(errors.values())),
        "error_types": dict(sorted(errors.items())),
        "latency_s": percentiles(latencies),
    }


def _classify_events(
    events: Sequence[Mapping[str, Any]],
    frozen: FrozenData,
) -> list[dict[str, Any]]:
    """Attach evaluator truth only after the observation snapshot exists."""
    classified: list[dict[str, Any]] = []
    for event in events:
        key = str(event["item_key"])
        classified.append(
            {
                **dict(event),
                "detector_promote": bool(frozen.scores[key]["promote"]),
                "truth_poison": bool(frozen.truth_poison[key]),
            }
        )
    return classified


def _decision_mix(events: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {
        "promote": 0,
        "refuse": 0,
        "correct_promotion": 0,
        "false_promotion": 0,
        "correct_refusal": 0,
        "false_refusal": 0,
    }
    for event in events:
        if not event.get("accepted"):
            continue
        poison = bool(event["truth_poison"])
        promote = bool(event["detector_promote"])
        counts["promote" if promote else "refuse"] += 1
        if promote and poison:
            counts["false_promotion"] += 1
        elif promote:
            counts["correct_promotion"] += 1
        elif poison:
            counts["correct_refusal"] += 1
        else:
            counts["false_refusal"] += 1
    return counts


def _shadow_confusion_512(frozen: FrozenData) -> dict[str, int]:
    events = [
        {
            "accepted": True,
            "detector_promote": bool(frozen.scores[key]["promote"]),
            "truth_poison": bool(frozen.truth_poison[key]),
        }
        for key in frozen.test_order
    ]
    return {"denominator": len(events), **_decision_mix(events)}


def _snapshot_system(
    system: System,
    dynamic_lo: int,
    dynamic_hi: int,
    snapshot_s: float,
) -> dict[str, Any]:
    """Copy all horizon-sensitive state before any draining or cancellation."""
    items = {
        int(item_id): copy.deepcopy(item)
        for item_id, item in system.items.items()
        if dynamic_lo <= int(item_id) < dynamic_hi
    }
    transitions = [
        copy.deepcopy(transition)
        for transition in system.transitions
        if dynamic_lo <= int(transition[1]) < dynamic_hi
        and float(transition[0]) <= snapshot_s
    ]
    by_item: dict[int, list[Sequence[Any]]] = {}
    for transition in transitions:
        by_item.setdefault(int(transition[1]), []).append(transition)
    for item_id, item in items.items():
        history = by_item.get(item_id, [])
        item["state"] = history[-1][3] if history else None
        item["was_visible"] = any(transition[3] in VISIBLE for transition in history)
        timestamps = item.get("ts")
        if isinstance(timestamps, Mapping):
            item["ts"] = {
                key: value
                for key, value in timestamps.items()
                if (
                    key == "deadline"
                    or not isinstance(value, (int, float))
                    or float(value) <= snapshot_s
                )
            }
    records = [
        copy.deepcopy(record)
        for record in system.verifier_records
        if dynamic_lo <= int(record.get("item_id", -1)) < dynamic_hi
    ]
    return {
        "snapshot_monotonic_s": snapshot_s,
        "items": items,
        "transitions": transitions,
        "verifier_records": records,
    }


def _exposure_seconds(
    item_id: int,
    transitions: Sequence[Sequence[Any]],
    measured_start_s: float,
    horizon_s: float,
) -> float:
    current: State | None = None
    cursor = measured_start_s
    exposure = 0.0
    for event_time, transitioned_id, _old, new in transitions:
        if int(transitioned_id) != int(item_id):
            continue
        event_time = float(event_time)
        if event_time <= measured_start_s:
            current = new
            continue
        if event_time > horizon_s:
            break
        if current in VISIBLE:
            exposure += max(0.0, event_time - cursor)
        current = new
        cursor = event_time
    if current in VISIBLE:
        exposure += max(0.0, horizon_s - cursor)
    return float(exposure)


def _live_outcomes(
    classified_events: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any],
    *,
    measured_start_s: float,
    horizon_s: float,
) -> dict[str, Any]:
    """Summarise observed lifecycle outcomes over the measured in-window cohort."""
    items = snapshot["items"]
    transitions = snapshot["transitions"]
    rows: list[dict[str, Any]] = []
    for event in classified_events:
        if not event.get("accepted_within_window"):
            continue
        item_id = int(event["item_id"])
        item = items.get(item_id)
        state = item.get("state") if isinstance(item, Mapping) else None
        exposure = _exposure_seconds(
            item_id,
            transitions,
            measured_start_s,
            horizon_s,
        )
        rows.append(
            {
                "truth_poison": bool(event["truth_poison"]),
                "detector_promote": bool(event["detector_promote"]),
                "ever_visible": bool(
                    exposure > 0
                    or (
                        isinstance(item, Mapping)
                        and item.get("was_visible") is True
                    )
                ),
                "at_end": state in VISIBLE,
                "exposure_seconds": exposure,
            }
        )

    def group(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "denominator": len(selected),
            "ever_visible": sum(bool(row["ever_visible"]) for row in selected),
            "at_end": sum(bool(row["at_end"]) for row in selected),
            "exposure_seconds": _maybe_percentiles(
                [float(row["exposure_seconds"]) for row in selected]
            ),
        }

    poison = [row for row in rows if row["truth_poison"]]
    clean = [row for row in rows if not row["truth_poison"]]
    false_promotions = [
        row for row in rows
        if row["truth_poison"] and row["detector_promote"]
    ]
    false_refusals = [
        row for row in rows
        if not row["truth_poison"] and not row["detector_promote"]
    ]
    return {
        "denominator": len(rows),
        "cohort": "measured admissions accepted by the fixed window horizon",
        "poison": group(poison),
        "clean": group(clean),
        "d1_false_promotion_realized": group(false_promotions),
        "d1_false_refusal_realized": group(false_refusals),
    }


def _record_values(
    records: Sequence[Mapping[str, Any]],
    field: str,
) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(field)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            values.append(float(value))
    return values


def _quick_panel_b_matrix(run_number: int) -> list[dict[str, Any]]:
    return [
        {
            **cell,
            "cell_id": f"quick-{cell['cell_id']}",
            "warmup_s": 0.2,
            "measured_s": 0.8,
            "admission_rate_per_s": 5.0,
        }
        for cell in panel_b_matrix(run_number)
        if cell["query_concurrency"] in {1, 4}
    ]


async def _run_panel_b_cell(
    backend: MilvusBackend,
    frozen: FrozenData,
    spec: Mapping[str, Any],
    *,
    cell_index: int,
    run_number: int,
    quick: bool,
    static_n: int = STATIC_N,
) -> dict[str, Any]:
    baseline = str(spec["baseline"])
    switches = BASELINE_SWITCHES[baseline]
    warmup_s = float(spec["warmup_s"])
    measured_s = float(spec["measured_s"])
    total_s = warmup_s + measured_s
    admission_rate = float(spec["admission_rate_per_s"])
    query_indices = query_order(run_number)
    sentinel_n = 5 if quick else SENTINEL_N
    sentinel_before = await _sentinel(
        backend, frozen.query_vectors, query_indices, sentinel_n
    )

    dynamic_lo = 1_000_000_000 + run_number * 10_000_000 + cell_index * 100_000
    offered_total = int(math.ceil(total_s * admission_rate))
    dynamic_hi = dynamic_lo + offered_total + 1
    provider = None if baseline == "B1" else FrozenReplay(frozen.scores)
    backend_view = PreinsertedNoopBackendView(backend)
    system = StaticPassthroughSystem(
        backend_view,
        mode="postfilter",
        Tp=TP_S,
        verify_cost=0.0,
        over_fetch=3,
        decouple=True,
        verifier_concurrency=VERIFIER_CONCURRENCY,
        verify=bool(switches["verify"]),
        sync=bool(switches["sync"]),
        deadline=bool(switches["deadline"]),
        verifier=provider,
        static_n=static_n,
    )

    admission_events: list[dict[str, Any]] = []
    admission_tasks: set[asyncio.Task[Any]] = set()
    query_latencies: list[float] = []
    query_error_types: Counter[str] = Counter()
    query_attempts = 0
    query_completed = 0
    query_completed_in_window = 0
    query_finished_after_window = 0
    query_ordinal = 0
    query_inflight = 0
    max_query_inflight = 0
    start = time.monotonic()
    measured_start = start + warmup_s
    horizon = start + total_s
    stop_queries = asyncio.Event()

    async def one_admission(event: dict[str, Any]) -> None:
        item_id = int(event["item_id"])
        key = str(event["item_key"])
        vector = frozen.embeddings[frozen.key_to_row[key]].tolist()
        try:
            # The storage RPC is outside System and outside the event loop.  In
            # particular, a slow insert cannot delay the independent deadline.
            await asyncio.to_thread(backend.insert, item_id, vector, True)
            event["store_insert_ack_s"] = time.monotonic() - start
            backend_view.mark_preinserted(item_id)
            # Evaluator truth is intentionally omitted.  B2's preinserted row
            # remains ineligible while its control state is None.
            system.admit(
                item_id,
                vector,
                verifier_key=key,
                verifier_input={
                    "execution_mode":
                        "score_once_frozen_decision_service_replay"
                },
            )
            event["accepted"] = True
            event["accepted_s"] = time.monotonic() - start
        except Exception as exc:
            event["error_type"] = type(exc).__name__

    async def admission_stream() -> None:
        for ordinal in range(offered_total):
            target_s = ordinal / admission_rate
            delay = start + target_s - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            dispatched_s = time.monotonic() - start
            key = frozen.test_order[ordinal % len(frozen.test_order)]
            event = {
                "ordinal": ordinal,
                "item_id": dynamic_lo + ordinal,
                "item_key": key,
                "target_s": target_s,
                "dispatched_s": dispatched_s,
                "dispatch_lag_s": max(0.0, dispatched_s - target_s),
                "measured": warmup_s <= target_s < total_s,
                "accepted": False,
                "accepted_s": None,
                "store_insert_ack_s": None,
                "error_type": None,
            }
            try:
                admission_events.append(event)
                task = asyncio.create_task(one_admission(event))
                admission_tasks.add(task)
            except Exception as exc:
                event["error_type"] = type(exc).__name__
        if admission_tasks:
            await asyncio.gather(*admission_tasks)

    async def query_client(_client_index: int) -> None:
        nonlocal query_attempts, query_completed, query_completed_in_window
        nonlocal query_finished_after_window, query_ordinal
        nonlocal query_inflight, max_query_inflight
        while not stop_queries.is_set():
            query_started = time.monotonic()
            relative = query_started - start
            if relative >= total_s:
                return
            ordinal = query_ordinal
            query_ordinal += 1
            query = frozen.query_vectors[
                query_indices[ordinal % len(query_indices)]
            ]
            measured = relative >= warmup_s
            if measured:
                query_attempts += 1
            query_inflight += 1
            max_query_inflight = max(max_query_inflight, query_inflight)
            try:
                await asyncio.to_thread(system.query, query.tolist(), TOP_K)
                if measured:
                    query_completed += 1
                    if time.monotonic() - start <= total_s:
                        query_completed_in_window += 1
                    else:
                        query_finished_after_window += 1
            except Exception as exc:
                if measured:
                    query_error_types[type(exc).__name__] += 1
                    if time.monotonic() - start > total_s:
                        query_finished_after_window += 1
            finally:
                elapsed = time.monotonic() - query_started
                query_inflight -= 1
                if measured:
                    query_latencies.append(elapsed)

    admission_driver = asyncio.create_task(admission_stream())
    query_drivers = [
        asyncio.create_task(query_client(client_index))
        for client_index in range(int(spec["query_concurrency"]))
    ]

    # The observation horizon is owned by one independent timer.  No workload
    # task is drained or cancelled before these three snapshots exist.
    await asyncio.sleep(max(0.0, horizon - time.monotonic()))
    snapshot_taken = time.monotonic()
    stop_queries.set()
    system_snapshot = _snapshot_system(
        system, dynamic_lo, dynamic_hi, horizon
    )
    admission_snapshot = copy.deepcopy(admission_events)
    verifier_snapshot = copy.deepcopy(system_snapshot["verifier_records"])

    orchestration_results = await asyncio.gather(
        admission_driver, *query_drivers, return_exceptions=True
    )
    orchestration_errors = [
        result
        for result in orchestration_results
        if isinstance(result, BaseException)
        and not isinstance(result, asyncio.CancelledError)
    ]

    measured_admissions = [
        event for event in admission_events if event["measured"]
    ]
    accepted_total = [event for event in admission_events if event["accepted"]]
    failed_total = [event for event in admission_events if not event["accepted"]]
    accepted_admissions = [
        event for event in measured_admissions if event["accepted"]
    ]
    accepted_in_window = [
        event
        for event in accepted_admissions
        if isinstance(event.get("accepted_s"), (int, float))
        and float(event["accepted_s"]) <= total_s
    ]
    failed_admissions = [
        event for event in measured_admissions if not event["accepted"]
    ]
    accepted_in_window_ids = {
        int(event["item_id"]) for event in accepted_in_window
    }
    states_at_end = {
        item_id: item.get("state")
        for item_id, item in system_snapshot["items"].items()
        if item_id in accepted_in_window_ids
    }
    transitions_at_end = [
        transition
        for transition in system_snapshot["transitions"]
        if int(transition[1]) in accepted_in_window_ids
    ]
    right_censored = sum(
        state == State.PROVISIONAL for state in states_at_end.values()
    )
    pending_not_visible = sum(state is None for state in states_at_end.values())
    deadline_hidden = len(
        {
            int(item_id)
            for _event_time, item_id, _old, new in transitions_at_end
            if new == State.HIDDEN
        }
    )

    b2_visibility_violations = 0
    if baseline == "B2":
        for item_id, item in system_snapshot["items"].items():
            if item_id not in accepted_in_window_ids:
                continue
            if item.get("was_visible"):
                record = item.get("verifier_record", {})
                if (
                    record.get("passes") is not True
                    or item.get("ts", {}).get("visible", -math.inf)
                    < item.get("ts", {}).get("verify_commit", math.inf)
                ):
                    b2_visibility_violations += 1

    b1_immediate_visibility_violations = 0
    if baseline == "B1":
        for event in accepted_in_window:
            item = system_snapshot["items"].get(int(event["item_id"]))
            if item is None or not item.get("was_visible"):
                b1_immediate_visibility_violations += 1
    b3_hidden_violations = (
        sum(
            1
            for _event_time, _item_id, _old, new in transitions_at_end
            if new == State.HIDDEN
        )
        if baseline == "B3"
        else 0
    )
    finite_deadline_violations = 0
    if baseline in {"B1", "B2", "B3"}:
        finite_deadline_violations = sum(
            isinstance(item.get("deadline"), (int, float))
            and not isinstance(item.get("deadline"), bool)
            and math.isfinite(float(item["deadline"]))
            for item_id, item in system_snapshot["items"].items()
            if item_id in accepted_in_window_ids
        )

    shutdown_errors = await system.shutdown()
    verifier_records = [dict(record) for record in system.verifier_records]
    status_counts = Counter(str(record.get("status")) for record in verifier_records)
    cleanup = await asyncio.to_thread(
        backend.delete_range, dynamic_lo, dynamic_hi
    )
    try:
        residual_ids = await asyncio.to_thread(
            backend.ids_in_range, dynamic_lo, dynamic_hi
        )
    except Exception:
        residual_ids = None
    sentinel_after = await _sentinel(
        backend, frozen.query_vectors, query_indices, sentinel_n
    )
    before_p95 = sentinel_before["latency_s"]["p95"]
    after_p95 = sentinel_after["latency_s"]["p95"]
    movement = (
        max(before_p95, after_p95) / min(before_p95, after_p95)
        if before_p95 > 0 and after_p95 > 0
        else None
    )

    measured_ids = {int(event["item_id"]) for event in accepted_admissions}
    measured_verifier_records = [
        record
        for record in verifier_records
        if int(record.get("item_id", -1)) in measured_ids
    ]
    verifier_total = len(verifier_records)
    verifier_terminal = sum(
        status_counts.get(status, 0)
        for status in ("COMMITTED", "CANCELLED", "FAILED")
    )
    snapshot_status_counts = Counter(
        str(record.get("status")) for record in verifier_snapshot
    )
    all_async_errors = [
        *orchestration_errors,
        *shutdown_errors,
    ]
    policy_checks = {
        "postfilter_static_passthrough_mode": (
            system.mode == "postfilter"
            and isinstance(system, StaticPassthroughSystem)
        ),
        "B1_no_verifier_calls": baseline != "B1" or verifier_total == 0,
        "B1_immediate_visibility": (
            baseline != "B1" or b1_immediate_visibility_violations == 0
        ),
        "B2_no_precommit_visibility": (
            baseline != "B2" or b2_visibility_violations == 0
        ),
        "B3_no_deadline": (
            baseline != "B3"
            or (
                not switches["deadline"]
                and b3_hidden_violations == 0
                and finite_deadline_violations == 0
            )
        ),
        "B4_independent_deadline": baseline != "B4" or system.decouple is True,
    }
    accounting = {
        "all_admissions_offered_equals_accepted_plus_failed": (
            len(admission_events) == len(accepted_total) + len(failed_total)
        ),
        "admission_offered_equals_accepted_plus_failed": (
            len(measured_admissions)
            == len(accepted_admissions) + len(failed_admissions)
        ),
        "query_attempts_equal_completed_plus_errors": (
            query_attempts
            == query_completed + sum(query_error_types.values())
        ),
        "verifier_records_terminal_after_shutdown": (
            verifier_total == verifier_terminal and not all_async_errors
        ),
        "verifier_records_match_accepted_admissions": (
            verifier_total
            == (0 if baseline == "B1" else len(accepted_total))
        ),
    }
    classified_measured = _classify_events(measured_admissions, frozen)
    accepted_by_id = {
        int(event["item_id"]): event for event in classified_measured
    }
    for event in accepted_by_id.values():
        event["accepted_within_window"] = (
            bool(event.get("accepted"))
            and isinstance(event.get("accepted_s"), (int, float))
            and float(event["accepted_s"]) <= total_s
        )

    return {
        "cell_id": spec["cell_id"],
        "position": cell_index + 1,
        "baseline": baseline,
        "query_concurrency": int(spec["query_concurrency"]),
        "system_mode": "postfilter_static_passthrough",
        "ef": PRIMARY_EF,
        "top_k": TOP_K,
        "tp_s": TP_S,
        "verifier_concurrency": VERIFIER_CONCURRENCY,
        "policy_switches": {**switches, "decouple": True},
        "warmup_s": warmup_s,
        "measured_s": measured_s,
        "actual_cell_elapsed_s": snapshot_taken - start,
        "fixed_horizon_s": total_s,
        "snapshot_before_drain": True,
        "horizon_admission_snapshot": {
            "offered": len(admission_snapshot),
            "accepted": sum(
                bool(event.get("accepted")) for event in admission_snapshot
            ),
        },
        "thread_pool_workers": THREAD_POOL_WORKERS,
        "async_error_types": sorted(
            type(error).__name__ for error in all_async_errors
        ),
        "admissions": {
            "offered_rate_per_s": admission_rate,
            "offered_including_warmup": len(admission_events),
            "accepted_including_warmup": len(accepted_total),
            "failed_including_warmup": len(failed_total),
            "offered": len(measured_admissions),
            "accepted": len(accepted_admissions),
            "accepted_within_window": len(accepted_in_window),
            "accepted_after_window": (
                len(accepted_admissions) - len(accepted_in_window)
            ),
            "failed": len(failed_admissions),
            "achieved_rate_per_s": len(accepted_in_window) / measured_s,
            "dispatch_lag_s": _maybe_percentiles(
                [float(event["dispatch_lag_s"]) for event in measured_admissions]
            ),
            "error_types": dict(
                sorted(
                    Counter(
                        str(event["error_type"])
                        for event in failed_admissions
                    ).items()
                )
            ),
            "dynamic_id_range": [dynamic_lo, dynamic_hi],
        },
        "queries": {
            "attempts": query_attempts,
            "completed": query_completed,
            "completed_within_window": query_completed_in_window,
            "finished_after_window": query_finished_after_window,
            "errors": int(sum(query_error_types.values())),
            "achieved_qps": query_completed_in_window / measured_s,
            "latency_s": _maybe_percentiles(query_latencies),
            "error_types": dict(sorted(query_error_types.items())),
            "max_inflight": max_query_inflight,
        },
        "verifier": {
            "execution": (
                "none_undefended"
                if baseline == "B1"
                else "measured-service trace replay, not online detector computation"
            ),
            "records_total_including_warmup": verifier_total,
            "measured_records": len(measured_verifier_records),
            "status_counts": dict(sorted(status_counts.items())),
            "window_snapshot_status_counts":
                dict(sorted(snapshot_status_counts.items())),
            "queue_wait_s": _maybe_percentiles(
                _record_values(measured_verifier_records, "queue_wait_s")
            ),
            "service_time_s": _maybe_percentiles(
                _record_values(measured_verifier_records, "service_time_s")
            ),
            "integrated_latency_s": _maybe_percentiles(
                _record_values(measured_verifier_records, "integrated_latency_s")
            ),
            "queue_depth_at_enqueue": _maybe_percentiles(
                _record_values(
                    measured_verifier_records, "queue_depth_at_enqueue"
                )
            ),
            "cancelled": int(status_counts.get("CANCELLED", 0)),
            "failed": int(status_counts.get("FAILED", 0)),
            "shutdown_error_types": sorted(
                type(error).__name__ for error in all_async_errors
            ),
        },
        "decision_mix": _decision_mix(
            [event for event in classified_measured if event["accepted"]]
        ),
        "shadow_confusion_512": _shadow_confusion_512(frozen),
        "live_outcomes": _live_outcomes(
            list(accepted_by_id.values()),
            system_snapshot,
            measured_start_s=measured_start,
            horizon_s=horizon,
        ),
        "truth_separation_checks": {
            "no_truth_field_in_pre_snapshot_events": not any(
                "truth_poison" in event for event in admission_snapshot
            ),
            "system_received_no_evaluator_truth": all(
                item.get("content_bad") is False
                for item in system_snapshot["items"].values()
            ),
            "decision_provider_has_no_truth_mapping": (
                provider is None or not hasattr(provider, "_truth_poison")
            ),
            "classification_ran_after_snapshot": True,
        },
        "lifecycle": {
            "deadline_hidden_at_window_end": deadline_hidden,
            "right_censored_provisional_at_window_end": right_censored,
            "pending_not_visible_at_window_end": pending_not_visible,
            "B2_precommit_visibility_violations": b2_visibility_violations,
            "B1_immediate_visibility_violations":
                b1_immediate_visibility_violations,
            "B3_hidden_transition_violations": b3_hidden_violations,
        },
        "sentinel": {
            "definition": (
                "identical sequential query burst over the same static HNSW "
                "collection after dynamic cleanup"
            ),
            "before": sentinel_before,
            "after": sentinel_after,
            "p95_movement_ratio": movement,
            "within_1_5x": movement is not None and movement <= 1.5,
        },
        "cleanup": {
            **cleanup,
            "residual_dynamic_row_count": (
                None if residual_ids is None else len(residual_ids)
            ),
            "residual_dynamic_ids": residual_ids,
            "live_tasks_after_shutdown": len(system.tasks),
            "live_timers_after_shutdown": len(system.timers),
            "physical_preinsert_count": len(backend_view.preinserted),
            "logical_noop_insert_count": len(backend_view.noop_insert_ids),
        },
        "policy_checks": policy_checks,
        "accounting_checks": accounting,
    }


async def run_panel_b(
    backend: MilvusBackend,
    frozen: FrozenData,
    *,
    run_number: int,
    quick: bool,
    static_n: int = STATIC_N,
) -> dict[str, Any]:
    matrix = (
        _quick_panel_b_matrix(run_number)
        if quick
        else panel_b_matrix(run_number)
    )
    cells: list[dict[str, Any]] = []
    for cell_index, spec in enumerate(matrix):
        print(
            f"Panel B {cell_index + 1}/{len(matrix)}: "
            f"{spec['baseline']} concurrency={spec['query_concurrency']}",
            flush=True,
        )
        cells.append(
            await _run_panel_b_cell(
                backend,
                frozen,
                spec,
                cell_index=cell_index,
                run_number=run_number,
                quick=quick,
                static_n=static_n,
            )
        )
    return {
        "formal_matrix_cell_count": 16,
        "executed_matrix_cell_count": len(cells),
        "baseline_order": list(baseline_order(run_number)),
        "concurrency_order": list(concurrency_order(run_number)),
        "run5_balance_note": (
            "run 5 repeats Williams row 1 and is reported separately"
            if run_number == 5
            else None
        ),
        "query_order_seed": query_order_seed(run_number),
        "query_order_ordinals": list(query_order(run_number)),
        "cells": cells,
    }


def _gate(gate_id: int, name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "name": name,
        "passed": bool(passed),
        "detail": detail,
    }


def _panel_a_primary(panel_a: Mapping[str, Any]) -> Mapping[str, Any]:
    for point in panel_a.get("ef_points", []):
        if point.get("ef") == PRIMARY_EF:
            return point
    raise E3Error("Panel A has no preregistered ef=64 primary point")


def _build_validity_gates(
    *,
    formal: bool,
    frozen: FrozenData,
    provenance: Mapping[str, Any],
    runtime_backend: Mapping[str, Any],
    population_record: Mapping[str, Any],
    panel_a: Mapping[str, Any],
    panel_b: Mapping[str, Any],
    previous_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    cells = list(panel_b.get("cells", []))
    expected_a_count = PANEL_A_LATENCY_N
    panel_a_points = list(panel_a.get("ef_points", []))
    panel_a_matrix_ok = (
        len(panel_a_points) == len(PANEL_A_EF)
        and [point.get("ef") for point in panel_a_points] == list(PANEL_A_EF)
        and all(
            point.get("measured_attempt_count") == expected_a_count
            for point in panel_a_points
        )
    )
    if not formal:
        panel_a_matrix_ok = False

    process_value = provenance.get("process_start_metric", {}).get("value")
    prior_process_values = [
        record.get("document", {})
        .get("provenance", {})
        .get("process_start_metric", {})
        .get("value")
        for record in previous_results
    ]
    prior_process_values = [
        value for value in prior_process_values if isinstance(value, (int, float))
    ]
    accepted_previous = [
        record
        for record in previous_results
        if record.get("document", {}).get("local_validity_accepted") is True
    ]
    prior_commits = [
        record.get("document", {}).get("provenance", {}).get("git", {}).get("commit")
        for record in accepted_previous
    ]
    prior_fingerprints = [
        record.get("document", {})
        .get("provenance", {})
        .get("runtime_fingerprint", {})
        .get("sha256")
        for record in accepted_previous
    ]
    current_commit = provenance.get("git", {}).get("commit")
    current_fingerprint = provenance.get("runtime_fingerprint", {}).get("sha256")

    accounting_ok = all(
        all(cell.get("accounting_checks", {}).values())
        and cell.get("async_error_types") == []
        for cell in cells
    )
    policy_ok = all(
        all(cell.get("policy_checks", {}).values())
        and all(cell.get("truth_separation_checks", {}).values())
        for cell in cells
    )
    cleanup_ok = all(
        cell.get("cleanup", {}).get("delete_confirmed") is True
        and cell.get("cleanup", {}).get("residual_dynamic_row_count") == 0
        and cell.get("cleanup", {}).get("live_tasks_after_shutdown") == 0
        and cell.get("cleanup", {}).get("live_timers_after_shutdown") == 0
        for cell in cells
    ) and runtime_backend.get("server_row_count_after_panels") == STATIC_N
    sentinel_ok = all(
        cell.get("sentinel", {}).get("within_1_5x") is True
        and cell.get("sentinel", {}).get("before", {}).get("errors") == 0
        and cell.get("sentinel", {}).get("after", {}).get("errors") == 0
        and cell.get("sentinel", {}).get("before", {}).get("started")
        == cell.get("sentinel", {}).get("before", {}).get("completed")
        and cell.get("sentinel", {}).get("after", {}).get("started")
        == cell.get("sentinel", {}).get("after", {}).get("completed")
        for cell in cells
    )
    backend_ok = (
        runtime_backend.get("deployment_mode") == "standalone"
        and "2.4.15" in str(runtime_backend.get("server_version"))
        and _index_is_ready(runtime_backend.get("index_info", {}))
        and runtime_backend.get("index_info", {}).get("indexed_rows")
        == STATIC_N
        and _index_is_ready(runtime_backend.get("index_info_after_panels", {}))
        and runtime_backend.get("index_info_after_panels", {}).get(
            "indexed_rows"
        ) >= STATIC_N
    )
    # Gate 7 is a byte-level reconstruction of the frozen execution matrix,
    # not merely a count.  The run number is encoded in every preregistered id.
    matrix_run_number = None
    if cells:
        try:
            matrix_run_number = int(str(cells[0].get("cell_id")).split("__", 1)[0][4:])
        except (TypeError, ValueError, IndexError):
            matrix_run_number = None
    expected_matrix = (
        panel_b_matrix(matrix_run_number)
        if matrix_run_number in RUN_NUMBERS
        else []
    )
    matrix_ok = (
        len(cells) == len(expected_matrix) == 16
        and all(
            cell.get("cell_id") == spec["cell_id"]
            and cell.get("baseline") == spec["baseline"]
            and cell.get("query_concurrency") == spec["query_concurrency"]
            and cell.get("ef") == spec["ef"]
            and cell.get("top_k") == spec["top_k"]
            and cell.get("tp_s") == spec["tp_s"]
            and cell.get("verifier_concurrency") == spec["verifier_concurrency"]
            and cell.get("warmup_s") == spec["warmup_s"]
            and cell.get("measured_s") == spec["measured_s"]
            and cell.get("system_mode") == "postfilter_static_passthrough"
            for cell, spec in zip(cells, expected_matrix)
        )
    )
    if not formal:
        matrix_ok = False

    return [
        _gate(
            1,
            "frozen_parent_hashes",
            set(frozen.parent_hashes)
            == set(FROZEN_PARENT_SHA256) | {TRANSITIVE_LABEL_PATH},
            dict(frozen.parent_hashes),
        ),
        _gate(
            2,
            "clean_tree_same_commit_runtime",
            provenance.get("git", {}).get("tracked_tree_clean") is True
            and all(value == current_commit for value in prior_commits)
            and all(value == current_fingerprint for value in prior_fingerprints),
            {
                "tracked_tree_clean":
                    provenance.get("git", {}).get("tracked_tree_clean"),
                "prior_formal_attempts_seen": len(previous_results),
                "prior_locally_accepted_results_seen": len(accepted_previous),
                "cross_run_final_check": "pending until five files exist",
            },
        ),
        _gate(3, "standalone_hnsw", backend_ok, runtime_backend),
        _gate(
            4,
            "independent_process_start",
            isinstance(process_value, (int, float))
            and process_value not in prior_process_values,
            {
                "current": process_value,
                "previous": prior_process_values,
                "cross_run_five_distinct_check": "pending until five files exist",
            },
        ),
        _gate(
            5,
            "static_population_and_queries",
            population_record.get("static_n") == STATIC_N
            and population_record.get("unique_primary_keys") == STATIC_N
            and panel_a.get("executed_query_count") == QUERY_N,
            {
                "static_n": population_record.get("static_n"),
                "unique_primary_keys":
                    population_record.get("unique_primary_keys"),
                "query_n": panel_a.get("executed_query_count"),
            },
        ),
        _gate(
            6,
            "panel_a_complete",
            panel_a_matrix_ok,
            {
                "ef": [point.get("ef") for point in panel_a_points],
                "attempt_counts": [
                    point.get("measured_attempt_count")
                    for point in panel_a_points
                ],
            },
        ),
        _gate(
            7,
            "panel_b_complete",
            matrix_ok,
            {
                "executed": len(cells),
                "formal_expected": 16,
                "exact_order_and_configuration": matrix_ok,
            },
        ),
        _gate(
            8,
            "cell_accounting_balances",
            accounting_ok,
            [
                cell["cell_id"]
                for cell in cells
                if not all(cell.get("accounting_checks", {}).values())
            ],
        ),
        _gate(
            9,
            "baseline_semantics",
            policy_ok,
            [
                cell["cell_id"]
                for cell in cells
                if (
                    not all(cell.get("policy_checks", {}).values())
                    or not all(cell.get("truth_separation_checks", {}).values())
                )
            ],
        ),
        _gate(
            10,
            "dynamic_cleanup",
            cleanup_ok,
            [
                cell["cell_id"]
                for cell in cells
                if not (
                    cell.get("cleanup", {}).get("delete_confirmed") is True
                    and cell.get("cleanup", {}).get(
                        "residual_dynamic_row_count"
                    ) == 0
                )
            ],
        ),
        _gate(
            11,
            "sentinel_p95_movement",
            sentinel_ok,
            {
                cell["cell_id"]:
                    cell.get("sentinel", {}).get("p95_movement_ratio")
                for cell in cells
            },
        ),
        _gate(
            12,
            "strict_json",
            True,
            "enforced by strict_json_bytes before exclusive creation",
        ),
    ]


async def run_experiment(
    *,
    run_number: int,
    uri: str,
    metrics_url: str | None = None,
    quick: bool,
) -> dict[str, Any]:
    run_number = _validate_run_number(run_number)
    formal = not quick
    started_utc = _utc_now()
    git = _git_provenance()
    fingerprint = runtime_fingerprint()
    environment = _environment_descriptor()
    frozen = load_frozen()
    previous = _previous_formal_results()
    if formal:
        _enforce_formal_stopping_rule(run_number, previous)

    if formal and not git["tracked_tree_clean"]:
        raise E3Error(
            "formal E3 refuses a dirty tracked tree; commit the runner first"
        )

    if metrics_url is None:
        parsed_uri = urlparse(uri)
        if not parsed_uri.hostname:
            raise E3Error("cannot derive metrics endpoint from Milvus URI")
        metrics_url = f"http://{parsed_uri.hostname}:9091/metrics"
    process_metric = _process_start_metric(metrics_url)
    executor = ThreadPoolExecutor(
        max_workers=THREAD_POOL_WORKERS,
        thread_name_prefix="w2d-e3-rpc",
    )
    asyncio.get_running_loop().set_default_executor(executor)
    static_n = 2_000 if quick else STATIC_N
    population = build_population(frozen.embeddings, static_n=static_n)
    collection = (
        f"{COLLECTION_PREFIX}{'quick_' if quick else ''}r{run_number}_"
        f"{int(time.time())}_{os.getpid()}"
    )
    backend: MilvusBackend | None = None
    final_drop: dict[str, Any] | None = None
    try:
        backend = MilvusBackend(
            dim=DIM,
            uri=uri,
            collection=collection,
            consistency_level=CONSISTENCY_LEVEL,
            index_type="HNSW",
            index_params={
                "M": HNSW_M,
                "efConstruction": HNSW_EF_CONSTRUCTION,
            },
            search_params={"ef": PRIMARY_EF},
        )
        server_info = backend.server_info(uri)
        if formal and (
            server_info.get("deployment_mode") != "standalone"
            or "2.4.15" not in str(server_info.get("server_version"))
        ):
            raise E3Error(f"wrong Milvus deployment/version: {server_info}")

        insertion = await asyncio.to_thread(
            _insert_population, backend, population
        )
        build_started = time.monotonic()
        initial_index_settlement = await asyncio.to_thread(
            _wait_settled_hnsw,
            backend,
            expected_indexed_rows=len(population),
        )
        index_build_s = time.monotonic() - build_started
        index_state_observed = initial_index_settlement["state_observed"]
        index_info = initial_index_settlement["index_info"]
        if not _index_is_ready(index_info):
            raise E3Error(f"HNSW is not the effective loaded index: {index_info}")
        row_count = await asyncio.to_thread(
            _collection_row_count, backend, len(population), 180.0
        )
        if formal and row_count != STATIC_N:
            raise E3Error(
                f"formal static row count is {row_count}, expected {STATIC_N}"
            )

        norms = np.linalg.norm(population, axis=1)
        population_record = {
            "static_n": int(len(population)),
            "formal_static_n": STATIC_N,
            "frozen_n": FROZEN_N,
            "synthetic_extension_n": int(len(population) - FROZEN_N),
            "synthetic_extension_seed": BACKGROUND_SEED,
            "graph_build_count": 1,
            "background_seed": BACKGROUND_SEED,
            "single_graph_build": True,
            "dimension": DIM,
            "dtype": str(population.dtype),
            "primary_key_range": [0, int(len(population))],
            "unique_primary_keys": int(len(population)),
            "normalization_max_abs_error": float(
                np.max(np.abs(norms - 1.0))
            ),
            "synthetic_extension_interpretation": (
                "systems pressure load; not a real-text corpus and not a "
                "detector-quality/attack-generalization denominator"
            ),
            "insert": insertion,
            "server_row_count_after_index": row_count,
            "index_build_s": index_build_s,
            "wait_index_state_observed": index_state_observed,
        }
        print(
            f"Panel A: {len(frozen.query_vectors) if formal else 12} queries, "
            f"population={len(population)}",
            flush=True,
        )
        panel_a = await run_panel_a(
            backend, population, frozen, quick=quick
        )
        panel_b = await run_panel_b(
            backend,
            frozen,
            run_number=run_number,
            quick=quick,
            static_n=static_n,
        )
        post_panel_index_settlement = await asyncio.to_thread(
            _wait_settled_hnsw,
            backend,
            minimum_indexed_rows=len(population),
        )
        index_state_observed_after = (
            post_panel_index_settlement["state_observed"]
        )
        index_wait_after_panels_s = post_panel_index_settlement["wait_s"]
        index_info_after = post_panel_index_settlement["index_info"]
        row_count_after_panels = await asyncio.to_thread(
            _collection_row_count, backend, len(population), 180.0
        )
        if not _index_is_ready(index_info_after):
            raise E3Error(
                "HNSW did not return to a finished loaded state after "
                f"Panel B cleanup: {index_info_after}"
            )
        if row_count_after_panels != len(population):
            raise E3Error(
                "Panel B cleanup did not restore the live collection to the "
                f"static population: {row_count_after_panels} != "
                f"{len(population)}"
            )

        runtime_backend = {
            **server_info,
            "implementation": "MilvusBackend",
            "metric": "COSINE",
            "vector_dim": DIM,
            "consistency_level": CONSISTENCY_LEVEL,
            "index_type_requested": "HNSW",
            "index_params_requested": {
                "M": HNSW_M,
                "efConstruction": HNSW_EF_CONSTRUCTION,
            },
            "search_primary": {"ef": PRIMARY_EF},
            "index_info": index_info,
            "index_info_after_panels": index_info_after,
            "index_state_observed_after_panels":
                index_state_observed_after,
            "index_wait_after_panels_s": index_wait_after_panels_s,
            "server_row_count_after_panels": row_count_after_panels,
            "collection": collection,
            "thread_pool": {
                "implementation": "concurrent.futures.ThreadPoolExecutor",
                "max_workers": THREAD_POOL_WORKERS,
                "purpose": "Milvus insert/search/cleanup RPC isolation",
            },
        }
        provenance = {
            "git": git,
            "runtime_fingerprint": fingerprint,
            "environment": environment,
            "process_start_metric": process_metric,
            "preregistration": {
                "parent_commit": "2c414c1",
                "document_commit": "e09b056",
                "amendment_a1_commit": "a4f31c6",
                "documents": [
                    "W2D-E3-PREREGISTRATION.md",
                    "W2D-E3-PREREGISTRATION-AMENDMENT-A1.md",
                ],
            },
        }
        gates = _build_validity_gates(
            formal=formal,
            frozen=frozen,
            provenance=provenance,
            runtime_backend=runtime_backend,
            population_record=population_record,
            panel_a=panel_a,
            panel_b=panel_b,
            previous_results=previous,
        )
        local_accepted = formal and all(gate["passed"] for gate in gates)
        primary = _panel_a_primary(panel_a)
        primary_recall = primary.get("overall", {}).get("recall_at_5")
        transfer_supported_this_run = (
            local_accepted
            and isinstance(primary_recall, (int, float))
            and primary_recall >= 0.95
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "experiment": "W2D-E3",
            "run_number": run_number,
            "formal": formal,
            "quick_nonformal": quick,
            "started_at_utc": started_utc,
            "finished_at_utc": _utc_now(),
            "status": (
                "quick_nonformal"
                if quick
                else ("locally_accepted" if local_accepted else "locally_rejected")
            ),
            "local_validity_accepted": local_accepted,
            "cross_run_acceptance": (
                "pending five-file verifier"
                if formal
                else "not applicable to quick mode"
            ),
            "transfer_condition": {
                "primary_ef": PRIMARY_EF,
                "per_run_recall_at_5": primary_recall,
                "required_minimum": 0.95,
                "supported_this_run": transfer_supported_this_run,
                "note": (
                    "failure is a reportable outcome and does not authorize "
                    "post-hoc substitution of ef=256"
                ),
            },
            "frozen_parent_hashes": dict(frozen.parent_hashes),
            "provenance": provenance,
            "configuration": {
                "static_n": static_n,
                "formal_static_n": STATIC_N,
                "hnsw": {
                    "M": HNSW_M,
                    "efConstruction": HNSW_EF_CONSTRUCTION,
                    "metric": "COSINE",
                    "consistency": CONSISTENCY_LEVEL,
                },
                "panel_a_ef": list(PANEL_A_EF),
                "panel_a_repeats": (
                    1 if quick else PANEL_A_REPEATS
                ),
                "panel_b_baseline_order": list(baseline_order(run_number)),
                "panel_b_concurrency_order": list(
                    concurrency_order(run_number)
                ),
                "thread_pool_workers": THREAD_POOL_WORKERS,
            },
            "runtime_backend": runtime_backend,
            "population": population_record,
            "panel_a": panel_a,
            "panel_b": panel_b,
            "validity_gates": gates,
            "interpretation_boundary": [
                "single-node Milvus Standalone only",
                "measured-service trace replay, not online detector computation",
                "synthetic extension is not a 100k real-text corpus",
                "no distributed scaling, production capacity, high availability, "
                "fault tolerance, unknown-attack robustness, or end-to-end RAG "
                "answer-safety claim",
            ],
        }
        # Exercise strict serialization before the collection is torn down, so a
        # non-finite measurement cannot be hidden by cleanup or by the CLI.
        strict_json_bytes(result)
        return result
    finally:
        if backend is not None:
            try:
                final_drop = await asyncio.to_thread(backend.drop)
            except Exception as exc:
                final_drop = {
                    "drop_confirmed": False,
                    "error_type": type(exc).__name__,
                }
            # ``result`` may already exist; attach final cleanup without masking
            # an earlier exception. This block is intentionally best-effort.
            if "result" in locals():
                result["final_collection_cleanup"] = final_drop
                cleanup_ok = bool(final_drop.get("drop_confirmed"))
                result["final_collection_cleanup"]["passed"] = cleanup_ok
                if not cleanup_ok:
                    for gate in result.get("validity_gates", []):
                        if gate.get("name") == "dynamic_cleanup":
                            gate["passed"] = False
                            gate["detail"] = {
                                "cell_cleanup": gate.get("detail"),
                                "final_collection_cleanup": final_drop,
                            }
                accepted_after_drop = bool(
                    formal
                    and cleanup_ok
                    and all(
                        gate.get("passed") is True
                        for gate in result.get("validity_gates", [])
                    )
                )
                result["local_validity_accepted"] = accepted_after_drop
                result["status"] = (
                    "quick_nonformal"
                    if quick
                    else (
                        "locally_accepted"
                        if accepted_after_drop
                        else "locally_rejected"
                    )
                )
                result["transfer_condition"]["supported_this_run"] = bool(
                    accepted_after_drop
                    and isinstance(
                        result["transfer_condition"].get("per_run_recall_at_5"),
                        (int, float),
                    )
                    and result["transfer_condition"]["per_run_recall_at_5"] >= 0.95
                )
                result["finished_at_utc"] = _utc_now()
                strict_json_bytes(result)


def _default_output(run_number: int, quick: bool) -> Path:
    name = (
        f"W2D-E3-QUICK-RUN-{run_number}.json"
        if quick
        else f"W2D-E3-RUN-{run_number:02d}.json"
    )
    return HERE / "results" / "w2d-e3" / name


def _parse_run_number(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("run id must be 1..5") from exc
    try:
        return _validate_run_number(parsed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run independent W2D-E3 Milvus Standalone experiment"
    )
    parser.add_argument(
        "--run-number",
        "--run-id",
        dest="run_number",
        required=True,
        type=_parse_run_number,
        help="formal run number 1..5 (--run-id is a compatibility alias)",
    )
    parser.add_argument("--uri", default="http://localhost:19530")
    parser.add_argument(
        "--metrics-url",
        default="http://127.0.0.1:9091/metrics",
        help="Milvus Prometheus endpoint carrying process_start_time_seconds",
    )
    parser.add_argument(
        "--out",
        help=(
            "exclusive output path; defaults to "
            "results/w2d-e3/W2D-E3-(QUICK-)RUN-N.json"
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="non-formal reduced code-path exercise; never admissible evidence",
    )
    args = parser.parse_args(argv)
    output = (
        Path(args.out).expanduser().resolve()
        if args.out
        else _default_output(args.run_number, args.quick)
    )
    authoritative_dir = (HERE / "results" / "w2d-e3").resolve()
    if not args.quick and output.parent != authoritative_dir:
        print(
            f"ERROR: formal output must be directly inside {authoritative_dir}",
            file=sys.stderr,
        )
        return 2
    if output.exists():
        print(f"ERROR: refusing to overwrite {output}", file=sys.stderr)
        return 2
    if not args.quick:
        try:
            _enforce_formal_stopping_rule(
                args.run_number,
                _previous_formal_results(),
            )
        except E3Error as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    try:
        result = asyncio.run(
            run_experiment(
                run_number=args.run_number,
                uri=args.uri,
                metrics_url=args.metrics_url,
                quick=args.quick,
            )
        )
        exclusive_write_json(output, result)
    except BaseException as exc:
        # Retain a formal technical failure under the requested attempt filename.
        # KeyboardInterrupt is re-raised after the record is attempted.
        try:
            failure_process_metric = _process_start_metric(args.metrics_url)
        except Exception:
            failure_process_metric = None
        failure = {
            "schema_version": SCHEMA_VERSION,
            "experiment": "W2D-E3",
            "run_number": args.run_number,
            "formal": not args.quick,
            "quick_nonformal": args.quick,
            "status": "technical_failure",
            "failed_at_utc": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "provenance": {
                "git": _git_provenance(),
                "runtime_fingerprint": runtime_fingerprint(),
                "environment": _environment_descriptor(),
                "process_start_metric": failure_process_metric,
            },
        }
        try:
            exclusive_write_json(output, failure)
            print(f"retained failure artifact: {output}", file=sys.stderr)
        except Exception as write_exc:
            print(
                f"ERROR: {exc}; additionally could not retain failure: {write_exc}",
                file=sys.stderr,
            )
        if isinstance(exc, KeyboardInterrupt):
            raise
        return 2

    print(f"wrote {'quick' if args.quick else 'formal'} E3 result: {output}")
    return 0 if (args.quick or result["local_validity_accepted"]) else 3


if __name__ == "__main__":
    raise SystemExit(main())
