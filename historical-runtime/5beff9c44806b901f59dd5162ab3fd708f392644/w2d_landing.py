#!/usr/bin/env python3
"""Exact, off-path W2D attack-landing evaluation.

This evaluator never imports or executes D1.  It consumes the protocol plan
frozen before test scoring and measures every test poison, one at a time,
against the plan's immutable 768-item retrieval background.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from w2d_metrics import file_sha256, rate_record, strict_json_load


SCHEMA_VERSION = "W2D-landing-v1"
EXPECTED_POPULATIONS = {
    "recipe": 128,
    "natural_cover_suffix_v1": 64,
}


class LandingError(ValueError):
    pass


def _finite_tree(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LandingError(f"{path} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_tree(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_tree(child, f"{path}[{index}]")
        return
    raise LandingError(f"{path} contains unsupported {type(value).__name__}")


def _exclusive_json_dump(value: Any, path: str | os.PathLike[str]) -> None:
    _finite_tree(value)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    with target.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _items(document: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(document, Mapping) or not isinstance(
        document.get("items"), list
    ):
        raise LandingError("safe input document has no items list")
    result: dict[str, Mapping[str, Any]] = {}
    for item in document["items"]:
        if not isinstance(item, Mapping):
            raise LandingError("safe input item is not an object")
        key = item.get("item_key")
        if not isinstance(key, str) or len(key) != 64 or key in result:
            raise LandingError("safe input contains a missing/duplicate key")
        result[key] = item
    return result


def _landing_keys(plan: Mapping[str, Any]) -> list[tuple[str, str]]:
    landing = plan.get("landing_plan")
    if not isinstance(landing, Mapping):
        raise LandingError("protocol plan has no landing_plan")
    raw = landing.get("records")
    if not isinstance(raw, list):
        raise LandingError("landing_plan.records must be a list")
    records: list[tuple[str, str]] = []
    for record in raw:
        if isinstance(record, str):
            key = record
            family = str(
                plan.get("planned_poison_queries", {})
                .get(key, {})
                .get("attack_family", "")
            )
        elif isinstance(record, Mapping):
            key = record.get("item_key", record.get("poison_item_key"))
            family = record.get("attack_family")
        else:
            raise LandingError("landing record must be a key or object")
        if not isinstance(key, str) or not isinstance(family, str):
            raise LandingError("landing record lacks item_key/attack_family")
        records.append((key, family))
    if len(records) != len({key for key, _family in records}):
        raise LandingError("landing plan contains a duplicate item")
    return records


def _query_embedding(plan: Mapping[str, Any], poison_key: str) -> np.ndarray:
    planned = plan.get("planned_poison_queries", {})
    if not isinstance(planned, Mapping) or poison_key not in planned:
        raise LandingError(f"{poison_key} has no planned query roles")
    roles = planned[poison_key].get("roles", {})
    associated = roles.get("attack_associated", {})
    query_key = associated.get("query_item_key")
    materials = plan.get("query_materials", {})
    if not isinstance(query_key, str) or not isinstance(materials, Mapping):
        raise LandingError(f"{poison_key} has no associated query material")
    material = materials.get(query_key)
    if not isinstance(material, Mapping):
        raise LandingError(f"query material {query_key} is absent")
    vector = np.asarray(material.get("embedding"), dtype=np.float64)
    if vector.shape != (384,) or not np.isfinite(vector).all():
        raise LandingError(f"query {query_key} has an invalid embedding")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0:
        raise LandingError(f"query {query_key} has invalid norm")
    vector = vector / norm
    expected = material.get("embedding_sha256")
    actual = hashlib.sha256(
        (
            json.dumps(
                material.get("embedding"),
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    if expected is not None and expected != actual:
        raise LandingError(f"query {query_key} embedding digest mismatch")
    return vector


def evaluate_landing(
    *,
    plan_path: str,
    inputs_path: str,
    embeddings_path: str,
    output_path: str | None = None,
) -> dict[str, Any]:
    plan = strict_json_load(plan_path)
    safe = strict_json_load(inputs_path)
    if not isinstance(plan, Mapping) or not isinstance(safe, Mapping):
        raise LandingError("plan and safe inputs must be JSON objects")

    provenance = plan.get("provenance", {})
    if not isinstance(provenance, Mapping) or not isinstance(
        provenance.get("inputs"), list
    ):
        raise LandingError("protocol provenance is absent")
    by_filename = {
        record.get("filename"): record
        for record in provenance["inputs"]
        if isinstance(record, Mapping)
    }
    for filename, path in (
        ("W2D-detector-inputs.json", inputs_path),
        ("W2D-detector-inputs.npz", embeddings_path),
    ):
        descriptor = by_filename.get(filename)
        expected = (
            descriptor.get("sha256")
            if isinstance(descriptor, Mapping)
            else descriptor
        )
        if expected != file_sha256(path):
            raise LandingError(f"{filename} does not match protocol provenance")

    by_key = _items(safe)
    background = plan.get("retrieval_background", {})
    if not isinstance(background, Mapping):
        raise LandingError("retrieval_background is absent")
    background_keys = background.get("item_keys")
    if not isinstance(background_keys, list) or len(background_keys) != 768:
        raise LandingError("retrieval background must contain exactly 768 keys")
    if len(background_keys) != len(set(background_keys)):
        raise LandingError("retrieval background contains duplicate keys")

    with np.load(embeddings_path, allow_pickle=False) as archive:
        if archive.files != ["embeddings"]:
            raise LandingError("safe NPZ must contain only embeddings")
        matrix = np.asarray(archive["embeddings"], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 384:
        raise LandingError("safe embedding matrix has wrong shape")

    def vector(key: str) -> np.ndarray:
        if key not in by_key:
            raise LandingError(f"safe item {key} is absent")
        row = by_key[key].get("embedding_row")
        if isinstance(row, bool) or not isinstance(row, int):
            raise LandingError(f"safe item {key} has invalid embedding row")
        value = matrix[row]
        norm = float(np.linalg.norm(value))
        if not np.isfinite(value).all() or not math.isfinite(norm) or norm <= 0:
            raise LandingError(f"safe item {key} has invalid vector")
        return value / norm

    background_vectors = {key: vector(key) for key in background_keys}
    planned = _landing_keys(plan)
    records: list[dict[str, Any]] = []
    for poison_key, family in planned:
        if poison_key in background_vectors:
            raise LandingError("poison item is present in retrieval background")
        query = _query_embedding(plan, poison_key)
        candidates = [
            (float(np.dot(query, embedding)), key)
            for key, embedding in background_vectors.items()
        ]
        candidates.append((float(np.dot(query, vector(poison_key))), poison_key))
        if any(not math.isfinite(score) for score, _key in candidates):
            raise LandingError("landing cosine is non-finite")
        ordered = sorted(candidates, key=lambda pair: (-pair[0], pair[1]))
        rank = next(
            index + 1
            for index, (_score, key) in enumerate(ordered)
            if key == poison_key
        )
        records.append(
            {
                "item_key": poison_key,
                "attack_family": family,
                "rank": rank,
                "candidate_rank": rank,
                "landed_top5": rank <= 5,
                "landed": rank <= 5,
                "top5_item_keys": [
                    key for _score, key in ordered[:5]
                ],
            }
        )

    populations: dict[str, Any] = {}
    for family, expected_n in EXPECTED_POPULATIONS.items():
        family_records = [
            record for record in records if record["attack_family"] == family
        ]
        if len(family_records) != expected_n:
            raise LandingError(
                f"{family} landing denominator {len(family_records)} != {expected_n}"
            )
        landed_n = sum(record["landed_top5"] for record in family_records)
        populations[family] = {
            **rate_record(landed_n, expected_n),
            "landed_n": landed_n,
            "n": expected_n,
        }

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "measurement": "B1 off-path exact-cosine attack landing",
        "top_k": 5,
        "tie_break": "descending cosine, then ascending opaque item_key",
        "candidate_population": (
            "one poison item at a time plus the frozen 768-item "
            "retrieval background"
        ),
        "protocol_plan_sha256": file_sha256(plan_path),
        "safe_inputs_sha256": file_sha256(inputs_path),
        "embeddings_sha256": file_sha256(embeddings_path),
        "retrieval_background_sha256": _sha256_json(background_keys),
        "populations": populations,
        "items": sorted(records, key=lambda record: record["item_key"]),
    }
    _finite_tree(artifact)
    if output_path is not None:
        _exclusive_json_dump(artifact, output_path)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    evaluate_landing(
        plan_path=args.plan,
        inputs_path=args.inputs,
        embeddings_path=args.embeddings,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
