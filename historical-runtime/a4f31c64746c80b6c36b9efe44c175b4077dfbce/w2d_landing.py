#!/usr/bin/env python3
"""Exact, off-path W2D attack-landing evaluation.

This evaluator never imports or executes D1.  It consumes the frozen protocol
plan and measures every test poison, one at a time, against the plan's
immutable 768-item retrieval background.  Amendment A9 records the plan's
actual construction timing.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from w2d_metrics import (
    file_sha256,
    rate_record,
    strict_json_load_bytes,
)
from w2d_snapshot import (
    OutputOwnership,
    OutputOwnershipError,
    assert_owned_output,
    exclusive_create_bytes,
    preserve_failed_output,
)


SCHEMA_VERSION = "W2D-landing-v1"
EXPECTED_POPULATIONS = {
    "recipe": 128,
    "natural_cover_suffix_v1": 64,
}
LANDING_FIELDS = frozenset(
    {
        "background_count",
        "background_id",
        "baseline",
        "candidate_rule",
        "metric",
        "off_path",
        "populations",
        "records",
        "result_requirements",
        "tie_break",
        "top_k",
    }
)
LANDING_RECORD_FIELDS = frozenset(
    {
        "landing_order_position_within_attack",
        "item_key",
        "source_group",
        "attack_family",
        "attack_variant",
        "topic",
        "query_role",
        "query_item_key",
        "retrieval_background_id",
        "candidate_added_alone_item_key",
        "candidate_corpus_size",
    }
)
LANDING_POPULATION_FIELDS = frozenset(
    {
        "item_keys",
        "expected_denominator",
        "failed_landing_policy",
    }
)
QUERY_MATERIAL_FIELDS = frozenset(
    {
        "attack_family",
        "attack_variant",
        "embedding",
        "embedding_encoding",
        "embedding_sha256",
        "query_item_key",
        "source_group",
        "text",
        "topic",
    }
)
EXPECTED_RESULT_REQUIREMENTS = {
    "per_item_fields": [
        "item_key",
        "attack_family",
        "landed",
        "top5_item_keys",
        "candidate_rank",
    ],
    "summary_fields": [
        "numerator",
        "denominator",
        "rate",
        "wilson95",
    ],
    "no_resampling": True,
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


def _assert_owned_output(ownership: OutputOwnership) -> None:
    try:
        assert_owned_output(ownership)
    except OutputOwnershipError as exc:
        raise LandingError(str(exc)) from exc


def _preserve_failed_output(ownership: OutputOwnership) -> None:
    """Validate a failed output but never delete its public pathname."""
    try:
        preserve_failed_output(ownership)
    except OutputOwnershipError as exc:
        raise LandingError(str(exc)) from exc


def _exclusive_json_dump(
    value: Any,
    path: str | os.PathLike[str],
) -> OutputOwnership:
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
    ).encode("utf-8")
    try:
        return exclusive_create_bytes(target, payload)
    except OutputOwnershipError as exc:
        raise LandingError(str(exc)) from exc


def _snapshot_bytes(path: str) -> tuple[bytes, str]:
    with open(path, "rb") as stream:
        payload = stream.read()
    return payload, hashlib.sha256(payload).hexdigest()


def _assert_frozen_paths_unchanged(
    frozen_paths: Mapping[str, str],
) -> None:
    for path, expected_sha256 in frozen_paths.items():
        try:
            actual_sha256 = file_sha256(path)
        except OSError as exc:
            raise LandingError(
                f"frozen landing input disappeared during evaluation: {path}"
            ) from exc
        if actual_sha256 != expected_sha256:
            raise LandingError(
                f"frozen landing input changed during evaluation: {path}"
            )


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


def _landing_records(
    plan: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    landing = plan.get("landing_plan")
    if not isinstance(landing, Mapping):
        raise LandingError("protocol plan has no landing_plan")
    if set(landing) != LANDING_FIELDS:
        raise LandingError("landing_plan fields differ from the producer schema")
    if (
        landing.get("baseline") != "B1"
        or landing.get("off_path") is not True
        or landing.get("metric") != "exact_cosine"
        or type(landing.get("top_k")) is not int
        or landing.get("top_k") != 5
        or landing.get("tie_break")
        != "descending cosine then ascending opaque item_key"
        or landing.get("background_id") != "frozen_reference_768"
        or type(landing.get("background_count")) is not int
        or landing.get("background_count") != 768
        or landing.get("candidate_rule")
        != (
            "add exactly one modified test poison passage to the immutable "
            "background; query with that item's attack_associated query"
        )
    ):
        raise LandingError("landing_plan constants differ from the producer")
    result_requirements = landing.get("result_requirements")
    if (
        not isinstance(result_requirements, Mapping)
        or result_requirements != EXPECTED_RESULT_REQUIREMENTS
        or type(result_requirements.get("no_resampling")) is not bool
    ):
        raise LandingError(
            "landing result requirements differ from the producer"
        )
    raw = landing.get("records")
    if not isinstance(raw, list):
        raise LandingError("landing_plan.records must be a list")
    if len(raw) != sum(EXPECTED_POPULATIONS.values()):
        raise LandingError("landing plan must contain exactly 192 records")
    records: list[Mapping[str, Any]] = []
    for record in raw:
        if not isinstance(record, Mapping):
            raise LandingError("landing record must be an object")
        if set(record) != LANDING_RECORD_FIELDS:
            raise LandingError(
                "landing record fields differ from the producer schema"
            )
        key = record.get("item_key")
        family = record.get("attack_family")
        query_key = record.get("query_item_key")
        if (
            not isinstance(key, str)
            or len(key) != 64
            or not isinstance(family, str)
            or not isinstance(query_key, str)
            or record.get("query_role") != "attack_associated"
            or query_key != key
            or record.get("candidate_added_alone_item_key") != key
            or record.get("retrieval_background_id")
            != "frozen_reference_768"
            or type(record.get("candidate_corpus_size")) is not int
            or record.get("candidate_corpus_size") != 769
            or type(record.get("landing_order_position_within_attack"))
            is not int
            or not isinstance(record.get("source_group"), str)
            or len(record["source_group"]) != 64
            or not isinstance(record.get("topic"), str)
            or not record["topic"]
            or not isinstance(record.get("attack_variant"), str)
            or not record["attack_variant"]
        ):
            raise LandingError(
                "landing record lacks its bound attack-associated query"
            )
        if family == "natural_cover_suffix_v1":
            if record["attack_variant"] != "natural_cover_suffix_v1":
                raise LandingError("natural-cover landing variant drifted")
        elif family == "recipe":
            if record["attack_variant"] not in {"T3", "T4"}:
                raise LandingError("recipe landing variant drifted")
        else:
            raise LandingError(f"unexpected landing family {family!r}")
        records.append(record)
    if len(records) != len({str(record["item_key"]) for record in records}):
        raise LandingError("landing plan contains a duplicate item")

    populations = landing.get("populations")
    if not isinstance(populations, Mapping) or set(populations) != set(
        EXPECTED_POPULATIONS
    ):
        raise LandingError("landing populations differ from the producer schema")
    offset = 0
    for family, expected_n in EXPECTED_POPULATIONS.items():
        family_records = records[offset : offset + expected_n]
        offset += expected_n
        if any(record["attack_family"] != family for record in family_records):
            raise LandingError("landing family blocks or counts drifted")
        for position, record in enumerate(family_records, start=1):
            if record["landing_order_position_within_attack"] != position:
                raise LandingError("landing order position drifted")
        population = populations[family]
        if (
            not isinstance(population, Mapping)
            or set(population) != LANDING_POPULATION_FIELDS
            or type(population.get("expected_denominator")) is not int
            or population.get("expected_denominator") != expected_n
            or population.get("failed_landing_policy")
            != "retain_as_failure_never_replace"
            or population.get("item_keys")
            != [record["item_key"] for record in family_records]
        ):
            raise LandingError(f"{family} landing population drifted")
    return records


def _query_embedding(
    material: Mapping[str, Any],
    record: Mapping[str, Any],
) -> np.ndarray:
    query_key = str(record["query_item_key"])
    if (
        set(material) != QUERY_MATERIAL_FIELDS
        or material.get("query_item_key") != query_key
        or material.get("attack_family") != record["attack_family"]
        or material.get("attack_variant") != record["attack_variant"]
        or material.get("source_group") != record["source_group"]
        or material.get("topic") != record["topic"]
        or material.get("embedding_encoding")
        != "strict-JSON float array in pinned model dimension"
        or not isinstance(material.get("text"), str)
        or not material["text"]
    ):
        raise LandingError(f"query material {query_key} differs from its record")
    raw_embedding = material.get("embedding")
    if (
        not isinstance(raw_embedding, list)
        or len(raw_embedding) != 384
        or any(
            type(component) is not float or not math.isfinite(component)
            for component in raw_embedding
        )
    ):
        raise LandingError(f"query {query_key} has an invalid embedding")
    vector = np.asarray(raw_embedding, dtype=np.float64)
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
    if not isinstance(expected, str) or expected != actual:
        raise LandingError(f"query {query_key} embedding digest mismatch")
    return vector


def evaluate_landing(
    *,
    plan_path: str,
    inputs_path: str,
    embeddings_path: str,
    output_path: str | None = None,
) -> dict[str, Any]:
    plan_bytes, plan_sha256 = _snapshot_bytes(plan_path)
    inputs_bytes, inputs_sha256 = _snapshot_bytes(inputs_path)
    embeddings_bytes, embeddings_sha256 = _snapshot_bytes(embeddings_path)
    frozen_paths = {
        plan_path: plan_sha256,
        inputs_path: inputs_sha256,
        embeddings_path: embeddings_sha256,
    }

    plan = strict_json_load_bytes(plan_bytes, source=plan_path)
    safe = strict_json_load_bytes(inputs_bytes, source=inputs_path)
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
        actual = (
            inputs_sha256
            if filename == "W2D-detector-inputs.json"
            else embeddings_sha256
        )
        if expected != actual:
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

    with np.load(io.BytesIO(embeddings_bytes), allow_pickle=False) as archive:
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
        if row < 0 or row >= matrix.shape[0]:
            raise LandingError(f"safe item {key} embedding row is out of range")
        value = matrix[row]
        norm = float(np.linalg.norm(value))
        if not np.isfinite(value).all() or not math.isfinite(norm) or norm <= 0:
            raise LandingError(f"safe item {key} has invalid vector")
        return value / norm

    background_vectors = {key: vector(key) for key in background_keys}
    planned = _landing_records(plan)
    materials = plan.get("query_materials")
    if not isinstance(materials, Mapping):
        raise LandingError("protocol plan has no query materials")
    planned_keys = {str(record["item_key"]) for record in planned}
    if set(materials) != planned_keys:
        raise LandingError(
            "query-material key universe differs from landing records"
        )
    records: list[dict[str, Any]] = []
    for planned_record in planned:
        poison_key = str(planned_record["item_key"])
        family = str(planned_record["attack_family"])
        query_key = str(planned_record["query_item_key"])
        if poison_key in background_vectors:
            raise LandingError("poison item is present in retrieval background")
        material = materials.get(query_key)
        if not isinstance(material, Mapping):
            raise LandingError(f"query material {query_key} is absent")
        query = _query_embedding(material, planned_record)
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

    _assert_frozen_paths_unchanged(frozen_paths)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "measurement": "B1 off-path exact-cosine attack landing",
        "top_k": 5,
        "tie_break": "descending cosine, then ascending opaque item_key",
        "candidate_population": (
            "one poison item at a time plus the frozen 768-item "
            "retrieval background"
        ),
        "protocol_plan_sha256": plan_sha256,
        "safe_inputs_sha256": inputs_sha256,
        "embeddings_sha256": embeddings_sha256,
        "retrieval_background_sha256": _sha256_json(background_keys),
        "populations": populations,
        "items": sorted(records, key=lambda record: record["item_key"]),
    }
    _finite_tree(artifact)
    if output_path is not None:
        ownership: OutputOwnership | None = None
        try:
            _assert_frozen_paths_unchanged(frozen_paths)
            ownership = _exclusive_json_dump(artifact, output_path)
            _assert_frozen_paths_unchanged(frozen_paths)
            _assert_owned_output(ownership)
        except BaseException as exc:
            if ownership is not None:
                try:
                    _preserve_failed_output(ownership)
                except LandingError as conflict:
                    raise conflict from exc
            raise
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
