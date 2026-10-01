#!/usr/bin/env python3
"""Build the immutable W2D A4 protocol plan.

The plan is an evaluator/orchestration artifact.  It consumes only the frozen
safe data pair, evaluator labels, the data-freeze manifest, and Amendment A4.
It deliberately has no detector, threshold, or protocol-result input.

Production usage::

    python w2d_protocol.py \
      --freeze data/w2d/W2D-DATA-FREEZE.json \
      --safe-json data/w2d/W2D-detector-inputs.json \
      --safe-npz data/w2d/W2D-detector-inputs.npz \
      --labels results/w2d/W2D-labels.json \
      --output results/w2d/W2D-PROTOCOL-PLAN.json

The output is strict JSON and is opened with exclusive creation.  Building a
second time is an error, including when the existing file is byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_FREEZE = HERE / "data" / "w2d" / "W2D-DATA-FREEZE.json"
DEFAULT_SAFE_JSON = HERE / "data" / "w2d" / "W2D-detector-inputs.json"
DEFAULT_SAFE_NPZ = HERE / "data" / "w2d" / "W2D-detector-inputs.npz"
DEFAULT_LABELS = HERE / "results" / "w2d" / "W2D-labels.json"
DEFAULT_OUTPUT = HERE / "results" / "w2d" / "W2D-PROTOCOL-PLAN.json"
DEFAULT_AMENDMENT = HERE / "W2D-PREREGISTRATION-AMENDMENT-A4.md"

SCHEMA_VERSION = "1.0"
CONSTRUCTION_VERSION = "W2D-A4-PROTOCOL-v1"

SEEDS = (1, 2, 3, 4, 5)
ATTACKS = ("recipe", "natural_cover_suffix_v1")
BACKLOGS = ("normal", "heavy")
BASELINES = ("B1", "B2", "B3", "B4")
VERIFIED_BASELINES = ("B2", "B3", "B4")
VERIFIED_ARMS = ("detector", "oracle")

POISON_PER_SEED = 6
CLEAN_PER_SEED = 6
HEAVY_FILLER_PER_SEED = 12
REFERENCE_COUNT = 768
TEST_COUNT = 512
RECIPE_TEST_COUNT = 128
NATURAL_TEST_COUNT = 64

CLEAN_COUNTS = {
    1: {"ordinary_clean": 4, "hard_negative_clean": 2},
    2: {"ordinary_clean": 3, "hard_negative_clean": 3},
    3: {"ordinary_clean": 4, "hard_negative_clean": 2},
    4: {"ordinary_clean": 3, "hard_negative_clean": 3},
    5: {"ordinary_clean": 4, "hard_negative_clean": 2},
}
FILLER_COUNTS = {
    1: {"ordinary_clean": 7, "hard_negative_clean": 5},
    2: {"ordinary_clean": 7, "hard_negative_clean": 5},
    3: {"ordinary_clean": 7, "hard_negative_clean": 5},
    4: {"ordinary_clean": 7, "hard_negative_clean": 5},
    5: {"ordinary_clean": 8, "hard_negative_clean": 4},
}

QUERY_ROLES = (
    "attack_associated",
    "heldout_same_topic",
    "negative_other_topic",
)

SAFE_ITEM_KEYS = frozenset(
    {
        "item_key",
        "normalized_text",
        "embedding_row",
        "source_evidence",
        "source_group",
        "reference",
        "model_revision",
    }
)


class ProtocolPlanError(RuntimeError):
    """The immutable A4 protocol plan could not be constructed."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(token: str) -> None:
    raise ProtocolPlanError(f"non-standard JSON constant {token!r}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolPlanError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _validate_json_value(value: Any, location: str = "$") -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolPlanError(f"{location} contains NaN or infinity")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProtocolPlanError(f"{location} has a non-string key")
            _validate_json_value(child, f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_value(child, f"{location}[{index}]")
        return
    raise ProtocolPlanError(
        f"{location} has unsupported JSON type {type(value).__name__}"
    )


def strict_json_load(path: os.PathLike[str] | str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as stream:
            value = json.load(
                stream,
                parse_constant=_reject_constant,
                object_pairs_hook=_unique_object,
            )
    except ProtocolPlanError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolPlanError(f"cannot read strict JSON {path}: {exc}") from exc
    _validate_json_value(value)
    return value


def _strict_json_bytes(value: Any) -> bytes:
    _validate_json_value(value)
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


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with open(path, "xb") as stream:
            created = True
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        if created:
            try:
                path.unlink()
            except OSError:
                pass
        raise


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolPlanError(message)


def _domain_digest(domain: str, item_key: str) -> str:
    return sha256_bytes(f"{domain}|{item_key}".encode("utf-8"))


def _ordered_candidates(
    records: Iterable[Mapping[str, Any]], domain: str
) -> list[Mapping[str, Any]]:
    return sorted(
        records,
        key=lambda record: (
            _domain_digest(domain, str(record["item_key"])),
            str(record["item_key"]),
        ),
    )


def _take_one(
    available: set[str],
    records: Iterable[Mapping[str, Any]],
    *,
    domain: str,
    description: str,
) -> Mapping[str, Any]:
    candidates = [
        record for record in records if str(record["item_key"]) in available
    ]
    ordered = _ordered_candidates(candidates, domain)
    if not ordered:
        raise ProtocolPlanError(f"no candidate remains for {description}")
    chosen = ordered[0]
    available.remove(str(chosen["item_key"]))
    return chosen


def _interleaved_strata(counts: Mapping[str, int]) -> list[str]:
    """A4's conservative deterministic order for an ambiguous within-seed mix.

    Ordinary and hard-negative items alternate, starting with ordinary; any
    surplus ordinary items follow.  The exact resulting sequence is recorded
    in the plan, so this convention cannot be changed after an outcome.
    """

    ordinary = int(counts["ordinary_clean"])
    hard = int(counts["hard_negative_clean"])
    result: list[str] = []
    while ordinary or hard:
        if ordinary:
            result.append("ordinary_clean")
            ordinary -= 1
        if hard:
            result.append("hard_negative_clean")
            hard -= 1
    return result


def _label_map(labels_document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = labels_document.get("items")
    _require(isinstance(raw, Mapping), "label artifact items must be a mapping")
    labels: dict[str, Mapping[str, Any]] = {}
    for key, value in raw.items():
        _require(isinstance(key, str) and key, "empty/non-string label item key")
        _require(isinstance(value, Mapping), f"label {key} is not an object")
        labels[key] = value
    return labels


def _load_and_validate_inputs(
    *,
    freeze_path: Path,
    safe_json_path: Path,
    safe_npz_path: Path,
    labels_path: Path,
) -> dict[str, Any]:
    import numpy as np

    freeze = strict_json_load(freeze_path)
    safe = strict_json_load(safe_json_path)
    labels_document = strict_json_load(labels_path)
    _require(isinstance(freeze, Mapping), "data freeze is not an object")
    _require(isinstance(safe, Mapping), "safe input JSON is not an object")
    _require(isinstance(labels_document, Mapping), "labels JSON is not an object")

    frozen_artifacts = freeze.get("artifacts")
    _require(isinstance(frozen_artifacts, Mapping), "freeze has no artifacts map")
    for path in (safe_json_path, safe_npz_path, labels_path):
        record = frozen_artifacts.get(path.name)
        _require(
            isinstance(record, Mapping) and isinstance(record.get("sha256"), str),
            f"{path.name} is absent from the data freeze",
        )
        actual = sha256_file(path)
        _require(
            actual == record["sha256"],
            f"{path.name} sha256 {actual} != frozen {record['sha256']}",
        )

    model = safe.get("model")
    _require(isinstance(model, Mapping), "safe input has no model record")
    model_revision = model.get("revision")
    model_dimension = model.get("dimension")
    _require(
        isinstance(model_revision, str) and model_revision,
        "safe model revision is missing",
    )
    _require(
        isinstance(model_dimension, int) and model_dimension > 0,
        "safe model dimension is invalid",
    )
    _require(
        safe.get("model_revision") == model_revision,
        "safe top-level/model revisions disagree",
    )
    _require(
        labels_document.get("model_revision") == model_revision,
        "safe and label model revisions disagree",
    )
    _require(
        freeze.get("model", {}).get("revision") == model_revision,
        "safe and freeze model revisions disagree",
    )

    embedding_record = safe.get("embedding_artifact")
    _require(
        isinstance(embedding_record, Mapping),
        "safe input has no embedding artifact record",
    )
    _require(
        embedding_record.get("filename") == safe_npz_path.name,
        "safe NPZ filename does not match its manifest",
    )
    _require(
        embedding_record.get("sha256") == sha256_file(safe_npz_path),
        "safe NPZ digest does not match its embedded manifest",
    )
    _require(
        embedding_record.get("array") == "embeddings",
        "safe NPZ array name is not embeddings",
    )

    try:
        with np.load(safe_npz_path, allow_pickle=False) as archive:
            _require(
                archive.files == ["embeddings"],
                f"safe NPZ has unexpected arrays {archive.files}",
            )
            embeddings = np.asarray(archive["embeddings"])
    except ProtocolPlanError:
        raise
    except Exception as exc:
        raise ProtocolPlanError(f"cannot read safe NPZ: {exc}") from exc
    _require(embeddings.dtype == np.float32, "safe embeddings are not float32")
    expected_shape = tuple(embedding_record.get("shape", ()))
    _require(
        embeddings.shape == expected_shape,
        f"safe embedding shape {embeddings.shape} != {expected_shape}",
    )
    _require(
        embeddings.ndim == 2 and embeddings.shape[1] == model_dimension,
        "safe embedding matrix/model dimension mismatch",
    )
    _require(bool(np.isfinite(embeddings).all()), "safe embeddings are non-finite")
    norms = np.linalg.norm(embeddings, axis=1)
    _require(
        bool(np.all(np.abs(norms - 1.0) <= 1e-4)),
        "safe embeddings are not unit-normalized",
    )

    raw_safe_items = safe.get("items")
    _require(isinstance(raw_safe_items, list), "safe items must be a list")
    safe_items: dict[str, Mapping[str, Any]] = {}
    rows: list[int] = []
    for record in raw_safe_items:
        _require(isinstance(record, Mapping), "safe item is not an object")
        _require(set(record) == SAFE_ITEM_KEYS, "safe item schema mismatch")
        key = record.get("item_key")
        _require(isinstance(key, str) and key, "safe item key is missing")
        _require(key not in safe_items, f"duplicate safe item key {key}")
        row = record.get("embedding_row")
        _require(isinstance(row, int), f"{key}: embedding row is not an integer")
        _require(
            record.get("model_revision") == model_revision,
            f"{key}: item model revision mismatch",
        )
        safe_items[key] = record
        rows.append(row)
    _require(
        sorted(rows) == list(range(len(rows))),
        "safe embedding rows are not a dense one-to-one index",
    )
    _require(
        len(safe_items) == embeddings.shape[0],
        "safe item/embedding row counts disagree",
    )

    labels = _label_map(labels_document)
    _require(
        set(labels) == set(safe_items),
        "safe and evaluator item-key sets differ",
    )
    source_groups: set[str] = set()
    for key, safe_record in safe_items.items():
        label = labels[key]
        _require(
            label.get("source_group") == safe_record.get("source_group"),
            f"{key}: safe/evaluator source groups disagree",
        )
        source_group = str(label.get("source_group", ""))
        _require(source_group, f"{key}: source group is missing")
        _require(
            source_group not in source_groups,
            f"source group {source_group} is allocated more than once",
        )
        source_groups.add(source_group)

    reference = safe.get("reference_index")
    _require(isinstance(reference, Mapping), "safe reference index is missing")
    ref_keys = reference.get("item_keys")
    ref_rows = reference.get("embedding_rows")
    ref_groups = reference.get("groups")
    _require(
        isinstance(ref_keys, list)
        and isinstance(ref_rows, list)
        and isinstance(ref_groups, list),
        "safe reference index arrays are missing",
    )
    _require(
        len(ref_keys) == len(ref_rows) == len(ref_groups) == REFERENCE_COUNT,
        f"retrieval/detector reference must contain {REFERENCE_COUNT} items",
    )
    _require(len(set(ref_keys)) == REFERENCE_COUNT, "reference keys repeat")
    for key, row, group in zip(ref_keys, ref_rows, ref_groups):
        _require(key in safe_items, f"unknown reference key {key}")
        _require(
            safe_items[key]["embedding_row"] == row,
            f"{key}: reference embedding row mismatch",
        )
        _require(
            safe_items[key]["source_group"] == group,
            f"{key}: reference source-group mismatch",
        )
        label = labels[key]
        _require(
            safe_items[key]["reference"] is True
            and label.get("split") == "reference"
            and label.get("stratum") == "clean_reference"
            and label.get("poison") is False,
            f"{key}: retrieval background is not frozen clean reference data",
        )

    sets = safe.get("sets")
    _require(isinstance(sets, Mapping), "safe sets record is missing")
    test_order = sets.get("test_score_order")
    _require(isinstance(test_order, list), "test score order is missing")
    _require(
        len(test_order) == TEST_COUNT and len(set(test_order)) == TEST_COUNT,
        f"detector-quality test denominator must be {TEST_COUNT}",
    )
    label_test_keys = {
        key for key, label in labels.items() if label.get("split") == "test"
    }
    _require(
        set(test_order) == label_test_keys,
        "test score order does not equal the frozen test population",
    )

    test_clean = [
        _joined_record(key, safe_items[key], labels[key])
        for key in test_order
        if labels[key].get("poison") is False
    ]
    recipe = [
        _joined_record(key, safe_items[key], labels[key])
        for key in test_order
        if labels[key].get("poison") is True
        and labels[key].get("attack_family") == "recipe"
    ]
    natural = [
        _joined_record(key, safe_items[key], labels[key])
        for key in test_order
        if labels[key].get("poison") is True
        and labels[key].get("attack_family") == "natural_cover_suffix_v1"
    ]
    _require(len(test_clean) == 320, "frozen test-clean population is not 320")
    _require(
        len(recipe) == RECIPE_TEST_COUNT,
        f"frozen recipe-test population is not {RECIPE_TEST_COUNT}",
    )
    _require(
        len(natural) == NATURAL_TEST_COUNT,
        f"frozen natural-cover population is not {NATURAL_TEST_COUNT}",
    )
    for record in recipe + natural:
        text = record.get("target_query")
        vector = record.get("target_query_embedding")
        _require(
            isinstance(text, str) and text and len(text.split()) == 30,
            f"{record['item_key']}: target query is not a frozen 30-word prefix",
        )
        _require(
            isinstance(vector, list) and len(vector) == model_dimension,
            f"{record['item_key']}: target-query vector dimension mismatch",
        )
        converted = np.asarray(vector, dtype=np.float64)
        _require(
            bool(np.isfinite(converted).all()),
            f"{record['item_key']}: target-query vector is non-finite",
        )
        _require(
            abs(float(np.linalg.norm(converted)) - 1.0) <= 1e-4,
            f"{record['item_key']}: target-query vector is not normalized",
        )

    return {
        "freeze": freeze,
        "safe": safe,
        "labels_document": labels_document,
        "labels": labels,
        "safe_items": safe_items,
        "embeddings": embeddings,
        "test_order": list(test_order),
        "test_clean": test_clean,
        "attack_items": {"recipe": recipe, "natural_cover_suffix_v1": natural},
        "model": {
            "name": model.get("name"),
            "revision": model_revision,
            "dimension": model_dimension,
        },
    }


def _joined_record(
    key: str,
    safe_record: Mapping[str, Any],
    label: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "item_key": key,
        "source_group": str(label["source_group"]),
        "embedding_row": int(safe_record["embedding_row"]),
        "topic": str(label.get("topic", "")),
        "split": str(label.get("split", "")),
        "stratum": str(label.get("stratum", "")),
        "poison": bool(label.get("poison")),
        "attack_family": label.get("attack_family"),
        "attack_variant": label.get("attack_variant"),
        "target_query": label.get("target_query"),
        "target_query_embedding": label.get("target_query_embedding"),
    }


def _select_attack_items(
    *,
    attack: str,
    records: Sequence[Mapping[str, Any]],
    topics: Sequence[str],
) -> dict[int, list[Mapping[str, Any]]]:
    available = {str(record["item_key"]) for record in records}
    selected: dict[int, list[Mapping[str, Any]]] = {seed: [] for seed in SEEDS}
    global_position = 0
    for seed in SEEDS:
        for _ in range(POISON_PER_SEED):
            topic = topics[global_position % len(topics)]
            variant: str | None = None
            if attack == "recipe":
                variant = "T3" if global_position % 2 == 0 else "T4"
            eligible = [
                record
                for record in records
                if record["topic"] == topic
                and (variant is None or record["attack_variant"] == variant)
            ]
            domain = (
                f"W2D-A4-candidate|poison|{attack}|{topic}|"
                f"{variant or attack}"
            )
            chosen = _take_one(
                available,
                eligible,
                domain=domain,
                description=(
                    f"seed {seed} {attack} topic {topic} "
                    f"variant {variant or attack}"
                ),
            )
            selected[seed].append(chosen)
            global_position += 1
    return selected


def _select_clean_role(
    *,
    role: str,
    counts: Mapping[int, Mapping[str, int]],
    records: Sequence[Mapping[str, Any]],
    available: set[str],
    topics: Sequence[str],
) -> dict[int, list[Mapping[str, Any]]]:
    selected: dict[int, list[Mapping[str, Any]]] = {seed: [] for seed in SEEDS}
    global_position = 0
    for seed in SEEDS:
        strata = _interleaved_strata(counts[seed])
        for stratum in strata:
            topic = topics[global_position % len(topics)]
            eligible = [
                record
                for record in records
                if record["topic"] == topic and record["stratum"] == stratum
            ]
            domain = f"W2D-A4-candidate|{role}|{topic}|{stratum}"
            chosen = _take_one(
                available,
                eligible,
                domain=domain,
                description=f"seed {seed} {role} topic {topic} {stratum}",
            )
            selected[seed].append(chosen)
            global_position += 1
    return selected


def _query_material(record: Mapping[str, Any]) -> dict[str, Any]:
    vector = [float(value) for value in record["target_query_embedding"]]
    vector_bytes = _strict_json_bytes(vector)
    return {
        "query_item_key": str(record["item_key"]),
        "source_group": str(record["source_group"]),
        "topic": str(record["topic"]),
        "attack_family": str(record["attack_family"]),
        "attack_variant": str(record["attack_variant"]),
        "text": str(record["target_query"]),
        "embedding": vector,
        "embedding_sha256": sha256_bytes(vector_bytes),
        "embedding_encoding": "strict-JSON float array in pinned model dimension",
    }


def _query_assignments(
    *,
    selected_by_attack: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]],
    attack_items: Mapping[str, Sequence[Mapping[str, Any]]],
    topics: Sequence[str],
) -> dict[str, Any]:
    selected_keys = {
        attack: {
            str(record["item_key"])
            for records in selected_by_attack[attack].values()
            for record in records
        }
        for attack in ATTACKS
    }
    assignments: dict[str, Any] = {}
    for attack in ATTACKS:
        unselected = [
            record
            for record in attack_items[attack]
            if str(record["item_key"]) not in selected_keys[attack]
        ]
        for seed in SEEDS:
            for record in selected_by_attack[attack][seed]:
                owner_key = str(record["item_key"])
                topic = str(record["topic"])
                next_topic = topics[(topics.index(topic) + 1) % len(topics)]
                same_candidates = [
                    candidate
                    for candidate in unselected
                    if candidate["topic"] == topic
                    and candidate["item_key"] != owner_key
                ]
                negative_candidates = [
                    candidate
                    for candidate in unselected
                    if candidate["topic"] == next_topic
                    and candidate["item_key"] != owner_key
                ]
                same = _ordered_candidates(
                    same_candidates,
                    f"W2D-A4-query-donor|heldout_same_topic|{owner_key}",
                )
                negative = _ordered_candidates(
                    negative_candidates,
                    f"W2D-A4-query-donor|negative_other_topic|{owner_key}",
                )
                _require(
                    bool(same),
                    f"{owner_key}: no unselected same-topic {attack} donor",
                )
                _require(
                    bool(negative),
                    f"{owner_key}: no unselected next-topic {attack} donor",
                )
                assignments[owner_key] = {
                    "poison_item_key": owner_key,
                    "seed": seed,
                    "attack_family": attack,
                    "roles": {
                        "attack_associated": {
                            "query_item_key": owner_key,
                            "donor_item_key": owner_key,
                            "donor_selected_for_runtime": True,
                            "topic": topic,
                        },
                        "heldout_same_topic": {
                            "query_item_key": str(same[0]["item_key"]),
                            "donor_item_key": str(same[0]["item_key"]),
                            "donor_selected_for_runtime": False,
                            "topic": topic,
                        },
                        "negative_other_topic": {
                            "query_item_key": str(negative[0]["item_key"]),
                            "donor_item_key": str(negative[0]["item_key"]),
                            "donor_selected_for_runtime": False,
                            "topic": next_topic,
                        },
                    },
                }
    _require(
        len(assignments) == len(SEEDS) * len(ATTACKS) * POISON_PER_SEED,
        "planned query assignment count is not 60",
    )
    return assignments


def _item_audit_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "item_key": str(record["item_key"]),
        "source_group": str(record["source_group"]),
        "topic": str(record["topic"]),
        "stratum": str(record["stratum"]),
        "attack_family": record.get("attack_family"),
        "attack_variant": record.get("attack_variant"),
    }


def _runtime_unit_id(seed: int, attack: str, backlog: str) -> str:
    return f"seed-{seed}__attack-{attack}__backlog-{backlog}"


def _cell_id(
    seed: int, attack: str, backlog: str, baseline: str, arm: str
) -> str:
    return (
        f"seed-{seed}__attack-{attack}__backlog-{backlog}"
        f"__baseline-{baseline}__arm-{arm}"
    )


def _build_runtime_grid(
    *,
    selected_by_attack: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]],
    clean_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    filler_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    runtime_units: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    units_by_id: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        for attack in ATTACKS:
            for backlog in BACKLOGS:
                unit_id = _runtime_unit_id(seed, attack, backlog)
                poison = list(selected_by_attack[attack][seed])
                clean = list(clean_by_seed[seed])
                filler = list(filler_by_seed[seed]) if backlog == "heavy" else []
                ordered_roles = (
                    [("filler", record) for record in filler]
                    + [("poison", record) for record in poison]
                    + [("clean", record) for record in clean]
                )
                injection = [
                    {
                        "injection_ordinal": ordinal,
                        "injection_at_s": 1.0,
                        "item_key": str(record["item_key"]),
                        "role": role,
                        "topic": str(record["topic"]),
                        "stratum": str(record["stratum"]),
                    }
                    for ordinal, (role, record) in enumerate(ordered_roles)
                ]
                unit = {
                    "runtime_unit_id": unit_id,
                    "seed": seed,
                    "attack_family": attack,
                    "backlog": backlog,
                    "poison_item_keys": [
                        str(record["item_key"]) for record in poison
                    ],
                    "clean_item_keys": [
                        str(record["item_key"]) for record in clean
                    ],
                    "filler_item_keys": [
                        str(record["item_key"]) for record in filler
                    ],
                    "query_poison_item_keys": [
                        str(record["item_key"]) for record in poison
                    ],
                    "injection_sequence": injection,
                    "reused_by_cells": [],
                }
                runtime_units.append(unit)
                units_by_id[unit_id] = unit

                baseline_arms = [("B1", "control")] + [
                    (baseline, arm)
                    for baseline in VERIFIED_BASELINES
                    for arm in VERIFIED_ARMS
                ]
                for baseline, arm in baseline_arms:
                    cell_id = _cell_id(seed, attack, backlog, baseline, arm)
                    unit["reused_by_cells"].append(cell_id)
                    cells.append(
                        {
                            "cell_id": cell_id,
                            "runtime_unit_id": unit_id,
                            "seed": seed,
                            "attack_family": attack,
                            "backlog": backlog,
                            "baseline": baseline,
                            "arm": arm,
                            "decision_source": (
                                "none_undefended"
                                if baseline == "B1"
                                else (
                                    "frozen_D1_promote_refuse"
                                    if arm == "detector"
                                    else "construction_truth"
                                )
                            ),
                            "service_replay": (
                                "shadow_only_no_state_effect"
                                if baseline == "B1"
                                else "frozen_D1_item_service_time"
                            ),
                            "execution_order_position": None,
                            "execution_order_key_sha256": None,
                        }
                    )

    execution_order: dict[str, list[str]] = {}
    cells_by_seed: dict[int, list[dict[str, Any]]] = {
        seed: [] for seed in SEEDS
    }
    for cell in cells:
        cells_by_seed[int(cell["seed"])].append(cell)
    for seed in SEEDS:
        ordered = sorted(
            cells_by_seed[seed],
            key=lambda cell: (
                sha256_bytes(
                    (
                        "W2D-cell-order|"
                        + str(seed)
                        + "|"
                        + str(cell["cell_id"])
                    ).encode("utf-8")
                ),
                str(cell["cell_id"]),
            ),
        )
        for position, cell in enumerate(ordered, start=1):
            cell["execution_order_position"] = position
            cell["execution_order_key_sha256"] = sha256_bytes(
                (
                    "W2D-cell-order|"
                    + str(seed)
                    + "|"
                    + str(cell["cell_id"])
                ).encode("utf-8")
            )
        execution_order[str(seed)] = [
            str(cell["cell_id"]) for cell in ordered
        ]

    _require(len(runtime_units) == 20, "runtime grid does not contain 20 units")
    _require(len(cells) == 140, "runtime grid does not contain 140 cells")
    _require(
        all(len(unit["reused_by_cells"]) == 7 for unit in runtime_units),
        "a runtime unit is not reused by exactly seven cells",
    )
    return runtime_units, cells, execution_order


def _build_landing_plan(
    attack_items: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    populations: dict[str, Any] = {}
    for attack in ATTACKS:
        ordered = _ordered_candidates(
            attack_items[attack], f"W2D-A4-landing-order|{attack}"
        )
        item_keys = [str(record["item_key"]) for record in ordered]
        expected = (
            RECIPE_TEST_COUNT if attack == "recipe" else NATURAL_TEST_COUNT
        )
        _require(len(item_keys) == expected, f"{attack}: landing count drifted")
        populations[attack] = {
            "item_keys": item_keys,
            "expected_denominator": expected,
            "failed_landing_policy": "retain_as_failure_never_replace",
        }
        for position, record in enumerate(ordered, start=1):
            records.append(
                {
                    "landing_order_position_within_attack": position,
                    "item_key": str(record["item_key"]),
                    "source_group": str(record["source_group"]),
                    "attack_family": attack,
                    "attack_variant": str(record["attack_variant"]),
                    "topic": str(record["topic"]),
                    "query_role": "attack_associated",
                    "query_item_key": str(record["item_key"]),
                    "retrieval_background_id": "frozen_reference_768",
                    "candidate_added_alone_item_key": str(record["item_key"]),
                    "candidate_corpus_size": REFERENCE_COUNT + 1,
                }
            )
    _require(len(records) == 192, "B1 landing plan does not contain 192 items")
    return {
        "baseline": "B1",
        "off_path": True,
        "metric": "exact_cosine",
        "top_k": 5,
        "tie_break": "descending cosine then ascending opaque item_key",
        "background_id": "frozen_reference_768",
        "background_count": REFERENCE_COUNT,
        "candidate_rule": (
            "add exactly one modified test poison passage to the immutable "
            "background; query with that item's attack_associated query"
        ),
        "records": records,
        "populations": populations,
        "result_requirements": {
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
        },
    }


def _provenance_record(role: str, path: Path) -> dict[str, Any]:
    return {
        "role": role,
        "filename": path.name,
        "sha256": sha256_file(path),
    }


def build_protocol_plan(
    *,
    freeze_path: os.PathLike[str] | str = DEFAULT_FREEZE,
    safe_json_path: os.PathLike[str] | str = DEFAULT_SAFE_JSON,
    safe_npz_path: os.PathLike[str] | str = DEFAULT_SAFE_NPZ,
    labels_path: os.PathLike[str] | str = DEFAULT_LABELS,
    amendment_path: os.PathLike[str] | str = DEFAULT_AMENDMENT,
    output_path: os.PathLike[str] | str = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Construct and exclusive-create the complete A4 protocol plan.

    The output-exists check precedes all input reads.  The function has no
    detector-output argument by design, and every choice is derived from
    evaluator labels plus domain-separated hashes.
    """

    freeze_path = Path(freeze_path)
    safe_json_path = Path(safe_json_path)
    safe_npz_path = Path(safe_npz_path)
    labels_path = Path(labels_path)
    amendment_path = Path(amendment_path)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(
            f"W2D protocol plan is no-overwrite: {output_path} already exists"
        )
    _require(amendment_path.is_file(), f"A4 amendment is missing: {amendment_path}")

    loaded = _load_and_validate_inputs(
        freeze_path=freeze_path,
        safe_json_path=safe_json_path,
        safe_npz_path=safe_npz_path,
        labels_path=labels_path,
    )
    safe = loaded["safe"]
    labels = loaded["labels"]
    topics_raw = loaded["freeze"].get("source", {}).get("topics")
    _require(
        isinstance(topics_raw, list)
        and len(topics_raw) > 1
        and len(set(topics_raw)) == len(topics_raw)
        and all(isinstance(topic, str) and topic for topic in topics_raw),
        "frozen topic order is missing or invalid",
    )
    topics = tuple(topics_raw)
    _require(
        all(record["topic"] in topics for record in loaded["test_clean"]),
        "test clean item has a topic outside the frozen topic order",
    )

    selected_by_attack = {
        attack: _select_attack_items(
            attack=attack,
            records=loaded["attack_items"][attack],
            topics=topics,
        )
        for attack in ATTACKS
    }
    available_clean = {
        str(record["item_key"]) for record in loaded["test_clean"]
    }
    clean_by_seed = _select_clean_role(
        role="injected_clean",
        counts=CLEAN_COUNTS,
        records=loaded["test_clean"],
        available=available_clean,
        topics=topics,
    )
    filler_by_seed = _select_clean_role(
        role="heavy_filler",
        counts=FILLER_COUNTS,
        records=loaded["test_clean"],
        available=available_clean,
        topics=topics,
    )

    selected_poison_keys = [
        str(record["item_key"])
        for attack in ATTACKS
        for seed in SEEDS
        for record in selected_by_attack[attack][seed]
    ]
    selected_clean_keys = [
        str(record["item_key"])
        for seed in SEEDS
        for record in clean_by_seed[seed]
    ]
    selected_filler_keys = [
        str(record["item_key"])
        for seed in SEEDS
        for record in filler_by_seed[seed]
    ]
    all_runtime_role_keys = (
        selected_poison_keys + selected_clean_keys + selected_filler_keys
    )
    _require(
        len(selected_poison_keys) == 60
        and len(set(selected_poison_keys)) == 60,
        "poison assignment is not 60 unique items",
    )
    _require(
        len(selected_clean_keys) == 30
        and len(set(selected_clean_keys)) == 30,
        "clean assignment is not 30 unique items",
    )
    _require(
        len(selected_filler_keys) == 60
        and len(set(selected_filler_keys)) == 60,
        "heavy-filler assignment is not 60 unique items",
    )
    _require(
        len(set(all_runtime_role_keys)) == len(all_runtime_role_keys),
        "runtime roles overlap instead of being disjoint",
    )

    planned_queries = _query_assignments(
        selected_by_attack=selected_by_attack,
        attack_items=loaded["attack_items"],
        topics=topics,
    )
    runtime_units, cells, execution_order = _build_runtime_grid(
        selected_by_attack=selected_by_attack,
        clean_by_seed=clean_by_seed,
        filler_by_seed=filler_by_seed,
    )
    landing_plan = _build_landing_plan(loaded["attack_items"])

    query_material_records = {
        str(record["item_key"]): _query_material(record)
        for attack in ATTACKS
        for record in loaded["attack_items"][attack]
    }
    _require(
        len(query_material_records) == 192,
        "query material does not cover all 192 test poison items",
    )
    for assignment in planned_queries.values():
        for role in QUERY_ROLES:
            query_key = assignment["roles"][role]["query_item_key"]
            _require(
                query_key in query_material_records,
                f"query assignment references unknown material {query_key}",
            )

    ref = safe["reference_index"]
    retrieval_key_payload = _strict_json_bytes(ref["item_keys"])
    retrieval_background = {
        "retrieval_background_id": "frozen_reference_768",
        "logical_roles": [
            "detector_reference",
            "W2D_fixed_retrieval_background",
        ],
        "selection_rule": (
            "exactly the 768 clean items already frozen in reference_index; "
            "no separate outcome-conditioned corpus"
        ),
        "item_keys": list(ref["item_keys"]),
        "embedding_rows": [int(row) for row in ref["embedding_rows"]],
        "source_groups": list(ref["groups"]),
        "count": len(ref["item_keys"]),
        "item_keys_sha256": sha256_bytes(retrieval_key_payload),
        "excluded_from_detector_quality_denominators": True,
    }

    item_assignments = []
    for seed in SEEDS:
        item_assignments.append(
            {
                "seed": seed,
                "poison_by_attack": {
                    attack: [
                        _item_audit_record(record)
                        for record in selected_by_attack[attack][seed]
                    ]
                    for attack in ATTACKS
                },
                "clean": [
                    _item_audit_record(record)
                    for record in clean_by_seed[seed]
                ],
                "heavy_filler": [
                    _item_audit_record(record)
                    for record in filler_by_seed[seed]
                ],
                "clean_stratum_counts": dict(CLEAN_COUNTS[seed]),
                "heavy_filler_stratum_counts": dict(FILLER_COUNTS[seed]),
            }
        )

    plan_code_path = Path(__file__).resolve()
    provenance = {
        "inputs": [
            _provenance_record("data_freeze", freeze_path),
            _provenance_record("safe_item_schema", safe_json_path),
            _provenance_record("safe_item_embeddings", safe_npz_path),
            _provenance_record("evaluator_only_labels_and_queries", labels_path),
            _provenance_record("protocol_plan_code", plan_code_path),
            _provenance_record("preregistration_amendment", amendment_path),
        ],
        "model": loaded["model"],
        "safe_construction_version": safe.get("construction_version"),
        "label_construction_version": loaded["labels_document"].get(
            "construction_version"
        ),
        "data_freeze_construction_version": loaded["freeze"].get(
            "construction_version"
        ),
        "selection_depends_only_on": [
            "frozen topic order",
            "frozen split/stratum/attack labels",
            "opaque item keys",
            "recorded SHA256 domain separators",
        ],
        "forbidden_selection_inputs": [
            "D1 outcome",
            "D1 measured service time",
            "D1 threshold",
            "retrieval landing outcome",
            "protocol outcome",
        ],
    }

    plan = {
        "schema_version": SCHEMA_VERSION,
        "construction_version": CONSTRUCTION_VERSION,
        "phase_constraint": (
            "constructed after D1 test scoring; the builder consumes only "
            "frozen data, labels, constants, opaque keys, and domain-separated "
            "hashes, with no D1 score, threshold, metric, landing, or protocol "
            "result input"
        ),
        "provenance": provenance,
        "constants": {
            "seeds": list(SEEDS),
            "topics_in_frozen_round_robin_order": list(topics),
            "attacks": list(ATTACKS),
            "backlogs": list(BACKLOGS),
            "baselines": list(BASELINES),
            "verified_arms": list(VERIFIED_ARMS),
            "poison_per_seed_per_attack": POISON_PER_SEED,
            "clean_per_seed_shared_by_attacks": CLEAN_PER_SEED,
            "heavy_filler_per_seed_shared_by_attacks": HEAVY_FILLER_PER_SEED,
            "deadline_T_p_s": 1.0,
            "window_horizon_s": 8.0,
            "injection_at_s": 1.0,
            "retrieval_top_k": 5,
            "primary_cell_count": 140,
            "detector_quality_unique_test_denominator": TEST_COUNT,
        },
        "selection_rules": {
            "topic_round_robin": (
                "each of poison, injected_clean, and heavy_filler has a "
                "separate zero-based cursor starting at the first frozen "
                "topic and advancing once per selected item across seeds 1..5"
            ),
            "candidate_order": (
                "ascending SHA256(domain + '|' + opaque_item_key), then "
                "ascending opaque_item_key"
            ),
            "candidate_domains": {
                "poison": (
                    "W2D-A4-candidate|poison|{attack}|{topic}|{variant}"
                ),
                "injected_clean": (
                    "W2D-A4-candidate|injected_clean|{topic}|{stratum}"
                ),
                "heavy_filler": (
                    "W2D-A4-candidate|heavy_filler|{topic}|{stratum}"
                ),
                "heldout_query_donor": (
                    "W2D-A4-query-donor|heldout_same_topic|{owner_item_key}"
                ),
                "negative_query_donor": (
                    "W2D-A4-query-donor|negative_other_topic|{owner_item_key}"
                ),
                "landing_order": "W2D-A4-landing-order|{attack}",
                "cell_order": "W2D-cell-order|{seed}|{cell_id}",
            },
            "recipe_variant_order": (
                "global recipe selection positions alternate T3,T4 starting "
                "with T3; an unavailable required variant aborts planning"
            ),
            "clean_stratum_order": (
                "within each seed alternate ordinary_clean and "
                "hard_negative_clean starting ordinary; append any ordinary "
                "surplus; exact sequence is retained in item_assignments"
            ),
            "reuse": (
                "clean and filler are shared by attacks; normal/heavy share "
                "poison and clean; all baselines/arms share each runtime unit"
            ),
            "replacement": "none after any outcome; failure aborts planning",
        },
        "retrieval_background": retrieval_background,
        "item_assignments": {"by_seed": item_assignments},
        "query_materials": query_material_records,
        "planned_poison_queries": planned_queries,
        "runtime_units": runtime_units,
        "cells": cells,
        "execution_order_by_seed": execution_order,
        "landing_plan": landing_plan,
        "reporting_populations": {
            "detector_quality": {
                "item_keys": list(loaded["test_order"]),
                "denominator": TEST_COUNT,
                "unique_source_groups": TEST_COUNT,
                "protocol_replay_may_not_enlarge": True,
            },
            "runtime_by_attack": {
                attack: {
                    "overall": {
                        "item_keys": [
                            str(record["item_key"])
                            for seed in SEEDS
                            for record in selected_by_attack[attack][seed]
                        ],
                        "denominator": len(SEEDS) * POISON_PER_SEED,
                    },
                    "landed_only": {
                        "definition": (
                            "overall item_keys intersect the separately "
                            "frozen B1 landing records where landed=true"
                        ),
                        "denominator": "observed intersection size; never resample",
                    },
                }
                for attack in ATTACKS
            },
        },
    }

    payload = _strict_json_bytes(plan)
    _exclusive_write(output_path, payload)
    return {
        "path": str(output_path),
        "sha256": sha256_bytes(payload),
        "runtime_units": len(runtime_units),
        "cells": len(cells),
        "landing_items": len(landing_plan["records"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    parser.add_argument("--safe-json", default=str(DEFAULT_SAFE_JSON))
    parser.add_argument("--safe-npz", default=str(DEFAULT_SAFE_NPZ))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    result = build_protocol_plan(
        freeze_path=args.freeze,
        safe_json_path=args.safe_json,
        safe_npz_path=args.safe_npz,
        labels_path=args.labels,
        amendment_path=args.amendment,
        output_path=args.output,
    )
    print(
        f"Wrote {result['path']} ({result['sha256']}); "
        f"{result['cells']} cells, {result['landing_items']} landing items"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
