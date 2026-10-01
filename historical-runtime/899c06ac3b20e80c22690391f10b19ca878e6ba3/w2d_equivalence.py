#!/usr/bin/env python3
"""Frozen Amendment A9 archive-to-formal equivalence gate."""

from __future__ import annotations

import copy
from typing import Any, Mapping


ARCHIVED_PLAN_SHA256 = (
    "cfcf57938019339ddc6573f66a0d345b42328654bd560ba8fe1197c9a2b7c5c3"
)
CURRENT_PLAN_SHA256 = (
    "8209e741d1d5899c2fee6205f5c3808b5a8c60b3f33a2a0dc3b3b1b4774aa262"
)
ARCHIVED_LANDING_SHA256 = (
    "88aa0cf6503209c3aef93535db46e7456fde0cab9661d870bd2ddf0121e7301d"
)
CURRENT_LANDING_SHA256 = (
    "438e9c494b58933b1a7df1cdee60a186fc6ccbba7c3f520beb424c9043014009"
)
ARCHIVED_PROTOCOL_CODE_SHA256 = (
    "2c830af0ca3c6e43b114c38843c594fda6d4158f36e603162d0e58a85558fd0d"
)
CURRENT_PROTOCOL_CODE_SHA256 = (
    "e328855b87fc296f46ac70f54a619e818b2ba714312f6afc984635abd85c4a08"
)
ARCHIVED_PHASE_CONSTRAINT = (
    "exclusive-create after the frozen data/labels and before the first "
    "test-item detector evaluation"
)
CURRENT_PHASE_CONSTRAINT = (
    "constructed after D1 test scoring; the builder consumes only frozen "
    "data, labels, constants, opaque keys, and domain-separated hashes, with "
    "no D1 score, threshold, metric, landing, or protocol result input"
)


class A9EquivalenceError(ValueError):
    """The A9 archive-to-formal replay equivalence contract failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A9EquivalenceError(message)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise A9EquivalenceError(f"{path} must be an object")
    return value


def _protocol_code_record(
    document: Mapping[str, Any], path: str
) -> tuple[int, Mapping[str, Any]]:
    provenance = _mapping(document.get("provenance"), f"{path}.provenance")
    inputs = provenance.get("inputs")
    if not isinstance(inputs, list):
        raise A9EquivalenceError(f"{path}.provenance.inputs must be a list")
    matches = [
        (index, record)
        for index, record in enumerate(inputs)
        if isinstance(record, Mapping)
        and record.get("role") == "protocol_plan_code"
        and record.get("filename") == "w2d_protocol.py"
    ]
    _require(
        len(matches) == 1,
        f"{path} must contain exactly one w2d_protocol.py provenance record",
    )
    return matches[0]


def validate_a9_replay_equivalence(
    *,
    archived_plan: Mapping[str, Any],
    current_plan: Mapping[str, Any],
    archived_landing: Mapping[str, Any],
    current_landing: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
) -> None:
    """Prove that A9 changed provenance only, never scientific fields."""

    expected_hashes = {
        "protocol_plan_pre_a9": ARCHIVED_PLAN_SHA256,
        "protocol_plan": CURRENT_PLAN_SHA256,
        "landing_pre_a9": ARCHIVED_LANDING_SHA256,
        "landing": CURRENT_LANDING_SHA256,
    }
    _require(
        all(artifact_hashes.get(role) == digest for role, digest in expected_hashes.items()),
        "A9 archived/current raw artifact hashes differ from Amendment A10",
    )
    _require(
        archived_plan.get("phase_constraint") == ARCHIVED_PHASE_CONSTRAINT,
        "archived plan phase constraint changed",
    )
    _require(
        current_plan.get("phase_constraint") == CURRENT_PHASE_CONSTRAINT,
        "current plan phase constraint is not the truthful A9 statement",
    )
    old_index, old_code = _protocol_code_record(
        archived_plan, "archived_plan"
    )
    new_index, new_code = _protocol_code_record(current_plan, "current_plan")
    _require(
        old_index == new_index,
        "protocol-code provenance record moved between plans",
    )
    _require(
        old_code.get("sha256") == ARCHIVED_PROTOCOL_CODE_SHA256,
        "archived plan protocol-code hash changed",
    )
    _require(
        new_code.get("sha256") == CURRENT_PROTOCOL_CODE_SHA256,
        "current plan protocol-code hash changed",
    )

    normalized_current_plan = copy.deepcopy(dict(current_plan))
    normalized_current_plan["phase_constraint"] = ARCHIVED_PHASE_CONSTRAINT
    normalized_inputs = normalized_current_plan["provenance"]["inputs"]
    normalized_inputs[new_index]["sha256"] = ARCHIVED_PROTOCOL_CODE_SHA256
    _require(
        normalized_current_plan == archived_plan,
        "A9 plan replay changed a field outside phase/code provenance",
    )

    _require(
        archived_landing.get("protocol_plan_sha256") == ARCHIVED_PLAN_SHA256,
        "archived landing does not bind the archived plan",
    )
    _require(
        current_landing.get("protocol_plan_sha256") == CURRENT_PLAN_SHA256,
        "current landing does not bind the current plan",
    )
    normalized_current_landing = copy.deepcopy(dict(current_landing))
    normalized_current_landing["protocol_plan_sha256"] = ARCHIVED_PLAN_SHA256
    _require(
        normalized_current_landing == archived_landing,
        "A9 landing replay changed a scientific field",
    )
