#!/usr/bin/env python3
"""Independent, fail-loud gates for a formal W2D replay bundle.

This verifier intentionally consumes only frozen artifacts plus the final
protocol result.  It never builds data, scores an item, or runs a protocol cell.
The runner may import the public constants and :func:`verify_bundle` before
publishing a result.

Usage:
    ./.venv312/bin/python verify_w2d.py \
        --manifest results/w2d/W2D-MANIFEST.json \
        --results results/w2d/W2D-PROTOCOL-RESULTS.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence
import unicodedata

from w2d_calibrate import CalibrationError, select_threshold
from w2d_metrics import (
    MetricsError,
    detector_quality_summary,
    file_sha256,
    rate_record,
    source_control_gate_summary,
    strict_json_load,
)
from verify_w2d_legacy import evaluate as evaluate_legacy_regression
from verify_w2d_legacy import (
    FROZEN_CANDIDATE_COMMIT,
    FROZEN_SAME_CODE_DIFFERENCE_SHA256,
    FROZEN_SOURCE_SHA256,
    LEGACY_GATE_COMMANDS,
    canonical_strict_json_sha256,
    legacy_gate_code_sha256,
    legacy_runtime_code_sha256,
    rerun_legacy_gates,
    validate_frozen_same_code_differences,
)


HERE = Path(__file__).resolve().parent
EXPECTED_ARTIFACT_ROOT = HERE
EXPECTED_FINGERPRINT_ROOT = HERE

MEASUREMENT_NAME = (
    "promotion-path replay driven by real D1 outputs and measured service times"
)
EXECUTION_MODE = "score_once_frozen_decision_service_replay"
FROZEN_SERVICE_REPLAY = "frozen_D1_item_service_time"
FROZEN_PROVIDER = "frozen_score_lookup"
RESULT_SCHEMA = "W2D-protocol-result-v2"
MANIFEST_SCHEMA = "W2D-manifest-v2"
EXPECTED_TEST_ITEMS = 512
EXPECTED_SOURCE_CONTROLS = 24
EXPECTED_NATURAL_COVER = 64
EXPECTED_RECIPE_TEST = 128
EXPECTED_RUNTIME_POISON_PER_FAMILY = 30
EXPECTED_RUNTIME_UNITS = 20
EXPECTED_PROTOCOL_CELLS = 140
EXPECTED_SEEDS = frozenset({1, 2, 3, 4, 5})
EXPECTED_CLEAN_STRATA = {
    1: {"ordinary_clean": 4, "hard_negative_clean": 2},
    2: {"ordinary_clean": 3, "hard_negative_clean": 3},
    3: {"ordinary_clean": 4, "hard_negative_clean": 2},
    4: {"ordinary_clean": 3, "hard_negative_clean": 3},
    5: {"ordinary_clean": 4, "hard_negative_clean": 2},
}
EXPECTED_FILLER_STRATA = {
    1: {"ordinary_clean": 7, "hard_negative_clean": 5},
    2: {"ordinary_clean": 7, "hard_negative_clean": 5},
    3: {"ordinary_clean": 7, "hard_negative_clean": 5},
    4: {"ordinary_clean": 7, "hard_negative_clean": 5},
    5: {"ordinary_clean": 8, "hard_negative_clean": 4},
}
OPAQUE_KEY_RE = re.compile(r"^[0-9a-f]{64}$")
FROZEN_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
FROZEN_DATA_CONSTRUCTION = "W2D-A6-v1"
A6_TOPICS = (
    "rec.sport.baseball",
    "sci.space",
    "comp.graphics",
    "talk.politics.mideast",
    "rec.autos",
    "sci.med",
    "soc.religion.christian",
    "misc.forsale",
)
A6_ORDINARY_AVAILABLE = {
    "rec.sport.baseball": 58,
    "sci.space": 47,
    "comp.graphics": 130,
    "talk.politics.mideast": 26,
    "rec.autos": 59,
    "sci.med": 61,
    "soc.religion.christian": 35,
    "misc.forsale": 247,
}
A6_ORDINARY_QUOTAS = {
    "rec.sport.baseball": {"calibration": 28, "test": 28},
    "sci.space": {"calibration": 24, "test": 23},
    "comp.graphics": {"calibration": 28, "test": 28},
    "talk.politics.mideast": {"calibration": 13, "test": 13},
    "rec.autos": {"calibration": 27, "test": 28},
    "sci.med": {"calibration": 27, "test": 28},
    "soc.religion.christian": {"calibration": 18, "test": 17},
    "misc.forsale": {"calibration": 27, "test": 27},
}


def _normalize_text(text: str) -> str:
    """Independently reproduce the frozen NFC/whitespace normalization."""

    return " ".join(unicodedata.normalize("NFC", text).split())


A6_NOMINAL_PER_TOPIC = {
    "clean_reference": 96,
    "calibration_ordinary_clean": 24,
    "calibration_hard_negative_clean": 8,
    "test_ordinary_clean": 24,
    "test_hard_negative_clean": 16,
    "calibration_recipe_T0": 8,
    "calibration_recipe_T1": 8,
    "calibration_recipe_T2": 8,
    "test_recipe_T3": 8,
    "test_recipe_T4": 8,
    "test_natural_cover_suffix": 8,
    "source_control_invalid_signature": 1,
    "source_control_provenance_conflict": 1,
    "source_control_unknown_source": 1,
}
A6_HARD_NEGATIVE_RULES = (
    "quoted_reply_lines",
    "reply_attribution_or_re_subject",
    "signature_footer_separator",
    "faq_boilerplate_or_repeated_lines",
    "cross_post_header",
    "nearest_topic_centroid_mismatch",
)

REQUIRED_ARTIFACTS = frozenset(
    {
        "legacy_authority_manifest",
        "legacy_authoritative_w2",
        "legacy_authoritative_w2r",
        "legacy_candidate_w2",
        "legacy_candidate_w2r",
        "legacy_same_code_repeat",
        "legacy_regression",
        "data_freeze",
        "safe_inputs_json",
        "safe_inputs_npz",
        "labels",
        "labels_checksum",
        "protocol_plan",
        "calibration_scores",
        "threshold",
        "test_scores",
        "source_controls",
        "source_control_gate",
        "detector_metrics",
        "landing",
    }
)

LEGACY_SOURCE_ARTIFACTS = {
    "authority_manifest": "legacy_authority_manifest",
    "authoritative_w2": "legacy_authoritative_w2",
    "authoritative_w2r": "legacy_authoritative_w2r",
    "candidate_w2": "legacy_candidate_w2",
    "candidate_w2r": "legacy_candidate_w2r",
    "same_code_repeat": "legacy_same_code_repeat",
}
EXPECTED_ARTIFACT_RELATIVE_PATHS = {
    "legacy_authority_manifest": "results/AUTHORITATIVE.json",
    "legacy_authoritative_w2": "results/W2-inmemory.json",
    "legacy_authoritative_w2r": "results/W2R-inmemory.json",
    "legacy_candidate_w2": "results/w2d/W2-legacy-regression-run-a.json",
    "legacy_candidate_w2r": "results/w2d/W2R-legacy-regression.json",
    "legacy_same_code_repeat": (
        "results/w2d/W2-legacy-regression-run-b-same-code.json"
    ),
    "legacy_regression": "results/w2d/W2D-LEGACY-REGRESSION.json",
    "data_freeze": "data/w2d/W2D-DATA-FREEZE.json",
    "safe_inputs_json": "data/w2d/W2D-detector-inputs.json",
    "safe_inputs_npz": "data/w2d/W2D-detector-inputs.npz",
    "labels": "results/w2d/W2D-labels.json",
    "labels_checksum": "results/w2d/W2D-labels.sha256",
    "protocol_plan": "results/w2d/W2D-PROTOCOL-PLAN.json",
    "calibration_scores": "results/w2d/W2D-calibration-scores.json",
    "threshold": "results/w2d/W2D-threshold.json",
    "test_scores": "results/w2d/W2D-test-scores.json",
    "source_controls": "results/w2d/W2D-source-controls.json",
    "source_control_gate": "results/w2d/W2D-source-control-gate.json",
    "detector_metrics": "results/w2d/W2D-detector-metrics.json",
    "landing": "results/w2d/W2D-LANDING.json",
}
FINGERPRINT_FILE_COMPONENTS = {
    "runtime": "functional_slice.py",
    "backend": "backend.py",
    "experiment_driver": "poison_exposure.py",
    "legacy_analysis": "analysis.py",
    "legacy_realtext_workload": "realtext_workload.py",
    "legacy_milvus_backend": "milvus_backend.py",
    "legacy_slice": "slice.py",
    "legacy_gate2": "gate2.py",
    "legacy_gate4": "gate4.py",
    "detector": "detector.py",
    "workload": "w2d_workload.py",
    "protocol": "w2d_protocol.py",
    "scorer": "w2d_scorer.py",
    "calibrator": "w2d_calibrate.py",
    "metrics": "w2d_metrics.py",
    "landing_evaluator": "w2d_landing.py",
    "runner": "w2d_runner.py",
    "verifier": "verify_w2d.py",
    "analyzer": "w2d_analyze.py",
    "preregistration": "W2D-PREREGISTRATION.md",
    "amendment_a1": "W2D-PREREGISTRATION-AMENDMENT-A1.md",
    "amendment_a2": "W2D-PREREGISTRATION-AMENDMENT-A2.md",
    "amendment_a3": "W2D-PREREGISTRATION-AMENDMENT-A3.md",
    "amendment_a4": "W2D-PREREGISTRATION-AMENDMENT-A4.md",
    "amendment_a5": "W2D-PREREGISTRATION-AMENDMENT-A5.md",
    "amendment_a6": "W2D-PREREGISTRATION-AMENDMENT-A6.md",
    "legacy_verifier": "verify_w2d_legacy.py",
    "legacy_verify_w2": "verify_w2.py",
    "legacy_verify_w2r": "verify_w2r.py",
    "legacy_test_runtime": "test_w2d_runtime.py",
    "legacy_test_w2": "test_w2.py",
    "legacy_test_invariants": "test_invariants.py",
    "legacy_gate_tests": "test_verify_w2d_legacy.py",
}
REQUIRED_FINGERPRINT_COMPONENTS = frozenset(
    {*FINGERPRINT_FILE_COMPONENTS, "model_revision"}
)
RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "measurement_name",
        "execution_mode",
        "runtime_fingerprint_sha256",
        "artifact_hashes",
        "detector_execution",
        "observation_horizon_s",
        "injection_at_s",
        "detector_denominator",
        "cells",
    }
)
CELL_FIELDS = frozenset(
    {
        "plan_cell_id",
        "attack_family",
        "arm",
        "baseline",
        "backlog",
        "seed",
        "protocol_items",
        "verifier_records",
        "lifecycle",
        "retrieval",
    }
)
PROTOCOL_ITEM_FIELDS = frozenset(
    {
        "item_key",
        "role",
        "ordinal",
        "detector_promote",
        "detector_service_ns",
        "score_record_sha256",
    }
)
LIFECYCLE_FIELDS = frozenset(
    {
        "item_key",
        "truth_poison",
        "detector_promote",
        "retry_count",
        "state_at_horizon",
        "visibility_status",
        "exposure_status",
        "exposure_observed_s",
        "unvetted_visibility_status",
        "unvetted_visibility_observed_s",
        "unavailable_status",
        "unavailable_observed_s",
        "quarantine_status",
        "readmission_count",
        "observed_at_horizon_s",
    }
)
VERIFIER_RECORD_FIELDS = frozenset(
    {
        "item_key",
        "status",
        "passes",
        "service_time_s",
        "queue_enter_s",
        "queue_start_s",
        "queue_wait_s",
        "queue_depth_at_enqueue",
        "queue_depth_at_start",
        "integrated_latency_s",
        "decision_commit_s",
    }
)
FINAL_STATES = frozenset(
    {"PROVISIONAL", "TRUSTED", "QUARANTINED", "HIDDEN", "REVOKED"}
)
EPISODE_STATUSES = frozenset(
    {"NOT_STARTED", "COMPLETED", "RIGHT_CENSORED", "NOT_APPLICABLE"}
)
PROVIDER_STATUSES = frozenset({"COMMITTED", "CANCELLED"})
CALIBRATION_SCORE_FIELDS = frozenset(
    {
        "item_key",
        "rep",
        "knn",
        "c_score",
        "family_s_affirms",
        "detector_service_ns",
    }
)
LANDING_FIELDS = frozenset(
    {
        "schema_version",
        "measurement",
        "top_k",
        "tie_break",
        "candidate_population",
        "protocol_plan_sha256",
        "safe_inputs_sha256",
        "embeddings_sha256",
        "retrieval_background_sha256",
        "populations",
        "items",
    }
)
LANDING_ITEM_FIELDS = frozenset(
    {
        "item_key",
        "attack_family",
        "rank",
        "candidate_rank",
        "landed_top5",
        "landed",
        "top5_item_keys",
    }
)


class W2DVerificationError(ValueError):
    """A formal bundle violated its frozen schema or a W2D gate."""


@dataclass
class VerificationReport:
    checks: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def passed(self, name: str) -> None:
        self.checks.append(name)

    def failed(self, message: str) -> None:
        self.failures.append(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise W2DVerificationError(message)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise W2DVerificationError(f"{path} must be a mapping")
    return value


def _sequence(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise W2DVerificationError(f"{path} must be a list")
    return value


def _exact_fields(value: Mapping[str, Any], expected: frozenset[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise W2DVerificationError(
            f"{path} schema mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _required_fields(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], path: str
) -> None:
    missing = set(expected) - set(value)
    if missing:
        raise W2DVerificationError(f"{path} missing required fields {sorted(missing)}")


def _sha256(value: Any, path: str) -> str:
    text = value if isinstance(value, str) else ""
    if not OPAQUE_KEY_RE.fullmatch(text):
        raise W2DVerificationError(f"{path} must be a lowercase SHA256")
    return text


def _item_key(value: Any, path: str) -> str:
    return _sha256(value, path)


def _strict_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise W2DVerificationError(f"{path} must be bool")
    return value


def _nonnegative_number(value: Any, path: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise W2DVerificationError(f"{path} must be finite and non-negative")
    return float(value)


def _recompute_a6_ordinary_quotas(
    available: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    """Independent implementation of Amendment A6's outcome-blind repair."""
    _require(
        set(available) == set(A6_TOPICS),
        "A6 ordinary availability does not cover the frozen topic set",
    )
    parsed: dict[str, int] = {}
    for topic in A6_TOPICS:
        value = available[topic]
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"A6 ordinary availability for {topic} is invalid",
        )
        parsed[topic] = value

    nominal = 24
    quotas: dict[str, dict[str, int]] = {}
    for topic in A6_TOPICS:
        capacity = parsed[topic]
        quotas[topic] = (
            {"calibration": nominal, "test": nominal}
            if capacity >= 2 * nominal
            else {
                "calibration": (capacity + 1) // 2,
                "test": capacity // 2,
            }
        )

    target = len(A6_TOPICS) * nominal
    for split in ("calibration", "test"):
        remaining = target - sum(quotas[topic][split] for topic in A6_TOPICS)
        while remaining:
            progressed = False
            for topic in A6_TOPICS:
                used = quotas[topic]["calibration"] + quotas[topic]["test"]
                if used >= parsed[topic]:
                    continue
                quotas[topic][split] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
            _require(
                progressed,
                f"A6 global {split} ordinary-clean quota is infeasible",
            )
    return quotas


def verify_data_freeze_a6(value: Any) -> dict[str, Mapping[str, Any]]:
    """Fail-loud, producer-independent checks for the A6 quota repair."""
    freeze = _mapping(value, "data_freeze")
    _require(freeze.get("schema_version") == "1.0", "data freeze schema mismatch")
    _require(
        freeze.get("construction_version") == FROZEN_DATA_CONSTRUCTION,
        "data freeze is not the frozen A6 construction",
    )
    provenance = _mapping(
        freeze.get("build_provenance"), "data_freeze.build_provenance"
    )
    _exact_fields(
        provenance,
        frozenset(
            {
                "git_commit",
                "git_dirty",
                "dirty_paths",
                "w2d_workload_py_sha256",
                "amendment_a6_sha256",
            }
        ),
        "data_freeze.build_provenance",
    )
    _require(
        isinstance(provenance["git_commit"], str)
        and bool(re.fullmatch(r"[0-9a-f]{40}", provenance["git_commit"]))
        and provenance["git_dirty"] is False
        and provenance["dirty_paths"] == [],
        "A6 production data was not built from a clean committed tree",
    )
    _sha256(
        provenance["w2d_workload_py_sha256"],
        "data_freeze.build_provenance.w2d_workload_py_sha256",
    )
    _sha256(
        provenance["amendment_a6_sha256"],
        "data_freeze.build_provenance.amendment_a6_sha256",
    )
    _require(
        freeze.get("nominal_per_topic") == A6_NOMINAL_PER_TOPIC,
        "data freeze nominal per-topic counts moved",
    )

    rule = _mapping(
        freeze.get("ordinary_clean_quota_rule"),
        "data_freeze.ordinary_clean_quota_rule",
    )
    _exact_fields(
        rule,
        frozenset(
            {
                "amendment",
                "available_after_reference",
                "quota",
                "global_calibration_total",
                "global_test_total",
                "topic_order",
            }
        ),
        "data_freeze.ordinary_clean_quota_rule",
    )
    _require(
        rule["amendment"] == "W2D-PREREGISTRATION-AMENDMENT-A6.md",
        "data freeze does not identify Amendment A6",
    )
    _require(
        rule["topic_order"] == list(A6_TOPICS),
        "A6 topic order changed",
    )
    available = _mapping(
        rule["available_after_reference"], "A6 available_after_reference"
    )
    _require(
        dict(available) == A6_ORDINARY_AVAILABLE,
        "A6 recorded ordinary-clean capacities differ from the preregistration",
    )
    recomputed = _recompute_a6_ordinary_quotas(available)
    quota = _mapping(rule["quota"], "A6 quota")
    _require(
        dict(quota) == recomputed == A6_ORDINARY_QUOTAS,
        "A6 ordinary-clean quota is not the independent recomputation",
    )
    _require(
        rule["global_calibration_total"] == 192
        and rule["global_test_total"] == 192
        and sum(row["calibration"] for row in recomputed.values()) == 192
        and sum(row["test"] for row in recomputed.values()) == 192,
        "A6 global ordinary-clean totals moved",
    )

    expected_per_topic = {
        topic: {
            **A6_NOMINAL_PER_TOPIC,
            "calibration_ordinary_clean": recomputed[topic]["calibration"],
            "test_ordinary_clean": recomputed[topic]["test"],
        }
        for topic in A6_TOPICS
    }
    _require(
        freeze.get("expected_per_topic") == expected_per_topic,
        "A6 expected per-topic ledger differs from its repaired quotas",
    )
    counts = _mapping(freeze.get("counts"), "data_freeze.counts")
    expected_counts = {
        "total_items": 1752,
        "unique_source_groups": 1752,
        "reference": 768,
        "calibration_quality": 448,
        "test_quality": 512,
        "source_controls": 24,
        "calibration_clean": 256,
        "calibration_recipe_poison": 192,
        "test_clean": 320,
        "test_recipe_poison": 128,
        "test_natural_cover_poison": 64,
    }
    _exact_fields(
        counts,
        frozenset({*expected_counts, "per_topic"}),
        "data_freeze.counts",
    )

    allocation = _sequence(freeze.get("allocation"), "data_freeze.allocation")
    _require(
        len(allocation) == expected_counts["total_items"],
        "A6 allocation does not contain exactly 1752 records",
    )
    item_keys: set[str] = set()
    source_groups: set[str] = set()
    allocation_by_key: dict[str, Mapping[str, Any]] = {}
    selected: Counter[tuple[str, str, str]] = Counter()
    per_topic: dict[str, Counter[str]] = {
        topic: Counter() for topic in A6_TOPICS
    }
    recomputed_counts: Counter[str] = Counter()
    hard_keys: dict[str, set[str]] = {"calibration": set(), "test": set()}
    hard_items: dict[str, dict[str, list[str]]] = {
        "calibration": {},
        "test": {},
    }
    hard_rule_counts: dict[str, Counter[str]] = {
        "calibration": Counter(),
        "test": Counter(),
    }
    allocation_fields = frozenset(
        {
            "item_key",
            "source_group",
            "source_id",
            "source_aliases",
            "topic",
            "split",
            "stratum",
            "attack_family",
            "attack_variant",
            "source_control_kind",
            "hard_negative_rules",
            "raw_sha256",
            "normalized_text_sha256",
        }
    )
    natural_cover_fields = frozenset(
        {"suffix_append_count", "longest_suffix_query_shared_ngram"}
    )
    for index, raw in enumerate(allocation):
        record = _mapping(raw, f"data_freeze.allocation[{index}]")
        expected_fields = allocation_fields
        if record.get("stratum") == "natural_cover_suffix_poison":
            expected_fields |= natural_cover_fields
        _exact_fields(
            record,
            expected_fields,
            f"data_freeze.allocation[{index}]",
        )
        item_key = _item_key(
            record["item_key"], f"data_freeze.allocation[{index}].item_key"
        )
        source_group = _sha256(
            record["source_group"],
            f"data_freeze.allocation[{index}].source_group",
        )
        _require(
            item_key not in item_keys and source_group not in source_groups,
            "A6 allocation reuses an item key or source group",
        )
        _require(
            item_key
            == hashlib.sha256(
                f"W2D-item|{source_group}".encode("utf-8")
            ).hexdigest(),
            "A6 item key is not derived from its source group",
        )
        source_id = record["source_id"]
        aliases = _sequence(
            record["source_aliases"],
            f"data_freeze.allocation[{index}].source_aliases",
        )
        _require(
            isinstance(source_id, str)
            and bool(source_id)
            and bool(aliases)
            and all(isinstance(alias, str) and bool(alias) for alias in aliases)
            and len(aliases) == len(set(aliases))
            and list(aliases) == sorted(aliases)
            and source_id in aliases,
            "A6 allocation source identity/alias ledger is malformed",
        )
        _sha256(
            record["raw_sha256"],
            f"data_freeze.allocation[{index}].raw_sha256",
        )
        _sha256(
            record["normalized_text_sha256"],
            f"data_freeze.allocation[{index}].normalized_text_sha256",
        )
        item_keys.add(item_key)
        source_groups.add(source_group)
        allocation_by_key[item_key] = record
        topic = record["topic"]
        split = record["split"]
        stratum = record["stratum"]
        _require(topic in A6_TOPICS, "A6 allocation contains an unknown topic")
        rules = _sequence(
            record["hard_negative_rules"],
            f"data_freeze.allocation[{index}].hard_negative_rules",
        )
        _require(
            len(rules) == len(set(rules))
            and all(rule in A6_HARD_NEGATIVE_RULES for rule in rules),
            "A6 allocation contains an unknown/duplicate hard-negative rule",
        )
        family = record["attack_family"]
        variant = record["attack_variant"]
        control = record["source_control_kind"]
        count_key: str
        if split == "reference" and stratum == "clean_reference":
            _require(
                family is None and variant is None and control is None,
                "A6 reference record carries attack/control metadata",
            )
            count_key = "clean_reference"
            recomputed_counts["reference"] += 1
        elif split in {"calibration", "test"} and stratum == "ordinary_clean":
            _require(
                not rules
                and family is None
                and variant is None
                and control is None,
                "A6 ordinary-clean record has a hard-negative rule or wrong split",
            )
            selected[(str(topic), str(split), "ordinary")] += 1
            count_key = f"{split}_ordinary_clean"
            recomputed_counts[f"{split}_clean"] += 1
        elif split in {"calibration", "test"} and stratum == "hard_negative_clean":
            _require(
                bool(rules)
                and family is None
                and variant is None
                and control is None,
                "A6 hard-negative record has no rule or wrong split",
            )
            selected[(str(topic), str(split), "hard")] += 1
            hard_keys[str(split)].add(item_key)
            hard_items[str(split)][item_key] = list(rules)
            hard_rule_counts[str(split)].update(rules)
            count_key = f"{split}_hard_negative_clean"
            recomputed_counts[f"{split}_clean"] += 1
        elif split == "calibration" and stratum == "recipe_poison":
            _require(
                family == "recipe"
                and variant in {"T0", "T1", "T2"}
                and control is None,
                "A6 calibration recipe record has an invalid family/variant",
            )
            count_key = f"calibration_recipe_{variant}"
            recomputed_counts["calibration_recipe_poison"] += 1
        elif split == "test" and stratum == "recipe_poison":
            _require(
                family == "recipe"
                and variant in {"T3", "T4"}
                and control is None,
                "A6 test recipe record has an invalid family/variant",
            )
            count_key = f"test_recipe_{variant}"
            recomputed_counts["test_recipe_poison"] += 1
        elif split == "test" and stratum == "natural_cover_suffix_poison":
            _require(
                family == "natural_cover_suffix_v1"
                and variant == "natural_cover_suffix_v1"
                and control is None,
                "A6 natural-cover record has an invalid family/variant",
            )
            shared_ngram = record["longest_suffix_query_shared_ngram"]
            _require(
                record["suffix_append_count"] == 1
                and isinstance(shared_ngram, int)
                and not isinstance(shared_ngram, bool)
                and 0 <= shared_ngram <= 5,
                "A6 natural-cover construction diagnostics moved",
            )
            count_key = "test_natural_cover_suffix"
            recomputed_counts["test_natural_cover_poison"] += 1
        elif (
            split == "implementation_control"
            and stratum == "source_family_negative_control"
        ):
            _require(
                family is None
                and variant is None
                and control
                in {
                    "invalid_signature",
                    "provenance_conflict",
                    "unknown_source",
                },
                "A6 source-control record has invalid metadata",
            )
            count_key = f"source_control_{control}"
            recomputed_counts["source_controls"] += 1
        else:
            raise W2DVerificationError(
                "A6 allocation contains an unknown split/stratum combination"
            )
        per_topic[str(topic)][count_key] += 1
        if split == "calibration":
            recomputed_counts["calibration_quality"] += 1
        elif split == "test":
            recomputed_counts["test_quality"] += 1

    recomputed_counts["total_items"] = len(allocation)
    recomputed_counts["unique_source_groups"] = len(source_groups)
    recomputed_document = {
        **{
            name: recomputed_counts[name]
            for name in expected_counts
        },
        "per_topic": {
            topic: dict(sorted(counter.items()))
            for topic, counter in per_topic.items()
        },
    }
    _require(
        recomputed_document == counts,
        "A6 counts/per-topic ledger is not the allocation recomputation",
    )
    for name, expected in expected_counts.items():
        _require(
            recomputed_document[name] == expected,
            f"A6 data-freeze count {name} is not {expected}",
        )
    _require(
        recomputed_document["per_topic"] == expected_per_topic,
        "A6 per-topic allocation differs from the repaired quotas",
    )

    classification = _mapping(
        freeze.get("hard_negative_classification"),
        "data_freeze.hard_negative_classification",
    )
    _exact_fields(
        classification,
        frozenset({"allowed_rules", "records"}),
        "data_freeze.hard_negative_classification",
    )
    _require(
        classification["allowed_rules"] == list(A6_HARD_NEGATIVE_RULES),
        "A6 hard-negative rule vocabulary moved",
    )
    classified: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(
        _sequence(
            classification["records"],
            "data_freeze.hard_negative_classification.records",
        )
    ):
        record = _mapping(
            raw, f"data_freeze.hard_negative_classification.records[{index}]"
        )
        _exact_fields(
            record,
            frozenset({"group_id", "topic", "rules"}),
            f"data_freeze.hard_negative_classification.records[{index}]",
        )
        group = _sha256(
            record["group_id"],
            f"data_freeze.hard_negative_classification.records[{index}].group_id",
        )
        rules = _sequence(
            record["rules"],
            f"data_freeze.hard_negative_classification.records[{index}].rules",
        )
        _require(
            group not in classified
            and record["topic"] in A6_TOPICS
            and len(rules) == len(set(rules))
            and all(rule in A6_HARD_NEGATIVE_RULES for rule in rules),
            "A6 hard-negative classification record is duplicate/malformed",
        )
        classified[group] = record

    eligibility = _mapping(
        freeze.get("base_eligibility"), "data_freeze.base_eligibility"
    )
    eligibility_records = _sequence(
        eligibility.get("records"), "data_freeze.base_eligibility.records"
    )
    eligible_topics: dict[str, str] = {}
    for index, raw in enumerate(eligibility_records):
        record = _mapping(
            raw, f"data_freeze.base_eligibility.records[{index}]"
        )
        group = _sha256(
            record.get("group_id"),
            f"data_freeze.base_eligibility.records[{index}].group_id",
        )
        _require(
            group not in eligible_topics and record.get("topic") in A6_TOPICS,
            "A6 base-eligibility record is duplicate or has an unknown topic",
        )
        eligible_topics[group] = str(record["topic"])
    _require(
        set(eligible_topics) == set(classified)
        and all(
            eligible_topics[group] == classified[group]["topic"]
            for group in classified
        ),
        "A6 hard-negative classification does not cover exact eligible inventory",
    )
    for record in allocation_by_key.values():
        group = str(record["source_group"])
        _require(
            group in classified
            and classified[group]["topic"] == record["topic"]
            and classified[group]["rules"] == record["hard_negative_rules"],
            "A6 allocation differs from the eligible hard-rule ledger",
        )

    reference_groups = {
        str(record["source_group"])
        for record in allocation_by_key.values()
        if record["split"] == "reference"
    }
    capacity_from_ledger = {
        topic: sum(
            record["topic"] == topic
            and group not in reference_groups
            and not record["rules"]
            for group, record in classified.items()
        )
        for topic in A6_TOPICS
    }
    _require(
        capacity_from_ledger == A6_ORDINARY_AVAILABLE,
        "A6 ordinary capacities do not recompute from eligible hard-rule ledger",
    )

    unallocated_rows = _sequence(
        freeze.get("eligible_unallocated"), "data_freeze.eligible_unallocated"
    )
    unallocated: dict[str, str] = {}
    for index, raw in enumerate(unallocated_rows):
        record = _mapping(raw, f"data_freeze.eligible_unallocated[{index}]")
        group = _sha256(
            record.get("group_id"),
            f"data_freeze.eligible_unallocated[{index}].group_id",
        )
        _require(
            group not in unallocated and record.get("topic") in A6_TOPICS,
            "A6 eligible-unallocated record is duplicate/malformed",
        )
        unallocated[group] = str(record["topic"])
    _require(
        set(unallocated) == set(classified) - source_groups
        and all(
            unallocated[group] == classified[group]["topic"]
            for group in unallocated
        ),
        "A6 eligible-unallocated ledger does not complement allocation",
    )

    source = _mapping(freeze.get("source"), "data_freeze.source")
    inventory = _mapping(
        source.get("inventory"), "data_freeze.source.inventory"
    )
    inventory_fields = {
        "raw_post_count",
        "canonical_group_count",
        "duplicate_noncanonical_count",
        "base_eligible_group_count",
        "base_excluded_group_count",
        "allocated_group_count",
        "eligible_unallocated_group_count",
    }
    _require(
        all(
            isinstance(inventory.get(name), int)
            and not isinstance(inventory.get(name), bool)
            and inventory[name] >= 0
            for name in inventory_fields
        ),
        "A6 source inventory contains an invalid count",
    )
    exclusions = _sequence(freeze.get("exclusions"), "data_freeze.exclusions")
    duplicate_n = sum(
        isinstance(row, Mapping)
        and row.get("reason") == "duplicate_noncanonical"
        for row in exclusions
    )
    base_excluded_n = sum(
        isinstance(row, Mapping) and row.get("reason") == "base_eligibility"
        for row in exclusions
    )
    _require(
        duplicate_n + base_excluded_n == len(exclusions),
        "A6 exclusions contain an unknown reason",
    )
    _require(
        source.get("raw_group_count") == inventory["canonical_group_count"]
        and inventory["raw_post_count"]
        == inventory["canonical_group_count"]
        + inventory["duplicate_noncanonical_count"]
        and inventory["canonical_group_count"]
        == inventory["base_eligible_group_count"]
        + inventory["base_excluded_group_count"]
        and inventory["base_eligible_group_count"] == len(classified)
        and inventory["allocated_group_count"] == len(allocation_by_key)
        and inventory["eligible_unallocated_group_count"] == len(unallocated)
        and inventory["base_eligible_group_count"]
        == inventory["allocated_group_count"]
        + inventory["eligible_unallocated_group_count"]
        and inventory["duplicate_noncanonical_count"] == duplicate_n
        and inventory["base_excluded_group_count"] == base_excluded_n,
        "A6 source/eligibility/exclusion inventory does not balance",
    )

    for topic in A6_TOPICS:
        for split in ("calibration", "test"):
            _require(
                selected[(topic, split, "ordinary")]
                == recomputed[topic][split],
                f"A6 selected ordinary count differs for {topic}/{split}",
            )
        _require(
            selected[(topic, "calibration", "hard")] == 8
            and selected[(topic, "test", "hard")] == 16,
            f"A6 selected hard-negative count differs for {topic}",
        )

    hard_summary = _mapping(
        freeze.get("hard_negative_selection"),
        "data_freeze.hard_negative_selection",
    )
    for split, expected in (("calibration", 64), ("test", 128)):
        summary = _mapping(
            hard_summary.get(split),
            f"data_freeze.hard_negative_selection.{split}",
        )
        _exact_fields(
            summary,
            frozenset({"n", "item_keys", "rule_counts", "items"}),
            f"data_freeze.hard_negative_selection.{split}",
        )
        keys = _sequence(
            summary.get("item_keys"),
            f"data_freeze.hard_negative_selection.{split}.item_keys",
        )
        _require(
            summary.get("n") == expected
            and len(keys) == expected
            and set(keys) == hard_keys[split],
            f"A6 {split} hard-negative summary differs from allocation",
        )
        _require(
            summary["rule_counts"]
            == dict(sorted(hard_rule_counts[split].items()))
            and summary["items"]
            == {
                key: hard_items[split][key]
                for key in sorted(hard_items[split])
            },
            f"A6 {split} hard-negative rule ledger differs from allocation",
        )
    return allocation_by_key


def canonical_record_sha256(record: Mapping[str, Any]) -> str:
    """Digest one strict-JSON score record for replay back-references."""
    try:
        payload = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise W2DVerificationError(f"score record is not strict JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def runtime_fingerprint_digest(components: Mapping[str, Mapping[str, Any]]) -> str:
    """Canonical aggregate over named component digests.

    The aggregate contains names as well as hashes, so swapping two component
    files cannot preserve the fingerprint.
    """
    normalized = {
        str(name): _sha256(_mapping(record, f"component {name}").get("sha256"),
                           f"component {name}.sha256")
        for name, record in components.items()
    }
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_runtime_fingerprint(
    components: Mapping[str, Mapping[str, Any]], base_dir: str | Path
) -> dict[str, Any]:
    """Build (but do not write) the independent fingerprint record.

    This helper exists so the runner and verifier use exactly one aggregate
    definition.  File entries use ``{"path": ...}``; the model revision uses
    ``{"value": ...}``.
    """
    root = Path(base_dir)
    frozen: dict[str, dict[str, str]] = {}
    for name, raw in components.items():
        record = _mapping(raw, f"component {name}")
        has_path = isinstance(record.get("path"), str)
        has_value = isinstance(record.get("value"), str)
        _require(has_path ^ has_value, f"component {name} needs exactly path or value")
        if has_path:
            path = Path(record["path"])
            actual = file_sha256(str(path if path.is_absolute() else root / path))
            frozen[str(name)] = {"path": record["path"], "sha256": actual}
        else:
            value = record["value"]
            _require(bool(value), f"component {name}.value must be non-empty")
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
            frozen[str(name)] = {"value": value, "sha256": digest}
    return {
        "algorithm": "sha256",
        "components": frozen,
        "sha256": runtime_fingerprint_digest(frozen),
    }


def _resolve(root: Path, raw: Any, path: str) -> Path:
    _require(isinstance(raw, str) and bool(raw), f"{path} must be a path string")
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else root / candidate


def _load_artifacts(
    manifest_path: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, str], dict[str, Path]]:
    entries = _mapping(manifest.get("artifact_hashes"), "manifest.artifact_hashes")
    _require(
        set(entries) == REQUIRED_ARTIFACTS,
        "manifest artifact role set mismatch; "
        f"missing={sorted(REQUIRED_ARTIFACTS - set(entries))}, "
        f"extra={sorted(set(entries) - REQUIRED_ARTIFACTS)}",
    )
    loaded: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    paths: dict[str, Path] = {}
    resolved_roles: dict[Path, str] = {}
    for name, raw in entries.items():
        record = _mapping(raw, f"manifest.artifact_hashes.{name}")
        _exact_fields(record, frozenset({"path", "sha256"}),
                      f"manifest.artifact_hashes.{name}")
        expected = _sha256(record["sha256"], f"artifact {name}.sha256")
        path = _resolve(manifest_path.parent, record["path"], f"artifact {name}.path")
        resolved = path.resolve()
        expected_resolved = (
            EXPECTED_ARTIFACT_ROOT / EXPECTED_ARTIFACT_RELATIVE_PATHS[str(name)]
        ).resolve()
        _require(
            resolved == expected_resolved,
            f"artifact role {name} points to {resolved}, expected {expected_resolved}",
        )
        _require(
            resolved not in resolved_roles,
            f"artifact roles {resolved_roles.get(resolved)!r} and {name!r} "
            "resolve to the same path",
        )
        resolved_roles[resolved] = str(name)
        _require(path.is_file(), f"artifact {name} is absent: {path}")
        actual = file_sha256(str(path))
        _require(actual == expected, f"artifact {name} sha256 {actual} != {expected}")
        hashes[str(name)] = actual
        paths[str(name)] = resolved
        if path.suffix.lower() == ".json":
            try:
                loaded[str(name)] = strict_json_load(str(path))
            except (MetricsError, json.JSONDecodeError, OSError) as exc:
                raise W2DVerificationError(
                    f"artifact {name} is not strict JSON: {exc}"
                ) from exc
        else:
            loaded[str(name)] = path
    return loaded, hashes, paths


def verify_a6_artifact_lineage(
    artifacts: Mapping[str, Any],
    hashes: Mapping[str, str],
    paths: Mapping[str, Path],
) -> None:
    freeze = _mapping(artifacts["data_freeze"], "data_freeze")
    allocation = verify_data_freeze_a6(freeze)
    model = _mapping(freeze.get("model"), "data_freeze.model")
    _require(
        model
        == {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "revision": FROZEN_MODEL_REVISION,
            "dimension": 384,
        },
        "data freeze model pin moved",
    )
    provenance = _mapping(
        freeze.get("build_provenance"), "data_freeze.build_provenance"
    )
    _require(
        provenance["w2d_workload_py_sha256"]
        == file_sha256(
            str(EXPECTED_FINGERPRINT_ROOT / "w2d_workload.py")
        )
        and provenance["amendment_a6_sha256"]
        == file_sha256(
            str(
                EXPECTED_FINGERPRINT_ROOT
                / "W2D-PREREGISTRATION-AMENDMENT-A6.md"
            )
        ),
        "A6 build provenance differs from the frozen builder/amendment bytes",
    )

    ledger = _mapping(freeze.get("artifacts"), "data_freeze.artifacts")
    expected_roles = {
        "safe_inputs_json": "safe_detector_schema",
        "safe_inputs_npz": "safe_full_item_embeddings",
        "labels": "evaluator_only_labels_queries_and_attack_family",
        "labels_checksum": "label_artifact_digest",
    }
    expected_filenames = {paths[role].name for role in expected_roles}
    _require(
        set(ledger) == expected_filenames,
        "data freeze artifact ledger is incomplete or contains an extra file",
    )
    for role, expected_role in expected_roles.items():
        filename = paths[role].name
        record = _mapping(
            ledger.get(filename), f"data_freeze.artifacts.{filename}"
        )
        _exact_fields(
            record,
            frozenset({"sha256", "role"}),
            f"data_freeze.artifacts.{filename}",
        )
        _require(
            record["sha256"] == hashes[role]
            and record["role"] == expected_role,
            f"data freeze does not bind the formal {role} artifact",
        )

    safe = _mapping(artifacts["safe_inputs_json"], "safe_inputs_json")
    labels = _mapping(artifacts["labels"], "labels")
    for name, document in (("safe inputs", safe), ("labels", labels)):
        _require(
            document.get("schema_version") == "1.0"
            and document.get("construction_version") == FROZEN_DATA_CONSTRUCTION
            and document.get("model_revision") == FROZEN_MODEL_REVISION,
            f"{name} does not identify the frozen A6/model construction",
        )
    _require(
        safe.get("model") == model,
        "safe-input model descriptor differs from the data freeze",
    )
    embedding = _mapping(
        safe.get("embedding_artifact"), "safe_inputs_json.embedding_artifact"
    )
    _exact_fields(
        embedding,
        frozenset({"filename", "sha256", "array", "dtype", "shape"}),
        "safe_inputs_json.embedding_artifact",
    )
    _require(
        embedding.get("filename") == paths["safe_inputs_npz"].name
        and embedding.get("sha256") == hashes["safe_inputs_npz"],
        "safe-input JSON does not bind the formal embedding artifact",
    )
    _require(
        embedding["array"] == "embeddings"
        and embedding["dtype"] == "float32"
        and embedding["shape"] == [1752, 384],
        "safe-input embedding descriptor moved",
    )
    try:
        import numpy as np

        with np.load(paths["safe_inputs_npz"], allow_pickle=False) as archive:
            _require(
                archive.files == ["embeddings"],
                "embedding NPZ array set moved",
            )
            matrix = np.asarray(archive["embeddings"])
            _require(
                matrix.dtype == np.dtype("float32")
                and matrix.shape == (1752, 384)
                and bool(np.isfinite(matrix).all()),
                "embedding NPZ dtype/shape/finiteness is invalid",
            )
            norms = np.linalg.norm(
                matrix.astype(np.float64, copy=False), axis=1
            )
            _require(
                bool(np.isfinite(norms).all())
                and bool(
                    np.allclose(
                        norms,
                        np.ones(1752, dtype=np.float64),
                        rtol=1e-4,
                        atol=1e-5,
                    )
                ),
                "embedding NPZ contains a zero or non-normalized row",
            )
    except (OSError, ValueError, TypeError) as exc:
        raise W2DVerificationError(
            f"embedding NPZ cannot be independently loaded: {exc}"
        ) from exc

    safe_rows = _sequence(safe.get("items"), "safe_inputs_json.items")
    safe_by_key: dict[str, Mapping[str, Any]] = {}
    safe_groups: set[str] = set()
    embedding_rows: set[int] = set()
    safe_item_fields = frozenset(
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
    for index, raw in enumerate(safe_rows):
        record = _mapping(raw, f"safe_inputs_json.items[{index}]")
        _exact_fields(
            record, safe_item_fields, f"safe_inputs_json.items[{index}]"
        )
        key = _item_key(
            record["item_key"], f"safe_inputs_json.items[{index}].item_key"
        )
        group = _sha256(
            record["source_group"],
            f"safe_inputs_json.items[{index}].source_group",
        )
        row = record["embedding_row"]
        _require(
            key not in safe_by_key
            and group not in safe_groups
            and isinstance(row, int)
            and not isinstance(row, bool)
            and row >= 0
            and row not in embedding_rows,
            "safe-input item/group/embedding-row mapping is not one-to-one",
        )
        _require(
            isinstance(record["normalized_text"], str)
            and bool(record["normalized_text"])
            and _normalize_text(record["normalized_text"])
            == record["normalized_text"]
            and isinstance(record["source_evidence"], list)
            and len(record["source_evidence"]) == 3
            and record["model_revision"] == FROZEN_MODEL_REVISION,
            "safe-input item has malformed scorer material",
        )
        safe_by_key[key] = record
        safe_groups.add(group)
        embedding_rows.add(row)
    _require(
        set(safe_by_key) == set(allocation)
        and embedding_rows == set(range(len(allocation))),
        "data-freeze allocation and safe-input key universe differ",
    )

    labels_by_key = _label_map(labels)
    _require(
        set(labels_by_key) == set(allocation),
        "data-freeze allocation and evaluator-label key universe differ",
    )
    for key, frozen in allocation.items():
        label = _mapping(labels_by_key[key], f"label {key}")
        safe_item = safe_by_key[key]
        for field_name in (
            "source_group",
            "topic",
            "split",
            "stratum",
            "attack_family",
            "attack_variant",
            "source_control_kind",
            "hard_negative_rules",
        ):
            _require(
                label.get(field_name) == frozen.get(field_name),
                f"label {key}.{field_name} differs from data-freeze allocation",
            )
        expected_poison = frozen["stratum"] in {
            "recipe_poison",
            "natural_cover_suffix_poison",
        }
        _require(
            label.get("poison") is expected_poison,
            f"label {key}.poison differs from the allocation stratum",
        )
        _require(
            safe_item["source_group"] == frozen["source_group"]
            and safe_item["reference"] is (frozen["split"] == "reference"),
            f"safe-input item {key} differs from data-freeze allocation",
        )
        _require(
            hashlib.sha256(
                safe_item["normalized_text"].encode("utf-8")
            ).hexdigest()
            == frozen["normalized_text_sha256"],
            f"safe-input item {key} text differs from the data-freeze allocation",
        )
        control = frozen["source_control_kind"]
        if control in {"invalid_signature", "unknown_source"}:
            expected_evidence = [False, True, FROZEN_MODEL_REVISION]
        elif control == "provenance_conflict":
            expected_evidence = [True, False, FROZEN_MODEL_REVISION]
        else:
            expected_evidence = [True, True, FROZEN_MODEL_REVISION]
        _require(
            safe_item["source_evidence"] == expected_evidence,
            f"safe-input item {key} source evidence differs from its allocation",
        )

    ordered_safe_keys = [str(record["item_key"]) for record in safe_rows]
    _require(
        ordered_safe_keys == sorted(ordered_safe_keys),
        "safe-input items are not in frozen opaque-key order",
    )
    reference = _mapping(
        safe.get("reference_index"), "safe_inputs_json.reference_index"
    )
    _exact_fields(
        reference,
        frozenset({"item_keys", "embedding_rows", "groups"}),
        "safe_inputs_json.reference_index",
    )
    expected_reference_keys = [
        key for key in ordered_safe_keys if allocation[key]["split"] == "reference"
    ]
    _require(
        reference["item_keys"] == expected_reference_keys
        and reference["embedding_rows"]
        == [safe_by_key[key]["embedding_row"] for key in expected_reference_keys]
        and reference["groups"]
        == [safe_by_key[key]["source_group"] for key in expected_reference_keys],
        "safe reference index differs from the A6 allocation",
    )

    sets = _mapping(safe.get("sets"), "safe_inputs_json.sets")
    _exact_fields(
        sets,
        frozenset(
            {
                "calibration_item_keys",
                "test_score_order",
                "source_control_item_keys",
                "warmup_reference_item_key",
            }
        ),
        "safe_inputs_json.sets",
    )
    calibration_keys = sorted(
        key for key, row in allocation.items() if row["split"] == "calibration"
    )
    test_keys = [
        key for key, row in allocation.items() if row["split"] == "test"
    ]
    test_score_order = sorted(
        test_keys,
        key=lambda key: hashlib.sha256(
            f"W2D-score-order|{key}".encode("utf-8")
        ).hexdigest(),
    )
    control_keys = sorted(
        key
        for key, row in allocation.items()
        if row["split"] == "implementation_control"
    )
    _require(
        sets["calibration_item_keys"] == calibration_keys
        and sets["test_score_order"] == test_score_order
        and sets["source_control_item_keys"] == control_keys
        and sets["warmup_reference_item_key"] == min(expected_reference_keys),
        "safe phase sets/reference warmup differ from the A6 allocation",
    )


def _verify_fingerprint(manifest_path: Path, manifest: Mapping[str, Any]) -> str:
    fp = _mapping(manifest.get("runtime_fingerprint"), "manifest.runtime_fingerprint")
    _exact_fields(
        fp, frozenset({"algorithm", "components", "sha256"}),
        "manifest.runtime_fingerprint",
    )
    _require(fp["algorithm"] == "sha256", "fingerprint algorithm must be sha256")
    components = _mapping(fp["components"], "runtime_fingerprint.components")
    _require(
        set(components) == REQUIRED_FINGERPRINT_COMPONENTS,
        "runtime fingerprint component set mismatch; "
        f"missing={sorted(REQUIRED_FINGERPRINT_COMPONENTS - set(components))}, "
        f"extra={sorted(set(components) - REQUIRED_FINGERPRINT_COMPONENTS)}",
    )
    resolved_components: dict[Path, str] = {}
    for name, raw in components.items():
        record = _mapping(raw, f"fingerprint component {name}")
        expected = _sha256(record.get("sha256"), f"component {name}.sha256")
        if "path" in record:
            _exact_fields(record, frozenset({"path", "sha256"}), f"component {name}")
            path = _resolve(manifest_path.parent, record["path"], f"component {name}.path")
            _require(path.is_file(), f"fingerprint component {name} is absent")
            resolved = path.resolve()
            expected_filename = FINGERPRINT_FILE_COMPONENTS.get(str(name))
            _require(
                expected_filename is not None
                and resolved
                == (EXPECTED_FINGERPRINT_ROOT / expected_filename).resolve(),
                f"fingerprint component {name} is not its declared source file",
            )
            _require(
                resolved not in resolved_components,
                f"fingerprint components {resolved_components.get(resolved)!r} "
                f"and {name!r} resolve to one file",
            )
            resolved_components[resolved] = str(name)
            actual = file_sha256(str(path))
        elif "value" in record:
            _exact_fields(record, frozenset({"value", "sha256"}), f"component {name}")
            _require(
                name == "model_revision"
                and record["value"] == FROZEN_MODEL_REVISION,
                f"component {name}.value is not the frozen model revision",
            )
            actual = hashlib.sha256(record["value"].encode("utf-8")).hexdigest()
        else:
            raise W2DVerificationError(
                f"fingerprint component {name} has neither path nor value"
            )
        _require(actual == expected, f"fingerprint component {name} moved")
    aggregate = runtime_fingerprint_digest(components)
    expected_aggregate = _sha256(fp["sha256"], "runtime_fingerprint.sha256")
    _require(aggregate == expected_aggregate, "runtime fingerprint aggregate mismatch")
    return aggregate


def _verify_legacy_regression(
    artifacts: Mapping[str, Any],
    hashes: Mapping[str, str],
    paths: Mapping[str, Path],
) -> None:
    source_hashes_before = {
        role: file_sha256(str(paths[artifact_name]))
        for role, artifact_name in LEGACY_SOURCE_ARTIFACTS.items()
    }
    runtime_sha256_before = legacy_runtime_code_sha256()
    gate_code_before = legacy_gate_code_sha256()
    legacy = _mapping(artifacts["legacy_regression"], "legacy_regression")
    _exact_fields(
        legacy,
        frozenset(
            {
                "schema_version",
                "amendment",
                "source_artifacts",
                "runtime_code_sha256",
                "git_commit",
                "gate_code_sha256",
                "replacement_gate",
                "existing_gate_executions",
            }
        ),
        "legacy_regression",
    )
    _require(
        legacy["schema_version"] == "W2D-legacy-regression-v2",
        "legacy regression schema mismatch",
    )
    _require(
        legacy["amendment"] == "W2D-PREREGISTRATION-AMENDMENT-A5.md",
        "legacy regression does not identify Amendment A5",
    )
    sources = _mapping(
        legacy["source_artifacts"], "legacy_regression.source_artifacts"
    )
    _require(
        set(sources) == set(LEGACY_SOURCE_ARTIFACTS),
        "legacy regression source set changed",
    )
    for role, artifact_name in LEGACY_SOURCE_ARTIFACTS.items():
        record = _mapping(sources[role], f"legacy source {role}")
        _exact_fields(
            record, frozenset({"filename", "sha256"}), f"legacy source {role}"
        )
        _require(
            record["sha256"] == hashes[artifact_name],
            f"legacy regression does not bind {artifact_name}",
        )
        _require(
            record["filename"] == paths[artifact_name].name,
            f"legacy regression filename for {artifact_name} moved",
        )
    _require(
        {
            role: hashes[artifact_name]
            for role, artifact_name in LEGACY_SOURCE_ARTIFACTS.items()
        }
        == FROZEN_SOURCE_SHA256,
        "legacy source hashes differ from Amendment A5",
    )
    _require(
        source_hashes_before == FROZEN_SOURCE_SHA256,
        "legacy source bytes moved before regression verification",
    )
    recomputed = evaluate_legacy_regression(
        authority_manifest=_mapping(
            artifacts["legacy_authority_manifest"],
            "legacy_authority_manifest",
        ),
        authoritative_w2=_mapping(
            artifacts["legacy_authoritative_w2"], "legacy_authoritative_w2"
        ),
        authoritative_w2r=_mapping(
            artifacts["legacy_authoritative_w2r"], "legacy_authoritative_w2r"
        ),
        candidate_w2=_mapping(
            artifacts["legacy_candidate_w2"], "legacy_candidate_w2"
        ),
        candidate_w2r=_mapping(
            artifacts["legacy_candidate_w2r"], "legacy_candidate_w2r"
        ),
        same_code_repeat=_mapping(
            artifacts["legacy_same_code_repeat"], "legacy_same_code_repeat"
        ),
        source_sha256={
            role: hashes[artifact_name]
            for role, artifact_name in LEGACY_SOURCE_ARTIFACTS.items()
        },
        expected_runtime_code_sha256=legacy_runtime_code_sha256(),
    )
    _require(
        legacy["replacement_gate"] == recomputed,
        "legacy replacement gate is not the independent recomputation",
    )
    _require(
        canonical_strict_json_sha256(
            recomputed["same_code_scheduling_differences"]
        )
        == FROZEN_SAME_CODE_DIFFERENCE_SHA256,
        "legacy same-code difference projection is not frozen",
    )
    validate_frozen_same_code_differences(
        recomputed["same_code_scheduling_differences"]
    )
    _require(
        legacy["runtime_code_sha256"] == runtime_sha256_before,
        "legacy regression runtime fingerprint differs from current code",
    )
    _require(
        legacy["gate_code_sha256"] == gate_code_before,
        "legacy regression gate-code hashes differ from current code",
    )
    _require(
        legacy["git_commit"] == FROZEN_CANDIDATE_COMMIT,
        "legacy regression is not bound to the frozen candidate commit",
    )
    _require(
        recomputed.get("passed") is True
        and recomputed.get("literal_A2_6_cross_execution_equality")
        == "INCONCLUSIVE",
        "legacy gate misstates the A2.6 result",
    )
    executions = _sequence(
        legacy["existing_gate_executions"],
        "legacy_regression.existing_gate_executions",
    )
    expected_executions = tuple(
        (
            script,
            LEGACY_SOURCE_ARTIFACTS[source_role]
            if source_role is not None
            else None,
        )
        for script, source_role in LEGACY_GATE_COMMANDS
    )
    _require(
        len(executions) == len(expected_executions),
        "legacy regression did not execute the exact eight existing gates",
    )
    for index, (raw, expected_execution) in enumerate(
        zip(executions, expected_executions)
    ):
        record = _mapping(raw, f"legacy gate execution {index}")
        _exact_fields(
            record,
            frozenset({"command", "returncode", "output_sha256", "passed"}),
            f"legacy gate execution {index}",
        )
        _require(
            record["returncode"] == 0 and record["passed"] is True,
            f"legacy gate execution {index} failed",
        )
        _sha256(
            record["output_sha256"],
            f"legacy gate execution {index}.output_sha256",
        )
        command = _sequence(
            record["command"], f"legacy gate execution {index}.command"
        )
        expected_script, artifact_name = expected_execution
        expected_length = 3 if artifact_name is not None else 2
        _require(
            len(command) == expected_length
            and command[0] == str(Path(sys.executable).resolve())
            and command[1] == expected_script,
            f"legacy gate execution {index} command changed",
        )
        if artifact_name is not None:
            _require(
                Path(str(command[2])).resolve() == paths[artifact_name],
                f"legacy gate execution {index} checked the wrong artifact",
            )
    rerun_legacy_gates(
        {
            role: paths[artifact_name]
            for role, artifact_name in LEGACY_SOURCE_ARTIFACTS.items()
        }
    )
    source_hashes_after = {
        role: file_sha256(str(paths[artifact_name]))
        for role, artifact_name in LEGACY_SOURCE_ARTIFACTS.items()
    }
    _require(
        source_hashes_after == source_hashes_before
        and legacy_runtime_code_sha256() == runtime_sha256_before
        and legacy_gate_code_sha256() == gate_code_before,
        "legacy source/runtime/gate code changed during independent rerun",
    )


def _label_map(value: Any) -> dict[str, Mapping[str, Any]]:
    root = _mapping(value, "labels")
    raw = root.get("items")
    if not isinstance(raw, Mapping):
        raise W2DVerificationError("labels.items must be an opaque-key mapping")
    result = {
        str(key): _mapping(record, f"labels.items.{key}")
        for key, record in raw.items()
    }
    for key in result:
        _item_key(key, f"labels key {key}")
    return result


def _score_map(value: Any, phase: str) -> dict[str, Mapping[str, Any]]:
    root = _mapping(value, f"{phase} scores")
    _require(root.get("schema_version") == "W2D-score-v1",
             f"{phase} score schema mismatch")
    _require(root.get("phase") == phase, f"expected {phase} score phase")
    _require(root.get("measurement_name") == MEASUREMENT_NAME,
             f"{phase} score measurement name mismatch")
    records = _sequence(root.get("items"), f"{phase} scores.items")
    result: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(records):
        record = _mapping(raw, f"{phase} scores.items[{index}]")
        key = _item_key(record.get("item_key"), f"{phase} score item_key")
        _require(key not in result, f"duplicate {phase} score key {key}")
        result[key] = record
    return result


def _verify_score_provenance(
    score: Mapping[str, Any],
    hashes: Mapping[str, str],
    fingerprint: Mapping[str, Any],
    *,
    phase: str,
) -> None:
    _require(score.get("safe_inputs_sha256") == hashes["safe_inputs_json"],
             f"{phase} safe-input digest mismatch")
    _require(score.get("embeddings_sha256") == hashes["safe_inputs_npz"],
             f"{phase} embedding digest mismatch")
    components = _mapping(fingerprint["components"], "fingerprint.components")
    _require(score.get("detector_py_sha256") == components["detector"]["sha256"],
             f"{phase} detector.py digest mismatch")
    _require(score.get("scorer_py_sha256") == components["scorer"]["sha256"],
             f"{phase} scorer.py digest mismatch")
    if phase in {"test", "source_control"}:
        _require(score.get("threshold_artifact_sha256") == hashes["threshold"],
                 f"{phase} threshold digest mismatch")


def _verify_detector_artifacts(
    artifacts: Mapping[str, Any],
    hashes: Mapping[str, str],
    fingerprint_record: Mapping[str, Any],
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    Mapping[str, Any],
]:
    labels = _label_map(artifacts["labels"])
    test_root = _mapping(artifacts["test_scores"], "test_scores")
    control_root = _mapping(artifacts["source_controls"], "source_controls")
    scores = _score_map(test_root, "test")
    controls = _score_map(control_root, "source_control")
    _verify_score_provenance(
        test_root, hashes, fingerprint_record, phase="test"
    )
    _verify_score_provenance(
        control_root, hashes, fingerprint_record, phase="source_control"
    )
    try:
        recomputed_metrics = detector_quality_summary(test_root, artifacts["labels"])
        control_gate = source_control_gate_summary(control_root, artifacts["labels"])
    except MetricsError as exc:
        raise W2DVerificationError(f"detector denominator/control gate failed: {exc}") from exc
    _require(control_gate.get("passed") is True, "source controls did not all refuse")

    metric_artifact = _mapping(artifacts["detector_metrics"], "detector_metrics")
    _required_fields(
        metric_artifact,
        {"schema_version", "scores_sha256", "labels_sha256", "metrics"},
        "detector_metrics",
    )
    _require(metric_artifact["scores_sha256"] == hashes["test_scores"],
             "detector metrics point to a different score artifact")
    _require(metric_artifact["labels_sha256"] == hashes["labels"],
             "detector metrics point to different labels")
    _require(metric_artifact["metrics"] == recomputed_metrics,
             "detector metrics are not the independent recomputation")

    gate_artifact = _mapping(
        artifacts["source_control_gate"], "source_control_gate"
    )
    _exact_fields(
        gate_artifact,
        frozenset({"schema_version", "scores_sha256", "labels_sha256", "gate"}),
        "source_control_gate",
    )
    _require(
        gate_artifact["schema_version"] == "W2D-source-control-gate-v1",
        "source-control gate schema mismatch",
    )
    _require(
        gate_artifact["scores_sha256"] == hashes["source_controls"],
        "source-control gate points to a different score artifact",
    )
    _require(
        gate_artifact["labels_sha256"] == hashes["labels"],
        "source-control gate points to different labels",
    )
    _require(
        gate_artifact["gate"] == control_gate,
        "source-control gate is not the independent recomputation",
    )
    _require(len(scores) == EXPECTED_TEST_ITEMS,
             f"test score denominator is {len(scores)}, expected {EXPECTED_TEST_ITEMS}")
    _require(len(controls) == EXPECTED_SOURCE_CONTROLS,
             f"source controls are {len(controls)}, expected {EXPECTED_SOURCE_CONTROLS}")
    return labels, scores, controls, recomputed_metrics


def _verify_calibration_lineage(
    artifacts: Mapping[str, Any],
    hashes: Mapping[str, str],
    fingerprint_record: Mapping[str, Any],
) -> None:
    calibration = _mapping(artifacts["calibration_scores"], "calibration_scores")
    calibration_map = _score_map(calibration, "calibration")
    _verify_score_provenance(
        calibration, hashes, fingerprint_record, phase="calibration"
    )
    _require(
        len(calibration_map) == 448,
        "calibration score denominator is not 448",
    )
    for key, record in calibration_map.items():
        _exact_fields(
            record, CALIBRATION_SCORE_FIELDS, f"calibration score {key}"
        )
        _strict_bool(
            record["family_s_affirms"],
            f"calibration score {key}.family_s_affirms",
        )
        for field_name in ("rep", "knn", "c_score"):
            value = record[field_name]
            _require(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value)),
                f"calibration score {key}.{field_name} must be finite numeric",
            )
        _require(
            isinstance(record["detector_service_ns"], int)
            and not isinstance(record["detector_service_ns"], bool)
            and record["detector_service_ns"] >= 0,
            f"calibration score {key}.detector_service_ns is invalid",
        )

    threshold = _mapping(artifacts["threshold"], "threshold")
    _required_fields(
        threshold,
        {
            "schema_version",
            "calibration_item_count",
            "calibration_source_group_count",
            "calibration_scores_sha256",
            "labels_sha256",
        },
        "threshold",
    )
    _require(
        threshold["schema_version"] == "W2D-threshold-v1",
        "threshold schema mismatch",
    )
    _require(
        threshold["calibration_item_count"] == 448
        and threshold["calibration_source_group_count"] == 448,
        "threshold was not selected from 448 unique calibration groups",
    )
    _require(
        threshold["calibration_scores_sha256"] == hashes["calibration_scores"],
        "threshold points to different calibration scores",
    )
    _require(
        threshold["labels_sha256"] == hashes["labels"],
        "threshold points to different labels",
    )
    try:
        recomputed = select_threshold(calibration, artifacts["labels"])
    except (CalibrationError, KeyError, TypeError, ValueError) as exc:
        raise W2DVerificationError(
            f"independent calibration recomputation failed: {exc}"
        ) from exc
    published_core = {
        key: value
        for key, value in threshold.items()
        if key not in {"calibration_scores_sha256", "labels_sha256"}
    }
    _require(
        published_core == recomputed,
        "threshold artifact is not the independent calibration recomputation",
    )


def _canonical_json_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise W2DVerificationError(f"value is not strict JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _verify_landing_artifact(
    artifacts: Mapping[str, Any],
    hashes: Mapping[str, str],
    labels: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    landing = _mapping(artifacts["landing"], "landing")
    _exact_fields(landing, LANDING_FIELDS, "landing")
    _require(
        landing["schema_version"] == "W2D-landing-v1",
        "landing schema mismatch",
    )
    _require(
        landing["top_k"] == 5,
        "landing artifact did not use top-5",
    )
    _require(
        landing["protocol_plan_sha256"] == hashes["protocol_plan"],
        "landing artifact points to a different protocol plan",
    )
    _require(
        landing["safe_inputs_sha256"] == hashes["safe_inputs_json"],
        "landing artifact points to different safe inputs",
    )
    _require(
        landing["embeddings_sha256"] == hashes["safe_inputs_npz"],
        "landing artifact points to different embeddings",
    )
    plan = _mapping(artifacts["protocol_plan"], "protocol_plan")
    background = _mapping(
        plan.get("retrieval_background"), "protocol_plan.retrieval_background"
    )
    background_keys = _sequence(
        background.get("item_keys"),
        "protocol_plan.retrieval_background.item_keys",
    )
    _require(
        len(background_keys) == 768
        and len(set(background_keys)) == len(background_keys),
        "retrieval background is not 768 unique items",
    )
    for index, key in enumerate(background_keys):
        _item_key(key, f"retrieval background item {index}")
    _require(
        landing["retrieval_background_sha256"]
        == _canonical_json_sha256(background_keys),
        "landing retrieval-background digest mismatch",
    )

    rows = _sequence(landing["items"], "landing.items")
    _require(len(rows) == 192, "landing artifact does not contain 192 test poisons")
    by_key: dict[str, Mapping[str, Any]] = {}
    counts: dict[str, tuple[int, int]] = {}
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"landing.items[{index}]")
        _exact_fields(row, LANDING_ITEM_FIELDS, f"landing.items[{index}]")
        key = _item_key(row["item_key"], f"landing.items[{index}].item_key")
        _require(key not in by_key, f"landing item {key} is duplicated")
        label = _mapping(labels.get(key), f"landing label {key}")
        _require(label.get("split") == "test", f"landing item {key} is not test")
        _require(
            _strict_bool(
                label.get("poison", label.get("is_poison")),
                f"landing label {key}.poison",
            ),
            f"landing item {key} is not poison",
        )
        family = row["attack_family"]
        _require(
            family in {"recipe", "natural_cover_suffix_v1"}
            and label.get("attack_family") == family,
            f"landing item {key} attack family differs from labels",
        )
        rank = row["rank"]
        _require(
            isinstance(rank, int)
            and not isinstance(rank, bool)
            and rank > 0
            and row["candidate_rank"] == rank,
            f"landing item {key} has an invalid rank",
        )
        landed = _strict_bool(row["landed_top5"], f"landing item {key}.landed_top5")
        _require(
            _strict_bool(row["landed"], f"landing item {key}.landed") == landed
            and landed == (rank <= 5),
            f"landing item {key} landed flag disagrees with rank",
        )
        top5 = _sequence(row["top5_item_keys"], f"landing item {key}.top5_item_keys")
        _require(
            len(top5) == 5 and len(set(top5)) == 5,
            f"landing item {key} does not report five unique top-5 keys",
        )
        for position, candidate in enumerate(top5):
            _item_key(candidate, f"landing item {key}.top5[{position}]")
        total, hits = counts.get(str(family), (0, 0))
        counts[str(family)] = (total + 1, hits + int(landed))
        by_key[key] = row

    expected_counts = {
        "recipe": EXPECTED_RECIPE_TEST,
        "natural_cover_suffix_v1": EXPECTED_NATURAL_COVER,
    }
    populations = _mapping(landing["populations"], "landing.populations")
    _require(
        set(populations) == set(expected_counts),
        "landing populations are not the exact two attack families",
    )
    for family, expected_n in expected_counts.items():
        total, hits = counts.get(family, (0, 0))
        _require(
            total == expected_n,
            f"landing {family} denominator is {total}, expected {expected_n}",
        )
        expected_summary = {
            **rate_record(hits, expected_n),
            "landed_n": hits,
            "n": expected_n,
        }
        _require(
            populations[family] == expected_summary,
            f"landing {family} summary is not recomputed from item rows",
        )

    landing_plan = _mapping(plan.get("landing_plan"), "protocol_plan.landing_plan")
    planned_rows = _sequence(
        landing_plan.get("records"), "protocol_plan.landing_plan.records"
    )
    planned_keys: set[str] = set()
    for index, raw in enumerate(planned_rows):
        if isinstance(raw, str):
            key = raw
        else:
            record = _mapping(raw, f"landing plan record {index}")
            key = record.get("item_key", record.get("poison_item_key"))
        planned_keys.add(_item_key(key, f"landing plan record {index}.item_key"))
    _require(
        len(planned_rows) == len(planned_keys) == 192
        and planned_keys == set(by_key),
        "landing artifact keys differ from the frozen landing plan",
    )
    return by_key


def _verify_plan(
    plan: Any,
    labels: Mapping[str, Mapping[str, Any]],
    landing_keys: set[str],
) -> tuple[
    dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]
]:
    root = _mapping(plan, "protocol_plan")
    units_raw = _sequence(root.get("runtime_units"), "protocol_plan.runtime_units")
    cells_raw = _sequence(root.get("cells"), "protocol_plan.cells")
    _require(len(units_raw) == EXPECTED_RUNTIME_UNITS,
             f"protocol plan has {len(units_raw)} runtime units, expected 20")
    _require(len(cells_raw) == EXPECTED_PROTOCOL_CELLS,
             f"protocol plan has {len(cells_raw)} cells, expected 140")
    units: dict[str, Mapping[str, Any]] = {}
    poison_by_family: dict[str, set[str]] = {
        "recipe": set(),
        "natural_cover_suffix_v1": set(),
    }
    poison_by_family_seed: dict[tuple[str, int], dict[str, set[str]]] = {}
    role_vectors: dict[
        tuple[str, int, str], dict[str, tuple[str, ...]]
    ] = {}
    for index, raw in enumerate(units_raw):
        unit = _mapping(raw, f"runtime_units[{index}]")
        _required_fields(
            unit,
            {
                "runtime_unit_id",
                "seed",
                "attack_family",
                "backlog",
                "poison_item_keys",
                "clean_item_keys",
                "filler_item_keys",
                "query_poison_item_keys",
                "injection_sequence",
            },
            f"runtime_units[{index}]",
        )
        uid = str(unit["runtime_unit_id"])
        _require(uid and uid not in units, f"duplicate/empty runtime_unit_id {uid!r}")
        sequence = _sequence(
            unit["injection_sequence"], f"runtime_unit {uid}.injection_sequence"
        )
        seen: set[str] = set()
        unit_poison: set[str] = set()
        ordered_by_role: dict[str, list[str]] = {
            "poison": [],
            "clean": [],
            "filler": [],
        }
        for position, event_raw in enumerate(sequence):
            event = _mapping(event_raw, f"runtime_unit {uid}.injection[{position}]")
            _required_fields(
                event, {"injection_ordinal", "item_key", "role"},
                f"runtime_unit {uid}.injection[{position}]",
            )
            key = _item_key(event["item_key"], f"runtime_unit {uid} item_key")
            _require(key not in seen, f"runtime unit {uid} repeats item {key}")
            seen.add(key)
            _require(
                event["injection_ordinal"] == position,
                f"runtime unit {uid} injection ordinal is not dense",
            )
            _require(event["role"] in {"poison", "clean", "filler"},
                     f"runtime unit {uid} has invalid role {event['role']!r}")
            ordered_by_role[str(event["role"])].append(key)
            label = _mapping(labels.get(key), f"runtime unit {uid} label {key}")
            truth = _strict_bool(
                label.get("poison", label.get("is_poison")),
                f"runtime unit {uid} label {key}.poison",
            )
            if event["role"] == "poison":
                _require(truth, f"runtime unit {uid} marks clean item {key} as poison")
                family = str(unit["attack_family"])
                _require(
                    family in poison_by_family
                    and label.get("attack_family") == family,
                    f"runtime unit {uid} poison {key} has the wrong attack family",
                )
                _require(
                    key in landing_keys,
                    f"runtime unit {uid} poison {key} has no landing record",
                )
                unit_poison.add(key)
                poison_by_family[family].add(key)
            else:
                _require(
                    not truth,
                    f"runtime unit {uid} marks poison item {key} as {event['role']}",
                )
                _require(
                    label.get("stratum")
                    in {"ordinary_clean", "hard_negative_clean"},
                    f"runtime unit {uid} clean/filler {key} has invalid stratum",
                )
        frozen_vectors: dict[str, list[str]] = {}
        for role, field_name in (
            ("poison", "poison_item_keys"),
            ("clean", "clean_item_keys"),
            ("filler", "filler_item_keys"),
        ):
            vector = _sequence(
                unit[field_name], f"runtime unit {uid}.{field_name}"
            )
            keys = [
                _item_key(value, f"runtime unit {uid}.{field_name}[{offset}]")
                for offset, value in enumerate(vector)
            ]
            _require(
                len(keys) == len(set(keys)) and keys == ordered_by_role[role],
                f"runtime unit {uid} {field_name} differs from injection sequence",
            )
            frozen_vectors[role] = keys
        _require(
            sequence
            == sorted(
                sequence,
                key=lambda event: (
                    {"filler": 0, "poison": 1, "clean": 2}[event["role"]],
                    event["injection_ordinal"],
                ),
            ),
            f"runtime unit {uid} injection order is not filler/poison/clean",
        )
        query_poison = _sequence(
            unit["query_poison_item_keys"],
            f"runtime unit {uid}.query_poison_item_keys",
        )
        _require(
            query_poison == frozen_vectors["poison"],
            f"runtime unit {uid} query poison order differs from injection",
        )
        _require(
            len(unit_poison) == len(frozen_vectors["poison"]) == 6,
            f"runtime unit {uid} does not contain exactly six poison items",
        )
        seed = unit["seed"]
        _require(
            isinstance(seed, int) and not isinstance(seed, bool),
            f"runtime unit {uid} seed must be an integer",
        )
        _require(seed in EXPECTED_SEEDS, f"runtime unit {uid} has invalid seed {seed}")
        backlog = str(unit["backlog"])
        _require(
            backlog in {"normal", "heavy"},
            f"runtime unit {uid} has invalid backlog {backlog!r}",
        )
        expected_filler_n = 12 if backlog == "heavy" else 0
        _require(
            len(frozen_vectors["clean"]) == 6
            and len(frozen_vectors["filler"]) == expected_filler_n,
            f"runtime unit {uid} does not have 6 clean/{expected_filler_n} filler",
        )
        clean_strata = Counter(
            str(_mapping(labels[key], f"label {key}")["stratum"])
            for key in frozen_vectors["clean"]
        )
        filler_strata = Counter(
            str(_mapping(labels[key], f"label {key}")["stratum"])
            for key in frozen_vectors["filler"]
        )
        _require(
            dict(clean_strata) == EXPECTED_CLEAN_STRATA[seed],
            f"runtime unit {uid} clean strata differ from A4",
        )
        expected_filler_strata = (
            EXPECTED_FILLER_STRATA[seed] if backlog == "heavy" else {}
        )
        _require(
            dict(filler_strata) == expected_filler_strata,
            f"runtime unit {uid} filler strata differ from A4",
        )
        family = str(unit["attack_family"])
        role_key = (family, seed, backlog)
        _require(role_key not in role_vectors, f"duplicate runtime tuple {role_key}")
        role_vectors[role_key] = {
            role: tuple(values) for role, values in frozen_vectors.items()
        }
        family_seed = (family, seed)
        by_backlog = poison_by_family_seed.setdefault(family_seed, {})
        _require(
            backlog not in by_backlog,
            f"runtime unit pair {family_seed} repeats backlog {backlog}",
        )
        by_backlog[backlog] = unit_poison
        units[uid] = unit

    for family, keys in poison_by_family.items():
        _require(
            len(keys) == EXPECTED_RUNTIME_POISON_PER_FAMILY,
            f"runtime plan has {len(keys)} unique {family} poisons, expected 30",
        )
    for family_seed, by_backlog in poison_by_family_seed.items():
        _require(
            set(by_backlog) == {"normal", "heavy"},
            f"runtime pair {family_seed} lacks normal/heavy backlog",
        )
        _require(
            by_backlog["normal"] == by_backlog["heavy"],
            f"runtime pair {family_seed} changes poison items across backlog",
        )
    for family in poison_by_family:
        family_seeds = {
            seed
            for (candidate_family, seed) in poison_by_family_seed
            if candidate_family == family
        }
        _require(
            family_seeds == EXPECTED_SEEDS,
            f"runtime plan seeds for {family} are {sorted(family_seeds)}",
        )

    expected_role_keys = {
        (family, seed, backlog)
        for family in poison_by_family
        for seed in EXPECTED_SEEDS
        for backlog in ("normal", "heavy")
    }
    _require(
        set(role_vectors) == expected_role_keys,
        "runtime plan does not contain the exact attack/seed/backlog product",
    )
    all_clean_by_seed: dict[int, set[str]] = {}
    all_filler_by_seed: dict[int, set[str]] = {}
    for seed in EXPECTED_SEEDS:
        recipe_normal = role_vectors[("recipe", seed, "normal")]
        recipe_heavy = role_vectors[("recipe", seed, "heavy")]
        natural_normal = role_vectors[
            ("natural_cover_suffix_v1", seed, "normal")
        ]
        natural_heavy = role_vectors[
            ("natural_cover_suffix_v1", seed, "heavy")
        ]
        clean_vectors = {
            recipe_normal["clean"],
            recipe_heavy["clean"],
            natural_normal["clean"],
            natural_heavy["clean"],
        }
        _require(
            len(clean_vectors) == 1,
            f"seed {seed} clean set/order changes across attack or backlog",
        )
        _require(
            recipe_normal["filler"] == natural_normal["filler"] == (),
            f"seed {seed} normal backlog unexpectedly has filler",
        )
        _require(
            recipe_heavy["filler"] == natural_heavy["filler"],
            f"seed {seed} heavy filler changes across attack",
        )
        all_clean_by_seed[seed] = set(recipe_normal["clean"])
        all_filler_by_seed[seed] = set(recipe_heavy["filler"])

    for left in EXPECTED_SEEDS:
        _require(
            not (all_clean_by_seed[left] & all_filler_by_seed[left]),
            f"seed {left} reuses an item as clean and filler",
        )
        for right in EXPECTED_SEEDS:
            if left >= right:
                continue
            _require(
                not (all_clean_by_seed[left] & all_clean_by_seed[right]),
                f"seeds {left}/{right} reuse injected clean items",
            )
            _require(
                not (all_filler_by_seed[left] & all_filler_by_seed[right]),
                f"seeds {left}/{right} reuse heavy filler items",
            )
            _require(
                not (all_clean_by_seed[left] & all_filler_by_seed[right])
                and not (all_filler_by_seed[left] & all_clean_by_seed[right]),
                f"seeds {left}/{right} cross-reuse clean/filler items",
            )

    cells: dict[str, Mapping[str, Any]] = {}
    per_unit: dict[str, list[tuple[str, str]]] = {}
    for index, raw in enumerate(cells_raw):
        cell = _mapping(raw, f"protocol_plan.cells[{index}]")
        _required_fields(
            cell,
            {
                "cell_id",
                "runtime_unit_id",
                "seed",
                "attack_family",
                "backlog",
                "baseline",
                "arm",
                "service_replay",
            },
            f"protocol_plan.cells[{index}]",
        )
        cid = str(cell["cell_id"])
        uid = str(cell["runtime_unit_id"])
        _require(cid and cid not in cells, f"duplicate/empty plan cell id {cid!r}")
        _require(uid in units, f"plan cell {cid} has unknown runtime unit {uid!r}")
        baseline, arm = cell["baseline"], cell["arm"]
        _require(baseline in {"B1", "B2", "B3", "B4"},
                 f"plan cell {cid} has invalid baseline")
        if baseline == "B1":
            _require(arm == "control", f"B1 plan cell {cid} must be control arm")
        else:
            _require(arm in {"detector", "oracle"},
                     f"plan cell {cid} has invalid arm {arm!r}")
            _require(cell["service_replay"] == FROZEN_SERVICE_REPLAY,
                     f"plan cell {cid} does not freeze item service times")
        unit = units[uid]
        for field in ("seed", "attack_family", "backlog"):
            _require(cell[field] == unit[field],
                     f"plan cell {cid} {field} differs from runtime unit")
        per_unit.setdefault(uid, []).append((baseline, arm))
        cells[cid] = cell
    expected = {
        ("B1", "control"),
        ("B2", "detector"), ("B3", "detector"), ("B4", "detector"),
        ("B2", "oracle"), ("B3", "oracle"), ("B4", "oracle"),
    }
    for uid, memberships in per_unit.items():
        _require(
            len(memberships) == len(expected) and set(memberships) == expected,
            f"runtime unit {uid} does not have the exact seven planned cells",
        )
    return units, cells


def _protocol_items(
    cell: Mapping[str, Any],
    unit: Mapping[str, Any],
    scores: Mapping[str, Mapping[str, Any]],
    path: str,
) -> list[Mapping[str, Any]]:
    items_raw = _sequence(cell["protocol_items"], f"{path}.protocol_items")
    sequence = _sequence(unit["injection_sequence"], f"{path}.plan_sequence")
    _require(len(items_raw) == len(sequence),
             f"{path} protocol item count differs from its runtime unit")
    items: list[Mapping[str, Any]] = []
    for index, (raw, event_raw) in enumerate(zip(items_raw, sequence)):
        item = _mapping(raw, f"{path}.protocol_items[{index}]")
        event = _mapping(event_raw, f"{path}.plan_sequence[{index}]")
        _exact_fields(item, PROTOCOL_ITEM_FIELDS, f"{path}.protocol_items[{index}]")
        key = _item_key(item["item_key"], f"{path}.protocol item key")
        _require(key in scores, f"{path} item {key} has no unique score-once record")
        _require(item["role"] == event["role"]
                 and item["ordinal"] == event["injection_ordinal"]
                 and key == event["item_key"],
                 f"{path} protocol item {index} differs from the frozen plan")
        score = scores[key]
        promote = _strict_bool(item["detector_promote"],
                               f"{path}.protocol_items[{index}].detector_promote")
        _require(promote == score["promote"],
                 f"{path} item {key} decision differs from score-once record")
        latency = item["detector_service_ns"]
        _require(
            isinstance(latency, int) and not isinstance(latency, bool) and latency >= 0,
            f"{path} item {key} detector_service_ns must be non-negative int",
        )
        _require(latency == score["detector_service_ns"],
                 f"{path} item {key} service time differs from score-once record")
        _require(
            item["score_record_sha256"] == canonical_record_sha256(score),
            f"{path} item {key} score-record digest mismatch",
        )
        items.append(item)
    return items


def _verify_provider_records(
    cell: Mapping[str, Any],
    item_map: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    path: str,
) -> dict[str, Mapping[str, Any]]:
    records = _sequence(cell["verifier_records"], f"{path}.verifier_records")
    if cell["baseline"] == "B1":
        _require(not records, f"{path} B1 control must not run a verifier")
        return {}
    by_key: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(records):
        record = _mapping(raw, f"{path}.verifier_records[{index}]")
        _exact_fields(
            record,
            VERIFIER_RECORD_FIELDS,
            f"{path}.verifier_records[{index}]",
        )
        key = _item_key(record["item_key"], f"{path}.verifier_records[{index}].item_key")
        _require(key not in by_key, f"{path} repeats verifier record {key}")
        _require(key in item_map, f"{path} verifier record {key} is not planned")
        status = record["status"]
        _require(status in PROVIDER_STATUSES,
                 f"{path} provider status {status!r} is not terminal/legal")
        if record["passes"] is not None:
            passes = _strict_bool(record["passes"], f"{path} record {key}.passes")
            if cell["arm"] == "detector":
                _require(passes == item_map[key]["detector_promote"],
                         f"{path} record {key} replayed a different decision")
            else:
                label = _mapping(labels.get(key), f"{path} label {key}")
                truth = _strict_bool(
                    label.get("poison", label.get("is_poison")),
                    f"{path} label {key}.poison",
                )
                _require(
                    passes is (not truth),
                    f"{path} oracle record {key} differs from evaluator truth",
                )
        if record["service_time_s"] is not None:
            service = _nonnegative_number(
                record["service_time_s"], f"{path} record {key}.service_time_s"
            )
            expected = item_map[key]["detector_service_ns"] / 1_000_000_000
            _require(
                math.isclose(service, expected, rel_tol=0.0, abs_tol=1e-12),
                f"{path} record {key} replayed a different service time",
            )
        queue_enter = _nonnegative_number(
            record["queue_enter_s"], f"{path} record {key}.queue_enter_s"
        )
        depth_at_enqueue = record["queue_depth_at_enqueue"]
        _require(
            isinstance(depth_at_enqueue, int)
            and not isinstance(depth_at_enqueue, bool)
            and depth_at_enqueue >= 0,
            f"{path} record {key}.queue_depth_at_enqueue is invalid",
        )
        queue_start = _nonnegative_number(
            record["queue_start_s"],
            f"{path} record {key}.queue_start_s",
            allow_none=True,
        )
        queue_wait = _nonnegative_number(
            record["queue_wait_s"],
            f"{path} record {key}.queue_wait_s",
            allow_none=True,
        )
        depth_at_start = record["queue_depth_at_start"]
        if queue_start is None:
            _require(
                queue_wait is None and depth_at_start is None,
                f"{path} record {key} has worker-start fields without a start",
            )
        else:
            _require(
                isinstance(depth_at_start, int)
                and not isinstance(depth_at_start, bool)
                and depth_at_start >= 0,
                f"{path} record {key}.queue_depth_at_start is invalid",
            )
            _require(
                queue_start >= queue_enter
                and math.isclose(
                    float(queue_wait),
                    queue_start - queue_enter,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ),
                f"{path} record {key} queue timing does not balance",
            )
        if status == "COMMITTED":
            _strict_bool(record["passes"], f"{path} committed record {key}.passes")
            _nonnegative_number(
                record["service_time_s"], f"{path} committed record {key}.service_time_s"
            )
            integrated = _nonnegative_number(
                record["integrated_latency_s"],
                f"{path} committed record {key}.integrated_latency_s",
            )
            commit = _nonnegative_number(
                record["decision_commit_s"],
                f"{path} committed record {key}.decision_commit_s",
            )
            _require(
                queue_start is not None
                and commit >= queue_start
                and math.isclose(
                    float(integrated),
                    commit - queue_enter,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ),
                f"{path} committed record {key} integrated timing does not balance",
            )
        else:
            _require(record["decision_commit_s"] is None,
                     f"{path} cancelled record {key} claims a commit")
            _nonnegative_number(
                record["integrated_latency_s"],
                f"{path} cancelled record {key}.integrated_latency_s",
                allow_none=True,
            )
        by_key[key] = record
    _require(set(by_key) == set(item_map),
             f"{path} verifier records do not exactly cover protocol items")
    return by_key


def _verify_lifecycle(
    cell: Mapping[str, Any],
    item_map: Mapping[str, Mapping[str, Any]],
    provider_records: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    horizon: float,
    path: str,
) -> None:
    rows = _sequence(cell["lifecycle"], f"{path}.lifecycle")
    by_key: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"{path}.lifecycle[{index}]")
        _exact_fields(row, LIFECYCLE_FIELDS, f"{path}.lifecycle[{index}]")
        key = _item_key(row["item_key"], f"{path}.lifecycle[{index}].item_key")
        _require(key not in by_key and key in item_map,
                 f"{path} lifecycle key {key} is duplicate or unplanned")
        label = _mapping(labels.get(key), f"label {key}")
        truth = label.get("poison")
        if truth is None:
            truth = label.get("is_poison")
        truth = _strict_bool(truth, f"label {key}.poison")
        promote = _strict_bool(
            row["detector_promote"], f"{path}.lifecycle[{index}].detector_promote"
        )
        _require(_strict_bool(row["truth_poison"], f"{path} lifecycle truth") == truth,
                 f"{path} lifecycle truth for {key} differs from evaluator label")
        _require(promote == item_map[key]["detector_promote"],
                 f"{path} lifecycle decision for {key} differs from score")
        _require(
            isinstance(row["retry_count"], int)
            and not isinstance(row["retry_count"], bool)
            and row["retry_count"] == 0,
            f"{path} item {key} violates the no-retry design",
        )
        _require(
            isinstance(row["readmission_count"], int)
            and not isinstance(row["readmission_count"], bool)
            and row["readmission_count"] >= 0,
            f"{path} item {key} has invalid readmission_count",
        )
        _require(row["state_at_horizon"] in FINAL_STATES,
                 f"{path} item {key} has invalid horizon state")
        _require(row["visibility_status"] in {"NOT_STARTED", "STARTED"},
                 f"{path} item {key} has invalid visibility status")
        for field_name in (
            "exposure_status",
            "unvetted_visibility_status",
            "unavailable_status",
            "quarantine_status",
        ):
            _require(row[field_name] in EPISODE_STATUSES,
                     f"{path} item {key} has invalid {field_name}")
        _nonnegative_number(
            row["exposure_observed_s"], f"{path} item {key}.exposure_observed_s",
            allow_none=True,
        )
        unvetted_observed = _nonnegative_number(
            row["unvetted_visibility_observed_s"],
            f"{path} item {key}.unvetted_visibility_observed_s",
            allow_none=True,
        )
        if row["unvetted_visibility_status"] == "NOT_STARTED":
            _require(
                unvetted_observed is None,
                f"{path} item {key} assigns time to an unstarted E_u episode",
            )
        else:
            _require(
                unvetted_observed is not None,
                f"{path} item {key} omits observed E_u time",
            )
        _nonnegative_number(
            row["unavailable_observed_s"], f"{path} item {key}.unavailable_observed_s",
            allow_none=True,
        )
        observed = _nonnegative_number(
            row["observed_at_horizon_s"], f"{path} item {key}.observed_at_horizon_s"
        )
        _require(math.isclose(observed, horizon, rel_tol=0.0, abs_tol=1e-9),
                 f"{path} item {key} was not accounted at the fixed horizon")
        if cell["baseline"] == "B2":
            _require(
                row["unvetted_visibility_status"] == "NOT_STARTED",
                f"{path} B2 item {key} started an unvetted-visible episode",
            )
        elif cell["baseline"] == "B1":
            _require(
                row["unvetted_visibility_status"] == "RIGHT_CENSORED",
                f"{path} B1 item {key} did not retain its open E_u episode",
            )
        elif cell["baseline"] == "B4":
            _require(
                row["unvetted_visibility_status"] == "COMPLETED"
                and float(unvetted_observed) <= 1.02,
                f"{path} B4 item {key} violates E_u <= 1.02 s",
            )

        decision_committed = (
            key in provider_records
            and provider_records[key]["status"] == "COMMITTED"
        )
        if (
            cell["arm"] == "detector"
            and decision_committed
            and not truth
            and not promote
        ):
            # Clean false positive: one terminal misquarantine, no retry or
            # readmission, with the unavailable/quarantine episode censored at
            # the fixed horizon.
            _require(row["state_at_horizon"] == "QUARANTINED",
                     f"{path} clean false positive {key} is not quarantined")
            _require(row["quarantine_status"] == "RIGHT_CENSORED"
                     and row["unavailable_status"] == "RIGHT_CENSORED",
                     f"{path} clean false positive {key} lost censoring")
            _require(row["readmission_count"] == 0,
                     f"{path} clean false positive {key} was retried/readmitted")
            expected_visibility = (
                "NOT_STARTED" if cell["baseline"] == "B2" else "STARTED"
            )
            _require(row["visibility_status"] == expected_visibility,
                     f"{path} clean false-positive visibility is inconsistent")

        if (
            cell["arm"] == "detector"
            and decision_committed
            and truth
            and promote
        ):
            # Poison false negative: the decision can commit before or after a
            # B4 hide, but after any readmission it remains exposed at horizon.
            _require(row["state_at_horizon"] == "TRUSTED"
                     and row["visibility_status"] == "STARTED",
                     f"{path} poison false negative {key} is not visible/trusted")
            _require(row["exposure_status"] == "RIGHT_CENSORED",
                     f"{path} poison false negative {key} was called contained")
            age = _nonnegative_number(
                row["exposure_observed_s"],
                f"{path} poison false negative {key}.exposure_observed_s",
            )
            _require(age <= horizon,
                     f"{path} poison false negative {key} exceeds the horizon")
            _require(row["quarantine_status"] == "NOT_STARTED",
                     f"{path} poison false negative {key} also claims quarantine")
        by_key[key] = row
    _require(set(by_key) == set(item_map),
             f"{path} lifecycle does not exactly cover protocol items")


def _verify_result(
    result: Any,
    artifacts: Mapping[str, Any],
    hashes: Mapping[str, str],
    fingerprint_sha256: str,
    labels: Mapping[str, Mapping[str, Any]],
    scores: Mapping[str, Mapping[str, Any]],
    landing_items: Mapping[str, Mapping[str, Any]],
    detector_metrics: Mapping[str, Any],
    units: Mapping[str, Mapping[str, Any]],
    plan_cells: Mapping[str, Mapping[str, Any]],
) -> None:
    root = _mapping(result, "results")
    _exact_fields(root, RESULT_FIELDS, "results")
    _require(root["schema_version"] == RESULT_SCHEMA, "result schema mismatch")
    _require(root["measurement_name"] == MEASUREMENT_NAME,
             "result measurement name is not the preregistered replay name")
    _require(root["execution_mode"] == EXECUTION_MODE,
             "result execution mode could be mistaken for online detector latency")
    _require(root["runtime_fingerprint_sha256"] == fingerprint_sha256,
             "result points to a different runtime fingerprint")
    result_hashes = _mapping(root["artifact_hashes"], "results.artifact_hashes")
    _require(set(result_hashes) == set(hashes),
             "result artifact hash set differs from the manifest")
    for name, expected in hashes.items():
        _require(result_hashes[name] == expected,
                 f"result artifact hash {name} differs from the manifest")
    execution = _mapping(root["detector_execution"], "results.detector_execution")
    _exact_fields(
        execution,
        frozenset({"protocol_provider", "detector_invocations_during_replay"}),
        "results.detector_execution",
    )
    _require(execution["protocol_provider"] == FROZEN_PROVIDER,
             "protocol provider is not the frozen score lookup")
    _require(
        isinstance(execution["detector_invocations_during_replay"], int)
        and not isinstance(execution["detector_invocations_during_replay"], bool)
        and execution["detector_invocations_during_replay"] == 0,
        "live detector execution occurred during replay",
    )
    horizon = _nonnegative_number(
        root["observation_horizon_s"], "results.observation_horizon_s"
    )
    injection_at = _nonnegative_number(root["injection_at_s"], "results.injection_at_s")
    _require(horizon == 8.0 and injection_at == 1.0 and injection_at < horizon,
             "result does not use the frozen 8 s horizon / 1 s injection")

    denominator = _mapping(root["detector_denominator"], "results.detector_denominator")
    _exact_fields(
        denominator,
        frozenset(
            {
                "score_once_item_count",
                "unique_source_group_count",
                "reported_detector_quality_denominator",
                "protocol_replay_observation_count",
            }
        ),
        "results.detector_denominator",
    )
    expected_items = int(detector_metrics["unique_item_count"])
    expected_groups = int(detector_metrics["unique_source_group_count"])
    _require(
        denominator["score_once_item_count"] == expected_items
        and denominator["unique_source_group_count"] == expected_groups
        and denominator["reported_detector_quality_denominator"] == expected_items,
        "detector denominator expanded or differs from the unique score-once set",
    )

    cells_raw = _sequence(root["cells"], "results.cells")
    _require(len(cells_raw) == len(plan_cells),
             "result cell count differs from the protocol plan")
    result_cells: dict[str, Mapping[str, Any]] = {}
    replay_vectors: dict[
        tuple[str, str, int, str], dict[str, tuple[tuple[Any, ...], ...]]
    ] = {}
    replay_observations = 0
    for index, raw in enumerate(cells_raw):
        cell = _mapping(raw, f"results.cells[{index}]")
        _exact_fields(cell, CELL_FIELDS, f"results.cells[{index}]")
        cid = str(cell["plan_cell_id"])
        _require(cid in plan_cells and cid not in result_cells,
                 f"result cell {cid!r} is absent from plan or duplicated")
        planned = plan_cells[cid]
        uid = str(planned["runtime_unit_id"])
        unit = units[uid]
        for field in ("seed", "attack_family", "backlog", "baseline", "arm"):
            _require(cell[field] == planned[field],
                     f"result cell {cid} {field} differs from plan")
        path = f"result cell {cid}"
        items = _protocol_items(cell, unit, scores, path)
        item_map = {str(item["item_key"]): item for item in items}
        _require(len(item_map) == len(items), f"{path} repeats a protocol item")
        provider_records = _verify_provider_records(
            cell, item_map, labels, path
        )
        _verify_lifecycle(
            cell, item_map, provider_records, labels, horizon, path
        )
        retrieval = _mapping(cell["retrieval"], f"{path}.retrieval")
        if cell["baseline"] == "B1":
            _required_fields(
                retrieval,
                {"attack_landing_numerator", "attack_landing_denominator"},
                f"{path}.retrieval",
            )
            numerator = retrieval["attack_landing_numerator"]
            denominator_value = retrieval["attack_landing_denominator"]
            _require(
                isinstance(numerator, int)
                and not isinstance(numerator, bool)
                and isinstance(denominator_value, int)
                and not isinstance(denominator_value, bool)
                and 0 <= numerator <= denominator_value,
                f"{path} has invalid B1 attack-landing counts",
            )
            poison_keys = [
                str(event["item_key"])
                for event in _sequence(
                    unit["injection_sequence"], f"{path}.plan_sequence"
                )
                if _mapping(event, f"{path}.plan event").get("role") == "poison"
            ]
            expected_landed = sum(
                int(
                    _strict_bool(
                        _mapping(
                            landing_items.get(key), f"{path} landing item {key}"
                        ).get("landed_top5"),
                        f"{path} landing item {key}.landed_top5",
                    )
                )
                for key in poison_keys
            )
            _require(
                denominator_value == len(poison_keys) == 6
                and numerator == expected_landed,
                f"{path} B1 landing counts differ from six frozen landing rows",
            )
        if cell["baseline"] != "B1":
            pair = (
                str(cell["attack_family"]),
                str(cell["backlog"]),
                int(cell["seed"]),
                str(cell["arm"]),
            )
            vector = tuple(
                (
                    item["item_key"],
                    item["role"],
                    item["ordinal"],
                    item["detector_promote"],
                    item["detector_service_ns"],
                    item["score_record_sha256"],
                )
                for item in items
            )
            replay_vectors.setdefault(pair, {})[str(cell["baseline"])] = vector
        if cell["arm"] == "detector":
            replay_observations += len(items)
        result_cells[cid] = cell
    _require(set(result_cells) == set(plan_cells),
             "result cells do not exactly equal the protocol-plan cells")
    for pair, by_baseline in replay_vectors.items():
        _require(set(by_baseline) == {"B2", "B3", "B4"},
                 f"replay pair {pair} lacks B2/B3/B4")
        _require(by_baseline["B2"] == by_baseline["B3"] == by_baseline["B4"],
                 f"replay pair {pair} changed decision/service vectors")
    _require(
        denominator["protocol_replay_observation_count"] == replay_observations,
        "reported replay observation count does not equal cell records",
    )
    expected_family_counts = {
        "natural_cover_suffix_v1": EXPECTED_NATURAL_COVER,
        "recipe": EXPECTED_RECIPE_TEST,
    }
    for family, expected in expected_family_counts.items():
        label_keys = {
            key
            for key, label in labels.items()
            if label.get("split") == "test"
            and label.get("attack_family") == family
            and label.get("poison", label.get("is_poison")) is True
        }
        _require(
            len(label_keys) == expected,
            f"labels retain {len(label_keys)} {family} test poisons, expected {expected}",
        )


def verify_bundle(
    manifest_path: str | Path, results_path: str | Path
) -> VerificationReport:
    """Verify the complete W2D bundle without mutating it."""
    report = VerificationReport()
    try:
        manifest_file = Path(manifest_path)
        try:
            manifest = strict_json_load(str(manifest_file))
            result = strict_json_load(str(results_path))
        except (MetricsError, json.JSONDecodeError, OSError) as exc:
            raise W2DVerificationError(f"strict JSON load failed: {exc}") from exc
        manifest = _mapping(manifest, "manifest")
        _exact_fields(
            manifest,
            frozenset(
                {
                    "schema_version",
                    "measurement_name",
                    "artifact_hashes",
                    "runtime_fingerprint",
                }
            ),
            "manifest",
        )
        _require(manifest["schema_version"] == MANIFEST_SCHEMA,
                 "manifest schema mismatch")
        _require(manifest["measurement_name"] == MEASUREMENT_NAME,
                 "manifest measurement name mismatch")
        artifacts, hashes, artifact_paths = _load_artifacts(
            manifest_file, manifest
        )
        checksum_line = artifact_paths["labels_checksum"].read_text(
            encoding="ascii"
        ).strip()
        _require(
            checksum_line
            == f"{hashes['labels']}  {artifact_paths['labels'].name}",
            "label checksum sidecar does not bind evaluator labels",
        )
        report.passed("strict JSON and every formal artifact hash")
        verify_a6_artifact_lineage(artifacts, hashes, artifact_paths)
        report.passed(
            "A6 quota recomputation, allocation invariants, and data lineage"
        )
        _verify_legacy_regression(artifacts, hashes, artifact_paths)
        report.passed(
            "legacy oracle replacement gate and same-code boundary disclosure"
        )
        fingerprint_sha = _verify_fingerprint(manifest_file, manifest)
        report.passed("independent runtime fingerprint and component hashes")
        fingerprint_record = _mapping(
            manifest["runtime_fingerprint"], "runtime_fingerprint"
        )
        _verify_calibration_lineage(artifacts, hashes, fingerprint_record)
        report.passed("calibration threshold recomputation and artifact lineage")
        labels, scores, _controls, detector_metrics = _verify_detector_artifacts(
            artifacts, hashes, fingerprint_record
        )
        report.passed("unique score-once denominator and 24 source controls")
        landing_items = _verify_landing_artifact(artifacts, hashes, labels)
        report.passed("192-item landing artifact and retrieval-background lineage")
        units, plan_cells = _verify_plan(
            artifacts["protocol_plan"], labels, set(landing_items)
        )
        report.passed("20 runtime units and 140 frozen protocol cells")
        _verify_result(
            result,
            artifacts,
            hashes,
            fingerprint_sha,
            labels,
            scores,
            landing_items,
            detector_metrics,
            units,
            plan_cells,
        )
        report.passed(
            "paired replay, score back-references, provider states, and FP/FN lifecycle"
        )
    except (W2DVerificationError, MetricsError, KeyError, TypeError, ValueError) as exc:
        report.failed(str(exc))
    return report


def verify_or_raise(manifest_path: str | Path, results_path: str | Path) -> None:
    report = verify_bundle(manifest_path, results_path)
    if not report.ok:
        raise W2DVerificationError("; ".join(report.failures))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args(argv)
    report = verify_bundle(args.manifest, args.results)
    for name in report.checks:
        print(f"  [PASS] {name}")
    for failure in report.failures:
        print(f"  [FAIL] {failure}")
    print(
        "\n==== verify_w2d: "
        + ("ALL GATES PASSED" if report.ok else "FAILURES ABOVE")
        + " ===="
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
