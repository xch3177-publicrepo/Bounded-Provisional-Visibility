#!/usr/bin/env python3
"""Independent, fail-loud gates for a formal W2D replay bundle.

This verifier intentionally consumes only frozen artifacts plus the final
protocol result.  It never builds data, scores an item, or runs a protocol cell.
The runner may import the public constants and :func:`verify_bundle` before
publishing a result.

Usage:
    ./.venv312/bin/python verify_w2d.py \
        --manifest results/w2d/W2D-E1-MANIFEST.json \
        --results results/w2d/W2D-E1-INMEMORY.json

The sibling ``W2D-MANIFEST-AUTHORITY.json`` is discovered deterministically.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
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
    strict_json_load_bytes,
)
from w2d_snapshot import (
    FileSnapshot,
    SnapshotError,
    SnapshotRegistry,
)
from w2d_backend_evidence import (
    BackendEvidenceError,
    summarize_expected_seed_state,
    validate_backend_evidence_pair,
)
from w2d_equivalence import validate_a9_replay_equivalence
from verify_w2d_legacy import evaluate as evaluate_legacy_regression
from verify_w2d_legacy import (
    FROZEN_CANDIDATE_COMMIT,
    FROZEN_SAME_CODE_DIFFERENCE_SHA256,
    FROZEN_SOURCE_SHA256,
    LEGACY_GATE_COMMANDS,
    LEGACY_GATE_FILES,
    LEGACY_RUNTIME_FILES,
    canonical_strict_json_sha256,
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
RESULT_SCHEMA = "W2D-protocol-result-v3"
MANIFEST_SCHEMA = "W2D-manifest-v2"
MANIFEST_AUTHORITY_SCHEMA = "W2D-manifest-authority-v1"
MANIFEST_AUTHORITY_BASENAME = "W2D-MANIFEST-AUTHORITY.json"
CANONICAL_MANIFEST_AUTHORITY_PATH = (
    HERE / "results" / "w2d" / MANIFEST_AUTHORITY_BASENAME
).resolve()
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
        "protocol_plan_pre_a9",
        "protocol_plan",
        "calibration_scores",
        "threshold",
        "test_scores",
        "source_controls",
        "source_control_gate",
        "detector_metrics",
        "landing_pre_a9",
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
    "protocol_plan_pre_a9": (
        "results/w2d/W2D-PROTOCOL-PLAN-preA7-landing-contract-failure.json"
    ),
    "protocol_plan": "results/w2d/W2D-PROTOCOL-PLAN.json",
    "calibration_scores": "results/w2d/W2D-calibration-scores.json",
    "threshold": "results/w2d/W2D-threshold.json",
    "test_scores": "results/w2d/W2D-test-scores.json",
    "source_controls": "results/w2d/W2D-source-controls.json",
    "source_control_gate": "results/w2d/W2D-source-control-gate.json",
    "detector_metrics": "results/w2d/W2D-detector-metrics.json",
    "landing_pre_a9": "results/w2d/W2D-LANDING-preA9-lineage.json",
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
    "snapshot_registry": "w2d_snapshot.py",
    "backend_evidence": "w2d_backend_evidence.py",
    "a9_equivalence": "w2d_equivalence.py",
    "preregistration": "W2D-PREREGISTRATION.md",
    "amendment_a1": "W2D-PREREGISTRATION-AMENDMENT-A1.md",
    "amendment_a2": "W2D-PREREGISTRATION-AMENDMENT-A2.md",
    "amendment_a3": "W2D-PREREGISTRATION-AMENDMENT-A3.md",
    "amendment_a4": "W2D-PREREGISTRATION-AMENDMENT-A4.md",
    "amendment_a5": "W2D-PREREGISTRATION-AMENDMENT-A5.md",
    "amendment_a6": "W2D-PREREGISTRATION-AMENDMENT-A6.md",
    "amendment_a7": "W2D-PREREGISTRATION-AMENDMENT-A7.md",
    "amendment_a8": "W2D-PREREGISTRATION-AMENDMENT-A8.md",
    "amendment_a9": "W2D-PREREGISTRATION-AMENDMENT-A9.md",
    "amendment_a10": "W2D-PREREGISTRATION-AMENDMENT-A10.md",
    "amendment_a11": "W2D-PREREGISTRATION-AMENDMENT-A11.md",
    "amendment_a12": "W2D-PREREGISTRATION-AMENDMENT-A12.md",
    "amendment_a13": "W2D-PREREGISTRATION-AMENDMENT-A13.md",
    "amendment_a14": "W2D-PREREGISTRATION-AMENDMENT-A14.md",
    "amendment_a15": "W2D-PREREGISTRATION-AMENDMENT-A15.md",
    "a15_query_schedule_test": "test_w2d_query_schedule.py",
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
        "manifest_sha256",
        "authority_sha256",
        "artifact_hashes",
        "detector_execution",
        "runtime_backend",
        "runtime_backend_evidence",
        "observation_horizon_s",
        "injection_at_s",
        "detector_denominator",
        "cells",
    }
)
RUNTIME_BACKEND_FIELDS = frozenset(
    {
        "backend",
        "implementation",
        "evidence_level",
        "deployment_mode",
        "vector_dim",
        "metric",
        "index_type_requested",
        "index_type_effective",
        "consistency_level",
        "pymilvus_version",
        "milvus_lite_version",
    }
)
RUNTIME_BACKEND_COMBINATIONS = (
    {
        "backend": "inmemory",
        "implementation": "InMemoryBackend",
        "evidence_level": "exact_in_memory",
        "deployment_mode": "process_local",
        "vector_dim": 384,
        "metric": "COSINE",
        "index_type_requested": "exact_bruteforce",
        "index_type_effective": "exact_bruteforce",
        "consistency_level": "strong_in_process",
        "pymilvus_version": None,
        "milvus_lite_version": None,
    },
    {
        "backend": "milvus_lite",
        "implementation": "MilvusBackend",
        "evidence_level": "milvus_lite_flat",
        "deployment_mode": "lite",
        "vector_dim": 384,
        "metric": "COSINE",
        "index_type_requested": "FLAT",
        "index_type_effective": "FLAT",
        "consistency_level": "Strong",
        "pymilvus_version": "2.4.15",
        "milvus_lite_version": "2.4.12",
    },
)
EXECUTION_TIER_CONTRACTS = {
    "E1_INMEMORY": {
        "runner_backend_name": "inmemory",
        "manifest_filename": "W2D-E1-MANIFEST.json",
        "result_filename": "W2D-E1-INMEMORY.json",
        "runtime_backend": RUNTIME_BACKEND_COMBINATIONS[0],
    },
    "E2_MILVUS_LITE": {
        "runner_backend_name": "milvus",
        "manifest_filename": "W2D-E2-MANIFEST.json",
        "result_filename": "W2D-E2-MILVUS.json",
        "runtime_backend": RUNTIME_BACKEND_COMBINATIONS[1],
    },
}
RUNNER_BACKEND_TO_TIER = {
    contract["runner_backend_name"]: tier
    for tier, contract in EXECUTION_TIER_CONTRACTS.items()
}
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
        "arrival_s",
        "transitions",
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
LIFECYCLE_TRANSITION_FIELDS = frozenset(
    {"t_s", "old_state", "new_state"}
)
RETRIEVAL_FIELDS = frozenset(
    {
        "events",
        "overall",
        "landed_only",
        "attack_landing_numerator",
        "attack_landing_denominator",
    }
)
RETRIEVAL_EVENT_FIELDS = frozenset(
    {
        "query_started_s",
        "t_s",
        "query_role",
        "query_ordinal",
        "returned_top5_item_keys",
        "eligible_top5_item_keys",
        "poisonfree_top5_item_keys",
        "poison_item_keys",
        "landed_poison_item_keys",
        "hit_overall",
        "hit_landed_only",
        "eligible_recall_at_5",
        "poisonfree_displacement_at_5",
    }
)
RETRIEVAL_SUMMARY_FIELDS = frozenset(
    {
        "attack_associated",
        "heldout_same_topic",
        "negative_other_topic",
    }
)
RETRIEVAL_ROLE_SUMMARY_FIELDS = frozenset(
    {
        "poisoned_retrieval_count",
        "query_count",
        "poisoned_retrieval_rate",
        "cumulative_displaced_positions",
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


LEGACY_GATE_METRICS_FILENAME = "w2d_metrics.py"
LEGACY_GATE_METRICS_ARCHIVED_SHA256 = (
    "27b775e40904354f46d3237420820f926667ffcaa91da1572dbd87cde839025d"
)
LEGACY_GATE_METRICS_A6_SHA256 = (
    "d379f132da17eb966f32cc04c6fe5cf31b697a097d56515a264031950f91fc3b"
)
LEGACY_GATE_RUNTIME_TEST_FILENAME = "test_w2d_runtime.py"
LEGACY_GATE_RUNTIME_TEST_ARCHIVED_SHA256 = (
    "d53c2861b7774f4e4d6dafa7aa467764dd215d3519bffe1b30a42ab8c4a72fc6"
)
LEGACY_GATE_RUNTIME_TEST_A12_SHA256 = (
    "05fd48acc80adc6411bf4861ba1a4d58c8a7ae5d52d1ece34e355f80c941d9b4"
)
LEGACY_GATE_REGRESSION_TEST_FILENAME = "test_verify_w2d_legacy.py"
LEGACY_GATE_REGRESSION_TEST_ARCHIVED_SHA256 = (
    "0a07113434630db2fafee4ef12c4f1a5e5792b882eab589173d1f6c555561ed2"
)
LEGACY_GATE_REGRESSION_TEST_A12_SHA256 = (
    "5733c1d3b13cf53d5654fdcfa3abf4a9d751dfa54de6af2729d7a064de8ae888"
)
LEGACY_RUNTIME_ARCHIVED_SHA256 = (
    "f403c4e58c2526a70ada0dbaaa1d22c5cead42fb97b0e888ff70da9ad3a429e7"
)
LEGACY_RUNTIME_A11_SHA256 = (
    "3a599541311c27c524b0add29461534272b4e84b62a7cfa1becc8cfca1509fea"
)
LEGACY_RUNTIME_ARCHIVED_FILE_SHA256 = {
    "poison_exposure.py": (
        "7b08cd02814f6bc3e9f475f0234de29f7371365df5b524378c1b52bf7e4a4ff7"
    ),
    "functional_slice.py": (
        "f6baee5e1e1f90c52650c656ed5236fac8398dcd40f619ca11cb864fbd094a4b"
    ),
    "backend.py": (
        "f9343e8df7de9afc7884d6a673aad6fec821d45ab9be2f2099c93a4d82269e28"
    ),
    "realtext_workload.py": (
        "f718e7b5c3173fb62cca850f6b3e994007051247fb863962d6b1bfc6c75fa9e6"
    ),
    "analysis.py": (
        "51bbc2a89eaac05e7479925cba3efac79921eabf40963f090eda3f551714eae1"
    ),
    "milvus_backend.py": (
        "b516103deecfd31b5e7b6673621e139c9b18bf2e607d57fc75c5d2678bb500a5"
    ),
}
LEGACY_RUNTIME_A11_FILE_SHA256 = {
    **LEGACY_RUNTIME_ARCHIVED_FILE_SHA256,
    "poison_exposure.py": (
        "94a778d266b94a2aa1496792794fc131e0f2dc8137b95032f359e52c33a2df37"
    ),
}
LEGACY_RUNTIME_A15_SHA256 = (
    "46f7b421e80f554830ac4a21b1789b8ef8cfae1b5f4c7cb81e5d311b03e2aeee"
)
LEGACY_RUNTIME_A15_FILE_SHA256 = {
    **LEGACY_RUNTIME_A11_FILE_SHA256,
    "poison_exposure.py": (
        "2f7db9f18af2a210c96a54088fcf17de93557332a54c6489c92346c5890d870a"
    ),
}


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


@dataclass(frozen=True)
class VerifiedBundleSnapshot:
    """The exact byte instances consumed by one successful verification.

    The parsed documents, their hashes, and every later analysis value all
    derive from these snapshots.  Live paths are revisited only by the
    registry's explicit drift check, never as a second source of scientific
    values.
    """

    registry: SnapshotRegistry
    authority_path: Path
    manifest_path: Path
    results_path: Path
    authority_snapshot: FileSnapshot
    manifest_snapshot: FileSnapshot
    result_snapshot: FileSnapshot
    authority: Mapping[str, Any]
    manifest: Mapping[str, Any]
    result: Mapping[str, Any]
    artifacts: Mapping[str, Any]
    artifact_hashes: Mapping[str, str]
    artifact_paths: Mapping[str, Path]
    artifact_snapshots: Mapping[str, FileSnapshot]
    component_snapshots: Mapping[str, FileSnapshot]


@dataclass(frozen=True)
class ManifestAuthorityBinding:
    """One captured authority/manifest pair and its selected execution tier."""

    registry: SnapshotRegistry
    authority_path: Path
    manifest_path: Path
    authority_snapshot: FileSnapshot
    manifest_snapshot: FileSnapshot
    authority: Mapping[str, Any]
    manifest: Mapping[str, Any]
    execution_tier: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise W2DVerificationError(message)


def validate_legacy_gate_code_compatibility(
    archived: Mapping[str, Any],
    current: Mapping[str, Any],
) -> None:
    """Validate the exact A8/A12 legacy-gate code transitions."""

    if not isinstance(archived, Mapping) or not isinstance(current, Mapping):
        raise W2DVerificationError(
            "legacy gate-code ledgers must both be mappings"
        )
    archived_keys = set(archived)
    current_keys = set(current)
    _require(
        archived_keys == current_keys,
        "legacy gate-code file set differs from the archived execution",
    )
    _require(
        LEGACY_GATE_METRICS_FILENAME in archived_keys,
        "legacy gate-code ledger omits w2d_metrics.py",
    )
    for filename in sorted(archived_keys):
        if not isinstance(filename, str):
            raise W2DVerificationError(
                "legacy gate-code filenames must be strings"
            )
        archived_sha256 = _sha256(
            archived[filename], f"legacy gate-code archived.{filename}"
        )
        current_sha256 = _sha256(
            current[filename], f"legacy gate-code current.{filename}"
        )
        if filename == LEGACY_GATE_METRICS_FILENAME:
            _require(
                archived_sha256 == LEGACY_GATE_METRICS_ARCHIVED_SHA256
                and current_sha256 == LEGACY_GATE_METRICS_A6_SHA256,
                "legacy w2d_metrics.py hash is not the exact A8 "
                "archived-to-A6 transition",
            )
        elif filename == LEGACY_GATE_RUNTIME_TEST_FILENAME:
            _require(
                archived_sha256
                == LEGACY_GATE_RUNTIME_TEST_ARCHIVED_SHA256
                and current_sha256 == LEGACY_GATE_RUNTIME_TEST_A12_SHA256,
                "legacy test_w2d_runtime.py hash is not the exact A12 "
                "raw-evidence transition",
            )
        elif filename == LEGACY_GATE_REGRESSION_TEST_FILENAME:
            _require(
                archived_sha256
                == LEGACY_GATE_REGRESSION_TEST_ARCHIVED_SHA256
                and current_sha256
                == LEGACY_GATE_REGRESSION_TEST_A12_SHA256,
                "legacy test_verify_w2d_legacy.py hash is not the exact A12 "
                "archived-runtime transition",
            )
        else:
            _require(
                archived_sha256 == current_sha256,
                f"legacy gate-code file changed outside A8/A12: {filename}",
            )


def validate_legacy_runtime_code_compatibility(
    current_sha256: Any,
    current_files: Mapping[str, Any],
) -> None:
    """Accept only A15's exact runtime ledger and its registered A11 delta."""

    digest = _sha256(current_sha256, "current legacy runtime digest")
    if not isinstance(current_files, Mapping):
        raise W2DVerificationError(
            "current legacy runtime file ledger must be a mapping"
        )
    normalized = {
        filename: _sha256(
            current_files.get(filename),
            f"current legacy runtime.{filename}",
        )
        for filename in LEGACY_RUNTIME_A15_FILE_SHA256
    }
    _require(
        set(current_files) == set(LEGACY_RUNTIME_A15_FILE_SHA256),
        "current legacy runtime file set differs from A15",
    )
    _require(
        digest == LEGACY_RUNTIME_A15_SHA256
        and normalized == LEGACY_RUNTIME_A15_FILE_SHA256,
        "current legacy runtime differs outside the exact A15 "
        "poison_exposure schedule transition",
    )
    _require(
        {
            filename
            for filename in LEGACY_RUNTIME_A11_FILE_SHA256
            if LEGACY_RUNTIME_A11_FILE_SHA256[filename]
            != LEGACY_RUNTIME_A15_FILE_SHA256[filename]
        }
        == {"poison_exposure.py"},
        "registered A11-to-A15 legacy-runtime transition is not single-file",
    )


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


def _strict_snapshot_json(snapshot: FileSnapshot, name: str) -> Any:
    try:
        return strict_json_load_bytes(
            snapshot.payload,
            source=str(snapshot.path),
        )
    except (MetricsError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise W2DVerificationError(
            f"{name} is not strict JSON: {exc}"
        ) from exc


def _verify_canonical_authority_at_head(
    authority_snapshot: FileSnapshot,
) -> None:
    """Require the canonical repository authority to equal its HEAD bytes.

    Temporary fixtures use the same basename in another directory and remain
    portable.  Only the one canonical repository path carries the committed
    trust-anchor requirement.
    """

    if authority_snapshot.path != CANONICAL_MANIFEST_AUTHORITY_PATH:
        return
    repository = HERE.parent
    try:
        relative = authority_snapshot.path.relative_to(repository).as_posix()
        head_payload = subprocess.check_output(
            ["git", "show", f"HEAD:{relative}"],
            cwd=repository,
            stderr=subprocess.DEVNULL,
        )
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--", relative],
            cwd=repository,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise W2DVerificationError(
            "canonical manifest authority is not committed at HEAD"
        ) from exc
    _require(
        head_payload == authority_snapshot.payload and not status.strip(),
        "canonical manifest authority differs from its committed HEAD bytes",
    )


def _validate_manifest_authority_documents(
    *,
    authority_path: Path,
    manifest_path: Path,
    authority_snapshot: FileSnapshot,
    manifest_snapshot: FileSnapshot,
    authority: Mapping[str, Any],
    manifest: Mapping[str, Any],
    check_head: bool,
) -> str:
    _require(
        authority_path.name == MANIFEST_AUTHORITY_BASENAME,
        "manifest authority does not use its canonical basename",
    )
    _require(
        authority_path.parent == manifest_path.parent,
        "manifest authority and selected manifest are not siblings",
    )
    _exact_fields(
        authority,
        frozenset(
            {
                "schema_version",
                "measurement_name",
                "manifests",
            }
        ),
        "manifest_authority",
    )
    _require(
        authority["schema_version"] == MANIFEST_AUTHORITY_SCHEMA,
        "manifest authority schema mismatch",
    )
    _require(
        authority["measurement_name"] == MEASUREMENT_NAME,
        "manifest authority measurement name mismatch",
    )
    entries = _mapping(authority["manifests"], "manifest_authority.manifests")
    _require(
        set(entries) == set(EXECUTION_TIER_CONTRACTS),
        "manifest authority must bind exactly every preregistered execution tier",
    )
    execution_tier = manifest.get("execution_tier")
    _require(
        type(execution_tier) is str
        and execution_tier in EXECUTION_TIER_CONTRACTS,
        "manifest execution tier is not preregistered",
    )
    for tier, contract in EXECUTION_TIER_CONTRACTS.items():
        candidate = _mapping(
            entries[tier],
            f"manifest_authority.manifests.{tier}",
        )
        _exact_fields(
            candidate,
            frozenset({"path", "sha256"}),
            f"manifest_authority.manifests.{tier}",
        )
        _require(
            candidate["path"] == contract["manifest_filename"],
            f"manifest authority entry {tier} does not use its canonical basename",
        )
        _sha256(
            candidate["sha256"],
            f"manifest_authority.manifests.{tier}.sha256",
        )
    entry = _mapping(
        entries[execution_tier],
        f"manifest_authority.manifests.{execution_tier}",
    )
    _exact_fields(
        entry,
        frozenset({"path", "sha256"}),
        f"manifest_authority.manifests.{execution_tier}",
    )
    tier_contract = EXECUTION_TIER_CONTRACTS[execution_tier]
    expected_manifest_filename = tier_contract["manifest_filename"]
    _require(
        entry["path"] == expected_manifest_filename,
        "manifest authority entry does not use the tier's canonical manifest basename",
    )
    _require(
        manifest_path.name == expected_manifest_filename,
        "selected manifest does not use the execution tier's canonical basename",
    )
    _require(
        (authority_path.parent / entry["path"]).resolve() == manifest_path,
        "manifest authority entry resolves to a different manifest",
    )
    _require(
        _sha256(entry["sha256"], "manifest_authority manifest sha256")
        == manifest_snapshot.sha256,
        "manifest authority sha256 differs from the captured manifest bytes",
    )
    if check_head:
        _verify_canonical_authority_at_head(authority_snapshot)
    return execution_tier


def _capture_authority_manifest_set(
    *,
    authority_path: Path,
    authority: Mapping[str, Any],
    registry: SnapshotRegistry,
) -> None:
    """Capture and identity-check every manifest named by one authority.

    The selected manifest is not enough: the authority is a two-tier
    pre-run commitment, so verification must also fail if its unselected
    sibling is absent, changed, or relabelled.
    """

    entries = _mapping(
        authority["manifests"], "manifest_authority.manifests"
    )
    manifest_fields = frozenset(
        {
            "schema_version",
            "measurement_name",
            "execution_tier",
            "expected_result_filename",
            "expected_runtime_backend",
            "artifact_hashes",
            "runtime_fingerprint",
        }
    )
    for tier, contract in EXECUTION_TIER_CONTRACTS.items():
        entry = _mapping(
            entries[tier],
            f"manifest_authority.manifests.{tier}",
        )
        manifest_path = (
            authority_path.parent / contract["manifest_filename"]
        ).resolve()
        snapshot = registry.capture(manifest_path)
        _require(
            snapshot.sha256
            == _sha256(
                entry["sha256"],
                f"manifest_authority.manifests.{tier}.sha256",
            ),
            f"manifest authority sha256 differs from captured {tier} bytes",
        )
        document = _mapping(
            _strict_snapshot_json(snapshot, f"{tier} manifest"),
            f"manifest_authority.{tier}_manifest",
        )
        _exact_fields(
            document,
            manifest_fields,
            f"manifest_authority.{tier}_manifest",
        )
        _require(
            document["schema_version"] == MANIFEST_SCHEMA
            and document["measurement_name"] == MEASUREMENT_NAME
            and document["execution_tier"] == tier
            and document["expected_result_filename"]
            == contract["result_filename"],
            f"manifest authority {tier} identity changed",
        )
        expected_backend = _verified_runtime_backend_descriptor(
            document["expected_runtime_backend"],
            f"manifest_authority.{tier}_manifest.expected_runtime_backend",
        )
        _require(
            expected_backend == contract["runtime_backend"],
            f"manifest authority {tier} backend identity changed",
        )


def capture_manifest_authority_binding(
    manifest_path: str | Path,
    *,
    registry: SnapshotRegistry | None = None,
) -> ManifestAuthorityBinding:
    """Capture and validate one authority→manifest byte chain.

    The authority location is deterministic: it is the canonical authority
    basename beside the tier-specific manifest.  This keeps copied test bundles
    portable without letting a caller substitute a second authority path.
    """

    active_registry = registry if registry is not None else SnapshotRegistry()
    manifest_file = Path(manifest_path).resolve()
    authority_file = manifest_file.parent / MANIFEST_AUTHORITY_BASENAME
    authority_snapshot = active_registry.capture(authority_file)
    manifest_snapshot = active_registry.capture(manifest_file)
    _require(
        authority_snapshot.path != manifest_snapshot.path,
        "manifest authority and manifest resolve to one formal input path",
    )
    authority = _mapping(
        _strict_snapshot_json(authority_snapshot, "manifest authority"),
        "manifest_authority",
    )
    manifest = _mapping(
        _strict_snapshot_json(manifest_snapshot, "manifest"),
        "manifest",
    )
    execution_tier = _validate_manifest_authority_documents(
        authority_path=authority_snapshot.path,
        manifest_path=manifest_snapshot.path,
        authority_snapshot=authority_snapshot,
        manifest_snapshot=manifest_snapshot,
        authority=authority,
        manifest=manifest,
        check_head=True,
    )
    _capture_authority_manifest_set(
        authority_path=authority_snapshot.path,
        authority=authority,
        registry=active_registry,
    )
    return ManifestAuthorityBinding(
        registry=active_registry,
        authority_path=authority_snapshot.path,
        manifest_path=manifest_snapshot.path,
        authority_snapshot=authority_snapshot,
        manifest_snapshot=manifest_snapshot,
        authority=authority,
        manifest=manifest,
        execution_tier=execution_tier,
    )


def _load_artifacts(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    registry: SnapshotRegistry,
    occupied_paths: Mapping[Path, str],
) -> tuple[
    dict[str, Any],
    dict[str, str],
    dict[str, Path],
    dict[str, FileSnapshot],
]:
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
    snapshots: dict[str, FileSnapshot] = {}
    resolved_roles = dict(occupied_paths)
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
            f"formal inputs {resolved_roles.get(resolved)!r} and "
            f"artifact {name!r} "
            "resolve to the same path",
        )
        resolved_roles[resolved] = f"artifact:{name}"
        snapshot = registry.capture(resolved)
        actual = snapshot.sha256
        _require(actual == expected, f"artifact {name} sha256 {actual} != {expected}")
        hashes[str(name)] = actual
        paths[str(name)] = resolved
        snapshots[str(name)] = snapshot
        if path.suffix.lower() == ".json":
            loaded[str(name)] = _strict_snapshot_json(
                snapshot, f"artifact {name}"
            )
        else:
            # Non-JSON formal values are consumed from ``snapshots``.  Keep
            # the resolved path here only for filename/role checks that do not
            # read its contents.
            loaded[str(name)] = resolved
    return loaded, hashes, paths, snapshots


def _load_component_snapshots(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    registry: SnapshotRegistry,
    occupied_paths: Mapping[Path, str],
) -> dict[str, FileSnapshot]:
    fingerprint = _mapping(
        manifest.get("runtime_fingerprint"),
        "manifest.runtime_fingerprint",
    )
    components = _mapping(
        fingerprint.get("components"),
        "runtime_fingerprint.components",
    )
    _require(
        set(components) == REQUIRED_FINGERPRINT_COMPONENTS,
        "runtime fingerprint component set mismatch; "
        f"missing={sorted(REQUIRED_FINGERPRINT_COMPONENTS - set(components))}, "
        f"extra={sorted(set(components) - REQUIRED_FINGERPRINT_COMPONENTS)}",
    )
    result: dict[str, FileSnapshot] = {}
    resolved_roles = dict(occupied_paths)
    for name, raw in components.items():
        record = _mapping(raw, f"fingerprint component {name}")
        has_path = "path" in record
        has_value = "value" in record
        _require(
            has_path ^ has_value,
            f"fingerprint component {name} needs exactly path or value",
        )
        if has_value:
            continue
        path = _resolve(
            manifest_path.parent,
            record["path"],
            f"fingerprint component {name}.path",
        )
        resolved = path.resolve()
        expected_filename = FINGERPRINT_FILE_COMPONENTS.get(str(name))
        _require(
            expected_filename is not None
            and resolved
            == (EXPECTED_FINGERPRINT_ROOT / expected_filename).resolve(),
            f"fingerprint component {name} is not its declared source file",
        )
        _require(
            resolved not in resolved_roles,
            f"formal inputs {resolved_roles.get(resolved)!r} and "
            f"fingerprint component {name!r} resolve to one file",
        )
        resolved_roles[resolved] = f"component:{name}"
        result[str(name)] = registry.capture(resolved)
    return result


def verify_a6_artifact_lineage(
    artifacts: Mapping[str, Any],
    hashes: Mapping[str, str],
    paths: Mapping[str, Path],
    *,
    artifact_snapshots: Mapping[str, FileSnapshot] | None = None,
    component_snapshots: Mapping[str, FileSnapshot] | None = None,
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
    if component_snapshots is None:
        workload_sha256 = file_sha256(
            str(EXPECTED_FINGERPRINT_ROOT / "w2d_workload.py")
        )
        amendment_a6_sha256 = file_sha256(
            str(
                EXPECTED_FINGERPRINT_ROOT
                / "W2D-PREREGISTRATION-AMENDMENT-A6.md"
            )
        )
    else:
        workload_sha256 = component_snapshots["workload"].sha256
        amendment_a6_sha256 = component_snapshots["amendment_a6"].sha256
    _require(
        provenance["w2d_workload_py_sha256"]
        == workload_sha256
        and provenance["amendment_a6_sha256"]
        == amendment_a6_sha256,
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

        npz_source: Any = paths["safe_inputs_npz"]
        if artifact_snapshots is not None:
            npz_source = io.BytesIO(
                artifact_snapshots["safe_inputs_npz"].payload
            )
        with np.load(npz_source, allow_pickle=False) as archive:
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


def _verify_fingerprint(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    component_snapshots: Mapping[str, FileSnapshot],
) -> str:
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
            snapshot = component_snapshots.get(str(name))
            _require(
                snapshot is not None and snapshot.path == resolved,
                f"fingerprint component {name} lacks its captured byte instance",
            )
            actual = snapshot.sha256
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


def _component_snapshot_by_filename(
    component_snapshots: Mapping[str, FileSnapshot],
    filename: str,
) -> FileSnapshot:
    matches = [
        snapshot
        for name, snapshot in component_snapshots.items()
        if FINGERPRINT_FILE_COMPONENTS.get(name) == filename
    ]
    _require(
        len(matches) == 1,
        f"runtime fingerprint lacks one byte snapshot for {filename}",
    )
    return matches[0]


def _legacy_runtime_sha256_from_snapshots(
    component_snapshots: Mapping[str, FileSnapshot],
) -> str:
    digest = hashlib.sha256()
    for filename in sorted(LEGACY_RUNTIME_FILES):
        snapshot = _component_snapshot_by_filename(
            component_snapshots, filename
        )
        digest.update(filename.encode("utf-8"))
        digest.update(snapshot.payload)
    return digest.hexdigest()


def _legacy_runtime_file_sha256_from_snapshots(
    component_snapshots: Mapping[str, FileSnapshot],
) -> dict[str, str]:
    return {
        filename: _component_snapshot_by_filename(
            component_snapshots, filename
        ).sha256
        for filename in LEGACY_RUNTIME_FILES
    }


def _legacy_gate_sha256_from_snapshots(
    component_snapshots: Mapping[str, FileSnapshot],
) -> dict[str, str]:
    return {
        filename: _component_snapshot_by_filename(
            component_snapshots, filename
        ).sha256
        for filename in LEGACY_GATE_FILES
    }


def _verify_legacy_regression(
    artifacts: Mapping[str, Any],
    hashes: Mapping[str, str],
    paths: Mapping[str, Path],
    artifact_snapshots: Mapping[str, FileSnapshot],
    component_snapshots: Mapping[str, FileSnapshot],
    registry: SnapshotRegistry,
) -> None:
    source_hashes_before = {
        role: artifact_snapshots[artifact_name].sha256
        for role, artifact_name in LEGACY_SOURCE_ARTIFACTS.items()
    }
    current_runtime_sha256 = _legacy_runtime_sha256_from_snapshots(
        component_snapshots
    )
    current_runtime_files = _legacy_runtime_file_sha256_from_snapshots(
        component_snapshots
    )
    validate_legacy_runtime_code_compatibility(
        current_runtime_sha256,
        current_runtime_files,
    )
    gate_code_before = _legacy_gate_sha256_from_snapshots(
        component_snapshots
    )
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
        expected_runtime_code_sha256=LEGACY_RUNTIME_ARCHIVED_SHA256,
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
        legacy["runtime_code_sha256"] == LEGACY_RUNTIME_ARCHIVED_SHA256,
        "legacy regression runtime fingerprint differs from its archived "
        "candidate code",
    )
    validate_legacy_gate_code_compatibility(
        _mapping(
            legacy["gate_code_sha256"],
            "legacy_regression.gate_code_sha256",
        ),
        gate_code_before,
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
    # The legacy command-line gates are path-only historical programs.  Their
    # scientific projection above consumes only captured bytes.  Bracket the
    # unavoidable path-based subprocess compatibility check with the complete
    # registry drift gate so a persistent substitution cannot pass.
    registry.assert_unchanged()
    rerun_legacy_gates(
        {
            role: paths[artifact_name]
            for role, artifact_name in LEGACY_SOURCE_ARTIFACTS.items()
        }
    )
    registry.assert_unchanged()


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
        _require(
            queue_enter <= 8.0,
            f"{path} record {key} entered the queue after the fixed horizon",
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
                and queue_start <= 8.0
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
                and commit <= 8.0
                and math.isclose(
                    float(integrated),
                    commit - queue_enter,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ),
                f"{path} committed record {key} integrated timing does not balance",
            )
            worker_service = commit - queue_start
            _require(
                worker_service + 1e-6 >= float(record["service_time_s"])
                and worker_service <= 8.0,
                f"{path} committed record {key} worker service does not "
                "cover the frozen service within the fixed horizon",
            )
            _require(
                math.isclose(
                    float(integrated),
                    float(record["queue_wait_s"]) + worker_service,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ),
                f"{path} committed record {key} queue/service closure fails",
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


def _top5_keys(
    value: Any,
    *,
    allowed_keys: set[str],
    path: str,
) -> list[str]:
    rows = _sequence(value, path)
    _require(len(rows) == 5, f"{path} must contain exactly five keys")
    keys = [_item_key(raw, f"{path}[{index}]") for index, raw in enumerate(rows)]
    _require(len(set(keys)) == 5, f"{path} contains duplicate keys")
    _require(set(keys) <= allowed_keys, f"{path} contains a key outside this cell")
    return keys


def _verify_rate_record(
    raw: Any, expected: Mapping[str, Any], path: str
) -> None:
    record = _mapping(raw, path)
    _exact_fields(
        record,
        frozenset({"defined", "numerator", "denominator", "rate", "wilson95"}),
        path,
    )
    _strict_bool(record["defined"], f"{path}.defined")
    for field_name in ("numerator", "denominator"):
        _require(
            type(record[field_name]) is int,
            f"{path}.{field_name} must be an integer",
        )
    _require(record == expected, f"{path} does not recompute from its events")


def _verify_retrieval(
    *,
    cell: Mapping[str, Any],
    item_map: Mapping[str, Mapping[str, Any]],
    lifecycle_rows: Mapping[str, Mapping[str, Any]],
    landing_items: Mapping[str, Mapping[str, Any]],
    background_keys: set[str],
    horizon: float,
    path: str,
) -> None:
    retrieval = _mapping(cell["retrieval"], f"{path}.retrieval")
    _exact_fields(retrieval, RETRIEVAL_FIELDS, f"{path}.retrieval")
    events = _sequence(retrieval["events"], f"{path}.retrieval.events")
    _require(
        len(events) == 191,
        f"{path} retrieval schedule must contain exactly 191 events",
    )
    poison_keys = {
        key for key, item in item_map.items() if item["role"] == "poison"
    }
    _require(len(poison_keys) == 6, f"{path} does not have six poison items")
    allowed_keys = set(background_keys) | set(item_map)
    expected_roles = (
        "attack_associated",
        "heldout_same_topic",
        "negative_other_topic",
    )

    def visible_during_query(
        key: str,
        started: float,
        completed: float,
    ) -> bool:
        """Whether the validated control-state ledger permits this returned key."""

        if key in background_keys:
            return True
        row = lifecycle_rows[key]
        interval_end = min(completed, horizon)
        cursor = float(row["arrival_s"])
        state: str | None = None
        for transition in row["transitions"]:
            transition_time = float(transition["t_s"])
            if state in {"PROVISIONAL", "TRUSTED"}:
                if max(started, cursor) <= min(interval_end, transition_time) + 1e-9:
                    return True
            state = str(transition["new_state"])
            cursor = transition_time
        return (
            state in {"PROVISIONAL", "TRUSTED"}
            and max(started, cursor) <= interval_end + 1e-9
        )

    recomputed: list[dict[str, Any]] = []
    previous_completed = 0.0
    for position, raw in enumerate(events):
        event_path = f"{path}.retrieval.events[{position}]"
        event = _mapping(raw, event_path)
        _exact_fields(event, RETRIEVAL_EVENT_FIELDS, event_path)
        started = _nonnegative_number(
            event["query_started_s"], f"{event_path}.query_started_s"
        )
        completed = _nonnegative_number(event["t_s"], f"{event_path}.t_s")
        _require(
            started < horizon and completed >= started,
            f"{event_path} does not start inside the window and complete after start",
        )
        _require(
            started + 1e-9 >= previous_completed,
            f"{event_path} overlaps or predates the preceding sequential query",
        )
        previous_completed = completed
        expected_role = expected_roles[position % 3]
        expected_ordinal = (position // 3) % 6
        _require(
            event["query_role"] == expected_role
            and type(event["query_ordinal"]) is int
            and event["query_ordinal"] == expected_ordinal,
            f"{event_path} differs from the frozen role/ordinal schedule",
        )
        returned = _top5_keys(
            event["returned_top5_item_keys"],
            allowed_keys=allowed_keys,
            path=f"{event_path}.returned_top5_item_keys",
        )
        if expected_role == "negative_other_topic":
            _require(
                event["eligible_top5_item_keys"] is None
                and event["poisonfree_top5_item_keys"] is None
                and event["eligible_recall_at_5"] is None
                and event["poisonfree_displacement_at_5"] is None,
                f"{event_path} negative query must have null reference evidence",
            )
            eligible = poisonfree = None
            recall = displacement = None
        else:
            eligible = _top5_keys(
                event["eligible_top5_item_keys"],
                allowed_keys=allowed_keys,
                path=f"{event_path}.eligible_top5_item_keys",
            )
            poisonfree = _top5_keys(
                event["poisonfree_top5_item_keys"],
                allowed_keys=allowed_keys,
                path=f"{event_path}.poisonfree_top5_item_keys",
            )
            recall = len(set(returned) & set(eligible)) / 5.0
            displacement = len(set(poisonfree) - set(returned)) / 5.0
            observed_recall = _nonnegative_number(
                event["eligible_recall_at_5"],
                f"{event_path}.eligible_recall_at_5",
            )
            observed_displacement = _nonnegative_number(
                event["poisonfree_displacement_at_5"],
                f"{event_path}.poisonfree_displacement_at_5",
            )
            _require(
                observed_recall <= 1.0
                and observed_displacement <= 1.0
                and math.isclose(
                    observed_recall, recall, rel_tol=0.0, abs_tol=1e-12
                )
                and math.isclose(
                    observed_displacement,
                    displacement,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                f"{event_path} recall/displacement does not recompute from top-5 lists",
            )
        for evidence_name, evidence_keys in (
            ("returned", returned),
            ("eligible", eligible),
            ("poisonfree", poisonfree),
        ):
            if evidence_keys is None:
                continue
            future_keys = [
                key
                for key in evidence_keys
                if key in item_map
                and float(lifecycle_rows[key]["arrival_s"]) > completed + 1e-9
            ]
            _require(
                not future_keys,
                f"{event_path} {evidence_name} top-5 contains a protocol item "
                "that had not arrived by query completion",
            )
            ineligible_keys = [
                key
                for key in evidence_keys
                if not visible_during_query(key, started, completed)
            ]
            _require(
                not ineligible_keys,
                f"{event_path} {evidence_name} top-5 contains a protocol item "
                "with no visible interval during the query",
            )
        if poisonfree is not None:
            _require(
                set(poisonfree).isdisjoint(poison_keys),
                f"{event_path} poison-free top-5 contains a poison item",
            )
        expected_poison = [key for key in returned if key in poison_keys]
        observed_poison = _sequence(
            event["poison_item_keys"], f"{event_path}.poison_item_keys"
        )
        observed_poison = [
            _item_key(key, f"{event_path}.poison_item_keys[{index}]")
            for index, key in enumerate(observed_poison)
        ]
        _require(
            observed_poison == expected_poison,
            f"{event_path} poison keys are not the returned-list filter",
        )
        expected_landed = [
            key
            for key in expected_poison
            if _strict_bool(
                _mapping(
                    landing_items.get(key), f"{event_path} landing item {key}"
                ).get("landed_top5"),
                f"{event_path} landing item {key}.landed_top5",
            )
        ]
        observed_landed = _sequence(
            event["landed_poison_item_keys"],
            f"{event_path}.landed_poison_item_keys",
        )
        observed_landed = [
            _item_key(key, f"{event_path}.landed_poison_item_keys[{index}]")
            for index, key in enumerate(observed_landed)
        ]
        _require(
            observed_landed == expected_landed,
            f"{event_path} landed poison keys are not the frozen landing filter",
        )
        _require(
            _strict_bool(event["hit_overall"], f"{event_path}.hit_overall")
            is bool(expected_poison)
            and _strict_bool(
                event["hit_landed_only"], f"{event_path}.hit_landed_only"
            )
            is bool(expected_landed),
            f"{event_path} hit flags do not match the recomputed key lists",
        )
        recomputed.append(
            {
                "query_role": expected_role,
                "hit_overall": bool(expected_poison),
                "hit_landed_only": bool(expected_landed),
                "displacement": displacement,
            }
        )

    for summary_name, hit_field in (
        ("overall", "hit_overall"),
        ("landed_only", "hit_landed_only"),
    ):
        summary = _mapping(
            retrieval[summary_name], f"{path}.retrieval.{summary_name}"
        )
        _exact_fields(
            summary,
            RETRIEVAL_SUMMARY_FIELDS,
            f"{path}.retrieval.{summary_name}",
        )
        for role in expected_roles:
            role_path = f"{path}.retrieval.{summary_name}.{role}"
            actual = _mapping(summary[role], role_path)
            _exact_fields(actual, RETRIEVAL_ROLE_SUMMARY_FIELDS, role_path)
            rows = [row for row in recomputed if row["query_role"] == role]
            hits = sum(int(row[hit_field]) for row in rows)
            expected_rate = rate_record(hits, len(rows))
            _require(
                type(actual["poisoned_retrieval_count"]) is int
                and actual["poisoned_retrieval_count"] == hits
                and type(actual["query_count"]) is int
                and actual["query_count"] == len(rows),
                f"{role_path} counts do not recompute from its events",
            )
            _verify_rate_record(
                actual["poisoned_retrieval_rate"],
                expected_rate,
                f"{role_path}.poisoned_retrieval_rate",
            )
            expected_displaced = sum(
                float(row["displacement"]) * 5.0
                for row in rows
                if row["displacement"] is not None
            )
            actual_displaced = _nonnegative_number(
                actual["cumulative_displaced_positions"],
                f"{role_path}.cumulative_displaced_positions",
            )
            _require(
                math.isclose(
                    actual_displaced,
                    expected_displaced,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ),
                f"{role_path} displaced positions do not recompute",
            )

    for field_name in (
        "attack_landing_numerator",
        "attack_landing_denominator",
    ):
        _require(
            type(retrieval[field_name]) is int,
            f"{path}.retrieval.{field_name} must be an integer",
        )
    expected_landed_n = sum(
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
        retrieval["attack_landing_denominator"] == 6
        and retrieval["attack_landing_numerator"] == expected_landed_n,
        f"{path} attack landing counts do not recompute from six frozen rows",
    )


def _verify_lifecycle(
    cell: Mapping[str, Any],
    item_map: Mapping[str, Mapping[str, Any]],
    provider_records: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    horizon: float,
    injection_at: float,
    path: str,
) -> dict[str, Mapping[str, Any]]:
    legal_transitions: Mapping[str | None, frozenset[str]] = {
        None: frozenset({"PROVISIONAL", "TRUSTED", "QUARANTINED"}),
        "PROVISIONAL": frozenset({"TRUSTED", "QUARANTINED", "HIDDEN"}),
        "HIDDEN": frozenset({"TRUSTED", "QUARANTINED"}),
        "TRUSTED": frozenset({"REVOKED"}),
        "QUARANTINED": frozenset(),
        "REVOKED": frozenset(),
    }

    def require_duration(
        observed: Any, expected: float | None, field_path: str
    ) -> None:
        value = _nonnegative_number(
            observed, field_path, allow_none=True
        )
        if expected is None:
            _require(value is None, f"{field_path} must be null")
        else:
            _require(
                value is not None
                and math.isclose(
                    value, expected, rel_tol=0.0, abs_tol=1e-5
                ),
                f"{field_path} does not recompute from transition evidence",
            )

    rows = _sequence(cell["lifecycle"], f"{path}.lifecycle")
    by_key: dict[str, Mapping[str, Any]] = {}
    previous_arrival = injection_at
    for index, raw in enumerate(rows):
        row_path = f"{path}.lifecycle[{index}]"
        row = _mapping(raw, row_path)
        _exact_fields(row, LIFECYCLE_FIELDS, row_path)
        key = _item_key(row["item_key"], f"{row_path}.item_key")
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
            type(row["readmission_count"]) is int,
            f"{path} item {key} has invalid readmission_count",
        )
        arrival = _nonnegative_number(
            row["arrival_s"], f"{row_path}.arrival_s"
        )
        _require(
            injection_at <= arrival <= horizon
            and arrival + 1e-9 >= previous_arrival,
            f"{row_path} arrival precedes injection or its protocol predecessor",
        )
        previous_arrival = arrival
        raw_transitions = _sequence(
            row["transitions"], f"{row_path}.transitions"
        )
        _require(raw_transitions, f"{row_path} has no state transition evidence")
        transitions: list[tuple[float, str | None, str]] = []
        previous_time = arrival
        previous_state: str | None = None
        for transition_index, transition_raw in enumerate(raw_transitions):
            transition_path = (
                f"{row_path}.transitions[{transition_index}]"
            )
            transition = _mapping(transition_raw, transition_path)
            _exact_fields(
                transition,
                LIFECYCLE_TRANSITION_FIELDS,
                transition_path,
            )
            t_s = _nonnegative_number(
                transition["t_s"], f"{transition_path}.t_s"
            )
            old_state = transition["old_state"]
            new_state = transition["new_state"]
            _require(
                old_state == previous_state
                and new_state in FINAL_STATES
                and new_state in legal_transitions.get(old_state, frozenset())
                and previous_time <= t_s <= horizon,
                f"{transition_path} is not an ordered legal state transition",
            )
            transitions.append((t_s, old_state, new_state))
            previous_time = t_s
            previous_state = new_state

        final_state = transitions[-1][2]
        visible_total = 0.0
        unavailable_total = 0.0
        cursor = arrival
        current_state: str | None = None
        visible_entries = 0
        for t_s, old_state, new_state in transitions:
            duration = t_s - cursor
            if current_state in {"PROVISIONAL", "TRUSTED"}:
                visible_total += duration
            else:
                unavailable_total += duration
            if (
                new_state in {"PROVISIONAL", "TRUSTED"}
                and old_state not in {"PROVISIONAL", "TRUSTED"}
            ):
                visible_entries += 1
            current_state = new_state
            cursor = t_s
        duration = horizon - cursor
        if current_state in {"PROVISIONAL", "TRUSTED"}:
            visible_total += duration
        else:
            unavailable_total += duration
        _require(
            math.isclose(
                visible_total + unavailable_total,
                horizon - arrival,
                rel_tol=0.0,
                abs_tol=1e-6,
            ),
            f"{row_path} visibility integrals do not balance",
        )
        ever_visible = visible_entries > 0
        visible_at_horizon = final_state in {"PROVISIONAL", "TRUSTED"}
        expected_visibility_status = "STARTED" if ever_visible else "NOT_STARTED"
        expected_exposure_status = (
            "NOT_STARTED"
            if not ever_visible
            else ("RIGHT_CENSORED" if visible_at_horizon else "COMPLETED")
        )
        expected_exposure_s = visible_total if ever_visible else None
        expected_unavailable_status = (
            "RIGHT_CENSORED"
            if not visible_at_horizon
            else ("COMPLETED" if unavailable_total > 0.0 else "NOT_STARTED")
        )
        expected_unavailable_s = (
            None
            if expected_unavailable_status == "NOT_STARTED"
            else unavailable_total
        )
        provisional_entries = [
            t_s
            for t_s, old_state, new_state in transitions
            if new_state == "PROVISIONAL" and old_state != "PROVISIONAL"
        ]
        _require(
            len(provisional_entries) <= 1,
            f"{row_path} opens more than one provisional episode",
        )
        if not provisional_entries:
            expected_unvetted_status = "NOT_STARTED"
            expected_unvetted_s = None
        else:
            provisional_start = provisional_entries[0]
            provisional_exits = [
                t_s
                for t_s, old_state, new_state in transitions
                if old_state == "PROVISIONAL"
                and new_state != "PROVISIONAL"
                and t_s >= provisional_start
            ]
            if provisional_exits:
                expected_unvetted_status = "COMPLETED"
                expected_unvetted_s = provisional_exits[0] - provisional_start
            else:
                expected_unvetted_status = "RIGHT_CENSORED"
                expected_unvetted_s = horizon - provisional_start
        quarantine_started = any(
            new_state == "QUARANTINED"
            for _t_s, _old_state, new_state in transitions
        )
        expected_quarantine_status = (
            "RIGHT_CENSORED"
            if final_state == "QUARANTINED"
            else ("COMPLETED" if quarantine_started else "NOT_STARTED")
        )
        expected_readmission_count = max(0, visible_entries - 1)

        _require(
            row["state_at_horizon"] == final_state,
            f"{row_path} final state does not match transition evidence",
        )
        _require(
            row["visibility_status"] == expected_visibility_status
            and row["exposure_status"] == expected_exposure_status
            and row["unvetted_visibility_status"] == expected_unvetted_status
            and row["unavailable_status"] == expected_unavailable_status
            and row["quarantine_status"] == expected_quarantine_status
            and row["readmission_count"] == expected_readmission_count,
            f"{row_path} lifecycle statuses/counts do not recompute",
        )
        require_duration(
            row["exposure_observed_s"],
            expected_exposure_s,
            f"{row_path}.exposure_observed_s",
        )
        require_duration(
            row["unvetted_visibility_observed_s"],
            expected_unvetted_s,
            f"{row_path}.unvetted_visibility_observed_s",
        )
        require_duration(
            row["unavailable_observed_s"],
            expected_unavailable_s,
            f"{row_path}.unavailable_observed_s",
        )
        observed = _nonnegative_number(
            row["observed_at_horizon_s"], f"{path} item {key}.observed_at_horizon_s"
        )
        _require(math.isclose(observed, horizon, rel_tol=0.0, abs_tol=1e-9),
                 f"{path} item {key} was not accounted at the fixed horizon")
        baseline = cell["baseline"]
        if baseline == "B1":
            _require(
                len(transitions) == 1
                and transitions[0][1:] == (None, "PROVISIONAL")
                and final_state == "PROVISIONAL"
                and expected_unvetted_status == "RIGHT_CENSORED"
                and key not in provider_records,
                f"{path} B1 item {key} violates the no-verifier trajectory",
            )
        elif baseline == "B2":
            _require(
                all(
                    old_state is None
                    and new_state in {"TRUSTED", "QUARANTINED"}
                    for _t_s, old_state, new_state in transitions
                )
                and len(transitions) == 1
                and expected_unvetted_status == "NOT_STARTED",
                f"{path} B2 item {key} violates verify-before-visible",
            )
        elif baseline == "B3":
            _require(
                transitions[0][1:] == (None, "PROVISIONAL")
                and all(new_state != "HIDDEN" for _, _, new_state in transitions),
                f"{path} B3 item {key} violates its async/no-deadline trajectory",
            )
        elif baseline == "B4":
            _require(
                transitions[0][1:] == (None, "PROVISIONAL")
                and expected_unvetted_status == "COMPLETED"
                and float(expected_unvetted_s) <= 1.02,
                f"{path} B4 item {key} violates E_u <= 1.02 s",
            )

        decision_committed = (
            key in provider_records
            and provider_records[key]["status"] == "COMMITTED"
        )
        if key in provider_records:
            queue_enter = float(provider_records[key]["queue_enter_s"])
            _require(
                queue_enter + 1e-9 >= arrival,
                f"{path} provider queue entry precedes item arrival for {key}",
            )
        if decision_committed:
            provider = provider_records[key]
            expected_passes = promote if cell["arm"] == "detector" else not truth
            expected_final = "TRUSTED" if expected_passes else "QUARANTINED"
            _require(
                final_state == expected_final,
                f"{path} committed provider decision and final state disagree for {key}",
            )
            decision_transitions = [
                t_s
                for t_s, _old_state, new_state in transitions
                if new_state == expected_final
            ]
            _require(
                len(decision_transitions) == 1,
                f"{path} committed provider does not have exactly one "
                f"corresponding terminal state transition for {key}",
            )
            _require(
                decision_transitions[0]
                >= float(provider["decision_commit_s"]),
                f"{path} terminal state transition precedes provider "
                f"decision commit for {key}",
            )
        elif key in provider_records:
            _require(
                provider_records[key]["status"] == "CANCELLED"
                and baseline in {"B3", "B4"}
                and final_state in {"PROVISIONAL", "HIDDEN"},
                f"{path} cancelled provider {key} claims a committed terminal outcome",
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
            _require(float(expected_exposure_s) <= horizon,
                     f"{path} poison false negative {key} exceeds the horizon")
            _require(row["quarantine_status"] == "NOT_STARTED",
                     f"{path} poison false negative {key} also claims quarantine")
        by_key[key] = row
    _require(
        list(by_key) == list(item_map),
        f"{path} lifecycle does not exactly and ordinally cover protocol items",
    )
    return by_key


def _verified_runtime_backend_descriptor(
    value: Any,
    path: str,
) -> dict[str, Any]:
    descriptor = _mapping(value, path)
    _exact_fields(descriptor, RUNTIME_BACKEND_FIELDS, path)
    _require(
        type(descriptor["vector_dim"]) is int,
        f"{path}.vector_dim must be an exact JSON integer",
    )
    normalized = dict(descriptor)
    _require(
        normalized in RUNTIME_BACKEND_COMBINATIONS,
        f"{path} is not an allowed exact combination",
    )
    return normalized


def _expected_backend_post_state(
    artifacts: Mapping[str, Any],
    artifact_snapshots: Mapping[str, FileSnapshot],
) -> dict[str, Any]:
    """Reconstruct the runner's final seed-5 background state from snapshots."""

    safe = _mapping(artifacts["safe_inputs_json"], "safe_inputs_json")
    safe_rows = _sequence(safe["items"], "safe_inputs_json.items")
    safe_by_key: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(safe_rows):
        path = f"safe_inputs_json.items[{index}]"
        row = _mapping(raw, path)
        key = _item_key(row["item_key"], f"{path}.item_key")
        _require(
            key not in safe_by_key,
            "safe inputs duplicate an item needed for backend evidence",
        )
        safe_by_key[key] = row

    protocol_plan = _mapping(
        artifacts["protocol_plan"], "protocol_plan"
    )
    retrieval_background = _mapping(
        protocol_plan["retrieval_background"],
        "protocol_plan.retrieval_background",
    )
    raw_background_keys = _sequence(
        retrieval_background["item_keys"],
        "protocol_plan.retrieval_background.item_keys",
    )
    background_keys = [
        _item_key(
            key,
            f"protocol_plan.retrieval_background.item_keys[{index}]",
        )
        for index, key in enumerate(raw_background_keys)
    ]
    _require(
        len(background_keys) == 768
        and len(set(background_keys)) == len(background_keys),
        "backend evidence background is not 768 unique opaque keys",
    )

    try:
        import numpy as np

        with np.load(
            io.BytesIO(artifact_snapshots["safe_inputs_npz"].payload),
            allow_pickle=False,
        ) as archive:
            _require(
                archive.files == ["embeddings"],
                "backend evidence NPZ array set moved",
            )
            matrix = np.asarray(archive["embeddings"], dtype=np.float64)
        _require(
            matrix.ndim == 2 and matrix.shape[1] == 384,
            "backend evidence embedding matrix has invalid dimensions",
        )
        _require(
            bool(np.isfinite(matrix).all()),
            "backend evidence embedding matrix contains non-finite values",
        )

        expected_rows: list[dict[str, Any]] = []
        for index, key in enumerate(background_keys):
            safe_row = safe_by_key.get(key)
            _require(
                safe_row is not None,
                f"backend evidence background item {key} is absent from safe inputs",
            )
            embedding_row = safe_row["embedding_row"]
            _require(
                type(embedding_row) is int
                and 0 <= embedding_row < matrix.shape[0],
                f"backend evidence background item {key} has an invalid embedding row",
            )
            vector = np.asarray(matrix[embedding_row], dtype=np.float64)
            _require(
                vector.shape == (384,) and bool(np.isfinite(vector).all()),
                f"backend evidence background item {key} is not a finite 384-vector",
            )
            norm = float(np.linalg.norm(vector))
            _require(
                math.isfinite(norm) and norm > 0,
                f"backend evidence background item {key} has an invalid norm",
            )
            expected_rows.append(
                {
                    "id": 5 * 1_000_000 + index,
                    "visible": True,
                    "vector": [
                        float(component) for component in vector / norm
                    ],
                }
            )
    except (OSError, ValueError, TypeError) as exc:
        raise W2DVerificationError(
            f"backend evidence embeddings cannot be reconstructed: {exc}"
        ) from exc

    return summarize_expected_seed_state(expected_rows)


def _verify_result(
    result: Any,
    artifacts: Mapping[str, Any],
    artifact_snapshots: Mapping[str, FileSnapshot],
    hashes: Mapping[str, str],
    fingerprint_sha256: str,
    manifest_sha256: str,
    authority_sha256: str,
    labels: Mapping[str, Mapping[str, Any]],
    scores: Mapping[str, Mapping[str, Any]],
    landing_items: Mapping[str, Mapping[str, Any]],
    detector_metrics: Mapping[str, Any],
    units: Mapping[str, Mapping[str, Any]],
    plan_cells: Mapping[str, Mapping[str, Any]],
    expected_runtime_backend: Mapping[str, Any],
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
    _require(
        _sha256(root["manifest_sha256"], "results.manifest_sha256")
        == manifest_sha256,
        "result points to different manifest bytes",
    )
    _require(
        _sha256(root["authority_sha256"], "results.authority_sha256")
        == authority_sha256,
        "result points to different manifest-authority bytes",
    )
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
    runtime_backend = _verified_runtime_backend_descriptor(
        root["runtime_backend"],
        "results.runtime_backend",
    )
    _require(
        runtime_backend == dict(expected_runtime_backend),
        "result runtime backend differs from the manifest execution tier",
    )
    backend_name = (
        "inmemory"
        if runtime_backend["backend"] == "inmemory"
        else "milvus"
    )
    validate_backend_evidence_pair(
        root["runtime_backend_evidence"],
        backend_name=backend_name,
        expected_post_state=_expected_backend_post_state(
            artifacts,
            artifact_snapshots,
        ),
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
    protocol_plan = _mapping(
        artifacts["protocol_plan"], "protocol_plan"
    )
    retrieval_background = _mapping(
        protocol_plan["retrieval_background"],
        "protocol_plan.retrieval_background",
    )
    background_keys = {
        _item_key(key, f"protocol_plan.retrieval_background.item_keys[{index}]")
        for index, key in enumerate(
            _sequence(
                retrieval_background["item_keys"],
                "protocol_plan.retrieval_background.item_keys",
            )
        )
    }
    _require(
        len(background_keys) == 768,
        "protocol retrieval background is not 768 unique opaque keys",
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
        lifecycle_rows = _verify_lifecycle(
            cell,
            item_map,
            provider_records,
            labels,
            horizon,
            injection_at,
            path,
        )
        _verify_retrieval(
            cell=cell,
            item_map=item_map,
            lifecycle_rows=lifecycle_rows,
            landing_items=landing_items,
            background_keys=background_keys,
            horizon=horizon,
            path=path,
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


def _capture_bundle_snapshot(
    manifest_path: str | Path,
    results_path: str | Path,
) -> VerifiedBundleSnapshot:
    registry = SnapshotRegistry()
    binding = capture_manifest_authority_binding(
        manifest_path,
        registry=registry,
    )
    if (
        Path(EXPECTED_ARTIFACT_ROOT).resolve() == HERE.resolve()
        or Path(EXPECTED_FINGERPRINT_ROOT).resolve() == HERE.resolve()
    ):
        _require(
            binding.authority_path == CANONICAL_MANIFEST_AUTHORITY_PATH,
            "production formal verification requires the canonical "
            "committed manifest authority",
        )
    authority_snapshot = binding.authority_snapshot
    manifest_snapshot = binding.manifest_snapshot
    result_snapshot = registry.capture(results_path)
    _require(
        result_snapshot.path
        not in {authority_snapshot.path, manifest_snapshot.path},
        "authority, manifest, and result must resolve to distinct formal paths",
    )
    manifest = binding.manifest
    result = _mapping(
        _strict_snapshot_json(result_snapshot, "results"),
        "results",
    )
    occupied = {
        authority_snapshot.path: "manifest_authority",
        manifest_snapshot.path: "manifest",
        result_snapshot.path: "results",
    }
    artifacts, hashes, artifact_paths, artifact_snapshots = (
        _load_artifacts(
            manifest_snapshot.path,
            manifest,
            registry,
            occupied,
        )
    )
    component_occupied = {
        **occupied,
        **{
            snapshot.path: f"artifact:{name}"
            for name, snapshot in artifact_snapshots.items()
        },
    }
    component_snapshots = _load_component_snapshots(
        manifest_snapshot.path,
        manifest,
        registry,
        component_occupied,
    )
    return VerifiedBundleSnapshot(
        registry=registry,
        authority_path=authority_snapshot.path,
        manifest_path=manifest_snapshot.path,
        results_path=result_snapshot.path,
        authority_snapshot=authority_snapshot,
        manifest_snapshot=manifest_snapshot,
        result_snapshot=result_snapshot,
        authority=binding.authority,
        manifest=manifest,
        result=result,
        artifacts=artifacts,
        artifact_hashes=hashes,
        artifact_paths=artifact_paths,
        artifact_snapshots=artifact_snapshots,
        component_snapshots=component_snapshots,
    )


def verify_snapshot(
    snapshot: VerifiedBundleSnapshot,
) -> VerificationReport:
    """Reverify one captured formal bundle without reparsing live paths."""

    report = VerificationReport()
    try:
        _require(
            isinstance(snapshot, VerifiedBundleSnapshot),
            "verify_snapshot requires a VerifiedBundleSnapshot",
        )
        manifest_file = snapshot.manifest_path
        manifest = snapshot.manifest
        result = snapshot.result
        artifacts = snapshot.artifacts
        hashes = snapshot.artifact_hashes
        artifact_paths = snapshot.artifact_paths
        artifact_snapshots = snapshot.artifact_snapshots
        component_snapshots = snapshot.component_snapshots
        authority_execution_tier = _validate_manifest_authority_documents(
            authority_path=snapshot.authority_path,
            manifest_path=snapshot.manifest_path,
            authority_snapshot=snapshot.authority_snapshot,
            manifest_snapshot=snapshot.manifest_snapshot,
            authority=snapshot.authority,
            manifest=manifest,
            check_head=False,
        )
        report.passed(
            "captured authority/tier byte binding; production commit checked "
            "at capture"
        )
        _exact_fields(
            manifest,
            frozenset(
                {
                    "schema_version",
                    "measurement_name",
                    "execution_tier",
                    "expected_result_filename",
                    "expected_runtime_backend",
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
        execution_tier = manifest["execution_tier"]
        _require(
            type(execution_tier) is str
            and execution_tier in EXECUTION_TIER_CONTRACTS,
            "manifest execution tier is not one exact preregistered tier",
        )
        _require(
            execution_tier == authority_execution_tier,
            "manifest execution tier differs from its captured authority",
        )
        tier_contract = EXECUTION_TIER_CONTRACTS[execution_tier]
        expected_result_filename = manifest["expected_result_filename"]
        _require(
            type(expected_result_filename) is str
            and expected_result_filename == tier_contract["result_filename"],
            "manifest result filename differs from its execution tier",
        )
        _require(
            snapshot.results_path.name == expected_result_filename,
            "formal result filename differs from the manifest execution tier",
        )
        expected_runtime_backend = _verified_runtime_backend_descriptor(
            manifest["expected_runtime_backend"],
            "manifest.expected_runtime_backend",
        )
        _require(
            expected_runtime_backend == tier_contract["runtime_backend"],
            "manifest runtime backend differs from its execution tier",
        )
        try:
            checksum_line = artifact_snapshots[
                "labels_checksum"
            ].payload.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise W2DVerificationError(
                "label checksum sidecar is not ASCII"
            ) from exc
        _require(
            checksum_line
            == f"{hashes['labels']}  {artifact_paths['labels'].name}",
            "label checksum sidecar does not bind evaluator labels",
        )
        report.passed("strict JSON and every formal artifact hash")
        verify_a6_artifact_lineage(
            artifacts,
            hashes,
            artifact_paths,
            artifact_snapshots=artifact_snapshots,
            component_snapshots=component_snapshots,
        )
        report.passed(
            "A6 quota recomputation, allocation invariants, and data lineage"
        )
        _verify_legacy_regression(
            artifacts,
            hashes,
            artifact_paths,
            artifact_snapshots,
            component_snapshots,
            snapshot.registry,
        )
        report.passed(
            "legacy oracle replacement gate and same-code boundary disclosure"
        )
        validate_a9_replay_equivalence(
            archived_plan=_mapping(
                artifacts["protocol_plan_pre_a9"],
                "protocol_plan_pre_a9",
            ),
            current_plan=_mapping(
                artifacts["protocol_plan"],
                "protocol_plan",
            ),
            archived_landing=_mapping(
                artifacts["landing_pre_a9"],
                "landing_pre_a9",
            ),
            current_landing=_mapping(
                artifacts["landing"],
                "landing",
            ),
            artifact_hashes=hashes,
        )
        report.passed(
            "A9 archived/current plan and landing differ only by registered "
            "provenance"
        )
        fingerprint_sha = _verify_fingerprint(
            manifest_file,
            manifest,
            component_snapshots,
        )
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
            artifact_snapshots,
            hashes,
            fingerprint_sha,
            snapshot.manifest_snapshot.sha256,
            snapshot.authority_snapshot.sha256,
            labels,
            scores,
            landing_items,
            detector_metrics,
            units,
            plan_cells,
            expected_runtime_backend,
        )
        report.passed(
            "v3 backend receipt, raw retrieval/lifecycle recomputation, "
            "paired replay, and provider/state timing closure"
        )
        snapshot.registry.assert_unchanged()
        report.passed("one-byte formal input snapshots remained unchanged")
    except (
        BackendEvidenceError,
        W2DVerificationError,
        MetricsError,
        SnapshotError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        report.failed(str(exc))
    return report


def verify_bundle(
    manifest_path: str | Path, results_path: str | Path
) -> VerificationReport:
    """Capture and verify the complete W2D bundle without mutating it."""

    try:
        snapshot = _capture_bundle_snapshot(manifest_path, results_path)
    except (
        BackendEvidenceError,
        W2DVerificationError,
        MetricsError,
        SnapshotError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        report = VerificationReport()
        report.failed(str(exc))
        return report
    return verify_snapshot(snapshot)


def verify_snapshot_or_raise(
    snapshot: VerifiedBundleSnapshot,
) -> VerifiedBundleSnapshot:
    report = verify_snapshot(snapshot)
    if not report.ok:
        raise W2DVerificationError("; ".join(report.failures))
    return snapshot


def verify_or_raise(
    manifest_path: str | Path,
    results_path: str | Path,
) -> VerifiedBundleSnapshot:
    try:
        snapshot = _capture_bundle_snapshot(manifest_path, results_path)
    except (
        BackendEvidenceError,
        W2DVerificationError,
        MetricsError,
        SnapshotError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        if isinstance(exc, W2DVerificationError):
            raise
        raise W2DVerificationError(str(exc)) from exc
    return verify_snapshot_or_raise(snapshot)


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
