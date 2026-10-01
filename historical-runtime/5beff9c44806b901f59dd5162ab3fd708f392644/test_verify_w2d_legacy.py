#!/usr/bin/env python3
"""Offline tests for the Amendment-A5 legacy-oracle gate."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from w2d_metrics import file_sha256, strict_json_dump, strict_json_load
from verify_w2d_legacy import (
    FROZEN_CANDIDATE_COMMIT,
    FROZEN_SAME_CODE_DIFFERENCE_SHA256,
    FROZEN_SOURCE_SHA256,
    LEGACY_GATE_COMMANDS,
    LegacyGateError,
    canonical_strict_json_sha256,
    evaluate,
    legacy_runtime_code_sha256,
    rerun_legacy_gates,
)


def _cell(baseline: str, backlog: str, tp: float, seed: int) -> dict:
    b1 = baseline == "B1"
    b2 = baseline == "B2"
    return {
        "baseline": baseline,
        "backlog": backlog,
        "Tp": tp,
        "seed": seed,
        "false_promoted_k": 0,
        "poisoned_retrievals_craft": 6 if b1 else 0,
        "poisoned_retrievals_targeted": 1 if b1 else 0,
        "poisoned_retrievals_negative": 0,
        "poison_never_visible_n": 6 if b2 else 0,
        "attack_coverage_craft": 1.0 if b1 else 0.0,
        "attack_coverage_target": 1.0 / 6.0 if b1 else 0.0,
        "cond_craft_prr": 1.0 if b1 else 0.0,
        "cond_craft_qualified_q": 6,
        "D_H_craft_positions": 6.0 if b1 else 0.0,
        "D_H_target_positions": 1.0 if b1 else 0.0,
        "foreign_candidates": 0,
        "candidate_queries": 191,
        "underfill_queries": 0,
        "residual_at_start": 0,
        "residual_rows": 0,
        "live_tasks": 0,
        "live_timers": 0,
        "bg_errors": [],
        "Ep_right_censored_n": 6 if b1 else 0,
        "Ep_n": 6 if baseline in {"B3", "B4"} else 0,
        "Ep_p50": None if b1 else 0.2,
        "Ep_max": None if b1 else min(tp, 0.2),
        "Eu_started_n": 0 if b2 else 6,
        "Eu_completed_n": 0 if b2 or b1 else 6,
        "Eu_right_censored_n": 6 if b1 else 0,
        "Eu_completed_p50": 0.1,
        "Eu_completed_max": 0.1,
        "Df_clean_p50": 0.1,
        "Df_trusted_p50": 0.1,
        "clean_expired_n": 0,
        "clean_gap_completed_n": 0,
        "clean_gap_right_censored_n": 0,
        "clean_gap_p50": None,
        "false_promoted_ids": [],
        "false_promoted_offsets": [],
        "sweep_attempts": 1,
        "delete_errors": 0,
        "inv_I3_no_expiry_promotion": True,
        "inv_I7_prov_once": True,
        "poison_readmitted_after_hide_n": 0,
        "sets": {
            role: {
                "per_query": {
                    "0": {
                        "n": 1,
                        "hits": int(b1 and role != "negative"),
                    }
                }
            }
            for role in ("craft", "target", "negative")
        },
    }


def _document(*, realtext: bool = False) -> dict:
    cells = []
    for seed in range(1, 6):
        for backlog in ("normal", "heavy"):
            for baseline in ("B1", "B2", "B3", "B4"):
                cells.append(_cell(baseline, backlog, 1.0, seed))
            for tp in (0.15, 0.3, 2.0):
                cells.append(_cell("B4", backlog, tp, seed))
    config = {
        "seeds": [1, 2, 3, 4, 5],
        "backlog": {
            "normal": {"conc": 8, "items": 0},
            "heavy": {"conc": 1, "items": 12},
        },
        "tp_sweep": [0.15, 0.3, 1.0, 2.0],
        "n_poison": 6,
        "qps": 24,
        "dur": 8.0,
        "tp": 1.0,
    }
    result = {
        "run_id": "fixture",
        "workload": "realtext" if realtext else "synthetic",
        "workload_provenance": {
            "model": "all-MiniLM-L6-v2" if realtext else None
        },
        "dataset_sha256": "d" * 64 if realtext else "s" * 64,
        "git_commit": FROZEN_CANDIDATE_COMMIT,
        "git_dirty": False,
        "runtime_code_sha256": "f" * 64,
        "smoke": False,
        "admissible": True,
        "config": config,
        "metrics": {"cells": cells},
    }
    result["config_sha256"] = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return result


class LegacyGateTest(unittest.TestCase):
    def setUp(self):
        self.old_w2 = _document()
        self.old_w2r = _document(realtext=True)
        self.new_w2 = deepcopy(self.old_w2)
        self.new_w2r = deepcopy(self.old_w2r)
        self.repeat = deepcopy(self.new_w2)
        repeat_cell = next(
            cell
            for cell in self.repeat["metrics"]["cells"]
            if cell["baseline"] == "B4"
            and cell["backlog"] == "normal"
            and cell["Tp"] == 0.3
            and cell["seed"] == 4
        )
        repeat_cell["Ep_p50"] = 0.201
        self.authority = {
            "files": {
                "w2_inmemory": "W2-inmemory.json",
                "w2r_inmemory": "W2R-inmemory.json",
            },
            "sha256": {
                "W2-inmemory.json": FROZEN_SOURCE_SHA256[
                    "authoritative_w2"
                ],
                "W2R-inmemory.json": FROZEN_SOURCE_SHA256[
                    "authoritative_w2r"
                ],
            },
        }

    def evaluate(self):
        return evaluate(
            authority_manifest=self.authority,
            authoritative_w2=self.old_w2,
            authoritative_w2r=self.old_w2r,
            candidate_w2=self.new_w2,
            candidate_w2r=self.new_w2r,
            same_code_repeat=self.repeat,
            source_sha256=FROZEN_SOURCE_SHA256,
            expected_runtime_code_sha256="f" * 64,
            enforce_frozen_same_code_differences=False,
        )

    def test_non_b1_same_code_boundary_drift_is_recorded_not_granted_equality(self):
        target = next(
            cell
            for cell in self.repeat["metrics"]["cells"]
            if cell["baseline"] == "B4"
            and cell["backlog"] == "normal"
            and cell["Tp"] == 0.3
            and cell["seed"] == 4
        )
        target["poisoned_retrievals_craft"] = 2
        target["sets"]["craft"]["per_query"]["0"]["hits"] = 1
        result = self.evaluate()
        self.assertTrue(result["passed"])
        self.assertEqual(
            result["literal_A2_6_cross_execution_equality"], "INCONCLUSIVE"
        )
        self.assertTrue(result["same_code_scheduling_differences"])

    def test_b1_drift_is_still_a_hard_failure(self):
        target = next(
            cell
            for cell in self.new_w2["metrics"]["cells"]
            if cell["baseline"] == "B1"
        )
        target["poisoned_retrievals_craft"] -= 1
        with self.assertRaisesRegex(LegacyGateError, "B1"):
            self.evaluate()

    def test_repeat_config_drift_is_a_hard_failure(self):
        self.repeat["config"]["verify_cost"] = 99
        self.repeat["config_sha256"] = hashlib.sha256(
            json.dumps(
                self.repeat["config"], sort_keys=True
            ).encode("utf-8")
        ).hexdigest()[:16]
        with self.assertRaisesRegex(LegacyGateError, "identity/config"):
            self.evaluate()

    def test_missing_runtime_hash_is_a_hard_failure(self):
        for document in (self.new_w2, self.new_w2r, self.repeat):
            document["runtime_code_sha256"] = None
        with self.assertRaisesRegex(LegacyGateError, "runtime fingerprint"):
            self.evaluate()

    def test_unknown_baseline_cannot_hide_in_a_70_cell_count(self):
        target = next(
            cell
            for cell in self.new_w2["metrics"]["cells"]
            if cell["baseline"] == "B4" and cell["Tp"] == 0.3
        )
        target["baseline"] = "X"
        with self.assertRaisesRegex(LegacyGateError, "70-cell grid"):
            self.evaluate()

    def test_b4_missing_exposure_is_a_hard_failure(self):
        for cell in self.new_w2["metrics"]["cells"]:
            if cell["baseline"] == "B4":
                cell["Ep_max"] = None
        with self.assertRaisesRegex(LegacyGateError, "B4 exposure"):
            self.evaluate()

    def test_w2r_geometry_drift_is_a_hard_failure(self):
        self.new_w2r["workload_geometry"] = {"changed": True}
        with self.assertRaisesRegex(LegacyGateError, "data/geometry"):
            self.evaluate()

    def test_query_denominator_drift_is_a_hard_failure(self):
        target = next(
            cell
            for cell in self.new_w2["metrics"]["cells"]
            if cell["baseline"] == "B3"
        )
        target["candidate_queries"] = 999
        with self.assertRaisesRegex(LegacyGateError, "query-event denominator"):
            self.evaluate()

    def test_source_role_alias_is_a_hard_failure(self):
        source_hashes = dict(FROZEN_SOURCE_SHA256)
        source_hashes["same_code_repeat"] = source_hashes["candidate_w2"]
        with self.assertRaisesRegex(LegacyGateError, "aliased"):
            evaluate(
                authority_manifest=self.authority,
                authoritative_w2=self.old_w2,
                authoritative_w2r=self.old_w2r,
                candidate_w2=self.new_w2,
                candidate_w2r=self.new_w2r,
                same_code_repeat=self.repeat,
                source_sha256=source_hashes,
                expected_runtime_code_sha256="f" * 64,
                enforce_frozen_same_code_differences=False,
            )

    def test_frozen_production_counterexample_has_exact_projection(self):
        root = Path(__file__).resolve().parent
        paths = {
            "authority_manifest": root / "results" / "AUTHORITATIVE.json",
            "authoritative_w2": root / "results" / "W2-inmemory.json",
            "authoritative_w2r": root / "results" / "W2R-inmemory.json",
            "candidate_w2": (
                root / "results" / "w2d" / "W2-legacy-regression-run-a.json"
            ),
            "candidate_w2r": (
                root / "results" / "w2d" / "W2R-legacy-regression.json"
            ),
            "same_code_repeat": (
                root
                / "results"
                / "w2d"
                / "W2-legacy-regression-run-b-same-code.json"
            ),
        }
        loaded = {
            role: strict_json_load(str(path)) for role, path in paths.items()
        }
        result = evaluate(
            **loaded,
            source_sha256={
                role: file_sha256(str(path)) for role, path in paths.items()
            },
            expected_runtime_code_sha256=legacy_runtime_code_sha256(),
        )
        self.assertEqual(
            canonical_strict_json_sha256(
                result["same_code_scheduling_differences"]
            ),
            FROZEN_SAME_CODE_DIFFERENCE_SHA256,
        )

    def test_rerun_uses_current_interpreter_and_exact_eight_commands(self):
        paths = {
            "candidate_w2": "/tmp/candidate-w2.json",
            "candidate_w2r": "/tmp/candidate-w2r.json",
            "same_code_repeat": "/tmp/repeat.json",
        }
        passed = {
            "command": [],
            "returncode": 0,
            "output_sha256": "0" * 64,
            "passed": True,
        }
        with patch(
            "verify_w2d_legacy._run_gate",
            side_effect=lambda command: {**passed, "command": list(command)},
        ) as run:
            executions = rerun_legacy_gates(paths)
        self.assertEqual(len(executions), len(LEGACY_GATE_COMMANDS))
        self.assertEqual(run.call_count, len(LEGACY_GATE_COMMANDS))
        self.assertTrue(
            all(
                Path(row["command"][0]).resolve()
                == Path(sys.executable).resolve()
                for row in executions
            )
        )

    def test_strict_artifact_writer_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            strict_json_dump({"passed": True}, str(path))
            with self.assertRaises(FileExistsError):
                strict_json_dump({"passed": False}, str(path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
