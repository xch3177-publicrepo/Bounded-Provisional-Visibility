#!/usr/bin/env python3
"""Tests for the machine-enforced A9 replay equivalence gate."""

from __future__ import annotations

import copy
from pathlib import Path
import unittest

from w2d_equivalence import (
    A9EquivalenceError,
    ARCHIVED_LANDING_SHA256,
    ARCHIVED_PLAN_SHA256,
    CURRENT_LANDING_SHA256,
    CURRENT_PLAN_SHA256,
    validate_a9_replay_equivalence,
)
from w2d_metrics import file_sha256, strict_json_load


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "w2d"


class A9ReplayEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.archived_plan = strict_json_load(
            str(
                RESULTS
                / "W2D-PROTOCOL-PLAN-preA7-landing-contract-failure.json"
            )
        )
        cls.current_plan = strict_json_load(
            str(RESULTS / "W2D-PROTOCOL-PLAN.json")
        )
        cls.archived_landing = strict_json_load(
            str(RESULTS / "W2D-LANDING-preA9-lineage.json")
        )
        cls.current_landing = strict_json_load(
            str(RESULTS / "W2D-LANDING.json")
        )
        cls.hashes = {
            "protocol_plan_pre_a9": file_sha256(
                str(
                    RESULTS
                    / "W2D-PROTOCOL-PLAN-preA7-landing-contract-failure.json"
                )
            ),
            "protocol_plan": file_sha256(
                str(RESULTS / "W2D-PROTOCOL-PLAN.json")
            ),
            "landing_pre_a9": file_sha256(
                str(RESULTS / "W2D-LANDING-preA9-lineage.json")
            ),
            "landing": file_sha256(str(RESULTS / "W2D-LANDING.json")),
        }

    def _validate(self, **changes) -> None:
        values = {
            "archived_plan": copy.deepcopy(self.archived_plan),
            "current_plan": copy.deepcopy(self.current_plan),
            "archived_landing": copy.deepcopy(self.archived_landing),
            "current_landing": copy.deepcopy(self.current_landing),
            "artifact_hashes": dict(self.hashes),
        }
        values.update(changes)
        validate_a9_replay_equivalence(**values)

    def test_frozen_production_pair_passes(self) -> None:
        self.assertEqual(self.hashes["protocol_plan_pre_a9"], ARCHIVED_PLAN_SHA256)
        self.assertEqual(self.hashes["protocol_plan"], CURRENT_PLAN_SHA256)
        self.assertEqual(
            self.hashes["landing_pre_a9"], ARCHIVED_LANDING_SHA256
        )
        self.assertEqual(self.hashes["landing"], CURRENT_LANDING_SHA256)
        self._validate()

    def test_raw_hash_change_fails(self) -> None:
        hashes = dict(self.hashes)
        hashes["landing"] = "0" * 64
        with self.assertRaisesRegex(A9EquivalenceError, "raw artifact hashes"):
            self._validate(artifact_hashes=hashes)

    def test_plan_scientific_change_fails(self) -> None:
        plan = copy.deepcopy(self.current_plan)
        plan["landing_plan"]["top_k"] = 6
        with self.assertRaisesRegex(A9EquivalenceError, "outside phase"):
            self._validate(current_plan=plan)

    def test_unregistered_phase_or_code_change_fails(self) -> None:
        plan = copy.deepcopy(self.current_plan)
        plan["phase_constraint"] += " changed"
        with self.assertRaisesRegex(A9EquivalenceError, "truthful A9"):
            self._validate(current_plan=plan)
        plan = copy.deepcopy(self.current_plan)
        for record in plan["provenance"]["inputs"]:
            if record.get("role") == "protocol_plan_code":
                record["sha256"] = "0" * 64
        with self.assertRaisesRegex(A9EquivalenceError, "code hash"):
            self._validate(current_plan=plan)

    def test_landing_scientific_change_fails(self) -> None:
        landing = copy.deepcopy(self.current_landing)
        landing["items"][0]["rank"] += 1
        with self.assertRaisesRegex(A9EquivalenceError, "scientific field"):
            self._validate(current_landing=landing)


if __name__ == "__main__":
    unittest.main(verbosity=2)
