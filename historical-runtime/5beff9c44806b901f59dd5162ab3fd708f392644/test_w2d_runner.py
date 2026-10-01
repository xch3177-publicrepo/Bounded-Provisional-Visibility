#!/usr/bin/env python3
"""Focused offline tests for W2D replay orchestration and estimands."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from functional_slice import State, VerifierRequest
import w2d_runner as runner


def key(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


class ReplayBankTest(unittest.IsolatedAsyncioTestCase):
    async def test_oracle_changes_only_decision_not_service(self):
        poison = key("poison")
        clean = key("clean")
        scores = {
            poison: {
                "promote": True,
                "detector_service_ns": 1_250_000,
                "family_s_affirms": True,
            },
            clean: {
                "promote": False,
                "detector_service_ns": 2_500_000,
                "family_s_affirms": True,
            },
        }
        truth = {poison: True, clean: False}
        detector = runner.ReplayBank(
            arm="detector",
            scores=scores,
            poison_truth=truth,
            score_artifact_sha256="a" * 64,
        )
        oracle = runner.ReplayBank(
            arm="oracle",
            scores=scores,
            poison_truth=truth,
            score_artifact_sha256="a" * 64,
        )
        payload = {"execution_mode": runner.EXECUTION_MODE}
        detector_poison = await detector(VerifierRequest(poison, payload))
        oracle_poison = await oracle(VerifierRequest(poison, payload))
        detector_clean = await detector(VerifierRequest(clean, payload))
        oracle_clean = await oracle(VerifierRequest(clean, payload))
        self.assertTrue(detector_poison.passes)
        self.assertFalse(oracle_poison.passes)
        self.assertFalse(detector_clean.passes)
        self.assertTrue(oracle_clean.passes)
        self.assertEqual(
            detector_poison.service_time_s, oracle_poison.service_time_s
        )
        self.assertEqual(
            detector_clean.service_time_s, oracle_clean.service_time_s
        )

    async def test_replay_rejects_non_replay_payload(self):
        item = key("item")
        bank = runner.ReplayBank(
            arm="detector",
            scores={
                item: {
                    "promote": True,
                    "detector_service_ns": 1,
                    "family_s_affirms": True,
                }
            },
            poison_truth={item: False},
            score_artifact_sha256="b" * 64,
        )
        with self.assertRaises(runner.RunnerError):
            await bank(VerifierRequest(item, {"text": "would be live input"}))


class EstimandTest(unittest.TestCase):
    def test_unvetted_episode_uses_provisional_state_not_total_visibility(self):
        completed = runner._unvetted_visibility_episode(
            transitions=[
                (1.0, 1, None, State.PROVISIONAL),
                (1.4, 1, State.PROVISIONAL, State.TRUSTED),
            ],
            horizon_abs=8.0,
        )
        censored = runner._unvetted_visibility_episode(
            transitions=[
                (1.0, 1, None, State.PROVISIONAL),
                (8.1, 1, State.PROVISIONAL, State.HIDDEN),
            ],
            horizon_abs=8.0,
        )
        absent = runner._unvetted_visibility_episode(
            transitions=[(1.0, 1, None, State.TRUSTED)],
            horizon_abs=8.0,
        )
        self.assertEqual(completed[0], "COMPLETED")
        self.assertAlmostEqual(completed[1], 0.4)
        self.assertEqual(censored, ("RIGHT_CENSORED", 7.0))
        self.assertEqual(absent, ("NOT_STARTED", None))

    @staticmethod
    def _cell():
        poison = [
            {
                "item_key": key(f"p{index}"),
                "false_promotion": index == 0,
                "ever_visible": True,
                "right_censored": index == 0,
                "restricted_visible_time_s": 7.0 if index == 0 else 0.1,
                "state_at_horizon": "TRUSTED" if index == 0 else "QUARANTINED",
                "landed_top5": index < 3,
            }
            for index in range(6)
        ]
        clean = [
            {
                "false_positive": index == 0,
                "never_visible": index == 0,
                "durable_visible": index != 0,
                "restricted_unavailable_time_s": 8.0 if index == 0 else 0.0,
                "quarantine_right_censored": index == 0,
            }
            for index in range(6)
        ]
        events = [
            {
                "query_role": "attack_associated",
                "hit_overall": True,
                "hit_landed_only": False,
                "poisonfree_displacement_at_5": 0.2,
            },
            {
                "query_role": "heldout_same_topic",
                "hit_overall": False,
                "hit_landed_only": False,
                "poisonfree_displacement_at_5": 0.0,
            },
            {
                "query_role": "negative_other_topic",
                "hit_overall": False,
                "hit_landed_only": False,
                "poisonfree_displacement_at_5": None,
            },
        ]
        return {
            "lifecycle": {"poison": poison, "clean": clean},
            "retrieval": {"events": events},
        }

    def test_censoring_uses_restricted_totals_not_duration_median(self):
        overall = runner._population_summary([self._cell()], landed_only=False)
        self.assertEqual(overall["false_promotion_incidence"]["numerator"], 1)
        self.assertEqual(overall["false_promotion_incidence"]["denominator"], 6)
        self.assertEqual(overall["right_censored"]["numerator"], 1)
        self.assertAlmostEqual(overall["restricted_exposure_total_s"], 7.5)
        self.assertNotIn("median", overall)
        landed = runner._population_summary(
            [self._cell()], landed_only=True
        )
        self.assertEqual(landed["poison_item_episode_n"], 3)

    def test_clean_false_positive_stays_in_denominator(self):
        summary = runner._clean_summary([self._cell()])
        self.assertEqual(
            summary["false_positive_misquarantine"]["numerator"], 1
        )
        self.assertEqual(
            summary["false_positive_misquarantine"]["denominator"], 6
        )
        self.assertEqual(summary["right_censored_quarantine_n"], 1)
        self.assertEqual(summary["restricted_unavailable_total_s"], 8.0)

    def test_formal_output_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            runner._exclusive_json_dump({"finite": 1.0}, path)
            with self.assertRaises(FileExistsError):
                runner._exclusive_json_dump({"finite": 2.0}, path)

    def test_postflight_rejects_artifact_drift(self):
        initial = {"safe_inputs_json": "a" * 64}
        with (
            mock.patch.object(runner.analysis, "git_dirty", return_value=False),
            mock.patch.object(
                runner,
                "_validate_manifest_binding",
                return_value=(
                    {"safe_inputs_json": "b" * 64},
                    "c" * 64,
                    {},
                ),
            ),
        ):
            with self.assertRaisesRegex(
                runner.RunnerError, "snapshot changed during replay"
            ):
                runner._assert_run_snapshot_unchanged(
                    paths={},
                    manifest_path="unused.json",
                    expected_hashes=initial,
                    expected_runtime_fingerprint_sha256="c" * 64,
                )

    def test_postflight_rejects_runtime_drift(self):
        initial = {"safe_inputs_json": "a" * 64}
        with (
            mock.patch.object(runner.analysis, "git_dirty", return_value=False),
            mock.patch.object(
                runner,
                "_validate_manifest_binding",
                return_value=(initial, "d" * 64, {}),
            ),
        ):
            with self.assertRaisesRegex(
                runner.RunnerError, "snapshot changed during replay"
            ):
                runner._assert_run_snapshot_unchanged(
                    paths={},
                    manifest_path="unused.json",
                    expected_hashes=initial,
                    expected_runtime_fingerprint_sha256="c" * 64,
                )

    def test_postflight_rejects_dirty_code_before_revalidation(self):
        with (
            mock.patch.object(runner.analysis, "git_dirty", return_value=True),
            mock.patch.object(
                runner.analysis,
                "git_dirty_files",
                return_value=["w2d_runner.py"],
            ),
            mock.patch.object(
                runner, "_validate_manifest_binding"
            ) as validate,
        ):
            with self.assertRaisesRegex(
                runner.RunnerError, "changed during replay"
            ):
                runner._assert_run_snapshot_unchanged(
                    paths={},
                    manifest_path="unused.json",
                    expected_hashes={},
                    expected_runtime_fingerprint_sha256="c" * 64,
                )
            validate.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
