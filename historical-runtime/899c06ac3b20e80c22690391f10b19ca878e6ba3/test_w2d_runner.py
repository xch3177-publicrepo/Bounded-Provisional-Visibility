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
from w2d_equivalence import A9EquivalenceError


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
    def test_run_grid_rejects_noncanonical_authority_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "W2D-E1-MANIFEST.json"
            with mock.patch.object(
                runner.analysis,
                "git_dirty",
                side_effect=AssertionError(
                    "execution must reject before inspecting the worktree"
                ),
            ):
                with self.assertRaisesRegex(
                    runner.RunnerError,
                    "canonical committed tier manifest",
                ):
                    asyncio.run(
                        runner.run_grid(
                            paths={},
                            manifest_path=manifest,
                            backend_name="inmemory",
                        )
                    )

    @staticmethod
    def _minimal_tier_manifests(root: Path) -> dict[str, Path]:
        manifests: dict[str, Path] = {}
        for tier, contract in runner.EXECUTION_TIER_CONTRACTS.items():
            path = root / contract["manifest_filename"]
            path.write_bytes(
                runner._strict_json_bytes(
                    {
                        "schema_version": runner.MANIFEST_SCHEMA,
                        "measurement_name": runner.MEASUREMENT_NAME,
                        "execution_tier": tier,
                        "expected_result_filename": contract[
                            "result_filename"
                        ],
                        "expected_runtime_backend": dict(
                            contract["runtime_backend"]
                        ),
                        "artifact_hashes": {},
                        "runtime_fingerprint": {"sha256": "a" * 64},
                    }
                )
            )
            manifests[tier] = path
        return manifests

    @staticmethod
    def _raw_retrieval_event():
        return {
            "query_started_s": 0.0,
            "t": 0.001,
            "kind": "craft",
            "qidx": 0,
            "returned_ids": [0, 1, 2, 3, 4],
            "eligible_topk_ids": [0, 1, 2, 3, 4],
            "poisonfree_topk_ids": [0, 1, 2, 3, 4],
            "poison_ids_retrieved": [],
            "recall": 1.0,
            "displacement": 0.0,
        }

    def test_retrieval_evidence_requires_five_unique_mapped_ids(self):
        class Frozen:
            poison_truth = {}
            landed = {}

        id_to_key = {index: key(f"background-{index}") for index in range(5)}
        record = runner._retrieval_records(
            raw_events=[self._raw_retrieval_event()],
            id_to_key=id_to_key,
            frozen=Frozen(),
        )[0]
        self.assertEqual(
            record["returned_top5_item_keys"],
            [id_to_key[index] for index in range(5)],
        )
        for invalid in ([0, 1, 2, 3], [0, 1, 2, 3, 3], [0, 1, 2, 3, 9]):
            event = self._raw_retrieval_event()
            event["returned_ids"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(
                runner.RunnerError
            ):
                runner._retrieval_records(
                    raw_events=[event],
                    id_to_key=id_to_key,
                    frozen=Frozen(),
                )

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

    def test_postflight_rejects_artifact_and_code_drift(self):
        for filename in ("artifact.json", "runtime.py"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / filename
                path.write_bytes(b"frozen\n")
                registry = runner.SnapshotRegistry()
                registry.capture(path)
                path.write_bytes(b"changed\n")
                with self.assertRaisesRegex(
                    runner.RunnerError, "snapshot changed during replay"
                ):
                    runner._assert_snapshot_registry_unchanged(registry)

    def test_postflight_rejects_dirty_code_before_registry_check(self):
        registry = mock.Mock(spec=runner.SnapshotRegistry)
        with mock.patch.object(
            runner,
            "_uncached_git_dirty_files",
            return_value=["w2d_runner.py"],
        ):
            with self.assertRaisesRegex(
                runner.RunnerError, "changed during replay"
            ):
                runner._assert_formal_snapshots_unchanged(registry)
            registry.assert_unchanged.assert_not_called()

    def test_postflight_worktree_query_bypasses_analysis_cache(self):
        runner.analysis._GIT["dirty"] = False
        runner.analysis._GIT["files"] = []
        with mock.patch.object(
            runner.subprocess,
            "check_output",
            return_value=b" M w2d_runner.py\n",
        ) as check:
            self.assertEqual(
                runner._uncached_git_dirty_files(),
                ["w2d_runner.py"],
            )
        check.assert_called_once()

    def test_fingerprint_markdown_is_part_of_formal_clean_tree(self):
        with mock.patch.object(
            runner.subprocess,
            "check_output",
            return_value=(
                b"?? prototype/W2D-PREREGISTRATION-AMENDMENT-A12.md\n"
            ),
        ):
            self.assertEqual(
                runner._uncached_git_dirty_files(),
                ["prototype/W2D-PREREGISTRATION-AMENDMENT-A12.md"],
            )

    def test_manifest_build_rejects_uncommitted_code_before_reading_inputs(self):
        with mock.patch.object(
            runner,
            "_uncached_git_dirty_files",
            return_value=["w2d_runner.py"],
        ):
            with self.assertRaisesRegex(
                runner.RunnerError,
                "manifest requires a committed code tree",
            ):
                runner.build_manifest(
                    paths={},
                    manifest_path="unused-manifest.json",
                )

    def test_authority_binds_both_canonical_manifest_byte_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifests = self._minimal_tier_manifests(root)
            with mock.patch.object(
                runner,
                "_uncached_git_dirty_files",
                return_value=[],
            ):
                authority = runner.build_manifest_authority(
                    manifest_paths=manifests
                )
            self.assertEqual(
                set(authority["manifests"]),
                set(runner.EXECUTION_TIER_CONTRACTS),
            )
            for tier, manifest in manifests.items():
                self.assertEqual(
                    authority["manifests"][tier],
                    {
                        "path": manifest.name,
                        "sha256": hashlib.sha256(
                            manifest.read_bytes()
                        ).hexdigest(),
                    },
                )
            authority_path = root / runner.MANIFEST_AUTHORITY_BASENAME
            self.assertTrue(authority_path.is_file())

    def test_runner_binding_rejects_backend_or_authority_manifest_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifests = self._minimal_tier_manifests(root)
            with mock.patch.object(
                runner,
                "_uncached_git_dirty_files",
                return_value=[],
            ):
                runner.build_manifest_authority(manifest_paths=manifests)

            paths = {
                "output": str(
                    root
                    / runner.EXECUTION_TIER_CONTRACTS["E1_INMEMORY"][
                        "result_filename"
                    ]
                )
            }
            patches = (
                mock.patch.object(runner, "ARTIFACT_NAMES", {}),
                mock.patch.object(runner, "_capture_artifacts", return_value={}),
                mock.patch.object(runner, "_artifact_hashes", return_value={}),
                mock.patch.object(runner, "_artifact_documents", return_value={}),
                mock.patch.object(
                    runner,
                    "_capture_runtime_fingerprint",
                    return_value=({"sha256": "a" * 64}, {}),
                ),
                mock.patch.object(
                    runner,
                    "_validate_artifact_chain",
                    return_value={},
                ),
            )
            for patcher in patches:
                patcher.start()
            try:
                binding = runner._validate_manifest_binding(
                    paths=paths,
                    manifest_path=manifests["E1_INMEMORY"],
                    backend_name="inmemory",
                )
                self.assertEqual(
                    binding.manifest_sha256,
                    hashlib.sha256(
                        manifests["E1_INMEMORY"].read_bytes()
                    ).hexdigest(),
                )
                with self.assertRaisesRegex(
                    runner.RunnerError,
                    "manifest identity changed",
                ):
                    runner._validate_manifest_binding(
                        paths={
                            "output": str(
                                root
                                / runner.EXECUTION_TIER_CONTRACTS[
                                    "E2_MILVUS_LITE"
                                ]["result_filename"]
                            )
                        },
                        manifest_path=manifests["E1_INMEMORY"],
                        backend_name="milvus",
                    )

                manifests["E1_INMEMORY"].write_bytes(
                    manifests["E1_INMEMORY"].read_bytes() + b" "
                )
                with self.assertRaisesRegex(
                    runner.RunnerError,
                    "authority binding failed",
                ):
                    runner._validate_manifest_binding(
                        paths=paths,
                        manifest_path=manifests["E1_INMEMORY"],
                        backend_name="inmemory",
                    )
            finally:
                for patcher in reversed(patches):
                    patcher.stop()

    def test_failed_verified_output_is_preserved_for_quarantine(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            registry = runner.SnapshotRegistry()
            with mock.patch.object(
                runner, "verify_or_raise", side_effect=ValueError("reject")
            ):
                with self.assertRaisesRegex(ValueError, "reject"):
                    runner._write_verified_result(
                        result={"finite": 1.0},
                        output_path=output,
                        manifest_path=Path(tmp) / "manifest.json",
                        registry=registry,
                    )
            self.assertTrue(output.exists())
            self.assertIn(b'"finite": 1.0', output.read_bytes())

    def test_foreign_output_replacement_survives_failed_verification(self):
        replacement = b"foreign replacement survives\n"
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            registry = runner.SnapshotRegistry()

            def replace_then_fail(_manifest, result_path):
                target = Path(result_path)
                target.unlink()
                target.write_bytes(replacement)
                raise ValueError("reject")

            with mock.patch.object(
                runner, "verify_or_raise", side_effect=replace_then_fail
            ):
                with self.assertRaisesRegex(
                    runner.OutputOwnershipError, "ownership conflict"
                ):
                    runner._write_verified_result(
                        result={"finite": 1.0},
                        output_path=output,
                        manifest_path=Path(tmp) / "manifest.json",
                        registry=registry,
                    )
            self.assertEqual(output.read_bytes(), replacement)

    def test_runtime_backend_provenance_is_exact_and_omits_uri(self):
        self.assertEqual(
            runner._runtime_backend_descriptor("inmemory"),
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
        )
        backend = mock.Mock()
        backend.index_info.return_value = {"index_type_effective": "FLAT"}
        backend.server_info.return_value = {"deployment_mode": "lite"}
        with mock.patch.object(
            runner.importlib_metadata,
            "version",
            side_effect=["2.4.15", "2.4.12"],
        ):
            milvus = runner._runtime_backend_descriptor(
                "milvus", backend=backend, uri="/tmp/formal.db"
            )
        self.assertEqual(
            milvus,
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
        self.assertNotIn("uri", milvus)

    def test_milvus_provenance_rejects_non_flat_effective_index(self):
        backend = mock.Mock()
        backend.index_info.return_value = {
            "index_type_effective": "AUTOINDEX"
        }
        with self.assertRaisesRegex(runner.RunnerError, "effective FLAT"):
            runner._runtime_backend_descriptor(
                "milvus", backend=backend, uri="/tmp/formal.db"
            )

    def test_pre_runtime_chain_fails_closed_on_a9_mutation(self):
        documents = {
            "plan_pre_a9": {},
            "plan": {},
            "landing_pre_a9": {},
            "landing": {},
        }
        with mock.patch.object(
            runner,
            "validate_a9_replay_equivalence",
            side_effect=A9EquivalenceError("scientific field moved"),
        ) as gate:
            with self.assertRaisesRegex(
                runner.RunnerError,
                "A9 replay equivalence failed",
            ):
                runner._validate_artifact_chain(
                    paths={},
                    hashes={},
                    documents=documents,
                    artifact_snapshots={},
                    runtime_snapshots={},
                    registry=runner.SnapshotRegistry(),
                )
        gate.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
