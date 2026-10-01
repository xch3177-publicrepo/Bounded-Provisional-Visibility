#!/usr/bin/env python3
"""Tests for formal one-byte-instance snapshots."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from w2d_snapshot import (
    OutputOwnershipError,
    SnapshotRegistry,
    SnapshotError,
    assert_owned_output,
    assert_snapshots_unchanged,
    exclusive_create_bytes,
    preserve_failed_output,
    snapshot_file,
    snapshot_files,
)


class SnapshotTest(unittest.TestCase):
    def test_payload_and_hash_are_the_same_byte_instance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            path.write_bytes(b'{"value":1}\n')
            snapshot = snapshot_file(path)
            self.assertEqual(snapshot.payload, b'{"value":1}\n')
            assert_snapshots_unchanged([snapshot])

    def test_content_or_identity_replacement_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            path.write_bytes(b'{"value":1}\n')
            snapshot = snapshot_file(path)
            path.write_bytes(b'{"value":2}\n')
            with self.assertRaisesRegex(SnapshotError, "changed after snapshot"):
                assert_snapshots_unchanged([snapshot])
            path.unlink()
            path.write_bytes(snapshot.payload)
            with self.assertRaisesRegex(SnapshotError, "changed after snapshot"):
                assert_snapshots_unchanged([snapshot])

    def test_duplicate_resolved_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input"
            path.write_bytes(b"x")
            with self.assertRaisesRegex(SnapshotError, "duplicate"):
                snapshot_files([path, path])

    def test_registry_reuses_one_resolved_byte_instance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "input"
            alias = root / "." / "input"
            path.write_bytes(b"first\n")
            registry = SnapshotRegistry()
            first = registry.capture(path)
            path.write_bytes(b"second\n")
            second = registry.capture(alias)
            self.assertIs(first, second)
            self.assertEqual(second.payload, b"first\n")
            with self.assertRaisesRegex(SnapshotError, "changed after snapshot"):
                registry.assert_unchanged()

    def test_failed_owned_output_is_preserved_for_explicit_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "output.json"
            ownership = exclusive_create_bytes(path, b'{"value":1}\n')
            assert_owned_output(ownership)
            preserve_failed_output(ownership)
            self.assertEqual(path.read_bytes(), b'{"value":1}\n')

    def test_replaced_or_modified_output_survives_failure_handling(self) -> None:
        replacement = b"user replacement must survive\n"
        for mode in ("different inode", "changed payload"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "output.json"
                ownership = exclusive_create_bytes(path, b'{"value":1}\n')
                if mode == "different inode":
                    path.unlink()
                path.write_bytes(replacement)
                with self.assertRaisesRegex(
                    OutputOwnershipError, "output ownership conflict"
                ):
                    preserve_failed_output(ownership)
                self.assertEqual(path.read_bytes(), replacement)

    def test_exclusive_create_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "output.json"
            path.write_bytes(b"existing\n")
            with self.assertRaises(FileExistsError):
                exclusive_create_bytes(path, b"replacement\n")
            self.assertEqual(path.read_bytes(), b"existing\n")

    def test_substitution_before_failure_check_survives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "output.json"
            ownership = exclusive_create_bytes(path, b'{"value":1}\n')
            replacement = b"user replacement must survive\n"
            path.unlink()
            path.write_bytes(replacement)
            with self.assertRaisesRegex(
                OutputOwnershipError, "output ownership conflict"
            ):
                preserve_failed_output(ownership)
            self.assertEqual(path.read_bytes(), replacement)

    def test_failure_handling_never_calls_unlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "output.json"
            ownership = exclusive_create_bytes(path, b'{"value":1}\n')
            with patch.object(Path, "unlink") as unlink:
                preserve_failed_output(ownership)
            unlink.assert_not_called()
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
