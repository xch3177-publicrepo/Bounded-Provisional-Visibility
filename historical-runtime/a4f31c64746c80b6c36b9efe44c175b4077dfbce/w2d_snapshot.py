#!/usr/bin/env python3
"""One-byte-instance snapshots for formal W2D readers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
from typing import Iterable


class SnapshotError(RuntimeError):
    """A formal input could not be read as one stable byte instance."""


class OutputOwnershipError(RuntimeError):
    """A writer can no longer prove that an output belongs to this invocation."""


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    payload: bytes
    sha256: str
    device: int
    inode: int


@dataclass(frozen=True)
class OutputOwnership:
    """Identity and payload of an exclusively-created regular-file output."""

    path: Path
    sha256: str
    device: int
    inode: int


def snapshot_file(path: str | os.PathLike[str]) -> FileSnapshot:
    target = Path(path).resolve()
    try:
        with target.open("rb") as stream:
            before = os.fstat(stream.fileno())
            payload = stream.read()
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise SnapshotError(f"cannot snapshot {target}: {exc}") from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if not stat.S_ISREG(after.st_mode):
        raise SnapshotError(f"formal snapshot is not a regular file: {target}")
    if identity_before != identity_after or len(payload) != after.st_size:
        raise SnapshotError(f"file changed while being snapshotted: {target}")
    return FileSnapshot(
        path=target,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        device=after.st_dev,
        inode=after.st_ino,
    )


def snapshot_files(
    paths: Iterable[str | os.PathLike[str]],
) -> dict[Path, FileSnapshot]:
    result: dict[Path, FileSnapshot] = {}
    for path in paths:
        snapshot = snapshot_file(path)
        if snapshot.path in result:
            raise SnapshotError(f"duplicate formal snapshot path: {snapshot.path}")
        result[snapshot.path] = snapshot
    return result


class SnapshotRegistry:
    """Capture each resolved formal path once and reuse that byte instance."""

    def __init__(self) -> None:
        self._snapshots: dict[Path, FileSnapshot] = {}

    def capture(
        self,
        path: str | os.PathLike[str],
    ) -> FileSnapshot:
        resolved = Path(path).resolve()
        existing = self._snapshots.get(resolved)
        if existing is not None:
            return existing
        snapshot = snapshot_file(resolved)
        self._snapshots[resolved] = snapshot
        return snapshot

    def values(self) -> tuple[FileSnapshot, ...]:
        return tuple(self._snapshots.values())

    def assert_unchanged(self) -> None:
        assert_snapshots_unchanged(self.values())


def assert_snapshots_unchanged(
    snapshots: Iterable[FileSnapshot],
) -> None:
    for snapshot in snapshots:
        current = snapshot_file(snapshot.path)
        if (
            current.sha256 != snapshot.sha256
            or current.device != snapshot.device
            or current.inode != snapshot.inode
        ):
            raise SnapshotError(
                f"formal input changed after snapshot: {snapshot.path}"
            )


def _ownership_error(
    ownership: OutputOwnership,
    reason: str,
) -> OutputOwnershipError:
    return OutputOwnershipError(
        "output ownership conflict: preserving "
        f"{ownership.path}: {reason}"
    )


def assert_owned_output(ownership: OutputOwnership) -> None:
    """Require the path to name the exact regular-file payload we created."""

    try:
        current = ownership.path.lstat()
    except FileNotFoundError as exc:
        raise _ownership_error(ownership, "created output disappeared") from exc
    if not stat.S_ISREG(current.st_mode):
        raise _ownership_error(
            ownership, "output path no longer names a regular file"
        )
    try:
        snapshot = snapshot_file(ownership.path)
    except SnapshotError as exc:
        raise _ownership_error(
            ownership, "output payload cannot be snapshotted"
        ) from exc
    if (snapshot.device, snapshot.inode) != (
        ownership.device,
        ownership.inode,
    ):
        raise _ownership_error(ownership, "output path names a different file")
    if snapshot.sha256 != ownership.sha256:
        raise _ownership_error(ownership, "created output payload changed")


def preserve_failed_output(ownership: OutputOwnership) -> None:
    """Validate but never delete an output from a failed invocation.

    POSIX has no atomic compare-and-unlink operation.  Once a pathname is
    public, any check followed by unlink has a final substitution window.
    Failure handling therefore leaves the path untouched; a caller may
    quarantine it only through an explicit later administrative action.
    """
    assert_owned_output(ownership)


def exclusive_create_bytes(
    path: str | os.PathLike[str],
    payload: bytes,
) -> OutputOwnership:
    """Exclusive-create, fsync, and return proof of the exact output payload."""

    if type(payload) is not bytes:
        raise TypeError("exclusive output payload must be bytes")
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    ownership: OutputOwnership | None = None
    try:
        with target.open("xb") as stream:
            created = os.fstat(stream.fileno())
            if not stat.S_ISREG(created.st_mode):
                raise OutputOwnershipError(
                    "exclusive output is not a newly created regular file"
                )
            ownership = OutputOwnership(
                path=target,
                sha256=hashlib.sha256(payload).hexdigest(),
                device=created.st_dev,
                inode=created.st_ino,
            )
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        assert_owned_output(ownership)
        return ownership
    except BaseException as exc:
        if ownership is not None:
            try:
                preserve_failed_output(ownership)
            except OutputOwnershipError as conflict:
                raise conflict from exc
        raise
