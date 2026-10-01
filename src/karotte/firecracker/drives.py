"""Per-run drives, made with the host's e2fsprogs: the scratch drive under the
guest's overlay, the io drive that carries inputs in and /out back, and one
read-only drive per mounted directory or file.

``mkfs.ext4 -d <dir>`` and ``debugfs`` need no root. Reading /out back with
``debugfs rdump`` after the VM exits copies only regular files and
directories, with the host user's ownership and default modes.
"""

import os
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MIB = 1 << 20
GIB = 1 << 30

IO_DRIVE_BYTES = 16 * GIB
"""Sparse size of the io drive; /out holds the transcript and artifacts."""

_SBIN_DIRS = ("/usr/sbin", "/sbin", "/usr/local/sbin")


class DriveError(RuntimeError):
    pass


def e2fs_tool(name: str) -> str:
    """Path of an e2fsprogs tool; user PATHs often lack the sbin dirs."""
    path = shutil.which(name) or shutil.which(name, path=os.pathsep.join(_SBIN_DIRS))
    if path is None:
        raise DriveError(f"{name} not found; install e2fsprogs")
    return path


def _run(argv: list[str]) -> None:
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise DriveError(
            f"{' '.join(argv)} failed: {result.stderr.strip() or result.stdout.strip()}"
        )


def _sparse_file(path: Path, size: int) -> None:
    with path.open("wb") as f:
        f.truncate(size)


def make_scratch_drive(path: Path, size: int) -> None:
    """An empty sparse ext4 of ``size`` bytes: the hard cap on what the guest
    can write to the host's disk."""
    _sparse_file(path, size)
    _run(
        [
            e2fs_tool("mkfs.ext4"),
            "-q",
            "-F",
            "-L",
            "karotte-scratch",
            "-O",
            "^has_journal",
            "-E",
            "lazy_itable_init=1",
            str(path),
        ]
    )


def make_io_drive(path: Path, content: Path, size: int = IO_DRIVE_BYTES) -> None:
    _sparse_file(path, size)
    _run(
        [
            e2fs_tool("mkfs.ext4"),
            "-q",
            "-F",
            "-L",
            "karotte-io",
            "-O",
            "^has_journal",
            "-E",
            "root_owner=0:0",
            "-d",
            str(content),
            str(path),
        ]
    )


@dataclass(frozen=True)
class MountDrive:
    path: Path
    kind: Literal["dir", "file"]
    name: str
    """For a file, its name on the drive; ``.`` for a directory."""


def _tree_size(root: Path) -> tuple[int, int]:
    total = 0
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        count += len(dirnames)
        for name in filenames:
            count += 1
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total, count


def mount_drive_size(content_bytes: int, entries: int) -> int:
    size = int(content_bytes * 1.25) + entries * 8192 + 64 * MIB
    return -(-size // MIB) * MIB


def make_mount_drive(path: Path, source: Path) -> MountDrive:
    """A read-only drive holding a copy of ``source`` as it is now."""
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".mount-") as tmp:
        if source.is_dir():
            content, kind, name = source, "dir", "."
        else:
            content, kind, name = Path(tmp), "file", source.name
            shutil.copyfile(source, content / name)
            # Keep it executable if it was; setuid and setgid stay behind.
            os.chmod(content / name, stat.S_IMODE(source.stat().st_mode) & 0o777)
        content_bytes, entries = _tree_size(content)
        _sparse_file(path, mount_drive_size(content_bytes, entries))
        _run(
            [
                e2fs_tool("mkfs.ext4"),
                "-q",
                "-F",
                "-O",
                "^has_journal",
                "-N",
                str(entries + 64),
                "-E",
                "root_owner=0:0",
                "-d",
                str(content),
                str(path),
            ]
        )
    _run([e2fs_tool("debugfs"), "-w", "-R", "rmdir /lost+found", str(path)])
    return MountDrive(path=path, kind=kind, name=name)


def read_exit_code(io_drive: Path) -> int | None:
    """What the guest's run exited with, or ``None`` if it never finished."""
    result = subprocess.run(
        [e2fs_tool("debugfs"), "-R", "cat /status/exit_code", str(io_drive)],
        capture_output=True,
        text=True,
        check=False,
    )
    text = result.stdout.strip()
    return int(text) if text.lstrip("-").isdigit() else None


def copy_out(io_drive: Path, dest: Path, scratch_dir: Path) -> None:
    """Copy the guest's /out into ``dest``."""
    with tempfile.TemporaryDirectory(dir=scratch_dir, prefix=".out-") as tmp:
        # rdump warns it can't chown as non-root, and copies anyway.
        subprocess.run(
            [e2fs_tool("debugfs"), "-R", f"rdump /out {tmp}", str(io_drive)],
            capture_output=True,
            check=False,
        )
        copy_plain(Path(tmp) / "out", dest)


def copy_plain(src: Path, dest: Path) -> None:
    """Copy regular files and directories only: no symlinks, devices, owners
    or mode bits from the guest."""
    if not src.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(src):
        rel = Path(dirpath).relative_to(src)
        target_dir = dest / rel
        # A link already on the host at a path the guest names would carry
        # the copy outside ``dest``.
        if rel != Path(".") and target_dir.is_symlink():
            dirnames.clear()
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        dirnames[:] = [
            d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))
        ]
        for name in filenames:
            path = os.path.join(dirpath, name)
            if stat.S_ISREG(os.lstat(path).st_mode):
                out = target_dir / name
                if out.is_symlink():
                    out.unlink()
                shutil.copyfile(path, out)
