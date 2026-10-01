"""Writes that are on disk when the call returns."""

import itertools
import os
from pathlib import Path

from loguru import logger


def write_durably(path: Path, text: str) -> None:
    """Write ``text`` and fsync it, so it is on disk before the sandbox can be
    torn down. A VM powered off right after the run loses unsynced writes.

    Creates missing parent directories. A file's fsync doesn't cover its
    directory entry, so the parent is synced too, and so is the parent of
    every directory created here."""
    created = list(itertools.takewhile(lambda p: not p.exists(), path.parents))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        _ = f.write(text)
        f.flush()
        os.fsync(f.fileno())
    for directory in dict.fromkeys([path.parent, *(d.parent for d in created)]):
        _fsync_directory(directory)


def _fsync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError as exc:
        # Some filesystems refuse fsync on a directory (EINVAL).
        logger.debug(f"Could not fsync {directory}: {exc}")
    finally:
        os.close(fd)
