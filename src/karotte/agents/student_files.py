"""Reading files a CLI agent wrote as the student, without following links."""

import os
import stat
from collections.abc import Callable, Generator
from pathlib import Path

from loguru import logger

from karotte.subprocess import demoted_uid_gid
from karotte.untrusted_paths import DIR_FLAGS, FILE_FLAGS, open_at, walk_to_parent

_MAX_LINE_BYTES = 16 * 1024**2
_MAX_FILE_BYTES = 256 * 1024**2
_SCAN_ENTRIES = 10_000
_SCAN_BYTES = 256 * 1024**2


class ScanBudget:
    """What one scan of the student's folders may still read: folders and files
    opened plus entries listed, and bytes, so files the student plants can't stall it."""

    def __init__(self) -> None:
        self.entries: int = _SCAN_ENTRIES
        self.bytes: int = _SCAN_BYTES
        self._warned: bool = False

    def take_entry(self) -> bool:
        """Spends one entry; False once the entries or bytes are spent."""
        if self.entries > 0 and self.bytes > 0:
            self.entries -= 1
            return True
        if not self._warned:
            self._warned = True
            logger.warning(
                "Stopped scanning student files at {} entries or {} bytes",
                _SCAN_ENTRIES,
                _SCAN_BYTES,
            )
        return False


def open_student_file(path: Path) -> int | None:
    """The file, opened without following a link anywhere on its path, or None
    while it doesn't exist. Raises OSError unless it is a regular file the
    student owns."""
    try:
        parent, name = walk_to_parent(path.absolute())
    except FileNotFoundError:
        return None
    try:
        fd = open_at(parent, name, FILE_FLAGS)
    except FileNotFoundError:
        return None
    finally:
        os.close(parent)
    info = os.fstat(fd)
    uid = demoted_uid_gid()
    if not stat.S_ISREG(info.st_mode) or (uid is not None and info.st_uid != uid):
        os.close(fd)
        raise OSError(f"{path} is not a regular file the student owns")
    return fd


def student_file_lines(
    path: Path, max_bytes: int = _MAX_FILE_BYTES, budget: ScanBudget | None = None
) -> Generator[str, None, None]:
    """The file's lines up to `max_bytes` and what `budget` has left, skipping
    overlong ones; none if it is missing or refused."""
    if budget is not None:
        if not budget.take_entry():
            return
        max_bytes = min(max_bytes, budget.bytes)
    try:
        fd = open_student_file(path)
    except OSError as e:
        logger.warning("Not reading {}: {}", path, e)
        return
    if fd is None:
        return
    read = 0
    try:
        with os.fdopen(fd, "rb") as f:
            while read < max_bytes:
                line = f.readline(min(_MAX_LINE_BYTES + 1, max_bytes - read))
                if not line:
                    return
                read += len(line)
                if len(line) > _MAX_LINE_BYTES:
                    while line and not line.endswith(b"\n") and read < max_bytes:
                        line = f.readline(min(_MAX_LINE_BYTES, max_bytes - read))
                        read += len(line)
                    logger.warning(
                        "Skipping a line over {} bytes in {}", _MAX_LINE_BYTES, path
                    )
                    continue
                if line.endswith(b"\n"):
                    yield line[:-1].decode("utf-8", errors="replace")
                elif read < max_bytes:
                    yield line.decode("utf-8", errors="replace")
    finally:
        if budget is not None:
            budget.bytes -= read
    if read >= max_bytes:
        logger.warning("Stopped reading {} after {} bytes", path, max_bytes)


def student_file_first_line(path: Path, budget: ScanBudget) -> str | None:
    """The file's first line, read no further than the longest line allowed."""
    lines = student_file_lines(path, _MAX_LINE_BYTES + 1, budget)
    try:
        return next(lines, None)
    finally:
        lines.close()


def student_subdirs(path: Path, budget: ScanBudget) -> list[str]:
    """Names of the directories directly in `path`; none if it is missing or
    reached through a link."""
    return _entries(path, lambda e: e.is_dir(follow_symlinks=False), budget)


def student_files_in(path: Path, budget: ScanBudget) -> list[str]:
    """Names of the regular files directly in `path`; none if it is missing or
    reached through a link."""
    return _entries(path, lambda e: e.is_file(follow_symlinks=False), budget)


def _entries(
    path: Path, keep: Callable[[os.DirEntry[str]], bool], budget: ScanBudget
) -> list[str]:
    if not budget.take_entry():
        return []
    try:
        parent, name = walk_to_parent(path.absolute())
    except OSError:
        return []
    try:
        fd = open_at(parent, name, DIR_FLAGS)
    except OSError:
        return []
    finally:
        os.close(parent)
    names: list[str] = []
    try:
        # `keep` may stat an entry through `fd`, so it runs before `fd` closes.
        with os.scandir(fd) as entries:
            for entry in entries:
                if not budget.take_entry():
                    break
                if keep(entry):
                    names.append(entry.name)
    finally:
        os.close(fd)
    return names
