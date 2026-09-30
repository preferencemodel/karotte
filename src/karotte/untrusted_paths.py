"""Filesystem primitives for a path a student controls.

Nothing here follows a symlink, and `probe` never raises: stating *through*
whatever the student left at a name is how root ends up reading a file it was
never meant to, or dying on a `PermissionError` the caller did not expect.
"""

import os
from pathlib import Path
from typing import Final

FILE_FLAGS: Final = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_NOCTTY
DIR_FLAGS: Final = os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY


def probe(path: Path | str) -> os.stat_result | None:
    """What sits at exactly this path, or None if it cannot be looked at.

    A symlink reports as a symlink rather than as its target, and every failure
    a student can provoke — an unreadable parent, a loop, a missing or
    non-directory component — comes back as None.
    """
    try:
        return os.lstat(path)
    except (OSError, ValueError):
        return None


def open_at(dir_fd: int, name: str, flags: int) -> int:
    """Open `name` inside `dir_fd`, refusing a symlink at that name."""
    return os.open(name, flags, dir_fd=dir_fd)


def lstat_at(dir_fd: int, name: str) -> os.stat_result:
    return os.lstat(name, dir_fd=dir_fd)


def walk_to_parent(path: Path) -> tuple[int, str]:
    """Open the path's parent directory component by component, refusing a
    symlink along the way. Returns (parent dir fd, final component name).

    A raised `OSError` carries the walked-to component in `filename`.
    """
    names = list(path.parts[1:])
    if not names:
        raise ValueError("path must not be the filesystem root")
    final = names.pop()

    fd = os.open("/", DIR_FLAGS)
    walked = "/"
    for name in names:
        walked = os.path.join(walked, name)
        try:
            next_fd = open_at(fd, name, DIR_FLAGS)
        except OSError as e:
            os.close(fd)
            e.filename = walked
            raise
        except BaseException:
            os.close(fd)
            raise
        os.close(fd)
        fd = next_fd
    return fd, final
