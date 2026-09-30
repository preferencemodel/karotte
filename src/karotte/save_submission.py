import errno
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from loguru import logger

from karotte.student_misbehavior import StudentMisbehaviorError
from karotte.untrusted_paths import DIR_FLAGS as _DIR_FLAGS
from karotte.untrusted_paths import FILE_FLAGS as _FILE_FLAGS
from karotte.untrusted_paths import lstat_at, open_at, walk_to_parent

SUBMISSIONS_DIR: Final = Path("~/.config/karotte/submissions").expanduser()

_CHUNK_SIZE = 1 << 20

_DEST_ERRNOS: Final = frozenset(
    {errno.ENAMETOOLONG, errno.EEXIST, errno.ENOSPC, errno.EDQUOT, errno.EFBIG}
)


@dataclass
class _CopyState:
    allow_symlinks: bool
    max_file_bytes: int
    max_total_bytes: int
    max_entries: int
    max_depth: int
    total_bytes: int = 0
    entries: int = 0

    def count_entry(self) -> None:
        self.entries += 1
        if self.entries > self.max_entries:
            raise StudentMisbehaviorError(
                f"submission has more than {self.max_entries} entries"
            )


def save_submission(
    source: Path | str,
    dest: Path | str | None = None,
    *,
    allow_symlinks: bool = False,
    max_file_bytes: int = 256 * 1024**2,
    max_total_bytes: int = 1024**3,
    max_entries: int = 10_000,
    max_depth: int = 32,
) -> Path:
    """Copy a student submission (file or directory) to `dest` — or, when `dest`
    is omitted, into a fresh directory under `SUBMISSIONS_DIR` that gets
    returned — without trusting its contents; everything is created root-only
    (0700 dirs, 0600 files) and `dest` must not exist. Symlinks (unless
    `allow_symlinks`), special files, oversized or too-deep trees, and
    student-provokable dest failures (disk exhaustion, name collisions,
    over-long paths) raise :class:`StudentMisbehaviorError`; call
    `kill_processes` first so no student process mutates the tree mid-copy.

    Missing submissions are not treated as student misbehavior. It is up to the
    judge to determine how to handle a missing submission.
    """
    source = Path(os.path.abspath(source))
    created_dir: Path | None = None
    if dest is None:
        with _dest_failure_is_misbehavior(str(SUBMISSIONS_DIR)):
            SUBMISSIONS_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
            created_dir = Path(tempfile.mkdtemp(dir=SUBMISSIONS_DIR))
        dest = created_dir / source.name
    else:
        dest = Path(dest)
        if os.path.lexists(dest):
            raise FileExistsError(f"{dest} already exists")

    state = _CopyState(
        allow_symlinks=allow_symlinks,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
        max_entries=max_entries,
        max_depth=max_depth,
    )

    try:
        _copy_source(source, dest, state)
    except BaseException:
        if created_dir is not None:
            shutil.rmtree(created_dir, ignore_errors=True)
        elif os.path.lexists(dest):
            if dest.is_dir() and not dest.is_symlink():
                shutil.rmtree(dest, ignore_errors=True)
            else:
                dest.unlink(missing_ok=True)
        raise
    return created_dir or dest


def _copy_source(source: Path, dest: Path, state: _CopyState) -> None:
    parent_fd, final = _walk_to_parent(source)
    try:
        try:
            st = lstat_at(parent_fd, final)
        except FileNotFoundError:
            logger.info(f"Nothing handed in at {source}")
            return
        except OSError as e:
            raise StudentMisbehaviorError(f"cannot stat {source}: {e.strerror}") from e
        if stat.S_ISLNK(st.st_mode):
            raise StudentMisbehaviorError(f"{source} is a symlink")
        if stat.S_ISDIR(st.st_mode):
            fd = _open_at(parent_fd, final, _DIR_FLAGS, str(source))
            try:
                with _dest_failure_is_misbehavior(str(dest)):
                    os.mkdir(dest, 0o700)
                _copy_dir(fd, dest, 1, state, rel="")
            finally:
                os.close(fd)
        elif stat.S_ISREG(st.st_mode):
            fd = _open_at(parent_fd, final, _FILE_FLAGS, str(source))
            try:
                _copy_file(fd, dest, state, str(source))
            finally:
                os.close(fd)
        else:
            raise StudentMisbehaviorError(f"{source} is not a regular file")
    finally:
        os.close(parent_fd)


def _walk_to_parent(source: Path) -> tuple[int, str]:
    try:
        return walk_to_parent(source)
    except OSError as e:
        raise _cannot_open(e, e.filename or str(source)) from e


def _copy_dir(
    dir_fd: int, dest_dir: Path, depth: int, state: _CopyState, rel: str
) -> None:
    if depth > state.max_depth:
        raise StudentMisbehaviorError(f"submission is nested too deep at {rel}")

    names = []
    with os.scandir(dir_fd) as entries:
        for entry in entries:
            state.count_entry()
            names.append(entry.name)

    for name in sorted(names):
        rel_child = f"{rel}/{name}" if rel else name
        dest_child = dest_dir / name
        st = _lstat_at(dir_fd, name, rel_child)

        if stat.S_ISLNK(st.st_mode):
            if not state.allow_symlinks:
                raise StudentMisbehaviorError(f"{rel_child} is a symlink")
            with _dest_failure_is_misbehavior(rel_child):
                os.symlink(os.readlink(name, dir_fd=dir_fd), dest_child)
        elif stat.S_ISDIR(st.st_mode):
            fd = _open_at(dir_fd, name, _DIR_FLAGS, rel_child)
            try:
                with _dest_failure_is_misbehavior(rel_child):
                    os.mkdir(dest_child, 0o700)
                _copy_dir(fd, dest_child, depth + 1, state, rel_child)
            finally:
                os.close(fd)
        elif stat.S_ISREG(st.st_mode):
            fd = _open_at(dir_fd, name, _FILE_FLAGS, rel_child)
            try:
                _copy_file(fd, dest_child, state, rel_child)
            finally:
                os.close(fd)
        else:
            raise StudentMisbehaviorError(f"{rel_child} is not a regular file")


def _copy_file(fd: int, dest: Path, state: _CopyState, rel: str) -> None:
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode):
        raise StudentMisbehaviorError(f"{rel} is not a regular file")
    if st.st_size > state.max_file_bytes:
        raise StudentMisbehaviorError(
            f"{rel} is too large ({st.st_size} > {state.max_file_bytes} bytes)"
        )

    file_bytes = 0
    with _dest_failure_is_misbehavior(rel):
        out_fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with _dest_failure_is_misbehavior(rel):
            while chunk := os.read(fd, _CHUNK_SIZE):
                file_bytes += len(chunk)
                state.total_bytes += len(chunk)
                if file_bytes > state.max_file_bytes:
                    raise StudentMisbehaviorError(
                        f"{rel} is too large (> {state.max_file_bytes} bytes)"
                    )
                if state.total_bytes > state.max_total_bytes:
                    raise StudentMisbehaviorError(
                        f"submission total size exceeds {state.max_total_bytes} bytes"
                    )
                if chunk.count(0) == len(chunk):
                    os.lseek(out_fd, len(chunk), os.SEEK_CUR)
                else:
                    view = memoryview(chunk)
                    while view:
                        view = view[os.write(out_fd, view) :]
            os.ftruncate(out_fd, file_bytes)
    finally:
        os.close(out_fd)


@contextmanager
def _dest_failure_is_misbehavior(rel: str) -> Iterator[None]:
    try:
        yield
    except OSError as e:
        if e.errno in _DEST_ERRNOS:
            raise StudentMisbehaviorError(f"cannot save {rel}: {e.strerror}") from e
        raise


def _lstat_at(dir_fd: int, name: str, described: str) -> os.stat_result:
    try:
        return lstat_at(dir_fd, name)
    except FileNotFoundError as e:
        raise StudentMisbehaviorError(f"{described} does not exist") from e
    except OSError as e:
        raise StudentMisbehaviorError(f"cannot stat {described}: {e.strerror}") from e


def _open_at(dir_fd: int, name: str, flags: int, described: str) -> int:
    try:
        return open_at(dir_fd, name, flags)
    except OSError as e:
        raise _cannot_open(e, described) from e


def _cannot_open(e: OSError, described: str) -> StudentMisbehaviorError:
    if e.errno == errno.ELOOP:
        return StudentMisbehaviorError(f"{described} is a symlink")
    if e.errno == errno.ENOENT:
        return StudentMisbehaviorError(f"{described} does not exist")
    if e.errno == errno.ENOTDIR:
        return StudentMisbehaviorError(f"{described} is not a directory")
    return StudentMisbehaviorError(f"cannot open {described}: {e.strerror}")
