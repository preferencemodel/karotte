"""User mounts staged by the launcher and copied into place in the guest.

A VM runtime whose bind mounts ignore guest ownership and mode (Apple
`container`'s virtiofs) mounts each user path under a root-only directory
instead of at its target. At start, root copies it to the target, with the
host's mode minus setuid/setgid. A read-write mount's copy belongs to the
student, who may write it as through a bind mount. A read-only mount's copy
belongs to root and is made read-only in the guest too: no group or other
write bits, and a read-only bind mount over it. At the end, read-write mounts
are copied back: regular files and directories only, from a target with no
link anywhere in its path, so a link the student planted can't redirect the
write on the host or pull in a directory it points at.

The launcher lists the mounts in ``KAROTTE_STAGED_MOUNTS``; without it,
nothing here runs.
"""

import json
import os
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from loguru import logger

from karotte.container import demoted_uid_gid
from karotte.process_utils import kill_processes
from karotte.trusted_bin import trusted_binary

STAGED_MOUNTS_ENV_VAR = "KAROTTE_STAGED_MOUNTS"

_MODE_BITS = 0o777


@dataclass(frozen=True)
class StagedMount:
    source: str
    """Where the launcher mounted the host path, under a root-only directory."""
    target: str
    """Where the task expects it."""
    writable: bool


def encode(mounts: list[StagedMount]) -> str:
    return json.dumps([asdict(m) for m in mounts])


def staged_mounts() -> list[StagedMount]:
    value = os.environ.get(STAGED_MOUNTS_ENV_VAR)
    if not value:
        return []
    return [StagedMount(**m) for m in json.loads(value)]


_copied: dict[str, list[Path]] = {}
"""What :func:`copy_in` put under each writable directory mount's target,
relative to it, so :func:`copy_back` can tell a file the student deleted from
one that was never there."""


def copy_in() -> None:
    """Copy every staged mount to its target, outer targets before the mounts
    nested in them."""
    mounts = sorted(staged_mounts(), key=lambda m: len(Path(m.target).parts))
    student = _student_id() if any(m.writable for m in mounts) else None
    for mount in mounts:
        source, target = Path(mount.source), Path(mount.target)
        owner = student if mount.writable and student is not None else 0
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            _ = shutil.copytree(
                source,
                target,
                symlinks=True,
                copy_function=shutil.copyfile,
                dirs_exist_ok=True,
            )
            _own_as(source, target, owner)
            if mount.writable:
                _copied[mount.target] = _entries(source)
        else:
            _ = shutil.copyfile(source, target)
            _chown(target, source.lstat(), owner)
        if not mount.writable:
            _make_read_only(target)
        logger.info(f"Copied mount {mount.target} into place")


@contextmanager
def copied_back() -> Iterator[None]:
    """Copy the writable staged mounts back to the host on the way out."""
    try:
        yield
    finally:
        copy_back()


def copy_back() -> None:
    writable = [m for m in staged_mounts() if m.writable]
    if not writable:
        return
    # A student process still running could swap a path component for a link
    # between the check and the copy; nothing the student starts outlives the
    # run anyway.
    student = _student_id()
    if student is not None:
        kill_processes(student)
    targets = [Path(m.target) for m in staged_mounts()]
    for mount in writable:
        target = Path(mount.target)
        # Another mount inside this one goes back to its own source only.
        nested = {
            t.relative_to(target)
            for t in targets
            if t != target and t.is_relative_to(target)
        }
        try:
            skipped = _copy_back(target, Path(mount.source), nested)
            # Not through a link the student made: what's missing there says
            # nothing about what the student deleted.
            if not (target.is_symlink() or _student_link_above(target)):
                _remove_deleted(
                    target, Path(mount.source), nested, _copied.get(mount.target, [])
                )
        except OSError as exc:
            logger.warning(f"Could not copy mount {mount.target} back: {exc}")
            continue
        if skipped:
            logger.warning(
                f"Mount {mount.target}: skipped {skipped} links and special files on copy-back"
            )
        logger.info(f"Copied mount {mount.target} back to the host")


def _student_id() -> int | None:
    try:
        return demoted_uid_gid()
    except RuntimeError as exc:
        logger.warning(
            f"No student to give writable mounts to ({exc}); they stay root's"
        )
        return None


def _own_as(source: Path, target: Path, owner: int) -> None:
    """Give every copied entry to ``owner`` (uid and gid) with the source's
    mode bits."""
    for dirpath, dirnames, filenames in os.walk(source):
        rel = Path(dirpath).relative_to(source)
        for name in [".", *dirnames, *filenames]:
            src = source / rel / name
            _chown(target / rel / name, src.lstat(), owner)


def _make_read_only(target: Path) -> None:
    """Keep the student from writing a read-only mount's copy: take away group
    and other write bits (the copy is root's), then bind-mount it over itself
    read-only, as a read-only bind mount would have been."""
    _strip_write_bits(target)
    mount = trusted_binary("mount")
    for argv in (
        [mount, "--bind", str(target), str(target)],
        [mount, "-o", "remount,ro,bind", str(target)],
    ):
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            logger.warning(
                f"Could not mount {target} read-only ({result.stderr.strip()}); its write bits are removed instead"
            )
            return


def _strip_write_bits(target: Path) -> None:
    paths = [target]
    if target.is_dir() and not target.is_symlink():
        for dirpath, dirnames, filenames in os.walk(target):
            paths += [Path(dirpath) / name for name in (*dirnames, *filenames)]
    for path in paths:
        st = path.lstat()
        if not stat.S_ISLNK(st.st_mode):
            os.chmod(path, stat.S_IMODE(st.st_mode) & ~(stat.S_IWGRP | stat.S_IWOTH))


def _chown(path: Path, source_stat: os.stat_result, owner: int) -> None:
    os.lchown(path, owner, owner)
    if not stat.S_ISLNK(source_stat.st_mode):
        os.chmod(path, source_stat.st_mode & _MODE_BITS)


def _entries(root: Path) -> list[Path]:
    """Every entry under ``root``, relative to it, without following links."""
    entries: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        entries += [rel / name for name in (*dirnames, *filenames)]
    return entries


def _under(rel: Path, nested: set[Path]) -> bool:
    return any(rel == n or rel.is_relative_to(n) for n in nested)


def _remove_deleted(
    source: Path, dest: Path, nested: set[Path], copied: list[Path]
) -> None:
    """Remove from ``dest`` what was copied in and the student deleted, as a
    bind mount would have. Only entries copied in: a file added on the host
    during the run stays. Deepest first, so a deleted directory is empty by
    the time it is removed; one that isn't (the host added to it) stays."""
    if not source.is_dir() or source.is_symlink():
        return
    for rel in sorted(copied, key=lambda p: len(p.parts), reverse=True):
        if _under(rel, nested) or os.path.lexists(source / rel):
            continue
        path = dest / rel
        try:
            st = path.lstat()
        except FileNotFoundError:
            continue
        try:
            if stat.S_ISDIR(st.st_mode):
                path.rmdir()
            else:
                path.unlink()
        except OSError as exc:
            logger.warning(f"Could not remove {path}, deleted in the sandbox: {exc}")


def _copy_back(source: Path, dest: Path, nested: set[Path] | None = None) -> int:
    """Copy regular files and directories from ``source`` over ``dest``,
    without following links on either side, leaving out the ``nested`` paths
    (relative to ``source``). Returns how many entries were skipped."""
    nested = nested or set()
    # Before is_dir(), which follows a link: a target the student swapped for
    # a link to a directory would otherwise be walked, as root, and exported.
    # A link the student made higher up counts too: with /workdir/results a
    # link to a root-only directory, /workdir/results/data is that directory's
    # entry. copy_back has killed the student's processes, so nothing swaps a
    # component in between.
    if source.is_symlink() or _student_link_above(source):
        return 1
    if not source.is_dir():
        if not source.is_file():
            return 1
        _copy_file_back(source, dest)
        return 0
    skipped = 0
    for dirpath, dirnames, filenames in os.walk(source):
        rel = Path(dirpath).relative_to(source)
        dirnames[:] = [d for d in dirnames if not _under(rel / d, nested)]
        out_dir = dest / rel
        if out_dir.is_symlink():
            skipped += 1
            dirnames.clear()
            continue
        out_dir.mkdir(exist_ok=True)
        for name in list(dirnames):
            if (Path(dirpath) / name).is_symlink():
                skipped += 1
                dirnames.remove(name)
        for name in filenames:
            if _under(rel / name, nested):
                continue
            path = Path(dirpath) / name
            st = path.lstat()
            if not stat.S_ISREG(st.st_mode):
                skipped += 1
                continue
            _copy_file_back(path, out_dir / name)
    return skipped


def _student_link_above(path: Path) -> bool:
    """Whether a directory above ``path`` is a link root didn't make. Root's
    own (``/var`` on macOS, links in the image) are left alone: the student
    can't create a link owned by root."""
    for parent in path.parents:
        try:
            st = parent.lstat()
        except OSError:
            return True
        if stat.S_ISLNK(st.st_mode) and st.st_uid != 0:
            return True
    return False


def _copy_file_back(source: Path, dest: Path) -> None:
    """Write a temporary file beside ``dest`` and rename it over ``dest``: a
    copy cut short (an interrupt, the VM killed) leaves the host's file as it
    was rather than truncated. The rename also replaces a link at ``dest``
    instead of writing through it."""
    in_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(in_fd, "rb") as src:
        mode = os.fstat(src.fileno()).st_mode & _MODE_BITS
        out_fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.")
        try:
            with os.fdopen(out_fd, "wb") as out:
                os.fchmod(out.fileno(), mode)
                shutil.copyfileobj(src, out)
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp, dest)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
