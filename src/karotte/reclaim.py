"""Free what a uid still holds once its processes are gone."""

import contextlib
import ctypes
import errno
import os
import stat
import time
import warnings
from functools import cache
from pathlib import Path
from typing import final

from loguru import logger

from karotte.container import is_containerized
from karotte.file_quota import QUOTA_DIR
from karotte.memory_watch import SYSVIPC_SHM_PATH
from karotte.student_misbehavior import StudentMisbehaviorError

_IPC_RMID = 0
_IPC_STAT = 2
_IPC_PRIVATE = 0
_IPC_CREAT = 0o1000

_MAX_PROBED_SHMID = 1 << 20
"""How many ids the walk checks when there is no ``/proc/sysvipc/shm`` to read.
A uid can churn ids to push its own past this cap and keep that segment."""

SWEEP_TIMEOUT_SECONDS = 600.0
"""How long a sweep may run before it counts as the uid stalling it. Well above
a full sweep of the largest tree a uid may hold, and well under the grading
timeout that would otherwise fail the run as infra."""

_CAP_IPC_OWNER = 15

MOUNTS_PATH = Path("/proc/self/mounts")
STATUS_PATH = Path("/proc/self/status")

DEFAULT_INCLUDE = (Path("/"),)

DEFAULT_EXCLUDE = (Path("/root"), QUOTA_DIR)
"""Spared unless the caller says otherwise: root's own files, and the file
quota's upper layers — those hold the same files as the overlays mounted over
them, and reaching behind a mounted overlay to change its upper layer is not
something overlayfs supports."""

EXCLUDE_ENV_VAR = "KAROTTE_RECLAIM_EXCLUDE"
"""Extra paths every sweep spares, ``os.pathsep``-separated. Set by an image
that hosts things karotte did not create but must outlive a sweep, e.g. an
external harness's log mounts or the agent it installed as the student."""


def _env_excludes() -> tuple[Path, ...]:
    raw = os.environ.get(EXCLUDE_ENV_VAR, "")
    return tuple(Path(part) for part in raw.split(os.pathsep) if part)


_VIRTUAL_FSTYPES = frozenset(
    {
        "autofs",
        "binfmt_misc",
        "bpf",
        "cgroup",
        "cgroup2",
        "configfs",
        "debugfs",
        "devpts",
        "devtmpfs",
        "efivarfs",
        "fusectl",
        "hugetlbfs",
        "mqueue",
        "nsfs",
        "proc",
        "pstore",
        "rpc_pipefs",
        "securityfs",
        "selinuxfs",
        "sysfs",
        "tracefs",
    }
)


class ReclaimError(StudentMisbehaviorError):
    """Some of what the uid owns could not be deleted."""


@final
class _Deadline:
    """When the sweep has to give up."""

    __slots__ = ("_timeout", "_end")

    def __init__(self, timeout: float | None):
        self._timeout = timeout
        self._end = None if timeout is None else time.monotonic() + timeout

    def passed(self) -> bool:
        return self._end is not None and time.monotonic() > self._end

    def __str__(self) -> str:
        return f"{self._timeout:g}s"


@final
class _IpcPerm(ctypes.Structure):
    """The head of ``struct ipc_perm``, five 32-bit fields on every Linux ABI
    karotte runs on."""

    _fields_ = (
        ("key", ctypes.c_int32),
        ("uid", ctypes.c_uint32),
        ("gid", ctypes.c_uint32),
        ("cuid", ctypes.c_uint32),
        ("cgid", ctypes.c_uint32),
    )


@final
class _ShmidDs(ctypes.Structure):
    """``struct shmid_ds``, of which only the permissions are read. The tail is
    room for whatever the ABI puts after them."""

    _fields_ = (("perm", _IpcPerm), ("_rest", ctypes.c_byte * 256))


def delete_files(
    uid: int,
    include: tuple[Path, ...] | None = None,
    exclude: tuple[Path, ...] = DEFAULT_EXCLUDE,
    extend_exclude: tuple[Path, ...] = (),
    directories: tuple[Path, ...] | None = None,
    timeout: float | None = SWEEP_TIMEOUT_SECONDS,
) -> int:
    """Delete everything ``uid`` owns under ``include`` (by default the whole
    filesystem, sparing read-only and virtual filesystems, the
    :data:`DEFAULT_EXCLUDE` paths, anything at or under an
    ``exclude``/``extend_exclude`` path, and the paths named in
    :data:`EXCLUDE_ENV_VAR`), plus the SysV shared-memory segments it created. Returns how many things were removed; raises
    :class:`ReclaimError` if anything else of the uid's survived, or if the
    sweep ran past ``timeout`` seconds (``None`` waits as long as it takes).

    Sweeping the whole filesystem is refused outside a karotte container, where
    the uid is likely a real user of the machine.
    """
    if directories is not None:
        if include is not None:
            raise TypeError("pass either include or directories, not both")
        warnings.warn(
            "delete_files(directories=...) is deprecated; use include=...",
            DeprecationWarning,
            stacklevel=2,
        )
        include = directories
    if include is None:
        include = DEFAULT_INCLUDE
    roots = tuple(os.path.abspath(path) for path in include)
    if "/" in roots and not is_containerized():
        logger.warning(
            f"Not sweeping the whole filesystem outside a karotte container: uid {uid} may be a real user of this machine"
        )
        return 0
    excluded = tuple(
        os.path.abspath(path) for path in exclude + extend_exclude + _env_excludes()
    )
    mounts = _mount_table(require_mounts="/" in roots)
    deadline = _Deadline(timeout)
    removed, failed = _delete_owned(uid, roots, excluded, mounts, deadline)
    removed += _remove_sysv_segments(uid)
    if removed:
        logger.info(f"Deleted {removed} thing(s) owned by uid {uid}")
    if failed:
        shown = ", ".join(failed[:10])
        if len(failed) > 10:
            shown += f", and {len(failed) - 10} more"
        raise ReclaimError(
            f"Could not delete {len(failed)} path(s) owned by uid {uid}: {shown}"
        )
    return removed


_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC


@final
class _Frame:
    """A directory being swept. Only the top of the stack holds an open fd and
    its path; a parent gets both back on the way out."""

    __slots__ = (
        "fd",
        "path",
        "name",
        "owned",
        "names",
        "next",
        "dev",
        "ino",
        "sweepable",
    )

    def __init__(
        self,
        fd: int,
        path: str,
        name: str,
        owned: bool,
        names: list[str],
        sweepable: bool,
    ):
        self.fd = fd
        self.path = path
        self.name = name
        self.owned = owned
        self.names = names
        self.sweepable = sweepable
        self.next = 0
        st = os.fstat(fd)
        self.dev, self.ino = st.st_dev, st.st_ino


@final
class _Mounts:
    """Which filesystem a path sits on and whether that one may be swept.
    Read-only and virtual filesystems may not."""

    __slots__ = ("_sweepable", "_points")

    def __init__(self, sweepable: dict[str, bool]):
        self._sweepable = sweepable
        self._points = tuple(point for point, ok in sweepable.items() if ok)

    def point(self, path: str) -> bool:
        """Whether something is mounted exactly at ``path``, which makes it
        none of a sweep's business to remove."""
        return path in self._sweepable

    def at(self, path: str, inherited: bool) -> bool:
        """Whether what sits at ``path`` may be swept: the mount there says so,
        or, with nothing mounted there, the filesystem it is part of."""
        return self._sweepable.get(path, inherited)

    def under(self, path: str) -> bool:
        """Whether a sweepable filesystem is mounted somewhere below ``path``,
        which is reason enough to walk through a filesystem that is not."""
        prefix = path if path.endswith("/") else path + "/"
        return any(point.startswith(prefix) for point in self._points)

    def around(self, path: str) -> bool:
        """Whether the filesystem holding ``path`` may be swept, taken from the
        longest mount point it sits under. For a sweep root, which need not be
        a mount point itself."""
        longest = ""
        sweepable = True
        for point, ok in self._sweepable.items():
            if len(point) < len(longest):
                continue
            if path == point or path.startswith(point.rstrip("/") + "/"):
                longest, sweepable = point, ok
        return sweepable


def _mount_table(require_mounts: bool) -> _Mounts:
    """Every mount point, and whether its filesystem may be swept."""
    try:
        lines = MOUNTS_PATH.read_text().splitlines()
    except OSError as exc:
        if require_mounts:
            raise ReclaimError(
                f"cannot read {MOUNTS_PATH} to find unsweepable mounts: {exc}"
            ) from exc
        return _Mounts({})
    sweepable: dict[str, bool] = {}
    for line in lines:
        fields = line.split()
        if len(fields) < 4:
            continue
        fstype, options = fields[2], fields[3]
        sweepable[_unescape(fields[1])] = not (
            fstype in _VIRTUAL_FSTYPES or "ro" in options.split(",")
        )
    return _Mounts(sweepable)


def _unescape(field: str) -> str:
    """Undo the octal escapes /proc/self/mounts uses for whitespace."""
    escapes = (("\\011", "\t"), ("\\012", "\n"), ("\\040", " "), ("\\134", "\\"))
    for escape, char in escapes:
        field = field.replace(escape, char)
    return field


def _delete_owned(
    uid: int,
    roots: tuple[str, ...],
    excluded: tuple[str, ...],
    mounts: _Mounts,
    deadline: _Deadline,
) -> tuple[int, list[str]]:
    removed = 0
    failed: list[str] = []
    for root in roots:
        removed += _sweep(uid, root, excluded, mounts, failed, deadline, removed)
    return removed, failed


def _is_excluded(path: str, excluded: tuple[str, ...]) -> bool:
    return any(path == e or path.startswith(e + "/") for e in excluded)


def _child_path(parent: str, name: str) -> str:
    return f"/{name}" if parent == "/" else f"{parent}/{name}"


def _parent_path(frame: _Frame) -> str:
    return frame.path[: -len(frame.name) - 1] or "/"


def _sweep(
    uid: int,
    root: str,
    excluded: tuple[str, ...],
    mounts: _Mounts,
    failed: list[str],
    deadline: _Deadline,
    already_removed: int = 0,
) -> int:
    """Delete with dir-fd-relative syscalls only: each component is bounded by
    NAME_MAX, so a deep tree cannot push a call past PATH_MAX."""
    try:
        fd = os.open(root, _DIR_FLAGS)
    except OSError:
        return 0
    try:
        names = os.listdir(fd)
    except OSError:
        os.close(fd)
        return 0
    removed = 0
    stack = [_Frame(fd, root, root, False, names, mounts.around(root))]
    try:
        while stack:
            frame = stack[-1]
            if deadline.passed():
                done = already_removed + removed
                raise ReclaimError(
                    f"Sweeping what uid {uid} owns took longer than {deadline}; deleted {done} thing(s) before giving up at {frame.path}"
                )
            if frame.next < len(frame.names):
                name = frame.names[frame.next]
                frame.next += 1
                removed += _sweep_entry(
                    uid, frame, name, stack, excluded, mounts, failed
                )
                continue
            stack.pop()
            if not stack:
                os.close(frame.fd)
                break
            parent = stack[-1]
            parent.path = _parent_path(frame)
            try:
                parent.fd = _reopen_parent(frame, parent)
            except OSError:
                failed.append(parent.path)
                return removed
            if not frame.owned:
                continue
            try:
                os.rmdir(frame.name, dir_fd=parent.fd)
                removed += 1
            except OSError as exc:
                if exc.errno not in (errno.ENOENT, errno.ENOTEMPTY):
                    failed.append(frame.path)
    except BaseException:
        for frame in stack:
            if frame.fd >= 0:
                with contextlib.suppress(OSError):
                    os.close(frame.fd)
        raise
    return removed


def _sweep_entry(
    uid: int,
    frame: _Frame,
    name: str,
    stack: list[_Frame],
    excluded: tuple[str, ...],
    mounts: _Mounts,
    failed: list[str],
) -> int:
    path = _child_path(frame.path, name)
    if _is_excluded(path, excluded):
        return 0
    try:
        st = os.stat(name, dir_fd=frame.fd, follow_symlinks=False)
    except FileNotFoundError:
        return 0
    except OSError:
        if not frame.sweepable:
            return 0
        failed.append(path)
        return 0
    if stat.S_ISDIR(st.st_mode):
        return _enter(uid, frame, name, path, stack, mounts, failed)
    if st.st_uid != uid or not mounts.at(path, frame.sweepable):
        return 0
    try:
        os.unlink(name, dir_fd=frame.fd)
        return 1
    except FileNotFoundError:
        return 0
    except OSError:
        failed.append(path)
        return 0


def _enter(
    uid: int,
    parent: _Frame,
    name: str,
    path: str,
    stack: list[_Frame],
    mounts: _Mounts,
    failed: list[str],
) -> int:
    """Descend even into other uids' dirs: uid files can live there. A dir we
    cannot see into might hold the uid's files, so it fails the reclaim. A
    filesystem that may not be swept is walked only to reach one that may, and
    nothing in it counts as missed."""
    sweepable = mounts.at(path, parent.sweepable)
    if not sweepable and not mounts.under(path):
        return 0
    try:
        fd = os.open(name, _DIR_FLAGS | os.O_NOFOLLOW, dir_fd=parent.fd)
    except FileNotFoundError:
        return 0
    except OSError:
        if sweepable:
            failed.append(path)
        return 0
    entry_uid = os.fstat(fd).st_uid
    try:
        names = os.listdir(fd)
    except OSError:
        os.close(fd)
        if sweepable:
            failed.append(path)
        return 0
    owned = sweepable and entry_uid == uid and not mounts.point(path)
    child = _Frame(fd, path, name, owned, names, sweepable)
    os.close(parent.fd)
    parent.fd = -1
    parent.path = ""
    stack.append(child)
    return 0


def _reopen_parent(frame: _Frame, parent: _Frame) -> int:
    """Climb back out through ".." and check we surfaced where we dove in."""
    try:
        fd = os.open("..", _DIR_FLAGS, dir_fd=frame.fd)
    finally:
        os.close(frame.fd)
    st = os.fstat(fd)
    if (st.st_dev, st.st_ino) != (parent.dev, parent.ino):
        os.close(fd)
        raise OSError(errno.ESTALE, "directory moved during sweep", parent.path)
    return fd


def _remove_sysv_segments(uid: int) -> int:
    """Mark the segments the uid created for destruction.

    Charged to the creator (``cuid``), consistent with how the memory watch
    weighs them: the owner field is the segment's to rewrite, the creator is
    not.
    """
    shmids = _listed_segments(uid)
    if shmids is None:
        return _remove_unlisted_segments(uid)
    return sum(1 for shmid in shmids if _shmctl_rmid(shmid))


def _listed_segments(uid: int) -> list[int] | None:
    """The uid's segments as ``/proc/sysvipc/shm`` lists them, or ``None``
    where the kernel does not publish the file, as gVisor's does not."""
    try:
        rows = SYSVIPC_SHM_PATH.read_text().splitlines()[1:]
    except OSError:
        return None
    shmids: list[int] = []
    for row in rows:
        fields = row.split()
        try:
            shmid, creator = int(fields[1]), int(fields[9])
        except (IndexError, ValueError):
            continue
        if creator == uid:
            shmids.append(shmid)
    return shmids


def _remove_unlisted_segments(uid: int) -> int:
    """Destroy the uid's segments on a kernel that lists none of them, as
    gVisor's does not. Reading a segment's creator needs read permission on it,
    so the walk runs as the uid in a child unless we hold CAP_IPC_OWNER."""
    ceiling = _allocation_ceiling()
    if ceiling is None:
        return 0
    last = min(ceiling, _MAX_PROBED_SHMID)
    if ceiling > last:
        logger.warning(
            f"Only the first {last} SysV ids were walked, of {ceiling} handed out; segments the uid parked above that stay"
        )
    if os.geteuid() == uid or _holds_ipc_owner():
        return _destroy_owned_segments(uid, last)
    if os.geteuid() != 0:
        logger.warning(
            f"Cannot become uid {uid} to find its SysV segments, and this kernel lists none"
        )
        return 0
    return _destroy_owned_segments_as_uid(uid, last)


def _holds_ipc_owner() -> bool:
    """Whether we may read any segment's creator, which also reaches the ones a
    uid made unreadable to itself."""
    try:
        status = STATUS_PATH.read_text()
    except OSError:
        return False
    for line in status.splitlines():
        if line.startswith("CapEff:"):
            return bool(int(line.split(":")[1], 16) >> _CAP_IPC_OWNER & 1)
    return False


def _destroy_owned_segments(uid: int, last: int) -> int:
    """Destroy every segment up to ``last`` that ``uid`` created. Runs with
    credentials that may read the creator, the uid's own or CAP_IPC_OWNER."""
    destroyed = 0
    for shmid in range(last + 1):
        perm = _shmctl_stat(shmid)
        if perm is not None and perm.cuid == uid and _shmctl_rmid(shmid, quiet=True):
            destroyed += 1
    return destroyed


def _destroy_owned_segments_as_uid(uid: int, last: int) -> int:
    """Run :func:`_destroy_owned_segments` in a child that has become the uid,
    and read back how many fell. The child only makes syscalls, since logging
    or allocating across a fork in a threaded process risks deadlock."""
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        destroyed = -1
        try:
            os.setgroups([])
            os.setgid(uid)
            os.setuid(uid)
            destroyed = _destroy_owned_segments(uid, last)
        except BaseException:
            destroyed = -1
        finally:
            try:
                _ = os.write(write_fd, str(destroyed).encode())
            except OSError:
                pass
            os._exit(0)
    os.close(write_fd)
    try:
        with os.fdopen(read_fd, "rb") as reported:
            answer = reported.read()
    finally:
        _ = os.waitpid(pid, 0)
    try:
        destroyed = int(answer)
    except ValueError:
        destroyed = -1
    if destroyed < 0:
        logger.warning(f"Could not walk the SysV id space as uid {uid}")
        return 0
    return destroyed


def _allocation_ceiling() -> int | None:
    """An id above every one in use, learned by taking the next one: ids only
    ever climb. A uid that has pinned every segment the kernel allows leaves
    none to take, and the walk then runs to its own bound instead."""
    libc = _libc()
    if libc is None:
        return None
    ctypes.set_errno(0)
    shmid = libc.shmget(_IPC_PRIVATE, 1, _IPC_CREAT | 0o600)
    if shmid < 0:
        if ctypes.get_errno() != errno.ENOSPC:
            logger.warning(
                f"Could not take a SysV id to bound the walk: errno {ctypes.get_errno()}"
            )
            return None
        return _MAX_PROBED_SHMID
    _ = libc.shmctl(shmid, _IPC_RMID, None)
    return shmid


def _shmctl_stat(shmid: int) -> _IpcPerm | None:
    """The segment's permissions, or ``None`` if there is no such segment or it
    will not say."""
    libc = _libc()
    if libc is None:
        return None
    segment = _ShmidDs()
    if libc.shmctl(shmid, _IPC_STAT, ctypes.byref(segment)) != 0:
        return None
    return segment.perm


def _shmctl_rmid(shmid: int, quiet: bool = False) -> bool:
    """``shmctl(IPC_RMID)`` as a syscall of our own, rather than a call out to
    ``ipcrm``, so path and symlink hijacks have nothing to grab. A walk of the
    id space asks after ids that were never handed out, and says nothing when
    they turn out not to be there."""
    libc = _libc()
    if libc is None:
        return False
    if libc.shmctl(shmid, _IPC_RMID, None) != 0:
        if not quiet:
            logger.warning(
                f"Could not remove SysV segment {shmid}: errno {ctypes.get_errno()}"
            )
        return False
    return True


@cache
def _libc() -> ctypes.CDLL | None:
    try:
        return ctypes.CDLL(None, use_errno=True)
    except OSError as exc:
        logger.warning(f"No libc to call shmctl through: {exc}")
        return None
