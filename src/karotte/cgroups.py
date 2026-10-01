"""Create and enforce a student cgroup on whichever cgroup version the sandbox offers."""

from __future__ import annotations

import os
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

PROC_MOUNTS = Path("/proc/mounts")
HARNESS_LEAF = "karotte_harness"
"""Where the harness parks itself so the parent may delegate controllers."""

_FREEZE_TIMEOUT = 5.0
_REQUIRED = ("memory", "pids")

_NO_LIMIT_THRESHOLD = 1 << 60
"""Values at or past this read back as "no limit set"."""


@dataclass(frozen=True)
class Mount:
    path: Path
    fstype: str
    controllers: frozenset[str]
    read_only: bool

    @property
    def is_cgroup2(self) -> bool:
        return self.fstype == "cgroup2"


def parse_mounts(text: str) -> list[Mount]:
    """The cgroup mounts in a ``/proc/mounts`` dump.

    A v1 line names its controllers among the mount options; a v2 line has no
    controller list at all, so its options must not be read as one.
    """
    mounts: list[Mount] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[2] not in ("cgroup", "cgroup2"):
            continue
        options = parts[3].split(",")
        controllers = (
            frozenset()
            if parts[2] == "cgroup2"
            else frozenset(o for o in options if not _is_mount_flag(o))
        )
        mounts.append(Mount(Path(parts[1]), parts[2], controllers, "ro" in options))
    return mounts


_MOUNT_FLAGS = frozenset(
    {"rw", "ro", "nosuid", "nodev", "noexec", "relatime", "noatime", "seclabel"}
)


def _is_mount_flag(option: str) -> bool:
    return option in _MOUNT_FLAGS or option.startswith(("name=", "mode=", "size="))


def read_mounts() -> list[Mount]:
    try:
        return parse_mounts(PROC_MOUNTS.read_text())
    except OSError as exc:
        logger.warning(f"Could not read {PROC_MOUNTS}: {exc}")
        return []


def own_cgroup_dir(v2_root: Path, proc_self_cgroup: str | None = None) -> Path:
    """The v2 group our own processes are in, which is what may delegate.

    A karotte process that an earlier one moved into the harness leaf resolves
    to the group above it, so every karotte process creates the student group
    in the same place instead of nesting an unlimited one under the leaf.
    """
    if proc_self_cgroup is None:
        try:
            proc_self_cgroup = Path("/proc/self/cgroup").read_text()
        except OSError:
            return v2_root
    for line in proc_self_cgroup.splitlines():
        parts = line.split(":")
        if len(parts) == 3 and parts[0] == "0":
            path = v2_root / parts[2].lstrip("/")
            while path.name == HARNESS_LEAF and path != v2_root:
                path = path.parent
            return path
    return v2_root


@dataclass
class StudentCgroup:
    """A created group, plus the handful of things we do to it."""

    path: Path
    join_paths: list[Path]
    _freezer: Path | None = None
    _memory_limit_file: str = "memory.max"
    _no_memory_limit: str = "max"
    _swap_limit_file: str | None = "memory.swap.max"

    def join_self(self) -> None:
        """Put the calling process in the group so its descendants inherit membership."""
        for path in self.join_paths:
            (path / "cgroup.procs").write_text(str(os.getpid()))

    def preexec(self) -> Callable[[], None]:
        return self.join_self

    def set_memory_limit(self, nbytes: int | None) -> bool:
        """Cap the group's memory, or lift the cap when ``nbytes`` is ``None``."""
        value = self._no_memory_limit if nbytes is None else str(nbytes)
        if not self._write(self.path / self._memory_limit_file, value):
            return False
        # Without this, on a host with swap the group pages out past its cap
        # instead of being OOM-killed. Absent when swap accounting is off.
        if self._swap_limit_file is not None:
            swap = self.path / self._swap_limit_file
            if swap.exists():
                _ = self._write(swap, "max" if nbytes is None else "0")
        return True

    def swap_limit(self) -> str | None:
        """The raw swap cap, or ``None`` where this layout has no swap file."""
        if self._swap_limit_file is None:
            return None
        try:
            return (self.path / self._swap_limit_file).read_text().strip()
        except OSError:
            return None

    def set_swap_limit(self, value: str) -> bool:
        """Put back a raw swap cap that :meth:`swap_limit` read."""
        if self._swap_limit_file is None:
            return False
        return self._write(self.path / self._swap_limit_file, value)

    def set_process_limit(self, count: int | None) -> bool:
        """Cap the group's process count, or lift the cap when ``count`` is ``None``."""
        pids_dir = self._controller_dir("pids")
        return self._write(
            pids_dir / "pids.max", "max" if count is None else str(count)
        )

    def memory_limit(self) -> int | None:
        """The cap in force, or ``None`` where there is none (or none readable).

        v2 spells unlimited ``max``; v1 accepts ``-1`` but reads it back as
        PAGE_COUNTER_MAX, hence the threshold.
        """
        return self._read_limit(self.path / self._memory_limit_file)

    def process_limit(self) -> int | None:
        """The cap in force, or ``None`` where there is none (or none readable)."""
        return self._read_limit(self._controller_dir("pids") / "pids.max")

    def inherited_memory_limit(self) -> int | None:
        """The tightest memory cap on the groups above this one, or ``None``."""
        limits: list[int] = []
        path = self.path.parent
        while (path / self._memory_limit_file).exists():
            limit = self._read_limit(path / self._memory_limit_file)
            if limit is not None:
                limits.append(limit)
            path = path.parent
        return min(limits, default=None)

    @staticmethod
    def _read_limit(path: Path) -> int | None:
        try:
            raw = path.read_text().strip()
        except OSError:
            return None
        if raw == "max":
            return None
        try:
            value = int(raw)
        except ValueError:
            return None
        return None if value < 0 or value >= _NO_LIMIT_THRESHOLD else value

    def member_pids(self) -> list[int]:
        try:
            text = (self.join_paths[0] / "cgroup.procs").read_text()
        except (OSError, IndexError):
            return []
        return [int(pid) for pid in text.split() if pid.isdigit()]

    def kill_all(self) -> int:
        """Kill everything in the group; returns how many processes are left.

        Overridden per layout: v2 has one atomic write, v1 has to freeze first.
        """
        raise NotImplementedError

    def destroy(self) -> None:
        for path in self.join_paths:
            try:
                path.rmdir()
            except OSError:
                pass

    def _controller_dir(self, controller: str) -> Path:
        """Where a controller's files live. v2 keeps them all in one group;
        v1 spreads them across a hierarchy each."""
        del controller
        return self.path

    @staticmethod
    def _write(path: Path, value: str) -> bool:
        try:
            path.write_text(value)
            return True
        except OSError as exc:
            logger.warning(f"Could not write {value} to {path}: {exc}")
            return False


class _V2StudentCgroup(StudentCgroup):
    def kill_all(self) -> int:
        """One write, and the kernel drops the whole subtree at once."""
        if not self._write(self.path / "cgroup.kill", "1"):
            return len(self.member_pids())
        time.sleep(0.05)
        return len(self.member_pids())


class _V1StudentCgroup(StudentCgroup):
    def _controller_dir(self, controller: str) -> Path:
        for path in self.join_paths:
            if path.parent.name == controller:
                return path
        return self.path

    def kill_all(self) -> int:
        """Freeze, then kill what is inside.

        v1 has no ``cgroup.kill``; freezing first stops the list growing as we
        walk it, since a frozen process cannot fork.
        """
        if self._freezer is None:
            logger.warning("No freezer controller; killing by enumeration alone")
            return self._kill_listed()

        state = self._freezer / "freezer.state"
        if not self._write(state, "FROZEN"):
            return self._kill_listed()

        deadline = time.monotonic() + _FREEZE_TIMEOUT
        while time.monotonic() < deadline:
            try:
                if state.read_text().strip() == "FROZEN":
                    break
            except OSError:
                break
            time.sleep(0.05)

        remaining = self._kill_listed()
        # Thaw so the corpses can be reaped; a frozen zombie never leaves.
        _ = self._write(state, "THAWED")
        time.sleep(0.05)
        return len(self.member_pids()) if remaining else 0

    def _kill_listed(self) -> int:
        pids = self.member_pids()
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                continue
        return len(pids)


class V2Cgroup:
    """Unified hierarchy: one directory per group, all controllers together."""

    version: int = 2
    parent: Path

    def __init__(self, parent: Path) -> None:
        self.parent = parent

    def create(self, name: str) -> StudentCgroup:
        path = self.parent / name
        path.mkdir(exist_ok=True)
        self._delegate()
        self._kill_as_a_group(path)
        return _V2StudentCgroup(path=path, join_paths=[path])

    @staticmethod
    def _kill_as_a_group(path: Path) -> None:
        """An OOM kill in the group takes every process in it, not the largest
        alone, so no half-killed process tree is left behind."""
        try:
            (path / "memory.oom.group").write_text("1")
        except OSError as exc:
            logger.warning(f"Could not set memory.oom.group on {path}: {exc}")

    def _delegate(self) -> None:
        """Enable the controllers on children.

        Only the real root may both hold processes and delegate, so the harness
        moves into a leaf of its own first.
        """
        if self._has_procs():
            leaf = self.parent / HARNESS_LEAF
            try:
                leaf.mkdir(exist_ok=True)
                self._move_procs(self.parent, leaf)
            except OSError as exc:
                logger.warning(f"Could not move the harness to {leaf}: {exc}")

        subtree = self.parent / "cgroup.subtree_control"
        try:
            subtree.write_text(" ".join(f"+{c}" for c in _REQUIRED))
        except OSError as exc:
            logger.warning(f"Could not delegate cgroup controllers: {exc}")

    def _has_procs(self) -> bool:
        try:
            return bool((self.parent / "cgroup.procs").read_text().split())
        except OSError:
            return False

    @staticmethod
    def _move_procs(src: Path, dst: Path) -> None:
        """Move every process in ``src`` into ``dst``, one pid per write."""
        try:
            pids = (src / "cgroup.procs").read_text().split()
        except OSError:
            return
        target = dst / "cgroup.procs"
        for pid in pids:
            try:
                target.write_text(pid)
            except OSError:
                continue


class V1Cgroup:
    """Split hierarchies: one directory per controller, per group."""

    version: int = 1
    roots: dict[str, Path]

    def __init__(self, roots: dict[str, Path]) -> None:
        self.roots = roots

    def create(self, name: str) -> StudentCgroup:
        join_paths: list[Path] = []
        for path in self.roots.values():
            group = path / name
            group.mkdir(exist_ok=True)
            join_paths.append(group)

        memory = self.roots.get("memory")
        freezer = self.roots.get("freezer")
        return _V1StudentCgroup(
            path=(memory / name) if memory else join_paths[0],
            join_paths=join_paths,
            _freezer=(freezer / name) if freezer else None,
            _memory_limit_file="memory.limit_in_bytes",
            _no_memory_limit="-1",
            # memory.memsw.limit_in_bytes must stay at or above the limit, so
            # it can't simply be zeroed; v1 hosts keep swap as they are.
            _swap_limit_file=None,
        )


_student_groups: dict[int, StudentCgroup] = {}


def register_student_cgroup(uid: int, group: StudentCgroup) -> None:
    """Make ``uid``'s group findable by the kill path."""
    _student_groups[uid] = group


def unregister_student_cgroup(uid: int) -> None:
    """Forget ``uid``'s group, e.g. after destroying it."""
    _ = _student_groups.pop(uid, None)


def student_cgroup(uid: int) -> StudentCgroup | None:
    """The registered group confining ``uid``, or ``None``. Never creates one:
    a group no session has joined holds nothing to kill."""
    return _student_groups.get(uid)


def detect_cgroups(
    mounts: list[Mount] | None = None,
    own_cgroup: Path | None = None,
) -> V2Cgroup | V1Cgroup | None:
    """The usable cgroup backend, or ``None`` if this sandbox has none.

    Prefers v2 only when it carries the controllers: a controller belongs to
    one hierarchy at a time, so mounting memory and pids on v1 leaves a
    writable but useless cgroup2.
    """
    mounts = read_mounts() if mounts is None else mounts

    for mount in mounts:
        if not mount.is_cgroup2 or mount.read_only:
            continue
        parent = own_cgroup if own_cgroup is not None else own_cgroup_dir(mount.path)
        try:
            available = (parent / "cgroup.controllers").read_text().split()
        except OSError:
            continue
        if all(c in available for c in _REQUIRED):
            return V2Cgroup(parent)

    roots = {
        controller: mount.path
        for mount in mounts
        if not mount.is_cgroup2 and not mount.read_only
        for controller in mount.controllers
        if controller in (*_REQUIRED, "freezer")
    }
    if all(c in roots for c in _REQUIRED):
        return V1Cgroup(roots)

    return None
