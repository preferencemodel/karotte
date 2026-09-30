"""Watch what a uid is holding, in RAM or in files, and reap it past a limit."""

import os
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from karotte.process_utils import kill_processes

PROC_PATH = Path("/proc")
PROC_MOUNTS_PATH = Path("/proc/mounts")
SYSVIPC_SHM_PATH = Path("/proc/sysvipc/shm")

TEMP_DIRS = (Path("/tmp"), Path("/var/tmp"), Path("/dev/shm"))
"""World-writable directories a uid can hold RAM in, when they are tmpfs."""

POLL_INTERVAL_S = 0.1

MAX_UNCAPPED_PROCESSES = 10_000
"""Walk bound when the process cap is lifted. Weighing memory still has to
visit every process, so a big cohort makes the poll slower, not the reading
blinder — up to here, past which the reading gives up."""

MAX_TMPFS_ENTRIES = 10_000

MAX_UNCAPPED_FILES = 1_000_000
"""Walk bound for a file watch whose count cap is lifted, for the same reason
as ``MAX_UNCAPPED_PROCESSES``: the walk stays bounded for cost, but nobody is
blamed for its size."""

FILE_POLL_INTERVAL_S = 1.0
"""Slower than the memory poll: the walk visits every entry the uid may hold."""

# Entries another uid owns cost the same walk, so give up at this multiple of a
# budget rather than reading them all.
_VISITS_PER_BUDGET = 2

PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


@dataclass(frozen=True)
class _Counted:
    """One bounded count: bytes weighed, entries owned by the uid, entries visited."""

    bytes_held: int
    owned: int
    walked: int
    budget: int | None
    named: str

    @property
    def failure(self) -> str | None:
        """Why this count could not be bounded, if it could not.

        No budget means nothing to fail: the walk stays bounded for cost, but
        nobody is blamed for its size."""
        if self.budget is None or self.owned <= self.budget:
            return None
        return f"left more than {self.budget} {self.named}"


@dataclass(frozen=True)
class _Reading:
    """One weigh of everything a uid is keeping in RAM."""

    bytes_held: int
    counts: tuple[_Counted, ...]

    @property
    def failure(self) -> str | None:
        return next((count.failure for count in self.counts if count.failure), None)


_PROCESSES_NAMED = "processes running"
_TMPFS_NAMED = "entries in the RAM-backed temp directories"
_DISK_NAMED = "files on disk"


def _resident_bytes(pid: str) -> int:
    """RSS from ``statm``: cheap, but counts a shared page once per process
    mapping it."""
    try:
        with open(PROC_PATH / pid / "statm") as f:
            # Total, then resident, in pages.
            return int(f.read().split()[1]) * PAGE_SIZE
    except (OSError, IndexError, ValueError):
        return 0


def _pss_rollup(pid: str) -> int | None:
    """Pss from ``smaps_rollup``, which splits shared pages across their
    mappers but costs a page-table walk. ``None`` where it cannot be read
    (e.g. gVisor has no ``smaps_rollup``)."""
    try:
        with open(PROC_PATH / pid / "smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        pass
    return None


_SMAPS_VMA = re.compile(r"\A[0-9a-f]+-[0-9a-f]+ \S+ \S+ (\S+) (\d+) *(.*)")

_DEVICE_MEMORY_MAPPINGS = frozenset({"anon_inode:[vfio-device]"})
"""Accelerator memory mapped through VFIO, which gVisor reports as resident."""


@dataclass
class _Vma:
    """One mapping being read out of ``smaps``: the file behind it, if any."""

    file: str | None
    rss: int = 0
    anonymous: int = 0
    device_memory: bool = False


def _smaps_reading(pid: str) -> tuple[int, dict[str, int]] | None:
    """Private bytes and file-backed bytes by mapped file, from per-VMA
    ``smaps``, skipping device memory. ``None`` where it cannot be read."""
    vmas: list[_Vma] = []
    try:
        with open(PROC_PATH / pid / "smaps") as f:
            for line in f:
                if match := _SMAPS_VMA.match(line):
                    device, inode, path = match.groups()
                    vmas.append(
                        _Vma(
                            f"{device}:{inode}" if inode != "0" else None,
                            device_memory=path.rstrip() in _DEVICE_MEMORY_MAPPINGS,
                        )
                    )
                elif not vmas:
                    return None
                elif line.startswith("Rss:"):
                    vmas[-1].rss = int(line.split()[1]) * 1024
                elif line.startswith("Anonymous:"):
                    vmas[-1].anonymous = int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        return None
    private = 0
    by_file: dict[str, int] = {}
    for vma in vmas:
        if vma.device_memory:
            # pages copied out of a private mapping of the device are RAM
            private += min(vma.anonymous, vma.rss)
            continue
        if vma.file is None:
            private += vma.rss
        else:
            # CoW'd pages in a file mapping are this process's own
            private += min(vma.anonymous, vma.rss)
            by_file[vma.file] = by_file.get(vma.file, 0) + max(
                vma.rss - vma.anonymous, 0
            )
    return private, by_file


def _precise_bytes(pid: str, shared_files: dict[str, int]) -> int:
    """What one process holds, without counting a shared page once per mapper:
    Pss, or anonymous bytes with the file-backed bytes merged into
    ``shared_files`` so each file is charged once per uid. A process neither
    source can read charges nothing — its RSS confirms nothing."""
    pss = _pss_rollup(pid)
    if pss is not None:
        return pss
    reading = _smaps_reading(pid)
    if reading is None:
        return 0
    private, by_file = reading
    for key, held in by_file.items():
        shared_files[key] = max(shared_files.get(key, 0), held)
    return private


def _processes(
    uid: int, *, precise: bool = False, budget: int | None = None
) -> _Counted:
    """Weigh what the uid's processes hold resident.

    The walk is bounded by ``budget`` (or ``MAX_UNCAPPED_PROCESSES`` when there
    is no cap), so a lifted cap never weighs the whole process table.
    """
    walk_bound = MAX_UNCAPPED_PROCESSES if budget is None else budget
    bytes_held = 0
    owned = 0
    walked = 0
    shared_files: dict[str, int] = {}
    try:
        # iterate, don't list: a million-pid listing can't be bounded
        listing = os.scandir(PROC_PATH)
    except OSError:
        return _Counted(0, 0, 0, budget, _PROCESSES_NAMED)
    with listing:
        for entry in listing:
            if not entry.name.isdigit():
                continue
            walked += 1
            if walked > walk_bound * _VISITS_PER_BUDGET:
                break
            try:
                if entry.stat().st_uid != uid:
                    continue
            except OSError:
                continue
            # counted on the uid alone: it may die before its statm is read
            owned += 1
            # one past, so a cohort exactly on the budget still passes
            if owned > walk_bound:
                break
            if precise:
                bytes_held += _precise_bytes(entry.name, shared_files)
            else:
                bytes_held += _resident_bytes(entry.name)
    bytes_held += sum(shared_files.values())
    return _Counted(bytes_held, owned, walked, budget, _PROCESSES_NAMED)


def _tmpfs_temp_dirs() -> list[Path]:
    """The temp directories that are RAM-backed on this machine."""
    return [directory for directory in TEMP_DIRS if not disk_backed(directory)]


_RAM_FSTYPES = frozenset({"tmpfs", "ramfs"})


def disk_backed(path: Path) -> bool:
    """Whether ``path`` sits on a disk-backed filesystem rather than a
    RAM-backed one. A path that cannot be classified counts as disk-backed,
    erring toward watching it."""
    try:
        mounts = PROC_MOUNTS_PATH.read_text().splitlines()
    except OSError:
        return True
    fstypes = {
        fields[1]: fields[2] for line in mounts if len(fields := line.split()) >= 3
    }
    best = ""
    for mount in fstypes:
        prefix = mount.rstrip("/") + "/"
        if (str(path) + "/").startswith(prefix) and len(mount) > len(best):
            best = mount
    return not best or fstypes[best] not in _RAM_FSTYPES


def _ram_scratch(uid: int, directories: list[Path]) -> _Counted:
    """Weigh what the uid holds in ``directories``, at most ``MAX_TMPFS_ENTRIES`` entries."""
    return _owned_files(
        uid,
        directories,
        budget=MAX_TMPFS_ENTRIES,
        uncapped_bound=MAX_TMPFS_ENTRIES,
        named=_TMPFS_NAMED,
    )


def _owned_files(
    uid: int,
    directories: list[Path],
    *,
    budget: int | None,
    uncapped_bound: int,
    named: str,
) -> _Counted:
    """Weigh and count what the uid owns under ``directories``, walking at most
    ``budget`` (or ``uncapped_bound`` when there is no budget) owned entries."""
    walk_bound = uncapped_bound if budget is None else budget
    bytes_held = 0
    owned = 0
    walked = 0
    stack = [str(directory) for directory in directories]
    while stack:
        try:
            # scandir, not os.walk: os.walk lists a directory in full first
            listing = os.scandir(stack.pop())
        except OSError:
            continue
        with listing:
            # directories count too: a tree of empty ones costs the same walk
            for entry in listing:
                walked += 1
                if walked > walk_bound * _VISITS_PER_BUDGET:
                    return _Counted(bytes_held, owned, walked, budget, named)
                try:
                    is_dir = entry.is_dir(follow_symlinks=False)
                    st = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                # descend even into other uids' dirs: uid can still write there
                if is_dir:
                    stack.append(entry.path)
                if st.st_uid != uid:
                    continue
                owned += 1
                # one past, so a tree exactly on the budget still passes
                if owned > walk_bound:
                    return _Counted(bytes_held, owned, walked, budget, named)
                if not is_dir:
                    bytes_held += st.st_blocks * 512
    return _Counted(bytes_held, owned, walked, budget, named)


def _sysv_shm_bytes(uid: int) -> int:
    """What the uid holds in SysV shared-memory segments.

    Charged to the creator (``cuid``) rather than the current owner (``uid``),
    which is not the uid's to set but the segment's: ``shmctl(IPC_SET)`` lets an
    unprivileged creator rewrite the owner to any value at all, and a segment
    that answers to nobody is one the cap never sees.
    """
    try:
        rows = SYSVIPC_SHM_PATH.read_text().splitlines()[1:]
    except OSError:
        return 0
    total = 0
    for row in rows:
        fields = row.split()
        try:
            size, creator = int(fields[3]), int(fields[9])
        except (IndexError, ValueError):
            continue
        if creator == uid:
            total += size
    return total


def _weigh(
    uid: int, *, precise: bool = False, max_processes: int | None = None
) -> _Reading:
    """Everything the uid is keeping in RAM: resident memory, tmpfs files, SysV segments."""
    counts = (
        _processes(uid, precise=precise, budget=max_processes),
        _ram_scratch(uid, _tmpfs_temp_dirs()),
    )
    held = sum(count.bytes_held for count in counts) + _sysv_shm_bytes(uid)
    return _Reading(held, counts)


@dataclass
class MemoryWatch:
    """What watching a uid's memory found."""

    max_bytes: int | None
    peak_bytes: int = 0
    over_bytes: int | None = None
    """What tripped the limit, if it was tripped."""
    unbounded: str | None = None
    """Why a reading could not be bounded, if one could not."""

    @property
    def failure(self) -> str | None:
        """Why the uid was reaped, if it was."""
        if self.over_bytes is None or self.max_bytes is None:
            return self.unbounded
        return (
            f"held {self.over_bytes / 1024**3:.1f}GiB in RAM at once, past the "
            f"{self.max_bytes / 1024**3:.1f}GiB limit"
        )


@dataclass
class RunningWatch:
    """A started watch, stopped by calling :meth:`stop`."""

    watch: MemoryWatch
    _stop: threading.Event = field(repr=False)
    _thread: threading.Thread = field(repr=False)

    def stop(self) -> MemoryWatch:
        self._stop.set()
        self._thread.join()
        return self.watch


def start_watch(
    uid: int,
    max_bytes: int | None = None,
    *,
    max_processes: int | None = None,
    poll_interval: float = POLL_INTERVAL_S,
) -> RunningWatch:
    """SIGKILL every process ``uid`` owns if what it holds in RAM goes past
    ``max_bytes``, or it runs more than ``max_processes`` at once.

    Weighs tmpfs files and SysV segments too, since those outlive their process
    and reaping does not free them; without ``/proc`` the watch never trips.
    The watch keeps running after a reap: a violation that persists (the uid's
    tmpfs files survive its processes) reaps whatever is spawned after it too.
    """
    watch = MemoryWatch(max_bytes=max_bytes)
    stop = threading.Event()

    def weigh() -> None:
        violating = False
        while not stop.wait(poll_interval):
            reading = _weigh(uid, max_processes=max_processes)
            if watch.max_bytes is not None and reading.bytes_held > watch.max_bytes:
                # RSS over-counts shared pages, so confirm before reaping.
                reading = _weigh(uid, precise=True, max_processes=max_processes)
            watch.peak_bytes = max(watch.peak_bytes, reading.bytes_held)
            over = watch.max_bytes is not None and reading.bytes_held > watch.max_bytes
            if not over and reading.failure is None:
                violating = False
                continue
            if over:
                watch.over_bytes = reading.bytes_held
            watch.unbounded = reading.failure
            # Log on entering violation only; one that persists would flood.
            if not violating:
                logger.warning(f"Reaping UID {uid}: it {watch.failure}")
            violating = True
            try:
                kill_processes(uid)
            except RuntimeError as e:
                logger.error(f"Could not reap UID {uid} after it {watch.failure}: {e}")

    thread = threading.Thread(target=weigh, daemon=True)
    thread.start()
    return RunningWatch(watch, stop, thread)


@dataclass
class FileWatch:
    """What watching a uid's files found."""

    max_bytes: int | None
    peak_bytes: int = 0
    over_bytes: int | None = None
    """What tripped the limit, if it was tripped."""
    unbounded: str | None = None
    """Why a reading could not be bounded, if one could not."""

    @property
    def failure(self) -> str | None:
        """Why the uid was reaped, if it was."""
        if self.over_bytes is None or self.max_bytes is None:
            return self.unbounded
        return (
            f"held {self.over_bytes / 1024**3:.1f}GiB in files at once, past the "
            f"{self.max_bytes / 1024**3:.1f}GiB limit"
        )


@dataclass
class RunningFileWatch:
    """A started file watch, stopped by calling :meth:`stop`."""

    watch: FileWatch
    _stop: threading.Event = field(repr=False)
    _thread: threading.Thread = field(repr=False)

    def stop(self) -> FileWatch:
        self._stop.set()
        self._thread.join()
        return self.watch


def start_file_watch(
    uid: int,
    paths: tuple[Path, ...] | list[Path],
    *,
    max_bytes: int | None = None,
    max_count: int | None = None,
    poll_interval: float = FILE_POLL_INTERVAL_S,
) -> RunningFileWatch:
    """SIGKILL every process ``uid`` owns if its files under ``paths`` outgrow
    ``max_bytes`` or ``max_count``, whichever caps are set. The watch keeps
    running after a reap: killing the processes does not shrink the files, so
    whatever is spawned next to a persisting violation is reaped too."""
    watch = FileWatch(max_bytes=max_bytes)
    stop = threading.Event()
    directories = [Path(path) for path in paths]

    def weigh() -> None:
        violating = False
        while not stop.wait(poll_interval):
            counted = _owned_files(
                uid,
                directories,
                budget=max_count,
                uncapped_bound=MAX_UNCAPPED_FILES,
                named=_DISK_NAMED,
            )
            watch.peak_bytes = max(watch.peak_bytes, counted.bytes_held)
            over = max_bytes is not None and counted.bytes_held > max_bytes
            if not over and counted.failure is None:
                violating = False
                continue
            if over:
                watch.over_bytes = counted.bytes_held
            watch.unbounded = counted.failure
            # Log on entering violation only; one that persists would flood.
            if not violating:
                logger.warning(f"Reaping UID {uid}: it {watch.failure}")
            violating = True
            try:
                kill_processes(uid)
            except RuntimeError as e:
                logger.error(f"Could not reap UID {uid} after it {watch.failure}: {e}")

    thread = threading.Thread(target=weigh, daemon=True)
    thread.start()
    return RunningFileWatch(watch, stop, thread)


@contextmanager
def watch_memory(
    uid: int,
    max_bytes: int,
    *,
    max_processes: int | None = None,
    poll_interval: float = POLL_INTERVAL_S,
) -> Iterator[MemoryWatch]:
    """:func:`start_watch` scoped to a block."""
    running = start_watch(
        uid, max_bytes, max_processes=max_processes, poll_interval=poll_interval
    )
    try:
        yield running.watch
    finally:
        running.stop()
