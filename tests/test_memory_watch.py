"""Tests for :mod:`karotte.memory_watch`.

The watch has to hold under a hostile submission, so most of what is tested here
is a reading that cannot be outrun: a fork bomb or a flood of temp files makes an
unbounded reading slower than the interval between readings, and a uid that can
do that can allocate through the gap unwatched.
"""

import os
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from karotte import memory_watch
from karotte.memory_watch import (
    FileWatch,
    MemoryWatch,
    _Counted,  # pyright: ignore[reportPrivateUsage]
    _processes,  # pyright: ignore[reportPrivateUsage]
    _pss_rollup,  # pyright: ignore[reportPrivateUsage]
    _ram_scratch,  # pyright: ignore[reportPrivateUsage]
    _Reading,  # pyright: ignore[reportPrivateUsage]
    _resident_bytes,  # pyright: ignore[reportPrivateUsage]
    _sysv_shm_bytes,  # pyright: ignore[reportPrivateUsage]
    _tmpfs_temp_dirs,  # pyright: ignore[reportPrivateUsage]
    _weigh,  # pyright: ignore[reportPrivateUsage]
    disk_backed,
    start_file_watch,
    start_watch,
    watch_memory,
)

needs_proc = pytest.mark.skipif(
    not Path("/proc/self").is_dir(), reason="reads /proc, which is Linux-only"
)

OTHER_UID = os.getuid() + 12345


class TestWeighingProcesses:
    def test_it_weighs_nothing_where_there_is_no_proc(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_proc(_path: object) -> Iterator[os.DirEntry[str]]:
            raise OSError("no /proc")

        monkeypatch.setattr(os, "scandir", no_proc)

        assert _processes(os.getuid()).bytes_held == 0

    @needs_proc
    def test_it_finds_what_the_uid_holds_resident(self) -> None:
        held = _processes(os.getuid()).bytes_held

        with open("/proc/self/statm") as f:
            mine = int(f.read().split()[1]) * memory_watch.PAGE_SIZE
        # Pss splits shared pages, so our own share is at most our RSS.
        assert held >= mine > 0
        pss = _pss_rollup("self")
        assert pss is not None
        assert pss <= mine

    @needs_proc
    def test_it_ignores_what_another_uid_holds(self) -> None:
        assert _processes(OTHER_UID).bytes_held == 0

    @needs_proc
    def test_the_walk_gives_up_rather_than_read_a_table_it_does_not_own(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(memory_watch, "MAX_UNCAPPED_PROCESSES", 1)

        # Twice the budget, and one past that before it gives up.
        assert _processes(OTHER_UID).walked == 3

    @needs_proc
    def test_an_explicit_budget_overrides_the_default(self) -> None:
        assert _processes(OTHER_UID, budget=1).walked == 3

    def test_no_budget_bounds_the_walk_but_blames_nobody(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A lifted process cap must not fall back to the walk bound as a cap."""
        monkeypatch.setattr(memory_watch, "MAX_UNCAPPED_PROCESSES", 5)
        for pid in range(20):
            (tmp_path / str(pid)).mkdir()
        monkeypatch.setattr(memory_watch, "PROC_PATH", tmp_path)

        counted = _processes(os.getuid(), budget=None)

        assert counted.owned > 5
        assert counted.failure is None


class TestWeighingOneProcess:
    """Summing RSS counts a page once per process mapping it, so a parent and
    its forks read as many times the memory the machine actually gave them.
    Pss splits each shared page across its mappers instead."""

    def proc(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        rollup: str | None = None,
        statm: str | None = None,
    ) -> str:
        directory = tmp_path / "7"
        directory.mkdir()
        if rollup is not None:
            (directory / "smaps_rollup").write_text(rollup)
        if statm is not None:
            (directory / "statm").write_text(statm)
        monkeypatch.setattr(memory_watch, "PROC_PATH", tmp_path)
        return "7"

    ROLLUP: str = "0-0 ---p 0 00:00 0 [rollup]\nRss:  409600 kB\nPss:  51200 kB\n"
    STATM: str = f"100000 {409600 * 1024 // memory_watch.PAGE_SIZE} 0 0 0 0 0"

    def test_a_precise_reading_takes_pss(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pid = self.proc(tmp_path, monkeypatch, rollup=self.ROLLUP, statm=self.STATM)

        assert _pss_rollup(pid) == 51200 * 1024

    def test_the_cheap_reading_leaves_the_rollup_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pss costs a page-table walk, so it is not what every poll pays for."""
        pid = self.proc(tmp_path, monkeypatch, rollup=self.ROLLUP, statm=self.STATM)

        assert _resident_bytes(pid) == 409600 * 1024

    def test_a_rollup_without_pss_is_no_reading(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pid = self.proc(
            tmp_path,
            monkeypatch,
            rollup="0-0 ---p 0 00:00 0 [rollup]\nRss:  400 kB\n",
            statm="100 25 10 0 0 0 0",
        )

        assert _pss_rollup(pid) is None

    def test_a_process_that_went_away_weighs_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ = self.proc(tmp_path, monkeypatch)

        assert _resident_bytes("7") == 0
        assert _pss_rollup("7") is None

    @needs_proc
    def test_it_reads_a_real_process(self) -> None:
        assert _resident_bytes("self") > 0


class TestWeighingACohort:
    """gVisor has no ``smaps_rollup``, so the precise reading falls back to
    per-VMA ``smaps``: anonymous pages per process, each mapped file once per
    uid. Falling back to RSS instead reaps N processes sharing one mmap'd file
    as if they held N copies."""

    def cohort(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        smaps_by_pid: dict[str, str],
    ) -> None:
        for pid, smaps in smaps_by_pid.items():
            directory = tmp_path / pid
            directory.mkdir()
            (directory / "smaps").write_text(smaps)
        monkeypatch.setattr(memory_watch, "PROC_PATH", tmp_path)

    def vma(
        self, rss_kb: int, *, anon_kb: int = 0, inode: int = 0, path: str = ""
    ) -> str:
        return (
            f"00400000-7fff0000 r--p 00000000 08:02 {inode} {path}\n"
            f"Rss: {rss_kb} kB\n"
            f"Anonymous: {anon_kb} kB\n"
            "VmFlags: rd mr \n"
        )

    def test_a_file_the_cohort_shares_is_charged_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        smaps = self.vma(4 << 20, inode=42, path="/weights.safetensors") + self.vma(
            40 << 10, anon_kb=40 << 10
        )
        self.cohort(tmp_path, monkeypatch, {"7": smaps, "8": smaps, "9": smaps})

        held = _processes(os.getuid(), precise=True).bytes_held

        assert held == (4 << 30) + 3 * (40 << 20)

    def test_anonymous_pages_are_charged_to_each_process(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        smaps = self.vma(1 << 20, anon_kb=1 << 20)
        self.cohort(tmp_path, monkeypatch, {"7": smaps, "8": smaps})

        assert _processes(os.getuid(), precise=True).bytes_held == 2 << 30

    def test_distinct_files_are_charged_separately(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.cohort(
            tmp_path,
            monkeypatch,
            {
                "7": self.vma(1 << 20, inode=1, path="/a"),
                "8": self.vma(1 << 20, inode=2, path="/b"),
            },
        )

        assert _processes(os.getuid(), precise=True).bytes_held == 2 << 30

    def test_the_largest_mapping_of_a_file_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.cohort(
            tmp_path,
            monkeypatch,
            {
                "7": self.vma(1 << 20, inode=1, path="/a"),
                "8": self.vma(2 << 20, inode=1, path="/a"),
            },
        )

        assert _processes(os.getuid(), precise=True).bytes_held == 2 << 30

    def test_cow_pages_in_a_file_mapping_stay_private(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        smaps = self.vma(100, anon_kb=40, inode=7, path="/lib.so")
        self.cohort(tmp_path, monkeypatch, {"7": smaps, "8": smaps})

        held = _processes(os.getuid(), precise=True).bytes_held

        assert held == (2 * 40 + 60) * 1024

    def test_vfio_device_memory_is_not_charged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """gVisor reports a TPU's VFIO device memory as resident, but it is not RAM."""
        smaps = (
            self.vma(8 << 20, inode=1, path="anon_inode:[vfio-device]")
            + self.vma(1 << 20, inode=42, path="/libtpu.so")
            + self.vma(1 << 20, anon_kb=1 << 20)
        )
        self.cohort(tmp_path, monkeypatch, {"7": smaps})

        assert _processes(os.getuid(), precise=True).bytes_held == 2 << 30

    def test_pages_copied_out_of_vfio_device_memory_are_charged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A write to a private mapping of the device copies the page into RAM."""
        device = "anon_inode:[vfio-device]"
        smaps = self.vma(2 << 20, inode=1, path=device) + self.vma(
            2 << 20, anon_kb=2 << 20, inode=1, path=device
        )
        self.cohort(tmp_path, monkeypatch, {"7": smaps})

        assert _processes(os.getuid(), precise=True).bytes_held == 2 << 30

    def test_other_anon_inode_mappings_are_charged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.cohort(
            tmp_path,
            monkeypatch,
            {"7": self.vma(1 << 20, inode=7, path="anon_inode:[io_uring]")},
        )

        assert _processes(os.getuid(), precise=True).bytes_held == 1 << 30

    def test_the_rollup_is_preferred_over_smaps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.cohort(
            tmp_path, monkeypatch, {"7": self.vma(4 << 20, inode=42, path="/a")}
        )
        (tmp_path / "7" / "smaps_rollup").write_text(
            "0-0 ---p 0 00:00 0 [rollup]\nRss: 400 kB\nPss: 100 kB\n"
        )

        assert _processes(os.getuid(), precise=True).bytes_held == 100 * 1024

    def test_without_a_precise_source_nothing_is_confirmed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A confirmation that re-reads the RSS that tripped the limit confirms
        nothing; reaping on it kills legitimate work."""
        directory = tmp_path / "7"
        directory.mkdir()
        (directory / "statm").write_text("100 25 0 0 0 0 0")
        monkeypatch.setattr(memory_watch, "PROC_PATH", tmp_path)

        assert _processes(os.getuid()).bytes_held == 25 * memory_watch.PAGE_SIZE
        assert _processes(os.getuid(), precise=True).bytes_held == 0


class TestWeighingRamScratch:
    """tmpfs pages belong to no process, so a per-process weigh cannot see them.
    Left out, they are RAM a uid can hold — and pin past its own reaping —
    without the cap noticing."""

    def test_what_the_uid_holds_in_a_ram_backed_directory_counts(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "bomb").write_bytes(b"\0" * (4 << 20))

        assert _ram_scratch(os.getuid(), [tmp_path]).bytes_held >= (4 << 20)

    def test_it_ignores_what_another_uid_holds(self, tmp_path: Path) -> None:
        (tmp_path / "roots").write_bytes(b"\0" * (1 << 20))

        assert _ram_scratch(OTHER_UID, [tmp_path]).bytes_held == 0

    def test_it_descends_into_directories_another_uid_owns(
        self, tmp_path: Path
    ) -> None:
        """A root-owned directory under /tmp is somewhere the uid can still put
        a file."""
        nested = tmp_path / "roots"
        nested.mkdir()
        (nested / "mine").write_bytes(b"\0" * (1 << 20))

        assert _ram_scratch(os.getuid(), [tmp_path]).bytes_held >= (1 << 20)

    def test_a_directory_that_cannot_be_read_is_skipped(self, tmp_path: Path) -> None:
        assert _ram_scratch(os.getuid(), [tmp_path / "gone"]).bytes_held == 0


class TestTheReadingCannotBeOutrun:
    def flood(self, directory: Path, entries: int) -> Path:
        directory.mkdir(exist_ok=True)
        for i in range(entries):
            (directory / f"f{i}").touch()
        return directory

    def test_the_walk_stops_at_the_budget(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(memory_watch, "MAX_TMPFS_ENTRIES", 50)
        shm = self.flood(tmp_path / "shm", 400)

        # One past, so a tree sitting exactly on the budget is still told from
        # one running past it.
        assert _ram_scratch(os.getuid(), [shm]).owned == 51

    def test_a_flood_of_entries_is_a_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(memory_watch, "MAX_TMPFS_ENTRIES", 50)
        shm = self.flood(tmp_path / "shm", 400)

        failure = _ram_scratch(os.getuid(), [shm]).failure

        assert failure is not None
        assert "more than 50 entries" in failure

    def test_a_tree_within_the_budget_is_left_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(memory_watch, "MAX_TMPFS_ENTRIES", 50)
        shm = self.flood(tmp_path / "shm", 20)

        assert _ram_scratch(os.getuid(), [shm]).failure is None

    def test_empty_files_are_counted_though_they_hold_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`st_blocks` is 0 for an empty file, so a flood that blinds the reading
        costs nothing against the byte cap. Counting entries is the only thing
        that sees it."""
        monkeypatch.setattr(memory_watch, "MAX_TMPFS_ENTRIES", 50)
        shm = self.flood(tmp_path / "shm", 400)

        scratch = _ram_scratch(os.getuid(), [shm])

        assert scratch.bytes_held == 0
        assert scratch.failure is not None

    def test_empty_directories_count_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A tree of empty directories costs the walk what a tree of empty files
        does, so counting only files would leave the same door open."""
        monkeypatch.setattr(memory_watch, "MAX_TMPFS_ENTRIES", 50)
        shm = tmp_path / "shm"
        shm.mkdir()
        for i in range(400):
            (shm / f"d{i}").mkdir()

        assert _ram_scratch(os.getuid(), [shm]).failure is not None

    def test_entries_another_uid_owns_are_not_a_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Whatever root left behind would otherwise fail a uid that wrote
        nothing at all."""
        monkeypatch.setattr(memory_watch, "MAX_TMPFS_ENTRIES", 50)
        shm = self.flood(tmp_path / "shm", 400)

        scratch = _ram_scratch(OTHER_UID, [shm])

        # Still bounded — the walk gave up — but nobody is blamed for it.
        assert scratch.walked == 101
        assert scratch.failure is None

    def test_a_tree_the_uid_owns_is_its_failure_however_much_else_is_there(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The budget counts what the uid owns, so root's leftovers neither
        fail it nor use up its allowance."""
        monkeypatch.setattr(memory_watch, "MAX_TMPFS_ENTRIES", 50)
        shm = self.flood(tmp_path / "shm", 51)

        assert _ram_scratch(os.getuid(), [shm]).failure is not None


class TestTheBudget:
    def test_a_count_within_its_budget_is_no_failure(self) -> None:
        assert _Counted(0, 10, 10, 50, "entries").failure is None

    def test_only_what_the_uid_owns_counts_against_the_budget(self) -> None:
        """A walk that visited far more than the budget is not a failure when
        almost none of what it visited belongs to the uid."""
        assert _Counted(0, 3, 4000, 50, "entries").failure is None

    def test_what_the_uid_owns_past_the_budget_is_a_failure(self) -> None:
        assert _Counted(0, 51, 51, 50, "entries").failure is not None

    def test_no_budget_is_never_a_failure(self) -> None:
        assert _Counted(0, 10_000, 10_000, None, "entries").failure is None


class TestFindingTheRamBackedDirectories:
    def test_only_tmpfs_mounts_among_the_temp_dirs_are_ram_backed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        disk, shm = tmp_path / "var-tmp", tmp_path / "shm"
        mounts = tmp_path / "mounts"
        mounts.write_text(
            f"overlay / overlay rw 0 0\ntmpfs {shm} tmpfs rw 0 0\n/dev/sda1 {disk} ext4 rw 0 0\n"
        )
        monkeypatch.setattr(memory_watch, "PROC_MOUNTS_PATH", mounts)
        monkeypatch.setattr(memory_watch, "TEMP_DIRS", (disk, shm))

        assert _tmpfs_temp_dirs() == [shm]

    def test_there_are_none_without_a_mount_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(memory_watch, "PROC_MOUNTS_PATH", tmp_path / "gone")

        assert _tmpfs_temp_dirs() == []


class TestWeighingSysVSegments:
    """SysV segments belong to no process either, and outlive the one that made
    them."""

    def rows(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
        path = tmp_path / "shm"
        path.write_text(text)
        monkeypatch.setattr(memory_watch, "SYSVIPC_SHM_PATH", path)

    HEADER: str = "key shmid perms size cpid lpid nattch uid gid cuid cgid\n"

    def row(self, shmid: int, size: int, uid: int, cuid: int) -> str:
        return f"0 {shmid} 600 {size} 1 1 0 {uid} {uid} {cuid} {cuid}\n"

    def test_it_sums_the_segments_the_uid_created(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        uid = os.getuid()
        self.rows(
            tmp_path,
            monkeypatch,
            self.HEADER
            + self.row(7, 4096, uid, uid)
            + self.row(8, 8192, uid, uid)
            + self.row(9, 1024, OTHER_UID, OTHER_UID),
        )

        assert _sysv_shm_bytes(uid) == 4096 + 8192

    def test_handing_the_segment_to_another_uid_does_not_hide_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``shmctl(IPC_SET)`` lets an unprivileged creator rewrite ``uid`` to
        anything it likes; ``cuid`` is the one field it cannot touch."""
        uid = os.getuid()
        self.rows(tmp_path, monkeypatch, self.HEADER + self.row(7, 4096, 65534, uid))

        assert _sysv_shm_bytes(uid) == 4096

    def test_a_segment_another_uid_created_is_not_charged_to_this_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same rewrite in reverse: being handed a segment is not creating
        it, so it is not this uid's to answer for."""
        uid = os.getuid()
        self.rows(
            tmp_path, monkeypatch, self.HEADER + self.row(7, 4096, uid, OTHER_UID)
        )

        assert _sysv_shm_bytes(uid) == 0

    def test_a_row_without_a_creator_column_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        uid = os.getuid()
        self.rows(
            tmp_path,
            monkeypatch,
            f"key shmid perms size cpid lpid nattch uid gid\n0 7 600 4096 1 1 0 {uid} {uid}\n",
        )

        assert _sysv_shm_bytes(uid) == 0

    def test_a_row_it_cannot_read_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.rows(tmp_path, monkeypatch, "key shmid perms size\nnonsense\n")

        assert _sysv_shm_bytes(os.getuid()) == 0

    def test_there_are_none_without_the_proc_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(memory_watch, "SYSVIPC_SHM_PATH", tmp_path / "gone")

        assert _sysv_shm_bytes(os.getuid()) == 0


class TestOneReading:
    def test_it_sums_processes_tmpfs_and_sysv(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def processes(_uid: int, **_kwargs: object) -> _Counted:
            return _Counted(1000, 1, 1, 10, "procs")

        def scratch(_uid: int, _directories: list[Path]) -> _Counted:
            return _Counted(200, 1, 1, 10, "entries")

        monkeypatch.setattr(memory_watch, "_processes", processes)
        monkeypatch.setattr(memory_watch, "_ram_scratch", scratch)
        monkeypatch.setattr(memory_watch, "_tmpfs_temp_dirs", list)

        def sysv(_uid: int) -> int:
            return 70

        monkeypatch.setattr(memory_watch, "_sysv_shm_bytes", sysv)

        assert _weigh(os.getuid()).bytes_held == 1270

    def test_a_reading_that_could_not_be_bounded_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def flooded(_uid: int, **_kwargs: object) -> _Counted:
            return _Counted(0, 20, 20, 10, "processes running")

        monkeypatch.setattr(memory_watch, "_processes", flooded)

        failure = _weigh(os.getuid()).failure

        assert failure is not None
        assert "more than 10 processes running" in failure

    def test_a_process_limit_becomes_the_walk_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        budgets: list[int | None] = []

        def processes(
            _uid: int, *, budget: int | None = None, **_kwargs: object
        ) -> _Counted:
            budgets.append(budget)
            return _Counted(0, 0, 0, budget or 0, "procs")

        monkeypatch.setattr(memory_watch, "_processes", processes)

        _ = _weigh(os.getuid(), max_processes=64)

        assert budgets == [64]


class TestTheWatch:
    def readings(self, monkeypatch: pytest.MonkeyPatch, *canned: _Reading) -> None:
        """Hand the watch one canned reading per poll, repeating the last."""
        queue = list(canned)

        def weigh(_uid: int, **_kwargs: object) -> _Reading:
            return queue.pop(0) if len(queue) > 1 else queue[0]

        monkeypatch.setattr(memory_watch, "_weigh", weigh)

    def tiered(
        self, monkeypatch: pytest.MonkeyPatch, cheap_bytes: int, precise_bytes: int
    ) -> list[bool]:
        """Answer the cheap and the precise reading differently. Returns the
        readings asked for, ``True`` for each precise one."""
        asked: list[bool] = []

        def weigh(_uid: int, *, precise: bool = False, **_kwargs: object) -> _Reading:
            asked.append(precise)
            return _Reading(precise_bytes if precise else cheap_bytes, ())

        monkeypatch.setattr(memory_watch, "_weigh", weigh)
        return asked

    def reaped(self, monkeypatch: pytest.MonkeyPatch) -> list[int]:
        killed: list[int] = []
        monkeypatch.setattr(memory_watch, "kill_processes", killed.append)
        return killed

    def held(self, bytes_held: int) -> _Reading:
        return _Reading(bytes_held, ())

    def wait_for(self, watch: MemoryWatch, seconds: float = 2.0) -> None:
        deadline = time.monotonic() + seconds
        while watch.failure is None and time.monotonic() < deadline:
            time.sleep(0.005)

    def test_it_records_the_peak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.readings(
            monkeypatch, self.held(10), self.held(500), self.held(20), self.held(20)
        )
        _ = self.reaped(monkeypatch)

        with watch_memory(os.getuid(), 1 << 30, poll_interval=0.001) as watch:
            time.sleep(0.2)

        assert watch.peak_bytes == 500
        assert watch.failure is None

    def test_it_reaps_what_goes_past_the_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.readings(monkeypatch, self.held(3 << 30))
        killed = self.reaped(monkeypatch)

        with watch_memory(1234, 2 << 30, poll_interval=0.001) as watch:
            self.wait_for(watch)

        assert set(killed) == {1234}
        assert watch.failure is not None
        assert "3.0GiB" in watch.failure
        assert "2.0GiB" in watch.failure

    def test_a_cheap_reading_over_the_limit_is_not_enough_to_reap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Summed RSS counts a shared page once per process holding it, so a
        forking job reads as several times its size. Reaping on that alone
        kills legitimate work."""
        asked = self.tiered(monkeypatch, cheap_bytes=3 << 30, precise_bytes=1 << 30)
        killed = self.reaped(monkeypatch)

        with watch_memory(1234, 2 << 30, poll_interval=0.001) as watch:
            time.sleep(0.1)

        assert killed == []
        assert watch.failure is None
        assert True in asked

    def test_a_precise_reading_over_the_limit_reaps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ = self.tiered(monkeypatch, cheap_bytes=3 << 30, precise_bytes=3 << 30)
        killed = self.reaped(monkeypatch)

        with watch_memory(1234, 2 << 30, poll_interval=0.001) as watch:
            self.wait_for(watch)

        assert set(killed) == {1234}

    def test_a_reading_under_the_limit_never_pays_for_the_precise_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pss costs a page-table walk per process, so the common case must not
        pay it or the reading gets slower than the interval between readings."""
        asked = self.tiered(monkeypatch, cheap_bytes=1 << 20, precise_bytes=1 << 20)
        _ = self.reaped(monkeypatch)

        with watch_memory(1234, 2 << 30, poll_interval=0.001):
            time.sleep(0.1)

        assert asked
        assert True not in asked

    def test_it_keeps_enforcing_after_a_reap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """tmpfs files and SysV segments survive the kill, so the uid can stay
        over the limit; a watch that stops after one reap leaves whatever is
        spawned afterwards unwatched."""
        self.readings(monkeypatch, self.held(3 << 30))
        killed = self.reaped(monkeypatch)

        with watch_memory(1234, 2 << 30, poll_interval=0.001):
            deadline = time.monotonic() + 2.0
            while len(killed) < 2 and time.monotonic() < deadline:
                time.sleep(0.005)

        assert len(killed) >= 2

    def test_it_reaps_a_reading_it_could_not_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        flood = _Counted(0, 400, 400, 50, "entries in the temp directories")
        self.readings(monkeypatch, _Reading(1 << 20, (flood,)))
        killed = self.reaped(monkeypatch)

        with watch_memory(1234, 1 << 30, poll_interval=0.001) as watch:
            self.wait_for(watch)

        assert set(killed) == {1234}
        assert watch.failure == "left more than 50 entries in the temp directories"

    def test_without_a_byte_limit_no_reading_is_over(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A watch capping only processes must not reap on bytes, however many."""
        self.readings(monkeypatch, self.held(300 << 30))
        killed = self.reaped(monkeypatch)

        running = start_watch(1234, max_processes=64, poll_interval=0.001)
        time.sleep(0.1)
        watch = running.stop()

        assert killed == []
        assert watch.failure is None
        assert watch.peak_bytes == 300 << 30

    def test_without_a_process_limit_a_crowd_is_not_reaped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Memory capped, process cap lifted: a crowd past the walk bound must
        not trip a reap."""
        crowd = _Counted(1 << 20, 600, 600, None, "processes running")
        self.readings(monkeypatch, _Reading(1 << 20, (crowd,)))
        killed = self.reaped(monkeypatch)

        with watch_memory(1234, 2 << 30, poll_interval=0.001) as watch:
            time.sleep(0.1)

        assert killed == []
        assert watch.failure is None

    def test_more_processes_than_the_limit_reaps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def processes(
            _uid: int, *, budget: int | None = None, **_kwargs: object
        ) -> _Counted:
            budget = budget or 0
            return _Counted(0, budget + 1, budget + 1, budget, "processes running")

        monkeypatch.setattr(memory_watch, "_processes", processes)
        monkeypatch.setattr(memory_watch, "_tmpfs_temp_dirs", list)
        killed = self.reaped(monkeypatch)

        running = start_watch(1234, max_processes=64, poll_interval=0.001)
        self.wait_for(running.watch)
        watch = running.stop()

        assert set(killed) == {1234}
        assert watch.failure == "left more than 64 processes running"

    def test_a_reap_that_will_not_go_through_does_not_take_the_watch_down(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def will_not_reap(_uid: int) -> None:
            raise RuntimeError("would not be reaped")

        self.readings(monkeypatch, self.held(3 << 30))
        monkeypatch.setattr(memory_watch, "kill_processes", will_not_reap)

        with watch_memory(1234, 2 << 30, poll_interval=0.001) as watch:
            self.wait_for(watch)

        assert watch.failure is not None

    def test_it_weighs_nothing_after_the_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        taken: list[int] = []

        def weigh(_uid: int, **_kwargs: object) -> _Reading:
            taken.append(1)
            return _Reading(0, ())

        monkeypatch.setattr(memory_watch, "_weigh", weigh)

        with watch_memory(os.getuid(), 1 << 30, poll_interval=0.001):
            time.sleep(0.05)
        after_the_block = len(taken)
        time.sleep(0.05)

        assert len(taken) == after_the_block

    def test_it_weighs_nothing_after_stop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        taken: list[int] = []

        def weigh(_uid: int, **_kwargs: object) -> _Reading:
            taken.append(1)
            return _Reading(0, ())

        monkeypatch.setattr(memory_watch, "_weigh", weigh)

        running = start_watch(os.getuid(), 1 << 30, poll_interval=0.001)
        time.sleep(0.05)
        _ = running.stop()
        after_stop = len(taken)
        time.sleep(0.05)

        assert len(taken) == after_stop

    def test_what_the_block_raises_is_not_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.readings(monkeypatch, self.held(0))

        with pytest.raises(ValueError):
            with watch_memory(os.getuid(), 1 << 30, poll_interval=0.001):
                raise ValueError("from the block")


class TestTheFileWatch:
    """The same bounded walk as the RAM one, but over disk directories and
    with the byte and count budgets coming from the caller."""

    def reaped(self, monkeypatch: pytest.MonkeyPatch) -> list[int]:
        killed: list[int] = []
        monkeypatch.setattr(memory_watch, "kill_processes", killed.append)
        return killed

    def wait_for(self, watch: FileWatch, seconds: float = 2.0) -> None:
        deadline = time.monotonic() + seconds
        while watch.failure is None and time.monotonic() < deadline:
            time.sleep(0.005)

    def test_it_reaps_files_past_the_byte_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        killed = self.reaped(monkeypatch)
        (tmp_path / "big").write_bytes(b"\0" * (1 << 20))

        running = start_file_watch(
            os.getuid(), [tmp_path], max_bytes=1 << 10, poll_interval=0.001
        )
        self.wait_for(running.watch)
        watch = running.stop()

        assert set(killed) == {os.getuid()}
        assert watch.failure is not None
        assert "in files" in watch.failure

    def test_within_the_limits_nothing_is_reaped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        killed = self.reaped(monkeypatch)
        (tmp_path / "small").write_bytes(b"\0" * 1024)

        running = start_file_watch(
            os.getuid(),
            [tmp_path],
            max_bytes=1 << 30,
            max_count=50,
            poll_interval=0.001,
        )
        time.sleep(0.1)
        watch = running.stop()

        assert killed == []
        assert watch.failure is None
        assert watch.peak_bytes >= 1024

    def test_more_files_than_the_count_reaps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        killed = self.reaped(monkeypatch)
        for i in range(5):
            (tmp_path / f"f{i}").touch()

        running = start_file_watch(
            os.getuid(), [tmp_path], max_count=3, poll_interval=0.001
        )
        self.wait_for(running.watch)
        watch = running.stop()

        assert set(killed) == {os.getuid()}
        assert watch.failure == "left more than 3 files on disk"

    def test_it_keeps_enforcing_after_a_reap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Killing the processes does not shrink the files, so the violation
        persists and later processes must be reaped too."""
        killed = self.reaped(monkeypatch)
        (tmp_path / "big").write_bytes(b"\0" * (1 << 20))

        running = start_file_watch(
            os.getuid(), [tmp_path], max_bytes=1 << 10, poll_interval=0.001
        )
        deadline = time.monotonic() + 2.0
        while len(killed) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        _ = running.stop()

        assert len(killed) >= 2

    def test_another_uids_files_do_not_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        killed = self.reaped(monkeypatch)
        (tmp_path / "big").write_bytes(b"\0" * (1 << 20))

        running = start_file_watch(
            OTHER_UID,
            [tmp_path],
            max_bytes=1 << 10,
            max_count=3,
            poll_interval=0.001,
        )
        time.sleep(0.1)
        watch = running.stop()

        assert killed == []
        assert watch.failure is None

    def test_no_count_cap_bounds_the_walk_but_blames_nobody(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A lifted count cap must not fall back to the walk bound as a cap."""
        monkeypatch.setattr(memory_watch, "MAX_UNCAPPED_FILES", 5)
        killed = self.reaped(monkeypatch)
        for i in range(20):
            (tmp_path / f"f{i}").touch()

        running = start_file_watch(
            os.getuid(), [tmp_path], max_bytes=1 << 30, poll_interval=0.001
        )
        time.sleep(0.1)
        watch = running.stop()

        assert killed == []
        assert watch.failure is None

    def test_it_weighs_nothing_after_stop(self, tmp_path: Path) -> None:
        running = start_file_watch(os.getuid(), [tmp_path], max_bytes=1 << 30)
        _ = running.stop()


class TestDiskBacked:
    def test_a_tmpfs_path_is_not_disk_backed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mounts = tmp_path / "mounts"
        mounts.write_text(
            f"overlay / overlay rw 0 0\ntmpfs {tmp_path}/shm tmpfs rw 0 0\n"
        )
        monkeypatch.setattr(memory_watch, "PROC_MOUNTS_PATH", mounts)

        assert not disk_backed(tmp_path / "shm")
        assert not disk_backed(tmp_path / "shm" / "nested")
        assert disk_backed(tmp_path / "elsewhere")

    def test_without_a_mount_table_everything_counts_as_disk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Erring toward watching: a path we cannot classify still gets walked."""
        monkeypatch.setattr(memory_watch, "PROC_MOUNTS_PATH", tmp_path / "gone")

        assert disk_backed(tmp_path)


class TestTheVerdict:
    def test_a_watch_that_saw_nothing_has_no_failure(self) -> None:
        assert MemoryWatch(max_bytes=1 << 30).failure is None

    def test_going_over_is_reported_before_an_unbounded_reading(self) -> None:
        """Both trip the same reap, and the bytes are the more useful of the
        two to be told about."""
        watch = MemoryWatch(max_bytes=1 << 30, over_bytes=2 << 30, unbounded="flooded")

        assert watch.failure is not None
        assert "2.0GiB" in watch.failure
