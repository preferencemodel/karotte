"""Tests for :mod:`karotte.reclaim`."""

import ctypes
import errno
import os
import tracemalloc
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

from karotte import reclaim
from karotte.file_quota import QUOTA_DIR
from karotte.reclaim import (
    DEFAULT_EXCLUDE,
    ReclaimError,
    _allocation_ceiling,  # pyright: ignore[reportPrivateUsage]
    _child_path,  # pyright: ignore[reportPrivateUsage]
    _remove_sysv_segments,  # pyright: ignore[reportPrivateUsage]
    delete_files,
)
from karotte.student_misbehavior import StudentMisbehaviorError

not_root = pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permissions")

OTHER_UID = os.getuid() + 12345


@pytest.fixture
def warnings() -> Iterator[list[str]]:
    """What the code under test warned about."""
    messages: list[str] = []
    handler = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    yield messages
    logger.remove(handler)


def _open_dir(name: str | Path, dir_fd: int | None = None) -> int:
    return os.open(name, os.O_RDONLY | os.O_DIRECTORY, dir_fd=dir_fd)


def _create_file(name: str, dir_fd: int) -> None:
    os.close(os.open(name, os.O_WRONLY | os.O_CREAT, 0o600, dir_fd=dir_fd))


def _grow_past_path_max(base: Path) -> tuple[int, int]:
    """Chain directories until the tail sits just under PATH_MAX; a 255-char
    child of the returned fd then crosses it. Returns (fd, dirs created)."""
    path_max = os.pathconf("/", "PC_PATH_MAX")
    target = path_max - 200
    fd = _open_dir(base)
    length = len(str(base))
    count = 0
    while length < target:
        name = "d" * max(1, min(target - length - 1, 200))
        os.mkdir(name, dir_fd=fd)
        nfd = _open_dir(name, fd)
        os.close(fd)
        fd = nfd
        length += 1 + len(name)
        count += 1
    return fd, count


@pytest.fixture(autouse=True)
def no_real_segments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Never touch the machine's actual SysV segments from a test: no listing
    to read, no id to be had for a walk, and no capability over them."""
    monkeypatch.setattr(reclaim, "SYSVIPC_SHM_PATH", tmp_path / "no-shm")
    monkeypatch.setattr(reclaim, "STATUS_PATH", tmp_path / "no-status")
    monkeypatch.setattr(reclaim, "_allocation_ceiling", lambda: None)


class TestDeletingFiles:
    def test_it_deletes_what_the_uid_owns(self, tmp_path: Path) -> None:
        (tmp_path / "mine").write_text("x")
        (tmp_path / "mine-too").write_text("y")

        assert delete_files(os.getuid(), (tmp_path,)) == 2
        assert list(tmp_path.iterdir()) == []

    def test_it_leaves_what_another_uid_owns(self, tmp_path: Path) -> None:
        (tmp_path / "roots").write_text("x")

        assert delete_files(OTHER_UID, (tmp_path,)) == 0
        assert (tmp_path / "roots").exists()

    def test_it_empties_an_owned_tree_bottom_up(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        (nested / "mine").write_text("x")

        removed = delete_files(os.getuid(), (tmp_path,))

        assert removed == 3  # the file and both directories
        assert list(tmp_path.iterdir()) == []

    def test_the_named_directories_themselves_survive(self, tmp_path: Path) -> None:
        """Deleting /tmp itself would be a step too far."""
        (tmp_path / "mine").write_text("x")

        _ = delete_files(os.getuid(), (tmp_path,))

        assert tmp_path.is_dir()

    def test_a_missing_directory_is_skipped(self, tmp_path: Path) -> None:
        assert delete_files(os.getuid(), (tmp_path / "gone",)) == 0

    def test_a_symlink_is_removed_not_followed(self, tmp_path: Path) -> None:
        """A student can point a link anywhere; what dies is the link."""
        target = tmp_path / "target"
        target.mkdir()
        (target / "precious").write_text("x")
        trap = tmp_path / "trap"
        trap.mkdir()
        (trap / "link").symlink_to(target)

        removed = delete_files(os.getuid(), (trap,))

        assert removed == 1
        assert (target / "precious").exists()
        assert not (trap / "link").exists()

    @not_root
    def test_it_raises_when_an_owned_file_cannot_be_deleted(
        self, tmp_path: Path
    ) -> None:
        locked = tmp_path / "locked"
        locked.mkdir()
        (locked / "stuck").write_text("x")
        (tmp_path / "mine").write_text("y")
        locked.chmod(0o500)

        try:
            with pytest.raises(ReclaimError, match="stuck"):
                delete_files(os.getuid(), (tmp_path,))
        finally:
            locked.chmod(0o700)
        assert not (tmp_path / "mine").exists()  # the sweep still finishes

    @not_root
    def test_it_raises_when_an_owned_dir_cannot_be_emptied(
        self, tmp_path: Path
    ) -> None:
        secret = tmp_path / "secret"
        secret.mkdir()
        (secret / "hidden").write_text("x")
        secret.chmod(0o000)

        try:
            with pytest.raises(ReclaimError, match="secret"):
                delete_files(os.getuid(), (tmp_path,))
        finally:
            secret.chmod(0o700)

    def test_a_dir_holding_another_uids_files_is_left_without_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        keep = tmp_path / "keep"
        keep.mkdir()

        def refuse(path: str, *, dir_fd: int | None = None) -> None:  # pyright: ignore[reportUnusedParameter]
            raise OSError(errno.ENOTEMPTY, "not empty", path)

        monkeypatch.setattr(os, "rmdir", refuse)

        assert delete_files(os.getuid(), (tmp_path,)) == 0
        assert keep.is_dir()

    def test_a_file_past_path_max_is_deleted(self, tmp_path: Path) -> None:
        fd, dirs = _grow_past_path_max(tmp_path)
        _create_file("f" * 255, fd)
        os.close(fd)

        removed = delete_files(os.getuid(), (tmp_path,))

        assert removed == dirs + 1
        assert list(tmp_path.iterdir()) == []

    def test_a_dir_past_path_max_is_deleted_with_its_content(
        self, tmp_path: Path
    ) -> None:
        fd, dirs = _grow_past_path_max(tmp_path)
        os.mkdir("g" * 255, dir_fd=fd)
        inner = _open_dir("g" * 255, fd)
        _create_file("stash", inner)
        os.close(inner)
        os.close(fd)

        removed = delete_files(os.getuid(), (tmp_path,))

        assert removed == dirs + 2
        assert list(tmp_path.iterdir()) == []

    def test_a_tree_deeper_than_the_python_stack_is_deleted(
        self, tmp_path: Path
    ) -> None:
        depth = 1500
        fd = _open_dir(tmp_path)
        for _ in range(depth):
            os.mkdir("d", dir_fd=fd)
            nfd = _open_dir("d", fd)
            os.close(fd)
            fd = nfd
        _create_file("stash", fd)
        os.close(fd)

        removed = delete_files(os.getuid(), (tmp_path,))

        assert removed == depth + 1
        assert list(tmp_path.iterdir()) == []

    def test_memory_grows_linearly_with_tree_depth(self, tmp_path: Path) -> None:
        depth = 600
        name = "d" * 255
        fd = _open_dir(tmp_path)
        try:
            for _ in range(depth):
                os.mkdir(name, dir_fd=fd)
                nfd = _open_dir(name, fd)
                os.close(fd)
                fd = nfd
        except OSError as e:
            if e.errno != errno.ENAMETOOLONG:
                raise
            pytest.skip(f"filesystem caps path length (overlayfs?): {e}")
        finally:
            os.close(fd)

        tracemalloc.start()
        try:
            removed = delete_files(os.getuid(), (tmp_path,))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert removed == depth
        assert peak < 4 * 2**20

    def test_an_unstattable_entry_fails_the_reclaim(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If we cannot prove an entry isn't the uid's, the reclaim is not done."""
        (tmp_path / "mystery").write_text("x")
        real_stat = os.stat

        def flaky(path: object, *args: object, **kwargs: object) -> os.stat_result:
            if path == "mystery":
                raise OSError(errno.EIO, "I/O error", path)
            return real_stat(path, *args, **kwargs)  # pyright: ignore[reportArgumentType]

        monkeypatch.setattr(os, "stat", flaky)

        with pytest.raises(ReclaimError, match="mystery"):
            delete_files(os.getuid(), (tmp_path,))

    @not_root
    def test_an_unopenable_dir_fails_the_reclaim(self, tmp_path: Path) -> None:
        """We cannot prove an unlistable dir holds nothing of the uid's."""
        sealed = tmp_path / "sealed"
        sealed.mkdir()
        sealed.chmod(0o000)

        try:
            with pytest.raises(ReclaimError, match="sealed"):
                delete_files(os.getuid(), (tmp_path,))
        finally:
            sealed.chmod(0o700)


class TestExcludingPaths:
    def test_an_excluded_file_survives_while_its_siblings_die(
        self, tmp_path: Path
    ) -> None:
        submission = tmp_path / "submission"
        submission.write_text("keep")
        (tmp_path / "junk").write_text("x")

        removed = delete_files(os.getuid(), (tmp_path,), exclude=(submission,))

        assert removed == 1
        assert submission.read_text() == "keep"
        assert not (tmp_path / "junk").exists()

    def test_an_excluded_dir_keeps_its_whole_subtree(self, tmp_path: Path) -> None:
        submission = tmp_path / "submission"
        (submission / "nested").mkdir(parents=True)
        (submission / "nested" / "part").write_text("keep")
        (tmp_path / "junk").write_text("x")

        removed = delete_files(os.getuid(), (tmp_path,), exclude=(submission,))

        assert removed == 1
        assert (submission / "nested" / "part").read_text() == "keep"
        assert not (tmp_path / "junk").exists()

    def test_paths_named_in_the_env_var_survive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        logs = tmp_path / "logs"
        (logs / "agent").mkdir(parents=True)
        (logs / "agent" / "trajectory.json").write_text("keep")
        agent = tmp_path / "installed-agent"
        agent.mkdir()
        (agent / "bin").write_text("keep")
        (tmp_path / "junk").write_text("x")
        monkeypatch.setenv(
            reclaim.EXCLUDE_ENV_VAR, os.pathsep.join((str(logs), str(agent)))
        )

        removed = delete_files(os.getuid(), (tmp_path,))

        assert removed == 1
        assert (logs / "agent" / "trajectory.json").read_text() == "keep"
        assert (agent / "bin").read_text() == "keep"
        assert not (tmp_path / "junk").exists()

    def test_the_env_var_adds_to_explicit_excludes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from_env = tmp_path / "from_env"
        from_env.write_text("keep")
        explicit = tmp_path / "explicit"
        explicit.write_text("keep")
        (tmp_path / "junk").write_text("x")
        monkeypatch.setenv(reclaim.EXCLUDE_ENV_VAR, str(from_env))

        removed = delete_files(os.getuid(), (tmp_path,), exclude=(explicit,))

        assert removed == 1
        assert from_env.read_text() == "keep"
        assert explicit.read_text() == "keep"

    def test_an_empty_or_unset_env_var_excludes_nothing_extra(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "junk").write_text("x")
        monkeypatch.setenv(reclaim.EXCLUDE_ENV_VAR, "")

        assert delete_files(os.getuid(), (tmp_path,)) == 1

        (tmp_path / "junk").write_text("x")
        monkeypatch.delenv(reclaim.EXCLUDE_ENV_VAR)

        assert delete_files(os.getuid(), (tmp_path,)) == 1

    def test_dirs_leading_to_an_excluded_file_survive_without_error(
        self, tmp_path: Path
    ) -> None:
        deep = tmp_path / "a" / "b"
        deep.mkdir(parents=True)
        submission = deep / "submission"
        submission.write_text("keep")
        (tmp_path / "a" / "junk").write_text("x")

        removed = delete_files(os.getuid(), (tmp_path,), exclude=(submission,))

        assert removed == 1
        assert submission.read_text() == "keep"

    def test_a_later_sweep_without_exclude_removes_the_leftovers(
        self, tmp_path: Path
    ) -> None:
        deep = tmp_path / "a" / "b"
        deep.mkdir(parents=True)
        submission = deep / "submission"
        submission.write_text("keep")

        _ = delete_files(os.getuid(), (tmp_path,), exclude=(submission,))

        assert delete_files(os.getuid(), (tmp_path,)) == 3
        assert list(tmp_path.iterdir()) == []

    def test_a_missing_exclude_changes_nothing(self, tmp_path: Path) -> None:
        (tmp_path / "junk").write_text("x")

        removed = delete_files(os.getuid(), (tmp_path,), exclude=(tmp_path / "ghost",))

        assert removed == 1
        assert list(tmp_path.iterdir()) == []

    def test_an_excluded_swept_root_is_left_alone(self, tmp_path: Path) -> None:
        (tmp_path / "junk").write_text("x")

        assert delete_files(os.getuid(), (tmp_path,), exclude=(tmp_path,)) == 0
        assert (tmp_path / "junk").exists()

    def test_an_unnormalized_exclude_still_matches(self, tmp_path: Path) -> None:
        submission = tmp_path / "submission"
        submission.write_text("keep")

        crooked = tmp_path / "elsewhere" / ".." / "submission"
        removed = delete_files(os.getuid(), (tmp_path,), exclude=(crooked,))

        assert removed == 0
        assert submission.read_text() == "keep"

    def test_the_file_quotas_upper_layers_are_spared_by_default(self) -> None:
        """They hold the overlays' own files, which the sweep reaches through
        the overlays instead."""
        assert QUOTA_DIR in DEFAULT_EXCLUDE

    def test_extend_exclude_composes_with_exclude(self, tmp_path: Path) -> None:
        first = tmp_path / "first"
        first.write_text("keep")
        second = tmp_path / "second"
        second.write_text("keep")
        (tmp_path / "junk").write_text("x")

        removed = delete_files(
            os.getuid(), (tmp_path,), exclude=(first,), extend_exclude=(second,)
        )

        assert removed == 1
        assert first.exists() and second.exists()


class TestIncludeArgument:
    def test_directories_is_a_deprecated_alias_for_include(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "junk").write_text("x")

        with pytest.deprecated_call():
            removed = delete_files(os.getuid(), directories=(tmp_path,))

        assert removed == 1

    def test_include_and_directories_together_are_rejected(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(TypeError, match="not both"):
            _ = delete_files(os.getuid(), include=(tmp_path,), directories=(tmp_path,))

    def test_child_paths_of_the_filesystem_root_are_well_formed(self) -> None:
        assert _child_path("/", "tmp") == "/tmp"
        assert _child_path("/tmp", "junk") == "/tmp/junk"


class TestMountSkips:
    @pytest.fixture
    def swept(self, tmp_path: Path) -> Path:
        """A dir to sweep that does not hold the mounts fixture file."""
        swept = tmp_path / "swept"
        swept.mkdir()
        return swept

    def mounts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, text: str
    ) -> None:
        path = tmp_path / "mounts"
        path.write_text(text)
        monkeypatch.setattr(reclaim, "MOUNTS_PATH", path)

    def test_a_read_only_mount_is_not_swept(
        self, tmp_path: Path, swept: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data = swept / "data"
        data.mkdir()
        (data / "file").write_text("x")
        self.mounts(monkeypatch, tmp_path, f"squash {data} squashfs ro,relatime 0 0\n")

        assert delete_files(os.getuid(), (swept,)) == 0
        assert (data / "file").exists()

    def test_a_virtual_filesystem_is_not_swept(
        self, tmp_path: Path, swept: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_proc = swept / "proc"
        fake_proc.mkdir()
        (fake_proc / "entry").write_text("x")
        self.mounts(monkeypatch, tmp_path, f"proc {fake_proc} proc rw,relatime 0 0\n")

        assert delete_files(os.getuid(), (swept,)) == 0
        assert (fake_proc / "entry").exists()

    def test_a_writable_disk_mount_is_swept(
        self, tmp_path: Path, swept: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (swept / "junk").write_text("x")
        self.mounts(monkeypatch, tmp_path, f"overlay {swept} overlay rw 0 0\n")

        assert delete_files(os.getuid(), (swept,)) == 1

    def test_a_mount_point_with_spaces_is_unescaped(
        self, tmp_path: Path, swept: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data = swept / "da ta"
        data.mkdir()
        (data / "file").write_text("x")
        escaped = str(data).replace(" ", "\\040")
        self.mounts(monkeypatch, tmp_path, f"squash {escaped} squashfs ro 0 0\n")

        assert delete_files(os.getuid(), (swept,)) == 0
        assert (data / "file").exists()

    def test_a_sweepable_mount_below_an_unsweepable_one_is_still_swept(
        self, tmp_path: Path, swept: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The tmpfs a uid can fill hangs off /dev, which is itself virtual."""
        dev = swept / "dev"
        shm = dev / "shm"
        shm.mkdir(parents=True)
        (dev / "node").write_text("x")
        (shm / "junk").write_text("x")
        self.mounts(
            monkeypatch,
            tmp_path,
            f"devtmpfs {dev} devtmpfs rw 0 0\nshm {shm} tmpfs rw 0 0\n",
        )

        assert delete_files(os.getuid(), (swept,)) == 1
        assert (dev / "node").exists()
        assert list(shm.iterdir()) == []

    def test_a_mount_point_itself_is_not_removed(
        self, tmp_path: Path, swept: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Emptying one is the sweep's job; unmounting it is not."""
        shm = swept / "shm"
        shm.mkdir()
        (shm / "junk").write_text("x")
        self.mounts(monkeypatch, tmp_path, f"shm {shm} tmpfs rw 0 0\n")

        assert delete_files(os.getuid(), (swept,)) == 1
        assert shm.is_dir()

    @not_root
    def test_an_unopenable_dir_on_an_unsweepable_mount_is_not_missed(
        self, tmp_path: Path, swept: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing there was ours to delete, so nothing there was left behind."""
        data = swept / "data"
        sealed = data / "sealed"
        sealed.mkdir(parents=True)
        sealed.chmod(0o000)
        self.mounts(monkeypatch, tmp_path, f"squash {data} squashfs ro 0 0\n")

        try:
            assert delete_files(os.getuid(), (swept,)) == 0
        finally:
            sealed.chmod(0o700)

    def test_sweeping_everything_without_mount_info_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setattr(reclaim, "MOUNTS_PATH", tmp_path / "gone")

        def never(*_args: object, **_kwargs: object) -> tuple[int, list[str]]:
            raise AssertionError("swept anyway")

        monkeypatch.setattr(reclaim, "_delete_owned", never)

        with pytest.raises(ReclaimError, match="mounts"):
            _ = delete_files(os.getuid())


class TestSweepingEverything:
    def test_it_is_refused_outside_a_container(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Outside one, the uid is likely a real user of the machine."""
        monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)

        def never(*_args: object, **_kwargs: object) -> tuple[int, list[str]]:
            raise AssertionError("swept anyway")

        monkeypatch.setattr(reclaim, "_delete_owned", never)

        assert delete_files(os.getuid()) == 0

    def test_a_named_root_is_swept_outside_a_container(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)
        (tmp_path / "junk").write_text("x")

        assert delete_files(os.getuid(), (tmp_path,)) == 1


class TestSweepDeadline:
    """A student can grow a tree faster than the sweep can walk it."""

    def clock(self, monkeypatch: pytest.MonkeyPatch, step: float = 1.0) -> None:
        """A monotonic clock that jumps ``step`` seconds every time it is read."""
        ticks = iter(range(0, 10_000))

        def monotonic() -> float:
            return next(ticks) * step

        monkeypatch.setattr(reclaim, "time", SimpleNamespace(monotonic=monotonic))

    def tree(self, base: Path, count: int = 20) -> None:
        for i in range(count):
            (base / f"f{i}").write_text("x")

    def test_a_sweep_past_the_deadline_is_given_up_on(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.tree(tmp_path)
        self.clock(monkeypatch)

        with pytest.raises(ReclaimError, match="took longer than 5s"):
            _ = delete_files(os.getuid(), (tmp_path,), timeout=5)

    def test_giving_up_says_how_far_it_got(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A timeout can also be a slow disk, so it has to be diagnosable."""
        self.tree(tmp_path)
        self.clock(monkeypatch)

        with pytest.raises(ReclaimError) as caught:
            _ = delete_files(os.getuid(), (tmp_path,), timeout=5)

        assert "deleted 5 thing(s)" in str(caught.value)
        assert str(tmp_path) in str(caught.value)

    def test_giving_up_scores_the_run_as_misbehavior(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.tree(tmp_path)
        self.clock(monkeypatch)

        with pytest.raises(StudentMisbehaviorError):
            _ = delete_files(os.getuid(), (tmp_path,), timeout=5)

    def test_a_sweep_inside_the_deadline_is_left_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.tree(tmp_path, count=5)
        self.clock(monkeypatch)

        assert delete_files(os.getuid(), (tmp_path,), timeout=1_000) == 5

    def test_no_timeout_lets_a_slow_sweep_finish(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.tree(tmp_path)
        self.clock(monkeypatch, step=1_000.0)

        assert delete_files(os.getuid(), (tmp_path,), timeout=None) == 20

    def test_the_deadline_spans_the_named_roots_together(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Otherwise each extra root buys the student another full timeout."""
        first, second = tmp_path / "a", tmp_path / "b"
        first.mkdir()
        second.mkdir()
        self.tree(first)
        self.tree(second)
        self.clock(monkeypatch)

        with pytest.raises(ReclaimError, match="took longer than"):
            _ = delete_files(os.getuid(), (first, second), timeout=15)

    def open_fds(self) -> int:
        """How many fds this process holds; /dev/fd on macOS, /proc on Linux."""
        for listing in ("/proc/self/fd", "/dev/fd"):
            if Path(listing).is_dir():
                return len(os.listdir(listing))
        pytest.skip("no way to count open fds here")

    def test_it_leaves_no_open_fds_behind(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The walk holds a dir fd; unwinding past it must not leak one."""
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        self.tree(deep)
        self.clock(monkeypatch)

        before = self.open_fds()
        with pytest.raises(ReclaimError):
            _ = delete_files(os.getuid(), (tmp_path,), timeout=5)

        assert self.open_fds() == before

    def test_the_default_gives_a_real_sweep_room(self) -> None:
        assert reclaim.SWEEP_TIMEOUT_SECONDS == 600.0


class TestRemovingSysVSegments:
    HEADER: str = "key shmid perms size cpid lpid nattch uid gid cuid cgid\n"

    def rows(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
        path = tmp_path / "shm"
        path.write_text(text)
        monkeypatch.setattr(reclaim, "SYSVIPC_SHM_PATH", path)

    def row(self, shmid: int, uid: int, cuid: int) -> str:
        return f"0 {shmid} 600 4096 1 1 0 {uid} {uid} {cuid} {cuid}\n"

    def removed(self, monkeypatch: pytest.MonkeyPatch) -> list[int]:
        gone: list[int] = []

        def rmid(shmid: int) -> bool:
            gone.append(shmid)
            return True

        monkeypatch.setattr(reclaim, "_shmctl_rmid", rmid)
        return gone

    def test_it_removes_the_segments_the_uid_created(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        uid = os.getuid()
        self.rows(
            tmp_path,
            monkeypatch,
            self.HEADER + self.row(7, uid, uid) + self.row(9, OTHER_UID, OTHER_UID),
        )
        gone = self.removed(monkeypatch)

        assert _remove_sysv_segments(uid) == 1
        assert gone == [7]

    def test_a_handed_over_segment_is_still_the_creators(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``shmctl(IPC_SET)`` lets the creator rewrite the owner; ``cuid`` is
        the field it cannot touch."""
        uid = os.getuid()
        self.rows(tmp_path, monkeypatch, self.HEADER + self.row(7, 65534, uid))
        gone = self.removed(monkeypatch)

        assert _remove_sysv_segments(uid) == 1
        assert gone == [7]

    def test_a_failed_removal_is_not_counted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        uid = os.getuid()
        self.rows(tmp_path, monkeypatch, self.HEADER + self.row(7, uid, uid))

        def refuse(_shmid: int) -> bool:
            return False

        monkeypatch.setattr(reclaim, "_shmctl_rmid", refuse)

        assert _remove_sysv_segments(uid) == 0

    def test_there_are_none_without_the_proc_file_or_an_id_to_probe_from(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(reclaim, "SYSVIPC_SHM_PATH", tmp_path / "gone")

        assert _remove_sysv_segments(os.getuid()) == 0


class TestWalkingTheSysVIdSpace:
    """Where there is no ``/proc/sysvipc/shm``, as under gVisor, the segments
    are found an id at a time, by whoever is allowed to read them."""

    def id_space(
        self,
        monkeypatch: pytest.MonkeyPatch,
        segments: dict[int, int],
        ceiling: int,
        unreadable: tuple[int, ...] = (),
    ) -> list[int]:
        """Stand in for the kernel's id space. Returns the ids asked after."""
        asked: list[int] = []

        def stat(shmid: int) -> object | None:
            asked.append(shmid)
            if shmid in unreadable or shmid not in segments:
                return None
            return SimpleNamespace(cuid=segments[shmid])

        monkeypatch.setattr(reclaim, "_allocation_ceiling", lambda: ceiling)
        monkeypatch.setattr(reclaim, "_shmctl_stat", stat)
        return asked

    def removed(self, monkeypatch: pytest.MonkeyPatch) -> list[int]:
        gone: list[int] = []

        def rmid(shmid: int, quiet: bool = False) -> bool:  # pyright: ignore[reportUnusedParameter]
            gone.append(shmid)
            return True

        monkeypatch.setattr(reclaim, "_shmctl_rmid", rmid)
        return gone

    def test_it_destroys_only_what_the_uid_created(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """gVisor lets a uid destroy segments it did not create, so what the
        walk may remove is decided here rather than left to the kernel."""
        uid = os.getuid()
        asked = self.id_space(monkeypatch, {1: uid, 2: OTHER_UID, 3: uid}, ceiling=4)
        gone = self.removed(monkeypatch)

        assert _remove_sysv_segments(uid) == 2
        assert gone == [1, 3]
        assert asked == [0, 1, 2, 3, 4]

    def test_a_segment_that_will_not_say_who_made_it_is_left(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A uid can hide a segment from itself by making it unreadable; no
        walk can tell it apart from anyone else's."""
        uid = os.getuid()
        _ = self.id_space(monkeypatch, {1: uid}, ceiling=2, unreadable=(1,))
        gone = self.removed(monkeypatch)

        assert _remove_sysv_segments(uid) == 0
        assert gone == []

    def test_the_walk_is_bounded(
        self, monkeypatch: pytest.MonkeyPatch, warnings: list[str]
    ) -> None:
        """A uid that churned ids only to push its own out of reach does not
        get an unbounded walk out of us."""
        uid = os.getuid()
        monkeypatch.setattr(reclaim, "_MAX_PROBED_SHMID", 3)
        asked = self.id_space(monkeypatch, {9: uid}, ceiling=10_000)
        _ = self.removed(monkeypatch)

        assert _remove_sysv_segments(uid) == 0
        assert asked == [0, 1, 2, 3]
        assert any("stay" in message for message in warnings)

    def test_capable_of_reading_any_creator_it_walks_as_itself(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """CAP_IPC_OWNER reads every segment, including one the uid made
        unreadable to itself, which becoming the uid would not."""
        status = tmp_path / "status"
        status.write_text("CapEff:\t00000000a8248000\n")
        monkeypatch.setattr(reclaim, "STATUS_PATH", status)
        monkeypatch.setattr(os, "geteuid", lambda: 0)
        _ = self.id_space(monkeypatch, {1: OTHER_UID}, ceiling=2)
        gone = self.removed(monkeypatch)

        def forbidden(_uid: int, _last: int) -> int:
            raise AssertionError("demoted the walk with the capability in hand")

        monkeypatch.setattr(reclaim, "_destroy_owned_segments_as_uid", forbidden)

        assert _remove_sysv_segments(OTHER_UID) == 1
        assert gone == [1]

    def test_a_uid_that_cannot_be_become_is_reported(
        self, monkeypatch: pytest.MonkeyPatch, warnings: list[str]
    ) -> None:
        _ = self.id_space(monkeypatch, {1: OTHER_UID}, ceiling=2)
        gone = self.removed(monkeypatch)
        monkeypatch.setattr(os, "geteuid", lambda: os.getuid() + 1)

        assert _remove_sysv_segments(OTHER_UID) == 0
        assert gone == []
        assert any("Cannot become uid" in message for message in warnings)

    def test_a_full_segment_table_leaves_no_id_to_bound_the_walk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A uid holding every segment the kernel allows would otherwise be the
        one case the walk gives up on."""

        class FullTable:
            def shmget(self, _key: int, _size: int, _flags: int) -> int:
                ctypes.set_errno(errno.ENOSPC)
                return -1

        monkeypatch.setattr(reclaim, "_libc", lambda: FullTable())

        assert _allocation_ceiling() == reclaim._MAX_PROBED_SHMID  # pyright: ignore[reportPrivateUsage]
