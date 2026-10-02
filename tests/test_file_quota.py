"""Tests for :mod:`karotte.file_quota`.

The unit tests fake the mount commands, since loop and overlay mounts need
Linux and root; the end-to-end test at the bottom really mounts and really
runs into ENOSPC, and only runs as root on Linux.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from loguru import logger

from karotte import file_quota
from karotte.file_quota import (
    FileQuota,
    ensure_file_quota,
    mount_file_quota,
)

GIB = 1024**3


@pytest.fixture
def quota_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "karotte_quota"
    monkeypatch.setattr(file_quota, "QUOTA_DIR", directory)
    monkeypatch.setattr(file_quota, "trusted_binary", str)
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    return directory


@pytest.fixture
def commands(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    ran: list[list[str]] = []
    monkeypatch.setattr(file_quota, "_run", ran.append)
    return ran


@pytest.fixture
def log_messages():
    messages: list[str] = []
    handler = logger.add(lambda m: messages.append(m.record["message"]), level="INFO")
    yield messages
    logger.remove(handler)


class TestMountingTheQuota:
    @pytest.mark.usefixtures("quota_dir", "commands")
    def test_the_log_says_the_quota_was_created(
        self, tmp_path: Path, log_messages: list[str]
    ) -> None:
        workdir = tmp_path / "workdir"
        workdir.mkdir()

        _ = mount_file_quota((workdir,), 5 * GIB, None)

        assert (
            f"Created loop file quota of {5 * GIB} bytes over ['{workdir}']"
            in log_messages
        )

    @pytest.mark.usefixtures("quota_dir")
    def test_it_overlays_every_path_on_one_budget(
        self, tmp_path: Path, commands: list[list[str]]
    ) -> None:
        workdir, temp = tmp_path / "workdir", tmp_path / "tmp"
        for path in (workdir, temp):
            path.mkdir()

        quota = mount_file_quota((workdir, temp), 1 * GIB, 100_000)

        assert quota == FileQuota(1 * GIB, (workdir, temp))
        mkfs, loop, overlay_a, overlay_b = commands
        assert mkfs[0] == "mkfs.ext4"
        assert "-m" in mkfs and "0" in mkfs
        assert "-N" in mkfs and "100000" in mkfs
        assert loop[:3] == ["mount", "-o", "loop"]
        assert f"lowerdir={workdir}" in overlay_a[-2]
        assert overlay_a[-1] == str(workdir)
        assert f"lowerdir={temp}" in overlay_b[-2]

    def test_the_byte_budget_is_the_filesystem_size(
        self, tmp_path: Path, quota_dir: Path, commands: list[list[str]]
    ) -> None:
        workdir = tmp_path / "workdir"
        workdir.mkdir()

        _ = mount_file_quota((workdir,), 1 * GIB, None)

        assert (quota_dir / "quota.img").stat().st_size == 1 * GIB
        assert not any("-N" in command for command in commands)

    @pytest.mark.usefixtures("commands")
    def test_the_upper_mirrors_the_lower_root(
        self, tmp_path: Path, quota_dir: Path
    ) -> None:
        """A 1777 workdir must stay 1777 once overlaid."""
        workdir = tmp_path / "workdir"
        workdir.mkdir()
        workdir.chmod(0o1777)

        _ = mount_file_quota((workdir,), 1 * GIB, None)

        import stat as stat_module

        mode = stat_module.S_IMODE((quota_dir / "mnt" / "upper0").stat().st_mode)
        assert mode == 0o1777

    @pytest.mark.usefixtures("commands")
    def test_it_writes_the_manifest_for_remounting(
        self, tmp_path: Path, quota_dir: Path
    ) -> None:
        workdir = tmp_path / "workdir"
        workdir.mkdir()

        _ = mount_file_quota((workdir,), 1 * GIB, None)

        manifest = json.loads((quota_dir / "manifest.json").read_text())
        assert manifest == {"paths": [str(workdir)]}

    @pytest.mark.usefixtures("quota_dir")
    def test_without_root_there_is_no_quota(
        self,
        tmp_path: Path,
        commands: list[list[str]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(os, "geteuid", lambda: 1000)

        assert mount_file_quota((tmp_path,), 1 * GIB, None) is None
        assert commands == []

    @pytest.mark.usefixtures("quota_dir", "commands")
    def test_a_second_quota_is_refused(self, tmp_path: Path) -> None:
        workdir = tmp_path / "workdir"
        workdir.mkdir()
        _ = mount_file_quota((workdir,), 1 * GIB, None)

        assert mount_file_quota((workdir,), 2 * GIB, None) is None

    def test_a_failed_mount_leaves_nothing_behind(
        self,
        tmp_path: Path,
        quota_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Sandboxes without CAP_SYS_ADMIN fail here; the caller falls back to
        the watchdog and a later attempt must not find leftovers."""

        def refuse(argv: list[str]) -> None:
            if argv[0] == "mount":
                raise RuntimeError("mount: permission denied")

        monkeypatch.setattr(file_quota, "_run", refuse)
        workdir = tmp_path / "workdir"
        workdir.mkdir()

        assert mount_file_quota((workdir,), 1 * GIB, None) is None
        assert not (quota_dir / "quota.img").exists()
        assert not (quota_dir / "manifest.json").exists()

    def test_a_failed_overlay_unmounts_the_rest(
        self,
        tmp_path: Path,
        quota_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ran: list[list[str]] = []

        def refuse_overlays(argv: list[str]) -> None:
            ran.append(argv)
            if argv[0] == "mount" and "-t" in argv:
                raise RuntimeError("overlay not supported")

        monkeypatch.setattr(file_quota, "_run", refuse_overlays)
        workdir = tmp_path / "workdir"
        workdir.mkdir()

        assert mount_file_quota((workdir,), 1 * GIB, None) is None
        assert ["umount", "-R", str(quota_dir / "mnt")] in ran
        assert not (quota_dir / "quota.img").exists()


class TestSubmountPreservation:
    """An overlay's lower layer hides mounts beneath it (e.g. read-only
    squashfs data mounts under /workdir), so they must be rebound
    into the merged view."""

    def test_submounts_are_rebound_into_the_merged_view(
        self,
        tmp_path: Path,
        quota_dir: Path,
        commands: list[list[str]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        workdir = tmp_path / "workdir"
        sample = workdir / "data" / "sample"
        sample.mkdir(parents=True)

        def submounts(path: Path) -> list[Path]:
            return [sample] if path == workdir else []

        monkeypatch.setattr(file_quota, "_submounts_under", submounts)

        quota = mount_file_quota((workdir,), 1 * GIB, None)

        assert quota is not None
        aside = quota_dir / "mnt" / "keep0-0"
        bind_aside = commands.index(["mount", "--rbind", str(sample), str(aside)])
        overlay = commands.index(
            next(c for c in commands if c[0] == "mount" and "-t" in c)
        )
        rebind = commands.index(["mount", "--rbind", str(aside), str(sample)])
        detach = commands.index(["umount", "-l", str(aside)])
        assert bind_aside < overlay < rebind < detach
        assert not aside.exists()

    @pytest.mark.usefixtures("quota_dir")
    def test_without_submounts_nothing_extra_is_mounted(
        self,
        tmp_path: Path,
        commands: list[list[str]],
    ) -> None:
        workdir = tmp_path / "workdir"
        workdir.mkdir()

        assert mount_file_quota((workdir,), 1 * GIB, None) is not None
        assert not any("--rbind" in command for command in commands)

    def test_only_topmost_submounts_are_rebound(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An rbind carries nested mounts along, so only mount roots count."""
        workdir = tmp_path / "workdir"
        mounts = [
            Path(f"{workdir}/a"),
            Path(f"{workdir}/a/b"),
            Path(f"{workdir}/c"),
            Path(f"{tmp_path}/elsewhere"),
            workdir,
        ]
        monkeypatch.setattr(file_quota, "mount_points", lambda: mounts)

        submounts = file_quota._submounts_under(workdir)  # pyright: ignore[reportPrivateUsage]

        assert submounts == [Path(f"{workdir}/a"), Path(f"{workdir}/c")]


class TestEnsuringTheQuota:
    def manifest(self, quota_dir: Path, paths: list[Path]) -> None:
        quota_dir.mkdir(parents=True, exist_ok=True)
        _ = (quota_dir / "manifest.json").write_text(
            json.dumps({"paths": [str(p) for p in paths]})
        )

    @pytest.mark.usefixtures("quota_dir")
    def test_without_a_manifest_it_does_nothing(
        self, commands: list[list[str]]
    ) -> None:
        ensure_file_quota()

        assert commands == []

    def test_it_remounts_what_is_missing(
        self,
        tmp_path: Path,
        quota_dir: Path,
        commands: list[list[str]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        workdir = tmp_path / "workdir"
        workdir.mkdir()
        self.manifest(quota_dir, [workdir])
        (quota_dir / "mnt").mkdir()

        def nothing_mounted(_path: object) -> bool:
            return False

        monkeypatch.setattr(os.path, "ismount", nothing_mounted)

        ensure_file_quota()

        loop, overlay = commands
        assert loop[:3] == ["mount", "-o", "loop"]
        assert overlay[-1] == str(workdir)

    def test_it_leaves_mounted_mounts_alone(
        self,
        tmp_path: Path,
        quota_dir: Path,
        commands: list[list[str]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self.manifest(quota_dir, [tmp_path / "workdir"])

        def all_mounted(_path: object) -> bool:
            return True

        monkeypatch.setattr(os.path, "ismount", all_mounted)

        ensure_file_quota()

        assert commands == []

    def test_it_raises_when_a_recorded_mount_cannot_be_restored(
        self,
        tmp_path: Path,
        quota_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Grading without the quota mounted would see none of the student's
        writes, so this must fail loudly rather than proceed."""
        self.manifest(quota_dir, [tmp_path / "workdir"])

        def nothing_mounted(_path: object) -> bool:
            return False

        monkeypatch.setattr(os.path, "ismount", nothing_mounted)

        def refuse(_argv: list[str]) -> None:
            raise RuntimeError("mount: no such device")

        monkeypatch.setattr(file_quota, "_run", refuse)

        with pytest.raises(RuntimeError, match="remount the file quota"):
            ensure_file_quota()


class TestReenteringTheWorkingDirectory:
    @staticmethod
    def record_chdirs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
        entered: list[str] = []
        monkeypatch.setattr(os, "chdir", lambda path: entered.append(str(path)))  # pyright: ignore[reportUnknownLambdaType]
        return entered

    @pytest.mark.usefixtures("quota_dir", "commands")
    def test_mounting_re_enters_the_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workdir = tmp_path / "workdir"
        (workdir / "project").mkdir(parents=True)
        monkeypatch.chdir(workdir / "project")
        entered = self.record_chdirs(monkeypatch)

        assert mount_file_quota((workdir,), 1 * GIB, None) is not None

        assert [Path(p) for p in entered] == [workdir / "project"]

    @pytest.mark.usefixtures("commands")
    def test_remounting_re_enters_the_cwd(
        self, tmp_path: Path, quota_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workdir = tmp_path / "workdir"
        workdir.mkdir()
        quota_dir.mkdir(parents=True, exist_ok=True)
        _ = (quota_dir / "manifest.json").write_text(
            json.dumps({"paths": [str(workdir)]})
        )
        (quota_dir / "mnt").mkdir()
        monkeypatch.chdir(workdir)
        entered = self.record_chdirs(monkeypatch)
        monkeypatch.setattr(os.path, "ismount", lambda _path: False)  # pyright: ignore[reportUnknownLambdaType]

        ensure_file_quota()

        assert [Path(p) for p in entered] == [workdir]

    @pytest.mark.usefixtures("quota_dir", "commands")
    def test_a_cwd_still_on_the_hidden_directory_is_an_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulates a mount over the cwd that re-entering did not move us out of:
        the path getcwd reports resolves to a different directory than ``.``."""
        workdir = tmp_path / "workdir"
        lower, merged = workdir / "lower", workdir / "merged"
        lower.mkdir(parents=True)
        merged.mkdir()
        monkeypatch.chdir(lower)
        monkeypatch.setattr(os, "getcwd", lambda: str(merged))
        monkeypatch.setattr(os, "chdir", lambda _path: None)  # pyright: ignore[reportUnknownLambdaType]

        with pytest.raises(RuntimeError, match="working directory"):
            _ = mount_file_quota((workdir,), 1 * GIB, None)


needs_mkfs_ext4 = pytest.mark.skipif(
    not Path("/sbin/mkfs.ext4").exists() and not Path("/usr/sbin/mkfs.ext4").exists(),
    reason="needs mkfs.ext4 to really mount a quota",
)


@pytest.mark.requires_root
@needs_mkfs_ext4
def test_a_real_quota_stops_writes_at_the_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(file_quota, "QUOTA_DIR", tmp_path / "karotte_quota")
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "existing").write_text("already here")

    quota = mount_file_quota((workdir,), 8 << 20, None)
    assert quota is not None
    try:
        # The lower layer shows through.
        assert (workdir / "existing").read_text() == "already here"

        # Writes within the budget land; writes past it are refused.
        (workdir / "small").write_bytes(b"\0" * (1 << 20))
        with pytest.raises(OSError):
            with open(workdir / "big", "wb") as f:
                _ = f.write(b"\0" * (16 << 20))
                os.fsync(f.fileno())

        # The lower layer stayed untouched: everything new is in the upper.
        _ = subprocess.run(["umount", str(workdir)], check=True)
        assert not (workdir / "small").exists()
        assert (workdir / "existing").exists()
    finally:
        _ = subprocess.run(["umount", str(workdir)], capture_output=True)
        _ = subprocess.run(
            ["umount", str(tmp_path / "karotte_quota" / "mnt")], capture_output=True
        )


@pytest.mark.requires_root
@needs_mkfs_ext4
def test_a_real_quota_keeps_submounts_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read-only data mount under the quota path must survive the overlay."""
    monkeypatch.setattr(file_quota, "QUOTA_DIR", tmp_path / "karotte_quota")
    workdir = tmp_path / "workdir"
    sample = workdir / "data" / "sample"
    sample.mkdir(parents=True)
    subprocess.run(["mount", "-t", "tmpfs", "tmpfs", str(sample)], check=True)
    (sample / "shard.parquet").write_text("rows")
    subprocess.run(["mount", "-o", "remount,ro", str(sample)], check=True)

    try:
        quota = mount_file_quota((workdir,), 8 << 20, None)
        assert quota is not None

        # The data mount still shows through, still read-only.
        assert (sample / "shard.parquet").read_text() == "rows"
        with pytest.raises(OSError):
            (sample / "nope").write_text("denied")

        # And the quota still caps writes elsewhere under the path.
        with pytest.raises(OSError):
            with open(workdir / "big", "wb") as f:
                _ = f.write(b"\0" * (16 << 20))
                os.fsync(f.fileno())
    finally:
        for target in (sample, workdir, sample, tmp_path / "karotte_quota" / "mnt"):
            _ = subprocess.run(["umount", "-l", str(target)], capture_output=True)


class TestPremountedQuota:
    """A privileged helper may loop-mount the sized filesystem before karotte
    runs (k8s: the device cgroup denies loop devices in-container); then only
    the overlays are mounted here and the filesystem's size is the budget."""

    @pytest.fixture
    def premounted(self, quota_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        mount_point = quota_dir / "mnt"
        mount_point.mkdir(parents=True)

        def fake_ismount(path: str | Path) -> bool:
            return Path(path) == mount_point

        monkeypatch.setattr(os.path, "ismount", fake_ismount)
        fs_size = 7 * GIB
        real_statvfs = os.statvfs

        def fake_statvfs(path: str | Path) -> os.statvfs_result:
            if Path(path) == mount_point:
                fields = list(real_statvfs(quota_dir))
                return os.statvfs_result((4096, 4096, fs_size // 4096, *fields[3:]))
            return real_statvfs(path)

        monkeypatch.setattr(os, "statvfs", fake_statvfs)
        return mount_point

    @pytest.mark.usefixtures("premounted")
    def test_no_image_is_created_and_the_fs_size_wins(
        self, tmp_path: Path, quota_dir: Path, commands: list[list[str]]
    ) -> None:
        workdir = tmp_path / "workdir"
        workdir.mkdir()

        quota = mount_file_quota((workdir,), 5 * GIB, 100_000)

        assert quota is not None
        assert quota.size_bytes == 7 * GIB  # the helper's filesystem, not the ask
        assert not (quota_dir / "quota.img").exists()
        assert not any(command[0] == "mkfs.ext4" for command in commands)
        assert not any("loop" in command for command in commands)
        assert any(
            command[0] == "mount" and command[-1] == str(workdir)
            for command in commands
        )

    @pytest.mark.usefixtures("commands")
    def test_the_log_says_the_quota_was_adopted(
        self, tmp_path: Path, premounted: Path, log_messages: list[str]
    ) -> None:
        """So a run's log tells the two paths apart without comparing sizes."""
        workdir = tmp_path / "workdir"
        workdir.mkdir()

        _ = mount_file_quota((workdir,), 5 * GIB, None)

        assert (
            f"Adopted pre-mounted file quota of {7 * GIB} bytes at {premounted} over ['{workdir}']"
            in log_messages
        )

    @pytest.mark.usefixtures("premounted")
    def test_overlay_failure_leaves_the_helpers_mount_alone(
        self, tmp_path: Path, quota_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workdir = tmp_path / "workdir"
        workdir.mkdir()
        ran: list[list[str]] = []

        def failing_run(argv: list[str]) -> None:
            ran.append(argv)
            if argv[0] == "mount" and "-t" in argv:
                raise RuntimeError("overlay refused")

        monkeypatch.setattr(file_quota, "_run", failing_run)

        assert mount_file_quota((workdir,), 5 * GIB, None) is None
        unmounted = [command[-1] for command in ran if command[0] == "umount"]
        assert str(quota_dir / "mnt") not in unmounted


@pytest.mark.requires_root
@needs_mkfs_ext4
def test_a_real_quota_keeps_the_working_directory_on_the_merged_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A process sitting in the workdir when the overlay lands must not be left under it."""
    monkeypatch.setattr(file_quota, "QUOTA_DIR", tmp_path / "karotte_quota")
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    quota = mount_file_quota((workdir,), 8 << 20, None)
    assert quota is not None
    try:
        assert os.stat(".").st_dev == os.stat(workdir).st_dev

        # safetensors: absolute temp file, renamed to a relative path
        (workdir / "model.tmp").write_bytes(b"weights")
        os.rename(workdir / "model.tmp", "model.safetensors")
        assert (workdir / "model.safetensors").read_bytes() == b"weights"

        Path("relative").write_text("counted")
        os.chdir(tmp_path)
        _ = subprocess.run(["umount", str(workdir)], check=True)
        assert not (workdir / "relative").exists()
    finally:
        os.chdir(tmp_path)
        _ = subprocess.run(["umount", str(workdir)], capture_output=True)
        _ = subprocess.run(
            ["umount", str(tmp_path / "karotte_quota" / "mnt")], capture_output=True
        )


@pytest.mark.requires_root
@needs_mkfs_ext4
def test_a_real_quota_is_where_a_child_of_a_stale_parent_lands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parent forked before the overlay (e.g. an HTTP MCP server) sits in the
    hidden lower directory for good; the preexec puts each child it spawns on
    the merged view anyway. The parent here is made stale on purpose by
    disabling the mount-time re-entry."""
    from karotte.subprocess import make_demote_fn

    monkeypatch.setattr(file_quota, "QUOTA_DIR", tmp_path / "karotte_quota")
    monkeypatch.setattr(file_quota, "_reenter_cwd", lambda *_args: None)  # pyright: ignore[reportUnknownLambdaType]
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setenv("KAROTTE_WORKDIR", str(workdir))
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", str(os.geteuid()))

    quota = mount_file_quota((workdir,), 8 << 20, None)
    assert quota is not None
    probe = "import os; print(os.stat('.').st_dev == os.stat(os.getcwd()).st_dev)"
    try:
        # The parent really is stale...
        assert os.stat(".").st_dev != os.stat(workdir).st_dev
        inherited = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True
        )
        assert inherited.stdout.strip() == "False"

        # ...and the preexec puts the child on the merged view regardless...
        fixed = subprocess.run(
            [sys.executable, "-c", probe],
            preexec_fn=make_demote_fn(),
            capture_output=True,
            text=True,
        )
        assert fixed.stdout.strip() == "True", fixed.stderr

        # ...without taking a caller's own cwd away from it.
        elsewhere = subprocess.run(
            [sys.executable, "-c", "import os; print(os.getcwd())"],
            cwd=tmp_path,
            preexec_fn=make_demote_fn(),
            capture_output=True,
            text=True,
        )
        assert Path(elsewhere.stdout.strip()) == tmp_path, elsewhere.stderr
    finally:
        os.chdir(tmp_path)
        _ = subprocess.run(["umount", str(workdir)], capture_output=True)
        _ = subprocess.run(
            ["umount", str(tmp_path / "karotte_quota" / "mnt")], capture_output=True
        )
