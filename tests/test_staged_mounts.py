import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from karotte.staged_mounts import (
    STAGED_MOUNTS_ENV_VAR,
    StagedMount,
    copied_back,
    copy_back,
    copy_in,
    encode,
)


@pytest.fixture
def chowned(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, int, int]]:
    """Record lchown calls instead of making them: tests don't run as root."""
    calls: list[tuple[Path, int, int]] = []

    def lchown(path: Path, uid: int, gid: int) -> None:
        calls.append((Path(path), uid, gid))

    monkeypatch.setattr(os, "lchown", lchown)
    return calls


@pytest.fixture(autouse=True)
def mounts(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Record mount commands instead of running them: tests don't run as root."""
    calls: list[list[str]] = []

    def run(argv: list[str], **_: object) -> SimpleNamespace:
        calls.append(argv)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("karotte.staged_mounts.subprocess.run", run)
    return calls


def stage(monkeypatch: pytest.MonkeyPatch, *mounts: StagedMount) -> None:
    monkeypatch.setenv(STAGED_MOUNTS_ENV_VAR, encode(list(mounts)))


def mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


class TestCopyIn:
    def test_a_writable_mount_belongs_to_the_student(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        chowned: list[tuple[Path, int, int]],
    ):
        """As through a bind mount, the student may write it; mounted over
        /workdir it would otherwise take away the student's workdir."""
        monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
        source = tmp_path / "staged"
        source.mkdir()
        (source / "out.txt").write_text("x")
        ro_source = tmp_path / "ro"
        ro_source.mkdir()
        target, ro_target = tmp_path / "workdir", tmp_path / "data"
        stage(
            monkeypatch,
            StagedMount(str(source), str(target), writable=True),
            StagedMount(str(ro_source), str(ro_target), writable=False),
        )

        copy_in()

        owners = {p: (u, g) for p, u, g in chowned}
        assert owners[target / "."] == owners[target / "out.txt"] == (1000, 1000)
        assert owners[ro_target / "."] == (0, 0)

    def test_nothing_happens_without_the_env_var(
        self, monkeypatch: pytest.MonkeyPatch, chowned: list[tuple[Path, int, int]]
    ):
        monkeypatch.delenv(STAGED_MOUNTS_ENV_VAR, raising=False)

        copy_in()
        copy_back()

        assert chowned == []

    def test_copies_a_directory_owned_by_root_without_setuid(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        chowned: list[tuple[Path, int, int]],
    ):
        source = tmp_path / "staged"
        (source / "sub").mkdir(parents=True)
        (source / "sub" / "tool").write_text("#!/bin/sh\n")
        (source / "sub" / "tool").chmod(0o4755)
        (source / "secret").write_text("s")
        (source / "secret").chmod(0o600)
        (source / "link").symlink_to("secret")
        target = tmp_path / "workdir" / "data"
        stage(monkeypatch, StagedMount(str(source), str(target), writable=False))

        copy_in()

        assert (target / "sub" / "tool").read_text() == "#!/bin/sh\n"
        assert mode(target / "sub" / "tool") == 0o755
        assert mode(target / "secret") == 0o600
        assert os.readlink(target / "link") == "secret"
        assert {p for p, _, _ in chowned} == {
            target,
            target / "sub",
            target / "sub" / "tool",
            target / "secret",
            target / "link",
        }
        assert all((u, g) == (0, 0) for _, u, g in chowned)

    @pytest.mark.usefixtures("chowned")
    def test_a_read_only_mount_is_read_only_in_the_guest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mounts: list[list[str]]
    ):
        """`:ro` only made the staging mount read-only; a world-writable
        source stayed student-writable in its copy."""
        source = tmp_path / "staged"
        (source / "sub").mkdir(parents=True)
        (source / "sub").chmod(0o777)
        (source / "open").write_text("x")
        (source / "open").chmod(0o666)
        target = tmp_path / "data"
        stage(monkeypatch, StagedMount(str(source), str(target), writable=False))

        copy_in()

        assert mode(target / "sub") == 0o755
        assert mode(target / "open") == 0o644
        assert [m[1:] for m in mounts] == [
            ["--bind", str(target), str(target)],
            ["-o", "remount,ro,bind", str(target)],
        ]

    @pytest.mark.usefixtures("chowned")
    def test_a_writable_mount_keeps_its_modes_and_no_mount(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mounts: list[list[str]]
    ):
        source = tmp_path / "staged"
        source.mkdir()
        (source / "open").write_text("x")
        (source / "open").chmod(0o666)
        target = tmp_path / "results"
        stage(monkeypatch, StagedMount(str(source), str(target), writable=True))

        copy_in()

        assert mode(target / "open") == 0o666
        assert mounts == []

    def test_copies_a_single_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        chowned: list[tuple[Path, int, int]],
    ):
        source = tmp_path / "staged" / "app.yaml"
        source.parent.mkdir()
        source.write_text("a: 1")
        source.chmod(0o2640)
        target = tmp_path / "etc" / "app.yaml"
        stage(monkeypatch, StagedMount(str(source), str(target), writable=False))

        copy_in()

        assert target.read_text() == "a: 1"
        assert mode(target) == 0o640
        assert chowned == [(target, 0, 0)]


@pytest.fixture(autouse=True)
def killed(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    monkeypatch.setattr("karotte.staged_mounts.kill_processes", calls.append)
    return calls


class TestCopyBack:
    def test_a_link_the_student_made_above_the_target_stops_copy_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """/workdir/results swapped for a link to a root-only directory: its
        entry passes as the target, but root mustn't export it."""
        secret = tmp_path / "root-only"
        (secret / "data").mkdir(parents=True)
        (secret / "data" / "key").write_text("secret")
        workdir = tmp_path / "workdir"
        workdir.mkdir()
        (workdir / "results").symlink_to(secret)
        host = tmp_path / "host"
        host.mkdir()
        stage(
            monkeypatch,
            StagedMount(str(host), str(workdir / "results" / "data"), writable=True),
        )

        copy_back()

        assert list(host.iterdir()) == []

    def test_a_copy_cut_short_leaves_the_host_file_as_it_was(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Ctrl-C or a killed VM partway through mustn't truncate the user's file."""
        host = tmp_path / "host"
        host.mkdir()
        (host / "data.csv").write_text("original")
        target = tmp_path / "workdir"
        target.mkdir()
        (target / "data.csv").write_text("new")
        stage(monkeypatch, StagedMount(str(host), str(target), writable=True))

        def interrupted(_src: object, _dst: object) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr("karotte.staged_mounts.shutil.copyfileobj", interrupted)
        with pytest.raises(KeyboardInterrupt):
            copy_back()

        assert (host / "data.csv").read_text() == "original"
        assert [p.name for p in host.iterdir()] == ["data.csv"]

    @pytest.mark.usefixtures("chowned")
    def test_files_the_student_deleted_are_deleted_on_the_host(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """As through a bind mount; a file added on the host meanwhile stays."""
        host = tmp_path / "host"
        (host / "gone_dir").mkdir(parents=True)
        (host / "gone_dir" / "f").write_text("x")
        (host / "gone.txt").write_text("x")
        (host / "kept.txt").write_text("x")
        target = tmp_path / "workdir"
        stage(monkeypatch, StagedMount(str(host), str(target), writable=True))
        copy_in()
        (host / "added_on_host.txt").write_text("x")
        (target / "gone.txt").unlink()
        (target / "gone_dir" / "f").unlink()
        (target / "gone_dir").rmdir()

        copy_back()

        assert sorted(p.name for p in host.iterdir()) == [
            "added_on_host.txt",
            "kept.txt",
        ]

    @pytest.mark.usefixtures("chowned")
    def test_a_nested_mount_goes_back_to_its_own_source_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        outer, inner = tmp_path / "outer", tmp_path / "inner"
        outer.mkdir()
        inner.mkdir()
        (inner / "data.csv").write_text("old")
        target = tmp_path / "workdir"
        stage(
            monkeypatch,
            # Listed inner first: the outer copy must not land over it.
            StagedMount(str(inner), str(target / "data"), writable=True),
            StagedMount(str(outer), str(target), writable=True),
        )
        copy_in()
        assert (target / "data" / "data.csv").read_text() == "old"
        (target / "data" / "data.csv").write_text("new")

        copy_back()

        assert (inner / "data.csv").read_text() == "new"
        assert not (outer / "data").exists()

    def test_the_students_processes_are_killed_before_copy_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, killed: list[int]
    ):
        """None left to swap a path for a link mid-copy."""
        monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
        stage(
            monkeypatch,
            StagedMount(str(tmp_path / "h"), str(tmp_path / "t"), writable=True),
        )

        copy_back()

        assert killed == [1000]

    def test_writable_mounts_are_copied_back_on_exit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        host = tmp_path / "host"
        host.mkdir()
        (host / "old").write_text("old")
        target = tmp_path / "results"
        (target / "nested").mkdir(parents=True)
        (target / "nested" / "out.txt").write_text("result")
        (target / "run.sh").write_text("x")
        (target / "run.sh").chmod(0o6755)
        stage(monkeypatch, StagedMount(str(host), str(target), writable=True))

        with copied_back():
            pass

        assert (host / "nested" / "out.txt").read_text() == "result"
        assert mode(host / "run.sh") == 0o755
        assert (host / "old").read_text() == "old"

    def test_read_only_mounts_are_not_copied_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        host = tmp_path / "host"
        host.mkdir()
        target = tmp_path / "data"
        target.mkdir()
        (target / "new").write_text("n")
        stage(monkeypatch, StagedMount(str(host), str(target), writable=False))

        copy_back()

        assert list(host.iterdir()) == []

    def test_a_target_swapped_for_a_directory_link_is_not_walked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """is_dir() and os.walk() follow a link at the root: copy-back would
        export, as root, whatever directory the student pointed it at."""
        outside = tmp_path / "root_only"
        outside.mkdir()
        (outside / "harness_secret").write_text("secret")
        host = tmp_path / "host"
        host.mkdir()
        target = tmp_path / "results"
        target.symlink_to(outside)
        stage(monkeypatch, StagedMount(str(host), str(target), writable=True))

        copy_back()

        assert list(host.iterdir()) == []

    def test_links_are_never_followed_or_copied(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "harness_secret").write_text("secret")
        host = tmp_path / "host"
        host.mkdir()
        target = tmp_path / "results"
        target.mkdir()
        (target / "file_link").symlink_to(outside / "harness_secret")
        (target / "dir_link").symlink_to(outside)
        (target / "real").write_text("ok")
        stage(monkeypatch, StagedMount(str(host), str(target), writable=True))

        copy_back()

        assert sorted(p.name for p in host.iterdir()) == ["real"]

    def test_a_host_side_link_is_replaced_not_written_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        victim = tmp_path / "victim"
        victim.write_text("keep")
        host = tmp_path / "host"
        host.mkdir()
        (host / "out").symlink_to(victim)
        target = tmp_path / "results"
        target.mkdir()
        (target / "out").write_text("new")
        stage(monkeypatch, StagedMount(str(host), str(target), writable=True))

        copy_back()

        assert victim.read_text() == "keep"
        assert not (host / "out").is_symlink()
        assert (host / "out").read_text() == "new"

    def test_a_single_writable_file_is_copied_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        host = tmp_path / "host" / "state.json"
        host.parent.mkdir()
        host.write_text("{}")
        target = tmp_path / "state.json"
        target.write_text('{"done": true}')
        stage(monkeypatch, StagedMount(str(host), str(target), writable=True))

        copy_back()

        assert host.read_text() == '{"done": true}'

    def test_copy_back_runs_when_the_run_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        host = tmp_path / "host"
        host.mkdir()
        target = tmp_path / "results"
        target.mkdir()
        (target / "partial").write_text("p")
        stage(monkeypatch, StagedMount(str(host), str(target), writable=True))

        with pytest.raises(RuntimeError), copied_back():
            raise RuntimeError("run failed")

        assert (host / "partial").read_text() == "p"
