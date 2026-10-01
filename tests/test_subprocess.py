"""Tests for karotte.subprocess helpers."""

import contextlib
import os
import pwd
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest

import karotte.subprocess as karotte_subprocess
from karotte.subprocess import (
    is_harness_secret_env,
    make_demote_fn,
    make_preexec,
    reset_pid_namespace_probe,
    scrub_harness_secrets,
    student_env,
    student_identity_env,
    student_session_command,
    trusted_binary,
    wrap_to_disable_networking,
)
from tests.conftest import register_harness_secrets, register_platform_tooling_dirs

_SPARE_UID = 60123
"""Uid the demoted children below run as. It need not exist in /etc/passwd;
nothing here looks it up, and a uid of its own keeps them clear of whatever
else is on the box."""


@pytest.fixture(autouse=True)
def _no_ipc_namespace(monkeypatch: pytest.MonkeyPatch):  # pyright: ignore[reportUnusedFunction]
    """The IPC probe forks a real child; tests opt in to availability explicitly."""
    monkeypatch.setattr("karotte.subprocess._ipcns_probed", True)
    monkeypatch.setattr("karotte.subprocess._ipcns_available", False)


# ---------------------------------------------------------------------------
# trusted_binary
# ---------------------------------------------------------------------------


def test_trusted_binary_resolves_to_a_root_owned_system_dir():
    """A known system binary resolves to an absolute path under a trusted dir,
    never a bare name that ``$PATH`` (with its student-writable entries) governs."""
    resolved = trusted_binary("unshare")
    assert os.path.isabs(resolved)
    assert resolved.startswith(("/usr/", "/bin/", "/sbin/"))


def test_trusted_binary_ignores_a_poisoned_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A binary planted on ``$PATH`` must not be picked up: resolution consults
    only the fixed trusted search path, so a student-writable PATH entry is inert."""
    fake = tmp_path / "unshare"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ.get('PATH', '')}")

    assert trusted_binary("unshare") != str(fake)


def test_trusted_binary_raises_when_absent_rather_than_using_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A binary that exists only on ``$PATH`` is treated as missing, not run."""
    fake = tmp_path / "definitely-not-a-real-binary"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ.get('PATH', '')}")

    with pytest.raises(FileNotFoundError, match="definitely-not-a-real-binary"):
        trusted_binary("definitely-not-a-real-binary")


# ---------------------------------------------------------------------------
# make_demote_fn
# ---------------------------------------------------------------------------


def test_make_demote_fn_returns_none_without_demote_id(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)
    monkeypatch.delenv("KAROTTE_DEMOTE_ID", raising=False)
    assert make_demote_fn() is None


def test_make_demote_fn_raises_in_container_without_demote_id(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
    monkeypatch.delenv("KAROTTE_DEMOTE_ID", raising=False)
    with pytest.raises(RuntimeError, match="KAROTTE_DEMOTE_ID"):
        make_demote_fn()


def test_make_demote_fn_returns_callable_with_demote_id(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    fn = make_demote_fn()
    assert callable(fn)


def test_make_demote_fn_calls_setgroups_setgid_setuid(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    fn = make_demote_fn()
    assert fn is not None

    with (
        patch("karotte.subprocess.os.setgroups") as mock_setgroups,
        patch("karotte.subprocess.os.setgid") as mock_setgid,
        patch("karotte.subprocess.os.setuid") as mock_setuid,
    ):
        fn()

    mock_setgroups.assert_called_once_with([])
    mock_setgid.assert_called_once_with(1000)
    mock_setuid.assert_called_once_with(1000)


def test_student_processes_are_the_first_oom_victims(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Both ways a student process starts raise its OOM score before it runs."""
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    score = tmp_path / "oom_score_adj"
    demote = make_demote_fn()
    assert demote is not None
    root_preexec = karotte_subprocess._make_root_preexec(1000)  # pyright: ignore[reportPrivateUsage]

    for preexec in (demote, root_preexec):
        score.unlink(missing_ok=True)
        with (
            patch("karotte.subprocess.os.setsid"),
            patch("karotte.subprocess.os.setgroups"),
            patch("karotte.subprocess.os.setgid"),
            patch("karotte.subprocess.os.setuid"),
        ):
            preexec()
        assert score.read_text() == "1000"


def test_an_unwritable_oom_score_does_not_stop_the_student(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A sandbox without the proc file still runs the student."""
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    monkeypatch.setattr(
        "karotte.subprocess.OOM_SCORE_ADJ", str(tmp_path / "missing" / "oom_score_adj")
    )
    demote = make_demote_fn()
    assert demote is not None
    with (
        patch("karotte.subprocess.os.setgroups"),
        patch("karotte.subprocess.os.setgid"),
        patch("karotte.subprocess.os.setuid") as mock_setuid,
    ):
        demote()
    mock_setuid.assert_called_once_with(1000)


@pytest.mark.skipif(sys.platform != "linux", reason="needs /proc")
def test_a_child_inherits_the_raised_oom_score(monkeypatch: pytest.MonkeyPatch):
    """Against the real proc file: raising needs no privilege, and what the
    session's first process sets, everything it forks inherits."""
    monkeypatch.setattr("karotte.subprocess.OOM_SCORE_ADJ", "/proc/self/oom_score_adj")
    out = subprocess.run(
        ["sh", "-c", "sh -c 'cat /proc/self/oom_score_adj'"],
        preexec_fn=karotte_subprocess._prefer_for_oom_kill,  # pyright: ignore[reportPrivateUsage]
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "1000"


def test_make_demote_fn_clears_groups_before_dropping_privileges(
    monkeypatch: pytest.MonkeyPatch,
):
    """setgroups must be called before setgid/setuid — once UID is non-root,
    setgroups will fail with EPERM."""
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    fn = make_demote_fn()
    assert fn is not None

    call_order: list[str] = []
    with (
        patch(
            "karotte.subprocess.os.setgroups",
            side_effect=lambda _: call_order.append("setgroups"),  # pyright: ignore[reportUnknownLambdaType]
        ),
        patch(
            "karotte.subprocess.os.setgid",
            side_effect=lambda _: call_order.append("setgid"),  # pyright: ignore[reportUnknownLambdaType]
        ),
        patch(
            "karotte.subprocess.os.setuid",
            side_effect=lambda _: call_order.append("setuid"),  # pyright: ignore[reportUnknownLambdaType]
        ),
    ):
        fn()

    assert call_order == ["setgroups", "setgid", "setuid"]


def test_make_demote_fn_does_not_call_setsid(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    fn = make_demote_fn()
    assert fn is not None

    with (
        patch("karotte.subprocess.os.setsid") as mock_setsid,
        patch("karotte.subprocess.os.setgroups"),
        patch("karotte.subprocess.os.setgid"),
        patch("karotte.subprocess.os.setuid"),
    ):
        fn()

    mock_setsid.assert_not_called()


def test_make_demote_fn_captures_demote_id_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
):
    """The UID should be captured when make_demote_fn() is called, not when
    the returned function is invoked."""
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    fn = make_demote_fn()
    assert fn is not None

    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "2000")

    with (
        patch("karotte.subprocess.os.setgroups"),
        patch("karotte.subprocess.os.setgid") as mock_setgid,
        patch("karotte.subprocess.os.setuid") as mock_setuid,
    ):
        fn()

    mock_setgid.assert_called_once_with(1000)
    mock_setuid.assert_called_once_with(1000)


def test_make_demote_fn_chowns_no_fds_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    """An fd this process merely inherited is not ours to give away, so a
    caller that names nothing hands over nothing."""
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    fn = make_demote_fn()
    assert fn is not None

    with (
        patch("karotte.subprocess.os.fchown") as mock_fchown,
        patch("karotte.subprocess.os.setgroups"),
        patch("karotte.subprocess.os.setgid"),
        patch("karotte.subprocess.os.setuid"),
    ):
        fn()

    mock_fchown.assert_not_called()


def test_make_demote_fn_chowns_named_fds_before_dropping_privileges(
    monkeypatch: pytest.MonkeyPatch,
):
    """fchown on a pipe we opened only works while the child is still root, so
    it has to happen before setuid."""
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    fn = make_demote_fn(0, 1, 2)
    assert fn is not None

    call_order: list[str] = []
    with (
        patch(
            "karotte.subprocess.os.fchown",
            side_effect=lambda fd, _uid, _gid: call_order.append(f"fchown{fd}"),  # pyright: ignore[reportUnknownLambdaType]
        ) as mock_fchown,
        patch("karotte.subprocess.os.setgroups"),
        patch("karotte.subprocess.os.setgid"),
        patch(
            "karotte.subprocess.os.setuid",
            side_effect=lambda _: call_order.append("setuid"),  # pyright: ignore[reportUnknownLambdaType]
        ),
    ):
        fn()

    assert call_order == ["fchown0", "fchown1", "fchown2", "setuid"]
    assert mock_fchown.call_args_list == [
        ((0, 1000, 1000),),
        ((1, 1000, 1000),),
        ((2, 1000, 1000),),
    ]


def test_make_demote_fn_returns_none_with_fds_but_nothing_to_demote_to(
    monkeypatch: pytest.MonkeyPatch,
):
    """Off the container the parent is not root and the child is not demoted,
    so the fds already belong to whoever will use them."""
    monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)
    monkeypatch.delenv("KAROTTE_DEMOTE_ID", raising=False)
    assert make_demote_fn(0, 1, 2) is None


def test_make_demote_fn_drops_to_an_explicit_uid_over_the_demote_id(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    fn = make_demote_fn(uid_gid=900)
    assert fn is not None

    with (
        patch("karotte.subprocess.os.setgroups"),
        patch("karotte.subprocess.os.setgid") as mock_setgid,
        patch("karotte.subprocess.os.setuid") as mock_setuid,
    ):
        fn()

    mock_setgid.assert_called_once_with(900)
    mock_setuid.assert_called_once_with(900)


def test_make_demote_fn_chowns_fds_to_the_explicit_uid(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    fn = make_demote_fn(1, 2, uid_gid=900)
    assert fn is not None

    with (
        patch("karotte.subprocess.os.fchown") as mock_fchown,
        patch("karotte.subprocess.os.setgroups"),
        patch("karotte.subprocess.os.setgid"),
        patch("karotte.subprocess.os.setuid"),
    ):
        fn()

    assert mock_fchown.call_args_list == [((1, 900, 900),), ((2, 900, 900),)]


def test_make_demote_fn_with_an_explicit_uid_still_returns_none_off_the_container(
    monkeypatch: pytest.MonkeyPatch,
):
    """An explicit uid says who to become, not whether to demote: with nothing
    to demote to there is no root to drop from either."""
    monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)
    monkeypatch.delenv("KAROTTE_DEMOTE_ID", raising=False)
    assert make_demote_fn(uid_gid=900) is None


# ---------------------------------------------------------------------------
# make_preexec
# ---------------------------------------------------------------------------


def test_make_preexec_returns_none_without_demote_id(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)
    monkeypatch.delenv("KAROTTE_DEMOTE_ID", raising=False)
    assert make_preexec() is None


def test_make_preexec_raises_in_container_without_demote_id(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
    monkeypatch.delenv("KAROTTE_DEMOTE_ID", raising=False)
    with pytest.raises(RuntimeError, match="KAROTTE_DEMOTE_ID"):
        make_preexec()


def test_make_preexec_returns_callable_with_demote_id(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    fn = make_preexec()
    assert callable(fn)


def test_make_preexec_calls_setsid_setgroups_setgid_setuid(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    fn = make_preexec()
    assert fn is not None

    with (
        patch("karotte.subprocess.os.setsid") as mock_setsid,
        patch("karotte.subprocess.os.setgroups") as mock_setgroups,
        patch("karotte.subprocess.os.setgid") as mock_setgid,
        patch("karotte.subprocess.os.setuid") as mock_setuid,
    ):
        fn()

    mock_setsid.assert_called_once()
    mock_setgroups.assert_called_once_with([])
    mock_setgid.assert_called_once_with(1000)
    mock_setuid.assert_called_once_with(1000)


def test_make_preexec_passes_an_explicit_uid_through(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    fn = make_preexec(uid_gid=900)
    assert fn is not None

    with (
        patch("karotte.subprocess.os.setsid"),
        patch("karotte.subprocess.os.setgroups"),
        patch("karotte.subprocess.os.setgid") as mock_setgid,
        patch("karotte.subprocess.os.setuid") as mock_setuid,
    ):
        fn()

    mock_setgid.assert_called_once_with(900)
    mock_setuid.assert_called_once_with(900)


def test_make_preexec_captures_demote_id_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
):
    """The UID should be captured when make_preexec() is called, not when
    the returned function is invoked."""
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    fn = make_preexec()
    assert fn is not None

    # Change it after capturing
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "2000")

    with (
        patch("karotte.subprocess.os.setsid"),
        patch("karotte.subprocess.os.setgroups"),
        patch("karotte.subprocess.os.setgid") as mock_setgid,
        patch("karotte.subprocess.os.setuid") as mock_setuid,
    ):
        fn()

    # Should use the value at capture time (1000), not current (2000)
    mock_setgid.assert_called_once_with(1000)
    mock_setuid.assert_called_once_with(1000)


def test_make_preexec_chowns_named_fds_before_dropping_privileges(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    fn = make_preexec(1, 2)
    assert fn is not None

    call_order: list[str] = []
    with (
        patch("karotte.subprocess.os.setsid"),
        patch(
            "karotte.subprocess.os.fchown",
            side_effect=lambda fd, _uid, _gid: call_order.append(f"fchown{fd}"),  # pyright: ignore[reportUnknownLambdaType]
        ),
        patch("karotte.subprocess.os.setgroups"),
        patch("karotte.subprocess.os.setgid"),
        patch(
            "karotte.subprocess.os.setuid",
            side_effect=lambda _: call_order.append("setuid"),  # pyright: ignore[reportUnknownLambdaType]
        ),
    ):
        fn()

    assert call_order == ["fchown1", "fchown2", "setuid"]


# ---------------------------------------------------------------------------
# Mount isolation
# ---------------------------------------------------------------------------

MountCall = tuple[object, ...]


def _flags(call: MountCall) -> int:
    """The mount flags of a recorded call."""
    flags = call[4]
    assert isinstance(flags, int)
    return flags


@pytest.fixture
def mount_trace(monkeypatch: pytest.MonkeyPatch) -> list[MountCall]:
    """Record the namespace and mount calls a demote fn makes, in order."""
    trace: list[MountCall] = []

    def record(*args: object) -> None:
        trace.append(("mount", *args))

    def unshare(flags: int) -> None:
        trace.append(("unshare", flags))

    monkeypatch.setattr(karotte_subprocess, "_ensure_libc", lambda: None)
    monkeypatch.setattr(karotte_subprocess, "_mount", record)
    monkeypatch.setattr("karotte.subprocess.os.unshare", unshare, raising=False)
    return trace


def _run_demote(fn: Callable[[], None], trace: list[MountCall]) -> None:
    with (
        patch(
            "karotte.subprocess.os.setgroups",
            side_effect=lambda _: trace.append(("setgroups",)),  # pyright: ignore[reportUnknownLambdaType]
        ),
        patch(
            "karotte.subprocess.os.setgid",
            side_effect=lambda uid: trace.append(("setgid", uid)),  # pyright: ignore[reportUnknownLambdaType]
        ),
        patch(
            "karotte.subprocess.os.setuid",
            side_effect=lambda uid: trace.append(("setuid", uid)),  # pyright: ignore[reportUnknownLambdaType]
        ),
    ):
        fn()


def test_mounts_are_left_alone_unless_asked_for(
    monkeypatch: pytest.MonkeyPatch, mount_trace: list[MountCall]
):
    """The student's shell and the graded run share the harness's mounts; only
    a build opts out."""
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    fn = make_demote_fn()
    assert fn is not None

    _run_demote(fn, mount_trace)

    assert not [call for call in mount_trace if call[0] == "mount"]


def test_mount_isolation_happens_before_the_privilege_drop(
    monkeypatch: pytest.MonkeyPatch, mount_trace: list[MountCall]
):
    """CLONE_NEWNS and mount(2) both need CAP_SYS_ADMIN, which setuid takes."""
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    fn = make_demote_fn(uid_gid=900, ephemeral_dirs=("/tmp",))
    assert fn is not None

    _run_demote(fn, mount_trace)

    kinds = [call[0] for call in mount_trace]
    assert kinds.index("unshare") < kinds.index("mount")
    assert kinds.index("mount") < kinds.index("setuid")
    assert mount_trace[0] == ("unshare", karotte_subprocess.CLONE_NEWNS)
    # Before anything else, or the mounts below propagate to the harness.
    assert mount_trace[1] == (
        "mount",
        "none",
        "/",
        None,
        karotte_subprocess.MS_REC | karotte_subprocess.MS_PRIVATE,
    )


def test_every_named_directory_gets_a_fresh_tmpfs(
    monkeypatch: pytest.MonkeyPatch, mount_trace: list[MountCall]
):
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    named = ("/tmp", "/var/tmp", "/dev/shm")
    fn = make_demote_fn(ephemeral_dirs=named)
    assert fn is not None

    _run_demote(fn, mount_trace)

    tmpfs = {
        call[2] for call in mount_trace if call[0] == "mount" and call[3] == "tmpfs"
    }
    assert tmpfs == set(named)


def test_a_read_only_directory_is_bound_read_only_and_noexec(
    monkeypatch: pytest.MonkeyPatch, mount_trace: list[MountCall]
):
    """Read-only so the build cannot leave the student a payload, noexec so it
    cannot run one the student left for it."""
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    fn = make_demote_fn(read_only_dirs=("/workdir",))
    assert fn is not None

    _run_demote(fn, mount_trace)

    bind, remount = [
        _flags(call)
        for call in mount_trace
        if call[0] == "mount" and call[2] == "/workdir"
    ]
    # Recursive, or data mounts under the workdir vanish from the build.
    assert bind & karotte_subprocess.MS_BIND
    assert bind & karotte_subprocess.MS_REC
    for flag in (
        karotte_subprocess.MS_REMOUNT,
        karotte_subprocess.MS_RDONLY,
        karotte_subprocess.MS_NOEXEC,
        karotte_subprocess.MS_NOSUID,
    ):
        assert remount & flag


def test_a_covered_directory_gets_an_empty_read_only_tmpfs(
    monkeypatch: pytest.MonkeyPatch, mount_trace: list[MountCall]
):
    """Read-only at mount time is the whole point: a writable cover is worse
    than the mount it replaced, since the child can write its own library into
    it and load that instead."""
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    fn = make_demote_fn(covered_dirs=("/opt/platform-tooling",))
    assert fn is not None

    _run_demote(fn, mount_trace)

    (cover,) = [
        call
        for call in mount_trace
        if call[0] == "mount" and call[2] == "/opt/platform-tooling"
    ]
    assert cover[1] == "tmpfs" and cover[3] == "tmpfs"
    for flag in (
        karotte_subprocess.MS_RDONLY,
        karotte_subprocess.MS_NOEXEC,
        karotte_subprocess.MS_NOSUID,
        karotte_subprocess.MS_NODEV,
    ):
        assert _flags(cover) & flag


def test_covering_alone_is_enough_to_build_the_mount_namespace(
    monkeypatch: pytest.MonkeyPatch, mount_trace: list[MountCall]
):
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    fn = make_demote_fn(covered_dirs=("/opt/platform-tooling",))
    assert fn is not None

    _run_demote(fn, mount_trace)

    kinds = [call[0] for call in mount_trace]
    assert mount_trace[0] == ("unshare", karotte_subprocess.CLONE_NEWNS)
    assert kinds.index("mount") < kinds.index("setuid")


def test_a_covered_directory_is_not_an_ephemeral_one(
    monkeypatch: pytest.MonkeyPatch, mount_trace: list[MountCall]
):
    """The two look alike — both are a fresh tmpfs — and only one of them is
    safe to hand a child that can map executable pages."""
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    fn = make_demote_fn(
        ephemeral_dirs=("/tmp",), covered_dirs=("/opt/platform-tooling",)
    )
    assert fn is not None

    _run_demote(fn, mount_trace)

    writable = {
        call[2]
        for call in mount_trace
        if call[0] == "mount" and not _flags(call) & karotte_subprocess.MS_RDONLY
    }
    assert "/tmp" in writable
    assert "/opt/platform-tooling" not in writable


def test_a_cover_that_fails_kills_the_launch(monkeypatch: pytest.MonkeyPatch):
    """Nothing downstream probes for a directory still being in reach, so a
    skipped cover is a silently reopened hole."""
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    monkeypatch.setattr(karotte_subprocess, "_ensure_libc", lambda: None)

    def unshare(flags: int) -> None:
        assert flags

    monkeypatch.setattr("karotte.subprocess.os.unshare", unshare, raising=False)

    def refuse(_source: str, target: str, *_args: object) -> None:
        if target == "/opt/platform-tooling":
            raise OSError(1, "cover refused")

    monkeypatch.setattr(karotte_subprocess, "_mount", refuse)
    fn = make_demote_fn(covered_dirs=("/opt/platform-tooling",))
    assert fn is not None

    with pytest.raises(OSError, match="cover refused"):
        _run_demote(fn, [])


def test_no_platform_tooling_dirs_without_a_plugin(monkeypatch: pytest.MonkeyPatch):
    register_platform_tooling_dirs(monkeypatch)
    assert karotte_subprocess.platform_tooling_dirs() == ()


def test_platform_tooling_dirs_names_only_what_is_there(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A missing path makes mount(2) fail, and this fails closed: naming one
    would take every launch on a machine without the platform's mount."""
    register_platform_tooling_dirs(
        monkeypatch, a=(str(tmp_path / "there"), str(tmp_path / "absent"))
    )
    (tmp_path / "there").mkdir()

    assert karotte_subprocess.platform_tooling_dirs() == (str(tmp_path / "there"),)


def test_a_broken_platform_tooling_plugin_is_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    register_platform_tooling_dirs(
        monkeypatch, a=ImportError("gone"), b=(str(tmp_path),)
    )
    assert karotte_subprocess.platform_tooling_dirs() == (str(tmp_path),)


def test_platform_tooling_dirs_from_several_plugins_are_deduplicated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    register_platform_tooling_dirs(
        monkeypatch,
        a=(str(tmp_path / "y"), str(tmp_path / "x")),
        b=(str(tmp_path / "x"),),
    )
    assert karotte_subprocess.platform_tooling_dirs() == (
        str(tmp_path / "x"),
        str(tmp_path / "y"),
    )


def test_nothing_to_bind_read_only_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch, mount_trace: list[MountCall]
):
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    fn = make_demote_fn(ephemeral_dirs=("/tmp",))
    assert fn is not None

    _run_demote(fn, mount_trace)

    assert not [
        call
        for call in mount_trace
        if call[0] == "mount" and _flags(call) & karotte_subprocess.MS_BIND
    ]


def test_mount_isolation_off_the_container_still_demotes_to_nothing(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)
    monkeypatch.delenv("KAROTTE_DEMOTE_ID", raising=False)
    assert make_demote_fn(ephemeral_dirs=("/tmp",)) is None


def test_make_preexec_passes_mount_isolation_through(
    monkeypatch: pytest.MonkeyPatch, mount_trace: list[MountCall]
):
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    fn = make_preexec(ephemeral_dirs=("/tmp",))
    assert fn is not None

    with patch("karotte.subprocess.os.setsid"):
        _run_demote(fn, mount_trace)

    assert ("unshare", karotte_subprocess.CLONE_NEWNS) in mount_trace


def test_make_preexec_passes_covering_through(
    monkeypatch: pytest.MonkeyPatch, mount_trace: list[MountCall]
):
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    fn = make_preexec(covered_dirs=("/opt/platform-tooling",))
    assert fn is not None

    with patch("karotte.subprocess.os.setsid"):
        _run_demote(fn, mount_trace)

    assert "/opt/platform-tooling" in [
        call[2] for call in mount_trace if call[0] == "mount"
    ]


def test_a_mount_that_fails_kills_the_launch(monkeypatch: pytest.MonkeyPatch):
    """Carrying on would run the build with the shared /tmp it was supposed to
    lose, which is the whole point of the isolation."""
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    monkeypatch.setattr(karotte_subprocess, "_ensure_libc", lambda: None)

    def unshare(*flags: int) -> None:
        assert flags

    def refuse(*args: object) -> None:
        raise OSError(1, f"Operation not permitted: {args}")

    monkeypatch.setattr("karotte.subprocess.os.unshare", unshare, raising=False)

    monkeypatch.setattr(karotte_subprocess, "_mount", refuse)
    fn = make_demote_fn(ephemeral_dirs=("/tmp",))
    assert fn is not None

    with pytest.raises(OSError):
        fn()


# ---------------------------------------------------------------------------
# IPC namespace
# ---------------------------------------------------------------------------


def test_make_demote_fn_unshares_ipc_before_dropping_privileges(
    monkeypatch: pytest.MonkeyPatch,
):
    """unshare(CLONE_NEWIPC) needs CAP_SYS_ADMIN, so it must happen while the
    child is still root."""
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    monkeypatch.setattr("karotte.subprocess._ipcns_available", True)
    fn = make_demote_fn()
    assert fn is not None

    call_order: list[str] = []
    with (
        patch(
            "karotte.subprocess._try_unshare_ipc",
            side_effect=lambda: call_order.append("unshare_ipc"),
        ),
        patch(
            "karotte.subprocess.os.setgroups",
            side_effect=lambda _: call_order.append("setgroups"),  # pyright: ignore[reportUnknownLambdaType]
        ),
        patch("karotte.subprocess.os.setgid"),
        patch(
            "karotte.subprocess.os.setuid",
            side_effect=lambda _: call_order.append("setuid"),  # pyright: ignore[reportUnknownLambdaType]
        ),
    ):
        fn()

    assert call_order == ["unshare_ipc", "setgroups", "setuid"]


def test_make_demote_fn_skips_ipc_unshare_where_the_kernel_refused(
    monkeypatch: pytest.MonkeyPatch,
):
    """A kernel that refused the probe would refuse every child the same way;
    the demotion itself must still go through."""
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    fn = make_demote_fn()
    assert fn is not None

    with (
        patch("karotte.subprocess._try_unshare_ipc") as mock_ipc,
        patch("karotte.subprocess.os.setgroups"),
        patch("karotte.subprocess.os.setgid") as mock_setgid,
        patch("karotte.subprocess.os.setuid") as mock_setuid,
    ):
        fn()

    mock_ipc.assert_not_called()
    mock_setgid.assert_called_once_with(1000)
    mock_setuid.assert_called_once_with(1000)


def test_try_unshare_ipc_unshares_the_ipc_namespace(monkeypatch: pytest.MonkeyPatch):
    seen: list[int] = []
    monkeypatch.setattr("karotte.subprocess.os.unshare", seen.append, raising=False)
    monkeypatch.setattr("karotte.subprocess.os.CLONE_NEWIPC", 0x08000000, raising=False)

    karotte_subprocess._try_unshare_ipc()  # pyright: ignore[reportPrivateUsage]

    assert seen == [0x08000000]


def test_try_unshare_ipc_swallows_a_refusal(monkeypatch: pytest.MonkeyPatch):
    """It runs in a preexec_fn, where raising kills the launch — worse than a
    shared IPC namespace."""

    def refuse(_flags: int) -> None:
        raise PermissionError

    monkeypatch.setattr("karotte.subprocess.os.unshare", refuse, raising=False)
    monkeypatch.setattr("karotte.subprocess.os.CLONE_NEWIPC", 0x08000000, raising=False)

    karotte_subprocess._try_unshare_ipc()  # pyright: ignore[reportPrivateUsage]


def test_ipc_probe_reports_available_when_a_child_can_unshare(
    monkeypatch: pytest.MonkeyPatch,
):
    karotte_subprocess.reset_ipc_namespace_probe()
    monkeypatch.setattr("karotte.subprocess.os.unshare", lambda _f: None, raising=False)  # pyright: ignore[reportUnknownLambdaType]
    runs: list[list[str]] = []

    def fake_run(
        argv: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        runs.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr("karotte.subprocess.subprocess.run", fake_run)

    assert karotte_subprocess.ipc_namespace_available() is True
    assert karotte_subprocess.ipc_namespace_available() is True
    assert len(runs) == 1, "the probe forks a process; once is enough"


def test_ipc_probe_reports_unavailable_when_the_preexec_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    """A refused unshare in the probe child surfaces here as SubprocessError."""
    karotte_subprocess.reset_ipc_namespace_probe()
    monkeypatch.setattr("karotte.subprocess.os.unshare", lambda _f: None, raising=False)  # pyright: ignore[reportUnknownLambdaType]

    def fake_run(
        *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.SubprocessError("Exception occurred in preexec_fn.")

    monkeypatch.setattr("karotte.subprocess.subprocess.run", fake_run)

    assert karotte_subprocess.ipc_namespace_available() is False
    assert karotte_subprocess.ipc_namespace_available() is False


def test_ipc_probe_reports_unavailable_without_os_unshare(
    monkeypatch: pytest.MonkeyPatch,
):
    """No os.unshare (non-Linux) means no namespace and no probe fork."""
    karotte_subprocess.reset_ipc_namespace_probe()
    monkeypatch.delattr("karotte.subprocess.os.unshare", raising=False)

    def fake_run(
        *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        raise AssertionError("must not fork")

    monkeypatch.setattr("karotte.subprocess.subprocess.run", fake_run)

    assert karotte_subprocess.ipc_namespace_available() is False


@pytest.mark.requires_root
def test_demoted_child_gets_a_fresh_ipc_namespace(monkeypatch: pytest.MonkeyPatch):
    """A demoted child starts with no SysV objects, and whatever it creates
    dies with its last process — its IPC namespace is its own."""
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", str(_SPARE_UID))
    karotte_subprocess.reset_ipc_namespace_probe()
    if not karotte_subprocess.ipc_namespace_available():
        pytest.skip("kernel refuses unshare(CLONE_NEWIPC) here")

    proc = subprocess.run(
        [sys.executable, "-c", "import os; print(os.readlink('/proc/self/ns/ipc'))"],
        stdout=subprocess.PIPE,
        preexec_fn=make_demote_fn(1),
        check=True,
    )

    assert proc.stdout.decode().strip() != os.readlink("/proc/self/ns/ipc")


# ---------------------------------------------------------------------------
# Handing pipes to a demoted child, for real
# ---------------------------------------------------------------------------

_REOPEN_PROBE = (
    "for p in /dev/stdin /dev/stdout /proc/self/fd/0 /proc/self/fd/1; do "
    '  if (exec 3<>"$p") 2>/dev/null; then echo "$p OK"; else echo "$p DENIED"; fi; '
    "done"
)
"""Shell that reports, on stderr, whether it can reopen its own standard
streams by path — what a Dart, C or shell server does to get at them."""


def _probe_reopen(preexec_fn: Callable[[], None] | None) -> str:
    proc = subprocess.Popen(
        ["/bin/sh", "-c", f"{_REOPEN_PROBE} >&2"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=preexec_fn,
    )
    _, stderr = proc.communicate(b"")
    return stderr.decode()


@pytest.mark.requires_root
def test_demoted_child_cannot_reopen_pipes_it_was_not_given(
    monkeypatch: pytest.MonkeyPatch,
):
    """The bug this guards against: a pipe opened by a root parent keeps its
    root-owned inode, and the demoted child is permission-checked on open like
    anyone else — even for the fds it already holds."""
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", str(_SPARE_UID))

    report = _probe_reopen(make_demote_fn())

    assert "DENIED" in report, report


@pytest.mark.requires_root
def test_demoted_child_can_reopen_the_pipes_it_was_given(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", str(_SPARE_UID))

    report = _probe_reopen(make_demote_fn(0, 1, 2))

    assert "DENIED" not in report, report


@pytest.mark.requires_root
def test_named_pipes_end_up_owned_by_the_demoted_uid(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", str(_SPARE_UID))

    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os; print(os.fstat(1).st_uid, os.fstat(2).st_uid)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=make_demote_fn(1),
    )
    stdout, _ = proc.communicate()

    # fd 1 was named, fd 2 was not.
    assert stdout.split() == [str(_SPARE_UID).encode(), b"0"]


# ---------------------------------------------------------------------------
# wrap_to_disable_networking
# ---------------------------------------------------------------------------


def test_wrap_to_disable_networking_outside_gvisor(monkeypatch: pytest.MonkeyPatch):
    """Outside gVisor, --map-current-user is included so the inside uid
    matches the outer uid."""
    monkeypatch.setenv("KAROTTE_SANDBOX", "runc")
    assert wrap_to_disable_networking(["echo", "hi"]) == [
        trusted_binary("unshare"),
        "--user",
        "--net",
        "--map-current-user",
        "--",
        "echo",
        "hi",
    ]


def test_wrap_to_disable_networking_under_gvisor(monkeypatch: pytest.MonkeyPatch):
    """Under gVisor, --map-current-user is omitted because gVisor's procfs
    doesn't support uid_map / gid_map writes."""
    monkeypatch.setenv("KAROTTE_SANDBOX", "gvisor")
    assert wrap_to_disable_networking(["echo", "hi"]) == [
        trusted_binary("unshare"),
        "--user",
        "--net",
        "--",
        "echo",
        "hi",
    ]


def test_wrap_to_disable_networking_preserves_argv(monkeypatch: pytest.MonkeyPatch):
    """Original argv is appended after `--` so flags in the wrapped command
    aren't interpreted by `unshare(1)`."""
    monkeypatch.setenv("KAROTTE_SANDBOX", "runc")
    wrapped = wrap_to_disable_networking(["bash", "-c", "--help"])
    sep = wrapped.index("--")
    assert wrapped[sep + 1 :] == ["bash", "-c", "--help"]


# ---------------------------------------------------------------------------
# student_session_command
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _forget_pidns_probe():  # pyright: ignore[reportUnusedFunction]
    """The probe result is cached process-wide; each test gets a fresh one."""
    reset_pid_namespace_probe()
    yield
    reset_pid_namespace_probe()


def _probe_returns(monkeypatch: pytest.MonkeyPatch, returncode: int) -> list[list[str]]:
    """Stub the ``unshare`` probe and record the argv it was asked to run."""
    seen: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs: object):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, returncode)

    monkeypatch.setattr("karotte.subprocess.subprocess.run", fake_run)
    return seen


def test_student_session_runs_under_a_pid_namespace_on_gvisor(
    monkeypatch: pytest.MonkeyPatch,
):
    """The namespace is the guaranteed killer: SIGKILLing its init reaps every
    process inside atomically, including fork-and-die chains no ``/proc`` sweep
    can see."""
    monkeypatch.setenv("KAROTTE_SANDBOX", "gvisor")
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    _ = _probe_returns(monkeypatch, 0)

    argv, _preexec = student_session_command(["bash"], disable_networking=False)

    assert argv[0] == trusted_binary("unshare")
    for flag in ("--pid", "--fork", "--mount-proc", "--kill-child=SIGKILL"):
        assert flag in argv, f"{flag} missing from {argv}"
    assert argv[argv.index("--") + 1 :] == ["bash"]


def test_student_session_demotes_via_unshare_not_the_preexec(
    monkeypatch: pytest.MonkeyPatch,
):
    """``unshare`` needs CAP_SYS_ADMIN to make the namespace, so it has to still
    be root when it runs; it drops to the student itself via ``--setuid``. A
    preexec that demoted first would leave it unable to create the namespace."""
    monkeypatch.setenv("KAROTTE_SANDBOX", "gvisor")
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    _ = _probe_returns(monkeypatch, 0)

    argv, preexec = student_session_command(["bash"], disable_networking=False)

    assert argv[argv.index("--setuid") + 1] == "1000"
    assert argv[argv.index("--setgid") + 1] == "1000"
    assert preexec is not None
    with (
        patch("karotte.subprocess.os.setsid"),
        patch("karotte.subprocess.os.setgroups") as mock_setgroups,
        patch("karotte.subprocess.os.setgid") as mock_setgid,
        patch("karotte.subprocess.os.setuid") as mock_setuid,
    ):
        preexec()
    mock_setuid.assert_not_called()
    mock_setgid.assert_not_called()
    # Supplementary groups still have to go before unshare drops the uid.
    mock_setgroups.assert_called_once_with([])


def test_student_session_keeps_the_sweep_fallback_on_runc(
    monkeypatch: pytest.MonkeyPatch,
):
    """A PID namespace on runc leaves an incoherent ``/proc`` (it is masked, so
    ``--mount-proc`` cannot replace it), which breaks the ``ps``/``pkill`` that
    students rely on. runc keeps demoting in the preexec and reaping by sweep."""
    monkeypatch.setenv("KAROTTE_SANDBOX", "runc")
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    probes = _probe_returns(monkeypatch, 0)

    argv, preexec = student_session_command(["bash"], disable_networking=False)

    assert argv == ["bash"]
    assert probes == [], "runc should not even probe for a PID namespace"
    assert preexec is not None
    with (
        patch("karotte.subprocess.os.setsid"),
        patch("karotte.subprocess.os.setgroups"),
        patch("karotte.subprocess.os.setgid") as mock_setgid,
        patch("karotte.subprocess.os.setuid") as mock_setuid,
    ):
        preexec()
    mock_setuid.assert_called_once_with(1000)
    mock_setgid.assert_called_once_with(1000)


def test_student_session_falls_back_when_the_namespace_cannot_be_made(
    monkeypatch: pytest.MonkeyPatch,
):
    """A gVisor pod without CAP_SYS_ADMIN can't make the namespace. Degrade to
    the best-effort sweep rather than refusing to run, and demote in the preexec
    again since no ``unshare`` will do it."""
    monkeypatch.setenv("KAROTTE_SANDBOX", "gvisor")
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    _ = _probe_returns(monkeypatch, 1)

    argv, preexec = student_session_command(["bash"], disable_networking=False)

    assert argv == ["bash"]
    assert preexec is not None
    with (
        patch("karotte.subprocess.os.setsid"),
        patch("karotte.subprocess.os.setgroups"),
        patch("karotte.subprocess.os.setgid"),
        patch("karotte.subprocess.os.setuid") as mock_setuid,
    ):
        preexec()
    mock_setuid.assert_called_once_with(1000)


def test_student_session_probes_only_once(monkeypatch: pytest.MonkeyPatch):
    """The probe forks a process; a session start is not the place to repeat it."""
    monkeypatch.setenv("KAROTTE_SANDBOX", "gvisor")
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    probes = _probe_returns(monkeypatch, 0)

    for _ in range(3):
        _ = student_session_command(["bash"], disable_networking=False)

    assert len(probes) == 1


def test_student_session_nests_the_network_namespace_inside(
    monkeypatch: pytest.MonkeyPatch,
):
    """The network namespace stays exactly as it was, just nested inside the PID
    namespace: it is unshared by the demoted student, so it keeps being an
    unprivileged user namespace rather than a root-owned one."""
    monkeypatch.setenv("KAROTTE_SANDBOX", "gvisor")
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    _ = _probe_returns(monkeypatch, 0)

    argv, _preexec = student_session_command(["bash"], disable_networking=True)

    inner = argv[argv.index("--") + 1 :]
    assert inner == wrap_to_disable_networking(["bash"])
    assert "--net" not in argv[: argv.index("--")], (
        "the outer unshare runs as root; a --net there would not be the "
        "unprivileged, user-namespace-owned netns we have today"
    )


def test_student_session_outside_a_container_is_untouched(
    monkeypatch: pytest.MonkeyPatch,
):
    """No demotion target means no grading to protect and no uid for --setuid."""
    monkeypatch.setenv("KAROTTE_SANDBOX", "gvisor")
    monkeypatch.delenv("KAROTTE_DEMOTE_ID", raising=False)
    monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)
    probes = _probe_returns(monkeypatch, 0)

    argv, preexec = student_session_command(["bash"], disable_networking=False)

    assert argv == ["bash"]
    assert preexec is None
    assert probes == []


def test_student_session_chowns_named_fds(monkeypatch: pytest.MonkeyPatch):
    """The pipes still have to reach the student, whoever ends up demoting."""
    monkeypatch.setenv("KAROTTE_SANDBOX", "gvisor")
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    _ = _probe_returns(monkeypatch, 0)

    _argv, preexec = student_session_command(["bash"], 0, 1, disable_networking=False)

    assert preexec is not None
    chowned: list[int] = []
    with (
        patch("karotte.subprocess.os.setsid"),
        patch(
            "karotte.subprocess.os.fchown",
            side_effect=lambda fd, _uid, _gid: chowned.append(fd),  # pyright: ignore[reportUnknownLambdaType]
        ),
        patch("karotte.subprocess.os.setgroups"),
    ):
        preexec()
    assert chowned == [0, 1]


def test_student_session_pidns_preexec_unshares_ipc(monkeypatch: pytest.MonkeyPatch):
    """With ``unshare(1)`` doing the demotion, the preexec enters the IPC
    namespace itself — it is the only part that still runs as root."""
    monkeypatch.setenv("KAROTTE_SANDBOX", "gvisor")
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    _ = _probe_returns(monkeypatch, 0)
    monkeypatch.setattr("karotte.subprocess._ipcns_available", True)

    _argv, preexec = student_session_command(["bash"], disable_networking=False)

    assert preexec is not None
    with (
        patch("karotte.subprocess.os.setsid"),
        patch("karotte.subprocess.os.setgroups"),
        patch("karotte.subprocess._try_unshare_ipc") as mock_ipc,
    ):
        preexec()
    mock_ipc.assert_called_once_with()


# ---------------------------------------------------------------------------
# student_identity_env
# ---------------------------------------------------------------------------


def test_student_identity_env_empty_without_demotion(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)
    monkeypatch.delenv("KAROTTE_DEMOTE_ID", raising=False)
    assert student_identity_env() == {}


def test_student_identity_env_reads_the_passwd_entry(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", str(os.getuid()))
    pw = pwd.getpwuid(os.getuid())

    assert student_identity_env() == {
        "HOME": pw.pw_dir,
        "USER": pw.pw_name,
        "LOGNAME": pw.pw_name,
    }


def test_student_identity_env_falls_back_to_workdir(monkeypatch: pytest.MonkeyPatch):
    """A uid with no passwd entry still has files somewhere writable."""
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", str(_SPARE_UID))
    monkeypatch.setenv("KAROTTE_WORKDIR", "/workdir")

    assert student_identity_env() == {"HOME": "/workdir"}


def test_student_identity_env_empty_when_nothing_to_point_at(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", str(_SPARE_UID))
    monkeypatch.delenv("KAROTTE_WORKDIR", raising=False)

    assert student_identity_env() == {}


# ---------------------------------------------------------------------------
# scrub_harness_secrets / student_env
# ---------------------------------------------------------------------------


@pytest.fixture
def backend_token(monkeypatch: pytest.MonkeyPatch) -> None:
    register_harness_secrets(monkeypatch, internal=("BACKEND_TOKEN",))


def test_nothing_is_a_harness_secret_without_a_plugin(
    monkeypatch: pytest.MonkeyPatch,
):
    register_harness_secrets(monkeypatch)
    assert not is_harness_secret_env("BACKEND_TOKEN")


@pytest.mark.usefixtures("backend_token")
def test_installed_packages_register_harness_secrets():
    assert is_harness_secret_env("BACKEND_TOKEN")


def test_a_broken_harness_secret_plugin_is_skipped(monkeypatch: pytest.MonkeyPatch):
    register_harness_secrets(
        monkeypatch, a=ImportError("gone"), b=("BACKEND_TOKEN", "OTHER_TOKEN")
    )
    assert is_harness_secret_env("BACKEND_TOKEN")
    assert is_harness_secret_env("OTHER_TOKEN")


@pytest.mark.parametrize(
    "name",
    [
        "PATH",
        "HOME",
        "KAROTTE_WORKDIR",
        "KAROTTE_PROXY_URL",
        # Credential-shaped, and deliberately NOT withheld: environments hand
        # the student secrets on purpose, so the set is explicit, not a pattern.
        "HF_TOKEN",
        "OPENAI_API_KEY",
        "DB_PASSWORD",
    ],
)
@pytest.mark.usefixtures("backend_token")
def test_other_names_are_left_alone(name: str):
    assert not is_harness_secret_env(name)


@pytest.mark.usefixtures("backend_token")
def test_scrub_drops_secrets_and_keeps_the_rest():
    env = {
        "PATH": "/usr/bin",
        "BACKEND_TOKEN": "eyJ...",
        "HF_TOKEN": "hf_x",
        "KAROTTE_WORKDIR": "/workdir",
    }
    assert scrub_harness_secrets(env) == {
        "PATH": "/usr/bin",
        "HF_TOKEN": "hf_x",
        "KAROTTE_WORKDIR": "/workdir",
    }
    # A copy, never the input.
    assert "BACKEND_TOKEN" in env


@pytest.mark.usefixtures("backend_token")
def test_student_env_is_the_scrubbed_environment_plus_identity(
    monkeypatch: pytest.MonkeyPatch,
):
    """A harness credential in this process's environment must not reach the
    student, whose shell is built from it."""
    monkeypatch.setenv("BACKEND_TOKEN", "eyJ...")
    monkeypatch.setenv("KAROTTE_WORKDIR", "/workdir")
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", str(_SPARE_UID))

    env = student_env()

    assert "BACKEND_TOKEN" not in env
    assert env["KAROTTE_WORKDIR"] == "/workdir"
    assert env["HOME"] == "/workdir"  # student_identity_env's override wins
    assert env["PATH"] == os.environ["PATH"]


# ---------------------------------------------------------------------------
# The child re-enters its working directory by name
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _drop_privileges_patched():
    with (
        patch("karotte.subprocess.os.setsid"),
        patch("karotte.subprocess.os.setgroups"),
        patch("karotte.subprocess.os.setgid"),
        patch("karotte.subprocess.os.setuid"),
    ):
        yield


def test_the_demoted_child_re_enters_its_cwd_by_name(
    monkeypatch: pytest.MonkeyPatch,
):
    """The parent's cwd may be the lower directory hidden under the quota
    overlay (it forked before the mount, as the HTTP MCP server does), so the
    child resolves the path again rather than keeping the inherited inode."""
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    fn = make_demote_fn()
    assert fn is not None

    with (
        _drop_privileges_patched(),
        patch("karotte.subprocess.os.getcwd", return_value="/workdir"),
        patch("karotte.subprocess.os.chdir") as chdir,
    ):
        fn()

    chdir.assert_called_once_with("/workdir")


def test_the_root_preexec_re_enters_its_cwd_too(monkeypatch: pytest.MonkeyPatch):
    """On gVisor the session is launched through ``unshare`` from a child that
    stays root; the cwd is fixed there as well, or bash under a PID namespace
    would be the one path left in the hidden directory."""
    monkeypatch.setattr(karotte_subprocess, "ipc_namespace_available", lambda: False)
    fn = karotte_subprocess._make_root_preexec(1000)  # pyright: ignore[reportPrivateUsage]

    with (
        _drop_privileges_patched(),
        patch("karotte.subprocess.os.getcwd", return_value="/workdir"),
        patch("karotte.subprocess.os.chdir") as chdir,
    ):
        fn()

    chdir.assert_called_once_with("/workdir")


def test_a_callers_cwd_survives_the_preexec(tmp_path: Path):
    """CPython applies ``cwd=`` before ``preexec_fn``, so re-entering a fixed
    KAROTTE_WORKDIR would have overridden it; re-entering the current directory
    is a no-op for a path the caller just set."""
    result = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getcwd())"],
        cwd=tmp_path,
        preexec_fn=karotte_subprocess._reenter_cwd,  # pyright: ignore[reportPrivateUsage]
        capture_output=True,
        text=True,
        check=True,
    )
    assert Path(result.stdout.strip()).resolve() == tmp_path.resolve()


def test_a_cwd_that_cannot_be_re_entered_does_not_kill_the_launch(
    monkeypatch: pytest.MonkeyPatch,
):
    """Not a security boundary: failing leaves the child where it was, which
    is what every launch did before."""
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
    fn = make_demote_fn()
    assert fn is not None

    with (
        _drop_privileges_patched(),
        patch("karotte.subprocess.os.getcwd", side_effect=FileNotFoundError),
        patch("karotte.subprocess.os.chdir") as chdir,
    ):
        fn()

    chdir.assert_not_called()
