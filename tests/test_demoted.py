import asyncio
import errno
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil
import pytest

from karotte.demoted import (
    OPEN_REFUSED_EXIT,
    check_access,
    communicate_or_kill,
    open_regular_as_stdin,
    open_regular_as_stdout,
    reap,
)
from karotte.subprocess import make_demote_fn

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="posix only")


def _children() -> set[int]:
    return {p.pid for p in psutil.Process().children(recursive=True)}


@pytest.mark.asyncio
async def test_cancelled_communicate_kills_the_child():
    proc = await asyncio.create_subprocess_exec(
        "/bin/sleep", "30", stdout=asyncio.subprocess.PIPE
    )
    task = asyncio.create_task(communicate_or_kill(proc))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert proc.returncode is not None


@pytest.mark.asyncio
async def test_reap_returns_when_nobody_reads_stdout():
    """Python 3.12's wait() also waits for EOF on stdout, which a paused reader never sees."""
    proc = await asyncio.create_subprocess_exec(
        "/usr/bin/yes", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    assert proc.stdout is not None
    async with asyncio.timeout(5):
        while not proc.stdout._paused:  # pyright: ignore[reportAttributeAccessIssue]
            await asyncio.sleep(0.01)
    async with asyncio.timeout(5):
        await reap(proc)
    assert proc.returncode == -signal.SIGKILL


@pytest.mark.asyncio
async def test_check_access_times_out_and_leaves_no_child(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("karotte.demoted.TEST_PATH", "/bin/sh")
    spawned: list[asyncio.subprocess.Process] = []
    create_subprocess_exec = asyncio.create_subprocess_exec

    async def recording_create_subprocess_exec(*args: Any, **kwargs: Any):
        proc = await create_subprocess_exec(*args, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", recording_create_subprocess_exec
    )
    with pytest.raises(TimeoutError):
        await check_access("-c", "sleep 30", timeout_s=0.2)
    [proc] = spawned
    assert proc.returncode == -signal.SIGKILL
    assert proc.pid not in _children()


@pytest.mark.asyncio
async def test_open_regular_as_stdin_feeds_the_file(tmp_path: Path):
    file_path = tmp_path / "f.txt"
    file_path.write_text("hello\n")
    proc = await asyncio.create_subprocess_exec(
        "/bin/cat",
        preexec_fn=open_regular_as_stdin(str(file_path), None),
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await communicate_or_kill(proc)
    assert proc.returncode == 0
    assert stdout == b"hello\n"


def _make(tmp_path: Path, kind: str) -> Path:
    path = tmp_path / kind
    if kind == "fifo":
        os.mkfifo(path)
    elif kind == "directory":
        path.mkdir()
    return path


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "reason"),
    [
        ("fifo", b"Not a regular file"),
        ("missing", b"No such file or directory"),
        ("directory", b"Not a regular file"),
    ],
)
async def test_open_regular_as_stdin_refuses_without_blocking(
    tmp_path: Path, kind: str, reason: bytes
):
    proc = await asyncio.create_subprocess_exec(
        "/bin/cat",
        preexec_fn=open_regular_as_stdin(str(_make(tmp_path, kind)), None),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await communicate_or_kill(proc, timeout_s=5)
    assert (proc.returncode, stderr) == (OPEN_REFUSED_EXIT, reason)


@pytest.mark.asyncio
async def test_open_regular_as_stdout_truncates_and_writes(tmp_path: Path):
    file_path = tmp_path / "f.txt"
    file_path.write_text("a much longer old content\n")
    proc = await asyncio.create_subprocess_exec(
        "/bin/cat",
        preexec_fn=open_regular_as_stdout(str(file_path), None),
        stdin=asyncio.subprocess.PIPE,
    )
    await communicate_or_kill(proc, b"new\n")
    assert proc.returncode == 0
    assert file_path.read_text() == "new\n"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "reason"),
    [
        ("fifo", b"Not a regular file"),
        ("missing", b"No such file or directory"),
        ("directory", b"Is a directory"),
    ],
)
async def test_open_regular_as_stdout_refuses_without_blocking(
    tmp_path: Path, kind: str, reason: bytes
):
    proc = await asyncio.create_subprocess_exec(
        "/bin/cat",
        preexec_fn=open_regular_as_stdout(str(_make(tmp_path, kind)), None),
        stdin=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await communicate_or_kill(proc, b"new\n", timeout_s=5)
    assert (proc.returncode, stderr) == (OPEN_REFUSED_EXIT, reason)


@pytest.mark.asyncio
async def test_open_regular_as_stdout_refuses_when_truncate_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    file_path = tmp_path / "f.txt"
    file_path.write_text("old\n")

    def failing_ftruncate(_fd: int, _length: int) -> None:
        raise OSError(errno.EFBIG, "File too large")

    monkeypatch.setattr(os, "ftruncate", failing_ftruncate)
    proc = await asyncio.create_subprocess_exec(
        "/bin/cat",
        preexec_fn=open_regular_as_stdout(str(file_path), None),
        stdin=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await communicate_or_kill(proc, b"new\n", timeout_s=5)
    assert (proc.returncode, stderr) == (OPEN_REFUSED_EXIT, b"File too large")
    assert file_path.read_text() == "old\n"


@pytest.mark.asyncio
async def test_hanging_open_is_killed_even_if_the_parent_ignores_sigalrm(
    tmp_path: Path, hanging_open: Callable[[Path], None]
):
    file_path = tmp_path / "f.txt"
    file_path.write_text("hello\n")
    hanging_open(file_path)
    previous = signal.signal(signal.SIGALRM, signal.SIG_IGN)
    try:
        proc = await asyncio.create_subprocess_exec(
            "/bin/cat",
            preexec_fn=open_regular_as_stdin(str(file_path), None),
            stdout=asyncio.subprocess.PIPE,
        )
        await communicate_or_kill(proc, timeout_s=5)
    finally:
        signal.signal(signal.SIGALRM, previous)
    assert proc.returncode == -signal.SIGALRM


@pytest.mark.asyncio
async def test_open_timer_does_not_outlive_exec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    file_path = tmp_path / "f.txt"
    file_path.write_text("hello\n")
    monkeypatch.setattr("karotte.demoted.OPEN_TIMEOUT_S", 0.3)
    proc = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        "sleep 1; cat",
        preexec_fn=open_regular_as_stdin(str(file_path), None),
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await communicate_or_kill(proc, timeout_s=5)
    assert (proc.returncode, stdout) == (0, b"hello\n")


@pytest.mark.asyncio
async def test_open_regular_as_stdin_does_not_mask_demote_failure(tmp_path: Path):
    file_path = tmp_path / "f.txt"
    file_path.write_text("hello\n")

    def failing_demote() -> None:
        raise PermissionError("setuid failed")

    with pytest.raises(subprocess.SubprocessError):
        await asyncio.create_subprocess_exec(
            "/bin/cat",
            preexec_fn=open_regular_as_stdin(str(file_path), failing_demote),
        )


@pytest.mark.requires_root
@pytest.mark.asyncio
@pytest.mark.parametrize(("mode", "readable"), [(0o600, False), (0o644, True)])
async def test_open_regular_as_stdin_opens_as_the_demoted_uid(
    monkeypatch: pytest.MonkeyPatch, mode: int, readable: bool
):
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "60123")
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        os.chmod(d, 0o755)
        file_path = Path(d) / "f.txt"
        file_path.write_text("hello\n")
        file_path.chmod(mode)
        proc = await asyncio.create_subprocess_exec(
            "/bin/cat",
            preexec_fn=open_regular_as_stdin(str(file_path), make_demote_fn()),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await communicate_or_kill(proc)

    if readable:
        assert (proc.returncode, stdout) == (0, b"hello\n")
    else:
        assert (proc.returncode, stderr) == (OPEN_REFUSED_EXIT, b"Permission denied")


@pytest.mark.requires_root
@pytest.mark.asyncio
@pytest.mark.parametrize(("mode", "writable"), [(0o644, False), (0o666, True)])
async def test_open_regular_as_stdout_opens_as_the_demoted_uid(
    monkeypatch: pytest.MonkeyPatch, mode: int, writable: bool
):
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "60123")
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        os.chmod(d, 0o755)
        file_path = Path(d) / "f.txt"
        file_path.write_text("hello\n")
        file_path.chmod(mode)
        proc = await asyncio.create_subprocess_exec(
            "/bin/cat",
            preexec_fn=open_regular_as_stdout(str(file_path), make_demote_fn()),
            stdin=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await communicate_or_kill(proc, b"new\n")
        content = file_path.read_text()

    if writable:
        assert (proc.returncode, content) == (0, "new\n")
    else:
        assert (proc.returncode, stderr, content) == (
            OPEN_REFUSED_EXIT,
            b"Permission denied",
            "hello\n",
        )


_CLOSED_STDIN_SCRIPT = """\
import os, subprocess, sys
from karotte.demoted import open_regular_as_stdin

os.close(0)
r = subprocess.run(
    ["/bin/cat"], preexec_fn=open_regular_as_stdin(sys.argv[1], None), capture_output=True
)
sys.stdout.buffer.write(r.stdout)
sys.exit(r.returncode)
"""


def test_open_regular_as_stdin_works_when_the_parent_has_no_stdin(tmp_path: Path):
    file_path = tmp_path / "f.txt"
    file_path.write_text("hello\n")
    r = subprocess.run(
        [sys.executable, "-c", _CLOSED_STDIN_SCRIPT, str(file_path)],
        capture_output=True,
        timeout=30,
    )
    assert (r.returncode, r.stdout, r.stderr) == (0, b"hello\n", b"")
