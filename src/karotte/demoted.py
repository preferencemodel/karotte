"""Bounded helpers for the demoted subprocesses the file tools spawn."""

import asyncio
import errno
import os
import signal
import stat
from collections.abc import Callable
from typing import NoReturn

from karotte.subprocess import make_demote_fn

TEST_PATH = "/usr/bin/test"
SUBPROCESS_TIMEOUT_S = 30.0
OPEN_REFUSED_EXIT = 125
OPEN_TIMEOUT_S = 5.0


def kill_quietly(proc: asyncio.subprocess.Process) -> None:
    try:
        proc.kill()
    except ProcessLookupError:
        pass


async def reap(proc: asyncio.subprocess.Process) -> None:
    """Kill the process if it is still running and collect it."""
    if proc.returncode is None:
        kill_quietly(proc)
        # On Python 3.12, wait() also waits for EOF on unread pipes (cpython#119710).
        for fd in (1, 2):
            if pipe := proc._transport.get_pipe_transport(fd):  # pyright: ignore[reportAttributeAccessIssue]
                pipe.close()
        await proc.wait()


async def communicate_or_kill(
    proc: asyncio.subprocess.Process,
    input_bytes: bytes | None = None,
    timeout_s: float | None = None,
) -> tuple[bytes, bytes]:
    try:
        async with asyncio.timeout(timeout_s or SUBPROCESS_TIMEOUT_S):
            return await proc.communicate(input_bytes)
    finally:
        await reap(proc)


async def check_access(
    flag: str, file_path: str, timeout_s: float | None = None
) -> bool:
    """Run ``test <flag> <file_path>`` as the demoted uid."""
    proc = await asyncio.create_subprocess_exec(
        TEST_PATH, flag, file_path, preexec_fn=make_demote_fn()
    )
    try:
        async with asyncio.timeout(timeout_s or SUBPROCESS_TIMEOUT_S):
            return await proc.wait() == 0
    finally:
        await reap(proc)


async def drain_bounded(stream: asyncio.StreamReader, cap: int) -> bytes:
    """Read to EOF, keeping at most ``cap`` bytes, so the writer never blocks."""
    buf = bytearray()
    while chunk := await stream.read(65536):
        if len(buf) < cap:
            buf += chunk
    return bytes(buf[:cap])


def _refuse(reason: str) -> NoReturn:
    try:
        os.write(2, reason.encode())
    except OSError:
        pass
    os._exit(OPEN_REFUSED_EXIT)


def _open_regular_onto(
    path: str,
    flags: int,
    target_fd: int,
    demote: Callable[[], None] | None,
    truncate: bool = False,
) -> Callable[[], None]:
    def preexec() -> None:
        if demote is not None:
            demote()
        # Popen blocks the event loop until exec, so a hung open would freeze the server.
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.setitimer(signal.ITIMER_REAL, OPEN_TIMEOUT_S)
        try:
            fd = os.open(path, flags | os.O_NONBLOCK | os.O_NOCTTY)
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                _refuse("Not a regular file")
            os.set_blocking(fd, True)
            if truncate:
                os.ftruncate(fd, 0)
            if fd == target_fd:
                os.set_inheritable(fd, True)
            else:
                os.dup2(fd, target_fd)
                os.close(fd)
        except OSError as e:
            # A FIFO without a reader, a socket, or a device without a driver.
            if e.errno == errno.ENXIO:
                _refuse("Not a regular file")
            _refuse(e.strerror or "Cannot open")
        # Interval timers survive exec.
        signal.setitimer(signal.ITIMER_REAL, 0)

    return preexec


def open_regular_as_stdin(
    path: str, demote: Callable[[], None] | None
) -> Callable[[], None]:
    """preexec_fn that demotes, then opens ``path`` as the child's stdin.

    Unless ``path`` opens as a regular file within ``OPEN_TIMEOUT_S``, the child fails
    before exec in a way :func:`open_error` recognizes.
    """
    return _open_regular_onto(path, os.O_RDONLY, 0, demote)


def open_regular_as_stdout(
    path: str, demote: Callable[[], None] | None
) -> Callable[[], None]:
    """Like :func:`open_regular_as_stdin`, but opens ``path`` truncated as stdout."""
    return _open_regular_onto(path, os.O_WRONLY, 1, demote, truncate=True)


def open_error(
    file_path: object, returncode: int | None, stderr: bytes
) -> Exception | None:
    """The error for a child launched with ``open_regular_as_*``, if its open failed."""
    if returncode == OPEN_REFUSED_EXIT:
        reason = stderr.decode("utf-8", errors="replace").strip()
        return ValueError(f"{reason or 'Cannot open'}: {file_path}")
    if returncode == -signal.SIGALRM:
        return TimeoutError(f"Timed out opening: {file_path}")
    return None
