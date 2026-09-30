"""Bounded helpers for the demoted subprocesses the file tools spawn."""

import asyncio

from karotte.subprocess import make_demote_fn

TEST_PATH = "/usr/bin/test"
SUBPROCESS_TIMEOUT_S = 30.0


def kill_quietly(proc: asyncio.subprocess.Process) -> None:
    try:
        proc.kill()
    except ProcessLookupError:
        pass


async def reap(proc: asyncio.subprocess.Process) -> None:
    """Kill the process if it is still running and collect it."""
    if proc.returncode is None:
        kill_quietly(proc)
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
