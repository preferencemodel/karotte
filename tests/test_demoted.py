import asyncio
import sys

import psutil
import pytest

from karotte.demoted import check_access, communicate_or_kill

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
async def test_check_access_times_out_and_leaves_no_child(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("karotte.demoted.TEST_PATH", "/bin/sh")
    before = _children()
    with pytest.raises(TimeoutError):
        await check_access("-c", "sleep 30", timeout_s=0.2)
    assert _children() - before == set()
