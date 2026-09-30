import asyncio
import base64
import os
import stat
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from karotte.demoted import drain_bounded as _drain_bounded
from karotte.tools.view_image_file import (
    BASE64_PATH,
    ViewImageFileConfig,
    _b64_encoded_len,  # pyright: ignore[reportPrivateUsage]
    _read_capped,  # pyright: ignore[reportPrivateUsage]
    view_image_file,
)


def pillow_installed() -> bool:
    try:
        import PIL  # noqa: F401  # pyright: ignore[reportUnusedImport, reportMissingImports]

        return True
    except ImportError:
        return False


@pytest.mark.skipif(pillow_installed(), reason="Pillow is installed")
@pytest.mark.asyncio
async def test_prints_instructions_if_pillow_not_installed(resource_dir: Path):
    """Test that an ImportError with instructions is raised if Pillow is not installed."""
    with pytest.raises(ImportError):
        await view_image_file()((resource_dir / "image_1px.png").absolute())


@pytest.mark.skipif(not pillow_installed(), reason="Pillow is not installed")
@pytest.mark.usefixtures("gnu_base64")
@pytest.mark.asyncio
async def test_ignores_malicious_base64_in_path(
    resource_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A malicious ``base64`` earlier in PATH must not be executed."""
    malicious_dir = tmp_path / "malicious_bin"
    malicious_dir.mkdir()
    malicious_base64 = malicious_dir / "base64"
    marker = tmp_path / "pwned"
    malicious_base64.write_text(f"#!/bin/sh\ntouch {marker}\n")
    malicious_base64.chmod(0o755)

    monkeypatch.setenv("PATH", f"{malicious_dir}:{os.environ.get('PATH', '')}")

    result = await view_image_file()((resource_dir / "image_1px.png").absolute())

    assert not marker.exists(), "Malicious base64 was executed!"
    assert result.content is not None


# ---------------------------------------------------------------------------
# Byte caps: size gate, bounded read, timeout
# ---------------------------------------------------------------------------

_GNU_BASE64_SHIM = """\
#!/usr/bin/env python3
import base64, sys

sys.stdout.write(base64.b64encode(open(sys.argv[3], "rb").read()).decode() + "\\n")
"""


@pytest.fixture
def gnu_base64(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
):
    """The tool hardcodes GNU `base64 -w 0 <file>`; macOS base64 supports neither
    the flag nor a file argument, so swap in a shim where the real one is missing."""
    probe = subprocess.run(
        [BASE64_PATH, "-w", "0", "/dev/null"], capture_output=True, check=False
    )
    if probe.returncode == 0:
        yield
        return
    shim = tmp_path_factory.mktemp("bin") / "base64"
    shim.write_text(_GNU_BASE64_SHIM)
    shim.chmod(0o755)
    monkeypatch.setattr("karotte.tools.view_image_file.BASE64_PATH", str(shim))
    yield


class _FakeRegularStat:
    st_mode: int = stat.S_IFREG | 0o644
    st_size: int = 100


def _stat_faking_regular_file(target: Path):
    """os.stat replacement reporting ``target`` as a small regular file,
    delegating everything else to the real stat."""
    real_stat = os.stat

    def fake(path: object, *args: object, **kwargs: object):
        if isinstance(path, (str, Path)) and str(path) == str(target):
            return _FakeRegularStat()
        return real_stat(path, *args, **kwargs)  # pyright: ignore[reportArgumentType]

    return fake


def test_b64_encoded_len_matches_real_base64():
    for n in (0, 1, 2, 3, 4, 100, 65537):
        assert _b64_encoded_len(n) == len(base64.b64encode(b"x" * n))


@pytest.mark.asyncio
async def test_drain_bounded_caps_but_consumes_to_eof():
    reader = asyncio.StreamReader()
    reader.feed_data(b"e" * 100_000)
    reader.feed_eof()

    out = await _drain_bounded(reader, 1000)

    assert len(out) == 1000
    assert reader.at_eof()


@pytest.mark.asyncio
async def test_read_capped_returns_all_under_cap():
    reader = asyncio.StreamReader()
    reader.feed_data(b"abc")
    reader.feed_eof()

    data, hit_cap = await _read_capped(reader, 1000)

    assert data == b"abc"
    assert not hit_cap


@pytest.mark.asyncio
async def test_read_capped_exactly_at_cap_is_not_flagged():
    reader = asyncio.StreamReader()
    reader.feed_data(b"x" * 1000)
    reader.feed_eof()

    data, hit_cap = await _read_capped(reader, 1000)

    assert data == b"x" * 1000
    assert not hit_cap


@pytest.mark.asyncio
async def test_read_capped_stops_just_past_cap():
    reader = asyncio.StreamReader()
    reader.feed_data(b"x" * 100_000)
    reader.feed_eof()

    data, hit_cap = await _read_capped(reader, 1000)

    assert hit_cap
    assert len(data) == 1000


@pytest.mark.skipif(not pillow_installed(), reason="Pillow is not installed")
@pytest.mark.asyncio
async def test_oversized_file_rejected_before_read(tmp_path: Path):
    big = tmp_path / "big.png"
    big.write_bytes(b"\x00" * (6 * 1024 * 1024))

    with pytest.raises(ValueError, match="too large") as exc_info:
        await view_image_file()(big.absolute())

    text = str(exc_info.value)
    assert str(6 * 1024 * 1024) in text
    assert str(5 * 1024 * 1024) in text


@pytest.mark.skipif(not pillow_installed(), reason="Pillow is not installed")
@pytest.mark.asyncio
async def test_max_file_bytes_is_configurable(tmp_path: Path):
    tool = view_image_file()
    small = ViewImageFileConfig(max_file_bytes=1024)
    f = tmp_path / "f.png"
    f.write_bytes(b"\x00" * 2048)

    with (
        patch.object(tool, "config", small),
        pytest.raises(ValueError, match="too large"),
    ):
        await tool(f.absolute())


@pytest.mark.skipif(not pillow_installed(), reason="Pillow is not installed")
@pytest.mark.usefixtures("gnu_base64")
@pytest.mark.asyncio
async def test_file_exactly_at_cap_passes_the_size_gate(tmp_path: Path):
    at_cap = tmp_path / "at_cap.png"
    at_cap.write_bytes(b"\x00" * (5 * 1024 * 1024))

    with pytest.raises(Exception) as exc_info:
        await view_image_file()(at_cap.absolute())

    # Rejected by PIL as not-an-image, NOT by the size gate.
    assert "too large" not in str(exc_info.value).lower()


@pytest.mark.skipif(not pillow_installed(), reason="Pillow is not installed")
@pytest.mark.usefixtures("gnu_base64")
@pytest.mark.asyncio
async def test_small_valid_image_roundtrip(resource_dir: Path):
    result = await view_image_file()((resource_dir / "image_1px.png").absolute())

    assert result.content is not None


@pytest.mark.skipif(not pillow_installed(), reason="Pillow is not installed")
@pytest.mark.usefixtures("gnu_base64")
@pytest.mark.asyncio
async def test_read_stays_bounded_when_stat_gate_is_bypassed(tmp_path: Path):
    """A symlink flip between the stat and the read defeats the size gate; the
    capped read must still reject the file instead of buffering it whole."""
    big = tmp_path / "big.bin"
    big.write_bytes(b"\x00" * (20 * 1024 * 1024))

    with (
        patch("karotte.tools.view_image_file.os.stat", _stat_faking_regular_file(big)),
        pytest.raises(ValueError, match="too large"),
    ):
        await view_image_file()(big.absolute())


@pytest.mark.skipif(not pillow_installed(), reason="Pillow is not installed")
@pytest.mark.usefixtures("gnu_base64")
@pytest.mark.asyncio
async def test_fifo_swapped_in_after_stat_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A FIFO flipped in after the checks must time out, not hang the call on
    the blocked open."""
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    monkeypatch.setattr("karotte.demoted.SUBPROCESS_TIMEOUT_S", 0.5)

    with patch(
        "karotte.tools.view_image_file.os.stat", _stat_faking_regular_file(fifo)
    ):
        with pytest.raises(RuntimeError, match="Timed out"):
            await view_image_file()(fifo.absolute())
