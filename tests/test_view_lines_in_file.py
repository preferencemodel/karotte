import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastmcp import Client, FastMCP
from mcp.types import TextContent

from karotte.tools.view_lines_in_file import (
    _MAX_CONTENT_BYTES,  # pyright: ignore[reportPrivateUsage]
    view_lines_in_file,
)
from karotte.truncation import json_encoded_len


def _access_stub(responses: dict[str, bool]) -> AsyncMock:
    async def _check(flag: str, _file_path: str) -> bool:
        return responses[flag]

    return AsyncMock(side_effect=_check)


@pytest.mark.asyncio
async def test_raises_error_if_file_path_is_not_absolute():
    with pytest.raises(ValueError, match="File path must be absolute"):
        await view_lines_in_file()(Path("relative/path"), from_line=1, to_line=2)


@pytest.mark.asyncio
async def test_raises_error_if_file_does_not_exist(tmp_path: Path):
    file_path = tmp_path / "nonexistent.txt"

    with pytest.raises(FileNotFoundError):
        await view_lines_in_file()(file_path, from_line=1, to_line=2)


@pytest.mark.asyncio
async def test_raises_error_if_from_line_is_less_than_1(tmp_path: Path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("Line 1\nLine 2\nLine 3")

    with pytest.raises(
        ValueError, match="`from_line` must be greater than or equal to 1"
    ):
        await view_lines_in_file()(file_path, from_line=0, to_line=2)


@pytest.mark.asyncio
async def test_raises_error_if_to_line_is_less_than_from_line(tmp_path: Path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("Line 1\nLine 2\nLine 3")

    with pytest.raises(
        ValueError, match="`to_line` must be greater than or equal to `from_line`"
    ):
        await view_lines_in_file()(file_path, from_line=2, to_line=1)


@pytest.mark.asyncio
async def test_read_all_lines(tmp_path: Path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("Line 1\nLine 2\nLine 3")

    result = await view_lines_in_file()(file_path, from_line=1, to_line=3)

    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["file_path"] == str(file_path)
    assert json.loads(result.content[0].text)["from_line"] == 1
    assert json.loads(result.content[0].text)["to_line"] == 3
    assert json.loads(result.content[0].text)["content"] == "Line 1\nLine 2\nLine 3"


@pytest.mark.asyncio
async def test_lines_in_the_middle(tmp_path: Path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("Line 1\nLine 2\nLine 3\nLine 4")

    result = await view_lines_in_file()(file_path, from_line=2, to_line=3)

    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["file_path"] == str(file_path)
    assert json.loads(result.content[0].text)["from_line"] == 2
    assert json.loads(result.content[0].text)["to_line"] == 3
    assert json.loads(result.content[0].text)["content"] == "Line 2\nLine 3\n"


@pytest.mark.asyncio
async def test_to_line_larger_then_number_of_lines(tmp_path: Path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("Line 1\nLine 2\nLine 3")

    result = await view_lines_in_file()(file_path, from_line=1, to_line=4)

    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["file_path"] == str(file_path)
    assert json.loads(result.content[0].text)["from_line"] == 1
    assert json.loads(result.content[0].text)["to_line"] == 4
    assert json.loads(result.content[0].text)["content"] == "Line 1\nLine 2\nLine 3"


@pytest.mark.asyncio
async def test_raises_error_if_requesting_too_many_lines(tmp_path: Path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("Line 1\nLine 2\nLine 3")

    with pytest.raises(ValueError, match="limit is 1000 lines"):
        await view_lines_in_file()(file_path, from_line=1, to_line=1001)


@pytest.mark.asyncio
async def test_ignores_malicious_sed_in_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A malicious ``sed`` earlier in PATH must not be executed."""
    malicious_dir = tmp_path / "malicious_bin"
    malicious_dir.mkdir()
    malicious_sed = malicious_dir / "sed"
    marker = tmp_path / "pwned"
    malicious_sed.write_text(f"#!/bin/sh\ntouch {marker}\n")
    malicious_sed.chmod(0o755)

    monkeypatch.setenv("PATH", f"{malicious_dir}:{os.environ.get('PATH', '')}")

    file_path = tmp_path / "test.txt"
    file_path.write_text("Line 1\nLine 2\nLine 3")

    result = await view_lines_in_file()(file_path, from_line=1, to_line=3)

    assert not marker.exists(), "Malicious sed was executed!"
    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["content"] == "Line 1\nLine 2\nLine 3"


@pytest.mark.asyncio
async def test_sed_failure_includes_exit_code_and_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """When sed fails, the error message should include the exit code and stderr."""
    fake_sed = tmp_path / "fake_sed"
    fake_sed.write_text("#!/bin/sh\necho 'some error' >&2\nexit 2\n")
    fake_sed.chmod(0o755)

    file_path = tmp_path / "test.txt"
    file_path.write_text("Line 1\nLine 2\nLine 3")

    monkeypatch.setattr("karotte.tools.view_lines_in_file.SED_PATH", str(fake_sed))

    with pytest.raises(RuntimeError, match="exit code 2"):
        await view_lines_in_file()(file_path, from_line=1, to_line=3)

    with pytest.raises(RuntimeError, match="some error"):
        await view_lines_in_file()(file_path, from_line=1, to_line=3)


@pytest.mark.asyncio
async def test_sed_failure_includes_exit_code_when_stderr_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """When sed fails with empty stderr (the original bug), the error should still
    include the exit code and file path so the failure is diagnosable."""
    fake_sed = tmp_path / "fake_sed"
    fake_sed.write_text("#!/bin/sh\nexit 1\n")
    fake_sed.chmod(0o755)

    file_path = tmp_path / "test.txt"
    file_path.write_text("Line 1\nLine 2\nLine 3")

    monkeypatch.setattr("karotte.tools.view_lines_in_file.SED_PATH", str(fake_sed))

    with pytest.raises(RuntimeError, match="exit code 1"):
        await view_lines_in_file()(file_path, from_line=1, to_line=3)

    with pytest.raises(RuntimeError, match=str(file_path)):
        await view_lines_in_file()(file_path, from_line=1, to_line=3)


@pytest.mark.asyncio
async def test_from_line_beyond_eof_reports_total_lines(tmp_path: Path):
    """When from_line is past the end of the file, the error should
    tell the agent how many lines the file actually has."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("Line 1\nLine 2\nLine 3\n")

    with pytest.raises(ValueError, match="3 lines") as exc_info:
        await view_lines_in_file()(file_path, from_line=50, to_line=60)

    assert "50" in str(exc_info.value)


@pytest.mark.asyncio
async def test_from_line_beyond_eof_counts_lines_without_trailing_newline(
    tmp_path: Path,
):
    """A file whose last line has no trailing newline still reports the
    correct total (awk counts the partial last line as a record)."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("Line 1\nLine 2\nLine 3")

    with pytest.raises(ValueError, match="3 lines"):
        await view_lines_in_file()(file_path, from_line=50, to_line=60)


@pytest.mark.asyncio
async def test_from_line_beyond_eof_on_empty_file_reports_zero_lines(tmp_path: Path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("")

    with pytest.raises(ValueError, match="0 lines"):
        await view_lines_in_file()(file_path, from_line=2, to_line=3)


@pytest.mark.asyncio
async def test_from_line_beyond_eof_does_not_read_file_into_server_process(
    tmp_path: Path,
):
    """The past-EOF error path must count lines via a streaming subprocess,
    never by reading the file into this (root) server process."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("Line 1\n")

    with patch.object(
        Path,
        "read_text",
        side_effect=AssertionError("read_text called in server process"),
    ):
        with pytest.raises(ValueError, match="1 lines"):
            await view_lines_in_file()(file_path, from_line=5, to_line=6)


@pytest.mark.asyncio
async def test_from_line_beyond_eof_degrades_when_awk_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """If the line count can't be taken, the error still reports the requested
    range but omits the count instead of failing differently."""
    fake_awk = tmp_path / "fake_awk"
    fake_awk.write_text("#!/bin/sh\nexit 1\n")
    fake_awk.chmod(0o755)
    monkeypatch.setattr("karotte.tools.view_lines_in_file.AWK_PATH", str(fake_awk))

    file_path = tmp_path / "test.txt"
    file_path.write_text("Line 1\n")

    with pytest.raises(ValueError, match="requested lines 50-60") as exc_info:
        await view_lines_in_file()(file_path, from_line=50, to_line=60)

    assert "has" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_from_line_beyond_eof_degrades_on_unparseable_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake_awk = tmp_path / "fake_awk"
    fake_awk.write_text("#!/bin/sh\necho banana\n")
    fake_awk.chmod(0o755)
    monkeypatch.setattr("karotte.tools.view_lines_in_file.AWK_PATH", str(fake_awk))

    file_path = tmp_path / "test.txt"
    file_path.write_text("Line 1\n")

    with pytest.raises(ValueError, match="requested lines 50-60") as exc_info:
        await view_lines_in_file()(file_path, from_line=50, to_line=60)

    assert "banana" not in str(exc_info.value)
    assert "has" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_requesting_too_many_lines_reports_how_many_requested(tmp_path: Path):
    """The 'too many lines' error should say how many were requested
    and what the limit is."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("Line 1")

    with pytest.raises(ValueError, match="1500 lines") as exc_info:
        await view_lines_in_file()(file_path, from_line=1, to_line=1500)

    assert "1000" in str(exc_info.value)


@pytest.mark.asyncio
async def test_access_is_checked_via_demoted_subprocess(tmp_path: Path):
    """Existence and permission must be verified through _check_access (a
    demoted subprocess), not via a privileged is_file() check in this process."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("Line 1\nLine 2\nLine 3")

    mock_check = AsyncMock(return_value=True)

    with patch("karotte.tools.view_lines_in_file._check_access", mock_check):
        await view_lines_in_file()(file_path, from_line=1, to_line=3)

    assert "-r" in [call.args[0] for call in mock_check.call_args_list]


@pytest.mark.asyncio
async def test_inaccessible_dir_reports_permission_never_not_found(tmp_path: Path):
    """A caller who cannot search the directory must get a permission error
    whether or not the file exists, so existence cannot be probed as an oracle.
    Existence must not be checked, and the message must not imply the file exists."""
    file_path = tmp_path / "secret.txt"

    stub = _access_stub({"-r": False, "-x": False})
    with patch("karotte.tools.view_lines_in_file._check_access", stub):
        with pytest.raises(PermissionError) as exc_info:
            await view_lines_in_file()(file_path, from_line=1, to_line=3)

    assert "-e" not in [call.args[0] for call in stub.call_args_list]
    message = str(exc_info.value).lower()
    assert "not found" not in message
    assert "file" not in message


@pytest.mark.asyncio
async def test_existing_but_unreadable_file_reports_permission(tmp_path: Path):
    """When the directory is searchable but the file is not readable, the
    caller gets a permission error rather than the file contents."""
    file_path = tmp_path / "test.txt"

    stub = _access_stub({"-r": False, "-x": True, "-e": True})
    with patch("karotte.tools.view_lines_in_file._check_access", stub):
        with pytest.raises(PermissionError):
            await view_lines_in_file()(file_path, from_line=1, to_line=3)


@pytest.mark.asyncio
async def test_missing_file_in_searchable_dir_reports_not_found(tmp_path: Path):
    """When the caller can search the directory, a genuinely missing file is
    reported as not found (revealing this leaks nothing they can't already list)."""
    file_path = tmp_path / "test.txt"

    stub = _access_stub({"-r": False, "-x": True, "-e": False})
    with patch("karotte.tools.view_lines_in_file._check_access", stub):
        with pytest.raises(FileNotFoundError):
            await view_lines_in_file()(file_path, from_line=1, to_line=3)


@pytest.mark.asyncio
async def test_structured_content_passes_mcp_output_schema_validation(tmp_path: Path):
    # Tests for https://github.com/PrefectHQ/fastmcp/issues/3528
    file_path = tmp_path / "test.txt"
    file_path.write_text("Line 1\nLine 2\nLine 3")

    server = FastMCP("test")
    instance = view_lines_in_file()
    server.tool(instance.__call__, name="view_lines_in_file")

    async with Client(server) as client:
        await client.call_tool(
            "view_lines_in_file",
            {"file_path": str(file_path), "from_line": 1, "to_line": 3},
        )


@pytest.mark.asyncio
async def test_huge_single_line_file_is_bounded(tmp_path: Path):
    # A model can write a multi-hundred-MB file with no newlines; the `max_lines`
    # cap bounds line COUNT only, so the tool must also bound the BYTES it buffers
    # into the root MCP-server process or it OOMs the server and wedges grading.
    # Output is capped and marked truncated instead of buffering the whole line.
    file_path = tmp_path / "bigline.txt"
    file_path.write_text("A" * (3 * 1024 * 1024))  # 3 MiB, single line, no newline

    stub = _access_stub({"-r": True})
    with patch("karotte.tools.view_lines_in_file._check_access", stub):
        result = await view_lines_in_file()(file_path, from_line=1, to_line=1)

    assert result.structured_content is not None
    content = result.structured_content["content"]
    assert json_encoded_len(content) <= _MAX_CONTENT_BYTES
    assert "truncated" in content


@pytest.mark.asyncio
async def test_cap_counts_json_encoded_bytes(tmp_path: Path):
    # Control characters JSON-escape to 6 bytes each, so a file well under the
    # raw cap still has to be cut to fit the budget of the serialized result.
    file_path = tmp_path / "control.txt"
    file_path.write_bytes(b"\x01" * (_MAX_CONTENT_BYTES // 2))

    stub = _access_stub({"-r": True})
    with patch("karotte.tools.view_lines_in_file._check_access", stub):
        result = await view_lines_in_file()(file_path, from_line=1, to_line=1)

    assert result.structured_content is not None
    content = result.structured_content["content"]
    assert json_encoded_len(content) <= _MAX_CONTENT_BYTES
    assert content.startswith("\x01" * 1000)
    assert "truncated" in content


@pytest.mark.asyncio
async def test_truncation_may_cut_a_multibyte_character(tmp_path: Path):
    file_path = tmp_path / "multibyte.txt"
    file_path.write_text("é" * (_MAX_CONTENT_BYTES + 1))

    stub = _access_stub({"-r": True})
    with patch("karotte.tools.view_lines_in_file._check_access", stub):
        result = await view_lines_in_file()(file_path, from_line=1, to_line=1)

    assert result.structured_content is not None
    content = result.structured_content["content"]
    assert content.startswith("éé")
    assert "\ufffd" not in content
    assert "truncated" in content


@pytest.mark.asyncio
async def test_non_utf8_file_is_refused(tmp_path: Path):
    file_path = tmp_path / "image.png"
    file_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\xff" * 32)

    stub = _access_stub({"-r": True})
    with (
        patch("karotte.tools.view_lines_in_file._check_access", stub),
        pytest.raises(ValueError, match="Not a UTF-8 text file"),
    ):
        await view_lines_in_file()(file_path, from_line=1, to_line=1)


@pytest.mark.asyncio
async def test_large_stderr_does_not_deadlock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # stderr is only drained after stdout hits EOF. A child that fills the stderr
    # pipe buffers before writing stdout blocks on write(2), so stdout never gets
    # data and both sides wait on each other forever.
    fake_sed = tmp_path / "fake_sed"
    fake_sed.write_text(
        "#!/bin/sh\nhead -c 1048576 /dev/zero >&2\nprintf 'Line 1\\n'\n"
    )
    fake_sed.chmod(0o755)

    file_path = tmp_path / "test.txt"
    file_path.write_text("Line 1\n")

    monkeypatch.setattr("karotte.tools.view_lines_in_file.SED_PATH", str(fake_sed))

    stub = _access_stub({"-r": True})
    with patch("karotte.tools.view_lines_in_file._check_access", stub):
        result = await asyncio.wait_for(
            view_lines_in_file()(file_path, from_line=1, to_line=1), timeout=10
        )

    assert result.structured_content is not None
    assert result.structured_content["content"] == "Line 1\n"


@pytest.mark.asyncio
async def test_cancellation_kills_sed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Cancelling the tool call while it waits on stdout must not leave the sed
    # child running (or the stderr drain task orphaned).
    pid_file = tmp_path / "pid"
    fake_sed = tmp_path / "fake_sed"
    fake_sed.write_text(f"#!/bin/sh\necho $$ > {pid_file}\nexec sleep 30\n")
    fake_sed.chmod(0o755)

    file_path = tmp_path / "test.txt"
    file_path.write_text("Line 1\n")

    monkeypatch.setattr("karotte.tools.view_lines_in_file.SED_PATH", str(fake_sed))

    stub = _access_stub({"-r": True})
    with patch("karotte.tools.view_lines_in_file._check_access", stub):
        task = asyncio.create_task(
            view_lines_in_file()(file_path, from_line=1, to_line=1)
        )
        async with asyncio.timeout(10):
            while not pid_file.exists() or not pid_file.read_text().strip():
                await asyncio.sleep(0.01)
            pid = int(pid_file.read_text())

            task.cancel()
            done, _ = await asyncio.wait({task}, timeout=5)
            assert task in done, "cancellation did not finish within 5s"
            assert task.cancelled()

            while True:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_rejects_fifo(tmp_path: Path):
    """A FIFO passes `test -r`, but sed would block in open(2) forever."""
    fifo = tmp_path / "probe_fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="Not a regular file"):
        await view_lines_in_file()(fifo, from_line=1, to_line=20)


@pytest.mark.asyncio
async def test_rejects_character_device():
    """/dev/zero never yields EOF; sed would accumulate it forever."""
    with pytest.raises(ValueError, match="Not a regular file"):
        await view_lines_in_file()(Path("/dev/zero"), from_line=1, to_line=20)


@pytest.mark.asyncio
async def test_rejects_directory():
    with pytest.raises(ValueError, match="Not a regular file"):
        await view_lines_in_file()(Path("/tmp"), from_line=1, to_line=20)


@pytest.mark.asyncio
async def test_reads_through_symlink_to_regular_file(tmp_path: Path):
    """The regular-file check must follow symlinks, not reject them."""
    target = tmp_path / "target.txt"
    target.write_text("Line 1\nLine 2\n")
    link = tmp_path / "link.txt"
    link.symlink_to(target)

    result = await view_lines_in_file()(link, from_line=1, to_line=2)

    assert result.structured_content is not None
    assert result.structured_content["content"] == "Line 1\nLine 2\n"
