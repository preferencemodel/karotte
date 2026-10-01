import asyncio
import os
import stat
from pathlib import Path
from typing import final

from fastmcp.tools.tool import ToolResult
from karotte import ToolBase, demoted
from karotte.demoted import check_access as _check_access
from karotte.demoted import (
    communicate_or_kill,
    drain_bounded,
    open_error,
    open_regular_as_stdin,
    reap,
)
from karotte.subprocess import make_demote_fn
from karotte.text_files import decode_text
from karotte.truncation import head_within_json_bytes, json_encoded_len
from pydantic import BaseModel

SED_PATH = "/usr/bin/sed"
AWK_PATH = "/usr/bin/awk"

_MAX_CONTENT_BYTES = 384 * 1024
_MAX_STDERR_BYTES = 16 * 1024
_TRUNCATION_NOTE = (
    f"\n\n[... output truncated: exceeded {_MAX_CONTENT_BYTES} bytes; "
    "request a narrower line range ...]"
)


async def _count_lines(file_path: str) -> int | None:
    """Exact line count via a demoted, *streaming* ``awk`` — never reads the file
    into this (root) server process. Returns ``None`` if the count can't be taken
    so callers can degrade to a message without it. Using ``read_text()`` here
    instead would let a model point the tool at a multi-GB file and OOM the server,
    voiding the episode."""
    proc = await asyncio.create_subprocess_exec(
        AWK_PATH,
        "END{print NR}",
        preexec_fn=open_regular_as_stdin(file_path, make_demote_fn()),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, _ = await communicate_or_kill(proc)
    except TimeoutError:
        return None
    if proc.returncode != 0:
        return None
    try:
        return int(stdout.decode("utf-8", errors="replace").strip())
    except ValueError:
        return None


class ViewLinesInFileConfig(BaseModel):
    max_lines: int = 1000


@final
class view_lines_in_file(ToolBase[ViewLinesInFileConfig]):
    config_schema = ViewLinesInFileConfig

    async def __call__(
        self, file_path: Path, from_line: int, to_line: int
    ) -> ToolResult:
        """Read lines from a file.

        The file_path must be an absolute path. Lines are 1-indexed (first line is 1). Both from_line and to_line are inclusive.

        You can read up to 1000 lines at a time. For larger files, make multiple calls with different ranges.

        To read the beginning of a file, use from_line=1. To continue reading, use from_line=(previous to_line + 1).
        """

        if from_line < 1:
            raise ValueError("`from_line` must be greater than or equal to 1")

        if to_line < from_line:
            raise ValueError("`to_line` must be greater than or equal to `from_line`")

        requested = to_line - from_line + 1
        if requested > self.config.max_lines:
            raise ValueError(
                f"Requested {requested} lines, but the limit is {self.config.max_lines} lines per call"
            )

        if not file_path.is_absolute():
            raise ValueError(f"File path must be absolute: {str(file_path)!r}")

        path_str = str(file_path)

        if not await _check_access("-r", path_str):
            if not await _check_access("-x", str(file_path.parent)):
                raise PermissionError(f"Permission denied: {file_path}")
            if not await _check_access("-e", path_str):
                raise FileNotFoundError(f"File not found: {file_path}")
            raise PermissionError(f"Permission denied: {file_path}")

        if not stat.S_ISREG(os.stat(path_str).st_mode):
            raise ValueError(f"Not a regular file: {file_path}")

        proc = await asyncio.create_subprocess_exec(
            SED_PATH,
            "-n",
            f"{from_line},{to_line}p;{to_line + 1}q",
            preexec_fn=open_regular_as_stdin(path_str, make_demote_fn()),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Bound what is buffered into this (root) process; sed is killed once
        # the cap is hit. stderr is drained concurrently so sed never blocks on it.
        assert proc.stdout is not None
        assert proc.stderr is not None
        stderr_task = asyncio.create_task(drain_bounded(proc.stderr, _MAX_STDERR_BYTES))
        try:
            async with asyncio.timeout(demoted.SUBPROCESS_TIMEOUT_S):
                buf = bytearray()
                while len(buf) <= _MAX_CONTENT_BYTES:
                    chunk = await proc.stdout.read(65536)
                    if not chunk:
                        break
                    buf += chunk
                truncated = len(buf) > _MAX_CONTENT_BYTES
                if truncated:
                    del buf[_MAX_CONTENT_BYTES:]
                    await reap(proc)

                try:
                    stderr = await stderr_task
                except Exception:  # noqa: BLE001 - stderr is diagnostic only
                    stderr = b""
                await proc.wait()
        except TimeoutError:
            raise RuntimeError(f"Timed out reading {file_path}") from None
        finally:
            stderr_task.cancel()
            await reap(proc)

        if err := open_error(file_path, proc.returncode, stderr):
            raise err

        # A cap-triggered kill yields a nonzero/negative returncode that is not an
        # error; only surface sed failures when we did NOT truncate.
        if not truncated and proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            parts = [f"Failed to read {file_path} (exit code {proc.returncode})"]
            if stderr_text:
                parts.append(stderr_text)
            raise RuntimeError(". ".join(parts))

        decoded = decode_text(bytes(buf), file_path, complete=not truncated)
        content = head_within_json_bytes(
            decoded, _MAX_CONTENT_BYTES - json_encoded_len(_TRUNCATION_NOTE)
        )
        if truncated or len(content) < len(decoded):
            truncated = True
            content += _TRUNCATION_NOTE

        if not truncated and not content and from_line > 1:
            total = await _count_lines(path_str)
            have = f" has {total} lines" if total is not None else ""
            raise ValueError(
                f"File {file_path}{have}, but you requested lines {from_line}-{to_line}"
            )

        return ToolResult(
            structured_content={
                "file_path": str(file_path),
                "from_line": from_line,
                "to_line": to_line,
                "content": content,
            }
        )
