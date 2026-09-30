import asyncio
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from fastmcp.tools.tool import ToolResult
from karotte.demoted import check_access as _check_access
from karotte.demoted import communicate_or_kill as _communicate_or_kill
from karotte.subprocess import make_demote_fn
from karotte.text_files import decode_text
from karotte.truncation import head_within_json_bytes, json_encoded_len

_MAX_FILE_BYTES = 10 * 1024 * 1024
_DIFF_CONTEXT_LINES = 3
_MAX_DIFF_OCCURRENCES = 50
_MAX_DIFF_BYTES = 64 * 1024
_DIFF_TRUNCATION_NOTE = "\n(diff truncated)\n"


@dataclass
class _Block:
    """A run of replacements and the whole lines of ``old_content`` they touch."""

    start: int
    end: int
    replaced: int
    new_text: str


def _line_start(text: str, pos: int) -> int:
    return text.rfind("\n", 0, pos) + 1


def _line_end(text: str, pos: int) -> int:
    nl = text.find("\n", pos)
    return len(text) if nl == -1 else nl + 1


def _window(content: str, blocks: list[_Block]) -> tuple[int, int]:
    """The context-padded span of a hunk, snapped to line boundaries."""
    ctx_start = blocks[0].start
    for _ in range(_DIFF_CONTEXT_LINES):
        if ctx_start == 0:
            break
        ctx_start = _line_start(content, ctx_start - 1)
    ctx_end = blocks[-1].end
    for _ in range(_DIFF_CONTEXT_LINES):
        if ctx_end >= len(content):
            break
        ctx_end = _line_end(content, ctx_end)
    return ctx_start, ctx_end


def _emit(out: list[str], prefix: str, text: str) -> int:
    parts = text.split("\n")
    for part in parts[:-1]:
        out.append(f"{prefix}{part}\n")
    if parts[-1]:
        out.append(f"{prefix}{parts[-1]}\n\\ No newline at end of file\n")
        return len(parts)
    return len(parts) - 1


def _replacement_blocks(content: str, old: str, new: str, count: int) -> list[_Block]:
    """Blocks for as many of the first ``count`` replacements as fit the cap.

    Scans only as far as those reach, since ``replace_all`` on a repetitive file
    has millions of matches; a block needing more than the cap allows is dropped
    whole so a shown block stays exact.
    """
    blocks: list[_Block] = []
    remaining = count
    shown = 0
    match = content.find(old) if remaining > 0 else -1
    while match != -1 and shown < _MAX_DIFF_OCCURRENCES:
        start = _line_start(content, match)
        end = _line_end(content, match + len(old) - 1)
        replaced = 0
        last_match_end = 0
        while True:
            while match != -1 and match < end:
                if shown + replaced == _MAX_DIFF_OCCURRENCES:
                    return blocks
                end = max(end, _line_end(content, match + len(old) - 1))
                last_match_end = match + len(old)
                replaced += 1
                remaining -= 1
                match = content.find(old, match + len(old)) if remaining > 0 else -1
            # A replacement that ate the block's last newline joins the next
            # line onto it; the diff has to show that line as changed too.
            ends_with_newline = (
                new.endswith("\n")
                if last_match_end == end
                else content[end - 1] == "\n"
            )
            if ends_with_newline or end >= len(content):
                break
            end = _line_end(content, end)
        blocks.append(
            _Block(start, end, replaced, content[start:end].replace(old, new, replaced))
        )
        shown += replaced
    return blocks


def _replacement_diff(
    old_content: str, old: str, new: str, count: int, filename: str
) -> str:
    """Unified diff of the first ``count`` replacements of ``old`` by ``new``.

    ``difflib`` is not used: it omits the ``\\ No newline at end of file``
    marker, which makes its output ambiguous on a file without a final newline.
    """
    blocks = _replacement_blocks(old_content, old, new, count)

    hunks: list[list[_Block]] = []
    for block in blocks:
        if hunks and (
            old_content.count("\n", hunks[-1][-1].end, block.start)
            <= 2 * _DIFF_CONTEXT_LINES
        ):
            hunks[-1].append(block)
        else:
            hunks.append([block])

    out = [f"--- {filename}\n", f"+++ {filename}\n"]
    delta = 0
    old_line = 1
    counted_to = 0
    for hunk in hunks:
        ctx_start, ctx_end = _window(old_content, hunk)

        body: list[str] = []
        old_n = new_n = _emit(body, " ", old_content[ctx_start : hunk[0].start])
        for k, block in enumerate(hunk):
            old_n += _emit(body, "-", old_content[block.start : block.end])
            new_n += _emit(body, "+", block.new_text)
            gap_end = hunk[k + 1].start if k + 1 < len(hunk) else ctx_end
            gap_n = _emit(body, " ", old_content[block.end : gap_end])
            old_n += gap_n
            new_n += gap_n

        old_line += old_content.count("\n", counted_to, ctx_start)
        counted_to = ctx_start
        out.append(f"@@ -{old_line},{old_n} +{old_line + delta},{new_n} @@\n")
        out.extend(body)
        delta += new_n - old_n

    shown = sum(block.replaced for block in blocks)
    if shown < count:
        out.append(f"({count - shown} more replacement(s) not shown)\n")

    diff = "".join(out)
    kept = head_within_json_bytes(
        diff, _MAX_DIFF_BYTES - json_encoded_len(_DIFF_TRUNCATION_NOTE)
    )
    if len(kept) < len(diff):
        return kept + _DIFF_TRUNCATION_NOTE
    return diff


async def replace_in_file(
    file_path: Path,
    old: str,
    new: str,
    replace_all: bool = False,
) -> ToolResult:
    """Perform exact string replacement in a file.

    The file_path must be an absolute path.

    By default, only the first occurrence of `old` is replaced. Set `replace_all=True` to replace every occurrence.

    The replacement will FAIL if `old` is not found in the file.
    Use `view_lines_in_file` to see the exact file contents, including whitespace and indentation, before attempting the replacement.

    When replacing code, preserve the exact indentation (tabs/spaces) as it appears in the file.
    """
    if not file_path.is_absolute():
        raise ValueError(f"File path must be absolute: {str(file_path)!r}")

    if old == "":
        raise ValueError("String to replace must not be empty")

    path_str = str(file_path)

    # Verify file exists and has read/write permission via demoted subprocess
    if not await _check_access("-e", path_str):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not await _check_access("-r", path_str):
        raise PermissionError(f"No read permission for file: {file_path}")

    if not await _check_access("-w", path_str):
        raise PermissionError(f"No write permission for file: {file_path}")

    # Fast honest-path gate; the read below stays bounded even when a race
    # defeats this check.
    try:
        st = os.stat(path_str)
    except OSError as e:
        raise OSError(f"Failed to stat file {file_path}: {e}") from e
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f"Not a regular file: {file_path}")
    if st.st_size > _MAX_FILE_BYTES:
        raise ValueError(
            f"File too large to edit in place: {file_path} is {st.st_size} bytes "
            + f"(limit {_MAX_FILE_BYTES} bytes)"
        )

    # Read via a demoted subprocess for the same TOCTOU reason as the write
    # below: reading directly from this (root) process would bypass the
    # permission check if the path were swapped after `_check_access`.
    # `head -c` (not `cat`) keeps the read bounded even if a symlink flip
    # defeats the os.stat() gate above; anything hitting the cap is rejected.
    read_proc = await asyncio.create_subprocess_exec(
        "/usr/bin/head",
        "-c",
        str(_MAX_FILE_BYTES + 1),
        path_str,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        preexec_fn=make_demote_fn(),
    )
    try:
        stdout, stderr = await _communicate_or_kill(read_proc)
    except TimeoutError:
        raise OSError(f"Timed out reading file {file_path}") from None
    if read_proc.returncode != 0:
        raise OSError(
            f"Failed to read file {file_path}: {stderr.decode('utf-8', errors='replace').strip()}"
        )
    if len(stdout) > _MAX_FILE_BYTES:
        raise ValueError(
            f"File too large to edit in place: {file_path} exceeds "
            + f"{_MAX_FILE_BYTES} bytes (read cap)"
        )
    old_content = decode_text(stdout, file_path)

    if old not in old_content:
        raise ValueError(f"String not found in file: {file_path}")

    # The read cap does not bound the output: replace_all amplifies it by the
    # occurrence count.
    occurrences = old_content.count(old) if replace_all else 1
    new_bytes = len(new.encode("utf-8"))
    projected = len(stdout) + occurrences * (new_bytes - len(old.encode("utf-8")))
    if projected > _MAX_FILE_BYTES:
        raise ValueError(
            f"Replacement would grow {file_path} to ~{projected} bytes, over the "
            + f"{_MAX_FILE_BYTES}-byte limit ({occurrences} occurrence(s) x "
            + f"{new_bytes} bytes of replacement text). "
            + "Narrow the match or replace in smaller steps."
        )

    if replace_all:
        new_content = old_content.replace(old, new)
    else:
        new_content = old_content.replace(old, new, 1)

    diff_output = _replacement_diff(old_content, old, new, occurrences, str(file_path))

    # Write via a demoted subprocess so the kernel enforces permissions at
    # write time. Writing directly from this (root) process would be subject
    # to a TOCTOU race: the path could be swapped for a symlink to a
    # privileged file between the `_check_access` calls and the write.
    proc = await asyncio.create_subprocess_exec(
        "/usr/bin/tee",
        path_str,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
        preexec_fn=make_demote_fn(),
    )
    try:
        _, stderr = await _communicate_or_kill(proc, new_content.encode("utf-8"))
    except TimeoutError:
        raise OSError(f"Timed out writing file {file_path}") from None
    if proc.returncode != 0:
        raise OSError(
            f"Failed to write file {file_path}: {stderr.decode('utf-8', errors='replace').strip()}"
        )

    return ToolResult(
        structured_content={
            "result": "Replacement successful",
            "diff": diff_output,
        }
    )
