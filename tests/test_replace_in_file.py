import asyncio
import os
import random
import re
import stat
import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from karotte.tools.replace_in_file import (
    _MAX_DIFF_BYTES,  # pyright: ignore[reportPrivateUsage]
    _MAX_DIFF_OCCURRENCES,  # pyright: ignore[reportPrivateUsage]
    _replacement_diff,  # pyright: ignore[reportPrivateUsage]
    replace_in_file,
)
from karotte.truncation import json_encoded_len


class _FakeRegularStat:
    st_mode: int = stat.S_IFREG | 0o644
    st_size: int = 100


def _fake_stat(_path: str) -> _FakeRegularStat:
    return _FakeRegularStat()


def _bypass_stat_gate():
    """Simulate a symlink flip after the checks: access passes, stat lies."""
    return (
        patch(
            "karotte.tools.replace_in_file._check_access", AsyncMock(return_value=True)
        ),
        patch("karotte.tools.replace_in_file.os.stat", _fake_stat),
    )


@pytest.mark.asyncio
async def test_raises_error_if_file_path_is_not_absolute():
    with pytest.raises(ValueError, match="File path must be absolute"):
        await replace_in_file(Path("relative/path"), "old", "new")


@pytest.mark.asyncio
async def test_quoted_path_shows_its_quotes_in_the_error():
    # Interpolated bare this reads as a correctly-quoted absolute path; repr
    # makes the quotes visibly part of the value.
    with pytest.raises(ValueError, match=r"'\"/workdir/solution.py\"'"):
        await replace_in_file(Path('"/workdir/solution.py"'), "old", "new")


@pytest.mark.asyncio
async def test_replaces_first_occurrence_by_default(tmp_path: Path):
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("old old old")

    await replace_in_file(file_path, "old", "new")

    assert file_path.read_text() == "new old old"


@pytest.mark.asyncio
async def test_replaces_all_occurrences_when_replace_all_is_true(tmp_path: Path):
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("old old old")

    await replace_in_file(file_path, "old", "new", replace_all=True)

    assert file_path.read_text() == "new new new"


@pytest.mark.asyncio
async def test_file_does_not_exist(tmp_path: Path):
    file_path = tmp_path / "nonexistent.txt"

    with pytest.raises(FileNotFoundError):
        await replace_in_file(file_path, "old", "new")


@pytest.mark.asyncio
async def test_search_string_not_found_raises_error(tmp_path: Path):
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("hello world")

    with pytest.raises(ValueError, match="String not found in file"):
        await replace_in_file(file_path, "xyz", "abc")


@pytest.mark.asyncio
async def test_empty_file_raises_error_when_string_not_found(tmp_path: Path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("")

    with pytest.raises(ValueError, match="String not found in file"):
        await replace_in_file(file_path, "old", "new")


@pytest.mark.asyncio
async def test_empty_old_string_raises_error(tmp_path: Path):
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("hello world")

    with pytest.raises(ValueError, match="String to replace must not be empty"):
        await replace_in_file(file_path, "", "new")

    assert file_path.read_text() == "hello world"


@pytest.mark.asyncio
async def test_empty_old_string_raises_error_replace_all(tmp_path: Path):
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("hello world")

    with pytest.raises(ValueError, match="String to replace must not be empty"):
        await replace_in_file(file_path, "", "new", replace_all=True)

    assert file_path.read_text() == "hello world"


@pytest.mark.asyncio
async def test_empty_old_string_raises_error_on_empty_file(tmp_path: Path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("")

    with pytest.raises(ValueError, match="String to replace must not be empty"):
        await replace_in_file(file_path, "", "new")

    assert file_path.read_text() == ""


@pytest.mark.asyncio
async def test_empty_replacement_string(tmp_path: Path):
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("hello old world old test")

    await replace_in_file(file_path, "old ", "", replace_all=True)

    assert file_path.read_text() == "hello world test"


@pytest.mark.asyncio
async def test_special_regex_characters(tmp_path: Path):
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("Price: $10.99 (sale)")

    await replace_in_file(file_path, "$10.99", "$8.99")

    assert file_path.read_text() == "Price: $8.99 (sale)"


@pytest.mark.asyncio
async def test_multiline_content_single_replacement(tmp_path: Path):
    file_path = tmp_path / "test_file.txt"
    content = "line 1\nold content\nline 3\nold content again"
    file_path.write_text(content)

    await replace_in_file(file_path, "old content", "new content")

    expected = "line 1\nnew content\nline 3\nold content again"
    assert file_path.read_text() == expected


@pytest.mark.asyncio
async def test_multiline_content_replace_all(tmp_path: Path):
    file_path = tmp_path / "test_file.txt"
    content = "line 1\nold content\nline 3\nold content again"
    file_path.write_text(content)

    await replace_in_file(file_path, "old content", "new content", replace_all=True)

    expected = "line 1\nnew content\nline 3\nnew content again"
    assert file_path.read_text() == expected


@pytest.mark.asyncio
async def test_unicode_content(tmp_path: Path):
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("héllo wörld émojis 🚀")

    await replace_in_file(file_path, "wörld", "universe")

    assert file_path.read_text() == "héllo universe émojis 🚀"


@pytest.mark.asyncio
async def test_partial_match_single_replacement(tmp_path: Path):
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("older oldish old")

    await replace_in_file(file_path, "old", "new")

    assert file_path.read_text() == "newer oldish old"


@pytest.mark.asyncio
async def test_partial_match_replace_all(tmp_path: Path):
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("older oldish old")

    await replace_in_file(file_path, "old", "new", replace_all=True)

    assert file_path.read_text() == "newer newish new"


@pytest.mark.asyncio
async def test_case_sensitive_replacement(tmp_path: Path):
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("Old old OLD")

    await replace_in_file(file_path, "old", "new")

    assert file_path.read_text() == "Old new OLD"


@pytest.mark.asyncio
async def test_newline_characters(tmp_path: Path):
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("line1\nold\nline3")

    await replace_in_file(file_path, "\nold\n", "\nnew\n")

    assert file_path.read_text() == "line1\nnew\nline3"


@pytest.mark.asyncio
async def test_file_truncated_when_content_shrinks(tmp_path: Path):
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("very long content that will be replaced")

    await replace_in_file(file_path, "very long content that will be replaced", "short")

    assert file_path.read_text() == "short"


@pytest.mark.asyncio
async def test_file_with_trailing_newline(tmp_path: Path):
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("line1\nline2\n")

    await replace_in_file(file_path, "line2", "line3")

    assert file_path.read_text() == "line1\nline3\n"


@pytest.mark.asyncio
async def test_file_path_with_spaces(tmp_path: Path):
    """Test that file paths with spaces are handled correctly."""
    file_path = tmp_path / "test file with spaces.txt"
    file_path.write_text("hello world")

    await replace_in_file(file_path, "hello", "goodbye")

    assert file_path.read_text() == "goodbye world"


@pytest.mark.asyncio
async def test_returns_diff_in_result(tmp_path: Path):
    """Test that the result includes a unified diff."""
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("hello world")

    result = await replace_in_file(file_path, "hello", "goodbye")

    assert result.structured_content is not None
    assert "diff" in result.structured_content
    diff = result.structured_content["diff"]
    assert "-hello world" in diff
    assert "+goodbye world" in diff


@pytest.mark.asyncio
async def test_verifies_permissions_via_demoted_subprocess(tmp_path: Path):
    """Permission checks should go through _check_access (demoted subprocess),
    not through direct Python file-access checks."""
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("hello world")

    mock_check = AsyncMock(return_value=True)

    with patch("karotte.tools.replace_in_file._check_access", mock_check):
        await replace_in_file(file_path, "hello", "goodbye")

    checked_flags = [call.args[0] for call in mock_check.call_args_list]
    assert "-e" in checked_flags
    assert "-r" in checked_flags
    assert "-w" in checked_flags


@pytest.mark.asyncio
async def test_ignores_malicious_test_in_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A malicious ``test`` binary earlier in PATH must not be executed."""
    malicious_dir = tmp_path / "malicious_bin"
    malicious_dir.mkdir()
    malicious_test = malicious_dir / "test"
    marker = tmp_path / "pwned"
    malicious_test.write_text(f"#!/bin/sh\ntouch {marker}\n")
    malicious_test.chmod(0o755)

    monkeypatch.setenv("PATH", f"{malicious_dir}:{os.environ.get('PATH', '')}")

    file_path = tmp_path / "test_file.txt"
    file_path.write_text("hello world")

    await replace_in_file(file_path, "hello", "goodbye")

    assert not marker.exists(), "Malicious test binary was executed!"
    assert file_path.read_text() == "goodbye world"


@pytest.mark.asyncio
async def test_refuses_file_over_size_cap(tmp_path: Path):
    """A file larger than the buffering cap is rejected BEFORE its contents are
    read into the (root) server process — the model-triggerable OOM-void lever."""
    file_path = tmp_path / "huge.txt"
    file_path.write_bytes(b"x" * (11 * 1024 * 1024))

    with pytest.raises(ValueError, match="too large") as exc_info:
        await replace_in_file(file_path, "x", "y")

    message = str(exc_info.value)
    assert str(11 * 1024 * 1024) in message
    assert str(10 * 1024 * 1024) in message


@pytest.mark.asyncio
async def test_accepts_file_exactly_at_size_cap(tmp_path: Path):
    """The cap is exclusive: a file of exactly the limit is still editable."""
    file_path = tmp_path / "at_cap.txt"
    file_path.write_bytes(b"a" * (10 * 1024 * 1024 - 3) + b"old")

    await replace_in_file(file_path, "old", "new")

    assert file_path.read_bytes().endswith(b"new")


@pytest.mark.asyncio
async def test_refuses_symlink_to_non_regular_file(tmp_path: Path):
    """The regular-file check follows symlinks, so a symlink can't smuggle a
    FIFO or device past the guard."""
    fifo_path = tmp_path / "fifo"
    os.mkfifo(fifo_path)
    link_path = tmp_path / "link"
    link_path.symlink_to(fifo_path)

    with pytest.raises(ValueError, match="Not a regular file"):
        await replace_in_file(link_path, "old", "new")


@pytest.mark.asyncio
async def test_file_removed_after_access_check_is_reported(tmp_path: Path):
    """If the file vanishes between the access checks and the stat, the
    failure is reported as a stat error rather than an unhandled exception."""
    file_path = tmp_path / "gone.txt"

    with patch(
        "karotte.tools.replace_in_file._check_access", AsyncMock(return_value=True)
    ):
        with pytest.raises(OSError, match="Failed to stat"):
            await replace_in_file(file_path, "old", "new")


@pytest.mark.asyncio
async def test_refuses_non_regular_file(tmp_path: Path):
    """A non-regular path (e.g. a FIFO / device) is rejected before the read, so
    an unbounded stream like /dev/zero cannot be buffered into the server."""
    fifo_path = tmp_path / "fifo"
    os.mkfifo(fifo_path)

    with pytest.raises(ValueError, match="Not a regular file"):
        await replace_in_file(fifo_path, "old", "new")


@pytest.mark.asyncio
async def test_read_stays_bounded_when_stat_gate_is_bypassed(tmp_path: Path):
    """A symlink flip between the stat and the read defeats the size gate; the
    capped read must still reject the file instead of buffering it."""
    file_path = tmp_path / "huge.txt"
    file_path.write_bytes(b"x" * (11 * 1024 * 1024))

    check_access, stat_gate = _bypass_stat_gate()
    with check_access, stat_gate:
        with pytest.raises(ValueError, match="read cap"):
            await replace_in_file(file_path, "x", "y")

    assert file_path.stat().st_size == 11 * 1024 * 1024


@pytest.mark.asyncio
async def test_fifo_swapped_in_after_stat_is_refused_without_waiting(tmp_path: Path):
    """A FIFO flipped in after the regular-file check must be refused at open,
    not block the read until the timeout."""
    fifo_path = tmp_path / "fifo"
    os.mkfifo(fifo_path)

    check_access, stat_gate = _bypass_stat_gate()
    with check_access, stat_gate:
        with pytest.raises(ValueError, match="^Not a regular file: "):
            async with asyncio.timeout(5):
                await replace_in_file(fifo_path, "old", "new")


@pytest.mark.asyncio
async def test_fifo_swapped_in_before_write_is_refused_without_waiting(
    tmp_path: Path,
):
    file_path = tmp_path / "f.txt"
    file_path.write_text("old")

    def swap_then_diff(*args: object, **kwargs: object) -> str:
        file_path.unlink()
        os.mkfifo(file_path)
        return _replacement_diff(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    with patch("karotte.tools.replace_in_file._replacement_diff", swap_then_diff):
        with pytest.raises(ValueError, match="^Not a regular file: "):
            async with asyncio.timeout(5):
                await replace_in_file(file_path, "old", "new")


@pytest.mark.asyncio
async def test_replace_all_amplification_is_refused(tmp_path: Path):
    """A file under the read cap can still blow past it once replace_all
    multiplies the replacement by the occurrence count."""
    file_path = tmp_path / "amp.txt"
    file_path.write_text("a" * (1024 * 1024))

    with pytest.raises(ValueError, match="over the .*-byte limit"):
        await replace_in_file(file_path, "a", "b" * 64, replace_all=True)

    assert file_path.read_text() == "a" * (1024 * 1024)


@pytest.mark.asyncio
async def test_a_shrinking_replace_all_is_still_allowed(tmp_path: Path):
    """The guard bounds growth, it does not forbid replace_all."""
    file_path = tmp_path / "shrink.txt"
    file_path.write_text("longneedle " * 1000)

    await replace_in_file(file_path, "longneedle", "x", replace_all=True)
    assert file_path.read_text() == "x " * 1000


@pytest.mark.asyncio
async def test_amplification_is_measured_in_bytes_not_characters(tmp_path: Path):
    """A 3-byte replacement character counts as 3 against the byte cap."""
    file_path = tmp_path / "multibyte.txt"
    file_path.write_text("a" * (1024 * 1024))

    with pytest.raises(ValueError, match="over the .*-byte limit"):
        await replace_in_file(file_path, "a", "€" * 5, replace_all=True)


@pytest.mark.asyncio
async def test_non_utf8_file_is_refused(tmp_path: Path):
    file_path = tmp_path / "image.png"
    file_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\xff" * 32)

    with (
        patch(
            "karotte.tools.replace_in_file._check_access", AsyncMock(return_value=True)
        ),
        pytest.raises(ValueError, match="Not a UTF-8 text file"),
    ):
        await replace_in_file(file_path, "PNG", "JPG")


NO_NEWLINE = "\\ No newline at end of file"


def _apply_unified_diff(content: str, diff: str) -> str:
    """Apply a unified diff the way ``patch`` would, rejecting a malformed one."""
    lines = diff.split("\n")[:-1]
    assert lines[0].startswith("--- ") and lines[1].startswith("+++ "), diff
    if lines[-1].startswith("("):
        lines = lines[:-1]
    out: list[str] = []
    consumed = 0
    index = 2
    while index < len(lines):
        header = re.fullmatch(
            r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", lines[index]
        )
        assert header is not None, lines[index]
        old_start = int(header.group(1))
        old_count = 1 if header.group(2) is None else int(header.group(2))
        new_count = 1 if header.group(4) is None else int(header.group(4))
        index += 1
        old_side: list[str] = []
        new_side: list[str] = []
        last = " "

        def drop_newline(sides: list[list[str]]) -> None:
            for side in sides:
                assert side and side[-1].endswith("\n"), diff
                side[-1] = side[-1][:-1]

        while len(old_side) < old_count or len(new_side) < new_count:
            line = lines[index]
            index += 1
            if line == NO_NEWLINE:
                drop_newline(
                    [old_side, new_side]
                    if last == " "
                    else [_side(last, old_side, new_side)]
                )
                continue
            last = line[0]
            assert last in " -+", diff
            if last in " -":
                old_side.append(line[1:] + "\n")
            if last in " +":
                new_side.append(line[1:] + "\n")
        while index < len(lines) and lines[index] == NO_NEWLINE:
            index += 1
            drop_newline(
                [old_side, new_side]
                if last == " "
                else [_side(last, old_side, new_side)]
            )
        offset = _offset_of_line(content, old_start)
        assert offset >= consumed, diff
        out.append(content[consumed:offset])
        assert content[offset:].startswith("".join(old_side)), diff
        consumed = offset + len("".join(old_side))
        out.append("".join(new_side))
    out.append(content[consumed:])
    return "".join(out)


def _side(prefix: str, old_side: list[str], new_side: list[str]) -> list[str]:
    return new_side if prefix == "+" else old_side


def _offset_of_line(content: str, line_number: int) -> int:
    offset = 0
    for _ in range(line_number - 1):
        offset = content.index("\n", offset) + 1
    return offset


def _diff_of(content: str, old: str, new: str, replace_all: bool) -> tuple[str, str]:
    count = content.count(old) if replace_all else 1
    return content.replace(old, new, count), _replacement_diff(
        content, old, new, count, "f"
    )


def test_diff_shows_the_joined_line_when_a_replacement_eats_a_newline():
    content = "x\nfoo\nbar\n"
    new_content, diff = _diff_of(content, "foo\n", "foo ", False)
    assert "-bar" in diff and "+foo bar" in diff
    assert _apply_unified_diff(content, diff) == new_content


def _assert_diff_is_exact(content: str, old: str, new: str, diff: str) -> None:
    """Applying the diff makes exactly the replacements it claims to show."""
    note = re.search(r"\((\d+) more replacement", diff)
    hidden = int(note.group(1)) if note else 0
    shown = content.count(old) - hidden
    assert _apply_unified_diff(content, diff) == content.replace(old, new, shown)


def test_diff_beyond_the_occurrence_cap_stays_exact():
    content = "".join(f"a{i}\n" for i in range(_MAX_DIFF_OCCURRENCES + 10))
    _, diff = _diff_of(content, "a", "bb", True)
    assert "more replacement(s) not shown" in diff
    _assert_diff_is_exact(content, "a", "bb", diff)


def test_a_line_needing_more_than_the_cap_is_not_shown_half_done():
    content = "a" * (_MAX_DIFF_OCCURRENCES + 5) + "\n"
    _, diff = _diff_of(content, "a", "b", True)
    assert "@@" not in diff
    _assert_diff_is_exact(content, "a", "b", diff)


def test_diff_at_the_cap_boundary_on_a_shared_line():
    content = "a" * (_MAX_DIFF_OCCURRENCES - 1) + "\n" + "a\n"
    _, diff = _diff_of(content, "a", "b", True)
    _assert_diff_is_exact(content, "a", "b", diff)


def test_diff_of_a_repetitive_file_is_not_quadratic():
    content = "".join(f"L{i % 200}\n" for i in range(30_000)) + "a\n"
    start = time.monotonic()
    new_content, diff = _diff_of(content, "a\n", "b\n", False)
    assert time.monotonic() - start < 1.0
    assert _apply_unified_diff(content, diff) == new_content


@pytest.mark.parametrize("seed", range(500))
def test_diff_applies_to_the_replaced_content(seed: int):
    rng = random.Random(seed)
    content = "".join(rng.choices("ab\n ", k=rng.randint(1, 40)))
    start = rng.randint(0, len(content) - 1)
    old = content[start : rng.randint(start + 1, len(content))]
    new = "".join(rng.choices("ab\n ", k=rng.randint(0, 5)))
    replace_all = rng.random() < 0.5
    new_content, diff = _diff_of(content, old, new, replace_all)
    if "more replacement(s) not shown" in diff or "(diff truncated)" in diff:
        return
    assert _apply_unified_diff(content, diff) == new_content


def test_diff_of_millions_of_matches_scans_only_what_it_shows():
    content = "a" * (5 * 1024 * 1024)
    start = time.monotonic()
    _, diff = _diff_of(content, "a", "b", True)
    assert time.monotonic() - start < 1.0
    assert "@@" not in diff
    _assert_diff_is_exact(content, "a", "b", diff)


def test_stripping_every_space_from_a_big_file_returns_a_small_diff():
    content = "".join(f"  line {i} = {i} + {i};\n" for i in range(8_000))
    assert len(content) > 165_000
    _, diff = _diff_of(content, " ", "", True)
    assert json_encoded_len(diff) <= _MAX_DIFF_BYTES
    assert "more replacement(s) not shown" in diff
    _assert_diff_is_exact(content, " ", "", diff)


def test_a_huge_replacement_is_cut_to_the_diff_budget():
    content = "a\n"
    _, diff = _diff_of(content, "a", "b" * (2 * _MAX_DIFF_BYTES), False)
    assert json_encoded_len(diff) <= _MAX_DIFF_BYTES
    assert diff.endswith("(diff truncated)\n")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("binary", "message"),
    [("HEAD_PATH", "Timed out reading file"), ("TEE_PATH", "Timed out writing file")],
)
async def test_stalled_subprocess_times_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stalling_binary: str,
    binary: str,
    message: str,
):
    file_path = tmp_path / "f.txt"
    file_path.write_text("old")
    monkeypatch.setattr(f"karotte.tools.replace_in_file.{binary}", stalling_binary)
    monkeypatch.setattr("karotte.demoted.SUBPROCESS_TIMEOUT_S", 0.5)

    with pytest.raises(OSError, match=message):
        async with asyncio.timeout(10):
            await replace_in_file(file_path, "old", "new")


@pytest.mark.asyncio
async def test_hanging_open_times_out(
    tmp_path: Path, hanging_open: Callable[[Path], None]
):
    file_path = tmp_path / "f.txt"
    file_path.write_text("old")
    hanging_open(file_path)

    with pytest.raises(TimeoutError, match="^Timed out opening: "):
        async with asyncio.timeout(10):
            await replace_in_file(file_path, "old", "new")


@pytest.mark.asyncio
async def test_missing_tee_leaves_the_file_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    file_path = tmp_path / "f.txt"
    file_path.write_text("old")
    monkeypatch.setattr(
        "karotte.tools.replace_in_file.TEE_PATH", str(tmp_path / "missing-tee")
    )

    with pytest.raises(OSError, match="is not available"):
        await replace_in_file(file_path, "old", "new")
    assert file_path.read_text() == "old"
