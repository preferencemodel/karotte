import os
from pathlib import Path

import pytest

from karotte.agents.student_files import (
    ScanBudget,
    open_student_file,
    student_file_first_line,
    student_file_lines,
    student_files_in,
    student_subdirs,
)


def test_lines_of_a_file(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    path.write_text('{"a": 1}\n{"b": 2}\n')
    assert list(student_file_lines(path)) == ['{"a": 1}', '{"b": 2}']


def test_a_missing_file_has_no_lines(tmp_path: Path):
    assert list(student_file_lines(tmp_path / "nope.jsonl")) == []


def test_a_linked_file_is_not_read(tmp_path: Path):
    secret = tmp_path / "secret"
    secret.write_text("secret\n")
    (tmp_path / "log.jsonl").symlink_to(secret)
    assert list(student_file_lines(tmp_path / "log.jsonl")) == []


def test_a_file_behind_a_linked_directory_is_not_read(tmp_path: Path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "log.jsonl").write_text("secret\n")
    (tmp_path / "linked").symlink_to(elsewhere)
    assert list(student_file_lines(tmp_path / "linked" / "log.jsonl")) == []


def test_a_file_the_student_does_not_own_is_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "karotte.agents.student_files.demoted_uid_gid", lambda: os.getuid() + 1
    )
    path = tmp_path / "log.jsonl"
    path.write_text("x\n")
    assert list(student_file_lines(path)) == []
    with pytest.raises(OSError, match="not a regular file the student owns"):
        _ = open_student_file(path)


def test_a_fifo_is_not_read(tmp_path: Path):
    os.mkfifo(tmp_path / "fifo")
    assert list(student_file_lines(tmp_path / "fifo")) == []


def test_open_returns_none_while_the_file_does_not_exist(tmp_path: Path):
    assert open_student_file(tmp_path / "nope") is None


def test_subdirs(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "file").write_text("x")
    (tmp_path / "link").symlink_to(tmp_path / "a")
    assert sorted(student_subdirs(tmp_path, ScanBudget())) == ["a", "b"]


def test_subdirs_of_a_missing_or_linked_directory(tmp_path: Path):
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "a").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real")
    assert student_subdirs(tmp_path / "nope", ScanBudget()) == []
    assert student_subdirs(tmp_path / "link", ScanBudget()) == []


def test_files_in(tmp_path: Path):
    (tmp_path / "a.jsonl").write_text("x")
    (tmp_path / "dir").mkdir()
    (tmp_path / "link.jsonl").symlink_to(tmp_path / "a.jsonl")
    os.mkfifo(tmp_path / "fifo")
    assert student_files_in(tmp_path, ScanBudget()) == ["a.jsonl"]


def test_an_overlong_line_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("karotte.agents.student_files._MAX_LINE_BYTES", 10)
    path = tmp_path / "log.jsonl"
    path.write_text("short\n" + "x" * 35 + "\nafter\n")
    assert list(student_file_lines(path)) == ["short", "after"]


def test_reading_stops_at_the_byte_limit(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    path.write_text("one\ntwo\nthree\n")
    assert list(student_file_lines(path, max_bytes=8)) == ["one", "two"]


def test_a_huge_line_without_newline_is_not_held_in_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("karotte.agents.student_files._MAX_LINE_BYTES", 1024)
    path = tmp_path / "sparse.jsonl"
    with path.open("wb") as f:
        f.truncate(1024**3)
    assert list(student_file_lines(path, max_bytes=1024**2)) == []


def test_a_scan_lists_no_more_entries_than_its_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("karotte.agents.student_files._SCAN_ENTRIES", 1 + 3)
    for name in "abcde":
        (tmp_path / name).mkdir()
    assert len(student_subdirs(tmp_path, ScanBudget())) == 3


def test_a_scan_shares_its_budget_between_listings_and_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("karotte.agents.student_files._SCAN_ENTRIES", 1 + 2)
    (tmp_path / "a").mkdir()
    (tmp_path / "log.jsonl").write_text("x\n")
    budget = ScanBudget()
    assert student_subdirs(tmp_path, budget) == ["a"]
    assert list(student_file_lines(tmp_path / "log.jsonl", budget=budget)) == []


def test_a_scan_stops_reading_once_its_bytes_are_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("karotte.agents.student_files._SCAN_BYTES", 8)
    (tmp_path / "one").write_text("one\ntwo\n")
    (tmp_path / "two").write_text("three\n")
    budget = ScanBudget()
    assert list(student_file_lines(tmp_path / "one", budget=budget)) == ["one", "two"]
    assert list(student_file_lines(tmp_path / "two", budget=budget)) == []


def test_the_first_line(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    path.write_text("a\nb\n")
    assert student_file_first_line(path, ScanBudget()) == "a"


def test_an_overlong_first_line_is_not_replaced_by_the_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("karotte.agents.student_files._MAX_LINE_BYTES", 10)
    path = tmp_path / "log.jsonl"
    path.write_text("x" * 35 + "\nsecond\n")
    assert student_file_first_line(path, ScanBudget()) is None


def test_the_first_line_of_a_huge_file_costs_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("karotte.agents.student_files._MAX_LINE_BYTES", 1024)
    monkeypatch.setattr("karotte.agents.student_files._SCAN_BYTES", 1024**2)
    path = tmp_path / "sparse.jsonl"
    with path.open("wb") as f:
        f.truncate(1024**3)
    budget = ScanBudget()
    assert student_file_first_line(path, budget) is None
    assert 1024**2 - budget.bytes <= 1025
