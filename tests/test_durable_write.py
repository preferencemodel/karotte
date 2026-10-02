from pathlib import Path

import pytest

from karotte.durable_write import write_durably


def test_the_file_and_its_directory_are_fsynced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A VM powered off right after the run loses writes still in the page cache."""
    synced: list[int] = []
    monkeypatch.setattr("karotte.durable_write.os.fsync", synced.append)
    path = tmp_path / "transcript.json"

    write_durably(path, "{}")

    assert path.read_text() == "{}"
    # The file and its directory, whose entry for it the file's fsync misses.
    assert len(synced) == 2


def test_created_directories_are_fsynced_up_to_an_existing_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    synced: list[Path] = []
    monkeypatch.setattr("karotte.durable_write._fsync_directory", synced.append)
    path = tmp_path / "a" / "b" / "transcript.json"

    write_durably(path, "{}")

    assert path.read_text() == "{}"
    # b holds the file, a holds b, and tmp_path holds a.
    assert synced == [tmp_path / "a" / "b", tmp_path / "a", tmp_path]
