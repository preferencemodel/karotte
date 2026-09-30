"""Tests for the primitives that touch a path a student controls."""

import errno
import os
import stat
from collections.abc import Iterator
from pathlib import Path

import pytest

from karotte.untrusted_paths import (
    DIR_FLAGS,
    FILE_FLAGS,
    open_at,
    probe,
    walk_to_parent,
)

needs_unprivileged = pytest.mark.skipif(
    os.geteuid() == 0, reason="root reads through a 0o000 directory"
)


@pytest.fixture
def unstattable(tmp_path: Path) -> Iterator[Path]:
    """A path whose target sits behind a directory nobody may traverse."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "secret").write_text("x")
    vault.chmod(0o000)
    yield vault / "secret"
    vault.chmod(0o700)


class TestProbe:
    def test_a_regular_file(self, tmp_path: Path):
        path = tmp_path / "main.c"
        path.write_text("int main(void) { return 0; }")
        st = probe(path)
        assert st is not None
        assert stat.S_ISREG(st.st_mode)

    def test_a_directory(self, tmp_path: Path):
        st = probe(tmp_path)
        assert st is not None
        assert stat.S_ISDIR(st.st_mode)

    def test_a_symlink_is_reported_as_itself(self, tmp_path: Path):
        target = tmp_path / "target"
        target.write_text("x")
        link = tmp_path / "link"
        link.symlink_to(target)
        st = probe(link)
        assert st is not None
        assert stat.S_ISLNK(st.st_mode)

    def test_a_dangling_symlink(self, tmp_path: Path):
        link = tmp_path / "link"
        link.symlink_to(tmp_path / "nowhere")
        st = probe(link)
        assert st is not None
        assert stat.S_ISLNK(st.st_mode)

    def test_a_missing_path(self, tmp_path: Path):
        assert probe(tmp_path / "nowhere") is None

    def test_a_path_under_a_file(self, tmp_path: Path):
        path = tmp_path / "file"
        path.write_text("x")
        assert probe(path / "child") is None

    @needs_unprivileged
    def test_an_unreadable_parent(self, unstattable: Path):
        assert probe(unstattable) is None

    @needs_unprivileged
    def test_a_symlink_to_an_unstattable_target(
        self, tmp_path: Path, unstattable: Path
    ):
        """`Path.is_file` stats through the link and raises here."""
        link = tmp_path / "link"
        link.symlink_to(unstattable)
        with pytest.raises(PermissionError):
            link.is_file()
        st = probe(link)
        assert st is not None
        assert stat.S_ISLNK(st.st_mode)

    def test_a_path_with_a_nul_byte(self, tmp_path: Path):
        assert probe(f"{tmp_path}/ma\0in.c") is None


class TestOpenAt:
    def test_opens_a_file(self, tmp_path: Path):
        (tmp_path / "main.c").write_text("hello")
        dir_fd = os.open(tmp_path, DIR_FLAGS)
        try:
            fd = open_at(dir_fd, "main.c", FILE_FLAGS)
            try:
                assert os.read(fd, 16) == b"hello"
            finally:
                os.close(fd)
        finally:
            os.close(dir_fd)

    def test_refuses_a_symlink(self, tmp_path: Path):
        (tmp_path / "target").write_text("hello")
        (tmp_path / "link").symlink_to(tmp_path / "target")
        dir_fd = os.open(tmp_path, DIR_FLAGS)
        try:
            with pytest.raises(OSError) as raised:
                open_at(dir_fd, "link", FILE_FLAGS)
            assert raised.value.errno in (errno.ELOOP, errno.EMLINK)
        finally:
            os.close(dir_fd)


class TestWalkToParent:
    def test_returns_the_parent_and_the_final_name(self, tmp_path: Path):
        fd, final = walk_to_parent(tmp_path / "main.c")
        try:
            assert final == "main.c"
            assert os.fstat(fd).st_ino == tmp_path.stat().st_ino
        finally:
            os.close(fd)

    def test_refuses_a_symlinked_component(self, tmp_path: Path):
        real = tmp_path / "real"
        real.mkdir()
        (tmp_path / "link").symlink_to(real)
        with pytest.raises(OSError) as raised:
            walk_to_parent(tmp_path / "link" / "main.c")
        assert raised.value.filename == str(tmp_path / "link")

    def test_refuses_the_filesystem_root(self):
        with pytest.raises(ValueError):
            walk_to_parent(Path("/"))
