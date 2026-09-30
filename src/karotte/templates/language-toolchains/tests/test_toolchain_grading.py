"""Tests for the grading half of the toolchain machinery: filling a task's
pinned argv in, and deciding what counts as a submitted file."""

import os
from pathlib import Path

import pytest

from environment import toolchain_grading
from environment.toolchain_grading import is_regular_file, resolve_argv

needs_unprivileged = pytest.mark.skipif(
    os.geteuid() == 0, reason="root reads through a 0o000 directory"
)

SOURCE = Path("/build/main.erl")
ARTIFACT = Path("/build/artifact")
BUILD_DIR = Path("/build")


def _resolve(*argv: str) -> list[str]:
    return resolve_argv(argv, SOURCE, ARTIFACT, BUILD_DIR)


class TestResolveArgv:
    def test_fills_in_every_placeholder(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(toolchain_grading, "student_python", lambda: Path("/py"))
        assert _resolve("{python}", "{source}", "{artifact}", "{build_dir}") == [
            "/py",
            "/build/main.erl",
            "/build/artifact",
            "/build",
        ]

    def test_leaves_other_braces_alone(self):
        """A task can pin argv carrying Erlang terms, JSON or an awk program;
        `str.format` blew up on all three."""
        logger = "[{handler,default,logger_std_h,#{config=>#{type=>standard_error}}}]"
        assert _resolve("escript", "-eval", logger, "{source}") == [
            "escript",
            "-eval",
            logger,
            "/build/main.erl",
        ]

    def test_doubled_braces_are_not_an_escape(self):
        # Not written literally: this file is a jinja template.
        doubled = "{" * 2 + " print $1 " + "}" * 2
        assert _resolve("awk", doubled) == ["awk", doubled]

    def test_an_unknown_placeholder_is_left_alone(self):
        assert _resolve("{nope}") == ["{nope}"]

    def test_python_is_only_looked_up_when_asked_for(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """`student_python` refuses an image with more than one managed CPython,
        which a compiled cell must not be made to care about."""

        def refuse() -> Path:
            raise RuntimeError("two managed CPythons")

        monkeypatch.setattr(toolchain_grading, "student_python", refuse)
        assert _resolve("{artifact}", "#{a=>1}") == ["/build/artifact", "#{a=>1}"]
        with pytest.raises(RuntimeError):
            _resolve("{python}")


@pytest.fixture
def unstattable(tmp_path: Path):
    """A file behind a directory nobody may traverse."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "secret").write_text("x")
    vault.chmod(0o000)
    yield vault / "secret"
    vault.chmod(0o700)


class TestIsRegularFile:
    def test_a_plain_file(self, tmp_path: Path):
        path = tmp_path / "server.java"
        path.write_text("class Server {}")
        assert is_regular_file(path) is True

    def test_a_directory(self, tmp_path: Path):
        assert is_regular_file(tmp_path) is False

    def test_a_missing_file(self, tmp_path: Path):
        assert is_regular_file(tmp_path / "server.java") is False

    def test_a_symlink_to_a_real_file(self, tmp_path: Path):
        (tmp_path / "target").write_text("class Server {}")
        link = tmp_path / "server.java"
        link.symlink_to(tmp_path / "target")
        assert is_regular_file(link) is False

    def test_a_dangling_symlink(self, tmp_path: Path):
        link = tmp_path / "server.java"
        link.symlink_to(tmp_path / "nowhere")
        assert is_regular_file(link) is False

    @needs_unprivileged
    def test_a_symlink_the_grader_cannot_stat_through(
        self, tmp_path: Path, unstattable: Path
    ):
        """What a student plants to make the grader die instead of scoring 0."""
        link = tmp_path / "server.java"
        link.symlink_to(unstattable)
        assert is_regular_file(link) is False
