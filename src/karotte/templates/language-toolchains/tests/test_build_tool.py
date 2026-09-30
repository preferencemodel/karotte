"""Tests for the `build` MCP tool.

Everything runs off the container: builds are not demoted (no KAROTTE_DEMOTE_ID)
and the compile is a shell stand-in, so what is checked is the tool's
contract — the grading build path, the published outputs, and the refusals.
"""

import asyncio
import json
import shlex
from pathlib import Path

import karotte.tool_base
import pytest
from fastmcp.tools.tool import ToolResult
from karotte.tool_base import ToolConfigWriter
from mcp.types import TextContent

from environment import toolchain_grading
from environment.tools.build import STDERR_NAME, STDOUT_NAME, BuildConfig, build


def _parse(result: ToolResult) -> dict:
    assert isinstance(result.content[0], TextContent)
    return json.loads(result.content[0].text)


def _call(tool: build) -> dict:
    return _parse(asyncio.run(tool()))


@pytest.fixture
def workdir(tmp_path, monkeypatch) -> Path:
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.setattr(toolchain_grading, "STUDENT_WORKDIR", workdir)
    monkeypatch.setattr(toolchain_grading, "BUILD_ROOT", tmp_path / "grader")
    monkeypatch.setattr(karotte.tool_base, "CONFIGS_DIR", tmp_path / "configs")
    return workdir


@pytest.fixture
def make_tool(workdir, tmp_path):
    def _make(submissions: list[dict] | None = None, **config) -> build:
        entries = []
        for entry in submissions or [{}]:
            entry = dict(entry)
            entry.setdefault("source", "main.c")
            entry.setdefault("build", [["/bin/sh", "-c", "cp {source} {artifact}"]])
            entries.append(entry)
        config.setdefault("output_dir", str(tmp_path / "out"))
        ToolConfigWriter(tmp_path / "configs").write(
            "build", BuildConfig(submissions=entries, **config)
        )
        return build()

    return _make


class TestASuccessfulBuild:
    def test_publishes_the_artifact_and_the_logs(self, workdir, make_tool, tmp_path):
        (workdir / "main.c").write_bytes(b"int main() {}")

        content = _call(make_tool())

        assert content["success"] is True
        artifact = tmp_path / "out" / "artifact"
        assert artifact.read_bytes() == b"int main() {}"
        assert content["artifact"] == str(artifact)
        assert (tmp_path / "out" / STDOUT_NAME).exists()
        assert (tmp_path / "out" / STDERR_NAME).exists()

    def test_says_how_grading_will_run_it(self, workdir, make_tool, tmp_path):
        (workdir / "main.c").write_bytes(b"int main() {}")

        content = _call(make_tool())

        assert content["run_command"] == shlex.join([str(tmp_path / "out/artifact")])

    def test_replaces_what_the_previous_build_published(
        self, workdir, make_tool, tmp_path
    ):
        (workdir / "main.c").write_bytes(b"int main() {}")
        stale = tmp_path / "out" / "stale"
        stale.parent.mkdir()
        stale.touch()

        _call(make_tool())

        assert not stale.exists()

    def test_keep_built_companions_are_published_too(
        self, workdir, make_tool, tmp_path
    ):
        (workdir / "main.c").write_bytes(b"int main() {}")
        tool = make_tool(
            submissions=[
                {
                    "build": [
                        ["/bin/sh", "-c", "cp {source} {artifact} && touch helper.beam"]
                    ],
                    "keep_built": ["*.beam"],
                }
            ]
        )

        content = _call(tool)

        assert content["success"] is True
        assert (tmp_path / "out" / "helper.beam").exists()

    def test_the_build_dir_does_not_outlive_the_call(
        self, workdir, make_tool, tmp_path
    ):
        (workdir / "main.c").write_bytes(b"int main() {}")

        _call(make_tool())

        assert list((tmp_path / "grader").iterdir()) == []


class TestTwoAcceptedSubmissions:
    """The BEAM cell's shape: the tool must pick and refuse exactly as
    `chosen_submission` does at grading."""

    CANDIDATES = [
        {"source": "main.erl"},
        {
            "source": "main.ex",
            "build": [["/bin/sh", "-c", "cp {source} {artifact}"]],
            "artifact": "artifact.ex",
        },
    ]

    def test_whichever_file_the_student_wrote_is_the_one_built(
        self, workdir, make_tool, tmp_path
    ):
        (workdir / "main.ex").write_bytes(b"defmodule Server do end")
        tool = make_tool(submissions=self.CANDIDATES)

        content = _call(tool)

        assert content["success"] is True
        assert content["artifact"] == str(tmp_path / "out" / "artifact.ex")

    def test_both_files_at_once_are_refused_the_way_grading_refuses(
        self, workdir, make_tool
    ):
        (workdir / "main.erl").write_bytes(b"-module(main).")
        (workdir / "main.ex").write_bytes(b"defmodule Server do end")
        tool = make_tool(submissions=self.CANDIDATES)

        content = _call(tool)

        assert content["success"] is False
        assert "one file" in content["error"]
        assert "main.erl" in content["error"]
        assert "main.ex" in content["error"]

    def test_a_missing_submission_names_every_accepted_file(self, workdir, make_tool):
        tool = make_tool(submissions=self.CANDIDATES)

        content = _call(tool)

        assert content["success"] is False
        assert "main.erl or main.ex" in content["error"]


class TestAFailingBuild:
    def test_reports_the_error_and_publishes_the_logs(
        self, workdir, make_tool, tmp_path
    ):
        (workdir / "main.c").write_bytes(b"int main() {}")
        tool = make_tool(
            submissions=[{"build": [["/bin/sh", "-c", "echo oops >&2; exit 3"]]}]
        )

        content = _call(tool)

        assert content["success"] is False
        assert "exited 3" in content["error"]
        assert "oops" in content["stderr_tail"]
        assert (tmp_path / "out" / STDERR_NAME).read_bytes() == b"oops\n"
        assert "artifact" not in content
        assert not (tmp_path / "out" / "artifact").exists()

    def test_a_missing_submission_is_named(self, workdir, make_tool):
        content = _call(make_tool())

        assert content["success"] is False
        assert "main.c" in content["error"]

    def test_an_output_dir_the_student_owns_is_refused(self, workdir, make_tool):
        (workdir / "main.c").write_bytes(b"int main() {}")
        tool = make_tool(output_dir=str(workdir / "out"))

        content = _call(tool)

        assert content["success"] is False
        assert "output_dir" in content["error"]


class TestConfigShape:
    def test_a_task_can_write_it_from_its_submissions(self):
        submissions = [
            toolchain_grading.Submission(
                source="main.erl", build=(("erlc", "{source}"),)
            ),
            toolchain_grading.Submission(
                source="main.ex",
                build=(("elixirc", "{source}"),),
                keep_built=("*.beam",),
            ),
        ]

        config = BuildConfig.from_submissions(submissions)

        assert config.candidates() == submissions

    def test_allow_unsafe_survives_the_round_trip(self):
        submissions = [
            toolchain_grading.Submission(
                source="main.rs",
                build=(("rustc", "{source}"),),
                allow_unsafe=True,
            )
        ]

        config = BuildConfig.from_submissions(submissions)

        assert config.candidates() == submissions

    def test_a_build_with_nothing_to_run_is_rejected(self):
        with pytest.raises(ValueError):
            BuildConfig(submissions=[{"source": "main.c", "build": []}])

    def test_no_submissions_at_all_is_rejected(self):
        with pytest.raises(ValueError):
            BuildConfig(submissions=[])

    def test_the_tool_passes_discovery_validation(self):
        from karotte.mcp_servers.discover_tools import _validate_tool

        import environment.tools.build as module

        _validate_tool(module, "build")

    def test_the_docstring_names_every_accepted_file_and_the_outputs(
        self, workdir, make_tool, tmp_path
    ):
        tool = make_tool(submissions=[{"source": "main.erl"}, {"source": "main.ex"}])

        doc = tool.__call__.__doc__
        assert doc is not None
        assert "main.erl or main.ex" in doc
        assert str(tmp_path / "out" / "artifact") in doc
