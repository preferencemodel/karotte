"""The `build` MCP tool: the only way the student reaches a compiler.

The toolchain is never on the student's PATH — `install_toolchain` grants it
to the builder uid alone. This tool picks the submission the way grading does
(`chosen_submission` over the task's accepted files), runs the exact grading
build (`toolchain_grading.build_submission`, demoted to the builder), and
publishes the artifact and the compiler's output to a root-owned directory
the student can read but not write.
"""

import asyncio
import fnmatch
import shlex
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import final

from fastmcp.tools.tool import ToolResult
from karotte import ToolBase
from pydantic import BaseModel, Field

from environment import toolchain_grading
from environment.toolchain_grading import Submission

STDOUT_NAME = "build.stdout"
STDERR_NAME = "build.stderr"


class SubmissionConfig(BaseModel):
    """One `Submission`, spelled so a task can write it with `ToolConfigWriter`."""

    source: str
    build: list[list[str]] = Field(min_length=1)
    run: list[str] = ["{artifact}"]
    artifact: str = "artifact"
    build_env: dict[str, str] = {}
    keep_built: list[str] = []
    allow_unsafe: bool = False

    @classmethod
    def from_submission(cls, submission: Submission) -> "SubmissionConfig":
        return cls(
            source=submission.source,
            build=[list(command) for command in submission.build],
            run=list(submission.run),
            artifact=submission.artifact,
            build_env=dict(submission.build_env),
            keep_built=list(submission.keep_built),
            allow_unsafe=submission.allow_unsafe,
        )

    def submission(self) -> Submission:
        return Submission(
            source=self.source,
            build=tuple(tuple(command) for command in self.build),
            run=tuple(self.run),
            artifact=self.artifact,
            build_env=dict(self.build_env),
            keep_built=tuple(self.keep_built),
            allow_unsafe=self.allow_unsafe,
        )


class BuildConfig(BaseModel):
    """The task's accepted submissions — one for most cells, one per language
    for the BEAM cell — plus where the tool publishes what it built."""

    submissions: list[SubmissionConfig] = Field(min_length=1)
    output_dir: str = "/opt/build"

    @classmethod
    def from_submissions(
        cls, submissions: Iterable[Submission], **overrides
    ) -> "BuildConfig":
        return cls(
            submissions=[
                SubmissionConfig.from_submission(submission)
                for submission in submissions
            ],
            **overrides,
        )

    def candidates(self) -> list[Submission]:
        return [entry.submission() for entry in self.submissions]


def _tool_result(
    success: bool,
    output_dir: Path,
    error: str | None = None,
    artifact: Path | None = None,
    run_command: str | None = None,
    stderr_tail: str = "",
) -> ToolResult:
    content: dict[str, str | bool] = {"success": success}
    if error is not None:
        content["error"] = error
    if artifact is not None:
        content["artifact"] = str(artifact)
    if run_command is not None:
        content["run_command"] = run_command
    content["stdout_file"] = str(output_dir / STDOUT_NAME)
    content["stderr_file"] = str(output_dir / STDERR_NAME)
    if stderr_tail:
        content["stderr_tail"] = stderr_tail
    return ToolResult(structured_content=content)


@final
class build(ToolBase[BuildConfig]):
    """Builds the student's submission exactly as grading will."""

    config_schema = BuildConfig

    def __init__(self):
        super().__init__()
        self._lock = asyncio.Lock()
        self._set_docstring()

    async def __call__(self) -> ToolResult:
        """THIS DOCSTRING IS SET DYNAMICALLY BELOW.
        f-strings dont work in docstrings, so we need to set it dynamically."""
        async with self._lock:
            return await asyncio.to_thread(self._build)

    def _build(self) -> ToolResult:
        output_dir = Path(self.config.output_dir)

        # Under the student's control it stops being a place the grader's
        # bytes can be trusted to sit.
        workdir = toolchain_grading.STUDENT_WORKDIR.resolve()
        if workdir == output_dir.resolve() or workdir in output_dir.resolve().parents:
            return _tool_result(
                False, output_dir, error=f"output_dir {output_dir} is inside {workdir}"
            )

        # The grader's pick, refusals included: missing and ambiguous
        # submissions fail here exactly as they will at grading.
        submission, why_not = toolchain_grading.chosen_submission(
            self.config.candidates()
        )
        if submission is None:
            self._publish_logs(output_dir, b"", b"")
            return _tool_result(False, output_dir, error=why_not)

        source = toolchain_grading.student_source(submission)
        build_dir = toolchain_grading.make_build_dir()
        try:
            staged = toolchain_grading.stage_submission(submission, build_dir)
            result = toolchain_grading.build_submission(submission, staged, build_dir)
            self._publish_logs(output_dir, result.stdout, result.stderr)
            tail = toolchain_grading._tail(result.stderr)
            if result.error is not None:
                return _tool_result(
                    False, output_dir, error=result.error, stderr_tail=tail
                )

            artifact = self._publish_outputs(submission, build_dir, output_dir)
            run_command = shlex.join(
                toolchain_grading.resolve_argv(
                    submission.run, source, artifact, output_dir
                )
            )
            return _tool_result(
                True,
                output_dir,
                artifact=artifact,
                run_command=run_command,
                stderr_tail=tail,
            )
        finally:
            shutil.rmtree(build_dir, ignore_errors=True)

    @staticmethod
    def _publish_logs(output_dir: Path, stdout: bytes, stderr: bytes) -> None:
        """Replace the output directory with one holding only this build's
        logs; readable for the student, root's to write."""
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        output_dir.chmod(0o755)
        for name, data in ((STDOUT_NAME, stdout), (STDERR_NAME, stderr)):
            log = output_dir / name
            log.write_bytes(data)
            log.chmod(0o644)

    @staticmethod
    def _publish_outputs(
        submission: Submission, build_dir: Path, output_dir: Path
    ) -> Path:
        """Copy the artifact and its `keep_built` companions out of the build
        directory, modes and all — `build_submission` already sealed them."""
        for entry in sorted(build_dir.iterdir()):
            if entry.is_symlink():
                continue
            wanted = entry.name == submission.artifact or any(
                fnmatch.fnmatch(entry.name, pattern)
                for pattern in submission.keep_built
            )
            if wanted:
                shutil.copy2(entry, output_dir / entry.name)
        return output_dir / submission.artifact

    def _set_docstring(self):
        output_dir = Path(self.config.output_dir)
        sources = " or ".join(entry.source for entry in self.config.submissions)
        artifacts = " or ".join(
            str(output_dir / artifact)
            for artifact in dict.fromkeys(
                entry.artifact for entry in self.config.submissions
            )
        )
        build.__call__.__doc__ = (
            f"Build your submission with the exact commands that will be used later.\n\n"
            f"Compiles {sources} from the working directory (exactly one must "
            f"exist); there is no other way to run the compiler. Takes no "
            f"arguments.\n"
            f"On success the artifact is published at {artifacts}. Either way the "
            f"compiler's output is written to {output_dir / STDOUT_NAME} "
            f"and {output_dir / STDERR_NAME}. "
            f"Each call replaces the previous build's output."
        )
