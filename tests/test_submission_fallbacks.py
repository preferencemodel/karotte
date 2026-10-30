"""`on_invalid_submission` and `on_final_submission`, seen from a run: what is on
disk when the step gets scored."""

import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, final, override

import pytest

from karotte import Step, Task, UnreapableCohortError
from karotte.agents import BuiltinSource
from karotte.evaluation_runner import EvaluationRunner
from karotte.judges.judge import Judge
from karotte.mcp_servers.http_mcp_server import HttpMcpServer
from karotte.restore_submission import restore_best_submission
from karotte.schemas.chat import Message
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.scoring import Scoring
from karotte.schemas.transcript import (
    MessageAddedEvent,
    ScoringEvent,
    Transcript,
)
from karotte.tools.validate import validate
from karotte.validation import ValidationState, load_state
from tests.conftest import TestTask, ValidateEnv

JUDGE_SCORE = 0.25

Episode = Callable[[validate], Awaitable[None]]
"""What the "model" does in a run: it is given the validate tool, and may write
the submission and call the tool."""


class _FixedJudge(Judge):
    @override
    def evaluate(self, transcript: Transcript) -> Scoring:
        return Scoring(JUDGE_SCORE, {}, True)


@dataclass
class Outcome:
    on_disk_at_scoring: str | None
    """The submission's text when the step was scored, None if it had no file."""
    scores: list[float]
    """The score of every `ScoringEvent` the run emitted."""
    score: float | None
    """The run's final score."""


Run = Callable[..., Awaitable[Outcome]]


async def run_episode(
    *,
    config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
    monkeypatch: pytest.MonkeyPatch,
    env: ValidateEnv,
    episode: Episode,
    validate_settings: dict[str, Any],
    extra_paths: tuple[Path, ...],
) -> Outcome:
    """Run a one-step task, with the validate tool, whose "model" is `episode`."""
    seen: list[str | None] = []

    class _Step(Step):
        @property
        @override
        def instructions(self) -> str:
            return "Write the answer."

        @property
        @override
        def judge(self) -> Judge:
            return _FixedJudge()

        @property
        @override
        def submission_paths(self) -> tuple[Path, ...]:
            return (env.submission, *extra_paths)

        @override
        def pre_scoring_hook(self) -> None:
            seen.append(
                env.submission.read_text() if env.submission.is_file() else None
            )

    @final
    class _Task(Task):
        id = "fallback-task"

        @property
        @override
        def system_prompt(self) -> str | None:
            return None

        @property
        @override
        def tools(self) -> list[str]:
            return ["validate"]

        @property
        @override
        def steps(self) -> list[Step]:
            return [_Step(config)]

        @override
        def configure_tools(self) -> None:
            env.write_config(
                submission_paths=self.submission_paths, **validate_settings
            )

    async def collect(self: BuiltinSource, _messages: object, _tools: object):  # pyright: ignore[reportUnusedParameter]
        await episode(validate())
        yield MessageAddedEvent(message=Message(role="assistant", content="Done."))

    monkeypatch.setattr(BuiltinSource, "collect", collect)
    config.task_id = _Task.id
    config.transcript_file = None
    config.mcp_server_config.port = mcp_server.config.port

    runner = EvaluationRunner(config, _Task(config))
    events = [event async for event in runner.run()]
    assert len(seen) == 1
    return Outcome(
        on_disk_at_scoring=seen[0],
        scores=[e.scoring.score for e in events if isinstance(e, ScoringEvent)],
        score=runner.run_state.score,
    )


@pytest.fixture
def run(
    sample_config: EvaluationRunConfig,
    validate_env: ValidateEnv,
    mcp_server: HttpMcpServer,
    monkeypatch: pytest.MonkeyPatch,
) -> Run:
    async def go(
        episode: Episode,
        validate_settings: dict[str, Any] | None = None,
        extra_paths: tuple[Path, ...] = (),
        **config: str,
    ) -> Outcome:
        return await run_episode(
            config=sample_config.model_copy(update=config),
            mcp_server=mcp_server,
            monkeypatch=monkeypatch,
            env=validate_env,
            episode=episode,
            validate_settings=validate_settings or {},
            extra_paths=extra_paths,
        )

    return go


def read_tree(root: Path) -> dict[str, str]:
    """The text of every file under `root`, by its path relative to `root`."""
    return {
        path.relative_to(root).as_posix(): path.read_text()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.asyncio
async def test_restore_best_puts_the_best_copy_back_when_the_submission_is_missing(
    run: Run, validate_env: ValidateEnv
):
    async def episode(tool: validate) -> None:
        validate_env.write("0.7")
        _ = await tool()
        validate_env.submission.unlink()

    outcome = await run(episode, on_invalid_submission="restore_best")

    assert outcome.on_disk_at_scoring == "0.7"


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param("reject not a number", id="scorer-rejects-it"),
        pytest.param("fail model did not load", id="scorer-reports-a-crash"),
    ],
)
@pytest.mark.asyncio
async def test_restore_best_replaces_a_submission_the_scorer_says_is_invalid(
    run: Run, validate_env: ValidateEnv, invalid: str
):
    async def episode(tool: validate) -> None:
        validate_env.write("0.7")
        _ = await tool()
        validate_env.write(invalid)

    outcome = await run(episode, on_invalid_submission="restore_best")

    assert outcome.on_disk_at_scoring == "0.7"


@pytest.mark.asyncio
async def test_restore_best_leaves_a_valid_submission_alone_even_when_it_is_worse(
    run: Run, validate_env: ValidateEnv
):
    async def episode(tool: validate) -> None:
        validate_env.write("0.7")
        _ = await tool()
        validate_env.write("0.3")

    outcome = await run(episode, on_invalid_submission="restore_best")

    assert outcome.on_disk_at_scoring == "0.3"


@pytest.mark.asyncio
async def test_nothing_is_restored_unless_asked_for(
    run: Run, validate_env: ValidateEnv
):
    async def episode(tool: validate) -> None:
        validate_env.write("0.7")
        _ = await tool()
        validate_env.write("reject not a number")

    outcome = await run(episode)

    assert outcome.on_disk_at_scoring == "reject not a number"


@pytest.mark.asyncio
async def test_restore_best_without_a_valid_copy_changes_nothing(
    run: Run, validate_env: ValidateEnv
):
    async def episode(tool: validate) -> None:
        validate_env.write("reject not a number")
        _ = await tool()

    outcome = await run(episode, on_invalid_submission="restore_best")

    assert outcome.on_disk_at_scoring == "reject not a number"


@pytest.mark.asyncio
async def test_keep_best_scores_the_best_copy_even_when_the_submission_is_valid(
    run: Run, validate_env: ValidateEnv
):
    async def episode(tool: validate) -> None:
        validate_env.write("0.7")
        _ = await tool()
        validate_env.write("0.3")

    outcome = await run(episode, on_final_submission="keep_best")

    assert outcome.on_disk_at_scoring == "0.7"


@pytest.mark.asyncio
async def test_keep_best_without_a_valid_copy_scores_the_submission_as_it_is(
    run: Run, validate_env: ValidateEnv
):
    async def episode(_tool: validate) -> None:
        validate_env.write("0.3")

    outcome = await run(episode, on_final_submission="keep_best")

    assert outcome.on_disk_at_scoring == "0.3"


@pytest.mark.asyncio
async def test_restore_puts_back_every_path_exactly_as_it_was_graded(
    run: Run, validate_env: ValidateEnv, tmp_path: Path
):
    extras = tmp_path / "extras"
    (extras / "sub").mkdir(parents=True)

    async def episode(tool: validate) -> None:
        validate_env.write("0.5")
        (extras / "a.txt").write_text("a")
        (extras / "sub" / "b.txt").write_text("b")
        _ = await tool()
        validate_env.write("reject changed")
        (extras / "a.txt").unlink()
        (extras / "sub" / "b.txt").write_text("changed")
        (extras / "junk.txt").write_text("junk")

    outcome = await run(
        episode, extra_paths=(extras,), on_invalid_submission="restore_best"
    )

    assert outcome.on_disk_at_scoring == "0.5"
    assert read_tree(extras) == {"a.txt": "a", "sub/b.txt": "b"}


@pytest.mark.asyncio
async def test_restore_reports_that_it_put_a_copy_back(validate_env: ValidateEnv):
    tool = validate_env.configure()
    validate_env.write("0.7")
    _ = await tool()
    validate_env.write("0.1")

    restored = await restore_best_submission(
        (validate_env.submission,), only_if_invalid=False
    )

    assert restored is True
    assert validate_env.submission.read_text() == "0.7"


@pytest.mark.asyncio
async def test_restore_does_not_report_success_when_every_path_was_skipped(
    validate_env: ValidateEnv,
):
    tool = validate_env.configure()
    validate_env.write("0.7")
    _ = await tool()
    shutil.rmtree(validate_env.submission.parent)

    restored = await restore_best_submission(
        (validate_env.submission,), only_if_invalid=False
    )

    assert restored is False


@pytest.mark.asyncio
async def test_restore_leaves_a_path_the_tool_never_graded_alone(
    validate_env: ValidateEnv, tmp_path: Path
):
    namesake = tmp_path / "elsewhere" / "answer.txt"
    namesake.parent.mkdir()
    namesake.write_text("not graded")
    tool = validate_env.configure()
    validate_env.write("0.7")
    _ = await tool()

    restored = await restore_best_submission((namesake,), only_if_invalid=False)

    assert restored is False
    assert namesake.read_text() == "not graded"


@pytest.mark.asyncio
async def test_validate_never_scores_the_step(run: Run, validate_env: ValidateEnv):
    async def episode(tool: validate) -> None:
        validate_env.write("0.9")
        _ = await tool()
        _ = await tool()

    outcome = await run(episode)

    assert outcome.scores == [JUDGE_SCORE]
    assert outcome.score == JUDGE_SCORE


@pytest.mark.asyncio
async def test_a_symlink_left_at_the_submission_path_is_replaced_not_followed(
    run: Run, validate_env: ValidateEnv, tmp_path: Path
):
    elsewhere = tmp_path / "elsewhere.txt"
    elsewhere.write_text("not the submission")

    async def episode(tool: validate) -> None:
        validate_env.write("0.7")
        _ = await tool()
        validate_env.submission.unlink()
        validate_env.submission.symlink_to(elsewhere)

    outcome = await run(episode, on_invalid_submission="restore_best")

    assert outcome.on_disk_at_scoring == "0.7"
    assert elsewhere.read_text() == "not the submission"


@pytest.mark.asyncio
async def test_the_student_is_stopped_before_the_copy_goes_back(
    run: Run, validate_env: ValidateEnv, monkeypatch: pytest.MonkeyPatch
):
    stops: list[tuple[int, bool]] = []

    def record_stop(uid: int) -> None:
        stops.append((uid, validate_env.submission.exists()))

    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "4242")
    monkeypatch.setattr("karotte.restore_submission.kill_processes", record_stop)

    async def episode(tool: validate) -> None:
        validate_env.write("0.7")
        _ = await tool()
        validate_env.submission.unlink()

    outcome = await run(episode, on_invalid_submission="restore_best")

    # Stopped as user 4242, while the submission was still missing.
    assert stops == [(4242, False)]
    assert outcome.on_disk_at_scoring == "0.7"


@pytest.mark.asyncio
async def test_nothing_is_stopped_or_checked_when_there_is_no_copy_to_restore(
    run: Run, validate_env: ValidateEnv, monkeypatch: pytest.MonkeyPatch
):
    stopped: list[int] = []
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "4242")
    monkeypatch.setattr("karotte.restore_submission.kill_processes", stopped.append)

    async def episode(tool: validate) -> None:
        validate_env.write("reject not a number")
        _ = await tool()

    _ = await run(episode, on_invalid_submission="restore_best")

    assert stopped == []
    assert validate_env.scorer_runs == 1


@pytest.mark.parametrize(
    ("text", "validate_settings"),
    [
        pytest.param("hang", {"timeout_s": 0.5}, id="scorer-times-out"),
        pytest.param("crash", {}, id="scorer-crashes"),
        pytest.param("garbage", {}, id="scorer-writes-no-result"),
    ],
)
@pytest.mark.asyncio
async def test_restore_best_keeps_the_submission_when_the_dev_scorer_cannot_say(
    run: Run,
    validate_env: ValidateEnv,
    text: str,
    validate_settings: dict[str, Any],
):
    async def episode(tool: validate) -> None:
        validate_env.write("0.7")
        _ = await tool()
        validate_env.write(text)

    outcome = await run(
        episode,
        validate_settings=validate_settings,
        on_invalid_submission="restore_best",
    )

    assert outcome.on_disk_at_scoring == text


@pytest.mark.asyncio
async def test_restore_best_keeps_a_submission_that_changes_under_the_copy(
    run: Run,
    validate_env: ValidateEnv,
    after_copying: Callable[[Path, Callable[[], object]], None],
):
    rewrites: list[str] = []

    def rewrite() -> None:
        rewrites.append(f"0.{len(rewrites) + 2}")
        validate_env.write(rewrites[-1])

    async def episode(tool: validate) -> None:
        validate_env.write("0.7")
        _ = await tool()
        validate_env.write("0.1")
        after_copying(validate_env.submission, rewrite)

    outcome = await run(episode, on_invalid_submission="restore_best")

    assert rewrites, "the check never copied the submission"
    assert outcome.on_disk_at_scoring == rewrites[-1]


@pytest.mark.asyncio
async def test_the_student_is_stopped_before_the_submission_is_checked(
    run: Run, validate_env: ValidateEnv, monkeypatch: pytest.MonkeyPatch
):
    runs_when_stopped: list[int] = []

    def record_stop(_uid: int) -> None:
        runs_when_stopped.append(validate_env.scorer_runs)

    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "4242")
    monkeypatch.setattr("karotte.restore_submission.kill_processes", record_stop)

    async def episode(tool: validate) -> None:
        validate_env.write("0.7")
        _ = await tool()

    _ = await run(episode, on_invalid_submission="restore_best")

    assert runs_when_stopped == [1]
    assert validate_env.scorer_runs == 2


@pytest.mark.asyncio
async def test_the_check_leaves_the_validate_state_as_the_last_call_left_it(
    run: Run, validate_env: ValidateEnv
):
    state_before_check: list[ValidationState] = []

    async def episode(tool: validate) -> None:
        validate_env.write("0.7")
        _ = await tool()
        state_before_check.append(load_state())

    _ = await run(episode, on_invalid_submission="restore_best")

    assert validate_env.scorer_runs == 2
    assert load_state() == state_before_check[0]


@pytest.mark.asyncio
async def test_a_new_run_starts_without_the_snapshots_of_an_earlier_one(
    run: Run, karotte_state: Path
):
    stale = karotte_state / "validate_snapshots" / "tmpold" / "answer.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("0.99")
    left_over: list[bool] = []

    async def episode(_tool: validate) -> None:
        left_over.append(stale.exists())

    _ = await run(episode)

    assert left_over == [False]


def test_the_options_need_the_validate_tool(sample_config: EvaluationRunConfig):
    config = sample_config.model_copy(update={"on_invalid_submission": "restore_best"})

    with pytest.raises(ValueError, match="need the validate tool"):
        _ = EvaluationRunner(config, TestTask(config))


def test_the_options_need_a_task_with_one_step(
    sample_config: EvaluationRunConfig, monkeypatch: pytest.MonkeyPatch
):
    # TestTask has two steps.
    monkeypatch.setattr(TestTask, "tools", property(lambda _self: ["validate"]))
    config = sample_config.model_copy(update={"on_final_submission": "keep_best"})

    with pytest.raises(ValueError, match="task with one step"):
        _ = EvaluationRunner(config, TestTask(config))


@pytest.mark.asyncio
async def test_a_student_that_cannot_be_stopped_is_scored_as_it_is(
    run: Run, validate_env: ValidateEnv, monkeypatch: pytest.MonkeyPatch
):
    def cannot_stop(_uid: int) -> None:
        raise UnreapableCohortError("processes survived")

    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "4242")
    monkeypatch.setattr("karotte.restore_submission.kill_processes", cannot_stop)

    async def episode(tool: validate) -> None:
        validate_env.write("0.7")
        _ = await tool()
        validate_env.submission.unlink()

    outcome = await run(episode, on_invalid_submission="restore_best")

    assert outcome.on_disk_at_scoring is None
    assert outcome.score == JUDGE_SCORE
