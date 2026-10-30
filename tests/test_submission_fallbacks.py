"""`on_invalid_submission` and `on_final_submission`, seen from a run: what is on
disk when the step gets scored."""

import os
import shutil
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, final, override

import pytest

from karotte import Step, Task, ToolConfigWriter, UnreapableCohortError
from karotte.agents import BuiltinSource
from karotte.evaluation_runner import EvaluationRunner
from karotte.judges.judge import Judge
from karotte.mcp_servers.http_mcp_server import HttpMcpServer
from karotte.restore_submission import restore_best_submission
from karotte.save_submission import save_submission
from karotte.schemas.chat import Message
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.scoring import Scoring
from karotte.schemas.transcript import (
    Event,
    MessageAddedEvent,
    ScoringEvent,
    Transcript,
)
from karotte.tools.validate import ValidateConfig, validate
from karotte.validation import load_state

JUDGE_SCORE = 0.25

Episode = Callable[[validate, Path], Awaitable[None]]


class _FixedJudge(Judge):
    @override
    def evaluate(self, transcript: Transcript) -> Scoring:
        return Scoring(JUDGE_SCORE, {}, True)


@dataclass
class Outcome:
    on_disk_at_scoring: str | None
    events: list[Event]
    score: float | None


async def run_episode(
    *,
    config: EvaluationRunConfig,
    mcp_server: HttpMcpServer,
    monkeypatch: pytest.MonkeyPatch,
    dev_scorer: Path,
    tmp_path: Path,
    episode: Episode,
    validate_settings: dict[str, Any],
    extra_paths: tuple[Path, ...],
) -> Outcome:
    """Run a one-step task whose "model" is `episode`: it gets the validate tool
    and the submission path, and may write files and call the tool."""
    submission = tmp_path / "work" / "answer.txt"
    submission.parent.mkdir()
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
            return (submission, *extra_paths)

        @override
        def pre_scoring_hook(self) -> None:
            seen.append(submission.read_text() if submission.is_file() else None)

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
            settings: dict[str, Any] = {
                "dev_command": [
                    sys.executable,
                    str(dev_scorer),
                    str(tmp_path / "counter"),
                    str(submission),
                ],
                "metric": "accuracy",
                "submission_paths": self.submission_paths,
                "min_interval_s": 0,
            } | validate_settings
            _ = ToolConfigWriter().write("validate", ValidateConfig(**settings))

    async def collect(self: BuiltinSource, _messages: object, _tools: object):  # pyright: ignore[reportUnusedParameter]
        await episode(validate(), submission)
        yield MessageAddedEvent(message=Message(role="assistant", content="Done."))

    monkeypatch.setattr(BuiltinSource, "collect", collect)
    config.task_id = _Task.id
    config.transcript_file = None
    config.mcp_server_config.port = mcp_server.config.port

    runner = EvaluationRunner(config, _Task(config))
    events = [event async for event in runner.run()]
    assert len(seen) == 1
    return Outcome(seen[0], events, runner.run_state.score)


@pytest.fixture
def run(
    sample_config: EvaluationRunConfig,
    karotte_state: Path,  # pyright: ignore[reportUnusedParameter]
    mcp_server: HttpMcpServer,
    monkeypatch: pytest.MonkeyPatch,
    dev_scorer: Path,
    tmp_path: Path,
) -> Callable[..., Awaitable[Outcome]]:
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
            dev_scorer=dev_scorer,
            tmp_path=tmp_path,
            episode=episode,
            validate_settings=validate_settings or {},
            extra_paths=extra_paths,
        )

    return go


def scorer_runs(tmp_path: Path) -> int:
    counter = tmp_path / "counter"
    return len(counter.read_text().splitlines()) if counter.exists() else 0


@pytest.mark.asyncio
async def test_restore_best_puts_the_best_copy_back_when_the_submission_is_missing(
    run: Callable[..., Awaitable[Outcome]],
):
    async def episode(tool: validate, submission: Path) -> None:
        submission.write_text("0.7")
        _ = await tool()
        os.unlink(submission)

    outcome = await run(episode, on_invalid_submission="restore_best")

    assert outcome.on_disk_at_scoring == "0.7"


@pytest.mark.asyncio
async def test_restore_best_puts_the_best_copy_back_when_the_submission_is_rejected(
    run: Callable[..., Awaitable[Outcome]],
):
    async def episode(tool: validate, submission: Path) -> None:
        submission.write_text("0.7")
        _ = await tool()
        submission.write_text("reject not a number")

    outcome = await run(episode, on_invalid_submission="restore_best")

    assert outcome.on_disk_at_scoring == "0.7"


@pytest.mark.asyncio
async def test_restore_best_leaves_a_valid_submission_alone_even_when_it_is_worse(
    run: Callable[..., Awaitable[Outcome]],
):
    async def episode(tool: validate, submission: Path) -> None:
        submission.write_text("0.7")
        _ = await tool()
        submission.write_text("0.3")

    outcome = await run(episode, on_invalid_submission="restore_best")

    assert outcome.on_disk_at_scoring == "0.3"


@pytest.mark.asyncio
async def test_nothing_is_restored_unless_asked_for(
    run: Callable[..., Awaitable[Outcome]],
):
    async def episode(tool: validate, submission: Path) -> None:
        submission.write_text("0.7")
        _ = await tool()
        submission.write_text("reject not a number")

    outcome = await run(episode)

    assert outcome.on_disk_at_scoring == "reject not a number"


@pytest.mark.asyncio
async def test_restore_best_without_a_valid_copy_changes_nothing(
    run: Callable[..., Awaitable[Outcome]],
):
    async def episode(tool: validate, submission: Path) -> None:
        submission.write_text("reject not a number")
        _ = await tool()

    outcome = await run(episode, on_invalid_submission="restore_best")

    assert outcome.on_disk_at_scoring == "reject not a number"


@pytest.mark.asyncio
async def test_keep_best_scores_the_best_copy_even_when_the_submission_is_valid(
    run: Callable[..., Awaitable[Outcome]],
):
    async def episode(tool: validate, submission: Path) -> None:
        submission.write_text("0.7")
        _ = await tool()
        submission.write_text("0.3")

    outcome = await run(episode, on_final_submission="keep_best")

    assert outcome.on_disk_at_scoring == "0.7"


@pytest.mark.asyncio
async def test_keep_best_without_a_valid_copy_scores_the_submission_as_it_is(
    run: Callable[..., Awaitable[Outcome]],
):
    async def episode(_tool: validate, submission: Path) -> None:
        submission.write_text("0.3")

    outcome = await run(episode, on_final_submission="keep_best")

    assert outcome.on_disk_at_scoring == "0.3"


@pytest.mark.asyncio
async def test_restore_puts_back_every_path_exactly_as_it_was_graded(
    run: Callable[..., Awaitable[Outcome]], tmp_path: Path
):
    extras = tmp_path / "extras"
    (extras / "sub").mkdir(parents=True)

    async def episode(tool: validate, submission: Path) -> None:
        submission.write_text("0.5")
        (extras / "a.txt").write_text("a")
        (extras / "sub" / "b.txt").write_text("b")
        _ = await tool()
        submission.write_text("reject changed")
        (extras / "a.txt").unlink()
        (extras / "sub" / "b.txt").write_text("changed")
        (extras / "junk.txt").write_text("junk")

    outcome = await run(
        episode, extra_paths=(extras,), on_invalid_submission="restore_best"
    )

    assert outcome.on_disk_at_scoring == "0.5"
    assert sorted(p.relative_to(extras).as_posix() for p in extras.rglob("*")) == [
        "a.txt",
        "sub",
        "sub/b.txt",
    ]
    assert (extras / "a.txt").read_text() == "a"
    assert (extras / "sub" / "b.txt").read_text() == "b"


@pytest.mark.asyncio
async def test_restore_reports_that_it_put_a_copy_back(
    run: Callable[..., Awaitable[Outcome]],
):
    restored: list[bool] = []

    async def episode(tool: validate, submission: Path) -> None:
        submission.write_text("0.7")
        _ = await tool()
        submission.write_text("0.1")
        restored.append(
            await restore_best_submission((submission,), only_if_invalid=False)
        )

    _ = await run(episode)

    assert restored == [True]


@pytest.mark.asyncio
async def test_restore_does_not_report_success_when_every_path_was_skipped(
    run: Callable[..., Awaitable[Outcome]],
):
    restored: list[bool] = []

    async def episode(tool: validate, submission: Path) -> None:
        submission.write_text("0.7")
        _ = await tool()
        shutil.rmtree(submission.parent)
        restored.append(
            await restore_best_submission((submission,), only_if_invalid=False)
        )

    _ = await run(episode)

    assert restored == [False]


@pytest.mark.asyncio
async def test_validate_never_scores_the_step(
    run: Callable[..., Awaitable[Outcome]],
):
    async def episode(tool: validate, submission: Path) -> None:
        submission.write_text("0.9")
        _ = await tool()
        _ = await tool()

    outcome = await run(episode)

    scorings = [e for e in outcome.events if isinstance(e, ScoringEvent)]
    assert [s.scoring.score for s in scorings] == [JUDGE_SCORE]
    assert outcome.score == JUDGE_SCORE


@pytest.mark.asyncio
async def test_a_symlink_left_at_the_submission_path_is_replaced_not_followed(
    run: Callable[..., Awaitable[Outcome]], tmp_path: Path
):
    elsewhere = tmp_path / "elsewhere.txt"
    elsewhere.write_text("not the submission")

    async def episode(tool: validate, submission: Path) -> None:
        submission.write_text("0.7")
        _ = await tool()
        submission.unlink()
        submission.symlink_to(elsewhere)

    outcome = await run(episode, on_invalid_submission="restore_best")

    assert outcome.on_disk_at_scoring == "0.7"
    assert elsewhere.read_text() == "not the submission"


@pytest.mark.asyncio
async def test_the_student_is_stopped_before_the_copy_goes_back(
    run: Callable[..., Awaitable[Outcome]],
    monkeypatch: pytest.MonkeyPatch,
):
    on_disk_when_stopped: list[tuple[int, bool]] = []
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "4242")

    async def episode(tool: validate, submission: Path) -> None:
        submission.write_text("0.7")
        _ = await tool()
        submission.unlink()
        monkeypatch.setattr(
            "karotte.restore_submission.kill_processes",
            lambda uid: on_disk_when_stopped.append((uid, submission.exists())),  # pyright: ignore[reportUnknownLambdaType]
        )

    outcome = await run(episode, on_invalid_submission="restore_best")

    assert on_disk_when_stopped == [(4242, False)]
    assert outcome.on_disk_at_scoring == "0.7"


@pytest.mark.asyncio
async def test_the_student_is_left_alone_when_there_is_no_copy_to_restore(
    run: Callable[..., Awaitable[Outcome]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    stopped: list[int] = []
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "4242")

    async def episode(tool: validate, submission: Path) -> None:
        submission.write_text("reject not a number")
        _ = await tool()
        monkeypatch.setattr("karotte.restore_submission.kill_processes", stopped.append)

    _ = await run(episode, on_invalid_submission="restore_best")

    assert stopped == []
    assert scorer_runs(tmp_path) == 1


@pytest.mark.asyncio
async def test_restore_best_keeps_a_valid_submission_when_the_check_times_out(
    run: Callable[..., Awaitable[Outcome]],
):
    async def episode(tool: validate, submission: Path) -> None:
        submission.write_text("0.7")
        _ = await tool()
        submission.write_text("hang")

    outcome = await run(
        episode,
        validate_settings={"timeout_s": 0.5},
        on_invalid_submission="restore_best",
    )

    assert outcome.on_disk_at_scoring == "hang"


@pytest.mark.asyncio
async def test_restore_best_keeps_the_submission_when_the_dev_scorer_crashes(
    run: Callable[..., Awaitable[Outcome]],
):
    async def episode(tool: validate, submission: Path) -> None:
        submission.write_text("0.7")
        _ = await tool()
        submission.write_text("crash")

    outcome = await run(episode, on_invalid_submission="restore_best")

    assert outcome.on_disk_at_scoring == "crash"


@pytest.mark.asyncio
async def test_restore_best_keeps_the_submission_when_the_scorer_writes_no_result(
    run: Callable[..., Awaitable[Outcome]],
):
    async def episode(tool: validate, submission: Path) -> None:
        submission.write_text("0.7")
        _ = await tool()
        submission.write_text("garbage")

    outcome = await run(episode, on_invalid_submission="restore_best")

    assert outcome.on_disk_at_scoring == "garbage"


@pytest.mark.asyncio
async def test_restore_best_keeps_a_submission_that_changes_under_the_copy(
    run: Callable[..., Awaitable[Outcome]],
    monkeypatch: pytest.MonkeyPatch,
):
    async def episode(tool: validate, submission: Path) -> None:
        submission.write_text("0.7")
        _ = await tool()
        submission.write_text("0.1")
        copies: list[Path | str] = []

        def copy_then_rewrite(
            source: Path | str, dest: Path | str | None = None, **limits: Any
        ):
            saved = save_submission(source, dest, **limits)
            copies.append(source)
            submission.write_text(f"0.{len(copies) + 1}")
            return saved

        monkeypatch.setattr("karotte.validation.save_submission", copy_then_rewrite)

    outcome = await run(episode, on_invalid_submission="restore_best")

    assert outcome.on_disk_at_scoring == "0.3"


@pytest.mark.asyncio
async def test_restore_best_replaces_a_submission_the_scorer_says_crashed(
    run: Callable[..., Awaitable[Outcome]],
):
    async def episode(tool: validate, submission: Path) -> None:
        submission.write_text("0.7")
        _ = await tool()
        submission.write_text("fail model did not load")

    outcome = await run(episode, on_invalid_submission="restore_best")

    assert outcome.on_disk_at_scoring == "0.7"


@pytest.mark.asyncio
async def test_the_student_is_stopped_before_the_submission_is_checked(
    run: Callable[..., Awaitable[Outcome]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    runs_when_stopped: list[int] = []
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "4242")

    async def episode(tool: validate, submission: Path) -> None:
        submission.write_text("0.7")
        _ = await tool()
        monkeypatch.setattr(
            "karotte.restore_submission.kill_processes",
            lambda _uid: runs_when_stopped.append(scorer_runs(tmp_path)),  # pyright: ignore[reportUnknownLambdaType]
        )

    _ = await run(episode, on_invalid_submission="restore_best")

    assert runs_when_stopped == [1]
    assert scorer_runs(tmp_path) == 2


@pytest.mark.asyncio
async def test_the_check_does_not_use_up_a_call_or_move_the_interval(
    run: Callable[..., Awaitable[Outcome]],
    tmp_path: Path,
):
    last_call_at: list[float | None] = []

    async def episode(tool: validate, submission: Path) -> None:
        submission.write_text("0.7")
        _ = await tool()
        last_call_at.append(load_state().last_call_at)

    _ = await run(episode, on_invalid_submission="restore_best")

    state_after_check = load_state()
    assert scorer_runs(tmp_path) == 2
    assert state_after_check.calls == 1
    assert state_after_check.last_call_at == last_call_at[0]


@pytest.mark.asyncio
async def test_a_new_run_starts_without_the_snapshots_of_an_earlier_one(
    run: Callable[..., Awaitable[Outcome]], karotte_state: Path
):
    stale = karotte_state / "validate_snapshots" / "tmpold" / "answer.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("0.99")
    left_over: list[bool] = []

    async def episode(_tool: validate, _submission: Path) -> None:
        left_over.append(stale.exists())

    _ = await run(episode)

    assert left_over == [False]


def test_the_options_need_the_validate_tool(sample_config: EvaluationRunConfig):
    from tests.conftest import TestTask

    config = sample_config.model_copy(update={"on_invalid_submission": "restore_best"})

    with pytest.raises(ValueError, match="need the validate tool"):
        _ = EvaluationRunner(config, TestTask(config))


def test_the_options_need_a_task_with_one_step(
    sample_config: EvaluationRunConfig, monkeypatch: pytest.MonkeyPatch
):
    from tests.conftest import TestTask

    # TestTask has two steps.
    monkeypatch.setattr(TestTask, "tools", property(lambda _self: ["validate"]))
    config = sample_config.model_copy(update={"on_final_submission": "keep_best"})

    with pytest.raises(ValueError, match="task with one step"):
        _ = EvaluationRunner(config, TestTask(config))


@pytest.mark.asyncio
async def test_restore_leaves_a_path_the_tool_never_graded_alone(
    run: Callable[..., Awaitable[Outcome]], tmp_path: Path
):
    namesake = tmp_path / "elsewhere" / "answer.txt"
    namesake.parent.mkdir()
    namesake.write_text("not graded")
    restored: list[bool] = []

    async def episode(tool: validate, submission: Path) -> None:
        submission.write_text("0.7")
        _ = await tool()
        restored.append(
            await restore_best_submission((namesake,), only_if_invalid=False)
        )

    _ = await run(episode)

    assert restored == [False]
    assert namesake.read_text() == "not graded"


@pytest.mark.asyncio
async def test_a_student_that_cannot_be_stopped_is_scored_as_it_is(
    run: Callable[..., Awaitable[Outcome]],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("KAROTTE_DEMOTE_ID", "4242")

    async def episode(tool: validate, submission: Path) -> None:
        submission.write_text("0.7")
        _ = await tool()
        submission.unlink()

        def cannot_stop(_uid: int) -> None:
            raise UnreapableCohortError("processes survived")

        monkeypatch.setattr("karotte.restore_submission.kill_processes", cannot_stop)

    outcome = await run(episode, on_invalid_submission="restore_best")

    assert outcome.on_disk_at_scoring is None
    assert outcome.score == JUDGE_SCORE
