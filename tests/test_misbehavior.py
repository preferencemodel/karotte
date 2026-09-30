from collections.abc import AsyncGenerator, Iterable
from typing import Any

import pytest

from karotte.evaluation_runner import EvaluationRunner
from karotte.judges.judge import Judge
from karotte.process_utils import UnreapableCohortError
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.http_mcp_server_config import HttpMcpServerConfig
from karotte.schemas.scoring import Scoring
from karotte.schemas.transcript import ErrorEvent, Event, ScoringEvent, Transcript
from karotte.step import Step
from karotte.student_misbehavior import StudentMisbehaviorError
from karotte.task import Task


def _make_config() -> EvaluationRunConfig:
    return EvaluationRunConfig(
        run_id="test_run",
        task_id="fake-task",
        model="",
        model_api_key="",
        mcp_server_config=HttpMcpServerConfig(),
        transcript_file=None,
    )


class PassingJudge(Judge):
    def evaluate(self, transcript: Transcript) -> Scoring:
        return Scoring(score=1.0, metadata={}, continue_task=True)


class MisbehaviorJudge(Judge):
    def evaluate(self, transcript: Transcript) -> Scoring:
        raise StudentMisbehaviorError("judge saw tampering")


class FakeStep(Step):
    judge_cls: type[Judge] = PassingJudge

    @property
    def instructions(self) -> str:
        return "Do the thing."

    @property
    def judge(self) -> Judge:
        return self.judge_cls()


class MisbehavingHookStep(FakeStep):
    def pre_scoring_hook(self) -> None:
        raise StudentMisbehaviorError("student deleted the grader")


class MisbehaviorJudgeStep(FakeStep):
    judge_cls: type[Judge] = MisbehaviorJudge


class FakeTask(Task):
    id: str = "fake-task"

    @property
    def system_prompt(self) -> str | None:
        return None

    @property
    def steps(self) -> Iterable[Step]:
        return [FakeStep(self.config)]

    @property
    def tools(self) -> list[str]:
        return []


class FakeAgent:
    async def run_step(self, _content: str, **_kwargs: Any) -> AsyncGenerator[Event]:
        for event in ():
            yield event


def _make_runner() -> EvaluationRunner:
    config = _make_config()
    runner = EvaluationRunner(config, FakeTask(config))
    runner._agent = FakeAgent()  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    runner.transcript = Transcript(run_id=config.run_id)
    return runner


async def _scoring_from_step(runner: EvaluationRunner, step: Step) -> Scoring:
    events = [event async for event in runner._execute_step(step, 0)]  # pyright: ignore[reportPrivateUsage]
    scoring_events = [e for e in events if isinstance(e, ScoringEvent)]
    assert len(scoring_events) == 1
    return scoring_events[0].scoring


@pytest.mark.asyncio
async def test_misbehavior_in_pre_scoring_hook_scores_zero():
    runner = _make_runner()

    scoring = await _scoring_from_step(runner, MisbehavingHookStep(runner.config))

    assert scoring.score == 0.0
    assert scoring.continue_task is False
    assert "student deleted the grader" in scoring.metadata["misbehavior"]


@pytest.mark.asyncio
async def test_unreapable_cohort_scores_zero():
    """The reap tasks do in ``pre_scoring_hook``: a cohort it cannot verify dead
    scores 0 rather than failing the run as a tool error."""

    class UnreapableStep(FakeStep):
        def pre_scoring_hook(self) -> None:
            raise UnreapableCohortError(
                "Failed to reap all processes owned by UID 1000"
            )

    runner = _make_runner()

    scoring = await _scoring_from_step(runner, UnreapableStep(runner.config))

    assert scoring.score == 0.0
    assert "Failed to reap" in scoring.metadata["misbehavior"]


@pytest.mark.asyncio
async def test_misbehavior_in_judge_scores_zero():
    runner = _make_runner()

    scoring = await _scoring_from_step(runner, MisbehaviorJudgeStep(runner.config))

    assert scoring.score == 0.0
    assert scoring.continue_task is False
    assert "judge saw tampering" in scoring.metadata["misbehavior"]


@pytest.mark.asyncio
async def test_undecodable_name_in_misbehavior_stays_serializable():
    """A submission entry whose name is not valid UTF-8 reaches the message as
    surrogates; the scoring event must still serialize."""
    name = b"bad\xff\xfelink".decode("utf-8", "surrogateescape")

    class UndecodableNameStep(FakeStep):
        def pre_scoring_hook(self) -> None:
            raise StudentMisbehaviorError(f"{name} is a symlink")

    runner = _make_runner()

    scoring = await _scoring_from_step(runner, UndecodableNameStep(runner.config))

    assert scoring.score == 0.0
    assert scoring.metadata["misbehavior"] == r"bad\udcff\udcfelink is a symlink"
    _ = ScoringEvent(scoring=scoring).model_dump_json()
    _ = runner.transcript.model_dump_json()


@pytest.mark.asyncio
async def test_undecodable_text_in_error_event_stays_serializable():
    """The same bytes reaching an uncaught exception must not break the error
    event that reports the failure."""
    name = b"bad\xff\xfelink".decode("utf-8", "surrogateescape")

    runner = _make_runner()

    class FailingClient:
        async def __aenter__(self) -> None:
            raise RuntimeError(f"cannot stat {name}")

        async def __aexit__(self, *_: object) -> None:
            return None

    runner._prepared_mcp_client = FailingClient  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]

    events = [event async for event in runner.run()]

    error_events = [e for e in events if isinstance(e, ErrorEvent)]
    assert len(error_events) == 1
    assert error_events[0].message == r"cannot stat bad\udcff\udcfelink"
    _ = error_events[0].model_dump_json()
    _ = runner.transcript.model_dump_json()


@pytest.mark.asyncio
async def test_regular_exceptions_still_propagate():
    class BrokenStep(FakeStep):
        def pre_scoring_hook(self) -> None:
            raise RuntimeError("env bug")

    runner = _make_runner()

    with pytest.raises(RuntimeError, match="env bug"):
        async for _ in runner._execute_step(BrokenStep(runner.config), 0):  # pyright: ignore[reportPrivateUsage]
            pass
