"""`Step.submission_paths` is the declaration; `Task.submission_paths` is what
a backend reads.

The step that takes a hand-in names it, and the task collects those paths so
`karotte tasks list --json` can report them per task.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, final, override

import pytest

import karotte.load_tasks
from karotte import Step, StepConfig, Task, create_task
from karotte.cli.tasks import list_tasks
from karotte.judges.always_pass_judge import AlwaysPassJudge
from karotte.schemas.evaluation_run_config import EvaluationRunConfig

ANSWER = Path("/workdir/data/answer.txt")
WEIGHTS = Path("/workdir/data/weights.safetensors")


class TalkingStep(Step):
    """A step whose answer is the final message — nothing is handed in."""

    @property
    @override
    def instructions(self) -> str:
        return "Say something."

    @property
    @override
    def judge(self) -> AlwaysPassJudge:
        return AlwaysPassJudge()


class FileStep(TalkingStep):
    @property
    @override
    def submission_paths(self) -> tuple[Path, ...]:
        return (ANSWER,)


class TwoFileStep(TalkingStep):
    @property
    @override
    def submission_paths(self) -> tuple[Path, ...]:
        return (ANSWER, WEIGHTS)


def _task(*step_classes: type[Step]) -> Task:
    @final
    class SubmissionTask(Task):
        id = "submission-task"

        @property
        @override
        def system_prompt(self) -> str | None:
            return None

        @property
        @override
        def tools(self) -> list[str]:
            return ["bash"]

        @property
        @override
        def steps(self) -> list[Step]:
            return [cls(config=self.config) for cls in step_classes]

    return SubmissionTask(
        EvaluationRunConfig(run_id="", task_id="", model="", model_api_key="")
    )


def test_a_computed_relative_submission_path_is_rejected_when_read():
    class SloppyStep(TalkingStep):
        @property
        @override
        def submission_paths(self) -> tuple[Path, ...]:
            return (Path("answer.txt"),)

    with pytest.raises(ValueError, match="must be absolute"):
        _ = SloppyStep(config=None).submission_paths  # pyright: ignore[reportArgumentType]


def test_a_fixed_relative_submission_path_is_rejected_at_class_definition():
    with pytest.raises(ValueError, match="must be absolute"):
        type("SloppyStep", (TalkingStep,), {"submission_paths": (Path("answer.txt"),)})


def test_a_fixed_submission_path_that_is_not_a_tuple_is_rejected():
    with pytest.raises(TypeError, match="must be a tuple"):
        type("SloppyStep", (TalkingStep,), {"submission_paths": [ANSWER]})


def test_a_computed_submission_path_that_is_not_a_tuple_is_rejected_when_read():
    class SloppyStep(TalkingStep):
        @property
        @override
        def submission_paths(self) -> tuple[Path, ...]:
            return [ANSWER]  # pyright: ignore[reportReturnType]

    with pytest.raises(TypeError, match="must be a tuple"):
        _ = SloppyStep(config=None).submission_paths  # pyright: ignore[reportArgumentType]


def test_validation_keeps_the_declared_property_intact():
    class Documented(TalkingStep):
        @property
        @override
        def submission_paths(self) -> tuple[Path, ...]:
            """Where the answer goes."""
            return (ANSWER,)

    prop = Documented.__dict__["submission_paths"]
    assert prop.__doc__ == "Where the answer goes."
    assert prop.fget.__name__ == "submission_paths"
    assert getattr(prop.fget, "__override__", False)


def test_submission_paths_without_property_raises():
    """A forgotten @property leaves a bound method where a tuple is read."""

    def submission_paths(_self: Any) -> tuple[Path, ...]:
        return (ANSWER,)

    with pytest.raises(TypeError, match="submission_paths"):
        type("BadStep", (TalkingStep,), {"submission_paths": submission_paths})


def test_an_abstract_submission_paths_stays_abstract():
    """Absolute-path validation wraps the declared property; the wrapper must
    not launder away the abstractness, or subclasses that never implement it
    instantiate and silently hand in nothing."""

    class NeedsAHandIn(TalkingStep, ABC):
        @property
        @abstractmethod
        @override
        def submission_paths(self) -> tuple[Path, ...]: ...

    class Forgetful(NeedsAHandIn, ABC):
        pass

    assert "submission_paths" in NeedsAHandIn.__abstractmethods__
    assert "submission_paths" in Forgetful.__abstractmethods__
    with pytest.raises(TypeError, match="submission_paths"):
        _ = Forgetful(config=None)  # pyright: ignore[reportArgumentType, reportAbstractUsage]


def test_an_implemented_abstract_submission_path_is_still_validated():
    class NeedsAHandIn(TalkingStep, ABC):
        @property
        @abstractmethod
        @override
        def submission_paths(self) -> tuple[Path, ...]: ...

    class Sloppy(NeedsAHandIn):
        @property
        @override
        def submission_paths(self) -> tuple[Path, ...]:
            return (Path("answer.txt"),)

    with pytest.raises(ValueError, match="must be absolute"):
        _ = Sloppy(config=None).submission_paths  # pyright: ignore[reportArgumentType]


def test_step_declares_no_submission_by_default():
    assert TalkingStep(config=None).submission_paths is None  # pyright: ignore[reportArgumentType]


def test_task_collects_submission_paths_from_its_steps_without_repeats():
    task = _task(TalkingStep, FileStep, TwoFileStep)
    assert task.submission_paths == (ANSWER, WEIGHTS)


def test_task_without_hand_ins_has_no_submission_paths():
    assert _task(TalkingStep).submission_paths == ()


def test_created_task_carries_the_step_config_submission_paths():
    cls = create_task(
        id="factory-task",
        tools=["bash"],
        system_prompt=None,
        steps=[
            StepConfig(instructions="Say something.", judge=AlwaysPassJudge()),
            StepConfig(
                instructions="Write a file.",
                judge=AlwaysPassJudge(),
                submission_paths=(ANSWER,),
            ),
        ],
    )
    task = cls(EvaluationRunConfig(run_id="", task_id="", model="", model_api_key=""))
    assert task.submission_paths == (ANSWER,)


def test_create_task_rejects_a_relative_submission_path():
    with pytest.raises(ValueError, match="must be absolute"):
        create_task(
            id="factory-task",
            tools=["bash"],
            system_prompt=None,
            steps=[
                StepConfig(
                    instructions="Write a file.",
                    judge=AlwaysPassJudge(),
                    submission_paths=(Path("answer.txt"),),
                )
            ],
        )


def test_list_tasks_json_reports_submission_paths(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    task = _task(TalkingStep, FileStep)
    monkeypatch.setattr(
        karotte.load_tasks, "_get_task_loader", lambda: lambda: [type(task)]
    )

    list_tasks(json_output=True)

    tasks = json.loads(capsys.readouterr().out)
    assert tasks[0]["submission_paths"] == [str(ANSWER)]
