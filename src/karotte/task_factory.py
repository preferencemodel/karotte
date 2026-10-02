from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from karotte.judges import Judge
from karotte.schemas import EvaluationRunConfig
from karotte.schemas.data_mount import DataMount
from karotte.step import Step
from karotte.task import Task


@dataclass(kw_only=True)
class StepConfig:
    """Configuration for a dynamically created Step.

    `instructions` and `judge` accept either a static value or a callable
    that receives the `EvaluationRunConfig` for config-dependent behaviour
    (e.g., appending hints when `config.use_hints` is set).
    """

    instructions: str | Callable[[EvaluationRunConfig], str]
    judge: Judge | Callable[[EvaluationRunConfig], Judge]
    submission_paths: tuple[Path, ...] | None = None
    pre_scoring_hook: Callable[[EvaluationRunConfig], None] | None = None
    post_hook: Callable[[EvaluationRunConfig], None] | None = None


def create_task(
    *,
    id: str,
    tools: list[str],
    steps: list[StepConfig],
    system_prompt: str | None | Callable[[EvaluationRunConfig], str | None],
    required_hardware: str | None = None,
    scoring_time_limit_seconds: float | None = None,
    data_mounts: list[DataMount] | None = None,
    configure_tools: Callable[[], None] | None = None,
    pre_hook: Callable[[EvaluationRunConfig], dict[str, Any]] | None = None,
) -> type[Task]:
    """Creates a `Task` subclass.

    `system_prompt` accepts a static value or a callable that receives the
    `EvaluationRunConfig`. Pass `None` to send no system prompt.
    """
    _mounts = data_mounts or []
    step_classes = [_make_step_class(i, sc) for i, sc in enumerate(steps)]

    attributes: dict[str, Any] = {
        "id": id,
        "tools": property(lambda self: tools),
        "data_mounts": property(lambda self, _m=_mounts: _m),
        "steps": property(
            lambda self, _cls=step_classes: [c(config=self.config) for c in _cls]
        ),
        "system_prompt": property(
            (lambda self, _f=system_prompt: _f(self.config))
            if callable(system_prompt)
            else (lambda self, _v=system_prompt: _v)
        ),
    }

    if required_hardware is not None:
        attributes["required_hardware"] = property(lambda self: required_hardware)

    if scoring_time_limit_seconds is not None:
        attributes["scoring_time_limit_seconds"] = property(
            lambda self: scoring_time_limit_seconds
        )

    if configure_tools is not None:
        attributes["configure_tools"] = lambda self, _f=configure_tools: _f()  # pyright: ignore[reportUnknownLambdaType]

    if pre_hook is not None:
        attributes["pre_hook"] = lambda self, _f=pre_hook: _f(self.config)  # pyright: ignore[reportUnknownLambdaType]

    class_name = id.replace("-", "_")
    return type(class_name, (Task,), attributes)


def _make_step_class(index: int, config: StepConfig) -> type[Step]:
    instr = config.instructions
    jdg = config.judge

    attrs: dict[str, Any] = {
        "instructions": property(
            (lambda self, _f=instr: _f(self.config))
            if callable(instr)
            else (lambda self, _v=instr: _v)
        ),
        "judge": property(
            (lambda self, _f=jdg: _f(self.config))
            if callable(jdg)
            else (lambda self, _v=jdg: _v)
        ),
    }

    if config.submission_paths is not None:
        attrs["submission_paths"] = config.submission_paths

    if config.pre_scoring_hook is not None:
        hook = config.pre_scoring_hook
        attrs["pre_scoring_hook"] = lambda self, _h=hook: _h(self.config)  # pyright: ignore[reportUnknownLambdaType]

    if config.post_hook is not None:
        hook = config.post_hook
        attrs["post_hook"] = lambda self, _h=hook: _h(self.config)  # pyright: ignore[reportUnknownLambdaType]

    return type(f"step_{index + 1}", (Step,), attrs)
