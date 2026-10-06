from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path
from types import FunctionType
from typing import Any, Final, final

from loguru import logger

from karotte.judges import Judge
from karotte.schemas import EvaluationRunConfig
from karotte.schemas.scoring import Scoring
from karotte.schemas.transcript import Transcript
from karotte.student_misbehavior import StudentMisbehaviorError, misbehavior_scoring


def _absolute(paths: object, cls: str) -> tuple[Path, ...] | None:
    if paths is None:
        return None
    if not isinstance(paths, tuple):
        raise TypeError(
            f"{cls}.submission_paths must be a tuple of paths, got {type(paths).__name__}."
        )
    for path in paths:
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError(f"{cls}.submission_paths must be absolute, got '{path}'.")
    return paths


class Step(ABC):
    def __init__(self, config: EvaluationRunConfig):
        self.config: Final = config

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "submission_paths" not in cls.__dict__:
            return
        declared = cls.__dict__["submission_paths"]
        if isinstance(declared, FunctionType):
            raise TypeError(
                f"{cls.__module__}.{cls.__qualname__}.submission_paths is missing "
                + "the @property decorator."
            )
        if not isinstance(declared, property):
            _ = _absolute(declared, cls.__qualname__)
            return
        fget = declared.fget
        # Wrapping an abstract property drops it from `__abstractmethods__`.
        if fget is None or getattr(fget, "__isabstractmethod__", False):
            return

        @wraps(fget)
        def validated(self: "Step") -> tuple[Path, ...] | None:
            return _absolute(fget(self), type(self).__qualname__)

        cls.submission_paths = property(  # pyright: ignore[reportAttributeAccessIssue]
            validated, declared.fset, declared.fdel, declared.__doc__
        )

    @property
    @abstractmethod
    def instructions(self) -> str:
        """The user message that starts the step."""

    @property
    @abstractmethod
    def judge(self) -> Judge:
        """Scores the step. Read after `pre_scoring_hook` has run."""

    @property
    def submission_paths(self) -> tuple[Path, ...] | None:
        """Where the student writes this step's answers (if they are file-based)."""
        return None

    def pre_scoring_hook(self) -> None:
        """Called before the step gets scored."""
        return None

    def post_hook(self) -> None:
        """Called after the step is completed."""
        return None

    @final
    def score(self, transcript: Transcript, step_index: int) -> Scoring:
        """Run the pre-scoring hook and judge, scoring misbehavior as 0."""
        try:
            self.pre_scoring_hook()
            return self.judge.evaluate(transcript)
        except StudentMisbehaviorError as e:
            logger.warning(
                "Student misbehavior, scoring step {} as 0: {}", step_index, e
            )
            return misbehavior_scoring(e)
