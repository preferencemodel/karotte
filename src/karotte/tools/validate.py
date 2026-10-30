import asyncio
import math
import time
from pathlib import Path
from typing import Any, final

from fastmcp.tools.tool import ToolResult
from karotte import ToolBase
from karotte.validation import (
    Grade,
    SnapshotLimits,
    ValidationState,
    grade_submission,
    load_state,
    record_grade,
    save_state,
)
from pydantic import BaseModel, model_validator


class ValidateConfig(BaseModel):
    """Configures the `validate` tool for a task."""

    dev_command: list[str]
    """The dev scorer's argv. Karotte appends the snapshot directory and the
    path of the JSON file to write."""
    metric: str
    """Name of the dev metric, shown to the student."""
    higher_is_better: bool = True
    submission_paths: tuple[Path, ...]
    """What gets graded: the absolute paths the step's student hands in. Use
    the task's `submission_paths`."""
    max_calls: int = 5
    min_interval_s: float = 60.0
    """Least time between two graded calls."""
    timeout_s: float = 60.0
    """Longest a dev scorer may run on one submission."""
    snapshot_limits: SnapshotLimits = SnapshotLimits()
    """Overrides for the limits on copying a submission. Use the ones the
    step's final judge copies with."""

    @model_validator(mode="after")
    def _check(self) -> "ValidateConfig":
        if not self.dev_command:
            raise ValueError("dev_command must not be empty")
        if not self.submission_paths:
            raise ValueError("submission_paths must not be empty")
        for path in self.submission_paths:
            if not path.is_absolute():
                raise ValueError(f"submission_paths must be absolute, got '{path}'")
        names = [path.name for path in self.submission_paths]
        if len(set(names)) != len(names):
            raise ValueError(
                "submission_paths must have distinct file names: the dev scorer "
                + "finds each submission under its own name"
            )
        if self.max_calls < 0 or self.min_interval_s < 0 or self.timeout_s <= 0:
            raise ValueError(
                "max_calls and min_interval_s must not be negative and timeout_s must be positive"
            )
        return self

    async def grade(self, paths: tuple[Path, ...]) -> Grade:
        """Run the dev scorer on a copy of `paths`."""
        return await grade_submission(
            paths,
            dev_command=self.dev_command,
            timeout_s=self.timeout_s,
            snapshot_limits=self.snapshot_limits,
        )


def _three_significant_figures(value: float | None) -> float | None:
    return None if value is None else float(f"{value:.3g}")


def _delta(current: float | None, previous: float | None) -> float | None:
    """The change between two metrics as the student sees them. Taking it from
    the rounded values means successive deltas add nothing finer than the
    3 significant figures `dev_metric` already shows."""
    shown_current = _three_significant_figures(current)
    shown_previous = _three_significant_figures(previous)
    if shown_current is None or shown_previous is None:
        return None
    return shown_current - shown_previous


@final
class validate(ToolBase[ValidateConfig]):
    config_schema = ValidateConfig

    def __init__(self) -> None:
        super().__init__()
        self._lock = asyncio.Lock()

    async def __call__(self) -> ToolResult:
        """Score your current submission on a dev split that is separate from the final test.

        Takes no arguments: it grades the files you were told to hand in, as they are now. The result is a JSON object:
        `tier` is `valid` (the dev score is in `dev_metric`), `no_valid_output` (the submission is missing or was rejected, see `reason`) or `crash` (it failed when it was run);
        `metric` names the score and says which direction is better; `delta_vs_prev_valid` is the change since your last valid call; `best_dev_metric` is your best so far;
        `calls_left` is how many calls remain.
        The number of calls is limited and calls must be spaced apart. A call that is refused, or that finds validation unavailable, says why in `reason` and does not use up a call.
        The dev score estimates, but is not, your final score.
        """
        async with self._lock:
            state = load_state()
            if state.calls >= self.config.max_calls:
                return self._refused(state, "budget spent")
            if state.last_call_at is not None:
                wait = state.last_call_at + self.config.min_interval_s - time.time()
                if wait > 0:
                    return self._refused(
                        state,
                        f"too soon: wait {math.ceil(wait)} s before calling again",
                    )

            grade = await self.config.grade(self.config.submission_paths)
            state.last_call_at = time.time()
            if grade.doubt == "scorer_failed":
                # The environment's fault: the call is not counted, and the
                # student learns nothing but that validation did not run.
                save_state(state)
                return self._refused(state, "validation is unavailable")
            delta = _delta(grade.dev_metric, state.last_valid_metric)
            state.calls += 1
            record_grade(state, grade, higher_is_better=self.config.higher_is_better)
            save_state(state)
            return self._result(state, grade, delta)

    def _refused(self, state: ValidationState, reason: str) -> ToolResult:
        return self._result(
            state, Grade(state.last_tier or "no_valid_output", reason=reason)
        )

    def _result(
        self, state: ValidationState, grade: Grade, delta: float | None = None
    ) -> ToolResult:
        direction = "higher" if self.config.higher_is_better else "lower"
        payload: dict[str, Any] = {
            "tier": grade.tier,
            "dev_metric": _three_significant_figures(grade.dev_metric),
            "metric": f"{self.config.metric}, {direction} is better",
            "reason": grade.reason,
            "delta_vs_prev_valid": _three_significant_figures(delta),
            "best_dev_metric": _three_significant_figures(state.best_metric),
            "elapsed_s": round(grade.elapsed_s, 1),
            "calls_left": max(self.config.max_calls - state.calls, 0),
        }
        return ToolResult(structured_content=payload)
