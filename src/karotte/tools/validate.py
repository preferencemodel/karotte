import asyncio
import math
import time
from typing import Any, final

from fastmcp.tools.tool import ToolResult
from karotte import ToolBase
from karotte.validation import (
    ValidateConfig,
    ValidationState,
    grade_submission,
    load_state,
    record_grade,
    save_state,
)

__all__ = ["ValidateConfig", "validate"]


def _now() -> float:
    return time.time()


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
                wait = state.last_call_at + self.config.min_interval_s - _now()
                if wait > 0:
                    return self._refused(
                        state,
                        f"too soon: wait {math.ceil(wait)} s before calling again",
                    )

            grade = await grade_submission(self.config, self.config.submission_paths)
            state.last_call_at = _now()
            if grade.doubt == "scorer_failed":
                # The environment's fault: the call is not counted, and the
                # student learns nothing but that validation did not run.
                save_state(state)
                return self._refused(state, "validation is unavailable")
            delta = _delta(grade.dev_metric, state.last_valid_metric)
            state.calls += 1
            record_grade(self.config, state, grade)
            save_state(state)
            return self._result(
                state,
                tier=grade.tier,
                dev_metric=grade.dev_metric,
                delta=delta,
                reason=grade.reason,
                elapsed_s=grade.elapsed_s,
            )

    def _refused(self, state: ValidationState, reason: str) -> ToolResult:
        return self._result(
            state,
            tier=state.last_tier or "no_valid_output",
            dev_metric=None,
            delta=None,
            reason=reason,
            elapsed_s=0.0,
        )

    def _result(
        self,
        state: ValidationState,
        *,
        tier: str,
        dev_metric: float | None,
        delta: float | None,
        reason: str | None,
        elapsed_s: float,
    ) -> ToolResult:
        direction = "higher" if self.config.higher_is_better else "lower"
        payload: dict[str, Any] = {
            "tier": tier,
            "dev_metric": _three_significant_figures(dev_metric),
            "metric": f"{self.config.metric}, {direction} is better",
            "reason": reason,
            "delta_vs_prev_valid": _three_significant_figures(delta),
            "best_dev_metric": _three_significant_figures(state.best_metric),
            "elapsed_s": round(elapsed_s, 1),
            "calls_left": max(self.config.max_calls - state.calls, 0),
        }
        return ToolResult(structured_content=payload)
