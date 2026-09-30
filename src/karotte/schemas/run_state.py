import datetime as dt
from typing import Any

from pydantic import BaseModel

from . import RunStatus
from .scoring import Score
from .transcript import (
    AnswersSubmittedEvent,
    Event,
    MetadataEvent,
    ScoringEvent,
    StepCompletedEvent,
    StepStartedEvent,
    TaskCompletedEvent,
    TaskStartedEvent,
    TokenUsageEvent,
)


class RunState(BaseModel):
    """A helper class for tracking a run's state as events happen.

    Initialize with a `TaskStartedEvent` and `apply` events to update the state."""

    run_id: str
    task_id: str
    status: RunStatus
    start_time: dt.datetime
    score: Score | None = None
    n_steps: int
    current_step: int
    metadata: dict[str, Any] = {}
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_cache_read_tokens: int | None = None
    total_cache_write_tokens: int | None = None

    def __init__(self, task_started_event: TaskStartedEvent):
        super().__init__(
            status="running",
            start_time=task_started_event.timestamp,
            current_step=0,
            **task_started_event.model_dump(),
        )

    @property
    def is_terminal(self) -> bool:
        return self.status in ("passed", "failed", "error")

    def apply(self, event: Event):
        match event:
            case MetadataEvent():
                self.metadata.update(event.metadata)
            case StepStartedEvent():
                self.current_step = event.step
            case ScoringEvent():
                self.score = event.scoring.score
            case TaskCompletedEvent():
                self.status = event.status
            case TokenUsageEvent():
                self.total_input_tokens = (
                    self.total_input_tokens or 0
                ) + event.input_tokens
                self.total_output_tokens = (
                    self.total_output_tokens or 0
                ) + event.output_tokens
                if event.cache_read_tokens is not None:
                    self.total_cache_read_tokens = (
                        self.total_cache_read_tokens or 0
                    ) + event.cache_read_tokens
                if event.cache_write_tokens is not None:
                    self.total_cache_write_tokens = (
                        self.total_cache_write_tokens or 0
                    ) + event.cache_write_tokens
            case AnswersSubmittedEvent():
                # Answers can be retrieved from `transcript.answers`
                pass
            case StepCompletedEvent():
                # Nothing to do. `StepStartedEvent` already increments the step.
                pass
            case _:
                pass
