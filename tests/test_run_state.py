import pytest

from karotte.schemas.run_state import RunState
from karotte.schemas.scoring import Scoring
from karotte.schemas.transcript import (
    MetadataEvent,
    ScoringEvent,
    StepStartedEvent,
    TaskCompletedEvent,
    TaskPreHookCompletedEvent,
    TaskStartedEvent,
    TokenUsageEvent,
)


@pytest.fixture
def run_state():
    return RunState(
        task_started_event=TaskStartedEvent(
            run_id="run_id",
            task_id="task_id",
            n_steps=42,
        )
    )


def test_init(run_state: RunState):
    assert run_state.run_id == "run_id"
    assert run_state.task_id == "task_id"
    assert run_state.n_steps == 42
    assert run_state.status == "running"
    assert run_state.start_time
    assert run_state.score is None
    assert run_state.total_input_tokens is None
    assert run_state.total_output_tokens is None
    assert run_state.total_cache_read_tokens is None
    assert run_state.total_cache_write_tokens is None


def test_metadata_event(run_state: RunState):
    metadata = {"key": "value"}
    run_state.apply(MetadataEvent(metadata=metadata))

    assert run_state.metadata == metadata


def test_task_pre_hook_completed_event(run_state: RunState):
    metadata = {"key": "value"}
    run_state.apply(TaskPreHookCompletedEvent(metadata=metadata))

    assert run_state.metadata == metadata


def test_step_started_event(run_state: RunState):
    run_state.apply(StepStartedEvent(step=5))

    assert run_state.current_step == 5


def test_scoring_event(run_state: RunState):
    score = 0.5
    run_state.apply(
        ScoringEvent(scoring=Scoring(score=score, metadata={}, continue_task=True))
    )

    assert run_state.score == score


def test_task_completed_event(run_state: RunState):
    run_state.apply(TaskCompletedEvent(status="passed"))

    assert run_state.status == "passed"


def test_token_usage_accumulation(run_state: RunState):
    """Test that token usage is accumulated across multiple TokenUsageEvents."""
    # First API call
    run_state.apply(
        TokenUsageEvent(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=200,
            cache_write_tokens=300,
        )
    )

    assert run_state.total_input_tokens == 100
    assert run_state.total_output_tokens == 50
    assert run_state.total_cache_read_tokens == 200
    assert run_state.total_cache_write_tokens == 300

    # Second API call
    run_state.apply(
        TokenUsageEvent(
            input_tokens=150,
            output_tokens=75,
            cache_read_tokens=250,
            cache_write_tokens=350,
        )
    )

    assert run_state.total_input_tokens == 250
    assert run_state.total_output_tokens == 125
    assert run_state.total_cache_read_tokens == 450
    assert run_state.total_cache_write_tokens == 650


def test_is_terminal_running(run_state: RunState):
    assert run_state.is_terminal is False


def test_is_terminal_passed(run_state: RunState):
    run_state.apply(TaskCompletedEvent(status="passed"))
    assert run_state.is_terminal is True


def test_is_terminal_failed(run_state: RunState):
    run_state.apply(TaskCompletedEvent(status="failed"))
    assert run_state.is_terminal is True


def test_is_terminal_error(run_state: RunState):
    run_state.apply(TaskCompletedEvent(status="error"))
    assert run_state.is_terminal is True


def test_token_usage_with_none_cache_tokens(run_state: RunState):
    """Test that None cache tokens are handled correctly - remain None when not provided."""
    run_state.apply(
        TokenUsageEvent(
            input_tokens=100,
            output_tokens=50,
        )
    )

    assert run_state.total_input_tokens == 100
    assert run_state.total_output_tokens == 50
    # When no cache tokens are provided, fields remain None
    assert run_state.total_cache_read_tokens is None
    assert run_state.total_cache_write_tokens is None


def test_token_usage_mixed_cache_tokens(run_state: RunState):
    """Test accumulation when some events have cache tokens and others don't."""
    # First call with cache tokens
    run_state.apply(
        TokenUsageEvent(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=200,
            cache_write_tokens=300,
        )
    )

    # Second call without cache tokens
    run_state.apply(
        TokenUsageEvent(
            input_tokens=100,
            output_tokens=50,
        )
    )

    assert run_state.total_input_tokens == 200
    assert run_state.total_output_tokens == 100
    assert run_state.total_cache_read_tokens == 200
    assert run_state.total_cache_write_tokens == 300
