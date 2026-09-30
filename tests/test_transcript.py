from karotte.schemas.chat import Message
from karotte.schemas.transcript import (
    AnswersSubmittedEvent,
    ErrorEvent,
    MessageAddedEvent,
    TokenUsageEvent,
    Transcript,
)


def test_retrieving_messages():
    transcript = Transcript(run_id="123")

    transcript.events.append(MessageAddedEvent(message=Message(content="Hello")))

    transcript.events.append(MessageAddedEvent(message=Message(content="World")))

    messages = transcript.messages

    assert messages == [Message(content="Hello"), Message(content="World")]


def test_retrieve_answers():
    transcript = Transcript(run_id="123")

    transcript.events.append(AnswersSubmittedEvent(answers={"a": "b"}))
    transcript.events.append(AnswersSubmittedEvent(answers={"a": "c"}))

    assert transcript.answers == {"a": "c"}


def test_error_event_backwards_compatibility():
    """Test that ErrorEvent without traceback field can be deserialized (backwards compatibility)."""
    # Simulate old ErrorEvent without traceback field
    error_event = ErrorEvent(
        exception_type="ValueError",
        message="Something went wrong",
    )

    assert error_event.exception_type == "ValueError"
    assert error_event.message == "Something went wrong"
    assert error_event.traceback is None


def test_error_event_with_traceback():
    """Test that ErrorEvent with traceback field works correctly."""
    error_event = ErrorEvent(
        exception_type="RuntimeError",
        message="Test error",
        traceback="Traceback (most recent call last):\n  File ...",
    )

    assert error_event.exception_type == "RuntimeError"
    assert error_event.message == "Test error"
    assert error_event.traceback == "Traceback (most recent call last):\n  File ..."


def test_token_usage_event():
    """Test that TokenUsageEvent can be created with all token types."""
    event = TokenUsageEvent(
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=200,
        cache_write_tokens=300,
    )

    assert event.type == "token_usage"
    assert event.input_tokens == 100
    assert event.output_tokens == 50
    assert event.cache_read_tokens == 200
    assert event.cache_write_tokens == 300


def test_token_usage_event_without_cache():
    """Test that TokenUsageEvent works without cache tokens."""
    event = TokenUsageEvent(
        input_tokens=100,
        output_tokens=50,
    )

    assert event.input_tokens == 100
    assert event.output_tokens == 50
    assert event.cache_read_tokens is None
    assert event.cache_write_tokens is None
