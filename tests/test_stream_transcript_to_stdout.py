import json
from io import StringIO
from unittest.mock import patch

import pytest
from mcp.types import CallToolResult

from karotte.schemas.chat import (
    ChatCompletionMessageToolCall,
    Delta,
    Function,
    Message,
)
from karotte.schemas.scoring import Scoring
from karotte.schemas.transcript import (
    AnswersSubmittedEvent,
    ErrorEvent,
    MessageAddedEvent,
    MessageChunkEvent,
    MessageChunkResetEvent,
    ResourceMetrics,
    ResourceSample,
    ScoringEvent,
    StepCompletedEvent,
    StepStartedEvent,
    TaskCompletedEvent,
    TaskPreHookCompletedEvent,
    TaskStartedEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from karotte.transcript_streaming.stream_transcript_to_stdout import (
    stream_transcript_to_stdout,
)


@pytest.mark.asyncio
async def test_bash_tool_call_with_valid_json():
    """Test that bash tool calls with valid JSON are printed correctly."""
    tool_call = ChatCompletionMessageToolCall(
        id="test_id",
        function=Function(
            name="bash",
            arguments='{"command": "echo hello"}',
        ),
        type="function",
    )
    event = ToolCallStartedEvent(tool_call=tool_call)

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert "🔧 Calling Bash:" in output
    assert "echo hello" in output


@pytest.mark.asyncio
async def test_bash_tool_call_with_invalid_json():
    """Test that bash tool calls with invalid JSON are handled gracefully."""
    # This simulates the guard case from the user's example
    tool_call = ChatCompletionMessageToolCall(
        id="test_id",
        function=Function(
            name="bash",
            arguments='{"command": "cat > /wor',  # Invalid JSON - unclosed string
        ),
        type="function",
    )
    event = ToolCallStartedEvent(tool_call=tool_call)

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert "🔧 Calling Bash:" in output
    assert "Invalid arguments:" in output
    assert '{"command": "cat > /wor' in output


@pytest.mark.asyncio
async def test_bash_tool_call_with_missing_command_key():
    """Test that bash tool calls with valid JSON but missing command key are handled."""
    tool_call = ChatCompletionMessageToolCall(
        id="test_id",
        function=Function(
            name="bash",
            arguments='{"other_field": "value"}',  # Valid JSON but no "command" key
        ),
        type="function",
    )
    event = ToolCallStartedEvent(tool_call=tool_call)

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert "🔧 Calling Bash:" in output
    assert "Invalid arguments:" in output
    assert '{"other_field": "value"}' in output


@pytest.mark.asyncio
async def test_bash_tool_call_with_multiline_command():
    """Test that bash tool calls with multiline commands print each line."""
    multiline_command = "echo line1\necho line2\necho line3"
    tool_call = ChatCompletionMessageToolCall(
        id="test_id",
        function=Function(
            name="bash",
            arguments=json.dumps({"command": multiline_command}),
        ),
        type="function",
    )
    event = ToolCallStartedEvent(tool_call=tool_call)

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert "🔧 Calling Bash:" in output
    assert "echo line1" in output
    assert "echo line2" in output
    assert "echo line3" in output


@pytest.mark.asyncio
async def test_bash_tool_call_with_empty_json():
    """Test that bash tool calls with empty JSON object are handled."""
    tool_call = ChatCompletionMessageToolCall(
        id="test_id",
        function=Function(
            name="bash",
            arguments="{}",  # Valid JSON but empty
        ),
        type="function",
    )
    event = ToolCallStartedEvent(tool_call=tool_call)

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert "🔧 Calling Bash:" in output
    assert "Invalid arguments:" in output
    assert "{}" in output


@pytest.mark.asyncio
async def test_non_bash_tool_call():
    """Test that non-bash tool calls use the default formatting."""
    tool_call = ChatCompletionMessageToolCall(
        id="test_id",
        function=Function(
            name="other_tool",
            arguments='{"param": "value"}',
        ),
        type="function",
    )
    event = ToolCallStartedEvent(tool_call=tool_call)

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert "🔧 Calling tool: other_tool" in output
    assert '{"param": "value"}' in output


@pytest.mark.asyncio
async def test_bash_tool_call_with_completely_malformed_json():
    """Test bash tool calls with completely malformed JSON."""
    tool_call = ChatCompletionMessageToolCall(
        id="test_id",
        function=Function(
            name="bash",
            arguments="not json at all {[}]",  # Completely invalid
        ),
        type="function",
    )
    event = ToolCallStartedEvent(tool_call=tool_call)

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert "🔧 Calling Bash:" in output
    assert "Invalid arguments:" in output
    assert "not json at all {[}]" in output


@pytest.mark.asyncio
async def test_bash_tool_call_with_empty_string_arguments():
    """Test bash tool calls when arguments is an empty string."""
    tool_call = ChatCompletionMessageToolCall(
        id="test_id",
        function=Function(
            name="bash",
            arguments="",  # Empty string - will be parsed as empty object by json.loads
        ),
        type="function",
    )
    event = ToolCallStartedEvent(tool_call=tool_call)

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert "🔧 Calling Bash:" in output
    assert "Invalid arguments:" in output


@pytest.mark.asyncio
async def test_user_message_with_square_brackets_preserved():
    """Test that square brackets in user messages are preserved and not stripped by Rich markup."""
    message = Message(role="user", content="text1[text2]text3")
    event = MessageAddedEvent(message=message)

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert "text1[text2]text3" in output


@pytest.mark.asyncio
async def test_task_started_with_square_brackets_preserved():
    """Test that square brackets in task_id are preserved."""
    event = TaskStartedEvent(
        run_id="run_1",
        task_id="task[with]brackets",
        n_steps=10,
    )

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert "task[with]brackets" in output


@pytest.mark.asyncio
async def test_task_pre_hook_metadata_with_square_brackets_preserved():
    """Test that square brackets in metadata keys and values are preserved."""
    event = TaskPreHookCompletedEvent(
        metadata={"key[1]": "value[2]", "normal_key": "text1[text2]text3"}
    )

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert "key[1]" in output
    assert "value[2]" in output
    assert "text1[text2]text3" in output


@pytest.mark.asyncio
async def test_bash_command_with_square_brackets_preserved():
    """Test that square brackets in bash commands are preserved."""
    tool_call = ChatCompletionMessageToolCall(
        id="test_id",
        function=Function(
            name="bash",
            arguments=json.dumps({"command": "echo text1[text2]text3"}),
        ),
        type="function",
    )
    event = ToolCallStartedEvent(tool_call=tool_call)

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert "text1[text2]text3" in output


@pytest.mark.asyncio
async def test_other_tool_with_square_brackets_preserved():
    """Test that square brackets in tool name and arguments are preserved."""
    tool_call = ChatCompletionMessageToolCall(
        id="test_id",
        function=Function(
            name="tool[name]",
            arguments='{"key": "text1[text2]text3"}',
        ),
        type="function",
    )
    event = ToolCallStartedEvent(tool_call=tool_call)

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert "tool[name]" in output
    assert "text1[text2]text3" in output


@pytest.mark.asyncio
async def test_tool_completed_with_square_brackets_preserved():
    """Test that square brackets in tool result structuredContent are preserved."""
    result = CallToolResult(
        content=[],
        structuredContent={"key[1]": "value[2]", "result": "text1[text2]text3"},
    )
    event = ToolCallCompletedEvent(tool_call_id="test_id", result=result)

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert "key[1]" in output
    assert "value[2]" in output
    assert "text1[text2]text3" in output


@pytest.mark.asyncio
async def test_answers_submitted_with_square_brackets_preserved():
    """Test that square brackets in answers are preserved."""
    event = AnswersSubmittedEvent(
        answers={"question[1]": "answer[2]", "q2": "text1[text2]text3"}
    )

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert "question[1]" in output
    assert "answer[2]" in output
    assert "text1[text2]text3" in output


@pytest.mark.asyncio
async def test_scoring_metadata_with_square_brackets_preserved():
    """Test that square brackets in scoring metadata are preserved."""
    scoring = Scoring(
        score=0.5,
        metadata={"key[1]": "value[2]", "info": "text1[text2]text3"},
        continue_task=True,
    )
    event = ScoringEvent(scoring=scoring)

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert "key[1]" in output
    assert "value[2]" in output
    assert "text1[text2]text3" in output


@pytest.mark.asyncio
async def test_scoring_event_with_resource_metrics():
    """Test that resource metrics are output as JSON in scoring events."""
    metrics = ResourceMetrics(
        samples=[ResourceSample(timestamp_ms=0, cpu_percent=10.5, memory_mb=400.0)],
        peak_cpu_percent=15.0,
        avg_cpu_percent=10.5,
        peak_memory_mb=420.0,
        avg_memory_mb=400.0,
    )
    scoring = Scoring(score=1.0, metadata={}, continue_task=True)
    event = ScoringEvent(scoring=scoring, resource_metrics=metrics)

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert "Resource Metrics" in output
    # Verify the output contains JSON with the metrics data (Rich may wrap lines)
    unwrapped = output.replace("\n", "")
    assert '"peak_cpu_percent":15.0' in unwrapped
    assert '"avg_memory_mb":400.0' in unwrapped


@pytest.mark.asyncio
async def test_scoring_event_without_resource_metrics():
    """Test that resource metrics section is omitted when not present."""
    scoring = Scoring(score=1.0, metadata={}, continue_task=True)
    event = ScoringEvent(scoring=scoring)

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert "Resource Metrics" not in output


@pytest.mark.asyncio
async def test_error_event_with_square_brackets_preserved():
    """Test that square brackets in error messages are preserved."""
    event = ErrorEvent(
        exception_type="Error[Type]",
        message="text1[text2]text3",
    )

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert "Error[Type]" in output
    assert "text1[text2]text3" in output


@pytest.mark.asyncio
async def test_error_event_with_traceback():
    """Test that ErrorEvent with traceback displays the full traceback."""
    traceback_text = """Traceback (most recent call last):
  File "test.py", line 10, in <module>
    raise ValueError("Something went wrong")
ValueError: Something went wrong"""

    event = ErrorEvent(
        exception_type="ValueError",
        message="Something went wrong",
        traceback=traceback_text,
    )

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert "💥 Error occurred:" in output
    assert "ValueError" in output
    assert "Something went wrong" in output
    assert "Full Traceback:" in output
    assert 'File "test.py", line 10' in output
    assert "raise ValueError" in output


@pytest.mark.asyncio
async def test_error_event_without_traceback():
    """Test that ErrorEvent without traceback does not display traceback section."""
    event = ErrorEvent(
        exception_type="ValueError",
        message="Something went wrong",
        traceback=None,
    )

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert "💥 Error occurred:" in output
    assert "ValueError" in output
    assert "Something went wrong" in output
    assert "Full Traceback:" not in output


@pytest.mark.asyncio
async def test_error_event_traceback_with_square_brackets_preserved():
    """Test that square brackets in traceback are preserved."""
    traceback_text = """Traceback (most recent call last):
  File "test.py", line 10, in process[data]
    items[0] = value[key]
KeyError: 'missing[key]'"""

    event = ErrorEvent(
        exception_type="KeyError[Type]",
        message="missing[key]",
        traceback=traceback_text,
    )

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert "KeyError[Type]" in output
    assert "missing[key]" in output
    assert "Full Traceback:" in output
    assert "process[data]" in output
    assert "items[0] = value[key]" in output


@pytest.mark.asyncio
async def test_step_started_event():
    """Test that StepStartedEvent prints the step number."""
    event = StepStartedEvent(step=0)

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert "Starting step 1" in output


@pytest.mark.asyncio
async def test_message_chunk_event():
    """Test that MessageChunkEvent streams content to stdout."""
    delta = Delta(content="Hello world", role=None, tool_calls=None)
    event = MessageChunkEvent(delta=delta)

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert "Student:" in output
    assert "Assistant" not in output
    assert "Hello world" in output


@pytest.mark.asyncio
async def test_message_chunk_event_with_square_brackets_preserved():
    """Test that square brackets in message chunks are preserved (uses sys.stdout.write, not Rich)."""
    delta = Delta(content="text1[text2]text3", role=None, tool_calls=None)
    event = MessageChunkEvent(delta=delta)

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert "text1[text2]text3" in output


@pytest.mark.asyncio
async def test_message_chunk_event_empty_content():
    """Test that MessageChunkEvent with empty content does not print."""
    delta = Delta(content=None, role=None, tool_calls=None)
    event = MessageChunkEvent(delta=delta)

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert "Student:" not in output


@pytest.mark.asyncio
async def test_tool_completed_without_structured_content():
    """Test that ToolCallCompletedEvent without structuredContent prints header only."""
    result = CallToolResult(content=[], structuredContent=None)
    event = ToolCallCompletedEvent(tool_call_id="test_id", result=result)

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert "✅ Tool call completed:" in output


@pytest.mark.asyncio
async def test_step_completed_event():
    """Test that StepCompletedEvent does not crash (it's a no-op)."""
    event = StepCompletedEvent(step=0)

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    # StepCompletedEvent is a no-op, so output should be empty
    assert output == ""


async def _render(*events: object) -> str:
    async def event_generator():
        for event in events:
            yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())  # pyright: ignore[reportArgumentType]
        return mock_stdout.getvalue()


def _scoring(score: float, continue_task: bool, **metadata: str) -> ScoringEvent:
    return ScoringEvent(
        scoring=Scoring(score=score, metadata=metadata, continue_task=continue_task)
    )


@pytest.mark.asyncio
async def test_task_completed_prints_a_summary():
    output = await _render(
        TaskStartedEvent(run_id="r", task_id="t", n_steps=2),
        StepStartedEvent(step=0),
        _scoring(1.0, True),
        StepStartedEvent(step=1),
        _scoring(0.5, True),
        TaskCompletedEvent(status="passed"),
    )

    assert "Result: passed, 2/2 steps passed, final score 0.5" in output


@pytest.mark.asyncio
async def test_failed_task_summary_counts_only_passed_steps():
    output = await _render(
        TaskStartedEvent(run_id="r", task_id="t", n_steps=2),
        StepStartedEvent(step=0),
        _scoring(0.0, False),
        TaskCompletedEvent(status="failed"),
    )

    assert "Result: failed, 0/2 steps passed, final score 0.0" in output


@pytest.mark.asyncio
async def test_summary_without_scores_or_step_count():
    output = await _render(TaskCompletedEvent(status="error"))

    assert "Result: error, 0 steps passed, final score N/A" in output


@pytest.mark.asyncio
async def test_continue_task_is_not_printed_after_the_last_step():
    output = await _render(
        TaskStartedEvent(run_id="r", task_id="t", n_steps=2),
        StepStartedEvent(step=0),
        _scoring(1.0, True),
        StepStartedEvent(step=1),
        _scoring(1.0, True),
    )

    assert output.count("Continue task?") == 1


@pytest.mark.asyncio
async def test_continue_task_is_printed_when_the_step_count_is_unknown():
    output = await _render(
        TaskStartedEvent(run_id="r", task_id="t", n_steps=-1),
        StepStartedEvent(step=0),
        _scoring(1.0, True),
    )

    assert "Continue task? -> Yes" in output


@pytest.mark.asyncio
async def test_empty_scoring_metadata_is_not_printed():
    assert "Metadata:" not in await _render(_scoring(1.0, True))
    assert "Metadata:" in await _render(_scoring(1.0, True, reason="ok"))


@pytest.mark.asyncio
async def test_streamed_message_ends_with_a_newline():
    output = await _render(
        MessageChunkEvent(delta=Delta(content="Hello", role=None, tool_calls=None)),
        MessageAddedEvent(message=Message(role="assistant", content="Hello")),
    )

    assert output.endswith("Hello\n")


@pytest.mark.asyncio
async def test_streamed_message_ending_in_a_newline_gets_no_extra_one():
    output = await _render(
        MessageChunkEvent(delta=Delta(content="Hello\n", role=None, tool_calls=None)),
        MessageChunkResetEvent(),
    )

    assert output.endswith("Hello\n")
    assert not output.endswith("\n\n")


@pytest.mark.asyncio
async def test_unknown_event_type():
    """Test that unknown event types print an error message."""
    from pydantic import BaseModel

    class UnknownEvent(BaseModel):
        type: str = "unknown"

    event = UnknownEvent()

    async def event_generator():
        yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())  # pyright: ignore[reportArgumentType]
        output = mock_stdout.getvalue()

    assert "Unknown event type: UnknownEvent" in output


@pytest.mark.asyncio
async def test_all_event_types_are_handled():
    """Test that all event types in the Event union can be processed without crashing.

    This test ensures that whenever a new event type is added to the Event union,
    it is properly handled in stream_transcript_to_stdout.
    """
    from typing import get_args

    from karotte.schemas.transcript import Event, TokenUsageEvent

    # Get all event types from the Event union
    event_types = get_args(Event)

    # Create a minimal instance of each event type
    test_events = []

    for event_type in event_types:
        if event_type == TaskStartedEvent:
            test_events.append(
                TaskStartedEvent(run_id="test", task_id="test_task", n_steps=1)
            )
        elif event_type == TaskPreHookCompletedEvent:
            test_events.append(TaskPreHookCompletedEvent(metadata={}))
        elif event_type == StepStartedEvent:
            test_events.append(StepStartedEvent(step=0))
        elif event_type == MessageChunkEvent:
            test_events.append(
                MessageChunkEvent(
                    delta=Delta(content="test", role=None, tool_calls=None)
                )
            )
        elif event_type == MessageChunkResetEvent:
            test_events.append(MessageChunkResetEvent())
        elif event_type == MessageAddedEvent:
            test_events.append(
                MessageAddedEvent(message=Message(role="user", content="test"))
            )
        elif event_type == ToolCallStartedEvent:
            tool_call = ChatCompletionMessageToolCall(
                id="test_id",
                function=Function(name="test_tool", arguments="{}"),
                type="function",
            )
            test_events.append(ToolCallStartedEvent(tool_call=tool_call))
        elif event_type == ToolCallCompletedEvent:
            result = CallToolResult(content=[], structuredContent={})
            test_events.append(
                ToolCallCompletedEvent(tool_call_id="test_id", result=result)
            )
        elif event_type == AnswersSubmittedEvent:
            test_events.append(AnswersSubmittedEvent(answers={}))
        elif event_type == ScoringEvent:
            scoring = Scoring(score=1.0, metadata={}, continue_task=True)
            test_events.append(ScoringEvent(scoring=scoring))
        elif event_type == StepCompletedEvent:
            test_events.append(StepCompletedEvent(step=0))
        elif event_type == TaskCompletedEvent:
            test_events.append(TaskCompletedEvent(status="passed"))
        elif event_type == ErrorEvent:
            test_events.append(
                ErrorEvent(exception_type="TestError", message="test message")
            )
        elif event_type == TokenUsageEvent:
            test_events.append(TokenUsageEvent(input_tokens=100, output_tokens=50))
        elif hasattr(event_type, "__name__") and event_type.__name__ == "MetadataEvent":
            # MetadataEvent is handled by TaskPreHookCompletedEvent which is a subclass
            # Skip it as it's abstract
            continue
        else:
            # If a new event type is added and not handled here, this will fail
            pytest.fail(f"Unhandled event type in test: {event_type.__name__}")

    # Process all events - should not crash
    async def event_generator():
        for event in test_events:
            yield event

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    # The test passes if no exception was raised
    # We don't assert "Unknown event type" because all types should be handled
    assert "Unknown event type" not in output


@pytest.mark.asyncio
async def test_scoring_metadata_display_is_truncated():
    scoring = Scoring(
        score=0.5,
        metadata={"stdout": "HEAD" + "x" * 1_000_000 + "TAIL"},
        continue_task=True,
    )

    async def event_generator():
        yield ScoringEvent(scoring=scoring)

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert len(output) < 200_000
    assert "HEAD" in output
    assert "TAIL" in output
    assert "truncated" in output


@pytest.mark.asyncio
async def test_structured_content_display_is_truncated():
    result = CallToolResult(content=[], structuredContent={"output": "x" * 1_000_000})

    async def event_generator():
        yield ToolCallCompletedEvent(tool_call_id="test_id", result=result)

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert len(output) < 200_000
    assert "truncated" in output


@pytest.mark.asyncio
async def test_user_message_display_is_truncated():
    message = Message(
        content="x" * 1_000_000,
        role="user",
        tool_calls=None,
        reasoning_content=None,
        tool_call_id=None,
    )

    async def event_generator():
        yield MessageAddedEvent(message=message)

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert len(output) < 200_000
    assert "truncated" in output


@pytest.mark.asyncio
async def test_error_traceback_display_is_truncated():
    async def event_generator():
        yield ErrorEvent(
            exception_type="TestError",
            message="m" * 1_000_000,
            traceback="t" * 1_000_000,
        )

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        await stream_transcript_to_stdout(event_generator())
        output = mock_stdout.getvalue()

    assert len(output) < 400_000
    assert "truncated" in output


@pytest.mark.asyncio
async def test_rendering_a_display_cap_sized_message_is_fast():
    import time

    async def event_generator():
        yield ErrorEvent(
            exception_type="TestError", message="m" * 100_000, traceback=""
        )

    start = time.perf_counter()
    with patch("sys.stdout", new_callable=StringIO):
        await stream_transcript_to_stdout(event_generator())

    assert time.perf_counter() - start < 3
