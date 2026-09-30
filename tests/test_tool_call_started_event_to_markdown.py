from karotte.schemas.chat import ChatCompletionMessageToolCall, Function
from karotte.schemas.transcript import ToolCallStartedEvent
from karotte.transcript_markdown import (
    convert_tool_call_started_event_to_markdown,
)


def test_tool_call_with_invalid_json_does_not_crash():
    """Test that tool calls with invalid JSON arguments are handled gracefully."""
    tool_call = ChatCompletionMessageToolCall(
        id="test_id",
        function=Function(
            name="bash",
            arguments='{"command": "cat > /wor',  # Invalid JSON - unclosed string
        ),
        type="function",
    )
    event = ToolCallStartedEvent(tool_call=tool_call)

    # This should not raise JSONDecodeError
    result = convert_tool_call_started_event_to_markdown(event)

    assert "bash" in result
    assert "Invalid" in result or "Error" in result


def test_submit_answers_with_non_dict_argument_does_not_crash():
    """Test that submit_answers with non-dict arguments returns early with error."""
    tool_call = ChatCompletionMessageToolCall(
        id="test_id",
        function=Function(
            name="submit_answers",
            arguments='"not an object"',  # A string, not a dict
        ),
        type="function",
    )
    event = ToolCallStartedEvent(tool_call=tool_call)

    # This should not raise AttributeError
    result = convert_tool_call_started_event_to_markdown(event)

    assert "Submitting Answers" in result
    assert "Invalid answer format" in result


def test_submit_answers_with_string_answers_value_does_not_crash():
    """Test that submit_answers handles the case where answers value is a string (double-serialized)."""
    tool_call = ChatCompletionMessageToolCall(
        id="test_id",
        function=Function(
            name="submit_answers",
            arguments='{"answers": "{\\"solution\\": \\"some text\\"}"}',
        ),
        type="function",
    )
    event = ToolCallStartedEvent(tool_call=tool_call)

    # This should not raise AttributeError
    result = convert_tool_call_started_event_to_markdown(event)

    assert "Submitting Answers" in result
    assert "Invalid answer format" in result
