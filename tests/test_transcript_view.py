from typing import override

import pytest
from mcp.types import CallToolResult, TextContent
from textual.app import App, ComposeResult
from textual.widgets import TextArea

from karotte.schemas.chat import ChatCompletionMessageToolCall, Delta, Function, Message
from karotte.schemas.scoring import Scoring
from karotte.schemas.transcript import (
    AnswersSubmittedEvent,
    ErrorEvent,
    MessageAddedEvent,
    MessageChunkEvent,
    MessageChunkResetEvent,
    ScoringEvent,
    StepCompletedEvent,
    StepStartedEvent,
    TaskCompletedEvent,
    TaskPreHookCompletedEvent,
    TaskStartedEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from karotte.terminal.transcript_view import TranscriptView


class TranscriptViewTestApp(App[None]):
    """Test app that wraps TranscriptView."""

    @override
    def compose(self) -> ComposeResult:
        yield TranscriptView(id="transcript-view")


@pytest.mark.asyncio
async def test_initial_message_disappears_after_first_event_in_pretty_mode():
    """Test that 'Waiting for connection...' disappears after first event in pretty mode."""
    app = TranscriptViewTestApp()

    async with app.run_test() as pilot:
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        transcript_view.set_view_mode("pretty")

        # Verify initial message is shown (visible, not hidden)
        visible_initial = [
            m
            for m in transcript_view.query(".initial-message")
            if "hidden" not in m.classes
        ]
        assert len(visible_initial) == 1
        assert "Waiting for connection..." in str(visible_initial[0].render())

        # Send an event that renders in pretty mode
        event = TaskStartedEvent(
            run_id="run_123",
            task_id="task_456",
            n_steps=5,
        )
        event_json = event.model_dump_json()

        transcript_view.append_event(event_json)
        await pilot.pause()

        # Initial message should be hidden
        visible_initial = [
            m
            for m in transcript_view.query(".initial-message")
            if "hidden" not in m.classes
        ]
        assert len(visible_initial) == 0


@pytest.mark.asyncio
async def test_initial_message_disappears_after_silent_event_in_pretty_mode():
    """Test that 'Waiting for connection...' disappears even after events that don't render."""
    app = TranscriptViewTestApp()

    async with app.run_test() as pilot:
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        transcript_view.set_view_mode("pretty")

        # Verify initial message is shown
        visible_initial = [
            m
            for m in transcript_view.query(".initial-message")
            if "hidden" not in m.classes
        ]
        assert len(visible_initial) == 1

        # Send a StepStartedEvent - this doesn't render anything in pretty mode
        event = StepStartedEvent(step=0)
        event_json = event.model_dump_json()

        transcript_view.append_event(event_json)
        await pilot.pause()

        # Initial message should be hidden
        visible_initial = [
            m
            for m in transcript_view.query(".initial-message")
            if "hidden" not in m.classes
        ]
        assert len(visible_initial) == 0


@pytest.mark.asyncio
async def test_append_event_json_mode_skips_message_chunk_event():
    """Test that MessageChunkEvent is not added to the view."""
    app = TranscriptViewTestApp()

    async with app.run_test() as pilot:
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        transcript_view.set_view_mode("json")

        event = MessageChunkEvent(
            delta=Delta(
                content="Hello, world!",
                role="assistant",
                tool_calls=None,
            )
        )
        event_json = event.model_dump_json()

        transcript_view.append_event(event_json)
        await pilot.pause()

        json_area = transcript_view.query_one("#transcript-json", TextArea)

        # MessageChunkEvent should NOT be added
        assert json_area.text == "Waiting for connection..."


@pytest.mark.asyncio
async def test_append_event_json_mode_skips_message_chunk_reset_event():
    """Test that MessageChunkResetEvent is not added to the JSON view."""
    app = TranscriptViewTestApp()

    async with app.run_test() as pilot:
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        transcript_view.set_view_mode("json")

        event = MessageChunkResetEvent()
        event_json = event.model_dump_json()

        transcript_view.append_event(event_json)
        await pilot.pause()

        json_area = transcript_view.query_one("#transcript-json", TextArea)

        # MessageChunkResetEvent should NOT be added
        assert json_area.text == "Waiting for connection..."


@pytest.mark.asyncio
async def test_pretty_mode_skips_message_chunk_reset_event():
    """Test that MessageChunkResetEvent produces no output in pretty mode."""
    app = TranscriptViewTestApp()

    async with app.run_test() as pilot:
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        transcript_view.set_view_mode("pretty")

        event = MessageChunkResetEvent()
        event_json = event.model_dump_json()

        transcript_view.append_event(event_json)
        await pilot.pause()

        # convert_event_to_markdown should return empty string
        assert transcript_view.convert_event_to_markdown(event) == ""


@pytest.mark.asyncio
async def test_pretty_mode_skips_task_pre_hook_completed_event():
    """Test that TaskPreHookCompletedEvent produces no output in pretty mode."""
    app = TranscriptViewTestApp()

    async with app.run_test() as pilot:
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        transcript_view.set_view_mode("pretty")

        event = TaskPreHookCompletedEvent(metadata={"key": "value"})
        event_json = event.model_dump_json()

        transcript_view.append_event(event_json)
        await pilot.pause()

        # convert_event_to_markdown should return empty string
        assert transcript_view.convert_event_to_markdown(event) == ""


@pytest.mark.asyncio
async def test_append_event_json_mode_adds_task_started_event():
    """Test that TaskStartedEvent is added to the view."""
    app = TranscriptViewTestApp()

    async with app.run_test() as pilot:
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        transcript_view.set_view_mode("json")

        event = TaskStartedEvent(
            run_id="run_123",
            task_id="task_456",
            n_steps=5,
        )
        event_json = event.model_dump_json()

        transcript_view.append_event(event_json)
        await pilot.pause()

        json_area = transcript_view.query_one("#transcript-json", TextArea)

        assert "task_started" in json_area.text
        assert "run_123" in json_area.text
        assert "task_456" in json_area.text


@pytest.mark.asyncio
async def test_append_event_json_mode_adds_message_added_event():
    """Test that MessageAddedEvent is added to the view."""
    app = TranscriptViewTestApp()

    async with app.run_test() as pilot:
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        transcript_view.set_view_mode("json")

        event = MessageAddedEvent(
            message=Message(
                content="Test message content",
                role="assistant",
            )
        )
        event_json = event.model_dump_json()

        transcript_view.append_event(event_json)
        await pilot.pause()

        json_area = transcript_view.query_one("#transcript-json", TextArea)

        assert "message_added" in json_area.text
        assert "Test message content" in json_area.text


@pytest.mark.asyncio
async def test_append_event_json_mode_adds_tool_call_started_event():
    """Test that ToolCallStartedEvent is added to the view."""
    app = TranscriptViewTestApp()

    async with app.run_test() as pilot:
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        transcript_view.set_view_mode("json")

        event = ToolCallStartedEvent(
            tool_call=ChatCompletionMessageToolCall(
                id="call_123",
                function=Function(name="bash", arguments='{"command": "ls -la"}'),
                type="function",
            )
        )
        event_json = event.model_dump_json()

        transcript_view.append_event(event_json)
        await pilot.pause()

        json_area = transcript_view.query_one("#transcript-json", TextArea)

        assert "tool_call_started" in json_area.text
        assert "bash" in json_area.text


@pytest.mark.asyncio
async def test_append_event_json_mode_adds_tool_call_completed_event():
    """Test that ToolCallCompletedEvent is added to the view."""
    app = TranscriptViewTestApp()

    async with app.run_test() as pilot:
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        transcript_view.set_view_mode("json")

        event = ToolCallCompletedEvent(
            tool_call_id="call_123",
            result=CallToolResult(
                content=[TextContent(type="text", text="file1.txt\nfile2.txt")]
            ),
        )
        event_json = event.model_dump_json()

        transcript_view.append_event(event_json)
        await pilot.pause()

        json_area = transcript_view.query_one("#transcript-json", TextArea)

        assert "tool_call_completed" in json_area.text
        assert "file1.txt" in json_area.text


@pytest.mark.asyncio
async def test_append_event_json_mode_adds_scoring_event():
    """Test that ScoringEvent is added to the view."""
    app = TranscriptViewTestApp()

    async with app.run_test() as pilot:
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        transcript_view.set_view_mode("json")

        event = ScoringEvent(
            scoring=Scoring(
                score=0.85,
                metadata={"judge_name": "test_judge"},
                continue_task=False,
            )
        )
        event_json = event.model_dump_json()

        transcript_view.append_event(event_json)
        await pilot.pause()

        json_area = transcript_view.query_one("#transcript-json", TextArea)

        assert "scoring" in json_area.text
        assert "0.85" in json_area.text


@pytest.mark.asyncio
async def test_append_event_json_mode_adds_error_event():
    """Test that ErrorEvent is added to the view."""
    app = TranscriptViewTestApp()

    async with app.run_test() as pilot:
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        transcript_view.set_view_mode("json")

        event = ErrorEvent(
            exception_type="RuntimeError",
            message="Something went wrong",
        )
        event_json = event.model_dump_json()

        transcript_view.append_event(event_json)
        await pilot.pause()

        json_area = transcript_view.query_one("#transcript-json", TextArea)

        assert "error" in json_area.text
        assert "Something went wrong" in json_area.text


@pytest.mark.asyncio
async def test_append_event_json_mode_adds_task_completed_event():
    """Test that TaskCompletedEvent is added to the view."""
    app = TranscriptViewTestApp()

    async with app.run_test() as pilot:
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        transcript_view.set_view_mode("json")

        event = TaskCompletedEvent(status="passed")
        event_json = event.model_dump_json()

        transcript_view.append_event(event_json)
        await pilot.pause()

        json_area = transcript_view.query_one("#transcript-json", TextArea)

        assert "task_completed" in json_area.text
        assert "passed" in json_area.text


@pytest.mark.asyncio
async def test_append_event_json_mode_adds_step_started_event():
    """Test that StepStartedEvent is added to the view."""
    app = TranscriptViewTestApp()

    async with app.run_test() as pilot:
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        transcript_view.set_view_mode("json")

        event = StepStartedEvent(step=0)
        event_json = event.model_dump_json()

        transcript_view.append_event(event_json)
        await pilot.pause()

        json_area = transcript_view.query_one("#transcript-json", TextArea)

        assert "step_started" in json_area.text


@pytest.mark.asyncio
async def test_append_event_json_mode_adds_step_completed_event():
    """Test that StepCompletedEvent is added to the view."""
    app = TranscriptViewTestApp()

    async with app.run_test() as pilot:
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        transcript_view.set_view_mode("json")

        event = StepCompletedEvent(step=0)
        event_json = event.model_dump_json()

        transcript_view.append_event(event_json)
        await pilot.pause()

        json_area = transcript_view.query_one("#transcript-json", TextArea)

        assert "step_completed" in json_area.text


@pytest.mark.asyncio
async def test_append_event_json_mode_adds_answers_submitted_event():
    """Test that AnswersSubmittedEvent is added to the view."""
    app = TranscriptViewTestApp()

    async with app.run_test() as pilot:
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        transcript_view.set_view_mode("json")

        event = AnswersSubmittedEvent(answers={"q1": "a1"})
        event_json = event.model_dump_json()

        transcript_view.append_event(event_json)
        await pilot.pause()

        json_area = transcript_view.query_one("#transcript-json", TextArea)

        assert "answers_submitted" in json_area.text
        assert "q1" in json_area.text


@pytest.mark.asyncio
async def test_append_event_logs_mode_adds_raw_event():
    """Test that logs mode appends raw event text to the TextArea."""
    app = TranscriptViewTestApp()

    async with app.run_test() as pilot:
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        transcript_view.set_view_mode("logs")

        event = TaskStartedEvent(
            run_id="run_123",
            task_id="task_456",
            n_steps=5,
        )
        event_json = event.model_dump_json()

        transcript_view.append_event(event_json)
        await pilot.pause()

        logs_area = transcript_view.query_one("#transcript-logs", TextArea)

        # Logs mode appends raw JSON
        assert "run_123" in logs_area.text
        assert "task_456" in logs_area.text


@pytest.mark.asyncio
async def test_logs_mode_textarea_is_read_only():
    """Test that logs mode TextArea is read-only."""
    app = TranscriptViewTestApp()

    async with app.run_test():
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        logs_area = transcript_view.query_one("#transcript-logs", TextArea)

        assert logs_area.read_only is True


@pytest.mark.asyncio
async def test_json_mode_textarea_is_read_only():
    """Test that JSON mode TextArea is read-only."""
    app = TranscriptViewTestApp()

    async with app.run_test():
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        json_area = transcript_view.query_one("#transcript-json", TextArea)

        assert json_area.read_only is True


# =============================================================================
# Tests for streaming in pretty mode
# =============================================================================


@pytest.mark.asyncio
async def test_pretty_mode_chunk_creates_streaming_widget():
    """Test that a MessageChunkEvent creates a streaming widget in pretty mode."""
    app = TranscriptViewTestApp()

    async with app.run_test() as pilot:
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        transcript_view.set_view_mode("pretty")

        event = MessageChunkEvent(
            delta=Delta(content="Hello", role="assistant", tool_calls=None)
        )
        transcript_view.append_event(event.model_dump_json())
        await pilot.pause()

        assert transcript_view._streaming_widget is not None  # pyright: ignore[reportPrivateUsage]
        assert transcript_view._streaming_buffer == "Hello"  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_pretty_mode_chunks_accumulate():
    """Test that multiple chunks accumulate in the streaming buffer."""
    app = TranscriptViewTestApp()

    async with app.run_test() as pilot:
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        transcript_view.set_view_mode("pretty")

        for text in ["Hello", " ", "world"]:
            event = MessageChunkEvent(
                delta=Delta(content=text, role=None, tool_calls=None)
            )
            transcript_view.append_event(event.model_dump_json())

        await pilot.pause()

        assert transcript_view._streaming_buffer == "Hello world"  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_pretty_mode_chunk_with_no_content_is_ignored():
    """Test that a chunk with no content does not create a streaming widget."""
    app = TranscriptViewTestApp()

    async with app.run_test() as pilot:
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        transcript_view.set_view_mode("pretty")

        event = MessageChunkEvent(
            delta=Delta(content=None, role="assistant", tool_calls=None)
        )
        transcript_view.append_event(event.model_dump_json())
        await pilot.pause()

        assert transcript_view._streaming_widget is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_pretty_mode_reset_clears_streaming():
    """Test that MessageChunkResetEvent clears the streaming buffer."""
    app = TranscriptViewTestApp()

    async with app.run_test() as pilot:
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        transcript_view.set_view_mode("pretty")

        # Stream some content
        event = MessageChunkEvent(
            delta=Delta(content="partial content", role="assistant", tool_calls=None)
        )
        transcript_view.append_event(event.model_dump_json())
        await pilot.pause()

        assert transcript_view._streaming_buffer == "partial content"  # pyright: ignore[reportPrivateUsage]

        # Reset
        reset = MessageChunkResetEvent()
        transcript_view.append_event(reset.model_dump_json())
        await pilot.pause()

        assert transcript_view._streaming_buffer == ""  # pyright: ignore[reportPrivateUsage]
        # Widget still exists (ready for next attempt)
        assert transcript_view._streaming_widget is not None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_pretty_mode_reset_without_streaming_is_noop():
    """Test that MessageChunkResetEvent without an active stream is a no-op."""
    app = TranscriptViewTestApp()

    async with app.run_test() as pilot:
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        transcript_view.set_view_mode("pretty")

        reset = MessageChunkResetEvent()
        transcript_view.append_event(reset.model_dump_json())
        await pilot.pause()

        assert transcript_view._streaming_widget is None  # pyright: ignore[reportPrivateUsage]
        assert transcript_view._streaming_buffer == ""  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_pretty_mode_message_added_replaces_streaming():
    """Test that an assistant MessageAddedEvent removes the streaming widget."""
    app = TranscriptViewTestApp()

    async with app.run_test() as pilot:
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        transcript_view.set_view_mode("pretty")

        # Stream some content
        chunk = MessageChunkEvent(
            delta=Delta(content="streaming text", role="assistant", tool_calls=None)
        )
        transcript_view.append_event(chunk.model_dump_json())
        await pilot.pause()

        assert transcript_view._streaming_widget is not None  # pyright: ignore[reportPrivateUsage]

        # Send the final MessageAddedEvent
        msg = MessageAddedEvent(
            message=Message(role="assistant", content="streaming text")
        )
        transcript_view.append_event(msg.model_dump_json())
        await pilot.pause()

        # Streaming widget should be gone
        assert transcript_view._streaming_widget is None  # pyright: ignore[reportPrivateUsage]
        assert transcript_view._streaming_buffer == ""  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_pretty_mode_user_message_does_not_affect_streaming():
    """Test that a user MessageAddedEvent does not remove the streaming widget."""
    app = TranscriptViewTestApp()

    async with app.run_test() as pilot:
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        transcript_view.set_view_mode("pretty")

        # Stream some content
        chunk = MessageChunkEvent(
            delta=Delta(content="streaming", role="assistant", tool_calls=None)
        )
        transcript_view.append_event(chunk.model_dump_json())
        await pilot.pause()

        # Send a user message (shouldn't clear streaming)
        msg = MessageAddedEvent(message=Message(role="user", content="user input"))
        transcript_view.append_event(msg.model_dump_json())
        await pilot.pause()

        # Streaming widget should still be there
        assert transcript_view._streaming_widget is not None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_pretty_mode_clear_transcript_resets_streaming():
    """Test that clear_transcript resets streaming state."""
    app = TranscriptViewTestApp()

    async with app.run_test() as pilot:
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        transcript_view.set_view_mode("pretty")

        chunk = MessageChunkEvent(
            delta=Delta(content="some text", role="assistant", tool_calls=None)
        )
        transcript_view.append_event(chunk.model_dump_json())
        await pilot.pause()

        assert transcript_view._streaming_widget is not None  # pyright: ignore[reportPrivateUsage]

        transcript_view.clear_transcript()
        await pilot.pause()

        assert transcript_view._streaming_widget is None  # pyright: ignore[reportPrivateUsage]
        assert transcript_view._streaming_buffer == ""  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_pretty_mode_retry_flow():
    """Test the full retry flow: chunks -> reset -> new chunks -> message added."""
    app = TranscriptViewTestApp()

    async with app.run_test() as pilot:
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        transcript_view.set_view_mode("pretty")

        # First attempt chunks
        for text in ["partial", " ", "response"]:
            chunk = MessageChunkEvent(
                delta=Delta(content=text, role=None, tool_calls=None)
            )
            transcript_view.append_event(chunk.model_dump_json())
        await pilot.pause()

        assert transcript_view._streaming_buffer == "partial response"  # pyright: ignore[reportPrivateUsage]

        # Reset (retry)
        transcript_view.append_event(MessageChunkResetEvent().model_dump_json())
        await pilot.pause()

        assert transcript_view._streaming_buffer == ""  # pyright: ignore[reportPrivateUsage]

        # Second attempt chunks
        for text in ["good", " ", "response"]:
            chunk = MessageChunkEvent(
                delta=Delta(content=text, role=None, tool_calls=None)
            )
            transcript_view.append_event(chunk.model_dump_json())
        await pilot.pause()

        assert transcript_view._streaming_buffer == "good response"  # pyright: ignore[reportPrivateUsage]

        # Final message
        msg = MessageAddedEvent(
            message=Message(role="assistant", content="good response")
        )
        transcript_view.append_event(msg.model_dump_json())
        await pilot.pause()

        assert transcript_view._streaming_widget is None  # pyright: ignore[reportPrivateUsage]


class TestSimplifyEventForDisplay:
    """Tests for the simplify_event_for_display function."""

    def test_replaces_image_data_with_placeholder(self):
        """Test that image content data is replaced with a placeholder."""
        from mcp.types import ImageContent

        from karotte.terminal.transcript_view import simplify_event_for_display

        large_base64 = "iVBORw0KGgo" + "A" * 200_000  # Large base64 image data
        event = ToolCallCompletedEvent(
            tool_call_id="call_123",
            result=CallToolResult(
                content=[
                    ImageContent(type="image", data=large_base64, mimeType="image/png")
                ]
            ),
        )

        result = simplify_event_for_display(event)
        result_json = result.model_dump_json()

        assert "<base64-encoded image>" in result_json
        assert large_base64 not in result_json
        # Result should be much smaller than original
        assert len(result_json) < len(event.model_dump_json())

    def test_preserves_text_content(self):
        """Test that text content is not modified."""
        from karotte.terminal.transcript_view import simplify_event_for_display

        text = "some output text"
        event = ToolCallCompletedEvent(
            tool_call_id="call_123",
            result=CallToolResult(content=[TextContent(type="text", text=text)]),
        )

        result = simplify_event_for_display(event)

        # Should return the same event (no images to simplify)
        assert isinstance(result, ToolCallCompletedEvent)
        assert result.result is not None
        assert result.result.content is not None
        content_item = result.result.content[0]
        assert isinstance(content_item, TextContent)
        assert content_item.text == text


class TestSimplifyMessageAddedEvent:
    """Tests for simplifying MessageAddedEvent with image_url content."""

    def test_replaces_image_url_data_with_placeholder(self):
        """Test that image_url base64 data is replaced with a placeholder."""
        from karotte.schemas.chat import Message
        from karotte.schemas.transcript import MessageAddedEvent
        from karotte.terminal.transcript_view import simplify_event_for_display

        large_base64 = "iVBORw0KGgo" + "A" * 200_000
        event = MessageAddedEvent(
            message=Message(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{large_base64}"},
                    }
                ],
                role="user",
            )
        )

        result = simplify_event_for_display(event)
        result_json = result.model_dump_json()

        assert "<base64-encoded image>" in result_json
        assert large_base64 not in result_json
        # Result should be much smaller than original
        assert len(result_json) < len(event.model_dump_json())

    def test_preserves_text_content_in_message(self):
        """Test that text content in messages is not modified."""
        from karotte.schemas.chat import Message
        from karotte.schemas.transcript import MessageAddedEvent
        from karotte.terminal.transcript_view import simplify_event_for_display

        event = MessageAddedEvent(
            message=Message(
                content=[{"type": "text", "text": "Hello world"}],
                role="user",
            )
        )

        result = simplify_event_for_display(event)

        # Should return the same event (no images to simplify)
        assert isinstance(result, MessageAddedEvent)
        assert result.message.content == event.message.content
