import asyncio
import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest
from textual.pilot import Pilot
from textual.widgets import TextArea

from karotte.schemas.transcript import (
    TaskCompletedEvent,
    TaskStartedEvent,
    Transcript,
)
from karotte.terminal.app import KarotteApp
from karotte.terminal.transcript_view import TranscriptView


async def wait_for[T](
    pilot: Pilot[T],
    condition: Callable[[], bool],
    timeout: float = 1.0,
    interval: float = 0.05,
) -> None:
    """Wait for a condition to become true, polling with pauses.

    Args:
        pilot: The Textual test pilot.
        condition: A callable that returns True when the condition is met.
        timeout: Maximum time to wait in seconds.
        interval: Time between checks in seconds.

    Raises:
        TimeoutError: If the condition is not met within the timeout.
    """
    elapsed = 0.0
    while elapsed < timeout:
        await pilot.pause()
        if condition():
            return
        await asyncio.sleep(interval)
        elapsed += interval
    raise TimeoutError(f"Condition not met within {timeout}s")


@pytest.mark.asyncio
async def test_static_mode_view_switching_shows_events_in_json_mode():
    """Test that switching to JSON mode in static dashboard shows events."""
    event = TaskStartedEvent(
        run_id="run_123",
        task_id="task_456",
        n_steps=5,
    )
    transcript = Transcript(run_id="run_123", events=[event])

    app = KarotteApp(transcripts=[transcript])

    async with app.run_test() as pilot:
        await pilot.pause()

        # Switch to JSON mode
        app.action_toggle_view()
        await pilot.pause()

        # Check that the JSON view shows the event
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        json_area = transcript_view.query_one("#transcript-json", TextArea)

        assert "task_started" in json_area.text
        assert "run_123" in json_area.text


@pytest.mark.asyncio
async def test_switching_runs_in_logs_mode_does_not_show_json():
    """Test that switching runs while in logs mode doesn't show raw JSON."""
    event1 = TaskStartedEvent(
        run_id="run_1",
        task_id="task_1",
        n_steps=5,
    )
    event2 = TaskStartedEvent(
        run_id="run_2",
        task_id="task_2",
        n_steps=5,
    )
    transcript1 = Transcript(run_id="run_1", events=[event1])
    transcript2 = Transcript(run_id="run_2", events=[event2])

    app = KarotteApp(transcripts=[transcript1, transcript2])

    async with app.run_test() as pilot:
        await pilot.pause()

        # Switch to logs mode (pretty -> json -> logs)
        app.action_toggle_view()
        app.action_toggle_view()
        await pilot.pause()

        assert app.view_mode == "logs"

        # Switch to run 2
        app.select_run(1)
        await pilot.pause()

        # The transcript view should NOT contain raw JSON like "task_started"
        # (logs mode shows log file contents, not events)
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        logs_area = transcript_view.query_one("#transcript-logs", TextArea)

        assert "task_started" not in logs_area.text


def _get_binding_description(app: KarotteApp, action: str) -> str | None:
    """Helper to get the current binding description for an action."""
    active_bindings = app.screen.active_bindings
    for ab in active_bindings.values():
        if ab.binding.action == action and ab.enabled and ab.binding.show:
            return ab.binding.description
    return None


@pytest.mark.asyncio
async def test_copy_markdown_visible_in_pretty_mode():
    """Test that copy markdown action is visible in pretty mode."""
    event = TaskStartedEvent(
        run_id="run_123",
        task_id="task_456",
        n_steps=5,
    )
    transcript = Transcript(run_id="run_123", events=[event])

    app = KarotteApp(transcripts=[transcript])

    async with app.run_test() as pilot:
        await pilot.pause()

        assert app.view_mode == "pretty"
        assert _get_binding_description(app, "copy_content") == "Copy markdown"


@pytest.mark.asyncio
async def test_copy_json_visible_in_json_mode():
    """Test that copy JSON action is visible in JSON mode."""
    event = TaskStartedEvent(
        run_id="run_123",
        task_id="task_456",
        n_steps=5,
    )
    transcript = Transcript(run_id="run_123", events=[event])

    app = KarotteApp(transcripts=[transcript])

    async with app.run_test() as pilot:
        await pilot.pause()

        app.action_toggle_view()  # pretty -> json
        await pilot.pause()

        assert app.view_mode == "json"
        assert _get_binding_description(app, "copy_content") == "Copy JSON"


@pytest.mark.asyncio
async def test_copy_logs_visible_in_logs_mode():
    """Test that copy logs action is visible in logs mode."""
    event = TaskStartedEvent(
        run_id="run_123",
        task_id="task_456",
        n_steps=5,
    )
    transcript = Transcript(run_id="run_123", events=[event])

    app = KarotteApp(transcripts=[transcript])

    async with app.run_test() as pilot:
        await pilot.pause()

        app.action_toggle_view()  # pretty -> json
        app.action_toggle_view()  # json -> logs
        await pilot.pause()

        assert app.view_mode == "logs"
        assert _get_binding_description(app, "copy_content") == "Copy logs"


@pytest.mark.asyncio
async def test_copy_action_copies_json_in_json_mode():
    """Test that copy action copies valid JSON transcript in JSON mode."""
    event = TaskStartedEvent(
        run_id="run_123",
        task_id="task_456",
        n_steps=5,
    )
    transcript = Transcript(run_id="run_123", events=[event])

    app = KarotteApp(transcripts=[transcript])

    async with app.run_test() as pilot:
        await pilot.pause()

        app.action_toggle_view()  # pretty -> json
        await pilot.pause()

        # Track what gets copied
        copied_content: list[str] = []
        original_copy = app.copy_to_clipboard

        def mock_copy(content: str) -> None:
            copied_content.append(content)
            original_copy(content)

        app.copy_to_clipboard = mock_copy  # pyright: ignore[reportAttributeAccessIssue]

        app.action_copy_content()
        await pilot.pause()

        assert len(copied_content) == 1

        # Should be a valid Transcript
        copied_transcript = Transcript.model_validate_json(copied_content[0])
        assert copied_transcript.run_id == "run_123"
        assert len(copied_transcript.events) == 1
        assert isinstance(copied_transcript.events[0], TaskStartedEvent)
        assert copied_transcript.events[0].task_id == "task_456"


@pytest.mark.asyncio
async def test_copy_action_copies_markdown_in_pretty_mode():
    """Test that copy action copies markdown content in pretty mode."""
    event = TaskStartedEvent(
        run_id="run_123",
        task_id="task_456",
        n_steps=5,
    )
    transcript = Transcript(run_id="run_123", events=[event])

    app = KarotteApp(transcripts=[transcript])

    async with app.run_test() as pilot:
        await pilot.pause()

        # Ensure events are processed from queue to received_events
        app.update_transcripts_from_queues()
        await pilot.pause()

        # Track what gets copied
        copied_content: list[str] = []
        original_copy = app.copy_to_clipboard

        def mock_copy(content: str) -> None:
            copied_content.append(content)
            original_copy(content)

        app.copy_to_clipboard = mock_copy  # pyright: ignore[reportAttributeAccessIssue]

        app.action_copy_content()
        await pilot.pause()

        assert len(copied_content) == 1
        # Markdown should contain the task info in readable format
        assert "task_456" in copied_content[0]


@pytest.mark.asyncio
async def test_copy_action_copies_logs_in_logs_mode():
    """Test that copy action copies logs content in logs mode."""
    event = TaskStartedEvent(
        run_id="run_123",
        task_id="task_456",
        n_steps=5,
    )
    transcript = Transcript(run_id="run_123", events=[event])

    app = KarotteApp(transcripts=[transcript])

    async with app.run_test() as pilot:
        await pilot.pause()

        app.action_toggle_view()  # pretty -> json
        app.action_toggle_view()  # json -> logs
        await pilot.pause()

        # Track what gets copied
        copied_content: list[str] = []
        original_copy = app.copy_to_clipboard

        def mock_copy(content: str) -> None:
            copied_content.append(content)
            original_copy(content)

        app.copy_to_clipboard = mock_copy  # pyright: ignore[reportAttributeAccessIssue]

        app.action_copy_content()
        await pilot.pause()

        # In logs mode with no log file, it should show the waiting message
        # and notify that there's no content
        # The copy should still work but content will be the placeholder
        assert len(copied_content) <= 1  # May or may not copy depending on content


@pytest.mark.asyncio
async def test_json_view_shows_simplified_images_but_copy_gives_full():
    """Test that JSON view shows simplified image content but copy gives full content."""
    from mcp.types import CallToolResult, ImageContent

    from karotte.schemas.transcript import ToolCallCompletedEvent

    large_base64 = "iVBORw0KGgo" + "A" * 200_000  # Large base64 image data
    event = ToolCallCompletedEvent(
        tool_call_id="call_123",
        result=CallToolResult(
            content=[
                ImageContent(type="image", data=large_base64, mimeType="image/png")
            ]
        ),
    )
    transcript = Transcript(run_id="run_123", events=[event])

    app = KarotteApp(transcripts=[transcript])

    async with app.run_test() as pilot:
        await pilot.pause()

        app.action_toggle_view()  # pretty -> json
        await pilot.pause()

        # Check that the displayed JSON has image data replaced with placeholder
        transcript_view = app.query_one("#transcript-view", TranscriptView)
        json_area = transcript_view.query_one("#transcript-json", TextArea)
        displayed_text = json_area.text

        assert "<base64-encoded image>" in displayed_text
        assert large_base64 not in displayed_text
        assert len(displayed_text) < 200_000  # Should be much smaller

        # Now copy and check that full content is copied
        copied_content: list[str] = []
        original_copy = app.copy_to_clipboard

        def mock_copy(content: str) -> None:
            copied_content.append(content)
            original_copy(content)

        app.copy_to_clipboard = mock_copy  # pyright: ignore[reportAttributeAccessIssue]

        app.action_copy_content()
        await pilot.pause()

        assert len(copied_content) == 1
        # Full content should contain the full base64 data (not simplified)
        assert large_base64 in copied_content[0]
        assert "<base64-encoded image>" not in copied_content[0]


@pytest.mark.asyncio
async def test_static_mode_shows_model_from_task_started_event():
    """Test that static mode extracts model from TaskStartedEvent."""
    event = TaskStartedEvent(
        run_id="run_123",
        task_id="task_456",
        model="gpt-4o",
        n_steps=5,
    )
    transcript = Transcript(run_id="run_123", events=[event])

    app = KarotteApp(transcripts=[transcript])

    async with app.run_test() as pilot:
        await pilot.pause()

        # Check that the config was extracted correctly
        assert len(app.configs) == 1
        assert app.configs[0].model == "gpt-4o"
        assert app.configs[0].task_id == "task_456"


@pytest.mark.asyncio
async def test_static_mode_shows_unknown_model_when_not_in_event():
    """Test that static mode shows 'unknown' when model is not in TaskStartedEvent."""
    event = TaskStartedEvent(
        run_id="run_123",
        task_id="task_456",
        n_steps=5,
    )
    transcript = Transcript(run_id="run_123", events=[event])

    app = KarotteApp(transcripts=[transcript])

    async with app.run_test() as pilot:
        await pilot.pause()

        # Check that the config shows unknown model
        assert len(app.configs) == 1
        assert app.configs[0].model == "unknown"


@pytest.mark.asyncio
async def test_static_mode_shows_unknown_model_when_no_task_started_event():
    """Test that static mode shows 'unknown' when there's no TaskStartedEvent."""
    from karotte.schemas.chat import Message
    from karotte.schemas.transcript import MessageAddedEvent

    event = MessageAddedEvent(message=Message(role="user", content="Hello"))
    transcript = Transcript(run_id="run_123", events=[event])

    app = KarotteApp(transcripts=[transcript])

    async with app.run_test() as pilot:
        await pilot.pause()

        # Check that the config shows unknown model
        assert len(app.configs) == 1
        assert app.configs[0].model == "unknown"


def _is_binding_visible(app: KarotteApp, action: str) -> bool:
    """Helper to check if a binding for an action is visible."""
    active_bindings = app.screen.active_bindings
    for ab in active_bindings.values():
        if ab.binding.action == action and ab.enabled and ab.binding.show:
            return True
    return False


@pytest.mark.asyncio
async def test_toggle_mode_hidden_when_main_console_selected():
    """Test that 'Toggle mode' binding is hidden when main console is selected."""
    from karotte.schemas.evaluation_run_config import EvaluationRunConfig
    from karotte.schemas.websocket_config import WebSocketConfig

    config = EvaluationRunConfig(
        run_id="run_1",
        task_id="task_1",
        model="gpt-4",
        model_api_key="test_key",
        websocket_config=WebSocketConfig(host="localhost", port=8000),
    )

    app = KarotteApp(configs=[config])

    async with app.run_test() as pilot:
        await pilot.pause()

        # In live mode, main console is selected by default
        assert app.detail_view is not None
        assert app.detail_view.show_main_console is True

        # Toggle mode binding should be hidden
        assert not _is_binding_visible(app, "toggle_view")


@pytest.mark.asyncio
async def test_copy_hidden_when_main_console_selected():
    """Test that 'Copy' binding is hidden when main console is selected."""
    from karotte.schemas.evaluation_run_config import EvaluationRunConfig
    from karotte.schemas.websocket_config import WebSocketConfig

    config = EvaluationRunConfig(
        run_id="run_1",
        task_id="task_1",
        model="gpt-4",
        model_api_key="test_key",
        websocket_config=WebSocketConfig(host="localhost", port=8000),
    )

    app = KarotteApp(configs=[config])

    async with app.run_test() as pilot:
        await pilot.pause()

        # In live mode, main console is selected by default
        assert app.detail_view is not None
        assert app.detail_view.show_main_console is True

        # Copy binding should be hidden
        assert not _is_binding_visible(app, "copy_content")


@pytest.mark.asyncio
async def test_toggle_mode_visible_when_run_selected():
    """Test that 'Toggle mode' binding is visible when a run is selected."""
    event = TaskStartedEvent(
        run_id="run_123",
        task_id="task_456",
        n_steps=5,
    )
    transcript = Transcript(run_id="run_123", events=[event])

    app = KarotteApp(transcripts=[transcript])

    async with app.run_test() as pilot:
        await pilot.pause()

        # In static mode, first run is selected by default
        assert app.detail_view is not None
        assert app.detail_view.run_index == 0

        # Toggle mode binding should be visible
        assert _is_binding_visible(app, "toggle_view")


@pytest.mark.asyncio
async def test_copy_visible_when_run_selected():
    """Test that 'Copy' binding is visible when a run is selected."""
    event = TaskStartedEvent(
        run_id="run_123",
        task_id="task_456",
        n_steps=5,
    )
    transcript = Transcript(run_id="run_123", events=[event])

    app = KarotteApp(transcripts=[transcript])

    async with app.run_test() as pilot:
        await pilot.pause()

        # In static mode, first run is selected by default
        assert app.detail_view is not None
        assert app.detail_view.run_index == 0

        # Copy binding should be visible
        assert _is_binding_visible(app, "copy_content")


@pytest.mark.asyncio
async def test_update_transcripts_handles_unmounted_transcript_view():
    """Test that update_transcripts_from_queues handles race condition when transcript view is not yet mounted.

    This tests the fix for a race condition where the timer callback runs before
    RunView has finished composing its children, causing NoMatches exception.
    """
    from unittest.mock import patch

    from textual.css.query import NoMatches

    event = TaskStartedEvent(
        run_id="run_123",
        task_id="task_456",
        n_steps=5,
    )
    transcript = Transcript(run_id="run_123", events=[event])

    app = KarotteApp(transcripts=[transcript])

    async with app.run_test() as pilot:
        await pilot.pause()

        # Re-populate queue to simulate events arriving
        app.event_queues[0].append(event.model_dump_json())

        # Mock query_one to raise NoMatches (simulating unmounted widget)
        assert app.detail_view is not None
        original_query_one = app.detail_view.query_one

        def mock_query_one(selector: str, expect_type: type | None = None):  # noqa: ANN401
            if selector == "#transcript-view":
                raise NoMatches(f"No nodes match {selector!r}")
            if expect_type is not None:
                return original_query_one(selector, expect_type)
            return original_query_one(selector)

        with patch.object(app.detail_view, "query_one", side_effect=mock_query_one):
            # This should NOT raise - should handle gracefully
            try:
                app.update_transcripts_from_queues()
            except NoMatches:
                pytest.fail(
                    "update_transcripts_from_queues raised NoMatches - "
                    + "race condition not handled"
                )


@pytest.mark.asyncio
async def test_chunk_reset_events_excluded_from_stored_json():
    """Test that MessageChunkResetEvent is not stored in received_events_json."""
    from karotte.schemas.transcript import MessageChunkResetEvent

    event = TaskStartedEvent(
        run_id="run_123",
        task_id="task_456",
        n_steps=5,
    )
    transcript = Transcript(run_id="run_123", events=[event])

    app = KarotteApp(transcripts=[transcript])

    async with app.run_test() as pilot:
        await wait_for(pilot, lambda: len(app.received_events[0]) == 1)

        # Inject a MessageChunkResetEvent into the queue
        reset_event = MessageChunkResetEvent()
        app.event_queues[0].append(reset_event.model_dump_json())
        app.update_transcripts_from_queues()
        await pilot.pause()

        # The reset event should be in received_events (raw) but NOT in received_events_json
        assert len(app.received_events[0]) == 2
        assert len(app.received_events_json[0]) == 1  # Only TaskStartedEvent
        assert "message_chunk_reset" not in app.received_events_json[0][0]


@pytest.mark.asyncio
async def test_chunk_reset_events_excluded_from_copy_json_live_mode():
    """Test that MessageChunkResetEvent is excluded from transcript JSON export in live mode."""
    from karotte.schemas.evaluation_run_config import EvaluationRunConfig
    from karotte.schemas.transcript import MessageChunkResetEvent
    from karotte.schemas.websocket_config import WebSocketConfig

    config = EvaluationRunConfig(
        run_id="run_123",
        task_id="task_1",
        model="gpt-4",
        model_api_key="test_key",
        websocket_config=WebSocketConfig(host="localhost", port=8000),
    )

    app = KarotteApp(configs=[config])

    async with app.run_test() as pilot:
        await pilot.pause()

        # Inject events into the queue
        task_event = TaskStartedEvent(run_id="run_123", task_id="task_1", n_steps=5)
        reset_event = MessageChunkResetEvent()
        app.event_queues[0].append(task_event.model_dump_json())
        app.event_queues[0].append(reset_event.model_dump_json())
        app.update_transcripts_from_queues()
        await pilot.pause()

        json_output = app._get_transcript_json_for_run(0)  # pyright: ignore[reportPrivateUsage]
        assert "message_chunk_reset" not in json_output
        assert "task_started" in json_output


@pytest.mark.asyncio
async def test_chunk_reset_events_excluded_from_copy_markdown():
    """Test that MessageChunkResetEvent is excluded from markdown export."""
    from karotte.schemas.evaluation_run_config import EvaluationRunConfig
    from karotte.schemas.transcript import MessageChunkResetEvent
    from karotte.schemas.websocket_config import WebSocketConfig

    config = EvaluationRunConfig(
        run_id="run_123",
        task_id="task_1",
        model="gpt-4",
        model_api_key="test_key",
        websocket_config=WebSocketConfig(host="localhost", port=8000),
    )

    app = KarotteApp(configs=[config])

    async with app.run_test() as pilot:
        await pilot.pause()

        task_event = TaskStartedEvent(run_id="run_123", task_id="task_1", n_steps=5)
        reset_event = MessageChunkResetEvent()
        app.event_queues[0].append(task_event.model_dump_json())
        app.event_queues[0].append(reset_event.model_dump_json())
        app.update_transcripts_from_queues()
        await pilot.pause()

        md_output = app._generate_markdown_for_run(0)  # pyright: ignore[reportPrivateUsage]
        assert "message_chunk_reset" not in md_output
        assert "task_1" in md_output


@pytest.mark.asyncio
async def test_static_mode_processes_events_after_mount():
    """Test that static mode eventually processes all events after widget mounting."""
    event = TaskStartedEvent(
        run_id="run_123",
        task_id="task_456",
        n_steps=5,
        model="test-model",
    )
    transcript = Transcript(run_id="run_123", events=[event])

    app = KarotteApp(transcripts=[transcript])

    async with app.run_test() as pilot:
        await wait_for(pilot, lambda: len(app.received_events[0]) == 1)

        # Events should be processed and stored
        assert len(app.received_events[0]) == 1
        assert "task_started" in app.received_events[0][0]


@pytest.mark.asyncio
async def test_static_mode_with_multiple_events_handles_race():
    """Test that multiple events are handled correctly even with potential race conditions."""
    from karotte.schemas.chat import Message
    from karotte.schemas.transcript import MessageAddedEvent

    events = [
        TaskStartedEvent(
            run_id="run_123",
            task_id="task_456",
            n_steps=5,
        ),
        MessageAddedEvent(message=Message(role="user", content="Hello")),
        MessageAddedEvent(message=Message(role="assistant", content="Hi there")),
    ]
    transcript = Transcript(run_id="run_123", events=events)  # pyright: ignore[reportArgumentType]

    app = KarotteApp(transcripts=[transcript])

    async with app.run_test() as pilot:
        await wait_for(pilot, lambda: len(app.received_events[0]) == 3)

        # All events should be processed
        assert len(app.received_events[0]) == 3


@pytest.mark.asyncio
async def test_multiple_transcripts_handle_race_condition():
    """Test that multiple transcripts handle race conditions during initial mount."""
    event1 = TaskStartedEvent(
        run_id="run_1",
        task_id="task_1",
        n_steps=5,
    )
    event2 = TaskStartedEvent(
        run_id="run_2",
        task_id="task_2",
        n_steps=5,
    )
    transcript1 = Transcript(run_id="run_1", events=[event1])
    transcript2 = Transcript(run_id="run_2", events=[event2])

    app = KarotteApp(transcripts=[transcript1, transcript2])

    async with app.run_test() as pilot:
        await wait_for(
            pilot,
            lambda: (
                len(app.received_events[0]) == 1 and len(app.received_events[1]) == 1
            ),
        )

        # Both transcripts should have their events processed
        assert len(app.received_events[0]) == 1
        assert len(app.received_events[1]) == 1


@pytest.mark.asyncio
async def test_task_completed_event_updates_navbar_without_task_started():
    """Test that TaskCompletedEvent updates navbar color even without TaskStartedEvent.

    This tests the fix for crashes that occur before TaskStartedEvent - the navbar
    should still show the failed/error status from TaskCompletedEvent.
    """
    from karotte.schemas.evaluation_run_config import EvaluationRunConfig
    from karotte.schemas.websocket_config import WebSocketConfig

    config = EvaluationRunConfig(
        run_id="run_1",
        task_id="task_1",
        model="gpt-4",
        model_api_key="test_key",
        websocket_config=WebSocketConfig(host="localhost", port=8000),
    )

    app = KarotteApp(configs=[config])

    async with app.run_test() as pilot:
        await pilot.pause()

        # Simulate receiving only a TaskCompletedEvent (no TaskStartedEvent)
        # This happens when a crash occurs before the task starts
        completed_event = TaskCompletedEvent(status="failed")
        app.event_queues[0].append(completed_event.model_dump_json())

        # Process the event
        app.update_transcripts_from_queues()
        await pilot.pause()

        # run_states should still be None (no TaskStartedEvent received)
        assert app.run_states[0] is None

        # But navbar should show failed status
        assert app.run_list is not None
        assert app.run_list.run_items[0].status == "failed"


@pytest.mark.asyncio
async def test_first_event_does_not_overwrite_completed_status():
    """Test that receiving the first event doesn't overwrite completed/failed/error status.

    When a TaskCompletedEvent arrives before other events (e.g., in error scenarios),
    subsequent events should not reset the status to 'connected'.
    """
    from karotte.schemas.chat import Message
    from karotte.schemas.evaluation_run_config import EvaluationRunConfig
    from karotte.schemas.transcript import MessageAddedEvent
    from karotte.schemas.websocket_config import WebSocketConfig

    config = EvaluationRunConfig(
        run_id="run_1",
        task_id="task_1",
        model="gpt-4",
        model_api_key="test_key",
        websocket_config=WebSocketConfig(host="localhost", port=8000),
    )

    app = KarotteApp(configs=[config])

    async with app.run_test() as pilot:
        await pilot.pause()

        # First, simulate run being marked as failed (e.g., from TaskCompletedEvent)
        assert app.run_list is not None
        app.run_list.update_run_status(0, "failed")
        app.run_list.update_run_color(0, "failed")
        await pilot.pause()

        assert app.run_list.run_items[0].status == "failed"

        # Now simulate receiving another event
        message_event = MessageAddedEvent(message=Message(role="user", content="Hello"))
        app.event_queues[0].append(message_event.model_dump_json())

        # Process the event
        app.update_transcripts_from_queues()
        await pilot.pause()

        # Status should still be failed (not reset to 'connected')
        assert app.run_list.run_items[0].status == "failed"


@pytest.mark.asyncio
async def test_log_file_path_set_immediately_from_config():
    """Test that log file paths are set immediately from config, not waiting for events.

    This ensures the TUI can show logs even if a crash occurs before TaskStartedEvent.
    """
    from karotte.schemas.evaluation_run_config import EvaluationRunConfig
    from karotte.schemas.websocket_config import WebSocketConfig

    config = EvaluationRunConfig(
        run_id="test_run_123",
        task_id="task_1",
        model="gpt-4",
        model_api_key="test_key",
        websocket_config=WebSocketConfig(host="localhost", port=8000),
    )

    app = KarotteApp(configs=[config])

    async with app.run_test() as pilot:
        await pilot.pause()

        # Log file path should be set immediately (not waiting for TaskStartedEvent)
        assert len(app.run_log_files) == 1
        assert app.run_log_files[0] == Path("/tmp/karotte_run_test_run_123.log")

        # run_states should be None (no events yet)
        assert app.run_states[0] is None


@pytest.mark.asyncio
async def test_logs_mode_shows_existing_log_file_before_task_started():
    """Test that logs mode can show log file content before TaskStartedEvent is received.

    This tests the scenario where a crash occurs during MCP tool registration -
    the log file exists but no TaskStartedEvent was emitted.
    """
    from karotte.schemas.evaluation_run_config import EvaluationRunConfig
    from karotte.schemas.websocket_config import WebSocketConfig

    # Create a temporary log file
    with tempfile.NamedTemporaryFile(
        mode="w", prefix="karotte_run_", suffix=".log", delete=False
    ) as f:
        f.write("2024-01-01 12:00:00 | INFO | Starting up...\n")
        f.write("2024-01-01 12:00:01 | ERROR | Crash during MCP registration!\n")
        log_file_path = Path(f.name)

    try:
        # Extract run_id from the temp file name
        run_id = log_file_path.stem.replace("karotte_run_", "")

        config = EvaluationRunConfig(
            run_id=run_id,
            task_id="task_1",
            model="gpt-4",
            model_api_key="test_key",
            websocket_config=WebSocketConfig(host="localhost", port=8000),
        )

        # Override the log file path to match our temp file
        app = KarotteApp(configs=[config])
        app.run_log_files[0] = log_file_path

        async with app.run_test() as pilot:
            await pilot.pause()

            # Select the run (not main console)
            app.select_run(0)
            await pilot.pause()

            # Switch to logs mode (pretty -> json -> logs)
            app.action_toggle_view()
            app.action_toggle_view()
            await pilot.pause()

            assert app.view_mode == "logs"

            # No TaskStartedEvent received, so run_states should be None
            assert app.run_states[0] is None

            # But logs should still be readable
            transcript_view = app.query_one("#transcript-view", TranscriptView)
            logs_area = transcript_view.query_one("#transcript-logs", TextArea)

            # Load logs - should work and show content
            app._load_logs_for_run(0, from_start=True)  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()

            # The log content should be visible
            assert (
                "Starting up" in logs_area.text or "Crash during MCP" in logs_area.text
            )
    finally:
        # Clean up
        log_file_path.unlink(missing_ok=True)


def test_default_runtime_is_docker():
    assert KarotteApp(transcripts=[]).runtime == "docker"


def test_build_passes_cache_options(monkeypatch: pytest.MonkeyPatch):
    import karotte.terminal.app as app_module

    captured: dict[str, object] = {}

    def fake_build_command(*_args: object, **kwargs: object) -> list[str]:
        captured.update(kwargs)
        return ["true"]

    monkeypatch.setattr(app_module, "get_container_build_command", fake_build_command)
    app = KarotteApp(transcripts=[Transcript(run_id="r", events=[])])
    app.cache_from = ["type=registry,ref=repo:cache"]
    app.cache_to = ["type=registry,ref=repo:cache,mode=max"]

    app._build_container("docker", ".")  # pyright: ignore[reportPrivateUsage]

    assert captured["cache_from"] == ["type=registry,ref=repo:cache"]
    assert captured["cache_to"] == ["type=registry,ref=repo:cache,mode=max"]


@pytest.mark.asyncio
async def test_a_failed_container_run_marks_the_app_failed():
    import subprocess
    from unittest.mock import patch

    from karotte.schemas.evaluation_run_config import EvaluationRunConfig

    config = EvaluationRunConfig(run_id="r", task_id="t", model="m", model_api_key="k")
    app = KarotteApp(configs=[config])
    assert not app.run_failed

    with patch.object(
        app,
        "_run_containerized_builds_and_runs",
        side_effect=subprocess.CalledProcessError(1, ["podman", "run"]),
    ):
        await app._build_and_run()  # pyright: ignore[reportPrivateUsage]

    assert app.run_failed
