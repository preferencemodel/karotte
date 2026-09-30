from typing import Any, override

from loguru import logger
from pydantic import TypeAdapter
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import RichLog, Static, TextArea

from karotte.schemas.transcript import (
    AnswersSubmittedEvent,
    ErrorEvent,
    Event,
    MessageAddedEvent,
    MessageChunkEvent,
    MessageChunkResetEvent,
    MetadataEvent,
    ScoringEvent,
    StepCompletedEvent,
    StepStartedEvent,
    TaskCompletedEvent,
    TaskStartedEvent,
    TokenUsageEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from karotte.transcript_markdown import (
    convert_error_event_to_markdown,
    convert_message_added_event_to_markdown,
    convert_scoring_event_to_markdown,
    convert_task_started_event_to_markdown,
    convert_tool_call_completed_event_to_markdown,
    convert_tool_call_started_event_to_markdown,
)


class AutoScrollingVerticalScroll(VerticalScroll):
    """A VerticalScroll that auto-scrolls to bottom until user scrolls away."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._auto_scroll: bool = True
        self._programmatic_scroll: bool = False

    @override
    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        """Track scroll position to manage auto-scroll behavior."""
        super().watch_scroll_y(old_value, new_value)

        # Ignore programmatic scrolls (from scroll_end)
        if self._programmatic_scroll:
            return

        # User scrolled: check if they're at the bottom
        at_bottom = new_value >= self.max_scroll_y - 1
        self._auto_scroll = at_bottom

    def scroll_to_end_if_auto(self) -> None:
        """Scroll to end if auto-scroll is enabled, after layout is complete."""
        if self._auto_scroll:
            self.call_after_refresh(self._do_scroll_end)

    def _do_scroll_end(self) -> None:
        """Actually perform the scroll to end."""
        self._programmatic_scroll = True
        self.scroll_end(animate=False)
        self._programmatic_scroll = False

    def reset_auto_scroll(self) -> None:
        """Reset auto-scroll to enabled (e.g., when clearing content)."""
        self._auto_scroll = True


def _simplify_tool_call_completed_event(
    event: ToolCallCompletedEvent,
) -> ToolCallCompletedEvent:
    """Simplify ToolCallCompletedEvent by replacing image data with placeholder."""
    from copy import deepcopy

    from mcp.types import ImageContent

    if not event.result or not event.result.content:
        return event

    # Check if any content needs simplification (contains images)
    has_images = any(isinstance(item, ImageContent) for item in event.result.content)

    if not has_images:
        return event

    # Create a deep copy and replace image data with placeholder
    simplified_event = deepcopy(event)
    if simplified_event.result and simplified_event.result.content:
        for item in simplified_event.result.content:
            if isinstance(item, ImageContent):
                item.data = "<base64-encoded image>"

    return simplified_event


def _simplify_message_added_event(
    event: MessageAddedEvent,
) -> MessageAddedEvent:
    """Simplify MessageAddedEvent by replacing image_url base64 data with placeholder."""
    from copy import deepcopy

    if not event.message.content or not isinstance(event.message.content, list):
        return event

    # Check if any content needs simplification (contains image_url with base64)
    def is_base64_image(item: object) -> bool:
        return (
            isinstance(item, dict)
            and item.get("type") == "image_url"
            and isinstance(item.get("image_url"), dict)
            and str(item["image_url"].get("url", "")).startswith("data:")
        )

    has_base64_images = any(is_base64_image(item) for item in event.message.content)

    if not has_base64_images:
        return event

    # Create a deep copy and replace image_url data with placeholder
    simplified_event = deepcopy(event)
    if simplified_event.message.content and isinstance(
        simplified_event.message.content, list
    ):
        for item in simplified_event.message.content:
            if is_base64_image(item):
                item["image_url"]["url"] = "<base64-encoded image>"

    return simplified_event


def simplify_event_for_display(
    event: ToolCallCompletedEvent | MessageAddedEvent,
) -> ToolCallCompletedEvent | MessageAddedEvent:
    """Simplify event outputs for display.

    Returns a copy of the event with image content replaced by a placeholder
    to avoid displaying large base64-encoded data.

    Handles:
    - ToolCallCompletedEvent: replaces ImageContent.data with placeholder
    - MessageAddedEvent: replaces image_url base64 data with placeholder
    """
    if isinstance(event, ToolCallCompletedEvent):
        return _simplify_tool_call_completed_event(event)
    else:
        return _simplify_message_added_event(event)


class TranscriptView(VerticalScroll):
    """Scrollable view showing the transcript of a run."""

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ):
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self.auto_scroll: bool = True
        self.view_mode: str = "pretty"  # "json", "logs", or "pretty"
        self._has_events: bool = False
        # Streaming state for pretty mode
        self._streaming_widget: RichLog | None = None
        self._streaming_buffer: str = ""

    @override
    def compose(self) -> ComposeResult:
        # TextArea for JSON mode (read-only with syntax highlighting)
        with AutoScrollingVerticalScroll(
            id="transcript-json-container", classes="hidden"
        ):
            yield TextArea(
                "Waiting for connection...",
                id="transcript-json",
                read_only=True,
                language="json",
            )
        # TextArea for Logs mode (read-only, no syntax highlighting)
        with AutoScrollingVerticalScroll(
            id="transcript-logs-container", classes="hidden"
        ):
            yield TextArea(
                "Waiting for connection...",
                id="transcript-logs",
                read_only=True,
            )
        # Container for individual RichLog widgets in pretty mode (visible by default)
        with AutoScrollingVerticalScroll(id="transcript-events-container"):
            yield Static("Waiting for connection...", classes="initial-message")

    def set_view_mode(self, view_mode: str) -> None:
        """Set the view mode: json, logs, or pretty."""
        self.view_mode = view_mode
        # Show/hide appropriate container based on mode
        json_container = self.query_one(
            "#transcript-json-container", AutoScrollingVerticalScroll
        )
        logs_container = self.query_one(
            "#transcript-logs-container", AutoScrollingVerticalScroll
        )
        events_container = self.query_one(
            "#transcript-events-container", AutoScrollingVerticalScroll
        )

        json_container.add_class("hidden")
        logs_container.add_class("hidden")
        events_container.add_class("hidden")

        if view_mode == "json":
            json_container.remove_class("hidden")
        elif view_mode == "logs":
            logs_container.remove_class("hidden")
        else:
            events_container.remove_class("hidden")

    def set_json_content(self, content: str) -> None:
        """Set the full JSON content (for JSON view mode)."""
        json_area = self.query_one("#transcript-json", TextArea)
        json_container = self.query_one(
            "#transcript-json-container", AutoScrollingVerticalScroll
        )
        json_area.text = content if content else "Waiting for connection..."
        json_container.scroll_to_end_if_auto()

    def append_event(self, event_json: str) -> None:
        """Append a new event to the transcript."""
        if self.view_mode == "json":
            self._append_event_json(event_json)
        elif self.view_mode == "logs":
            self._append_event_logs(event_json)
        else:
            self._append_event_pretty(event_json)

    def _append_event_json(self, event_json: str) -> None:
        """Append event in JSON mode with indentation and truncation."""
        try:
            event: Event = TypeAdapter(Event).validate_json(event_json)

            if isinstance(event, (MessageChunkEvent, MessageChunkResetEvent)):
                return

            # Simplify events with images for display
            if isinstance(event, (ToolCallCompletedEvent, MessageAddedEvent)):
                event = simplify_event_for_display(event)

            formatted_json = event.model_dump_json(indent=2)
        except Exception:
            formatted_json = event_json

        json_area = self.query_one("#transcript-json", TextArea)
        json_container = self.query_one(
            "#transcript-json-container", AutoScrollingVerticalScroll
        )
        current = json_area.text

        if current == "Waiting for connection...":
            json_area.text = formatted_json
        else:
            json_area.text = f"{current}\n\n{formatted_json}"
        json_container.scroll_to_end_if_auto()

    def _append_event_logs(self, event: str) -> None:
        """Append event in logs mode (raw text)."""
        logs_area = self.query_one("#transcript-logs", TextArea)
        logs_container = self.query_one(
            "#transcript-logs-container", AutoScrollingVerticalScroll
        )
        current = logs_area.text

        if current == "Waiting for connection...":
            logs_area.text = event
        else:
            logs_area.text = f"{current}\n\n{event}"
        logs_container.scroll_to_end_if_auto()

    def _append_event_pretty(self, event: str) -> None:
        """Append event in pretty mode with markdown rendering."""
        events_container = self.query_one(
            "#transcript-events-container", AutoScrollingVerticalScroll
        )

        try:
            event_obj = TypeAdapter(Event).validate_json(event)

            # Hide initial message on first event
            if not self._has_events:
                self._has_events = True
                for msg in events_container.query(".initial-message"):
                    msg.add_class("hidden")

            # Handle streaming events
            if isinstance(event_obj, MessageChunkEvent):
                self._handle_chunk_event(event_obj, events_container)
                return

            if isinstance(event_obj, MessageChunkResetEvent):
                self._handle_chunk_reset_event(events_container)
                return

            # When a MessageAddedEvent arrives for an assistant message,
            # remove the streaming widget — the full message replaces it.
            if (
                isinstance(event_obj, MessageAddedEvent)
                and event_obj.message.role == "assistant"
            ):
                self._finalize_streaming()

            formatted_event = self.convert_event_to_markdown(event_obj)

            if formatted_event:
                self._mount_richlog_for_markdown(formatted_event, events_container)

        except Exception as e:
            # If parsing fails, log and skip
            logger.error(f"Error parsing markdown: {type(e).__name__}: {e}")

    def _handle_chunk_event(
        self,
        event: MessageChunkEvent,
        events_container: AutoScrollingVerticalScroll,
    ) -> None:
        """Handle a streaming chunk by appending to the streaming widget."""
        if not event.delta.content:
            return

        if self._streaming_widget is None:
            # Create the streaming widget with a header
            self._streaming_widget = RichLog(
                auto_scroll=False,
                markup=True,
                wrap=True,
                highlight=False,
                classes="event-richlog streaming-richlog",
            )
            events_container.mount(self._streaming_widget)
            self._streaming_widget.write("[bold]🗣️  Student[/bold]\n")
            self._streaming_buffer = ""

        self._streaming_buffer += event.delta.content
        # Clear and re-render the full buffer as plain text.
        # RichLog.clear() + write() is the simplest way to update in place.
        self._streaming_widget.clear()
        from rich.text import Text

        self._streaming_widget.write("[bold]🗣️  Student[/bold]\n")
        self._streaming_widget.write(Text(self._streaming_buffer))
        events_container.scroll_to_end_if_auto()

    def _handle_chunk_reset_event(
        self, events_container: AutoScrollingVerticalScroll
    ) -> None:
        """Handle a reset event by clearing the streaming widget."""
        if self._streaming_widget is not None:
            self._streaming_widget.clear()
            self._streaming_widget.write("[bold]🗣️  Student[/bold]\n")
            self._streaming_buffer = ""
            events_container.scroll_to_end_if_auto()

    def _finalize_streaming(self) -> None:
        """Remove the streaming widget and indicator, resetting streaming state."""
        if self._streaming_widget is not None:
            self._streaming_widget.remove()
            self._streaming_widget = None
            self._streaming_buffer = ""

    def _mount_richlog_for_markdown(
        self, formatted_event: str, events_container: AutoScrollingVerticalScroll
    ) -> None:
        """Create a RichLog widget for a fully rendered markdown event."""
        from rich.markdown import Markdown as RichMarkdown

        event_log = RichLog(
            auto_scroll=False,
            markup=True,
            wrap=True,
            highlight=False,
            classes="event-richlog",
        )

        # Mount the widget first
        events_container.mount(event_log)

        # Simple approach: extract only the FIRST header, render rest as markdown
        lines = formatted_event.split("\n")
        first_header = None
        content_start_idx = 0

        # Find first header line
        for i, line in enumerate(lines):
            if line.startswith("#"):
                hash_count = len(line) - len(line.lstrip("#"))
                first_header = line[hash_count:].strip()
                content_start_idx = i + 1
                break

        # Render first header as bold text if found
        if first_header:
            event_log.write(f"[bold]{first_header}[/bold]\n")

        # Render rest as markdown (preserves code blocks)
        content = "\n".join(lines[content_start_idx:])
        if content.strip():
            md = RichMarkdown(
                content,
                justify="left",
                code_theme="monokai",
                inline_code_lexer="text",
                inline_code_theme="monokai",
            )
            event_log.write(md)

        # Auto-scroll to bottom if user hasn't scrolled away
        events_container.scroll_to_end_if_auto()

    def clear_transcript(self) -> None:
        """Clear the console content."""
        json_area = self.query_one("#transcript-json", TextArea)
        json_container = self.query_one(
            "#transcript-json-container", AutoScrollingVerticalScroll
        )
        logs_area = self.query_one("#transcript-logs", TextArea)
        logs_container = self.query_one(
            "#transcript-logs-container", AutoScrollingVerticalScroll
        )
        events_container = self.query_one(
            "#transcript-events-container", AutoScrollingVerticalScroll
        )

        json_area.text = "Waiting for connection..."
        json_container.reset_auto_scroll()
        logs_area.text = "Waiting for connection..."
        logs_container.reset_auto_scroll()
        # Remove all event RichLogs and add initial message back
        events_container.remove_children()
        events_container.mount(
            Static("Waiting for connection...", classes="initial-message")
        )
        events_container.reset_auto_scroll()
        self._has_events = False
        # Reset streaming state (widgets already removed by remove_children)
        self._streaming_widget = None
        self._streaming_buffer = ""

    def convert_event_to_markdown(self, event: Event) -> str:
        """Convert an event to markdown format using existing conversion functions."""
        markdown = ""
        match event:
            case TaskStartedEvent():
                markdown = convert_task_started_event_to_markdown(event)
            case MessageAddedEvent():
                markdown = convert_message_added_event_to_markdown(event)
            case ToolCallStartedEvent():
                markdown = convert_tool_call_started_event_to_markdown(event)
            case ToolCallCompletedEvent():
                markdown = convert_tool_call_completed_event_to_markdown(event)
            case ScoringEvent():
                markdown = convert_scoring_event_to_markdown(event)
            case ErrorEvent():
                markdown = convert_error_event_to_markdown(event)
            case (
                TaskCompletedEvent()
                | MessageChunkEvent()
                | MessageChunkResetEvent()
                | MetadataEvent()
                | StepCompletedEvent()
                | StepStartedEvent()
                | AnswersSubmittedEvent()
                | TokenUsageEvent()
            ):
                return ""  # Skip these events

        # Ensure proper spacing for markdown lists by adding blank line before and after
        if markdown:
            markdown = "\n" + markdown + "\n"
        return markdown
