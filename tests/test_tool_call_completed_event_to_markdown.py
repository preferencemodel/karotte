from pathlib import Path

import pytest
from mcp.types import CallToolResult, ImageContent, TextContent

from karotte.schemas.transcript import ToolCallCompletedEvent
from karotte.tools.view_lines_in_file import view_lines_in_file
from karotte.transcript_markdown import (
    convert_tool_call_completed_event_to_markdown,
)


@pytest.mark.asyncio
async def test_does_not_crash_when_used_with_view_lines_in_file_tool(tmp_path: Path):
    test_file = tmp_path / "test_file.txt"

    test_file.write_text("\n".join(str(x) for x in range(1, 10)))

    result = await view_lines_in_file()(test_file, from_line=2, to_line=6)

    convert_tool_call_completed_event_to_markdown(
        ToolCallCompletedEvent(
            tool_call_id="",
            result=CallToolResult(
                content=[], structuredContent=result.structured_content
            ),
        )
    )


def test_image_content_shows_mime_type():
    event = ToolCallCompletedEvent(
        tool_call_id="test",
        result=CallToolResult(
            content=[
                ImageContent(type="image", data="base64data", mimeType="image/png")
            ],
        ),
    )

    markdown = convert_tool_call_completed_event_to_markdown(event)

    assert "[Image: image/png]" in markdown


def test_image_content_with_empty_mime_type_shows_unknown():
    event = ToolCallCompletedEvent(
        tool_call_id="test",
        result=CallToolResult(
            content=[ImageContent(type="image", data="base64data", mimeType="")],
        ),
    )

    markdown = convert_tool_call_completed_event_to_markdown(event)

    assert "[Image: unknown]" in markdown


def test_text_content_shows_actual_text():
    event = ToolCallCompletedEvent(
        tool_call_id="test",
        result=CallToolResult(
            content=[TextContent(type="text", text="hello")],
        ),
    )

    markdown = convert_tool_call_completed_event_to_markdown(event)

    assert "hello" in markdown
    assert "```" in markdown


def test_multiple_content_items():
    event = ToolCallCompletedEvent(
        tool_call_id="test",
        result=CallToolResult(
            content=[
                ImageContent(type="image", data="base64data", mimeType="image/jpeg"),
                TextContent(type="text", text="hello"),
                ImageContent(type="image", data="moredata", mimeType="image/gif"),
            ],
        ),
    )

    markdown = convert_tool_call_completed_event_to_markdown(event)

    assert "[Image: image/jpeg]" in markdown
    assert "hello" in markdown
    assert "[Image: image/gif]" in markdown


def test_structured_content_takes_precedence_over_content():
    """When structuredContent is present, content should be ignored."""
    event = ToolCallCompletedEvent(
        tool_call_id="test",
        result=CallToolResult(
            content=[
                ImageContent(type="image", data="base64data", mimeType="image/png")
            ],
            structuredContent={"key": "value"},
        ),
    )

    markdown = convert_tool_call_completed_event_to_markdown(event)

    assert "[Image:" not in markdown
    assert "**key**" in markdown


def test_success_result_shows_checkmark():
    event = ToolCallCompletedEvent(
        tool_call_id="test",
        result=CallToolResult(
            content=[TextContent(type="text", text="success")],
            isError=False,
        ),
    )

    markdown = convert_tool_call_completed_event_to_markdown(event)

    assert "##### ✅ Result" in markdown


def test_error_result_shows_x():
    event = ToolCallCompletedEvent(
        tool_call_id="test",
        result=CallToolResult(
            content=[TextContent(type="text", text="error message")],
            isError=True,
        ),
    )

    markdown = convert_tool_call_completed_event_to_markdown(event)

    assert "##### ❌ Error" in markdown


def test_bash_tool_filters_empty_stdout():
    event = ToolCallCompletedEvent(
        tool_call_id="test",
        result=CallToolResult(
            content=[],
            structuredContent={"stdout": "", "stderr": "some error"},
        ),
    )

    markdown = convert_tool_call_completed_event_to_markdown(event, tool_name="bash")

    assert "**stdout**" not in markdown
    assert "**stderr**" in markdown
    assert "some error" in markdown


def test_bash_tool_keeps_nonempty_stdout():
    event = ToolCallCompletedEvent(
        tool_call_id="test",
        result=CallToolResult(
            content=[],
            structuredContent={"stdout": "hello world", "stderr": ""},
        ),
    )

    markdown = convert_tool_call_completed_event_to_markdown(event, tool_name="bash")

    assert "**stdout**" in markdown
    assert "hello world" in markdown


def test_karotte_resource_metrics_filtered_from_structured_content():
    """Test that _karotte_resource_metrics is not displayed in structured content."""

    event = ToolCallCompletedEvent(
        tool_call_id="test",
        result=CallToolResult(
            content=[],
            structuredContent={
                "stdout": "output",
                "_karotte_resource_metrics": {
                    "samples": [],
                    "peak_cpu_percent": 50.0,
                    "avg_cpu_percent": 25.0,
                    "peak_memory_mb": 100.0,
                    "avg_memory_mb": 80.0,
                },
            },
        ),
    )

    markdown = convert_tool_call_completed_event_to_markdown(event)

    assert "**stdout**" in markdown
    assert "_karotte_resource_metrics" not in markdown
    assert "peak_cpu_percent" not in markdown


def test_resource_metrics_displayed_when_present():
    """Test that resource metrics summary is displayed when metrics in structuredContent."""
    event = ToolCallCompletedEvent(
        tool_call_id="test",
        result=CallToolResult(
            content=[TextContent(type="text", text="output")],
            structuredContent={
                "output": "output",
                "_karotte_resource_metrics": {
                    "samples": [
                        {"timestamp_ms": 0, "cpu_percent": 20.0, "memory_mb": 100.0},
                        {"timestamp_ms": 100, "cpu_percent": 30.0, "memory_mb": 120.0},
                    ],
                    "peak_cpu_percent": 30.0,
                    "avg_cpu_percent": 25.0,
                    "peak_memory_mb": 120.0,
                    "avg_memory_mb": 110.0,
                },
            },
        ),
    )

    markdown = convert_tool_call_completed_event_to_markdown(event)

    assert "**Container Resources**" in markdown
    assert "CPU 25.0%" in markdown
    assert "peak 30.0%" in markdown
    assert "Memory 110 MB" in markdown
    assert "peak 120 MB" in markdown


def test_resource_metrics_not_displayed_when_no_samples():
    """Test that resource metrics are not displayed when there are no samples."""
    event = ToolCallCompletedEvent(
        tool_call_id="test",
        result=CallToolResult(
            content=[TextContent(type="text", text="output")],
            structuredContent={
                "output": "output",
                "_karotte_resource_metrics": {
                    "samples": [],
                    "peak_cpu_percent": 0.0,
                    "avg_cpu_percent": 0.0,
                    "peak_memory_mb": 0.0,
                    "avg_memory_mb": 0.0,
                },
            },
        ),
    )

    markdown = convert_tool_call_completed_event_to_markdown(event)

    assert "**Container Resources**" not in markdown


def test_resource_metrics_not_displayed_when_not_in_structured_content():
    """Test that resource metrics section is skipped when not in structuredContent."""
    event = ToolCallCompletedEvent(
        tool_call_id="test",
        result=CallToolResult(
            content=[TextContent(type="text", text="output")],
            structuredContent={"output": "output"},
        ),
    )

    markdown = convert_tool_call_completed_event_to_markdown(event)

    assert "**Container Resources**" not in markdown


def test_resource_metrics_not_displayed_when_no_structured_content():
    """Test that resource metrics section is skipped when no structuredContent."""
    event = ToolCallCompletedEvent(
        tool_call_id="test",
        result=CallToolResult(
            content=[TextContent(type="text", text="output")],
        ),
    )

    markdown = convert_tool_call_completed_event_to_markdown(event)

    assert "**Container Resources**" not in markdown
