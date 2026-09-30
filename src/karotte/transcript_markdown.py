"""Markdown rendering of transcript events."""

import json
from textwrap import dedent
from typing import Any

from karotte.schemas.transcript import (
    ErrorEvent,
    MessageAddedEvent,
    ResourceMetrics,
    ScoringEvent,
    TaskStartedEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)


def add_code_block_to_markdown(content: Any, language: str = "") -> str:
    markdown = f"```{language}\n"

    if isinstance(content, str):
        markdown += "\n".join(content.splitlines())
        if content:
            markdown += "\n"
    elif isinstance(content, dict | list):
        markdown += _format_json(content) + "\n"
    else:
        markdown += f"{content}\n"

    markdown += "```\n\n"

    return markdown


def _format_json(content: dict[Any, Any] | list[Any]) -> str:
    try:
        return json.dumps(content, indent=2)
    except (TypeError, ValueError):
        return str(content)


def convert_error_event_to_markdown(event: ErrorEvent) -> str:
    markdown = dedent(f"""
        ### 💥 Error

        **Exception Type:** `{event.exception_type}`

        **Message:**
        """)

    markdown += add_code_block_to_markdown(event.message)

    if event.traceback:
        markdown += "\n**Traceback:**\n"
        markdown += add_code_block_to_markdown(event.traceback)

    return markdown


def convert_message_added_event_to_markdown(event: MessageAddedEvent):
    markdown = ""

    match event.message.role:
        case "user" | "system":
            markdown += "#### 📋 Instructions\n\n"
        case "assistant":
            if event.message.content:
                markdown += "#### 🗣️  Student\n\n"
        case "tool":
            return ""
        case _:
            markdown += f"#### {event.message.role.capitalize()}\n\n"

    content = event.message.content
    if not content:
        return markdown
    if isinstance(content, str):
        for line in content.split("\n"):
            markdown += f"{line}\n"
    else:
        for part in content:
            markdown += _content_part_to_markdown(part)

    return markdown


def _content_part_to_markdown(part: dict[str, Any]) -> str:
    match part.get("type"):
        case "text":
            return f"{part.get('text', '')}\n\n"
        case "image_url":
            url = str((part.get("image_url") or {}).get("url", ""))
            mime_type = (
                url.removeprefix("data:").split(";", 1)[0]
                if url.startswith("data:")
                else "unknown"
            )
            return f"[Image: {mime_type}]\n\n"
        case other:
            return f"[{other}]\n\n"


def convert_scoring_event_to_markdown(event: ScoringEvent) -> str:
    scoring = event.scoring
    icon = "✔️" if scoring.continue_task else "❌"
    markdown = f"#### {icon} Scoring"

    markdown += f"\nScore: {scoring.score}\n\n"
    markdown += f"Continue Task: {'Yes' if scoring.continue_task else 'No'}\n\n"

    markdown += _metadata_to_markdown(scoring.metadata)

    if event.resource_metrics and event.resource_metrics.samples:
        m = event.resource_metrics
        markdown += f"\n**Container Resources**: CPU {m.avg_cpu_percent:.1f}% (peak {m.peak_cpu_percent:.1f}%) | Memory {m.avg_memory_mb:.0f} MB (peak {m.peak_memory_mb:.0f} MB)\n"

    return markdown


def _metadata_to_markdown(metadata: Any) -> str:
    if not metadata:
        return ""

    # Skip empty dicts
    if isinstance(metadata, dict) and len(metadata) == 0:
        return ""

    result = "**Metadata:**\n\n"

    if isinstance(metadata, dict):
        for key, value in metadata.items():
            result += f"**{key}**\n"
            result += add_code_block_to_markdown(value)
        return result

    result += add_code_block_to_markdown(metadata)
    return result


def convert_task_started_event_to_markdown(event: TaskStartedEvent):
    markdown = dedent(f"""
        #### Run Started

        Run ID: {event.run_id if event.run_id else "N/A"}

        Task: {event.task_id}

        Model: {event.model if event.model else "N/A"}

        Steps: {event.n_steps}
        """)

    return markdown


def convert_tool_call_completed_event_to_markdown(
    event: ToolCallCompletedEvent,
    is_submit_answers_call: bool = False,
    tool_name: str | None = None,
) -> str:
    if is_submit_answers_call and not event.result.isError:
        return ""

    if event.result.isError:
        markdown = "##### ❌ Error\n\n"
    else:
        markdown = "##### ✅ Result\n\n"

    from mcp.types import ImageContent, TextContent

    if event.result.structuredContent:
        for key, value in event.result.structuredContent.items():
            # Filter out empty stdout for bash tool
            if tool_name == "bash" and key == "stdout" and value == "":
                continue
            # Filter out resource metrics since we display it separately
            if key == "_karotte_resource_metrics":
                continue
            markdown += f"**{key}**\n"
            markdown += add_code_block_to_markdown(value)
    else:
        for item in event.result.content:
            if isinstance(item, ImageContent):
                mime_type = item.mimeType or "unknown"
                markdown += f"[Image: {mime_type}]\n\n"
            elif isinstance(item, TextContent):
                markdown += add_code_block_to_markdown(item.text)
            else:
                markdown += f"[{type(item).__name__}]\n\n"

    # Display resource metrics if present in structuredContent
    if (
        event.result.structuredContent
        and "_karotte_resource_metrics" in event.result.structuredContent
    ):
        metrics_obj = event.result.structuredContent["_karotte_resource_metrics"]
        if isinstance(metrics_obj, dict) and metrics_obj.get("samples"):
            m = ResourceMetrics.model_validate(metrics_obj)
            markdown += f"\n**Container Resources**: CPU {m.avg_cpu_percent:.1f}% (peak {m.peak_cpu_percent:.1f}%) | Memory {m.avg_memory_mb:.0f} MB (peak {m.peak_memory_mb:.0f} MB)\n"

    return markdown


def convert_tool_call_started_event_to_markdown(event: ToolCallStartedEvent):
    if event.tool_call.function.name == "submit_answers":
        return _convert_submit_answers_call_to_markdown(event)

    markdown = dedent(f"""
        #### 🔧 {event.tool_call.function.name}

        """)

    if event.tool_call.function.arguments:
        try:
            arguments = json.loads(event.tool_call.function.arguments)
        except json.JSONDecodeError:
            markdown += "**Invalid arguments**\n"
            markdown += add_code_block_to_markdown(event.tool_call.function.arguments)
            return markdown

        language = None

        if "command" in arguments and arguments["command"].startswith("python -c"):
            language = "python"

        for key, value in arguments.items():
            markdown += f"**{key}**\n"
            markdown += add_code_block_to_markdown(str(value), language or "")

    return markdown


def _convert_submit_answers_call_to_markdown(event: ToolCallStartedEvent):
    markdown = dedent("""
        #### 📨 Submitting Answers

        """)

    arguments = json.loads(event.tool_call.function.arguments or "{}")

    if not isinstance(arguments, dict):
        markdown += "**❗ Invalid answer format**\n"
        markdown += add_code_block_to_markdown(str(arguments))
        return markdown

    if len(arguments) == 1 and "answers" in arguments:
        answers = arguments["answers"]
    else:
        answers = arguments

    if not isinstance(answers, dict):
        markdown += "**❗ Invalid answer format**\n"
        markdown += add_code_block_to_markdown(str(answers))
        return markdown

    if answers:
        for key, value in answers.items():
            markdown += f"**{key}**\n"
            markdown += add_code_block_to_markdown(str(value))

    return markdown
