from typing import Any

from mcp.types import CallToolResult, ContentBlock, ImageContent, TextContent

from karotte.schemas.chat import Message
from karotte.schemas.transcript import MessageAddedEvent, ToolCallCompletedEvent


def tool_result_events(
    tool_call_id: str, result: CallToolResult
) -> tuple[ToolCallCompletedEvent, MessageAddedEvent]:
    """A tool result and the tool message the model reads; an image's bytes are kept only in the message."""
    if not result.content:
        content: list[dict[str, Any]] = []
    else:
        first = result.content[0]
        content = [_to_chat_content_part(first)]
        if isinstance(first, ImageContent):
            result = result.model_copy(
                update={
                    "content": [
                        first.model_copy(update={"data": ""}),
                        *result.content[1:],
                    ]
                }
            )
    return (
        ToolCallCompletedEvent(tool_call_id=tool_call_id, result=result),
        MessageAddedEvent(
            message=Message(role="tool", content=content, tool_call_id=tool_call_id)
        ),
    )


def _to_chat_content_part(block: ContentBlock) -> dict[str, Any]:
    """Project an MCP content block onto a chat-completions content part.

    MCP blocks carry protocol-level fields (`annotations`, `_meta`) that are
    addressed to the host, not the model. Strict OpenAI-compatible endpoints
    (e.g. Fireworks) forbid unknown keys in content parts and reject the whole
    request, so this whitelists the fields a provider understands rather than
    dumping the model.
    """
    if isinstance(block, ImageContent):
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{block.mimeType};base64,{block.data}"},
        }

    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text}

    # Audio, resource links and embedded resources have no chat equivalent;
    # hand the model the block's own JSON instead of dropping it.
    return {
        "type": "text",
        "text": block.model_dump_json(by_alias=True, exclude_none=True),
    }
