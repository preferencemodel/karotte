from mcp.types import CallToolResult, ImageContent, TextContent

from karotte.agents.tool_results import tool_result_events


def test_image_bytes_are_kept_only_in_the_tool_message():
    image = ImageContent(type="image", data="Ym9ndXM=", mimeType="image/png")
    result = CallToolResult(content=[image], isError=False)

    completed, message = tool_result_events("call_1", result)

    assert completed.tool_call_id == "call_1"
    assert completed.result.content == [
        ImageContent(type="image", data="", mimeType="image/png")
    ]
    assert message.message.role == "tool"
    assert message.message.tool_call_id == "call_1"
    assert message.message.content == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,Ym9ndXM="}}
    ]


def test_caller_result_is_not_modified():
    image = ImageContent(type="image", data="Ym9ndXM=", mimeType="image/png")
    result = CallToolResult(content=[image], isError=False)

    _ = tool_result_events("call_1", result)

    assert image.data == "Ym9ndXM="


def test_text_result_is_kept_in_both():
    result = CallToolResult(
        content=[TextContent(type="text", text="hi")],
        structuredContent={"stdout": "hi"},
        isError=True,
    )

    completed, message = tool_result_events("call_1", result)

    assert completed.result == result
    assert message.message.content == [{"type": "text", "text": "hi"}]


def test_empty_result():
    result = CallToolResult(content=[], isError=False)

    completed, message = tool_result_events("call_1", result)

    assert completed.result == result
    assert message.message.content == []
