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


def test_mixed_image_text_image_preserves_all_model_content_and_redacts_all_images():
    image = ImageContent(type="image", data="Ym9ndXM=", mimeType="image/png")
    text = TextContent(type="text", text='{"receipt_id":"opaque-receipt"}')
    result = CallToolResult(content=[image, text, image], isError=False)
    completed, message = tool_result_events("call_mixed", result)
    assert message.message.content == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,Ym9ndXM="}},
        {"type": "text", "text": text.text},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,Ym9ndXM="}},
    ]
    assert [
        block.data
        for block in completed.result.content
        if isinstance(block, ImageContent)
    ] == ["", ""]
    assert result.content == [image, text, image]
    assert image.data == "Ym9ndXM="


def test_text_followed_by_image_preserves_order():
    result = CallToolResult(
        content=[
            TextContent(type="text", text="receipt"),
            ImageContent(type="image", data="YQ==", mimeType="image/png"),
        ]
    )
    completed, message = tool_result_events("call_text_first", result)
    assert message.message.content == [
        {"type": "text", "text": "receipt"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ=="}},
    ]
    assert isinstance(completed.result.content[1], ImageContent)
    assert completed.result.content[1].data == ""
