import json
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from fastmcp import Client, FastMCP
from fastmcp.client.transports.memory import FastMCPTransport

from karotte.agents.message_loop import (
    _execute_tool_calls,  # pyright: ignore[reportPrivateUsage]
)
from karotte.schemas.chat import ChatCompletionMessageToolCall, Function
from karotte.schemas.transcript import Event, ToolCallCompletedEvent


@pytest_asyncio.fixture
async def mcp_client() -> AsyncGenerator[Client[FastMCPTransport]]:
    server = FastMCP("test")

    @server.tool
    def echo(text: str) -> str:  # pyright: ignore[reportUnusedFunction]
        return text

    @server.tool
    def ping() -> str:  # pyright: ignore[reportUnusedFunction]
        return "pong"

    async with Client(FastMCPTransport(server)) as client:
        yield client


async def _completed(
    tool_call: ChatCompletionMessageToolCall, client: Client[FastMCPTransport]
) -> ToolCallCompletedEvent:
    events: list[Event] = [
        event
        async for event in _execute_tool_calls(
            [tool_call],
            client,  # pyright: ignore[reportArgumentType]
        )
    ]
    return next(e for e in events if isinstance(e, ToolCallCompletedEvent))


def _call(name: str, arguments: str) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id="call_1", type="function", function=Function(name=name, arguments=arguments)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    ["[]", "1", '"x"', json.dumps(json.dumps({"text": "hi"}))],
    ids=["list", "number", "string", "double_encoded_object"],
)
async def test_non_object_arguments_give_an_error_result(
    mcp_client: Client[FastMCPTransport], arguments: str
):
    tool_call = _call("echo", arguments)

    completed = await _completed(tool_call, mcp_client)

    assert completed.result.isError
    assert isinstance(json.loads(tool_call.function.arguments), dict)


@pytest.mark.asyncio
async def test_invalid_json_gives_an_error_result(
    mcp_client: Client[FastMCPTransport],
):
    tool_call = _call("echo", '{"text": "hi"')

    completed = await _completed(tool_call, mcp_client)

    assert completed.result.isError
    assert isinstance(json.loads(tool_call.function.arguments), dict)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments", "text"),
    [("echo", '{"text": "hi"}', "hi"), ("ping", "null", "pong"), ("ping", "", "pong")],
    ids=["object", "null", "empty"],
)
async def test_valid_arguments_still_call_the_tool(
    mcp_client: Client[FastMCPTransport], name: str, arguments: str, text: str
):
    completed = await _completed(_call(name, arguments), mcp_client)

    assert not completed.result.isError
    assert completed.result.content[0].text == text  # pyright: ignore[reportAttributeAccessIssue]
