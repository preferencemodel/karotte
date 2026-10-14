"""Opaque reasoning state must reach the next request on providers that take it back."""

import json
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, final, override

import litellm
import pytest
from litellm import CustomStreamWrapper
from litellm.types.utils import Delta as LiteLlmDelta
from litellm.types.utils import ModelResponseStream, StreamingChoices, Usage

from karotte.agents.builtin_source import BuiltinSource
from karotte.schemas.chat import Message
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.transcript import MessageAddedEvent

_SIGNATURE = "EqQBCkgIARABGAIiQL+opaque/signature=="
_SUMMARY = "Let me list the files first."
_SECOND_SIGNATURE = "EqQBsecond/signature=="
_SECOND_SUMMARY = "Then read them."


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


_THINK_THEN_CALL_BASH = b"".join(
    [
        _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5-5",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            },
        ),
        _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            },
        ),
        _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": _SUMMARY},
            },
        ),
        _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": _SIGNATURE},
            },
        ),
        _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            },
        ),
        _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "thinking_delta", "thinking": _SECOND_SUMMARY},
            },
        ),
        _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "signature_delta", "signature": _SECOND_SIGNATURE},
            },
        ),
        _sse("content_block_stop", {"type": "content_block_stop", "index": 1}),
        _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "bash",
                    "input": {},
                },
            },
        ),
        _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "input_json_delta", "partial_json": '{"cmd": "ls"}'},
            },
        ),
        _sse("content_block_stop", {"type": "content_block_stop", "index": 2}),
        _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 20},
            },
        ),
        _sse("message_stop", {"type": "message_stop"}),
    ]
)


type Serve = Callable[[bytes], list[dict[str, Any]]]


@pytest.fixture
def fake_provider(monkeypatch: pytest.MonkeyPatch) -> Iterator[Serve]:
    """Answers every call with the given SSE stream; returns the request bodies."""
    servers: list[HTTPServer] = []

    def serve(stream: bytes) -> list[dict[str, Any]]:
        bodies: list[dict[str, Any]] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("content-length", 0))
                bodies.append(json.loads(self.rfile.read(n)))
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                self.wfile.write(stream)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        servers.append(server)
        threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        ).start()

        acompletion = litellm.acompletion

        async def to_server(**params: Any) -> Any:
            return await acompletion(
                **params, api_base=f"http://127.0.0.1:{server.server_port}"
            )

        monkeypatch.setattr("litellm.acompletion", to_server)
        return bodies

    yield serve
    for server in servers:
        server.shutdown()


def _config(model: str, effort: str | None = None) -> EvaluationRunConfig:
    return EvaluationRunConfig(
        run_id="run",
        task_id="task",
        model=model,
        model_api_key="test_key",
        reasoning_effort=effort,
    )


_READ_FILE: Any = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
}


async def _turn(source: BuiltinSource, messages: list[Message]) -> Message:
    added = [
        event
        async for event in source.collect(messages, [_READ_FILE])
        if isinstance(event, MessageAddedEvent)
    ]
    assert len(added) == 1
    return added[0].message


@pytest.mark.asyncio
async def test_claude_thinking_is_sent_back_with_its_signature(fake_provider: Serve):
    bodies = fake_provider(_THINK_THEN_CALL_BASH)
    source = BuiltinSource(_config("claude-sonnet-5-5"))
    user = Message(role="user", content="List the files.")

    assistant = await _turn(source, [user])
    assert assistant.reasoning_content == _SUMMARY + _SECOND_SUMMARY
    assert assistant.tool_calls is not None

    stored = MessageAddedEvent.model_validate_json(
        MessageAddedEvent(message=assistant).model_dump_json()
    ).message
    tool_result = Message(role="tool", content="a.txt", tool_call_id="toolu_1")
    await _turn(source, [user, stored, tool_result])

    replayed = bodies[1]["messages"][1]
    assert replayed["role"] == "assistant"
    assert replayed["content"][:2] == [
        {"type": "thinking", "thinking": _SUMMARY, "signature": _SIGNATURE},
        {
            "type": "thinking",
            "thinking": _SECOND_SUMMARY,
            "signature": _SECOND_SIGNATURE,
        },
    ]
    assert replayed["content"][-1]["type"] == "tool_use"


_ENCRYPTED = "gAAAAABopaque-encrypted-reasoning=="
_OPENAI_SUMMARY = "I should read the file."
_REASONING_ITEM = {
    "id": "rs_1",
    "type": "reasoning",
    "summary": [{"type": "summary_text", "text": _OPENAI_SUMMARY}],
    "encrypted_content": _ENCRYPTED,
}
_FUNCTION_CALL_ITEM = {
    "id": "fc_1",
    "type": "function_call",
    "status": "completed",
    "call_id": "call_1",
    "name": "read_file",
    "arguments": '{"path": "a.txt"}',
}
_RESPONSE = {
    "id": "resp_1",
    "object": "response",
    "created_at": 1735689600,
    "model": "gpt-5.6-sol",
}


def _responses_event(data: dict[str, Any]) -> bytes:
    return _sse(data["type"], data)


_REASON_THEN_CALL_READ_FILE = b"".join(
    [
        _responses_event(
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {**_RESPONSE, "status": "in_progress", "output": []},
            }
        ),
        _responses_event(
            {
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": {"id": "rs_1", "type": "reasoning", "summary": []},
            }
        ),
        _responses_event(
            {
                "type": "response.reasoning_summary_text.delta",
                "sequence_number": 2,
                "item_id": "rs_1",
                "output_index": 0,
                "summary_index": 0,
                "delta": _OPENAI_SUMMARY,
            }
        ),
        _responses_event(
            {
                "type": "response.output_item.done",
                "sequence_number": 3,
                "output_index": 0,
                "item": _REASONING_ITEM,
            }
        ),
        _responses_event(
            {
                "type": "response.output_item.added",
                "sequence_number": 4,
                "output_index": 1,
                "item": {
                    **_FUNCTION_CALL_ITEM,
                    "status": "in_progress",
                    "arguments": "",
                },
            }
        ),
        _responses_event(
            {
                "type": "response.function_call_arguments.delta",
                "sequence_number": 5,
                "item_id": "fc_1",
                "output_index": 1,
                "delta": _FUNCTION_CALL_ITEM["arguments"],
            }
        ),
        _responses_event(
            {
                "type": "response.output_item.done",
                "sequence_number": 6,
                "output_index": 1,
                "item": _FUNCTION_CALL_ITEM,
            }
        ),
        _responses_event(
            {
                "type": "response.completed",
                "sequence_number": 7,
                "response": {
                    **_RESPONSE,
                    "status": "completed",
                    "output": [_REASONING_ITEM, _FUNCTION_CALL_ITEM],
                    "usage": {
                        "input_tokens": 10,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 20,
                        "output_tokens_details": {"reasoning_tokens": 15},
                        "total_tokens": 30,
                    },
                },
            }
        ),
    ]
)


@pytest.mark.parametrize(
    "model",
    ["openai/gpt-5.6-sol", "meta/muse-spark-1.3", "xai/grok-4.7", "xai/grok-4.6"],
)
@pytest.mark.asyncio
async def test_responses_reasoning_is_sent_back_with_its_encrypted_content(
    model: str, fake_provider: Serve
):
    bodies = fake_provider(_REASON_THEN_CALL_READ_FILE)
    source = BuiltinSource(_config(model, "high"))
    user = Message(role="user", content="Read a.txt.")

    assistant = await _turn(source, [user])
    assert assistant.reasoning_content == _OPENAI_SUMMARY
    assert assistant.tool_calls is not None

    stored = MessageAddedEvent.model_validate_json(
        MessageAddedEvent(message=assistant).model_dump_json()
    ).message
    tool_result = Message(
        role="tool", content="hello", tool_call_id=assistant.tool_calls[0].id
    )
    await _turn(source, [user, stored, tool_result])

    replayed = [item for item in bodies[1]["input"] if item.get("type") != "message"]
    assert [item["type"] for item in replayed] == [
        "reasoning",
        "function_call",
        "function_call_output",
    ]
    assert replayed[0]["id"] == "rs_1"
    assert replayed[0]["encrypted_content"] == _ENCRYPTED


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-6.1-sol",
        "meta/muse-spark-1.3",
        "xai/grok-4.7",
        "xai/grok-4.6",
        "xai/grok-4.3",
    ],
)
@pytest.mark.asyncio
async def test_reasoning_travels_encrypted_instead_of_stored(
    model: str, fake_provider: Serve
):
    bodies = fake_provider(_REASON_THEN_CALL_READ_FILE)
    source = BuiltinSource(_config(model))

    await _turn(source, [Message(role="user", content="Read a.txt.")])

    assert bodies[0]["include"] == ["reasoning.encrypted_content"]
    assert bodies[0]["store"] is False


@pytest.mark.parametrize("model", ["xai/grok-4.7", "xai/grok-4.3"])
@pytest.mark.asyncio
async def test_grok_system_prompt_is_sent_as_an_input_message(
    model: str, fake_provider: Serve
):
    bodies = fake_provider(_REASON_THEN_CALL_READ_FILE)
    source = BuiltinSource(_config(model))

    await _turn(
        source,
        [
            Message(role="system", content="Be terse."),
            Message(role="user", content="Read a.txt."),
        ],
    )

    assert "instructions" not in bodies[0]
    assert bodies[0]["input"][0] == {
        "type": "message",
        "role": "system",
        "content": [{"type": "input_text", "text": "Be terse."}],
    }


@final
class _ScriptedStream(CustomStreamWrapper):
    def __init__(self, chunks: list[ModelResponseStream]):  # pyright: ignore[reportMissingSuperCall]
        self._chunks = chunks

    @override
    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk


def _chunk(
    delta: LiteLlmDelta, finish_reason: str | None = None
) -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl-1",
        created=1735689600,
        model="m",
        choices=[StreamingChoices(index=0, delta=delta, finish_reason=finish_reason)],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        if finish_reason
        else None,
    )


@pytest.mark.parametrize(
    "model",
    [
        "gemini/gemini-3-pro",
        "vertex_ai/gemini-3.8-flash",
        "together_ai/moonshotai/Kimi-K3",
        "fireworks_ai/accounts/fireworks/models/glm-5p3",
    ],
)
@pytest.mark.asyncio
async def test_other_providers_neither_keep_nor_send_reasoning_state(
    model: str, monkeypatch: pytest.MonkeyPatch
):
    chunks = [
        _chunk(
            LiteLlmDelta(
                role="assistant",
                reasoning_content="thought",
                thinking_blocks=[
                    {"type": "thinking", "thinking": "thought", "signature": "sig"}
                ],
            )
        ),
        _chunk(
            LiteLlmDelta(
                content="done",
                reasoning_items=[{"id": "rs_1", "type": "reasoning", "summary": []}],
            ),
            finish_reason="stop",
        ),
    ]

    async def scripted(**_params: Any) -> _ScriptedStream:
        return _ScriptedStream(chunks)

    monkeypatch.setattr("litellm.acompletion", scripted)
    source = BuiltinSource(_config(model))
    user = Message(role="user", content="Go.")

    assistant = await _turn(source, [user])
    assert assistant.reasoning_content == "thought"
    assert assistant.thinking_blocks is None
    assert assistant.reasoning_items is None

    sent = source.get_completion_params([user, assistant], [])["messages"][1]
    assert "thinking_blocks" not in sent
    assert "reasoning_items" not in sent
