"""Tests for keeping the provider's raw chunks on blocked turns.

The chunks go through litellm's real parser, and whole turns through karotte's
builtin source. A litellm upgrade that moves the parser fails here.
"""

import inspect
import json
import re
from typing import Any
from unittest.mock import patch

import httpx
import litellm
import pytest
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    ModelResponseIterator as VertexIterator,
)
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

from karotte.agents import provider_chunks
from karotte.schemas.transcript import Event, MessageAddedEvent
from tests.test_evaluation_runner import (
    _collect,  # pyright: ignore[reportPrivateUsage]
    _create_runner_with_tools_and_transcript,  # pyright: ignore[reportPrivateUsage]
)

provider_chunks.install_provider_chunk_taps()

# Last chunk of a Vertex reply blocked for recitation.
VERTEX_RECITATION = {
    "candidates": [
        {
            "content": {"role": "model", "parts": [{"text": ""}]},
            "finishReason": "RECITATION",
            "finishMessage": "The response was blocked for unauthorized citations.",
            "citationMetadata": {
                "citations": [
                    {"startIndex": 0, "endIndex": 412, "uri": "https://example.com"}
                ]
            },
            "index": 0,
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 10,
        "candidatesTokenCount": 0,
        "totalTokenCount": 10,
    },
    "responseId": "r1",
}

# Vertex blocking the prompt itself: a single chunk with no candidates.
VERTEX_PROMPT_BLOCK = {
    "promptFeedback": {
        "blockReason": "SAFETY",
        "blockReasonMessage": "The prompt is blocked due to safety",
    },
    "usageMetadata": {"promptTokenCount": 10, "totalTokenCount": 10},
    "responseId": "r2",
}

VERTEX_TEXT = {
    "candidates": [
        {"content": {"role": "model", "parts": [{"text": "hello"}]}, "index": 0}
    ],
    "responseId": "r3",
}

VERTEX_STOP = {
    "candidates": [
        {
            "content": {"role": "model", "parts": [{"text": "done"}]},
            "finishReason": "STOP",
            "index": 0,
        }
    ],
    "responseId": "r4",
}


class _Logging:
    """Stub for what litellm's Vertex iterator reads from its logging object."""

    optional_params: dict[str, Any] = {}
    model_call_details: dict[str, Any] = {}


def _vertex_parse(chunk: dict[str, Any]) -> Any:
    iterator = VertexIterator(
        streaming_response=iter(()),
        sync_stream=True,
        logging_obj=_Logging(),  # pyright: ignore[reportArgumentType]
    )
    return iterator.chunk_parser(chunk)


def test_vertex_reply_block_keeps_the_chunk_as_sent():
    chunks = provider_chunks.tap_provider_chunks()
    _vertex_parse(VERTEX_TEXT)
    parsed = _vertex_parse(VERTEX_RECITATION)

    assert parsed.choices[0].finish_reason == "content_filter"  # all litellm reports
    assert list(chunks) == [VERTEX_TEXT, VERTEX_RECITATION]


def test_vertex_prompt_block_keeps_the_chunk_as_sent():
    chunks = provider_chunks.tap_provider_chunks()
    parsed = _vertex_parse(VERTEX_PROMPT_BLOCK)

    assert parsed.choices[0].finish_reason == "content_filter"
    assert list(chunks) == [VERTEX_PROMPT_BLOCK]


def test_only_the_last_chunks_are_kept():
    chunks = provider_chunks.tap_provider_chunks()
    for _ in range(10):
        _vertex_parse(VERTEX_TEXT)
    _vertex_parse(VERTEX_PROMPT_BLOCK)

    assert len(chunks) == provider_chunks.MAX_KEPT_CHUNKS
    assert chunks[-1] == VERTEX_PROMPT_BLOCK


def test_a_kept_chunk_is_a_copy():
    chunks = provider_chunks.tap_provider_chunks()
    feedback = {"blockReason": "SAFETY"}
    _vertex_parse({**VERTEX_PROMPT_BLOCK, "promptFeedback": feedback})
    feedback["blockReason"] = "changed"

    assert chunks[0]["promptFeedback"]["blockReason"] == "SAFETY"


def test_parsing_without_a_turn_still_works():
    provider_chunks._current_turn_chunks.set(None)  # pyright: ignore[reportPrivateUsage]
    assert _vertex_parse(VERTEX_RECITATION).choices[0].finish_reason == "content_filter"


def test_install_is_idempotent():
    parser = VertexIterator.chunk_parser
    provider_chunks.install_provider_chunk_taps()
    assert VertexIterator.chunk_parser is parser


def test_a_parser_litellm_moved_is_skipped():
    assert not provider_chunks._tap_parser(  # pyright: ignore[reportPrivateUsage]
        "litellm.llms.no_such_provider", "ModelResponseIterator"
    )


# Whole turns through karotte's builtin source and the real litellm, over a fake
# transport.

GEMINI = "vertex_ai/gemini-3.1-pro-preview"


def _sse(*chunks: dict[str, Any]) -> bytes:
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode()


async def _collect_over_transport(body: bytes) -> list[Event]:
    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )

    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    real_acompletion = litellm.acompletion

    async def acompletion(**kwargs: Any) -> Any:
        return await real_acompletion(**kwargs, client=client)

    runner = _create_runner_with_tools_and_transcript(GEMINI)
    with (
        patch("litellm.acompletion", acompletion),
        patch.object(
            VertexBase, "_ensure_access_token_async", return_value=("tok", "proj")
        ),
    ):
        return [event async for event in _collect(runner)]


def _message_added(events: list[Event]) -> MessageAddedEvent:
    (event,) = [e for e in events if isinstance(e, MessageAddedEvent)]
    return event


@pytest.mark.asyncio
async def test_a_blocked_turn_keeps_the_chunks():
    event = _message_added(await _collect_over_transport(_sse(VERTEX_PROMPT_BLOCK)))

    assert event.finish_reason == "content_filter"
    assert event.raw == {"provider_chunks": [VERTEX_PROMPT_BLOCK]}


@pytest.mark.asyncio
async def test_a_delivered_turn_keeps_no_chunks():
    event = _message_added(await _collect_over_transport(_sse(VERTEX_STOP)))

    assert event.finish_reason == "stop"
    assert event.raw is None


def _turn(finish_reason: str) -> list[dict[str, Any]]:
    """A turn with thinking, text and a tool call, ending in `finish_reason`."""

    def parts(*p: dict[str, Any]) -> dict[str, Any]:
        return {"candidates": [{"content": {"role": "model", "parts": list(p)}}]}

    return [
        parts({"text": "Let me check.", "thought": True}),
        parts({"text": 'Running it: é 😀 {"x": 1}'}),
        parts(
            {
                "functionCall": {"name": "bash", "args": {"command": "ls"}},
                "thoughtSignature": "c2lnbmF0dXJl",
            }
        ),
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": ""}]},
                    "finishReason": finish_reason,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        },
    ]


def _without_raw(events: list[Event]) -> str:
    """The events as JSON, without timestamps, `raw` or litellm's random tool call ids."""
    dumped = [e.model_dump(mode="json", exclude={"timestamp", "raw"}) for e in events]
    return re.sub(r"call_[0-9a-f]+", "call_ID", json.dumps(dumped))


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["STOP", "SAFETY"])
async def test_the_tap_changes_nothing_else(
    monkeypatch: pytest.MonkeyPatch, finish_reason: str
):
    body = _sse(*_turn(finish_reason))
    tapped = await _collect_over_transport(body)
    monkeypatch.setattr(
        VertexIterator, "chunk_parser", inspect.unwrap(VertexIterator.chunk_parser)
    )
    untapped = await _collect_over_transport(body)

    assert _without_raw(tapped) == _without_raw(untapped)
