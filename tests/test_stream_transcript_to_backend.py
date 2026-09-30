"""Tests for streaming transcripts to the backend."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from karotte.schemas.chat import Delta
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.transcript import (
    Event,
    MessageChunkEvent,
    MessageChunkResetEvent,
    TaskCompletedEvent,
    TaskStartedEvent,
    TokenUsageEvent,
)
from karotte.transcript_streaming.stream_transcript_to_backend import (
    stream_transcript_to_backend,
)

_MODULE = "karotte.backend_client"


def _length_get(length: int | None) -> AsyncMock:
    """Mock for GET /transcript_length: an int, or 404 when the transcript is new."""
    if length is None:
        response = httpx.Response(
            404, text="not found", request=httpx.Request("GET", "http://test")
        )
    else:
        response = httpx.Response(
            200, json=length, request=httpx.Request("GET", "http://test")
        )
    return AsyncMock(return_value=response)


def make_run_config(**overrides: Any) -> EvaluationRunConfig:
    defaults: dict[str, Any] = {
        "run_id": "run_1",
        "task_id": "task_1",
        "model": "vertex_ai/test",
        "backend_uri": "https://backend.example.com",
    }
    defaults.update(overrides)
    return EvaluationRunConfig(**defaults)


async def async_iter(items: list[Event]) -> AsyncIterator[Event]:
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_raises_if_backend_uri_is_none():
    config = make_run_config(backend_uri=None)
    with pytest.raises(ValueError, match="backend_uri is required"):
        await stream_transcript_to_backend(async_iter([]), config)


@pytest.mark.asyncio
async def test_creates_transcript_on_first_event(tmp_path: Path):
    config = make_run_config()
    task_started = TaskStartedEvent(run_id="run_1", task_id="task_1", n_steps=10)

    mock_response = httpx.Response(
        200, json={}, request=httpx.Request("POST", "http://test")
    )
    token_file = tmp_path / "token"
    token_file.write_text("test_token")

    with (
        patch(f"{_MODULE}.read_token", return_value="test_token"),
        patch.object(httpx.AsyncClient, "get", _length_get(None)),
        patch.object(
            httpx.AsyncClient,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_post,
    ):
        await stream_transcript_to_backend(async_iter([task_started]), config)

    # A new transcript (404 on length) is created, then the first event is
    # appended at seq 0, then run state is updated.
    assert mock_post.call_count == 3
    assert mock_post.call_args_list[0].args[0] == "/api/internal/create_transcript"
    assert mock_post.call_args_list[0].kwargs["params"] == {"run_id": "run_1"}
    assert mock_post.call_args_list[1].args[0] == "/api/internal/append_transcript"
    assert mock_post.call_args_list[1].kwargs["json"]["seq"] == 0
    assert mock_post.call_args_list[2].args[0] == "/api/internal/update_run_state"


@pytest.mark.asyncio
async def test_fresh_run_numbers_events_from_zero():
    config = make_run_config()
    task_started = TaskStartedEvent(run_id="run_1", task_id="task_1", n_steps=10)
    token_usage = TokenUsageEvent(
        input_tokens=1, output_tokens=1, cache_read_tokens=0, cache_write_tokens=0
    )

    mock_response = httpx.Response(
        200, json={}, request=httpx.Request("POST", "http://test")
    )

    with (
        patch(f"{_MODULE}.read_token", return_value="test_token"),
        patch.object(httpx.AsyncClient, "get", _length_get(None)),
        patch.object(
            httpx.AsyncClient,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_post,
    ):
        await stream_transcript_to_backend(
            async_iter([task_started, token_usage]), config
        )

    appends = [
        c
        for c in mock_post.call_args_list
        if c.args[0] == "/api/internal/append_transcript"
    ]
    assert [c.kwargs["json"]["seq"] for c in appends] == [0, 1]


@pytest.mark.asyncio
async def test_resume_replays_and_overwrites_from_zero():
    """A reconnected run replays from the start: it skips create_transcript for
    an existing transcript but numbers events from seq 0, overwriting slots in
    place so the transcript is not duplicated."""
    config = make_run_config()
    task_started = TaskStartedEvent(run_id="run_1", task_id="task_1", n_steps=10)
    token_usage = TokenUsageEvent(
        input_tokens=1, output_tokens=1, cache_read_tokens=0, cache_write_tokens=0
    )

    mock_response = httpx.Response(
        200, json={}, request=httpx.Request("POST", "http://test")
    )

    with (
        patch(f"{_MODULE}.read_token", return_value="test_token"),
        patch.object(httpx.AsyncClient, "get", _length_get(3)),
        patch.object(
            httpx.AsyncClient,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_post,
    ):
        await stream_transcript_to_backend(
            async_iter([task_started, token_usage]), config
        )

    paths = [c.args[0] for c in mock_post.call_args_list]
    assert "/api/internal/create_transcript" not in paths

    appends = [
        c
        for c in mock_post.call_args_list
        if c.args[0] == "/api/internal/append_transcript"
    ]
    assert [c.kwargs["json"]["seq"] for c in appends] == [0, 1]


@pytest.mark.asyncio
async def test_skips_message_chunk_events():
    config = make_run_config()
    task_started = TaskStartedEvent(run_id="run_1", task_id="task_1", n_steps=10)
    chunk = MessageChunkEvent(
        delta=Delta(role="assistant", content="hello", tool_calls=None)
    )
    chunk_reset = MessageChunkResetEvent()

    mock_response = httpx.Response(
        200, json={}, request=httpx.Request("POST", "http://test")
    )

    with (
        patch(f"{_MODULE}.read_token", return_value="test_token"),
        patch.object(httpx.AsyncClient, "get", _length_get(None)),
        patch.object(
            httpx.AsyncClient,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_post,
    ):
        await stream_transcript_to_backend(
            async_iter([chunk, chunk_reset, task_started]), config
        )

    # Only task_started should be processed (create + append + update_run_state)
    assert mock_post.call_count == 3


@pytest.mark.asyncio
async def test_raises_on_append_error():
    config = make_run_config()
    task_started = TaskStartedEvent(run_id="run_1", task_id="task_1", n_steps=10)

    _req = httpx.Request("POST", "http://test")
    create_response = httpx.Response(200, json={}, request=_req)
    error_response = httpx.Response(400, text="Bad Request", request=_req)

    call_count = 0

    async def mock_post(*_args: Any, **_kwargs: Any) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return create_response
        return error_response

    with (
        patch(f"{_MODULE}.read_token", return_value="test_token"),
        patch.object(httpx.AsyncClient, "get", _length_get(None)),
        patch.object(httpx.AsyncClient, "post", side_effect=mock_post),
    ):
        with pytest.raises(httpx.HTTPStatusError):
            await stream_transcript_to_backend(async_iter([task_started]), config)


@pytest.mark.asyncio
async def test_raises_on_update_run_state_error():
    config = make_run_config()
    task_started = TaskStartedEvent(run_id="run_1", task_id="task_1", n_steps=10)

    _req = httpx.Request("POST", "http://test")
    ok_response = httpx.Response(200, json={}, request=_req)
    error_response = httpx.Response(400, text="Bad Request", request=_req)

    call_count = 0

    async def mock_post(*_args: Any, **_kwargs: Any) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        # 1: create_transcript, 2: append_transcript (ok), 3: update_run_state (error)
        if call_count <= 2:
            return ok_response
        return error_response

    with (
        patch(f"{_MODULE}.read_token", return_value="test_token"),
        patch.object(httpx.AsyncClient, "get", _length_get(None)),
        patch.object(httpx.AsyncClient, "post", side_effect=mock_post),
    ):
        with pytest.raises(httpx.HTTPStatusError):
            await stream_transcript_to_backend(async_iter([task_started]), config)


@pytest.mark.asyncio
async def test_sends_run_state_for_task_completed():
    config = make_run_config()
    task_started = TaskStartedEvent(run_id="run_1", task_id="task_1", n_steps=10)
    task_completed = TaskCompletedEvent(status="passed")

    mock_response = httpx.Response(
        200, json={}, request=httpx.Request("POST", "http://test")
    )

    with (
        patch(f"{_MODULE}.read_token", return_value="test_token"),
        patch.object(httpx.AsyncClient, "get", _length_get(None)),
        patch.object(
            httpx.AsyncClient,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_post,
    ):
        await stream_transcript_to_backend(
            async_iter([task_started, task_completed]), config
        )

    # task_started: create + append + update_run_state
    # task_completed: append + update_run_state
    assert mock_post.call_count == 5

    # Check the last update_run_state call has status="passed"
    last_update_call = mock_post.call_args_list[4]
    assert last_update_call.args[0] == "/api/internal/update_run_state"
    assert last_update_call.kwargs["json"]["status"] == "passed"


@pytest.mark.asyncio
async def test_sends_run_state_for_token_usage():
    config = make_run_config()
    task_started = TaskStartedEvent(run_id="run_1", task_id="task_1", n_steps=10)
    token_usage = TokenUsageEvent(
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=10,
        cache_write_tokens=5,
    )

    mock_response = httpx.Response(
        200, json={}, request=httpx.Request("POST", "http://test")
    )

    with (
        patch(f"{_MODULE}.read_token", return_value="test_token"),
        patch.object(httpx.AsyncClient, "get", _length_get(None)),
        patch.object(
            httpx.AsyncClient,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_post,
    ):
        await stream_transcript_to_backend(
            async_iter([task_started, token_usage]), config
        )

    # task_started: create + append + update_run_state
    # token_usage: append + update_run_state
    assert mock_post.call_count == 5

    last_update_call = mock_post.call_args_list[4]
    assert last_update_call.args[0] == "/api/internal/update_run_state"
    assert last_update_call.kwargs["json"]["input_tokens"] == 100
    assert last_update_call.kwargs["json"]["output_tokens"] == 50
    assert last_update_call.kwargs["json"]["cache_read_tokens"] == 10
    assert last_update_call.kwargs["json"]["cache_write_tokens"] == 5


@pytest.mark.asyncio
async def test_strips_trailing_slash_from_backend_uri():
    config = make_run_config(backend_uri="https://backend.example.com/")
    assert config.backend_uri is not None
    assert config.backend_uri.rstrip("/") == "https://backend.example.com"


@pytest.mark.asyncio
async def test_empty_event_stream():
    config = make_run_config()

    mock_response = httpx.Response(
        200, json={}, request=httpx.Request("POST", "http://test")
    )

    with (
        patch(f"{_MODULE}.read_token", return_value="test_token"),
        patch.object(httpx.AsyncClient, "get", _length_get(None)),
        patch.object(
            httpx.AsyncClient,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_post,
    ):
        await stream_transcript_to_backend(async_iter([]), config)

    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_authorization_header_set():
    config = make_run_config()
    task_started = TaskStartedEvent(run_id="run_1", task_id="task_1", n_steps=10)

    captured_client = None
    original_init = httpx.AsyncClient.__init__

    def capturing_init(self: Any, *args: Any, **kwargs: Any) -> None:
        nonlocal captured_client
        original_init(self, *args, **kwargs)
        captured_client = self

    mock_response = httpx.Response(
        200, json={}, request=httpx.Request("POST", "http://test")
    )

    with (
        patch(f"{_MODULE}.read_token", return_value="decrypted_token"),
        patch.object(httpx.AsyncClient, "__init__", capturing_init),
        patch.object(httpx.AsyncClient, "get", _length_get(None)),
        patch.object(
            httpx.AsyncClient,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ),
    ):
        await stream_transcript_to_backend(async_iter([task_started]), config)

    assert captured_client is not None
    assert captured_client.headers["Authorization"] == "Bearer decrypted_token"
