"""Tests for BackendClient's per-attempt deadline and retry budget."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import anyio
import httpx
import pytest
from loguru import logger
from tenacity import stop_after_attempt

from karotte import backend_client
from karotte.backend_client import BackendClient

_MODULE = "karotte.backend_client"


def _ok_response() -> httpx.Response:
    return httpx.Response(200, json={}, request=httpx.Request("POST", "http://test"))


@pytest.mark.asyncio
async def test_append_transcript_sends_seq_when_provided():
    mock_post = AsyncMock(return_value=_ok_response())
    with (
        patch(f"{_MODULE}.read_token", return_value="test_token"),
        patch.object(httpx.AsyncClient, "post", mock_post),
    ):
        client = BackendClient("http://test")
        await client.append_transcript("run_1", {"type": "x"}, seq=5)
        await client.aclose()

    assert mock_post.call_args.kwargs["json"] == {"event": {"type": "x"}, "seq": 5}


@pytest.mark.asyncio
async def test_append_transcript_omits_seq_when_absent():
    mock_post = AsyncMock(return_value=_ok_response())
    with (
        patch(f"{_MODULE}.read_token", return_value="test_token"),
        patch.object(httpx.AsyncClient, "post", mock_post),
    ):
        client = BackendClient("http://test")
        await client.append_transcript("run_1", {"type": "x"})
        await client.aclose()

    assert mock_post.call_args.kwargs["json"] == {"event": {"type": "x"}}


@pytest.mark.asyncio
async def test_get_transcript_length_returns_count():
    response = httpx.Response(200, json=7, request=httpx.Request("GET", "http://test"))
    with (
        patch(f"{_MODULE}.read_token", return_value="test_token"),
        patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)),
    ):
        client = BackendClient("http://test")
        length = await client.get_transcript_length("run_1")
        await client.aclose()

    assert length == 7


@pytest.mark.asyncio
async def test_get_transcript_length_returns_none_when_missing():
    response = httpx.Response(
        404, text="not found", request=httpx.Request("GET", "http://test")
    )
    with (
        patch(f"{_MODULE}.read_token", return_value="test_token"),
        patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=response)),
    ):
        client = BackendClient("http://test")
        length = await client.get_transcript_length("run_1")
        await client.aclose()

    assert length is None


def test_timeout_error_is_retryable():
    # anyio.fail_after raises a builtin TimeoutError, which must be retried
    # rather than crashing the run.
    assert backend_client._is_retryable(TimeoutError())  # pyright: ignore[reportPrivateUsage]


def test_retry_budget_spans_realistic_outage():
    assert backend_client._RETRY_BUDGET_SECONDS >= 20 * 60  # pyright: ignore[reportPrivateUsage]


def test_backoff_is_capped():
    assert backend_client._RETRY_BACKOFF_MAX_SECONDS <= 30  # pyright: ignore[reportPrivateUsage]


_SAFETY_NET_SECONDS = 10


async def _hang() -> None:
    """A backend that never answers: waits until the attempt is cancelled."""
    await anyio.Event().wait()


@pytest.mark.asyncio
async def test_each_attempt_is_bounded_by_hard_deadline():
    """A request that hangs is cut off per attempt and retried.

    Without a hard per-attempt deadline a single hung request eats the whole
    retry budget. A deadline of 0 cancels each attempt at its first await, and
    the budget is a fixed number of attempts, so no real time is involved.
    """
    call_count = 0

    async def hanging_post(*_args: Any, **_kwargs: Any) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        await _hang()
        return _ok_response()

    with (
        patch(f"{_MODULE}.read_token", return_value="test_token"),
        patch.object(backend_client, "_REQUEST_TIMEOUT_SECONDS", 0),
        patch.object(backend_client, "_RETRY_BACKOFF_MAX_SECONDS", 0),
        patch.object(
            backend_client,
            "stop_after_delay",
            lambda _delay: stop_after_attempt(3),  # pyright: ignore[reportUnknownLambdaType]
        ),
        patch.object(httpx.AsyncClient, "post", side_effect=hanging_post),
    ):
        client = BackendClient("http://test")
        with pytest.raises(TimeoutError):
            # Only reached if the per-attempt deadline is gone: fail, don't hang.
            with anyio.fail_after(_SAFETY_NET_SECONDS):
                await client.append_transcript("run_1", {})
        await client.aclose()

    assert call_count == 3, "each hung attempt is cut off and retried"


@pytest.mark.asyncio
async def test_retries_then_succeeds_after_transient_hang():
    """Attempts that hang recover once the backend responds within the deadline."""
    call_count = 0

    async def flaky_post(*_args: Any, **_kwargs: Any) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            await _hang()
        return _ok_response()

    with (
        patch(f"{_MODULE}.read_token", return_value="test_token"),
        patch.object(backend_client, "_REQUEST_TIMEOUT_SECONDS", 0),
        patch.object(backend_client, "_RETRY_BACKOFF_MAX_SECONDS", 0),
        patch.object(httpx.AsyncClient, "post", side_effect=flaky_post),
    ):
        client = BackendClient("http://test")
        with anyio.fail_after(_SAFETY_NET_SECONDS):
            await client.append_transcript("run_1", {})
        await client.aclose()

    assert call_count == 3


@pytest.mark.asyncio
async def test_non_retryable_status_is_not_retried():
    """A 4xx is a genuine failure and must propagate without retrying."""
    call_count = 0

    async def bad_request_post(*_args: Any, **_kwargs: Any) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(
            400, text="Bad Request", request=httpx.Request("POST", "http://test")
        )

    with (
        patch(f"{_MODULE}.read_token", return_value="test_token"),
        patch.object(httpx.AsyncClient, "post", side_effect=bad_request_post),
    ):
        client = BackendClient("http://test")
        with pytest.raises(httpx.HTTPStatusError):
            await client.append_transcript("run_1", {})
        await client.aclose()

    assert call_count == 1


@pytest.mark.asyncio
async def test_transport_error_is_retried():
    """A ReadError (stale keep-alive connection) is retried."""
    call_count = 0

    async def broken_then_ok(*_args: Any, **_kwargs: Any) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise httpx.ReadError("connection reset")
        return _ok_response()

    with (
        patch(f"{_MODULE}.read_token", return_value="test_token"),
        patch.object(backend_client, "_RETRY_BACKOFF_MAX_SECONDS", 0.01),
        patch.object(httpx.AsyncClient, "post", side_effect=broken_then_ok),
    ):
        client = BackendClient("http://test")
        await client.append_transcript("run_1", {})
        await client.aclose()

    assert call_count == 2


@pytest.mark.asyncio
async def test_transport_error_forces_fresh_connection():
    """A transport error drops the poisoned connection pool before retrying."""
    call_count = 0

    async def broken_then_ok(*_args: Any, **_kwargs: Any) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise httpx.ReadError("connection reset")
        return _ok_response()

    with (
        patch(f"{_MODULE}.read_token", return_value="test_token"),
        patch.object(backend_client, "_RETRY_BACKOFF_MAX_SECONDS", 0.01),
        patch.object(httpx.AsyncClient, "post", side_effect=broken_then_ok),
    ):
        client = BackendClient("http://test")
        first_inner = client._client  # pyright: ignore[reportPrivateUsage]
        await client.append_transcript("run_1", {})

        assert client._client is not first_inner  # pyright: ignore[reportPrivateUsage]
        assert first_inner.is_closed, "poisoned client was not closed"
        await client.aclose()


@pytest.fixture
def error_log() -> Iterator[list[str]]:
    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="ERROR")
    yield messages
    logger.remove(handler_id)


@pytest.mark.asyncio
async def test_post_error_log_names_exception_type(error_log: list[str]):
    """ReadError and TimeoutError stringify to '', so the type must be logged."""

    async def broken_post(*_args: Any, **_kwargs: Any) -> httpx.Response:
        raise httpx.ReadError("")

    with (
        patch(f"{_MODULE}.read_token", return_value="test_token"),
        patch.object(backend_client, "_RETRY_BUDGET_SECONDS", 0.01),
        patch.object(backend_client, "_RETRY_BACKOFF_MAX_SECONDS", 0.01),
        patch.object(httpx.AsyncClient, "post", side_effect=broken_post),
    ):
        client = BackendClient("http://test")
        with pytest.raises(httpx.ReadError):
            await client.append_transcript("run_1", {})
        await client.aclose()

    assert any("httpx.ReadError" in m for m in error_log)


@pytest.mark.asyncio
async def test_post_error_log_includes_response_body(error_log: list[str]):
    async def bad_request_post(*_args: Any, **_kwargs: Any) -> httpx.Response:
        return httpx.Response(
            400,
            text='{"detail": "seq 5 out of range"}',
            request=httpx.Request("POST", "http://test"),
        )

    with (
        patch(f"{_MODULE}.read_token", return_value="test_token"),
        patch.object(httpx.AsyncClient, "post", side_effect=bad_request_post),
    ):
        client = BackendClient("http://test")
        with pytest.raises(httpx.HTTPStatusError):
            await client.append_transcript("run_1", {})
        await client.aclose()

    assert any("seq 5 out of range" in m for m in error_log)


@pytest.mark.asyncio
async def test_get_error_log_includes_response_body(error_log: list[str]):
    async def server_error_get(*_args: Any, **_kwargs: Any) -> httpx.Response:
        return httpx.Response(
            500,
            text='{"detail": "db write failed"}',
            request=httpx.Request("GET", "http://test"),
        )

    with (
        patch(f"{_MODULE}.read_token", return_value="test_token"),
        patch.object(backend_client, "_RETRY_BUDGET_SECONDS", 0.01),
        patch.object(backend_client, "_RETRY_BACKOFF_MAX_SECONDS", 0.01),
        patch.object(httpx.AsyncClient, "get", side_effect=server_error_get),
    ):
        client = BackendClient("http://test")
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_transcript_length("run_1")
        await client.aclose()

    assert any("db write failed" in m for m in error_log)


@pytest.mark.asyncio
async def test_non_transport_error_keeps_connection():
    """A genuine 4xx failure must not tear down a healthy connection pool."""

    async def bad_request_post(*_args: Any, **_kwargs: Any) -> httpx.Response:
        return httpx.Response(
            400, text="Bad Request", request=httpx.Request("POST", "http://test")
        )

    with (
        patch(f"{_MODULE}.read_token", return_value="test_token"),
        patch.object(httpx.AsyncClient, "post", side_effect=bad_request_post),
    ):
        client = BackendClient("http://test")
        first_inner = client._client  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(httpx.HTTPStatusError):
            await client.append_transcript("run_1", {})

        assert client._client is first_inner  # pyright: ignore[reportPrivateUsage]
        await client.aclose()


def test_token_is_read_from_the_default_path(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("KAROTTE_BACKEND_TOKEN_PATH", raising=False)
    with patch.object(Path, "read_text", autospec=True, return_value="tok\n") as read:
        assert backend_client.read_token() == "tok"
    assert read.call_args.args[0] == Path("/var/run/secrets/service-account-token")


def test_token_path_can_be_overridden(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    token_file = tmp_path / "token"
    token_file.write_text("custom\n")
    monkeypatch.setenv("KAROTTE_BACKEND_TOKEN_PATH", str(token_file))
    assert backend_client.read_token() == "custom"


def test_no_token_without_a_token_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KAROTTE_BACKEND_TOKEN_PATH", str(tmp_path / "missing"))
    assert backend_client.read_token() is None


@pytest.mark.asyncio
async def test_requests_carry_the_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    token_file = tmp_path / "token"
    token_file.write_text("tok")
    monkeypatch.setenv("KAROTTE_BACKEND_TOKEN_PATH", str(token_file))
    with patch.object(
        httpx.AsyncClient, "post", AsyncMock(return_value=_ok_response())
    ):
        client = BackendClient("http://test")
        await client.create_transcript("run_1")
        assert client._client.headers["Authorization"] == "Bearer tok"  # pyright: ignore[reportPrivateUsage]
        await client.aclose()


@pytest.mark.asyncio
async def test_requests_are_unauthenticated_without_a_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("KAROTTE_BACKEND_TOKEN_PATH", str(tmp_path / "missing"))
    with patch.object(
        httpx.AsyncClient, "post", AsyncMock(return_value=_ok_response())
    ):
        client = BackendClient("http://test")
        await client.create_transcript("run_1")
        assert "Authorization" not in client._client.headers  # pyright: ignore[reportPrivateUsage]
        await client.aclose()
