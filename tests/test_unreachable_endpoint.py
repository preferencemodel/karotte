import socket
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiohttp
import httpx
import litellm
import litellm.exceptions
import pytest

from karotte.agents.builtin_source import (
    LLM_RETRY_MAX_ATTEMPTS,
    UNREACHABLE_RETRY_MAX_ATTEMPTS,
    ModelEndpointUnreachableError,
    _unreachable,  # pyright: ignore[reportPrivateUsage]
)
from karotte.evaluation_runner import EvaluationRunner
from karotte.mcp_servers.http_mcp_server import HttpMcpServer
from karotte.schemas.chat import Message
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.http_mcp_server_config import HttpMcpServerConfig
from karotte.schemas.transcript import (
    ErrorEvent,
    MessageAddedEvent,
    TaskCompletedEvent,
    Transcript,
)
from tests.conftest import TestTask, find_free_port
from tests.test_evaluation_runner import (
    _collect,  # pyright: ignore[reportPrivateUsage]
    _create_runner_with_tools_and_transcript,  # pyright: ignore[reportPrivateUsage]
    _make_mock_model_response,  # pyright: ignore[reportPrivateUsage]
    _make_mock_stream,  # pyright: ignore[reportPrivateUsage]
)

_URL = "https://nonexistent.invalid/v1/messages"


def _chain(*excs: BaseException) -> BaseException:
    """Link ``excs`` outermost first, each caused by the next."""
    for outer, inner in zip(excs, excs[1:]):
        outer.__cause__ = inner
    return excs[0]


def _litellm_wrapped(*causes: BaseException) -> BaseException:
    """The shape litellm raises for a transport failure: a 500 around the cause."""
    outer = litellm.exceptions.InternalServerError(
        message=f"AnthropicException - {causes[0]}",
        model="test",
        llm_provider="anthropic",
    )
    return _chain(outer, *causes)


def _connect_error(cause: BaseException) -> BaseException:
    error = httpx.ConnectError(str(cause), request=httpx.Request("POST", _URL))
    return _litellm_wrapped(error, cause)


def _dns_error(errno: int = socket.EAI_NONAME) -> BaseException:
    return _connect_error(socket.gaierror(errno, "Name or service not known"))


def _refused_error() -> BaseException:
    return _connect_error(ConnectionRefusedError(111, "Connection refused"))


def test_dns_not_found_is_unreachable():
    unreachable = _unreachable(_dns_error())
    assert unreachable is not None
    assert not unreachable.permanent
    assert str(unreachable) == (
        f"Could not reach {_URL}: DNS lookup failed (Name or service not known)"
    )


def test_dns_no_data_is_unreachable():
    assert _unreachable(_dns_error(socket.EAI_NODATA)) is not None


def test_temporary_dns_failure_is_transient():
    assert _unreachable(_dns_error(socket.EAI_AGAIN)) is None


def test_connection_refused_is_unreachable():
    unreachable = _unreachable(_refused_error())
    assert unreachable is not None
    assert str(unreachable) == f"Could not reach {_URL}: connection refused"


def test_openai_client_shape_is_unreachable():
    import openai

    request = httpx.Request("POST", "https://nonexistent.invalid/v1/chat/completions")
    error = _litellm_wrapped(
        openai.APIConnectionError(request=request),
        httpx.ConnectError("dns", request=request),
        socket.gaierror(socket.EAI_NONAME, "Name or service not known"),
    )
    unreachable = _unreachable(error)
    assert unreachable is not None
    assert "https://nonexistent.invalid/v1/chat/completions" in str(unreachable)


@pytest.mark.parametrize(
    "cause",
    [
        httpx.InvalidURL("/nonexistent.invalid/v1/messages"),
        httpx.UnsupportedProtocol("Request URL has an unsupported protocol 'ftp://'."),
        aiohttp.NonHttpUrlClientError("ftp://x.invalid/v1/messages"),
    ],
    ids=["invalid", "unsupported-protocol", "non-http"],
)
def test_invalid_url_is_permanent(cause: BaseException):
    unreachable = _unreachable(_litellm_wrapped(cause))
    assert unreachable is not None
    assert unreachable.permanent
    assert "invalid URL" in str(unreachable)


@pytest.mark.parametrize(
    "error",
    [
        litellm.exceptions.InternalServerError(
            message="AnthropicError - Overloaded", model="t", llm_provider="anthropic"
        ),
        _litellm_wrapped(httpx.ReadTimeout("timed out")),
        _litellm_wrapped(httpx.RemoteProtocolError("peer closed connection")),
        _connect_error(ConnectionResetError(104, "Connection reset by peer")),
        _connect_error(TimeoutError("connect timed out")),
    ],
    ids=["overloaded", "read-timeout", "protocol", "reset", "connect-timeout"],
)
def test_transient_errors_are_not_unreachable(error: BaseException):
    assert _unreachable(error) is None


async def _drive(errors: list[BaseException]) -> tuple[int, AsyncMock]:
    """Run one turn whose calls raise ``errors`` in order, then succeed."""
    runner = _create_runner_with_tools_and_transcript()
    calls = 0

    async def mock_acompletion(**_kwargs):  # pyright: ignore[reportMissingParameterType]
        nonlocal calls
        calls += 1
        if errors:
            raise errors.pop(0)
        return _make_mock_stream([])

    sleep = AsyncMock()
    with (
        patch("litellm.acompletion", side_effect=mock_acompletion),
        patch(
            "litellm.stream_chunk_builder",
            return_value=_make_mock_model_response("ok"),
        ),
        patch("asyncio.sleep", sleep),
    ):
        async for _ in _collect(runner):
            pass
    return calls, sleep


@pytest.mark.asyncio
async def test_dns_failure_gives_up_quickly_with_a_clear_message():
    errors = [_dns_error() for _ in range(LLM_RETRY_MAX_ATTEMPTS)]
    with pytest.raises(ModelEndpointUnreachableError) as excinfo:
        await _drive(errors)

    assert str(excinfo.value) == (
        f"Could not reach {_URL}: DNS lookup failed (Name or service not known)"
    )
    assert len(errors) == LLM_RETRY_MAX_ATTEMPTS - UNREACHABLE_RETRY_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_unreachable_backoff_is_short():
    runner = _create_runner_with_tools_and_transcript()

    async def mock_acompletion(**_kwargs):  # pyright: ignore[reportMissingParameterType]
        raise _refused_error()

    sleep = AsyncMock()
    with (
        patch("litellm.acompletion", side_effect=mock_acompletion),
        patch("asyncio.sleep", sleep),
        pytest.raises(ModelEndpointUnreachableError),
    ):
        async for _ in _collect(runner):
            pass

    waits = [call.args[0] for call in sleep.await_args_list]
    assert len(waits) == UNREACHABLE_RETRY_MAX_ATTEMPTS - 1
    assert sum(waits) <= 15


@pytest.mark.asyncio
async def test_invalid_url_is_not_retried():
    errors = [_litellm_wrapped(httpx.InvalidURL("/nonexistent.invalid/v1/messages"))]
    with pytest.raises(ModelEndpointUnreachableError, match="invalid URL"):
        await _drive(errors)
    assert errors == []


@pytest.mark.asyncio
async def test_connection_blip_recovers():
    calls, _ = await _drive([_refused_error(), _dns_error()])
    assert calls == 3


@pytest.mark.asyncio
async def test_temporary_dns_failure_keeps_the_full_retry_budget():
    calls, _ = await _drive(
        [_dns_error(socket.EAI_AGAIN) for _ in range(LLM_RETRY_MAX_ATTEMPTS - 1)]
    )
    assert calls == LLM_RETRY_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_earlier_transient_errors_do_not_use_up_the_unreachable_budget():
    overloaded = [
        litellm.exceptions.InternalServerError(
            message="AnthropicError - Overloaded", model="t", llm_provider="anthropic"
        )
        for _ in range(8)
    ]
    refused = [_refused_error() for _ in range(UNREACHABLE_RETRY_MAX_ATTEMPTS - 1)]
    calls, _ = await _drive([*overloaded, *refused])
    assert calls == 8 + UNREACHABLE_RETRY_MAX_ATTEMPTS


def _claude_runner(tmp_path: Path, port: int = 8080) -> EvaluationRunner:
    config = EvaluationRunConfig(
        run_id="test_run",
        task_id="test-task",
        model="claude-sonnet-4-20250514",
        model_api_key="test_key",
        mcp_server_config=HttpMcpServerConfig(host="0.0.0.0", port=port),
        transcript_file=(tmp_path / "transcript.json").as_posix(),
    )
    return EvaluationRunner(config, TestTask(config))


def _resolver_answers_nxdomain() -> bool:
    try:
        socket.getaddrinfo("nonexistent.invalid", 80)
    except socket.gaierror as e:
        return e.errno in (socket.EAI_NONAME, socket.EAI_NODATA)
    return False


@pytest.mark.asyncio
async def test_real_litellm_call_to_unresolvable_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    if not _resolver_answers_nxdomain():
        pytest.skip("the resolver does not answer NXDOMAIN for .invalid")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://nonexistent.invalid")
    monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)
    runner = _claude_runner(tmp_path)
    runner.transcript = Transcript(run_id="test_run")
    runner.transcript.events.append(
        MessageAddedEvent(message=Message(role="user", content="Hello"))
    )
    runner.tools = []

    with (
        patch("karotte.agents.builtin_source._retry_wait", return_value=0),
        pytest.raises(ModelEndpointUnreachableError) as excinfo,
    ):
        async for _ in _collect(runner):
            pass

    assert str(excinfo.value).startswith(
        "Could not reach http://nonexistent.invalid/v1/messages: DNS lookup failed"
    )


@pytest.mark.asyncio
async def test_refused_endpoint_ends_the_transcript_with_task_completed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mcp_server: HttpMcpServer
):
    port = find_free_port()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{port}")
    monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)
    runner = _claude_runner(tmp_path, mcp_server.config.port)

    with patch("karotte.agents.builtin_source._retry_wait", return_value=0):
        _ = [event async for event in runner.run()]

    transcript = Transcript.model_validate_json(
        (tmp_path / "transcript.json").read_text()
    )
    error, completed = transcript.events[-2:]
    assert isinstance(error, ErrorEvent)
    assert error.exception_type == "ModelEndpointUnreachableError"
    assert error.message == (
        f"Could not reach http://127.0.0.1:{port}/v1/messages: connection refused"
    )
    assert isinstance(completed, TaskCompletedEvent)
    assert completed.status == "error"
