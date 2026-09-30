"""Async HTTP client for the run backend's `/api/internal/` endpoints."""

import functools
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import anyio
import httpx
from loguru import logger
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_delay,
    wait_exponential,
)

from karotte.schemas.transcript import MessageAddedEvent

DEFAULT_TOKEN_PATH = Path("/var/run/secrets/service-account-token")


def read_token() -> str | None:
    """Bearer token for the backend, or None without a token file.

    Re-read on every request because the token rotates.
    """
    path = Path(os.environ.get("KAROTTE_BACKEND_TOKEN_PATH", DEFAULT_TOKEN_PATH))
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return None


# Per-attempt deadline; httpx's own timeouts don't always fire on a hung backend.
_REQUEST_TIMEOUT_SECONDS = 30.0

# Total retry time per request, long enough to ride out a multi-minute outage.
_RETRY_BUDGET_SECONDS = 25 * 60

# Cap on the exponential backoff between attempts.
_RETRY_BACKOFF_MAX_SECONDS = 10


def _is_retryable(exc: BaseException) -> bool:
    """Return True for connection errors, timeouts, and 5XX responses."""
    if isinstance(exc, httpx.TransportError):
        return True
    # anyio.fail_after raises a builtin TimeoutError for a hung request.
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500:
        return True
    return False


def _with_retries(
    func: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """Retry ``func`` on transient failures within a bounded time budget.

    The retry policy is rebuilt per call so the module-level tuning constants
    are read at call time (and can be patched in tests).
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        retrying = AsyncRetrying(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_delay(_RETRY_BUDGET_SECONDS),
            wait=wait_exponential(multiplier=1, max=_RETRY_BACKOFF_MAX_SECONDS),
            reraise=True,
        )
        return await retrying(func, *args, **kwargs)

    return wrapper


class BackendClient:
    """Async client for the backend's internal eval-run endpoints.

    Use as an async context manager:

        async with BackendClient(base_url) as client:
            await client.create_transcript(run_id)
    """

    _base_url: str
    _client: httpx.AsyncClient

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = self._build_client()

    def _build_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(5.0, read=15.0),
        )

    async def _reset_on_transport_error(self, exc: BaseException) -> None:
        """Drop the connection pool after a transport error before retrying.

        A ReadError/BrokenResourceError (or an attempt we abandoned via the hard
        deadline) leaves a stale keep-alive connection in the pool; reusing it
        fails again. Rebuild the client so the retry reconnects fresh. Genuine
        failures (4xx/5xx) leave the healthy pool untouched.
        """
        if isinstance(exc, (httpx.TransportError, TimeoutError)):
            await self._client.aclose()
            self._client = self._build_client()

    async def __aenter__(self) -> "BackendClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _refresh_auth(self) -> None:
        token = read_token()
        if token is None:
            self._client.headers.pop("Authorization", None)
        else:
            self._client.headers["Authorization"] = f"Bearer {token}"

    @_with_retries
    async def _post(self, path: str, **kwargs: Any) -> httpx.Response:
        self._refresh_auth()
        try:
            with anyio.fail_after(_REQUEST_TIMEOUT_SECONDS):
                response = await self._client.post(path, **kwargs)
            response.raise_for_status()
            return response
        except Exception as e:
            logger.error(f"Error posting {path} - {describe_error(e)}")
            await self._reset_on_transport_error(e)
            raise

    @_with_retries
    async def _get(
        self,
        path: str,
        ignore_statuses: tuple[int, ...] = (),
        **kwargs: Any,
    ) -> httpx.Response:
        self._refresh_auth()
        try:
            with anyio.fail_after(_REQUEST_TIMEOUT_SECONDS):
                response = await self._client.get(path, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in ignore_statuses:
                logger.error(f"Error getting {path} - {describe_error(e)}")
            raise
        except Exception as e:
            logger.error(f"Error getting {path} - {describe_error(e)}")
            await self._reset_on_transport_error(e)
            raise

    async def create_transcript(self, run_id: str) -> None:
        await self._post(
            "/api/internal/create_transcript",
            params={"run_id": run_id},
        )

    async def get_message(self, run_id: str) -> MessageAddedEvent | None:
        try:
            response = await self._get(
                "/api/internal/get_message",
                params={"run_id": run_id},
                ignore_statuses=(404,),
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        return MessageAddedEvent.model_validate(response.json())

    async def delete_message(self, run_id: str) -> bool:
        response = await self._post(
            "/api/internal/delete_message",
            params={"run_id": run_id},
        )
        return response.json()

    async def get_transcript_length(self, run_id: str) -> int | None:
        """Number of events already stored, or None if the transcript is new."""
        try:
            response = await self._get(
                "/api/internal/transcript_length",
                params={"run_id": run_id},
                ignore_statuses=(404,),
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        return response.json()

    async def append_transcript(
        self, run_id: str, event: dict[str, Any], seq: int | None = None
    ) -> None:
        body: dict[str, Any] = {"event": event}
        if seq is not None:
            # The seq (array index) makes the write idempotent server-side: a
            # retry of an already-stored event rewrites its slot instead of
            # duplicating it.
            body["seq"] = seq
        await self._post(
            "/api/internal/append_transcript",
            params={"run_id": run_id},
            json=body,
        )

    async def update_run_state(
        self,
        run_id: str,
        status: str,
        score: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cache_read_tokens: int | None = None,
        cache_write_tokens: int | None = None,
    ) -> None:
        await self._post(
            "/api/internal/update_run_state",
            params={"run_id": run_id},
            json={
                "status": status,
                "score": score,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_tokens": cache_read_tokens,
                "cache_write_tokens": cache_write_tokens,
            },
        )


def full_typename(exc: BaseException) -> str:
    cls = type(exc)
    module = cls.__module__
    qualname = cls.__qualname__
    if module == "builtins":
        return qualname
    return f"{module}.{qualname}"


def describe_error(exc: BaseException) -> str:
    """Loggable description; transport/timeout errors often stringify to ''."""
    description = f"{full_typename(exc)}: {exc}"
    if isinstance(exc, httpx.HTTPStatusError):
        description += f" - response body: {exc.response.text[:2000]}"
    return description
