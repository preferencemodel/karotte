import asyncio
import random
import re
import socket
import warnings
from collections.abc import AsyncGenerator, Iterator
from dataclasses import dataclass
from typing import Any, final

import aiohttp
import httpx
import litellm
import litellm.exceptions
from litellm import (
    ChatCompletionToolParam,
    Choices,
    CustomStreamWrapper,
)
from litellm.types.utils import ModelResponse, ModelResponseStream, StreamingChoices
from loguru import logger

from karotte.model_spec import spec_for
from karotte.providers import (
    TOOL_CALL_REPAIRS,
    ProviderAuth,
    cache_tokens,
    provider_for,
    proxy_api_base,
    repair_tool_calls,
    service_tier_mode,
    vertex_served_tier,
)
from karotte.schemas.chat import Delta, Message
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.transcript import (
    Event,
    MessageAddedEvent,
    MessageChunkEvent,
    MessageChunkResetEvent,
    TokenUsageEvent,
)

# litellm's success logging for Responses API streams dumps a chat usage dict into ResponseAPIUsage.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

# Retry configuration for transient LLM errors (overloaded, rate-limited, etc.)
# 16 retries lead to a maximum wait time of around 10 minutes
LLM_RETRY_MAX_ATTEMPTS = 16
LLM_RETRY_WAIT_MIN_S = 1
LLM_RETRY_WAIT_MAX_S = 60
LLM_RETRY_AFTER_MAX_S = 300

_RETRYABLE_EXCEPTIONS = (
    litellm.exceptions.InternalServerError,
    litellm.exceptions.RateLimitError,
    litellm.exceptions.ServiceUnavailableError,
    litellm.exceptions.Timeout,
    litellm.exceptions.APIConnectionError,
)

# A 401 is usually permanent, but providers do sometimes return false negatives.
AUTH_RETRY_MAX_ATTEMPTS = 3

# Unknown host or refused connection: a few quick tries ride out a blip.
UNREACHABLE_RETRY_MAX_ATTEMPTS = 4

_DNS_NOT_FOUND = {socket.EAI_NONAME, socket.EAI_NODATA}
_INVALID_URL_ERRORS = (
    httpx.InvalidURL,
    httpx.UnsupportedProtocol,
    aiohttp.InvalidURL,
    aiohttp.NonHttpUrlClientError,
)

_RETRY_AFTER_BODY_RE = re.compile(r'"retry_after"\s*:\s*(\d+(?:\.\d+)?)')


def _is_retryable(exc: Exception) -> bool:
    """Whether an LLM call failed for a reason worth trying again."""
    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return True
    status_code = getattr(exc, "status_code", None)
    return isinstance(status_code, int) and status_code >= 500


def _retry_limit(exc: Exception) -> int:
    """How many attempts in total an LLM call failing with ``exc`` gets."""
    if isinstance(exc, litellm.exceptions.AuthenticationError):
        return AUTH_RETRY_MAX_ATTEMPTS
    return LLM_RETRY_MAX_ATTEMPTS if _is_retryable(exc) else 1


class ModelEndpointUnreachableError(Exception):
    """The model endpoint could not be reached at all."""


@dataclass(frozen=True)
class _Unreachable:
    reason: str
    url: str | None
    permanent: bool

    def __str__(self) -> str:
        return f"Could not reach {self.url or 'the model endpoint'}: {self.reason}"


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _request_url(exc: BaseException) -> str | None:
    if not isinstance(exc, httpx.RequestError):
        return None
    try:
        return str(exc.request.url)
    except RuntimeError:
        return None


def _unreachable(exc: BaseException) -> _Unreachable | None:
    """Why the endpoint could not be reached, if the connection itself failed in a
    way that retrying won't soon fix; litellm reports these as a generic 500."""
    chain = list(_exception_chain(exc))
    url = next((u for e in chain if (u := _request_url(e))), None)
    for e in chain:
        if isinstance(e, _INVALID_URL_ERRORS):
            return _Unreachable(
                "invalid URL (does it start with http:// or https://?)",
                url or str(e),
                permanent=True,
            )
        if isinstance(e, socket.gaierror) and e.errno in _DNS_NOT_FOUND:
            return _Unreachable(
                f"DNS lookup failed ({e.strerror})", url, permanent=False
            )
        if isinstance(e, ConnectionRefusedError):
            return _Unreachable("connection refused", url, permanent=False)
    return None


def _retry_after(exc: Exception) -> float | None:
    """The provider's requested retry delay in seconds, if it sent one."""
    headers = getattr(exc, "litellm_response_headers", None)
    raw = headers.get("retry-after") if headers else None
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass

    match = _RETRY_AFTER_BODY_RE.search(str(exc))
    return float(match.group(1)) if match else None


def _retry_wait(attempt: int, retry_after: float | None = None) -> float:
    """Wait before retry attempt number (1-based): the provider's requested delay
    when it sent one, otherwise exponential backoff with jitter."""
    if retry_after is not None:
        return min(max(retry_after, 0.0), LLM_RETRY_AFTER_MAX_S)
    exp = min(2 ** (attempt - 1), LLM_RETRY_WAIT_MAX_S)
    return min(
        LLM_RETRY_WAIT_MAX_S,
        max(LLM_RETRY_WAIT_MIN_S, exp + random.uniform(0, exp * 0.5)),
    )


def _tap_traffic_types(stream: CustomStreamWrapper) -> list[str]:
    """Collect Gemini's ``usageMetadata.trafficType``, which litellm drops from the
    chunks it yields but sets on the provider chunks it builds them from."""
    seen: list[str] = []
    create = stream.chunk_creator

    def tap(chunk: Any) -> Any:
        fields = (getattr(chunk, "_hidden_params", None) or {}).get(
            "provider_specific_fields"
        ) or {}
        if traffic_type := fields.get("traffic_type"):
            seen.append(str(traffic_type))
        return create(chunk=chunk)

    stream.chunk_creator = tap
    return seen


def _provider_auth(config: EvaluationRunConfig) -> ProviderAuth:
    return ProviderAuth(api_key=config.model_api_key)


@final
class BuiltinSource:
    """The builtin agent: karotte produces the model's response itself by driving
    the litellm turn loop -- streaming completion, chunk assembly, retries on
    transient errors, and per-provider request quirks."""

    def __init__(self, config: EvaluationRunConfig) -> None:
        self.config = config
        self._spec = spec_for(config.model)
        self._provider = provider_for(self._spec)
        # Sticky for the run: flipping back to the default tier risks another
        # replica, and with it the prompt cache.
        self._escalated = False
        self._service_tier: str | None = None

    async def collect(
        self,
        messages: list[Message],
        tools: list[ChatCompletionToolParam],
    ) -> AsyncGenerator[Event]:
        unreachable_attempts = 0
        for attempt in range(1, LLM_RETRY_MAX_ATTEMPTS + 1):
            chunks: list[ModelResponseStream] = []

            try:
                response = await litellm.acompletion(
                    **self.get_completion_params(messages, tools)
                )
                assert isinstance(response, CustomStreamWrapper)
                traffic_types = _tap_traffic_types(response)

                async for chunk in response:
                    assert isinstance(chunk, ModelResponseStream)
                    assert isinstance(chunk.choices[0], StreamingChoices)
                    chunks.append(chunk)
                    yield MessageChunkEvent(
                        delta=Delta(**chunk.choices[0].delta.model_dump())
                    )

            except Exception as e:
                if (
                    not self._escalated
                    and service_tier_mode() == "auto"
                    and self._provider.escalates_on(self._spec, e)
                ):
                    logger.warning(
                        "{model} ran out of provider capacity; asking for priority for the rest of the run",
                        model=self.config.model,
                    )
                    self._escalated = True
                unreachable = _unreachable(e)
                if unreachable is None:
                    tries, limit, cause = attempt, _retry_limit(e), type(e).__name__
                else:
                    unreachable_attempts += 1
                    tries, cause = unreachable_attempts, str(unreachable)
                    limit = (
                        1 if unreachable.permanent else UNREACHABLE_RETRY_MAX_ATTEMPTS
                    )
                if tries < limit and attempt < LLM_RETRY_MAX_ATTEMPTS:
                    wait = _retry_wait(tries, _retry_after(e))
                    logger.warning(
                        "LLM call failed: {cause}; retrying (attempt {next} of {max}) after {wait:.1f}s...",
                        cause=cause,
                        next=tries + 1,
                        max=limit,
                        wait=wait,
                    )
                    yield MessageChunkResetEvent()
                    await asyncio.sleep(wait)
                    continue
                if unreachable is not None:
                    raise ModelEndpointUnreachableError(str(unreachable)) from e
                raise

            full_response = litellm.stream_chunk_builder(chunks)
            assert isinstance(full_response, ModelResponse)
            choice = full_response.choices[0]
            assert isinstance(choice, Choices)

            message = Message(**choice.message.model_dump())

            for name in self._spec.tool_call_repairs:
                message.tool_calls = repair_tool_calls(
                    message.tool_calls, TOOL_CALL_REPAIRS[name]
                )

            usage = getattr(full_response, "usage", None)

            # litellm normalizes a provider safety refusal (e.g. Claude's native
            # 'refusal' stop_reason) to the OpenAI 'content_filter' finish_reason.
            finish_reason = choice.finish_reason
            if finish_reason == "content_filter":
                logger.warning(
                    "Model {model} declined the request (finish_reason=content_filter).",
                    model=self.config.model,
                )

            yield MessageAddedEvent(message=message, finish_reason=finish_reason)

            if usage:
                cache_read_tokens, cache_write_tokens = cache_tokens(usage)
                yield TokenUsageEvent(
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=cache_write_tokens,
                    service_tier=vertex_served_tier(traffic_types[-1])
                    if traffic_types
                    else self._service_tier,
                )
            return

    def get_completion_params(
        self,
        messages: list[Message],
        tools: list[ChatCompletionToolParam],
    ) -> dict[str, Any]:
        serialized_messages = self._serialize_messages(messages)
        spec = self._spec

        completion_params: dict[str, Any] = {
            "stream": True,
            "stream_options": {"include_usage": True},
            "model": spec.litellm_model,
            "messages": serialized_messages,
            "tools": tools,
            "tool_choice": "auto" if tools else None,
            "max_tokens": spec.max_output_tokens,
            "timeout": 300.0,
            # this is needed for together ai and other such providers, doesn't seem to break
            # claude api calls, so doing it for everyone
            "allowed_openai_params": [
                "tools",
                "tool_choice",
                *spec.extra_allowed_openai_params,
            ],
        }

        completion_params.update(
            self._provider.request_params(spec, _provider_auth(self.config))
        )

        if api_base := proxy_api_base(self._provider):
            completion_params["api_base"] = api_base

        self._provider.apply_prompt_caching(
            self.config.run_id, serialized_messages, completion_params
        )

        self._provider.apply_reasoning(
            spec, self.config.applied_reasoning_effort, completion_params
        )

        self._service_tier = self._provider.apply_service_tier(
            spec, self._escalated, completion_params
        )

        return completion_params

    def _serialize_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Serializes transcript messages to dicts."""
        return [message.model_dump() for message in messages]
