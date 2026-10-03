"""Per-provider request shaping and response repair for a litellm turn loop.

The base class is a complete implementation, so an unregistered provider still
authenticates and reaches the proxy, and simply gets no prompt caching.
"""

import functools
import json
import logging
import os
import re
from collections.abc import Callable
from typing import Any, Final, Literal, override

from pydantic import BaseModel

from karotte.model_spec import ModelSpec, ToolCallRepair
from karotte.schemas.chat import ChatCompletionMessageToolCall, Function

_logger = logging.getLogger(__name__)


class ProviderAuth(BaseModel, frozen=True):
    api_key: str | None = None


SERVICE_TIER_ENV: Final = "KAROTTE_INFERENCE_SERVICE_TIER"
"""``priority`` always asks for the provider's priority tier, ``auto`` does once the
conversation hits the provider's capacity error; unset uses the default tier."""

ServiceTierMode = Literal["auto", "priority"]


def service_tier_mode() -> ServiceTierMode | None:
    raw = os.environ.get(SERVICE_TIER_ENV, "").strip().lower()
    if raw == "auto" or raw == "priority":
        return raw
    if raw:
        _warn_unknown_service_tier(raw)
    return None


@functools.cache
def _warn_unknown_service_tier(value: str) -> None:
    _logger.warning("unknown %s=%r, ignoring it", SERVICE_TIER_ENV, value)


def _wants_priority(escalated: bool) -> bool:
    mode = service_tier_mode()
    return mode == "priority" or (mode == "auto" and escalated)


class Provider:
    """How one API provider's requests are built."""

    proxied: bool = True
    """Whether KAROTTE_PROXY_URL routing applies."""

    def request_params(self, _spec: ModelSpec, auth: ProviderAuth) -> dict[str, Any]:
        """Provider-specific params, chiefly how the call authenticates."""
        return {"api_key": auth.api_key}

    def apply_prompt_caching(
        self,
        _cache_key: str,
        _messages: list[dict[str, Any]],
        _params: dict[str, Any],
    ) -> None:
        """Ask for prompt caching. ``cache_key`` names the conversation for
        providers that key their cache on the caller (one per run or review)."""
        return None

    def escalates_on(self, _spec: ModelSpec, _exc: Exception) -> bool:
        """Whether ``exc`` should move the rest of the conversation to a higher
        service tier. Only a caller that keeps per-conversation state acts on it."""
        return False

    def apply_service_tier(
        self, _spec: ModelSpec, _escalated: bool, _params: dict[str, Any]
    ) -> str | None:
        """Ask for a service tier; returns the tier asked for, None for the default."""
        return None

    def apply_reasoning(
        self, spec: ModelSpec, effort: str | None, params: dict[str, Any]
    ) -> None:
        """Ask the model to think and to return its reasoning."""
        request = dict(spec.reasoning_request)
        if (extra_body := request.pop("extra_body", None)) is not None:
            merge_extra_body(params, extra_body)
        params.update(request)
        if effort is not None:
            # litellm maps a Gemini effort to a thinking config that already includes
            # summaries and 400s if we send our own, so the spec's thinkingConfig is
            # dropped when an effort level is sent.
            params.pop("thinkingConfig", None)
            # litellm >=1.83.13 auto-routes tool-using gpt-5.4+ calls through
            # OpenAI's Responses API, so reasoning_effort coexists with tools.
            params["reasoning_effort"] = effort


class AnthropicProvider(Provider):
    proxied: bool = False

    @override
    def apply_prompt_caching(
        self,
        _cache_key: str,
        messages: list[dict[str, Any]],
        _params: dict[str, Any],
    ) -> None:
        _add_inline_cache_control(messages)


class OpenAIProvider(Provider):
    @override
    def apply_prompt_caching(
        self,
        cache_key: str,
        _messages: list[dict[str, Any]],
        params: dict[str, Any],
    ) -> None:
        # litellm 1.94 lists prompt_cache_key as a known OpenAI param but never
        # plumbs it into the request, so a top-level kwarg is dropped on the
        # wire. extra_body is forwarded verbatim on both Chat Completions and
        # the Responses bridge.
        merge_extra_body(params, {"prompt_cache_key": cache_key})


class XaiProvider(Provider):
    @override
    def apply_prompt_caching(
        self,
        cache_key: str,
        _messages: list[dict[str, Any]],
        params: dict[str, Any],
    ) -> None:
        merge_extra_headers(params, {"x-grok-conv-id": cache_key})


def _is_vertex_gemini(spec: ModelSpec) -> bool:
    return spec.model.startswith("vertex_ai/gemini-")


class VertexProvider(Provider):
    proxied: bool = False

    @override
    def request_params(self, spec: ModelSpec, auth: ProviderAuth) -> dict[str, Any]:
        return {}

    @override
    def escalates_on(self, spec: ModelSpec, exc: Exception) -> bool:
        # 429 RESOURCE_EXHAUSTED is Vertex running out of shared capacity.
        return _is_vertex_gemini(spec) and getattr(exc, "status_code", None) == 429

    @override
    def apply_service_tier(
        self, spec: ModelSpec, escalated: bool, params: dict[str, Any]
    ) -> str | None:
        if not (_is_vertex_gemini(spec) and _wants_priority(escalated)):
            return None
        # Priority PayGo only, never Provisioned Throughput.
        merge_extra_headers(
            params,
            {
                "X-Vertex-AI-LLM-Request-Type": "shared",
                "X-Vertex-AI-LLM-Shared-Request-Type": "priority",
            },
        )
        return "priority"


def vertex_served_tier(traffic_type: str) -> str | None:
    """The tier a Vertex ``usageMetadata.trafficType`` was billed at; None is the default tier."""
    tier = traffic_type.lower().removeprefix("on_demand").removeprefix("_")
    return tier or None


class TogetherAiProvider(Provider):
    @override
    def apply_reasoning(
        self, spec: ModelSpec, effort: str | None, params: dict[str, Any]
    ) -> None:
        super().apply_reasoning(spec, effort, params)
        # Kimi only streams reasoning_content when reasoning is on; litellm
        # forwards extra_body and remaps the returned `reasoning` field.
        if spec.model.startswith("together_ai/moonshotai/Kimi-K"):
            merge_extra_body(params, {"reasoning": {"enabled": True}})


class FireworksProvider(Provider):
    @override
    def apply_prompt_caching(
        self,
        cache_key: str,
        _messages: list[dict[str, Any]],
        params: dict[str, Any],
    ) -> None:
        # Fireworks caches per replica and routes on this header, so without it a
        # conversation's turns spread over replicas and miss each other's prefix.
        merge_extra_headers(params, {"x-session-affinity": cache_key})

    @override
    def escalates_on(self, _spec: ModelSpec, exc: Exception) -> bool:
        # 503 is Fireworks' load shedding, which Priority is admitted ahead of. A 429
        # is the account's rate limit, which Priority does not lift.
        return getattr(exc, "status_code", None) == 503

    @override
    def apply_service_tier(
        self, _spec: ModelSpec, escalated: bool, params: dict[str, Any]
    ) -> str | None:
        if not _wants_priority(escalated):
            return None
        # extra_body: litellm's fireworks_ai param list lacks service_tier.
        merge_extra_body(params, {"service_tier": "priority"})
        return "priority"


_PROVIDERS: Final[dict[str, Provider]] = {
    "anthropic": AnthropicProvider(),
    "openai": OpenAIProvider(),
    "xai": XaiProvider(),
    "vertex_ai": VertexProvider(),
    "together_ai": TogetherAiProvider(),
    "fireworks_ai": FireworksProvider(),
}

_FALLBACK_PROVIDER: Final = Provider()


def provider_for(spec: ModelSpec) -> Provider:
    return _PROVIDERS.get(spec.provider, _FALLBACK_PROVIDER)


PROXY_PLACEHOLDER_KEY = "model_api_key"
"""Key sent through a proxy that needs none, since litellm refuses to send no key."""


def proxy_api_base(provider: Provider) -> str | None:
    """The proxy's OpenAI-compatible endpoint when KAROTTE_PROXY_URL routes ``provider``."""
    proxy = os.environ.get("KAROTTE_PROXY_URL")
    if proxy and provider.proxied:
        return f"{proxy.rstrip('/')}/v1"
    return None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
    if value is not None:
        _logger.warning(
            "unexpected token count type %s: %r", type(value).__name__, value
        )
    return None


def cache_tokens(usage: object) -> tuple[int | None, int | None]:
    """Cache read/write token counts from a litellm usage object, whichever of
    the Anthropic, OpenAI-shaped or flat-Together shapes it carries."""
    read = _int_or_none(getattr(usage, "cache_read_input_tokens", None))
    write = _int_or_none(getattr(usage, "cache_creation_input_tokens", None))
    details = getattr(usage, "prompt_tokens_details", None)
    if read is None:
        read = _int_or_none(getattr(details, "cached_tokens", None))
    if read is None:
        read = _int_or_none(getattr(usage, "cached_tokens", None))
    if write is None:
        write = _int_or_none(getattr(details, "cache_write_tokens", None))
    return read, write


def merge_extra_headers(params: dict[str, Any], headers: dict[str, str]) -> None:
    """Merge headers into params' extra_headers without clobbering existing ones."""
    existing = params.get("extra_headers") or {}
    assert isinstance(existing, dict)
    params["extra_headers"] = {**existing, **headers}


def merge_extra_body(params: dict[str, Any], body: dict[str, Any]) -> None:
    """Merge fields into params' extra_body without clobbering existing ones."""
    existing = params.get("extra_body") or {}
    assert isinstance(existing, dict)
    params["extra_body"] = {**existing, **body}


def _add_inline_cache_control(messages: list[dict[str, Any]]) -> None:
    """Add inline cache_control to the last message for direct Anthropic API calls.

    See: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
    """
    for message in reversed(messages):
        if isinstance(message["content"], str):
            message["content"] = [{"type": "text", "text": message["content"]}]
        if isinstance(message["content"], list) and message["content"]:
            message["content"][-1]["cache_control"] = {  # pyright: ignore[reportArgumentType]
                "type": "ephemeral",
                "ttl": "1h",
            }
            break


_DEEPSEEK_KNOWN_TOKENS = [
    "<｜tool▁call▁begin｜>",
    "<｜tool▁call▁end｜>",
    "<｜tool▁sep｜>",
    "<｜tool▁outputs▁begin｜>",
    "<｜tool▁outputs▁end｜>",
]


def _strip_deepseek_tokens(text: str) -> str:
    """Remove the special tokens DeepSeek leaks into tool call arguments.

    Only the known ones, to avoid false positives.
    """
    for token in _DEEPSEEK_KNOWN_TOKENS:
        text = text.replace(token, "")
    return text


def _unwrap_json_string(value: object) -> object:
    """The string inside a JSON-encoded string, or ``value`` unchanged."""
    if not isinstance(value, str):
        return value
    if not (value.startswith('"') and value.endswith('"')):
        return value
    try:
        inner = json.loads(value)
    except json.JSONDecodeError:
        return value
    return inner if isinstance(inner, str) else value


def _unwrap_double_encoding(args: dict[str, Any]) -> dict[str, Any]:
    """Undo xAI's double JSON-encoding of string tool-call arguments.

    Grok emits every *string* argument as a JSON-encoded string, so
    ``{"file_path": "/workdir/x"}`` arrives as ``{"file_path": "\\"/workdir/x\\""}``
    -- a path whose first character is a quote, which every absolute-path check
    rejects. Numbers are unaffected.
    """
    unwrapped = {k: _unwrap_json_string(v) for k, v in args.items()}
    string_keys = [k for k, v in args.items() if isinstance(v, str)]
    # One unchanged string means this call is not double-encoded at all.
    if not string_keys or any(unwrapped[k] == args[k] for k in string_keys):
        return args
    return unwrapped


_STRAY_QUOTE_PREFIX: Final = re.compile(r'^"?(?:/")?[\s;,]*(/[^\s"]*)$')


def _strip_stray_quote_prefix(value: object) -> object:
    """Drop the quote-and-separator debris Grok glues onto an absolute path.

    The same double-encoding that produces ``"\\"/workdir/x\\""`` also misplaces
    the closing quote, leaving values like ``";/tmp/a.png`` or ``", /workdir/b``.
    Only a value that is otherwise a single whitespace-free absolute path is
    touched, which leaves shell commands and prose alone.
    """
    if not isinstance(value, str):
        return value
    match = _STRAY_QUOTE_PREFIX.match(value)
    return match.group(1) if match else value


def _fix_xai_arguments(text: str) -> str:
    """Repair the ways Grok mangles the quoting of string tool-call arguments."""
    try:
        args = json.loads(text)
    except json.JSONDecodeError:
        # Malformed JSON is handled downstream in the tool-call executor.
        return text
    if not isinstance(args, dict):
        return text

    fixed = _unwrap_double_encoding(args)
    fixed = {k: _strip_stray_quote_prefix(v) for k, v in fixed.items()}
    return text if fixed == args else json.dumps(fixed)


TOOL_CALL_REPAIRS: Final[dict[ToolCallRepair, Callable[[str], str]]] = {
    "deepseek": _strip_deepseek_tokens,
    "xai": _fix_xai_arguments,
}


def repair_tool_calls(
    tool_calls: list[ChatCompletionMessageToolCall] | None,
    repair: Callable[[str], str],
) -> list[ChatCompletionMessageToolCall] | None:
    """Apply a provider repair to every tool call's arguments."""
    if not tool_calls:
        return tool_calls

    repaired = []
    for tc in tool_calls:
        if tc.function.arguments:
            tc = ChatCompletionMessageToolCall(
                id=tc.id,
                type=tc.type,
                function=Function(
                    name=tc.function.name,
                    arguments=repair(tc.function.arguments),
                ),
            )
        repaired.append(tc)
    return repaired
