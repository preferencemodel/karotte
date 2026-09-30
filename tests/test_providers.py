"""The per-provider request shaping ``karotte.schemas`` ships for downstream
callers that run their own litellm loop."""

import json
from typing import Any
from unittest.mock import Mock

import pytest
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

from karotte.model_spec import spec_for
from karotte.providers import (
    SERVICE_TIER_ENV,
    AnthropicProvider,
    FireworksProvider,
    OpenAIProvider,
    Provider,
    ProviderAuth,
    TogetherAiProvider,
    VertexProvider,
    XaiProvider,
    cache_tokens,
    provider_for,
    service_tier_mode,
    vertex_served_tier,
)


class TestRequestParams:
    def test_the_api_key_is_the_only_default_param(self):
        params = Provider().request_params(
            spec_for("openai/gpt-5.6"), ProviderAuth(api_key="k")
        )
        assert params == {"api_key": "k"}

    def test_vertex_leaves_project_and_location_to_litellm(self):
        params = VertexProvider().request_params(
            spec_for("vertex_ai/claude-sonnet-5"), ProviderAuth()
        )
        assert params == {}

    def test_litellm_reads_vertex_project_and_location_from_the_env(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("VERTEXAI_PROJECT", "proj")
        monkeypatch.setenv("VERTEXAI_LOCATION", "global")
        assert VertexBase.get_vertex_ai_project({}) == "proj"
        assert VertexBase.get_vertex_ai_location({}) == "global"

    def test_vertex_gemini_leaves_the_tier_to_apply_service_tier(self):
        params = VertexProvider().request_params(
            spec_for("vertex_ai/gemini-3.1-pro-preview"), ProviderAuth()
        )
        assert params == {}

    def test_vertex_sends_no_api_key(self):
        params = VertexProvider().request_params(
            spec_for("vertex_ai/gemini-3.1-pro-preview"), ProviderAuth(api_key="k")
        )
        assert "api_key" not in params


class TestPromptCaching:
    def test_openai_keys_the_cache_on_the_caller_s_key(self):
        params: dict[str, Any] = {}
        OpenAIProvider().apply_prompt_caching("review-7", [], params)
        assert params == {"extra_body": {"prompt_cache_key": "review-7"}}

    def test_xai_pins_routing_on_the_caller_s_key(self):
        params: dict[str, Any] = {}
        XaiProvider().apply_prompt_caching("review-7", [], params)
        assert params == {"extra_headers": {"x-grok-conv-id": "review-7"}}

    def test_anthropic_marks_the_last_message(self):
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
        ]
        AnthropicProvider().apply_prompt_caching("review-7", messages, {})
        assert messages[0]["content"] == "s"
        assert messages[-1]["content"][-1]["cache_control"] == {
            "type": "ephemeral",
            "ttl": "1h",
        }

    @pytest.mark.parametrize("provider", [TogetherAiProvider(), Provider()])
    def test_implicit_cache_providers_touch_nothing(self, provider: Provider):
        messages: list[dict[str, Any]] = [{"role": "user", "content": "u"}]
        params: dict[str, Any] = {}
        provider.apply_prompt_caching("review-7", messages, params)
        assert params == {}
        assert "cache_control" not in json.dumps(messages)

    def test_fireworks_pins_routing_on_the_caller_s_key(self):
        params: dict[str, Any] = {}
        FireworksProvider().apply_prompt_caching("review-7", [], params)
        assert params == {"extra_headers": {"x-session-affinity": "review-7"}}


_GEMINI = spec_for("vertex_ai/gemini-3.1-pro-preview")
_VERTEX_CLAUDE = spec_for("vertex_ai/claude-sonnet-5")
_GLM = spec_for("fireworks_ai/accounts/fireworks/models/glm-5p3")
_VERTEX_PRIORITY_HEADERS = {
    "X-Vertex-AI-LLM-Request-Type": "shared",
    "X-Vertex-AI-LLM-Shared-Request-Type": "priority",
}
_MODES = [
    (None, False, None),
    (None, True, None),
    ("auto", False, None),
    ("auto", True, "priority"),
    ("priority", False, "priority"),
    (" PRIORITY ", False, "priority"),
    ("standard", True, None),
    ("bogus", True, None),
]


def _set_mode(monkeypatch: pytest.MonkeyPatch, mode: str | None) -> None:
    if mode is None:
        monkeypatch.delenv(SERVICE_TIER_ENV, raising=False)
    else:
        monkeypatch.setenv(SERVICE_TIER_ENV, mode)


class TestServiceTier:
    @pytest.mark.parametrize(
        ("mode", "expected"),
        [(None, None), ("auto", "auto"), (" Priority", "priority"), ("", None)],
    )
    def test_mode_comes_from_one_env_var(
        self, monkeypatch: pytest.MonkeyPatch, mode: str | None, expected: str | None
    ):
        _set_mode(monkeypatch, mode)
        assert service_tier_mode() == expected

    def test_unknown_mode_warns_once_and_is_unset(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        _set_mode(monkeypatch, "turbo")
        assert service_tier_mode() is None
        assert service_tier_mode() is None
        assert [
            r.getMessage() for r in caplog.records if "turbo" in r.getMessage()
        ] == [f"unknown {SERVICE_TIER_ENV}='turbo', ignoring it"]

    def test_the_old_fireworks_var_is_gone(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("KAROTTE_FIREWORKS_SERVICE_TIER", "priority")
        params: dict[str, Any] = {}
        assert FireworksProvider().apply_service_tier(_GLM, False, params) is None
        assert params == {}

    def test_escalation_triggers(self):
        assert FireworksProvider().escalates_on(_GLM, Mock(status_code=503))
        assert not FireworksProvider().escalates_on(_GLM, Mock(status_code=429))
        assert not FireworksProvider().escalates_on(_GLM, Mock(status_code=500))
        assert VertexProvider().escalates_on(_GEMINI, Mock(status_code=429))
        assert not VertexProvider().escalates_on(_GEMINI, Mock(status_code=503))
        assert not VertexProvider().escalates_on(_VERTEX_CLAUDE, Mock(status_code=429))
        assert not Provider().escalates_on(_GLM, Mock(status_code=503))

    @pytest.mark.parametrize(("mode", "escalated", "tier"), _MODES)
    def test_fireworks_asks_for_priority_per_the_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mode: str | None,
        escalated: bool,
        tier: str | None,
    ):
        _set_mode(monkeypatch, mode)
        params: dict[str, Any] = {"extra_body": {"reasoning": True}}
        assert FireworksProvider().apply_service_tier(_GLM, escalated, params) == tier
        expected: dict[str, Any] = {"reasoning": True}
        if tier:
            expected["service_tier"] = tier
        assert params == {"extra_body": expected}

    @pytest.mark.parametrize(("mode", "escalated", "tier"), _MODES)
    def test_vertex_gemini_asks_for_priority_per_the_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mode: str | None,
        escalated: bool,
        tier: str | None,
    ):
        _set_mode(monkeypatch, mode)
        params: dict[str, Any] = {"extra_headers": {"x-other": "1"}}
        assert VertexProvider().apply_service_tier(_GEMINI, escalated, params) == tier
        expected = {"x-other": "1"}
        if tier:
            expected |= _VERTEX_PRIORITY_HEADERS
        assert params == {"extra_headers": expected}

    def test_vertex_non_gemini_never_asks_for_priority(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _set_mode(monkeypatch, "priority")
        params: dict[str, Any] = {}
        assert VertexProvider().apply_service_tier(_VERTEX_CLAUDE, True, params) is None
        assert params == {}

    def test_other_providers_leave_the_tier_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _set_mode(monkeypatch, "priority")
        params: dict[str, Any] = {}
        assert Provider().apply_service_tier(_GLM, True, params) is None
        assert params == {}

    @pytest.mark.parametrize(
        ("traffic_type", "tier"),
        [
            ("ON_DEMAND_PRIORITY", "priority"),
            ("ON_DEMAND", None),
            ("ON_DEMAND_FLEX", "flex"),
            ("PROVISIONED_THROUGHPUT", "provisioned_throughput"),
        ],
    )
    def test_vertex_served_tier(self, traffic_type: str, tier: str | None):
        assert vertex_served_tier(traffic_type) == tier


class TestProviderFor:
    def test_together_gets_its_own_provider(self):
        spec = spec_for("together_ai/zai-org/GLM-5.3")
        assert type(provider_for(spec)) is TogetherAiProvider

    def test_fireworks_gets_its_own_provider(self):
        spec = spec_for("fireworks_ai/accounts/fireworks/models/glm-5p3")
        assert type(provider_for(spec)) is FireworksProvider


class TestCacheTokens:
    def test_reads_anthropic_flat_fields(self):
        usage = Mock(
            cache_read_input_tokens=100,
            cache_creation_input_tokens=25,
            prompt_tokens_details=None,
        )
        assert cache_tokens(usage) == (100, 25)

    def test_reads_openai_shaped_details(self):
        usage = Mock(
            cache_read_input_tokens=None,
            cache_creation_input_tokens=None,
            prompt_tokens_details=Mock(cached_tokens=4096, cache_write_tokens=None),
        )
        assert cache_tokens(usage) == (4096, None)

    def test_reads_gpt56_cache_write_tokens(self):
        usage = Mock(
            cache_read_input_tokens=None,
            cache_creation_input_tokens=None,
            prompt_tokens_details=Mock(cached_tokens=2048, cache_write_tokens=512),
        )
        assert cache_tokens(usage) == (2048, 512)

    def test_reads_together_s_flat_cached_tokens(self):
        """Together's non-reasoning models put ``cached_tokens`` at the top of
        ``usage`` with no details object at all."""
        usage = Mock(
            cache_read_input_tokens=None,
            cache_creation_input_tokens=None,
            prompt_tokens_details=None,
            cached_tokens=777,
        )
        assert cache_tokens(usage) == (777, None)

    def test_prefers_anthropic_fields_over_details(self):
        usage = Mock(
            cache_read_input_tokens=100,
            cache_creation_input_tokens=25,
            prompt_tokens_details=Mock(cached_tokens=999, cache_write_tokens=999),
        )
        assert cache_tokens(usage) == (100, 25)

    def test_ignores_non_int_garbage(self):
        assert cache_tokens(Mock()) == (None, None)
