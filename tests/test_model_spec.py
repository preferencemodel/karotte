"""Tests for the model catalog: how an id resolves to a spec, and which
provider that spec selects."""

from typing import Any

import pytest

from karotte.model_catalog import CATALOG_MODEL_IDS
from karotte.model_spec import ReasoningEffort, api_key_env_var, spec_for
from karotte.providers import (
    AnthropicProvider,
    OpenAIProvider,
    Provider,
    VertexProvider,
    XaiProvider,
    provider_for,
)


class TestProviderResolution:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("claude-opus-5", "anthropic"),
            ("anthropic/claude-opus-5", "anthropic"),
            ("vertex_ai/claude-opus-4-6", "vertex_ai"),
            ("openai/gpt-5.6", "openai"),
            ("together_ai/moonshotai/Kimi-K3", "together_ai"),
            ("fireworks_ai/accounts/fireworks/models/glm-5p3", "fireworks_ai"),
            ("pt/checkpoint", "pt"),
            ("gpt-5.6", ""),
            ("some-unknown-model", ""),
        ],
    )
    def test_provider(self, model: str, expected: str):
        assert spec_for(model).provider == expected

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("claude-opus-5", AnthropicProvider),
            ("openai/gpt-5.6", OpenAIProvider),
            ("xai/grok-4.7", XaiProvider),
            ("xai/grok-4.6", XaiProvider),
            ("vertex_ai/gemini-3.1-pro-preview", VertexProvider),
            # A provider nobody registered still authenticates and proxies.
            ("mistral/mistral-medium-3.5", Provider),
            ("some-unknown-model", Provider),
        ],
    )
    def test_provider_for(self, model: str, expected: type[Provider]):
        assert type(provider_for(spec_for(model))) is expected

    def test_only_anthropic_and_vertex_skip_the_proxy(self):
        assert not provider_for(spec_for("claude-opus-5")).proxied
        assert not provider_for(spec_for("vertex_ai/gemini-3.1-pro-preview")).proxied
        assert provider_for(spec_for("openai/gpt-5.6")).proxied
        assert provider_for(spec_for("some-unknown-model")).proxied


class TestLitellmModel:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            # Every bare Claude id names its provider, so litellm never has to
            # know a new release before we can run it.
            ("claude-fable-5-1", "anthropic/claude-fable-5-1"),
            ("claude-fable-5", "anthropic/claude-fable-5"),
            ("claude-sonnet-5", "anthropic/claude-sonnet-5"),
            ("claude-haiku-5-5", "anthropic/claude-haiku-5-5"),
            ("claude-opus-5", "anthropic/claude-opus-5"),
            ("claude-opus-4-5-20251101", "anthropic/claude-opus-4-5-20251101"),
            ("anthropic/claude-opus-5", "anthropic/claude-opus-5"),
            ("vertex_ai/claude-opus-4-6", "vertex_ai/claude-opus-4-6"),
            ("openai/gpt-5.6", "openai/gpt-5.6"),
            # GPT-6 goes through the Responses bridge litellm only wires for gpt-5.
            ("openai/gpt-6-astra", "openai/responses/gpt-6-astra"),
            ("openai/gpt-6-sol", "openai/responses/gpt-6-sol"),
            ("openai/gpt-6-luna", "openai/responses/gpt-6-luna"),
            # Muse Spark only returns reasoning summaries on the Responses API.
            ("meta/muse-spark-1.3", "meta/responses/muse-spark-1.3"),
            ("meta/muse-spark-1.2", "meta/responses/muse-spark-1.2"),
            (
                "fireworks_ai/accounts/fireworks/models/glm-5p3",
                "fireworks_ai/accounts/fireworks/models/glm-5p3",
            ),
        ],
    )
    def test_litellm_model(self, model: str, expected: str):
        assert spec_for(model).litellm_model == expected


class TestExtraAllowedOpenAIParams:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("meta/muse-spark-1.3", ("reasoning_effort",)),
            # litellm's cost map has no responses/gpt-6-* bridge names.
            ("openai/gpt-6-astra", ("reasoning_effort",)),
            ("openai/gpt-6-sol", ("reasoning_effort",)),
            ("openai/gpt-6-luna", ("reasoning_effort",)),
            ("openai/gpt-5.6", ()),
            # litellm's cost map predates 3.8 Flash; 3.7 is in it.
            ("vertex_ai/gemini-3.8-flash", ("reasoning_effort",)),
            ("vertex_ai/gemini-3.7-flash", ()),
            # litellm's together_ai param list has no reasoning_effort.
            ("together_ai/deepseek-ai/DeepSeek-V4-Pro-0813", ("reasoning_effort",)),
            ("together_ai/zai-org/GLM-5.3", ("reasoning_effort",)),
            (
                "fireworks_ai/accounts/fireworks/models/glm-5p3",
                ("reasoning_effort",),
            ),
            (
                "fireworks_ai/accounts/fireworks/models/glm-5p3-flash",
                ("reasoning_effort",),
            ),
            ("together_ai/Qwen/Qwen3.8-2.4T-A95B", ("reasoning_effort",)),
            ("together_ai/moonshotai/Kimi-K3", ("reasoning_effort",)),
            ("together_ai/zai-org/GLM-5.2", ()),
            ("together_ai/moonshotai/Kimi-K2.6", ()),
            ("claude-opus-5", ()),
        ],
    )
    def test_extra_allowed_openai_params(self, model: str, expected: tuple[str, ...]):
        assert spec_for(model).extra_allowed_openai_params == expected


class TestMaxOutputTokens:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("openai/gpt-6-astra", 128000),
            ("openai/gpt-6-sol", 128000),
            ("openai/gpt-6-luna", 128000),
            ("openai/gpt-5.6", 128000),
            ("claude-fable-5-1", 128000),
            ("claude-haiku-5-5", 128000),
            ("vertex_ai/gemini-3.8-flash", 64000),
            ("together_ai/Qwen/Qwen3.8-2.4T-A95B", 128000),
            ("together_ai/zai-org/GLM-5.3", 64000),
            # Fireworks documents no output ceiling for it, so it keeps the default.
            ("fireworks_ai/accounts/fireworks/models/glm-5p3-flash", 64000),
        ],
    )
    def test_max_output_tokens(self, model: str, expected: int):
        assert spec_for(model).max_output_tokens == expected


class TestSamplingParams:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("claude-fable-5-1", False),
            ("claude-fable-5", False),
            ("claude-opus-4-7", False),
            ("claude-opus-4-8", False),
            ("claude-opus-5", False),
            ("claude-sonnet-5", False),
            ("claude-haiku-5-5", False),
            ("anthropic/claude-fable-5", False),
            ("anthropic/claude-opus-4-6", True),
            ("claude-opus-4-6", True),
            ("claude-haiku-4-5-20251001", True),
            ("openai/gpt-5.6", True),
            ("openai/gpt-6-astra", False),
            ("some-unknown-model", True),
        ],
    )
    def test_supports_sampling_params(self, model: str, expected: bool):
        assert spec_for(model).supports_sampling_params == expected


class TestReasoningEffort:
    """The lowest and highest effort each model's API accepts. Both ends are
    only sent when a run asks for them by name."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("claude-fable-5-1", ("low", "max")),
            ("claude-fable-5", ("low", "max")),
            ("claude-opus-5", ("low", "max")),
            ("claude-sonnet-5", ("low", "max")),
            ("claude-haiku-5-5", ("low", "max")),
            ("claude-opus-4-8", ("low", "max")),
            ("claude-opus-4-7", ("low", "max")),
            ("claude-opus-4-6", ("low", "max")),
            ("claude-sonnet-4-6", ("low", "max")),
            ("anthropic/claude-opus-5", ("low", "max")),
            ("vertex_ai/claude-opus-4-6", ("low", "max")),
            # litellm would turn effort into legacy budget_tokens thinking here.
            ("claude-opus-4-5-20251101", (None, None)),
            ("claude-sonnet-4-5-20250929", (None, None)),
            ("claude-haiku-4-5-20251001", (None, None)),
            # gpt-5.5 and up reject "minimal".
            ("openai/gpt-6-astra", ("low", "max")),
            ("openai/gpt-6-sol", ("low", "max")),
            ("openai/gpt-6-luna", ("low", "max")),
            ("openai/gpt-5.6", ("low", "max")),
            ("openai/gpt-5.6-sol", ("low", "max")),
            ("openai/gpt-5.6-2026-04-01", ("low", "max")),
            ("openai/gpt-5.5", ("low", "xhigh")),
            ("openai/gpt-5.5-pro", ("low", "xhigh")),
            ("openai/gpt-5.4", ("low", "xhigh")),
            ("openai/gpt-5.3-codex", ("low", "high")),
            ("openai/gpt-5.2", ("low", "xhigh")),
            ("openai/gpt-5.1", ("low", "high")),
            ("openai/gpt-5", ("minimal", "high")),
            ("gpt-5.5", ("low", "xhigh")),
            # The o-series predates "minimal" and stops at "high".
            ("openai/o1", ("low", "high")),
            ("openai/o3", ("low", "high")),
            # Siblings that reject the parameter their family takes.
            ("openai/o1-mini", (None, None)),
            ("openai/o1-preview", (None, None)),
            # thinkingLevel has no xhigh or max, and 3.7 Flash dropped minimal.
            ("vertex_ai/gemini-3.1-pro-preview", ("low", "high")),
            ("vertex_ai/gemini-3.8-flash", ("low", "high")),
            ("vertex_ai/gemini-3.7-flash", ("low", "high")),
            ("gemini/gemini-3.8-flash", ("low", "high")),
            ("gemini/gemini-3.1-pro-preview", ("low", "high")),
            # litellm has no effort mapping for ids outside its cost map.
            ("gemini/gemini-3-pro", (None, None)),
            ("vertex_ai/gemini-3-pro", (None, None)),
            ("together_ai/moonshotai/Kimi-K2.6", (None, None)),
            ("together_ai/deepseek-ai/DeepSeek-V3.1", (None, None)),
            ("together_ai/zai-org/GLM-5.2", (None, None)),
            # Together rounds low and medium up to high on V4-Pro; GLM-5.3 takes low.
            ("together_ai/deepseek-ai/DeepSeek-V4-Pro-0813", ("high", "max")),
            ("together_ai/zai-org/GLM-5.3", ("low", "max")),
            ("fireworks_ai/accounts/fireworks/models/glm-5p3", ("low", "max")),
            ("fireworks_ai/accounts/fireworks/models/glm-5p3-flash", ("low", "max")),
            ("together_ai/Qwen/Qwen3.8-2.4T-A95B", ("low", "xhigh")),
            # Kimi K3 takes low, high and max; K2.x answers 400 to the parameter.
            ("together_ai/moonshotai/Kimi-K3", ("low", "max")),
            ("minimax/MiniMax-M3", (None, None)),
            # xhigh is 4.6 and up; xAI documents no range for 4.3.
            ("xai/grok-4.7", ("low", "xhigh")),
            ("xai/grok-4.6", ("low", "xhigh")),
            ("xai/grok-4.5", (None, None)),
            ("xai/grok-4.3", (None, None)),
            # max shipped for 1.3 only.
            ("meta/muse-spark-1.2", ("minimal", "xhigh")),
            ("meta/muse-spark-1.3", ("minimal", "max")),
            ("mistral/mistral-medium-3.5", (None, None)),
            ("some-unknown-model", (None, None)),
        ],
    )
    def test_effort_range(self, model: str, expected: tuple[str | None, str | None]):
        spec = spec_for(model)
        assert (spec.min_reasoning_effort, spec.max_reasoning_effort) == expected

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("claude-fable-5-1", ("low", "medium", "high", "xhigh", "max")),
            ("claude-opus-5", ("low", "medium", "high", "xhigh", "max")),
            ("claude-haiku-5-5", ("low", "medium", "high", "xhigh", "max")),
            ("claude-opus-4-7", ("low", "medium", "high", "xhigh", "max")),
            # The 4.6 generation predates xhigh.
            ("claude-opus-4-6", ("low", "medium", "high", "max")),
            ("claude-sonnet-4-6", ("low", "medium", "high", "max")),
            ("openai/o1", ("low", "medium", "high")),
            ("openai/o3", ("low", "medium", "high")),
            ("openai/gpt-5", ("minimal", "low", "medium", "high")),
            ("openai/gpt-5.1", ("low", "medium", "high")),
            ("openai/gpt-5.2", ("low", "medium", "high", "xhigh")),
            ("openai/gpt-5.3-codex", ("low", "medium", "high")),
            ("openai/gpt-5.4", ("low", "medium", "high", "xhigh")),
            ("openai/gpt-5.5", ("low", "medium", "high", "xhigh")),
            ("openai/gpt-5.6-sol", ("low", "medium", "high", "xhigh", "max")),
            ("openai/gpt-6-astra", ("low", "medium", "high", "xhigh", "max")),
            ("openai/gpt-6-sol", ("low", "medium", "high", "xhigh", "max")),
            ("openai/gpt-6-luna", ("low", "medium", "high", "xhigh", "max")),
            ("vertex_ai/gemini-3.1-pro-preview", ("low", "medium", "high")),
            ("vertex_ai/gemini-3.8-flash", ("low", "medium", "high")),
            ("gemini/gemini-3.7-flash", ("low", "medium", "high")),
            ("xai/grok-4.7", ("low", "medium", "high", "xhigh")),
            ("xai/grok-4.6", ("low", "medium", "high", "xhigh")),
            ("meta/muse-spark-1.2", ("minimal", "low", "medium", "high", "xhigh")),
            (
                "meta/muse-spark-1.3",
                ("minimal", "low", "medium", "high", "xhigh", "max"),
            ),
            ("together_ai/deepseek-ai/DeepSeek-V4-Pro-0813", ("high", "max")),
            ("together_ai/zai-org/GLM-5.3", ("low", "high", "max")),
            # Fireworks spells the same checkpoint glm-5p3.
            (
                "fireworks_ai/accounts/fireworks/models/glm-5p3",
                ("low", "high", "max"),
            ),
            (
                "fireworks_ai/accounts/fireworks/models/glm-5p3-flash",
                ("low", "high", "max"),
            ),
            ("together_ai/Qwen/Qwen3.8-2.4T-A95B", ("low", "medium", "xhigh")),
            ("together_ai/moonshotai/Kimi-K3", ("low", "high", "max")),
            ("together_ai/moonshotai/Kimi-K2.6", ()),
            ("together_ai/zai-org/GLM-5.2", ()),
            ("minimax/MiniMax-M3", ()),
            ("xai/grok-4.5", ()),
            ("claude-opus-4-5-20251101", ()),
            ("some-unknown-model", ()),
        ],
    )
    def test_effort_levels(self, model: str, expected: tuple[str, ...]):
        assert spec_for(model).reasoning_effort_levels == expected

    @pytest.mark.parametrize("model", CATALOG_MODEL_IDS)
    def test_range_is_the_ends_of_the_levels(self, model: str):
        spec = spec_for(model)
        levels = spec.reasoning_effort_levels
        assert spec.min_reasoning_effort == (levels[0] if levels else None)
        assert spec.max_reasoning_effort == (levels[-1] if levels else None)

    @pytest.mark.parametrize(
        ("model", "level", "expected"),
        [
            ("claude-opus-5", None, None),
            ("claude-opus-5", "min", "low"),
            ("claude-opus-5", "max", "max"),
            ("openai/gpt-5.2", "min", "low"),
            ("openai/gpt-5.2", "max", "xhigh"),
            ("together_ai/moonshotai/Kimi-K2.6", "min", None),
            ("together_ai/moonshotai/Kimi-K2.6", "max", None),
            ("together_ai/deepseek-ai/DeepSeek-V4-Pro-0813", "min", "high"),
            ("together_ai/zai-org/GLM-5.3", "max", "max"),
            ("together_ai/Qwen/Qwen3.8-2.4T-A95B", "max", "xhigh"),
            ("together_ai/moonshotai/Kimi-K3", "min", "low"),
            ("together_ai/moonshotai/Kimi-K3", "max", "max"),
            # A provider's own level name passes through untouched.
            ("claude-opus-5", "medium", "medium"),
            ("openai/gpt-5.5", "xhigh", "xhigh"),
        ],
    )
    def test_reasoning_effort_value(
        self, model: str, level: ReasoningEffort | None, expected: str | None
    ):
        assert spec_for(model).reasoning_effort_value(level) == expected


class TestToolCallRepairs:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("together_ai/deepseek-ai/DeepSeek-V3.1", ("deepseek",)),
            ("together_ai/deepseek-ai/DeepSeek-V4-Pro-0813", ("deepseek",)),
            ("together_ai/zai-org/GLM-5.3", ()),
            ("together_ai/Qwen/Qwen3.8-2.4T-A95B", ()),
            ("xai/grok-4.7", ("xai",)),
            ("xai/grok-4.6", ("xai",)),
            ("openai/gpt-5.6", ()),
            ("openai/gpt-6-astra", ()),
            ("claude-opus-5", ()),
        ],
    )
    def test_tool_call_repairs(self, model: str, expected: tuple[str, ...]):
        assert spec_for(model).tool_call_repairs == expected


class TestApiKeyPolicy:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("claude-opus-5", True),
            ("openai/gpt-5.6", True),
            ("vertex_ai/gemini-3.1-pro-preview", False),
            ("__karotte_special__/training", False),
            ("pt/checkpoint", False),
        ],
    )
    def test_requires_api_key(self, model: str, expected: bool):
        assert spec_for(model).requires_api_key == expected

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("claude-fable-5", "ANTHROPIC_API_KEY"),
            ("anthropic/claude-opus-4-5", "ANTHROPIC_API_KEY"),
            ("openai/gpt-5.5", "OPENAI_API_KEY"),
            ("gemini/gemini-3-pro-preview", "GEMINI_API_KEY"),
            ("mistral/mistral-medium-3.5", "MISTRAL_API_KEY"),
            ("xai/grok-4.3", "XAI_API_KEY"),
            ("together_ai/moonshotai/Kimi-K3", "TOGETHERAI_API_KEY"),
            ("some-model", "ANTHROPIC_API_KEY"),
            ("vertex_ai/gemini-3-pro-preview", None),
        ],
    )
    def test_api_key_env_var(self, model: str, expected: str | None):
        assert api_key_env_var(model) == expected


def test_unknown_model_gets_safe_defaults():
    spec = spec_for("some-unknown-model")

    assert spec.max_output_tokens == 64000
    assert spec.min_reasoning_effort is None
    assert spec.max_reasoning_effort is None
    assert spec.tool_call_repairs == ()
    assert spec.supports_sampling_params is True


class TestClaudeFamilyRules:
    """Claude 4.6 and later are one family for karotte's purposes, matched by
    prefix, so a point release or dated snapshot inherits everything without a
    code change. The 4.5 generation takes ``budget_tokens`` and stays out."""

    @pytest.mark.parametrize(
        "model",
        [
            "claude-sonnet-5-1",
            "claude-opus-5-2",
            "claude-fable-5-3",
            "claude-haiku-5-6",
            "claude-opus-4-8-20260901",
            "anthropic/claude-sonnet-5-1",
            "vertex_ai/claude-sonnet-5-1",
        ],
    )
    def test_a_point_release_inherits_its_family(self, model: str):
        spec = spec_for(model)
        assert spec.max_output_tokens == 128000
        assert spec.reasoning_request == {
            "thinking": {"type": "adaptive", "display": "summarized"}
        }
        assert (spec.min_reasoning_effort, spec.max_reasoning_effort) == ("low", "max")

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-5-20251101",
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-20250514",
            "claude-opus-4-1",
        ],
    )
    def test_the_previous_generation_stays_out(self, model: str):
        spec = spec_for(model)
        assert spec.max_output_tokens == 64000
        assert spec.reasoning_request == {}
        assert (spec.min_reasoning_effort, spec.max_reasoning_effort) == (None, None)
        assert spec.supports_sampling_params is True


class TestDisplayNames:
    """The names a picker shows: the model's, derived from the id so a new model
    needs no second table, and
    its provider's, from a per-provider table."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("claude-fable-5-1", "Fable 5.1"),
            ("claude-fable-5", "Fable 5"),
            ("claude-opus-5", "Opus 5"),
            ("claude-opus-4-8", "Opus 4.8"),
            ("claude-sonnet-4-6", "Sonnet 4.6"),
            ("claude-haiku-5-5", "Haiku 5.5"),
            # Dated snapshots drop the date.
            ("claude-opus-4-5-20251101", "Opus 4.5"),
            ("claude-haiku-4-5-20251001", "Haiku 4.5"),
            ("anthropic/claude-opus-5", "Opus 5"),
            ("vertex_ai/claude-opus-4-6", "Opus 4.6"),
            # A point release nobody has listed yet still reads right.
            ("claude-sonnet-5-1", "Sonnet 5.1"),
            ("openai/o1", "O1"),
            ("openai/o3", "O3"),
            ("openai/gpt-5", "GPT-5"),
            ("openai/gpt-5.6", "GPT-5.6"),
            ("openai/gpt-5.6-sol", "GPT-5.6 Sol"),
            ("openai/gpt-6-astra", "GPT-6 Astra"),
            ("openai/gpt-6-sol", "GPT-6 Sol"),
            ("openai/gpt-6-luna", "GPT-6 Luna"),
            ("together_ai/moonshotai/Kimi-K3", "Kimi K3"),
            ("together_ai/moonshotai/Kimi-K2.6", "Kimi K2.6"),
            ("together_ai/deepseek-ai/DeepSeek-V3.1", "DeepSeek V3.1"),
            ("together_ai/deepseek-ai/DeepSeek-V4-Pro-0813", "DeepSeek V4 Pro 0813"),
            ("together_ai/zai-org/GLM-5.2", "GLM 5.2"),
            ("together_ai/zai-org/GLM-5.3", "GLM 5.3"),
            # Fireworks' spelling reads back as the vendor's.
            ("fireworks_ai/accounts/fireworks/models/glm-5p3", "GLM 5.3"),
            ("fireworks_ai/accounts/fireworks/models/glm-5p3-flash", "GLM 5.3 Flash"),
            ("fireworks_ai/accounts/fireworks/models/kimi-k2p5", "Kimi K2.5"),
            ("together_ai/Qwen/Qwen3.8-2.4T-A95B", "Qwen3.8 2.4T A95B"),
            ("minimax/MiniMax-M2.5", "MiniMax M2.5"),
            ("vertex_ai/gemini-3.8-flash", "Gemini 3.8 Flash"),
            ("vertex_ai/gemini-3.7-flash", "Gemini 3.7 Flash"),
            ("vertex_ai/gemini-3.1-pro-preview", "Gemini 3.1 Pro Preview"),
            ("xai/grok-4.7", "Grok 4.7"),
            ("xai/grok-4.6", "Grok 4.6"),
            ("meta/muse-spark-1.3", "Muse Spark 1.3"),
            ("__karotte_special__/training", "Training"),
            ("pt/checkpoint-7", "Checkpoint 7"),
            ("some-unknown-model", "Some Unknown Model"),
        ],
    )
    def test_model_display_name(self, model: str, expected: str):
        assert spec_for(model).model_display_name == expected

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("claude-opus-5", "Anthropic"),
            ("anthropic/claude-opus-5", "Anthropic"),
            ("vertex_ai/claude-opus-4-6", "Google Vertex AI"),
            ("vertex_ai/gemini-3.7-flash", "Google Vertex AI"),
            ("openai/gpt-5.6", "OpenAI"),
            ("together_ai/moonshotai/Kimi-K3", "Together AI"),
            ("fireworks_ai/accounts/fireworks/models/kimi-k2p5", "Fireworks AI"),
            ("xai/grok-4.6", "xAI"),
            ("minimax/MiniMax-M3", "MiniMax"),
            ("meta/muse-spark-1.3", "Meta"),
            ("mistral/mistral-medium-3.5", "Mistral"),
            ("__karotte_special__/training", "Training"),
            ("pt/checkpoint-7", "Training"),
            # A provider nobody has named yet reads as its slug, an unknown bare
            # id as nothing.
            ("newhost/some-model", "Newhost"),
            ("some-unknown-model", ""),
        ],
    )
    def test_provider_display_name(self, model: str, expected: str):
        assert spec_for(model).provider_display_name == expected


class TestReasoningRequest:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("meta/muse-spark-1.3", {"extra_body": {"reasoning_summary": "detailed"}}),
            ("openai/gpt-6-astra", {"extra_body": {"reasoning_summary": "detailed"}}),
            ("openai/gpt-6-sol", {"extra_body": {"reasoning_summary": "detailed"}}),
            ("openai/gpt-6-luna", {"extra_body": {"reasoning_summary": "detailed"}}),
            # litellm's string-to-Reasoning fold has no "max" branch, so every
            # model with a max level needs the summary to take the dict path.
            ("openai/gpt-5.6", {"extra_body": {"reasoning_summary": "detailed"}}),
            ("openai/gpt-5.6-sol", {"extra_body": {"reasoning_summary": "detailed"}}),
            ("openai/gpt-5.5", {}),
            ("xai/grok-4.6", {}),
            ("together_ai/Qwen/Qwen3.8-2.4T-A95B", {}),
        ],
    )
    def test_reasoning_request(self, model: str, expected: dict[str, Any]):
        assert spec_for(model).reasoning_request == expected
