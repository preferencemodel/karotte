"""Golden locks on every model-conditional code path in karotte.

Each golden covers one seam across the same model list: karotte's catalog plus
extra ids that reach branches no catalogued model does.

Regenerate after an intentional change with:

    UPDATE_GOLDEN=1 pytest tests/test_model_goldens.py

and review the golden diff carefully.
"""

from collections.abc import Callable
from dataclasses import asdict
from typing import Any, final, override

import litellm
import pytest
from litellm import CustomStreamWrapper
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)
from litellm.types.utils import Delta as LiteLlmDelta
from litellm.types.utils import Function as LiteLlmFunction
from litellm.utils import get_optional_params
from pydantic import ValidationError

from karotte.agents.builtin_source import BuiltinSource
from karotte.agents.models import resolve_model
from karotte.judges.rubric_judge import RubricJudge
from karotte.model_catalog import CATALOG_MODEL_IDS
from karotte.model_spec import ReasoningEffort
from karotte.schemas.chat import Message
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.transcript import MessageAddedEvent

type Golden = Callable[[str, Any], None]

# Ids that reach a karotte branch the catalog doesn't cover.
_EXTRA_MODELS = (
    "anthropic/claude-opus-5",
    "openai/gpt-5.3-codex",
    "openai/gpt-5.6-2026-04-01",
    "fireworks_ai/accounts/fireworks/models/kimi-k2p5",
    "mistral/mistral-medium-3.5",
    "gemini/gemini-3-pro",
    "vertex_ai/claude-opus-4-6",
    "pt/some-training-checkpoint",
    "some-unknown-model",
)

MODELS = CATALOG_MODEL_IDS + _EXTRA_MODELS


def _config(model: str, **overrides: Any) -> EvaluationRunConfig:
    return EvaluationRunConfig(
        run_id="golden-run",
        task_id="golden-task",
        model=model,
        model_api_key="test_key",
        **overrides,
    )


def _completion_params(model: str) -> dict[str, Any]:
    """Request params for a one-message, no-tool turn.

    Tool serialization is model-independent and already pinned by
    ``llm_completion_params.json``; leaving it out keeps the diff readable.
    """
    source = BuiltinSource(_config(model))
    return source.get_completion_params([Message(role="user", content="Go.")], [])


def test_golden_completion_params_by_model(
    monkeypatch: pytest.MonkeyPatch, assert_matches_golden: Golden
):
    """Model routing, max_tokens, reasoning effort, thinking display, provider
    request bodies, and prompt-caching placement, for every model."""
    monkeypatch.delenv("KAROTTE_PROXY_URL", raising=False)

    assert_matches_golden(
        "completion_params_by_model.json",
        {model: _completion_params(model) for model in MODELS},
    )


def test_golden_completion_params_by_model_with_proxy(
    monkeypatch: pytest.MonkeyPatch, assert_matches_golden: Golden
):
    """Which models get routed through KAROTTE_PROXY_URL."""
    monkeypatch.setenv("KAROTTE_PROXY_URL", "https://proxy.example/")

    assert_matches_golden(
        "completion_params_by_model_with_proxy.json",
        {model: _completion_params(model) for model in MODELS},
    )


_LITELLM_REASONING_KEYS = (
    "reasoning_effort",
    "thinking",
    "output_config",
    "thinkingConfig",
)

_REASONING_KEYS = ("reasoning_effort", "thinking")


def _litellm_reasoning_params(model: str, level: ReasoningEffort | None) -> Any:
    """What litellm makes of the effort karotte sends, or the error it raises.

    karotte leaves ``litellm.drop_params`` off, so a level a model does not
    accept aborts the run rather than being quietly dropped.
    """
    source = BuiltinSource(_config(model, reasoning_effort=level))
    params = source.get_completion_params([Message(role="user", content="Go.")], [])
    try:
        name, provider, _, _ = litellm.get_llm_provider(model=params["model"])
        mapped = get_optional_params(
            model=name,
            custom_llm_provider=provider,
            max_tokens=params["max_tokens"],
            allowed_openai_params=params["allowed_openai_params"],
            **{k: v for k, v in params.items() if k in _REASONING_KEYS},
        )
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"
    return {k: v for k, v in mapped.items() if k in _LITELLM_REASONING_KEYS}


def test_golden_reasoning_effort_by_model(
    monkeypatch: pytest.MonkeyPatch, assert_matches_golden: Golden
):
    """Per model and effort setting: what karotte asks for, and what litellm
    turns it into on the wire."""
    monkeypatch.delenv("KAROTTE_PROXY_URL", raising=False)

    assert_matches_golden(
        "reasoning_effort_by_model.json",
        {
            model: {
                str(level): {
                    "sent_by_karotte": BuiltinSource(
                        _config(model, reasoning_effort=level)
                    )
                    .get_completion_params([Message(role="user", content="Go.")], [])
                    .get("reasoning_effort"),
                    "mapped_by_litellm": _litellm_reasoning_params(model, level),
                }
                for level in (None, "min", "max")
            }
            for model in MODELS
        },
    )


def test_gemini_gets_thought_summaries_without_an_effort_level():
    """Without an effort level litellm sends no thinking config, so karotte asks
    for thought summaries itself; with one, litellm asking too is a conflict."""
    params = _completion_params("vertex_ai/gemini-3.7-flash")
    assert params["thinkingConfig"] == {"includeThoughts": True}

    source = BuiltinSource(
        _config("vertex_ai/gemini-3.7-flash", reasoning_effort="max")
    )
    with_effort = source.get_completion_params(
        [Message(role="user", content="Go.")], []
    )
    assert "thinkingConfig" not in with_effort

    assert "thinkingConfig" not in _completion_params("claude-fable-5")


def test_gemini_3_8_flash_effort_gets_past_litellms_cost_map_check():
    """litellm validates reasoning_effort on Gemini against its cost map, which
    predates 3.8 Flash, so without an explicit allowed_openai_params entry the
    effort would abort the run. Allowed through, litellm still maps it onto
    thinkingLevel exactly as it does for 3.7."""
    for model in ("vertex_ai/gemini-3.8-flash", "vertex_ai/gemini-3.7-flash"):
        mapped = _litellm_reasoning_params(model, "max")
        assert mapped["thinkingConfig"] == {
            "thinkingLevel": "high",
            "includeThoughts": True,
        }, model
        assert _litellm_reasoning_params(model, "min")["thinkingConfig"] == {
            "thinkingLevel": "low",
            "includeThoughts": True,
        }, model


def test_grok_4_7_effort_gets_past_litellms_cost_map_check():
    """litellm 1.94 predates Grok 4.7 and raises UnsupportedParamsError for
    reasoning_effort on any xai id outside its cost map, so 4.7 needs an
    explicit allowed_openai_params entry where 4.6 does not."""
    for model in ("xai/grok-4.7", "xai/grok-4.6"):
        for level, effort in (("min", "low"), ("max", "xhigh")):
            assert _litellm_reasoning_params(model, level) == {
                "reasoning_effort": effort
            }, (model, level)


def test_muse_spark_asks_for_reasoning_summaries():
    """Meta hides reasoning on chat completions and only returns summaries on the
    Responses API, so Muse Spark goes through litellm's Responses bridge."""
    for model in ("meta/muse-spark-1.3", "meta/muse-spark-1.2"):
        params = _completion_params(model)
        assert params["model"] == model.replace("meta/", "meta/responses/")
        assert params["extra_body"] == {"reasoning_summary": "detailed"}


def test_muse_spark_effort_gets_past_litellms_unknown_model_check():
    """litellm looks the bridge name responses/<model> up in its cost map and
    finds nothing, so without an explicit allowed_openai_params entry the
    effort would abort the run."""
    for model, top in (
        ("meta/muse-spark-1.3", "max"),
        ("meta/muse-spark-1.2", "xhigh"),
    ):
        assert _litellm_reasoning_params(model, "min") == {
            "reasoning_effort": "minimal"
        }
        assert _litellm_reasoning_params(model, "max") == {"reasoning_effort": top}


def test_gpt_6_goes_through_the_responses_bridge():
    """litellm's built-in OpenAI Responses bridge only matches "gpt-5." names, so
    GPT-6 is routed there explicitly; Chat Completions would send max_tokens,
    which OpenAI reasoning models answer 400 to."""
    for model in ("openai/gpt-6-astra", "openai/gpt-6-sol", "openai/gpt-6-luna"):
        params = _completion_params(model)
        assert params["model"] == model.replace("openai/", "openai/responses/")
        # Without reasoning_summary litellm builds no reasoning object and
        # silently drops the effort; the cache key shares extra_body because
        # litellm drops it as a top-level kwarg. Both confirmed by wire capture.
        assert params["extra_body"] == {
            "reasoning_summary": "detailed",
            "prompt_cache_key": "golden-run",
        }
        assert "temperature" not in params
        assert _litellm_reasoning_params(model, "max") == {"reasoning_effort": "max"}
        assert _litellm_reasoning_params(model, "min") == {"reasoning_effort": "low"}


def test_together_effort_gets_past_litellms_param_check():
    """litellm carries no reasoning_effort in its together_ai param list and
    raises UnsupportedParamsError rather than forwarding it, so the Together
    models that take one need an explicit allowed_openai_params entry."""
    assert _litellm_reasoning_params(
        "together_ai/deepseek-ai/DeepSeek-V4-Pro-0813", "min"
    ) == {"reasoning_effort": "high"}
    assert _litellm_reasoning_params("together_ai/zai-org/GLM-5.3", "max") == {
        "reasoning_effort": "max"
    }
    assert _litellm_reasoning_params("together_ai/Qwen/Qwen3.8-2.4T-A95B", "max") == {
        "reasoning_effort": "xhigh"
    }
    assert _litellm_reasoning_params("together_ai/moonshotai/Kimi-K3", "min") == {
        "reasoning_effort": "low"
    }
    # Models without an effort range send nothing, so there is nothing to allow.
    assert _litellm_reasoning_params("together_ai/zai-org/GLM-5.2", "max") == {}


def test_tools_are_offered_when_the_task_has_them(monkeypatch: pytest.MonkeyPatch):
    """The no-tool params above would not notice tool_choice regressing."""
    monkeypatch.delenv("KAROTTE_PROXY_URL", raising=False)
    tool: Any = {
        "type": "function",
        "function": {"name": "echo", "parameters": {"type": "object"}},
    }
    params = BuiltinSource(_config("claude-opus-5")).get_completion_params(
        [Message(role="user", content="Go.")], [tool]
    )
    assert params["tools"] == [tool]
    assert params["tool_choice"] == "auto"


# A DeepSeek special token inside a value that is itself JSON-encoded the way
# xAI double-encodes strings, so both repairs have something to find.
_DIRTY_ARGUMENTS = '{"text": "\\"<\uff5ctool\u2581call\u2581begin\uff5c>hello\\""}'


@final
class _ScriptedStream(CustomStreamWrapper):
    """Passes the source's isinstance check while yielding scripted chunks."""

    def __init__(self, chunks: list[ModelResponseStream]):  # pyright: ignore[reportMissingSuperCall]
        self._chunks = chunks

    @override
    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk


def _dirty_tool_call_chunks(model: str) -> list[ModelResponseStream]:
    def chunk(
        delta: LiteLlmDelta, finish_reason: str | None = None
    ) -> ModelResponseStream:
        return ModelResponseStream(
            id="chatcmpl-golden",
            created=1735689600,
            model=model,
            choices=[
                StreamingChoices(index=0, delta=delta, finish_reason=finish_reason)
            ],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
            if finish_reason
            else None,
        )

    return [
        chunk(LiteLlmDelta(role="assistant", content="Calling echo.")),
        chunk(
            LiteLlmDelta(
                tool_calls=[
                    ChatCompletionDeltaToolCall(
                        id="call_1",
                        type="function",
                        index=0,
                        function=LiteLlmFunction(
                            name="echo", arguments=_DIRTY_ARGUMENTS
                        ),
                    )
                ]
            )
        ),
        chunk(LiteLlmDelta(), finish_reason="tool_calls"),
    ]


async def _repaired_arguments(model: str, monkeypatch: pytest.MonkeyPatch) -> str:
    async def scripted_acompletion(**_kwargs: Any) -> _ScriptedStream:
        return _ScriptedStream(_dirty_tool_call_chunks(model))

    monkeypatch.setattr("litellm.acompletion", scripted_acompletion)

    source = BuiltinSource(_config(model))
    added = [
        event
        async for event in source.collect([Message(role="user", content="Go.")], [])
        if isinstance(event, MessageAddedEvent)
    ]
    assert len(added) == 1
    tool_calls = added[0].message.tool_calls
    assert tool_calls is not None
    return tool_calls[0].function.arguments


@pytest.mark.asyncio
async def test_golden_tool_call_repair_by_model(
    monkeypatch: pytest.MonkeyPatch, assert_matches_golden: Golden
):
    """Which providers get their tool-call arguments rewritten on the way in
    (DeepSeek special tokens stripped, xAI's double JSON-encoding undone)."""
    assert_matches_golden(
        "tool_call_repair_by_model.json",
        {
            "sent_by_the_model": _DIRTY_ARGUMENTS,
            "seen_by_the_tool": {
                model: await _repaired_arguments(model, monkeypatch) for model in MODELS
            },
        },
    )


def test_golden_cli_agent_model_resolution(assert_matches_golden: Golden):
    """The provider-native name and credential env var a CLI agent subprocess is
    handed for each model."""
    assert_matches_golden(
        "cli_agent_model_resolution.json",
        {model: asdict(resolve_model(model, "test_key")) for model in MODELS},
    )


def _rubric_judge_request(model: str, monkeypatch: pytest.MonkeyPatch) -> Any:
    judge = RubricJudge(
        rubric=[{"criterion": "is correct", "weight": 1.0}],
        model=model,
        api_key="test_key",
    )
    captured: dict[str, Any] = {}

    def completion(**kwargs: Any) -> litellm.ModelResponse:
        captured.update(kwargs)
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": "YES\nthe criterion is met",
                    }
                }
            ]
        )

    monkeypatch.setattr("litellm.completion", completion)

    is_met, reasoning = judge._evaluate_criterion("some content", "is correct")  # pyright: ignore[reportPrivateUsage]
    return {
        "model": captured["model"],
        "max_tokens": captured["max_tokens"],
        "temperature": captured.get("temperature", "<omitted>"),
        "parsed": [is_met, reasoning],
    }


def test_golden_rubric_judge_request_by_model(
    monkeypatch: pytest.MonkeyPatch, assert_matches_golden: Golden
):
    """Which models the rubric judge omits sampling params for."""
    assert_matches_golden(
        "rubric_judge_request_by_model.json",
        {model: _rubric_judge_request(model, monkeypatch) for model in MODELS},
    )


def _api_key_required(model: str) -> bool:
    try:
        EvaluationRunConfig(run_id="golden-run", task_id="golden-task", model=model)
    except ValidationError:
        return True
    return False


def test_golden_run_config_model_policy(assert_matches_golden: Golden):
    """Which agent drives each model, which ones take the training rollout path,
    and which ones must be given an API key."""
    assert_matches_golden(
        "config_policy_by_model.json",
        {
            model: {
                "resolved_agent": _config(model).resolved_agent,
                "is_special_training_model": _config(model).is_special_training_model,
                "api_key_required": _api_key_required(model),
            }
            for model in MODELS
        },
    )
