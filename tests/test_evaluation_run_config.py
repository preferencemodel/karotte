from typing import Any

import pytest
from pydantic import ValidationError

from karotte.schemas.evaluation_run_config import EvaluationRunConfig


def test_resolved_agent_defaults_to_builtin():
    config = EvaluationRunConfig(
        run_id="r", task_id="t", model="gpt-4o", model_api_key="k"
    )
    assert config.resolved_agent == "builtin"


def test_resolved_agent_maps_pt_prefix_to_external():
    config = EvaluationRunConfig(run_id="r", task_id="t", model="pt/foo")
    assert config.resolved_agent == "external"


def test_explicit_agent_overrides_pt_mapping():
    config = EvaluationRunConfig(
        run_id="r", task_id="t", model="pt/foo", agent="mistral-vibe"
    )
    assert config.resolved_agent == "mistral-vibe"


def test_vertex_ai_model_does_not_require_api_key():
    """Vertex AI models should work without model_api_key."""
    config = EvaluationRunConfig(
        run_id="test_run",
        task_id="test_task",
        model="vertex_ai/gemini-3-pro-preview",
    )
    assert config.model_api_key is None
    assert config.model == "vertex_ai/gemini-3-pro-preview"


def test_non_vertex_ai_model_requires_api_key():
    """Non-Vertex AI models should raise error without model_api_key."""
    with pytest.raises(ValidationError) as exc_info:
        EvaluationRunConfig(
            run_id="test_run",
            task_id="test_task",
            model="claude-sonnet-4-5-20250929",
        )
    assert "model_api_key is required" in str(exc_info.value)


def test_non_vertex_ai_model_with_api_key_succeeds():
    """Non-Vertex AI models should work with model_api_key provided."""
    config = EvaluationRunConfig(
        run_id="test_run",
        task_id="test_task",
        model="claude-sonnet-4-5-20250929",
        model_api_key="sk-ant-test-key",
    )
    assert config.model_api_key == "sk-ant-test-key"


def test_env_var_reference_passes_validation():
    """$ENV_VAR format should be accepted by the schema (resolution happens at parse time)."""
    config = EvaluationRunConfig(
        run_id="test_run",
        task_id="test_task",
        model="claude-sonnet-4-5-20250929",
        model_api_key="$ANTHROPIC_API_KEY",
    )
    assert config.model_api_key == "$ANTHROPIC_API_KEY"


def test_vertex_ai_model_with_api_key_also_succeeds():
    """Vertex AI models should also work if api_key is provided."""
    config = EvaluationRunConfig(
        run_id="test_run",
        task_id="test_task",
        model="vertex_ai/gemini-3-pro-preview",
        model_api_key="optional-key",
    )
    assert config.model_api_key == "optional-key"


def _config_with_extra(extra: object) -> EvaluationRunConfig:
    return EvaluationRunConfig(
        run_id="test_run",
        task_id="test_task",
        model="vertex_ai/gemini-3-pro-preview",
        extra_config={"extra_artifact_paths": extra},
    )


def test_extra_artifact_paths_defaults_to_empty():
    config = EvaluationRunConfig(
        run_id="test_run",
        task_id="test_task",
        model="vertex_ai/gemini-3-pro-preview",
    )
    assert config.extra_artifact_paths == []


def test_extra_artifact_paths_normalizes_string_to_list():
    config = _config_with_extra("/workdir/data/reasoning_summary.md")
    assert config.extra_artifact_paths == ["/workdir/data/reasoning_summary.md"]


def test_extra_artifact_paths_accepts_list_of_strings():
    config = _config_with_extra(["/a.md", "/b.md"])
    assert config.extra_artifact_paths == ["/a.md", "/b.md"]


def test_extra_artifact_paths_rejects_non_string_entry():
    with pytest.raises(ValidationError, match="string or list of strings"):
        _config_with_extra([123])


def test_extra_artifact_paths_rejects_wrong_type():
    with pytest.raises(ValidationError, match="string or list of strings"):
        _config_with_extra({"not": "a list"})


def _config_with_instr_extra(extra: dict[Any, Any]) -> EvaluationRunConfig:
    return EvaluationRunConfig(
        run_id="test_run",
        task_id="test_task",
        model="vertex_ai/gemini-3-pro-preview",
        extra_config=extra,
    )


def test_resolve_step_instructions_no_extra_config():
    config = EvaluationRunConfig(
        run_id="test_run",
        task_id="test_task",
        model="vertex_ai/gemini-3-pro-preview",
    )
    assert config.resolve_step_instructions("base", 0) == "base"


def test_extra_task_instructions_string_appends_to_every_step():
    config = _config_with_instr_extra({"extra_task_instructions": "be careful"})
    assert config.resolve_step_instructions("step one", 0) == "step one\n\nbe careful"
    assert config.resolve_step_instructions("step two", 1) == "step two\n\nbe careful"


def test_extra_task_instructions_list_appends_per_step():
    config = _config_with_instr_extra({"extra_task_instructions": ["first", "second"]})
    assert config.resolve_step_instructions("a", 0) == "a\n\nfirst"
    assert config.resolve_step_instructions("b", 1) == "b\n\nsecond"


def test_extra_task_instructions_list_missing_index_is_noop():
    config = _config_with_instr_extra({"extra_task_instructions": ["only first"]})
    assert config.resolve_step_instructions("a", 0) == "a\n\nonly first"
    assert config.resolve_step_instructions("b", 1) == "b"


def test_extra_task_instructions_empty_string_is_noop():
    config = _config_with_instr_extra({"extra_task_instructions": ""})
    assert config.resolve_step_instructions("a", 0) == "a"


def test_extra_task_instructions_rejects_wrong_type():
    with pytest.raises(ValidationError, match="string or list of strings"):
        _config_with_instr_extra({"extra_task_instructions": 123})


def test_extra_task_instructions_rejects_non_string_entry():
    with pytest.raises(ValidationError, match="string or list of strings"):
        _config_with_instr_extra({"extra_task_instructions": ["ok", 5]})


def test_task_instructions_override_string_replaces_every_step():
    config = _config_with_instr_extra({"task_instructions_override": "do this instead"})
    assert config.resolve_step_instructions("original", 0) == "do this instead"
    assert config.resolve_step_instructions("original", 1) == "do this instead"


def test_task_instructions_override_list_replaces_per_step():
    config = _config_with_instr_extra({"task_instructions_override": ["one", "two"]})
    assert config.resolve_step_instructions("a", 0) == "one"
    assert config.resolve_step_instructions("b", 1) == "two"


def test_task_instructions_override_list_missing_index_keeps_original():
    config = _config_with_instr_extra({"task_instructions_override": ["one"]})
    assert config.resolve_step_instructions("a", 0) == "one"
    assert config.resolve_step_instructions("b", 1) == "b"


def test_task_instructions_override_empty_string_replaces():
    config = _config_with_instr_extra({"task_instructions_override": ""})
    assert config.resolve_step_instructions("a", 0) == ""


def test_task_instructions_override_takes_precedence_over_extra():
    config = _config_with_instr_extra(
        {
            "task_instructions_override": "override",
            "extra_task_instructions": "appended",
        }
    )
    assert config.resolve_step_instructions("original", 0) == "override"


def test_task_instructions_override_rejects_wrong_type():
    with pytest.raises(ValidationError, match="string or list of strings"):
        _config_with_instr_extra({"task_instructions_override": 123})


def _config(**kwargs: Any) -> EvaluationRunConfig:
    return EvaluationRunConfig(
        run_id="test_run",
        task_id="test_task",
        model="vertex_ai/gemini-3-pro-preview",
        **kwargs,
    )


def test_resolve_step_time_limit_defaults_to_none():
    config = _config()
    assert config.resolve_step_time_limit(0) is None
    assert config.resolve_step_time_limit(5) is None


def test_resolve_step_time_limit_single_value_applies_to_every_step():
    config = _config(step_time_limit_seconds=30)
    assert config.resolve_step_time_limit(0) == 30
    assert config.resolve_step_time_limit(3) == 30


def test_resolve_step_time_limit_list_is_indexed_per_step():
    config = _config(step_time_limit_seconds=[10, 20])
    assert config.resolve_step_time_limit(0) == 10
    assert config.resolve_step_time_limit(1) == 20


def test_resolve_step_time_limit_list_out_of_range_is_unlimited():
    config = _config(step_time_limit_seconds=[10])
    assert config.resolve_step_time_limit(0) == 10
    assert config.resolve_step_time_limit(1) is None


def test_on_step_time_limit_defaults_to_error():
    assert _config().on_step_time_limit == "error"


def test_on_step_time_limit_rejects_unknown_value():
    with pytest.raises(ValidationError):
        _config(on_step_time_limit="ignore")


def test_inject_time_remaining_counter_defaults_to_true():
    assert _config().inject_time_remaining_counter is True


def test_resolve_step_context_window_limit_defaults_to_none():
    config = _config()
    assert config.resolve_step_context_window_limit(0) is None
    assert config.resolve_step_context_window_limit(5) is None


def test_resolve_step_context_window_limit_single_value_applies_to_every_step():
    config = _config(step_context_window_limit=100_000)
    assert config.resolve_step_context_window_limit(0) == 100_000
    assert config.resolve_step_context_window_limit(3) == 100_000


def test_resolve_step_context_window_limit_list_is_indexed_per_step():
    config = _config(step_context_window_limit=[10_000, 20_000])
    assert config.resolve_step_context_window_limit(0) == 10_000
    assert config.resolve_step_context_window_limit(1) == 20_000


def test_resolve_step_context_window_limit_list_out_of_range_is_unlimited():
    config = _config(step_context_window_limit=[10_000])
    assert config.resolve_step_context_window_limit(0) == 10_000
    assert config.resolve_step_context_window_limit(1) is None


def test_on_step_context_window_limit_defaults_to_error():
    assert _config().on_step_context_window_limit == "error"


def test_on_step_context_window_limit_rejects_unknown_value():
    with pytest.raises(ValidationError):
        _config(on_step_context_window_limit="ignore")


def test_inject_context_remaining_counter_defaults_to_true():
    assert _config().inject_context_remaining_counter is True


class TestAppliedReasoningEffort:
    """What actually reaches the provider, which is nothing unless the builtin
    agent is driving a model whose API takes the parameter."""

    def _config(
        self, model: str = "claude-opus-5", **overrides: Any
    ) -> EvaluationRunConfig:
        return EvaluationRunConfig(
            run_id="r", task_id="t", model=model, model_api_key="k", **overrides
        )

    def test_unset_sends_nothing(self):
        assert self._config().applied_reasoning_effort is None

    @pytest.mark.parametrize(("level", "expected"), [("min", "low"), ("max", "max")])
    def test_resolves_to_the_provider_name(self, level: str, expected: str):
        config = self._config(reasoning_effort=level)
        assert config.applied_reasoning_effort == expected

    def test_model_without_the_parameter_sends_nothing(self):
        config = self._config(
            model="together_ai/moonshotai/Kimi-K2.6", reasoning_effort="max"
        )
        assert config.applied_reasoning_effort is None

    def test_a_provider_level_is_sent_as_is(self):
        config = self._config(reasoning_effort="medium")
        assert config.applied_reasoning_effort == "medium"

    @pytest.mark.parametrize(
        ("model", "level", "accepted"),
        [
            ("together_ai/moonshotai/Kimi-K3", "medium", "low, high, max"),
            ("claude-opus-4-6", "xhigh", "low, medium, high, max"),
            ("openai/gpt-5.5", "banana", "low, medium, high, xhigh"),
        ],
    )
    def test_a_level_the_model_lacks_is_rejected(
        self, model: str, level: str, accepted: str
    ):
        with pytest.raises(ValidationError, match=accepted):
            self._config(model=model, reasoning_effort=level)

    def test_a_level_on_a_model_without_the_parameter_is_rejected(self):
        with pytest.raises(ValidationError, match="no reasoning_effort"):
            self._config(
                model="together_ai/moonshotai/Kimi-K2.6", reasoning_effort="low"
            )

    def test_min_and_max_are_never_rejected(self):
        self._config(model="together_ai/moonshotai/Kimi-K2.6", reasoning_effort="min")
        self._config(model="together_ai/moonshotai/Kimi-K2.6", reasoning_effort="max")

    @pytest.mark.parametrize("agent", ["mistral-vibe", "external"])
    def test_other_agents_never_send_it(self, agent: str):
        config = self._config(agent=agent, reasoning_effort="max")
        assert config.applied_reasoning_effort is None

    def test_fake_model_never_sends_it(self):
        config = self._config(use_fake_model=True, reasoning_effort="max")
        assert config.applied_reasoning_effort is None


def test_unknown_fields_are_ignored():
    config = EvaluationRunConfig.model_validate(
        {
            "run_id": "r",
            "task_id": "t",
            "model": "m",
            "model_api_key": "k",
            "some_unknown_field": "x",
        }
    )
    assert "some_unknown_field" not in config.model_dump()
