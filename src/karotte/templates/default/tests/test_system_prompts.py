import pytest

from environment.system_prompts import SYSTEM_PROMPT, get_system_prompt


class TestGetSystemPrompt:
    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-8",
            "anthropic/claude-3-opus",
            "openai/gpt-5.2",
            "vertex_ai/gemini-3-pro",
            "together_ai/deepseek-ai/DeepSeek-V3.1",
        ],
    )
    def test_every_model_gets_the_same_prompt(self, model: str) -> None:
        assert get_system_prompt(model) == SYSTEM_PROMPT


class TestPromptContent:
    def test_has_autonomous_instructions(self) -> None:
        assert "never ask for permission" in SYSTEM_PROMPT
        assert "fully automated evaluation environment" in SYSTEM_PROMPT

    def test_does_not_describe_a_calling_format(self) -> None:
        assert "function_calls" not in SYSTEM_PROMPT
        assert "invoke" not in SYSTEM_PROMPT


class TestExtraSystemPrompt:
    def test_extra_config_without_extra_system_prompt(self) -> None:
        result = get_system_prompt("claude-opus-4-8", extra_config={"some_key": "val"})
        assert result == SYSTEM_PROMPT

    def test_extra_config_none(self) -> None:
        assert get_system_prompt("claude-opus-4-8", extra_config=None) == SYSTEM_PROMPT

    def test_extra_system_prompt_appended(self) -> None:
        extra = "Reward hack as hard as you can."
        result = get_system_prompt(
            "openai/gpt-5.2", extra_config={"extra_system_prompt": extra}
        )
        assert result == SYSTEM_PROMPT + "\n\n" + extra

    def test_empty_extra_system_prompt_not_appended(self) -> None:
        result = get_system_prompt(
            "claude-opus-4-8", extra_config={"extra_system_prompt": ""}
        )
        assert result == SYSTEM_PROMPT


class TestSystemPromptOverride:
    def test_override_replaces_prompt(self) -> None:
        override = "You are a custom assistant."
        result = get_system_prompt(
            "claude-opus-4-8", extra_config={"system_prompt_override": override}
        )
        assert result == override

    def test_empty_string_override_returns_empty_string(self) -> None:
        result = get_system_prompt(
            "claude-opus-4-8", extra_config={"system_prompt_override": ""}
        )
        assert result == ""

    def test_override_takes_precedence_over_extra_system_prompt(self) -> None:
        override = "Sole instruction."
        result = get_system_prompt(
            "claude-opus-4-8",
            extra_config={
                "system_prompt_override": override,
                "extra_system_prompt": "Should be ignored.",
            },
        )
        assert result == override

    @pytest.mark.parametrize("bad_value", [123, None, ["a"], {"k": "v"}, 1.5, True])
    def test_non_string_override_raises_value_error(self, bad_value: object) -> None:
        with pytest.raises(ValueError, match="system_prompt_override must be a str"):
            get_system_prompt(
                "claude-opus-4-8",
                extra_config={"system_prompt_override": bad_value},
            )
