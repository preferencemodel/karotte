"""Tests for the Grok Build CLI agent: its (subprocess-free) config and argv,
and the `--output-format streaming-messages-json` -> transcript-event adapter,
driven by a `grok -p` log recorded against a scripted model
(tests/resources/grok_build/)."""

import json
import tomllib
from pathlib import Path

import pytest

from karotte.agents import cli_agent_types
from karotte.agents.cli_agent import AGENTS_BIN_DIR
from karotte.agents.grok_build import GrokBuildAgent
from karotte.agents.grok_build.adapter import parse_stream
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.transcript import (
    MessageAddedEvent,
    TokenUsageEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)

_FIXTURE = Path(__file__).parent / "resources" / "grok_build" / "step_0.ndjson"


def _agent(model: str = "xai/grok-4.7", **kwargs: object) -> GrokBuildAgent:
    config = EvaluationRunConfig.model_validate(
        {
            "run_id": "r",
            "task_id": "t",
            "model": model,
            "model_api_key": "xai-abc",
            **kwargs,
        }
    )
    return GrokBuildAgent(config)


def _write_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent: GrokBuildAgent
) -> dict[str, object]:
    agent._grok_home = tmp_path / ".grok_home"  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr("karotte.agents.grok_build.agent.demoted_uid_gid", lambda: None)
    agent._write_config()  # pyright: ignore[reportPrivateUsage]
    return tomllib.loads((tmp_path / ".grok_home" / "config.toml").read_text())


class TestGrokBuildAgent:
    def test_is_registered(self):
        assert cli_agent_types()["grok-build"] is GrokBuildAgent

    def test_install_pins_version_and_checksums(self):
        script = " ".join(GrokBuildAgent.install())
        assert f"grok-{GrokBuildAgent.version}-linux-" in script
        assert "sha256sum -c" in script
        assert f"{AGENTS_BIN_DIR}/grok" in script

    def test_argv_creates_then_resumes_one_session(self):
        agent = _agent(turn_limit=7)
        first = agent._argv("do the thing")  # pyright: ignore[reportPrivateUsage]
        assert first[0] == f"{AGENTS_BIN_DIR}/grok"
        assert first[first.index("-p") + 1] == "do the thing"
        assert first[first.index("-m") + 1] == "grok-4.7"
        assert first[first.index("--max-turns") + 1] == "7"
        assert "--always-approve" in first
        session_id = first[first.index("--session-id") + 1]

        agent._step_index = 1  # pyright: ignore[reportPrivateUsage]
        second = agent._argv("next")  # pyright: ignore[reportPrivateUsage]
        assert "--session-id" not in second
        assert second[second.index("--resume") + 1] == session_id

    def test_argv_omits_turn_limit_when_unset(self):
        assert "--max-turns" not in _agent()._argv("x")  # pyright: ignore[reportPrivateUsage]

    def test_env_sets_credentials_and_grok_home(self):
        env = _agent()._env()  # pyright: ignore[reportPrivateUsage]
        assert env["XAI_API_KEY"] == "xai-abc"
        assert env["GROK_HOME"].endswith(".grok_home")
        assert env["GROK_DISABLE_AUTOUPDATER"] == "1"
        assert AGENTS_BIN_DIR in env["PATH"]
        assert env["HOME"] != "/root"
        assert env["USER"] == "student"

    def test_env_uses_placeholder_key_behind_keyless_proxy(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("KAROTTE_PROXY_URL", "http://proxy:4000")
        env = _agent(model_api_key=None)._env()  # pyright: ignore[reportPrivateUsage]
        assert env["XAI_API_KEY"] == "model_api_key"

    def test_config_talks_to_xai_without_proxy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("KAROTTE_PROXY_URL", raising=False)
        config = _write_config(tmp_path, monkeypatch, _agent())
        model = config["model"]["grok-4.7"]  # pyright: ignore[reportIndexIssue]
        assert model["base_url"] == "https://api.x.ai/v1"
        assert model["env_key"] == "XAI_API_KEY"
        assert config["models"]["session_summary"] == "grok-4.7"  # pyright: ignore[reportIndexIssue]
        assert config["features"]["telemetry"] == "off"  # pyright: ignore[reportIndexIssue]
        assert "url" in config["mcp_servers"]["karotte"]  # pyright: ignore[reportIndexIssue]

    def test_config_routes_any_provider_through_proxy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("KAROTTE_PROXY_URL", "http://proxy:4000/")
        agent = _agent(model="anthropic/claude-sonnet-5-5")
        config = _write_config(tmp_path, monkeypatch, agent)
        model = config["model"]["claude-sonnet-5-5"]  # pyright: ignore[reportIndexIssue]
        assert model["base_url"] == "http://proxy:4000/v1"
        assert model["env_key"] == "ANTHROPIC_API_KEY"

    def test_config_rejects_non_xai_model_without_proxy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("KAROTTE_PROXY_URL", raising=False)
        with pytest.raises(ValueError, match="only reach xAI models"):
            _write_config(tmp_path, monkeypatch, _agent(model="openai/gpt-5"))


class TestParseStream:
    def _events(self):
        return list(parse_stream(_FIXTURE.read_text().splitlines()))

    def test_event_sequence(self):
        assert [type(e).__name__ for e in self._events()] == [
            "MessageAddedEvent",  # assistant: text + use_tool call
            "TokenUsageEvent",
            "ToolCallStartedEvent",
            "ToolCallCompletedEvent",
            "MessageAddedEvent",  # MCP tool result
            "MessageAddedEvent",  # assistant: thinking + shell call
            "TokenUsageEvent",
            "ToolCallStartedEvent",
            "ToolCallCompletedEvent",
            "MessageAddedEvent",  # shell tool error
            "MessageAddedEvent",  # final answer
            "TokenUsageEvent",
        ]

    def test_messages(self):
        messages = [
            e.message for e in self._events() if isinstance(e, MessageAddedEvent)
        ]
        assert [m.role for m in messages] == [
            "assistant",
            "tool",
            "assistant",
            "tool",
            "assistant",
        ]
        first = messages[0]
        assert first.content == "calling echo"
        assert first.tool_calls is not None
        call = first.tool_calls[0]
        assert call.function.name == "use_tool"
        assert json.loads(call.function.arguments) == {
            "tool_name": "karotte__echo",
            "tool_input": {"text": "hi"},
        }
        assert messages[2].reasoning_content == "thinking about bash"
        assert messages[-1].content == "All done."

    def test_tool_results_pair_with_calls(self):
        events = self._events()
        started = [
            e.tool_call.id for e in events if isinstance(e, ToolCallStartedEvent)
        ]
        completed = [e for e in events if isinstance(e, ToolCallCompletedEvent)]
        assert started == [c.tool_call_id for c in completed] == ["call_a", "call_b"]
        assert completed[0].result.isError is False
        assert completed[1].result.isError is True

    def test_usage_is_reported_not_estimated(self):
        usage = [e for e in self._events() if isinstance(e, TokenUsageEvent)]
        assert [(u.input_tokens, u.output_tokens) for u in usage] == [
            (100, 7),
            (110, 7),
            (120, 7),
        ]
        assert not any(u.estimated for u in usage)

    def test_usage_input_includes_cache_buckets(self):
        line = json.dumps(
            {
                "type": "assistant",
                "parent_tool_use_id": None,
                "message": {
                    "content": [{"type": "text", "text": "x"}],
                    "usage": {
                        "input_tokens": 10,
                        "cache_read_input_tokens": 90,
                        "cache_creation_input_tokens": 5,
                        "output_tokens": 3,
                    },
                },
            }
        )
        usage = next(e for e in parse_stream([line]) if isinstance(e, TokenUsageEvent))
        assert usage.input_tokens == 105
        assert usage.cache_read_tokens == 90
        assert usage.cache_write_tokens == 5

    def test_skips_subagent_and_malformed_lines(self):
        subagent = json.dumps(
            {
                "type": "assistant",
                "parent_tool_use_id": "call_x",
                "message": {"content": [{"type": "text", "text": "child"}]},
            }
        )
        assert list(parse_stream(["not json", "[]", subagent, ""])) == []
