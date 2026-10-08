"""Tests for the Grok Build CLI agent: its (subprocess-free) config and argv,
and the `--output-format streaming-messages-json` -> transcript-event adapter,
driven by a `grok -p` log recorded against a scripted model
(tests/resources/grok_build/)."""

import asyncio
import json
import sys
import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from mcp.types import ImageContent, TextContent

from karotte.agents import cli_agent_types
from karotte.agents.agent import StepTimeLimitReachedError
from karotte.agents.cli_agent import AGENTS_BIN_DIR
from karotte.agents.grok_build import GrokBuildAgent
from karotte.agents.grok_build.adapter import parse_stream
from karotte.providers import PROXY_PLACEHOLDER_KEY
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.transcript import (
    Event,
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
        first = agent._argv(Path("step_0.prompt"))  # pyright: ignore[reportPrivateUsage]
        assert first[0] == f"{AGENTS_BIN_DIR}/grok"
        assert first[first.index("--prompt-file") + 1] == "step_0.prompt"
        assert "-p" not in first
        assert first[first.index("-m") + 1] == "grok-4.7"
        assert first[first.index("--max-turns") + 1] == "7"
        assert "--always-approve" in first
        session_id = first[first.index("--session-id") + 1]

        agent._step_index = 1  # pyright: ignore[reportPrivateUsage]
        second = agent._argv(Path("step_1.prompt"))  # pyright: ignore[reportPrivateUsage]
        assert "--session-id" not in second
        assert second[second.index("--resume") + 1] == session_id

    def test_argv_omits_turn_limit_when_unset(self):
        assert "--max-turns" not in _agent()._argv(Path("x"))  # pyright: ignore[reportPrivateUsage]

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

    @pytest.mark.asyncio
    async def test_start_without_proxy_forwards_and_withholds_the_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("KAROTTE_PROXY_URL", raising=False)
        agent = _agent()
        agent._grok_home = tmp_path / ".grok_home"  # pyright: ignore[reportPrivateUsage]
        monkeypatch.setattr(
            "karotte.agents.grok_build.agent.demoted_uid_gid", lambda: None
        )
        await agent.start(MagicMock())
        try:
            assert agent.model_url.startswith("http://127.0.0.1:")
            config = tomllib.loads(
                (tmp_path / ".grok_home" / "config.toml").read_text()
            )
            model = config["model"]["grok-4.7"]
            assert model["base_url"] == f"{agent.model_url}/v1"
            env = agent._env()  # pyright: ignore[reportPrivateUsage]
            assert env["XAI_API_KEY"] == PROXY_PLACEHOLDER_KEY
            assert "xai-abc" not in env.values()
        finally:
            await agent.stop()


class TestGrokBuildSampling:
    def _model_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: object
    ) -> dict[str, Any]:
        monkeypatch.setenv("KAROTTE_PROXY_URL", "http://proxy:4000")
        model = str(kwargs.pop("model", "xai/grok-4.7"))
        agent = _agent(model=model, agent="grok-build", **kwargs)
        config: dict[str, Any] = _write_config(tmp_path, monkeypatch, agent)
        return config["model"][model.split("/", 1)[1]]

    def test_sends_fixed_temperature(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        assert self._model_table(tmp_path, monkeypatch)["temperature"] == 0.7

    def test_effort_defaults_to_high(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        menu = self._model_table(tmp_path, monkeypatch)["reasoning_efforts"]
        assert menu == [
            {"value": "low"},
            {"value": "medium"},
            {"value": "high", "default": True},
            {"value": "xhigh"},
        ]

    @pytest.mark.parametrize(
        ("requested", "sent"), [("medium", "medium"), ("min", "low"), ("max", "xhigh")]
    )
    def test_effort_follows_the_run_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        requested: str,
        sent: str,
    ):
        table = self._model_table(tmp_path, monkeypatch, reasoning_effort=requested)
        defaults = [o["value"] for o in table["reasoning_efforts"] if o.get("default")]
        assert defaults == [sent]


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

    def test_read_file_image_becomes_an_image(self):
        line = json.dumps(
            {
                "type": "user",
                "parent_tool_use_id": None,
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_img",
                            "content": json.dumps(
                                {
                                    "type": "ReadFile",
                                    "ImageContent": {
                                        "data": "/9j/4AAQ",
                                        "mime_type": "image/jpeg",
                                    },
                                }
                            ),
                            "is_error": False,
                        }
                    ]
                },
            }
        )

        completed, message = list(parse_stream([line]))

        assert isinstance(completed, ToolCallCompletedEvent)
        assert completed.tool_call_id == "call_img"
        assert completed.result.content == [
            ImageContent(type="image", data="", mimeType="image/jpeg")
        ]
        assert isinstance(message, MessageAddedEvent)
        assert message.message.role == "tool"
        assert message.message.content == [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ"},
            }
        ]

    def test_other_read_file_results_stay_text(self):
        text = json.dumps({"type": "ReadFile", "FileContent": "hello"})
        line = json.dumps(
            {
                "type": "user",
                "parent_tool_use_id": None,
                "message": {
                    "content": [
                        {"type": "tool_result", "tool_use_id": "c", "content": text}
                    ]
                },
            }
        )

        completed, message = list(parse_stream([line]))

        assert isinstance(completed, ToolCallCompletedEvent)
        assert completed.result.content == [TextContent(type="text", text=text)]
        assert isinstance(message, MessageAddedEvent)
        assert message.message.content == text


_FAKE_GROK = """
import pathlib, sys, time
lines = pathlib.Path(sys.argv[1]).read_text().splitlines()
gate = pathlib.Path(sys.argv[2])
for i, line in enumerate(lines):
    print(line, flush=True)
    if i == 1:
        # Hold the process open until the test has seen the first turn.
        while not gate.exists():
            time.sleep(0.01)
time.sleep(float(sys.argv[3]))
"""


class TestGrokBuildRunStep:
    """run_step against a stand-in `grok` that replays the recorded stream."""

    def _agent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        stream: Path = _FIXTURE,
        linger_seconds: float = 0.0,
    ) -> GrokBuildAgent:
        monkeypatch.setenv("KAROTTE_WORKDIR", str(tmp_path))
        monkeypatch.setattr(
            "karotte.agents.grok_build.agent.make_preexec",
            lambda *_fds: None,  # pyright: ignore[reportUnknownLambdaType]
        )
        agent = _agent(save_artifacts=False)
        agent._grok_home = tmp_path / ".grok_home"  # pyright: ignore[reportPrivateUsage]
        agent._grok_home.mkdir()  # pyright: ignore[reportPrivateUsage]
        script = tmp_path / "fake_grok.py"
        script.write_text(_FAKE_GROK)
        argv = [
            sys.executable,
            str(script),
            str(stream),
            str(tmp_path / "gate"),
            str(linger_seconds),
        ]
        monkeypatch.setattr(agent, "_argv", lambda _instructions: argv)  # pyright: ignore[reportUnknownLambdaType]
        return agent

    @pytest.mark.asyncio
    async def test_yields_each_turn_before_grok_exits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = self._agent(tmp_path, monkeypatch)
        events = agent.run_step("do the thing")
        assert isinstance(await anext(events), MessageAddedEvent)  # instructions
        # The first assistant turn arrives while the stand-in is still blocked
        # on the gate, i.e. before the process has exited.
        first_turn = await asyncio.wait_for(anext(events), 10)
        assert isinstance(first_turn, MessageAddedEvent)
        assert first_turn.message.role == "assistant"
        assert not (tmp_path / "gate").exists()

        (tmp_path / "gate").touch()
        rest = [event async for event in events]
        assert (
            len(rest) == len(list(parse_stream(_FIXTURE.read_text().splitlines()))) - 1
        )
        log = tmp_path / ".grok_home" / "step_0.ndjson"
        assert log.read_text().splitlines() == _FIXTURE.read_text().splitlines()
        prompt = tmp_path / ".grok_home" / "step_0.prompt"
        assert prompt.read_text() == "do the thing"

    @pytest.mark.asyncio
    async def test_time_limit_keeps_output_so_far(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / "gate").touch()
        agent = self._agent(tmp_path, monkeypatch, linger_seconds=30)

        events = [
            event
            async for event in agent.run_step(
                "do the thing", time_limit_seconds=2, on_time_limit="score"
            )
        ]

        expected = list(parse_stream(_FIXTURE.read_text().splitlines()))
        assert len(events) == 1 + len(expected)

    @pytest.mark.asyncio
    async def test_time_limit_error_raises_after_yielding(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / "gate").touch()
        agent = self._agent(tmp_path, monkeypatch, linger_seconds=30)

        seen: list[Event] = []
        with pytest.raises(StepTimeLimitReachedError):
            async for event in agent.run_step(
                "do the thing", time_limit_seconds=2, on_time_limit="error"
            ):
                seen.append(event)
        assert any(
            isinstance(e, MessageAddedEvent) and e.message.content == "All done."
            for e in seen
        )

    @pytest.mark.asyncio
    async def test_parses_lines_longer_than_a_read_chunk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / "gate").touch()
        big = "x" * 300_000
        stream = tmp_path / "big.ndjson"
        stream.write_text(
            json.dumps(
                {
                    "type": "assistant",
                    "parent_tool_use_id": None,
                    "message": {"content": [{"type": "text", "text": big}]},
                }
            )
            + "\n"
        )
        agent = self._agent(tmp_path, monkeypatch, stream=stream)

        events = [event async for event in agent.run_step("do the thing")]

        answers = [
            e.message.content
            for e in events
            if isinstance(e, MessageAddedEvent) and e.message.role == "assistant"
        ]
        assert answers == [big]
