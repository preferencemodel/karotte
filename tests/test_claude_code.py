"""Tests for the Claude Code CLI agent: its config, argv and env, and the
`--output-format stream-json` -> transcript-event adapter, driven by a two-step
`claude -p` log (tests/resources/claude_code/)."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from karotte.agents import cli_agent_types
from karotte.agents.claude_code import ClaudeCodeAgent
from karotte.agents.claude_code.adapter import parse_stream
from karotte.agents.cli_agent import AGENTS_BIN_DIR
from karotte.providers import PROXY_PLACEHOLDER_KEY
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.transcript import (
    Event,
    MessageAddedEvent,
    TokenUsageEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)

_FIXTURES = Path(__file__).parent / "resources" / "claude_code"


def _lines(name: str) -> list[str]:
    return (_FIXTURES / name).read_text().splitlines()


def _agent(**kwargs: object) -> ClaudeCodeAgent:
    config = EvaluationRunConfig.model_validate(
        {
            "run_id": "r",
            "task_id": "t",
            "model": "claude-sonnet-5-5",
            "model_api_key": "sk-ant-real",
            **kwargs,
        }
    )
    return ClaudeCodeAgent(config)


def _value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


class TestClaudeCodeAgent:
    def test_is_registered(self):
        assert cli_agent_types()["claude-code"] is ClaudeCodeAgent

    def test_install_pins_version_and_checksums(self):
        script = " ".join(ClaudeCodeAgent.install())
        assert f"claude-code-releases/{ClaudeCodeAgent.version}/linux-" in script
        assert "sha256sum -c" in script
        assert f"{AGENTS_BIN_DIR}/claude" in script

    def test_reads_the_prompt_from_stdin(self):
        assert ClaudeCodeAgent.prompt_input == "stdin"

    def test_argv_creates_then_resumes_one_session(self):
        agent = _agent(turn_limit=7, reasoning_effort="max")
        first = agent._argv("x", None)  # pyright: ignore[reportPrivateUsage]
        assert first[:2] == [f"{AGENTS_BIN_DIR}/claude", "-p"]
        assert "x" not in first
        assert _value(first, "--model") == "claude-sonnet-5-5"
        assert _value(first, "--output-format") == "stream-json"
        assert "--include-partial-messages" in first
        assert _value(first, "--thinking-display") == "summarized"
        assert _value(first, "--permission-mode") == "bypassPermissions"
        assert _value(first, "--max-turns") == "7"
        assert _value(first, "--effort") == "max"
        assert "--strict-mcp-config" in first
        assert (
            _value(first, "--disallowedTools") == "WebSearch,WebFetch,AskUserQuestion"
        )
        session_id = _value(first, "--session-id")

        agent._step_index = 1  # pyright: ignore[reportPrivateUsage]
        second = agent._argv("x", None)  # pyright: ignore[reportPrivateUsage]
        assert "--session-id" not in second
        assert _value(second, "--resume") == session_id

    def test_argv_omits_unset_limits_and_effort(self):
        argv = _agent()._argv("x", None)  # pyright: ignore[reportPrivateUsage]
        assert "--max-turns" not in argv
        assert "--effort" not in argv

    def test_strips_a_provider_prefix_from_the_model(self):
        argv = _agent(model="anthropic/claude-sonnet-5-5")._argv("x", None)  # pyright: ignore[reportPrivateUsage]
        assert _value(argv, "--model") == "claude-sonnet-5-5"

    def test_mcp_config_points_at_the_tool_server(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("KAROTTE_WORKDIR", str(tmp_path))
        monkeypatch.setattr("karotte.agents.cli_agent.demoted_uid_gid", lambda: None)
        agent = _agent()
        agent._write_config()  # pyright: ignore[reportPrivateUsage]
        config_path = Path(
            _value(agent._argv("x", None), "--mcp-config")  # pyright: ignore[reportPrivateUsage]
        )
        assert config_path.parent == tmp_path / ".claude_home"
        server = json.loads(config_path.read_text())["mcpServers"]["karotte"]
        assert server == {"type": "http", "url": agent.mcp_url}

    def test_env_points_claude_at_its_home_and_model_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("KAROTTE_PROXY_URL", "http://proxy:4000/")
        env = _agent()._env()  # pyright: ignore[reportPrivateUsage]
        assert env["CLAUDE_CONFIG_DIR"].endswith(".claude_home")
        assert env["ANTHROPIC_BASE_URL"] == "http://proxy:4000"
        assert env["ANTHROPIC_API_KEY"] == "sk-ant-real"
        assert env["DISABLE_AUTOUPDATER"] == "1"
        assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"

    @pytest.mark.asyncio
    async def test_start_without_proxy_forwards_and_withholds_the_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("KAROTTE_PROXY_URL", raising=False)
        monkeypatch.setenv("KAROTTE_WORKDIR", str(tmp_path))
        monkeypatch.setattr("karotte.agents.cli_agent.demoted_uid_gid", lambda: None)
        agent = _agent()
        await agent.start(MagicMock())
        try:
            env = agent._env()  # pyright: ignore[reportPrivateUsage]
            assert env["ANTHROPIC_BASE_URL"].startswith("http://127.0.0.1:")
            assert env["ANTHROPIC_API_KEY"] == PROXY_PLACEHOLDER_KEY
            assert "sk-ant-real" not in env.values()
        finally:
            await agent.stop()

    def test_end_of_output_keeps_the_unfinished_response_and_its_results(self):
        agent = _agent()
        parse = agent._line_parser("x")  # pyright: ignore[reportPrivateUsage]
        # Cut after two tool calls have run, before the response's message_stop.
        for i, line in enumerate(_lines("step_0.ndjson")[:100]):
            _ = list(parse(line, i))
        ended = list(agent._output_ended())  # pyright: ignore[reportPrivateUsage]
        (message,) = _messages(ended, "assistant")
        assert message.message.content
        assert message.message.reasoning_content
        calls = [c.id for c in message.message.tool_calls or []]
        assert len(calls) == 2
        assert [
            e.tool_call.id for e in ended if isinstance(e, ToolCallStartedEvent)
        ] == calls
        assert [
            e.tool_call_id for e in ended if isinstance(e, ToolCallCompletedEvent)
        ] == calls
        assert list(agent._output_ended()) == []  # pyright: ignore[reportPrivateUsage]

    def test_end_of_output_before_any_content_keeps_only_the_usage(self):
        agent = _agent()
        parse = agent._line_parser("x")  # pyright: ignore[reportPrivateUsage]
        start = {
            "type": "stream_event",
            "event": {
                "type": "message_start",
                "message": {"usage": {"input_tokens": 5}},
            },
        }
        _ = list(parse(json.dumps(start), 0))
        ended = list(agent._output_ended())  # pyright: ignore[reportPrivateUsage]
        assert not [e for e in ended if isinstance(e, MessageAddedEvent)]
        (usage,) = [e for e in ended if isinstance(e, TokenUsageEvent)]
        assert usage.input_tokens == 5

    def test_end_of_output_after_a_finished_response_adds_nothing(self):
        agent = _agent()
        parse = agent._line_parser("x")  # pyright: ignore[reportPrivateUsage]
        for i, line in enumerate(_lines("step_0.ndjson")):
            _ = list(parse(line, i))
        assert list(agent._output_ended()) == []  # pyright: ignore[reportPrivateUsage]


def _messages(events: list[Event], role: str) -> list[MessageAddedEvent]:
    return [
        e for e in events if isinstance(e, MessageAddedEvent) and e.message.role == role
    ]


class TestParseStream:
    def test_one_message_per_model_response(self):
        events = list(parse_stream(_lines("step_0.ndjson")))
        assistant = _messages(events, "assistant")
        assert len(assistant) == 3
        assert assistant[-1].message.content
        assert "42" in str(assistant[-1].message.content)

    def test_tool_calls_pair_with_their_results(self):
        events = list(parse_stream(_lines("step_0.ndjson")))
        started = [e.tool_call for e in events if isinstance(e, ToolCallStartedEvent)]
        completed = [e for e in events if isinstance(e, ToolCallCompletedEvent)]
        assert [c.function.name for c in started] == [
            "mcp__karotte__add",
            "Read",
            "Bash",
            "Write",
        ]
        assert [c.tool_call_id for c in completed] == [c.id for c in started]
        assert json.loads(started[0].function.arguments) == {"a": 17, "b": 25}

    def test_tool_calls_follow_the_message_that_makes_them(self):
        events = list(parse_stream(_lines("step_0.ndjson")))
        first_call = next(
            i for i, e in enumerate(events) if isinstance(e, ToolCallStartedEvent)
        )
        first_assistant = next(
            i
            for i, e in enumerate(events)
            if isinstance(e, MessageAddedEvent) and e.message.role == "assistant"
        )
        assert first_assistant < first_call

    def test_results_that_arrive_mid_response_follow_their_calls(self):
        events = list(parse_stream(_lines("step_0.ndjson")))
        started = {
            e.tool_call.id: i
            for i, e in enumerate(events)
            if isinstance(e, ToolCallStartedEvent)
        }
        for i, e in enumerate(events):
            if isinstance(e, ToolCallCompletedEvent):
                assert started[e.tool_call_id] < i

    def test_an_image_result_keeps_its_bytes_only_in_the_message(self):
        events = list(parse_stream(_lines("step_0.ndjson")))
        completed = [e for e in events if isinstance(e, ToolCallCompletedEvent)]
        image_result = completed[1].result.content[0]
        assert image_result.type == "image"
        assert image_result.data == ""
        tool_messages = _messages(events, "tool")
        part = tool_messages[1].message.content[0]  # pyright: ignore[reportOptionalSubscript]
        assert part["image_url"]["url"].startswith("data:image/png;base64,iVBOR")  # pyright: ignore[reportArgumentType]

    def test_a_result_with_text_and_an_image_keeps_both(self):
        record = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [
                            {"type": "text", "text": "caption"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "QUJD",
                                },
                            },
                        ],
                    }
                ],
            },
        }
        events = list(parse_stream([json.dumps(record)]))
        (completed,) = [e for e in events if isinstance(e, ToolCallCompletedEvent)]
        assert [
            b.model_dump(include={"type", "data"}) for b in completed.result.content
        ] == [
            {"type": "text"},
            {"type": "image", "data": ""},
        ]
        (message,) = _messages(events, "tool")
        assert message.message.content == [
            {"type": "text", "text": "caption"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
        ]

    def test_usage_is_reported_once_per_response_and_adds_up_to_claudes_total(
        self,
    ):
        lines = _lines("step_0.ndjson")
        total = json.loads(lines[-1])["usage"]
        events = list(parse_stream(lines))
        usage = [e for e in events if isinstance(e, TokenUsageEvent)]
        assert len(usage) == 3
        assert sum(u.output_tokens for u in usage) == total["output_tokens"]
        assert sum(u.input_tokens for u in usage) == (
            total["input_tokens"]
            + total["cache_read_input_tokens"]
            + total["cache_creation_input_tokens"]
        )
        assert not any(u.estimated for u in usage)

    def test_usage_input_includes_cache_buckets(self):
        events = list(parse_stream(_lines("step_1.ndjson")))
        (usage,) = [e for e in events if isinstance(e, TokenUsageEvent)]
        assert usage.input_tokens > 0
        assert usage.cache_read_tokens is not None
        assert usage.cache_write_tokens is not None
        assert usage.input_tokens >= usage.cache_read_tokens + usage.cache_write_tokens

    def test_resumed_step_answers_from_the_session(self):
        events = list(parse_stream(_lines("step_1.ndjson")))
        (answer,) = _messages(events, "assistant")
        assert answer.message.content == "42"

    def test_keeps_the_thinking_summary(self):
        events = list(parse_stream(_lines("step_0.ndjson")))
        first, *rest = _messages(events, "assistant")
        assert first.message.reasoning_content
        assert "7 × 11 × 13" in first.message.reasoning_content
        assert all(m.message.reasoning_content is None for m in rest)

    def test_skips_subagent_and_malformed_lines(self):
        subagent = json.dumps(
            {
                "type": "assistant",
                "parent_tool_use_id": "toolu_1",
                "message": {"id": "m", "content": [{"type": "text", "text": "x"}]},
            }
        )
        assert list(parse_stream(["not json", "[1]", subagent, ""])) == []
