"""Tests for the Claude Code CLI agent: its config, argv and env, and the
`--output-format stream-json` -> transcript-event adapter, driven by a two-step
`claude -p` log (tests/resources/claude_code/)."""

import json
import re
import subprocess
import time
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

_SELF_HOSTED = "hosted_vllm/Qwen/Qwen3-8B"


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


def _in_step(
    agent: ClaudeCodeAgent,
    deadline: float | None = None,
    context_window_limit: int | None = None,
) -> ClaudeCodeAgent:
    """`agent` as `run_step` leaves it just before it starts `claude`."""
    agent._step_deadline = deadline  # pyright: ignore[reportPrivateUsage]
    agent._step_context_window_limit = context_window_limit  # pyright: ignore[reportPrivateUsage]
    return agent


def _hook_notes(argv: list[str]) -> dict[str, str]:
    """Runs each `--settings` hook the way Claude Code does, and returns the
    note each one hands the model, by event."""
    notes: dict[str, str] = {}
    for event, entries in json.loads(_value(argv, "--settings"))["hooks"].items():
        (entry,) = entries
        assert "matcher" not in entry  # every tool
        (hook,) = entry["hooks"]
        ran = subprocess.run(
            ["sh", "-c", hook["command"]], capture_output=True, text=True, check=True
        )
        output = json.loads(ran.stdout)["hookSpecificOutput"]
        assert output["hookEventName"] == event
        notes[event] = output["additionalContext"]
    return notes


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
        assert "--include-hook-events" in first
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

    @pytest.mark.asyncio
    async def test_a_self_hosted_model_reaches_its_server_through_the_forwarder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("KAROTTE_PROXY_URL", "http://vllm:8000")
        monkeypatch.setenv("KAROTTE_WORKDIR", str(tmp_path))
        monkeypatch.setattr("karotte.agents.cli_agent.demoted_uid_gid", lambda: None)
        agent = _agent(model=_SELF_HOSTED, model_api_key="server-key")
        await agent.start(MagicMock())
        try:
            forwarder = agent._forwarder  # pyright: ignore[reportPrivateUsage]
            assert forwarder is not None
            assert forwarder.upstream == "http://vllm:8000"
            env = agent._env()  # pyright: ignore[reportPrivateUsage]
            assert env["ANTHROPIC_BASE_URL"] == forwarder.url
            assert env["ANTHROPIC_API_KEY"] == PROXY_PLACEHOLDER_KEY
            argv = agent._argv("x", None)  # pyright: ignore[reportPrivateUsage]
            assert _value(argv, "--model") == "Qwen/Qwen3-8B"
        finally:
            await agent.stop()

    @pytest.mark.asyncio
    async def test_a_self_hosted_model_without_a_proxy_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("KAROTTE_PROXY_URL", raising=False)
        monkeypatch.setenv("KAROTTE_WORKDIR", str(tmp_path))
        monkeypatch.setattr("karotte.agents.cli_agent.demoted_uid_gid", lambda: None)
        agent = _agent(model=_SELF_HOSTED)
        with pytest.raises(ValueError, match="KAROTTE_PROXY_URL"):
            await agent.start(MagicMock())

    def test_only_a_self_hosted_model_points_the_model_aliases_at_itself(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("KAROTTE_PROXY_URL", "http://vllm:8000")
        monkeypatch.delenv("ANTHROPIC_DEFAULT_HAIKU_MODEL", raising=False)
        hosted = _agent(model=_SELF_HOSTED)._env()  # pyright: ignore[reportPrivateUsage]
        claude = _agent()._env()  # pyright: ignore[reportPrivateUsage]
        assert hosted["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "Qwen/Qwen3-8B"
        assert hosted["CLAUDE_CODE_SUBAGENT_MODEL"] == "Qwen/Qwen3-8B"
        assert "ANTHROPIC_DEFAULT_HAIKU_MODEL" not in claude

    def test_only_a_self_hosted_model_is_called_without_a_thinking_request(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("KAROTTE_PROXY_URL", "http://vllm:8000")
        monkeypatch.delenv("MAX_THINKING_TOKENS", raising=False)
        hosted = _agent(model=_SELF_HOSTED)._env()  # pyright: ignore[reportPrivateUsage]
        claude = _agent()._env()  # pyright: ignore[reportPrivateUsage]
        assert hosted["MAX_THINKING_TOKENS"] == "0"
        assert "MAX_THINKING_TOKENS" not in claude

    def test_only_a_self_hosted_model_is_called_without_an_effort(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("KAROTTE_PROXY_URL", "http://vllm:8000")
        monkeypatch.delenv("CLAUDE_CODE_EFFORT_LEVEL", raising=False)
        hosted = _agent(model=_SELF_HOSTED)
        claude = _agent(reasoning_effort="max")
        assert hosted._env()["CLAUDE_CODE_EFFORT_LEVEL"] == "unset"  # pyright: ignore[reportPrivateUsage]
        assert "--effort" not in hosted._argv("x", None)  # pyright: ignore[reportPrivateUsage]
        assert "CLAUDE_CODE_EFFORT_LEVEL" not in claude._env()  # pyright: ignore[reportPrivateUsage]

    def test_a_timed_step_tells_the_model_its_time_left_after_every_tool_call(self):
        agent = _in_step(_agent(), deadline=time.time() + 100)
        notes = _hook_notes(agent._argv("x", None))  # pyright: ignore[reportPrivateUsage]
        assert set(notes) == {"PostToolUse", "PostToolUseFailure"}
        for note in notes.values():
            match = re.fullmatch(r"Time remaining: (\d+) seconds", note)
            assert match is not None
            assert 95 <= int(match[1]) <= 100

    def test_the_time_left_stops_at_zero(self):
        agent = _in_step(_agent(), deadline=time.time() - 50)
        notes = _hook_notes(agent._argv("x", None))  # pyright: ignore[reportPrivateUsage]
        assert set(notes.values()) == {"Time remaining: 0 seconds"}

    def test_no_time_left_note_without_a_time_limit_or_with_the_counter_off(self):
        untimed = _agent()._argv("x", None)  # pyright: ignore[reportPrivateUsage]
        off = _in_step(
            _agent(inject_time_remaining_counter=False), deadline=time.time() + 100
        )
        assert "--settings" not in untimed
        assert "--settings" not in off._argv("x", None)  # pyright: ignore[reportPrivateUsage]

    def test_a_run_may_set_claude_codes_own_settings_but_not_where_it_calls(self):
        agent_env = {
            "BASH_DEFAULT_TIMEOUT_MS": "3600000",
            "ANTHROPIC_BASE_URL": "http://elsewhere",
        }
        env = _agent(extra_config={"agent_env": agent_env})._env()  # pyright: ignore[reportPrivateUsage]
        assert env["BASH_DEFAULT_TIMEOUT_MS"] == "3600000"
        assert env["ANTHROPIC_BASE_URL"] != "http://elsewhere"

    def test_only_a_self_hosted_model_is_given_the_steps_context_window(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("KAROTTE_PROXY_URL", "http://vllm:8000")
        monkeypatch.delenv("CLAUDE_CODE_MAX_CONTEXT_TOKENS", raising=False)
        hosted = _in_step(
            _agent(model=_SELF_HOSTED), context_window_limit=196608
        )._env()  # pyright: ignore[reportPrivateUsage]
        claude = _in_step(_agent(), context_window_limit=196608)._env()  # pyright: ignore[reportPrivateUsage]
        unlimited = _agent(model=_SELF_HOSTED)._env()  # pyright: ignore[reportPrivateUsage]
        assert hosted["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "196608"
        assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in claude
        assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in unlimited

    def test_reports_the_error_claude_ended_the_step_with(self):
        agent = _agent()
        parse = agent._line_parser("hi")  # pyright: ignore[reportPrivateUsage]
        assert agent._reported_error() is None  # pyright: ignore[reportPrivateUsage]
        result = {
            "type": "result",
            "subtype": "success",
            "is_error": True,
            "result": "API Error: 400 thinking is not supported",
        }
        _ = list(parse(json.dumps(result), 0))
        assert (
            agent._reported_error()  # pyright: ignore[reportPrivateUsage]
            == "API Error: 400 thinking is not supported"
        )

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

    def test_a_tool_hooks_note_stays_with_the_result_it_followed(self):
        # As Claude Code 2.1.292 writes them with --include-hook-events.
        lines = [
            _hook_response("PostToolUse", "Time remaining: 299 seconds"),
            _tool_result("toolu_0", "hello", is_error=False),
            _hook_response("PostToolUseFailure", "Time remaining: 298 seconds"),
            _tool_result("toolu_1", "Exit code 3", is_error=True),
            _tool_result("toolu_2", "no hook ran", is_error=False),
        ]
        first, second, third = _messages(list(parse_stream(lines)), "tool")
        assert first.message.content == [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "Time remaining: 299 seconds"},
        ]
        assert second.message.content == [
            {"type": "text", "text": "Exit code 3"},
            {"type": "text", "text": "Time remaining: 298 seconds"},
        ]
        assert third.message.content == [{"type": "text", "text": "no hook ran"}]

    def test_only_tool_hooks_notes_join_a_tool_result(self):
        lines = [
            _hook_response("SessionStart", "Welcome"),
            _tool_result("toolu_0", "hello", is_error=False),
        ]
        (message,) = _messages(list(parse_stream(lines)), "tool")
        assert message.message.content == [{"type": "text", "text": "hello"}]


def _hook_response(event: str, context: str) -> str:
    output = json.dumps(
        {"hookSpecificOutput": {"hookEventName": event, "additionalContext": context}}
    )
    return json.dumps(
        {
            "type": "system",
            "subtype": "hook_response",
            "hook_name": f"{event}:Bash",
            "hook_event": event,
            "output": output,
            "stdout": output,
            "stderr": "",
            "exit_code": 0,
            "outcome": "success",
        }
    )


def _tool_result(tool_use_id: str, text: str, *, is_error: bool) -> str:
    block = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": text,
        "is_error": is_error,
    }
    return json.dumps({"type": "user", "message": {"role": "user", "content": [block]}})
