"""Tests for the Codex CLI agent: its config, argv and env, and the
`codex exec --json` -> transcript-event adapter, driven by a two-step log
(tests/resources/codex/)."""

import json
import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from karotte.agents import cli_agent_types
from karotte.agents.cli_agent import AGENTS_BIN_DIR
from karotte.agents.codex import CodexAgent
from karotte.agents.codex.adapter import Session, StreamParser, parse_stream
from karotte.providers import PROXY_PLACEHOLDER_KEY
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.transcript import (
    Event,
    MessageAddedEvent,
    TokenUsageEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)

_FIXTURES = Path(__file__).parent / "resources" / "codex"
_THREAD = "01a141fd-76c6-7d92-8a94-779c40ac6c65"
_ZERO = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)


def _lines(name: str) -> list[str]:
    return (_FIXTURES / name).read_text().splitlines()


def _agent(**kwargs: object) -> CodexAgent:
    config = EvaluationRunConfig.model_validate(
        {
            "run_id": "r",
            "task_id": "t",
            "model": "openai/gpt-6.1-sol",
            "model_api_key": "sk-real",
            **kwargs,
        }
    )
    return CodexAgent(config)


def _toml(agent: CodexAgent) -> dict[str, Any]:
    return tomllib.loads(agent._config_files()["config.toml"])  # pyright: ignore[reportPrivateUsage]


class TestCodexAgent:
    def test_is_registered(self):
        assert cli_agent_types()["codex"] is CodexAgent

    def test_install_pins_version_and_checksums_for_both_binaries(self):
        script = " ".join(CodexAgent.install())
        tag = f"rust-v{CodexAgent.version}"
        for binary in ("codex", "codex-code-mode-host"):
            assert f"{tag}/{binary}-$arch-unknown-linux-musl.tar.gz" in script
            assert f"{AGENTS_BIN_DIR}/{binary}" in script
        assert script.count("sha256sum -c") == 2

    def test_install_leaves_the_binaries_to_root(self):
        commands = CodexAgent.install()
        assert all("tar --no-same-owner -xzf" in command for command in commands)

    def test_capabilities(self):
        assert CodexAgent.prompt_input == "stdin"
        assert CodexAgent.applies_reasoning_effort
        assert not CodexAgent.supports_temperature
        assert not CodexAgent.supports_turn_limit

    def test_argv_starts_a_session_then_resumes_it(self):
        agent = _agent()
        first = agent._argv("x", None)  # pyright: ignore[reportPrivateUsage]
        assert first == [
            f"{AGENTS_BIN_DIR}/codex",
            "exec",
            "--json",
            "--dangerously-bypass-approvals-and-sandbox",
            "-",
        ]
        agent._session.thread_id = _THREAD  # pyright: ignore[reportPrivateUsage]
        second = agent._argv("x", None)  # pyright: ignore[reportPrivateUsage]
        assert second == [*first[:-1], "resume", _THREAD, "-"]

    def test_config_points_at_the_model_and_tool_server(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("KAROTTE_PROXY_URL", "http://proxy:4000/")
        agent = _agent(reasoning_effort="max")
        config = _toml(agent)
        assert config["model"] == "gpt-6.1-sol"
        assert config["model_provider"] == "karotte"
        provider = config["model_providers"]["karotte"]
        assert provider["base_url"] == "http://proxy:4000/v1"
        assert provider["env_key"] == "OPENAI_API_KEY"
        assert config["model_reasoning_effort"] == "max"
        assert config["model_reasoning_summary"] == "detailed"
        assert config["mcp_servers"]["karotte"]["url"] == agent.mcp_url

    def test_config_turns_off_what_reaches_past_the_model(self):
        config = _toml(_agent())
        assert config["web_search"] == "disabled"
        assert config["analytics"]["enabled"] is False
        assert config["otel"]["metrics_exporter"] == "none"
        assert config["feedback"]["enabled"] is False
        assert config["features"]["unbounded_connection_retries"] is False
        assert config["shell_environment_policy"]["ignore_default_excludes"] is False
        assert "model_reasoning_effort" not in config

    def test_env_points_codex_at_its_home(self):
        env = _agent()._env()  # pyright: ignore[reportPrivateUsage]
        assert env["CODEX_HOME"].endswith(".codex_home")
        assert env["OPENAI_API_KEY"] == "sk-real"

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
            assert env["OPENAI_API_KEY"] == PROXY_PLACEHOLDER_KEY
            assert "sk-real" not in env.values()
            config = tomllib.loads(
                (tmp_path / ".codex_home" / "config.toml").read_text()
            )
            assert config["model_providers"]["karotte"]["base_url"].startswith(
                "http://127.0.0.1:"
            )
        finally:
            await agent.stop()

    def test_tool_call_ids_follow_the_agents_step(self):
        agent = _agent()
        agent._step_index = 3  # pyright: ignore[reportPrivateUsage]
        parse = agent._line_parser("x")  # pyright: ignore[reportPrivateUsage]
        events = [
            e for i, line in enumerate(_lines("step_0.ndjson")) for e in parse(line, i)
        ]
        ids = {e.tool_call.id for e in events if isinstance(e, ToolCallStartedEvent)}
        assert "step3_item_2" in ids

    def test_end_of_output_flushes_the_current_step(self):
        agent = _agent()
        parse = agent._line_parser("x")  # pyright: ignore[reportPrivateUsage]
        for i, line in enumerate(_lines("step_0.ndjson")[:4]):
            _ = list(parse(line, i))
        ended = list(agent._output_ended())  # pyright: ignore[reportPrivateUsage]
        (message,) = _messages(ended, "assistant")
        assert message.message.reasoning_content


def _messages(events: list[Event], role: str) -> list[MessageAddedEvent]:
    return [
        e for e in events if isinstance(e, MessageAddedEvent) and e.message.role == role
    ]


class TestParseStream:
    def test_records_the_session(self):
        session = Session()
        _ = list(parse_stream(_lines("step_0.ndjson"), session))
        assert session.thread_id == _THREAD

    def test_each_tool_call_gets_its_own_assistant_message(self):
        events = list(parse_stream(_lines("step_0.ndjson")))
        assistant = _messages(events, "assistant")
        names = [
            [c.function.name for c in m.message.tool_calls or []] for m in assistant
        ]
        assert names == [
            ["exec_command"],
            ["mcp__karotte__add"],
            ["apply_patch"],
            ["exec_command"],
            [],
        ]
        assert "7 × 11 × 13" in str(assistant[-1].message.content)

    def test_text_and_reasoning_go_with_the_next_call(self):
        events = list(parse_stream(_lines("step_0.ndjson")))
        first = _messages(events, "assistant")[0].message
        assert str(first.content).startswith("I’ll check whether 1001 is prime")
        assert first.reasoning_content
        assert first.reasoning_content.startswith("**Checking tool setup**")

    def test_tool_calls_pair_with_their_results(self):
        events = list(parse_stream(_lines("step_0.ndjson")))
        started = [e.tool_call for e in events if isinstance(e, ToolCallStartedEvent)]
        completed = [e for e in events if isinstance(e, ToolCallCompletedEvent)]
        assert [c.tool_call_id for c in completed] == [c.id for c in started]
        positions = {id(e): i for i, e in enumerate(events)}
        for call, done in zip(started, completed, strict=True):
            call_event = next(
                e
                for e in events
                if isinstance(e, ToolCallStartedEvent) and e.tool_call is call
            )
            assert positions[id(call_event)] < positions[id(done)]

    def test_shell_results_carry_output_and_failure(self):
        events = list(parse_stream(_lines("step_0.ndjson")))
        started = [e.tool_call for e in events if isinstance(e, ToolCallStartedEvent)]
        completed = [e for e in events if isinstance(e, ToolCallCompletedEvent)]
        assert "rg --files" in json.loads(started[0].function.arguments)["command"]
        assert completed[0].result.isError
        assert not completed[3].result.isError
        assert "result.txt" in completed[3].result.content[0].text  # pyright: ignore[reportAttributeAccessIssue]

    def test_mcp_and_file_change_calls(self):
        events = list(parse_stream(_lines("step_0.ndjson")))
        started = [e.tool_call for e in events if isinstance(e, ToolCallStartedEvent)]
        completed = [e for e in events if isinstance(e, ToolCallCompletedEvent)]
        assert json.loads(started[1].function.arguments) == {"a": 17, "b": 25}
        assert completed[1].result.content[0].text == "42"  # pyright: ignore[reportAttributeAccessIssue]
        assert json.loads(started[2].function.arguments) == {
            "changes": [{"path": "/workdir/result.txt", "kind": "add"}]
        }
        assert not completed[2].result.isError

    def test_an_mcp_result_with_text_and_an_image_keeps_both(self):
        call = {
            "id": "item_1",
            "type": "mcp_tool_call",
            "server": "karotte",
            "tool": "screenshot",
            "arguments": {},
            "result": {
                "content": [
                    {"type": "text", "text": "caption"},
                    {"type": "image", "data": "QUJD", "mimeType": "image/png"},
                ]
            },
            "error": None,
            "status": "completed",
        }
        events = list(parse_stream([_started(call), _completed(call)]))
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

    def test_usage_is_per_step_although_codex_reports_session_totals(self):
        session = Session()
        first = list(parse_stream(_lines("step_0.ndjson"), session))
        second = list(parse_stream(_lines("step_1.ndjson"), session))
        (usage_0,) = [e for e in first if isinstance(e, TokenUsageEvent)]
        (usage_1,) = [e for e in second if isinstance(e, TokenUsageEvent)]
        assert (usage_0.input_tokens, usage_0.output_tokens) == (45144, 369)
        assert (usage_0.cache_read_tokens, usage_0.cache_write_tokens) == (33590, 11542)
        assert (usage_1.input_tokens, usage_1.output_tokens) == (11606, 5)
        assert (usage_1.cache_read_tokens, usage_1.cache_write_tokens) == (11542, 61)

    def test_a_new_thread_starts_usage_from_zero(self):
        session = Session()
        _ = list(parse_stream(_lines("step_0.ndjson"), session))
        other = [
            line.replace(_THREAD, "another-thread") for line in _lines("step_1.ndjson")
        ]
        (usage,) = [
            e for e in parse_stream(other, session) if isinstance(e, TokenUsageEvent)
        ]
        assert (usage.input_tokens, usage.output_tokens) == (56750, 374)

    def test_totals_without_usage_never_count_negative(self):
        session = Session()
        _ = list(parse_stream(_lines("step_0.ndjson"), session))
        lines = [
            json.dumps({"type": "thread.started", "thread_id": _THREAD}),
            # Codex reports zeros when no usage update reached it during the turn.
            json.dumps({"type": "turn.completed", "usage": dict.fromkeys(_ZERO, 0)}),
            *_lines("step_1.ndjson"),
        ]
        usage = [
            e for e in parse_stream(lines, session) if isinstance(e, TokenUsageEvent)
        ]
        assert [(u.input_tokens, u.output_tokens) for u in usage] == [
            (0, 0),
            (11606, 5),
        ]

    def test_resumed_step_answers_from_the_session(self):
        events = list(parse_stream(_lines("step_1.ndjson")))
        (answer,) = _messages(events, "assistant")
        assert answer.message.content == "42"

    def test_skips_error_items_and_malformed_lines(self):
        error = json.dumps(
            {
                "type": "item.completed",
                "item": {"id": "i", "type": "error", "message": "x"},
            }
        )
        assert list(parse_stream(["not json", "[1]", error, ""])) == []

    def test_tool_call_ids_differ_between_steps(self):
        session = Session()
        first = list(parse_stream(_lines("step_0.ndjson"), session, step=0))
        second = list(parse_stream(_lines("step_0.ndjson"), session, step=1))
        first_ids = {
            e.tool_call.id for e in first if isinstance(e, ToolCallStartedEvent)
        }
        second_ids = {
            e.tool_call.id for e in second if isinstance(e, ToolCallStartedEvent)
        }
        assert len(first_ids) == 4
        assert first_ids.isdisjoint(second_ids)
        assert "step0_item_2" in first_ids

    def test_a_retried_error_does_not_split_text_from_the_next_call(self):
        lines = [
            _completed({"id": "item_0", "type": "agent_message", "text": "Listing."}),
            json.dumps({"type": "error", "message": "Reconnecting... 1/5"}),
            _completed(_command("item_1")),
        ]
        (message,) = _messages(list(parse_stream(lines)), "assistant")
        assert message.message.content == "Listing."
        assert message.message.tool_calls

    def test_tool_items_without_an_id_stay_apart(self):
        first, second = _command(None), _command(None)
        lines = [
            _started(first),
            _completed(first),
            _started(second),
            _completed(second),
        ]
        events = list(parse_stream(lines))
        started = [
            e.tool_call.id for e in events if isinstance(e, ToolCallStartedEvent)
        ]
        completed = [
            e.tool_call_id for e in events if isinstance(e, ToolCallCompletedEvent)
        ]
        assert len(set(started)) == 2
        assert completed == started

    def test_concurrent_calls_share_one_message_followed_by_their_results(self):
        a, b = _command("a"), _command("b")
        lines = [
            _completed({"id": "t", "type": "agent_message", "text": "Both."}),
            _started(a),
            _started(b),
            _completed(a),
            _completed(b),
        ]
        messages = [
            e.message for e in parse_stream(lines) if isinstance(e, MessageAddedEvent)
        ]
        assert [(m.role, m.tool_call_id) for m in messages] == [
            ("assistant", None),
            ("tool", "step0_a"),
            ("tool", "step0_b"),
        ]
        assert messages[0].content == "Both."
        assert [c.id for c in messages[0].tool_calls or []] == ["step0_a", "step0_b"]

    def test_calls_run_one_after_another_get_a_message_each(self):
        a, b = _command("a"), _command("b")
        lines = [_started(a), _completed(a), _started(b), _completed(b)]
        roles = [
            e.message.role
            for e in parse_stream(lines)
            if isinstance(e, MessageAddedEvent)
        ]
        assert roles == ["assistant", "tool", "assistant", "tool"]

    def test_flush_sends_running_calls_with_the_results_they_got(self):
        parse = StreamParser(Session(), 0)
        a, b = _command("a"), _command("b")
        events = [
            e
            for i, line in enumerate([_started(a), _started(b), _completed(a)])
            for e in parse(line, i)
        ]
        assert not events
        flushed = list(parse.flush())
        (message,) = _messages(flushed, "assistant")
        assert [c.id for c in message.message.tool_calls or []] == [
            "step0_a",
            "step0_b",
        ]
        assert [
            e.tool_call_id for e in flushed if isinstance(e, ToolCallCompletedEvent)
        ] == ["step0_a"]

    def test_flush_keeps_what_a_killed_step_left_unsent(self):
        parse = StreamParser(Session(), 0)
        lines = _lines("step_0.ndjson")[:4]
        assert not _messages(
            [e for i, line in enumerate(lines) for e in parse(line, i)], "assistant"
        )
        (message,) = _messages(list(parse.flush()), "assistant")
        assert str(message.message.content).startswith("I’ll check whether 1001")
        assert message.message.reasoning_content


def _command(item_id: str | None) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "command_execution",
        "command": "ls",
        "aggregated_output": "",
        "exit_code": 0,
        "status": "completed",
    }


def _started(item: dict[str, Any]) -> str:
    return json.dumps({"type": "item.started", "item": item})


def _completed(item: dict[str, Any]) -> str:
    return json.dumps({"type": "item.completed", "item": item})
