"""Tests for the Antigravity CLI agent: its config, argv and env, and the
`agy --output-format stream-json` -> transcript-event adapter, driven by a
two-step log (tests/resources/agy/)."""

import json
from pathlib import Path
from typing import Any

import pytest

from karotte.agents import cli_agent_types
from karotte.agents.agy import AgyAgent
from karotte.agents.agy.adapter import Session, StreamParser, parse_stream
from karotte.agents.cli_agent import AGENTS_BIN_DIR
from karotte.model_forwarder import GeminiForwarder
from karotte.providers import PROXY_PLACEHOLDER_KEY
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.transcript import (
    Event,
    MessageAddedEvent,
    TokenUsageEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)

_FIXTURES = Path(__file__).parent / "resources" / "agy"
_CONVERSATION = "241912c8-06a8-4ea2-b753-284e74119b90"


def _lines(name: str) -> list[str]:
    return (_FIXTURES / name).read_text().splitlines()


def _agent(**kwargs: object) -> AgyAgent:
    config = EvaluationRunConfig.model_validate(
        {
            "run_id": "r",
            "task_id": "t",
            "model": "gemini/gemini-3.8-flash",
            "model_api_key": "real-key",
            **kwargs,
        }
    )
    return AgyAgent(config)


def _settings(agent: AgyAgent) -> dict[str, Any]:
    files = agent._config_files()  # pyright: ignore[reportPrivateUsage]
    return json.loads(files["antigravity-cli/settings.json"])


class TestAgyAgent:
    def test_is_registered(self):
        assert cli_agent_types()["agy"] is AgyAgent

    def test_install_pins_version_and_checksums_per_architecture(self):
        script = " ".join(AgyAgent.install())
        assert f"/{AgyAgent.version}-{AgyAgent.build}/" in script
        assert "/linux-x64/cli_linux_x64.tar.gz" in script
        assert "/linux-arm/cli_linux_arm64.tar.gz" in script
        assert f"{AGENTS_BIN_DIR}/agy" in script
        assert script.count("sha256sum -c") == 1

    def test_capabilities(self):
        assert AgyAgent.prompt_input == "stdin"
        assert AgyAgent.applies_reasoning_effort
        assert not AgyAgent.supports_temperature
        assert not AgyAgent.supports_turn_limit
        assert AgyAgent.providers == ("gemini", "vertex_ai")

    def test_prompt_is_one_stream_json_user_event(self):
        text = _agent()._prompt_text("/model hi\nthere")  # pyright: ignore[reportPrivateUsage]
        assert text.count("\n") == 1
        assert json.loads(text) == {
            "event": "user",
            "message": {"content": "/model hi\nthere"},
        }

    def test_argv_starts_a_conversation_then_resumes_it(self):
        agent = _agent()
        first = agent._argv("x", None)  # pyright: ignore[reportPrivateUsage]
        assert first == [
            f"{AGENTS_BIN_DIR}/agy",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--dangerously-skip-permissions",
            "--disable-slash-commands",
            "--print-timeout",
            "0",
            "--model",
            "gemini-3.8-flash",
        ]
        agent._session.conversation_id = _CONVERSATION  # pyright: ignore[reportPrivateUsage]
        second = agent._argv("x", None)  # pyright: ignore[reportPrivateUsage]
        assert second == [*first, "--conversation", _CONVERSATION]

    def test_argv_sends_the_reasoning_effort(self):
        agent = _agent(model="vertex_ai/gemini-4-argon", reasoning_effort="max")
        argv = agent._argv("x", None)  # pyright: ignore[reportPrivateUsage]
        assert argv[-4:] == ["--model", "gemini-4-argon", "--effort", "high"]

    def test_settings_declare_the_run_model_and_turn_off_telemetry(self):
        settings = _settings(_agent(model="vertex_ai/gemini-4-argon"))
        assert settings["modelProvider"] == "gemini"
        assert settings["enableTelemetry"] is False
        assert settings["customModelsConfig"]["customModels"] == {
            "gemini-4-argon": {
                "modelId": "gemini-4-argon",
                "modelName": "gemini-api://gemini-4-argon",
                "label": "gemini-4-argon",
                "apiProvider": "API_PROVIDER_GOOGLE_GEMINI",
            }
        }

    def test_mcp_config_points_at_the_tool_server(self):
        agent = _agent()
        files = agent._config_files()  # pyright: ignore[reportPrivateUsage]
        assert json.loads(files["config/mcp_config.json"]) == {
            "mcpServers": {"karotte": {"serverUrl": agent.mcp_url}}
        }

    def test_env_points_agy_at_the_model_and_its_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("KAROTTE_WORKDIR", str(tmp_path))
        monkeypatch.setenv("KAROTTE_PROXY_URL", "http://proxy:4000/")
        env = _agent()._env()  # pyright: ignore[reportPrivateUsage]
        assert env["HOME"] == str(tmp_path)
        assert env["GOOGLE_GEMINI_BASE_URL"] == "http://proxy:4000"
        assert env["GEMINI_API_KEY"] == "real-key"
        assert env["AGY_CLI_DISABLE_AUTO_UPDATE"] == "1"

    def test_gemini_models_forward_with_the_api_key(self):
        forwarder = _agent()._make_forwarder()  # pyright: ignore[reportPrivateUsage]
        assert isinstance(forwarder, GeminiForwarder)
        assert forwarder.upstream == "https://generativelanguage.googleapis.com"

    def test_gemini_models_without_a_key_have_no_forwarder(self):
        agent = _agent(model_api_key=None)
        assert agent._make_forwarder() is None  # pyright: ignore[reportPrivateUsage]

    def test_vertex_models_forward_with_the_harness_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        made = object()
        monkeypatch.setattr("karotte.agents.agy.agent.vertex_forwarder", lambda: made)
        agent = _agent(model="vertex_ai/gemini-4-argon", model_api_key=None)
        assert agent._make_forwarder() is made  # pyright: ignore[reportPrivateUsage]

    def test_a_forwarded_agent_gets_a_placeholder_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("KAROTTE_WORKDIR", str(tmp_path))
        monkeypatch.delenv("KAROTTE_PROXY_URL", raising=False)
        agent = _agent()
        agent._forwarder = agent._make_forwarder()  # pyright: ignore[reportPrivateUsage]
        env = agent._env()  # pyright: ignore[reportPrivateUsage]
        assert env["GEMINI_API_KEY"] == PROXY_PLACEHOLDER_KEY
        assert "real-key" not in env.values()

    def test_tool_call_ids_follow_the_agents_step(self):
        agent = _agent()
        agent._step_index = 3  # pyright: ignore[reportPrivateUsage]
        parse = agent._line_parser("x")  # pyright: ignore[reportPrivateUsage]
        events = [
            e for i, line in enumerate(_lines("step_0.ndjson")) for e in parse(line, i)
        ]
        ids = {e.tool_call.id for e in events if isinstance(e, ToolCallStartedEvent)}
        assert "step3_2" in ids

    def test_end_of_output_flushes_the_current_response(self):
        agent = _agent()
        parse = agent._line_parser("x")  # pyright: ignore[reportPrivateUsage]
        for i, line in enumerate(_lines("step_0.ndjson")[:5]):
            _ = list(parse(line, i))
        ended = list(agent._output_ended())  # pyright: ignore[reportPrivateUsage]
        (message,) = _messages(ended, "assistant")
        assert [c.function.name for c in message.message.tool_calls or []] == [
            "run_command"
        ]


def _messages(events: list[Event], role: str) -> list[MessageAddedEvent]:
    return [
        e for e in events if isinstance(e, MessageAddedEvent) and e.message.role == role
    ]


def _started(events: list[Event]) -> list[ToolCallStartedEvent]:
    return [e for e in events if isinstance(e, ToolCallStartedEvent)]


def _completed(events: list[Event]) -> list[ToolCallCompletedEvent]:
    return [e for e in events if isinstance(e, ToolCallCompletedEvent)]


class TestParseStream:
    def test_records_the_conversation(self):
        session = Session()
        _ = list(parse_stream(_lines("step_0.ndjson"), session))
        assert session.conversation_id == _CONVERSATION

    def test_each_model_response_is_one_message_with_its_calls(self):
        events = list(parse_stream(_lines("step_0.ndjson")))
        names = [
            [c.function.name for c in m.message.tool_calls or []]
            for m in _messages(events, "assistant")
        ]
        assert names == [
            ["run_command", "view_file"],
            ["mcp__karotte__add"],
            ["write_to_file"],
            [],
        ]

    def test_the_final_message_is_the_streamed_text(self):
        events = list(parse_stream(_lines("step_0.ndjson")))
        final = _messages(events, "assistant")[-1].message
        assert final.content == (
            "### `ls` Output\n```\nnotes.txt\n```\n\n### Sum\n**42** "
            "(17 + 25 via `karotte` `add`)\n\nThe sum has been written to "
            "[result.txt](file:///workdir/result.txt).\n"
        )

    def test_results_follow_the_message_that_called_them(self):
        events = list(parse_stream(_lines("step_0.ndjson")))
        open_calls: set[str] = set()
        for event in events:
            if not isinstance(event, MessageAddedEvent):
                continue
            message = event.message
            if message.role == "assistant":
                assert not open_calls
                open_calls = {c.id for c in message.tool_calls or []}
            elif message.role == "tool":
                assert message.tool_call_id in open_calls
                open_calls.discard(message.tool_call_id or "")
        assert [c.tool_call_id for c in _completed(events)] == [
            s.tool_call.id for s in _started(events)
        ]

    def test_tool_arguments_and_results(self):
        events = list(parse_stream(_lines("step_0.ndjson")))
        started = [s.tool_call for s in _started(events)]
        completed = _completed(events)
        assert json.loads(started[0].function.arguments) == {"CommandLine": "ls"}
        assert completed[0].result.content[0].text == "notes.txt\r\n"  # pyright: ignore[reportAttributeAccessIssue]
        assert json.loads(started[2].function.arguments) == {"a": 17, "b": 25}
        assert completed[2].result.content[0].text == "42"  # pyright: ignore[reportAttributeAccessIssue]
        assert not completed[3].result.content
        assert not any(c.result.isError for c in completed)

    def test_usage_is_per_model_response(self):
        events = list(parse_stream(_lines("step_0.ndjson")))
        usage = [e for e in events if isinstance(e, TokenUsageEvent)]
        assert [
            (u.input_tokens, u.output_tokens, u.cache_read_tokens) for u in usage
        ] == [
            (11658, 725, 0),
            (12728, 182, 7319),
            (12955, 371, 10970),
            (13466, 391, 10985),
        ]

    def test_a_resumed_step_counts_only_its_own_usage(self):
        events = list(parse_stream(_lines("step_1.ndjson")))
        (usage,) = [e for e in events if isinstance(e, TokenUsageEvent)]
        assert (usage.input_tokens, usage.output_tokens) == (14151, 44)
        (answer,) = _messages(events, "assistant")
        assert answer.message.content == "42\n"

    def test_a_failed_tool_is_an_error_result(self):
        lines = [
            _tool(2, "ACTIVE", {}),
            _tool(
                2,
                "ERROR",
                {"error": {"type": "TOOL_ERROR", "message": "invalid arguments"}},
            ),
        ]
        (completed,) = _completed(list(parse_stream(lines)))
        assert completed.result.isError
        assert completed.result.content[0].text == "invalid arguments"  # pyright: ignore[reportAttributeAccessIssue]

    def test_tool_call_ids_differ_between_steps(self):
        session = Session()
        first = list(parse_stream(_lines("step_0.ndjson"), session, step=0))
        second = list(parse_stream(_lines("step_0.ndjson"), session, step=1))
        first_ids = {s.tool_call.id for s in _started(first)}
        assert first_ids == {"step0_2", "step0_3", "step0_5", "step0_7"}
        assert first_ids.isdisjoint({s.tool_call.id for s in _started(second)})

    def test_skips_malformed_lines_and_other_steps(self):
        other = json.dumps(
            {
                "event": "step_update",
                "step_update": {
                    "step_index": 0,
                    "state": "DONE",
                    "step_type": "checkpoint",
                },
            }
        )
        assert list(parse_stream(["not json", "[1]", other, ""])) == []

    def test_flush_keeps_what_a_killed_step_left_unsent(self):
        parse = StreamParser(Session(), 0)
        lines = _lines("step_0.ndjson")[:5]
        assert not _messages(
            [e for i, line in enumerate(lines) for e in parse(line, i)], "assistant"
        )
        flushed = list(parse.flush())
        (message,) = _messages(flushed, "assistant")
        assert [c.id for c in message.message.tool_calls or []] == ["step0_2"]
        assert [c.tool_call_id for c in _completed(flushed)] == ["step0_2"]
        assert list(parse.flush()) == []


def _tool(index: int, state: str, info: dict[str, Any]) -> str:
    return json.dumps(
        {
            "event": "step_update",
            "step_update": {
                "step_index": index,
                "state": state,
                "step_type": "tool",
                "tool_name": "run_command",
                "tool_info": {
                    "name": "run_command",
                    "parameters": {"CommandLine": "x"},
                    **info,
                },
            },
        }
    )
