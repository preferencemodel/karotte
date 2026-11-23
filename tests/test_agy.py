"""Tests for the Antigravity CLI agent: its config, argv and env, and the
`agy --output-format stream-json` -> transcript-event adapter, driven by a
two-step log (tests/resources/agy/)."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from karotte.agents import cli_agent_types
from karotte.agents.agy import AgyAgent
from karotte.agents.agy.adapter import (
    Session,
    StreamParser,
    parse_stream,
    transcript_usage,
)
from karotte.agents.cli_agent import AGENTS_BIN_DIR
from karotte.model_forwarder import GeminiForwarder
from karotte.providers import PROXY_PLACEHOLDER_KEY
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.transcript import (
    ContextCompactedEvent,
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
        release = (
            "https://github.com/google-antigravity/antigravity-cli/releases/download/"
            f"{AgyAgent.version}"
        )
        assert f"{release}/agy_cli_linux_$arch.tar.gz" in script
        assert "x86_64) arch=x64;" in script
        assert "aarch64) arch=arm64;" in script
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
        substitutes: list[Mapping[str, str] | None] = []

        def vertex_forwarder(subs: Mapping[str, str] | None = None) -> object:
            substitutes.append(subs)
            return made

        monkeypatch.setattr(
            "karotte.agents.agy.agent.vertex_forwarder", vertex_forwarder
        )
        agent = _agent(model="vertex_ai/gemini-4-argon", model_api_key=None)
        assert agent._make_forwarder() is made  # pyright: ignore[reportPrivateUsage]
        assert substitutes == [
            {"gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite"}
        ]

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
        assert not any(c.result.is_error for c in completed)

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
        assert completed.result.is_error
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
                    "step_type": "system_message",
                },
            }
        )
        assert list(parse_stream(["not json", "[1]", other, ""])) == []

    def test_checkpoint_marks_a_compaction_before_the_next_response(self):
        lines = [
            _step(11, "user_input"),
            _step(12, "system_message", duration_seconds=0.000248),
            _CHECKPOINT,
            _step(
                14, "agent_response", text_delta="OK\n", usage={"input_tokens": 10948}
            ),
        ]

        events = list(parse_stream(lines))

        assert [type(e) for e in events] == [
            TokenUsageEvent,
            ContextCompactedEvent,
            MessageAddedEvent,
            TokenUsageEvent,
        ]

    def test_checkpoint_mid_tool_loop_follows_the_results(self):
        lines = [
            _step(
                1, "agent_response", text_delta="Listing.", usage={"input_tokens": 5}
            ),
            _tool(2, "DONE", {"output": "a.txt"}),
            _CHECKPOINT,
            _step(14, "agent_response", text_delta="Done.", usage={"input_tokens": 7}),
        ]

        events = list(parse_stream(lines))

        assert [type(e) for e in events] == [
            MessageAddedEvent,
            TokenUsageEvent,
            ToolCallStartedEvent,
            ToolCallCompletedEvent,
            MessageAddedEvent,
            TokenUsageEvent,
            ContextCompactedEvent,
            MessageAddedEvent,
            TokenUsageEvent,
        ]

    def test_the_compaction_summary_is_a_helper_call(self):
        events = list(parse_stream([_CHECKPOINT]))
        (usage,) = [e for e in events if isinstance(e, TokenUsageEvent)]
        assert usage.source == "helper"
        assert (usage.input_tokens, usage.output_tokens, usage.cache_read_tokens) == (
            275589,
            675,
            208129,
        )

    def test_model_responses_are_the_agents_own(self):
        events = list(parse_stream(_lines("step_0.ndjson")))
        usage = [e for e in events if isinstance(e, TokenUsageEvent)]
        assert {u.source for u in usage} == {"agent"}

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


def _step(index: int, step_type: str, **fields: Any) -> str:
    return json.dumps(
        {
            "event": "step_update",
            "step_update": {
                "step_index": index,
                "state": "DONE",
                "step_type": step_type,
                **fields,
            },
        }
    )


_CHECKPOINT = _step(
    7,
    "checkpoint",
    duration_seconds=8.541613,
    usage={
        "input_tokens": 67460,
        "output_tokens": 675,
        "thinking_tokens": 0,
        "cache_read_tokens": 208129,
        "total_tokens": 68135,
    },
)


_SUBAGENT = "fcff31b0-7d48-41d3-ab6f-36175f296ea5"
_WORKER = "8a4d7844-c16b-49f8-9318-c8d89b0e1f11"


def _transcript(workdir: Path, conversation: str) -> Path:
    return (
        workdir
        / ".gemini"
        / "antigravity-cli"
        / "brain"
        / conversation
        / ".system_generated"
        / "logs"
        / "transcript_full.jsonl"
    )


def _write_transcript(
    workdir: Path, conversation: str, rows: int | None = None
) -> None:
    fixture = (_FIXTURES / "brain" / conversation).joinpath(
        ".system_generated", "logs", "transcript_full.jsonl"
    )
    path = _transcript(workdir, conversation)
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = fixture.read_text().splitlines()[:rows]
    _ = path.write_text("\n".join(kept) + "\n")


def _run_step(agent: AgyAgent, fixture: str | list[str]) -> list[Event]:
    lines = _lines(fixture) if isinstance(fixture, str) else fixture
    parse = agent._line_parser("x")  # pyright: ignore[reportPrivateUsage]
    events = [e for i, line in enumerate(lines) for e in parse(line, i)]
    return events + list(agent._output_ended())  # pyright: ignore[reportPrivateUsage]


def _usage(events: list[Event], source: str) -> tuple[int, int, int]:
    usage = [e for e in events if isinstance(e, TokenUsageEvent) and e.source == source]
    return (
        sum(u.input_tokens for u in usage),
        sum(u.output_tokens for u in usage),
        sum(u.cache_read_tokens or 0 for u in usage),
    )


class TestSubagentUsage:
    @pytest.fixture
    def workdir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setenv("KAROTTE_WORKDIR", str(tmp_path))
        return tmp_path

    def test_a_subagent_is_counted_from_its_transcript(self, workdir: Path):
        _write_transcript(workdir, _SUBAGENT)
        events = _run_step(_agent(), "subagent_step.ndjson")
        assert _usage(events, "subagent") == (14946, 535, 5596)

    def test_subagent_usage_follows_the_steps_own(self, workdir: Path):
        _write_transcript(workdir, _SUBAGENT)
        events = _run_step(_agent(), "subagent_step.ndjson")
        usage = [e for e in events if isinstance(e, TokenUsageEvent)]
        assert [u.source for u in usage] == ["agent"] * 4 + ["subagent"] * 3
        assert _usage(events, "agent") == (41835, 271, 24797)

    def test_a_swarm_worker_is_counted_from_its_transcript(self, workdir: Path):
        _write_transcript(workdir, _WORKER)
        events = _run_step(_agent(), "swarm_step.ndjson")
        assert _usage(events, "subagent") == (21477, 84, 8629)

    def test_a_running_worker_is_counted_on_later_steps_once(self, workdir: Path):
        agent = _agent()
        _write_transcript(workdir, _WORKER, rows=3)
        first = _run_step(agent, "swarm_step.ndjson")
        _write_transcript(workdir, _WORKER)
        second = _run_step(agent, "step_1.ndjson")
        third = _run_step(agent, "step_1.ndjson")
        assert _usage(first, "subagent") == (10675, 38, 0)
        assert _usage(second, "subagent") == (10802, 46, 8629)
        assert _usage(third, "subagent") == (0, 0, 0)

    @pytest.mark.usefixtures("workdir")
    def test_a_missing_transcript_counts_nothing(self):
        events = _run_step(_agent(), "subagent_step.ndjson")
        assert _usage(events, "subagent") == (0, 0, 0)

    def test_the_stdout_log_uri_is_never_read(self, workdir: Path):
        _write_transcript(workdir / "elsewhere", _SUBAGENT)
        logged = _transcript(workdir / "elsewhere", _SUBAGENT).as_uri()
        lines: list[str] = []
        for line in _lines("subagent_step.ndjson"):
            record = json.loads(line)
            info = record.get("step_update", {}).get("subagent_info", {})
            for subagent in info.get("subagents", []):
                subagent["log_uri"] = logged
            lines.append(json.dumps(record))
        events = _run_step(_agent(), lines)
        assert _usage(events, "subagent") == (0, 0, 0)

    def test_only_uuid_conversation_ids_are_read(self, workdir: Path):
        secret = workdir / "secret" / ".system_generated" / "logs"
        secret.mkdir(parents=True)
        _ = (secret / "transcript_full.jsonl").write_text(
            '{"step_index":1,"type":"PLANNER_RESPONSE","input_tokens":5}\n'
        )
        lines = [
            line.replace(_WORKER, "../../../secret")
            for line in _lines("swarm_step.ndjson")
        ]
        events = _run_step(_agent(), lines)
        assert _usage(events, "subagent") == (0, 0, 0)

    @pytest.mark.usefixtures("workdir")
    def test_a_swarm_output_nested_too_deep_names_no_worker(self):
        lines: list[str] = []
        for line in _lines("swarm_step.ndjson"):
            record = json.loads(line)
            info = record.get("step_update", {}).get("tool_info", {})
            if info.get("name") == "swarm":
                info["output"] = '{"a":' * 200_000
            lines.append(json.dumps(record))
        events = _run_step(_agent(), lines)
        assert _usage(events, "subagent") == (0, 0, 0)

    def test_only_rows_that_can_carry_usage_are_parsed(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        parsed: list[str] = []

        def parse_json_line(line: str, _i: int, _label: str) -> dict[str, Any] | None:
            parsed.append(line)
            return json.loads(line)

        monkeypatch.setattr(
            "karotte.agents.agy.adapter.parse_json_line", parse_json_line
        )
        fixture = (_FIXTURES / "brain" / _SUBAGENT).joinpath(
            ".system_generated", "logs", "transcript_full.jsonl"
        )
        lines = fixture.read_text().splitlines()
        _ = list(transcript_usage(_SUBAGENT, lines, set()))
        assert len(parsed) == 3
        assert all("PLANNER_RESPONSE" in line for line in parsed)

    def test_a_subagents_compaction_summary_is_a_helper_call(self, workdir: Path):
        _write_transcript(workdir, _SUBAGENT)
        row = {
            "step_index": 7,
            "source": "SYSTEM",
            "type": "CHECKPOINT",
            "status": "DONE",
            "input_tokens": 67460,
            "cache_read_tokens": 208129,
            "output_tokens": 675,
        }
        with _transcript(workdir, _SUBAGENT).open("a") as f:
            _ = f.write(json.dumps(row) + "\n")
        events = _run_step(_agent(), "subagent_step.ndjson")
        assert _usage(events, "helper") == (275589, 675, 208129)
        assert _usage(events, "subagent") == (14946, 535, 5596)
