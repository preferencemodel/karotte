"""Tests for the Muse Code CLI agent: its config, argv and env, and the
session log -> transcript-event adapter, driven by a real two-step log
(tests/resources/muse_code/)."""

import json
from pathlib import Path
from typing import Any

import pytest
from mcp.types import ImageContent, TextContent

from karotte.agents import cli_agent_types
from karotte.agents.cli_agent import AGENTS_BIN_DIR
from karotte.agents.muse_code import MuseCodeAgent
from karotte.agents.muse_code.adapter import StreamParser
from karotte.model_forwarder import MuseForwarder
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

_FIXTURES = Path(__file__).parent / "resources" / "muse_code"


def _lines(name: str) -> list[str]:
    return (_FIXTURES / name).read_text().splitlines()


def _parse(lines: list[str]) -> list[Event]:
    parse = StreamParser()
    events = [e for i, line in enumerate(lines) for e in parse(line, i)]
    return [*events, *parse.flush()]


def _agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: object
) -> MuseCodeAgent:
    monkeypatch.setenv("KAROTTE_WORKDIR", str(tmp_path))
    monkeypatch.delenv("KAROTTE_PROXY_URL", raising=False)
    config = EvaluationRunConfig.model_validate(
        {
            "run_id": "r",
            "task_id": "t",
            "model": "meta/muse-spark-1.3",
            "model_api_key": "real-key",
            **kwargs,
        }
    )
    return MuseCodeAgent(config)


def _settings(agent: MuseCodeAgent) -> dict[str, Any]:
    files = agent._config_files()  # pyright: ignore[reportPrivateUsage]
    return json.loads(files["config/muse/settings.json"])


class TestMuseCodeAgent:
    def test_is_registered(self):
        assert cli_agent_types()["muse-code"] is MuseCodeAgent

    def test_install_pins_version_and_checksums_per_architecture(self):
        script = " ".join(MuseCodeAgent.install())
        assert (
            "https://lookaside.facebook.com/lookaside/muse/download/?channel=muse"
            f"&version={MuseCodeAgent.version}&file=muse-$arch-linux"
        ) in script
        assert "x86_64) arch=x86;" in script
        assert "aarch64) arch=aarch64;" in script
        assert f"{AGENTS_BIN_DIR}/muse" in script
        assert script.count("sha256sum -c") == 1

    def test_capabilities(self):
        assert MuseCodeAgent.prompt_input == "file"
        assert MuseCodeAgent.events_from == "file"
        assert MuseCodeAgent.applies_reasoning_effort
        assert not MuseCodeAgent.supports_temperature
        assert not MuseCodeAgent.supports_turn_limit
        assert MuseCodeAgent.providers == ("meta",)

    def test_every_step_runs_in_the_same_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = _agent(tmp_path, monkeypatch)
        prompt = tmp_path / "step_0.prompt"
        first = agent._argv("x", prompt)  # pyright: ignore[reportPrivateUsage]
        assert first == [
            f"{AGENTS_BIN_DIR}/muse",
            "exec",
            "--json",
            "--yolo",
            "--session-id",
            agent._session_id,  # pyright: ignore[reportPrivateUsage]
            "--model",
            "muse-spark-1.3",
            "--prompt-file",
            str(prompt),
        ]
        agent._step_index = 1  # pyright: ignore[reportPrivateUsage]
        assert agent._argv("x", prompt) == first  # pyright: ignore[reportPrivateUsage]

    def test_argv_sends_the_reasoning_effort(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = _agent(tmp_path, monkeypatch, reasoning_effort="max")
        argv = agent._argv("x", tmp_path / "p")  # pyright: ignore[reportPrivateUsage]
        assert argv[-2:] == ["--reasoning-effort", "max"]

    def test_settings_pin_the_model_url_and_the_tool_server(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = _agent(tmp_path, monkeypatch)
        agent._forwarder = agent._make_forwarder()  # pyright: ignore[reportPrivateUsage]
        agent._forwarder.url = "http://127.0.0.1:1234"  # pyright: ignore[reportOptionalMemberAccess, reportPrivateUsage]
        assert _settings(agent) == {
            "schema_version": 1,
            "endpoint_transport": {
                "base_url": "http://127.0.0.1:1234/v1",
                "auth": "bearer",
            },
            "mcpServers": {
                "karotte": {"type": "streamable-http", "url": agent.mcp_url}
            },
            "telemetry": {"enabled": False},
        }

    def test_env_keeps_muse_in_its_home_with_a_placeholder_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = _agent(tmp_path, monkeypatch)
        agent._forwarder = agent._make_forwarder()  # pyright: ignore[reportPrivateUsage]
        env = agent._env()  # pyright: ignore[reportPrivateUsage]
        assert env["HOME"] == str(tmp_path)
        assert env["XDG_CONFIG_HOME"] == str(tmp_path / ".muse_home" / "config")
        assert env["XDG_DATA_HOME"] == str(tmp_path / ".muse_home" / "data")
        assert env["META_API_KEY"] == PROXY_PLACEHOLDER_KEY
        assert "real-key" not in env.values()

    def test_forwards_to_meta_with_the_key_and_the_models_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        forwarder = _agent(tmp_path, monkeypatch)._make_forwarder()  # pyright: ignore[reportPrivateUsage]
        assert isinstance(forwarder, MuseForwarder)
        assert forwarder.upstream == "https://api.meta.ai"
        assert forwarder._api_key == "real-key"  # pyright: ignore[reportPrivateUsage]
        assert forwarder._models["data"] == [  # pyright: ignore[reportPrivateUsage]
            {
                "id": "muse-spark-1.3",
                "object": "model",
                "metadata": {"muse-code": {"limit": {"output": 128000}}},
            }
        ]

    def test_forwards_to_a_proxy_even_without_a_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = _agent(tmp_path, monkeypatch, model_api_key=None)
        monkeypatch.setenv("KAROTTE_PROXY_URL", "http://proxy:4000/")
        forwarder = agent._make_forwarder()  # pyright: ignore[reportPrivateUsage]
        assert isinstance(forwarder, MuseForwarder)
        assert forwarder.upstream == "http://proxy:4000"
        assert forwarder._api_key == PROXY_PLACEHOLDER_KEY  # pyright: ignore[reportPrivateUsage]

    def test_finds_the_session_log_and_keeps_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = _agent(tmp_path, monkeypatch)
        assert agent._event_log() is None  # pyright: ignore[reportPrivateUsage]
        sessions = tmp_path / ".muse_home" / "data" / "muse" / "sessions"
        log = sessions / "2026" / "11" / "13" / agent._session_id / "session.jsonl"  # pyright: ignore[reportPrivateUsage]
        log.parent.mkdir(parents=True)
        log.touch()
        assert agent._event_log() == log  # pyright: ignore[reportPrivateUsage]

        decoy = sessions / "2000" / "01" / "01" / agent._session_id / "session.jsonl"  # pyright: ignore[reportPrivateUsage]
        decoy.parent.mkdir(parents=True)
        decoy.touch()
        assert agent._event_log() == log  # pyright: ignore[reportPrivateUsage]

    def test_a_failed_turn_is_the_reported_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        agent = _agent(tmp_path, monkeypatch)
        parse = agent._line_parser("x")  # pyright: ignore[reportPrivateUsage]
        _ = list(
            parse(
                _run_event(
                    {
                        "kind": "terminal",
                        "terminal": "failed",
                        "reason": "API error 401",
                    }
                ),
                0,
            )
        )
        assert agent._reported_error() == "API error 401"  # pyright: ignore[reportPrivateUsage]


def _run_event(event: dict[str, Any]) -> str:
    return json.dumps(
        {
            "payload_type": "runtime.session",
            "payload": {"kind": "run", "event": event},
        }
    )


def _messages(events: list[Event], role: str) -> list[MessageAddedEvent]:
    return [
        e for e in events if isinstance(e, MessageAddedEvent) and e.message.role == role
    ]


def _completed(events: list[Event]) -> list[ToolCallCompletedEvent]:
    return [e for e in events if isinstance(e, ToolCallCompletedEvent)]


def _usage(events: list[Event]) -> list[tuple[int, int, int | None]]:
    return [
        (e.input_tokens, e.output_tokens, e.cache_read_tokens)
        for e in events
        if isinstance(e, TokenUsageEvent)
    ]


class TestStreamParser:
    def test_each_model_response_is_one_message_with_its_calls(self):
        events = _parse(_lines("step_0.ndjson"))
        messages = [m.message for m in _messages(events, "assistant")]
        assert [[c.function.name for c in m.tool_calls or []] for m in messages] == [
            [
                "mcp__karotte__add",
                "mcp__karotte__explode",
                "mcp__karotte__picture",
                "bash",
            ],
            [],
        ]
        assert messages[0].content == (
            "Calling your four requests and gathering each result."
        )
        assert isinstance(messages[1].content, str)
        assert messages[1].content.startswith("Results:\n\n- `add(2, 3)` → `5`")
        assert all(m.reasoning_content is None for m in messages)

    def test_tool_calls_keep_their_ids_and_arguments(self):
        events = _parse(_lines("step_0.ndjson"))
        calls = [e.tool_call for e in events if isinstance(e, ToolCallStartedEvent)]
        assert calls[0].id == "call_01a1d722c3497325be47b1493a1cbafd"
        assert json.loads(calls[0].function.arguments) == {"a": 2, "b": 3}
        assert json.loads(calls[3].function.arguments) == {
            "command": "ls /nonexistent",
            "description": "List nonexistent directory path",
        }

    def test_results_follow_the_message_that_called_them(self):
        events = _parse(_lines("step_0.ndjson"))
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
        assert not open_calls

    def test_results_carry_text_errors_and_images(self):
        add, explode, picture, ls = (
            c.result for c in _completed(_parse(_lines("step_0.ndjson")))
        )
        assert add.content == [TextContent(type="text", text="5")]
        assert not add.is_error
        assert explode.is_error
        assert explode.content == [
            TextContent(type="text", text="tool failed: Error executing tool explode")
        ]
        assert not picture.is_error
        assert isinstance(picture.content[1], ImageContent)
        assert picture.content[1].mime_type == "image/png"
        assert ls.is_error
        assert '"command": "ls /nonexistent"' in ls.content[0].text  # pyright: ignore[reportAttributeAccessIssue]

    def test_the_image_reaches_the_tool_message(self):
        events = _parse(_lines("step_0.ndjson"))
        picture = _messages(events, "tool")[2].message
        assert isinstance(picture.content, list)
        assert picture.content[1]["image_url"]["url"].startswith(
            "data:image/png;base64,iVBORw0KGgo"
        )

    def test_usage_counts_every_model_call_including_reminders(self):
        assert _usage(_parse(_lines("step_0.ndjson"))) == [
            (33298, 449, 0),
            (33985, 259, 33265),
            (18922, 799, 0),
            (5574, 443, 0),
            (4929, 259, 0),
        ]

    def test_a_later_step_has_only_its_own_messages(self):
        events = _parse(_lines("step_1.ndjson"))
        (write,) = [e.tool_call for e in events if isinstance(e, ToolCallStartedEvent)]
        assert write.function.name == "write_file"
        assert _messages(events, "assistant")[-1].message.content == (
            "`add(2, 3)` returned `5`.\n\nWrote `5` to [five.txt](/workdir/five.txt)."
        )
        assert len(_usage(events)) == 5

    def test_a_compaction_is_marked_and_its_usage_counted(self):
        events = _parse(_lines("compaction.ndjson"))
        assert sum(isinstance(e, ContextCompactedEvent) for e in events) == 1
        assert (2343, 701, 0) in _usage(events)
        compacted = next(
            i for i, e in enumerate(events) if isinstance(e, ContextCompactedEvent)
        )
        assert all(
            not isinstance(e, ToolCallCompletedEvent) for e in events[compacted:]
        )

    def test_skips_malformed_lines_and_other_records(self):
        other = _run_event({"kind": "context_block_diagnostic"})
        assert _parse(["not json", "[1]", "{}", other, ""]) == []

    def test_flush_keeps_what_a_killed_step_left_unsent(self):
        lines = _lines("step_0.ndjson")
        cut = next(
            i for i, line in enumerate(lines) if "tool_result_batch_committed" in line
        )
        events = _parse(lines[: cut + 1])
        assert len(_messages(events, "assistant")) == 1
        assert len(_completed(events)) == 4
