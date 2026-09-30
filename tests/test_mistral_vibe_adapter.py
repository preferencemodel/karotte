"""Tests for the Mistral Vibe NDJSON -> transcript-event adapter, driven by
`vibe --output streaming` logs with the system prompt omitted
(tests/resources/mistral_vibe/)."""

import json
from pathlib import Path

import pytest

from karotte.agents.mistral_vibe.adapter import parse_stream
from karotte.agents.mistral_vibe.agent import (
    MistralVibeAgent,
    _transcript_events,  # pyright: ignore[reportPrivateUsage]
)
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.transcript import (
    MessageAddedEvent,
    TokenUsageEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)

_FIXTURE_DIR = Path(__file__).parent / "resources" / "mistral_vibe"

MODEL = "mistral/mistral-medium-3-5"


def _fixture_lines(name: str) -> list[str]:
    return (_FIXTURE_DIR / name).read_text().splitlines()


class TestParseStream:
    def test_single_tool_call_step_event_sequence(self):
        events = list(parse_stream(_fixture_lines("step_0.ndjson"), model=MODEL))

        assert [type(e).__name__ for e in events] == [
            "MessageAddedEvent",  # vibe's own system prompt
            "MessageAddedEvent",  # user instructions
            "MessageAddedEvent",  # assistant, tool call only
            "TokenUsageEvent",
            "ToolCallStartedEvent",
            "ToolCallCompletedEvent",
            "MessageAddedEvent",  # tool result
            "MessageAddedEvent",  # final assistant answer
            "TokenUsageEvent",
        ]

    def test_roles_and_final_answer(self):
        events = list(parse_stream(_fixture_lines("step_0.ndjson"), model=MODEL))
        messages = [e.message for e in events if isinstance(e, MessageAddedEvent)]
        assert [m.role for m in messages] == [
            "system",
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        assert messages[-1].content == "path: /workdir/.venv/bin/python3"

    def test_tool_call_ids_match_results(self):
        events = list(parse_stream(_fixture_lines("step_1.ndjson"), model=MODEL))
        started = [e for e in events if isinstance(e, ToolCallStartedEvent)]
        completed = [e for e in events if isinstance(e, ToolCallCompletedEvent)]

        assert [s.tool_call.function.name for s in started] == ["bash", "write_file"]
        assert [c.tool_call_id for c in completed] == [s.tool_call.id for s in started]
        assert json.loads(started[0].tool_call.function.arguments) == {
            "command": "python --version"
        }
        assert "Python 3.12.11" in completed[0].result.content[0].text  # pyright: ignore[reportAttributeAccessIssue]

    def test_tool_result_becomes_tool_message(self):
        events = list(parse_stream(_fixture_lines("step_1.ndjson"), model=MODEL))
        tool_messages = [
            e.message
            for e in events
            if isinstance(e, MessageAddedEvent) and e.message.role == "tool"
        ]
        assert len(tool_messages) == 2
        assert all(m.tool_call_id for m in tool_messages)

    def test_token_usage_is_estimated_and_nonzero(self):
        events = list(parse_stream(_fixture_lines("step_1.ndjson"), model=MODEL))
        usages = [e for e in events if isinstance(e, TokenUsageEvent)]

        # One usage event per assistant message.
        n_assistant = sum(
            1
            for e in events
            if isinstance(e, MessageAddedEvent) and e.message.role == "assistant"
        )
        assert len(usages) == n_assistant == 3

        assert all(u.estimated for u in usages)
        # Every assistant turn produced text or tool-call arguments.
        assert all(u.output_tokens > 0 for u in usages)
        # Input grows with conversation history (system + user prompt first).
        assert usages[0].input_tokens > 0
        assert usages[0].input_tokens < usages[1].input_tokens
        assert usages[1].input_tokens < usages[2].input_tokens

    def test_unmapped_fields_go_into_raw(self):
        events = list(parse_stream(_fixture_lines("step_0.ndjson"), model=MODEL))
        tool_events = [
            e
            for e in events
            if isinstance(e, MessageAddedEvent) and e.message.role == "tool"
        ]
        assert tool_events[0].raw == {"name": "bash"}

    def test_reasoning_content_maps_to_message(self):
        line = json.dumps(
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "thinking...",
                "reasoning_signature": "sig",
            }
        )
        events = list(parse_stream([line]))
        message_event = next(e for e in events if isinstance(e, MessageAddedEvent))
        assert message_event.message.reasoning_content == "thinking..."
        assert message_event.raw == {"reasoning_signature": "sig"}

    def test_garbage_lines_are_skipped(self):
        lines = [
            "",
            "not json at all {{{",
            '["a", "json", "array"]',
            '{"role": "narrator", "content": "unknown role"}',
            '{"role": "assistant", "content": "still parsed"}',
        ]
        events = list(parse_stream(lines))
        messages = [e for e in events if isinstance(e, MessageAddedEvent)]
        assert len(messages) == 1
        assert messages[0].message.content == "still parsed"

    def test_tool_call_without_id_gets_synthetic_id(self):
        line = json.dumps(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"index": 0, "function": {"name": "bash", "arguments": "{}"}}
                ],
            }
        )
        events = list(parse_stream([line]))
        started = next(e for e in events if isinstance(e, ToolCallStartedEvent))
        assert started.tool_call.id

    def test_tool_result_without_id_yields_no_completed_event(self):
        line = json.dumps({"role": "tool", "content": "orphan result"})
        events = list(parse_stream([line]))
        assert not any(isinstance(e, ToolCallCompletedEvent) for e in events)
        messages = [e for e in events if isinstance(e, MessageAddedEvent)]
        assert messages[0].message.content == "orphan result"

    def test_unknown_model_still_estimates_tokens(self):
        line = json.dumps({"role": "assistant", "content": "some answer text"})
        events = list(parse_stream([line], model="acme/unknown-model"))
        usage = next(e for e in events if isinstance(e, TokenUsageEvent))
        assert usage.output_tokens > 0
        assert usage.estimated


class TestTranscriptEvents:
    """The agent-side filter on top of parse_stream."""

    def _instructions(self, name: str) -> str:
        user_line = json.loads(_fixture_lines(name)[1])
        assert user_line["role"] == "user"
        return user_line["content"]

    def test_drops_vibe_system_prompt_and_echoed_instructions(self):
        lines = _fixture_lines("step_0.ndjson")
        events = list(
            _transcript_events(lines, self._instructions("step_0.ndjson"), MODEL)
        )
        roles = [e.message.role for e in events if isinstance(e, MessageAddedEvent)]
        assert roles == ["assistant", "tool", "assistant"]

    def test_keeps_user_message_that_differs_from_instructions(self):
        lines = _fixture_lines("step_0.ndjson")
        events = list(_transcript_events(lines, "different instructions", MODEL))
        roles = [e.message.role for e in events if isinstance(e, MessageAddedEvent)]
        assert roles == ["user", "assistant", "tool", "assistant"]

    def test_token_usage_still_reported(self):
        lines = _fixture_lines("step_1.ndjson")
        events = list(
            _transcript_events(lines, self._instructions("step_1.ndjson"), MODEL)
        )
        assert sum(isinstance(e, TokenUsageEvent) for e in events) == 3


class TestEnv:
    def test_drops_pythonsafepath(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PYTHONSAFEPATH", "1")
        agent = MistralVibeAgent(
            EvaluationRunConfig(run_id="r", task_id="t", model=MODEL, model_api_key="k")
        )

        env = agent._env()  # pyright: ignore[reportPrivateUsage]

        assert "PYTHONSAFEPATH" not in env
        assert env["HOME"]
