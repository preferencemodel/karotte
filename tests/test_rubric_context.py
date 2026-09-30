"""Tests for RubricContext classes used by RubricJudge."""

from pathlib import Path

import pytest

from karotte.judges.rubric_context import (
    AnswersContext,
    FileContext,
    TranscriptContext,
)
from karotte.schemas.chat import ChatCompletionMessageToolCall, Function, Message
from karotte.schemas.transcript import (
    AnswersSubmittedEvent,
    MessageAddedEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
    Transcript,
)

try:
    from mcp.types import CallToolResult, TextContent
except ImportError:
    pytest.skip("mcp not available", allow_module_level=True)


# --- AnswersContext ---


class TestAnswersContext:
    def test_render_all_answers(self, transcript: Transcript) -> None:
        transcript.events.append(
            AnswersSubmittedEvent(answers={"q1": "answer1", "q2": "answer2"})
        )
        ctx = AnswersContext()
        result = ctx.render(transcript)

        assert "q1" in result
        assert "answer1" in result
        assert "q2" in result
        assert "answer2" in result

    def test_render_specific_key(self, transcript: Transcript) -> None:
        transcript.events.append(
            AnswersSubmittedEvent(answers={"q1": "answer1", "q2": "answer2"})
        )
        ctx = AnswersContext(key="q1")
        result = ctx.render(transcript)

        assert "answer1" in result
        assert "answer2" not in result

    def test_render_missing_key_returns_empty(self, transcript: Transcript) -> None:
        transcript.events.append(AnswersSubmittedEvent(answers={"q1": "answer1"}))
        ctx = AnswersContext(key="missing")
        result = ctx.render(transcript)

        assert result == ""

    def test_render_no_answers_returns_empty(self, transcript: Transcript) -> None:
        ctx = AnswersContext()
        result = ctx.render(transcript)

        assert result == ""


# --- FileContext ---


class TestFileContext:
    def test_render_reads_file(self, transcript: Transcript, tmp_path: Path) -> None:
        f = tmp_path / "hello.py"
        f.write_text("print('hello')")

        ctx = FileContext(paths=[str(f)])
        result = ctx.render(transcript)

        assert "print('hello')" in result

    def test_render_multiple_files(
        self, transcript: Transcript, tmp_path: Path
    ) -> None:
        f1 = tmp_path / "a.py"
        f1.write_text("file_a")
        f2 = tmp_path / "b.py"
        f2.write_text("file_b")

        ctx = FileContext(paths=[str(f1), str(f2)])
        result = ctx.render(transcript)

        assert "file_a" in result
        assert "file_b" in result

    def test_render_includes_file_path(
        self, transcript: Transcript, tmp_path: Path
    ) -> None:
        f = tmp_path / "hello.py"
        f.write_text("content")

        ctx = FileContext(paths=[str(f)])
        result = ctx.render(transcript)

        assert str(f) in result

    def test_render_empty_paths(self, transcript: Transcript) -> None:
        ctx = FileContext(paths=[])
        result = ctx.render(transcript)

        assert result == ""

    def test_render_missing_file_returns_error(self, transcript: Transcript) -> None:
        ctx = FileContext(paths=["/nonexistent/file.py"])
        result = ctx.render(transcript)

        assert "error" in result.lower() or result == ""


# --- TranscriptContext ---


class TestTranscriptContext:
    def _add_tool_call(
        self,
        transcript: Transcript,
        tool_name: str,
        arguments: str,
        result_text: str,
        tool_call_id: str = "tc1",
    ) -> None:
        """Helper to add a tool call started + completed event pair."""
        transcript.events.append(
            ToolCallStartedEvent(
                tool_call=ChatCompletionMessageToolCall(
                    id=tool_call_id,
                    function=Function(name=tool_name, arguments=arguments),
                    type="function",
                )
            )
        )
        transcript.events.append(
            ToolCallCompletedEvent(
                tool_call_id=tool_call_id,
                result=CallToolResult(
                    content=[TextContent(type="text", text=result_text)],
                    isError=False,
                ),
            )
        )

    def test_render_all_messages(self, transcript: Transcript) -> None:
        transcript.events.append(
            MessageAddedEvent(
                message=Message(content="Hello from assistant", role="assistant")
            )
        )
        transcript.events.append(
            MessageAddedEvent(message=Message(content="Tool result here", role="tool"))
        )

        ctx = TranscriptContext()
        result = ctx.render(transcript)

        assert "Hello from assistant" in result
        assert "Tool result here" in result

    def test_render_all_tool_calls_with_wildcard(self, transcript: Transcript) -> None:
        self._add_tool_call(transcript, "bash", '{"command": "ls"}', "file1.py", "tc1")
        self._add_tool_call(
            transcript,
            "submit_answers",
            '{"answers": {}}',
            "Submitted",
            "tc2",
        )

        ctx = TranscriptContext(tool="*")
        result = ctx.render(transcript)

        assert "bash" in result
        assert "file1.py" in result
        assert "submit_answers" in result
        assert "Submitted" in result

    def test_render_filter_by_tool(self, transcript: Transcript) -> None:
        self._add_tool_call(transcript, "bash", '{"command": "ls"}', "file1.py", "tc1")
        self._add_tool_call(
            transcript,
            "submit_answers",
            '{"answers": {}}',
            "Submitted",
            "tc2",
        )

        ctx = TranscriptContext(tool="bash")
        result = ctx.render(transcript)

        assert "ls" in result
        assert "file1.py" in result
        assert "submit_answers" not in result

    def test_render_last_n(self, transcript: Transcript) -> None:
        for i in range(5):
            self._add_tool_call(
                transcript,
                "bash",
                f'{{"command": "cmd{i}"}}',
                f"output{i}",
                f"tc{i}",
            )

        ctx = TranscriptContext(tool="bash", last=2)
        result = ctx.render(transcript)

        # Should only contain the last 2
        assert "cmd3" in result
        assert "cmd4" in result
        assert "cmd0" not in result
        assert "cmd1" not in result
        assert "cmd2" not in result

    def test_render_last_n_more_than_available(self, transcript: Transcript) -> None:
        self._add_tool_call(transcript, "bash", '{"command": "ls"}', "output", "tc1")

        ctx = TranscriptContext(tool="bash", last=10)
        result = ctx.render(transcript)

        assert "ls" in result
        assert "output" in result

    def test_render_last_n_messages(self, transcript: Transcript) -> None:
        for i in range(5):
            transcript.events.append(
                MessageAddedEvent(
                    message=Message(content=f"message{i}", role="assistant")
                )
            )

        ctx = TranscriptContext(last=2)
        result = ctx.render(transcript)

        assert "message3" in result
        assert "message4" in result
        assert "message0" not in result
        assert "message1" not in result
        assert "message2" not in result

    def test_render_non_string_message_content(self, transcript: Transcript) -> None:
        transcript.events.append(
            MessageAddedEvent(
                message=Message(
                    content=[{"type": "text", "text": "structured content"}],
                    role="assistant",
                )
            )
        )

        ctx = TranscriptContext()
        result = ctx.render(transcript)

        assert "structured content" in result

    def test_render_empty_transcript(self, transcript: Transcript) -> None:
        ctx = TranscriptContext()
        result = ctx.render(transcript)

        assert result == ""
