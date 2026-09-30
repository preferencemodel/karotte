import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

from mcp.types import TextContent

from karotte.schemas.transcript import (
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
    Transcript,
)


class RubricContext(ABC):
    """Base class for rubric context providers.

    A context provider extracts relevant text from the transcript
    (or external sources) to feed to the LLM for criterion evaluation.
    """

    @abstractmethod
    def render(self, transcript: Transcript) -> str: ...


@dataclass(frozen=True)
class AnswersContext(RubricContext):
    """Provides submitted answers as context.

    Args:
        key: Optional specific answer key. If None, all answers are concatenated.
    """

    key: str | None = None

    def render(self, transcript: Transcript) -> str:
        answers = transcript.answers
        if self.key is not None:
            value = answers.get(self.key, "")
            return value
        if not answers:
            return ""
        return "\n\n".join(f"{key}: {value}" for key, value in answers.items())


@dataclass(frozen=True)
class FileContext(RubricContext):
    """Provides file contents as context.

    Args:
        paths: List of absolute file paths to read.
    """

    paths: list[str]

    def render(self, transcript: Transcript) -> str:
        parts: list[str] = []
        for path in self.paths:
            try:
                with open(path) as f:
                    content = f.read()
                parts.append(f"# File '{path}':\n{content}")
            except Exception as e:
                parts.append(f"# File '{path}': Error reading file: {e}")
        return "\n\n".join(parts)


@dataclass(frozen=True)
class TranscriptContext(RubricContext):
    """Provides transcript messages/tool calls as context.

    Args:
        tool: If set, only include tool calls. Use a specific name to filter
              by tool, or "*" to include all tool calls.
        last: If set, only include the last N matching items.
    """

    tool: str | None = None
    last: int | None = None

    def render(self, transcript: Transcript) -> str:
        if self.tool is not None:
            return self._render_tool_calls(transcript)
        return self._render_messages(transcript)

    def _render_messages(self, transcript: Transcript) -> str:
        messages = transcript.messages
        if self.last is not None:
            messages = messages[-self.last :]
        if not messages:
            return ""
        parts: list[str] = []
        for msg in messages:
            content = (
                msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
            )
            parts.append(f"[{msg.role}]: {content}")
        return "\n".join(parts)

    def _render_tool_calls(self, transcript: Transcript) -> str:
        # Collect tool call started events and their matching completed events
        started: dict[str, ToolCallStartedEvent] = {}
        pairs: list[tuple[ToolCallStartedEvent, ToolCallCompletedEvent]] = []

        for event in transcript.events:
            if isinstance(event, ToolCallStartedEvent):
                if self.tool == "*" or event.tool_call.function.name == self.tool:
                    started[event.tool_call.id] = event
            elif isinstance(event, ToolCallCompletedEvent):
                if event.tool_call_id in started:
                    pairs.append((started.pop(event.tool_call_id), event))

        if self.last is not None:
            pairs = pairs[-self.last :]

        if not pairs:
            return ""

        parts: list[str] = []
        for start_event, end_event in pairs:
            func = start_event.tool_call.function
            result_texts = [
                c.text
                for c in (end_event.result.content or [])
                if isinstance(c, TextContent)
            ]
            result_str = "\n".join(result_texts)
            parts.append(
                f"Tool: {func.name}\nArguments: {func.arguments}\nResult:\n{result_str}"
            )
        return "\n\n".join(parts)
