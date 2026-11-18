import re

import pytest

from karotte.judges.regex_judge import RegexJudge
from karotte.schemas.chat import ChatCompletionMessageToolCall, Function, Message, Role
from karotte.schemas.transcript import (
    MessageAddedEvent,
    StepStartedEvent,
    Transcript,
)


def test_general_value_match(transcript: Transcript):
    expected_results = RegexJudge([re.compile(r".*23")])

    transcript.events.append(MessageAddedEvent(message=Message(content="203049823984")))

    scoring = expected_results.evaluate(transcript)

    assert scoring.score == 1
    assert scoring.metadata == {}
    assert scoring.continue_task is True


@pytest.mark.parametrize("role", ["user", "system", "tool", "function"])
@pytest.mark.parametrize(
    "content", ["answer: 42", [{"type": "text", "text": "answer: 42"}]]
)
def test_non_assistant_messages_do_not_match(
    transcript: Transcript, role: Role, content: str | list[dict[str, str]]
):
    transcript.events.extend(
        [
            MessageAddedEvent(message=Message(role=role, content=content)),
            MessageAddedEvent(message=Message(content="answer: 41")),
        ]
    )

    scoring = RegexJudge([re.compile("answer: 42")]).evaluate(transcript)

    assert scoring.score == 0
    assert scoring.metadata == {"answer: 42": "Transcript contains no match."}
    assert scoring.continue_task is False


@pytest.mark.parametrize(
    "message",
    [
        Message(content=None),
        Message(content=""),
        Message(content=[]),
        Message(content=None, reasoning_content="answer: 42"),
        Message(
            content=None,
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_1",
                    type="function",
                    function=Function(
                        name="bash", arguments='{"command": "answer: 42"}'
                    ),
                )
            ],
        ),
        Message(content=[{"type": "image_url", "image_url": {"url": "answer: 42"}}]),
        Message(content=[{"type": "text", "text": "wrong", "extra": "answer: 42"}]),
        Message(content=[{"type": "text", "text": None, "extra": "answer: 42"}]),
        Message(content=[{"type": "text", "text": 42}]),
    ],
)
def test_only_assistant_text_is_matched(transcript: Transcript, message: Message):
    transcript.events.append(MessageAddedEvent(message=message))

    scoring = RegexJudge([re.compile("42")]).evaluate(transcript)

    assert scoring.score == 0
    assert scoring.continue_task is False


def test_multipart_assistant_text_matches(transcript: Transcript):
    transcript.events.append(
        MessageAddedEvent(
            message=Message(
                content=[
                    {"type": "text", "text": "answer:"},
                    {"type": "image_url", "image_url": {"url": "image.png"}},
                    {"type": "text", "text": "42"},
                ]
            )
        )
    )

    scoring = RegexJudge([re.compile(r"answer:\s+42")]).evaluate(transcript)

    assert scoring.score == 1
    assert scoring.metadata == {}
    assert scoring.continue_task is True


def test_assistant_text_from_earlier_steps_is_included(transcript: Transcript):
    transcript.events.extend(
        [
            StepStartedEvent(step=0),
            MessageAddedEvent(message=Message(content="answer:")),
            StepStartedEvent(step=1),
            MessageAddedEvent(message=Message(role="user", content="next step")),
            MessageAddedEvent(message=Message(content="42")),
        ]
    )

    scoring = RegexJudge([re.compile(r"answer:\s+42")]).evaluate(transcript)

    assert scoring.score == 1
    assert scoring.continue_task is True


def test_empty_pattern_list_passes_without_assistant_text(transcript: Transcript):
    transcript.events.append(
        MessageAddedEvent(message=Message(role="user", content="answer: 42"))
    )

    scoring = RegexJudge([]).evaluate(transcript)

    assert scoring.score == 1
    assert scoring.metadata == {}
    assert scoring.continue_task is True


def test_general_value_mismatch(transcript: Transcript):
    expected_results = RegexJudge([re.compile(r"123")])

    transcript.events.append(MessageAddedEvent(message=Message(content="203049823984")))

    scoring = expected_results.evaluate(transcript)

    assert scoring.score == 0
    assert scoring.metadata == {"123": "Transcript contains no match."}
    assert scoring.continue_task is False


def test_general_value_mismatch_for_mutliple_values(transcript: Transcript):
    expected_results = RegexJudge(
        [re.compile(r"123"), re.compile(r"456"), re.compile(r"20304")]
    )

    transcript.events.append(MessageAddedEvent(message=Message(content="203049823984")))

    scoring = expected_results.evaluate(transcript)

    assert scoring.score == 0
    assert scoring.metadata == {
        "123": "Transcript contains no match.",
        "456": "Transcript contains no match.",
    }
    assert scoring.continue_task is False


def test_general_value_match_multiline(transcript: Transcript):
    expected_results = RegexJudge([re.compile(r".*2.*3", re.DOTALL)])

    transcript.events.append(MessageAddedEvent(message=Message(content="\n2\n03\n")))

    scoring = expected_results.evaluate(transcript)

    assert scoring.score == 1
    assert scoring.metadata == {}
    assert scoring.continue_task is True
