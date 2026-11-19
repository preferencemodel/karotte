import re
from pathlib import Path

import pytest

from karotte.judges.regex_judge import RegexJudge
from karotte.schemas.chat import ChatCompletionMessageToolCall, Function, Message, Role
from karotte.schemas.transcript import (
    MessageAddedEvent,
    StepStartedEvent,
    Transcript,
)

DOCS = Path(__file__).parent.parent / "docs/tasks"
FINAL_ANSWER = r"answer: 42\.?\s*\Z"
FINAL_PATH = r"path: \S*/python[\d.]*\s*\Z"


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


def _score(pattern: str, transcript: Transcript, replies: list[str]) -> float:
    transcript.events.extend(
        MessageAddedEvent(message=Message(content=reply)) for reply in replies
    )
    return RegexJudge([re.compile(pattern)]).evaluate(transcript).score


@pytest.mark.parametrize(
    "replies",
    [
        ["answer: 42"],
        ["answer: 42."],
        ["answer: 42\n"],
        ["6 * 7 is 42.\n\nanswer: 42"],
        ["Let me multiply them.", "answer: 42"],
    ],
)
def test_end_anchored_pattern_matches_a_reply_ending_with_the_answer(
    transcript: Transcript, replies: list[str]
):
    assert _score(FINAL_ANSWER, transcript, replies) == 1


@pytest.mark.parametrize(
    "replies",
    [
        pytest.param(["answer: 42, 43, 44"], id="list"),
        pytest.param(["answer: 42 or 44"], id="or"),
        pytest.param(["answer: 42-44"], id="range"),
        pytest.param(["answer: 42", "Actually, answer: 44"], id="changed"),
        pytest.param(["answer: 42.5"], id="decimal"),
        pytest.param(["answer: 420"], id="longer"),
        pytest.param(["\n".join(f"answer: {n}" for n in range(100))], id="guesses"),
    ],
)
def test_end_anchored_pattern_does_not_match_several_answers(
    transcript: Transcript, replies: list[str]
):
    assert _score(FINAL_ANSWER, transcript, replies) == 0


@pytest.mark.parametrize(
    ("reply", "score"),
    [
        ("path: /usr/bin/python3", 1),
        ("`which python3` prints it.\n\npath: /usr/local/bin/python3.12.", 1),
        ("path: /usr/bin/perl /usr/bin/python3", 0),
        ("path: /usr/bin/python3\npath: /usr/bin/perl", 0),
    ],
)
def test_end_anchored_path_pattern(transcript: Transcript, reply: str, score: float):
    assert _score(FINAL_PATH, transcript, [reply]) == score


@pytest.mark.parametrize(
    ("page", "pattern"),
    [("scoring.md", FINAL_ANSWER), ("tasks-and-steps.md", FINAL_PATH)],
)
def test_docs_show_the_end_anchored_patterns(page: str, pattern: str):
    assert f'r"{pattern}"' in (DOCS / page).read_text(encoding="utf-8")
