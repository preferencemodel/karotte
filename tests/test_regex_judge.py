import re

from karotte.judges.regex_judge import RegexJudge
from karotte.schemas.chat import Message
from karotte.schemas.transcript import (
    MessageAddedEvent,
    Transcript,
)


def test_general_value_match(transcript: Transcript):
    expected_results = RegexJudge([re.compile(r".*23")])

    transcript.events.append(MessageAddedEvent(message=Message(content="203049823984")))

    scoring = expected_results.evaluate(transcript)

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
