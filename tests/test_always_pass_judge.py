from karotte.judges.always_pass_judge import AlwaysPassJudge
from karotte.schemas.transcript import Transcript


def test_always_passes_with_empty_transcript(transcript: Transcript) -> None:
    judge = AlwaysPassJudge()

    scoring = judge.evaluate(transcript)

    assert scoring.score == 1.0
    assert scoring.metadata == {}
    assert scoring.continue_task is True


def test_always_passes_regardless_of_transcript_content(
    transcript: Transcript,
) -> None:
    from karotte.schemas.transcript import AnswersSubmittedEvent

    transcript.events.append(AnswersSubmittedEvent(answers={"a": "wrong"}))

    judge = AlwaysPassJudge()
    scoring = judge.evaluate(transcript)

    assert scoring.score == 1.0
    assert scoring.metadata == {}
    assert scoring.continue_task is True
