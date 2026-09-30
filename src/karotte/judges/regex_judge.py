import re
from collections.abc import Sequence
from typing import Final, override

from karotte.judges.judge import Judge
from karotte.schemas.scoring import Metadata, Score, Scoring
from karotte.schemas.transcript import Transcript


class RegexJudge(Judge):
    """The messages the model wrote must match the expected patterns via `re.search`.

    The patterns get matched against a concatenated string of all messages the
    model wrote.
    """

    def __init__(self, expected: Sequence[re.Pattern[str]]) -> None:
        self.expected: Final = expected

    @override
    def evaluate(self, transcript: Transcript) -> Scoring:
        full_transcript = "\n\n".join(
            str(m.content) for m in transcript.messages if m.content
        )

        score: Score = 1.0
        metadata: Metadata = {}

        for expected in self.expected:
            if not expected.search(full_transcript):
                score = 0.0
                metadata[expected.pattern] = "Transcript contains no match."

        return Scoring(score, metadata, score == 1.0)
