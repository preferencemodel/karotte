import re
from collections.abc import Sequence
from typing import Final, override

from karotte.judges.judge import Judge
from karotte.schemas.chat import Message
from karotte.schemas.scoring import Metadata, Score, Scoring
from karotte.schemas.transcript import Transcript


class RegexJudge(Judge):
    """The messages the model wrote must match the expected patterns via `re.search`.

    The patterns get matched against the concatenated text of all assistant
    messages. Other roles, reasoning, and non-text content are excluded.
    """

    def __init__(self, expected: Sequence[re.Pattern[str]]) -> None:
        self.expected: Final = expected

    @override
    def evaluate(self, transcript: Transcript) -> Scoring:
        full_transcript = "\n\n".join(
            text
            for m in transcript.messages
            if m.role == "assistant" and (text := _message_text(m))
        )

        score: Score = 1.0
        metadata: Metadata = {}

        for expected in self.expected:
            if not expected.search(full_transcript):
                score = 0.0
                metadata[expected.pattern] = "Transcript contains no match."

        return Scoring(score, metadata, score == 1.0)


def final_answer_pattern(answer: str, prefix: str = "answer:") -> re.Pattern[str]:
    """Match `answer` only if it is the last answer the model gave.

    For use with `RegexJudge`. The answer is escaped and must follow the last
    `prefix` in the assistant text, so listing several answers or changing the
    answer later doesn't pass. The answer can't continue into a longer value:
    `answer: 5.5`, `answer: 55` and `answer: 5,000` don't match `5`.
    """
    label = re.escape(prefix)
    return re.compile(
        rf"{label}\s*{re.escape(answer)}(?![.,]?\w)(?!.*{label})", re.DOTALL
    )


def _message_text(message: Message) -> str:
    if isinstance(message.content, str):
        return message.content
    return "\n\n".join(
        part["text"]
        for part in message.content or []
        if part.get("type") == "text" and isinstance(part.get("text"), str)
    )
