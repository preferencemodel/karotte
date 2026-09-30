from typing import override

from karotte.judges.judge import Judge
from karotte.schemas.scoring import Scoring
from karotte.schemas.transcript import Transcript


class AlwaysPassJudge(Judge):
    @override
    def evaluate(self, transcript: Transcript) -> Scoring:
        return Scoring(score=1.0, metadata={}, continue_task=True)
