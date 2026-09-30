from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from karotte.schemas.scoring import Scoring
from karotte.schemas.transcript import Transcript

if TYPE_CHECKING:
    from karotte.judges._composite_judge import AndJudge, OrJudge


class Judge(ABC):
    @abstractmethod
    def evaluate(self, transcript: Transcript) -> Scoring: ...

    def __and__(self, other: "Judge") -> "AndJudge":
        from karotte.judges._composite_judge import AndJudge

        if isinstance(self, AndJudge):
            return AndJudge([*self.judges, other])
        return AndJudge([self, other])

    def __or__(self, other: "Judge") -> "OrJudge":
        from karotte.judges._composite_judge import OrJudge

        if isinstance(self, OrJudge):
            return OrJudge([*self.judges, other])
        return OrJudge([self, other])
