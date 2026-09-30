from karotte.schemas.scoring import Scoring
from karotte.text import escape_surrogates


class StudentMisbehaviorError(Exception):
    """The student did something illegitimate. Raise this from scoring-time env
    code (pre_scoring_hook, judges) so karotte scores the run 0 instead of
    treating it as an infra error."""


def misbehavior_scoring(error: StudentMisbehaviorError) -> Scoring:
    """The message routinely quotes student-chosen filenames, so escape any
    surrogates before they land in metadata and break event serialization."""
    return Scoring(
        score=0.0,
        metadata={"misbehavior": escape_surrogates(str(error))},
        continue_task=False,
    )
