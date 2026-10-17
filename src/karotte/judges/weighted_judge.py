import math
from collections.abc import Mapping
from typing import Any, Final, override

from karotte.judges.judge import Judge
from karotte.schemas.scoring import Scoring
from karotte.schemas.transcript import Transcript


class WeightedJudge(Judge):
    """Combine component scores after mandatory gates pass.

    Components map names to ``(judge, weight)`` pairs. Gates map names to
    judges whose ``continue_task`` must be true before any component runs.
    A failed gate returns score 0 and stops the task, regardless of its score.

    The reward is the weighted mean of the component scores. Components must
    use comparable scales, with 0 as a failure reward when using gates.
    Scores are not clamped. Their ``continue_task``
    flags are recorded but do not veto the reward. The task continues when
    the reward is at least ``continue_threshold``.

    Both mappings are copied and evaluated in insertion order. Exceptions
    propagate, including student misbehavior for ``Step.score`` to handle.
    """

    def __init__(
        self,
        components: Mapping[str, tuple[Judge, float]],
        *,
        continue_threshold: float,
        gates: Mapping[str, Judge] | None = None,
    ) -> None:
        if not components:
            raise ValueError("WeightedJudge requires at least one component")
        if not math.isfinite(continue_threshold):
            raise ValueError("continue_threshold must be finite")

        self._components: Final = tuple(components.items())
        self._gates: Final = tuple((gates or {}).items())
        self.continue_threshold: Final = continue_threshold

        for name, _ in (*self._components, *self._gates):
            _check_name(name)
        for name, (_, weight) in self._components:
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"Weight for {name!r} must be finite and positive")
        try:
            total_weight = math.fsum(weight for _, (_, weight) in self._components)
        except OverflowError:
            raise ValueError("The sum of component weights must be finite") from None
        if not math.isfinite(total_weight):
            raise ValueError("The sum of component weights must be finite")
        self._total_weight: Final = total_weight

    @override
    def evaluate(self, transcript: Transcript) -> Scoring:
        gate_results: dict[str, dict[str, Any]] = {
            name: {"evaluated": False} for name, _ in self._gates
        }
        component_results: dict[str, dict[str, Any]] = {
            name: {
                "evaluated": False,
                "weight": weight,
                "normalized_weight": weight / self._total_weight,
            }
            for name, (_, weight) in self._components
        }
        metadata = {
            "gates": gate_results,
            "components": component_results,
            "continue_threshold": self.continue_threshold,
        }

        for name, judge in self._gates:
            scoring = judge.evaluate(transcript)
            gate_results[name] = _result(name, scoring)
            if not scoring.continue_task:
                return Scoring(score=0.0, metadata=metadata, continue_task=False)

        contributions: list[float] = []
        for name, (judge, weight) in self._components:
            scoring = judge.evaluate(transcript)
            component_results[name].update(_result(name, scoring))
            # Normalize before multiplying so a large weight and score do not
            # overflow even when their weighted mean is finite.
            contribution = (weight / self._total_weight) * scoring.score
            component_results[name]["contribution"] = contribution
            contributions.append(contribution)

        score = math.fsum(contributions)
        return Scoring(
            score=score,
            metadata=metadata,
            continue_task=score >= self.continue_threshold,
        )


def _check_name(name: object) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Judge names must be non-empty strings")


def _result(name: str, scoring: Scoring) -> dict[str, Any]:
    if not math.isfinite(scoring.score):
        raise ValueError(
            f"Judge {name!r} returned a non-finite score: {scoring.score!r}"
        )
    return {
        "evaluated": True,
        "score": scoring.score,
        "continue_task": scoring.continue_task,
        "metadata": scoring.metadata,
    }
