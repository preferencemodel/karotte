# pyright: reportImportCycles=false

import json
from enum import Enum
from typing import Any, Final, final, override

from karotte.judges.judge import Judge
from karotte.schemas.scoring import Scoring
from karotte.schemas.transcript import Transcript


class _ShortCircuitMode(Enum):
    AND = "AND"  # short-circuit on failure
    OR = "OR"  # short-circuit on success


def _format_judges_tree(
    root_type: str, results: list[dict[str, Any]], score: float, continue_task: bool
) -> str:
    """Format judge results as a tree structure."""
    icon = "✔ " if continue_task else "❗"
    lines: list[str] = [f"{icon} {root_type} (score: {score})", "│"]

    for i, result in enumerate(results):
        is_last = i == len(results) - 1
        prefix = "└─ " if is_last else "├─ "
        child_prefix = "   " if is_last else "│  "

        if result["evaluated"]:
            icon = "✔ " if result["continue_task"] else "❗"
            line = f"{prefix}{icon} {result['type']} (score: {result['score']})"
        else:
            line = f"{prefix}⊘  {result['type']} (not evaluated)"

        lines.append(line)

        has_children = False
        if result["evaluated"] and result.get("metadata"):
            metadata = result["metadata"]

            # Check if metadata is from a nested composite judge
            if (
                isinstance(metadata, dict)
                and "judges" in metadata
                and isinstance(metadata["judges"], str)
            ):
                # Nested composite: judges is already a formatted string
                # Skip the first line (root type) since we already printed it
                # Also skip the second line (empty line after root) since we add our own
                nested_lines = metadata["judges"].splitlines()[2:]
                if nested_lines:
                    # Add spacing line before first child
                    lines.append(child_prefix + "│")
                    for nested_line in nested_lines:
                        lines.append(f"{child_prefix}{nested_line}")
                    has_children = True
            else:
                # Regular metadata: format as tree entries
                metadata_lines = _format_metadata_tree(metadata, child_prefix)
                if metadata_lines:
                    lines.extend(metadata_lines)
                    has_children = True

        # Add spacing line after nested content if not the last item
        if has_children and not is_last:
            lines.append(child_prefix.rstrip())

    return "\n".join(lines)


def _format_metadata_tree(metadata: Any, prefix: str) -> list[str]:
    """Format metadata as tree entries under a judge."""
    if not metadata:
        return []

    if not isinstance(metadata, dict):
        return [f"{prefix}└─ metadata: {_format_value(metadata)}"]

    lines: list[str] = []
    items = list(metadata.items())

    for i, (key, value) in enumerate(items):
        is_last = i == len(items) - 1
        item_prefix = "└─ " if is_last else "├─ "
        lines.append(f"{prefix}{item_prefix}{key}: {_format_value(value)}")

    return lines


def _format_value(value: Any) -> str:
    """Format a value for display. Uses JSON for complex types."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str | int | float):
        return str(value)
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


class _CompositeJudge(Judge):
    """Base class for composite judges that evaluate multiple judges in sequence."""

    _display_name: str  # Subclasses must define this

    def __init__(self, judges: list[Judge], mode: _ShortCircuitMode) -> None:
        if not judges:
            raise ValueError("Composite judge requires at least one judge")
        self.judges: Final = judges
        self._mode: Final = mode

    @override
    def evaluate(self, transcript: Transcript) -> Scoring:
        results: list[dict[str, Any]] = []
        scoring: Scoring | None = None

        for i, judge in enumerate(self.judges):
            scoring = judge.evaluate(transcript)
            display_name = getattr(judge, "_display_name", type(judge).__name__)
            results.append(
                {
                    "type": display_name,
                    "evaluated": True,
                    "score": scoring.score,
                    "metadata": scoring.metadata,
                    "continue_task": scoring.continue_task,
                }
            )

            should_short_circuit = (
                scoring.continue_task is False and self._mode == _ShortCircuitMode.AND
            ) or (scoring.continue_task is True and self._mode == _ShortCircuitMode.OR)

            if should_short_circuit:
                for remaining in self.judges[i + 1 :]:
                    remaining_name = getattr(
                        remaining, "_display_name", type(remaining).__name__
                    )
                    results.append({"type": remaining_name, "evaluated": False})
                return Scoring(
                    score=scoring.score,
                    metadata={
                        "judges": _format_judges_tree(
                            self._display_name,
                            results,
                            scoring.score,
                            scoring.continue_task,
                        ),
                    },
                    continue_task=scoring.continue_task,
                )

        assert scoring is not None
        return Scoring(
            score=scoring.score,
            metadata={
                "judges": _format_judges_tree(
                    self._display_name,
                    results,
                    scoring.score,
                    scoring.continue_task,
                ),
            },
            continue_task=scoring.continue_task,
        )


@final
class AndJudge(_CompositeJudge):
    """Evaluates multiple judges in sequence. Short-circuits on first failure.

    Returns the first failing Scoring, or the last Scoring if all pass.
    """

    _display_name = "And"

    def __init__(self, judges: list[Judge]) -> None:
        super().__init__(judges, _ShortCircuitMode.AND)


@final
class OrJudge(_CompositeJudge):
    """Evaluates multiple judges in sequence. Short-circuits on first success.

    Returns the first passing Scoring, or the last Scoring if all fail.
    """

    _display_name = "Or"

    def __init__(self, judges: list[Judge]) -> None:
        super().__init__(judges, _ShortCircuitMode.OR)
