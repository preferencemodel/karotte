import re

import pytest

from karotte.judges._composite_judge import AndJudge, OrJudge
from karotte.judges.regex_judge import RegexJudge
from karotte.schemas.chat import Message
from karotte.schemas.transcript import MessageAddedEvent, Transcript


def _passing() -> RegexJudge:
    return RegexJudge([re.compile("found")])


def _failing() -> RegexJudge:
    return RegexJudge([re.compile("missing")])


def _seed(transcript: Transcript) -> None:
    transcript.events.append(MessageAddedEvent(message=Message(content="found")))


class TestEmptyJudgesList:
    def test_and_judge_rejects_empty_list(self) -> None:
        with pytest.raises(ValueError, match="at least one judge"):
            AndJudge([])

    def test_or_judge_rejects_empty_list(self) -> None:
        with pytest.raises(ValueError, match="at least one judge"):
            OrJudge([])


class TestAndJudge:
    def test_all_pass(self, transcript: Transcript) -> None:
        judge1 = _passing()
        judge2 = _passing()

        _seed(transcript)

        scoring = (judge1 & judge2).evaluate(transcript)

        assert scoring.continue_task is True
        assert scoring.score == 1.0
        judges_tree = scoring.metadata["judges"]
        assert judges_tree.startswith("✔  And (score: 1.0)\n")
        assert "✔  RegexJudge (score: 1.0)" in judges_tree
        assert "⊘" not in judges_tree  # no unevaluated judges

    def test_first_fails_short_circuits(self, transcript: Transcript) -> None:
        judge1 = _failing()
        judge2 = _passing()

        _seed(transcript)

        scoring = (judge1 & judge2).evaluate(transcript)

        assert scoring.continue_task is False
        assert scoring.score == 0.0
        judges_tree = scoring.metadata["judges"]
        assert judges_tree.startswith("❗ And (score: 0.0)\n")
        assert "❗ RegexJudge (score: 0.0)" in judges_tree
        assert "⊘  RegexJudge (not evaluated)" in judges_tree

    def test_second_fails(self, transcript: Transcript) -> None:
        judge1 = _passing()
        judge2 = _failing()

        _seed(transcript)

        scoring = (judge1 & judge2).evaluate(transcript)

        assert scoring.continue_task is False
        assert scoring.score == 0.0
        judges_tree = scoring.metadata["judges"]
        assert judges_tree.startswith("❗ And (score: 0.0)\n")
        assert "✔  RegexJudge (score: 1.0)" in judges_tree
        assert "❗ RegexJudge (score: 0.0)" in judges_tree

    def test_metadata_shown_for_failed_judge(self, transcript: Transcript) -> None:
        judge1 = _failing()
        judge2 = _passing()

        _seed(transcript)

        scoring = (judge1 & judge2).evaluate(transcript)

        judges_tree = scoring.metadata["judges"]
        assert "Transcript contains no match." in judges_tree


class TestOrJudge:
    def test_first_passes_short_circuits(self, transcript: Transcript) -> None:
        judge1 = _passing()
        judge2 = _failing()

        _seed(transcript)

        scoring = (judge1 | judge2).evaluate(transcript)

        assert scoring.continue_task is True
        assert scoring.score == 1.0
        judges_tree = scoring.metadata["judges"]
        assert judges_tree.startswith("✔  Or (score: 1.0)\n")
        assert "✔  RegexJudge (score: 1.0)" in judges_tree
        assert "⊘  RegexJudge (not evaluated)" in judges_tree

    def test_first_fails_second_passes(self, transcript: Transcript) -> None:
        judge1 = _failing()
        judge2 = _passing()

        _seed(transcript)

        scoring = (judge1 | judge2).evaluate(transcript)

        assert scoring.continue_task is True
        assert scoring.score == 1.0
        judges_tree = scoring.metadata["judges"]
        assert judges_tree.startswith("✔  Or (score: 1.0)\n")
        assert "❗ RegexJudge (score: 0.0)" in judges_tree
        assert "✔  RegexJudge (score: 1.0)" in judges_tree

    def test_all_fail(self, transcript: Transcript) -> None:
        judge1 = _failing()
        judge2 = _failing()

        _seed(transcript)

        scoring = (judge1 | judge2).evaluate(transcript)

        assert scoring.continue_task is False
        assert scoring.score == 0.0
        judges_tree = scoring.metadata["judges"]
        assert judges_tree.startswith("❗ Or (score: 0.0)\n")
        assert judges_tree.count("❗ RegexJudge") == 2


class TestNesting:
    def test_and_inside_or(self, transcript: Transcript) -> None:
        # (a & b) | c — if both a and b pass, or c passes
        judge_a = _passing()
        judge_b = _passing()
        judge_c = _failing()

        _seed(transcript)

        scoring = ((judge_a & judge_b) | judge_c).evaluate(transcript)

        assert scoring.continue_task is True
        judges_tree = scoring.metadata["judges"]
        assert judges_tree.startswith("✔  Or (score: 1.0)\n")
        # The And passed, so it shows as ✔
        assert "✔  And (score: 1.0)" in judges_tree
        # The nested judges should be shown
        assert "✔  RegexJudge (score: 1.0)" in judges_tree
        # c was not evaluated due to short-circuit
        assert "⊘  RegexJudge (not evaluated)" in judges_tree

    def test_or_inside_and(self, transcript: Transcript) -> None:
        # (a | b) & c — one of a or b must pass, and c must pass
        judge_a = _failing()
        judge_b = _passing()
        judge_c = _passing()

        _seed(transcript)

        scoring = ((judge_a | judge_b) & judge_c).evaluate(transcript)

        assert scoring.continue_task is True
        judges_tree = scoring.metadata["judges"]
        assert judges_tree.startswith("✔  And (score: 1.0)\n")
        assert "✔  Or (score: 1.0)" in judges_tree
        assert "✔  RegexJudge (score: 1.0)" in judges_tree


class TestFlattening:
    def test_and_flattens(self) -> None:
        judge1 = _passing()
        judge2 = _passing()
        judge3 = _passing()

        combined = judge1 & judge2 & judge3

        assert isinstance(combined, AndJudge)
        assert len(combined.judges) == 3

    def test_or_flattens(self) -> None:
        judge1 = _passing()
        judge2 = _passing()
        judge3 = _passing()

        combined = judge1 | judge2 | judge3

        assert isinstance(combined, OrJudge)
        assert len(combined.judges) == 3


class TestTreeFormat:
    def test_nested_composite_not_duplicated(self, transcript: Transcript) -> None:
        """Nested composite judges should not have their name printed twice."""
        judge_a = _passing()
        judge_b = _passing()
        judge_c = _passing()

        _seed(transcript)

        scoring = ((judge_a | judge_b) & judge_c).evaluate(transcript)

        judges_tree = scoring.metadata["judges"]
        # Or should only appear once (as a child), not twice
        assert judges_tree.count("Or") == 1

    def test_tree_structure_with_prefixes(self, transcript: Transcript) -> None:
        judge1 = _passing()
        judge2 = _passing()
        judge3 = _passing()

        _seed(transcript)

        scoring = (judge1 & judge2 & judge3).evaluate(transcript)

        judges_tree = scoring.metadata["judges"]
        assert judges_tree.startswith("✔  And (score: 1.0)\n")
        # First two should have ├─ prefix
        assert "├─ ✔  RegexJudge" in judges_tree
        # Last one should have └─ prefix
        assert "└─ ✔  RegexJudge" in judges_tree

    def test_empty_metadata_not_shown(self, transcript: Transcript) -> None:
        judge1 = _passing()

        _seed(transcript)

        scoring = (judge1 & judge1).evaluate(transcript)

        judges_tree = scoring.metadata["judges"]
        # Empty metadata should not appear
        assert "metadata: {}" not in judges_tree

    def test_empty_lines_around_nested_content(self, transcript: Transcript) -> None:
        """Spacing lines should appear before and after nested content."""
        judge_a = _failing()
        judge_b = _passing()
        judge_c = _passing()

        _seed(transcript)

        scoring = ((judge_a | judge_b) & judge_c).evaluate(transcript)

        judges_tree = scoring.metadata["judges"]
        lines = judges_tree.splitlines()

        # Find the Or line
        or_line_idx = next(i for i, line in enumerate(lines) if "Or" in line)

        # Next line should be a spacing line with continuation characters at each level
        spacing_line = lines[or_line_idx + 1]
        assert spacing_line.startswith("│")
        assert "├" not in spacing_line and "└" not in spacing_line  # No branch chars

        # Find the last RegexJudge inside Or (the one that passed)
        # After its content, there should be a spacing line before the next sibling
        last_or_child_idx = next(
            i
            for i, line in enumerate(lines)
            if "└─ ✔  RegexJudge" in line and i > or_line_idx
        )
        # After nested content ends, there should be a spacing line
        after_line = lines[last_or_child_idx + 1]
        assert after_line.startswith("│")
        assert "├" not in after_line and "└" not in after_line

    def test_empty_line_after_root(self, transcript: Transcript) -> None:
        """Empty line should appear after the root."""
        judge1 = _passing()
        judge2 = _passing()

        _seed(transcript)

        scoring = (judge1 & judge2).evaluate(transcript)

        judges_tree = scoring.metadata["judges"]
        lines = judges_tree.splitlines()

        # First line is the root
        assert "And (score: 1.0)" in lines[0]

        # Second line should be the continuation character
        assert lines[1] == "│"

        # Third line is the first child
        assert "RegexJudge" in lines[2]

    def test_metadata_directly_under_judge(self, transcript: Transcript) -> None:
        """Metadata should appear directly under the judge without blank line."""
        judge1 = _failing()
        judge2 = _passing()

        _seed(transcript)

        scoring = (judge1 & judge2).evaluate(transcript)

        judges_tree = scoring.metadata["judges"]
        lines = judges_tree.splitlines()

        # Find the failed judge line
        failed_idx = next(i for i, line in enumerate(lines) if "❗ RegexJudge" in line)

        # Next line should be the metadata (no blank line)
        assert "Transcript contains no match." in lines[failed_idx + 1]
